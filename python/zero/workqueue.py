"""The shared work queue of remote self-play (docs/az.md 5.3, 11.6): one pull model for a
single-game run's games and a co-training run's matches.

An iteration needs `units` of work in total: a single-game run's `games` (finished games, across
every machine), a co-training run's matches (its `games` events x their matches). Each **consumer**
- the run's own driver here ('local') and every remote worker ('w0', 'w1', ...) - takes work as it
has room under its **in-flight cap**, what it finishes in one iteration's running time:

    cap = clamp(ceil(inflight x rate x (lead + P)), lo, hi),  P = (units - sum rate x lead) / total rate

`rate` = the consumer's measured throughput (units finished per second of its running time, an
EWMA over its phases; one that had less work than its cap counts as if it had kept its cap busy
at the same latency), `lead` = its head start (a remote's phase begins as soon as the checkpoint
is written, before this machine's: an EWMA of the seconds it played before; this machine's: 0;
co-training: 0, the matches come with the drafting), `P` = the phase they all play together,
`total rate` = the sum over the consumers taking work now (a remote backing off or offline drops
out, so the others' shares grow at once), `inflight` = the run setting `selfplay_remote_inflight`
(default 1: each consumer holds one iteration's worth of its own throughput, so a game or match it
starts finishes within about one iteration of its running time), `lo` / `hi` its bounds (a few
games per driver group / its driver's slots). Before a consumer is measured its rate is assumed
proportional to its slots (`hi`), and a remote's cap is a quarter of what that gives (`BOOT`: a
remote slower than its slots say would otherwise start games that finish iterations later) until a
first measurement - its phase's end, or a few seconds of its status lines (zero.remoteplay). The
caps' sum is `inflight x units` (no bound binding).

`Shares` is that model; `split_games` cuts a driver's sample dict into its finished games (the
rows and the samples come out of the driver in the same order, each row counting its own
samples), so a game beyond an iteration's quota can wait - its samples with it - for the next
one; `lag_hist` is the staleness histogram (rows by how many checkpoints older than the
iteration's their search's network was).
"""
import math

import numpy as np

from .protocol import SAMPLE_KEYS

EWMA = 0.5                         # weight of a phase's measurement in a consumer's rate
MIN_RUN_S = 0.2                    # a phase shorter than this (running) measures nothing (the toys': ~1 s)
MIN_UTIL = 0.25                    # a consumer below this share of its cap: measured as if at 25%
BOOT = 0.25                        # an unmeasured remote's cap: this share of its slot-proportional one


def split_games(rows, s):
    """[(row, sample dict or None)] for the finished-game rows `rows` and the sample dict `s` the
    driver handed out with them (take_results + take_samples at the same point: game k's samples
    are rows[k]['samples'] consecutive samples after game k - 1's)."""
    out = []
    a = 0
    n = 0 if s is None else len(s['player'])
    off = None if s is None else np.asarray(s['opt_off'], np.int64)
    for r in rows:
        k = int(r.get('samples', 0) or 0)
        if not k:
            out.append((r, None))
            continue
        b = a + k
        if b > n:
            raise RuntimeError(f'split_games: game {r.get("gid")} has {k} samples, {n - a} left in the batch')
        lo, hi = int(off[a]), int(off[b])
        part = {}
        for key in SAMPLE_KEYS:
            if key not in s:
                continue
            v = s[key]
            if key == 'opt_off':
                part[key] = off[a:b + 1] - lo
            elif key in ('opt_action', 'pi'):
                part[key] = v[lo:hi]
            else:
                part[key] = v[a:b]
        out.append((r, part))
        a = b
    if a != n:
        raise RuntimeError(f'split_games: {n - a} samples left over after {len(rows)} games')
    return out


def lag_hist(net, it, into=None):
    """Rows of iteration `it`'s data by staleness: {lag: rows}, lag = (it - 1) - the row's `net`
    (0 = searched with the iteration's own checkpoint it - 1). Keys are strings (JSON)."""
    out = into if into is not None else {}
    if net is None or not len(net):
        return out
    lag = (it - 1) - np.asarray(net, np.int64)
    lag = np.maximum(lag, 0)
    for k, v in enumerate(np.bincount(lag)):
        if v:
            out[str(k)] = out.get(str(k), 0) + int(v)
    return out


def merge_hist(*hs):
    out = {}
    for h in hs:
        for k, v in (h or {}).items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def hist_summary(h):
    """{rows, mean, max, share_le1} of a lag histogram (None when empty)."""
    n = sum(h.values())
    if not n:
        return None
    mean = sum(int(k) * v for k, v in h.items()) / n
    return {'rows': n, 'mean': round(mean, 3), 'max': max(int(k) for k in h),
            'le1': round(sum(v for k, v in h.items() if int(k) <= 1) / n, 4)}


class Shares:
    """The consumers' throughput model and in-flight caps (module docstring). Not thread-safe:
    its owner (zero.remoteplay.RemotePlay) holds its lock around every call."""

    def __init__(self, inflight=1.0):
        self.f = float(inflight)
        self.units = None                          # units per iteration (None: caps = hi)
        self.lpu = None                            # leaves per unit (an EWMA over the phases' finished units)
        self.c = {}                                # name -> {lo, hi, rate, active}

    def consumer(self, name, lo, hi, cautious=False):
        """Consumer `name` with cap bounds [lo, hi]; `cautious` (a remote): a quarter of its guessed
        share until it is measured."""
        c = self.c.setdefault(name, {'rate': None, 'active': True, 'lead': 0.0})
        c['hi'] = max(1, int(hi))
        c['lo'] = max(1, min(int(lo), c['hi']))
        c['cautious'] = bool(cautious)
        return c

    def set_active(self, name, on):
        if name in self.c:
            self.c[name]['active'] = bool(on)

    def rates(self):
        """Each consumer's rate, measured or (not yet) assumed: proportional to its slots, scaled
        by the measured consumers' rate per slot (all unmeasured: the slots themselves)."""
        meas = [(c['rate'], c['hi']) for c in self.c.values() if c['rate']]
        per_slot = sum(r for r, _ in meas) / sum(h for _, h in meas) if meas else 1.0
        return {n: (c['rate'] if c['rate'] is not None else per_slot * c['hi']) for n, c in self.c.items()}

    def horizon(self):
        """P (seconds): the phase the active consumers play together after their head starts (None:
        no units)."""
        r = self.rates()
        act = [n for n, c in self.c.items() if c['active']]
        tot = sum(r[n] for n in act)
        if self.units is None or tot <= 0:
            return None
        return max(0.0, (self.units - sum(r[n] * self.c[n]['lead'] for n in act)) / tot)

    def caps(self):
        r = self.rates()
        p = self.horizon()
        out = {}
        for n, c in self.c.items():
            if not c['active']:
                out[n] = 0
            elif p is None:
                out[n] = c['hi']
            else:
                want = self.f * r[n] * (c['lead'] + p)
                if c['rate'] is None and c.get('cautious'):
                    want *= BOOT
                out[n] = max(c['lo'], min(c['hi'], math.ceil(want)))
        return out

    def provisional(self, name, leaves, run_s):
        """An unmeasured consumer's first rate, from `leaves` in `run_s` seconds of its running time
        so far (its status lines; the leaves per unit known): True if set."""
        c = self.c.get(name)
        if c is None or c['rate'] is not None or not self.lpu or run_s <= 0 or not leaves:
            return False
        c['rate'] = leaves / run_s / self.lpu
        return True

    def observe_phase(self, meas):
        """A phase's measurements {name: {units, run_s, util, lead, leaves}}: first the leaves per
        unit (every consumer's finished units together), then each consumer's rate. -> {name: the
        rate measured or None}."""
        tl = sum(m['leaves'] for m in meas.values() if m.get('units', 0) > 0 and m.get('leaves'))
        tu = sum(m['units'] for m in meas.values() if m.get('units', 0) > 0 and m.get('leaves'))
        if tu > 0 and tl > 0:
            v = tl / tu
            self.lpu = v if self.lpu is None else (1 - EWMA) * self.lpu + EWMA * v
        return {n: self.observe(n, m.get('units', 0), m.get('run_s', 0.0), m.get('util'), lead=m.get('lead'),
                                leaves=m.get('leaves')) for n, m in meas.items()}

    def observe(self, name, units, run_s, util, lead=None, leaves=None):
        """A phase of consumer `name`: `units` finished in `run_s` seconds of its running time (and
        `leaves` searched), its outstanding work `util` of its cap on average, `lead` seconds of that
        before this machine's phase began. The rate is its leaves per second over the leaves per
        unit (known once any consumer finished units: a consumer whose games are longer than a phase
        is measured too - slow, so its cap falls to `lo`), else its units per second. -> the rate
        measured (None: too short a phase, nothing to measure)."""
        c = self.c.get(name)
        if c is None:
            return None
        if lead is not None:
            v = max(0.0, float(lead))
            c['lead'] = v if not c.get('led') else (1 - EWMA) * c['lead'] + EWMA * v
            c['led'] = True
        if run_s < MIN_RUN_S:
            return None
        if leaves is not None and self.lpu:
            r = leaves / run_s / self.lpu
        elif units > 0:
            r = units / run_s
        else:
            return None
        u = max(MIN_UTIL, min(1.0, float(util))) if util is not None else 1.0
        r /= u
        c['rate'] = r if c['rate'] is None else (1 - EWMA) * c['rate'] + EWMA * r
        return r

    def state(self):
        out = {n: {'rate': None if c['rate'] is None else round(c['rate'], 6), 'lead': round(c['lead'], 2),
                   'lo': c['lo'], 'hi': c['hi']} for n, c in self.c.items()}
        if self.lpu:
            out['_leaves_per_unit'] = round(self.lpu, 1)
        return out

    def restore(self, state):
        """Rates and head starts from an earlier run of the same run directory (its last metrics
        row): the caps of its first phase follow them instead of the slots."""
        state = dict(state or {})
        lpu = state.pop('_leaves_per_unit', None)
        if lpu:
            self.lpu = float(lpu)
        for n, st in state.items():
            if isinstance(st, dict) and st.get('rate'):
                c = self.c.setdefault(n, {'rate': None, 'active': True, 'lead': 0.0, 'lo': 1, 'hi': 1})
                c['rate'] = float(st['rate'])
                c['lead'] = float(st.get('lead') or 0.0)
