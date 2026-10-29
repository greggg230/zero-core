"""The shared work queue of remote self-play (docs/az.md 5.3, 11.6): one pull model for a
single-game run's games and a co-training run's matches.

An iteration needs `units` of work in total: a single-game run's `games` (finished games, across
every machine), a co-training run's matches (its `games` events x their matches). Each **consumer**
- the run's own driver here ('local') and every remote worker ('w0', 'w1', ...) - takes work as it
has room under its **in-flight cap**:

    cap = clamp(ceil(inflight x units x rate / total rate), lo, hi)

`rate` = the consumer's measured throughput (units finished per second of its running time, an
EWMA over its phases; one that had less work than its cap counts as if it had kept its cap busy
at the same latency), `total rate` = the sum over the consumers taking work now (a remote backing
off or offline drops out, so the others' shares grow at once), `inflight` = the run setting
`selfplay_remote_inflight` (default 1: each consumer holds about one iteration's worth of its own
throughput, so a game or match it starts finishes within about one iteration), `lo` / `hi` its
bounds (a few games per driver group / its driver's slots). Before a consumer is measured its rate
is assumed proportional to its slots (`hi`). The caps' sum is `inflight x units`.

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
MIN_RUN_S = 0.5                    # a phase shorter than this (running) measures nothing
MIN_UTIL = 0.25                    # a consumer below this share of its cap: measured as if at 25%


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
        self.c = {}                                # name -> {lo, hi, rate, active}

    def consumer(self, name, lo, hi):
        c = self.c.setdefault(name, {'rate': None, 'active': True})
        c['hi'] = max(1, int(hi))
        c['lo'] = max(1, min(int(lo), c['hi']))
        return c

    def set_active(self, name, on):
        if name in self.c:
            self.c[name]['active'] = bool(on)

    def rates(self):
        """Each consumer's rate, measured or (not yet) assumed: proportional to its slots, scaled
        by the measured consumers' rate per slot (all unmeasured: the slots themselves)."""
        meas = [(c['rate'], c['hi']) for c in self.c.values() if c['rate']]
        per_slot = sum(r for r, _ in meas) / sum(h for _, h in meas) if meas else 1.0
        return {n: (c['rate'] if c['rate'] else per_slot * c['hi']) for n, c in self.c.items()}

    def caps(self):
        r = self.rates()
        act = [n for n, c in self.c.items() if c['active']]
        tot = sum(r[n] for n in act)
        out = {}
        for n, c in self.c.items():
            if not c['active']:
                out[n] = 0
            elif self.units is None or tot <= 0:
                out[n] = c['hi']
            else:
                want = math.ceil(self.f * self.units * r[n] / tot)
                out[n] = max(c['lo'], min(c['hi'], want))
        return out

    def observe(self, name, units, run_s, util):
        """A phase of consumer `name`: `units` finished in `run_s` seconds of its running time, its
        outstanding work `util` of its cap on average. -> the rate measured (None: too short)."""
        c = self.c.get(name)
        if c is None or run_s < MIN_RUN_S or units <= 0:
            return None
        u = max(MIN_UTIL, min(1.0, float(util))) if util is not None else 1.0
        r = units / run_s / u
        c['rate'] = r if not c['rate'] else (1 - EWMA) * c['rate'] + EWMA * r
        return r

    def state(self):
        return {n: {'rate': None if c['rate'] is None else round(c['rate'], 6), 'lo': c['lo'], 'hi': c['hi']}
                for n, c in self.c.items()}

    def restore(self, state):
        """Rates from an earlier run of the same run directory (its last metrics row): the caps of
        its first phase follow them instead of the slots."""
        for n, st in (state or {}).items():
            if isinstance(st, dict) and st.get('rate'):
                self.c.setdefault(n, {'rate': None, 'active': True, 'lo': 1, 'hi': 1})['rate'] = float(st['rate'])
