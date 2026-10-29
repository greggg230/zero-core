"""The shared work queue of remote self-play (docs/az.md 5.3, 11.6): one pull model for a
single-game run's games and a co-training run's matches.

An iteration needs `units` of work in total: a single-game run's `games` (finished games, across
every machine), a co-training run's matches (its `games` events x their matches). Each **consumer**
- the run's own driver here ('local') and every remote worker ('w0', 'w1', ...) - takes work as it
has room under its **in-flight cap**. The cap targets the consumer's **game latency**: a game it
starts should finish within about one iteration of its running time.

* **Measured per phase**: its throughput `rate` (units per second of its running time: its leaves
  per second over the leaves per finished unit, so a consumer whose games outlast a phase is
  measured too), its mean in-flight `n`, and so its latency `L = n / rate` (Little's law); its
  horizon `H = lead + P` (`lead`: its head start - a remote's phase begins as soon as the
  checkpoint is written, before this machine's: an EWMA of the seconds it played before; this
  machine's 0; co-training 0 - and `P = (units - sum rate x lead) / total rate`, the phase they all
  play together).
* **The target**: `inflight x H` (the run setting `selfplay_remote_inflight`, default 1), or `SLACK`
  (1.5) times that for a consumer whose latency hardly depends on its in-flight (`sens`, the
  elasticity of L to n, below 0.5: a GPU's batches are per-call bound, so fewer games in flight make
  each game little faster and cost most of its throughput - MTG Zero's delver1 desktop: 82 games in
  flight at 3.5k leaves/s, 192 at 6.5k, its games ~234 and ~282 s long).
* **The step**: after each phase `cap <- cap x (target / L) ** (1 / max(sens, 0.25))`, at most x2 or
  x0.5, within `[lo, hi]` (a few games per driver group; its driver's slots): a throughput-bound
  consumer (sens ~1: CPU workers, whose latency grows with their load) converges to `L = target`; a
  latency-bound one grows to its slots while its games finish within the target and falls to `lo`
  when they cannot (a worker slower than a phase). `sens` starts at the consumer's prior (0.25 for
  CUDA / MPS inference, 1 for the CPU) and follows its measurements (an EWMA over phases whose
  in-flight differed by 1.3x or more).
* **Before its first measurement** (a new run, a new worker): its throughput share of the units,
  `inflight x rate x H`, its rate assumed proportional to its slots (`hi`) and a remote's cap a
  quarter of that (`BOOT`) until a first measurement (its phase's end, or a few seconds of its
  status lines, zero.remoteplay). A restart takes the model's state from the last metrics row.
* **Who takes work now**: a remote backing off or offline is no consumer (cap 0), and each other
  consumer's cap is at least its throughput share of the work for who takes work now (`inflight x
  rate x (lead + P)` over the active consumers: this machine alone takes the whole quota, up to its
  slots) - at once; a phase spent mostly so (`TAKEOVER`) does not move the latency model.

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
SLACK = 1.5                        # a latency-bound consumer's target: its games within 1.5 x the horizon
LATENCY_BOUND = 0.5                # sens below this: latency-bound
SENS_SPREAD = 1.3                  # in-flight ratio between two phases that measures the sensitivity
SENS_GPU, SENS_CPU = 0.25, 1.0     # the sensitivity's priors (CUDA / MPS inference; the CPU)
STEP = 2.0                         # a cap moves at most x2 or x0.5 per phase
TAKEOVER = 0.2                     # a phase with a consumer out this share of its time: not a measure of the others' latency


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
    """The consumers' latency-targeted caps (module docstring). Not thread-safe: its owner
    (zero.remoteplay.RemotePlay) holds its lock around every call."""

    def __init__(self, inflight=1.0):
        self.f = float(inflight)
        self.units = None                          # units per iteration (None: caps = hi)
        self.lpu = None                            # leaves per unit (an EWMA over the phases' finished units)
        self.c = {}                                # name -> {lo, hi, rate, lead, cap (the model's), sens, ...}

    def consumer(self, name, lo, hi, cautious=False, gpu=False):
        """Consumer `name` with cap bounds [lo, hi]; `cautious` (a remote): a quarter of its guessed
        share until it is measured; `gpu`: CUDA / MPS inference (its latency's prior: hardly
        load-dependent)."""
        c = self.c.setdefault(name, {'rate': None, 'active': True, 'lead': 0.0, 'cap': None, 'sens': None,
                                     'hist': []})
        c['hi'] = max(1, int(hi))
        c['lo'] = max(1, min(int(lo), c['hi']))
        c['cautious'] = bool(cautious)
        c['sens0'] = SENS_GPU if gpu else SENS_CPU
        return c

    def set_active(self, name, on):
        if name in self.c:
            self.c[name]['active'] = bool(on)

    def sens(self, name):
        c = self.c[name]
        return c['sens'] if c['sens'] is not None else c['sens0']

    def rates(self):
        """Each consumer's rate, measured or (not yet) assumed: proportional to its slots, scaled
        by the measured consumers' rate per slot (all unmeasured: the slots themselves)."""
        meas = [(c['rate'], c['hi']) for c in self.c.values() if c['rate']]
        per_slot = sum(r for r, _ in meas) / sum(h for _, h in meas) if meas else 1.0
        return {n: (c['rate'] if c['rate'] is not None else per_slot * c['hi']) for n, c in self.c.items()}

    def horizon(self, everyone=False):
        """P (seconds): the phase the consumers taking work now (`everyone`: all of them) play
        together after their head starts (None: no units)."""
        r = self.rates()
        act = [n for n, c in self.c.items() if c['active'] or everyone]
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
                share = self.f * r[n] * (c['lead'] + p)    # its throughput share of the work, for who works now
                if c['cap'] is None:                       # not measured yet: that share
                    want = share * (BOOT if c['rate'] is None and c.get('cautious') else 1.0)
                else:                                      # the latency model's cap, the share at least
                    want = max(c['cap'], share)
                out[n] = max(c['lo'], min(c['hi'], int(math.ceil(want - 1e-6))))
        return out

    def provisional(self, name, leaves, run_s):
        """An unmeasured consumer's first rate, from `leaves` in `run_s` seconds of its running time
        so far (its status lines; the leaves per unit known): True if set."""
        c = self.c.get(name)
        if c is None or c['rate'] is not None or not self.lpu or run_s <= 0 or not leaves:
            return False
        c['rate'] = leaves / run_s / self.lpu
        return True

    def observe_phase(self, meas, phase_s=None, adjust=True):
        """A phase's measurements {name: {units, run_s, util, lead, leaves, cap}} (`cap`: its mean
        cap in the phase, `util` its outstanding work's mean share of it) and `phase_s`, how long the
        joint phase actually ran (this machine's phase; None: the model's P): the leaves per unit
        (every consumer's finished units together), each consumer's rate, then each one's latency
        against its target - inflight x (its head start + that phase) - and its next cap; with
        `adjust` False (a phase spent mostly with a consumer out: the others played its share too)
        the latency model keeps its caps. -> {name: {rate, inflight, latency, target, sens, step}
        measured, or None}."""
        tl = sum(m['leaves'] for m in meas.values() if m.get('units', 0) > 0 and m.get('leaves'))
        tu = sum(m['units'] for m in meas.values() if m.get('units', 0) > 0 and m.get('leaves'))
        if tu > 0 and tl > 0:
            v = tl / tu
            self.lpu = v if self.lpu is None else (1 - EWMA) * self.lpu + EWMA * v
        raw = {n: self.observe(n, m.get('units', 0), m.get('run_s', 0.0), m.get('util'), lead=m.get('lead'),
                               leaves=m.get('leaves')) for n, m in meas.items()}
        pall = float(phase_s) if phase_s is not None else self.horizon(everyone=True)
        out = {}
        for n, m in meas.items():
            c = self.c.get(n)
            r = raw.get(n)
            capm = m.get('cap')
            if c is None or r is None or pall is None or not capm or not adjust:
                out[n] = None if r is None else {'rate': round(r, 4)}
                continue
            u = min(1.0, float(m['util'])) if m.get('util') is not None else 1.0
            inflight = max(1e-9, u * capm)
            lat = inflight / r if r > 0 else float('inf')
            self._sens_update(c, inflight, lat)
            sens = self.sens(n)
            lead = max(0.0, float(m['lead'])) if m.get('lead') is not None else c['lead']
            target = max(MIN_RUN_S, self.f * (lead + pall)) * (SLACK if sens < LATENCY_BOUND else 1.0)
            q = target / lat if lat > 0 else STEP
            step = min(STEP, max(1.0 / STEP, q ** (1.0 / max(sens, 0.25)))) if q > 0 else 1.0 / STEP
            base = c['cap'] if c['cap'] is not None else capm
            c['cap'] = max(c['lo'], min(c['hi'], base * step))
            out[n] = {'rate': round(r, 4), 'inflight': round(inflight, 1), 'latency': round(lat, 2),
                      'target': round(target, 2), 'sens': round(sens, 3), 'step': round(step, 3)}
        return out

    def _sens_update(self, c, inflight, lat):
        """The elasticity of the consumer's latency to its in-flight, from this phase and the newest
        earlier one whose in-flight differed by SENS_SPREAD or more (an EWMA from its prior)."""
        if math.isfinite(lat) and lat > 0:
            for n0, l0 in reversed(c['hist']):
                if max(inflight / n0, n0 / inflight) >= SENS_SPREAD:
                    e = math.log(lat / l0) / math.log(inflight / n0)
                    e = max(0.0, min(1.0, e))
                    prev = c['sens'] if c['sens'] is not None else c['sens0']
                    c['sens'] = (1 - EWMA) * prev + EWMA * e
                    break
            c['hist'] = (c['hist'] + [(inflight, lat)])[-4:]

    def observe(self, name, units, run_s, util, lead=None, leaves=None):
        """A phase of consumer `name`: `units` finished in `run_s` seconds of its running time (and
        `leaves` searched), its outstanding work `util` of its cap on average, `lead` seconds of that
        before this machine's phase began. Its throughput is its leaves per second over the leaves
        per unit (known once any consumer finished units: a consumer whose games are longer than a
        phase is measured too), else its units per second; its `rate` (the share model's) counts it
        as if it had kept its cap busy. -> the throughput measured (None: too short a phase, nothing
        to measure)."""
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
        c['rate'] = r / u if c['rate'] is None else (1 - EWMA) * c['rate'] + EWMA * r / u
        return r

    def state(self):
        out = {n: {'rate': None if c['rate'] is None else round(c['rate'], 6), 'lead': round(c['lead'], 2),
                   'cap': None if c['cap'] is None else round(c['cap'], 2),
                   'sens': None if c['sens'] is None else round(c['sens'], 4), 'lo': c['lo'], 'hi': c['hi']}
               for n, c in self.c.items()}
        if self.lpu:
            out['_leaves_per_unit'] = round(self.lpu, 1)
        return out

    def restore(self, state):
        """The model's state from an earlier run of the same run directory (its last metrics row):
        the caps of its first phase follow it instead of the slots."""
        state = dict(state or {})
        lpu = state.pop('_leaves_per_unit', None)
        if lpu:
            self.lpu = float(lpu)
        for n, st in state.items():
            if isinstance(st, dict) and st.get('rate'):
                c = self.c.setdefault(n, {'rate': None, 'active': True, 'lead': 0.0, 'cap': None, 'sens': None,
                                          'hist': [], 'lo': 1, 'hi': 1, 'sens0': SENS_CPU})
                c['rate'] = float(st['rate'])
                c['lead'] = float(st.get('lead') or 0.0)
                if st.get('cap'):
                    c['cap'] = float(st['cap'])
                if st.get('sens') is not None:
                    c['sens'] = float(st['sens'])
