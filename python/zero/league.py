"""The self-play league (docs/az.md 14): a share of a run's self-play games puts a PAST checkpoint
of the run in one seat, so that a run (MTG Zero's Delver mirror) does not settle into what its
current network happens to do - fictitious self-play / AlphaStar's league, in its simplest form.

Run settings (zero.run.DEFAULTS; all off by default - a run without them plays exactly as before):

* `selfplay_league` (0.0): the share of self-play games that are league games (0 = off; <= 1).
* `selfplay_league_pool` ('2,4,8,16,32'): which past checkpoints may play - raw iteration offsets
  back from the network that self-plays (iteration i self-plays with checkpoint i-1, so offset 2 is
  checkpoint i-3), those that exist; 'all' = every earlier checkpoint (fictitious self-play).
* `selfplay_league_slots` (1): past checkpoints per iteration, each its own evaluator key ('L0',
  'L1', ...; one more network batch per group and key); picked from the pool every iteration.
* `selfplay_league_pick` ('uniform'): how the slots are picked from the pool - 'uniform', or 'pfsp'
  (prioritized fictitious self-play, AlphaStar's f_hard: weight (1 - x)^2 where x is the current
  network's recent score against that checkpoint, from the run's metrics; unmeasured 0.5).

**The seats.** League game t of every `cycle` games plays slot `t % slots` in seat `(t // slots) %
players`, the other seats the current network; the games are spread evenly (game g is a league
game when floor((g + 1) f) > floor(g f)), so a driver's started games hold exactly the share over
each cycle. A rolling driver's cyclic seat table is that cycle (`games[gid % len]`, the Rust
driver's), persistent across iterations: the keys stay, their weights follow the iteration's picks.
Both seats search with the run's self-play settings (noise, temperature, playout cap: the past
network plays as in its own self-play), but the past network's seat has `record` false: **only the
current network's decisions become samples** (`z` = the game's result from its seat, `q` its own
search's root value: the driver's records are per decider). The past network's moves are not the
current policy's - its visit targets would pull the policy back to its old habits and its values
are an older network's - so they are not trained on (AlphaStar trains the main agents only).

**Until a past checkpoint exists** (a new run's first iterations, or a pool reaching back further
than the run) a slot holds the current network: those games are mirror games recorded on one side
(`self` in the metrics).

Every past network is loaded through `zero.net.load` with the run's game: the encoding guard
(docs/az.md 10.4) refuses one of another encoding.
"""
import math
import os
import threading
from fractions import Fraction

import numpy as np

PICKS = ('uniform', 'pfsp')
DEFAULT_POOL = '2,4,8,16,32'
PRIOR_GAMES = 4.0                  # pfsp: the prior's weight (games at score 0.5)
DECAY = 0.7                        # pfsp: a measurement's weight per iteration of age
FLOOR = 1e-3                       # pfsp: the least weight (every candidate can be picked)
MAX_DEN = 1000                     # the share as a fraction of at most this denominator


def enabled(cfg):
    return float((cfg or {}).get('selfplay_league') or 0) > 0


def parse_pool(s):
    """The setting selfplay_league_pool -> 'all' or a tuple of distinct positive offsets; else
    ValueError."""
    if s is None or (isinstance(s, str) and s.strip() == ''):
        s = DEFAULT_POOL
    if isinstance(s, str) and s.strip().lower() == 'all':
        return 'all'
    items = s.split(',') if isinstance(s, str) else list(s)
    try:
        out = tuple(int(str(x).strip()) for x in items if str(x).strip())
    except ValueError:
        raise ValueError(f'selfplay_league_pool {s!r}: offsets ("2,4,8,16,32") or "all"') from None
    if not out or any(x < 1 for x in out) or len(set(out)) != len(out):
        raise ValueError(f'selfplay_league_pool {s!r}: distinct positive offsets ("2,4,8,16,32") or "all"')
    return out


def check_settings(cfg, where=''):
    """The league settings of a run config -> None, else SystemExit (`where`: a prefix)."""
    f = cfg.get('selfplay_league') or 0
    try:
        f = float(f)
    except (TypeError, ValueError):
        raise SystemExit(f'{where}selfplay_league {f!r}: a share in [0, 1]') from None
    if not (0.0 <= f <= 1.0) or math.isnan(f):
        raise SystemExit(f'{where}selfplay_league {f!r}: a share in [0, 1]')
    if f == 0:
        return
    try:
        parse_pool(cfg.get('selfplay_league_pool', DEFAULT_POOL))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    k = cfg.get('selfplay_league_slots', 1)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise SystemExit(f'{where}selfplay_league_slots {k!r}: a positive integer')
    if cfg.get('selfplay_league_pick', 'uniform') not in PICKS:
        raise SystemExit(f'{where}selfplay_league_pick {cfg.get("selfplay_league_pick")!r}: one of {", ".join(PICKS)}')
    if cfg.get('resolver'):
        raise SystemExit(f'{where}selfplay_league: single-game runs only (not with a resolver: co-training)')


def share(cfg):
    """The league share as a Fraction (a/b, b <= MAX_DEN)."""
    return Fraction(float(cfg.get('selfplay_league') or 0)).limit_denominator(MAX_DEN)


def kinds(cfg, n, counts):
    """Game g of `n` (its player count `counts[g]`): None (a mirror game) or (slot, seat of the past
    network). Exactly floor(n f) league games, spread evenly; slots in turn, seats in turn per slot."""
    f = share(cfg)
    k = int(cfg.get('selfplay_league_slots', 1))
    out, m = [], 0
    for g in range(n):
        if (g + 1) * f.numerator // f.denominator > g * f.numerator // f.denominator:
            out.append((m % k, (m // k) % int(counts[g])))
            m += 1
        else:
            out.append(None)
    return out


def cycle(cfg):
    """A rolling driver's cyclic table length: every slot and seat its share exactly once a cycle."""
    f = share(cfg)
    P = len(cfg['players'])
    seats = math.lcm(*[int(p) for p in cfg['players']])
    return f.denominator * int(cfg.get('selfplay_league_slots', 1)) * seats * P


def key(t):
    return f'L{t}'


def seat_of(seat, t):
    """The league seat of slot t: the run's self-play seat on key L<t> that records nothing."""
    return dict(seat, key=key(t), record=False)


def table(cfg, seat, counts):
    """The seat table of games with these player counts (one per game) and their kinds."""
    ks = kinds(cfg, len(counts), counts)
    games = []
    for p, kd in zip(counts, ks):
        row = [seat] * int(p)
        if kd is not None:
            row[kd[1]] = seat_of(seat, kd[0])
        games.append(row)
    return games, ks


def rolling_table(cfg, seat):
    """A rolling driver's cyclic seat table (and the kinds of its entries: game gid is entry gid %
    len); the player counts in turn, as the table without the league."""
    P = cfg['players']
    n = cycle(cfg)
    return table(cfg, seat, [P[t % len(P)] for t in range(n)])


def ckpt(i):
    return f'iter_{i:04d}'


class League:
    """A run's league (module docstring): the iterations' picks (`plan`, cached; thread-safe), the
    evaluators of the slots, the games' attribution and the metrics."""

    def __init__(self, R):
        c = R.cfg
        self.R = R
        self.f = float(c['selfplay_league'])
        self.pool = parse_pool(c.get('selfplay_league_pool', DEFAULT_POOL))
        self.k = int(c.get('selfplay_league_slots', 1))
        self.pick = c.get('selfplay_league_pick', 'uniform')
        self.seed = int(c.get('seed', 0))
        self.plans = {}
        self.mu = threading.Lock()

    def keys(self):
        return [key(t) for t in range(self.k)]

    def candidates(self, it):
        """The past checkpoints that may play iteration `it` (checkpoint it - 1 self-plays): newest
        first, those that exist."""
        from .run import model_path
        cur = it - 1
        js = range(cur - 1, -1, -1) if self.pool == 'all' else [cur - off for off in self.pool]
        out = []
        for j in js:
            if 0 <= j < cur and j not in out and os.path.exists(model_path(self.R.run, j)):
                out.append(j)
        return out

    def history(self, upto):
        """pfsp: {checkpoint: (score, games)} of the current networks against each, from the metrics
        rows of iterations <= upto, each weighted DECAY ** age."""
        from .run import read_metrics
        acc = {}
        try:
            rows = read_metrics(self.R.run)
        except (OSError, ValueError):
            rows = []
        for m in rows:
            if m.get('iter', upto + 1) > upto:
                continue
            lg = (m.get('selfplay') or {}).get('league') or {}
            w = DECAY ** (upto - m['iter'])
            for name, v in (lg.get('pool') or {}).items():
                s, g = acc.get(name, (0.0, 0.0))
                acc[name] = (s + w * v.get('score', 0.0), g + w * v.get('games', 0))
        return acc

    def weights(self, cands, it):
        if self.pick == 'uniform':
            return np.ones(len(cands))
        # (rows <= it - 2: the same plan whether it is made while checkpoint it - 1 is evaluated -
        # the remote's early phase - or when this machine's phase begins, and after a restart)
        h = self.history(it - 2)
        w = []
        for j in cands:
            s, g = h.get(ckpt(j), (0.0, 0.0))
            x = (s + 0.5 * PRIOR_GAMES) / (g + PRIOR_GAMES)
            w.append(max(FLOOR, (1.0 - x) ** 2))
        return np.asarray(w, dtype=np.float64)

    def plan(self, it):
        """Iteration `it`'s picks -> {'slots': [checkpoint per slot], 'candidates': [...], 'self':
        True when no past checkpoint exists (the slots hold it - 1), 'weights': pfsp's}. Made once per
        iteration (the same for this machine and the remote's workers: shipped with the phase)."""
        with self.mu:
            p = self.plans.get(it)
            if p is not None:
                return p
            cands = self.candidates(it)
            p = {'candidates': cands}
            if not cands:
                p['slots'] = [it - 1] * self.k
                p['self'] = True
            elif len(cands) <= self.k:
                p['slots'] = [cands[t % len(cands)] for t in range(self.k)]
            else:
                w = self.weights(cands, it)
                rng = np.random.default_rng([self.seed, it, 0x1EA9])
                pick = rng.choice(len(cands), size=self.k, replace=False, p=w / w.sum())
                p['slots'] = [cands[int(i)] for i in pick]
                if self.pick != 'uniform':
                    p['weights'] = {ckpt(j): round(float(x), 4) for j, x in zip(cands, w / w.sum())}
            self.plans = {i: q for i, q in self.plans.items() if i >= it - 2}
            self.plans[it] = p
            return p

    def files(self, it):
        """{key: the checkpoint's path relative to the run} of iteration `it`'s slots."""
        return {key(t): f'models/{ckpt(j)}.pt' for t, j in enumerate(self.plan(it)['slots'])}

    def evaluators(self, it, evs, root=None):
        """Point the slots' evaluators in `evs` (a dict, updated in place: kept across iterations by
        a rolling driver) at iteration `it`'s picks: the weights loaded into the existing evaluator
        when the network has the same config, else a new one. Each checkpoint is checked by the
        encoding guard (zero.net.load with the run's game). `root`: the directory holding models/
        (default the run's; a remote worker's mirror)."""
        c = self.R.cfg
        for k, rel in self.files(it).items():
            path = os.path.join(root or self.R.run, *rel.split('/'))
            evs[k] = load_into(evs.get(k), path, c)
        return evs

    @staticmethod
    def attribute(rows, ks, slots):
        """Annotate finished-game rows of a driver whose table has the kinds `ks` (gid % len) with
        their league game: row['league'] = [checkpoint, the current network's score] (2 players:
        its result; more: 1 - the past seat's), slots = the checkpoints in the slots now."""
        for r in rows:
            kd = ks[r['gid'] % len(ks)]
            if kd is not None:
                r['league'] = [ckpt(slots[kd[0]]), 1.0 - float(r['result'][kd[1]])]
        return rows


def load_into(ev, path, c):
    """A slot's evaluator holding the network at `path` (guarded): `ev`'s weights replaced when the
    network's config is the same, else a new NetEvaluator."""
    from .infer import NetEvaluator
    from .net import check_belief_head, load
    m, _ = load(path, 'cpu', game=c['game'])
    from .mcts import belief_tilts
    if float(c.get('selfplay_belief_tilt', 0.0) or 0.0) > 0 or belief_tilts(c, 'selfplay'):
        # a league opponent is the self-play seat: it takes the root's belief too, so its checkpoint
        # needs the head (HeadMissing: narrow selfplay_league_pool to checkpoints that have one)
        check_belief_head(m, c['game'], f'selfplay_belief_tilt (the league opponent {path}: a selfplay_league_pool '
                                        f'reaching before the auxiliary head was trained)')
    if ev is not None and getattr(ev.model, 'config', None) == m.config:
        ev.set_weights(m.state_dict())
        return ev
    return NetEvaluator(m, c['infer_device'], workers=c.get('infer_workers', 1))


def tally(rows, into=None):
    """{checkpoint: [score, games]} of the annotated rows (`attribute`), added to `into`."""
    into = {} if into is None else into
    for r in rows:
        lg = r.get('league')
        if lg:
            v = into.setdefault(lg[0], [0.0, 0])
            v[0] += lg[1]
            v[1] += 1
    return into


def merge(a, b):
    out = {k: list(v) for k, v in (a or {}).items()}
    for k, v in (b or {}).items():
        x = out.setdefault(k, [0.0, 0])
        x[0] += v[0]
        x[1] += v[1]
    return out


def metrics(plan, local, remote, games, pick):
    """An iteration's `selfplay.league`: the share of its games that were league games, the current
    network's score against them (and per checkpoint), the slots' checkpoints, the candidates."""
    allp = merge(local, remote)
    n = sum(v[1] for v in allp.values())
    s = sum(v[0] for v in allp.values())
    w = s / n if n else None
    out = {'share': round(n / games, 4) if games else 0.0, 'games': n, 'score': round(s, 2),
           'win': None if w is None else round(w, 4),
           'se': None if w is None else round(math.sqrt(max(w * (1 - w), 1e-9) / n), 4),
           'slots': {key(t): ckpt(j) for t, j in enumerate(plan['slots'])},
           'candidates': [ckpt(j) for j in plan['candidates']], 'pick': pick,
           'pool': {k: {'games': v[1], 'score': round(v[0], 2), 'win': round(v[0] / v[1], 4)}
                    for k, v in sorted(allp.items()) if v[1]},
           'local': sum(v[1] for v in (local or {}).values()), 'remote': sum(v[1] for v in (remote or {}).values())}
    if plan.get('self'):
        out['self'] = True
    if plan.get('weights'):
        out['weights'] = plan['weights']
    return out


def fmt(lg):
    """One log line of `metrics`."""
    vs = ', '.join(f'{k} {v["games"]} games {v["win"]:.3f}' for k, v in lg['pool'].items())
    s = (f"{lg['games']} league games ({100 * lg['share']:.1f}%), slots {', '.join(lg['slots'].values())}"
         + (' (no past checkpoint yet: the current network, recorded on one side)' if lg.get('self') else '')
         + f" of {len(lg['candidates'])} candidates ({lg['pick']})")
    if lg['win'] is not None:
        s += f"; the current network scores {lg['win']:.3f} +- {lg['se']:.3f} against them ({vs})"
    return s


def fork_needs(cfg, it):
    """fork_run: the checkpoints <= it a continuation at `it` with these settings may play as league
    opponents (every one back to the pool's furthest offset; 'all': every one)."""
    if not enabled(cfg):
        return set()
    pool = parse_pool(cfg.get('selfplay_league_pool', DEFAULT_POOL))
    lo = 0 if pool == 'all' else it - max(pool)
    return set(range(max(0, lo), it + 1))
