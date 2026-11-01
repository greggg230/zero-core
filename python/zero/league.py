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
* `selfplay_league_fixed` (''): FIXED opponents instead of the pool (docs/az.md 14.7) - frozen
  checkpoints ('path' or 'label=path', comma-separated; a label defaults to the file's stem), e.g. a
  best-response (exploitability) run whose learner meets only a frozen copy of another run's network
  (`selfplay_league` 1.0: every self-play game). Each is copied once into the run
  (`league/<label>.pt`, its sha256 recorded in `league/fixed.json`: another file under a recorded
  label is refused) and plays from that copy; the pool's offsets are not read. '' = off.

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

**Fixed opponents** (`selfplay_league_fixed`): the candidates are the fixed opponents (named
`fixed:<label>` in the plans, the metrics and the attribution) - all of them every iteration when
there are no more of them than slots, else picked as the pool's are; the slots' seats and records
exactly as above, so the learner's samples are its own decisions against them. A remote self-play
worker gets the run's copies with its mirror and announces their sha256 in its hello; a worker
without them, or with other files, is refused (zero/remoteplay.py).
"""
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from fractions import Fraction

import numpy as np

from .net import EncodingMismatch

PICKS = ('uniform', 'pfsp')
DEFAULT_POOL = '2,4,8,16,32'
PRIOR_GAMES = 4.0                  # pfsp: the prior's weight (games at score 0.5)
DECAY = 0.7                        # pfsp: a measurement's weight per iteration of age
FLOOR = 1e-3                       # pfsp: the least weight (every candidate can be picked)
MAX_DEN = 1000                     # the share as a fraction of at most this denominator
FIXED_PREFIX = 'fixed:'            # a fixed opponent's name (plans, metrics, attribution): 'fixed:<label>'
FIXED_DIR = 'league'               # the run's copies of its fixed opponents: <run>/league/<label>.pt ...
FIXED_RECORD = 'fixed.json'        # ... and their record: {label: {sha256, bytes, source, since, time}}
LABEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
ITER_RE = re.compile(r'^iter_(\d{4,})$')


class LeagueRefused(EncodingMismatch):
    """A fixed league opponent that cannot be played as recorded (missing, another file under its
    label, a copy that changed, another encoding, no belief head): the run is refused (exit status 78,
    not restarted by the supervisor), as for an Elo anchor (docs/az.md 20.1)."""


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


def parse_fixed(s):
    """The setting selfplay_league_fixed ('path,label=path,...', a list, or '' / None) -> [(label,
    path)] in order ([] = off). A label defaults to the file's stem; labels are distinct, of letters,
    digits, '.', '_' and '-' (the run's copy is league/<label>.pt). ValueError on a malformed one."""
    if s is None or (isinstance(s, str) and s.strip() == ''):
        return []
    if not isinstance(s, (str, list, tuple)):
        raise ValueError(f'selfplay_league_fixed {s!r}: "path" or "label=path" entries, comma-separated')
    items = s.split(',') if isinstance(s, str) else list(s)
    out = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        label, sep, path = x.partition('=')
        if not sep or any(ch in label for ch in ':/' + chr(92)):
            # (no label: 'E:/x/iter_0475.pt' - the file's stem)
            label, path = os.path.splitext(os.path.basename(x.replace(chr(92), '/')))[0], x
        label, path = label.strip(), path.strip()
        if not path or not LABEL_RE.match(label):
            raise ValueError(f'selfplay_league_fixed {s!r}: entries "path" or "label=path", a label of letters, '
                             f'digits, ".", "_" and "-" (got {label!r})')
        out.append((label, path))
    if not out:
        raise ValueError(f'selfplay_league_fixed {s!r}: no checkpoint named')
    labels = [lb for lb, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f'selfplay_league_fixed {s!r}: labels must be distinct ({labels})')
    return out


def fixed_name(label):
    """A fixed opponent's name in the plans, the metrics and the attribution ('fixed:<label>')."""
    return FIXED_PREFIX + label


def is_fixed(j):
    return isinstance(j, str) and j.startswith(FIXED_PREFIX)


def name_of(j):
    """A slot's checkpoint -> its name: 'iter_NNNN' for a past checkpoint j of the run, a fixed
    opponent's name as it is."""
    return j if isinstance(j, str) else ckpt(int(j))


def rel_of(j):
    """A slot's checkpoint -> its file, relative to the run (or to a remote worker's mirror)."""
    if is_fixed(j):
        return f'{FIXED_DIR}/{j[len(FIXED_PREFIX):]}.pt'
    return f'models/{ckpt(int(j))}.pt'


def slot_of(rel):
    """`rel_of`'s inverse (a remote worker reads its phase's slots so): models/iter_j.pt -> j,
    league/<label>.pt -> 'fixed:<label>'; anything else ValueError."""
    d, _, base = str(rel).replace(chr(92), '/').rpartition('/')
    stem, ext = os.path.splitext(base)
    if ext == '.pt' and d == FIXED_DIR and LABEL_RE.match(stem):
        return fixed_name(stem)
    m = ITER_RE.match(stem)
    if ext == '.pt' and d == 'models' and m:
        return int(m.group(1))
    raise ValueError(f'a league slot\'s checkpoint {rel!r}: models/iter_NNNN.pt or {FIXED_DIR}/<label>.pt')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fixed_path(root, label):
    """The copy of the fixed opponent `label` in a run (or a remote worker's mirror) `root`."""
    return os.path.join(root, FIXED_DIR, f'{label}.pt')


def fixed_shas(root, cfg):
    """{label: sha256 of <root>/league/<label>.pt, or None when it is missing} of the setting's fixed
    opponents ({} when off): what a remote worker announces in its hello (its mirror's copies)."""
    return {label: (sha256_file(fixed_path(root, label)) if os.path.exists(fixed_path(root, label)) else None)
            for label, _ in parse_fixed(cfg.get('selfplay_league_fixed'))}


def check_settings(cfg, where=''):
    """The league settings of a run config -> None, else SystemExit (`where`: a prefix)."""
    f = cfg.get('selfplay_league') or 0
    try:
        f = float(f)
    except (TypeError, ValueError):
        raise SystemExit(f'{where}selfplay_league {f!r}: a share in [0, 1]') from None
    if not (0.0 <= f <= 1.0) or math.isnan(f):
        raise SystemExit(f'{where}selfplay_league {f!r}: a share in [0, 1]')
    fx = cfg.get('selfplay_league_fixed') or ''
    if fx and f == 0:
        raise SystemExit(f'{where}selfplay_league_fixed {fx!r} with selfplay_league 0: selfplay_league is the share '
                         f'of the self-play games played against it (1.0: every game)')
    if f == 0:
        return
    try:
        parse_pool(cfg.get('selfplay_league_pool', DEFAULT_POOL))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    k = cfg.get('selfplay_league_slots', 1)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise SystemExit(f'{where}selfplay_league_slots {k!r}: a positive integer')
    if fx:
        try:
            fixed = parse_fixed(fx)
        except ValueError as e:
            raise SystemExit(f'{where}{e}') from None
        if parse_pool(cfg.get('selfplay_league_pool', DEFAULT_POOL)) != parse_pool(DEFAULT_POOL):
            raise SystemExit(f'{where}selfplay_league_pool {cfg.get("selfplay_league_pool")!r} with '
                             f'selfplay_league_fixed: the pool is the fixed opponents (its offsets are not read: '
                             f'leave it at its default {DEFAULT_POOL!r})')
        if k > len(fixed):
            raise SystemExit(f'{where}selfplay_league_slots {k} with {len(fixed)} fixed opponent(s) '
                             f'(selfplay_league_fixed {fx!r}): one slot each at most')
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
        # the fixed opponents (selfplay_league_fixed; [] = the pool's past checkpoints) and, once
        # `prepare` has copied / checked them, their sha256 by label
        self.fixed = parse_fixed(c.get('selfplay_league_fixed'))
        self.fixed_sha = {}

    def keys(self):
        return [key(t) for t in range(self.k)]

    def prepare(self, it=None):
        """The fixed opponents (selfplay_league_fixed) before the run plays: each copied into
        <run>/league/<label>.pt the first time (atomic; its sha256, size, source and the iteration it
        starts at recorded in <run>/league/fixed.json), on a restart the copy checked against the
        record - and the setting's file, when it still exists, too: another file under a recorded
        label is refused (give it another label) -, then every copy through the encoding guard (and
        the belief head when selfplay_belief_tilt is on). -> {label: sha256} ({} without fixed
        opponents); LeagueRefused / EncodingMismatch: the run is refused (exit 78). `it`: the
        iteration the run resumes at (the record's `since`)."""
        if not self.fixed:
            return {}
        c = self.R.cfg
        d = os.path.join(self.R.run, FIXED_DIR)
        rp = os.path.join(d, FIXED_RECORD)
        rec = {}
        if os.path.exists(rp):
            with open(rp, encoding='utf-8') as f:
                rec = json.load(f)
        for label, src in self.fixed:
            dst = fixed_path(self.R.run, label)
            known = rec.get(label)
            src_sha = sha256_file(src) if os.path.exists(src) else None
            if known is None:
                if src_sha is None:
                    raise LeagueRefused(f'selfplay_league_fixed: the fixed opponent {label!r} ({src}) does not exist')
                self._check_net(src, label)
                os.makedirs(d, exist_ok=True)
                tmp = dst + '.tmp'
                shutil.copyfile(src, tmp)
                if sha256_file(tmp) != src_sha:
                    os.remove(tmp)
                    raise LeagueRefused(f'selfplay_league_fixed: the copy of {label!r} ({src}) differs from its '
                                        f'source (the file changed while it was copied)')
                os.replace(tmp, dst)
                rec[label] = {'sha256': src_sha, 'bytes': os.path.getsize(dst), 'source': os.path.abspath(src),
                              'since': it, 'time': time.time()}
                with open(rp + '.tmp', 'w', encoding='utf-8') as f:
                    json.dump(rec, f, indent=1)
                os.replace(rp + '.tmp', rp)
                self.R.log(f'league: fixed opponent {label} ({src}, sha256 {src_sha[:12]}) copied to '
                           f'{FIXED_DIR}/{label}.pt')
            else:
                want = known.get('sha256')
                if src_sha is not None and src_sha != want:
                    raise LeagueRefused(f'selfplay_league_fixed: {src} (sha256 {src_sha[:12]}) is not the fixed '
                                        f'opponent {label!r} this run plays ({want[:12]}, from {known.get("source")}): '
                                        f'another file under the same label - give it another label')
                if not os.path.exists(dst):
                    if src_sha is None:
                        raise LeagueRefused(f'selfplay_league_fixed: the run\'s copy of {label!r} ({dst}) and its '
                                            f'source ({src}) are both missing')
                    shutil.copyfile(src, dst + '.tmp')
                    os.replace(dst + '.tmp', dst)
                    self.R.log(f'league: fixed opponent {label} copied again from {src} (the run\'s copy was missing)')
                got = sha256_file(dst)
                if got != want:
                    raise LeagueRefused(f'selfplay_league_fixed: the run\'s copy of {label!r} ({dst}, sha256 '
                                        f'{got[:12]}) is not the file it recorded ({want[:12]})')
            self._check_net(dst, label)
        self.fixed_sha = {label: rec[label]['sha256'] for label, _ in self.fixed}
        self.R.log('league: fixed opponents ' + ', '.join(f'{fixed_name(lb)} = {FIXED_DIR}/{lb}.pt (sha256 '
                                                          f'{sha[:12]})' for lb, sha in self.fixed_sha.items())
                   + f'; {100 * self.f:g}% of the self-play games against them, the pool not used')
        return self.fixed_sha

    def _check_net(self, path, label):
        """A fixed opponent's checkpoint: this build's encoding (EncodingMismatch) and, with the
        self-play seat's root belief on, the auxiliary head (net.HeadMissing) - as `load_into` checks
        every slot's network when it is loaded."""
        from .net import check_belief_head, check_checkpoint, load
        c = self.R.cfg
        check_checkpoint(path, c['game'])
        if float(c.get('selfplay_belief_tilt', 0.0) or 0.0) > 0:
            m, _ = load(path, 'cpu', game=c['game'])
            check_belief_head(m, c['game'], f'selfplay_belief_tilt (the fixed league opponent {label!r}, {path})')

    def candidates(self, it):
        """The past checkpoints that may play iteration `it` (checkpoint it - 1 self-plays): newest
        first, those that exist. With fixed opponents: their names ('fixed:<label>'), in order."""
        if self.fixed:
            return [fixed_name(lb) for lb, _ in self.fixed]
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
            s, g = h.get(name_of(j), (0.0, 0.0))
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
                    p['weights'] = {name_of(j): round(float(x), 4) for j, x in zip(cands, w / w.sum())}
            self.plans = {i: q for i, q in self.plans.items() if i >= it - 2}
            self.plans[it] = p
            return p

    def files(self, it):
        """{key: the checkpoint's path relative to the run} of iteration `it`'s slots."""
        return {key(t): rel_of(j) for t, j in enumerate(self.plan(it)['slots'])}

    def evaluators(self, it, evs, root=None):
        """Point the slots' evaluators in `evs` (a dict, updated in place: kept across iterations by
        a rolling driver) at iteration `it`'s picks: the weights loaded into the existing evaluator
        when the network has the same config, else a new one. Each checkpoint is checked by the
        encoding guard (zero.net.load with the run's game). `root`: the directory holding models/
        and league/ (default the run's; a remote worker's mirror)."""
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
                r['league'] = [name_of(slots[kd[0]]), 1.0 - float(r['result'][kd[1]])]
        return rows


def load_into(ev, path, c):
    """A slot's evaluator holding the network at `path` (guarded): `ev`'s weights replaced when the
    network's config is the same, else a new NetEvaluator."""
    from .infer import NetEvaluator
    from .net import check_belief_head, load
    m, _ = load(path, 'cpu', game=c['game'])
    if float(c.get('selfplay_belief_tilt', 0.0) or 0.0) > 0:
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


def metrics(plan, local, remote, games, pick, fixed=None):
    """An iteration's `selfplay.league`: the share of its games that were league games, the current
    network's score against them (and per checkpoint), the slots' checkpoints, the candidates; with
    fixed opponents (`fixed`: {label: sha256}) their sha256 (16 hex digits) as `fixed`."""
    allp = merge(local, remote)
    n = sum(v[1] for v in allp.values())
    s = sum(v[0] for v in allp.values())
    w = s / n if n else None
    out = {'share': round(n / games, 4) if games else 0.0, 'games': n, 'score': round(s, 2),
           'win': None if w is None else round(w, 4),
           'se': None if w is None else round(math.sqrt(max(w * (1 - w), 1e-9) / n), 4),
           'slots': {key(t): name_of(j) for t, j in enumerate(plan['slots'])},
           'candidates': [name_of(j) for j in plan['candidates']], 'pick': pick,
           'pool': {k: {'games': v[1], 'score': round(v[0], 2), 'win': round(v[0] / v[1], 4)}
                    for k, v in sorted(allp.items()) if v[1]},
           'local': sum(v[1] for v in (local or {}).values()), 'remote': sum(v[1] for v in (remote or {}).values())}
    if plan.get('self'):
        out['self'] = True
    if plan.get('weights'):
        out['weights'] = plan['weights']
    if fixed:
        out['fixed'] = {fixed_name(lb): sha[:16] for lb, sha in fixed.items()}
    return out


def fmt(lg):
    """One log line of `metrics`."""
    vs = ', '.join(f'{k} {v["games"]} games {v["win"]:.3f}' for k, v in lg['pool'].items())
    s = (f"{lg['games']} league games ({100 * lg['share']:.1f}%), slots {', '.join(lg['slots'].values())}"
         + (' (no past checkpoint yet: the current network, recorded on one side)' if lg.get('self') else '')
         + (' (fixed)' if lg.get('fixed') else f" of {len(lg['candidates'])} candidates ({lg['pick']})"))
    if lg['win'] is not None:
        s += f"; the current network scores {lg['win']:.3f} +- {lg['se']:.3f} against them ({vs})"
    return s


def fork_needs(cfg, it):
    """fork_run: the checkpoints <= it a continuation at `it` with these settings may play as league
    opponents (every one back to the pool's furthest offset; 'all': every one; fixed opponents: none
    of the run's - they are the run's copies in league/, which fork_run copies with the run)."""
    if not enabled(cfg) or parse_fixed(cfg.get('selfplay_league_fixed')):
        return set()
    pool = parse_pool(cfg.get('selfplay_league_pool', DEFAULT_POOL))
    lo = 0 if pool == 'all' else it - max(pool)
    return set(range(max(0, lo), it + 1))
