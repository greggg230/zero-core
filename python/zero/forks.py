"""Branched self-play games and archive starts (docs/az.md 24, docs/search.md 2.11).

Two opt-in ways to put positions the run's own self-play rarely reaches into its training data, both
played by the self-play driver itself (`DriverCfg::branch`, `DriverCfg::starts`):

* **Branched games** (KataGo's side games; run setting `selfplay_branch` = p, 0 = off): a share p of
  the games each spawn one branch game from a copy of their state at a uniformly drawn decision of
  theirs (a recording seat's searched decision, not minor); when the game ends its slot plays the
  branch: 1..`selfplay_branch_moves` deviation moves sampled from the network's priors at temperature
  `selfplay_branch_temp` (never recorded: no search, no policy target, no z; `selfplay_branch_differ` 1:
  the first one never the game's own move there), then normal self-play
  to the end, its rows' z the branch's own outcome. The game itself is untouched (its rows, its
  streams); none of its rows is recorded twice.
* **Near-tie branches** (`selfplay_branch_rule` 'neartie', docs/az.md 72, docs/search.md 2.11.1): the branch
  point is drawn only among decisions where the search could not separate the move played from a credible
  runner-up (|Q(a) - Q(b)| <= `selfplay_branch_eps`, b with >= `selfplay_branch_n_min` visits and prior >=
  `selfplay_branch_p_min`, Q(a) in [`selfplay_branch_q_lo`, `selfplay_branch_q_hi`]); the branch plays that
  runner-up (one deviation, no sampling), then normal self-play; a branch that reaches a state equivalent to one
  of its game's after the branch point (the game's `merge_key`) ends there with the game's outcome. The uniform
  rule's `selfplay_branch_moves` / `_temp` / `_differ` must stay at their defaults with it.
* **Archive starts** (`selfplay_archive` = an archive file, '' = off; `selfplay_archive_p` = the share
  of games): a game starts from an archived position instead of a new deal - an entry's `patch`
  merged into the run's spec (MTG Zero: its `start` block: a source game's seed and action prefix,
  the hidden cards nobody knows dealt again). The file (format 'zero-archive/1', made by the game's
  own builder) names the spec it was built and checked against (`base_spec`): another spec is refused.

**Quota.** A branch is a finished game of its own: it counts toward the iteration's `games` like any
other (so two arms with and without branching play the same number of games per iteration; a branch
starts mid-game, so it is shorter - the arm with branches plays fewer searched decisions per
iteration, which `selfplay.forks` reports). An archive start is one of the iteration's games.
Self-play's setups (the shared queue's) count only the games that started from one: a branch is
the second row of its game's setup (`setups`).

Remote self-play: the workers branch and start from the archive exactly as this machine (the run's
config.json; the archive file shipped into the mirror as `archive.json`); a worker whose checkout
plays them otherwise (an older one) is refused by its hello (`fingerprint`).
"""
import hashlib
import json
import os

FORMAT = 'zero-archive/1'
MIRROR_NAME = 'archive.json'           # the archive's name in a remote self-play mirror (zero/remoteplay.py)
KEYS = ('selfplay_branch', 'selfplay_branch_moves', 'selfplay_branch_temp', 'selfplay_branch_differ', 'selfplay_archive',
        'selfplay_archive_p', 'selfplay_branch_rule', 'selfplay_branch_eps', 'selfplay_branch_n_min', 'selfplay_branch_p_min',
        'selfplay_branch_q_lo', 'selfplay_branch_q_hi')
RULES = ('uniform', 'neartie')
# the near-tie rule's settings and their defaults (zero.run DEFAULTS; docs/az.md 72), the uniform rule's
NEARTIE = {'selfplay_branch_eps': 0.02, 'selfplay_branch_n_min': 4, 'selfplay_branch_p_min': 0.02,
           'selfplay_branch_q_lo': 0.1, 'selfplay_branch_q_hi': 0.9}
UNIFORM = {'selfplay_branch_moves': 3, 'selfplay_branch_temp': 2.0, 'selfplay_branch_differ': 0}


def branch_on(cfg):
    return float(cfg.get('selfplay_branch', 0.0) or 0.0) > 0.0


def rule_of(cfg):
    return str(cfg.get('selfplay_branch_rule', 'uniform') or 'uniform')


def archive_on(cfg):
    return bool(str(cfg.get('selfplay_archive', '') or ''))


def enabled(cfg):
    return branch_on(cfg) or archive_on(cfg)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_settings(cfg, where=''):
    """The settings' ranges (zero.run.resolve_config) -> None, else SystemExit."""
    p = cfg.get('selfplay_branch', 0.0)
    if not (_num(p) and 0.0 <= p <= 1.0):
        raise SystemExit(f'{where}selfplay_branch {p!r}: a share in [0, 1] of the games that branch (0 = off)')
    m = cfg.get('selfplay_branch_moves', 3)
    if not (isinstance(m, int) and not isinstance(m, bool) and m >= 1):
        raise SystemExit(f'{where}selfplay_branch_moves {m!r}: an integer >= 1 (deviation moves 1..N)')
    t = cfg.get('selfplay_branch_temp', 2.0)
    if not (_num(t) and t > 0):
        raise SystemExit(f'{where}selfplay_branch_temp {t!r}: a temperature > 0')
    d = cfg.get('selfplay_branch_differ', 0)
    if d not in (0, 1, True, False):
        raise SystemExit(f'{where}selfplay_branch_differ {d!r}: 0 or 1')
    a = cfg.get('selfplay_archive_p', 0.1)
    if not (_num(a) and 0.0 < a <= 1.0):
        raise SystemExit(f'{where}selfplay_archive_p {a!r}: a share in (0, 1] of the games')
    if not isinstance(cfg.get('selfplay_archive', ''), str):
        raise SystemExit(f'{where}selfplay_archive {cfg.get("selfplay_archive")!r}: a path (\'\' = off)')
    r = cfg.get('selfplay_branch_rule', 'uniform')
    if r not in RULES:
        raise SystemExit(f'{where}selfplay_branch_rule {r!r}: one of {", ".join(RULES)}')
    e = cfg.get('selfplay_branch_eps', 0.02)
    if not (_num(e) and e >= 0):
        raise SystemExit(f'{where}selfplay_branch_eps {e!r}: a Q gap >= 0 (win probability)')
    k = cfg.get('selfplay_branch_n_min', 4)
    if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1):
        raise SystemExit(f'{where}selfplay_branch_n_min {k!r}: an integer >= 1 (the runner-up\'s visits)')
    pm = cfg.get('selfplay_branch_p_min', 0.02)
    if not (_num(pm) and 0.0 <= pm <= 1.0):
        raise SystemExit(f'{where}selfplay_branch_p_min {pm!r}: a prior in [0, 1]')
    lo, hi = cfg.get('selfplay_branch_q_lo', 0.1), cfg.get('selfplay_branch_q_hi', 0.9)
    if not (_num(lo) and _num(hi) and 0.0 <= lo <= hi <= 1.0):
        raise SystemExit(f'{where}selfplay_branch_q_lo {lo!r} / selfplay_branch_q_hi {hi!r}: 0 <= q_lo <= q_hi <= 1')
    if branch_on(cfg):
        # a setting of the rule not in play does nothing: refused, never silently ignored
        if r == 'neartie':
            bad = [k for k, v in UNIFORM.items() if cfg.get(k, v) != v]
            if bad:
                raise SystemExit(f'{where}selfplay_branch_rule neartie: {", ".join(bad)} are the uniform rule\'s settings (a '
                                 f'near-tie branch plays one deviation, the runner-up): leave them at their defaults')
        else:
            bad = [k for k, v in NEARTIE.items() if cfg.get(k, v) != v]
            if bad:
                raise SystemExit(f'{where}{", ".join(bad)}: settings of selfplay_branch_rule neartie (the rule is {r!r})')
    if not enabled(cfg):
        return
    if cfg.get('resolver'):
        raise SystemExit(f'{where}selfplay_branch / selfplay_archive: single-game runs only (not co-training)')
    if L_enabled(cfg):
        raise SystemExit(f'{where}selfplay_branch / selfplay_archive: not with the self-play league (selfplay_league 0)')
    if branch_on(cfg) and not (int(cfg.get('rolling', 0) or 0) or cfg.get('selfplay_remote')):
        raise SystemExit(f'{where}selfplay_branch: needs rolling self-play (rolling 1 or selfplay_remote): '
                         f'a classic iteration plays a fixed list of games, its branches would come on top')


def L_enabled(cfg):
    from . import league as L
    return L.enabled(cfg)


def archive_path(run_dir, path):
    """An archive path: absolute as given; relative - the run directory's file if it has one (a remote
    self-play mirror's `archive.json`), else from the working directory."""
    if os.path.isabs(path):
        return path
    p = os.path.join(run_dir, path)
    return p if os.path.exists(p) else os.path.abspath(path)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def load_archive(path, game, spec):
    """The archive file at `path` for a run of `game` with spec `spec` (a JSON string or dict) -> {path,
    sha (of its entries), n, ids, kinds, specs: [(id, spec JSON)], meta}; SystemExit for a file that is
    not one, another game, a base spec other than the run's, no entries or a malformed entry."""
    from .games import spec_json
    try:
        with open(path, encoding='utf-8') as f:
            A = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f'selfplay_archive {path}: {type(e).__name__}: {e}') from None
    if not isinstance(A, dict) or A.get('format') != FORMAT:
        raise SystemExit(f'selfplay_archive {path}: not a {FORMAT} file')
    if A.get('game') != game:
        raise SystemExit(f'selfplay_archive {path}: built for game {A.get("game")!r}, the run plays {game!r}')
    base = json.loads(spec_json(spec))
    built = A.get('base_spec')
    built = json.loads(built) if isinstance(built, str) else built
    if _canon(built) != _canon(base):
        raise SystemExit(f'selfplay_archive {path}: built against another spec than the run\'s (rebuild it for '
                         f'this spec; its entries were checked on that one only)')
    entries = A.get('entries')
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f'selfplay_archive {path}: no entries')
    specs, ids, kinds = [], [], []
    for k, e in enumerate(entries):
        if not (isinstance(e, dict) and isinstance(e.get('id'), str) and isinstance(e.get('patch'), dict) and e['patch']):
            raise SystemExit(f'selfplay_archive {path}: entry {k} is not {{id, kind, patch}}')
        s = dict(base)
        s.update(e['patch'])
        specs.append((e['id'], _canon(s)))
        ids.append(e['id'])
        kinds.append(str(e.get('kind') or 'entry'))
    if len(set(ids)) != len(ids):
        raise SystemExit(f'selfplay_archive {path}: duplicate entry ids')
    sha = hashlib.sha256(_canon([[e['id'], e['patch']] for e in entries]).encode()).hexdigest()[:16]
    meta = {k: v for k, v in A.items() if k not in ('entries', 'base_spec')}
    return {'path': path, 'sha': sha, 'n': len(entries), 'ids': ids, 'kinds': kinds, 'specs': specs, 'meta': meta}


class Forks:
    """A run's branching and archive settings as its self-play drivers take them (`kwargs`)."""

    def __init__(self, run_dir, cfg):
        self.cfg = cfg
        self.branch = None
        if branch_on(cfg) and rule_of(cfg) == 'neartie':
            # the near-tie rule (docs/az.md 72): the driver's branch dict without the uniform rule's keys
            self.branch = {'p': float(cfg['selfplay_branch']), 'rule': 'neartie',
                           'eps': float(cfg.get('selfplay_branch_eps', 0.02)),
                           'n_min': int(cfg.get('selfplay_branch_n_min', 4)),
                           'p_min': float(cfg.get('selfplay_branch_p_min', 0.02)),
                           'q_lo': float(cfg.get('selfplay_branch_q_lo', 0.1)),
                           'q_hi': float(cfg.get('selfplay_branch_q_hi', 0.9))}
        elif branch_on(cfg):
            self.branch = {'p': float(cfg['selfplay_branch']), 'moves': int(cfg.get('selfplay_branch_moves', 3)),
                           'temp': float(cfg.get('selfplay_branch_temp', 2.0)),
                           'differ': bool(int(cfg.get('selfplay_branch_differ', 0) or 0))}
        self.archive = None
        if archive_on(cfg):
            path = archive_path(run_dir, str(cfg['selfplay_archive']))
            self.archive = load_archive(path, cfg['game'], cfg['spec'])
            self.archive['p'] = float(cfg.get('selfplay_archive_p', 0.1))

    def kwargs(self):
        """The self-play driver's `branch` / `starts` keyword arguments."""
        kw = {}
        if self.branch:
            kw['branch'] = dict(self.branch)
        if self.archive:
            kw['starts'] = {'p': self.archive['p'], 'specs': list(self.archive['specs'])}
        return kw

    def kinds(self):
        return self.archive['kinds'] if self.archive else []

    def describe(self):
        out = []
        if self.branch and self.branch.get('rule') == 'neartie':
            b = self.branch
            out.append(f'branch {b["p"]} of the games at near-tie decisions (|Q gap| <= {b["eps"]}, runner-up visits >= '
                       f'{b["n_min"]}, prior >= {b["p_min"]}, Q in [{b["q_lo"]}, {b["q_hi"]}]; the runner-up played, '
                       f'merges back on equal merge keys)')
        elif self.branch:
            b = self.branch
            out.append(f'branch {b["p"]} of the games ({b["moves"]} deviation moves at most, temperature {b["temp"]}'
                       + (", the first never the game's own move)" if b['differ'] else ')'))
        if self.archive:
            a = self.archive
            out.append(f'archive starts {a["p"]} of the games from {a["n"]} entries of {a["path"]} (sha {a["sha"]})')
        return '; '.join(out)


def setup(run_dir, cfg):
    """A run's `Forks`, None when both are off (nothing changes)."""
    return Forks(run_dir, cfg) if enabled(cfg) else None


def fingerprint(fk):
    """What a remote worker's hello reports (zero/remoteplay.py) and the run checks: '' when off."""
    if fk is None:
        return ''
    parts = []
    if fk.branch and fk.branch.get('rule') == 'neartie':
        # (the near-tie rule: a worker of an older checkout reports the uniform fingerprint and is refused)
        b = fk.branch
        parts.append(f'branch p={b["p"]} rule=neartie eps={b["eps"]} n_min={b["n_min"]} p_min={b["p_min"]} '
                     f'q_lo={b["q_lo"]} q_hi={b["q_hi"]}')
    elif fk.branch:
        b = fk.branch
        parts.append(f'branch p={b["p"]} moves={b["moves"]} temp={b["temp"]}' + (' differ' if b['differ'] else ''))
    if fk.archive:
        a = fk.archive
        parts.append(f'archive p={a["p"]} n={a["n"]} sha={a["sha"]}')
    return '; '.join(parts)


def setups(rows):
    """Rows that started from a setup (a queue_only driver's): a branch is its game's second row."""
    return sum(1 for r in rows if not r.get('branch'))


def new_tally():
    return {'games': {'game': 0, 'branch': 0, 'archive': 0}, 'samples': {'game': 0, 'branch': 0, 'archive': 0},
            'decisions': {'game': 0, 'branch': 0, 'archive': 0}, 'branch': {'moves': 0, 'played': 0, 'same_first': 0,
                                                                            'at_frac': 0.0, 'archive': 0},
            'archive': {}, 'neartie': new_neartie()}


def new_neartie():
    """The near-tie rule's tally (docs/az.md 72): games that counted eligible decisions and their sum, near-tie
    branches, the merged ones and their decisions to the merge, the samples of the merged ones, the gap sum, and
    the branch points' decision tags (tag -> [branches, merged])."""
    return {'games': 0, 'seen': 0, 'branches': 0, 'merges': 0, 'merge_decisions': 0, 'merge_samples': 0, 'gap': 0.0,
            'tags': {}}


def tally(rows, acc, kinds=()):
    """Add finished-game rows (accepted ones) to `acc` (new_tally)."""
    for r in rows:
        b = r.get('branch')
        k = 'branch' if b else 'archive' if r.get('start') is not None else 'game'
        acc['games'][k] += 1
        acc['samples'][k] += int(r.get('samples', 0) or 0)
        acc['decisions'][k] += int(r.get('decisions', 0) or 0)
        nt = acc.setdefault('neartie', new_neartie())
        if r.get('neartie') is not None:
            nt['games'] += 1
            nt['seen'] += int(r['neartie'])
        if b and b.get('alt') is not None:
            # a near-tie branch: merged or not, its decisions to the merge, its branch point's decision tag
            nt['branches'] += 1
            nt['gap'] += float(b.get('gap', 0.0) or 0.0)
            m = b.get('merged')
            tg = nt['tags'].setdefault(str(b.get('tag', 0)), [0, 0])
            tg[0] += 1
            if m is not None:
                nt['merges'] += 1
                nt['merge_decisions'] += int(m)
                nt['merge_samples'] += int(r.get('samples', 0) or 0)
                tg[1] += 1
        if b:
            t = acc['branch']
            t['moves'] += int(b['moves'])
            t['played'] += int(b['played'])
            t['same_first'] += int(bool(b['same_first']))
            t['at_frac'] += (b['at'] + 0.5) / max(1, b['of'])
            t['archive'] += int(b.get('start') is not None)
        elif r.get('start') is not None:
            j = int(r['start'])
            kind = kinds[j] if j < len(kinds) else 'entry'
            acc['archive'][kind] = acc['archive'].get(kind, 0) + 1
    return acc


def merge(a, b):
    """Two tallies summed (None: the other)."""
    if not a:
        return b
    if not b:
        return a
    out = new_tally()
    for part in ('games', 'samples', 'decisions', 'branch'):
        for k in out[part]:
            out[part][k] = a[part].get(k, 0) + b[part].get(k, 0)
    for src in (a, b):
        for k, v in src.get('archive', {}).items():
            out['archive'][k] = out['archive'].get(k, 0) + v
        nt = src.get('neartie') or {}
        for k in ('games', 'seen', 'branches', 'merges', 'merge_decisions', 'merge_samples', 'gap'):
            out['neartie'][k] += nt.get(k, 0)
        for tg, (nb, nm) in (nt.get('tags') or {}).items():
            cur = out['neartie']['tags'].setdefault(tg, [0, 0])
            cur[0] += nb
            cur[1] += nm
    return out


def summary(acc, fk=None):
    """The iteration's `selfplay.forks`: games, samples and searched decisions by origin, shares,
    the branches' means (deviation moves drawn / played, the first deviation equal to the game's own
    move, the branch point's place in its game), archive starts by kind."""
    g = acc['games']
    n = sum(g.values())
    nb = g['branch']
    t = acc['branch']
    out = {'games': dict(g), 'samples': dict(acc['samples']), 'decisions': dict(acc['decisions']),
           'share': {k: round(v / max(1, n), 4) for k, v in g.items()},
           'samples_per_game': {k: round(acc['samples'][k] / max(1, v), 1) for k, v in g.items()}}
    if nb:
        out['branch'] = {'moves': round(t['moves'] / nb, 3), 'played': round(t['played'] / nb, 3),
                         'same_first': round(t['same_first'] / nb, 4), 'at_frac': round(t['at_frac'] / nb, 4),
                         'of_archive': t['archive']}
    nt = acc.get('neartie') or {}
    if nt.get('games') or nt.get('branches'):
        # the near-tie rule (docs/az.md 72): eligible decisions per game, merges and their decisions to the merge, the
        # branch points' decision tags (most branches first)
        nb, nm = nt['branches'], nt['merges']
        out['neartie'] = {'games': nt['games'], 'seen': nt['seen'], 'seen_per_game': round(nt['seen'] / max(1, nt['games']), 2),
                          'branches': nb, 'merges': nm, 'merge_share': round(nm / max(1, nb), 4),
                          'decisions_to_merge': round(nt['merge_decisions'] / max(1, nm), 2),
                          'merge_samples': nt['merge_samples'], 'gap': round(nt['gap'] / max(1, nb), 5),
                          'tags': dict(sorted(nt['tags'].items(), key=lambda kv: (-kv[1][0], kv[0])))}
    if acc['archive']:
        out['archive'] = dict(sorted(acc['archive'].items()))
    if fk is not None and fk.archive:
        out['archive_sha'] = fk.archive['sha']
    return out


def fmt(s):
    """One log line of a summary."""
    g, sh = s['games'], s['share']
    line = (f"{g['game']} games, {g['branch']} branches ({100 * sh['branch']:.1f}%), {g['archive']} archive starts "
            f"({100 * sh['archive']:.1f}%); samples/game {s['samples_per_game']['game']} / "
            f"{s['samples_per_game']['branch']} / {s['samples_per_game']['archive']}")
    if 'branch' in s:
        b = s['branch']
        line += f"; branch moves {b['played']:.2f} played, first = game's {100 * b['same_first']:.0f}%"
    if 'neartie' in s:
        n = s['neartie']
        tags = ', '.join(f'{t}: {v[0]}' for t, v in list(n['tags'].items())[:6])
        line += (f"; near-tie {n['seen_per_game']} eligible/game, {n['merges']} of {n['branches']} branches merged "
                 f"({100 * n['merge_share']:.0f}%, {n['decisions_to_merge']} decisions to merge), mean gap {n['gap']}"
                 + (f", tags {tags}" if tags else ''))
    return line
