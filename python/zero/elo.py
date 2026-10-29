"""Elo ratings for checkpoints and reference players (as stpz / domz).

Results live in runs/<run>/elo/results.json as {"anchor": name, "pairs": {"a|b": [score_of_a,
games]}} with a < b (a shared win scores 0.5 each). Ratings are a maximum-likelihood
Bradley-Terry fit over all results, with the run's yardstick anchored at 1000 (the toy: the
heuristic bot; Caverna / Glass Road: the web app's JS bot). Refitting after every iteration
lets older ratings sharpen as more games involve them.

**One pool per move rule** (docs/az.md 5.4): a checkpoint searched with another rule for its
non-sampled moves (run setting eval_select, docs/search.md 3.3) is another player, so each rule
keeps its own pool - elo/ for the default 'visits' (every run's pool from before the setting),
elo/<rule>/ for another ('cq', 'cq-n2', ...; its results.json also holds {"select": rule,
"since": the first iteration it rated}) - with its own results.json, ratings.json, and the rule's
models/index.json + models/best.pt while it is the run's rule. A pool never holds another rule's
games, so a run that changes its rule leaves the old pool's files as they were.
"""
import json
import math
import os
import shutil
import time

import numpy as np

ANCHOR_ELO = 1000.0
SCALE = 400.0 / math.log(10.0)        # Elo points per unit of log-odds
OFFSETS = (1, 2, 4, 8, 16, 32)         # each checkpoint plays the ones this many iterations back


def ckpt_name(i):
    return f'iter_{i:04d}'


def pool_dir(run, tag='visits'):
    """The Elo pool directory of the move rule `tag` (docs/az.md 5.4): <run>/elo for 'visits',
    <run>/elo/<tag> for another."""
    d = os.path.join(run, 'elo')
    return d if tag in (None, '', 'visits') else os.path.join(d, tag)


def eval_tag(cfg):
    """The name of a run's evaluation move rule (`mcts.select_tag` of eval_select, its guard and
    the completed-Q transform): its evaluations' 'select' and its pool."""
    from .mcts import select_tag
    return select_tag(cfg.get('eval_select', 'visits'), cfg.get('eval_select_min_visits', 0),
                      cfg.get('cq_c_visit', 50.0), cfg.get('cq_c_scale', 1.0))


def active_pool(run):
    """(tag, directory) of the Elo pool a run's evaluations use now: its config.json's rule (the
    dashboards; a run without one: 'visits')."""
    try:
        with open(os.path.join(run, 'config.json'), encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    tag = eval_tag(cfg if isinstance(cfg, dict) else {})
    return tag, pool_dir(run, tag)


def row(rating, tag='visits'):
    """A checkpoint's Elo row in metrics / evals ({elo, se}, + the pool's rule unless 'visits')."""
    out = {'elo': round(rating[0], 1), 'se': round(rating[1], 1)}
    if tag not in (None, '', 'visits'):
        out['select'] = tag
    return out


def pool_note(tag):
    """' (pool <tag>)' for a log line, '' for the default pool."""
    return '' if tag in (None, '', 'visits') else f' (pool {tag})'


def parse_offsets(s):
    """The run setting elo_offsets ('1,2,4,8', a list, or '' / None) -> a tuple of distinct
    positive iteration offsets in the given order (() = unset), else ValueError."""
    if s in (None, ''):
        return ()
    items = s.split(',') if isinstance(s, str) else list(s)
    try:
        out = tuple(int(str(x).strip()) for x in items if str(x).strip())
    except ValueError:
        raise ValueError(f'elo_offsets {s!r}: positive integers, e.g. "1,2,4,8"') from None
    if not out or any(x < 1 for x in out) or len(set(out)) != len(out):
        raise ValueError(f'elo_offsets {s!r}: distinct positive integers, e.g. "1,2,4,8"')
    return out


def opponents(k, eval_every, offsets='', limit=0):
    """The checkpoints checkpoint k plays in its Elo matches (those that exist and are >= 0 are
    played): k - off x eval_every for the first `limit` of OFFSETS (0 = all) by default; with
    `offsets` (run setting elo_offsets), k - off in raw iterations whatever eval_every (`limit`
    is not used)."""
    raw = parse_offsets(offsets)
    if raw:
        return [k - off for off in raw]
    e = max(1, int(eval_every))
    return [k - off * e for off in (OFFSETS[:limit] if limit else OFFSETS)]


def _path(run, tag='visits'):
    return os.path.join(pool_dir(run, tag), 'results.json')


def load_results(run, anchor=None, tag='visits'):
    """A pool's results (`tag`: the move rule's pool, docs/az.md 5.4); a new pool: no pairs."""
    p = _path(run, tag)
    if os.path.exists(p):
        with open(p) as f:
            res = json.load(f)
    else:
        res = {'pairs': {}}
    if anchor is not None:
        res.setdefault('anchor', anchor)
    return res


def save_results(run, res, tag='visits'):
    os.makedirs(pool_dir(run, tag), exist_ok=True)
    tmp = _path(run, tag) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    _replace(tmp, _path(run, tag))


def add(res, a, b, score_a, n):
    """Record n games between a and b in which a scored score_a."""
    if a > b:
        a, b, score_a = b, a, n - score_a
    key = f'{a}|{b}'
    s, m = res['pairs'].get(key, [0.0, 0])
    res['pairs'][key] = [s + score_a, m + n]


def has(res, a, b):
    return f'{min(a, b)}|{max(a, b)}' in res['pairs']


def fit(res, prior_sd=8.0, iters=50, anchor=None):
    """-> {player: (elo, se, games)}. Newton's method on the Bradley-Terry log-likelihood with a
    weak Gaussian prior (keeps perfect records finite)."""
    anchor = anchor or res.get('anchor')
    players = sorted({p for k in res['pairs'] for p in k.split('|')})
    if anchor not in players:
        return {}
    free = [p for p in players if p != anchor]
    idx = {p: i for i, p in enumerate(free)}
    pairs = []
    games = {p: 0 for p in players}
    for k, (s, n) in res['pairs'].items():
        a, b = k.split('|')
        if n > 0:
            pairs.append((idx.get(a), idx.get(b), s, n))
            games[a] += n
            games[b] += n
    r = np.zeros(len(free))
    H = -np.eye(len(free)) / prior_sd ** 2
    for _ in range(iters):
        g = -r / prior_sd ** 2
        H = -np.eye(len(free)) / prior_sd ** 2
        for ia, ib, s, n in pairs:
            ra = r[ia] if ia is not None else 0.0
            rb = r[ib] if ib is not None else 0.0
            p = 1.0 / (1.0 + math.exp(-(ra - rb)))
            d = s - n * p
            w = n * p * (1 - p)
            if ia is not None:
                g[ia] += d
                H[ia, ia] -= w
            if ib is not None:
                g[ib] -= d
                H[ib, ib] -= w
            if ia is not None and ib is not None:
                H[ia, ib] += w
                H[ib, ia] += w
        if not len(free):
            break
        step = np.linalg.solve(H, g)
        r -= step
        if np.abs(step).max() < 1e-7:
            break
    out = {anchor: (ANCHOR_ELO, 0.0, games[anchor])}
    if len(free):
        cov = np.linalg.inv(-H)
        for p, i in idx.items():
            out[p] = (ANCHOR_ELO + SCALE * r[i], SCALE * math.sqrt(max(cov[i, i], 0.0)), games[p])
    return out


def write_ratings(run, ratings, anchor, tag='visits'):
    rows = {p: {'elo': round(e, 1), 'se': round(se, 1), 'games': g} for p, (e, se, g) in ratings.items()}
    d = pool_dir(run, tag)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, 'ratings.json.tmp')
    out = {'anchor': anchor, 'anchor_elo': ANCHOR_ELO, 'updated': time.time(), 'ratings': rows}
    if tag not in (None, '', 'visits'):
        out['select'] = tag
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    _replace(tmp, os.path.join(d, 'ratings.json'))


def _replace(tmp, dst, tries=40):
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.25)


def tag_checkpoints(run, ratings, latest, anchor, tag='visits', evaluated=None):
    """Refresh models/index.json (each checkpoint's Elo in the pool of the move rule `tag`, the
    run's rule) and copy the highest-rated checkpoint to models/best.pt. `evaluated` (a set of
    names; None = every rated one): the checkpoints that may be best - those that played the
    yardstick (elo_offsets rates others too: they are listed with "evaluated": false). Returns
    the best entry."""
    mdir = os.path.join(run, 'models')
    index = []
    for i in range(latest + 1):
        name = ckpt_name(i)
        if not os.path.exists(os.path.join(mdir, name + '.pt')) or name not in ratings:
            continue
        e, se, g = ratings[name]
        index.append({'iter': i, 'file': name + '.pt', 'elo': round(e, 1), 'se': round(se, 1), 'games': g})
        if evaluated is not None and name not in evaluated:
            index[-1]['evaluated'] = False
    cands = [r for r in index if r.get('evaluated', True)]
    best = max(cands, key=lambda r: r['elo']) if cands else None
    if best:
        shutil.copyfile(os.path.join(mdir, best['file']), os.path.join(mdir, 'best.pt.tmp'))
        _replace(os.path.join(mdir, 'best.pt.tmp'), os.path.join(mdir, 'best.pt'))
    refs = {p: {'elo': round(e, 1), 'se': round(se, 1)} for p, (e, se, g) in ratings.items() if not p.startswith('iter_')}
    tmp = os.path.join(mdir, 'index.json.tmp')
    out = {'anchor': f'{anchor}={ANCHOR_ELO:g}', 'best': best, 'references': refs, 'checkpoints': index}
    if tag not in (None, '', 'visits'):
        out['select'] = tag
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    _replace(tmp, os.path.join(mdir, 'index.json'))
    return best
