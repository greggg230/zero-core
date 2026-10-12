"""Elo ratings for checkpoints and reference players (as stpz / domz).

Results live in runs/<run>/elo/results.json as {"anchor": name, "pairs": {"a|b": [score_of_a,
games]}} with a < b (a shared win scores 0.5 each). Ratings are a maximum-likelihood
Bradley-Terry fit over all results, with the run's yardstick anchored at 1000 (the toy: the
heuristic bot; Caverna / Glass Road: the web app's JS bot). Refitting after every iteration
lets older ratings sharpen as more games involve them.
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


def _path(run):
    return os.path.join(run, 'elo', 'results.json')


def load_results(run, anchor=None):
    p = _path(run)
    if os.path.exists(p):
        with open(p) as f:
            res = json.load(f)
    else:
        res = {'pairs': {}}
    if anchor is not None:
        res.setdefault('anchor', anchor)
    return res


def save_results(run, res):
    os.makedirs(os.path.join(run, 'elo'), exist_ok=True)
    tmp = _path(run) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    _replace(tmp, _path(run))


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


def write_ratings(run, ratings, anchor):
    rows = {p: {'elo': round(e, 1), 'se': round(se, 1), 'games': g} for p, (e, se, g) in ratings.items()}
    os.makedirs(os.path.join(run, 'elo'), exist_ok=True)
    tmp = os.path.join(run, 'elo', 'ratings.json.tmp')
    with open(tmp, 'w') as f:
        json.dump({'anchor': anchor, 'anchor_elo': ANCHOR_ELO, 'updated': time.time(), 'ratings': rows}, f, indent=1)
    _replace(tmp, os.path.join(run, 'elo', 'ratings.json'))


def _replace(tmp, dst, tries=40):
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.25)


def tag_checkpoints(run, ratings, latest, anchor):
    """Refresh models/index.json (each checkpoint's Elo) and copy the highest-rated checkpoint
    to models/best.pt. Returns the best entry."""
    mdir = os.path.join(run, 'models')
    index = []
    for i in range(latest + 1):
        name = ckpt_name(i)
        if not os.path.exists(os.path.join(mdir, name + '.pt')) or name not in ratings:
            continue
        e, se, g = ratings[name]
        index.append({'iter': i, 'file': name + '.pt', 'elo': round(e, 1), 'se': round(se, 1), 'games': g})
    best = max(index, key=lambda r: r['elo']) if index else None
    if best:
        shutil.copyfile(os.path.join(mdir, best['file']), os.path.join(mdir, 'best.pt.tmp'))
        _replace(os.path.join(mdir, 'best.pt.tmp'), os.path.join(mdir, 'best.pt'))
    refs = {p: {'elo': round(e, 1), 'se': round(se, 1)} for p, (e, se, g) in ratings.items() if not p.startswith('iter_')}
    tmp = os.path.join(mdir, 'index.json.tmp')
    with open(tmp, 'w') as f:
        json.dump({'anchor': f'{anchor}={ANCHOR_ELO:g}', 'best': best, 'references': refs, 'checkpoints': index}, f, indent=1)
    _replace(tmp, os.path.join(mdir, 'index.json'))
    return best
