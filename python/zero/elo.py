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
    the rule's completed-Q transform: eval_select_c_visit / _c_scale, unset = cq_c_visit /
    cq_c_scale; then `mcts.forced_tag` of eval_root_min_visits, '' when off; then `mcts.view_tag` of
    eval_leaf_value_view, '-rv' for 'root'; then `mcts.belief_tag` of the evaluations' root belief,
    '-b<tilt>...' when eval_belief_tilt > 0; then `mcts.merge_tag` of eval_infoset_merge, '-ism' when on): its
    evaluations' 'select' and its pool. With
    eval_root_search 'gumbel' the move is the Gumbel search's: `mcts.gumbel_tag` (gumbel_m, the rule's
    transform, eval_gumbel_scale) instead of the rule's name, then the view's and the belief's suffixes
    ('gumbel-m16-v50-s0.3'; docs/search.md 3.7)."""
    from .mcts import (belief_settings, belief_tag, belief_tilts, forced_tag, gumbel_tag, merge_tag, select_settings, select_tag,
                       view_tag)
    ism = merge_tag(cfg.get('eval_infoset_merge', 0))
    if cfg.get('eval_root_search', 'puct') == 'gumbel':
        _, _, cv, cs = select_settings(cfg, 'eval_select')
        return gumbel_tag(cfg.get('gumbel_m', 16), cv, cs, cfg.get('eval_gumbel_scale', 0.0)) + view_tag(
            cfg.get('eval_leaf_value_view', 'mover')) + belief_tag(*belief_settings(cfg, 'eval'), tilts=belief_tilts(cfg, 'eval')) + ism
    return select_tag(*select_settings(cfg, 'eval_select')) + forced_tag(
        cfg.get('eval_root_min_visits', 0), cfg.get('root_min_visits_max_actions', 4)) + view_tag(
        cfg.get('eval_leaf_value_view', 'mover')) + belief_tag(*belief_settings(cfg, 'eval'), tilts=belief_tilts(cfg, 'eval')) + ism


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


def other_pools(run, tag='visits'):
    """The rules of a run's Elo pools other than `tag` (subdirectories of elo/ holding a
    results.json, plus 'visits' for elo/ itself): a run that has evaluated with another rule
    (docs/az.md 5.4)."""
    d = os.path.join(run, 'elo')
    try:
        subs = sorted(x for x in os.listdir(d) if os.path.exists(os.path.join(d, x, 'results.json')))
    except OSError:
        return []
    if os.path.exists(os.path.join(d, 'results.json')):
        subs = ['visits'] + subs
    return [x for x in subs if x != (tag or 'visits')]


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


def write_ratings(run, ratings, anchor, tag='visits', evaluated=None):
    """The pool's ratings.json. `evaluated` (a set of names; None = every checkpoint): the
    checkpoints that played the yardstick in this pool - any other rated checkpoint's row gets
    "evaluated": false (rated through Elo matches only: raw elo_offsets, a pool begun mid-run;
    the dashboards never call it the best)."""
    rows = {p: {'elo': round(e, 1), 'se': round(se, 1), 'games': g} for p, (e, se, g) in ratings.items()}
    if evaluated is not None:
        for p, r in rows.items():
            if p.startswith('iter_') and p not in evaluated:
                r['evaluated'] = False
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
    yardstick in this pool (raw elo_offsets, and a rule's pool begun mid-run, rate others through
    Elo matches only: they are listed with "evaluated": false). Returns the best entry."""
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


# ------------------------------------------------------------------ fixed anchors (docs/az.md 20.1)

ANCHOR_PREFIX = 'anchor:'


def parse_anchors(s):
    """The run setting elo_anchors ('path,label=path,...', a list, or '' / None) -> [(label, path)] in
    order (() = off). A label defaults to the file's stem; labels are distinct; the pool's player is
    `anchor_name(label)`. ValueError on a malformed setting."""
    if s in (None, ''):
        return []
    items = s.split(',') if isinstance(s, str) else list(s)
    out = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        label, sep, path = x.partition('=')
        if not sep or any(ch in label for ch in ':/' + chr(92)):
            label, path = os.path.splitext(os.path.basename(x))[0], x      # (no label: 'E:/x/iter_0192.pt')
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError(f'elo_anchors {s!r}: entries "path" or "label=path"')
        out.append((label, path))
    labels = [lb for lb, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f'elo_anchors {s!r}: labels must be distinct ({labels})')
    return out


def anchor_name(label):
    """The pool's player name of the anchor `label` (never a checkpoint's name: 'anchor:<label>')."""
    return ANCHOR_PREFIX + label


def is_anchor(name):
    return name.startswith(ANCHOR_PREFIX)


def fit_fixed(res, fixed, prior_sd=8.0, iters=50, pairs=None):
    """`fit` with the players `fixed` ({name: elo}) held at their ratings (the anchors) and every other
    player free (the yardstick included) -> {player: (elo, se, games)}, the fixed ones with se 0. The
    weak Gaussian prior (prior_sd log-odds) is centred on the fixed players' mean. `pairs`: the
    results' keys to use (None: all). Players without a path of games to a fixed one get the prior's
    centre and a huge se. {} when no fixed player has games (or none is given)."""
    keys = list(res['pairs']) if pairs is None else list(pairs)
    players = sorted({p for k in keys for p in k.split('|')} | set(fixed))
    fixed = {p: float(v) for p, v in fixed.items()}
    if not fixed:
        return {}
    r0 = {p: (v - ANCHOR_ELO) / SCALE for p, v in fixed.items()}
    centre = sum(r0.values()) / len(r0)
    free = [p for p in players if p not in fixed]
    idx = {p: i for i, p in enumerate(free)}
    games = {p: 0 for p in players}
    rows = []
    for k in keys:
        s, n = res['pairs'][k]
        a, b = k.split('|')
        if n > 0:
            rows.append((a, b, s, n))
            games[a] += n
            games[b] += n
    r = np.full(len(free), centre)
    H = -np.eye(len(free)) / prior_sd ** 2

    def rating(p):
        return r[idx[p]] if p in idx else r0[p]
    for _ in range(iters):
        if not len(free):
            break
        g = -(r - centre) / prior_sd ** 2
        H = -np.eye(len(free)) / prior_sd ** 2
        for a, b, s, n in rows:
            ia, ib = idx.get(a), idx.get(b)
            if ia is None and ib is None:
                continue
            p = 1.0 / (1.0 + math.exp(-(rating(a) - rating(b))))
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
        step = np.linalg.solve(H, g)
        r -= step
        if np.abs(step).max() < 1e-7:
            break
    out = {p: (fixed[p], 0.0, games[p]) for p in fixed}
    if len(free):
        cov = np.linalg.inv(-H)
        for p, i in idx.items():
            out[p] = (ANCHOR_ELO + SCALE * r[i], SCALE * math.sqrt(max(cov[i, i], 0.0)), games[p])
    return out


def anchored_path(run, tag='visits'):
    return os.path.join(pool_dir(run, tag), 'anchored.json')


def write_anchored(run, ratings, anchors, tag='visits'):
    """The pool's anchored.json: {'anchors': the fixed anchors {name: {elo, label, sha256, ...}}, 'ratings':
    {player: {elo, se, games}} of the fit with them fixed (`fit_fixed`), 'updated'} (docs/az.md 20.1)."""
    d = pool_dir(run, tag)
    os.makedirs(d, exist_ok=True)
    out = {'anchors': anchors, 'updated': time.time(),
           'ratings': {p: {'elo': round(e, 1), 'se': round(se, 1), 'games': g} for p, (e, se, g) in ratings.items()}}
    if tag not in (None, '', 'visits'):
        out['select'] = tag
    tmp = os.path.join(d, 'anchored.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    _replace(tmp, anchored_path(run, tag))
