"""Head-to-head match between two checkpoints, seats alternating: the readout of an A/B whose
yardstick is saturated (Caverna Zero's cvz2p wins 98-99.5% against the JS bot; docs/az.md 9).

  python -m <pkg>.az.h2h runs/cvz2p_cq/models/iter_0180.pt runs/cvz2p_vis/models/iter_0180.pt
      [--games 600 --sims 64 --chance-k 2 --device cuda --threads 4 --seed 0]
      [--select cq [--select-min-visits 2]] [--a-select cq] [--b-select visits] [--move-log 200]
      [--cq-c-scale 0.3] [--a-cq-c-scale 0.3 --b-cq-c-scale 1.0] [--a-cq-c-visit 50 ...]
  (python -m zero.h2h ... in zero-core: toy checkpoints)

Both seats search with the evaluation settings of run.py (`eval_sims`, chance_k, no noise, no
temperature); each side's move rule (docs/search.md 3.3) is its checkpoint's run's eval_select
(runs/<run>/config.json next to models/: its rule, guard and the rule's completed-Q transform -
eval_select_c_visit / _c_scale, unset: cq_c_visit / cq_c_scale; 'visits' without one), or --select /
--select-min-visits for both, --a-select / --b-select (and their --*-min-visits) for one side - the
same checkpoint with two rules is an A/B of the rule. The 'cq' rule's transform: --cq-c-visit /
--cq-c-scale for both sides, --a-cq-c-* / --b-cq-c-* for one (over the both-sides flag); a side
without one takes 50 / 1 when its rule is given here, else its run's - so `A A --select cq
--a-cq-c-scale 0.3 --b-cq-c-scale 1.0` is an A/B of the selection scale on one checkpoint. A
side's transform flag on a 'visits' side is an error, as are both-sides flags when no side plays
'cq'. The game comes from checkpoint A (`Net.config['game']`), its spec from the game package
(`GamePackage.specs`, e.g. Caverna {"full": true}, Glass Road {"intro": false}) unless --spec is
given. Prints one JSON line: A's score (+- 1 se), its mean score margin, the Elo difference, each
side's rule (`select_a`, `select_b`: e.g. 'cq-n4-v50-s0.3'). The models are only read.

The root's belief (docs/search.md 3.8): each side deals its search's worlds as its run evaluates
(eval_belief_tilt, belief_clip, belief_mix, belief_fast in its config.json; also with --select) unless --a-belief-tilt /
--b-belief-tilt (0 = uniform), --*-belief-clip, --*-belief-mix, --*-belief-fast say otherwise; a side
with a tilt needs a checkpoint with the auxiliary head (refused before the match: net.HeadMissing).
`A A --a-belief-tilt 1 --b-belief-tilt 0` is the evaluation-only A/B of the belief on one checkpoint.
Each side's rule name carries it ('-b1', 'select_a').
Root search (docs/search.md 3.7): each side searches as its checkpoint's run's evaluation does
(eval_root_search, whatever --select says: a run trained and evaluated with the Gumbel root search
plays its side with it), or --a-root-search / --b-root-search puct | gumbel. A Gumbel side's sigma is
its 'cq' rule's transform when it has one (the --*-cq-c-* flags, or its run's), else its run's
evaluation transform (eval_select_c_* / cq_c_*); its considered options --gumbel-m (default: its
run's gumbel_m, 16) and its noise --a-gumbel-scale / --b-gumbel-scale (default: its run's
eval_gumbel_scale, 0 = none). The side's move rule is then the Gumbel search's (`select_*`:
'gumbel-m16-v50-s0.3').

Per-side simulations (docs/az.md 20.4): --a-sims / --b-sims (default --sims), e.g. `A A --a-sims 128
--b-sims 32` (the output then names `sims_a`, `sims_b`).

Paired games and sequential testing (docs/az.md 20.3):

  --paired          games 2k and 2k + 1 share every random stream (the deal, shuffles, chance: the
                    driver's `paired`) with the seats swapped, so each deal is played once from each
                    seat by each side. The output adds `pairs` (both won / split / both lost / with a
                    draw, the pentanomial counts of a pair's points 0, 0.5, ..., 2), `se` becomes the
                    se of the pair scores (`se_games`: the binomial se of the games, as unpaired) and
                    `variance_ratio` = var(pair score) / (var(game) / 2), the variance the pairing left
                    (1 = none removed). --games must be even.
  --sprt E0,E1      a sequential probability ratio test of H0: A is E0 Elo better than B against H1:
                    E1 (logistic Elo; the normal-approximation GSPRT of the per-pair scores, or per-game
                    scores unpaired), error rates --alpha / --beta (0.05 / 0.05): batches of
                    --batch-games (default --games) are played until the log-likelihood ratio leaves
                    (ln(beta / (1 - alpha)), ln((1 - beta) / alpha)) or --max-games (default 4 x
                    --games) are played. `sprt` reports llr, the bounds and the decision 'H1' | 'H0' |
                    'undecided'.
  --target-se S     batches until `se` <= S (or --max-games).
Each batch b > 0 is seeded seed + 7919 b (batch 0: --seed, so a one-batch run is the plain match); the
scores, se and pairs are over every game played; `batches` lists each batch's games, score and llr
(cumulative), and a progress line per batch goes to stderr ('# batch {...}').
"""
import argparse
import json
import math
import os
import sys
import time

from . import games as G, package
from .infer import NetEvaluator, drive
from .mcts import (belief_settings, belief_tag, check_belief, check_root_search, check_select, check_select_transform,
                   forced_tag, gumbel_tag, select_settings, select_tag, select_transform, view_tag)
from .net import check_belief_head, load


def run_config(ckpt):
    """The config.json of a checkpoint runs/<run>/models/<file>.pt's run ({} without one)."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    try:
        with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    return c if isinstance(c, dict) else {}


def root_keys(ckpt, keys, root=None, m=None, scale=None):
    """A side's seat settings `keys` with its root search (docs/search.md 3.7): `root` 'puct' | 'gumbel'
    (None: its checkpoint's run's eval_root_search). A Gumbel side: gumbel_m `m` (None: its run's, 16),
    noise scale `scale` (None: its run's eval_gumbel_scale, 0), sigma = its 'cq' rule's transform in
    `keys` (cq_c_visit / cq_c_scale) when it has one, else its run's evaluation transform."""
    c = run_config(ckpt)
    root = root or c.get('eval_root_search', 'puct')
    keys = {k: v for k, v in keys.items() if not (k == 'root_search' or k.startswith('gumbel_'))}
    if root == 'puct':
        return keys
    m = int(c.get('gumbel_m', 16) if m is None else m)
    scale = float(c.get('eval_gumbel_scale', 0.0) if scale is None else scale)
    check_root_search(root, m, scale, what='--root-search')
    if keys.get('select', 'visits') == 'cq':
        cv, cs = keys['cq_c_visit'], keys['cq_c_scale']
    else:
        _, _, cv, cs = select_settings(c, 'eval_select')
    return dict(keys, root_search='gumbel', gumbel_m=m, gumbel_scale=scale, gumbel_c_visit=float(cv),
                gumbel_c_scale=float(cs))


def run_rule(ckpt):
    """The move-rule seat settings of a checkpoint's run (its config.json's eval_select, guard
    and the rule's completed-Q transform, `mcts.select_settings`; {} = the default) for a
    checkpoint runs/<run>/models/<file>.pt. An evaluation seat has no 'cq' target, so the rule's
    transform goes in as cq_c_visit / cq_c_scale (as `games.nn_seat` writes it). The run's
    evaluation root minimum visits (eval_root_min_visits, docs/search.md 3.5), leaf values' view
    (eval_leaf_value_view, docs/search.md 3.6) and root search (eval_root_search, `root_keys`,
    docs/search.md 3.7) come with it."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    try:
        with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(c, dict):
        return {}
    out = {}
    if c.get('eval_root_min_visits'):
        out = {'root_min_visits': int(c['eval_root_min_visits']),
               'root_min_visits_max_actions': int(c.get('root_min_visits_max_actions', 4))}
    if c.get('eval_leaf_value_view', 'mover') != 'mover':
        out['leaf_value_view'] = c['eval_leaf_value_view']
    tilt, clip, mix, fast = belief_settings(c, 'eval')
    if tilt:
        out.update(belief_tilt=tilt, belief_clip=clip, belief_mix=mix, belief_fast=fast)
    if c.get('eval_select', 'visits') == 'visits':
        return root_keys(ckpt, out)
    rule, mv, cv, cs = select_settings(c, 'eval_select')
    return root_keys(ckpt, dict({'select': rule, 'select_min_visits': mv, 'cq_c_visit': cv, 'cq_c_scale': cs}, **out))


def rule_keys(select=None, min_visits=0, c_visit=50.0, c_scale=1.0):
    """Seat settings of a move rule given on the command line (None: not given)."""
    if select is None:
        return None
    check_select(select, min_visits)
    if select == 'visits':
        return {}
    check_select_transform(select, float(c_visit), float(c_scale), what='--cq')
    return {'select': select, 'select_min_visits': int(min_visits), 'cq_c_visit': float(c_visit),
            'cq_c_scale': float(c_scale)}


def side_keys(ckpt, select=None, min_visits=0, c_visit=None, c_scale=None):
    """A side's move-rule seat settings: `select` given (rule_keys; a transform not given: 50 / 1)
    or its checkpoint's run's (run_rule), with the 'cq' rule's transform `c_visit` / `c_scale` where
    given (None: not given). A transform given for a 'visits' side is a ValueError."""
    if select is not None:
        keys = rule_keys(select, min_visits, 50.0 if c_visit is None else c_visit, 1.0 if c_scale is None else c_scale)
    else:
        keys = dict(run_rule(ckpt))
    if c_visit is None and c_scale is None:
        return keys
    if keys.get('select', 'visits') == 'visits':
        raise ValueError(f"{os.path.basename(ckpt)}: a completed-Q transform for a side whose rule is 'visits'")
    cv, cs = select_transform(keys['cq_c_visit'], keys['cq_c_scale'], c_visit, c_scale)
    check_select_transform('cq', cv, cs, what='--cq')
    return dict(keys, cq_c_visit=cv, cq_c_scale=cs)


def _tag(seat):
    if seat.get('root_search', 'puct') == 'gumbel':
        return gumbel_tag(seat.get('gumbel_m', 16), seat.get('gumbel_c_visit', 50.0), seat.get('gumbel_c_scale', 1.0),
                          seat.get('gumbel_scale', 1.0)) + view_tag(seat.get('leaf_value_view', 'mover')) + belief_tag(
            seat.get('belief_tilt', 0.0), seat.get('belief_clip'), seat.get('belief_mix', 0.0), seat.get('belief_fast', False))
    cv, cs = select_transform(seat.get('cq_c_visit', 50.0), seat.get('cq_c_scale', 1.0), seat.get('select_c_visit'),
                              seat.get('select_c_scale'))
    return select_tag(seat.get('select', 'visits'), seat.get('select_min_visits', 0), cv, cs) + forced_tag(
        seat.get('root_min_visits', 0), seat.get('root_min_visits_max_actions', 4)) + view_tag(
        seat.get('leaf_value_view', 'mover')) + belief_tag(seat.get('belief_tilt', 0.0), seat.get('belief_clip'),
                                                           seat.get('belief_mix', 0.0), seat.get('belief_fast', False))


def with_belief(keys, tilt=None, clip=None, mix=None, fast=None):
    """A side's seat keys with its root belief (docs/search.md 3.8) as given (None: as it was): tilt 0
    removes it (uniform worlds); clip / mix / fast change a side that has one (or is given one)."""
    keys = dict(keys)
    if tilt is not None:
        if float(tilt) == 0:
            for k in ('belief_tilt', 'belief_clip', 'belief_mix', 'belief_fast'):
                keys.pop(k, None)
            return keys
        keys.setdefault('belief_clip', 20.0)
        keys.setdefault('belief_mix', 0.0)
        keys.setdefault('belief_fast', False)
        keys['belief_tilt'] = float(tilt)
    if not keys.get('belief_tilt'):
        if (clip, mix, fast) != (None, None, None):
            raise ValueError('--*-belief-clip / -mix / -fast for a side without a root belief (give its --*-belief-tilt)')
        return keys
    if clip is not None:
        keys['belief_clip'] = float(clip)
    if mix is not None:
        keys['belief_mix'] = float(mix)
    if fast is not None:
        keys['belief_fast'] = bool(fast)
    check_belief(keys['belief_tilt'], keys['belief_clip'], keys['belief_mix'])
    return keys


def pair_stats(rows):
    """Paired games' statistics of side A (rows: result rows of a match whose games 2k and 2k + 1 are
    a pair, A in seat gid % 2): {'n', 'score', 'se', 'var', 'both_won', 'split', 'both_lost',
    'with_draw', 'pentanomial': counts of a pair's points [0, 0.5, 1, 1.5, 2]}; `score` the mean pair
    score (= the mean of the pairs' games), `se` the se of the pair scores, `var` their variance.
    Unpaired leftovers are ignored."""
    by = {r['gid']: float(r['result'][r['gid'] % 2]) for r in rows}
    pairs = [(by[g], by[g + 1]) for g in sorted(by) if g % 2 == 0 and g + 1 in by]
    out = {'n': len(pairs), 'both_won': 0, 'split': 0, 'both_lost': 0, 'with_draw': 0, 'pentanomial': [0] * 5}
    for x, y in pairs:
        if x == 1 and y == 1:
            out['both_won'] += 1
        elif x == 0 and y == 0:
            out['both_lost'] += 1
        elif {x, y} == {0.0, 1.0}:
            out['split'] += 1
        else:
            out['with_draw'] += 1
        out['pentanomial'][min(4, max(0, int(round(2 * (x + y)))))] += 1
    sc = [(x + y) / 2 for x, y in pairs]
    n = len(sc)
    mean = sum(sc) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in sc) / (n - 1) if n > 1 else 0.0
    out.update(score=round(mean, 4), se=round(math.sqrt(var / n), 4) if n else 0.0, var=var)
    return out


def expected_score(elo):
    """The logistic expected score of an `elo` difference."""
    return 1.0 / (1.0 + 10 ** (-float(elo) / 400.0))


def sprt_llr(obs, elo0, elo1):
    """The log-likelihood ratio of H1 (Elo difference elo1) against H0 (elo0) for observations `obs`
    (per-pair scores of paired games, or per-game scores) - the generalized SPRT's normal approximation
    (Van den Bergh; Fishtest's GSPRT): n (s1 - s0) (2 mean - s0 - s1) / (2 var), s = the expected scores,
    var the observations' variance (a tiny floor: identical observations)."""
    n = len(obs)
    if n < 2:
        return 0.0
    mean = sum(obs) / n
    var = max(sum((v - mean) ** 2 for v in obs) / n, 1e-6)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def sprt_bounds(alpha=0.05, beta=0.05):
    """(lower, upper) LLR bounds: below = accept H0, above = accept H1 (Wald)."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def observations(rows, paired):
    """Side A's per-pair scores (paired) or per-game scores (rows of a match, A in seat gid % 2)."""
    if paired:
        by = {r['gid']: float(r['result'][r['gid'] % 2]) for r in rows}
        return [(by[g] + by[g + 1]) / 2 for g in sorted(by) if g % 2 == 0 and g + 1 in by]
    return [float(r['result'][r['gid'] % 2]) for r in rows]


def _load_pair(a, b, device, game):
    # (the encoding guard, docs/az.md 10.4: A against the game played, B against its own game,
    # which must be the same)
    ma, _ = load(a, device, game=game)
    game = game or ma.config.get('game', 'toy')
    mb, _ = load(b, device)
    if mb.config.get('game', game) != game:
        raise ValueError(f'{a} plays {game}, {b} plays {mb.config.get("game")}')
    return ma, mb, game


def _seats(a, b, ma, mb, game, sims, c_puct, chance_k, seat_a, seat_b, sims_a, sims_b):
    sa = G.nn_seat('a', sims if sims_a is None else int(sims_a), c_puct, chance_k=chance_k)
    sb = G.nn_seat('b', sims if sims_b is None else int(sims_b), c_puct, chance_k=chance_k)
    sa.update(run_rule(a) if seat_a is None else seat_a)
    sb.update(run_rule(b) if seat_b is None else seat_b)
    for s, m, p in ((sa, ma, a), (sb, mb, b)):
        if s.get('belief_tilt'):
            check_belief_head(m, game, f'belief_tilt of {p}')     # (before any game: a loud refusal)
    return sa, sb


def _play(game, sa, sb, evs, games, spec, conc, seed, threads, groups, max_steps, move_log, paired):
    seats = [[sa, sb] if g % 2 == 0 else [sb, sa] for g in range(games)]
    kw = {'move_log': move_log} if move_log else {}
    if paired:
        kw['paired'] = True
    drv = G.Driver(game, seats, spec=spec, concurrency=conc or min(games, 1024), seed=seed, threads=threads,
                   groups=groups, max_steps=max_steps, **kw)
    drive(drv, evs)
    return drv.take_results()


def summary(rs, paired=False):
    """A's score (+- 1 se: the pair scores' with `paired`, else the games' binomial se), mean margin,
    Elo difference (+ its se, the delta method), unfinished games; paired: `pairs`, `se_games`,
    `variance_ratio` (var(pair score) / (var(game) / 2))."""
    n = len(rs)
    w = sum(r['result'][r['gid'] % 2] for r in rs) / n
    margin = sum(r['scores'][r['gid'] % 2] - r['scores'][1 - r['gid'] % 2] for r in rs) / n
    s = min(max(w, 1e-3), 1 - 1e-3)
    out = {'games': n, 'a_score': round(w, 4), 'se': round(math.sqrt(w * (1 - w) / n), 4),
           'a_margin': round(margin, 2), 'elo_diff': round(400 * math.log10(s / (1 - s)), 1),
           'unfinished': sum(1 for r in rs if not r['over'])}
    if paired:
        ps = pair_stats(rs)
        var = ps.pop('var')
        vg = sum((r['result'][r['gid'] % 2] - w) ** 2 for r in rs) / max(1, n - 1)
        out['se_games'] = out['se']
        if ps['n']:
            out['se'] = ps['se']
        out['variance_ratio'] = round(var / (vg / 2), 4) if vg > 0 and ps['n'] > 1 else None
        out['pairs'] = ps
    out['elo_se'] = round(400 / math.log(10) * out['se'] / (s * (1 - s)), 1)
    return out


def match(a, b, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None, seat_a=None, seat_b=None, max_steps=5000, move_log=0, rows=False, paired=False,
          sims_a=None, sims_b=None):
    """Checkpoint `a` vs checkpoint `b` (paths), `games` 2-player games, seats alternating. -> dict.
    `seat_a` / `seat_b`: each side's move-rule seat settings ({'select': 'cq', 'select_min_visits':
    2, ...}; {} = the default rule; None = its checkpoint's run's eval_select, `run_rule`).
    `move_log`: the driver's move log (the first N moves of each game in its row), `rows`: the
    games' result rows under 'rows'. `paired`: games 2k and 2k + 1 share their deal with the seats
    swapped (`games` even; the output's `pairs`, `se` of the pair scores: `summary`). `sims_a` /
    `sims_b`: a side's simulations (None: `sims`)."""
    if paired and games % 2:
        raise ValueError(f'paired games: an even number of games ({games})')
    ma, mb, game = _load_pair(a, b, device, game)
    spec = package.active().specs.get(game) if spec is None else spec
    sa, sb = _seats(a, b, ma, mb, game, sims, c_puct, chance_k, seat_a, seat_b, sims_a, sims_b)
    t0 = time.time()
    rs = _play(game, sa, sb, {'a': NetEvaluator(ma, device), 'b': NetEvaluator(mb, device)}, games, spec, conc, seed,
               threads, groups, max_steps, move_log, paired)
    out = dict({'a': a, 'b': b, 'game': game, 'sims': sims, 'chance_k': chance_k}, **summary(rs, paired))
    out.update(seconds=round(time.time() - t0, 1), select_a=_tag(sa), select_b=_tag(sb))
    _sides_sims(out, sims_a, sims_b)
    if paired:
        out['paired'] = True
    if rows:
        out['rows'] = sorted(rs, key=lambda r: r['gid'])
    return out


def _sides_sims(out, sims_a, sims_b):
    if sims_a is not None or sims_b is not None:
        out['sims_a'] = out['sims'] if sims_a is None else int(sims_a)
        out['sims_b'] = out['sims'] if sims_b is None else int(sims_b)


def sequential(a, b, batch_games=600, max_games=2400, sprt=None, alpha=0.05, beta=0.05, target_se=None, sims=64,
               chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None, spec=None, conc=None,
               seat_a=None, seat_b=None, max_steps=5000, paired=False, sims_a=None, sims_b=None, on_batch=None,
               rows=False):
    """`match` in batches of `batch_games` until decided: `sprt` (elo0, elo1) - the log-likelihood ratio
    (`sprt_llr`) leaves `sprt_bounds(alpha, beta)` - or `se` <= `target_se`, or `max_games` played. Batch k
    is seeded seed + 7919 k (batch 0: `seed`, the plain match's games). -> `match`'s dict over every game,
    plus 'batches' (games, a_score, se, llr per batch, cumulative), 'stop' (why it stopped) and with `sprt`
    'sprt' {elo0, elo1, alpha, beta, llr, lower, upper, decision 'H1' | 'H0' | 'undecided'}. `on_batch(dict)`:
    called after each batch with the cumulative result (progress lines). The game ids of batch k follow
    the earlier batches' (pairs stay 2j, 2j + 1)."""
    if sprt is None and target_se is None:
        raise ValueError('sequential: give sprt (elo0, elo1) and / or target_se')
    if paired and batch_games % 2:
        raise ValueError(f'paired games: an even batch ({batch_games})')
    if max_games < batch_games:
        raise ValueError(f'max_games {max_games} < batch_games {batch_games}')
    lo = hi = e0 = e1 = None
    if sprt is not None:
        e0, e1 = float(sprt[0]), float(sprt[1])
        if e1 <= e0:
            raise ValueError(f'sprt: elo1 ({e1}) must be > elo0 ({e0})')
        lo, hi = sprt_bounds(alpha, beta)
    ma, mb, game = _load_pair(a, b, device, game)
    spec = package.active().specs.get(game) if spec is None else spec
    sa, sb = _seats(a, b, ma, mb, game, sims, c_puct, chance_k, seat_a, seat_b, sims_a, sims_b)
    evs = {'a': NetEvaluator(ma, device), 'b': NetEvaluator(mb, device)}
    t0 = time.time()
    rs, batches, stop, llr = [], [], None, None
    k = 0
    while stop is None:
        n = min(batch_games, max_games - len(rs))
        if paired:
            n -= n % 2
        if n <= 0:
            stop = 'max_games'
            break
        got = _play(game, sa, sb, evs, n, spec, conc, seed + 7919 * k, threads, groups, max_steps, 0, paired)
        off = len(rs)
        rs += [dict(r, gid=r['gid'] + off) for r in got]
        cur = summary(rs, paired)
        row = {'games': len(rs), 'a_score': cur['a_score'], 'se': cur['se']}
        if sprt is not None:
            llr = sprt_llr(observations(rs, paired), e0, e1)
            row['llr'] = round(llr, 3)
            if llr >= hi:
                stop = 'sprt H1'
            elif llr <= lo:
                stop = 'sprt H0'
        if stop is None and target_se is not None and cur['se'] <= target_se:
            stop = 'target_se'
        if stop is None and len(rs) >= max_games:
            stop = 'max_games'
        batches.append(row)
        if on_batch is not None:
            on_batch(dict(row, seconds=round(time.time() - t0, 1), stop=stop))
        k += 1
    out = dict({'a': a, 'b': b, 'game': game, 'sims': sims, 'chance_k': chance_k}, **summary(rs, paired))
    out.update(seconds=round(time.time() - t0, 1), select_a=_tag(sa), select_b=_tag(sb), batches=batches, stop=stop)
    _sides_sims(out, sims_a, sims_b)
    if paired:
        out['paired'] = True
    if sprt is not None:
        out['sprt'] = {'elo0': e0, 'elo1': e1, 'alpha': alpha, 'beta': beta, 'llr': round(llr or 0.0, 3),
                       'lower': round(lo, 3), 'upper': round(hi, 3),
                       'decision': 'H1' if stop == 'sprt H1' else 'H0' if stop == 'sprt H0' else 'undecided'}
    if target_se is not None:
        out['target_se'] = target_se
    if rows:
        out['rows'] = sorted(rs, key=lambda r: r['gid'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--games', type=int, default=600)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--chance-k', type=int, default=2)
    ap.add_argument('--c-puct', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu', help="network device ('cuda' works; the GPU is shared)")
    ap.add_argument('--spec', default=None, help="game spec JSON (default: the game package's spec of the game)")
    ap.add_argument('--select', default=None, choices=['visits', 'cq'],
                    help="both sides' move rule (default: each checkpoint's run's eval_select)")
    ap.add_argument('--select-min-visits', type=int, default=0)
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-select', default=None, choices=['visits', 'cq'], help=f"side {side.upper()}'s move rule")
        ap.add_argument(f'--{side}-select-min-visits', type=int, default=None)
    ap.add_argument('--cq-c-visit', type=float, default=None,
                    help="the 'cq' rule's completed-Q transform, both sides (default: 50 / 1 with --select, else "
                         "each checkpoint's run's)")
    ap.add_argument('--cq-c-scale', type=float, default=None)
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-cq-c-visit', type=float, default=None,
                        help=f"side {side.upper()}'s 'cq' transform (over --cq-c-visit / --cq-c-scale)")
        ap.add_argument(f'--{side}-cq-c-scale', type=float, default=None)
    ap.add_argument('--move-log', type=int, default=0, help='log the first N moves of every game (with --rows)')
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-leaf-value-view', default=None, choices=['mover', 'root'],
                        help=f"side {side.upper()}'s leaf values' view (docs/search.md 3.6; default: its checkpoint's "
                             f"run's eval_leaf_value_view)")
        ap.add_argument(f'--{side}-belief-tilt', type=float, default=None,
                        help=f"side {side.upper()}'s root belief tilt (docs/search.md 3.8; 0 = uniform worlds; default: its "
                             f"checkpoint's run's eval_belief_tilt)")
        ap.add_argument(f'--{side}-belief-clip', type=float, default=None)
        ap.add_argument(f'--{side}-belief-mix', type=float, default=None)
        ap.add_argument(f'--{side}-belief-fast', type=int, default=None, choices=[0, 1])
    ap.add_argument('--torch-threads', type=int, default=1, help='torch intra-op threads (default 1)')
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-root-search', default=None, choices=['puct', 'gumbel'],
                        help=f"side {side.upper()}'s root search (docs/search.md 3.7; default: its checkpoint's run's "
                             f"eval_root_search)")
        ap.add_argument(f'--{side}-gumbel-scale', type=float, default=None,
                        help=f"side {side.upper()}'s Gumbel noise scale (default: its run's eval_gumbel_scale, 0)")
    ap.add_argument('--gumbel-m', type=int, default=None,
                    help="a Gumbel side's considered options (default: its run's gumbel_m, 16)")
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-sims', type=int, default=None, help=f"side {side.upper()}'s simulations (default --sims)")
    ap.add_argument('--paired', action='store_true',
                    help='games 2k and 2k + 1 share their deal with the seats swapped (docs/az.md 20.3; --games even)')
    ap.add_argument('--sprt', default=None, metavar='E0,E1',
                    help='sequential test of H0: A is E0 Elo better against H1: E1, in batches until decided or '
                         '--max-games (e.g. 0,20)')
    ap.add_argument('--alpha', type=float, default=0.05, help='SPRT false-H1 rate (default 0.05)')
    ap.add_argument('--beta', type=float, default=0.05, help='SPRT false-H0 rate (default 0.05)')
    ap.add_argument('--target-se', type=float, default=None, help='sequential: stop once se <= this')
    ap.add_argument('--batch-games', type=int, default=None, help='sequential: games per batch (default --games)')
    ap.add_argument('--max-games', type=int, default=None,
                    help='sequential: at most this many games (default 4 x --games)')
    a = ap.parse_args(argv)
    sprt = None
    if a.sprt is not None:
        try:
            sprt = tuple(float(x) for x in a.sprt.split(','))
            if len(sprt) != 2 or sprt[1] <= sprt[0]:
                raise ValueError
        except ValueError:
            ap.error(f'--sprt {a.sprt!r}: E0,E1 with E1 > E0 (e.g. 0,20)')
    seq = sprt is not None or a.target_se is not None
    if not seq and (a.batch_games is not None or a.max_games is not None):
        ap.error('--batch-games / --max-games: only with --sprt or --target-se')
    for k in ('games', 'sims', 'a_sims', 'b_sims', 'batch_games', 'max_games'):
        v = getattr(a, k)
        if v is not None and v < 1:
            ap.error(f'--{k.replace("_", "-")} must be >= 1')
    if a.paired and (a.batch_games or a.games) % 2:
        ap.error('--paired: an even number of games (per batch)')
    if seq and (a.max_games or 4 * a.games) < (a.batch_games or a.games):
        ap.error('--max-games < the games of one batch')
    import torch
    torch.set_num_threads(max(1, a.torch_threads))
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass
    sides = []
    for side, ckpt in (('a', a.a), ('b', a.b)):
        rule = getattr(a, f'{side}_select') or a.select
        mv = getattr(a, f'{side}_select_min_visits')
        mv = a.select_min_visits if mv is None else mv
        own = getattr(a, f'{side}_cq_c_visit'), getattr(a, f'{side}_cq_c_scale')
        both = a.cq_c_visit, a.cq_c_scale
        base = side_keys(ckpt, rule, mv)
        if base.get('select', 'visits') == 'visits':
            if own != (None, None):
                ap.error(f"--{side}-cq-c-*: side {side.upper()} plays 'visits' (no completed-Q transform)")
            sides.append(base)
            continue
        cv, cs = (o if o is not None else b for o, b in zip(own, both))
        sides.append(side_keys(ckpt, rule, mv, cv, cs))
    if (a.cq_c_visit, a.cq_c_scale) != (None, None) and not any(s.get('select') == 'cq' for s in sides):
        ap.error("--cq-c-visit / --cq-c-scale: no side plays 'cq'")
    for side, keys in zip(('a', 'b'), sides):
        view = getattr(a, f'{side}_leaf_value_view')
        if view == 'mover':
            keys.pop('leaf_value_view', None)
        elif view == 'root':
            keys['leaf_value_view'] = 'root'
    try:
        # (a side's root belief is its run's eval_belief_tilt even when --select replaced its rule, so an
        # A/B arm whose config sets it is dealt as it trained; the side's flags override it)
        sides = [with_belief(dict(k, **{x: v for x, v in run_rule(ckpt).items() if x.startswith('belief_')}),
                             *(getattr(a, f'{s}_belief_{x}') for x in ('tilt', 'clip', 'mix', 'fast')))
                 for s, k, ckpt in zip(('a', 'b'), sides, (a.a, a.b))]
    except ValueError as e:
        ap.error(str(e))
    for k, (side, ckpt) in enumerate((('a', a.a), ('b', a.b))):
        try:
            sides[k] = root_keys(ckpt, sides[k], getattr(a, f'{side}_root_search'), a.gumbel_m,
                                 getattr(a, f'{side}_gumbel_scale'))
        except ValueError as e:
            ap.error(f'side {side.upper()}: {e}')
        if getattr(a, f'{side}_gumbel_scale') is not None and sides[k].get('root_search') != 'gumbel':
            ap.error(f"--{side}-gumbel-scale: side {side.upper()} searches with PUCT")
    if seq:
        out = sequential(a.a, a.b, a.batch_games or a.games, a.max_games or 4 * a.games, sprt, a.alpha, a.beta,
                         a.target_se, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device, spec=a.spec,
                         seat_a=sides[0], seat_b=sides[1], paired=a.paired, sims_a=a.a_sims, sims_b=a.b_sims,
                         on_batch=lambda r: print('# batch ' + json.dumps(r), file=sys.stderr, flush=True))
        print(json.dumps(out), flush=True)
        return
    print(json.dumps(match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device,
                           spec=a.spec, seat_a=sides[0], seat_b=sides[1], move_log=a.move_log, paired=a.paired,
                           sims_a=a.a_sims, sims_b=a.b_sims)), flush=True)


if __name__ == '__main__':
    main()
