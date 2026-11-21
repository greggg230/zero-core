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
'cq'. The game comes from checkpoint A (`Net.config['game']`). Unless --spec is given the games
are played at A's run's evaluation spec, `Run.driver`'s rule: its `eval_spec`, else the game
package's eval spec of its self-play spec (`GamePackage.eval_specs`; MTG Zero: its fixed evaluation
decks) - without a run config.json, the game package's default spec (`GamePackage.specs`, e.g.
Caverna {"full": true}, Glass Road {"intro": false}). Each side's seat also takes its run's
`margin_w`, `redet` and `eval_cache` (`Run.nn_seat`'s; without a run config margin_w 0, redet on, no
cache), so an h2h of a run's checkpoints plays the games of its evaluation and of a zero.farm_h2h
at the same settings (docs/az.md 44.3; before 2026-11-12 these four were the package's defaults).
Prints one JSON line: A's score (+- 1 se), its mean score margin, the Elo difference, each
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

Information-set merging (docs/search.md 3.16, docs/az.md 30): --a-infoset-merge / --b-infoset-merge 0 | 1 (or
--infoset-merge for both) - each side's trees merge their nodes by the root decider's observation, or not (default:
its run's eval_infoset_merge). A merged side needs root-view leaf values (its run's eval_leaf_value_view 'root' or
--*-leaf-value-view root). `CK CK --a-infoset-merge 1 --b-infoset-merge 0` is the evaluation-only A/B of the merge
on one checkpoint; the side's rule name carries it ('-ism') and the output adds `merge` (the merged trees' links,
merges, aliases and growth leaves, summed over the drivers).

Per-side simulations (docs/az.md 20.4): --a-sims / --b-sims (default --sims), e.g. `A A --a-sims 128
--b-sims 32` (the output then names `sims_a`, `sims_b`).

One meaning with the farm (docs/az.md 68): the per-side options above are zero.seat_settings' (`add_side_args`; also
--a-gumbel-m / --b-gumbel-m over --gumbel-m), and a side's seat is its run's settings with its options as run settings
over them (`seat_settings.side_set`), `Run.nn_seat` of those (`sides_of`) - the seat a farm eval job's `{"net": gen,
"set": ...}` plays (zero.farm_h2h, MTG Zero's farm campaigns take the same options). So --select / --a-select replace
the move rule only (the side keeps its run's root minimum visits, leaf view, merging and value choices), a
--a-select-min-visits without a rule guards the side's own rule, and every combination Run.nn_seat refuses is refused
here too. --graphs and --lookahead are local only (no run setting: torch's CUDA graphs, the Python lookahead).

CUDA graphs (docs/az.md 10.6): --graphs 0 | 1 for both sides, --a-graphs / --b-graphs for one (over
--graphs); default: each side's NetEvaluator default (a conv network's on, a token network's env
ZERO_TOKEN_GRAPHS=1). Given, the output names `graphs_a` / `graphs_b` (each side's CUDA-graph
batches over its batches: 1.0 = every batch). `A A --a-graphs 1 --b-graphs 0` is the equivalence
check of a token network's graphs against its eager inference.

The exhaustive lookahead (docs/search.md 3.9): --a-lookahead / --b-lookahead SETTINGS (or --lookahead
for both) make a side a lookahead seat - its decisions decided by the lookahead per information set
(`zero.lookahead.parse`: e.g. `plies=6,worlds=8,topk=0,opp=best,mode=infoset,budget=200000`), its worlds
dealt by its seat's root belief, its final fallback (every budget stage over) its seat's MCTS search
with the settings above. The output adds `lookahead_a` / `lookahead_b` (the canonical settings),
`look` (the driver's lookahead counters: decisions, nodes, rows, fallbacks per stage...) and
`max_rows`: batches over --max-rows rows (default 1024 with a lookahead side, else whole) are evaluated
in pieces (`NetEvaluator(max_rows=)`, the CUDA graphs' shapes bounded).

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

Where the games are played (docs/az.md 69): --backend auto | farm | local (default auto). A game package may name
another place to play a match (`GamePackage.h2h_backend`; MTG Zero: zero-farm campaigns, its docs/ENGINE.md 217). Under
'auto' zero.h2h asks it, before any game, whether it can play exactly this match (its conditions: the package's) and
plays there if so, else here - never a fallback in the middle of a match; one stderr line says which and why
('h2h: backend local - --lookahead is local-only', 'h2h: backend farm - campaign ...'). 'farm': there, or refused (exit
2 with the reason). 'local': here, as before. A package without one plays here ('auto': its line says so). The JSON
line names its backend (`backend` 'local' | 'farm'); a farm match prints the same keys as a local one (the summary,
`select_a` / `select_b`, `sims_a` / `sims_b`, `batches` / `stop` / `sprt`, `paired`; `merge` null - the local
driver's counters - and none of the local-only `graphs_*`, `look`, `precision`, `sparse`) plus the backend's own
(`remote_match`), and --records keeps its games from their traced records in the local records' format. The two
backends' games are not bit-identical (torch here, ONNX Runtime there): compare series of one backend.
"""
import argparse
import json
import math
import os
import sys
import time

from . import gamerec as GR, games as G, lookahead as LA, package, seat_settings as SS
from .infer import NetEvaluator, drive
from .mcts import (belief_settings, belief_tag, belief_tilts, check_belief, check_belief_tilts, check_merge,
                   check_root_search, check_select, check_select_transform, forced_tag, gumbel_tag, merge_tag, select_settings, select_tag, select_transform,
                   view_tag)
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


def run_settings(ckpt):
    """A checkpoint's run's settings resolved as the run resolves them (`zero.run.resolve_config`: DEFAULTS < its
    preset < its config.json, as zero.farm_h2h does) - {} without a config.json. A config that no longer resolves
    (e.g. a preset since removed) gives its saved values over DEFAULTS, with a note on stderr."""
    c = run_config(ckpt)
    if not c:
        return {}
    from . import run as R
    try:
        return R.resolve_config(argparse.Namespace(**{k: None for k in R.DEFAULTS}), c)
    except (SystemExit, ValueError) as e:
        print(f'# note: the run config of {ckpt} does not resolve ({e}): its saved values over the defaults',
              file=sys.stderr, flush=True)
        return dict(R.DEFAULTS, **c)


def run_seat_keys(ckpt):
    """The evaluation seat keys `Run.nn_seat` takes from a checkpoint's run besides its move rule (`run_rule`):
    margin_w, redet, eval_cache (docs/az.md 44.3) - {} without a run config (the seat of before: margin_w 0, redet
    on, no cache)."""
    c = run_settings(ckpt)
    if not c:
        return {}
    return {'margin_w': c['margin_w'], 'redet': bool(c['redet']), 'eval_cache': c.get('eval_cache', 0)}


def run_eval_spec(ckpt, game):
    """The spec of a checkpoint's run's evaluation matches, `Run.driver`'s rule (its eval_spec, else the game
    package's eval spec of its self-play spec); without a run config the game package's default spec of `game`."""
    c = run_settings(ckpt)
    if not c:
        return package.active().specs.get(game)
    return c.get('eval_spec') or package.active().eval_spec_of(game, c['spec'])


def match_spec(a, b, game, spec=None):
    """The match's spec: `spec` if given, else A's run's evaluation spec (`run_eval_spec`; a note on stderr when
    B's run evaluates at another one)."""
    if spec is not None:
        return spec
    sa = run_eval_spec(a, game)
    if run_config(b) and G.spec_json(run_eval_spec(b, game)) != G.spec_json(sa):
        print(f"# note: B's run evaluates at another spec than A's; the match plays A's ({G.spec_json(sa)[:120]})",
              file=sys.stderr, flush=True)
    return sa


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
    if c.get('eval_infoset_merge'):
        out['infoset_merge'] = True                # (docs/search.md 3.16)
    if c.get('eval_value_choice_worlds'):
        # (value choices, docs/search.md 3.17)
        out.update(value_choice_worlds=int(c['eval_value_choice_worlds']),
                   value_choice_temp=float(c.get('eval_value_choice_temp', 0.0) or 0.0))
    tilt, clip, mix, fast = belief_settings(c, 'eval')
    tilts = belief_tilts(c, 'eval')
    if tilt or tilts:
        out.update(belief_tilt=tilt, belief_clip=clip, belief_mix=mix, belief_fast=fast)
    if tilts:
        out['belief_tilts'] = tilts                # (per-target tilts, docs/search.md 3.8.1)
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
    from .mcts import value_choice_tag
    ism = merge_tag(seat.get('infoset_merge', False)) + value_choice_tag(seat.get('value_choice_worlds', 0),
                                                                         seat.get('value_choice_temp', 0.0))
    if seat.get('root_search', 'puct') == 'gumbel':
        return gumbel_tag(seat.get('gumbel_m', 16), seat.get('gumbel_c_visit', 50.0), seat.get('gumbel_c_scale', 1.0),
                          seat.get('gumbel_scale', 1.0)) + view_tag(seat.get('leaf_value_view', 'mover')) + belief_tag(
            seat.get('belief_tilt', 0.0), seat.get('belief_clip'), seat.get('belief_mix', 0.0), seat.get('belief_fast', False), seat.get('belief_tilts')) + ism
    cv, cs = select_transform(seat.get('cq_c_visit', 50.0), seat.get('cq_c_scale', 1.0), seat.get('select_c_visit'),
                              seat.get('select_c_scale'))
    return select_tag(seat.get('select', 'visits'), seat.get('select_min_visits', 0), cv, cs) + forced_tag(
        seat.get('root_min_visits', 0), seat.get('root_min_visits_max_actions', 4)) + view_tag(
        seat.get('leaf_value_view', 'mover')) + belief_tag(seat.get('belief_tilt', 0.0), seat.get('belief_clip'),
                                                           seat.get('belief_mix', 0.0), seat.get('belief_fast', False), seat.get('belief_tilts')) + ism


MERGE_STATS = ('links', 'merged', 'aliases', 'growths')


def _merge_out(out, sa, sb, stats):
    """`merge` in a match's output when a side merges its trees (docs/search.md 3.16): the drivers' merged-tree
    counters summed (links, merges, aliases, growth leaves); a merged side whose trees made no link is reported
    on stderr (a setting that did nothing)."""
    if not (sa.get('infoset_merge') or sb.get('infoset_merge')):
        return
    m = {k: sum(int(st.get(f'merge_{k}', 0)) for st in stats) for k in MERGE_STATS}
    out['merge'] = m
    if m['links'] == 0:
        print('# WARNING: infoset_merge on but the merged trees made no link (docs/search.md 3.16)', file=sys.stderr,
              flush=True)


def with_belief(keys, tilt=None, clip=None, mix=None, fast=None, tilts=None):
    """A side's seat keys with its root belief (docs/search.md 3.8) as given (None: as it was): tilt 0
    removes it (uniform worlds) unless the side keeps per-target tilts; `tilts` ({target: alpha}, 3.8.1)
    replaces the side's per-target tilts ({} removes them); clip / mix / fast change a side that has a
    belief (or is given one)."""
    keys = dict(keys)
    if tilts is not None:
        check_belief_tilts(tilts, what='--*-belief-tilts')
        tilts = {k: float(v) for k, v in sorted(tilts.items()) if v}
        keys.pop('belief_tilts', None)
        if tilts:
            keys['belief_tilts'] = tilts
    if tilt is not None:
        keys['belief_tilt'] = float(tilt)
    if not keys.get('belief_tilt') and not keys.get('belief_tilts'):
        for k in ('belief_tilt', 'belief_clip', 'belief_mix', 'belief_fast'):
            keys.pop(k, None)
        if tilt is not None:
            return keys                            # (tilt 0 given: uniform worlds, as before)
    else:
        keys.setdefault('belief_tilt', 0.0)
        keys.setdefault('belief_clip', 20.0)
        keys.setdefault('belief_mix', 0.0)
        keys.setdefault('belief_fast', False)
    if not keys.get('belief_tilt') and not keys.get('belief_tilts'):
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
    # (each side's run's margin_w, redet and eval_cache: Run.nn_seat's evaluation seat, docs/az.md 44.3)
    sa = G.nn_seat('a', sims if sims_a is None else int(sims_a), c_puct, chance_k=chance_k, **run_seat_keys(a))
    sb = G.nn_seat('b', sims if sims_b is None else int(sims_b), c_puct, chance_k=chance_k, **run_seat_keys(b))
    sa.update(run_rule(a) if seat_a is None else seat_a)
    sb.update(run_rule(b) if seat_b is None else seat_b)
    for s, m, p in ((sa, ma, a), (sb, mb, b)):
        if s.get('belief_tilt') or s.get('belief_tilts'):
            check_belief_head(m, game, f'belief_tilt / belief_tilts of {p}')     # (before any game: a loud refusal)
    return sa, sb


def _rec(a, b, sa, sb):
    """The recording of a match's games (docs/az.md 59.3): (sink, A's player, B's player) while the thread records
    (`gamerec.recording`), else None (the match of before)."""
    sink = GR.active()
    if sink is None:
        return None
    return (sink, GR.net_player(a, sink.labels.get('a'), sa, _tag(sa)),
            GR.net_player(b, sink.labels.get('b'), sb, _tag(sb)))


def _play(game, sa, sb, evs, games, spec, conc, seed, threads, groups, max_steps, move_log, paired, stats_out=None,
          sparse=False, rec=None, rec_meta=None):
    seats = [[sa, sb] if g % 2 == 0 else [sb, sa] for g in range(games)]
    kw = {'move_log': move_log} if move_log else {}
    if paired:
        kw['paired'] = True
    if sparse:
        kw['sparse'] = True
    drv = G.Driver(game, seats, spec=spec, concurrency=conc or min(games, 1024), seed=seed, threads=threads,
                   groups=groups, max_steps=max_steps, **kw)
    if rec is not None:
        drv.set_game_log(True)                     # (game records, docs/az.md 59.3: the games themselves unchanged)
    drive(drv, evs)
    if stats_out is not None:
        stats_out.append(drv.stats())
    rows = drv.take_results()
    if rec is not None:
        sink, pa, pb = rec
        # (A in seat gid % 2; the rows' logs taken out: the caller's rows are a match's without records)
        sink.add_rows(rows, lambda g: [pa, pb] if g % 2 == 0 else [pb, pa],
                      dict({'driver_seed': int(seed), 'paired': bool(paired)}, **(rec_meta or {})))
    return rows


LOOK_SUMS = ('decisions', 'nodes', 'leaves', 'prior_rows', 'value_rows', 'value_dups', 'minor_steps', 'groups', 'cut', 'rounds', 'rows',
             'fallback_stage', 'fallback_last', 'fallback_mcts', 'minor_root', 'aborted_nodes', 'busy_s', 'guard_decisions',
             'guard_overrides')


def look_summary(stats):
    """The lookahead counters of the drivers' `stats()` (docs/search.md 3.9), summed (by_stage per stage,
    peak_games the max), with per-decision means; None when no driver had a lookahead decision."""
    ls = [st['look'] for st in stats if st.get('look')]
    if not ls:
        return None
    out = {k: sum(x[k] for x in ls) for k in LOOK_SUMS}
    out['busy_s'] = round(out['busy_s'], 2)
    out['peak_games'] = max(x['peak_games'] for x in ls)
    by = []
    for x in ls:
        for i, v in enumerate(x['by_stage']):
            if i >= len(by):
                by.append(0)
            by[i] += v
    out['by_stage'] = by
    d = max(1, out['decisions'])
    out['per_decision'] = {k: round(out[k] / d, 1) for k in ('nodes', 'leaves', 'rows', 'rounds')}
    return out


def look_keys(spec):
    """A side's lookahead seat key from its command-line settings (`lookahead.parse`; None: none)."""
    return None if spec is None else LA.parse(spec)


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
    # a game that chains its episodes (docs/az.md 42): the rows above are contests (matches), A's games beside them
    pg = G.summarize_games([g for r in rs for g in G.episode_rows(r, r['gid'] % 2)])
    if pg is not None:
        out['per_game'] = pg
    return out


def _evaluators(ma, mb, device, graphs, max_rows=None, precision=None):
    """Each side's NetEvaluator; `graphs` None (each one's default), a bool for both, or (a, b) (None:
    that side's default); `max_rows`: batches evaluated in pieces of at most that many rows; `precision` both sides'
    arithmetic (NetEvaluator's, the run setting infer_precision: docs/az.md 57; None: 'fp32')."""
    ga, gb = graphs if isinstance(graphs, (tuple, list)) else (graphs, graphs)
    kw = {} if max_rows is None else {'max_rows': int(max_rows)}
    if precision is not None and precision != 'fp32':
        kw['precision'] = precision
    return {'a': NetEvaluator(ma, device, graphs=ga, **kw), 'b': NetEvaluator(mb, device, graphs=gb, **kw)}


def _max_rows(sa, sb, max_rows):
    """The evaluators' max_rows: as given, else 1024 when a side is a lookahead seat (its batches hold
    up to `rows` per game in flight), else None (every batch whole)."""
    if max_rows is not None:
        return int(max_rows)
    return 1024 if (sa.get('lookahead') or sb.get('lookahead')) else None


def _look_out(out, sa, sb, stats, max_rows, evs=None):
    """The lookahead fields of a match's output (only with a lookahead side or max_rows); with a
    lookahead side also `perf` (where the time went: the drivers' search / assembly / the caller's
    wait, the evaluators' seconds, calls, rows)."""
    for k, sd in (('a', sa), ('b', sb)):
        if sd.get('lookahead'):
            out[f'lookahead_{k}'] = LA.settings(sd['lookahead'])
            out[f'lookahead_tag_{k}'] = LA.tag(sd['lookahead'])
    ls = look_summary(stats)
    if ls is not None:
        out['look'] = ls
        perf = {k: round(sum(st[k] for st in stats), 2) for k in ('busy_s', 'assemble_s', 'wait_s', 'wall_s')}
        perf.update({k: sum(st[k] for st in stats) for k in ('batches', 'leaves', 'sims', 'belief_rows', 'view_rows')})
        perf['threads'] = stats[0]['threads']
        for k, ev in (evs or {}).items():
            perf[f'eval_{k}'] = {'seconds': round(ev.seconds, 2), 'calls': ev.calls, 'rows': ev.leaves,
                                 'graph_calls': ev.graph_calls, 'pieces': ev.pieces}
        out['perf'] = perf
    if max_rows is not None:
        out['max_rows'] = max_rows


def _precision_out(out, evs):
    """`precision`: the sides' network arithmetic when it is not float32 (docs/az.md 57)."""
    p = {getattr(ev, 'precision', 'fp32') for ev in evs.values()}
    if p != {'fp32'}:
        out['precision'] = '/'.join(sorted(p))


def _graphs_used(out, evs, graphs):
    """`graphs_a` / `graphs_b`: each side's CUDA-graph batches over its batches (with `graphs` given)."""
    if graphs is None or (isinstance(graphs, (tuple, list)) and all(g is None for g in graphs)):
        return
    for k in ('a', 'b'):
        ev = evs[k]
        out[f'graphs_{k}'] = round(ev.graph_calls / ev.calls, 4) if ev.calls else None


def match(a, b, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None, seat_a=None, seat_b=None, max_steps=5000, move_log=0, rows=False, paired=False,
          sims_a=None, sims_b=None, graphs=None, max_rows=None, sparse=None, precision=None):
    """Checkpoint `a` vs checkpoint `b` (paths), `games` 2-player games, seats alternating. -> dict.
    `seat_a` / `seat_b`: each side's move-rule seat settings ({'select': 'cq', 'select_min_visits':
    2, ...}; {} = the default rule; None = its checkpoint's run's eval_select, `run_rule`).
    `move_log`: the driver's move log (the first N moves of each game in its row), `rows`: the
    games' result rows under 'rows'. `paired`: games 2k and 2k + 1 share their deal with the seats
    swapped (`games` even; the output's `pairs`, `se` of the pair scores: `summary`). `sims_a` /
    `sims_b`: a side's simulations (None: `sims`). `graphs`: the sides' CUDA graphs (`_evaluators`;
    given: `graphs_a` / `graphs_b` in the output). A seat dict with `lookahead` is a lookahead side
    (docs/search.md 3.9: the output's `lookahead_*`, `look`); `max_rows` the evaluators' (`_max_rows`);
    `sparse` the driver's sparse batches (None: with a lookahead side; docs/search.md 2.2). `precision`: both sides'
    network arithmetic ('fp32' | 'tf32' | 'bf16', docs/az.md 57; None: 'fp32'; another is named in the output)."""
    if paired and games % 2:
        raise ValueError(f'paired games: an even number of games ({games})')
    ma, mb, game = _load_pair(a, b, device, game)
    spec = match_spec(a, b, game, spec)            # (A's run's evaluation spec unless given: Run.driver's, docs/az.md 44.3)
    sa, sb = _seats(a, b, ma, mb, game, sims, c_puct, chance_k, seat_a, seat_b, sims_a, sims_b)
    mr = _max_rows(sa, sb, max_rows)
    sp = _sparse(sa, sb, sparse)
    evs = _evaluators(ma, mb, device, graphs, mr, precision)
    t0 = time.time()
    st = []
    rs = _play(game, sa, sb, evs, games, spec, conc, seed, threads, groups, max_steps, move_log, paired, st, sp,
               _rec(a, b, sa, sb))
    out = dict({'a': a, 'b': b, 'game': game, 'sims': sims, 'chance_k': chance_k}, **summary(rs, paired))
    out.update(seconds=round(time.time() - t0, 1), select_a=_tag(sa), select_b=_tag(sb))
    _sides_sims(out, sims_a, sims_b)
    _graphs_used(out, evs, graphs)
    _look_out(out, sa, sb, st, mr, evs)
    _merge_out(out, sa, sb, st)
    _precision_out(out, evs)
    if sp:
        out['sparse'] = True
    if paired:
        out['paired'] = True
    if rows:
        out['rows'] = sorted(rs, key=lambda r: r['gid'])
    return out


def _sparse(sa, sb, sparse):
    """The driver's sparse batches: as given, else on with a lookahead side (its batches are bound by
    the host's copies of the dense rows), else off (the batches of before)."""
    if sparse is not None:
        return bool(sparse)
    return bool(sa.get('lookahead') or sb.get('lookahead'))


def _sides_sims(out, sims_a, sims_b):
    if sims_a is not None or sims_b is not None:
        out['sims_a'] = out['sims'] if sims_a is None else int(sims_a)
        out['sims_b'] = out['sims'] if sims_b is None else int(sims_b)


def seq_bounds(sprt=None, alpha=0.05, beta=0.05, target_se=None, paired=False, batch_games=600, max_games=2400):
    """A sequential match's arguments checked (`sequential`'s rules) -> (elo0, elo1, lower, upper): the SPRT's
    hypotheses and LLR bounds (None without `sprt`); ValueError for a combination refused."""
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
    return e0, e1, lo, hi


def batch_loop(play, batch_games, max_games, paired, bounds, target_se=None, on_batch=None, t0=None):
    """zero.h2h's sequential schedule (docs/az.md 20.3) over `play(k, n, gid_offset)` -> batch k's rows (its n games,
    ids 0..n-1; the caller seeds batch k seed + 7919 k): batches until the SPRT of `bounds` (`seq_bounds`) decides,
    the score's se <= `target_se`, or `max_games` are played -> (rows over every batch - ids following the earlier
    batches', pairs kept 2j, 2j + 1 -, batches [{games, a_score, se, llr}], stop, llr). Local matches (`sequential`)
    and another backend's (docs/az.md 69) run this one loop."""
    e0, e1, lo, hi = bounds
    t0 = time.time() if t0 is None else t0
    rs, batches, stop, llr = [], [], None, None
    k = 0
    while stop is None:
        n = min(batch_games, max_games - len(rs))
        if paired:
            n -= n % 2
        if n <= 0:
            stop = 'max_games'
            break
        got = play(k, n, len(rs))
        off = len(rs)
        rs += [dict(r, gid=r['gid'] + off) for r in got]
        cur = summary(rs, paired)
        row = {'games': len(rs), 'a_score': cur['a_score'], 'se': cur['se']}
        if e0 is not None:
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
    return rs, batches, stop, llr


def _seq_tail(out, bounds, alpha, beta, llr, stop, target_se):
    """A sequential match's `sprt` (with an SPRT) and `target_se` fields."""
    e0, e1, lo, hi = bounds
    if e0 is not None:
        out['sprt'] = {'elo0': e0, 'elo1': e1, 'alpha': alpha, 'beta': beta, 'llr': round(llr or 0.0, 3),
                       'lower': round(lo, 3), 'upper': round(hi, 3),
                       'decision': 'H1' if stop == 'sprt H1' else 'H0' if stop == 'sprt H0' else 'undecided'}
    if target_se is not None:
        out['target_se'] = target_se


def sequential(a, b, batch_games=600, max_games=2400, sprt=None, alpha=0.05, beta=0.05, target_se=None, sims=64,
               chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None, spec=None, conc=None,
               seat_a=None, seat_b=None, max_steps=5000, paired=False, sims_a=None, sims_b=None, on_batch=None,
               rows=False, graphs=None, max_rows=None, sparse=None, precision=None):
    """`match` in batches of `batch_games` until decided: `sprt` (elo0, elo1) - the log-likelihood ratio
    (`sprt_llr`) leaves `sprt_bounds(alpha, beta)` - or `se` <= `target_se`, or `max_games` played. Batch k
    is seeded seed + 7919 k (batch 0: `seed`, the plain match's games). -> `match`'s dict over every game,
    plus 'batches' (games, a_score, se, llr per batch, cumulative), 'stop' (why it stopped) and with `sprt`
    'sprt' {elo0, elo1, alpha, beta, llr, lower, upper, decision 'H1' | 'H0' | 'undecided'}. `on_batch(dict)`:
    called after each batch with the cumulative result (progress lines). The game ids of batch k follow
    the earlier batches' (pairs stay 2j, 2j + 1). The schedule is `batch_loop`'s."""
    bounds = seq_bounds(sprt, alpha, beta, target_se, paired, batch_games, max_games)
    ma, mb, game = _load_pair(a, b, device, game)
    spec = match_spec(a, b, game, spec)            # (A's run's evaluation spec unless given: Run.driver's, docs/az.md 44.3)
    sa, sb = _seats(a, b, ma, mb, game, sims, c_puct, chance_k, seat_a, seat_b, sims_a, sims_b)
    mr = _max_rows(sa, sb, max_rows)
    sp = _sparse(sa, sb, sparse)
    evs = _evaluators(ma, mb, device, graphs, mr, precision)
    t0 = time.time()
    st = []
    rec = _rec(a, b, sa, sb)

    def play(k, n, off):
        return _play(game, sa, sb, evs, n, spec, conc, seed + 7919 * k, threads, groups, max_steps, 0, paired, st, sp,
                     rec, {'batch': k, 'gid_offset': off})
    rs, batches, stop, llr = batch_loop(play, batch_games, max_games, paired, bounds, target_se, on_batch, t0)
    out = dict({'a': a, 'b': b, 'game': game, 'sims': sims, 'chance_k': chance_k}, **summary(rs, paired))
    out.update(seconds=round(time.time() - t0, 1), select_a=_tag(sa), select_b=_tag(sb), batches=batches, stop=stop)
    _sides_sims(out, sims_a, sims_b)
    _graphs_used(out, evs, graphs)
    _look_out(out, sa, sb, st, mr, evs)
    _merge_out(out, sa, sb, st)
    _precision_out(out, evs)
    if sp:
        out['sparse'] = True
    if paired:
        out['paired'] = True
    _seq_tail(out, bounds, alpha, beta, llr, stop, target_se)
    if rows:
        out['rows'] = sorted(rs, key=lambda r: r['gid'])
    return out


def parser():
    """zero.h2h's command line (its per-side options: zero.seat_settings.add_side_args, shared with the farm's tools)."""
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
    ap.add_argument('--spec', default=None, help="game spec JSON (default: A's run's evaluation spec - its eval_spec, else "
                                                  "the game package's eval spec of its spec -, without a run config the "
                                                  "game package's spec of the game)")
    ap.add_argument('--move-log', type=int, default=0, help='log the first N moves of every game (with --rows)')
    SS.add_side_args(ap, local=True)
    ap.add_argument('--torch-threads', type=int, default=1, help='torch intra-op threads (default 1)')
    ap.add_argument('--precision', default=None, choices=['fp32', 'tf32', 'bf16'],
                    help="both sides' network arithmetic on CUDA (the run setting infer_precision, docs/az.md 57; "
                         "default fp32)")
    ap.add_argument('--conc', type=int, default=None,
                    help='games in flight at once (default: every game, at most 1024)')
    ap.add_argument('--sparse', type=int, default=None, choices=[0, 1],
                    help="the driver's sparse batches (default: on with a lookahead side, else off; docs/search.md 2.2)")
    ap.add_argument('--groups', type=int, default=2,
                    help="the driver's groups (default 2: one searches while another is evaluated)")
    ap.add_argument('--max-rows', type=int, default=None,
                    help='evaluate batches over this many rows in pieces (default: 1024 with a lookahead side, else whole)')
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
    ap.add_argument('--records', default=None, metavar='DIR',
                    help='keep every game as a game record (docs/az.md 59.3) under DIR/<--records-kind>/misc/ (a game '
                         'database ingests them: MTG Zero tools/gamedb.py ingest --records DIR); the games are the same')
    ap.add_argument('--records-kind', default='h2h', help="the records' kind (default 'h2h')")
    ap.add_argument('--label-a', default=None, help="side A's name in the records (default: its checkpoint id)")
    ap.add_argument('--label-b', default=None, help="side B's name in the records")
    ap.add_argument('--backend', default='auto', choices=BACKENDS,
                    help="where the games are played (docs/az.md 69): 'auto' (default) - the game package's other "
                         "backend (MTG Zero: the farm) when it can play exactly this match, else here, decided before "
                         "any game and said in one line; 'farm' - there or refused; 'local' - here")
    be = package.active().h2h_backend_of()
    if be is not None:
        be.add_args(ap)                            # (the backend's own options, e.g. MTG Zero's --farm-*)
    return ap


def side_settings(ckpt, sims, c_puct=1.0, chance_k=2):
    """A side's settings: its checkpoint's run's (`run_settings`; without a run config the evaluation seat's own
    defaults - margin_w 0, redet on, no cache, the 'visits' rule) with the match's eval_sims, c_puct and chance_k."""
    c = run_settings(ckpt) or {'margin_w': 0.0, 'redet': 1}
    return dict(c, eval_sims=int(sims), c_puct=c_puct, chance_k=chance_k)


def settings_seat(cfg, key='a'):
    """The evaluation seat of settings `cfg` (`Run.nn_seat(key, cfg['eval_sims'])`), without its key."""
    from . import run as R
    from .farm_eval import _Stub
    s = R.Run.nn_seat(_Stub(cfg), key, cfg['eval_sims'])
    s.pop('key', None)
    return s


def sides_of(a):
    """Both sides' seat dicts of a command line (`a`: `parser()`'s arguments): each side's settings (`side_settings`)
    with its per-side options as run settings over them (zero.seat_settings.side_set - the meaning a farm head-to-head's
    seat sets have too, docs/az.md 68), that settings' evaluation seat (Run.nn_seat), and its lookahead seat key
    -> [seat_a, seat_b]; ValueError for a combination refused."""
    cfgs = [side_settings(ck, a.sims, a.c_puct, a.chance_k) for ck in (a.a, a.b)]
    sets = SS.sets_of(a, cfgs[0], cfgs[1])
    out = []
    for k, side in enumerate('ab'):
        seat = settings_seat(SS.seat_cfg(cfgs[k], sets[k] or None), side)
        try:
            lk = look_keys(getattr(a, f'{side}_lookahead') or a.lookahead)
        except ValueError as e:
            raise ValueError(f'side {side.upper()}: {e}') from None
        if lk is not None:
            seat['lookahead'] = lk
        out.append(seat)
    return out


BACKENDS = ('auto', 'farm', 'local')


class Refused(ValueError):
    """--backend farm, and the match cannot be played there (the reason)."""


def backend_choice(a, sides, seq):
    """Where a command line's match is played (docs/az.md 69), decided before any game -> (name, why, runner):
    ('local', why, None) - here (`why`: asked, or what keeps it off the other backend) - or ('farm', what, runner)
    (the game package's `h2h_backend` plans it). Refused under --backend farm when it cannot."""
    if a.backend == 'local':
        return 'local', 'asked (--backend local)', None
    pkg = package.active()
    be = pkg.h2h_backend_of()
    if be is None or getattr(be, 'name', None) != 'farm':
        why = f'{pkg.title} has no farm backend (GamePackage.h2h_backend)'
        if a.backend == 'farm':
            raise Refused(why)
        return 'local', why, None
    ok, why, runner = be.plan(a, a.backend, sides, seq)
    if ok:
        return be.name, why, runner
    if a.backend == 'farm':
        raise Refused(why)
    return 'local', why, None


def _sink_of(a, ap):
    if not a.records:
        return None
    try:
        return GR.MatchSink(a.records, a.records_kind, labels={'a': a.label_a, 'b': a.label_b}, meta={'tool': 'h2h'})
    except ValueError as e:
        ap.error(str(e))


def main(argv=None):
    ap = parser()
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
    try:
        sides = sides_of(a)
    except ValueError as e:
        ap.error(str(e))
    if a.max_rows is not None and a.max_rows < 1:
        ap.error('--max-rows must be >= 1')
    if a.conc is not None and a.conc < 1:
        ap.error('--conc must be >= 1')
    if a.groups < 1:
        ap.error('--groups must be >= 1')
    graphs = None
    if (a.graphs, a.a_graphs, a.b_graphs) != (None, None, None):
        graphs = tuple(None if v is None else bool(v)
                       for v in (a.graphs if a.a_graphs is None else a.a_graphs,
                                 a.graphs if a.b_graphs is None else a.b_graphs))
    try:
        name, why, runner = backend_choice(a, sides, seq)
    except Refused as e:
        ap.exit(2, f'h2h: backend {a.backend} refused - {e}\n')
    print(f'h2h: backend {name} - {why}', file=sys.stderr, flush=True)
    sink = _sink_of(a, ap)
    if runner is not None:
        out = remote_match(runner, a, sides, seq, sprt, sink, name)
        print(json.dumps(out), flush=True)
    else:
        with GR.recording(sink):
            _main_play(a, seq, sprt, sides, graphs)
    if sink is not None:
        print(f'# game records: {sink.stats["games"]} games in {sink.stats["files"]} file(s) under {sink.dir()}',
              file=sys.stderr, flush=True)


def _main_play(a, seq, sprt, sides, graphs):
    if seq:
        out = sequential(a.a, a.b, a.batch_games or a.games, a.max_games or 4 * a.games, sprt, a.alpha, a.beta,
                         a.target_se, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device, spec=a.spec,
                         seat_a=sides[0], seat_b=sides[1], paired=a.paired, sims_a=a.a_sims, sims_b=a.b_sims,
                         on_batch=_progress, graphs=graphs,
                         max_rows=a.max_rows, conc=a.conc, groups=a.groups, sparse=a.sparse, precision=a.precision)
    else:
        out = match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device,
                    spec=a.spec, seat_a=sides[0], seat_b=sides[1], move_log=a.move_log, paired=a.paired,
                    sims_a=a.a_sims, sims_b=a.b_sims, graphs=graphs, max_rows=a.max_rows, conc=a.conc,
                    groups=a.groups, sparse=a.sparse, precision=a.precision)
    out['backend'] = 'local'                       # (docs/az.md 69: the line names where its games were played)
    print(json.dumps(out), flush=True)


def _progress(r):
    print('# batch ' + json.dumps(r), file=sys.stderr, flush=True)


def remote_match(runner, a, sides, seq=False, sprt=None, sink=None, name='farm'):
    """A command line's match played by another backend (docs/az.md 69): `runner.play(k, n, seed)` plays batch k
    (the plain match: batch 0, seeded --seed; a sequential match `batch_loop`'s batches, seeded seed + 7919 k), zero.h2h
    reads its rows as a local match's - the same JSON line (`summary`, the sides' rules, simulations, `batches`,
    `stop`, `sprt`, `paired`), `merge` null (the drivers' merged-tree counters are the local driver's), no
    `graphs_*` / `look` / `precision` / `sparse` (local only) - plus `runner.info()`. Both checkpoints are loaded here
    first (on the CPU: the encoding guard, the belief head of a side with a root belief - the local match's
    refusals); `sink`: each batch's traced records written as a local match writes its own (`src` the backend's)."""
    ma, mb, game = _load_pair(a.a, a.b, 'cpu', None)
    spec = match_spec(a.a, a.b, game, a.spec)
    sa, sb = _seats(a.a, a.b, ma, mb, game, a.sims, a.c_puct, a.chance_k, sides[0], sides[1], a.a_sims, a.b_sims)
    del ma, mb
    pa = pb = None
    if sink is not None:
        pa = GR.net_player(a.a, sink.labels.get('a'), sa, _tag(sa))
        pb = GR.net_player(a.b, sink.labels.get('b'), sb, _tag(sb))
    t0 = time.time()

    def play(k, n, off):
        seed = a.seed + 7919 * k
        got = runner.play(k, n, seed)
        if sorted(int(r['gid']) for r in got) != list(range(n)):
            raise RuntimeError(f'backend {name}: batch {k} came back with {len(got)} games, not ids 0..{n - 1}')
        if sink is not None:
            items = []
            for r in sorted(got, key=lambda r: int(r['gid'])):
                if not isinstance(r.get('rec'), dict):
                    raise RuntimeError(f'backend {name}: game {r["gid"]} has no record: --records cannot keep it')
                ps = [pa, pb] if int(r['gid']) % 2 == 0 else [pb, pa]
                items.append(GR.from_trace(r['rec'], G.spec_json(spec), sink.kind, ps, sink.it, name))
            meta = {'driver_seed': int(seed), 'paired': bool(a.paired), 'backend': name}
            if seq:
                meta.update(batch=k, gid_offset=off)
            sink.add(items, meta)
        return [{k2: r[k2] for k2 in ('gid', 'result', 'scores', 'over')} for r in got]
    if seq:
        bounds = seq_bounds(sprt, a.alpha, a.beta, a.target_se, a.paired, a.batch_games or a.games,
                            a.max_games or 4 * a.games)
        rs, batches, stop, llr = batch_loop(play, a.batch_games or a.games, a.max_games or 4 * a.games, a.paired, bounds,
                                            a.target_se, _progress, t0)
    else:
        rs = play(0, a.games, 0)
    out = dict({'a': a.a, 'b': a.b, 'game': game, 'sims': a.sims, 'chance_k': a.chance_k}, **summary(rs, a.paired))
    out.update(seconds=round(time.time() - t0, 1), select_a=_tag(sa), select_b=_tag(sb))
    if seq:
        out.update(batches=batches, stop=stop)
    _sides_sims(out, a.a_sims, a.b_sims)
    if sa.get('infoset_merge') or sb.get('infoset_merge'):
        out['merge'] = None                        # (the local driver's counters: not the backend's)
    if a.paired:
        out['paired'] = True
    if seq:
        _seq_tail(out, bounds, a.alpha, a.beta, llr, stop, a.target_se)
    out['backend'] = name
    out.update(runner.info())
    return out


if __name__ == '__main__':
    main()
