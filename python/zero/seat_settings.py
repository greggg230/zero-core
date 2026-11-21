"""Per-seat evaluation settings (docs/az.md 68): one side of an evaluation match playing other run settings than the
other side - zero.h2h's per-side options (--a-* / --b-*) and a zero-farm eval job seat's `set` (PROTOCOL.md 5.2), one
mapping, so a local head-to-head and a farm one cannot drift apart.

A side's settings are run settings: its evaluation seat is `Run.nn_seat(key, eval_sims)` of the match's settings with
the side's `set` over them (`dict(cfg, **set)`). `SEAT_KEYS` is every setting that seat reads - what a set may carry,
zero_worker settings::SEAT_READS on the worker's side (tests/test_seat_settings.py checks both against what Run.nn_seat
reads). An eval job's network seat `{"net": gen, "set": {...}}` plays its set (the worker refuses any other key, and a
build without the feature `seat_settings` refuses the seat); a seat without a set plays the job's settings, as before.

zero.h2h's per-side options (`add_side_args`: the flags, `SIDE_OPTIONS`: the run settings each one sets) become a side's
set by `side_set` - their meaning as zero.h2h documents it, written as run settings: zero.h2h builds its sides' seats
with it (`h2h.sides_of`), zero.farm_h2h and MTG Zero's farm campaigns build their seats' sets with it. `LOCAL_ONLY`
options have no run setting (torch's CUDA graphs, the Python lookahead): zero.h2h only. A per-side option zero.h2h gains
must be in one of the two (tests/test_seat_settings.py fails otherwise).

Imports only the search's checks (zero.mcts): light enough for command-line tools.
"""
from .mcts import (belief_settings, belief_tilts, check_belief, check_belief_tilts, check_merge, check_root_search,
                   check_select, check_select_transform, select_settings)

# the settings an evaluation seat reads (Run.nn_seat(key, c['eval_sims']), and eval_sims): a set's keys
SEAT_KEYS = (
    'eval_sims', 'c_puct', 'chance_k', 'margin_w', 'redet', 'cq_c_visit', 'cq_c_scale', 'eval_select',
    'eval_select_min_visits', 'eval_select_c_visit', 'eval_select_c_scale', 'eval_root_min_visits',
    'root_min_visits_max_actions', 'eval_leaf_value_view', 'eval_root_search', 'gumbel_m', 'eval_gumbel_scale',
    'eval_cache', 'eval_infoset_merge', 'eval_belief_tilt', 'belief_clip', 'belief_mix', 'belief_fast',
    'eval_belief_tilts', 'eval_value_choice_worlds', 'eval_value_choice_temp',
)

# zero.h2h's per-side options (the argparse dest after the side's prefix: --a-select -> 'select') -> the run settings
# `side_set` writes for it
SIDE_OPTIONS = {
    'sims': ('eval_sims',),
    'select': ('eval_select', 'eval_select_min_visits', 'eval_select_c_visit', 'eval_select_c_scale', 'cq_c_visit',
               'cq_c_scale'),
    'select_min_visits': ('eval_select_min_visits',),
    'cq_c_visit': ('eval_select_c_visit',),
    'cq_c_scale': ('eval_select_c_scale',),
    'leaf_value_view': ('eval_leaf_value_view',),
    'belief_tilt': ('eval_belief_tilt', 'belief_clip', 'belief_mix', 'belief_fast'),
    'belief_clip': ('belief_clip',),
    'belief_mix': ('belief_mix',),
    'belief_fast': ('belief_fast',),
    'belief_tilts': ('eval_belief_tilts',),
    'infoset_merge': ('eval_infoset_merge',),
    'root_search': ('eval_root_search',),
    'gumbel_m': ('gumbel_m',),
    'gumbel_scale': ('eval_gumbel_scale',),
}
# both-sides options (no side prefix) of the per-side ones that have one
BOTH_OPTIONS = ('select', 'select_min_visits', 'cq_c_visit', 'cq_c_scale', 'infoset_merge', 'gumbel_m')
# zero.h2h's per-side options with no run setting: never on the farm
LOCAL_ONLY = {
    'graphs': "CUDA graphs: torch's evaluator (NetEvaluator, docs/az.md 10.6)",
    'lookahead': 'the exhaustive lookahead: a Python seat (zero.lookahead, docs/search.md 3.9)',
}


def add_side_args(ap, local=True):
    """zero.h2h's per-side options on `ap` (zero.h2h; zero.farm_h2h and farm campaigns with `local` False: no
    `LOCAL_ONLY` option). Every default None = not given (the side plays its settings' value)."""
    sides = ('a', 'b')
    ap.add_argument('--select', default=None, choices=['visits', 'cq'],
                    help="both sides' move rule (default: each side's run's eval_select)")
    ap.add_argument('--select-min-visits', type=int, default=None,
                    help="both sides' 'cq' guard (default: 0 with --select, else each side's run's)")
    for side in sides:
        ap.add_argument(f'--{side}-select', default=None, choices=['visits', 'cq'], help=f"side {side.upper()}'s move rule")
        ap.add_argument(f'--{side}-select-min-visits', type=int, default=None)
    ap.add_argument('--cq-c-visit', type=float, default=None,
                    help="the 'cq' rule's completed-Q transform, both sides (default: 50 / 1 with --select, else "
                         "each side's run's)")
    ap.add_argument('--cq-c-scale', type=float, default=None)
    for side in sides:
        ap.add_argument(f'--{side}-cq-c-visit', type=float, default=None,
                        help=f"side {side.upper()}'s 'cq' transform (over --cq-c-visit / --cq-c-scale)")
        ap.add_argument(f'--{side}-cq-c-scale', type=float, default=None)
    for side in sides:
        ap.add_argument(f'--{side}-leaf-value-view', default=None, choices=['mover', 'root'],
                        help=f"side {side.upper()}'s leaf values' view (docs/search.md 3.6; default: its run's "
                             f"eval_leaf_value_view)")
        ap.add_argument(f'--{side}-belief-tilt', type=float, default=None,
                        help=f"side {side.upper()}'s root belief tilt (docs/search.md 3.8; 0 = uniform worlds; default: its "
                             f"run's eval_belief_tilt)")
        ap.add_argument(f'--{side}-belief-clip', type=float, default=None)
        ap.add_argument(f'--{side}-belief-mix', type=float, default=None)
        ap.add_argument(f'--{side}-belief-fast', type=int, default=None, choices=[0, 1])
        ap.add_argument(f'--{side}-belief-tilts', type=_json, default=None,
                        help=f"side {side.upper()}'s per-target tilts, JSON {{target: alpha}} (docs/search.md 3.8.1; {{}} = "
                             f"none; default: its run's eval_belief_tilts)")
    ap.add_argument('--infoset-merge', type=int, default=None, choices=[0, 1],
                    help="both sides' information-set merging (docs/search.md 3.16; default: each side's run's "
                         "eval_infoset_merge)")
    for side in sides:
        ap.add_argument(f'--{side}-infoset-merge', type=int, default=None, choices=[0, 1],
                        help=f"side {side.upper()}'s information-set merging (over --infoset-merge; needs root-view "
                             f"leaf values)")
    for side in sides:
        ap.add_argument(f'--{side}-root-search', default=None, choices=['puct', 'gumbel'],
                        help=f"side {side.upper()}'s root search (docs/search.md 3.7; default: its run's eval_root_search)")
        ap.add_argument(f'--{side}-gumbel-scale', type=float, default=None,
                        help=f"side {side.upper()}'s Gumbel noise scale (default: its run's eval_gumbel_scale, 0)")
        ap.add_argument(f'--{side}-gumbel-m', type=int, default=None,
                        help=f"side {side.upper()}'s Gumbel considered options (over --gumbel-m)")
    ap.add_argument('--gumbel-m', type=int, default=None,
                    help="a Gumbel side's considered options (default: its run's gumbel_m, 16)")
    for side in sides:
        ap.add_argument(f'--{side}-sims', type=int, default=None, help=f"side {side.upper()}'s simulations (default --sims)")
    if not local:
        return ap
    ap.add_argument('--graphs', type=int, default=None, choices=[0, 1],
                    help="both sides' CUDA graphs (default: the evaluator's: a conv network's on, a token network's "
                         "env ZERO_TOKEN_GRAPHS=1; docs/az.md 10.6)")
    for side in sides:
        ap.add_argument(f'--{side}-graphs', type=int, default=None, choices=[0, 1],
                        help=f"side {side.upper()}'s CUDA graphs (over --graphs)")
    ap.add_argument('--lookahead', default=None, metavar='SETTINGS',
                    help="both sides lookahead seats (docs/search.md 3.9; zero.lookahead.parse, e.g. "
                         "plies=6,worlds=8,opp=best,mode=infoset,budget=200000)")
    for side in sides:
        ap.add_argument(f'--{side}-lookahead', default=None, metavar='SETTINGS',
                        help=f"side {side.upper()} a lookahead seat (over --lookahead)")
    return ap


def _json(text):
    import json
    return json.loads(text)


def check_set(s, what='set'):
    """A seat's set: None, or a non-empty dict of `SEAT_KEYS` keys -> the dict (None for none); ValueError otherwise
    (the worker's refusal, here first)."""
    if s is None:
        return None
    if not isinstance(s, dict) or not s:
        raise ValueError(f'{what} {s!r}: a non-empty object {{<setting>: <value>}} (docs/az.md 68; none: leave it out)')
    bad = sorted(k for k in s if k not in SEAT_KEYS)
    if bad:
        raise ValueError(f'{what}: {bad} not seat settings - a seat\'s set holds only what its evaluation seat reads '
                         f'(Run.nn_seat: {", ".join(SEAT_KEYS)})')
    return dict(s)


def seat_cfg(cfg, s=None):
    """The settings a seat plays: `cfg` with its set `s` over them."""
    return dict(cfg, **(check_set(s) or {}))


def changed(cfg, over):
    """The keys of `over` whose value is not `cfg`'s (a set that restates a setting adds nothing)."""
    miss = object()
    return {k: v for k, v in over.items() if cfg.get(k, miss) != v or type(cfg.get(k, miss)) is not type(v)}


def _opt(a, side, k):
    return getattr(a, f'{side}_{k}', None)


def _pick(a, side, k):
    v = _opt(a, side, k)
    return getattr(a, k, None) if v is None and k in BOTH_OPTIONS else v


def side_rule(a, side, cfg):
    """The move rule side `side` plays ('visits' | 'cq'): its --{side}-select / --select, else its settings'."""
    return _pick(a, side, 'select') or cfg.get('eval_select', 'visits')


def side_set(a, side, cfg):
    """Side `side`'s ('a' | 'b') per-side options in `a` (a namespace with `add_side_args`' dests, None = not given) as
    run settings over `cfg` (the side's settings: a run's resolved settings) -> {setting: value}, only what differs from
    `cfg` ({}: the side plays cfg's evaluation seat). Their meaning (zero.h2h's):

    * --{s}-sims: eval_sims.
    * --{s}-select / --select: the move rule replaced - its guard 0 unless given and its completed-Q transform 50 / 1
      unless given (eval_select, eval_select_min_visits, eval_select_c_visit / _c_scale); a 'visits' side keeps its
      run's evaluation transform for a Gumbel root's sigma (cq_c_visit / cq_c_scale). The rest of the side's evaluation
      seat stays its run's (root minimum visits, leaf view, merging, value choices, belief, root search).
      --{s}-select-min-visits without a rule: the guard of the side's own rule (--select-min-visits: of each side
      that plays 'cq').
    * --{s}-cq-c-visit / -scale (over --cq-c-*): the 'cq' rule's transform (eval_select_c_*); refused for a side
      that plays 'visits' (a both-sides flag skips it).
    * --{s}-leaf-value-view, --{s}-infoset-merge / --infoset-merge: eval_leaf_value_view, eval_infoset_merge (a merged
      side needs root-view values).
    * --{s}-belief-tilt, -tilts, -clip, -mix, -fast: eval_belief_tilt, eval_belief_tilts (zeros dropped), belief_clip,
      belief_mix, belief_fast; a belief the side's run evaluates without takes clip 20, mix 0, full searches only
      unless given; -clip / -mix / -fast for a side without a belief are refused.
    * --{s}-root-search: eval_root_search; a Gumbel side's --{s}-gumbel-m (over --gumbel-m) and --{s}-gumbel-scale:
      gumbel_m, eval_gumbel_scale (either per-side flag refused for a PUCT side; --gumbel-m skips it).

    ValueError naming the option for a combination refused (the seat's own checks - Run.nn_seat's - run where it is
    built: zero.farm_eval.seat_dicts, the worker's check-config)."""
    S = side.upper()
    out = {}
    if _opt(a, side, 'sims') is not None:
        out['eval_sims'] = int(_opt(a, side, 'sims'))
    # the move rule (docs/search.md 3.3)
    rule = _pick(a, side, 'select')
    mv = _pick(a, side, 'select_min_visits')
    run_rule, _, run_cv, run_cs = select_settings(cfg, 'eval_select')
    if rule is not None:
        check_select(rule, 0 if mv is None else int(mv), what=f'side {S} select')
        if rule == 'visits':
            # (no 'cq' keys; a Gumbel root's sigma stays the run's evaluation transform, as zero.h2h.root_keys)
            out.update(eval_select='visits', eval_select_min_visits=0, eval_select_c_visit=None,
                       eval_select_c_scale=None, cq_c_visit=run_cv, cq_c_scale=run_cs)
        else:
            out.update(eval_select='cq', eval_select_min_visits=int(mv or 0), eval_select_c_visit=50.0,
                       eval_select_c_scale=1.0)
    elif mv is not None and (run_rule != 'visits' or _opt(a, side, 'select_min_visits') is not None):
        # (the guard of the side's own rule; the both-sides flag skips a 'visits' side, as the 'cq' transform's does)
        check_select(run_rule, int(mv), what=f'side {S} select_min_visits (its run\'s rule {run_rule!r})')
        out['eval_select_min_visits'] = int(mv)
    eff = rule or run_rule
    own = (_opt(a, side, 'cq_c_visit'), _opt(a, side, 'cq_c_scale'))
    if eff == 'visits':
        if own != (None, None):
            raise ValueError(f"--{side}-cq-c-*: side {S} plays 'visits' (no completed-Q transform)")
    else:
        cv, cs = _pick(a, side, 'cq_c_visit'), _pick(a, side, 'cq_c_scale')
        if cv is not None:
            out['eval_select_c_visit'] = float(cv)
        if cs is not None:
            out['eval_select_c_scale'] = float(cs)
        if (cv, cs) != (None, None):
            c = dict(cfg, **out)
            _, _, tv, ts = select_settings(c, 'eval_select')
            check_select_transform('cq', tv, ts, what=f'side {S} --cq')
    # leaf values' view, information-set merging (docs/search.md 3.6, 3.16)
    view = _opt(a, side, 'leaf_value_view')
    if view is not None:
        out['eval_leaf_value_view'] = view
    merge = _pick(a, side, 'infoset_merge')
    if merge is not None:
        out['eval_infoset_merge'] = int(merge)
    c = dict(cfg, **out)
    try:
        check_merge(int(bool(c.get('eval_infoset_merge', 0))), c.get('eval_leaf_value_view', 'mover'),
                    what=f'side {S} infoset_merge')
    except ValueError as e:
        raise ValueError(f"{e} (--{side}-leaf-value-view root, or its run's eval_leaf_value_view)") from None
    # the root's belief (docs/search.md 3.8)
    out.update(_belief_set(a, side, cfg))
    # the root search (docs/search.md 3.7)
    root = _opt(a, side, 'root_search')
    if root is not None:
        out['eval_root_search'] = root
    eff_root = root or cfg.get('eval_root_search', 'puct')
    m, scale = _pick(a, side, 'gumbel_m'), _opt(a, side, 'gumbel_scale')
    if eff_root == 'gumbel':
        if m is not None:
            out['gumbel_m'] = int(m)
        if scale is not None:
            out['eval_gumbel_scale'] = float(scale)
        c = dict(cfg, **out)
        try:
            check_root_search('gumbel', c.get('gumbel_m', 16), c.get('eval_gumbel_scale', 0.0), what='--root-search')
        except ValueError as e:
            raise ValueError(f'side {S}: {e}') from None
    else:
        for k in ('gumbel_scale', 'gumbel_m'):
            if _opt(a, side, k) is not None:
                raise ValueError(f"--{side}-{k.replace('_', '-')}: side {S} searches with PUCT")
    return changed(cfg, out)


def _belief_set(a, side, cfg):
    tilt, tilts = _opt(a, side, 'belief_tilt'), _opt(a, side, 'belief_tilts')
    clip, mix, fast = (_opt(a, side, k) for k in ('belief_clip', 'belief_mix', 'belief_fast'))
    run_tilt = belief_settings(cfg, 'eval')[0]
    run_tilts = belief_tilts(cfg, 'eval')
    out = {}
    new_tilts = run_tilts
    if tilts is not None:
        check_belief_tilts(tilts, what=f'--{side}-belief-tilts')
        new_tilts = {k: float(v) for k, v in sorted(tilts.items()) if v}
        out['eval_belief_tilts'] = new_tilts
    new_tilt = run_tilt
    if tilt is not None:
        new_tilt = float(tilt)
        out['eval_belief_tilt'] = new_tilt
    on = bool(new_tilt or new_tilts)
    if not on:
        if (clip, mix, fast) != (None, None, None):
            raise ValueError(f'--{side}-belief-clip / -mix / -fast for a side without a root belief (give its '
                             f'--{side}-belief-tilt)')
        return out
    if not (run_tilt or run_tilts):
        # (a belief its run evaluates without: clip 20, no uniform share, full searches only - zero.h2h.with_belief)
        out.update(belief_clip=20.0, belief_mix=0.0, belief_fast=0)
    if clip is not None:
        out['belief_clip'] = float(clip)
    if mix is not None:
        out['belief_mix'] = float(mix)
    if fast is not None:
        out['belief_fast'] = int(fast)
    _, bc, bm, _ = belief_settings(dict(cfg, **out), 'eval')
    check_belief(new_tilt, bc, bm)
    return out


def sets_of(a, cfg_a, cfg_b=None):
    """Both sides' sets (`side_set`) over their settings (`cfg_b` None: `cfg_a` - one job's settings) -> [set_a,
    set_b]; also refuses both-sides 'cq' transform flags when no side plays 'cq' (zero.h2h's rule)."""
    cfg_b = cfg_a if cfg_b is None else cfg_b
    out = [side_set(a, 'a', cfg_a), side_set(a, 'b', cfg_b)]
    cq = 'cq' in (side_rule(a, 'a', cfg_a), side_rule(a, 'b', cfg_b))
    if (getattr(a, 'cq_c_visit', None), getattr(a, 'cq_c_scale', None)) != (None, None) and not cq:
        raise ValueError("--cq-c-visit / --cq-c-scale: no side plays 'cq'")
    if getattr(a, 'select_min_visits', None) and getattr(a, 'select', None) is None and not cq:
        raise ValueError("--select-min-visits: no side plays 'cq' (give --select)")
    return out


def given(a, local=False):
    """The per-side (and both-sides) options given in `a` (`LOCAL_ONLY` ones only with `local`) -> {dest: value}."""
    out = {}
    for k in SIDE_OPTIONS:
        for d in (f'a_{k}', f'b_{k}') + ((k,) if k in BOTH_OPTIONS else ()):
            if getattr(a, d, None) is not None:
                out[d] = getattr(a, d)
    if local:
        for k in LOCAL_ONLY:
            for d in (f'a_{k}', f'b_{k}', k):
                if getattr(a, d, None) is not None:
                    out[d] = getattr(a, d)
    return out


def net_seat(gen, s=None):
    """A network seat of an eval job: {"net": gen}, with its set when it has one."""
    s = check_set(s or None)
    return {'net': int(gen), 'set': s} if s else {'net': int(gen)}
