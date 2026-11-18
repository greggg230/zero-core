"""The playout cap's full searches chosen by the prior's uncertainty (docs/az.md 60, docs/search.md 3.21).

Playout cap randomisation gives a self-play decision the full budget (`sims`) with probability `p_full` and the
fast one (`sims_fast`) otherwise; only the full searches are recorded. `p_full_rule` 'prior_uncertainty' spends the
same budget where the search is likely to change the move: the decision's chance of a full search is

    f = min(1, p_full_floor + (p_full - p_full_floor) * g(u) / p_full_u_mean),   u = 1 - max P0

P0 being the network's policy at the root (its own evaluation, before any noise) and g the shape `p_full_shape`
('linear' u, 'sqrt', 'square'). With `p_full_u_mean` the mean of g(u) over the seat's capped decisions, the mean of
f is p_full (less the little the cap at 1 cuts off): the same simulations per decision, at least the floor for
every decision. The decision's coin is drawn where the random rule draws it (the game's own stream), and compared
once the root's priors are known: the same games on every executor (this machine, the remote workers, the farm).

Run settings (zero.run.DEFAULTS): `p_full_rule` ('random' = off: the coin of before, nothing changes),
`p_full_floor` (0.05), `p_full_shape` ('linear'), `p_full_u_mean` (0 = unset; required with the rule: measure it
with the random rule's `selfplay.budget.g_mean`, which reports u's mean, or - for another shape - with the rule at
`p_full_floor` = `p_full`, which plays the coin's games exactly and reports that shape's mean; the rule's own games
read it slightly higher: docs/az.md 60.3). The
self-play seat's keys of the same names (games.nn_seat); evaluations have no playout cap.

Every self-play phase reports the playout cap of this machine's driver as `selfplay.budget` (decisions, the full
share, simulations per decision, the mean chance and the mean g(u), the share of recorded targets whose argmax is
not the prior's), with either rule.

Remote self-play: the workers draw the full searches exactly as this machine (the run's config.json); a worker whose
checkout draws them otherwise (an older one by the coin) is refused by its hello (`fingerprint`).
"""

KEYS = ('p_full_rule', 'p_full_floor', 'p_full_shape', 'p_full_u_mean')
RULES = ('random', 'prior_uncertainty')
SHAPES = ('linear', 'sqrt', 'square')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def enabled(cfg):
    return (cfg.get('p_full_rule') or 'random') != 'random'


def check_values(rule, floor, shape, u_mean, p_full, sims_fast=1):
    """The settings' ranges -> None, else ValueError (the driver's `FullRule::check`)."""
    if rule not in RULES:
        raise ValueError(f'p_full_rule {rule!r}: one of {RULES}')
    if rule == 'random':
        return
    if not sims_fast:
        raise ValueError("p_full_rule 'prior_uncertainty': a playout-cap seat's (sims_fast)")
    if shape not in SHAPES:
        raise ValueError(f'p_full_shape {shape!r}: one of {SHAPES}')
    if not (_num(p_full) and 0.0 < p_full <= 1.0):
        raise ValueError(f"p_full {p_full!r}: in (0, 1] with p_full_rule 'prior_uncertainty'")
    if not (_num(floor) and 0.0 <= floor <= p_full):
        raise ValueError(f'p_full_floor {floor!r}: a share in [0, p_full {p_full}] (every decision\'s least chance of '
                         f'a full search)')
    if not (_num(u_mean) and u_mean > 0.0 and u_mean == u_mean and u_mean != float('inf')):
        raise ValueError(f'p_full_u_mean {u_mean!r}: the mean > 0 of g(u) over the self-play decisions (measure it: '
                         f'selfplay.budget.g_mean; docs/az.md 60)')


def check_settings(cfg, where=''):
    """The run settings (zero.run.resolve_config) -> None, else SystemExit."""
    try:
        check_values(cfg.get('p_full_rule') or 'random', cfg.get('p_full_floor', 0.05), cfg.get('p_full_shape', 'linear'),
                     cfg.get('p_full_u_mean', 0.0), cfg.get('p_full'), cfg.get('sims_fast'))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    if enabled(cfg) and cfg.get('selfplay_root_search', 'puct') == 'gumbel':
        raise SystemExit(f"{where}p_full_rule 'prior_uncertainty': not built for selfplay_root_search 'gumbel'")


def seat_keys(cfg):
    """The self-play seat's keyword arguments of `games.nn_seat` ({} when off: the seat of before)."""
    if not enabled(cfg):
        return {}
    return dict(p_full_rule=cfg['p_full_rule'], p_full_floor=float(cfg.get('p_full_floor', 0.05)),
                p_full_shape=cfg.get('p_full_shape', 'linear'), p_full_u_mean=float(cfg['p_full_u_mean']))


def fingerprint(cfg):
    """What a remote worker's hello reports (zero/remoteplay.py) and the run checks: '' when off."""
    k = seat_keys(cfg)
    if not k:
        return ''
    return f'{k["p_full_rule"]} floor={k["p_full_floor"]} shape={k["p_full_shape"]} u_mean={k["p_full_u_mean"]}'


CAP = ('cap_decisions', 'cap_full', 'cap_sims', 'cap_g_e6', 'cap_f_e6', 'cap_changed', 'cap_targets')


def summary(st, st0=None):
    """A self-play phase's playout cap from its driver's counters (`st` - `st0`, SearchDriver.stats(); docs/search.md
    3.21) -> None without capped decisions, else {decisions, full, full_share, sims_per_decision, f_mean (the mean
    chance of a full search: p_full for the coin), g_mean (the mean of g(u): u for the coin), changed (the share of
    the recorded targets whose argmax is not the prior's argmax), targets}."""
    d = {k: st.get(k, 0) - (st0 or {}).get(k, 0) for k in CAP}
    n = d['cap_decisions']
    if n <= 0:
        return None
    return {'decisions': n, 'full': d['cap_full'], 'full_share': round(d['cap_full'] / n, 4),
            'sims_per_decision': round(d['cap_sims'] / n, 3), 'f_mean': round(d['cap_f_e6'] * 1e-6 / n, 4),
            'g_mean': round(d['cap_g_e6'] * 1e-6 / n, 4), 'changed': round(d['cap_changed'] / max(1, d['cap_targets']), 4),
            'targets': d['cap_targets']}


def fmt(s, cfg):
    """One log line of a summary."""
    rule = cfg.get('p_full_rule') or 'random'
    line = (f"{s['decisions']} capped decisions ({rule}): full {100 * s['full_share']:.1f}% (chance "
            f"{100 * s['f_mean']:.1f}%, p_full {100 * float(cfg.get('p_full', 0)):.1f}%), "
            f"{s['sims_per_decision']:.2f} sims per decision, mean g(u) {s['g_mean']:.4f}"
            + (f" (p_full_u_mean {float(cfg.get('p_full_u_mean') or 0):.4f})" if rule != 'random' else '')
            + f", targets off the prior's argmax {100 * s['changed']:.1f}%")
    return line


def drift(s, cfg, tol=0.1):
    """With the rule on: a message when the phase's mean chance of a full search is off p_full by more than `tol` of it
    (p_full_u_mean no longer the mean of g(u): the network's sharpness moved), else ''. The coin never drifts."""
    if not enabled(cfg):
        return ''
    p = float(cfg['p_full'])
    if abs(s['f_mean'] - p) <= tol * p:
        return ''
    return (f"the mean chance of a full search is {s['f_mean']:.4f}, p_full {p}: p_full_u_mean "
            f"{float(cfg['p_full_u_mean']):.4f} vs the phase's mean g(u) {s['g_mean']:.4f} - set p_full_u_mean to the "
            f"measured mean (docs/az.md 60.1)")
