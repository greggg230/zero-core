"""Light decisions in self-play (docs/search.md 3.10, docs/az.md 22).

A game declares classes of decisions that rarely matter (`SearchGame::LIGHT_CLASSES` and
`light_class`; MTG Zero: the priority decisions with an empty stack outside the decider's own main
phases, its "timing windows", by step and whose turn). A self-play seat with `sims_light` searches
the classes it selects with that small budget:

* `selfplay_sims_light` N > 0: a light decision whose playout-cap draw was full gets N simulations
  instead of `sims`; one whose draw was fast `selfplay_sims_light_fast` (0 = N) instead of
  `sims_fast`. 0 = off: nothing changes (the game's hook is never read).
* `selfplay_light_record` 0 (default): light decisions are never recorded - a minor decision's
  treatment: no root noise, no root floor or belief, no value-only record's next q from them. 1: a
  full draw is recorded as any full search, its policy target and root value from the N-simulation
  search (thinner targets instead of losing the samples); a fast draw is a fast search (unrecorded).
* `selfplay_light_classes`: which of the game's classes ('all', the default; else comma-separated
  tokens, left to right: a name or glob adds the classes it matches, '-' before one removes them -
  'all,-opp_end' = every class but the opponent's end step, 'opp_*' = the opponent's turn only).

Evaluation seats (the yardstick, random, Elo, anchors, h2h, replays, the eval remote) never read
these settings. The self-play league's opponents (zero/league.py) are self-play seats: they search
light decisions too. Remote self-play workers search from the mirror's config; a worker whose hello
reports another `fingerprint` (an older checkout: none) is refused (zero/remoteplay.py).

Metrics (the self-play line's `light`, from the finished games' result rows, this machine's and the
remote's): light decisions per game and their share of the searched decisions, recorded and
unrecorded light decisions per game, the simulations they took and saved per game, and the saved
share of the self-play search budget (saved / (all simulations + saved)).
"""
import fnmatch

KEYS = ('selfplay_sims_light', 'selfplay_sims_light_fast', 'selfplay_light_record', 'selfplay_light_classes')


def _int(cfg, k):
    v = cfg.get(k, 0)
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        raise ValueError(f'{k} {v!r}: an integer') from None


def select_classes(spec, names):
    """'all' | comma-separated tokens (a class name or a glob adds what it matches, '-<name or glob>'
    removes it; left to right, 'all' = every class) over the game's class `names` -> the selected
    classes' indices (sorted). ValueError for a token that matches no class, or an empty selection."""
    names = list(names)
    spec = str(spec if spec is not None else 'all').strip() or 'all'
    sel = set()
    for tok in (t.strip() for t in spec.split(',')):
        if not tok:
            raise ValueError(f'selfplay_light_classes {spec!r}: an empty token')
        neg = tok.startswith('-')
        pat = tok[1:] if neg else tok
        hit = set(range(len(names))) if pat == 'all' else {i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, pat)}
        if not hit:
            raise ValueError(f'selfplay_light_classes {spec!r}: {pat!r} matches none of the game\'s light classes '
                             f'{names}')
        sel = sel - hit if neg else sel | hit
    if not sel:
        raise ValueError(f'selfplay_light_classes {spec!r}: selects no class (of {names})')
    return sorted(sel)


def settings(cfg, info=None):
    """The run's light-decision settings resolved against the game's classes -> None when off
    (`selfplay_sims_light` 0), else {sims_light, sims_light_fast (None = sims_light), light_record,
    light_classes (indices), names (the selected classes' names)}. ValueError for a setting that would
    do nothing or not be light (the driver refuses those too: docs/search.md 3.10)."""
    sl, sf, rec = _int(cfg, 'selfplay_sims_light'), _int(cfg, 'selfplay_sims_light_fast'), _int(cfg, 'selfplay_light_record')
    cls = cfg.get('selfplay_light_classes', 'all')
    if sl < 0 or sf < 0:
        raise ValueError(f'selfplay_sims_light {sl} / selfplay_sims_light_fast {sf}: >= 0 (0 = off / = sims_light)')
    if rec not in (0, 1):
        raise ValueError(f'selfplay_light_record {rec}: 0 or 1')
    if not sl:
        bad = [k for k, v in (('selfplay_sims_light_fast', sf), ('selfplay_light_record', rec)) if v]
        if str(cls if cls is not None else 'all').strip() not in ('', 'all'):
            bad.append('selfplay_light_classes')
        if bad:
            raise ValueError(f'{" / ".join(bad)} without selfplay_sims_light: light decisions are off '
                             f'(selfplay_sims_light 0), the settings would do nothing')
        return None
    if info is None:                               # (a caller's stand-in Run without `info`)
        from .games import info as game_info
        info = game_info(cfg['game'])
    names = list((info or {}).get('light_classes') or [])
    if not names:
        raise ValueError(f'selfplay_sims_light {sl}: the game {(info or {}).get("name")!r} declares no light '
                         f'decisions (SearchGame::LIGHT_CLASSES)')
    idx = select_classes(cls, names)
    sims, fast = _int(cfg, 'sims'), _int(cfg, 'sims_fast')
    if sims and sl > sims:
        raise ValueError(f'selfplay_sims_light {sl} > sims {sims}: a light decision\'s budget is at most the one '
                         f'it replaces')
    if fast and (sf or sl) > fast:
        what = f'selfplay_sims_light_fast {sf}' if sf else f'selfplay_sims_light {sl} (no selfplay_sims_light_fast)'
        raise ValueError(f'{what} > sims_fast {fast}: a fast draw\'s light search would take more than the fast '
                         f'budget it replaces')
    if sf and not fast:
        raise ValueError(f'selfplay_sims_light_fast {sf} without sims_fast: every search is a full one')
    return {'sims_light': sl, 'sims_light_fast': sf or None, 'light_record': bool(rec), 'light_classes': idx,
            'names': [names[i] for i in idx]}


def check(cfg, info, where=''):
    """`settings` for the run's start (SystemExit with the reason, `where` a prefix) -> the settings
    or None."""
    try:
        return settings(cfg, info)
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None


def seat_keys(cfg, info=None):
    """The light keyword arguments of `games.nn_seat` for the run's self-play seat ({} when off: the
    seat of before, key for key)."""
    s = settings(cfg, info)
    if s is None:
        return {}
    return {k: s[k] for k in ('sims_light', 'sims_light_fast', 'light_record', 'light_classes')}


def fingerprint(cfg, info):
    """The run's light decisions as its self-play driver plays them ('' when off): what a remote
    worker's hello reports and the run checks (an older checkout reports none)."""
    s = settings(cfg, info)
    if s is None:
        return ''
    return (f'sims_light={s["sims_light"]},sims_light_fast={s["sims_light_fast"] or s["sims_light"]},'
            f'record={int(s["light_record"])},classes={"+".join(s["names"])}')


def describe(s):
    """A start-up line for the settings (`settings`'s dict)."""
    fast = s['sims_light_fast'] or s['sims_light']
    return (f'light decisions: {s["sims_light"]} simulations (full draws) / {fast} (fast draws), '
            f'{"recorded (full draws)" if s["light_record"] else "never recorded"}; classes {",".join(s["names"])}')


# ---------------------------------------------------------------- metrics

FIELDS = ('decisions', 'recorded', 'sims', 'saved', 'sims_all')


def new_tally():
    return {'games': 0, 'searched': 0, **{k: 0 for k in FIELDS}}


def tally(rows, acc=None):
    """Add finished games' result rows (their `light`, `decisions`) to `acc` -> acc. Rows without
    `light` (a driver without sims_light) are not counted."""
    acc = acc if acc is not None else new_tally()
    for r in rows:
        lt = r.get('light')
        if lt is None:
            continue
        acc['games'] += 1
        acc['searched'] += int(r.get('decisions', 0))
        for k in FIELDS:
            acc[k] += int(lt.get(k, 0))
    return acc


def merge(a, b):
    """Two tallies (either may be None) -> their sum (None if both are)."""
    if a is None:
        return dict(b) if b else None
    if not b:
        return a
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def summary(acc):
    """A tally -> the self-play line's `light` (per game; shares)."""
    acc = acc or new_tally()
    g = max(1, acc['games'])
    d = acc['decisions']
    return {'games': acc['games'],
            'per_game': round(d / g, 2),
            'share': round(d / max(1, acc['searched']), 4),
            'recorded_per_game': round(acc['recorded'] / g, 2),
            'unrecorded_per_game': round((d - acc['recorded']) / g, 2),
            'sims_per_game': round(acc['sims'] / g, 1),
            'saved_per_game': round(acc['saved'] / g, 1),
            'saved_share': round(acc['saved'] / max(1, acc['sims_all'] + acc['saved']), 4)}


def fmt(s):
    """The log line's text for a `summary`."""
    return (f'{s["per_game"]} light decisions per game ({100 * s["share"]:.1f}% of the searched), '
            f'{s["recorded_per_game"]} recorded / {s["unrecorded_per_game"]} unrecorded; '
            f'{s["saved_per_game"]:.0f} simulations saved per game ({100 * s["saved_share"]:.1f}% of the budget) '
            f'over {s["games"]} games')
