"""Heavy decisions in self-play (docs/search.md 3.11, docs/az.md 23).

A self-play seat searches deeper exactly where hidden information is about to be revealed to it: at
each of its decisions that is neither minor nor light, once the root is expanded, a probe
(`heavy::probe`, zero-core) steps the options of prior <= `selfplay_heavy_prior_max` in one world
determinized for the decider and follows a short line from each (other seats take their passive
option, the decider may deviate `selfplay_heavy_depth` times in all, `selfplay_heavy_horizon` steps,
no step into a later round); a line that reveals something to the decider beyond what passing would
(the game's `reveals` of the keys `step` pushes from the decider's view: its own draws and looks, a
hidden card made public - never another player's masked private draw) makes the decision **heavy**:

* `selfplay_sims_heavy` N > 0: a heavy decision whose playout-cap draw was full gets N simulations
  instead of `sims`; one whose draw was fast `selfplay_sims_heavy_fast` (required when `sims_fast` is
  set) instead of `sims_fast`. 0 = off: nothing changes (the probe never runs, the hooks are never
  read).
* `selfplay_chance_k_heavy` (-1 = the run's `chance_k`): `chance_k` inside heavy searches - an edge
  holding k sampled outcomes replays one of them, so a revealed card's subtree is searched again.
* `selfplay_root_min_visits_heavy` (0 = off): every root option of a heavy search gets at least this
  many visits, clipped to half its budget (`heavy_floor`), full and fast draws alike; the root
  floor's machinery (docs/search.md 3.5): the forced walks stay out of the recorded 'visits' targets,
  temperature sampling, the 'visits' rule and the root value q.
* `selfplay_heavy_prior_max` (1.0 = every option probed), `selfplay_heavy_depth` (2: the root option
  and one follow-up action), `selfplay_heavy_horizon` (12 steps), `selfplay_heavy_lines` (64 lines a
  probe).

Recorded as usual: a heavy full draw is a recorded full search whose targets come from the heavy
search. Light (zero/light.py) and minor decisions come first: a decision of a light class the seat
selects is light, never probed. Evaluation seats (the yardstick, random, Elo, anchors, h2h, replays,
the eval remote) never read these settings; the self-play league's opponents are self-play seats and
search heavy decisions too. Remote self-play workers search from the mirror's config; a worker whose
hello reports another `fingerprint` (an older checkout: none) is refused (zero/remoteplay.py).

Metrics (the self-play line's `heavy`, from the finished games' result rows, this machine's and the
remote's): heavy decisions per game and their share of the searched decisions, the probed share, the
recorded heavy decisions, the simulations spent on heavy decisions (per game and their share of all),
the root floor's forced walks, and the probe's cost (lines, steps and milliseconds per probed
decision).
"""
KEYS = ('selfplay_sims_heavy', 'selfplay_sims_heavy_fast', 'selfplay_chance_k_heavy', 'selfplay_root_min_visits_heavy',
        'selfplay_heavy_prior_max', 'selfplay_heavy_depth', 'selfplay_heavy_horizon', 'selfplay_heavy_lines')
# the probe's defaults (zero_search::heavy::ProbeCfg::default)
PROBE = {'selfplay_heavy_depth': 2, 'selfplay_heavy_horizon': 12, 'selfplay_heavy_lines': 64}


def _int(cfg, k, default=0):
    v = cfg.get(k, default)
    try:
        return int(default if v is None else v)
    except (TypeError, ValueError):
        raise ValueError(f'{k} {v!r}: an integer') from None


def _float(cfg, k, default):
    v = cfg.get(k, default)
    try:
        return float(default if v is None else v)
    except (TypeError, ValueError):
        raise ValueError(f'{k} {v!r}: a number') from None


def settings(cfg):
    """The run's heavy-decision settings -> None when off (`selfplay_sims_heavy` 0), else {sims_heavy,
    sims_heavy_fast (None without sims_fast), chance_k_heavy (None = chance_k), root_min_visits_heavy,
    heavy_prior_max, heavy_depth, heavy_horizon, heavy_lines}. ValueError for a setting that would do
    nothing or not be heavy (the driver refuses those too: docs/search.md 3.11)."""
    sh, sf = _int(cfg, 'selfplay_sims_heavy'), _int(cfg, 'selfplay_sims_heavy_fast')
    ck, k = _int(cfg, 'selfplay_chance_k_heavy', -1), _int(cfg, 'selfplay_root_min_visits_heavy')
    gate = _float(cfg, 'selfplay_heavy_prior_max', 1.0)
    probe = {k2: _int(cfg, k2, v) for k2, v in PROBE.items()}
    if sh < 0 or sf < 0 or k < 0 or ck < -1:
        raise ValueError(f'selfplay_sims_heavy {sh} / _fast {sf} / selfplay_root_min_visits_heavy {k}: >= 0; '
                         f'selfplay_chance_k_heavy {ck}: >= 0 or -1 (= chance_k)')
    if not sh:
        bad = [n for n, v in (('selfplay_sims_heavy_fast', sf), ('selfplay_root_min_visits_heavy', k)) if v]
        if ck != -1:
            bad.append('selfplay_chance_k_heavy')
        if gate != 1.0:
            bad.append('selfplay_heavy_prior_max')
        bad += [n for n, v in probe.items() if v != PROBE[n]]
        if bad:
            raise ValueError(f'{" / ".join(bad)} without selfplay_sims_heavy: heavy decisions are off '
                             f'(selfplay_sims_heavy 0), the settings would do nothing')
        return None
    sims, fast, chance_k = _int(cfg, 'sims'), _int(cfg, 'sims_fast'), _int(cfg, 'chance_k')
    if sims and sh < sims:
        raise ValueError(f'selfplay_sims_heavy {sh} < sims {sims}: a heavy decision\'s budget is at least the one it '
                         f'replaces')
    if fast and not sf:
        raise ValueError(f'selfplay_sims_heavy {sh} with sims_fast {fast}: set selfplay_sims_heavy_fast (a fast '
                         f'draw\'s heavy budget) explicitly')
    if sf and not fast:
        raise ValueError(f'selfplay_sims_heavy_fast {sf} without sims_fast: every search is a full one')
    if fast and sf < fast:
        raise ValueError(f'selfplay_sims_heavy_fast {sf} < sims_fast {fast}: a heavy decision\'s budget is at least '
                         f'the one it replaces')
    if ck == chance_k:
        raise ValueError(f'selfplay_chance_k_heavy {ck} = chance_k: the setting would do nothing (-1 = chance_k)')
    if not 0.0 < gate <= 1.0:
        raise ValueError(f'selfplay_heavy_prior_max {gate}: in (0, 1] (1 = every option probed)')
    if not 1 <= probe['selfplay_heavy_depth'] <= 4 or not 1 <= probe['selfplay_heavy_horizon'] <= 64 \
            or not 1 <= probe['selfplay_heavy_lines'] <= 4096:
        raise ValueError(f'selfplay_heavy_depth / _horizon / _lines {list(probe.values())}: 1..4, 1..64, 1..4096')
    if str(cfg.get('selfplay_root_search', 'puct')) == 'gumbel':
        raise ValueError('selfplay_sims_heavy with selfplay_root_search gumbel: exclusive (docs/search.md 3.11)')
    return {'sims_heavy': sh, 'sims_heavy_fast': sf or None, 'chance_k_heavy': None if ck == -1 else ck,
            'root_min_visits_heavy': k, 'heavy_prior_max': gate, 'heavy_depth': probe['selfplay_heavy_depth'],
            'heavy_horizon': probe['selfplay_heavy_horizon'], 'heavy_lines': probe['selfplay_heavy_lines']}


def check(cfg, where=''):
    """`settings` for the run's start (SystemExit with the reason, `where` a prefix) -> the settings or None."""
    try:
        return settings(cfg)
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None


def seat_keys(cfg):
    """The heavy keyword arguments of `games.nn_seat` for the run's self-play seat ({} when off: the seat
    of before, key for key)."""
    s = settings(cfg)
    return {} if s is None else dict(s)


def fingerprint(cfg):
    """The run's heavy decisions as its self-play driver plays them ('' when off): what a remote worker's
    hello reports and the run checks (an older checkout reports none)."""
    s = settings(cfg)
    if s is None:
        return ''
    return ','.join(f'{k}={s[k]}' for k in ('sims_heavy', 'sims_heavy_fast', 'chance_k_heavy', 'root_min_visits_heavy',
                                            'heavy_prior_max', 'heavy_depth', 'heavy_horizon', 'heavy_lines'))


def describe(s):
    """A start-up line for the settings (`settings`'s dict)."""
    fast = s['sims_heavy_fast'] or s['sims_heavy']
    ck = 'the run\'s chance_k' if s['chance_k_heavy'] is None else f'chance_k {s["chance_k_heavy"]}'
    return (f'heavy decisions: {s["sims_heavy"]} simulations (full draws) / {fast} (fast draws), {ck}, root floor '
            f'{s["root_min_visits_heavy"]}; probe: options of prior <= {s["heavy_prior_max"]}, depth {s["heavy_depth"]}, '
            f'horizon {s["heavy_horizon"]}, {s["heavy_lines"]} lines')


# ---------------------------------------------------------------- metrics

FIELDS = ('probed', 'decisions', 'recorded', 'sims', 'forced', 'lines', 'steps', 'probe_ns', 'sims_all')


def new_tally():
    return {'games': 0, 'searched': 0, **{k: 0 for k in FIELDS}}


def tally(rows, acc=None):
    """Add finished games' result rows (their `heavy`, `decisions`) to `acc` -> acc. Rows without `heavy`
    (a driver without sims_heavy) are not counted."""
    acc = acc if acc is not None else new_tally()
    for r in rows:
        h = r.get('heavy')
        if h is None:
            continue
        acc['games'] += 1
        acc['searched'] += int(r.get('decisions', 0))
        for k in FIELDS:
            acc[k] += int(h.get(k, 0))
    return acc


def merge(a, b):
    """Two tallies (either may be None) -> their sum (None if both are)."""
    if a is None:
        return dict(b) if b else None
    if not b:
        return a
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def summary(acc):
    """A tally -> the self-play line's `heavy` (per game; shares)."""
    acc = acc or new_tally()
    g = max(1, acc['games'])
    d, p = acc['decisions'], max(1, acc['probed'])
    return {'games': acc['games'],
            'per_game': round(d / g, 2),
            'share': round(d / max(1, acc['searched']), 4),
            'probed_share': round(acc['probed'] / max(1, acc['searched']), 4),
            'recorded_per_game': round(acc['recorded'] / g, 2),
            'sims_per_game': round(acc['sims'] / g, 1),
            'sims_share': round(acc['sims'] / max(1, acc['sims_all']), 4),
            'sims_all_per_game': round(acc['sims_all'] / g, 1),
            'forced_per_game': round(acc['forced'] / g, 1),
            'probe_lines': round(acc['lines'] / p, 2),
            'probe_steps': round(acc['steps'] / p, 2),
            'probe_ms': round(acc['probe_ns'] / p / 1e6, 4)}


def fmt(s):
    """The log line's text for a `summary`."""
    return (f'{s["per_game"]} heavy decisions per game ({100 * s["share"]:.1f}% of the searched, '
            f'{100 * s["probed_share"]:.1f}% probed), {s["recorded_per_game"]} recorded; {s["sims_per_game"]:.0f} '
            f'simulations per game on them ({100 * s["sims_share"]:.1f}% of {s["sims_all_per_game"]:.0f}), '
            f'{s["forced_per_game"]:.0f} forced; probe {s["probe_lines"]} lines / {s["probe_ms"]:.3f} ms a decision '
            f'over {s["games"]} games')
