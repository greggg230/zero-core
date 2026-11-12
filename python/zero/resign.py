"""Self-play resignation with a no-resign share (docs/az.md 27, docs/search.md 3.14; KataGo's).

A self-play game ends when `selfplay_resign_k` consecutive full searches of the self-play seat (the
searches that make its training records; never a fast or minor one) found one seat's root win estimate
>= 1 - `selfplay_resign_t` - the same seat each time -, scored as that seat's win, every record made so far
kept (their z the resignation's outcome). A share `selfplay_resign_noresign` of the games (a hash of the
driver seed and the game id: no random stream moves) never resigns: they keep the value head seeing the
decided positions and measure the rule - each reports where it would have resigned, and whether the
favoured seat then won (the live wrong-resignation rate, `selfplay.resign.wrong_rate`).

Run settings (zero.run.DEFAULTS): `selfplay_resign_t` (0 = off: nothing changes), `selfplay_resign_k`
(default 5), `selfplay_resign_noresign` (default 0.1; 1 = a dry run: no game resigns, every one reports).
The self-play seat's keys `resign_t`, `resign_k`, `resign_noresign` (games.nn_seat); evaluations, the
league's opponents and the Play tab never resign. Two-player single-game runs only (not co-training, not
the self-play league).

Remote self-play: the workers resign exactly as this machine (the run's config.json); a worker whose
checkout plays otherwise (an older one reports nothing) is refused by its hello (`fingerprint`).
"""

KEYS = ('selfplay_resign_t', 'selfplay_resign_k', 'selfplay_resign_noresign')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def enabled(cfg):
    return float(cfg.get('selfplay_resign_t', 0.0) or 0.0) > 0.0


def check_values(t, k, noresign, what='selfplay_resign'):
    """The settings' ranges -> None, else ValueError (the driver's `NnSeat::check_resign`)."""
    if not (_num(t) and 0.0 <= t < 0.5):
        raise ValueError(f'{what}_t {t!r}: a threshold in [0, 0.5) (0 = off; a search is decided when a seat\'s '
                         f'root win estimate is >= 1 - t)')
    if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1):
        raise ValueError(f'{what}_k {k!r}: an integer >= 1 (consecutive decided full searches)')
    if not (_num(noresign) and 0.0 <= noresign <= 1.0):
        raise ValueError(f'{what}_noresign {noresign!r}: a share in [0, 1] of the games that never resign')


def check_settings(cfg, where=''):
    """The run settings (zero.run.resolve_config) -> None, else SystemExit."""
    try:
        check_values(cfg.get('selfplay_resign_t', 0.0), cfg.get('selfplay_resign_k', 5),
                     cfg.get('selfplay_resign_noresign', 0.1))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    if not enabled(cfg):
        return
    if cfg.get('resolver'):
        raise SystemExit(f'{where}selfplay_resign_t: single-game runs only (not co-training: deferred terminals '
                         f'take their outcome from outside)')
    from . import league as L
    if L.enabled(cfg):
        raise SystemExit(f'{where}selfplay_resign_t: not with the self-play league (selfplay_league 0)')
    players = cfg.get('players', [2])
    if list(players) != [2] * len(players):
        raise SystemExit(f'{where}selfplay_resign_t: two-player games only (players {players})')


def seat_keys(cfg):
    """The self-play seat's resignation keyword arguments of `games.nn_seat` ({} when off: the seat of before)."""
    if not enabled(cfg):
        return {}
    return dict(resign_t=float(cfg['selfplay_resign_t']), resign_k=int(cfg.get('selfplay_resign_k', 5)),
                resign_noresign=float(cfg.get('selfplay_resign_noresign', 0.1)))


def fingerprint(cfg):
    """What a remote worker's hello reports (zero/remoteplay.py) and the run checks: '' when off."""
    k = seat_keys(cfg)
    if not k:
        return ''
    return f't={k["resign_t"]} k={k["resign_k"]} noresign={k["resign_noresign"]}'


def new_tally():
    return {'games': 0, 'resigned': 0, 'resigned_samples': 0, 'resigned_decisions': 0, 'noresign': 0,
            'noresign_samples': 0, 'noresign_decisions': 0, 'would': 0, 'wrong': 0, 'would_samples_after': 0,
            'would_decisions_after': 0, 'never': 0}


def tally(rows, acc):
    """Add finished-game rows (accepted ones) to `acc` (new_tally). A row without `resign` (a game whose seats do
    not resign) counts in `games` only. A chained game's contest (docs/az.md 42) counts its episodes, each with its
    own `resign`: the rule resigns a game of the contest, never the contest."""
    for r in (e for row in rows for e in (row.get('episodes') or [row])):
        acc['games'] += 1
        x = r.get('resign')
        if not x:
            continue
        samples, dec = int(r.get('samples', 0) or 0), int(r.get('decisions', 0) or 0)
        if x['resigned']:
            acc['resigned'] += 1
            acc['resigned_samples'] += samples
            acc['resigned_decisions'] += dec
        elif x['noresign']:
            acc['noresign'] += 1
            acc['noresign_samples'] += samples
            acc['noresign_decisions'] += dec
            if x.get('winner') is None:
                acc['never'] += 1
            elif r.get('over'):
                # where it would have resigned: the rows and searched decisions after that point, and whether the
                # favoured seat did win (a draw or a loss: a wrong resignation)
                acc['would'] += 1
                acc['wrong'] += int(r['result'][int(x['winner'])] < 1.0 - 1e-6)
                acc['would_samples_after'] += max(0, samples - int(x['records']))
                acc['would_decisions_after'] += max(0, dec - int(x['decision']))
    return acc


def merge(a, b):
    """Two tallies summed (None: the other)."""
    if not a:
        return b
    if not b:
        return a
    return {k: a.get(k, 0) + b.get(k, 0) for k in new_tally()}


def summary(acc):
    """The iteration's `selfplay.resign`: games resigned and their share; no-resign games, those where the rule
    would have fired (`would`) and the wrong ones among them (`wrong`, `wrong_rate`: the live estimate of the
    wrong-resignation rate); the share of a no-resign game's samples and searched decisions after the point
    where it would have resigned (`rows_saved`, `decisions_saved`: what resigning saves per game)."""
    n = acc['games']
    nr = acc['noresign']
    out = {'games': n, 'resigned': acc['resigned'], 'resigned_share': round(acc['resigned'] / max(1, n), 4),
           'noresign': nr, 'would': acc['would'], 'would_share': round(acc['would'] / max(1, nr), 4),
           'wrong': acc['wrong'], 'wrong_rate': round(acc['wrong'] / max(1, acc['would']), 4),
           'rows_saved': round(acc['would_samples_after'] / max(1, acc['noresign_samples']), 4),
           'decisions_saved': round(acc['would_decisions_after'] / max(1, acc['noresign_decisions']), 4),
           'samples_per_game': {'resigned': round(acc['resigned_samples'] / max(1, acc['resigned']), 1),
                                'noresign': round(acc['noresign_samples'] / max(1, nr), 1)}}
    return out


def fmt(s):
    """One log line of a summary."""
    return (f"{s['resigned']} of {s['games']} games resigned ({100 * s['resigned_share']:.1f}%); no-resign "
            f"{s['noresign']}: would have resigned {s['would']}, wrongly {s['wrong']} "
            f"({100 * s['wrong_rate']:.1f}%); a resignation saves {100 * s['rows_saved']:.1f}% of a game's rows, "
            f"{100 * s['decisions_saved']:.1f}% of its searched decisions")
