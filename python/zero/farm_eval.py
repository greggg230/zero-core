"""zero-farm eval jobs on the desktop's side (docs/az.md 42.2, zero-farm PROTOCOL.md 5.2): what an eval job's
games are in Python terms, their result rows, and the readout of a head-to-head.

An eval job plays games `gid_base .. gid_base + games` of an evaluation match between two seats, each
`{"net": <gen>}` or `{"bot": <name>}`, at a run's settings, exactly as `Run.play_match` plays a match (the
Elo matches', the yardstick's): seats `Run.nn_seat(key, c['eval_sims'])` (key 'm' in seats[0], 'o' in
seats[1]; a bot `games.seat_dict`), game g with `seats` for even g and the seats swapped for odd g,
`Run.driver(table, seed, conc=min(conc, games), paired=paired)` - the run's eval spec, threads, groups,
max_steps, sparse. `match_driver` builds that driver here (the eval gate's Python side,
tools/worker_eval_gate.py); the native worker (`zero_worker::evaljob`) plays the same games.

A result row (one JSON line per game, `row_json`): `{"gid", "pair", "seats", "result", "margin", "rounds",
"over"}` - `pair` gid // 2 (None unpaired), `seats` that game's seats in its order, `result` per seat in that
order, `margin` seat 0's score minus the best other seat's, `json.dumps(row, separators=(',', ':'))`.

`h2h_rows` turns rows into zero.h2h's (A in seat gid % 2) so `zero.h2h.summary` reads a farm head-to-head
exactly as a local one; `elo_interval` is the 95% interval of the Elo difference.
"""
import json
import math

SEP = (',', ':')


class _Stub:
    """A stand-in Run: `Run.nn_seat` and `Run.driver` (record False) read only `cfg`."""

    def __init__(self, cfg):
        self.cfg = cfg


def check_seats(seats):
    """The job's seats: two of {"net": gen} / {"bot": name}; an external bot (ext:) is refused."""
    if not isinstance(seats, (list, tuple)) or len(seats) != 2:
        raise ValueError(f'seats {seats!r}: two seats')
    for s in seats:
        if not isinstance(s, dict) or len(s) != 1 or not ({'net', 'bot'} & set(s)):
            raise ValueError(f'seat {s!r}: {{"net": <gen>}} or {{"bot": <name>}}')
        if 'net' in s and (isinstance(s['net'], bool) or not isinstance(s['net'], int) or s['net'] < 0):
            raise ValueError(f'seat {s!r}: net is a checkpoint generation')
        if 'bot' in s and (not isinstance(s['bot'], str) or not s['bot'] or s['bot'].startswith('ext:')):
            raise ValueError(f'seat {s!r}: a built-in bot (an ext: player runs on the desktop, not in a worker)')
    return [dict(s) for s in seats]


def seat_dicts(cfg, seats):
    """The driver seats of the job's two seats (`Run.nn_seat(key, cfg['eval_sims'])` / `games.seat_dict`)."""
    from . import games as G, run as R
    stub = _Stub(cfg)
    out = []
    for k, s in enumerate(check_seats(seats)):
        out.append(R.Run.nn_seat(stub, ('m', 'o')[k], cfg['eval_sims']) if 'net' in s else G.seat_dict(s['bot']))
    return out


def seats_of(seats, gid):
    """Game `gid`'s seats in its order."""
    return [seats[0], seats[1]] if gid % 2 == 0 else [seats[1], seats[0]]


def job_cfg(cfg, threads=None, groups=None, conc=None):
    """The settings with a job's own threads / groups / conc over the run's (as the worker reads them)."""
    c = dict(cfg)
    for k, v in (('threads', threads), ('groups', groups), ('conc', conc)):
        if v is not None:
            c[k] = int(v)
    return c


def match_driver(cfg, seats, games, seed, paired=True):
    """`Run.play_match`'s driver for `games` games (ids 0..games) of the match between `seats` at settings
    `cfg` (the eval job's: `job_cfg`). The evaluators: one per network seat key ('m' seats[0], 'o' seats[1])."""
    from . import run as R
    if not cfg.get('eval_spec'):
        raise ValueError('eval_spec is unset: resolve it first (resolve_eval_spec): a worker cannot run the game '
                         'package\'s eval_specs hook')
    sd = seat_dicts(cfg, seats)
    table = [[sd[0], sd[1]] if g % 2 == 0 else [sd[1], sd[0]] for g in range(games)]
    return R.Run.driver(_Stub(cfg), table, seed=seed, conc=min(int(cfg['conc']), games), paired=paired)


def resolve_eval_spec(cfg):
    """The settings with `eval_spec` resolved as `Run.driver` resolves it for a match (the run's, else the game
    package's eval spec of its self-play spec), as JSON text."""
    from . import package
    from .games import spec_json
    c = dict(cfg)
    c['eval_spec'] = spec_json(c.get('eval_spec') or package.active().eval_spec_of(c['game'], c['spec']))
    return c


def row(r, seats, paired):
    """A driver result row (`take_results`) -> the job's result row (a dict, the protocol's keys in order)."""
    g = int(r['gid'])
    sc = [float(x) for x in r['scores']]
    margin = sc[0] - max(sc[j] for j in range(1, len(sc))) if len(sc) > 1 else 0.0
    return {'gid': g, 'pair': g // 2 if paired else None, 'seats': seats_of(seats, g),
            'result': [float(x) for x in r['result']], 'margin': margin, 'rounds': int(r['rounds']),
            'over': bool(r['over'])}


def row_json(d):
    return json.dumps(d, separators=SEP)


def read_rows(text):
    """Result rows (dicts) of a result file's text."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def a_seat_of(rows, a):
    """Check that `a` (A's seat, e.g. {"net": 1}) sits in seat gid % 2 of every row (`seats` [A, B] in the job)."""
    for r in rows:
        if r['seats'][r['gid'] % 2] != a:
            raise ValueError(f'game {r["gid"]}: A ({a}) is not in seat {r["gid"] % 2}: {r["seats"]}')


def h2h_rows(rows):
    """Rows of a job whose seats are [A, B] -> zero.h2h's match rows (A in seat gid % 2): `scores` [margin, 0]
    gives `zero.h2h.summary` the same margin (seat 0's margin, A's sign by its seat)."""
    return [{'gid': r['gid'], 'result': r['result'], 'scores': [r['margin'], 0.0], 'over': r['over']} for r in rows]


def elo_of(s):
    s = min(max(float(s), 1e-3), 1 - 1e-3)
    return 400 * math.log10(s / (1 - s))


def elo_interval(score, se, z=1.959964):
    """The 95% interval of the Elo difference: the score's interval (score +- z se, kept in (0, 1)) through
    the logistic Elo of zero.h2h."""
    return round(elo_of(score - z * se), 1), round(elo_of(score + z * se), 1)


def summary(rows, paired, a=None):
    """zero.h2h.summary over a head-to-head's result rows (A = seats[0] of the job), plus `elo_95`."""
    from .h2h import summary as h2h_summary
    if a is not None:
        a_seat_of(rows, a)
    out = h2h_summary(h2h_rows(sorted(rows, key=lambda r: r['gid'])), paired)
    out['elo_95'] = list(elo_interval(out['a_score'], out['se']))
    return out
