"""zero-farm eval jobs on the desktop's side (docs/az.md 44.2, zero-farm PROTOCOL.md 5.2): what an eval job's
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

Traced rows (the eval setting `eval_trace`, docs/az.md 50; zero-farm analysis campaigns): each row also carries
`digest` (`trace_digest` of its record) and `rec` (the game's record, zero-core docs/search.md 2.17: seed, every real
move, result, margins, rounds, setup) - `replay_trace` plays it back through the game's engine (`replay_rows`) and
checks it against the row. Only the native worker writes them (a Python match refuses the setting).

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
    if cfg.get('eval_trace'):
        raise ValueError('eval_trace: traced rows are the native worker\'s (zero_worker evaljob); a Python match '
                         'plays the same games untraced (tracing changes no game, docs/search.md 2.17)')
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


def trace_digest(rec):
    """A game record's digest (docs/az.md 50): the first 16 hex of the SHA-256 of its compact JSON with sorted keys
    (`zero_worker::evaljob::traced_row_json` hashes serde_json's text of the same object, which is this text)."""
    import hashlib
    text = json.dumps(rec, sort_keys=True, separators=SEP, ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def games_digest(rows):
    """The digest of a set of traced games (a lease's, a campaign's): SHA-256 over 'gid:digest' lines in gid order,
    first 16 hex - the same games give the same digest whoever played them and however they were cut into leases."""
    import hashlib
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: int(r['gid'])):
        h.update(f"{int(r['gid'])}:{r['digest']}\n".encode())
    return h.hexdigest()[:16]


class TraceMismatch(ValueError):
    """A traced row that its own record does not support (a wrong digest, or a replay that refuses it)."""


def check_digest(row):
    """A traced row's `digest` is its record's (`trace_digest`) and the record is this game's (gid) -> the row;
    TraceMismatch otherwise."""
    rec = row.get('rec')
    if not isinstance(rec, dict) or not isinstance(row.get('digest'), str):
        raise TraceMismatch(f'game {row.get("gid")}: not a traced row (no rec / digest: eval_trace off?)')
    if rec.get('gid') != row.get('gid'):
        raise TraceMismatch(f'game {row.get("gid")}: its record is game {rec.get("gid")}')
    d = trace_digest(rec)
    if d != row['digest']:
        raise TraceMismatch(f'game {row["gid"]}: digest {row["digest"]} but its record hashes to {d}')
    return row


def replay_trace(row, game, spec, engine, seed, paired=True):
    """Replay a traced row's record through the game's engine (`engine.replay_rows(game, spec, record)`, zero-core
    docs/search.md 2.17: every move legal, the result, margins, rounds and setup the record's) and check it against
    the row: the game's seed is `game_seed_of(seed, gid, paired)` (the lease's base seed), the replayed result is the
    row's (seat order of the row) and the record has no samples (an eval game records none) -> the replay's facts;
    TraceMismatch on any difference."""
    import struct
    check_digest(row)
    rec = row['rec']
    g = int(row['gid'])
    want = engine.game_seed_of(int(seed), g, bool(paired))
    if rec.get('seed') != want:
        raise TraceMismatch(f'game {g}: record seed {rec.get("seed")}, the campaign\'s game seed is {want}')
    try:
        samples, info = engine.replay_rows(game, spec, json.dumps(rec, separators=SEP), False)
    except ValueError as e:
        raise TraceMismatch(f'game {g}: the replay refuses its record: {e}') from None
    if samples is not None or info.get('rows'):
        raise TraceMismatch(f'game {g}: its record has samples (an eval game records none)')
    f32 = lambda bits: struct.unpack('<f', struct.pack('<I', int(bits) & 0xffffffff))[0]
    rres = [float(f32(b)) for b in rec.get('result') or []]
    if rec.get('over') and [float(x) for x in row['result']] != rres:
        raise TraceMismatch(f'game {g}: row result {row["result"]}, record result {rres}')
    if bool(rec.get('over')) != bool(row.get('over')) or int(rec.get('rounds', -1)) != int(row.get('rounds', -2)):
        raise TraceMismatch(f'game {g}: row over/rounds {row.get("over")}/{row.get("rounds")}, record '
                            f'{rec.get("over")}/{rec.get("rounds")}')
    return info


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
