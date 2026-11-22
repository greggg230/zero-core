"""Self-play games dropped after a panic (docs/search.md 2.18, 2.19): the run's failure queue and its replay.

A panic inside one self-play game (an engine bug) costs that game, not the run: the driver drops it and keeps what
replays it (`Driver::take_failures`). Every executor's failures end up in one file, `<run>/game_failures.jsonl`,
one JSON object a line (`record`):

    t, iter, source ('local': this machine's driver; a remote's name, e.g. 'mac'; 'pool': the zero-farm pool -
    never a farm machine), game, game_id, driver_seed, game_seed, players, spec (JSON text), moves (the real moves
    before the panic, a trace record's steps), phase ('move' | 'search'), panic_move, tag, start, branch, message,
    site (the panic's file:line:col, else 'unknown'), engine_commit (the code's commit), build (the engine module's
    sha256[:24], or a farm worker's build_id), repro (the command below, for this line)

    python -m <package>.az.game_failure replay FILE [--line N] [--rollouts 64] [--seed 0] [--max-steps 20000]
    python -m <package>.az.game_failure list FILE

`replay` (from a checkout at the line's engine_commit, its engine module built): the game from its spec, players and
seed, its real moves, then where it panicked - a 'move' panic's action, or `--rollouts` random continuations of
that state, determinized half of the time, each step encoding the decider's row as a search leaf does
(`zero_search::record::replay_failure`). Exit 101 and the panic's text when it panics again (reproduced); 0 when
nothing panicked (not reproduced: a search panic may need the search's own state); 2 when the line cannot be
replayed. A test fault's panic names the setting that armed it (`[test fault NAME=VALUE]`, NAME ending in
_TEST_PANIC): the replay arms it the same way.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

FILE = 'game_failures.jsonl'
KEY_CHARS = 120
REPRODUCED = 101
SITE_RE = re.compile(r'panicked at ([^\s:]+(?::[^\s:]+)*?:\d+:\d+)')
FAULT_RE = re.compile(r'\[test fault ([A-Z0-9_]*_TEST_PANIC)=([^\]\s]+)\]')
_ID = {}


def norm(s):
    return re.sub(r'\d+', '#', str(s).strip())[:300]


def failure_key(site, message):
    """The dedup key of a failure (tools/fuzz_farm.py's in MTG Zero): its site and the first KEY_CHARS characters of
    its message's first non-empty line, digits masked."""
    lines = [x for x in str(message or '').splitlines() if x.strip()]
    return f'{site}\n{norm(lines[0] if lines else "")[:KEY_CHARS]}'


def site_of(f):
    """A driver failure's site (`site`, its panic hook's note), else the 'panicked at <file:line:col>' in its
    message, else 'unknown'."""
    if f.get('site'):
        return str(f['site'])
    m = SITE_RE.search(str(f.get('message') or ''))
    return m.group(1) if m else 'unknown'


def head_commit(root):
    """The commit of the checkout at `root` (cached), or None."""
    k = ('commit', root)
    if k not in _ID:
        try:
            _ID[k] = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                    timeout=30).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            _ID[k] = None
    return _ID[k]


def module_build(mod):
    """sha256[:24] of a compiled module's file (cached): the exact engine that played."""
    path = getattr(mod, '__file__', None)
    k = ('build', path)
    if k not in _ID:
        try:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for b in iter(lambda: f.read(1 << 20), b''):
                    h.update(b)
            _ID[k] = h.hexdigest()[:24]
        except (OSError, TypeError):
            _ID[k] = None
    return _ID[k]


def local_ids():
    """(engine_commit, build) of this process's game package: its checkout's HEAD and its engine module's hash."""
    from . import games as G
    from . import package
    try:
        root = package.active().root
    except Exception:                              # noqa: BLE001 - (no package: none)
        root = None
    try:
        build = module_build(G.engine())
    except Exception:                              # noqa: BLE001
        build = None
    return (head_commit(root) if root else None), build


def repro_cmd(pkg, path, line):
    return f'python -m {pkg}.az.game_failure replay "{os.path.abspath(path)}" --line {line}'


def _stamp(game):
    from . import package
    return package.layout_stamp(game)       # (an aliased game's layout record, docs/az.md 73)


def record(f, *, it, source, game, engine_commit, build, path, line, pkg, t=None):
    """One game_failures.jsonl line (module docs) of a driver failure `f` (`GameFailure::to_json`'s dict; a remote's
    may carry its own engine_commit / build, which win)."""
    return {'t': round(time.time() if t is None else t, 3), 'iter': it, 'source': source, 'game': game, **_stamp(game),
            'game_id': f.get('gid'), 'driver_seed': f.get('driver_seed'), 'game_seed': f.get('game_seed'),
            'players': f.get('players'), 'spec': f.get('spec'), 'moves': f.get('steps') or [],
            'phase': f.get('phase') or 'search', 'panic_move': f.get('panic_move'), 'tag': f.get('tag'),
            'start': f.get('start'), 'branch': f.get('branch'), 'message': str(f.get('message') or ''),
            'site': site_of(f), 'engine_commit': f.get('engine_commit') or engine_commit,
            'build': f.get('build') or build, 'repro': repro_cmd(pkg, path, line)}


def append(path, recs_of_line):
    """Append records to `path`: `recs_of_line(first_line_number)` -> the records (each its own line number's
    repro). Returns them."""
    n = 0
    if os.path.exists(path):
        with open(path, 'rb') as f:
            n = sum(1 for _ in f)
    recs = recs_of_line(n + 1)
    with open(path, 'a', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, separators=(',', ':')) + '\n')
    return recs


def driver_failure(rec):
    """A record (module docs) -> the driver's failure object `replay_failure` reads."""
    return {'gid': rec.get('game_id'), 'driver_seed': rec.get('driver_seed'), 'game_seed': rec['game_seed'],
            'players': rec['players'], 'spec': rec['spec'], 'tag': rec.get('tag'), 'start': rec.get('start'),
            'branch': rec.get('branch'), 'steps': rec.get('moves') or [], 'message': rec.get('message'),
            'phase': rec.get('phase'), 'panic_move': rec.get('panic_move')}


def read(path, line=None):
    with open(path, encoding='utf-8') as f:
        lines = [x for x in f.read().splitlines()]
    if line is None:
        return [json.loads(x) for x in lines if x.strip()]
    if not 1 <= line <= len(lines) or not lines[line - 1].strip():
        raise ValueError(f'{path}: no record on line {line} ({len(lines)} lines)')
    return json.loads(lines[line - 1])


def _record_game(rec):
    from . import package
    return package.record_game(rec, default='mtg')   # (resolved, docs/az.md 73; a record of before names none: MTG's)


def replay(rec, rollouts=64, seed=0, max_steps=20000, engine=None):
    """Replay a record (module docs) -> (reproduced, text): (True, the panic's message) or (False, the replay's
    facts as JSON). ValueError when it cannot be replayed."""
    from . import games as G
    for name, value in FAULT_RE.findall(rec.get('message') or ''):
        os.environ[name] = value                   # (a test fault: armed as it was when it panicked)
    E = engine or G.engine()
    try:
        out = E.replay_failure(_record_game(rec), json.dumps(driver_failure(rec)), int(rollouts), int(seed),
                               int(max_steps))
    except BaseException as e:                     # noqa: BLE001 - pyo3's PanicException is a BaseException
        if type(e).__name__ == 'PanicException':
            return True, str(e)
        raise
    return False, out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='game_failure', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replay')
    r.add_argument('file')
    r.add_argument('--line', type=int, help='1-based line (default: the last)')
    r.add_argument('--rollouts', type=int, default=64)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--max-steps', type=int, default=20000)
    ls = sub.add_parser('list')
    ls.add_argument('file')
    a = ap.parse_args(argv)
    if a.cmd == 'list':
        for k, x in enumerate(read(a.file), 1):
            print(f'{k}: iter {x.get("iter")} {x.get("source")} game {x.get("game_id")} at {x.get("site")}: '
                  f'{str(x.get("message"))[:160]}')
        return 0
    try:
        recs = read(a.file)
        line = a.line or len(recs)
        rec = read(a.file, line)
        ok, text = replay(rec, a.rollouts, a.seed, a.max_steps)
    except (OSError, ValueError, KeyError) as e:
        print(f'game_failure: cannot replay: {e}', file=sys.stderr)
        return 2
    if ok:
        print(f'REPRODUCED (line {line}, site {rec.get("site")}): {text}', flush=True)
        return REPRODUCED
    print(f'not reproduced (line {line}): {text}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
