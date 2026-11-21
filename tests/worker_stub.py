"""A stand-in for a game's worker executable in zero-core's tests (no toy worker binary exists): `info`,
`check-config <settings.json> [--eval] --json` and `driver-cfg <settings.json>` as a native worker answers them, and `eval` (an eval job's games: tests/farm_play.py). With
--eval every setting in zero_worker's EVAL_READS (crates/worker/src/settings.rs, read here) is `play`, every other one
`trainer`; for self-play a setting is `play` when crates/worker/src/run_settings.json classifies it so, else `trainer`;
`driver-cfg` prints the self-play `play` settings (the same driver for a run's full settings and for those alone).
Environment: WORKER_STUB_BUILD (the build id `info` names), WORKER_STUB_LOG (JSON lines: argv), WORKER_STUB_UNKNOWN
(comma-separated settings this build doesn't know - a build older than them: `unknown`, a failure, exit 1),
WORKER_STUB_FEATURES (comma-separated: `info`'s `features`; unset: none, a build before rolling jobs). With the feature
`setting_classes` (docs/az.md 61) check-config honours the settings' declared classes as zero_worker classify does for
a key this build doesn't know: declared `trainer` -> unused (`trainer`), anything else or undeclared -> `unknown`; the
`setting_classes` key itself gets no verdict. `check-config --eval --seats JSON` checks the seats as the worker parses
them: a network seat's `set` (docs/az.md 68) needs the feature `seat_settings` and keys of zero_worker's SEAT_READS
(else a driver error, exit 1)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rs_list(name):
    with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'settings.rs'), encoding='utf-8') as f:
        m = re.search(r'pub const ' + name + r': &\[&str\] = &\[(.*?)\];', f.read(), re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


def eval_reads():
    return _rs_list('EVAL_READS')


def seat_error(argv):
    """`--seats JSON` as the worker's EvalSeat::parse reads it -> its refusal, or None."""
    if '--seats' not in argv:
        return None
    for st in json.loads(argv[argv.index('--seats') + 1]):
        if 'set' not in st:
            continue
        if 'seat_settings' not in features():
            return f'seat {json.dumps(st)}: one key, net or bot'
        bad = [k for k in st['set'] if k not in _rs_list('SEAT_READS')]
        if bad or not st['set']:
            return f'seat {json.dumps(st)}: {bad} is not a seat setting'
    return None


_TABLE = []


def selfplay_play(k):
    if not _TABLE:
        with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'run_settings.json'), encoding='utf-8') as f:
            _TABLE.append(json.load(f)['keys'])
    return (_TABLE[0].get(k) or {}).get('class') == 'play'


def unknown():
    return [k for k in os.environ.get('WORKER_STUB_UNKNOWN', '').split(',') if k]


def features():
    return [f for f in os.environ.get('WORKER_STUB_FEATURES', '').split(',') if f]


def eval_job(argv):
    """`eval --settings F --seats JSON --games N --gid-base G --seed S [--paired] --out D --prefix P --model GEN=M ...`:
    the games as the native worker plays them (tests/farm_play.py), one result file of traced rows, the events `evals`
    and `end` on stdout. WORKER_STUB_EVAL_FAIL: exit 3 after a fatal `error` event, nothing played."""
    flags, models, paired = {}, {}, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--paired':
            paired = True
            i += 1
            continue
        if a == '--model':
            g, _, m = argv[i + 1].partition('=')
            models[int(g)] = m
        else:
            flags[a] = argv[i + 1]
        i += 2
    if os.environ.get('WORKER_STUB_EVAL_FAIL'):
        print(json.dumps({'ev': 'error', 'message': os.environ['WORKER_STUB_EVAL_FAIL'], 'fatal': True}))
        return 3
    import farm_play
    with open(flags['--settings'], encoding='utf-8') as f:
        settings = json.load(f)
    lo, n = int(flags['--gid-base']), int(flags['--games'])
    rows = farm_play.play_traced(settings, json.loads(flags['--seats']), lo, lo + n, int(flags['--seed']), paired, models)
    out = flags.get('--out', 'out')
    os.makedirs(out, exist_ok=True)
    name = f'{flags.get("--prefix", "e")}_000.jsonl'
    with open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':')) + '\n')
    print(json.dumps({'ev': 'evals', 'seq': 0, 'path': name, 'games': n, 'gids': [[lo, lo + n]]}))
    print(json.dumps({'ev': 'end', 'reason': 'done', 'stats': {}, 'unfinished': None}))
    return 0


def main(argv):
    log = os.environ.get('WORKER_STUB_LOG')
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'argv': argv}) + '\n')
    if argv[:1] == ['info']:
        print(json.dumps({'build_id': os.environ.get('WORKER_STUB_BUILD', ''), 'game': 'zero-toy',
                          'kinds': ['selfplay', 'eval', 'fuzz'], 'aw': 1,
                          'features': features()}))
        return 0
    if argv[:1] == ['check-config'] and '--json' in argv:
        with open(argv[1], encoding='utf-8') as f:
            cfg = json.load(f)
        decl = cfg.pop('setting_classes', None) if 'setting_classes' in features() else None
        took = [k for k in unknown() if k in cfg and (decl or {}).get(k) == 'trainer']
        old = [k for k in unknown() if k in cfg and k not in took]
        if '--eval' in argv:
            reads = eval_reads()
            st = lambda k: 'unknown' if k in old else 'play' if k in reads and k not in took else 'trainer'
            out = {'job': 'eval', 'game': cfg.get('game'), 'eval_reads': reads, 'seat_reads': _rs_list('SEAT_READS'),
                   'driver_error': seat_error(argv), 'failures': [k for k in reads if k not in cfg and k != 'game'] + old}
        else:
            st = lambda k: 'unknown' if k in old else 'play' if selfplay_play(k) and k not in took else 'trainer'
            out = {'game': cfg.get('game'), 'driver_error': None, 'failures': old}
        out['settings'] = [{'key': k, 'status': st(k), 'value': json.dumps(v),
                            'why': 'unknown to this build; declared trainer-only' if k in took else ''} for k, v in cfg.items()]
        print(json.dumps(out))
        return 1 if out['failures'] or out.get('driver_error') else 0
    if argv[:1] == ['eval']:
        return eval_job(argv[1:])
    if argv[:1] == ['driver-cfg']:
        with open(argv[1], encoding='utf-8') as f:
            cfg = json.load(f)
        print(json.dumps({k: cfg[k] for k in sorted(cfg) if selfplay_play(k)}, indent=1))
        return 0
    print(f'worker stub: {argv} is not stubbed', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
