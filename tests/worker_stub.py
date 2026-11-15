"""A stand-in for a game's worker executable in zero-core's tests (no toy worker binary exists): `info`,
`check-config <settings.json> [--eval] --json` and `driver-cfg <settings.json>` as a native worker answers them. With
--eval every setting in zero_worker's EVAL_READS (crates/worker/src/settings.rs, read here) is `play`, every other one
`trainer`; for self-play a setting is `play` when crates/worker/src/run_settings.json classifies it so, else `trainer`;
`driver-cfg` prints the self-play `play` settings (the same driver for a run's full settings and for those alone).
Environment: WORKER_STUB_BUILD (the build id `info` names), WORKER_STUB_LOG (JSON lines: argv), WORKER_STUB_UNKNOWN
(comma-separated settings this build doesn't know - a build older than them: `unknown`, a failure, exit 1)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def eval_reads():
    with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'settings.rs'), encoding='utf-8') as f:
        m = re.search(r'pub const EVAL_READS: &\[&str\] = &\[(.*?)\];', f.read(), re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


_TABLE = []


def selfplay_play(k):
    if not _TABLE:
        with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'run_settings.json'), encoding='utf-8') as f:
            _TABLE.append(json.load(f)['keys'])
    return (_TABLE[0].get(k) or {}).get('class') == 'play'


def unknown():
    return [k for k in os.environ.get('WORKER_STUB_UNKNOWN', '').split(',') if k]


def main(argv):
    log = os.environ.get('WORKER_STUB_LOG')
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'argv': argv}) + '\n')
    if argv[:1] == ['info']:
        print(json.dumps({'build_id': os.environ.get('WORKER_STUB_BUILD', ''), 'game': 'zero-toy',
                          'kinds': ['selfplay', 'eval', 'fuzz'], 'aw': 1}))
        return 0
    if argv[:1] == ['check-config'] and '--json' in argv:
        with open(argv[1], encoding='utf-8') as f:
            cfg = json.load(f)
        old = [k for k in unknown() if k in cfg]
        if '--eval' in argv:
            reads = eval_reads()
            st = lambda k: 'unknown' if k in old else 'play' if k in reads else 'trainer'
            out = {'job': 'eval', 'game': cfg.get('game'), 'eval_reads': reads, 'driver_error': None,
                   'failures': [k for k in reads if k not in cfg and k != 'game'] + old}
        else:
            st = lambda k: 'unknown' if k in old else 'play' if selfplay_play(k) else 'trainer'
            out = {'game': cfg.get('game'), 'driver_error': None, 'failures': old}
        out['settings'] = [{'key': k, 'status': st(k), 'value': json.dumps(v)} for k, v in cfg.items()]
        print(json.dumps(out))
        return 1 if out['failures'] else 0
    if argv[:1] == ['driver-cfg']:
        with open(argv[1], encoding='utf-8') as f:
            cfg = json.load(f)
        print(json.dumps({k: cfg[k] for k in sorted(cfg) if selfplay_play(k)}, indent=1))
        return 0
    print(f'worker stub: {argv} is not stubbed', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
