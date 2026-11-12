"""A stand-in for a game's worker executable in zero-core's tests (no toy worker binary exists): `info` and
`check-config <settings.json> --eval --json` as a native worker answers them - every setting in zero_worker's
EVAL_READS (crates/worker/src/settings.rs, read here) is `play`, every other one `trainer`. Environment:
WORKER_STUB_BUILD (the build id `info` names), WORKER_STUB_LOG (JSON lines: argv)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def eval_reads():
    with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'settings.rs'), encoding='utf-8') as f:
        m = re.search(r'pub const EVAL_READS: &\[&str\] = &\[(.*?)\];', f.read(), re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


def main(argv):
    log = os.environ.get('WORKER_STUB_LOG')
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'argv': argv}) + '\n')
    if argv[:1] == ['info']:
        print(json.dumps({'build_id': os.environ.get('WORKER_STUB_BUILD', ''), 'game': 'zero-toy',
                          'kinds': ['selfplay', 'eval', 'fuzz'], 'aw': 1}))
        return 0
    if argv[:1] == ['check-config'] and '--eval' in argv and '--json' in argv:
        with open(argv[1], encoding='utf-8') as f:
            cfg = json.load(f)
        reads = eval_reads()
        out = {'job': 'eval', 'game': cfg.get('game'), 'eval_reads': reads, 'driver_error': None,
               'failures': [k for k in reads if k not in cfg and k != 'game'],
               'settings': [{'key': k, 'status': 'play' if k in reads else 'trainer', 'value': json.dumps(v)}
                            for k, v in cfg.items()]}
        print(json.dumps(out))
        return 1 if out['failures'] else 0
    print(f'worker stub: {argv} is not stubbed', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
