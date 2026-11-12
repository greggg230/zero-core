"""The native worker's protocol (aw/1, zero-farm PROTOCOL.md 5) end to end with a real network (zero-farm phase
0b, docs/az.md 44): `<worker> serve` is given the same self-play job twice - once left alone, once with a
`yield` after its first finished games and a `resume` a few seconds later - and the two jobs' shards must be
byte-identical (deterministic settings: groups 1). Also checked: `hello` first and within 10 s, `yielded`
within 2 s of `yield`, `status` lines at least every 30 s, `end` with reason 'done' and no unfinished games,
stdout JSON only.

    python tools/worker_serve_check.py --worker <exe> --settings <config.json> --onnx <graph> --out <dir> \
        [--games 12 --seed 5 --threads 8 --set groups=1 --set conc=12 --hold 3]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time


def run(a, settings, out, do_yield):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    p = subprocess.Popen([a.worker, 'serve'], cwd=out, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(out, 'stderr.log'), 'w', encoding='utf-8'), text=True, encoding='utf-8',
                         bufsize=1)
    events = []
    times = {}
    lines = []

    def reader():
        for line in p.stdout:
            lines.append(line.rstrip('\n'))
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {'ev': '__not_json__', 'line': line}
            ev['_t'] = time.time() - t0
            events.append(ev)

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    def send(obj):
        p.stdin.write(json.dumps(obj) + '\n')
        p.stdin.flush()

    def wait_for(pred, timeout):
        end = time.time() + timeout
        while time.time() < end:
            for e in list(events):
                if pred(e):
                    return e
            time.sleep(0.01)
        return None

    hello = wait_for(lambda e: e['ev'] == 'hello', 10)
    times['hello_s'] = None if hello is None else round(hello['_t'], 3)
    send({'op': 'job', 'lease_id': 'l_check', 'kind': 'selfplay', 'output': 'shards', 'settings': settings, 'games': a.games, 'gid_base': 0,
          'seed': a.seed, 'model': os.path.abspath(a.onnx), 'reference': None, 'threads': a.threads, 'out_dir': 'out'})
    if do_yield:
        wait_for(lambda e: e['ev'] == 'fin', 3600)
        t = time.time()
        send({'op': 'yield'})
        y = wait_for(lambda e: e['ev'] == 'yielded', 30)
        times['yielded_after_s'] = None if y is None else round(time.time() - t, 3)
        n_fin = sum(e.get('n', 0) for e in events if e['ev'] == 'fin')
        time.sleep(a.hold)
        times['fins_while_yielded'] = sum(e.get('n', 0) for e in events if e['ev'] == 'fin') - n_fin
        send({'op': 'ping'})
        t = time.time()
        send({'op': 'resume'})
        r = wait_for(lambda e: e['ev'] == 'resumed', 60)
        times['resumed_after_s'] = None if r is None else round(time.time() - t, 3)
    end = wait_for(lambda e: e['ev'] in ('end', 'error'), 7200)
    try:
        p.stdin.close()
    except OSError:
        pass
    p.wait(60)
    th.join(5)
    st = [e['_t'] for e in events if e['ev'] == 'status']
    gaps = [b - a_ for a_, b in zip([0.0] + st, st + ([end['_t']] if end else []))]
    times['status_gap_max_s'] = round(max(gaps), 2) if gaps else None
    times['not_json'] = sum(1 for e in events if e['ev'] == '__not_json__')
    times['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out, 'events.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return end, events, times


def shards(d):
    import numpy as np
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith('.npz'):
            with np.load(os.path.join(d, name)) as z:
                out.append((name, {k: (z[k].dtype.str, z[k].shape, z[k].tobytes()) for k in z.files if k != 'meta'}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--worker', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=12)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--hold', type=float, default=3.0)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    with open(a.settings, encoding='utf-8') as f:
        settings = json.load(f)
    for s in a.set:
        k, v = s.split('=', 1)
        settings[k] = json.loads(v)
    res = {}
    for name, y in (('plain', False), ('yield', True)):
        end, events, times = run(a, settings, os.path.join(a.out, name), y)
        res[name] = dict(times, end={k: v for k, v in (end or {}).items() if k != '_t'})
        print(name, json.dumps(res[name]), flush=True)
    sa, sb = shards(os.path.join(a.out, 'plain', 'out')), shards(os.path.join(a.out, 'yield', 'out'))
    res['shards'] = len(sa)
    res['rows'] = sum(int(json.loads(e)['rows']) for e in open(os.path.join(a.out, 'plain', 'events.jsonl'), encoding='utf-8')
                      if '"shard"' in e)
    res['identical'] = sa == sb and len(sa) > 0
    y = res['yield']
    ok = (res['identical'] and all(res[n]['end'].get('reason') == 'done' and not res[n]['end'].get('unfinished', {}).get('gids')
                                   for n in ('plain', 'yield'))
          and all(res[n]['hello_s'] is not None and res[n]['hello_s'] <= 10 and res[n]['not_json'] == 0
                  and res[n]['status_gap_max_s'] <= 30 for n in ('plain', 'yield'))
          and y['yielded_after_s'] is not None and y['yielded_after_s'] <= 2 and y['fins_while_yielded'] == 0)
    res['pass'] = ok
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('plain', 'yield')}))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
