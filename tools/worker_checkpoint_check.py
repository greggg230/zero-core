"""The native worker's `checkpoint` op (aw/1, zero-farm PROTOCOL.md 5) with a real network (docs/az.md 42): a
self-play job is told after its first finished game to switch to `--model2` as generation `gen + 1`; the job must
end 'done' with every game, rows searched before the switch carrying the old id and rows after it the new one
(the remote worker's phase change: the games in flight go on, their searches restart with the new network).

    python tools/worker_checkpoint_check.py --worker <exe> --settings <config.json> --onnx <graph> \
        [--model2 <graph>] --out <dir> [--games 6 --seed 3 --threads 8 --set groups=1 --set conc=6]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--worker', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--model2', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--gen', type=int, default=80)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    import numpy as np
    with open(a.settings, encoding='utf-8') as f:
        settings = json.load(f)
    for s in a.set:
        k, v = s.split('=', 1)
        settings[k] = json.loads(v)
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    p = subprocess.Popen([a.worker, 'serve'], cwd=a.out, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(a.out, 'stderr.log'), 'w', encoding='utf-8'), text=True, encoding='utf-8',
                         bufsize=1)
    events = []
    threading.Thread(target=lambda: [events.append(json.loads(ln)) for ln in p.stdout if ln.strip()], daemon=True).start()

    def send(o):
        p.stdin.write(json.dumps(o) + '\n')
        p.stdin.flush()

    send({'op': 'job', 'lease_id': 'l_ck', 'kind': 'selfplay', 'settings': settings, 'games': a.games, 'gid_base': 0,
          'seed': a.seed, 'gen': a.gen, 'model': os.path.abspath(a.onnx), 'threads': a.threads, 'out_dir': 'out'})
    while not any(e['ev'] == 'fin' for e in events):
        if any(e['ev'] == 'error' for e in events):
            raise SystemExit(f'error: {events}')
        time.sleep(0.05)
    send({'op': 'checkpoint', 'gen': a.gen + 1, 'model': os.path.abspath(a.model2 or a.onnx)})
    while not any(e['ev'] in ('end', 'error') for e in events):
        time.sleep(0.2)
    p.stdin.close()
    p.wait(60)
    end = [e for e in events if e['ev'] in ('end', 'error')][-1]
    nets = {}
    for path in sorted(glob.glob(os.path.join(a.out, 'out', '*.npz'))):
        with np.load(path) as z:
            for v, c in zip(*np.unique(z['net'], return_counts=True)):
                nets[int(v)] = nets.get(int(v), 0) + int(c)
    res = {'end': end, 'rows_by_net': nets,
           'pass': end['ev'] == 'end' and end['reason'] == 'done' and not end['unfinished']['gids']
           and set(nets) == {a.gen, a.gen + 1}}
    print(json.dumps(res))
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
