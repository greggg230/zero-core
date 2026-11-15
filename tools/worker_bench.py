"""Self-play throughput of the native worker against the Python remote worker (zero-farm phase 0b, docs/az.md 44):
the same run settings (with the overrides given, e.g. a friend-class 8 threads), the same seed and network
(the native side its zero-onnx/1 graph through ONNX Runtime, the Python side the checkpoint through torch on
the CPU), each side a rolling self-play for `--minutes`, the runs interleaved for `--rounds` rounds (native,
Python, then Python, native, ...), the machine's load sampled every 10 s.

    python tools/worker_bench.py --root <game checkout> --package mtgz.az --config <run>/config.json \
        --checkpoint <ckpt> --onnx <its graph> --worker <exe> --out <dir> [--minutes 10 --rounds 2 \
        --set threads=8 --set groups=2 --set conc=64 --set torch_threads=1 --infer-threads 8]

Per run: leaves (network rows' leaves, the driver's counter), games finished, seconds, leaves/s, the peak
working set (MB), the CPU load (mean of the samples, %) and python processes on the machine. The native
side runs `<worker> serve` (its job: a huge game count, `quit` after the minutes); the Python side is
zero-core's remoteplay.Worker in a child process (`end` after the minutes).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time


def load_sampler(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.run(['wmic', 'cpu', 'get', 'loadpercentage'], capture_output=True, text=True, timeout=20).stdout
            vals = [int(x) for x in out.split() if x.isdigit()]
            tl = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=20).stdout
            py = sum(1 for ln in tl.splitlines() if ln.lower().startswith(('python.exe', 'python3.13.exe')))
            if vals:
                samples.append((sum(vals) / len(vals), py))
        except Exception:                          # noqa: BLE001 - a missed sample
            pass
        stop.wait(10)


def peak_wset(pid):
    """The peak working set (MB) of `pid` and its children, the largest (a venv's python.exe is a launcher: the
    interpreter is its child)."""
    try:
        import psutil
        p = psutil.Process(pid)
        return max(q.memory_info().peak_wset for q in [p] + p.children(recursive=True)) / 2 ** 20
    except Exception:                              # noqa: BLE001
        return None


def native(a, cfg, out, seed, net):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    p = subprocess.Popen([a.worker, 'serve'], cwd=out, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(out, 'stderr.log'), 'w', encoding='utf-8'), text=True, encoding='utf-8',
                         bufsize=1)
    events = []
    threading.Thread(target=lambda: [events.append(json.loads(ln)) for ln in p.stdout if ln.strip()], daemon=True).start()
    job = {'op': 'job', 'lease_id': 'bench', 'kind': 'selfplay', 'output': 'shards', 'settings': cfg, 'games': 1_000_000, 'gid_base': 0,
           'seed': seed, 'gen': net, 'model': os.path.abspath(a.onnx), 'threads': int(cfg['threads']), 'out_dir': 'out'}
    if a.infer_threads:
        job['infer_threads'] = a.infer_threads
    t0 = time.time()
    p.stdin.write(json.dumps(job) + '\n')
    p.stdin.flush()
    time.sleep(a.minutes * 60)
    p.stdin.write(json.dumps({'op': 'quit'}) + '\n')
    p.stdin.flush()
    while not any(e['ev'] in ('end', 'error') for e in events):
        time.sleep(0.2)
    secs = time.time() - t0
    p.wait(120)
    end = [e for e in events if e['ev'] in ('end', 'error')][-1]
    if end['ev'] == 'error':
        raise SystemExit(f'native worker: {end}')
    st = end['stats']
    return {'leaves': st['leaves'], 'games': st['games'], 'seconds': round(st['wall_s'], 1), 'wall_s': round(secs, 1),
            'peak_mb': st.get('peak_rss_mb'), 'infer_ms_p50': st.get('infer_ms_p50'), 'infer_batches': st.get('infer_batches'),
            'infer_ms': st.get('infer_ms')}


def python_child(a):
    """The Python remote worker for `--minutes` (in this child process)."""
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    from zero import remoteplay as RP
    it = a.iter

    class W(RP.Worker):
        def _stdin(self):
            threading.Event().wait()

        def emit(self, ev, **kw):
            super().emit(ev, **kw)
            if ev == 'fin':
                self.ctl.put({'cmd': 'accept', 'iter': kw['iter'], 'n': kw['n']})
            elif ev == 'end' and kw.get('iter') == it:
                with open(os.path.join(self.rdir, 'end.json'), 'w', encoding='utf-8') as f:
                    json.dump(kw, f)
                self.ctl.put({'cmd': 'quit'})

    w = W(a.mirror, 0)
    w.ctl.put({'cmd': 'phase', 'iter': it, 'model': f'models/iter_{it - 1:04d}.pt'})
    threading.Timer(a.minutes * 60, lambda: w.ctl.put({'cmd': 'end', 'iter': it})).start()
    os._exit(w.main())


def python_side(a, cfg, out, seed):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, 'models'))
    shutil.copy(a.checkpoint, os.path.join(out, 'models', f'iter_{a.iter - 1:04d}.pt'))
    with open(os.path.join(out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, 'selfplay.json'), 'w', encoding='utf-8') as f:
        json.dump({'mode': 'selfplay', 'locks': [], 'nice': 0, 'run_id': 'bench', 'mirror': 'bench', 'watchdog_s': 1e9,
                   'linger': False, 'test': {}}, f)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--python-child', '--mirror', out, '--root', a.root,
                          '--package', a.package, '--minutes', str(a.minutes), '--iter', str(a.iter)],
                         stdout=open(os.path.join(out, 'stdout.log'), 'w', encoding='utf-8'),
                         stderr=open(os.path.join(out, 'stderr.log'), 'w', encoding='utf-8'),
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    peak = 0.0
    while p.poll() is None:
        m = peak_wset(p.pid)
        peak = max(peak, m or 0.0)
        time.sleep(5)
    with open(os.path.join(out, 'end.json'), encoding='utf-8') as f:
        end = json.load(f)
    return {'leaves': end['leaves'], 'games': end['games'], 'seconds': end['seconds'], 'wall_s': round(time.time() - t0, 1),
            'peak_mb': round(peak) if peak else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root')
    ap.add_argument('--package')
    ap.add_argument('--config')
    ap.add_argument('--checkpoint')
    ap.add_argument('--onnx')
    ap.add_argument('--worker')
    ap.add_argument('--out')
    ap.add_argument('--minutes', type=float, default=10)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--iter', type=int, default=81)
    ap.add_argument('--infer-threads', type=int, default=None)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--python-child', action='store_true')
    ap.add_argument('--mirror')
    a = ap.parse_args()
    if a.python_child:
        return python_child(a)
    with open(a.config, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.update(eval_remote='', selfplay_remote='', resolver='', resolver_cfg={}, resolver_init='', infer_device='cpu')
    cfg['shard_rows'] = min(int(cfg['shard_rows']), 8192)
    for s in a.set:
        k, v = s.split('=', 1)
        cfg[k] = json.loads(v)
    seed = int(cfg['seed']) * 7919 + a.iter + 7_777_777
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for r in range(a.rounds):
        for side in (('native', 'python') if r % 2 == 0 else ('python', 'native')):
            stop, samples = threading.Event(), []
            th = threading.Thread(target=load_sampler, args=(stop, samples), daemon=True)
            th.start()
            out = os.path.join(a.out, f'r{r}-{side}')
            res = native(a, cfg, out, seed, a.iter - 1) if side == 'native' else python_side(a, cfg, out, seed)
            stop.set()
            th.join(30)
            res.update(side=side, round=r, leaves_per_s=round(res['leaves'] / res['seconds'], 1),
                       games_per_h=round(res['games'] / res['seconds'] * 3600, 1),
                       cpu_load_pct=round(sum(s[0] for s in samples) / len(samples), 1) if samples else None,
                       python_procs=round(sum(s[1] for s in samples) / len(samples), 1) if samples else None)
            runs.append(res)
            print(json.dumps(res), flush=True)
    settings = {k: cfg[k] for k in ('threads', 'groups', 'conc', 'torch_threads', 'infer_workers', 'sims', 'sims_fast',
                                    'p_full', 'sims_minor')}
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'settings': settings, 'infer_threads': a.infer_threads, 'minutes': a.minutes, 'runs': runs}, f, indent=1)


if __name__ == '__main__':
    main()
