"""Self-play throughput of the native worker against the Python remote worker (zero-farm phase 0b, docs/az.md 44):
the same run settings (with the overrides given, e.g. a friend-class 8 threads), the same seed and network
(the native side its zero-onnx/1 graph through ONNX Runtime, the Python side the checkpoint through torch on
the CPU), each side a rolling self-play for `--minutes`, the runs interleaved for `--rounds` rounds (native,
Python, then Python, native, ...), the machine's load sampled every 10 s.

    python tools/worker_bench.py --root <game checkout> --package mtgz.az --config <run>/config.json \
        --checkpoint <ckpt> --onnx <its graph> --worker <exe> --out <dir> [--minutes 10 --rounds 2 \
        --set threads=8 --set groups=2 --set conc=64 --set torch_threads=1 --infer-threads 8]

Per run: leaves (network rows' leaves, the driver's counter), games finished, rows (sample rows written), seconds,
leaves/s, games/h, rows/h, the network batches (count, rows per batch: mean and the histogram by powers of two,
milliseconds per batch: mean / p50 / p95), the peak working set (MB; macOS: the physical footprint), the CPU load
(mean of the samples, %) and python processes on the machine. The native side runs `<worker> serve` (its job: a huge
game count, `quit` after the minutes; ZERO_WORKER_BATCH_HIST=1 for its batches' rows); the Python side is zero-core's
remoteplay.Worker in a child process (`end` after the minutes; its evaluator timed per batch in the child).

macOS / Linux (docs/az.md 62): the load is the 1-minute load average over the cores (%), `--nice N` runs both sides
at that niceness (beside a live run), `--runtime-dir` gives the native side its runtime pack (`libonnxruntime.dylib`),
`--rolling` makes its job a rolling one (zero-farm PROTOCOL.md 10.5: the farm's self-play leases), `--native-job
key=JSON` adds a field to its job line (the agent's tuning fields `conc`, `groups`, `infer_threads`, `spinning`), and
`--sides native,python` picks the sides (default both). Beside a live run the numbers are contended: the RATIO of the
interleaved sides is what they measure.
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
    if os.name != 'nt':
        return load_sampler_posix(stop, samples)
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


def load_sampler_posix(stop, samples):
    """(load %, python processes) every 10 s: the 1-minute load average over the cores, `ps`'s python processes."""
    n = os.cpu_count() or 1
    while not stop.is_set():
        try:
            ps = subprocess.run(['ps', '-Ao', 'comm'], capture_output=True, text=True, timeout=20).stdout
            py = sum(1 for ln in ps.splitlines() if os.path.basename(ln.strip()).lower().startswith('python'))
            samples.append((100.0 * os.getloadavg()[0] / n, py))
        except Exception:                          # noqa: BLE001 - a missed sample
            pass
        stop.wait(10)


def lower_priority(nice):
    """A preexec_fn for a child at niceness at least `nice` (POSIX; a niced harness's children inherit its own), else
    None."""
    if not nice or os.name == 'nt':
        return None

    def f():
        try:
            if os.getpriority(os.PRIO_PROCESS, 0) < nice:
                os.setpriority(os.PRIO_PROCESS, 0, nice)
        except OSError:
            pass
    return f


def peak_wset(pid):
    """The peak working set (MB) of `pid` and its children, the largest (a venv's python.exe is a launcher: the
    interpreter is its child)."""
    try:
        import psutil
        p = psutil.Process(pid)
        key = 'peak_wset' if os.name == 'nt' else 'rss'
        return max(getattr(q.memory_info(), key) for q in [p] + p.children(recursive=True)) / 2 ** 20
    except Exception:                              # noqa: BLE001
        return None


def native(a, cfg, out, seed, net):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    p = subprocess.Popen([a.worker, 'serve'], cwd=out, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(out, 'stderr.log'), 'w', encoding='utf-8'), text=True, encoding='utf-8',
                         bufsize=1, env=dict(os.environ, ZERO_WORKER_BATCH_HIST='1'), preexec_fn=lower_priority(a.nice))
    events = []
    threading.Thread(target=lambda: [events.append(json.loads(ln)) for ln in p.stdout if ln.strip()], daemon=True).start()
    job = {'op': 'job', 'lease_id': 'bench', 'kind': 'selfplay', 'output': 'shards', 'settings': cfg, 'games': 1_000_000, 'gid_base': 0,
           'seed': seed, 'gen': net, 'model': os.path.abspath(a.onnx), 'threads': int(cfg['threads']), 'out_dir': 'out'}
    if a.infer_threads:
        job['infer_threads'] = a.infer_threads
    if a.runtime_dir:
        job['runtime_dir'] = os.path.abspath(a.runtime_dir)
    if a.rolling:
        job.update(rolling=True, flush_s=60)
    for kv in a.native_job:
        k, v = kv.split('=', 1)
        job[k] = json.loads(v)
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
    nb = st.get('infer_batches') or 0
    return {'leaves': st['leaves'], 'games': st['games'], 'rows': st.get('rows'), 'seconds': round(st['wall_s'], 1),
            'wall_s': round(secs, 1), 'peak_mb': st.get('peak_rss_mb'), 'infer_batches': nb,
            'batch_rows_mean': round(st['infer_rows'] / nb, 1) if nb else None, 'batch_rows_hist': st.get('batch_rows_hist'),
            'infer_ms_mean': round(st['infer_ms'] / nb, 2) if nb else None, 'infer_ms_p50': st.get('infer_ms_p50'),
            'infer_ms_p95': st.get('infer_ms_p95'), 'infer_ms': st.get('infer_ms'),
            'job': {k: job[k] for k in ('threads', 'infer_threads', 'conc', 'groups', 'rolling', 'spinning') if k in job}}


def python_child(a):
    """The Python remote worker for `--minutes` (in this child process)."""
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    from zero import remoteplay as RP
    it = a.iter

    batches = []                                   # (rows, seconds) per network batch

    class Timed:
        """The evaluator, each batch's rows and time recorded (`launch` + `result` in the pipelined drive)."""

        def __init__(self, ev):
            self.ev = ev

        def __getattr__(self, k):
            return getattr(self.ev, k)

        def __call__(self, lb):
            t = time.perf_counter()
            r = self.ev(lb)
            batches.append((lb.B, time.perf_counter() - t))
            return r

        def launch(self, lb):
            t = time.perf_counter()
            h = self.ev.launch(lb)
            return (h, lb.B, time.perf_counter() - t)

        def result(self, h):
            t = time.perf_counter()
            r = self.ev.result(h[0])
            batches.append((h[1], h[2] + time.perf_counter() - t))
            return r

    class W(RP.Worker):
        def _stdin(self):
            threading.Event().wait()

        def _load(self, it_, model, league=None):
            r = super()._load(it_, model, league)
            if self.evr is not None and not isinstance(self.evr, Timed):
                self.evr = Timed(self.evr)
            return r

        def emit(self, ev, **kw):
            super().emit(ev, **kw)
            if ev == 'fin':
                self.ctl.put({'cmd': 'accept', 'iter': kw['iter'], 'n': kw['n']})
            elif ev == 'end' and kw.get('iter') == it:
                kw = dict(kw, rows=kw.get('samples'), batches=batch_summary(batches))
                with open(os.path.join(self.rdir, 'end.json'), 'w', encoding='utf-8') as f:
                    json.dump(kw, f)
                self.ctl.put({'cmd': 'quit'})

    w = W(a.mirror, 0)
    w.ctl.put({'cmd': 'phase', 'iter': it, 'model': f'models/iter_{it - 1:04d}.pt'})
    threading.Timer(a.minutes * 60, lambda: w.ctl.put({'cmd': 'end', 'iter': it})).start()
    os._exit(w.main())


def batch_summary(batches):
    """Rows per batch (mean, histogram by powers of two) and milliseconds per batch (mean, p50, p95)."""
    if not batches:
        return {}
    rows = [b[0] for b in batches]
    ms = sorted(b[1] * 1e3 for b in batches)
    hist = {}
    for r in rows:
        k = str(1 << (max(1, int(r)).bit_length() - 1))
        hist[k] = hist.get(k, 0) + 1
    return {'infer_batches': len(batches), 'batch_rows_mean': round(sum(rows) / len(rows), 1), 'batch_rows_hist': hist,
            'infer_ms_mean': round(sum(ms) / len(ms), 2), 'infer_ms_p50': round(ms[len(ms) // 2], 3),
            'infer_ms_p95': round(ms[int(0.95 * (len(ms) - 1))], 3), 'infer_ms': round(sum(ms))}


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
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), preexec_fn=lower_priority(a.nice))
    peak = 0.0
    while p.poll() is None:
        m = peak_wset(p.pid)
        peak = max(peak, m or 0.0)
        time.sleep(5)
    with open(os.path.join(out, 'end.json'), encoding='utf-8') as f:
        end = json.load(f)
    return {'leaves': end['leaves'], 'games': end['games'], 'rows': end.get('rows'), 'seconds': end['seconds'],
            'wall_s': round(time.time() - t0, 1), 'peak_mb': round(peak) if peak else None, **(end.get('batches') or {})}


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
    ap.add_argument('--runtime-dir', help='the native side\'s runtime pack (its job\'s runtime_dir)')
    ap.add_argument('--rolling', action='store_true', help='the native side\'s job a rolling one (PROTOCOL.md 10.5)')
    ap.add_argument('--native-job', action='append', default=[], help='key=JSON: a field of the native job line')
    ap.add_argument('--nice', type=int, default=0, help='POSIX: both sides at this niceness')
    ap.add_argument('--sides', default='native,python', help='the sides to run, in round 0\'s order')
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
    sides = [x for x in a.sides.split(',') if x]
    for r in range(a.rounds):
        for side in (sides if r % 2 == 0 else sides[::-1]):
            stop, samples = threading.Event(), []
            th = threading.Thread(target=load_sampler, args=(stop, samples), daemon=True)
            th.start()
            out = os.path.join(a.out, f'r{r}-{side}')
            res = native(a, cfg, out, seed, a.iter - 1) if side == 'native' else python_side(a, cfg, out, seed)
            stop.set()
            th.join(30)
            res.update(side=side, round=r, leaves_per_s=round(res['leaves'] / res['seconds'], 1),
                       games_per_h=round(res['games'] / res['seconds'] * 3600, 1),
                       rows_per_h=round(res['rows'] / res['seconds'] * 3600) if res.get('rows') is not None else None,
                       cpu_load_pct=round(sum(s[0] for s in samples) / len(samples), 1) if samples else None,
                       python_procs=round(sum(s[1] for s in samples) / len(samples), 1) if samples else None)
            runs.append(res)
            print(json.dumps(res), flush=True)
    settings = {k: cfg.get(k) for k in ('threads', 'groups', 'conc', 'torch_threads', 'infer_workers', 'sims', 'sims_fast',
                                        'p_full', 'sims_minor')}
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'settings': settings, 'infer_threads': a.infer_threads, 'minutes': a.minutes, 'runs': runs}, f, indent=1)


if __name__ == '__main__':
    main()
