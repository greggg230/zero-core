"""CPU speed of a network's zero-onnx/1 graphs (ONNX Runtime, CPU EP, each packing) against torch on the CPU
(eager `NetEvaluator`, `forward_static`), on real batch rows (docs/az.md 41).

  python tools/onnx_bench.py --checkpoint <ckpt> --batches <file.npz> --onnx-dir <dir> [--sizes 16,64,256]
      [--threads 1,4,8] [--windows 3] [--rounds 2] [--packing mask,compact,packed,host] [--chunk 64]
      [--max-call-s 4] [--json out.json]

Every method answers the same row windows (consecutive rows of the file at `--windows` starting points per
batch size) from the driver's sparse rows to an evaluator's answer (priors, value, margin, belief rows'
auxiliary predictions): torch densifies the rows (as LeafBatch.x) and runs the eager evaluator, or packs
them on the host and runs forward_static; ONNX Runtime runs the graph (the 'host' packing packs on the host
first, numpy) and the worker's generic step (`onnx_parity.answer`); 'ort-packed-c64' is the 'packed' graph
run in pieces of `--chunk` rows grouped by their non-zero count (`onnx_parity.run_chunked`: a worker-side
policy that needs nothing of the network). Methods are interleaved (each round
runs every method on every window, in a rotating order) so the machine's background load biases all of
them alike; one warm-up call per method and window first. Reported: the median seconds per call and
leaves/s per method, with the machine's CPU load during each configuration (psutil, system-wide), the
sessions' load time and the files' sizes. A method whose warm-up call takes over `--max-call-s` (torch with
several threads on a saturated machine: its OpenMP threads collapse) is timed by that one call ('n': 1).
CPU only (CUDA_VISIBLE_DEVICES is cleared).
"""
import argparse
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load_info():
    import psutil
    procs = 0
    for p in psutil.process_iter(['name']):
        try:
            if (p.info['name'] or '').lower().startswith(('python', 'rustc', 'cargo')):
                procs += 1
        except Exception:
            pass
    return {'python_rustc_procs': procs, 'mem_free_gb': round(psutil.virtual_memory().available / 2 ** 30, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--batches', required=True)
    ap.add_argument('--onnx-dir', required=True)
    ap.add_argument('--sizes', default='16,64,256')
    ap.add_argument('--threads', default='1,4,8')
    ap.add_argument('--windows', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--packing', default='mask,compact,packed,host')
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--max-call-s', type=float, default=4.0)
    ap.add_argument('--methods', default=None, help='a subset of the methods (comma-separated)')
    ap.add_argument('--game', default=None)
    ap.add_argument('--allow-mismatch', action='store_true')
    ap.add_argument('--max-rows', type=int, default=4096)
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    import psutil
    from zero import onnx_export as OX, onnx_parity as OP
    from zero.infer import NetEvaluator, token_index
    model, info = OX.load_checkpoint(a.checkpoint, a.game, a.allow_mismatch)
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    packs = [p for p in a.packing.split(',') if p]
    paths = {}
    for p in packs:
        path = os.path.join(a.onnx_dir, f'{stem}.{p}.onnx')
        if not os.path.exists(path) or OP.read_metadata(path).get('checkpoint_sha256') != info['checkpoint_sha256']:
            OX.export(model, path, p, info)
        paths[p] = path
    rows = OP.Rows.load(a.batches, a.max_rows)
    pidx = model.present_index()
    out = {'checkpoint': a.checkpoint, 'batches': a.batches, 'rows_in_file': rows.B, 'torch': torch.__version__,
           'files': {p: os.path.getsize(v) for p, v in paths.items()}, 'session_load_s': {}, 'configs': []}
    import onnxruntime as ort
    out['onnxruntime'] = ort.__version__
    for p, path in paths.items():                       # session load time (median of 3, 4 threads)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            OP.session(path, 4)
            ts.append(time.perf_counter() - t0)
        out['session_load_s'][p] = round(float(np.median(ts)), 3)
    print(json.dumps({k: out[k] for k in ('files', 'session_load_s')}), flush=True)
    sizes = [int(s) for s in a.sizes.split(',')]
    threads = [int(t) for t in a.threads.split(',')]
    for t in threads:
        if t > 8:
            raise SystemExit('at most 8 threads (a shared machine)')
        torch.set_num_threads(t)
        ev = NetEvaluator(model, 'cpu')
        sess = {p: OP.session(path, t) for p, path in paths.items()}

        def torch_eager(sub):
            return OP.eager_answer(ev, sub)

        def torch_static(sub):
            return OP.static_answer(model, sub)

        def ort_run(p):
            def f(sub):
                return OP.answer(sess[p].run(None, OP.feeds(sub, p, pidx)), sub)
            return f
        methods = {'torch-eager': torch_eager, 'torch-static': torch_static}
        methods.update({f'ort-{p}': ort_run(p) for p in paths})
        if 'packed' in paths:
            methods[f'ort-packed-c{a.chunk}'] = lambda sub: OP.answer(OP.run_chunked(sess['packed'], sub, a.chunk), sub)
        if a.methods:
            methods = {k: v for k, v in methods.items() if k in a.methods.split(',')}
        for B in sizes:
            R = rows.B
            wins = [rows.take((((k * 7919 + B * 104729) % R) + np.arange(B)) % R) for k in range(a.windows)]
            pack_s = []
            for w in wins:                                  # the host packing's own cost ('host')
                t0 = time.perf_counter()
                P = OP.presence_sparse(w, pidx)
                token_index(P, np.zeros((w.B, int(P.sum(1).max())), np.int64))
                pack_s.append(time.perf_counter() - t0)
            times = {k: [] for k in methods}
            slow = set()
            for k, f in methods.items():                    # warm-up
                for w in wins:
                    t0 = time.perf_counter()
                    f(w)
                    dt = time.perf_counter() - t0
                    if dt > a.max_call_s:
                        slow.add(k)
                        times[k].append(dt)
                        break
            names = [k for k in methods if k not in slow]
            psutil.cpu_percent(None)
            t_start = time.perf_counter()
            for r in range(a.rounds):
                order = names[r % len(names):] + names[:r % len(names)]
                for w in wins:
                    for k in order:
                        t0 = time.perf_counter()
                        methods[k](w)
                        times[k].append(time.perf_counter() - t0)
            cpu = psutil.cpu_percent(None)
            ntok = [float(OP.presence_sparse(w, pidx).sum(1).mean()) for w in wins]
            cfg = {'B': B, 'threads': t, 'cpu_percent': cpu, 'seconds': round(time.perf_counter() - t_start, 1),
                   'mean_tokens': round(float(np.mean(ntok)), 1), 'host_pack_ms': round(1e3 * float(np.median(pack_s)), 3),
                   **load_info(), 'methods': {}}
            for k, ts in times.items():
                med = float(np.median(ts))
                cfg['methods'][k] = {'ms': round(med * 1e3, 2), 'leaves_s': round(B / med, 1), 'n': len(ts)}
            out['configs'].append(cfg)
            print(json.dumps(cfg), flush=True)
        del sess, ev
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
