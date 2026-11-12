"""Parity gate 2 (b), the evaluator's numerics (zero-farm phase 0b, docs/az.md 42): on the same recorded
network batches, the native worker's answers (`<worker> eval-batches`: ONNX Runtime + its Rust
post-processing) vs the Python reference (`zero.onnx_eval.OnnxEvaluator`: the same graph through
onnxruntime + numpy, `onnx_parity.answer`) - within ~1e-6 - and vs torch (`infer.NetEvaluator`, eager, CPU,
the checkpoint the graph was exported from) - within 2e-5 (parity gate 1's tolerance).

    python tools/onnx_worker_gate.py --root <game checkout> --package mtgz.az --onnx <graph> \
        --checkpoint <its checkpoint> --worker <native worker exe> --batches <recorded .npz>... \
        --out <dir> [--threads 4]

Per output (priors over each row's options, value [B, 8], margin [B, 8], the belief rows' aux): the max
absolute difference of each pair over every batch of every file, and whether the native answers are
bit-identical to the Python reference's. Exit 1 when a tolerance is crossed.
"""
import argparse
import json
import os
import subprocess
import sys

TOL_PY = 1e-6
TOL_TORCH = 2e-5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--batches', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from zero.infer import NetEvaluator
    from zero.net import load
    from zero.onnx_eval import OnnxEvaluator
    from zero.onnx_parity import Rows
    os.makedirs(a.out, exist_ok=True)
    ort_py = OnnxEvaluator(a.onnx, threads=a.threads)
    model, _ = load(a.checkpoint, 'cpu', game=ort_py.meta['game'])
    tev = NetEvaluator(model, 'cpu')
    keys = ('priors', 'value', 'margin', 'aux')
    worst = {f'{p}_{k}': 0.0 for p in ('native_vs_py', 'native_vs_torch', 'py_vs_torch') for k in keys}
    identical = True
    files = []
    for path in a.batches:
        out = os.path.join(a.out, os.path.basename(path).replace('.npz', '.native.npz'))
        p = subprocess.run([a.worker, 'eval-batches', '--model', a.onnx, '--batches', path, '--out', out, '--threads',
                            str(a.threads)], capture_output=True, text=True, encoding='utf-8')
        if p.returncode != 0:
            raise SystemExit(f'eval-batches failed: {p.stderr[-2000:]}')
        rows = Rows.load(path)
        bo = rows.batch_off
        py = {k: [] for k in keys}
        th = {k: [] for k in keys}
        for b in range(len(bo) - 1):
            sub = rows.take(np.arange(bo[b], bo[b + 1]))
            r = ort_py.answer_rows(sub)
            t = tev.evaluate(sub.dense(), sub.n_players, sub.opt_off, sub.opt_action, sub.extra)
            for d, x in ((py, r), (th, t)):
                for i, k in enumerate(keys):
                    if i < len(x):
                        d[k].append(np.asarray(x[i], np.float32).reshape(-1))
        with np.load(out) as z:
            nat = {k: z[k].reshape(-1) for k in keys}
        pyc = {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in py.items()}
        thc = {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in th.items()}
        frow = {'file': os.path.basename(path), 'batches': int(len(bo) - 1), 'rows': int(rows.B),
                'belief_rows': int((rows.extra == 2).sum())}
        for k in keys:
            if not (len(nat[k]) == len(pyc[k]) == len(thc[k])):
                raise SystemExit(f'{path} {k}: lengths native {len(nat[k])}, python {len(pyc[k])}, torch {len(thc[k])}')
            for name, x, y in (('native_vs_py', nat[k], pyc[k]), ('native_vs_torch', nat[k], thc[k]),
                               ('py_vs_torch', pyc[k], thc[k])):
                d = float(np.abs(x.astype(np.float64) - y).max()) if len(x) else 0.0
                worst[f'{name}_{k}'] = max(worst[f'{name}_{k}'], d)
                frow[f'{name}_{k}'] = d
            same = nat[k].tobytes() == pyc[k].tobytes()
            frow[f'bit_identical_{k}'] = same
            identical &= same
        files.append(frow)
        print(json.dumps(frow), flush=True)
    ok_py = all(v <= TOL_PY for k, v in worst.items() if k.startswith('native_vs_py'))
    ok_t = all(v <= TOL_TORCH for k, v in worst.items() if 'torch' in k)
    res = {'onnx': a.onnx, 'threads': a.threads, 'files': files, 'worst': worst, 'native_bit_identical_to_py': identical,
           'pass_vs_py_1e-6': ok_py, 'pass_vs_torch_2e-5': ok_t}
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({'worst': worst, 'native_bit_identical_to_py': identical, 'pass': ok_py and ok_t}))
    sys.exit(0 if ok_py and ok_t else 1)


if __name__ == '__main__':
    main()
