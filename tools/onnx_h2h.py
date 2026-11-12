"""Parity gate 2 (c), behaviour (zero-farm phase 0b, docs/az.md 42): a head-to-head of a checkpoint's
zero-onnx/1 graph through onnxruntime (`zero.onnx_eval.OnnxEvaluator`, side A) against the checkpoint itself
through torch (`infer.NetEvaluator`, side B) - the same network, so A's score should be 0.5 within its se.
zero.h2h's match (paired games, the evaluation seats) with side A's evaluator swapped; CPU only.

    python tools/onnx_h2h.py --root <game checkout> --package mtgz.az --onnx <graph> --checkpoint <ckpt> \
        [--games 200 --sims 32 --spec '<eval spec>' --select cq --cq-c-scale 0.3 --chance-k 0 --threads 8 \
         --onnx-threads 1 --max-steps 20000 --out result.json]
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--sims', type=int, default=32)
    ap.add_argument('--spec', default=None)
    ap.add_argument('--select', default='visits')
    ap.add_argument('--cq-c-visit', type=float, default=50.0)
    ap.add_argument('--cq-c-scale', type=float, default=1.0)
    ap.add_argument('--chance-k', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--groups', type=int, default=2)
    ap.add_argument('--onnx-threads', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    import torch
    torch.set_num_threads(1)
    from zero import h2h
    from zero.infer import NetEvaluator
    from zero.onnx_eval import OnnxEvaluator

    def evaluators(ma, mb, device, graphs, max_rows=None):
        ev = OnnxEvaluator(a.onnx, threads=a.onnx_threads, expect={'game': ma.config.get('game'), 'features': ma.features})
        return {'a': ev, 'b': NetEvaluator(mb, 'cpu')}

    h2h._evaluators = evaluators
    seat = {'select': a.select}
    if a.select == 'cq':
        seat.update(select_min_visits=0, cq_c_visit=a.cq_c_visit, cq_c_scale=a.cq_c_scale)
    out = h2h.match(a.checkpoint, a.checkpoint, games=a.games, sims=a.sims, chance_k=a.chance_k, seed=a.seed,
                    threads=a.threads, groups=a.groups, device='cpu', spec=a.spec, seat_a=dict(seat), seat_b=dict(seat),
                    max_steps=a.max_steps, paired=True, sparse=True)
    out['a'] = f'onnx {a.onnx} (onnxruntime, {a.onnx_threads} threads)'
    out['b'] = f'torch {a.checkpoint} (NetEvaluator, cpu)'
    print(json.dumps(out))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
