"""zero-farm's provider probe graph (zero-farm PROTOCOL.md 8: `probe-providers` runs "a small built-in graph and
self-test per candidate provider and adapter"): a small token network built as `zero.onnx_export` exports any run's
TokenNet - the `packed` packing (NonZero / TopK compaction, GatherElements / ScatterElements writing unique indices,
the policy's per-row sort by TopK, CumSum, identity digits decoded by Round / Cast, boolean-mask attention, exact GELU,
LayerNorm; opset 17, docs/az.md 41.2), a reader and an auxiliary head
(softmax and exp targets) - so a provider that runs it runs every operator a real network's graph uses; plus its
reference set (`write_reference`: torch's eager answers on the CPU). The worker embeds both files
(`crates/worker/probe/`); `--check GRAPH` lists the operators of a real graph the probe graph lacks.

    PYTHONPATH=python python tools/onnx_probe_graph.py crates/worker/probe [--check eoe1.onnx ...]

Deterministic: the same files from the same torch / onnxscript (the seeds are fixed). Rows are synthetic (random
features with valid presence flags and identity digits, 1-5 legal options, value-only and belief rows), in
batches of 1, 7, 64 and 120 rows: the worker's 64-row pieces and a two-piece batch.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from zero import onnx_export as OX  # noqa: E402
from zero.net import TokenNet  # noqa: E402
from zero.onnx_parity import Rows  # noqa: E402
from zero.protocol import ROW_BELIEF, ROW_LEAF, ROW_VALUE  # noqa: E402

# tests/test_net_data.py's TOK_LAYOUT with identity actions (MTG Zero's id_actions), more unit slots
LAYOUT = {'vocab': 12,
          'segments': [{'name': 'ctx', 'offset': 5, 'count': 1, 'width': 14, 'present': None, 'id': {'at': 1, 'radix': [12]}},
                       {'name': 'units', 'offset': 19 + 24, 'count': 12, 'width': 9, 'present': 0,
                        'id': {'at': 1, 'radix': [3, 4]}, 'actions': [[3, 1], [20, 2]], 'id_actions': 44},
                       {'name': 'cards', 'offset': 19 + 24 + 108, 'count': 3, 'width': 4, 'present': 3, 'actions': [[58, 1]]}],
          'bags': [{'name': 'zones', 'offset': 19, 'count': 2, 'width': 12}]}
F, A = 19 + 24 + 108 + 12 + 2, 64
AUX = [['belief', 6, 'multinomial'], ['rate', 3, 'poisson']]
READERS = [{'target': 'belief', 'width': 6, 'loss': 'multinomial', 'inputs': [[0, 5], [19, 6]], 'hidden': 16,
            'enter': 'trunk', 'belief': True}]
BATCHES = [1, 7, 64, 120]


def features(B, g):
    """Random features with valid presence flags and identity digits."""
    x = torch.randn(B, F, generator=g)
    x[:, 6:18] = 0
    x[torch.arange(B), 6 + torch.randint(12, (B,), generator=g)] = 1.0
    x[:, 19:43] = torch.randint(0, 3, (B, 24), generator=g).float() * 0.25
    # each row's units have distinct identities (a permutation of the 12), as a real game's objects do: two present
    # units with one identity would score the same identity action twice, a scatter to one slot with two values,
    # which ONNX leaves undefined (the CPU keeps the last write, GPU providers any; zero-farm AMD/CUDA work)
    ids = torch.stack([torch.randperm(12, generator=g) for _ in range(B)])
    for s in range(12):
        o = 43 + 9 * s
        x[:, o] = (torch.rand(B, generator=g) < 0.5).float()
        x[:, o + 1:o + 8] = 0
        x[torch.arange(B), o + 1 + ids[:, s] // 4] = 1.0
        x[torch.arange(B), o + 4 + ids[:, s] % 4] = 1.0
    for s in range(3):
        o = 151 + 4 * s
        x[:, o + 3] = (torch.rand(B, generator=g) < 0.7).float()
    x[x.abs() < 0.3] = 0      # sparse, as driver rows are
    return x


def net():
    torch.manual_seed(20261112)
    m = TokenNet(F, A, LAYOUT, d=32, layers=2, heads=4, ff=64, head=32, game='probe', aux=AUX, readers=READERS).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def rows(seed=5):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    B = sum(BATCHES)
    x = features(B, g).numpy()
    kinds = rng.choice([ROW_LEAF, ROW_VALUE, ROW_BELIEF], B, p=[0.6, 0.25, 0.15]).astype(np.uint8)
    kinds[0] = ROW_LEAF
    npl = rng.integers(2, 5, B)
    cnt = np.where(kinds == ROW_LEAF, rng.integers(1, 6, B), 0)
    off = np.concatenate([[0], np.cumsum(cnt)])
    act = np.concatenate([np.sort(rng.choice(A, c, replace=False)) for c in cnt] + [np.zeros(0, np.int64)]).astype(np.int32)
    r = Rows.from_dense(np.ascontiguousarray(x, np.float32), npl, off, act, kinds, source='probe-synthetic')
    r.batch_off = np.concatenate([[0], np.cumsum(BATCHES)]).astype(np.int64)
    return r


def ops(path):
    import onnx
    return collections.Counter(n.op_type for n in onnx.load(path, load_external_data=False).graph.node)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out', help='directory for probe.onnx and probe_reference.npz')
    ap.add_argument('--check', nargs='*', default=[], help='real graphs: the operators the probe graph lacks')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = net()
    info = {'checkpoint': 'probe', 'checkpoint_sha256': '0' * 64, 'checkpoint_iter': '0'}
    g = os.path.join(a.out, 'probe.onnx')
    OX.export(m, g, 'packed', info)
    r = rows()
    meta = OX.write_reference(m, [r], os.path.join(a.out, 'probe_reference.npz'), sum(BATCHES), info)
    chk = OX.check_reference(g, os.path.join(a.out, 'probe_reference.npz'), threads=1)
    mine = ops(g)
    out = {'graph': g, 'bytes': os.path.getsize(g), 'reference': meta, 'onnxruntime_cpu': chk, 'ops': dict(sorted(mine.items()))}
    for c in a.check:
        out.setdefault('missing', {})[os.path.basename(c)] = sorted(set(ops(c)) - set(mine))
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
