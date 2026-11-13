"""Record a short self-play's real network batches (the driver's sparse LeafBatches, leaf, value-only
and root belief rows) for the zero-onnx/1 parity gate and benchmark (python -m zero.onnx_parity,
tools/onnx_bench.py; docs/az.md 41).

  python tools/onnx_record.py --root <game checkout> --package mtgz --checkpoint <ckpt> --game mtg_eoe \
      --out batches.npz [--spec '<json>' --batches 60 --conc 96 --threads 4 --sims 16 --sims-fast 8 \
      --p-full 0.25 --belief-tilt 1 --view-samples 0.5 --seed 1 --every 1]

Imports only from `--root` (its pinned core: the recorder needs nothing newer than the driver's
sparse batches), CPU only: the checkpoint is loaded as the run loads it (`net.load`: the encoding
guard against the game build, the auxiliary head widened to the game's targets) and answers every
batch (eager `NetEvaluator`); every `--every`-th batch is kept (later batches hold later game states)
and the self-play is stopped after `--batches` kept batches. The seats search
with the root's belief (ROW_BELIEF rows) and root-view leaf values with view samples (ROW_VALUE rows)
so every row kind is in the file.

The file (npz): the rows of every batch in order - xs_off i64 [R+1], xs_idx i32, xs_val f32,
n_players i64 [R], player i32 [R], gid i64 [R], extra u8 [R], opt_off i64 [R+1], opt_action i32 -,
batch_off i64 [batches+1] (each batch's first row), and meta (JSON: game, features, actions,
checkpoint, its sha256, the settings, the engine module). A game with variable-length rows (docs/search.md 2.15)
also: t_off i64 [R+1], t_seg u16, t_slot u32, t_val f32 (each token's floats consecutive, as the driver sends them),
ov_off i64 [R+1], ov_opt i32, ov_tok i64, and meta `t_segs` [[segment, name, width, head slots], ...],
`option_pointers`, the counts; `--min-tail-rows N` / `--min-overrides N` record on (past `--batches`) until the
kept batches hold that many rows with a tail token / overrides, at most `--max-batches` kept batches (then it fails).
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time


VARLEN = ('t_off', 't_seg', 't_slot', 't_val', 'ov_off', 'ov_opt', 'ov_tok')


class Enough(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--game', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--spec', default=None, help='the game spec (default: the package\'s for the game)')
    ap.add_argument('--batches', type=int, default=60)
    ap.add_argument('--conc', type=int, default=96)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--torch-threads', type=int, default=4)
    ap.add_argument('--sims', type=int, default=16)
    ap.add_argument('--sims-fast', type=int, default=8)
    ap.add_argument('--p-full', type=float, default=0.25)
    ap.add_argument('--belief-tilt', type=float, default=1.0)
    ap.add_argument('--view-samples', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--every', type=int, default=1, help='keep every k-th batch')
    ap.add_argument('--min-tail-rows', type=int, default=0, help='record until the kept rows hold this many with a tail token')
    ap.add_argument('--min-overrides', type=int, default=0, help='record until the kept rows hold this many pointer overrides')
    ap.add_argument('--max-batches', type=int, default=0, help='fail past this many kept batches (default 20 x --batches)')
    a = ap.parse_args(argv)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    sys.path.insert(0, os.path.abspath(a.root))
    importlib.import_module(a.package)
    for m in (f'{a.package}.az', ):
        try:
            importlib.import_module(m)                    # (a game package registers itself in <pkg>.az)
        except ModuleNotFoundError:
            pass
    import numpy as np
    import torch
    from zero import games as G, package
    from zero.infer import NetEvaluator, drive
    from zero.net import load
    torch.set_num_threads(a.torch_threads)
    model, extra = load(a.checkpoint, 'cpu', game=a.game)
    ev = NetEvaluator(model, 'cpu')
    spec = a.spec if a.spec is not None else package.active().specs.get(a.game, '{}')
    seat = G.nn_seat('m', a.sims, selfplay=True, sims_fast=a.sims_fast, p_full=a.p_full, leaf_value_view='root',
                     view_samples=a.view_samples, belief_tilt=a.belief_tilt)
    drv = G.Driver(a.game, [[seat, seat]] * (a.conc * 4), spec=spec, concurrency=a.conc, seed=a.seed, record=True,
                   threads=a.threads, groups=1, sparse=True)
    keep, seen = [], [0]

    class Recorder:
        def __call__(self, lb):
            if not lb.sparse:
                raise RuntimeError('the driver sent a dense batch (sparse=True asked)')
            seen[0] += 1
            if seen[0] % a.every:
                return ev(lb)
            keep.append({k: np.array(lb.a[k]) for k in ('xs_off', 'xs_idx', 'xs_val', 'n_players', 'player', 'gid',
                                                         'extra', 'opt_off', 'opt_action') + VARLEN if k in lb.a})
            keep[-1]['F'] = int(lb.a['n_features'])
            return ev(lb)

    def counts():
        tr = sum(int((np.diff(b['t_off']) > 0).sum()) for b in keep if 't_off' in b)
        no = sum(int(b['ov_off'][-1]) for b in keep if 'ov_off' in b)
        return tr, no

    cap = a.max_batches or 20 * a.batches

    def check():
        if len(keep) >= a.batches:
            tr, no = counts()
            if tr >= a.min_tail_rows and no >= a.min_overrides:
                raise Enough()
            if len(keep) >= cap:
                raise SystemExit(f'{len(keep)} batches kept with {tr} tail rows and {no} overrides (asked '
                                 f'{a.min_tail_rows} / {a.min_overrides}): giving up')

    t0 = time.time()
    os.environ['ZERO_INFLIGHT'] = '1'
    try:
        drive(drv, {'m': Recorder()}, check=check)
    except Enough:
        pass
    secs = time.time() - t0
    tr, no = counts()
    if tr < a.min_tail_rows or no < a.min_overrides:
        raise SystemExit(f'the self-play ended with {tr} tail rows and {no} overrides (asked {a.min_tail_rows} / '
                         f'{a.min_overrides})')
    F = {b['F'] for b in keep}
    if F != {model.features}:
        raise SystemExit(f'batch widths {F} != the network\'s {model.features}')
    rows = [len(b['n_players']) for b in keep]
    nnz = [int(b['xs_off'][-1]) for b in keep]
    nopt = [int(b['opt_off'][-1]) for b in keep]
    xs_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['xs_off']) for b in keep]))]).astype(np.int64)
    opt_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['opt_off']) for b in keep]))]).astype(np.int64)
    out = {'xs_off': xs_off, 'xs_idx': np.concatenate([b['xs_idx'] for b in keep]).astype(np.int32),
           'xs_val': np.concatenate([b['xs_val'] for b in keep]).astype(np.float32),
           'n_players': np.concatenate([b['n_players'] for b in keep]).astype(np.int64),
           'player': np.concatenate([b['player'] for b in keep]).astype(np.int32),
           'gid': np.concatenate([b['gid'] for b in keep]).astype(np.int64),
           'extra': np.concatenate([b['extra'] for b in keep]).astype(np.uint8),
           'opt_off': opt_off, 'opt_action': np.concatenate([b['opt_action'] for b in keep]).astype(np.int32),
           'batch_off': np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)}
    assert out['xs_off'][-1] == len(out['xs_idx']) == sum(nnz) and out['opt_off'][-1] == sum(nopt)
    info = G.info(a.game)
    vmeta = {}
    if 't_off' in keep[0]:
        out['t_off'] = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['t_off']) for b in keep]))]).astype(np.int64)
        out['t_seg'] = np.concatenate([b['t_seg'] for b in keep]).astype(np.uint16)
        out['t_slot'] = np.concatenate([b['t_slot'] for b in keep]).astype(np.uint32)
        out['t_val'] = np.concatenate([b['t_val'] for b in keep]).astype(np.float32)
        vmeta['t_segs'] = [[int(x['seg']), str(x['name']), int(x['width']), int(x['head'])] for x in info['tails']]
        vmeta['tail_rows'], vmeta['tail_tokens'] = tr, int(out['t_off'][-1])
    if 'ov_off' in keep[0]:
        out['ov_off'] = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['ov_off']) for b in keep]))]).astype(np.int64)
        out['ov_opt'] = np.concatenate([b['ov_opt'] for b in keep]).astype(np.int32)
        out['ov_tok'] = np.concatenate([b['ov_tok'] for b in keep]).astype(np.int64)
        vmeta['option_pointers'], vmeta['overrides'] = True, no
    h = hashlib.sha256()
    with open(a.checkpoint, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    ex = out['extra']
    meta = {'game': a.game, 'features': model.features, 'actions': model.config['actions'],
            'checkpoint': os.path.abspath(a.checkpoint), 'checkpoint_sha256': h.hexdigest(),
            'aux_layout': [list(t) for t in (getattr(model, 'aux_layout', None) or [])],
            'engine': os.path.abspath(G.engine().__file__), 'spec': spec, 'seat': seat,
            'settings': {k: v for k, v in vars(a).items() if k not in ('root', 'out')}, 'seconds': round(secs, 1),
            'batches': len(keep), 'batches_seen': seen[0], 'rows': int(sum(rows)), 'rows_leaf': int((ex == 0).sum()),
            'rows_value': int((ex == 1).sum()), 'rows_belief': int((ex == 2).sum()), 'nnz': int(sum(nnz)), **vmeta}
    np.savez(a.out, meta=json.dumps(meta), **out)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('spec', 'seat')}))


if __name__ == '__main__':
    main()
