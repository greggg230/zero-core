"""Parity gate 1 of zero-onnx/1 (docs/az.md 41): a network's exported ONNX graph, run by ONNX Runtime's
CPU execution provider, against torch on real network batches - new code is right only if it matches.

  python -m zero.onnx_parity <checkpoint> --batches <file.npz> [<file.npz> ...] [--packing mask,compact,packed,host]
      [--sizes 1,7,64,256,1024] [--tol 2e-5] [--onnx-dir DIR] [--reuse] [--game G] [--allow-mismatch]
      [--threads 4] [--json out.json]

Batch files: recorded driver batches (tools/onnx_record.py: the driver's sparse rows with their kinds:
leaf, value-only, root belief) or sample shards (`data.read_shard`; a shard row with no options is a
value-only row, and every 8th leaf row is followed by a root belief copy of it, as the driver places
them). Cases per file: each batch size of `--sizes` (consecutive rows, wrapping around), as they are
and with NaN in every absent slot's floats (the present flag aside: a token network never reads them);
a packing that takes the host's token positions ('host') also gets junk in its padding positions;
and the file's first recorded batches as the driver sent them. A case of more than 64 rows is also
answered in pieces of 64 rows grouped by their non-zero count (`run_chunked`, the worker's policy;
reported as '<packing>-c64').

Each case compares what the worker would answer from the graph (`answer`: the generic step - each
row's legal logits softmaxed per segment, value / margin as they are, `aux` of the belief rows) with
torch's answers on the same rows: `forward_static` on the host-packed rows (a token network; the
CUDA graphs' entry point, `infer.TokenGraphRunner`'s arithmetic on the CPU) and the eager evaluator
(`infer.NetEvaluator`): priors, value, margin and the belief rows' auxiliary predictions, max abs
difference within `--tol` (2e-5: tests/test_token_graphs.py's graphs-vs-eager tolerance). Also: value
exactly 0 beyond n_players, one auxiliary row per belief row, and the graph deterministic (each case run
twice: the outputs bit for bit the same - a kernel that races on duplicate indices is caught here). Prints one JSON line per case and a
summary; exit 1 when a case fails.

Needs onnx, onnxscript and onnxruntime (CPU build); nothing on a run's path imports this module.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .infer import NetEvaluator, segment_softmax_np, token_index, token_presence
from .protocol import MAX_SEATS, ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows

TOL = 2e-5
SIZES = (1, 7, 64, 256, 1024)
CHUNK = 64


class Rows:
    """Network batch rows in the driver's sparse form (protocol.py): xs_off i64 [B+1], xs_idx i32,
    xs_val f32, n_players i64 [B], opt_off i64 [B+1], opt_action i32, extra u8 [B]; `features` F;
    `batch_off` the recorded batches' first rows (None: not recorded batches)."""

    def __init__(self, xs_off, xs_idx, xs_val, n_players, opt_off, opt_action, extra, features, batch_off=None,
                 source=''):
        self.xs_off = np.asarray(xs_off, np.int64)
        self.xs_idx = np.asarray(xs_idx, np.int32)
        self.xs_val = np.asarray(xs_val, np.float32)
        self.n_players = np.asarray(n_players, np.int64)
        self.opt_off = np.asarray(opt_off, np.int64)
        self.opt_action = np.asarray(opt_action, np.int32)
        self.extra = np.asarray(extra, np.uint8)
        self.F = int(features)
        self.batch_off = batch_off
        self.source = source

    @property
    def B(self):
        return len(self.n_players)

    def take(self, idx):
        """The rows `idx` (an int array: any order, repeats allowed), offsets rebased."""
        idx = np.asarray(idx, np.int64)
        xa, xb = self.xs_off[idx], self.xs_off[idx + 1]
        oa, ob = self.opt_off[idx], self.opt_off[idx + 1]
        sel = lambda a, b: np.concatenate([np.arange(u, v) for u, v in zip(a, b)]) if len(a) else np.zeros(0, np.int64)
        xi, oi = sel(xa, xb).astype(np.int64), sel(oa, ob).astype(np.int64)
        return Rows(np.concatenate([[0], np.cumsum(xb - xa)]), self.xs_idx[xi], self.xs_val[xi], self.n_players[idx],
                    np.concatenate([[0], np.cumsum(ob - oa)]), self.opt_action[oi], self.extra[idx], self.F,
                    source=self.source)

    def dense(self):
        x = np.zeros((self.B, self.F), np.float32)
        x[np.repeat(np.arange(self.B), np.diff(self.xs_off)), self.xs_idx] = self.xs_val
        return x

    @classmethod
    def from_dense(cls, x, n_players, opt_off, opt_action, extra, source=''):
        nz = x != 0                                    # (NaN != 0: kept)
        r, c = np.nonzero(nz)
        off = np.concatenate([[0], np.cumsum(nz.sum(1))])
        return cls(off, c.astype(np.int32), x[r, c], n_players, opt_off, opt_action, extra, x.shape[1], source=source)

    @classmethod
    def load(cls, path, max_rows=4096, belief_every=8):
        """A recorded batch file (tools/onnx_record.py) or a sample shard (at most `max_rows` rows of it:
        no-option rows are value-only rows; each `belief_every`-th leaf row gets a root belief copy after it)."""
        with np.load(path) as z:
            files = set(z.files)
            if 'xs_off' in files:
                meta = json.loads(str(z['meta']))
                return cls(z['xs_off'], z['xs_idx'], z['xs_val'], z['n_players'], z['opt_off'], z['opt_action'],
                           z['extra'], meta['features'], batch_off=z['batch_off'], source=os.path.basename(path))
        from .data import read_shard
        s = read_shard(path, dense=False)
        if 'x' in s:
            x = s['x']
            N = min(len(x), max_rows)
            get = lambda a, b: np.asarray(x[a:b], np.float32)
        else:
            from .data import stored_rows
            xs = stored_rows(s)
            N = min(xs.N, max_rows)
            get = lambda a, b: np.asarray(xs.dense(np.arange(a, b)), np.float32)
        off, act, npl = s['opt_off'], s['opt_action'], s['n_players']
        parts, k = [], 0
        for a in range(0, N, 512):
            b = min(N, a + 512)
            xx = get(a, b)
            rows, kinds, oo, aa = [], [], [0], []
            for i in range(a, b):
                c = int(off[i + 1] - off[i])
                rows.append(i - a)
                kinds.append(ROW_LEAF if c else ROW_VALUE)
                aa.append(act[off[i]:off[i + 1]])
                oo.append(oo[-1] + c)
                if c:
                    k += 1
                    if k % belief_every == 0:
                        rows.append(i - a)
                        kinds.append(ROW_BELIEF)
                        oo.append(oo[-1])
            parts.append(cls.from_dense(xx[rows], np.asarray(npl, np.int64)[a + np.asarray(rows)], np.asarray(oo),
                                        np.concatenate(aa).astype(np.int32), np.asarray(kinds, np.uint8)))
        return cat(parts, source=os.path.basename(path))


def cat(parts, source=''):
    """Rows of several Rows objects, in order."""
    xo = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(p.xs_off) for p in parts]))])
    oo = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(p.opt_off) for p in parts]))])
    return Rows(xo, np.concatenate([p.xs_idx for p in parts]), np.concatenate([p.xs_val for p in parts]),
                np.concatenate([p.n_players for p in parts]), oo, np.concatenate([p.opt_action for p in parts]),
                np.concatenate([p.extra for p in parts]), parts[0].F, source=source)


def absent_nan(model, rows):
    """`rows` with NaN in every absent slot's floats but its present flag (a token network's segments with
    a flag) - a token network must never read them."""
    x = rows.dense()
    for s in model.segs:
        if s['present'] is None:
            continue
        o, k, w = s['offset'], s['count'], s['width']
        v = x[:, o:o + k * w].reshape(-1, k, w)
        absent = v[..., s['present']] <= 0.5
        keep = v[..., s['present']].copy()
        v[absent] = np.nan
        v[..., s['present']] = keep
    return Rows.from_dense(x, rows.n_players, rows.opt_off, rows.opt_action, rows.extra, rows.source)


# ---------------------------------------------------------------- answers
def answer(outs, rows):
    """The worker's generic step on the graph's outputs (policy, value, margin, aux) -> what an evaluator
    returns: (priors [options] - each row's legal logits softmaxed per segment, float32 -, value [B, 8],
    margin [B, 8]) plus the belief rows' auxiliary answers when the batch has belief rows."""
    policy, value, margin, aux = outs
    r = np.repeat(np.arange(rows.B), np.diff(rows.opt_off))
    pri = segment_softmax_np(policy[r, rows.opt_action], rows.opt_off)
    br = belief_rows(rows.extra)
    out = (pri, value, margin)
    return out + (aux[br],) if len(br) else out


def feeds(rows, packing=None, pidx=None, junk=False, seed=0):
    """ONNX Runtime's inputs for `rows` (zero-onnx/1; 'host' also tok_full / tok_n from the host's packing,
    `junk`: padding positions hold random slots)."""
    f = {'x_ptr': rows.xs_off, 'x_idx': rows.xs_idx, 'x_val': rows.xs_val, 'n_players': rows.n_players,
         'belief': rows.extra == ROW_BELIEF}
    if packing == 'host':
        P = presence_sparse(rows, pidx)
        n = P.sum(1)
        T = max(2, int(n.max()))
        full = token_index(P, np.zeros((rows.B, T), np.int64))
        if junk:
            pad = np.arange(T)[None, :] >= n[:, None]
            full[pad] = np.random.default_rng(seed).integers(0, P.shape[1], size=int(pad.sum()))
        f['tok_full'] = full
        f['tok_n'] = n.astype(np.int64)
    return f


def run_chunked(sess, rows, chunk=64, packing=None, pidx=None):
    """The graph's outputs for `rows` evaluated in pieces of at most `chunk` rows, the rows grouped by their
    non-zero count (a worker-side policy that needs nothing of the network: a row's outputs depend only on
    it; rows of similar size pad less in a token network's attention, as the eager evaluator's chunks of
    64 rows sorted by token count) -> (policy, value, margin, aux) in the rows' order."""
    if rows.B <= chunk:
        return sess.run(None, feeds(rows, packing, pidx))
    order = np.argsort(np.diff(rows.xs_off), kind='stable')
    outs = None
    for a in range(0, rows.B, chunk):
        idx = order[a:a + chunk]
        o = sess.run(None, feeds(rows.take(idx), packing, pidx))
        if outs is None:
            outs = [np.empty((rows.B,) + v.shape[1:], v.dtype) for v in o]
        for dst, v in zip(outs, o):
            dst[idx] = v
    return outs


def presence_sparse(rows, pidx):
    """`infer.TokenGraphRunner.presence_sparse` from the sparse rows: [B, 1 + slots] bool."""
    pidx = np.asarray(pidx, np.int64)
    F_ = rows.F
    flag_col = np.full(F_, -1, np.int64)
    has = pidx >= 0
    flag_col[pidx[has]] = 1 + np.flatnonzero(has)
    P = np.zeros((rows.B, 1 + len(pidx)), bool)
    P[:, 0] = True
    P[:, 1 + np.flatnonzero(~has)] = True
    r = np.repeat(np.arange(rows.B), np.diff(rows.xs_off))
    col = flag_col[rows.xs_idx.astype(np.int64)]
    hit = (col >= 0) & (rows.xs_val > 0.5)
    P[r[hit], col[hit]] = True
    return P


@torch.inference_mode()
def static_answer(model, rows):
    """torch `TokenNet.forward_static` on the host-packed rows (`infer.token_presence` / `token_index`, padding
    0) with `infer.TokenGraphRunner`'s post-processing, on the CPU -> an evaluator's answer."""
    from .net import aux_answer
    x = rows.dense()
    P = token_presence(x, model.present_index())
    n = P.sum(1)
    full = token_index(P, np.zeros((rows.B, int(n.max())), np.int64))
    layout = list(getattr(model, 'aux_layout', None) or [])
    r = model.forward_static(torch.from_numpy(x), torch.from_numpy(rows.n_players), torch.from_numpy(full),
                             torch.from_numpy(n.astype(np.int64)), aux=bool(layout))
    S = model.seats
    value = np.zeros((rows.B, MAX_SEATS), np.float32)
    margin = np.zeros((rows.B, MAX_SEATS), np.float32)
    value[:, :S] = torch.softmax(r[1].float(), 1).numpy()
    margin[:, :S] = r[2].float().numpy()
    value[np.arange(MAX_SEATS)[None, :] >= rows.n_players[:, None]] = 0.0
    pol = r[0].float().numpy()
    aux = None
    if layout:
        pred = aux_answer(model, r[3])
        aux = torch.cat([pred[t[0]].float() for t in layout], 1).numpy()
    return answer((pol, value, margin, aux), rows)


def eager_answer(ev, rows):
    """The eager evaluator's answer (`NetEvaluator.evaluate` on the densified rows)."""
    return ev.evaluate(rows.dense(), rows.n_players, rows.opt_off.astype(np.int32), rows.opt_action, rows.extra)


def diffs(a, b):
    """Max abs difference per item (priors, value, margin[, aux]); NaN anywhere -> inf."""
    if len(a) != len(b):
        raise AssertionError(f'answers of {len(a)} and {len(b)} items')
    out = []
    for u, v in zip(a, b):
        if u.shape != v.shape:
            raise AssertionError(f'shapes {u.shape} vs {v.shape}')
        d = float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if u.size else 0.0
        out.append(d if np.isfinite(d) else float('inf'))
    return out


ITEMS = ('priors', 'value', 'margin', 'aux')


def session(path, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def check_invariants(got, rows):
    """value exactly 0 beyond n_players; one auxiliary row per belief row; priors sum to 1 per leaf."""
    value = got[1]
    if np.any(value[np.arange(MAX_SEATS)[None, :] >= rows.n_players[:, None]] != 0):
        return 'value not 0 beyond n_players'
    nb = len(belief_rows(rows.extra))
    if nb and (len(got) != 4 or got[3].shape[0] != nb):
        return 'auxiliary rows != belief rows'
    cnt = np.diff(rows.opt_off)
    if len(got[0]) and np.abs(np.add.reduceat(got[0], rows.opt_off[:-1][cnt > 0]) - 1).max() > 1e-5:
        return 'priors do not sum to 1'
    return None


def cases(model, rows, sizes, tokens, recorded=3):
    """(name, rows, junk) of the gate on `rows` (module docstring)."""
    R = rows.B
    out = []
    for s in sizes:
        start = (s * 7919) % R
        idx = (start + np.arange(s)) % R
        sub = rows.take(idx)
        out.append((f'B{s}', sub, False))
        if tokens:
            out.append((f'B{s}-nan', absent_nan(model, sub), True))
    if rows.batch_off is not None:
        bo = rows.batch_off
        for k in range(min(recorded, len(bo) - 1)):
            out.append((f'batch{k}', rows.take(np.arange(bo[k], bo[k + 1])), False))
    return out


def run_gate(model, onnx_paths, rows_list, sizes=SIZES, tol=TOL, threads=4, log=print):
    """The gate (module docstring): `onnx_paths` {packing: path} of `model`'s exports, `rows_list` Rows
    objects -> (passed, [result dicts])."""
    from .net import TokenNet
    tokens = isinstance(model, TokenNet)
    torch.set_num_threads(threads)
    ev = NetEvaluator(model, 'cpu')
    sess = {p: session(path, threads) for p, path in onnx_paths.items()}
    pidx = model.present_index() if tokens else None
    results, ok = [], True
    for rows in rows_list:
        if rows.F != model.features:
            raise SystemExit(f'{rows.source}: rows of {rows.F} features for a network of {model.features}')
        for name, sub, junk in cases(model, rows, sizes, tokens):
            eager = eager_answer(ev, sub)
            static = static_answer(model, sub) if tokens else None
            base = diffs(static, eager) if tokens else None
            runs = [(p, s, None) for p, s in sess.items()]
            if sub.B > CHUNK:
                runs += [(f'{p}-c{CHUNK}', s, p) for p, s in sess.items() if p != 'host']
            for p, s, chunked in runs:
                t0 = time.perf_counter()
                if chunked:                                       # (the worker's chunks: `run_chunked`)
                    outs = run_chunked(s, sub, CHUNK)
                    secs = time.perf_counter() - t0
                    again = run_chunked(s, sub, CHUNK)
                else:
                    f = feeds(sub, p, pidx, junk=junk and p == 'host')
                    outs = s.run(None, f)
                    secs = time.perf_counter() - t0
                    again = s.run(None, f)
                got = answer(outs, sub)
                bad = check_invariants(got, sub)
                if bad is None and not all(np.array_equal(u, v, equal_nan=True) for u, v in zip(outs, again)):
                    bad = 'not deterministic: the same inputs twice gave other outputs'
                d_e = diffs(got, eager)
                d_s = diffs(got, static) if tokens else None
                worst = max(d_e + (d_s or []))
                fail = bad is not None or worst > tol
                ok &= not fail
                r = {'source': rows.source, 'case': name, 'packing': p, 'rows': sub.B,
                     'leaf': int((sub.extra == ROW_LEAF).sum()), 'value_rows': int((sub.extra == ROW_VALUE).sum()),
                     'belief_rows': int((sub.extra == ROW_BELIEF).sum()),
                     'ort_vs_eager': dict(zip(ITEMS, d_e)), 'ort_vs_static': dict(zip(ITEMS, d_s)) if tokens else None,
                     'static_vs_eager': dict(zip(ITEMS, base)) if tokens else None, 'max': worst,
                     'ort_ms': round(secs * 1e3, 2), 'pass': not fail}
                if bad:
                    r['invariant'] = bad
                results.append(r)
                log(json.dumps(r))
    return ok, results


def summarize(results):
    """Per packing: cases, failures, the worst ORT-vs-torch difference per item."""
    out = {}
    for r in results:
        s = out.setdefault(r['packing'], {'cases': 0, 'failed': 0, 'worst': {k: 0.0 for k in ITEMS}})
        s['cases'] += 1
        s['failed'] += 0 if r['pass'] else 1
        for d in (r['ort_vs_eager'], r['ort_vs_static'] or {}):
            for k, v in d.items():
                s['worst'][k] = max(s['worst'][k], v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='zero-onnx/1 parity gate: ONNX Runtime (CPU EP) vs torch (docs/az.md 41)')
    ap.add_argument('checkpoint')
    ap.add_argument('--batches', nargs='+', required=True)
    ap.add_argument('--packing', default='mask,compact,packed,host')
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--onnx-dir', default=None, help='where the exports go (default: next to the checkpoint)')
    ap.add_argument('--reuse', action='store_true', help='reuse an export whose metadata names this checkpoint\'s sha256')
    ap.add_argument('--game', default=None)
    ap.add_argument('--allow-mismatch', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-rows', type=int, default=4096, help='rows read from a shard')
    ap.add_argument('--json', default=None, help='write the results here')
    a = ap.parse_args(argv)
    from . import onnx_export as OX
    model, info = OX.load_checkpoint(a.checkpoint, a.game, a.allow_mismatch)
    d = a.onnx_dir or os.path.dirname(os.path.abspath(a.checkpoint))
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    from .net import TokenNet
    packs = [p for p in a.packing.split(',') if p] if isinstance(model, TokenNet) else ['none']
    paths, export_s = {}, {}
    for p in packs:
        path = os.path.join(d, f'{stem}.{p}.onnx')
        if not (a.reuse and os.path.exists(path) and read_metadata(path).get('checkpoint_sha256') == info['checkpoint_sha256']):
            t0 = time.perf_counter()
            OX.export(model, path, p if p != 'none' else 'mask', info)
            export_s[p] = round(time.perf_counter() - t0, 1)
        paths[p] = path
    rows_list = [Rows.load(b, a.max_rows) for b in a.batches]
    sizes = [int(s) for s in a.sizes.split(',') if s]
    ok, results = run_gate(model, paths, rows_list, sizes, a.tol, a.threads)
    summ = {'checkpoint': a.checkpoint, 'sha256': info['checkpoint_sha256'], 'tol': a.tol, 'passed': ok,
            'packings': summarize(results), 'export_seconds': export_s,
            'onnx_bytes': {p: os.path.getsize(v) for p, v in paths.items()}}
    print(json.dumps(summ))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summ, 'results': results}, f, indent=1)
    print(f'ONNX PARITY {"PASSED" if ok else "FAILED"} (tol {a.tol:g})', file=sys.stderr)
    return 0 if ok else 1


def read_metadata(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


if __name__ == '__main__':
    sys.exit(main())
