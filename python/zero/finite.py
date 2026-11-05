"""The non-finite guard (docs/az.md 4.1): a NaN or inf in a sample shard, a training step or a
checkpoint fails the run (an exception: non-zero exit, the supervisor's crash path) instead of
poisoning it. MTG Zero's run mtg4 (2026-10-27) took one non-finite training step at iteration
30, saved a checkpoint whose every float tensor was NaN and ran on for 10 iterations of
turn-capped games and drawn matches before anyone noticed.

Checks (all vectorised; a float16 array is checked on its bits, exponent all ones = inf / NaN):
* `write_shard` (zero/data.py): x, pi, z, q, sc after the float16 cast (a value above 65504
  becomes inf); a bad shard is written as `*.bad.npz` (the window never loads it) and the run
  fails naming the array, rows, columns (x) and the shard's meta.
  A sparse shard (run setting shard_sparse, docs/az.md 32) is checked on its features' values
  (`x_val`), and described by the rows and columns of the dense rows.
* `Window`: the same arrays of every shard it loads (shards written before the guard too).
* `Run.train` (zero/run.py): the loss terms and the gradient norm after `clip_grad_norm_`,
  before `optimizer.step()`; the failing batch is saved to `<run>/debug/`.
* `net.save` and the optimizer checkpoint: every float tensor.
"""
import numpy as np
import torch

F16_EXP = 0x7C00                  # float16 exponent bits: all ones = inf or NaN
CHUNK = 1 << 20                   # elements per vectorised chunk (bounded temporaries)
SHARD_FLOATS = ('x', 'pi', 'z', 'q', 'sc', 'aux')       # ('aux': only in shards with auxiliary targets)


class NonFinite(RuntimeError):
    """A NaN / inf where the run needs finite numbers. Uncaught: the run exits non-zero."""


def nonfinite_rows(a, chunk=CHUNK):
    """Indices along the first axis of float array `a` whose entries include a NaN or inf
    (empty when all are finite; non-float arrays are finite)."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind != 'f' or a.size == 0:
        return np.zeros(0, np.int64)
    a2 = a.reshape(len(a), -1)
    step = max(1, chunk // max(1, a2.shape[1]))
    out = []
    if a2.dtype == np.float16:
        u = a2.view(np.uint16)
        buf = np.empty((min(step, len(u)), u.shape[1]), np.uint16)
        for s in range(0, len(u), step):
            b = buf[:len(u[s:s + step])]
            np.bitwise_and(u[s:s + step], F16_EXP, out=b)
            if b.max() == F16_EXP:
                out.append(s + np.flatnonzero((b == F16_EXP).any(1)))
    else:
        for s in range(0, len(a2), step):
            fin = np.isfinite(a2[s:s + step])
            if not fin.all():
                out.append(s + np.flatnonzero(~fin.all(1)))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def head(ix, n=12):
    """'[3, 7, 9]' or '[3, 7, ... (+40 more)]' (indices or names)."""
    ix = list(ix)
    return '[' + ', '.join(map(str, ix[:n])) + (f', ... (+{len(ix) - n} more)' if len(ix) > n else '') + ']'




def describe(name, a, rows, opt_off=None):
    """One line about the non-finite entries of array `name` (`rows`: nonfinite_rows(a)). x: the
    columns too; pi (one entry per option): the sample rows through `opt_off`."""
    a2 = np.asarray(a).reshape(len(a), -1)
    cols = np.zeros(a2.shape[1], bool)
    n = np.zeros(3, np.int64)
    for s in range(0, len(rows), 1024):
        sub = a2[rows[s:s + 1024]].astype(np.float32)
        cols |= (~np.isfinite(sub)).any(0)
        n += [np.isnan(sub).sum(), np.isposinf(sub).sum(), np.isneginf(sub).sum()]
    kinds = ', '.join(f'{k} {name}' for k, name in zip(n.tolist(), ('NaN', '+inf', '-inf')) if k)
    if name == 'pi' and opt_off is not None:
        samples = np.unique(np.searchsorted(np.asarray(opt_off), rows, side='right') - 1)
        return f'pi: {len(rows)} option entr(ies) ({kinds}) in sample row(s) {head(samples)}'
    out = f'{name}: {len(rows)} row(s) {head(rows)}'
    if name == 'x' or a2.shape[1] > 1:
        out += f', column(s) {head(np.flatnonzero(cols))}'
    return out + f' ({kinds})'


def describe_sparse_x(c, ent):
    """`describe` of a sparse shard's features (`x_ptr` / `x_idx` / `x_val`, zero/data.py) whose value
    entries `ent` are non-finite: the dense rows' line (rows, columns, kinds)."""
    ptr = np.asarray(c['x_ptr'], np.int64)
    rows = np.unique(np.searchsorted(ptr, ent, side='right') - 1)
    cols = np.unique(np.asarray(c['x_idx'])[ent])
    v = np.asarray(c['x_val'])[ent].astype(np.float32)
    n = [int(np.isnan(v).sum()), int(np.isposinf(v).sum()), int(np.isneginf(v).sum())]
    kinds = ', '.join(f'{k} {name}' for k, name in zip(n, ('NaN', '+inf', '-inf')) if k)
    return f'x: {len(rows)} row(s) {head(rows)}, column(s) {head(cols)} ({kinds})'


def shard_problems(c):
    """The non-finite entries of a shard's float arrays (stored dtypes: zero/data.py) -> one
    line per bad array; [] = clean. A sparse shard's features (`x_val`, no `x`): its values (a NaN /
    inf is never a zero), described as the dense rows would be."""
    probs = []
    for k in SHARD_FLOATS:
        if k == 'x' and 'x' not in c and 'x_val' in c:
            ent = nonfinite_rows(c['x_val'])
            if len(ent):
                probs.append(describe_sparse_x(c, ent))
            continue
        if k not in c:
            continue
        rows = nonfinite_rows(c[k])
        if len(rows):
            probs.append(describe(k, c[k], rows, c.get('opt_off')))
    return probs


def _walk(obj, prefix=''):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f'{prefix}.{k}' if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f'{prefix}[{i}]')


def nonfinite_tensors(state):
    """(names of the float tensors holding a NaN / inf, number of float tensors) in a (nested)
    state dict. One sync per device (an optimizer state mixes devices: AdamW keeps `step` on
    the CPU, its moments on the GPU)."""
    ts = [(n, t) for n, t in _walk(state) if t.is_floating_point() and t.numel()]
    by_dev = {}
    for n, t in ts:
        by_dev.setdefault(t.device, []).append((n, torch.isfinite(t).all()))
    bad = []
    for flags in by_dev.values():
        ok = torch.stack([f for _, f in flags]).tolist()
        bad += [n for (n, _), g in zip(flags, ok) if not g]
    return bad, len(ts)


def check_state(state, what):
    """Raise NonFinite if a float tensor of `state` holds a NaN / inf (`what`: the file it was
    going to be saved as)."""
    bad, n = nonfinite_tensors(state)
    if bad:
        raise NonFinite(f'refusing to save {what}: {len(bad)} of {n} float tensors hold NaN / inf {head(bad, 8)}')
