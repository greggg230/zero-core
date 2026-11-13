"""Training data (docs/az.md 3): sample shards and the RAM-bounded training window.

Shards: `data/iter_NNNN/<task>_SSS.npz` (uncompressed npz, written to a temporary name and
renamed: a crash never leaves a half-written shard; remote self-play's shards, `m<KK>-<sid>_SSS.npz`,
are compressed npz - zlib level 1, ~58x for MTG's sparse features - which `np.load` reads alike:
docs/az.md 5.3). Arrays: `x` float16 [N, F], `player`,
`n_players` int8, `opt_off` int32 [N+1], `opt_action` int16/int32, `pi` float16, `z` / `q` /
`sc` float16 [N, P] (P = most players in the shard), `net` int32, optionally `aux` float16 [N, W]
(the deciders' auxiliary targets: self-play with an auxiliary loss on, docs/az.md 15), and `meta` (a
JSON string: game, features, actions, rows, task info). Float16 targets are renormalised on load.
Non-finite guard (zero/finite.py, docs/az.md 4.1): `write_shard` checks x, pi, z, q, sc after the
float16 cast; a shard with a NaN / inf is written as `*.bad.npz` instead (the window never loads
it: `shard_paths` skips it) and the run fails (`NonFinite`). The `Window` checks every shard it
loads the same way, and remembers where each row came from (`provenance`).

`TargetStats`: sharpness of the recorded policy targets (self-play metrics).

Window: the newest iterations' shards loaded into RAM (`Window(paths, ...)`), every shard
subsampled at the same rate when the total exceeds the RAM budget - the rate from the shards'
row counts (their meta), each shard subsampled as it is loaded (remote self-play triples a run's
samples per iteration; the rows kept are those of loading every shard first); `batch(idx)` builds
torch tensors: features cut / zero-padded to the network's width, dense legal masks and policy
targets over the actions, value / margin targets over MAX_SEATS; a window with `aux_width` W > 0 also
`aux` [B, W] and `aux_mask` [B] (1 = the row's shard has auxiliary targets; a shard written without
them - before the targets were on, e.g. a fork's older window - gives zeros with mask 0, so the
auxiliary losses skip its rows and the shards stay usable). The shards load in parallel
(`threads`, default `WINDOW_THREADS`): inflating a compressed shard, its CRC and the copies
release the GIL, so the window of MTG's Delver run (40 shards, 265k rows, 17.9 GB) loads in a
fraction of the 41 s it took one shard at a time. Each shard's rows are drawn in path order
before any loading (the random stream of the sequential load) and its features written into
their place in the one preallocated `x`: the window is the same, array for array, and the peak
is the window plus the shards being loaded (it was twice the window: the per-shard copies, then
their concatenation).

WindowCache (run setting `window_cache`): the shards a run's windows keep between iterations, each
shard's full arrays as loaded, so a window loads only the shards it has not seen (an iteration's
window shares all but its newest iteration's shards with the last one). The window then points at
the cached shards' features (no copy: `batch` gathers its rows shard by shard) and is the window
of loading them all, array for array. Only while the window fits its budget (no subsample: the
window is exactly its shards); a window over budget empties the cache and loads as without it.
The cached shards stay in RAM between trainings (up to the budget), where a window was freed.

Sparse window (run setting `window_sparse`, `Window(sparse=True)`): each row's features kept as their
non-zeros (`SparseRows`: column indexes and float16 values; non-zero = the float16 bits, so a -0.0
is kept) - the same window, rows and batches bit for bit, in a fraction of the RAM for a sparse
encoding (MTG's rows are ~0.7% non-zero: the Delver line's 34 GB window in 1.8 GB). A shard's
features are converted on threads in chunks, read in place from the file when the shard is stored
uncompressed (np.memmap: np.load copies a member through a Python loop that holds the GIL); the
non-finite guard of `x` reads its non-zeros. With a WindowCache every shard is cached whole, also
when the window is over budget: the window is subsampled from the cache (the picks of loading)
and its rows' features read from their cached shards (`src_row`), so an iteration loads only its
new shards (the Delver line: 54 s -> 6.6 s). The budget still counts dense rows.

`batches` (run setting `train_prefetch`): a training's batches in order; with prefetch, built that
many steps ahead on a background thread (the window's gather, the float32 features and the dense
targets: ~60 ms a batch for MTG's 33,652 features) while the training step runs on the device. The
rows are drawn from the same generator in the same order, so the batches are the same.

Sparse shards (run setting `shard_sparse`, docs/az.md 32): a shard's features stored as their non-zeros
instead of `x` - `x_ptr` int64 [N+1], `x_idx` uint16 [nnz] (int32 past 65,536 columns), `x_val` float16
[nnz]: row r's non-zeros are x_idx / x_val[x_ptr[r]:x_ptr[r + 1]], columns ascending, non-zero = the
float16 bits (a -0.0 is kept), so the dense rows come back bit for bit - and in its meta `x_format`
'sparse', `x_version` 1 (`X_VERSION`: a newer one is refused), `x_cols` (the dense width) and `x_nnz`.
Every other array is the dense shard's. The writer is asked for it by its meta (`shard_format(cfg)`:
{'x_format': 'sparse'}; without it a shard is written as before, byte for byte); a ShardWriter so asked
keeps each added sample dict's features as their non-zeros (never the dense float16 copy of a whole
shard). Every reader takes both formats, and a window mixes them (a generation written before the
setting beside one written with it): `read_shard` densifies a sparse shard's features (all its rows;
`dense=False`: as stored), the window reads a sparse shard's rows straight into its dense rows or its
SparseRows (the same window either way), `shard_features` reads some rows of either, and the
non-finite guard reads a sparse shard's values (zero/finite.py).

Shard format v2 (variable-length rows, docs/az.md 43, docs/search.md 2.15): a game with tail segments or pointer
overrides writes sparse shards only (`x_version` 2; a dense shard with tails is refused: an older reader would read
`x` and ignore the tail), with version 1's members plus `t_ptr` int64 [N+1], `t_seg` uint16, `t_slot` uint32, `t_val`
float16 (each token's floats, dense), `ov_ptr` int64 [N+1], `ov_opt` int32, `ov_tok` int64, and in its meta `t_segs`
([[seg, name, width, head], ...]), `option_pointers`, `max_row_tokens`. Version 1 readers refuse it (their
`x_version` check); this build reads both (a v1 shard has empty tails) and refuses a newer one (R7). `read_shard`
returns the v2 members with the features; the features alone (`shard_features`) are refused for a shard with any
tail token or override unless `head_only` (R10). A window for a network with tail segments (`Window(layout=...)`)
loads them as `Window.tail` / `Window.ov` and refuses shards of other tail segments or without them (v1, R9); a
window for a network without them refuses a shard with any tail token or override (R11: never a silent drop).
"""
import glob
import json
import os
import queue
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .finite import NonFinite, shard_problems
from .protocol import has_rows
from . import rows as R
from .protocol import MAX_SEATS, concat_samples

EXT = '.npz'
BAD = '.bad.npz'                  # a quarantined shard (non-finite values): never loaded
# Shards loaded at once by a Window (env ZERO_WINDOW_THREADS overrides; 1 = one at a time).
WINDOW_THREADS = int(os.environ.get('ZERO_WINDOW_THREADS', '0')) or min(8, os.cpu_count() or 1)
X_SPARSE = 'sparse'               # meta `x_format` of a sparse shard (module docstring)
X_VERSION = 1                     # the sparse format's version of a shard without variable-length rows
X_VERSION_ROWS = 2                # ... of a shard with them (a game with tail segments / pointer overrides)
X_VERSION_MAX = 2                 # the newest version this build reads (R7: a newer one is refused)
X_SPARSE_KEYS = ('x_ptr', 'x_idx', 'x_val')
ROW_KEYS = ('t_ptr', 't_seg', 't_slot', 't_val', 'ov_ptr', 'ov_opt', 'ov_tok')    # format v2's members


def shard_format(cfg, info=None):
    """The meta keys that ask `write_shard` / `ShardWriter` for a run's shard format (run setting
    `shard_sparse`): {'x_format': 'sparse'}, or {} (dense: the meta, and the shard, of before). `info` (the game's
    `games.info`) of a game with tail segments or pointer overrides: format v2's meta (`rows_meta`) too, and dense
    shards are refused (R8)."""
    sparse = int(cfg.get('shard_sparse', 0) or 0)
    rm = rows_meta(info) if info is not None else {}
    if rm and not sparse:
        raise ValueError(f'game {info.get("name")!r} has variable-length rows (tail segments / pointer overrides): its '
                         f'shards are sparse only (shard_sparse 1; docs/az.md 43)')
    return dict({'x_format': X_SPARSE} if sparse else {}, **rm)


def rows_meta(info):
    """Format v2's meta of a game (`games.info`) with tail segments or pointer overrides: {'t_segs', 'option_pointers',
    'max_row_tokens'}; {} for a game without (the meta of before)."""
    if not info or not (info.get('tails') or info.get('option_pointers')):
        return {}
    return {'t_segs': R.info_segs(info), 'option_pointers': bool(info.get('option_pointers')),
            'max_row_tokens': info.get('max_row_tokens')}


def _rows_shard(meta):
    """Whether a writer's meta asks for format v2 (variable-length rows)."""
    return bool((meta or {}).get('t_segs')) or bool((meta or {}).get('option_pointers'))


def shard_widths(meta):
    """A v2 shard's tail segments {seg: (width, head, k)} from its meta `t_segs` (None for a v1 shard)."""
    t = (meta or {}).get('t_segs')
    if t is None:
        return None
    return {int(s): (int(w), int(h), k) for k, (s, _, w, h) in enumerate(t)}


def _wants_sparse(meta):
    """Whether a writer's meta asks for sparse features ('x_format': 'sparse' | 'dense' | absent)."""
    fmt = (meta or {}).get('x_format', 'dense')
    if fmt not in ('dense', X_SPARSE):
        raise ValueError(f"shard meta x_format {fmt!r}: 'dense' or {X_SPARSE!r}")
    return fmt == X_SPARSE


def _sparse_stored(files, meta, path):
    """Whether a shard (its member names, its meta) stores its features sparse; a sparse format of a
    newer version (or an incomplete one) is refused."""
    if 'x_val' not in files:
        return False
    v = int(meta.get('x_version', 1))
    if meta.get('x_format', X_SPARSE) != X_SPARSE or v > X_VERSION_MAX:
        raise ValueError(f'shard {path}: features stored as {meta.get("x_format")!r} version {v}; this build '
                         f'reads {X_SPARSE!r} up to version {X_VERSION_MAX} (a newer zero-core wrote it)')
    if any(k not in files for k in X_SPARSE_KEYS) or 'x_cols' not in meta:
        raise ValueError(f'shard {path}: a sparse shard without {X_SPARSE_KEYS} / meta x_cols')
    if v >= X_VERSION_ROWS and (any(k not in files for k in ROW_KEYS) or 't_segs' not in meta):
        raise ValueError(f'shard {path}: a version {v} shard without {ROW_KEYS} / meta t_segs')
    return True


def _shard_has_rows(p):
    """Whether a shard's arrays (`read_shard(dense=False)`) hold any tail token or override."""
    return any(k in p and len(p[k]) and int(p[k][-1]) > 0 for k in ('t_ptr', 'ov_ptr'))


def stored_rows(p, F=None):
    """A sparse shard's features (`read_shard(..., dense=False)` arrays and meta) as SparseRows of
    width F (None: the stored `x_cols`; narrower: the columns >= F dropped; wider: nothing to add),
    column indexes in SparseRows' dtype for F (`SparseRows.from_dense`'s)."""
    cols = int(p['meta']['x_cols'])
    F = cols if F is None else int(F)
    ptr, idx, val = p['x_ptr'].astype(np.int64, copy=False), p['x_idx'], p['x_val']
    if F < cols and len(idx) and int(idx.max()) >= F:
        keep = idx < F
        N = len(ptr) - 1
        cnt = np.bincount(np.repeat(np.arange(N), np.diff(ptr))[keep], minlength=N)
        ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt, dtype=np.int64)])
        idx, val = idx[keep], val[keep]
    itype = np.uint16 if F <= 1 << 16 else np.int32
    return SparseRows(ptr, idx.astype(itype, copy=False), val.astype(np.float16, copy=False), F)


def _narrow_int(a):
    lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
    for dt in (np.int8, np.int16, np.int32):
        if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return a.astype(dt)
    return a.astype(np.int64)


def compact(s):
    """Sample arrays (driver format) -> the stored dtypes."""
    return {'x': s['x'].astype(np.float16), **_compact_rest(s)}


def _compact_rest(s):
    """`compact` but the features (in the stored order of the arrays)."""
    n = s['n_players']
    P = int(n.max()) if len(n) else 1
    return {'player': s['player'].astype(np.int8),
            'n_players': n.astype(np.int8), 'opt_off': s['opt_off'].astype(np.int32),
            'opt_action': _narrow_int(s['opt_action']), 'pi': s['pi'].astype(np.float16),
            'z': s['z'][:, :P].astype(np.float16), 'q': s['q'][:, :P].astype(np.float16),
            'sc': s['sc'][:, :P].astype(np.float16), 'net': s['net'].astype(np.int32),
            **({'aux': s['aux'].astype(np.float16)} if 'aux' in s else {})}


def compact_rows(s, meta):
    """Format v2's members of the samples `s` (driver format; meta `t_segs` / `option_pointers` say which the game
    has: a tail game's samples carry `t_off` etc., an overrides game's `ov_off` etc. - refused when missing; the other
    kind empty): `t_val` as float16."""
    N = len(s['player'])
    if meta.get('t_segs'):
        if 't_off' not in s:
            raise ValueError('a shard of a game with tail segments: samples without their tails (t_off)')
        t = {'t_ptr': np.asarray(s['t_off'], np.int64), 't_seg': np.asarray(s['t_seg'], np.uint16),
             't_slot': np.asarray(s['t_slot'], np.uint32), 't_val': np.asarray(s['t_val']).astype(np.float16)}
    else:
        if 't_off' in s:
            raise ValueError('samples with tails for a shard whose meta declares no tail segments (t_segs)')
        t = {'t_ptr': np.zeros(N + 1, np.int64), 't_seg': np.zeros(0, np.uint16), 't_slot': np.zeros(0, np.uint32),
             't_val': np.zeros(0, np.float16)}
    if meta.get('option_pointers'):
        if 'ov_off' not in s:
            raise ValueError('a shard of a game with pointer overrides: samples without them (ov_off)')
        o = {'ov_ptr': np.asarray(s['ov_off'], np.int64), 'ov_opt': np.asarray(s['ov_opt'], np.int32),
             'ov_tok': np.asarray(s['ov_tok'], np.int64)}
    else:
        if 'ov_off' in s:
            raise ValueError('samples with pointer overrides for a shard whose meta declares none (option_pointers)')
        o = {'ov_ptr': np.zeros(N + 1, np.int64), 'ov_opt': np.zeros(0, np.int32), 'ov_tok': np.zeros(0, np.int64)}
    if len(t['t_ptr']) != N + 1 or len(o['ov_ptr']) != N + 1:
        raise ValueError(f'samples of {N} rows with tail / override offsets of {len(t["t_ptr"])} / {len(o["ov_ptr"])}')
    return {**t, **o}


def compact_sparse(s):
    """`compact` of a sparse shard: the features as `x_ptr` / `x_idx` / `x_val` (from the sample dict's
    SparseRows 'xs' when a ShardWriter made them, else from its dense 'x': the non-zeros of the float16
    cast, bit for bit) -> (arrays, the dense width)."""
    xs = s['xs'] if 'xs' in s else SparseRows.from_dense(s['x'], s['x'].shape[1])
    return {'x_ptr': xs.indptr, 'x_idx': xs.idx, 'x_val': xs.val, **_compact_rest(s)}, xs.F


def write_shard(path, s, meta, bad_path=None, compress=0):
    """Write the samples `s` to `path` (atomic) -> (rows, bytes). Non-finite values after the
    float16 cast: the shard goes to `bad_path` (default: `path` as *.bad.npz) and NonFinite is
    raised naming the arrays, rows, columns and the meta. `compress`: 0 = a plain npz (np.savez),
    1-9 = a compressed npz at that zlib level (remote self-play: 1). A meta with 'x_format': 'sparse'
    (`shard_format`): the features stored as their non-zeros (module docstring)."""
    if _rows_shard(meta):
        # format v2 (variable-length rows): sparse only (R8)
        if not _wants_sparse(meta):
            raise ValueError(f'shard {path}: tails / pointer overrides need a sparse shard (shard_sparse 1; R8)')
        c, cols = compact_sparse(s)
        c.update(compact_rows(s, meta))
        meta = dict(meta, rows=int(len(c['player'])), x_version=X_VERSION_ROWS, x_cols=int(cols), x_nnz=int(len(c['x_val'])))
    elif has_rows(s):
        raise ValueError(f'shard {path}: samples with tails / pointer overrides for a meta without format v2\'s '
                         f'(t_segs / option_pointers: `shard_format(cfg, info)`); they would be dropped')
    elif _wants_sparse(meta):
        c, cols = compact_sparse(s)
        meta = dict(meta, rows=int(len(c['player'])), x_version=X_VERSION, x_cols=int(cols), x_nnz=int(len(c['x_val'])))
    else:
        c = compact(s)
        meta = dict(meta, rows=int(len(c['player'])))
    probs = shard_problems(c)
    if probs:
        bad = bad_path or path[:-len(EXT)] + BAD
        os.makedirs(os.path.dirname(os.path.abspath(bad)), exist_ok=True)
        _save_npz(bad, c, dict(meta, nonfinite=probs, shard=path), compress)
        raise NonFinite(f'non-finite sample values in shard {os.path.basename(path)} (meta {json.dumps(meta)}): '
                        + '; '.join(probs) + f'. The samples were kept in {bad} (never loaded by the window); '
                        f'the run stops here')
    _save_npz(path, c, meta, compress)
    return meta['rows'], os.path.getsize(path)


def savez_level(path, level, arrays):
    """np.savez_compressed at zlib level `level` (numpy's own uses zlib's default, 6: several times
    slower on MTG's shards for a few percent of size). Readable by np.load."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=int(level)) as zf:
        for k, a in arrays.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)


def _save_npz(path, c, meta, compress=0):
    tmp = path + f'.{os.getpid()}.tmp.npz'
    if compress:
        savez_level(tmp, compress, dict(meta=np.array(json.dumps(meta)), **c))
    else:
        np.savez(tmp, meta=np.array(json.dumps(meta)), **c)
    for k in range(40):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if k == 39:
                raise
            import time
            time.sleep(0.25)


def read_shard(path, dense=True, head_only=False):
    """A shard's arrays and its meta (a dict). A sparse shard's features (module docstring) come
    densified as `x` float16 [N, x_cols], the dense shard's bit for bit (all its rows: the window
    reads a sparse shard's rows without that, `shard_features` some of them); `dense` False: as
    stored (`x_ptr` / `x_idx` / `x_val`; a dense shard's `x` as ever). A v2 shard's tail and override members
    (`t_ptr`, ..., `ov_tok`) come with the features either way; `head_only` drops them (the caller's explicit
    choice to read the heads alone, R10)."""
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != 'meta'}
        out['meta'] = json.loads(str(z['meta']))
    if _sparse_stored(out, out['meta'], path) and dense:
        xs = stored_rows(out)
        out['x'] = xs.dense(np.arange(xs.N))
        for k in X_SPARSE_KEYS:
            del out[k]
    if head_only:
        for k in ROW_KEYS:
            out.pop(k, None)
    return out


def shard_tail(p, widths=None):
    """A shard's (`read_shard`) tails and overrides as (`rows.Tail` or None, `rows.Overrides` or None): None for a
    v1 shard or a kind its meta does not declare. `widths` (the network's tail segments): refused (ValueError, R9)
    when they differ from the shard's `t_segs`."""
    meta = p['meta']
    sw = shard_widths(meta)
    if sw is None:
        return None, None
    if widths is not None and {s: v[:2] for s, v in sw.items()} != {int(s): tuple(v[:2]) for s, v in widths.items()}:
        raise ValueError(f'shard tail segments {meta.get("t_segs")} differ from the network\'s {widths} (R9)')
    tail = R.Tail(p['t_ptr'], p['t_seg'], p['t_slot'], p['t_val'], sw) if sw else None
    ov = R.Overrides(p['ov_ptr'], p['ov_opt'], p['ov_tok']) if meta.get('option_pointers') else None
    return tail, ov


def shard_features(path, rows=None, F=None, head_only=False):
    """Rows `rows` (None = all; an int array or a slice) of a shard's features as float16 [n, F] (F None
    = the stored width; else cut / zero-padded), from either format: a dense shard's read in place
    when it is stored uncompressed (only those rows), a sparse shard's scattered from its non-zeros. The features
    alone: refused (ValueError, R10) for a v2 shard with any tail token or override unless `head_only`."""
    with np.load(path) as z:
        meta = json.loads(str(z['meta']))
        sparse = _sparse_stored(z.files, meta, path)
        if sparse and not head_only and int(meta.get('x_version', 1)) >= X_VERSION_ROWS \
                and _shard_has_rows({k: z[k] for k in ('t_ptr', 'ov_ptr')}):
            raise ValueError(f'shard {path}: its rows have tail tokens / pointer overrides; the features alone would '
                             f'drop them (read_shard, or head_only=True; R10)')
        if sparse:
            p = {k: z[k] for k in X_SPARSE_KEYS}
        else:
            x = _stored_member(path, 'x')
            if x is None:
                x = z['x']
    if sparse:
        p['meta'] = meta
        xs = stored_rows(p, F)
        N, cols = xs.N, xs.F
    else:
        N, cols = x.shape
        F = cols if F is None else int(F)
    sel = np.arange(N)[rows] if rows is not None else np.arange(N)
    if sparse:
        return xs.dense(sel)
    out = np.zeros((len(sel), F), np.float16)
    w = min(F, cols)
    out[:, :w] = x[sel, :w]
    return out


def _stored_member(path, name):
    """Member `name` of an npz as a read-only np.memmap when it is stored uncompressed (a local
    shard), else None (compressed, absent, Fortran order, objects): its bytes in place in the file,
    read as they are used (np.load copies the whole member through a Python loop)."""
    import struct
    with zipfile.ZipFile(path) as zf:
        try:
            zi = zf.getinfo(name + '.npy')
        except KeyError:
            return None
        if zi.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, 'rb') as f:
        f.seek(zi.header_offset)
        h = f.read(30)
        if len(h) < 30 or h[:4] != b'PK\x03\x04':
            return None
        n, m = struct.unpack('<HH', h[26:30])         # the local header's name and extra lengths
        f.seek(zi.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        offset = f.tell()
    if fortran or dtype.hasobject or not shape or 0 in shape:
        return None
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def _read_shard_sparse(path, F, idx, threads):
    """read_shard for a sparse window: every array but `x`, whose rows `idx` (None = all) come as
    SparseRows 'xs' - from the file in place when stored uncompressed. The non-finite guard of x
    reads its non-zeros (a NaN / inf is never zero bits); a bad x is described from the dense rows.
    A sparse shard: its rows' non-zeros as stored (cut to F; `x_ptr` / `x_idx` / `x_val` stay in the
    result for the guard, which reads all of them)."""
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k not in ('meta', 'x')}
        out['meta'] = json.loads(str(z['meta']))
        if _sparse_stored(z.files, out['meta'], path):
            xs = stored_rows(out, F)
            out['xs'] = xs if idx is None else xs.take(idx)
            return out
        x = _stored_member(path, 'x')
        if x is None:
            x = z['x']
    if x.dtype != np.float16:                      # (shards store float16; the dense window's cast)
        x = np.asarray(x).astype(np.float16)
    try:
        xs = SparseRows.from_dense(x, F, idx, threads=threads)
        if not np.isfinite(xs.val).all():
            out['x'] = np.asarray(x)               # (shard_problems names the rows and columns)
    finally:
        del x
    out['xs'] = xs
    return out


def convert_shard(src, dst, sparse=True, compress=None, threads=1):
    """Write shard `src` again as `dst` with its features sparse (`sparse`) or dense (module docstring)
    -> (rows, bytes): the same rows, every other array as stored (an old generation shrunk for good, or
    a sparse one made readable by an older checkout). A dense uncompressed `src` is read in place in
    chunks (never its dense rows in RAM at once; on `threads`). `compress`: zlib level of `dst` (None:
    `src`'s - 1 if it is compressed, else 0). A shard with non-finite values is refused (NonFinite).
    `dst` must be another file (write it, then replace `src` with it)."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError(f'convert_shard: {dst} is the source')
    with zipfile.ZipFile(src) as zf:
        deflated = any(zi.compress_type != zipfile.ZIP_STORED for zi in zf.infolist())
    compress = (1 if deflated else 0) if compress is None else int(compress)
    with np.load(src) as z:
        meta0 = json.loads(str(z['meta']))
        was = _sparse_stored(z.files, meta0, src)
        rest = {k: z[k] for k in z.files if k not in ('meta', 'x') + X_SPARSE_KEYS}
        if was:
            xs = stored_rows(dict({k: z[k] for k in X_SPARSE_KEYS}, meta=meta0))
        else:
            x = _stored_member(src, 'x')
            if x is None:
                x = z['x']
    meta = {k: v for k, v in meta0.items() if k not in ('x_format', 'x_version', 'x_cols', 'x_nnz')}
    v2 = was and int(meta0.get('x_version', 1)) >= X_VERSION_ROWS
    if v2 and not sparse:
        raise ValueError(f'convert_shard: {src} is a version 2 shard (tails / pointer overrides): never dense (R8)')
    if sparse:
        if not was:
            xs = SparseRows.from_dense(x, x.shape[1], threads=threads)
        c = {'x_ptr': xs.indptr, 'x_idx': xs.idx, 'x_val': xs.val, **rest}
        meta.update(x_format=X_SPARSE, x_version=X_VERSION_ROWS if v2 else X_VERSION, x_cols=int(xs.F),
                    x_nnz=int(len(xs.val)))
    else:
        c = {'x': xs.dense(np.arange(xs.N)) if was else x, **rest}
    probs = shard_problems(c)
    if probs:
        raise NonFinite(f'shard {src}: non-finite values: ' + '; '.join(probs) + ' (not converted)')
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or '.', exist_ok=True)
    _save_npz(dst, c, meta, compress)
    return int(len(rest['player'])), os.path.getsize(dst)


def shard_paths(d):
    """The loadable shards of a data directory (not *.bad.npz, not *.tmp.npz)."""
    return sorted(p for p in glob.glob(os.path.join(d, '*' + EXT)) if not p.endswith((BAD, '.tmp' + EXT)))


def shard_rows(path):
    """A shard's sample count from its meta (only that member is read; a shard without `rows` in
    its meta: its arrays are counted)."""
    with np.load(path) as z:
        rows = json.loads(str(z['meta'])).get('rows')
        return int(rows) if rows is not None else int(len(z['player']))


class ShardWriter:
    """Collects driver sample dicts and writes a shard every `rows` samples (and at close).
    `bad_dir`: where a shard with non-finite values is quarantined, as
    `<data dir name>_<prefix>_SSS.bad.npz` (default: next to the good ones, `*.bad.npz`).
    `compress`: zlib level of the shards (0 = plain npz). `on_write(path, rows, bytes)`: called
    after each shard is written (remote self-play announces its shards so). A `meta` with
    'x_format': 'sparse' (`shard_format`, run setting shard_sparse): sparse shards, each added sample
    dict's features kept as their non-zeros as it is added (the dense float16 rows of a shard are
    never made)."""

    def __init__(self, d, prefix, rows=65536, meta=None, bad_dir=None, compress=0, on_write=None):
        os.makedirs(d, exist_ok=True)
        self.d, self.prefix, self.rows, self.meta, self.bad_dir = d, prefix, rows, dict(meta or {}), bad_dir
        self.compress, self.on_write = compress, on_write
        self.sparse = _wants_sparse(self.meta)
        self.buf = []
        self.nbuf = 0
        self.k = 0
        self.samples = 0
        self.bytes = 0
        self.shards = 0

    def add(self, s):
        if s is None or not len(s['player']):
            return
        if has_rows(s) and not (self.sparse and _rows_shard(self.meta)):
            raise ValueError('ShardWriter: samples with tails / pointer overrides need a sparse format v2 writer '
                             '(meta `shard_format(cfg, info)` with shard_sparse 1; R8)')
        if 'xs_val' in s and not self.sparse:
            raise ValueError('ShardWriter: samples given as their non-zeros (SampleRecords.arrays(sparse=True)) for a '
                             'dense shard writer')
        if self.sparse:
            if 'xs_val' in s:
                # (a SampleRecords' arrays as their non-zeros: no dense rows made, docs/az.md 44.4)
                pk = ('xs_off', 'xs_idx', 'xs_val', 'n_features')
                xs = SparseRows.from_pairs(*(s[k] for k in pk))
                s = {k: v for k, v in s.items() if k not in pk}
            else:
                x = s['x']
                s = {k: v for k, v in s.items() if k != 'x'}
                xs = SparseRows.from_dense(x, x.shape[1])
            s['xs'] = xs
        self.buf.append(s)
        self.nbuf += len(s['player'])
        if self.nbuf >= self.rows:
            self.flush()

    def flush(self):
        xs = [b.pop('xs') for b in self.buf] if self.sparse else None
        s = concat_samples(self.buf)
        self.buf, self.nbuf = [], 0
        if s is None:
            return
        if xs is not None:
            if len({p.F for p in xs}) != 1:
                raise ValueError(f'ShardWriter: sample dicts of different feature widths {sorted({p.F for p in xs})}')
            s = dict(s, xs=SparseRows.concat(xs, xs[0].F))
        name = f'{self.prefix}_{self.k:03d}'
        bad = None
        if self.bad_dir:
            bad = os.path.join(self.bad_dir, f'{os.path.basename(os.path.normpath(self.d))}_{name}{BAD}')
        path = os.path.join(self.d, name + EXT)
        kw = {'compress': self.compress} if self.compress else {}     # local shards: the call as it was
        rows, nbytes = write_shard(path, s, self.meta, bad, **kw)
        self.k += 1
        self.samples += rows
        self.bytes += nbytes
        self.shards += 1
        if self.on_write is not None:
            self.on_write(path, rows, nbytes)

    def close(self):
        self.flush()
        return {'samples': self.samples, 'bytes': self.bytes, 'shards': self.shards}


def target_sharpness(s):
    """Per sample of a driver sample dict: (options, normalised entropy H(pi) / ln(options),
    max share) of its policy target `pi`. Samples with one option are skipped (not recorded
    by the drivers anyway)."""
    off = np.asarray(s['opt_off'], np.int64)
    pi = np.asarray(s['pi'], np.float64)
    n = np.diff(off)
    keep = n >= 2
    if not keep.any() or not len(pi):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    seg = np.repeat(np.arange(len(n)), n)
    plogp = np.where(pi > 0, pi * np.log(np.where(pi > 0, pi, 1.0)), 0.0)
    h = -np.bincount(seg, plogp, minlength=len(n))
    mx = np.zeros(len(n))
    np.maximum.at(mx, seg, pi)
    nn = n[keep].astype(np.float64)
    return nn, h[keep] / np.log(nn), mx[keep]


class TargetStats:
    """Running means of the policy targets' sharpness over driver sample dicts."""

    def __init__(self):
        self.n = 0
        self.opts = 0.0
        self.h = 0.0
        self.mx = 0.0

    def add(self, s):
        if s is None or not len(s['player']):
            return
        nn, h, mx = target_sharpness(s)
        self.n += len(nn)
        self.opts += float(nn.sum())
        self.h += float(h.sum())
        self.mx += float(mx.sum())

    def summary(self, kind='visits'):
        k = max(1, self.n)
        return {'kind': kind, 'samples': self.n, 'options': round(self.opts / k, 3), 'h_norm': round(self.h / k, 4),
                'max_share': round(self.mx / k, 4)}


def _ranges(starts, lens):
    """Concatenated aranges [starts[i], starts[i] + lens[i])."""
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    return np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)


class SparseRows:
    """Feature rows kept as their non-zeros (a sparse window, module docstring): row r's columns are
    `idx[indptr[r]:indptr[r + 1]]` (ascending) and its values `val[...]` (float16). Non-zero means
    the float16 bits: a -0.0 is kept, so `dense` gives back the dense rows bit for bit."""

    CHUNK = 1024                                   # rows a conversion step (a bool mask of CHUNK x F)

    def __init__(self, indptr, idx, val, F):
        self.indptr, self.idx, self.val, self.F = indptr, idx, val, int(F)

    @property
    def N(self):
        return len(self.indptr) - 1

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.idx.nbytes + self.val.nbytes

    @classmethod
    def from_dense(cls, x, F, rows=None, threads=1):
        """The rows `rows` (None = all) of float16 `x` [N, w], their columns cut to F (a narrower `x` is
        zero-padded: nothing to keep). Chunks of CHUNK rows on up to `threads` threads (numpy releases
        the GIL). Another float `x` (a driver's float32 samples, ShardWriter): the rows of its float16
        cast, bit for bit - a float32 `x` without that cast (its non-zero bits, then the values cast:
        a value whose cast is +0.0 is dropped, a -0.0 kept)."""
        w = min(F, x.shape[1])
        n = len(x) if rows is None else len(rows)
        itype = np.uint16 if F <= 1 << 16 else np.int32

        def one(a):
            b = min(n, a + cls.CHUNK)
            xa = np.ascontiguousarray(x[a:b, :w] if rows is None else x[rows[a:b], :w])
            if xa.dtype == np.float32:
                f = np.flatnonzero(xa.view(np.uint32) != 0)
                v = xa.reshape(-1)[f].astype(np.float16)
                k = v.view(np.uint16) != 0
                f, v = f[k], v[k]
            else:
                if xa.dtype != np.float16:
                    xa = xa.astype(np.float16)
                f = np.flatnonzero(xa.view(np.uint16) != 0)
                v = xa.reshape(-1)[f]
            r = f // w
            return np.bincount(r, minlength=b - a), (f - r * w).astype(itype), v
        starts = range(0, n, cls.CHUNK)
        if threads > 1 and n > cls.CHUNK:
            with ThreadPoolExecutor(min(threads, len(starts))) as ex:
                got = list(ex.map(one, starts))
        else:
            got = [one(a) for a in starts]
        cnt = np.concatenate([g[0] for g in got]) if got else np.zeros(0, np.int64)
        indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt, dtype=np.int64)])
        idx = np.concatenate([g[1] for g in got]) if got else np.zeros(0, itype)
        val = np.concatenate([g[2] for g in got]) if got else np.zeros(0, np.float16)
        return cls(indptr, idx, val, F)

    @classmethod
    def from_pairs(cls, off, idx, val, F):
        """Rows given as their float32 non-zeros (`SampleRecords.arrays(sparse=True)`: `xs_off` [N + 1], `xs_idx`
        ascending in each row, `xs_val`) -> exactly `from_dense` of their dense float32 rows of width F, without
        those rows: the values cast to float16, a value whose cast is +0.0 dropped (a -0.0 kept), the columns in
        from_dense's index dtype (docs/az.md 44.4). A column >= F is refused."""
        F = int(F)
        off = np.asarray(off, np.int64)
        idx = np.asarray(idx)
        N = len(off) - 1
        if N < 0 or (N and (off[0] != 0 or off[-1] != len(idx))) or len(idx) != len(val):
            raise ValueError(f'SparseRows.from_pairs: offsets {off[:1]}..{off[-1:]} for {len(idx)} columns and '
                             f'{len(val)} values')
        if len(idx) and int(idx.max()) >= F:
            raise ValueError(f'SparseRows.from_pairs: column {int(idx.max())} of rows of {F} features')
        itype = np.uint16 if F <= 1 << 16 else np.int32
        v = np.asarray(val, np.float32).astype(np.float16)
        keep = v.view(np.uint16) != 0
        if keep.all():
            return cls(off.copy(), idx.astype(itype), v, F)
        cnt = np.bincount(np.repeat(np.arange(N), np.diff(off))[keep], minlength=N)
        indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt, dtype=np.int64)])
        return cls(indptr, idx[keep].astype(itype), v[keep], F)

    def _pos(self, rows):
        rows = np.asarray(rows, np.int64)
        s = self.indptr[rows]
        lens = self.indptr[rows + 1] - s
        return rows, lens, _ranges(s, lens)

    def take(self, rows):
        """The rows `rows`, in that order."""
        rows, lens, pos = self._pos(rows)
        indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens, dtype=np.int64)])
        return SparseRows(indptr, self.idx[pos], self.val[pos], self.F)

    @staticmethod
    def concat(parts, F):
        """The rows of `parts` one after the other."""
        nnz = np.cumsum([0] + [len(p.idx) for p in parts])
        indptr = np.concatenate([np.zeros(1, np.int64)] + [p.indptr[1:] + o for p, o in zip(parts, nnz[:-1])])
        itype = parts[0].idx.dtype if parts else np.uint16
        return SparseRows(indptr, np.concatenate([p.idx for p in parts]) if parts else np.zeros(0, itype),
                          np.concatenate([p.val for p in parts]) if parts else np.zeros(0, np.float16), F)

    def dense(self, rows, dtype=np.float16):
        """The dense rows `rows` [len(rows), F] in `dtype` (float16: the dense rows bit for bit; float32:
        their exact float32 values)."""
        out = np.zeros((len(rows), self.F), dtype)
        self.scatter(out, np.arange(len(rows)), rows)
        return out

    def scatter(self, out, at, rows):
        """Writes rows `rows`' non-zeros into the zeroed rows `at` of `out`."""
        rows, lens, pos = self._pos(rows)
        out[np.repeat(np.asarray(at, np.int64), lens), self.idx[pos]] = self.val[pos]


def _file_key(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _load_part(path, N, idx, F, xdst=None, AW=0, sparse=False, xthreads=1, older=None, rows=None):
    """One shard's part of a window: rows `idx` (None = all N), features cut / zero-padded to F
    written into `xdst` (a fresh array when None: then also returned as 'x'); `sparse`: instead the
    features as SparseRows 'xs' (converted on `xthreads` threads); `AW` > 0: the auxiliary
    targets too ('aux' float16 [rows, AW] and 'aux_mask': zeros and False for a shard without them);
    `older` (`auxtarget.older_columns`, docs/az.md 37): a shard of an older layout's width has its
    targets put in their places (the others zero), and 'aux_tmask' [rows, T] says which targets a row
    has.
    A sparse shard (`shard_sparse`): its rows' non-zeros scattered into `xdst` (the dense shard's rows
    bit for bit), or kept as they are stored for `sparse`.
    `rows` (a window for a network with tail segments / pointer overrides: {'widths': its tail segments, 'pointers'}):
    the shard's tails and overrides of the rows too ('tail' `rows.Tail` float16, 'ov' `rows.Overrides`), refused
    for a shard of other tail segments or without them (R9); None: a shard with any tail token or override is
    refused (R11)."""
    p = _read_shard_sparse(path, F, idx, xthreads) if sparse else read_shard(path, dense=False)
    probs = shard_problems(p)
    if probs:
        raise NonFinite(f'non-finite sample values in shard {path} (meta {json.dumps(p["meta"])}): '
                        + '; '.join(probs) + '. Move it out of the data directory (or rename it to '
                        '*.bad.npz) to train without it')
    if len(p['player']) != N:
        raise ValueError(f'shard {path}: {len(p["player"])} rows, its meta says {N}')
    out = {}
    if rows is not None:
        try:
            tail, ov = shard_tail(p, rows['widths'] or None)
        except ValueError as e:
            raise ValueError(f'shard {path}: {e}') from None
        if rows['widths'] and tail is None:
            raise ValueError(f'shard {path}: no tail tokens stored (a version 1 shard, or no tail segments), the '
                             f'network has tail segments {rows["widths"]} (R9: a v7 network trains on v2 shards only)')
        if rows['pointers'] and ov is None:
            raise ValueError(f'shard {path}: no pointer overrides stored, the network reads them (R9)')
        sel_rows = np.arange(N) if idx is None else idx
        if tail is not None:
            out['tail'] = tail.take(sel_rows)
        if ov is not None:
            out['ov'] = ov.take(sel_rows)
    elif _shard_has_rows(p):
        raise ValueError(f'shard {path}: rows with tail tokens / pointer overrides, the network has none - its window '
                         f'would drop them (R11; a network with the shard\'s tail segments reads them)')
    if sparse:
        out['xs'] = p.pop('xs')
    if idx is None:
        idx = np.arange(N)
        sel = slice(None)
    else:
        sel = idx
    if not sparse:
        if xdst is None:
            xdst = out['x'] = np.empty((len(idx), F), np.float16)
        if 'x_val' in p:                           # a sparse shard
            xdst[:] = 0
            stored_rows(p, F).scatter(xdst, np.arange(len(idx)), idx)
        else:
            w = min(F, p['x'].shape[1])
            xdst[:, :w] = p['x'][sel, :w]
            if w < F:
                xdst[:, w:] = 0
    off = p['opt_off'].astype(np.int64)
    lens = off[idx + 1] - off[idx]
    pos = _ranges(off[idx], lens)
    out.update(src=idx.astype(np.int32), player=p['player'][sel].astype(np.int64),
               n_players=p['n_players'][sel].astype(np.int64), lens=lens,
               opt_action=p['opt_action'][pos].astype(np.int64), pi=p['pi'][pos].astype(np.float32))
    for key in ('z', 'q', 'sc'):
        a = np.zeros((len(idx), MAX_SEATS), np.float32)
        v = p[key][sel].astype(np.float32)
        a[:, :v.shape[1]] = v
        out[key] = a
    if AW:
        if 'aux' in p:
            w = p['aux'].shape[1]
            if older is not None and w != AW and w in older['maps']:
                # a shard of an older, narrower layout (docs/az.md 37): its targets in their places
                cols, have = older['maps'][w]
                out['aux'] = np.zeros((len(idx), AW), np.float16)
                out['aux'][:, cols] = p['aux'][sel].astype(np.float16)
                out['aux_mask'] = np.ones(len(idx), bool)
                out['aux_tmask'] = np.repeat(have[None, :], len(idx), 0)
                return out
            if w != AW:
                raise ValueError(f'shard {path}: auxiliary targets of width {w}, the window wants {AW}'
                                 + (f' (or an older layout of {sorted(older["maps"])})' if older is not None else ''))
            out['aux'] = p['aux'][sel].astype(np.float16)
            out['aux_mask'] = np.ones(len(idx), bool)
        else:
            out['aux'] = np.zeros((len(idx), AW), np.float16)
            out['aux_mask'] = np.zeros(len(idx), bool)
        if older is not None:
            out['aux_tmask'] = np.repeat(out['aux_mask'][:, None], older['T'], 1)
    return out


def _subset_part(part, idx, xs=True):
    """A shard's full part (`_load_part` with idx None) cut to its rows `idx` (sorted): the part of
    loading the shard with `idx`, array for array (`xs` False: its features left out)."""
    lens = part['lens']
    off = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
    pos = _ranges(off[idx], lens[idx])
    out = {k: part[k][idx] for k in ('src', 'player', 'n_players', 'lens', 'z', 'q', 'sc', 'aux', 'aux_mask',
                                     'aux_tmask') if k in part}
    out.update(opt_action=part['opt_action'][pos], pi=part['pi'][pos])
    for k in ('tail', 'ov'):                      # (variable-length rows: the rows' tails and overrides)
        if k in part:
            out[k] = part[k].take(idx)
    if xs and 'xs' in part:
        out['xs'] = part['xs'].take(idx)
    return out


def _run_all(fn, n, threads):
    """[fn(k) for k in range(n)] on up to `threads` threads; the first failure in order is raised."""
    t = max(1, min(threads or WINDOW_THREADS, n))
    if t == 1:
        return [fn(k) for k in range(n)]
    with ThreadPoolExecutor(t) as ex:
        futs = [ex.submit(fn, k) for k in range(n)]
        return [f.result() for f in futs]


def batches(W, rng, n, size, device='cpu', prefetch=0, tokens=None, coo=False):
    """A training's `n` batches: `size` window rows drawn from `rng` (uniform, with replacement),
    one draw per batch in order -> (rows, W.batch(rows, device, tokens)). `prefetch` > 0: drawn and
    built up to that many batches ahead on a background thread (as CPU tensors, moved to `device` as
    they are taken) while the caller trains on the current one - the same draws, the same
    batches. A failure there is raised here; closing the generator (or dropping it) stops the
    thread. `coo` (a sparse window): each batch's features travel as their non-zeros and are
    scattered into the dense rows on `device` (`Window.batch`'s `coo`: the same rows, bit for bit)."""
    def build(idx, dev):                       # (a window-like `W` without the options: the call of before)
        return W.batch(idx, dev, tokens=tokens, coo=coo) if tokens is not None or coo else W.batch(idx, dev)

    def to(b):
        if device != 'cpu':
            # (a window's 'tail' / 'ov' are dicts of tensors: variable-length rows, docs/az.md 43)
            b = {k: ({j: u.to(device) for j, u in v.items()} if isinstance(v, dict) else v.to(device))
                 for k, v in b.items()}
        return dense_coo(b, W.F) if coo else b
    if not prefetch:
        for _ in range(n):
            idx = rng.integers(W.N, size=size)
            yield idx, to(build(idx, 'cpu')) if coo else build(idx, device)
        return
    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            for _ in range(n):
                idx = rng.integers(W.N, size=size)
                if not put((idx, build(idx, 'cpu'))):
                    return
        except BaseException as e:             # noqa: BLE001  (re-raised by the consumer)
            put(e)

    th = threading.Thread(target=produce, name='zero-batch-prefetch', daemon=True)
    th.start()
    try:
        for _ in range(n):
            while True:
                try:
                    item = q.get(timeout=1.0)
                    break
                except queue.Empty:
                    if not th.is_alive() and q.empty():
                        raise RuntimeError('the batch prefetch thread ended early')
            if isinstance(item, BaseException):
                raise item
            idx, b = item
            yield idx, to(b)
    finally:
        stop.set()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join(timeout=10)


def dense_coo(b, F):
    """A `coo` batch (`Window.batch`) with its features as dense float32 rows 'x' [B, F] on the batch's
    device, scattered from 'x_rows' / 'x_cols' / 'x_vals' (which it drops): `features(idx, float32)`
    bit for bit (a float16 value is exact in float32; every other float a zero)."""
    rows, cols, vals = b.pop('x_rows'), b.pop('x_cols'), b.pop('x_vals')
    x = torch.zeros((len(b['n_players']), F), dtype=torch.float32, device=vals.device)
    x.view(-1).index_copy_(0, rows * F + cols, vals.float())
    b['x'] = x
    return b


class WindowCache:
    """Shards kept between a run's windows (module docstring): path -> (its size and mtime, the
    number of features, its part with the features of all its rows)."""

    def __init__(self):
        self.parts = {}

    def clear(self):
        self.parts.clear()

    @property
    def nbytes(self):
        return sum(_nbytes(a) for _, _, part in self.parts.values() for a in part.values())


def _nbytes(a):
    """An array's bytes, or a `rows.Tail` / `rows.Overrides`'."""
    return a.nbytes() if callable(a.nbytes) else a.nbytes


class Window:
    """The training window in RAM. paths: shard files; `budget_rows`: keep at most this many
    samples (a uniform random share of every shard). A shard with a NaN / inf raises NonFinite.
    `threads`: shards loaded at once (default WINDOW_THREADS); the result does not depend on it.
    `cache`: a WindowCache (module docstring): used while the window fits its budget; `reused` =
    the rows it supplied. `aux_width` W > 0: the auxiliary targets too (module docstring; `aux_rows`
    = the rows that have them). `aux_older` (`auxtarget.older_columns`; a run whose auxiliary head was
    widened, docs/az.md 37): shards recorded under an older, narrower layout load too, their targets in
    their places, and `aux_tmask` [N, T] (else None) says which targets each row has.
    `layout` (the network's normalised token layout; variable-length rows, docs/az.md 43): with tail segments or
    pointer overrides the window loads the shards' tails and overrides (`tail`: `rows.Tail` over the window's rows,
    `ov`: `rows.Overrides`; None without), refusing shards of other tail segments or without them (R9); without, a
    shard with any tail token or override is refused (R11)."""

    def __init__(self, paths, num_features, num_actions, budget_rows=None, seed=0, threads=None, cache=None,
                 aux_width=0, sparse=False, aux_older=None, layout=None):
        self.F, self.A = num_features, num_actions
        self.AW = AW = int(aux_width)
        older = aux_older if AW else None
        okw = {'older': older} if older is not None else {}     # (none: the calls of before)
        self.layout = layout
        lt = R.layout_tails(layout) if layout is not None else {}
        pointers = bool(layout is not None and layout.get('option_pointers'))
        rspec = {'widths': lt, 'pointers': pointers}
        okw['rows'] = rspec if (lt or pointers) else None
        rkey = (('rows', tuple(sorted((s, v[:2]) for s, v in lt.items())), pointers),) if (lt or pointers) else ()
        self.paths = list(paths)
        self.reused = 0
        self._x = self._xs = self._csr = self._xsp = None
        rng = np.random.default_rng(seed)
        rows = [shard_rows(p) for p in self.paths]
        total = sum(rows)
        keep = 1.0 if not budget_rows or total <= budget_rows else budget_rows / total
        # Each shard's rows, drawn in path order (the random stream of loading one at a time).
        picks = [None if keep >= 1.0 else np.sort(rng.choice(N, size=max(1, int(round(N * keep))), replace=False))
                 for N in rows]
        kept = [N if idx is None else len(idx) for N, idx in zip(rows, picks)]
        self.starts = np.concatenate([[0], np.cumsum(kept)]).astype(np.int64)
        self.N = int(self.starts[-1])
        if self.N == 0:
            raise ValueError('empty training window')
        if sparse:
            T = threads or WINDOW_THREADS
            if cache is not None:
                # every shard cached whole (its rows as non-zeros), subsampled from the cache: the window
                # of loading the picks, also over budget
                ck = ('sparse', self.F, AW) + ((older['key'],) if older is not None else ()) + rkey
                keys = {path: (_file_key(path), ck) for path in self.paths}
                for path in list(cache.parts):
                    if path not in keys or cache.parts[path][:2] != keys[path]:
                        del cache.parts[path]
                todo = [k for k, path in enumerate(self.paths) if path not in cache.parts]
                todo.sort(key=lambda k: -rows[k])  # the big shards first (the order of the results is the paths')
                loaded = _run_all(lambda j: _load_part(self.paths[todo[j]], rows[todo[j]], None, self.F, AW=AW,
                                                       sparse=True, xthreads=T, **okw), len(todo), T)
                for k, part in zip(todo, loaded):
                    cache.parts[self.paths[k]] = keys[self.paths[k]] + (part,)
                self.reused = self.N - sum(kept[k] for k in todo)
                full = [cache.parts[path][2] for path in self.paths]
                # the features stay in the cached shards (`features` reads a row from its shard by
                # src_row); the other arrays cut to the picks
                parts = [q if idx is None else _subset_part(q, idx, xs=False) for q, idx in zip(full, picks)]
                self._xsp = [q['xs'] for q in full]
            else:
                # (a big shard's conversion on T threads too: it is the slowest part, and a small
                # shard's few chunks bound the threads in flight)
                parts = _run_all(lambda k: _load_part(self.paths[k], rows[k], picks[k], self.F, AW=AW, sparse=True,
                                                      xthreads=T, **okw), len(self.paths), T)
                self._csr = SparseRows.concat([q['xs'] for q in parts], self.F)
        elif cache is not None and keep >= 1.0:
            # (the auxiliary width only when on: the keys of before otherwise)
            ck = ((self.F, AW) + ((older['key'],) if older is not None else ())) if AW else self.F
            if rkey:                               # (variable-length rows: the keys of before without them)
                ck = (ck,) + rkey
            keys = {path: (_file_key(path), ck) for path in self.paths}
            for path in list(cache.parts):         # gone from the window, rewritten, another width
                if path not in keys or cache.parts[path][:2] != keys[path]:
                    del cache.parts[path]
            todo = [k for k, path in enumerate(self.paths) if path not in cache.parts]
            loaded = _run_all(lambda j: _load_part(self.paths[todo[j]], rows[todo[j]], None, self.F, AW=AW, **okw),
                              len(todo), threads)
            for k, part in zip(todo, loaded):
                cache.parts[self.paths[k]] = keys[self.paths[k]] + (part,)
            self.reused = self.N - sum(rows[k] for k in todo)
            parts = [cache.parts[path][2] for path in self.paths]
            self._xs = [q['x'] for q in parts]
        else:
            if cache is not None:
                cache.clear()                      # over budget: subsampled shards are not cached
            self._x = np.empty((self.N, self.F), np.float16)
            parts = _run_all(lambda k: _load_part(self.paths[k], rows[k], picks[k], self.F,
                                                  self._x[self.starts[k]:self.starts[k + 1]], AW=AW, **okw),
                             len(self.paths), threads)
        self.src_row = np.concatenate([q['src'] for q in parts])     # window row -> its row in its shard
        self.player = np.concatenate([q['player'] for q in parts])
        self.n_players = np.concatenate([q['n_players'] for q in parts])
        self.opt_off = np.concatenate([np.zeros(1, np.int64), np.cumsum(np.concatenate([q['lens'] for q in parts]))])
        self.opt_action = np.concatenate([q['opt_action'] for q in parts])
        self.pi = np.concatenate([q['pi'] for q in parts])
        if self.opt_action.max() >= self.A:
            raise ValueError(f'action id {self.opt_action.max()} >= the network\'s {self.A} actions')
        self.z = np.concatenate([q['z'] for q in parts])
        self.q = np.concatenate([q['q'] for q in parts])
        self.sc = np.concatenate([q['sc'] for q in parts])
        self.loaded = total
        self.aux = np.concatenate([q['aux'] for q in parts]) if AW else None
        self.aux_mask = np.concatenate([q['aux_mask'] for q in parts]) if AW else None
        self.aux_rows = int(self.aux_mask.sum()) if AW else 0
        self.aux_tmask = np.concatenate([q['aux_tmask'] for q in parts]) if older is not None else None
        self.tail = R.Tail.concat([q['tail'] for q in parts]) if lt else None
        self.ov = R.Overrides.concat([q['ov'] for q in parts]) if pointers else None
        if self.tail is not None and self.tail.N != self.N or self.ov is not None and self.ov.N != self.N:
            raise ValueError('internal: the window\'s tails / overrides do not cover its rows')
        xbytes = (self._x.nbytes if self._x is not None else self._csr.nbytes if self._csr is not None
                  else sum(a.nbytes for a in self._xsp) if self._xsp is not None
                  else sum(a.nbytes for a in self._xs))
        self.nbytes = xbytes + sum(a.nbytes for a in (self.player, self.n_players, self.opt_off, self.opt_action,
                                                      self.pi, self.z, self.q, self.sc)
                                   + ((self.aux, self.aux_mask) if AW else ())
                                   + ((self.aux_tmask,) if self.aux_tmask is not None else ()))
        self.nbytes += sum(r.nbytes() for r in (self.tail, self.ov) if r is not None)

    @property
    def x(self):
        """The features [N, F] float16 (from the cache or a sparse window: a dense copy - tests,
        debugging)."""
        if self._csr is not None or self._xsp is not None:
            return self.features(np.arange(self.N))
        return self._x if self._x is not None else np.concatenate(self._xs)

    def features(self, idx, dtype=np.float16):
        """The features of window rows `idx`, [len(idx), F] in `dtype` (float16; float32: their exact
        float32 values - a sparse window builds them directly)."""
        if self._csr is not None:
            return self._csr.dense(idx, dtype)
        if self._xsp is not None:                  # (a cached sparse window: each row in its shard)
            idx = np.asarray(idx, np.int64)
            sh = np.searchsorted(self.starts, idx, side='right') - 1
            src = self.src_row[idx]
            order = np.argsort(sh, kind='stable')
            cut = np.flatnonzero(np.diff(sh[order])) + 1
            out = np.zeros((len(idx), self.F), dtype)
            for g in np.split(order, cut):
                if len(g):
                    self._xsp[sh[g[0]]].scatter(out, g, src[g])
            return out
        if dtype != np.float16:
            return self.features(idx).astype(dtype)
        if self._x is not None:
            return self._x[idx]
        idx = np.asarray(idx, np.int64)
        sh = np.searchsorted(self.starts, idx, side='right') - 1
        order = np.argsort(sh, kind='stable')       # the rows grouped by shard
        cut = np.flatnonzero(np.diff(sh[order])) + 1
        out = np.empty((len(idx), self.F), np.float16)
        for g in np.split(order, cut):
            k = sh[g[0]]
            out[g] = self._xs[k][idx[g] - self.starts[k]]
        return out

    def provenance(self, idx):
        """Window rows -> [(shard path, its rows in that shard, sorted)]."""
        idx = np.asarray(idx, np.int64)
        sh = np.searchsorted(self.starts, idx, side='right') - 1
        return [(self.paths[k], np.unique(self.src_row[idx[sh == k]])) for k in np.unique(sh)]

    @property
    def sparse(self):
        """Whether the features are kept as their non-zeros (a sparse window)."""
        return self._csr is not None or self._xsp is not None

    def features_coo(self, idx):
        """A sparse window's features of rows `idx` as their non-zeros -> (rows [nnz] int64: each one's
        position in `idx`, cols [nnz] int64, vals [nnz] float16): `features(idx)`'s non-zeros (a -0.0
        too), grouped by shard."""
        idx = np.asarray(idx, np.int64)
        if self._csr is not None:
            groups = [(np.arange(len(idx)), self._csr, idx)]
        elif self._xsp is not None:
            sh = np.searchsorted(self.starts, idx, side='right') - 1
            src = self.src_row[idx]
            order = np.argsort(sh, kind='stable')
            cut = np.flatnonzero(np.diff(sh[order])) + 1
            groups = [(g, self._xsp[sh[g[0]]], src[g]) for g in np.split(order, cut) if len(g)]
        else:
            raise ValueError('features_coo: not a sparse window')
        R, C, V = [], [], []
        for at, part, rows in groups:
            rows, lens, pos = part._pos(rows)
            R.append(np.repeat(at, lens))
            C.append(part.idx[pos].astype(np.int64))
            V.append(part.val[pos])
        return np.concatenate(R), np.concatenate(C), np.concatenate(V)

    def use_layout(self, layout):
        """Resolve overrides with `layout` (a network's normalised token layout) from now on: its tail segments and
        pointer overrides must be the window's (R9; a window loaded with `rows.shard_layout` of its shards, e.g. a
        held-out set, takes the network's here)."""
        lt = {s: v[:2] for s, v in R.layout_tails(layout).items()}
        mine = {s: v[:2] for s, v in R.layout_tails(self.layout).items()} if self.layout is not None else {}
        if lt != mine or bool(layout.get('option_pointers')) != (self.ov is not None):
            raise ValueError(f'the network\'s tail segments {lt} / option_pointers {bool(layout.get("option_pointers"))} '
                             f'differ from the window\'s {mine} / {self.ov is not None} (R9)')
        self.layout = layout

    def row_inputs(self, idx, device='cpu', x=None):
        """The network's variable-length row inputs of window rows `idx` (docs/az.md 43): {'tail': the rows' tails
        (`rows.Tail.to_torch`), 'ov': their overrides resolved (`rows.resolve`; the head slots' presence from the
        float features `x` [len(idx), F], read here when None)} for the kinds the window has; {} for a window without
        them."""
        if self.tail is None and self.ov is None:
            return {}
        if self.layout is not None and self.layout.get('placeholder'):
            raise ValueError('Window.row_inputs: the window knows its shards\' tail segments only (use_layout(the '
                             'network\'s layout) first)')
        out = {}
        tail = self.tail.take(idx) if self.tail is not None else None
        if tail is not None:
            out['tail'] = tail.to_torch(device)
        if self.ov is not None:
            if x is None:
                x = self.features(idx, np.float32)
            off = self.opt_off
            lens = off[idx + 1] - off[idx]
            acts = self.opt_action[_ranges(off[idx], lens)]
            boff = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
            out['ov'] = R.ov_torch(R.resolve(tail, self.ov.take(idx), boff, acts, self.layout,
                                             present=R.presence(np.asarray(x), self.layout)), device)
        return out

    def batch(self, idx, device='cpu', tokens=None, coo=False):
        """The training batch of window rows `idx` (tensors on `device`). `tokens` (a token network's
        static forward, run setting train_static: (TokenNet.present_index() as an int64 array, T step))
        adds each row's tokens as `TokenNet.forward_static` takes them: 'full' [B, T] (their positions in
        the full token axis, the global token's 0 first, then padding 0) and 'ntok' [B], T the longest
        row's tokens rounded up to the step. `coo` (a sparse window): the features as their non-zeros
        'x_rows' / 'x_cols' / 'x_vals' (`features_coo`) instead of 'x' - `dense_coo` makes 'x' of them on the
        device -, the tokens read from them.
        A window with tails / overrides (`layout`): 'tail' (`rows.Tail.to_torch`: the network's `tail`, unpadded) and
        'ov' (`rows.resolve`d: the network's `ov`, a pointer to an absent head slot or a missing tail token refused);
        with `tokens`, each row's tail tokens follow its head tokens in 'full' (`rows.token_full`)."""
        B = len(idx)
        off = self.opt_off
        starts = off[idx]
        lens = off[idx + 1] - starts
        pos = _ranges(starts, lens)
        rows = np.repeat(np.arange(B), lens)
        acts = self.opt_action[pos]
        mask = np.zeros((B, self.A), bool)
        pi = np.zeros((B, self.A), np.float32)
        mask[rows, acts] = True
        pi[rows, acts] = self.pi[pos]
        pi /= np.maximum(pi.sum(1, keepdims=True), 1e-8)

        def norm(a):
            return a / np.maximum(a.sum(1, keepdims=True), 1e-8)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        if coo:
            xr, xc, xv = self.features_coo(idx)
            out = {'x_rows': t(xr), 'x_cols': t(xc), 'x_vals': t(xv)}
        else:
            x = self.features(idx, np.float32)
            out = {'x': t(x)}
        out.update({'mask': t(mask), 'pi': t(pi),
                    'z': t(norm(self.z[idx])), 'q': t(norm(self.q[idx])), 'sc': t(self.sc[idx]),
                    'n_players': t(self.n_players[idx])})
        if self.AW:
            out['aux'] = t(self.aux[idx].astype(np.float32))
            out['aux_mask'] = t(self.aux_mask[idx].astype(np.float32))
            if self.aux_tmask is not None:
                out['aux_tmask'] = t(self.aux_tmask[idx].astype(np.float32))
        tail = self.tail.take(idx) if self.tail is not None else None
        ov = self.ov.take(idx) if self.ov is not None else None
        if (tail is not None or ov is not None) and self.layout.get('placeholder'):
            raise ValueError("Window.batch: the window knows its shards' tail segments only (use_layout(the network's "
                             "layout) first)")
        if tokens is None and ov is not None:
            tokens_p = (R.present_index(self.layout), None)     # (the head slots' presence, for `resolve` only)
        else:
            tokens_p = tokens
        if tail is not None:
            out['tail'] = tail.to_torch(device)
        if tokens_p is not None:
            pidx, step = tokens_p
            has = pidx >= 0
            P = np.ones((B, 1 + len(pidx)), bool)          # (infer.token_presence / token_index)
            if coo:                                         # a flag is present when its value > 0.5
                P[:, 1:][:, has] = False
                slot = np.full(self.F, -1, np.int64)
                slot[pidx[has]] = np.flatnonzero(has) + 1
                c = slot[xc]
                on = (c >= 0) & (xv > 0.5)
                P[xr[on], c[on]] = True
            else:
                P[:, 1:][:, has] = x[:, pidx[has]] > 0.5
            if ov is not None:
                boff = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])     # (the batch's option segments)
                out['ov'] = R.ov_torch(R.resolve(tail, ov, boff, acts, self.layout, present=P), device)
            if tokens is None:
                return out
            if tail is not None:
                full, n = R.token_full(P, tail.counts(), 1 + len(pidx), step)
                out['full'], out['ntok'] = t(full), t(n)
                return out
            n = P.sum(1)
            T = -(-int(n.max()) // step) * step
            full = np.zeros((B, T), np.int64)
            r, c = np.nonzero(P)
            full[r, (np.cumsum(P, 1) - 1)[r, c]] = c
            out['full'], out['ntok'] = t(full), t(n.astype(np.int64))
        return out
