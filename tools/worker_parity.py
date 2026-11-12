"""Parity gate 2 (a), the plumbing (zero-farm phase 0b, docs/az.md 42): the native worker's self-play job vs
the Python remote worker on a run's real settings, both with the uniform evaluator (exact arithmetic in
both languages), the same seed and deterministic settings (groups 1: the batch order is then fixed) -
the sample rows and the first network batches sent to the evaluator must be byte-identical.

    python tools/worker_parity.py --root <game checkout> --package mtgz.az --config <run>/config.json \
        --checkpoint <a checkpoint of the run> --worker <native worker exe> --out <dir> \
        [--games 48 --batches 300 --set groups=1 --set shard_rows=8192 --iter 81]

The Python side is zero-core's `remoteplay.Worker` itself, in this process, fed its control lines
directly: `phase` (its `_load` builds the run's seat and driver from the mirror's config.json; the
evaluator is then replaced by a recording `UniformEvaluator`), an `accept` for every `fin`, `end` once
`--games` games are accepted. Two changes only: its setups stop at `--games` in all (`_topup`; the
native job's game count, the run's `cap` limits only the games in flight) and no stdin thread (the
control lines come from here; the watchdog is off). Its shards are those of `zero.data.ShardWriter`
(the mirror's shard_writer_thread setting as given). The native side: `<worker> selfplay --settings
<the same config.json> --uniform --games N --seed <the remote worker's seed> --net <iter - 1>
--record-batches B`. Compared: every shard's arrays and cut points, the meta keys of the format, and
the first B batches (every array). Exit 1 on a difference.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time


def py_side(a, mirror, it):
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    import numpy as np
    from zero import remoteplay as RP
    from zero.infer import UniformEvaluator

    rec = {'batches': [], 'n': a.batches}

    class Recording:
        def __init__(self):
            self.u = UniformEvaluator()

        def __call__(self, lb):
            if len(rec['batches']) < rec['n']:
                rec['batches'].append({k: np.array(lb.a[k]) for k in ('xs_off', 'xs_idx', 'xs_val', 'n_players', 'player',
                                                                        'gid', 'extra', 'opt_off', 'opt_action')})
            return self.u(lb)

    class W(RP.Worker):
        budget = a.games

        def _stdin(self):                          # control lines come from the harness
            threading.Event().wait()

        def _load(self, it_, model, league=None):
            super()._load(it_, model, league)
            self.evr = Recording()

        def _topup(self, ph):
            if ph['ending']:
                return
            cap = self.cap if self.cap is not None else int(self.cfg['conc'])
            room = cap - self.outstanding() - (len(self.held) if ph['closed'] else 0)
            room = min(room, self.budget - self.enq)
            for _ in range(max(0, room)):
                self.drv.enqueue(f'{self.sid}:{self.enq}', self.spec)
                self.enq += 1

        def emit(self, ev, **kw):
            super().emit(ev, **kw)
            if ev == 'fin':
                self.fins = getattr(self, 'fins', 0) + kw['n']
                self.ctl.put({'cmd': 'accept', 'iter': kw['iter'], 'n': kw['n']})
                if self.fins >= self.budget:
                    self.ctl.put({'cmd': 'end', 'iter': kw['iter']})
            elif ev == 'end' and kw.get('iter') == it:
                self.ctl.put({'cmd': 'quit'})

    w = W(mirror, 0)
    w.ctl.put({'cmd': 'phase', 'iter': it, 'model': f'models/iter_{it - 1:04d}.pt'})
    t0 = time.time()
    code = w.main()
    if code != 0:
        raise SystemExit(f'the Python remote worker failed (exit {code}): see {mirror}/w00.log')
    return rec['batches'], w.k, time.time() - t0


def save_batches(path, batches):
    import numpy as np
    out = {k: [] for k in ('xs_idx', 'xs_val', 'n_players', 'player', 'gid', 'extra', 'opt_action')}
    xs_off, opt_off, bo = [0], [0], [0]
    for b in batches:
        for k in out:
            out[k].append(b[k])
        xs_off.extend((xs_off[-1] + np.asarray(b['xs_off'][1:], np.int64)).tolist())
        opt_off.extend((opt_off[-1] + np.asarray(b['opt_off'][1:], np.int64)).tolist())
        bo.append(bo[-1] + len(b['n_players']))
    np.savez(path, meta=json.dumps({'source': 'python remote worker'}), xs_off=np.array(xs_off, np.int64),
             opt_off=np.array(opt_off, np.int64), batch_off=np.array(bo, np.int64),
             **{k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()})


def compare(py_shards, rs_shards, py_batches, rs_batch_file, nb):
    import numpy as np
    bad = []
    if len(py_shards) != len(rs_shards):
        bad.append(f'{len(py_shards)} Python shards, {len(rs_shards)} native')
    rows = 0
    fmt = ('game', 'features', 'actions', 'rows', 'x_format', 'x_version', 'x_cols', 'x_nnz')
    for p, r in zip(py_shards, rs_shards):
        with np.load(p) as zp, np.load(r) as zr:
            mp, mr = json.loads(str(zp['meta'])), json.loads(str(zr['meta']))
            for k in fmt:
                if mp.get(k) != mr.get(k):
                    bad.append(f'{os.path.basename(p)} meta {k}: {mp.get(k)} vs {mr.get(k)}')
            kp = [k for k in zp.files if k != 'meta']
            kr = [k for k in zr.files if k != 'meta']
            if kp != kr:
                bad.append(f'{os.path.basename(p)} members {kp} vs {kr}')
            for k in kp:
                if k in zr.files:
                    x, y = zp[k], zr[k]
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        bad.append(f'{os.path.basename(p)} {k}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}, '
                                   f'equal bytes {x.tobytes() == y.tobytes()}')
            rows += int(mp.get('rows', 0))
    with np.load(rs_batch_file) as z:
        bo = z['batch_off']
        nr = len(bo) - 1
        if nr != min(nb, len(py_batches)) or len(py_batches) < min(nb, nr):
            bad.append(f'batches recorded: Python {len(py_batches)}, native {nr}')
        for i in range(min(nr, len(py_batches))):
            a, b0, b1 = py_batches[i], bo[i], bo[i + 1]
            xo = z['xs_off'][b0:b1 + 1]
            oo = z['opt_off'][b0:b1 + 1]
            mine = {'xs_off': xo - xo[0], 'xs_idx': z['xs_idx'][xo[0]:xo[-1]], 'xs_val': z['xs_val'][xo[0]:xo[-1]],
                    'n_players': z['n_players'][b0:b1], 'player': z['player'][b0:b1], 'gid': z['gid'][b0:b1],
                    'extra': z['extra'][b0:b1], 'opt_off': oo - oo[0], 'opt_action': z['opt_action'][oo[0]:oo[-1]]}
            for k, v in mine.items():
                pv = np.asarray(a[k])
                if pv.astype(np.float64).tobytes() != np.asarray(v).astype(np.float64).tobytes() and not (
                        k == 'xs_val' and pv.tobytes() == np.asarray(v, np.float32).tobytes()):
                    bad.append(f'batch {i} {k} differs')
                    break
    return bad, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=48)
    ap.add_argument('--batches', type=int, default=300)
    ap.add_argument('--iter', type=int, default=81)
    ap.add_argument('--set', action='append', default=[], help='key=JSON value: a setting changed for both sides')
    ap.add_argument('--threads', type=int, default=None, help='both sides\' search threads (the config\'s if unset)')
    a = ap.parse_args()
    it = a.iter
    mirror = os.path.join(a.out, 'py')
    native = os.path.join(a.out, 'native')
    for d in (mirror, native):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    os.makedirs(os.path.join(mirror, 'models'))
    shutil.copy(a.checkpoint, os.path.join(mirror, 'models', f'iter_{it - 1:04d}.pt'))
    with open(a.config, encoding='utf-8') as f:
        cfg = json.load(f)
    # the mirror's config: the run's settings as a remote worker gets them (zero/remoteplay.py _prepare_mirror)
    cfg.update(eval_remote='', selfplay_remote='', resolver='', resolver_cfg={}, resolver_init='', infer_device='cpu')
    cfg['shard_rows'] = min(int(cfg['shard_rows']), 8192)
    for s in a.set:
        k, v = s.split('=', 1)
        cfg[k] = json.loads(v)
    if a.threads:
        cfg['threads'] = a.threads
    with open(os.path.join(mirror, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(mirror, 'selfplay.json'), 'w', encoding='utf-8') as f:
        json.dump({'mode': 'selfplay', 'locks': [], 'nice': 0, 'run_id': 'parity', 'mirror': 'parity',
                   'watchdog_s': 1e9, 'linger': False, 'test': {}}, f)
    seed = int(cfg['seed']) * 7919 + it + 7_777_777 * (0 + 1)
    print(f'settings: {a.config} + {a.set}; seed {seed}, net {it - 1}, {a.games} games', flush=True)
    batches, k, py_s = py_side(a, mirror, it)
    print(f'Python remote worker: {py_s:.1f} s', flush=True)
    save_batches(os.path.join(a.out, 'py_batches.npz'), batches)
    t0 = time.time()
    cmd = [a.worker, 'selfplay', '--settings', os.path.join(mirror, 'config.json'), '--uniform', '--games', str(a.games),
           '--seed', str(seed), '--net', str(it - 1), '--threads', str(cfg['threads']), '--out', native,
           '--record-batches', str(a.batches), '--record-file', os.path.join(a.out, 'native_batches.npz')]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    with open(os.path.join(a.out, 'native.log'), 'w', encoding='utf-8') as f:
        f.write(p.stdout + '\n--- stderr\n' + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f'the native worker failed ({p.returncode}): {p.stderr[-2000:]}')
    print(f'native worker: {time.time() - t0:.1f} s', flush=True)
    py_shards = sorted(glob.glob(os.path.join(mirror, 'data', f'iter_{it:04d}', '*.npz')))
    rs_shards = sorted(glob.glob(os.path.join(native, '*.npz')))
    bad, rows = compare(py_shards, rs_shards, batches, os.path.join(a.out, 'native_batches.npz'), a.batches)
    res = {'games': a.games, 'shards': len(py_shards), 'rows': rows, 'batches_compared': min(len(batches), a.batches),
           'differences': bad, 'identical': not bad, 'python_s': round(py_s, 1)}
    print(json.dumps(res))
    with open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
