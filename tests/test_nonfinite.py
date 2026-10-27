"""The non-finite guard (zero/finite.py, docs/az.md 4.1): a NaN / inf in a shard, a training
step or a checkpoint fails the run loudly (MTG Zero's mtg4 ran 10 iterations on a NaN network)."""
import glob
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from zero import _engine as E
from zero import games as G
from zero.data import ShardWriter, Window, compact, read_shard, shard_paths, write_shard
from zero.finite import NonFinite, nonfinite_rows
from zero.infer import UniformEvaluator, drive
from zero.net import Net, save
from zero.run import DEFAULTS, Run

from test_loop import PYPATH, ROOT, SMOKE

INFO = E.game_info_of('toy')
F, A = INFO['num_features'], INFO['num_actions']


@pytest.fixture(scope='module')
def samples():
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5)
    drv = G.Driver('toy', [[seat, seat]] * 8, concurrency=8, threads=1, seed=5, record=True)
    parts = []
    drive(drv, {'m': UniformEvaluator()}, on_batch=lambda lb: parts.append(drv.take_samples()))
    parts.append(drv.take_samples())
    from zero.protocol import concat_samples
    return concat_samples([p for p in parts if p is not None])


def test_nonfinite_rows_float16_bits_and_float32():
    a = np.zeros((300, 7), np.float16)
    assert len(nonfinite_rows(a, chunk=50)) == 0
    a[3, 2], a[250, 6], a[251, 0] = np.inf, np.nan, -np.inf
    a[10, 1] = 65504                               # the float16 maximum is finite
    assert nonfinite_rows(a, chunk=50).tolist() == [3, 250, 251]
    assert nonfinite_rows(a.astype(np.float32), chunk=50).tolist() == [3, 250, 251]
    assert nonfinite_rows(np.array([1.0, np.nan, 2.0])).tolist() == [1]
    assert len(nonfinite_rows(np.arange(5))) == 0


def test_write_shard_quarantines_float16_overflow(samples, tmp_path):
    s = dict(samples, x=samples['x'].copy())
    s['x'][4, 7] = 1e6                             # an encoder value above 65504: inf in float16
    d = tmp_path / 'data' / 'iter_0003'
    w = ShardWriter(str(d), 'c00', rows=10 ** 6, meta={'game': 'toy', 'iter': 3}, bad_dir=str(tmp_path / 'debug'))
    w.add(s)
    with pytest.raises(NonFinite) as e:
        w.close()
    msg = str(e.value)
    print(msg)
    assert 'shard c00_000.npz' in msg and '"iter": 3' in msg and '"game": "toy"' in msg
    assert 'x: 1 row(s) [4], column(s) [7] (1 +inf)' in msg
    bad = tmp_path / 'debug' / 'iter_0003_c00_000.bad.npz'
    assert bad.exists() and str(bad) in msg
    assert shard_paths(str(d)) == []               # nothing loadable was written
    q = read_shard(str(bad))
    assert len(q['player']) == len(s['player']) and q['meta']['nonfinite']


def test_write_shard_names_targets(samples, tmp_path):
    s = {k: v.copy() for k, v in samples.items()}
    s['z'][2, 1] = np.nan
    s['pi'][s['opt_off'][5] + 1] = np.inf
    p = str(tmp_path / 'c00_000.npz')
    with pytest.raises(NonFinite) as e:
        write_shard(p, s, {'game': 'toy'})
    msg = str(e.value)
    assert 'z: 1 row(s) [2], column(s) [1] (1 NaN)' in msg and 'pi: 1 option entr(ies) (1 +inf) in sample row(s) [5]' in msg
    assert os.path.exists(str(tmp_path / 'c00_000.bad.npz')) and not os.path.exists(p)
    assert shard_paths(str(tmp_path)) == []        # the window never loads *.bad.npz


def test_window_checks_old_shards_and_tells_provenance(samples, tmp_path):
    good = str(tmp_path / 'c00_000.npz')
    write_shard(good, samples, {'game': 'toy'})
    N = len(samples['player'])
    w = Window([good, good], F, A, budget_rows=N, seed=1)   # two copies, half of each kept
    rows = np.arange(w.N)
    for path, r in w.provenance(rows):
        assert path == good and r.max() < N
    sh = np.searchsorted(w.starts, rows, side='right') - 1
    for k in (0, 5, w.N - 1):                       # the window row is that shard row
        assert np.allclose(w.x[k], read_shard(w.paths[sh[k]])['x'][w.src_row[k]])
    # a shard written before the guard (raw npz): the window refuses it, naming it
    c = compact(samples)
    c['q'][9, 0] = np.nan
    old = str(tmp_path / 'c00_001.npz')
    np.savez(old, meta=np.array(json.dumps({'game': 'toy', 'iter': 7})), **c)
    with pytest.raises(NonFinite) as e:
        Window([good, old], F, A)
    assert f'shard {old}' in str(e.value) and 'q: 1 row(s) [9], column(s) [0] (1 NaN)' in str(e.value)


def test_checkpoint_refuses_nonfinite(tmp_path):
    m = Net(F, A, hidden=16, blocks=1)
    save(m, str(tmp_path / 'ok.pt'))
    with torch.no_grad():
        next(p for p in m.parameters() if p.dim() == 2)[0, 0] = float('nan')
    with pytest.raises(NonFinite, match=r'refusing to save .*bad\.pt: 1 of \d+ float tensors hold NaN / inf'):
        save(m, str(tmp_path / 'bad.pt'))
    assert not glob.glob(str(tmp_path / 'bad.pt*'))


def test_prune_keeps_quarantined_shards_and_debug(tmp_path):
    run = tmp_path / 'run'
    R = Run(str(run), dict(DEFAULTS, keep_data=2))
    for it in (1, 2, 5):
        d = run / 'data' / f'iter_{it:04d}'
        d.mkdir(parents=True)
        (d / 'c00_000.npz').write_bytes(b'x')
    (run / 'data' / 'iter_0001' / 'c00_001.bad.npz').write_bytes(b'bad')
    (run / 'debug').mkdir()
    (run / 'debug' / 'nonfinite_iter_0002_step_0001.pt').write_bytes(b'dbg')
    R.prune(5)
    assert sorted(os.listdir(run / 'data')) == ['iter_0005']
    assert sorted(os.listdir(run / 'debug')) == ['iter_0001_c00_001.bad.npz', 'nonfinite_iter_0002_step_0001.pt']


# ---------------------------------------------------------------- the loop end to end
RUNNER = textwrap.dedent('''
    import sys
    import numpy as np
    import zero.data as D, zero.net as N, zero.run as R
    mode = sys.argv[1]
    if mode in ('x-overflow', 'z-nan'):
        c0 = D.compact
        def compact(s):
            s = dict(s)
            k = 'x' if mode == 'x-overflow' else 'z'
            a = s[k].copy()
            a[3, 5 if k == 'x' else 1] = 1e6 if k == 'x' else np.nan
            s[k] = a
            return c0(s)
        D.compact = compact
    elif mode in ('loss-nan', 'grad-inf'):          # at the 3rd training step
        f0 = N.Net.forward
        calls = [0]
        def forward(self, x, n_players):
            p, v, m = f0(self, x, n_players)
            if self.training:
                calls[0] += 1
                if calls[0] == 3 and mode == 'loss-nan':      # batch row 5's value logits
                    v = v.clone()
                    v[5] = float('nan')
                elif calls[0] == 3:                           # a finite loss, an infinite gradient
                    m.register_hook(lambda g: g + float('inf'))
            return p, v, m
        N.Net.forward = forward
    R.main(sys.argv[2:])
''')


def run_injected(tmp_path, mode):
    run = tmp_path / mode
    script = tmp_path / 'inject.py'
    script.write_text(RUNNER)
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    p = subprocess.run([sys.executable, str(script), mode, '--run', str(run), *SMOKE, '--iters', '2'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=e)
    return run, p


def assert_failed_without_checkpoint(run, p):
    assert p.returncode != 0, p.stdout + p.stderr
    assert 'zero.finite.NonFinite' in p.stderr
    assert (run / 'models' / 'iter_0000.pt').exists() and not (run / 'models' / 'iter_0001.pt').exists()
    assert not (run / 'models' / 'opt_latest.pt').exists()
    with open(run / 'run_state.json') as f:
        assert json.load(f)['state'] == 'aborted'
    with open(run / 'log.txt', encoding='utf-8') as f:
        assert 'run aborted: NonFinite: ' in f.read()


@pytest.mark.parametrize('mode', ['x-overflow', 'z-nan'])
def test_loop_fails_on_nonfinite_samples(tmp_path, mode):
    run, p = run_injected(tmp_path, mode)
    assert_failed_without_checkpoint(run, p)
    msg = p.stderr.strip().splitlines()[-1]
    print(msg)
    k = 'x: 1 row(s) [3], column(s) [5] (1 +inf)' if mode == 'x-overflow' else 'z: 1 row(s) [3], column(s) [1] (1 NaN)'
    assert 'non-finite sample values in shard c00_000.npz' in msg and k in msg and '"iter": 1' in msg
    assert os.listdir(run / 'debug') == ['iter_0001_c00_000.bad.npz']
    assert shard_paths(str(run / 'data' / 'iter_0001')) == []


def test_loop_fails_on_nonfinite_loss_before_the_step(tmp_path):
    run, p = run_injected(tmp_path, 'loss-nan')
    assert_failed_without_checkpoint(run, p)
    msg = p.stderr.strip().splitlines()[-1]
    print(msg)
    # the NaN row's gradient is cut by the assignment: only the loss shows it
    assert 'iter 1 training step 3 of ' in msg and 'non-finite value before optimizer.step()' in msg
    assert 'batch inputs finite; weights finite; 1 of 64 batch rows have a non-finite per-sample loss, from: ' \
           f'{os.path.join("data", "iter_0001", "c00_000.npz")} rows [' in msg
    dbg = run / 'debug' / 'nonfinite_iter_0001_step_0003.pt'
    assert str(dbg) in msg
    d = torch.load(dbg, weights_only=False)
    assert d['iter'] == 1 and d['step'] == 3 and d['nonfinite'] == ['value']
    assert d['batch']['x'].shape == (64, F) and d['bad_batch_rows'].tolist() == [5]
    assert torch.isnan(d['per_sample']['value'][5]) and torch.isfinite(d['per_sample']['policy']).all()
    w = Window(d['shards'], F, A)                  # the saved provenance finds the rows again
    k = 5
    shard = read_shard(d['shards'][d['shard_of_row'][k]])
    x = torch.from_numpy(shard['x'][d['row_in_shard'][k]].astype(np.float32))
    assert torch.equal(d['batch']['x'][k], x) and w.N > 0
    assert all(torch.isfinite(v).all() for v in d['model_state'].values())


def test_loop_fails_on_nonfinite_gradient(tmp_path):
    run, p = run_injected(tmp_path, 'grad-inf')
    assert_failed_without_checkpoint(run, p)
    msg = p.stderr.strip().splitlines()[-1]
    assert 'iter 1 training step 3 of ' in msg and 'non-finite grad_norm before optimizer.step()' in msg
    assert 'every per-sample loss is finite (the gradient is not); the batch came from: ' in msg
    assert (run / 'debug' / 'nonfinite_iter_0001_step_0003.pt').exists()


def test_normal_run_has_no_debug_dir(tmp_path):
    run = tmp_path / 'ok'
    script = tmp_path / 'inject.py'
    script.write_text(RUNNER)
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    p = subprocess.run([sys.executable, str(script), 'none', '--run', str(run), *SMOKE, '--iters', '1'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=e)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (run / 'models' / 'iter_0001.pt').exists() and not (run / 'debug').exists()
