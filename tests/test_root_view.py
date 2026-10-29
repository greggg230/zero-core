"""Root-view leaf values (docs/search.md 3.6, docs/az.md 9.3): run settings selfplay_leaf_value_view /
eval_leaf_value_view / selfplay_view_samples / view_sample_target, seat keys leaf_value_view /
view_samples / view_sample_target, `play.view_pick` (= the Rust driver's), the Python reference tree
and driver against the Rust ones, the value-only samples and their targets, the Elo pool's name, toy
runs: off = the runs of before byte for byte; self-play on = value-only rows in the shards, trained
without a policy loss; evaluation only = the training byte for byte, another pool."""
import json
import os
import random
import sys
import types

import numpy as np
import pytest

from zero import elo, games as G, h2h, mcts, parity
from zero.play import NNSeat, PyDriver, view_pick
from zero.protocol import concat_samples, validate_samples
from zero.run import DEFAULTS, Run, check_eval_settings, read_metrics

from test_select import _Rand
from test_select_scale import _runs, _training_files

DASH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard')


# ------------------------------------------------------------------ settings, seats, names

def test_settings_seats_and_names(tmp_path):
    assert (DEFAULTS['selfplay_leaf_value_view'], DEFAULTS['eval_leaf_value_view'], DEFAULTS['selfplay_view_samples'],
            DEFAULTS['view_sample_target']) == ('mover', 'mover', 0.0, 'next_q')
    # off: the seat dict of before, key for key
    base = G.nn_seat('m', 8)
    assert base == G.nn_seat('m', 8, leaf_value_view='mover', view_samples=0.0, view_sample_target='z')
    assert not {'leaf_value_view', 'view_samples', 'view_sample_target'} & set(base)
    s = G.nn_seat('m', 8, selfplay=True, leaf_value_view='root', view_samples=0.5)
    assert (s['leaf_value_view'], s['view_samples'], s['view_sample_target']) == ('root', 0.5, 'next_q')
    for bad in (dict(leaf_value_view='decider'), dict(view_samples=1.5), dict(view_samples=-0.1),
                dict(view_samples=True), dict(view_samples=0.5, view_sample_target='q')):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
        with pytest.raises(ValueError):
            NNSeat('m', **bad)
    # the Rust driver takes the keys (and refuses bad ones)
    G.Driver('toy', [[s, s]], concurrency=1)
    for bad in (dict(s, leaf_value_view='x'), dict(s, view_sample_target='q'), dict(s, view_samples=2.0)):
        with pytest.raises(ValueError):
            G.Driver('toy', [[bad, bad]], concurrency=1)
    # run settings: validated
    for bad in ({'selfplay_leaf_value_view': 'x'}, {'eval_leaf_value_view': 'decider'}, {'selfplay_view_samples': 2},
                {'view_sample_target': 'q'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, selfplay_leaf_value_view='root', eval_leaf_value_view='root',
                             selfplay_view_samples=0.25, view_sample_target='z'))
    # Run.nn_seat: self-play's settings on the self-play seat, the evaluation's view on every
    # evaluation seat (which records nothing: no value-only samples there)
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, selfplay_leaf_value_view='root', selfplay_view_samples=0.25))
    sp = Run.nn_seat(r, 'm', 64, selfplay=True)
    assert (sp['leaf_value_view'], sp['view_samples'], sp['view_sample_target']) == ('root', 0.25, 'next_q')
    assert not {'leaf_value_view', 'view_samples'} & set(Run.nn_seat(r, 'm', 32))
    r.cfg = dict(DEFAULTS, eval_leaf_value_view='root')
    assert 'leaf_value_view' not in Run.nn_seat(r, 'm', 64, selfplay=True)
    ev = Run.nn_seat(r, 'm', 32)
    assert ev['leaf_value_view'] == 'root' and 'view_samples' not in ev
    # names: the evaluation's view names its Elo pool; self-play's never does
    assert mcts.view_tag('mover') == '' and mcts.view_tag('root') == '-rv'
    cq = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'eval_select_c_scale': 0.3}
    cases = [({}, 'visits'), ({'selfplay_leaf_value_view': 'root', 'selfplay_view_samples': 0.5}, 'visits'),
             ({'eval_leaf_value_view': 'root'}, 'visits-rv'), (dict(cq, eval_leaf_value_view='root'), 'cq-n4-v50-s0.3-rv'),
             (dict(cq, eval_leaf_value_view='root', eval_root_min_visits=4), 'cq-n4-v50-s0.3-f4-rv'),
             ({'eval_leaf_value_view': 'mover'}, 'visits')]
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k, (c, tag) in enumerate(cases):
        assert elo.eval_tag(c) == tag, c
        d = tmp_path / f'run{k}'
        os.makedirs(d / 'models')
        json.dump(c, open(d / 'config.json', 'w'))
        assert elo.active_pool(str(d)) == (tag, elo.pool_dir(str(d), tag))
        assert runfiles.elo_pool(str(d)) == elo.pool_dir(str(d), tag), c
        # h2h follows a run's evaluation view (and names it)
        rule = h2h.run_rule(str(d / 'models' / 'iter_0000.pt'))
        assert rule.get('leaf_value_view', 'mover') == c.get('eval_leaf_value_view', 'mover')
        assert h2h._tag(dict(G.nn_seat('a', 8), **rule)) == tag


def test_view_pick_matches_the_rust_driver():
    E = G.engine()
    rng = random.Random(3)
    for _ in range(3000):
        seed, gid, dec = rng.getrandbits(64), rng.getrandbits(32), rng.randrange(1 << 16)
        share = rng.choice((0.0, 0.1, 0.25, 0.5, 0.9, 1.0, rng.random()))
        assert view_pick(seed, gid, dec, share) == E.view_pick(seed, gid, dec, share), (seed, gid, dec, share)
    picked = sum(view_pick(7, g, d, 0.3) for g in range(50) for d in range(100))
    assert 1350 < picked < 1650


# ------------------------------------------------------------------ the trees

def test_rust_and_python_trees_take_the_same_root_view_values():
    """Both trees with root-view leaf values at toy positions (the stand-in network reads the row's
    features, so the viewer's give other values than the decider's): the mean visit distributions,
    completed-Q targets and root win at the Python-Python noise floor, the positions where nothing is
    left to chance searched identically; the setting changes the search (mover vs root view), and a
    Rust tree searching mover-view is told apart (power)."""
    r = parity.root_view(n=8, reps=16, sims=64)
    assert r['tv_rs_py'] <= 3 * r['tv_py_py'] + 0.003, r
    assert r['tv_cq_rs_py'] <= 2 * r['tv_cq_py_py'] + 0.01, r
    assert r['win_diff'] < 0.01, r
    assert r['tv_view'] > 10 * r['tv_py_py'] + 0.02, r          # the setting matters
    p = parity.root_view(n=8, reps=8, sims=64, rs_view=False)
    assert p['tv_rs_py'] > 5 * r['tv_py_py'] + 0.02, p


def test_tree_bindings_give_the_viewers_features():
    """`SearchTree.simulate(root_view=True)`: a 4-tuple whose last item is the viewer's features of the
    leaf when another seat decides it (`Game.encode(viewer)` of that state), else None; `expand` then
    reads the value relative to the viewer."""
    ev = parity.LinearEval(G.info('toy')['num_features'], G.info('toy')['num_actions'], seed=1, scale=0.15)
    g = parity.sample_positions('toy', 1, seed=4)[0]
    t = g.search_tree(True, 5)
    assert t.viewer == g.to_move
    seen = {True: 0, False: 0}
    for _ in range(64):
        r = t.simulate(g, 1.0, root_view=True)
        if r is None:
            continue
        p, x, acts, vx = r
        assert (vx is None) == (p == t.viewer)
        seen[vx is None] += 1
        pri, v, m = ev(x, acts, 2)
        if vx is not None:
            assert vx.shape == x.shape and not np.array_equal(vx, x)
            _, v, m = ev(vx, [], 2)
        t.expand(pri, v, m)
    assert seen[True] > 0 and seen[False] > 0, seen


# ------------------------------------------------------------------ the drivers

def _rows_checked(ev):
    """`ev` that also checks a root-view batch's layout: a value-only row (no options) follows its
    leaf's row - the same game, another seat (the tree's viewer)."""
    def f(lb):
        off, a = lb.opt_off, lb.a
        cnt = np.diff(off)
        for b in np.where(cnt == 0)[0]:
            assert b > 0 and cnt[b - 1] >= 2 and a['gid'][b] == a['gid'][b - 1] and a['player'][b] != a['player'][b - 1]
        f.view_rows += int((cnt == 0).sum())
        return ev(lb)
    f.view_rows = 0
    return f


def _selfplay(impl, view, share, target='next_q', games=16, sims=12, seed=5):
    """Toy self-play (every search full, so every decision is recorded) with the deterministic `_Rand`
    network: (result rows by gid, the samples of each game in order, value-only rows sent)."""
    ev = _rows_checked(_Rand(3))
    if impl == 'rust-driver':
        seat = G.nn_seat('m', sims, selfplay=True, sims_fast=None, p_full=1.0, leaf_value_view=view, view_samples=share,
                         view_sample_target=target)
        drv = G.Driver('toy', [[seat, seat]] * games, concurrency=games, seed=seed, record=True, threads=2)
    else:
        seat = NNSeat('m', sims, selfplay=True, impl=impl, leaf_value_view=view, view_samples=share,
                      view_sample_target=target)
        drv = PyDriver('toy', games, lambda g: [seat, seat], concurrency=games, seed=seed, record=True)
    parts = []
    while True:
        lb = drv.next_batch()
        if lb is None:
            break
        drv.submit(lb.id, *ev(lb))
        parts.append(drv.take_samples())
    parts.append(drv.take_samples())
    rows = sorted(drv.take_results(), key=lambda r: r['gid'])
    s = concat_samples([p for p in parts if p is not None and len(p['player'])])
    validate_samples(s)
    return rows, s, ev.view_rows


@pytest.mark.parametrize('impl', ['py', 'rs', 'rust-driver'])
def test_value_only_samples_and_their_targets(impl):
    """Value-only samples with every decision (2 players): one per decision record, right after it,
    from the other seat's view of the same state, no options, z in that seat's perspective; their q
    ('next_q') is that seat's next search's root value - its next decision record's q, the same
    search - or z if it never searches again; with 'z' q = z. The decisions' samples are those of
    the run without value-only samples, and the games the same. The Python driver's 'py' and 'rs'
    seats and the Rust driver alike."""
    for view in ('mover', 'root'):
        rows0, s0, _ = _selfplay(impl, view, 0.0)
        rows, s, vrows = _selfplay(impl, view, 1.0)
        assert [r['scores'] for r in rows] == [r['scores'] for r in rows0]
        assert (vrows > 0) == (view == 'root')
        n = np.diff(s['opt_off'])
        dec = n > 0
        assert dec.sum() == len(s0['player']) and (~dec).sum() == dec.sum()
        assert np.array_equal(s['x'][dec], s0['x']) and np.array_equal(s['pi'], s0['pi'])
        assert np.array_equal(s['q'][dec], s0['q']) and np.array_equal(s['z'][dec], s0['z'])
        # (a game's samples are consecutive; the observer's next decision record of the same game, if
        # any, is its next record after the value-only one - a later game's never has the same q)
        nexts = 0
        idx = np.arange(len(n))
        for i in idx[~dec]:
            d = i - 1
            assert dec[d] and s['player'][i] == 1 - s['player'][d]
            assert np.allclose(s['z'][i, :2], s['z'][d, :2][::-1])
            assert not np.array_equal(s['x'][i], s['x'][d])
            later = [j for j in idx[i + 1:] if dec[j] and s['player'][j] == s['player'][i]]
            if later and np.array_equal(s['q'][i], s['q'][later[0]]):
                nexts += 1
            elif not np.array_equal(s['q'][i], s['z'][i]):
                raise AssertionError(f'{impl} {view}: a value-only sample whose q is neither its next search nor z')
        assert nexts > 0.8 * (~dec).sum(), (nexts, (~dec).sum())
        _, sz, _ = _selfplay(impl, view, 1.0, 'z')
        vz = np.diff(sz['opt_off']) == 0
        assert np.array_equal(sz['q'][vz], sz['z'][vz])


@pytest.mark.parametrize('impl', ['py', 'rs', 'rust-driver'])
def test_root_view_changes_the_search_and_a_share_of_samples(impl):
    """Root-view values play other games than mover-view ones (the `_Rand` network's value reads the
    viewer's features); a share 0.5 records about half the value-only samples."""
    rows_m, _, _ = _selfplay(impl, 'mover', 0.0, games=24)
    rows_r, _, vr = _selfplay(impl, 'root', 0.0, games=24)
    assert vr > 0 and [r['scores'] for r in rows_m] != [r['scores'] for r in rows_r]
    _, s, _ = _selfplay(impl, 'root', 0.5, games=24)
    n = np.diff(s['opt_off'])
    v, d = int((n == 0).sum()), int((n > 0).sum())
    assert 0.3 * d < v < 0.7 * d, (v, d)


def test_python_and_rust_drivers_record_the_same_value_samples_rule():
    """The decisions that record value-only samples at a share: per game, decision k (its searched
    decisions before it) records them iff `view_pick(seed, gid, k, share)` - the hash the Python
    driver computes with `play.view_pick` (test_view_pick_matches_the_rust_driver)."""
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=None, p_full=1.0, view_samples=0.5)
    drv = G.Driver('toy', [[seat, seat]] * 8, concurrency=1, seed=11, record=True, threads=1, groups=1)
    from zero.infer import UniformEvaluator, drive
    parts = []
    drive(drv, {'m': UniformEvaluator()}, on_batch=lambda lb: parts.append(drv.take_samples()))
    parts.append(drv.take_samples())
    s = concat_samples([p for p in parts if p is not None and len(p['player'])])
    rows = drv.take_results()
    # one game at a time (concurrency 1): games finish in gid order; each game's records in order
    n = np.diff(s['opt_off'])
    a = 0
    for r in sorted(rows, key=lambda r: r['gid']):
        k = r['samples']
        dec_seen, want = 0, []
        got = []
        for i in range(a, a + k):
            if n[i] > 0:
                want.append(view_pick(11, r['gid'], dec_seen, 0.5))
                got.append(i + 1 < a + k and n[i + 1] == 0)
                dec_seen += 1
        assert got == want, r['gid']
        a += k
    assert a == len(n)


# ------------------------------------------------------------------ toy runs

def test_runs_off_selfplay_and_eval(tmp_path):
    """Toy-smoke runs (2 iterations, 'cq' targets):
    * the settings given at their defaults: every shard, checkpoint and Elo file of the run without
      them, byte for byte;
    * self-play's root view with value-only samples: shards with value-only rows (no options), the
      training's value loss on them reported, the evaluations' pool unchanged (elo/);
    * the evaluation's root view: every shard and checkpoint byte for byte (training untouched), the
      evaluations in the pool elo/visits-rv/ (zerodash's too)."""
    from zero.data import read_shard
    r = _runs(tmp_path, {
        'base': [],
        'zero': ['--selfplay-leaf-value-view', 'mover', '--eval-leaf-value-view', 'mover', '--selfplay-view-samples', '0',
                 '--view-sample-target', 'next_q'],
        'sp': ['--selfplay-leaf-value-view', 'root', '--selfplay-view-samples', '0.5'],
        'ev': ['--eval-leaf-value-view', 'root'],
    })
    a, b = _training_files(r['base']), _training_files(r['zero'])
    assert len(a) >= 6 and a == b
    for f in ('results.json', 'ratings.json'):
        ja, jb = (json.load(open(r[k] / 'elo' / f)) for k in ('base', 'zero'))
        ja.pop('updated', None), jb.pop('updated', None)
        assert ja == jb, f
    assert all('view' not in m['selfplay'] and 'value_view' not in m['train'] for m in read_metrics(str(r['zero']))
               if 'selfplay' in m)
    # self-play's root view and value-only samples
    sp = _training_files(r['sp'])
    assert sp['models/iter_0000.pt'] == a['models/iter_0000.pt'] and sp != a
    import glob
    shards = sorted(glob.glob(str(r['sp'] / 'data' / '**' / '*.npz'), recursive=True))
    assert shards
    nview = ndec = 0
    for p in shards:
        s = read_shard(p)
        n = np.diff(s['opt_off'])
        nview += int((n == 0).sum())
        ndec += int((n > 0).sum())
    assert 0.25 * ndec < nview < 0.75 * ndec, (nview, ndec)
    ms = [m for m in read_metrics(str(r['sp'])) if 'selfplay' in m]
    assert len(ms) == 2
    assert all(m['selfplay']['view']['rows'] > 0 and m['selfplay']['view']['samples'] > 0 for m in ms)
    assert all(m['train']['window_view_samples'] > 0 and np.isfinite(m['train']['value_view']) for m in ms)
    assert all(np.isfinite(m['train']['policy']) and m['train']['entropy'] > 0 for m in ms)
    assert elo.active_pool(str(r['sp']))[0] == 'visits' and os.path.exists(r['sp'] / 'elo' / 'ratings.json')
    # the evaluation's root view: the training of before, its own pool
    assert _training_files(r['ev']) == a
    assert elo.active_pool(str(r['ev']))[0] == 'visits-rv'
    assert os.path.isdir(r['ev'] / 'elo' / 'visits-rv') and not os.path.exists(r['ev'] / 'elo' / 'ratings.json')
    assert {m['eval_yard']['select'] for m in read_metrics(str(r['ev']))} == {'visits-rv'}
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k in r:
        assert runfiles.elo_pool(str(r[k])) == elo.active_pool(str(r[k]))[1], k
    cfg = json.load(open(r['sp'] / 'config.json'))
    assert (cfg['selfplay_leaf_value_view'], cfg['selfplay_view_samples'], cfg['eval_leaf_value_view']) == ('root', 0.5, 'mover')
