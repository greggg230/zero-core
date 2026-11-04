"""Information-set merging (docs/search.md 3.16, docs/az.md 30): the seat key, the run settings (eval_infoset_merge,
selfplay_infoset_merge) and the Elo pool name, the tree bindings, the drivers (the Rust driver's evaluation and
self-play seats, the Python driver's Rust-tree seats; the reference tree refuses it), h2h's flags, a rolling toy run
that self-plays merged (exact quotas, the `merge` metrics) and remote self-play (the workers merge as this machine; an
older checkout refused by its hello). The toy's other seat never has options that depend on hidden cards (the merge's
subset-armed nodes and growth leaves are tested in Rust on the counter game, `tree::tests` / `driver::tests`); here:
the plumbing."""
import json
import types

import numpy as np
import pytest

from zero import games as G, elo, h2h, mcts, parity
from zero.infer import UniformEvaluator, drive
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, Run, check_eval_settings, read_metrics

from test_loop import run_loop
from test_remoteplay import PAIR, check_quota, pair_env
from test_select import _ckpts


def test_settings_seats_and_names():
    assert DEFAULTS['eval_infoset_merge'] == 0 and DEFAULTS['selfplay_infoset_merge'] == 0
    # off: the seat dict of before, key for key
    base = G.nn_seat('m', 8, leaf_value_view='root')
    assert base == G.nn_seat('m', 8, leaf_value_view='root', infoset_merge=False) and 'infoset_merge' not in base
    s = G.nn_seat('m', 8, leaf_value_view='root', infoset_merge=True)
    assert s == dict(base, infoset_merge=True)
    sp = G.nn_seat('m', 8, selfplay=True, leaf_value_view='root', infoset_merge=True)
    assert sp == dict(G.nn_seat('m', 8, selfplay=True, leaf_value_view='root'), infoset_merge=True)
    for bad in (dict(infoset_merge=True), dict(infoset_merge=True, selfplay=True),
                dict(infoset_merge=2, leaf_value_view='root'), dict(infoset_merge=0.5, leaf_value_view='root')):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
    # the Rust driver takes the key on evaluation and self-play seats, and refuses it without root-view values
    G.Driver('toy', [[s, s]], concurrency=1)
    G.Driver('toy', [[sp, sp]], concurrency=1, record=True)
    for bad in (dict(s, leaf_value_view='mover'), dict(sp, leaf_value_view='mover')):
        with pytest.raises(ValueError):
            G.Driver('toy', [[bad, bad]], concurrency=1)
    # the run settings: validated; the evaluation seats take eval_infoset_merge, self-play's selfplay_infoset_merge
    for bad in ({'eval_infoset_merge': 1}, {'eval_infoset_merge': 2, 'eval_leaf_value_view': 'root'},
                {'selfplay_infoset_merge': 1}, {'selfplay_infoset_merge': 1, 'eval_leaf_value_view': 'root'},
                {'selfplay_infoset_merge': 0.5, 'selfplay_leaf_value_view': 'root'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, eval_infoset_merge=1, eval_leaf_value_view='root'))
    check_eval_settings(dict(DEFAULTS, selfplay_infoset_merge=1, selfplay_leaf_value_view='root'))
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, eval_infoset_merge=1, eval_leaf_value_view='root'))
    assert Run.nn_seat(r, 'm', 32)['infoset_merge'] is True
    assert 'infoset_merge' not in Run.nn_seat(r, 'm', 64, selfplay=True)
    r.cfg = dict(DEFAULTS, eval_leaf_value_view='root')
    assert 'infoset_merge' not in Run.nn_seat(r, 'm', 32)
    r.cfg = dict(DEFAULTS, selfplay_infoset_merge=1, selfplay_leaf_value_view='root')
    assert Run.nn_seat(r, 'm', 64, selfplay=True)['infoset_merge'] is True
    assert 'infoset_merge' not in Run.nn_seat(r, 'm', 32)
    r.cfg = dict(DEFAULTS, selfplay_leaf_value_view='root')
    off = Run.nn_seat(r, 'm', 64, selfplay=True)
    assert 'infoset_merge' not in off
    r.cfg = dict(DEFAULTS, selfplay_leaf_value_view='root', selfplay_infoset_merge=0)
    assert Run.nn_seat(r, 'm', 64, selfplay=True) == off       # (off: the seat of before, key for key)
    # names: the evaluation's merge names its Elo pool ('-ism' last)
    assert mcts.merge_tag(0) == '' and mcts.merge_tag(True) == '-ism'
    cq = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'eval_select_c_scale': 0.3, 'eval_leaf_value_view': 'root',
          'eval_belief_tilt': 1, 'belief_mix': 0.25}
    assert elo.eval_tag(cq) == 'cq-n4-v50-s0.3-rv-b1m0.25'
    assert elo.eval_tag(dict(cq, eval_infoset_merge=1)) == 'cq-n4-v50-s0.3-rv-b1m0.25-ism'
    assert elo.eval_tag(dict(cq, eval_infoset_merge=0)) == elo.eval_tag(cq)
    assert elo.eval_tag({'eval_root_search': 'gumbel', 'eval_leaf_value_view': 'root', 'eval_infoset_merge': 1}).endswith('-rv-ism')
    # the Python reference tree has no merged mode; a Python-driver seat of the Rust tree has
    with pytest.raises(ValueError):
        NNSeat('m', leaf_value_view='root', infoset_merge=True)
    with pytest.raises(ValueError):
        NNSeat('m', impl='rs', infoset_merge=True)
    assert NNSeat('m', impl='rs', leaf_value_view='root', infoset_merge=True).infoset_merge


def test_tree_bindings_merge_and_count():
    ev = parity.LinearEval(G.info('toy')['num_features'], G.info('toy')['num_actions'], seed=1, scale=0.15)
    stats = []
    for k, g in enumerate(parity.sample_positions('toy', 6, seed=4)):
        t = g.search_tree(True, 5 + k)
        assert not t.merge_on and t.merge_stats() is None
        t.set_merge(True)
        assert t.merge_on
        for _ in range(96):
            r = t.simulate(g, 1.0, root_view=True)
            if r is None:
                continue
            p, x, acts, vx = r
            pri, v, m = ev(x, acts, 2)
            assert len(pri) == len(acts)
            if vx is not None:
                _, v, m = ev(vx, [], 2)
            t.expand(pri, v, m)
        st = t.merge_stats()
        assert set(st) == {'links', 'encodes', 'merged', 'aliases', 'growths'}
        assert st['links'] > 0 and st['encodes'] >= st['links'] - st['merged'] and st['growths'] == 0
        stats.append(st)
        with pytest.raises(ValueError):
            t.set_merge(False)                     # (a tree with a search in it)
        assert sum(t.root_visits) == t.root_tot == 95
    assert sum(s['merged'] for s in stats) > 0, stats


def test_drivers_play_with_merged_seats():
    # the Rust driver: a merged evaluation seat against a plain root-view one; its stats count the merged trees
    a = G.nn_seat('a', 12, leaf_value_view='root', infoset_merge=True)
    b = G.nn_seat('b', 12, leaf_value_view='root')
    drv = G.Driver('toy', [[a, b] if g % 2 == 0 else [b, a] for g in range(12)], concurrency=12, seed=3, threads=2)
    drive(drv, {'a': UniformEvaluator(), 'b': UniformEvaluator()})
    st = drv.stats()
    assert len(drv.take_results()) == 12
    assert st['merge_links'] > 0 and st['merge_growths'] == 0
    assert {'merge_merged', 'merge_aliases'} <= set(st)
    # off: no merged tree, nothing counted
    drv = G.Driver('toy', [[b, b]] * 4, concurrency=4, seed=3, threads=1)
    drive(drv, {'b': UniformEvaluator()})
    assert drv.stats()['merge_links'] == 0
    # the Python driver's seats of the Rust tree
    seat = NNSeat('m', 12, impl='rs', leaf_value_view='root', infoset_merge=True)
    pd = PyDriver('toy', 6, lambda g: [seat, seat], concurrency=6, seed=2)
    ev = UniformEvaluator()
    while True:
        lb = pd.next_batch()
        if lb is None:
            break
        pd.submit(lb.id, *ev(lb)[:3])
    assert len(pd.take_results()) == 6


def test_h2h_flags_tags_and_counts(tmp_path, capsys):
    a, _ = _ckpts(tmp_path, {'eval_leaf_value_view': 'root'})
    rule = h2h.run_rule(a)
    assert rule.get('leaf_value_view') == 'root' and 'infoset_merge' not in rule
    capsys.readouterr()
    h2h.main([a, a, '--games', '4', '--sims', '8', '--chance-k', '0', '--threads', '1', '--paired',
              '--a-infoset-merge', '1', '--b-infoset-merge', '0'])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['select_a'] == 'visits-rv-ism' and out['select_b'] == 'visits-rv', out
    assert out['merge']['links'] > 0 and set(out['merge']) == {'links', 'merged', 'aliases', 'growths'}
    # a side without root-view values: refused before any game
    b, _ = _ckpts(tmp_path / 'plain')
    with pytest.raises(SystemExit):
        h2h.main([b, b, '--games', '2', '--sims', '4', '--a-infoset-merge', '1'])
    capsys.readouterr()
    h2h.main([b, b, '--games', '2', '--sims', '4', '--chance-k', '0', '--threads', '1', '--a-leaf-value-view', 'root',
              '--a-infoset-merge', '1'])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['select_a'] == 'visits-rv-ism' and out['select_b'] == 'visits' and 'merge' in out
    # a run that evaluates merged: its side comes merged (and named so) unless told otherwise
    c, _ = _ckpts(tmp_path / 'merged', {'eval_leaf_value_view': 'root', 'eval_infoset_merge': 1})
    assert h2h.run_rule(c)['infoset_merge'] is True
    capsys.readouterr()
    h2h.main([c, c, '--games', '2', '--sims', '4', '--chance-k', '0', '--threads', '1', '--b-infoset-merge', '0'])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (out['select_a'], out['select_b']) == ('visits-rv-ism', 'visits-rv'), out


def _selfplay(seat, games=24, threads=2, seed=4):
    """Toy self-play of `seat` on both sides, answered by a function of each row (deterministic) -> (rows sorted by
    gid, each (result, its samples), stats)."""
    from zero.workqueue import split_games
    drv = G.Driver('toy', [[seat, seat]] * games, concurrency=12, seed=seed, record=True, threads=threads)
    rows = []
    while True:
        lb = drv.next_batch()
        if lb is None:
            break
        n = len(lb.player)
        off = np.asarray(lb.opt_off)
        pri = np.concatenate([np.full(off[b + 1] - off[b], 1.0 / max(1, off[b + 1] - off[b]), np.float32)
                              for b in range(n)]) if n else np.zeros(0, np.float32)
        x = np.asarray(lb.x).reshape(n, -1)
        w = 1.0 / (1.0 + np.exp(-0.1 * (x * ((np.arange(x.shape[1]) * 7 % 13) - 6.0)).sum(axis=1)))
        v = np.zeros((n, 8), np.float32)
        v[:, 0], v[:, 1] = w, 1.0 - w
        drv.submit(lb.id, pri, v, np.zeros((n, 8), np.float32))
        rows.extend(split_games(drv.take_results(), drv.take_samples()))
    rows.extend(split_games(drv.take_results(), drv.take_samples()))
    return sorted(rows, key=lambda rs: rs[0]['gid']), drv.stats()


def test_selfplay_records_with_merged_seats():
    """The bindings' self-play seats (docs/search.md 3.16): merged self-play records are valid (pi a distribution over
    the options, q a win vector, z the result from the decider's view; value-only samples) and the same across
    thread counts; the merged trees link, merge and share the leaf's encoding; off, nothing is counted."""
    seat = G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=6, leaf_value_view='root',
                     view_samples=0.5, infoset_merge=True)
    rows, st = _selfplay(seat)
    assert len(rows) == 24 and all(r['over'] for r, _ in rows)
    dec = val = 0
    for r, smp in rows:
        assert smp is not None and len(smp['player']) == r['samples']
        off, pi, z, q = smp['opt_off'], smp['pi'], smp['z'], smp['q']
        n = r['n']
        for k in range(len(smp['player'])):
            assert np.allclose(z[k][:n], [r['result'][(int(smp['player'][k]) + j) % n] for j in range(n)])
            assert np.all(q[k][:n] >= -1e-6) and np.all(q[k][:n] <= 1 + 1e-6) and abs(q[k][:n].sum() - 1) < 1e-4
            p = pi[off[k]:off[k + 1]]
            if len(p):
                assert len(p) >= 2 and np.all(p >= 0) and abs(p.sum() - 1) < 1e-4
                dec += 1
            else:
                val += 1
    assert dec > 0 and val > 0
    assert st['merge_links'] > 0 and st['merge_merged'] > 0 and st['merge_shared'] > 0, st
    again, st1 = _selfplay(seat, threads=1)
    assert [(r['gid'], r['steps'], r['result'], r['samples']) for r, _ in again] == \
        [(r['gid'], r['steps'], r['result'], r['samples']) for r, _ in rows]
    for (_, a), (_, b) in zip(again, rows):
        assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ('x', 'z', 'q', 'pi', 'player'))
    assert (st1['merge_links'], st1['merge_shared']) == (st['merge_links'], st['merge_shared'])
    _, st0 = _selfplay(G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=6,
                                 leaf_value_view='root', view_samples=0.5))
    assert (st0['merge_links'], st0['merge_shared']) == (0, 0)


def check_merge_metrics(m):
    for r in m[1:]:
        mg = r['selfplay']['merge']
        assert set(mg) == {'links', 'merged', 'aliases', 'growths', 'shared', 'merged_per_leaf'}, mg
        assert mg['links'] > 0 and mg['merged'] > 0 and mg['shared'] > 0, mg


def test_a_rolling_run_selfplays_merged_with_exact_quotas(tmp_path):
    """A rolling toy run with selfplay_infoset_merge (and resignation, value-only samples): every iteration exactly its
    quota, the `merge` metrics; refused without root-view values; off by default (no metrics)."""
    run = tmp_path / 'ism'
    p = run_loop(run, '--iters', '3', '--rolling', '1', '--games', '96', '--conc', '32', '--selfplay-infoset-merge', '1',
                 '--selfplay-leaf-value-view', 'root', '--selfplay-view-samples', '0.5', '--selfplay-resign-t', '0.3',
                 '--selfplay-resign-k', '2')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    for r in m[1:]:
        assert r['selfplay']['games'] == 96, r['selfplay']['games']
        assert r['selfplay']['samples'] > 0 and r['selfplay']['view']['samples'] > 0
    check_merge_metrics(m)
    assert 'WARNING' not in (run / 'log.txt').read_text(encoding='utf-8')
    assert json.loads((run / 'config.json').read_text(encoding='utf-8'))['selfplay_infoset_merge'] == 1
    p = run_loop(tmp_path / 'bad', '--iters', '1', '--selfplay-infoset-merge', '1')
    assert p.returncode != 0 and 'selfplay_infoset_merge' in p.stderr, p.stderr[-1000:]
    run0 = tmp_path / 'off'
    p = run_loop(run0, '--iters', '1', '--rolling', '1')
    assert p.returncode == 0, p.stderr[-2000:]
    assert all('merge' not in r['selfplay'] for r in read_metrics(str(run0))[1:])


def test_remote_workers_merge_and_the_quotas_stay_exact(tmp_path):
    """Remote self-play (the remote 'local') with the local linger: the workers merge as this machine (their hello
    agrees: no refusal), every iteration exactly its quota."""
    run = tmp_path / 'rm'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--selfplay-local-linger', '1',
                 '--selfplay-infoset-merge', '1', '--selfplay-leaf-value-view', 'root', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 192)
    assert any(r['selfplay']['remote']['accepted'] > 0 for r in m[1:]), [r['selfplay']['remote'] for r in m[1:]]
    check_merge_metrics(m)
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')


def test_an_older_worker_is_refused(tmp_path):
    """A worker whose checkout does not report merging (the tests' no_merge: an older checkout, which would search
    its games unmerged in silence) is refused and killed; this machine plays the quota."""
    run = tmp_path / 'old'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--selfplay-infoset-merge', '1',
                 '--selfplay-leaf-value-view', 'root', env=pair_env(tmp_path, {'no_merge': 1}))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and 'information-set merging' in log, log[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 192)
    assert all(r['selfplay']['remote']['accepted'] == 0 for r in m[1:]), [r['selfplay']['remote'] for r in m[1:]]
