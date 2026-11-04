"""Information-set merging (docs/search.md 3.16, docs/az.md 30): the seat key, the run setting and its Elo pool name,
the tree bindings, the drivers (the Rust driver, the Python driver's Rust-tree seats; the reference tree refuses it)
and h2h's flags. The toy's other seat never has options that depend on hidden cards (the merge's subset-armed nodes
and growth leaves are tested in Rust on the counter game, `tree::tests` / `driver::tests`); here: the plumbing."""
import json
import types

import pytest

from zero import elo, games as G, h2h, mcts, parity
from zero.infer import UniformEvaluator, drive
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, Run, check_eval_settings

from test_select import _ckpts


def test_settings_seats_and_names():
    assert DEFAULTS['eval_infoset_merge'] == 0
    # off: the seat dict of before, key for key
    base = G.nn_seat('m', 8, leaf_value_view='root')
    assert base == G.nn_seat('m', 8, leaf_value_view='root', infoset_merge=False) and 'infoset_merge' not in base
    s = G.nn_seat('m', 8, leaf_value_view='root', infoset_merge=True)
    assert s == dict(base, infoset_merge=True)
    for bad in (dict(infoset_merge=True), dict(infoset_merge=True, leaf_value_view='root', selfplay=True),
                dict(infoset_merge=2, leaf_value_view='root'), dict(infoset_merge=0.5, leaf_value_view='root')):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
    # the Rust driver takes the key, and refuses it without root-view values or on a self-play seat
    G.Driver('toy', [[s, s]], concurrency=1)
    for bad in (dict(s, leaf_value_view='mover'), dict(s, selfplay=True)):
        with pytest.raises(ValueError):
            G.Driver('toy', [[bad, bad]], concurrency=1)
    # the run setting: validated; the evaluation seats take it, self-play's never
    for bad in ({'eval_infoset_merge': 1}, {'eval_infoset_merge': 2, 'eval_leaf_value_view': 'root'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, eval_infoset_merge=1, eval_leaf_value_view='root'))
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, eval_infoset_merge=1, eval_leaf_value_view='root'))
    assert Run.nn_seat(r, 'm', 32)['infoset_merge'] is True
    assert 'infoset_merge' not in Run.nn_seat(r, 'm', 64, selfplay=True)
    r.cfg = dict(DEFAULTS, eval_leaf_value_view='root')
    assert 'infoset_merge' not in Run.nn_seat(r, 'm', 32)
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
