"""Per-target tilts of the root's belief (docs/search.md 3.8.1, docs/az.md 38): run settings
selfplay_belief_tilts / eval_belief_tilts ({target: alpha}), the seat key belief_tilts, a seat taking the
root's belief with belief_tilt 0, the game reading a target's own tilt (`Belief::tilts`; the toy's
`next_card`), the refusals, the Elo pool's name, zero.h2h's per-side flag, the trees' API - and with none
set, the seat dicts, tags and runs of before."""
import json
import os
import types

import numpy as np
import pytest

from zero import elo, games as G, h2h, mcts, parity
from zero.infer import NetEvaluator
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, Run, belief_keys, check_eval_settings

from test_aux import TOY_AUX, nets
from test_belief import _checked


def test_settings_seats_names_and_refusals(tmp_path):
    assert (DEFAULTS['selfplay_belief_tilts'], DEFAULTS['eval_belief_tilts']) == ({}, {})
    # none set: the seat dicts and names of before
    assert belief_keys(dict(DEFAULTS), 'selfplay') == {} and belief_keys(dict(DEFAULTS), 'eval') == {}
    assert G.nn_seat('m', 8) == G.nn_seat('m', 8, belief_tilts={}) == G.nn_seat('m', 8, belief_tilts={'next_card': 0})
    assert mcts.belief_tag(1.0) == mcts.belief_tag(1.0, tilts={}) == '-b1'
    # a per-target tilt alone: the belief's keys, its tilt 0
    s = G.nn_seat('m', 8, belief_tilts={'next_card': 1.0})
    assert (s['belief_tilt'], s['belief_clip'], s['belief_mix'], s['belief_fast'], s['belief_tilts']) == \
        (0.0, 20.0, 0.0, False, {'next_card': 1.0})
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, selfplay_belief_tilts={'next_card': 0.5}, belief_mix=0.25))
    sp = Run.nn_seat(r, 'm', 64, selfplay=True)
    assert (sp['belief_tilt'], sp['belief_mix'], sp['belief_tilts']) == (0.0, 0.25, {'next_card': 0.5})
    assert not any(k.startswith('belief_') for k in Run.nn_seat(r, 'm', 32))
    # names: the evaluation's per-target tilts name its Elo pool
    assert mcts.belief_tag(0, tilts={'next_card': 1}) == '-b0-tnext_card1'
    assert mcts.belief_tag(1, 10.0, 0.25, False, {'b': 0.5, 'a': 0}) == '-b1c10m0.25-tb0.5'
    for c, tag in (({'eval_belief_tilts': {'next_card': 1.0}}, 'visits-b0-tnext_card1'),
                   ({'eval_belief_tilt': 1.0, 'eval_belief_tilts': {'next_card': 0.5}}, 'visits-b1-tnext_card0.5'),
                   ({'selfplay_belief_tilts': {'next_card': 1.0}}, 'visits'), ({'eval_belief_tilts': {}}, 'visits')):
        assert elo.eval_tag(c) == tag, c
    # refusals: Python (a dict of finite tilts >= 0), the Rust driver (a target of the game, once)
    for bad in ({'next_card': -1.0}, {'next_card': float('nan')}, {'next_card': True}, [('next_card', 1.0)]):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, belief_tilts=bad)
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, eval_belief_tilts=bad))
    G.Driver('toy', [[s, s]], concurrency=1)
    with pytest.raises(ValueError, match='not an auxiliary target'):
        G.Driver('toy', [[dict(s, belief_tilts={'deck': 1.0})] * 2], concurrency=1)
    # h2h: a side's per-target tilts from its run's config, the flag overriding them
    d = tmp_path / 'run'
    os.makedirs(d / 'models')
    json.dump(dict(DEFAULTS, eval_belief_tilts={'next_card': 1.0}), open(d / 'config.json', 'w'))
    ckpt = str(d / 'models' / 'iter_0001.pt')
    rule = h2h.run_rule(ckpt)
    assert rule['belief_tilts'] == {'next_card': 1.0} and rule['belief_tilt'] == 0.0
    assert h2h.with_belief(rule, tilts={}) == {k: v for k, v in rule.items() if not k.startswith('belief_')}
    assert h2h.with_belief({}, None, None, None, None, {'next_card': 2})['belief_tilts'] == {'next_card': 2.0}
    assert h2h.with_belief(rule, 0)['belief_tilts'] == {'next_card': 1.0}, 'tilt 0 keeps the per-target tilts'


def test_a_target_tilt_takes_the_belief_and_deals_as_belief_tilt_does():
    """A seat with belief_tilt 0 and `next_card` 1: one root belief row per full search, answered by the
    head (the Rust driver, the Python driver with both trees); the toy deals by `next_card`'s tilt as by
    belief_tilt (`Game.determinize_with(..., tilts=)` at the same odds, draw for draw)."""
    info = G.info('toy')
    ev = NetEvaluator(nets(TOY_AUX)['mlp'])
    kw = dict(belief_tilt=0.0, belief_tilts={'next_card': 1.0})
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5, **kw)
    drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=16, seed=3, record=True, threads=2)
    seen = _checked(drv, ev, info)
    n = len(drv.take_samples()['player'])
    assert drv.stats()['belief_rows'] == seen['rows'] == n > 0
    for impl in ('py', 'rs'):
        st = NNSeat('m', sims=8, selfplay=True, sims_fast=4, p_full=0.5, impl=impl, **kw)
        assert st.takes_belief(True) and not st.takes_belief(False)
        drv = PyDriver('toy', 16, lambda gid: [st, st], concurrency=16, seed=3, record=True)
        seen = _checked(drv, ev, info)
        assert drv.belief_rows == seen['rows'] == len(drv.take_samples()['player']) > 0, impl
    for g, pred, c, cards in parity.belief_positions(3, 0):
        v = g.to_move
        for s in range(50):
            a, b = g.clone(), g.clone()
            a.determinize_with(v, 7 + s, pred, 1.0, 20.0)
            b.determinize_with(v, 7 + s, pred, 0.0, 20.0, {'next_card': 1.0})
            assert np.array_equal(np.asarray(a.aux_target(v)), np.asarray(b.aux_target(v)))
            assert a.export_state() == b.export_state()
        with pytest.raises(ValueError):
            g.clone().determinize_with(v, 1, pred, 0.0, 20.0, {'next_card': 0.0})
        with pytest.raises(ValueError):
            g.clone().determinize_with(v, 1, pred, 1.0, 20.0, {'deck': 1.0})
    # the trees' API
    g, pred, _, _ = parity.belief_positions(1, 0)[0]
    t = g.search_tree(True, 5)
    t.set_belief(pred, g, 0.0, 20.0, 0.0, {'next_card': 1.0})
    assert np.allclose(t.belief, pred)
