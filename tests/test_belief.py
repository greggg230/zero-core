"""The root's belief (docs/search.md 3.8, docs/az.md 18): run settings selfplay_belief_tilt /
eval_belief_tilt / belief_clip / belief_mix / belief_fast, seat keys belief_tilt / belief_clip /
belief_mix / belief_fast, the leaf-batch protocol's root belief rows (ROW_BELIEF) and the answer's
`aux`, `NetEvaluator` answering them from the auxiliary head (a network without one: HeadMissing,
never a uniform deal in silence), the Python reference tree and driver against the Rust ones, the
toy's belief-weighted deal (its next round's card), the Elo pool's name, zero.h2h's per-side flags,
remote self-play refusing a checkout without it, toy runs: off = the runs of before byte for byte;
self-play and evaluation on; a run without the head refused."""
import json
import os
import random
import types

import numpy as np
import pytest
import torch

from zero import elo, games as G, h2h, mcts, parity
from zero.infer import NetEvaluator, UniformEvaluator, drive
from zero.net import HeadMissing, check_belief_head, make_net, save
from zero.play import NNSeat, PyDriver
from zero.protocol import ROW_BELIEF, ROW_LEAF, belief_rows, validate_batch, validate_evaluation
from zero.run import DEFAULTS, Run, check_eval_settings, read_metrics

from test_aux import TOY_AUX, nets
from test_loop import run_loop
from test_remoteplay import env_for
from test_select_scale import _runs, _training_files


# ------------------------------------------------------------------ settings, seats, names

def test_settings_seats_and_names(tmp_path):
    assert (DEFAULTS['selfplay_belief_tilt'], DEFAULTS['eval_belief_tilt'], DEFAULTS['belief_clip'],
            DEFAULTS['belief_mix'], DEFAULTS['belief_fast']) == (0.0, 0.0, 20.0, 0.0, 0)
    assert G.engine().BELIEF_CLIP == mcts.BELIEF_CLIP == 20.0
    # off: the seat dict of before, key for key
    base = G.nn_seat('m', 8)
    assert base == G.nn_seat('m', 8, belief_tilt=0.0, belief_clip=3.0, belief_mix=0.5, belief_fast=True)
    assert not any(k.startswith('belief_') for k in base)
    s = G.nn_seat('m', 8, belief_tilt=0.5, belief_mix=0.25)
    assert (s['belief_tilt'], s['belief_clip'], s['belief_mix'], s['belief_fast']) == (0.5, 20.0, 0.25, False)
    for bad in (dict(belief_tilt=-1.0), dict(belief_tilt=float('nan')), dict(belief_tilt=1.0, belief_clip=0.5),
                dict(belief_tilt=1.0, belief_mix=1.0), dict(belief_tilt=1.0, belief_mix=-0.1), dict(belief_tilt=True)):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
        with pytest.raises(ValueError):
            NNSeat('m', **bad)
    # the Rust driver takes the keys (and refuses bad ones)
    G.Driver('toy', [[s, s]], concurrency=1)
    for bad in (dict(s, belief_clip=0.5), dict(s, belief_mix=1.0), dict(s, belief_tilt=-2.0)):
        with pytest.raises(ValueError):
            G.Driver('toy', [[bad, bad]], concurrency=1)
    # run settings: validated
    for bad in ({'selfplay_belief_tilt': -1}, {'eval_belief_tilt': 1, 'belief_mix': 1.0},
                {'selfplay_belief_tilt': 1, 'belief_clip': 0.2}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, selfplay_belief_tilt=1.0, eval_belief_tilt=0.5, belief_mix=0.25, belief_fast=1))
    # Run.nn_seat: self-play's tilt on the self-play seat, the evaluation's on every evaluation seat; the
    # clip, mix and fast shared
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, selfplay_belief_tilt=1.0, belief_mix=0.25, belief_clip=10.0))
    sp = Run.nn_seat(r, 'm', 64, selfplay=True)
    assert (sp['belief_tilt'], sp['belief_clip'], sp['belief_mix'], sp['belief_fast']) == (1.0, 10.0, 0.25, False)
    assert not any(k.startswith('belief_') for k in Run.nn_seat(r, 'm', 32))
    r.cfg = dict(DEFAULTS, eval_belief_tilt=0.5)
    assert not any(k.startswith('belief_') for k in Run.nn_seat(r, 'm', 64, selfplay=True))
    assert Run.nn_seat(r, 'm', 32)['belief_tilt'] == 0.5
    # names: the evaluation's belief names its Elo pool; self-play's never does
    assert mcts.belief_tag() == '' and mcts.belief_tag(1.0) == '-b1'
    assert mcts.belief_tag(0.5, 10.0, 0.25, True) == '-b0.5c10m0.25f' and mcts.belief_tag(1, 20.0) == '-b1'
    cq = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'eval_select_c_scale': 0.3}
    cases = [({}, 'visits'), ({'selfplay_belief_tilt': 1.0, 'belief_mix': 0.25}, 'visits'),
             ({'eval_belief_tilt': 1.0}, 'visits-b1'), ({'eval_belief_tilt': 1.0, 'belief_mix': 0.25}, 'visits-b1m0.25'),
             (dict(cq, eval_belief_tilt=0.5, eval_leaf_value_view='root'), 'cq-n4-v50-s0.3-rv-b0.5'),
             ({'eval_belief_tilt': 0.0, 'belief_clip': 5.0}, 'visits')]
    for k, (c, tag) in enumerate(cases):
        assert elo.eval_tag(c) == tag, c
        d = tmp_path / f'run{k}'
        os.makedirs(d / 'models')
        json.dump(c, open(d / 'config.json', 'w'))
        assert elo.active_pool(str(d)) == (tag, elo.pool_dir(str(d), tag))


# ------------------------------------------------------------------ the toy's weighted deal

def _positions(n=6, seed=0):
    return parity.belief_positions(n, seed)


def test_game_deal_follows_the_belief_and_keeps_the_rest():
    """`Game.determinize_with` (the toy's next round's card): the believed card dealt at the deal's odds
    (`belief_weight`), a flat belief = the uniform deal's odds; the encoding of the decider unchanged;
    bad beliefs refused."""
    E = G.engine()
    for g, pred, c, cards in _positions(4):
        v = g.to_move
        x0 = np.asarray(g.encode(v))
        k = len(cards)
        for p, tilt, clip in ((pred, 1.0, 20.0), (pred, 0.5, 20.0), (pred, 1.0, 2.0), (np.full(6, 1 / 6, np.float32), 1.0, 20.0)):
            hits = 0
            for s in range(400):
                w = g.clone()
                w.determinize_with(v, 100 + s, p, tilt, clip)
                assert np.array_equal(np.asarray(w.encode(v)), x0), 'the decider sees the same'
                assert w.legal() == g.legal()
                hits += int(np.argmax(w.aux_target(v))) == c
            wt = {d: E.belief_weight(float(p[d]), 1.0 / k, tilt, clip) for d in cards}
            want = wt[c] / sum(wt.values())
            assert abs(hits / 400 - want) < 0.07, (tilt, clip, hits / 400, want)
    g = _positions(1)[0][0]
    for bad in (dict(belief=np.ones(5, np.float32)), dict(belief=np.full(6, np.nan, np.float32)),
                dict(belief=np.ones(6, np.float32), tilt=0.0), dict(belief=np.ones(6, np.float32), clip=0.5)):
        with pytest.raises(ValueError):
            g.clone().determinize_with(g.to_move, 1, **bad)


def test_rust_and_python_trees_deal_by_the_same_belief():
    """The Rust and Python trees with the same belief: mean visits, completed-Q targets and root win at the
    noise floor of two Python searches; the believed card's share of the leaves' worlds the deal's; the
    belief changes the search (vs the uniform deal: a power check)."""
    r = parity.belief(n=4, sims=48, reps=12)
    assert r['tv_rs_py'] < 2.5 * r['tv_py_py'] + 0.01, r
    assert r['tv_cq_rs_py'] < 2.5 * r['tv_cq_py_py'] + 0.02, r
    assert r['win_diff'] < 0.01, r
    assert abs(r['share'] - r['share_want']) < 0.05, r
    assert r['tv_belief'] > 3 * r['tv_py_py'], r
    # the trees' API: None = uniform, a belief set and cleared by advance
    g, pred, c, cards = _positions(1)[0]
    t = g.search_tree(True, 5)
    assert t.belief is None
    t.set_belief(pred, g, 1.0, 20.0, 0.25)
    assert np.allclose(t.belief, pred)
    with pytest.raises(ValueError):
        t.set_belief(pred, g, 1.0, 20.0, 1.0)
    t.set_belief(None, g)
    assert t.belief is None


# ------------------------------------------------------------------ the drivers and the evaluator

def _belief_seat(selfplay=True, fast=False, mix=0.0, impl=None, sims=8):
    kw = dict(belief_tilt=1.0, belief_mix=mix, belief_fast=fast)
    if impl is None:
        return G.nn_seat('m', sims, selfplay=selfplay, sims_fast=sims // 2 if selfplay else None,
                         p_full=0.5 if selfplay else 1.0, **kw)
    return NNSeat('m', sims=sims, selfplay=selfplay, sims_fast=sims // 2 if selfplay else None,
                  p_full=0.5 if selfplay else 1.0, impl=impl, **kw)


def _checked(drv, ev, info):
    """`drive` with every batch and answer validated -> (batches, belief rows seen)."""
    seen = {'b': 0, 'rows': 0}

    def answer(lb):
        validate_batch(lb, info['num_features'], info['num_actions'])
        out = ev(lb)
        validate_evaluation(lb, *out[:3], out[3] if len(out) > 3 else None, info['aux_width'])
        seen['b'] += 1
        seen['rows'] += len(belief_rows(lb.a.get('extra')))
        return out
    drive(drv, {'m': answer})
    return seen


@pytest.mark.parametrize('driver', ['rust', 'py', 'rs'])
def test_one_belief_row_per_full_search_answered_by_the_head(driver):
    """Self-play (p_full 0.5) with a network with the head: as many root belief rows as full searches
    (recorded decisions) - after a fresh root's own leaf row or alone -, every batch valid, its belief rows
    answered by the head (`NetEvaluator`'s fourth item); the Rust driver, the Python driver with the Python
    tree and with the Rust tree."""
    info = G.info('toy')
    m = nets(TOY_AUX)['mlp']
    ev = NetEvaluator(m)
    if driver == 'rust':
        seat = _belief_seat()
        drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=16, seed=3, record=True, threads=2)
    else:
        seat = _belief_seat(impl=driver)
        drv = PyDriver('toy', 16, lambda gid: [seat, seat], concurrency=16, seed=3, record=True)
    seen = _checked(drv, ev, info)
    smp = drv.take_samples()
    n = len(smp['player'])
    rows = drv.stats()['belief_rows'] if driver == 'rust' else drv.belief_rows
    assert rows == seen['rows'] == n > 0, (rows, seen, n)


def test_a_network_without_the_head_or_no_aux_answer_fails_loudly():
    seat = _belief_seat(selfplay=False)
    ev = NetEvaluator(nets()['mlp'])
    with pytest.raises(HeadMissing):
        drive(G.Driver('toy', [[seat, seat]] * 2, concurrency=2, threads=1), {'m': ev})
    with pytest.raises(ValueError, match='root belief rows'):
        drive(G.Driver('toy', [[seat, seat]] * 2, concurrency=2, threads=1), {'m': UniformEvaluator()})
    with pytest.raises(ValueError, match='root belief rows'):
        drive(PyDriver('toy', 2, lambda gid: [NNSeat('m', sims=8, belief_tilt=1.0)] * 2, concurrency=2),
              {'m': UniformEvaluator()})
    with pytest.raises(HeadMissing):
        check_belief_head(nets()['mlp'], 'toy')
    check_belief_head(nets(TOY_AUX)['tokens'], 'toy')
    with pytest.raises(HeadMissing):
        check_belief_head(nets([['other', 6, 'multinomial']])['mlp'], 'toy')
    # a batch without belief rows: the network's answers of before (no fourth item)
    x = np.stack([np.asarray(g.encode(g.to_move), np.float32) for g, *_ in _positions(3)])
    off = np.arange(len(x) + 1, dtype=np.int32) * 2
    acts = np.tile(np.array([0, 3], np.int32), len(x))
    ev1 = NetEvaluator(nets(TOY_AUX)['mlp'])
    assert len(ev1.evaluate(x, np.full(len(x), 2), off, acts)) == 3
    extra = np.zeros(len(x), np.uint8)
    extra[1] = ROW_BELIEF
    off2 = off.copy()
    off2[2:] -= 2                                  # (row 1 without options)
    r = ev1.evaluate(x, np.full(len(x), 2), off2, np.delete(acts, [2, 3]), extra)
    assert len(r) == 4 and r[3].shape == (1, 6) and np.isclose(r[3].sum(), 1.0)
    assert np.allclose(r[3][0], ev1.aux(x[1:2], np.full(1, 2))['next_card'][0], atol=1e-6), 'the head of that row'


def test_the_python_trees_mix_deals_a_share_uniformly():
    """The Python reference tree with a belief sharp on one card of the toy's next round: the believed card's
    share of the leaves' worlds (in the root's round) is the deal's odds w_c / sum w, and halfway to the
    uniform 1/k with belief_mix 0.5 (half the worlds dealt uniformly)."""
    info = G.info('toy')
    ev = parity.LinearEval(info['num_features'], info['num_actions'], seed=1, scale=0.15)
    E = G.engine()
    for g, pred, c, cards in _positions(3, seed=1):
        k = len(cards)
        w = {d: E.belief_weight(float(pred[d]), 1.0 / k, 1.0, 1000.0) for d in cards}
        deal = w[c] / sum(w.values())
        assert deal > 0.9
        for mix in (0.0, 0.5):
            worlds = []
            for r in range(40):
                parity.search_py(g, ev, 48, 500 + r, belief=(pred, 1.0, 1000.0, mix), worlds=worlds)
            share = float(np.mean([int(np.argmax(x)) == c for x in worlds]))
            want = (1 - mix) * deal + mix / k
            se = (want * (1 - want) / len(worlds)) ** 0.5
            assert abs(share - want) < 4 * se + 0.01, (mix, share, want, len(worlds))


# ------------------------------------------------------------------ zero.h2h

def test_h2h_sides_and_flags(tmp_path, capsys):
    info = G.info('toy')
    torch.manual_seed(1)
    with_head = make_net(info, 'toy', 'mlp', 16, 1, aux=TOY_AUX)
    torch.manual_seed(1)
    plain = make_net(info, 'toy', 'mlp', 16, 1)
    pa, pb = str(tmp_path / 'a.pt'), str(tmp_path / 'b.pt')
    save(with_head, pa, {})
    save(plain, pb, {})
    on = h2h.with_belief({}, 1.0)
    assert on == {'belief_tilt': 1.0, 'belief_clip': 20.0, 'belief_mix': 0.0, 'belief_fast': False}
    assert h2h.with_belief(on, 0) == {} and h2h.with_belief(on, None, 10.0, 0.25, 1)['belief_mix'] == 0.25
    with pytest.raises(ValueError):
        h2h.with_belief({}, None, 10.0)
    with pytest.raises(ValueError):
        h2h.with_belief({}, 1.0, None, 1.0)
    r = h2h.match(pa, pa, games=8, sims=8, seat_a=on, seat_b={}, conc=8)
    assert r['games'] == 8 and r['select_a'] == 'visits-b1' and r['select_b'] == 'visits'
    with pytest.raises(HeadMissing):
        h2h.match(pa, pb, games=2, sims=8, seat_a={}, seat_b=on)
    # a run's evaluation belief comes with its checkpoint (config.json next to models/)
    run = tmp_path / 'r'
    os.makedirs(run / 'models')
    save(with_head, str(run / 'models' / 'iter_0001.pt'), {})
    json.dump({'eval_belief_tilt': 0.5, 'belief_mix': 0.25}, open(run / 'config.json', 'w'))
    rr = h2h.run_rule(str(run / 'models' / 'iter_0001.pt'))
    assert (rr['belief_tilt'], rr['belief_mix'], rr['belief_clip']) == (0.5, 0.25, 20.0)
    # the command line: each side as its run evaluates - also with --select (the A/B driver's h2h) -, the
    # side's flags over it
    ck = str(run / 'models' / 'iter_0001.pt')
    capsys.readouterr()
    h2h.main([ck, ck, '--games', '2', '--sims', '4', '--chance-k', '0', '--select', 'cq', '--select-min-visits', '4',
              '--b-belief-tilt', '0'])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['select_a'] == 'cq-n4-b0.5m0.25' and out['select_b'] == 'cq-n4', out


# ------------------------------------------------------------------ toy runs

def test_runs_off_selfplay_eval_and_refused(tmp_path):
    """Toy-smoke runs (2 iterations, 'cq' targets):
    * the settings given at their defaults: every shard, checkpoint and Elo file of the run without them,
      byte for byte;
    * self-play's belief (the toy's head, aux_next_card_w 1): belief rows reported, the training's files
      differ from the head-only run's (other searches), the evaluations' pool unchanged;
    * the evaluation's belief: the head-only run's shards and checkpoints byte for byte (training
      untouched), the evaluations in the pool elo/visits-b1/;
    * a belief without the head: refused (exit 78, HeadMissing), nothing played."""
    r = _runs(tmp_path, {
        'base': [],
        'zero': ['--selfplay-belief-tilt', '0', '--eval-belief-tilt', '0', '--belief-clip', '20', '--belief-mix', '0',
                 '--belief-fast', '0'],
        'aux': ['--aux-next-card-w', '1'],
        'sp': ['--aux-next-card-w', '1', '--selfplay-belief-tilt', '1', '--belief-mix', '0.25'],
        'ev': ['--aux-next-card-w', '1', '--eval-belief-tilt', '1'],
    })
    a, b = _training_files(r['base']), _training_files(r['zero'])
    assert len(a) >= 6 and a == b
    for f in ('results.json', 'ratings.json'):
        ja, jb = (json.load(open(r[k] / 'elo' / f)) for k in ('base', 'zero'))
        ja.pop('updated', None), jb.pop('updated', None)
        assert ja == jb, f
    assert all('belief' not in m['selfplay'] for m in read_metrics(str(r['zero'])) if 'selfplay' in m)
    ax, sp = _training_files(r['aux']), _training_files(r['sp'])
    assert sp['models/iter_0000.pt'] == ax['models/iter_0000.pt'] and sp != ax
    ms = [m for m in read_metrics(str(r['sp'])) if 'selfplay' in m]
    assert len(ms) == 2 and all(0 < m['selfplay']['belief']['per_decision'] <= 1 for m in ms)
    assert elo.active_pool(str(r['sp']))[0] == 'visits'
    assert _training_files(r['ev']) == ax
    assert elo.active_pool(str(r['ev']))[0] == 'visits-b1'
    assert {m['eval_yard']['select'] for m in read_metrics(str(r['ev']))} == {'visits-b1'}
    p = run_loop(tmp_path / 'nohead', '--iters', '1', '--selfplay-belief-tilt', '1')
    assert p.returncode == 78, p.stdout + p.stderr
    assert 'HeadMissing' in p.stderr and 'auxiliary head' in p.stderr
    assert not os.path.exists(tmp_path / 'nohead' / 'data')


def test_remote_selfplay_refuses_a_checkout_without_the_belief(tmp_path):
    """A worker whose checkout has no root belief (its hello says so: it would deal uniformly in silence) is
    killed at its hello; this machine plays every game."""
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '2', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--aux-next-card-w', '1', '--selfplay-belief-tilt', '1', env=env_for(tmp_path, {'no_belief': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    for x in m[1:]:
        assert x['selfplay']['games'] == 200 and x['selfplay']['remote']['games'] == 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED: its checkout plays the root belief as None' in log
