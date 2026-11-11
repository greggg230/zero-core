"""Value choices (docs/search.md 3.17, docs/az.md 39): the run settings selfplay_ / eval_value_choice_worlds and _temp,
the seat keys value_choice_worlds / value_choice_temp, the refusals (Python and the Rust driver), the Elo pool's name,
zero.h2h's run rule, the remote hello's fingerprint - and with none set, the seat dicts, names and games of before. The
search itself (a one-ply lookahead at the decisions a game marks, never recorded) is tested in Rust
(`driver::tests::value_choices_are_valued_by_a_one_ply_lookahead_and_never_recorded`): the toy marks none."""
import json
import os
import types

import pytest

from zero import elo, games as G, h2h, mcts
from zero.infer import UniformEvaluator, drive
from zero.remoteplay import _value_choice_of
from zero.run import DEFAULTS, Run, check_eval_settings, value_choice_keys


def test_settings_seats_names_and_refusals(tmp_path):
    for k, v in (('selfplay_value_choice_worlds', 0), ('selfplay_value_choice_temp', 0.0),
                 ('eval_value_choice_worlds', 0), ('eval_value_choice_temp', 0.0)):
        assert DEFAULTS[k] == v
    # none set: the seat dicts and names of before
    assert value_choice_keys(dict(DEFAULTS), 'selfplay') == {} == value_choice_keys(dict(DEFAULTS), 'eval')
    assert G.nn_seat('m', 8) == G.nn_seat('m', 8, value_choice_worlds=0, value_choice_temp=0.5)
    assert mcts.value_choice_tag(0, 1.0) == '' and elo.eval_tag(dict(DEFAULTS)) == 'visits'
    assert _value_choice_of(dict(DEFAULTS)) == ''
    # set: the keys, the names
    s = G.nn_seat('m', 8, value_choice_worlds=4, value_choice_temp=0.05)
    assert (s['value_choice_worlds'], s['value_choice_temp']) == (4, 0.05)
    r = types.SimpleNamespace(cfg=dict(DEFAULTS, selfplay_value_choice_worlds=8, selfplay_value_choice_temp=0.03,
                                       eval_value_choice_worlds=4))
    sp = Run.nn_seat(r, 'm', 64, selfplay=True)
    ev = Run.nn_seat(r, 'm', 32)
    assert (sp['value_choice_worlds'], sp['value_choice_temp']) == (8, 0.03)
    assert (ev['value_choice_worlds'], ev['value_choice_temp']) == (4, 0.0)
    assert mcts.value_choice_tag(4) == '-vc4' and mcts.value_choice_tag(8, 0.03) == '-vc8t0.03'
    assert elo.eval_tag(dict(DEFAULTS, eval_value_choice_worlds=4)) == 'visits-vc4'
    assert elo.eval_tag(dict(DEFAULTS, selfplay_value_choice_worlds=4)) == 'visits'
    assert _value_choice_of(r.cfg) == '-vc8t0.03'
    # refusals: Python and the Rust driver
    for w, t in ((-1, 0.0), (1025, 0.0), (2.5, 0.0), (True, 0.0), (4, -0.1), (4, float('nan')), (4, float('inf'))):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, value_choice_worlds=w, value_choice_temp=t)
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, eval_value_choice_worlds=w, eval_value_choice_temp=t))
    with pytest.raises(ValueError, match='value_choice_temp'):
        G.Driver('toy', [[dict(s, value_choice_temp=-1.0)] * 2], concurrency=1)
    with pytest.raises(ValueError, match='lookahead'):
        G.Driver('toy', [[dict(G.nn_seat('m', 8), lookahead={'plies': 1}, value_choice_worlds=2)] * 2], concurrency=1)
    # h2h: a checkpoint's run gives its side the evaluations' value choices, and names them
    d = tmp_path / 'run'
    os.makedirs(d / 'models')
    json.dump(dict(DEFAULTS, eval_value_choice_worlds=4, eval_value_choice_temp=0.0), open(d / 'config.json', 'w'))
    rule = h2h.run_rule(str(d / 'models' / 'iter_0001.pt'))
    assert (rule['value_choice_worlds'], rule['value_choice_temp']) == (4, 0.0)
    assert h2h._tag(rule).endswith('-vc4')


def test_a_game_without_value_choices_plays_the_same_games_with_the_setting():
    """The toy marks no value choice: a seat with the setting plays exactly the games of one without (results, steps,
    records) and makes no value choice."""
    def play(**kw):
        seat = G.nn_seat('m', 8, selfplay=True, sims_fast=2, p_full=0.5, temp_moves=10, **kw)
        drv = G.Driver('toy', [[seat, seat]] * 6, concurrency=6, seed=11, record=True, threads=2)
        drive(drv, {'m': UniformEvaluator()})
        res = sorted(drv.take_results(), key=lambda r: r['gid'])
        smp = drv.take_samples()
        return [(r['result'], r['steps'], r['decisions']) for r in res], smp['x'].sum(), smp['pi'].sum(), drv.stats()
    a, b = play(), play(value_choice_worlds=4, value_choice_temp=0.05)
    assert a[:3] == b[:3]
    assert b[3]['value_choices'] == 0 and b[3]['value_choice_rows'] == 0
