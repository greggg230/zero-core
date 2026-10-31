"""Light decisions in self-play (zero/light.py, docs/search.md 3.10, docs/az.md 22) on the toy (its
light classes: the plow question and the counts decisions): the class selection, the settings
refused where they would do nothing or not be light, the self-play seat's keys and the evaluation
seats unchanged, a run's metrics (unrecorded and recorded), remote self-play's tallies and a worker
whose checkout searches otherwise refused."""
import json

import pytest

from zero import games as G, light as LT
from zero.run import DEFAULTS, Run, read_metrics

from test_loop import run_loop
from test_remoteplay import TOY, env_for

TOY_INFO = G.info('toy')


def cfg_of(**kw):
    c = dict(DEFAULTS, game='toy', sims=8, sims_fast=4)
    c.update(kw)
    return c


def test_the_toy_declares_its_classes():
    assert TOY_INFO['light_classes'] == ['plow', 'count']
    assert G.info('toy_team')['light_classes'] == []


@pytest.mark.parametrize('spec, want', [('all', [0, 1]), ('', [0, 1]), ('plow', [0]), ('count,plow', [0, 1]),
                                        ('all,-plow', [1]), ('*', [0, 1]), ('c*', [1]), ('all,-*', None),
                                        ('-count', None), ('sow', None), ('plow,,count', None)])
def test_class_selection(spec, want):
    if want is None:
        with pytest.raises(ValueError):
            LT.select_classes(spec, TOY_INFO['light_classes'])
    else:
        assert LT.select_classes(spec, TOY_INFO['light_classes']) == want


@pytest.mark.parametrize('kw, msg', [
    (dict(selfplay_light_record=1), 'without selfplay_sims_light'),
    (dict(selfplay_sims_light_fast=2), 'without selfplay_sims_light'),
    (dict(selfplay_light_classes='plow'), 'without selfplay_sims_light'),
    (dict(selfplay_sims_light=16), 'at most the one it replaces'),
    (dict(selfplay_sims_light=6), 'more than the fast budget'),
    (dict(selfplay_sims_light=4, selfplay_sims_light_fast=5), 'more than the fast budget'),
    (dict(selfplay_sims_light=2, selfplay_light_record=2), '0 or 1'),
    (dict(selfplay_sims_light=-1), '>= 0'),
    (dict(selfplay_sims_light=2, selfplay_light_classes='sow'), 'matches none'),
])
def test_settings_that_would_do_nothing_are_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        LT.settings(cfg_of(**kw), TOY_INFO)


def test_a_game_without_light_classes_is_refused():
    with pytest.raises(ValueError, match='declares no light decisions'):
        LT.settings(cfg_of(game='toy_team', selfplay_sims_light=2), G.info('toy_team'))


def test_the_seats(tmp_path):
    off = Run(str(tmp_path / 'off'), cfg_of())
    on = Run(str(tmp_path / 'on'), cfg_of(selfplay_sims_light=4, selfplay_sims_light_fast=1, selfplay_light_record=1,
                                          selfplay_light_classes='all,-plow'))
    assert off.light is None
    assert on.light == {'sims_light': 4, 'sims_light_fast': 1, 'light_record': True, 'light_classes': [1],
                        'names': ['count']}
    sp_off, sp_on = off.nn_seat('m', 8, selfplay=True), on.nn_seat('m', 8, selfplay=True)
    assert not any(k in sp_off for k in ('sims_light', 'sims_light_fast', 'light_record', 'light_classes'))
    assert dict(sp_on) == dict(sp_off, sims_light=4, sims_light_fast=1, light_record=True, light_classes=[1])
    # every evaluation seat is the seat of before, key for key
    assert on.nn_seat('m', 8) == off.nn_seat('m', 8) and on.nn_seat('o', 32) == off.nn_seat('o', 32)
    assert LT.fingerprint(on.cfg, TOY_INFO) == 'sims_light=4,sims_light_fast=1,record=1,classes=count'
    assert LT.fingerprint(off.cfg, TOY_INFO) == ''
    with pytest.raises(ValueError, match="self-play seat's setting"):
        G.nn_seat('m', 8, sims_light=2, light_classes=[0])


def light_rows(run):
    return [m['selfplay'] for m in read_metrics(str(run)) if 'selfplay' in m]


def test_a_run_with_unrecorded_light_decisions(tmp_path):
    run = tmp_path / 'u'
    p = run_loop(run, '--iters', '2', '--selfplay-sims-light', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    sp = light_rows(run)
    assert len(sp) == 2
    for s in sp:
        lt = s['light']
        assert lt['games'] == s['games'] == 24
        assert lt['per_game'] > 0 and 0 < lt['share'] < 1 and lt['recorded_per_game'] == 0
        assert lt['unrecorded_per_game'] == lt['per_game'] and lt['sims_per_game'] == pytest.approx(lt['per_game'], abs=0.05)
        assert lt['saved_per_game'] > 0 and 0 < lt['saved_share'] < 1
    c = json.load(open(run / 'config.json'))
    assert (c['selfplay_sims_light'], c['selfplay_sims_light_fast'], c['selfplay_light_record'],
            c['selfplay_light_classes']) == (1, 0, 0, 'all')
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'light decisions: 1 simulations (full draws) / 1 (fast draws), never recorded; classes plow,count' in log
    assert '"light": {"games": 24' in log


def test_a_run_with_recorded_light_decisions_and_one_without(tmp_path):
    run = tmp_path / 'r'
    p = run_loop(run, '--iters', '1', '--selfplay-sims-light', '4', '--selfplay-sims-light-fast', '1',
                 '--selfplay-light-record', '1', '--selfplay-light-classes', 'count')
    assert p.returncode == 0, p.stdout + p.stderr
    lt = light_rows(run)[0]['light']
    assert 0 < lt['recorded_per_game'] < lt['per_game']
    # off: the metrics of before (no `light`)
    run0 = tmp_path / 'o'
    p = run_loop(run0, '--iters', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    assert all('light' not in s for s in light_rows(run0))
    # a setting that would do nothing is refused at the start
    p = run_loop(tmp_path / 'x', '--iters', '1', '--selfplay-light-record', '1')
    assert p.returncode != 0 and 'without selfplay_sims_light' in p.stdout + p.stderr


def test_remote_selfplay_tallies_the_light_decisions(tmp_path):
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '3', *TOY, '--selfplay-remote', 'local', '--selfplay-sims-light', '2',
                 env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    remote = 0
    for x in m[1:]:
        sp = x['selfplay']
        assert sp['light']['games'] == sp['games'] + sp['remote']['games'], (sp['light'], sp['games'], sp['remote'])
        assert sp['light']['per_game'] > 0
        remote += sp['remote']['games']
    assert remote > 0
    rd = tmp_path / 'mirror' / 'sp'
    assert json.load(open(rd / 'config.json'))['selfplay_sims_light'] == 2


def test_remote_selfplay_refuses_a_checkout_without_light_decisions(tmp_path):
    """A worker whose checkout has no light decisions (its hello reports none: it would search every
    decision with the full budget in silence) is killed at its hello; this machine plays every game."""
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '2', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-sims-light', '2', env=env_for(tmp_path, {'no_light': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    for x in m[1:]:
        assert x['selfplay']['games'] == 200 and x['selfplay']['remote']['games'] == 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "REFUSED: its checkout searches light decisions as None, the run as 'sims_light=2" in log
