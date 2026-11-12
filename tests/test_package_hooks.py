"""A game package's run settings, self-play spec hook and remote evaluation inputs (docs/az.md 40): off unless a
package declares them; with them, the phase's spec reaches the driver (a rolling driver switches between phases), a
package setting is a run setting, and a remote evaluation gets the run's files it names."""
import json
import os
from types import SimpleNamespace

import pytest

from zero import package
from zero import run as RR
from zero.net import save


def toy_run(tmp_path, **kw):
    cfg = dict(RR.DEFAULTS, **RR.PRESETS['toy-smoke'])
    cfg.update(game='toy', games=6, conc=6, threads=1, groups=1, sims=4, sims_fast=2, **kw)
    run = tmp_path / 'r'
    (run / 'models').mkdir(parents=True)
    R = RR.Run(str(run), cfg)
    save(R.new_model(), RR.model_path(str(run), 0), {'iter': 0})
    return R


@pytest.fixture
def hook():
    pkg = package.active()
    old = dict(pkg.selfplay_specs)
    calls = []
    answers = {}

    def fn(R, it):
        calls.append(it)
        return answers.get(it)
    pkg.selfplay_specs['toy'] = fn
    try:
        yield calls, answers
    finally:
        pkg.selfplay_specs.clear()
        pkg.selfplay_specs.update(old)


def test_without_a_hook_selfplay_plays_the_runs_spec(tmp_path):
    R = toy_run(tmp_path, rolling=1)
    assert package.active().selfplay_spec_fn('toy') is None
    assert R.selfplay_spec(1) == R.cfg['spec']
    R.selfplay(1)
    assert R.roll_spec == R.cfg['spec']


def test_a_rolling_driver_switches_to_the_hooks_spec_between_phases(tmp_path, hook):
    calls, answers = hook
    R = toy_run(tmp_path, rolling=1)
    new = '{"food": [20, 0]}'
    answers.update({2: new, 3: new})
    R.selfplay(1)
    assert R.roll_spec == R.cfg['spec'], 'None: the run\'s spec'
    save(R.new_model(), RR.model_path(R.run, 1), {'iter': 1})
    R.selfplay(2)
    assert R.roll_spec == new
    save(R.new_model(), RR.model_path(R.run, 2), {'iter': 2})
    R.selfplay(3)
    log = open(os.path.join(R.run, 'log.txt'), encoding='utf-8').read()
    assert 'iter 2 self-play: the rolling driver plays the game package\'s new spec' in log
    assert calls == [1, 2, 3] and log.count('new spec') == 1
    answers[4] = 7
    save(R.new_model(), RR.model_path(R.run, 3), {'iter': 3})
    with pytest.raises(ValueError, match='selfplay_specs hook answered 7'):
        R.selfplay(4)
    answers[4] = '{"food": "x"}'
    with pytest.raises(Exception, match='set_spec'):
        R.selfplay(4)


def test_a_phases_driver_is_made_with_the_hooks_spec(tmp_path, hook, monkeypatch):
    calls, answers = hook
    R = toy_run(tmp_path, rolling=0)
    made = []
    real = R.driver

    def driver(*a, **kw):
        made.append(kw.get('spec'))
        return real(*a, **kw)
    monkeypatch.setattr(R, 'driver', driver)
    answers[2] = '{"food": [3, 0]}'
    R.selfplay(1)
    save(R.new_model(), RR.model_path(R.run, 1), {'iter': 1})
    R.selfplay(2)
    assert made == [None, '{"food": [3, 0]}']
    with pytest.raises(ValueError, match='self-play'):
        real([], seed=0, spec='{}')


def test_remote_selfplay_refuses_a_changed_spec(tmp_path, hook):
    calls, answers = hook
    R = toy_run(tmp_path, rolling=1)
    R.rsp = object()
    answers[1] = '{"food": [1, 0]}'
    with pytest.raises(RuntimeError, match='remote self-play'):
        R.selfplay(1)


def test_package_settings_are_run_settings(tmp_path):
    pkg = package.active()
    try:
        pkg.settings = {'probe_every': 0, 'probe_name': 'x'}
        RR.apply_package(pkg)
        assert RR.DEFAULTS['probe_every'] == 0 and RR.DEFAULTS['probe_name'] == 'x'
        a = RR.parse(['--run', 'x', '--probe-every', '3'])
        assert RR.resolve_config(a, {})['probe_every'] == 3
        pkg.settings = {'sims': 1}
        with pytest.raises(ValueError, match='names of the core'):
            RR.apply_package(pkg)
    finally:
        pkg.settings = {}
        RR.apply_package(pkg)
    assert 'probe_every' not in RR.DEFAULTS


def test_remote_evaluation_ships_the_packages_inputs(tmp_path, monkeypatch):
    from zero import remote
    run, mirror = tmp_path / 'run', tmp_path / 'mirror'
    (run / 'evaluators').mkdir(parents=True)
    mirror.mkdir()
    (run / 'evaluators' / 'v3.json').write_text('three')
    (run / 'a.txt').write_text('a')
    names = {3: ['evaluators/v3.json', 'a.txt'], 4: ['missing.txt']}
    monkeypatch.setattr(package.active(), 'remote_eval_inputs', {'toy': lambda R, it: names.get(it, [])})
    ev = remote.RemoteEval.__new__(remote.RemoteEval)      # (no thread: the shipping step alone)
    ev.R, ev.t, ev.rdir = SimpleNamespace(run=str(run), cfg={'game': 'toy'}), remote.Transport({'host': 'local'}), str(mirror)
    ev.inputs_shipped = {}
    ev._ship_inputs(3)
    assert (mirror / 'evaluators' / 'v3.json').read_text() == 'three' and (mirror / 'a.txt').read_text() == 'a'
    (mirror / 'a.txt').write_text('stale')
    ev._ship_inputs(3)
    assert (mirror / 'a.txt').read_text() == 'stale', 'unchanged since it was put: not put again'
    (run / 'a.txt').write_text('a, changed')
    os.utime(run / 'a.txt', ns=(1, 10 ** 18))
    ev._ship_inputs(3)
    assert (mirror / 'a.txt').read_text() == 'a, changed'
    with pytest.raises(RuntimeError, match='missing.txt'):
        ev._ship_inputs(4)
    ev._ship_inputs(5)
    for bad in ('/abs', '../up', 'a\\b', 'c:/x'):
        names[6] = [bad]
        with pytest.raises(ValueError, match='not a relative'):
            ev._ship_inputs(6)
    assert package.active().remote_eval_inputs_of('nogame', ev.R, 3) == []
    json.dumps(names)
