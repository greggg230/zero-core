"""A game package's own run settings, an extra evaluation's cadence of its own, and the extras of a remote
evaluation brought back (docs/az.md 40)."""
import json
import os
import threading
from types import SimpleNamespace

import pytest

from zero import package, remote
from zero import run as zrun
from zero.net import save
from zero.run import DEFAULTS, PRESETS, Run, model_path


def test_a_packages_settings_are_run_settings(tmp_path):
    """GamePackage.settings join DEFAULTS (a flag each, presets may set them, resolve_config keeps a saved one); a
    name the core has is refused, and so is a cadence setting that is not one of the package's integer settings.
    Without them DEFAULTS is the core's."""
    before = dict(DEFAULTS)
    pkg = package.GamePackage('tpkg', 'T', str(tmp_path), settings={'probe_every': 0, 'probe_opts': {'n': 4}},
                              evals_every={'probe': 'probe_every'},
                              presets={'probe-smoke': dict(PRESETS['toy-smoke'], probe_every=2)})
    try:
        zrun.apply_package(pkg)
        assert DEFAULTS['probe_every'] == 0 and DEFAULTS['probe_opts'] == {'n': 4}
        a = zrun.parse(['--run', 'x', '--probe-every', '3', '--probe-opts', '{"n": 8}'])
        assert a.probe_every == 3 and a.probe_opts == {'n': 8}
        assert zrun.resolve_config(a, {})['probe_every'] == 3
        assert zrun.resolve_config(zrun.parse(['--run', 'x', '--preset', 'probe-smoke']), {})['probe_every'] == 2
        assert zrun.resolve_config(zrun.parse(['--run', 'x']), {'probe_every': 5})['probe_every'] == 5
        with pytest.raises(ValueError, match="settings \\['iters'\\] are run settings of the core"):
            zrun.apply_package(package.GamePackage('tpkg', 'T', str(tmp_path), settings={'iters': 3}))
    finally:
        zrun.apply_package(package.active())
    assert DEFAULTS == before and 'probe_every' not in DEFAULTS
    for bad in ({'probe': 'nope'}, {'probe': 'probe_opts'}):
        with pytest.raises(ValueError, match='integer default'):
            package.GamePackage('tpkg', 'T', str(tmp_path), settings={'probe_every': 0, 'probe_opts': {}},
                                evals_every=bad)


def _probe_package(seen):
    """The active package with two extra evaluations of the toy: 'probe' (cadence `probe_every`) and 'always'.
    -> a restore function."""
    pkg = package.active()
    old = (dict(pkg.evals), dict(pkg.settings), dict(pkg.evals_every))
    pkg.evals['toy'] = {'probe': lambda r, it: seen.append(('probe', it)) or {'value': 0.5, 'n': 2},
                        'always': lambda r, it: seen.append(('always', it)) or {'value': 1.0}}
    pkg.settings['probe_every'] = 0
    pkg.evals_every['probe'] = 'probe_every'

    def restore():
        for d, o in zip((pkg.evals, pkg.settings, pkg.evals_every), old):
            d.clear()
            d.update(o)
    return restore


def test_an_extra_evaluation_with_a_cadence_runs_when_due(tmp_path):
    """evals_every: the evaluation runs on evaluated iterations that are multiples of its setting, never at 0 -
    silently (no log line, no extras key); one without an entry runs every time, as before."""
    run = tmp_path / 'x'
    run.mkdir()
    seen = []
    restore = _probe_package(seen)
    try:
        for every, its in ((0, (0, 1, 2, 4)), (2, (3, 4)), (5, (5, 7))):
            R = Run(str(run), dict(DEFAULTS, game='toy', probe_every=every))
            for it in its:
                m = {}
                R.extra_evals(it, m)
                due = every > 0 and it % every == 0
                assert set(m['extras']) == ({'probe', 'always'} if due else {'always'}), (every, it, m)
    finally:
        restore()
    assert [x for x in seen if x[0] == 'probe'] == [('probe', 4), ('probe', 5)]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'iter 4 probe: ' in log and 'iter 5 probe: ' in log and 'probe: skipped' not in log
    assert log.count("probe") == 2 and log.count("always") == 8
    # the core's own package declares nothing: every extra evaluation is due, DEFAULTS has no package setting
    assert package.CORE.settings == {} and package.CORE.eval_due('anything', {}, 3)


def test_a_remote_evaluation_brings_its_extras_back(tmp_path, monkeypatch):
    """A remote evaluation runs the package's extra evaluations with the rest of Run.evaluate (on the mirror, with the
    mirror's settings: the cadence setting shipped in config.json) and records their `extras` in the run's
    evals.jsonl beside eval_yard; the dashboard merges them into the iteration's row; the log names each. An
    iteration whose extra evaluation is not due has no extras."""
    from zero import dashboard
    seen = []
    restore = _probe_package(seen)
    try:
        cfg = dict(DEFAULTS, **PRESETS['toy-smoke'])
        cfg.update(game='toy', yardstick='random', eval_games=4, eval_random_games=0, elo_games=2, ref_games=2,
                   probe_every=2, eval_remote='local')
        run = tmp_path / 'runs' / 'r'
        (run / 'models').mkdir(parents=True)
        R = Run(str(run), cfg)
        for k in (0, 1, 2):
            save(R.new_model(), model_path(str(run), k), {'iter': k})
        monkeypatch.setenv('ZERO_REMOTE_WORKDIR', str(tmp_path / 'remote'))
        ev = remote.RemoteEval.__new__(remote.RemoteEval)          # (no thread: one evaluation at a time)
        ev.R, ev.name, ev.spec = R, 'local', remote.remote_spec('local')
        ev.t = remote.Transport(ev.spec)
        ev.rdir = ev.spec['workdir'].rstrip('/') + '/r'
        ev.module, ev.shipped, ev.anchors_shipped, ev.lock, ev.queued = 'zero.remote', None, set(), threading.Lock(), set()

        def run_module(module, args, timeout):
            # the remote side in this process (its package is this one): what `python -m zero.remote eval` prints
            assert module == 'zero.remote' and args[0] == 'eval'
            m = remote.evaluate_mirror(args[1], int(args[2]))
            return SimpleNamespace(returncode=0, stdout=remote.RESULT + json.dumps(m) + '\n', stderr='')
        ev.t.run_module = run_module
        for it in (1, 2):
            ev._evaluate(it)
    finally:
        restore()
    rows = [json.loads(line) for line in open(run / 'evals.jsonl', encoding='utf-8')]
    assert [r['iter'] for r in rows] == [1, 2]
    assert set(rows[0]['extras']) == {'always'} and set(rows[1]['extras']) == {'probe', 'always'}
    assert rows[1]['extras']['probe']['value'] == 0.5 and rows[1]['extras']['probe']['n'] == 2
    assert 'seconds' in rows[1]['extras']['probe'] and rows[1]['eval_yard']['games'] == 4
    assert json.load(open(tmp_path / 'remote' / 'r' / 'config.json'))['probe_every'] == 2
    assert [x for x in seen if x[0] == 'probe'] == [('probe', 2)]
    for m in ({'iter': 1}, {'iter': 2}):
        zrun.append_metrics(str(run), m)
    got = {m['iter']: m for m in dashboard._metrics(str(run))}
    assert got[2]['extras']['probe']['value'] == 0.5 and set(got[1]['extras']) == {'always'}
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'iter 2 probe (remote local): {"value": 0.5, "n": 2' in log and 'iter 1 probe' not in log
