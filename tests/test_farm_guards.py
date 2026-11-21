"""docs/az.md 70: the farm guards - a fork's own farm runs (tools/fork_run.py `fork_farm`, fork.json), pool settings that
do nothing (`inert_pool_settings`), and the reason a run's self-play stays on this machine beside a published farm build
(`selfplay_local_reason`: required at a fresh start and a fork's first start, a warning on a resumed run), end to end
on the toy game with a stand-in for the game package's `farm_build` hook."""
import json
import os
import subprocess
import sys

import pytest

from zero import run as R
from zero.tools import fork_run as FR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']
POOL = 'b' * 24


# ------------------------------------------------------------------ fork_run: the fork's own farm runs
def test_a_fork_gets_its_own_farm_runs(tmp_path):
    parent = {'selfplay_remote': 'mac,pool:legacy-r1', 'eval_remote': 'pool:legacy-r1-eval', 'pool_build': POOL}
    ch, notes = FR.fork_farm(parent, {'lr': 1e-4}, str(tmp_path / 'Meta3_AB.adapt'))
    assert ch == {'selfplay_remote': 'mac,pool:meta3-ab-adapt', 'eval_remote': 'pool:meta3-ab-adapt-eval'}
    assert any('OWN farm run pool:meta3-ab-adapt' in n and 'never' in n for n in notes)
    assert any('pool:meta3-ab-adapt-eval' in n for n in notes)
    # a name of its own (--farm NAME); none (this machine only; another host kept)
    ch, _ = FR.fork_farm(parent, {}, str(tmp_path / 'x'), 'arm-b')
    assert ch == {'selfplay_remote': 'mac,pool:arm-b', 'eval_remote': 'pool:arm-b-eval'}
    ch, notes = FR.fork_farm(parent, {}, str(tmp_path / 'x'), FR.FARM_NONE)
    assert ch == {'selfplay_remote': 'mac', 'eval_remote': ''} and any('--farm none' in n for n in notes)
    # an override is kept as given (its own pool, or none), never rewritten
    assert FR.fork_farm(parent, {'selfplay_remote': '', 'eval_remote': ''}, str(tmp_path / 'x')) == ({}, [])
    ch, _ = FR.fork_farm(parent, {'selfplay_remote': 'pool:mine'}, str(tmp_path / 'x'))
    assert ch == {'eval_remote': 'pool:x-eval'}
    # never the parent's: an override naming it, a fork named like it, a name that is not a slug
    for over in ({'selfplay_remote': 'pool:legacy-r1'}, {'eval_remote': 'pool:legacy-r1-eval'}):
        with pytest.raises(SystemExit, match='never writes into its parent'):
            FR.fork_farm(parent, over, str(tmp_path / 'x'))
    with pytest.raises(SystemExit, match="are the source's"):
        FR.fork_farm(parent, {}, str(tmp_path / 'legacy-r1'))
    with pytest.raises(SystemExit, match='must be a slug'):
        FR.fork_farm(parent, {}, str(tmp_path / 'x'), 'Not A Slug')
    with pytest.raises(SystemExit, match='names a farm pool'):
        FR.fork_farm(parent, {'selfplay_remote': 'pool:mine'}, str(tmp_path / 'x'), FR.FARM_NONE)
    # a parent without a farm: nothing to change (a name has nothing to name)
    assert FR.fork_farm({'selfplay_remote': 'mac'}, {}, str(tmp_path / 'x')) == ({}, [])
    assert FR.fork_farm({'selfplay_remote': 'mac'}, {}, str(tmp_path / 'x'), FR.FARM_NONE) == ({}, [])
    with pytest.raises(SystemExit, match='names no farm run'):
        FR.fork_farm({}, {}, str(tmp_path / 'x'), 'arm-b')
    assert FR.farm_slug('--A__b..C--') == 'a-b-c' and FR.farm_slug('x' * 80) == 'x' * 63 and FR.farm_slug('__') == ''
    assert FR.fork_farm(parent, {}, str(tmp_path / ('y' * 70)))[0]['eval_remote'] == f'pool:{"y" * 58}-eval'


def test_the_sources_selfplay_remotes_file_is_the_parents_farm_too(tmp_path):
    """The source's selfplay_remotes file (docs/az.md 34) replaces its setting and is never copied: a pool it names is
    the parent's farm run - the fork gets its own (or none), and an override naming it is refused."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'selfplay_remotes').write_text('# the farm only\npool:legacy-r1\n')
    assert FR.farm_runs({'selfplay_remote': ''}, str(src)) == {'selfplay': ['legacy-r1'], 'eval': None}
    ch, notes = FR.fork_farm({'selfplay_remote': ''}, {}, str(tmp_path / 'fork-a'), source=str(src))
    assert ch == {'selfplay_remote': 'pool:fork-a'} and 'pool:legacy-r1' in notes[0]
    assert FR.fork_farm({'selfplay_remote': ''}, {}, str(tmp_path / 'fork-a'), FR.FARM_NONE, str(src))[0] == \
        {'selfplay_remote': ''}
    with pytest.raises(SystemExit, match='never writes into its parent'):
        FR.fork_farm({}, {'selfplay_remote': 'pool:legacy-r1'}, str(tmp_path / 'fork-a'), source=str(src))


# ------------------------------------------------------------------ zero.run: the start's checks
def test_inert_pool_settings(tmp_path):
    cfg = dict(R.DEFAULTS, pool_build=POOL, pool_variants='gpu=fp16', pool_max_stale=4)
    assert R.inert_pool_settings(cfg) == ['pool_build', 'pool_max_stale', 'pool_variants']
    # the farm evaluation reads the build's settings, not the self-play pool's own
    assert R.inert_pool_settings(dict(cfg, eval_remote='pool:x-eval')) == ['pool_max_stale', 'pool_variants']
    assert R.inert_pool_settings(dict(cfg, selfplay_remote='mac,pool:x')) == []
    assert R.inert_pool_settings(dict(R.DEFAULTS)) == []
    (tmp_path / 'selfplay_remotes').write_text('pool:x\n')
    assert R.selfplay_pools(str(tmp_path), cfg) == ['pool:x'] and R.inert_pool_settings(cfg, str(tmp_path)) == []
    (tmp_path / 'selfplay_remotes').write_text('mac\n')    # (the file replaces the setting while it exists)
    assert R.selfplay_pools(str(tmp_path), dict(cfg, selfplay_remote='pool:y')) == []


def test_check_farm_use(tmp_path, monkeypatch):
    monkeypatch.delenv(R.LOCAL_REASON_ENV, raising=False)
    run = str(tmp_path)
    pub = lambda run, cfg: 'build b0 of commit c0 (a test)'  # noqa: E731
    lines = []
    cfg = dict(R.DEFAULTS)
    with pytest.raises(R.FarmUseRefused, match='selfplay-local-reason') as e:
        R.check_farm_use(run, cfg, True, lines.append, pub)
    assert isinstance(e.value, R.EncodingMismatch) and 'build b0 of commit c0' in str(e.value)    # (exit 78)
    R.check_farm_use(run, cfg, False, lines.append, pub)
    assert lines[-1].startswith('WARNING: a published farm build') and 'resumed run: not refused' in lines[-1]
    lines.clear()
    R.check_farm_use(run, dict(cfg, selfplay_local_reason='timing A/B'), True, lines.append, pub)
    assert lines == ['self-play on this machine only (selfplay_local_reason): timing A/B - beside a published farm build '
                     'of this commit (build b0 of commit c0 (a test))']
    monkeypatch.setenv(R.LOCAL_REASON_ENV, 'a test harness')
    R.check_farm_use(run, cfg, True, lines.append, lambda run, cfg: None)
    assert lines[-1] == f'self-play on this machine only (${R.LOCAL_REASON_ENV}): a test harness'
    monkeypatch.delenv(R.LOCAL_REASON_ENV)
    lines.clear()
    R.check_farm_use(run, cfg, True, lines.append, lambda run, cfg: None)       # no published build: nothing
    R.check_farm_use(run, dict(cfg, selfplay_remote='pool:x'), True, lines.append, pub)    # on the farm: nothing
    assert lines == []
    R.check_farm_use(run, dict(cfg, pool_build=POOL, selfplay_local_reason='x'), True, lines.append, pub)
    assert lines[0].startswith('WARNING: pool settings that do nothing: pool_build=') and 'yardsticks are off' in lines[0]
    # the game package's hook: none (the toy's) and one that fails count as no build
    assert R.check_farm_use(run, cfg, True, lines.append) is None
    from zero import package

    def broken(run, cfg):
        raise ValueError('no index')
    monkeypatch.setattr(package.active(), 'farm_build', broken)
    lines.clear()
    R.check_farm_use(run, cfg, True, lines.append)
    assert 'the farm build lookup failed (ValueError: no index)' in lines[0]


def test_first_start_and_the_fork_marker(tmp_path):
    run = str(tmp_path / 'r')
    os.makedirs(os.path.join(run, 'models'))
    assert R.first_start(run)                      # no checkpoint: fresh
    open(os.path.join(run, 'models', 'iter_0003.pt'), 'wb').close()
    assert not R.first_start(run)                  # resumed (every run of before fork.json too)
    FR.write_fork(run, {'kind': 'continuation', 'iter': 3, 'started': None})
    assert R.first_start(run)                      # a fork that has not started
    assert FR.mark_started(run) and FR.read_fork(run)['started'] and not R.first_start(run)
    assert not FR.mark_started(run)


# ------------------------------------------------------------------ end to end (the toy game, a stand-in hook)
WRAPPER = '''import sys
from zero import package
package.CORE.farm_build = lambda run, cfg: 'build b0b0 of commit c0ffee (a test)'
from zero import run
run.cli()
'''


def _run(tmp_path, run, *extra, hook=True, env=None):
    w = tmp_path / 'with_hook.py'
    w.write_text(WRAPPER)
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    e.pop(R.LOCAL_REASON_ENV, None)
    e.update(env or {})
    cmd = [sys.executable, str(w)] if hook else [sys.executable, '-m', 'zero.run']
    return subprocess.run(cmd + ['--run', str(run), *SMOKE, *extra], cwd=ROOT, capture_output=True, text=True,
                          timeout=300, env=e)


def _log(run):
    with open(os.path.join(run, 'log.txt'), encoding='utf-8') as f:
        return f.read()


def test_the_local_reason_rule_end_to_end(tmp_path, capsys):
    """A fresh start without a farm pool beside a published build: refused (exit 78, run_state 'refused', no checkpoint
    written); with selfplay_local_reason it runs. A run that started before (a resume, as every live run after PAUSE)
    only warns - also of its inert pool settings. Its fork's first start is refused again until a reason is given
    ($ZERO_SELFPLAY_LOCAL_REASON here), then fork.json is marked started."""
    from zero import runctl
    fresh = tmp_path / 'fresh'
    p = _run(tmp_path, fresh, '--iters', '1')
    assert p.returncode == 78, p.stdout + p.stderr
    assert 'build b0b0 of commit c0ffee' in p.stderr and runctl.read_state(str(fresh), 'run')['state'] == 'refused'
    assert not os.path.exists(fresh / 'models' / 'iter_0000.pt')
    p = _run(tmp_path, fresh, '--iters', '1', '--selfplay-local-reason', 'a toy test')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'self-play on this machine only (selfplay_local_reason): a toy test' in _log(fresh)
    # a run that started before the hook said anything: resumed, it warns and runs
    old = tmp_path / 'old'
    p = _run(tmp_path, old, '--iters', '1', hook=False)
    assert p.returncode == 0, p.stdout + p.stderr
    p = _run(tmp_path, old, '--iters', '2', '--pool-build', POOL)
    assert p.returncode == 0, p.stdout + p.stderr
    log = _log(old)
    assert 'WARNING: a published farm build exists for this commit (build b0b0' in log and 'not refused' in log
    assert f"WARNING: pool settings that do nothing: pool_build='{POOL}'" in log
    assert [r['iter'] for r in R.read_metrics(str(old))] == [0, 1, 2]
    # its fork: a first start again
    fork = tmp_path / 'fork'
    assert FR.main(['--source', str(old), '--dest', str(fork), '--iter', '2']) == 0
    assert FR.read_fork(str(fork))['started'] is None and FR.read_fork(str(fork))['iter'] == 2
    p = _run(tmp_path, fork, '--iters', '3')
    assert p.returncode == 78, p.stdout + p.stderr
    assert FR.read_fork(str(fork))['started'] is None
    p = _run(tmp_path, fork, '--iters', '3', env={R.LOCAL_REASON_ENV: 'the test harness'})
    assert p.returncode == 0, p.stdout + p.stderr
    assert FR.read_fork(str(fork))['started'] and not R.first_start(str(fork))
    assert f'(${R.LOCAL_REASON_ENV}): the test harness' in _log(fork)
    # a parent on the farm: the fork's config names its own farm runs, fork.json records it, the output says so
    with open(old / 'config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    with open(old / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(dict(cfg, selfplay_remote='pool:old', eval_remote='pool:old-eval'), f)
    farm = tmp_path / 'farm-fork'
    FR.continuation(str(old), str(farm), 2, {})
    out = capsys.readouterr().out
    with open(farm / 'config.json', encoding='utf-8') as f:
        got = json.load(f)
    assert (got['selfplay_remote'], got['eval_remote'], got['pool_build']) == ('pool:farm-fork', 'pool:farm-fork-eval', POOL)
    assert FR.read_fork(str(farm))['farm'] == {'mode': 'own', 'changes': {'selfplay_remote': 'pool:farm-fork',
                                                                           'eval_remote': 'pool:farm-fork-eval'}}
    assert 'FARM: the source self-plays on pool:old; the fork self-plays on its OWN farm run pool:farm-fork' in out
    with pytest.raises(SystemExit, match='never writes into its parent'):
        FR.continuation(str(old), str(tmp_path / 'bad'), 2, {'selfplay_remote': 'pool:old'})
    assert not os.path.exists(tmp_path / 'bad' / 'config.json'), 'refused before anything is copied'


def test_a_worker_build_that_refuses_refuses_the_runs_start(tmp_path):
    """docs/az.md 70: a farm pool whose settings check refuses definitively (here: no worker executable) refuses the
    run's start - exit 78, run_state 'refused', not restarted by the supervisor - before anything reaches the farm
    (the coordinator's address is a documentation address that is never called)."""
    from zero import runctl
    run = tmp_path / 'pooled'
    run.mkdir()
    FR.write_fork(str(run), {'kind': 'new', 'iter': 0, 'started': None})     # (as a fork's first start)
    p = _run(tmp_path, run, '--iters', '1', '--selfplay-remote', 'pool:toy-pooled', '--pool-build', POOL,
             '--pool-worker', str(tmp_path / 'nowhere.exe'), '--pool-url', 'http://192.0.2.1', hook=False,
             env={'ZEROFARM_ADMIN_TOKEN': 'a-test-token'})
    assert p.returncode == 78, p.stdout + p.stderr
    st = runctl.read_state(str(run), 'run')
    assert st['state'] == 'refused' and 'not an executable on this machine' in st['error']
    assert 'REFUSED at the run\'s start' in _log(run)
    assert FR.read_fork(str(run))['started'] is None, 'a first start refused by any check stays a first start'
