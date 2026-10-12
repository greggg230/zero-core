"""State detection on every fixture run, at the snapshot and in the other states (made by
editing the copy's files or changing which processes are alive / what time it is)."""
import json
import os
import time

from zerodash import procs, runfiles, state
from zerodash.procs import FixedLiveness, PsutilLiveness

# what the desktop showed when the snapshot was taken (2026-10-12 09:53 PDT)
AT_SNAPSHOT = {
    'stpete/az6_cq': ('running', 'supervisor.log'),
    'stpete/az5': ('stopped', 'supervisor.log'),
    'dominion/all1': ('paused', 'state files'),
    'dominion/rand6': ('paused', 'state files'),
    'caverna/cvz2p': ('paused', 'state files'),
    'caverna/cvz2p_cq': ('running', 'state files'),
    'glassroad/grz2p_cq': ('paused', 'state files'),
    'glassroad/grz2p_cq_s512': ('running', 'state files'),
}


def test_every_fixture_run_at_snapshot(fx):
    runs = fx.runs()
    assert set(runs) == set(AT_SNAPSHOT)
    for key, (st, source) in AT_SNAPSHOT.items():
        r = runs[key]
        assert (r['state'], r['source']) == (st, source), (key, r['state'], r['detail'])
        assert not r['hidden'], key


def test_snapshot_details(fx):
    runs = fx.runs()
    # St. Pete az6_cq: supervisor + run alive per supervisor.log; self-play of iteration 59
    r = runs['stpete/az6_cq']
    assert r['run_alive'] and r['supervisor_alive']
    assert r['iteration']['current'] == 59 and r['iteration']['phase'] == 'selfplay'
    assert r['iteration']['last'] == 58
    assert not r['pause_supported'] and not r['can_pause'] and not r['can_resume']
    assert r['eval']['key'] == 'eval_shrewd' and r['eval']['win'] == 1.0 and r['eval']['iter'] == 58
    assert r['stale_after_s'] > 2 * 3600          # 1.5 x its ~85 min iterations: an hour of silent self-play is normal
    # az5: clean exit after its STOP file
    r = runs['stpete/az5']
    assert r['stop'] and r['detail'] == 'Ended by STOP.'
    # Glass Road paused by hand during self-play; the supervisor waits
    r = runs['glassroad/grz2p_cq']
    assert r['pause'] == {'time': 1791820487.3917716, 'by': 'tools/pause_run.py', 'kind': 'other'}
    assert r['supervisor_alive'] and not r['run_alive'] and r['can_resume'] and not r['can_pause']
    assert r['iteration']['current'] == 125 and r['iteration']['paused_in'].startswith('PAUSED (self-play)')
    assert r['eval']['label'] == 'JS bot' and r['eval']['win'] == 0.675 and r['eval']['iter'] == 123
    assert r['elo']['iter'] == 123 and round(r['elo']['elo']) == 1162 and len(r['elo']['series']) > 10
    assert r['throughput']['iter'] == 124 and round(r['throughput']['games_per_h']) == 36328
    # Glass Road s512: running, progress.json gives the phase and the games
    r = runs['glassroad/grz2p_cq_s512']
    it = r['iteration']
    assert (it['current'], it['phase'], it['phase_detail']) == (126, 'selfplay', '2,941 / 3,000 games')
    assert abs(it['phase_frac'] - 2941 / 3000) < 1e-9 and r['can_pause']
    # Caverna cvz2p paused during evaluation of iteration 171
    r = runs['caverna/cvz2p']
    assert (r['iteration']['current'], r['iteration']['phase']) == (171, 'eval')
    # Dominion: yardstick BMU, positions/s as leaves/s, games_per_hour; an evaluation pending on the Mac
    r = runs['dominion/rand6']
    assert r['eval']['key'] == 'eval_bmu' and r['eval']['iter'] == 240 and r['eval']['pending']['iter'] == 249
    assert round(r['throughput']['leaves_per_s']) == 135437 and round(r['throughput']['games_per_h']) == 93540
    r = runs['dominion/all1']
    assert r['iteration']['current'] == 282 and r['eval']['iter'] == 279 and 'pending' not in r['eval']


def test_nothing_alive(fx):
    """After a reboot: runs that were mid-run are crashed, paused runs say no supervisor waits."""
    runs = fx.runs(fx.dash(alive={}))
    assert runs['stpete/az6_cq']['state'] == 'crashed'          # supervisor.log ends with 'started ... pid'
    assert runs['stpete/az5']['state'] == 'stopped'
    assert runs['caverna/cvz2p_cq']['state'] == 'crashed'       # run_state still 'running'
    assert runs['glassroad/grz2p_cq_s512']['state'] == 'crashed'
    for k in ('glassroad/grz2p_cq', 'caverna/cvz2p', 'dominion/all1', 'dominion/rand6'):
        assert runs[k]['state'] == 'paused' and 'No supervisor' in runs[k]['detail'], k
    assert not any(r['can_pause'] for r in runs.values())


def test_stale_when_live_but_silent(fx):
    later = fx.now + 3 * 3600
    runs = fx.runs(fx.dash(now=later))
    assert runs['caverna/cvz2p_cq']['state'] == 'stale'
    assert runs['glassroad/grz2p_cq_s512']['state'] == 'stale'
    assert runs['stpete/az6_cq']['state'] == 'stale'             # 4 h without a log line > 1.5 x 85 min
    assert runs['glassroad/grz2p_cq']['state'] == 'paused'       # paused runs are never stale
    assert runs['caverna/cvz2p_cq']['can_pause']                 # a stale run can still be paused


def test_pausing_and_stopping(fx):
    fx.write_json('glassroad', 'grz2p_cq_s512', 'PAUSE', {'time': fx.now, 'by': 'tools/pause_run.py'})
    open(fx.path('caverna', 'cvz2p_cq', 'STOP'), 'w').close()
    runs = fx.runs()
    assert runs['glassroad/grz2p_cq_s512']['state'] == 'pausing' and runs['glassroad/grz2p_cq_s512']['can_resume']
    assert runs['caverna/cvz2p_cq']['state'] == 'stopping' and runs['caverna/cvz2p_cq']['stop']


def test_resuming_starting_restarting(fx):
    # PAUSE removed, the supervisor (alive) still says paused -> resuming
    os.remove(fx.path('glassroad', 'grz2p_cq', 'PAUSE'))
    # the supervisor launched a new run that has not written its state yet -> starting
    fx.edit_json('dominion', 'all1', 'supervisor_state.json', state='running', run_pid=999991, since=fx.now - 5)
    os.remove(fx.path('dominion', 'all1', 'PAUSE'))
    # the run crashed, the supervisor waits to restart it
    fx.edit_json('caverna', 'cvz2p', 'supervisor_state.json', state='restarting', since=fx.now - 30)
    os.remove(fx.path('caverna', 'cvz2p', 'PAUSE'))
    runs = fx.runs()
    assert runs['glassroad/grz2p_cq']['state'] == 'resuming'
    assert runs['dominion/all1']['state'] == 'starting'
    r = runs['caverna/cvz2p']
    assert r['state'] == 'restarting' and r['label'] == 'Crashed - restarting'


def test_restart_loop_from_supervisor_log(fx):
    """St. Pete (no state files): three crashes in the hour and a live supervisor."""
    log = fx.path('stpete', 'az6_cq', 'supervisor.log')
    with open(log, 'a', encoding='utf-8') as f:
        for k, t in enumerate(('09:01:00', '09:15:00', '09:30:00'), 1):
            f.write(f'2026-10-12 {t} stpz.run exited with code 1; killed 0 orphaned workers\n')
            f.write(f'2026-10-12 {t} crash #{k} in the last hour; restarting in 60s\n')
    r = fx.runs()['stpete/az6_cq']
    assert r['state'] == 'restarting' and r['label'] == 'Restart loop' and r['crashes_1h'] == 3


def test_crashed_and_finished_with_state_files(fx):
    fx.edit_json('glassroad', 'grz2p_cq_s512', 'run_state.json', state='aborted')
    fx.edit_json('caverna', 'cvz2p_cq', 'run_state.json', state='done')
    runs = fx.runs(fx.dash(alive={}))
    assert runs['glassroad/grz2p_cq_s512']['state'] == 'crashed'
    assert runs['glassroad/grz2p_cq_s512']['detail'].startswith('Aborted')
    assert runs['caverna/cvz2p_cq']['state'] == 'stopped' and runs['caverna/cvz2p_cq']['detail'] == 'Finished.'


def test_pid_reuse_is_not_alive(fx):
    """A live process with a state file's pid but another creation time is someone else."""
    alive = {int(k): v + 500 for k, v in fx.snap['alive'].items()}
    runs = fx.runs(fx.dash(alive=alive))
    assert runs['caverna/cvz2p_cq']['state'] == 'crashed'
    assert runs['glassroad/grz2p_cq']['state'] == 'paused' and not runs['glassroad/grz2p_cq']['supervisor_alive']


def test_unknown_liveness_judges_by_output(fx):
    runs = fx.runs(fx.dash(liveness=FixedLiveness(unknown=True)))
    assert runs['caverna/cvz2p_cq']['state'] == 'running' and runs['caverna/cvz2p_cq']['run_alive'] is None
    assert runs['glassroad/grz2p_cq']['state'] == 'paused'
    assert runs['stpete/az5']['state'] == 'stopped'
    runs = fx.runs(fx.dash(liveness=FixedLiveness(unknown=True), now=fx.now + 5 * 3600))
    assert runs['caverna/cvz2p_cq']['state'] == 'unknown'


def test_hidden_after_days_without_live_processes(fx):
    later = fx.now + 4 * 86400
    runs = fx.runs(fx.dash(now=later))
    assert runs['stpete/az5']['hidden']                      # stopped, untouched for 4+ days
    assert not runs['glassroad/grz2p_cq']['hidden']          # paused, its supervisor waits
    runs = fx.runs(fx.dash(now=later, alive={}))
    assert all(r['hidden'] for r in runs.values())
    snap = fx.dash(now=later, alive={}).snapshot()
    assert all(p['hidden_count'] == len(p['runs']) for p in snap['projects'])


def test_discovery_skips_non_runs_and_follows_junctions(fx, tmp_path):
    root_a, root_b = tmp_path / 'a', tmp_path / 'b'
    (root_a / 'ladder').mkdir(parents=True)                  # no config.json / metrics.jsonl: not a run
    (root_a / 'notes.txt').write_text('x')
    os.makedirs(root_b)
    run = root_b / 'az6_cq'
    import shutil
    shutil.copytree(fx.dir('stpete', 'az6_cq'), run)
    link = root_a / 'az6_cq'
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(run), str(link))
    else:
        os.symlink(run, link)
    try:
        cfg = dict(fx.cfg)
        st = dict(next(p for p in cfg['projects'] if p['id'] == 'stpete'), roots=[str(root_a), str(root_b)])
        cfg['projects'] = [st]
        dash = fx.dash(cfg=cfg)
        runs, errors = dash.discover(st)
        assert [r['id'] for r in runs] == ['az6_cq'] and not errors
        assert os.path.normcase(runs[0]['dir']) == os.path.normcase(str(link))   # shown under the first root
        assert os.path.isdir(link) and os.path.isdir(run)                       # nothing removed
    finally:                      # remove the link itself (never a recursive delete through it)
        os.rmdir(link) if os.name == 'nt' else os.unlink(link)
    assert os.path.isfile(run / 'metrics.jsonl')


# ------------------------------------------------------------------ pieces

def test_check_creation_times():
    assert procs.check(100.0, started=100.4)
    assert not procs.check(100.0, started=103.0)
    assert procs.check(100.0, near=102.0)                    # created 2 s before it was logged
    assert not procs.check(100.0, near=2000.0)               # a much older process: pid reused
    assert not procs.check(100.0, near=50.0)                 # created after it was logged
    assert procs.check(False) is False and procs.check(None) is None


def test_psutil_liveness_on_this_process():
    import psutil
    lv = PsutilLiveness()
    ct = psutil.Process().create_time()
    assert lv.alive(os.getpid(), started=ct)
    assert not lv.alive(os.getpid(), started=ct - 100)
    assert lv.alive(os.getpid(), near=ct + 1)
    assert not lv.alive(4_000_000_000)


def test_supervisor_log_parsing():
    lines = ['2026-10-12 08:40:23 supervisor 138696 starting: E:/stpete-zero/runs/az6_cq --iters 100000',
             '2026-10-12 08:40:23 started stpz.run pid 93496',
             '2026-10-12 09:00:00 stpz.run exited with code 1; killed 2 orphaned workers',
             '2026-10-12 09:00:00 crash #1 in the last hour; restarting in 60s',
             '2026-10-12 09:01:00 started stpz.run pid 1234']
    info = runfiles.supervisor_info(runfiles.supervisor_events(lines))
    assert (info['sup_pid'], info['run_pid'], info['crashes_1h'], info['last_exit']) == (138696, 1234, 1, 1)
    assert info['last']['kind'] == 'run_start'
    assert info['sup_t'] == time.mktime(time.strptime('2026-10-12 08:40:23', '%Y-%m-%d %H:%M:%S'))


def test_log_events_and_flags():
    ref = time.mktime(time.strptime('2026-10-12 00:10:00', '%Y-%m-%d %H:%M:%S'))
    ev = runfiles.log_events(['23:59:00 iter 5 selfplay: {}', '00:05:00 iter 5 train: {}', 'garbage',
                              '00:06:00 STOP file found; exiting', '00:06:01 done'], ref)
    assert [e['what'] for e in ev] == ['selfplay', 'train', None, None]
    assert ev[0]['t'] == ref - 11 * 60                        # yesterday 23:59, not today
    f = runfiles.log_flags(ev)
    assert f['done'] and f['stop_found'] and not f['aborted']


def test_metrics_tail_tolerates_a_partial_line(tmp_path):
    p = tmp_path / 'metrics.jsonl'
    p.write_text('{"iter": 1, "time": 1, "eval_yard": {"win": 0.5, "se": 0.1}}\n{"iter": 2, "time": 2}\n{"iter": 3, "ti',
                 encoding='utf-8')
    s = runfiles.metrics_summary(str(tmp_path))
    assert s['last']['iter'] == 2 and s['evals']['eval_yard']['iter'] == 1


def test_state_labels_cover_all_states():
    assert set(state.LABELS) == set(state.SEVERITY)
    assert set(state.ACTIVE) <= set(state.LABELS)


def test_summaries_are_json(fx):
    for r in fx.runs().values():
        json.loads(json.dumps(r))
