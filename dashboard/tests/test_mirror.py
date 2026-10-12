"""St. Pete runs on the Mac seen through their desktop mirror (stpete-zero tools/mac_sync.py):
a run dir with MAC_SYNC + mac_state.json; state from mac_state.json (never from this machine's
processes); pause / resume by writing / removing the mirror's PAUSE (mac_sync forwards it)."""
import json
import os

from zerodash.summary import MIRROR_LAG_S, MIRROR_STALE_S

SUP = {'pid': 51234, 'started': 1791822000.5, 'since': 1791822001.0, 'state': 'running', 'run_pid': 51240,
       'args': [], 'updated': 1791822001.0}
RUN = {'pid': 51240, 'started': 1791822000.9, 'since': 1791822003.0, 'state': 'running', 'updated': 1791822003.0}


def make_mirror(fx, run='az6_cq', run_alive=True, sup_alive=True, checked_ago=8, rs=RUN, ss=SUP):
    d = fx.dir('stpete', run)
    with open(os.path.join(d, 'MAC_SYNC'), 'w', encoding='utf-8') as f:
        json.dump({'host': 'greggkeithley@100.95.248.116', 'remote': f'/Users/greggkeithley/stpete-zero/runs/{run}',
                   'ctl_mirrored': {'PAUSE': None, 'STOP': None}, 'last_pass': '2026-10-12 09:53:12'}, f)
    with open(os.path.join(d, 'mac_state.json'), 'w', encoding='utf-8') as f:
        json.dump({'run': rs, 'run_alive': run_alive, 'supervisor': ss, 'supervisor_alive': sup_alive,
                   'pause': False, 'stop': False, 'host': 'greggkeithley@100.95.248.116',
                   'remote': f'/Users/greggkeithley/stpete-zero/runs/{run}', 'checked': fx.now - checked_ago}, f)
    return d


def test_mirror_running_on_the_mac(fx):
    make_mirror(fx)
    r = fx.runs(fx.dash(alive={}))['stpete/az6_cq']       # nothing alive here: the Mac's state counts
    assert r['state'] == 'running' and r['run_alive'] and r['supervisor_alive'] and r['source'] == 'state files'
    assert r['remote']['kind'] == 'mirror' and r['remote']['host'] == 'greggkeithley@100.95.248.116'
    assert r['remote']['probe_age_s'] == 8 and r['remote']['error'] is None
    assert r['pause_supported'] and r['can_pause']          # a mirror honours PAUSE (pause: "auto")
    assert not fx.runs()['stpete/az5']['pause_supported']   # an old local run: STOP only


def test_mirror_pause_and_resume_write_the_mirror(fx):
    d = make_mirror(fx)
    dash = fx.dash(alive={})
    res = dash.pause('stpete', 'az6_cq', by='zerodash 100.64.0.7')
    assert res['changed'] and res['run']['state'] == 'pausing'
    with open(os.path.join(d, 'PAUSE'), encoding='utf-8') as f:
        assert json.load(f)['by'] == 'zerodash 100.64.0.7'
    # mac_sync forwards it; the Mac's run pauses and exits; the next pass says so
    make_mirror(fx, rs=dict(RUN, state='paused'), run_alive=False, ss=dict(SUP, state='paused', run_pid=None))
    assert fx.runs(dash)['stpete/az6_cq']['state'] == 'paused'
    res = dash.resume('stpete', 'az6_cq')
    assert res['changed'] and not os.path.exists(os.path.join(d, 'PAUSE'))
    assert res['run']['state'] == 'resuming'


def test_mirror_in_pause_all_and_resume_all(fx):
    d = make_mirror(fx)
    dash = fx.dash()
    res = dash.pause_all()
    assert 'stpete/az6_cq' in [x['key'] for x in res['paused']] and res['skipped'] == []
    assert dash.resume_all()['resumed'] and not os.path.exists(os.path.join(d, 'PAUSE'))


def test_mirror_with_a_stale_sync(fx):
    make_mirror(fx, checked_ago=MIRROR_STALE_S + 600)
    r = fx.runs()['stpete/az6_cq']
    assert r['run_alive'] is None and 'min ago' in r['remote']['error']
    r = fx.runs(fx.dash(now=fx.now + 5 * 3600))['stpete/az6_cq']
    assert r['state'] == 'unknown'


def test_mirror_without_mac_state_does_not_use_local_pids(fx):
    """The mirrored supervisor.log names the Mac's pids: never checked on this machine."""
    d = make_mirror(fx)
    os.remove(os.path.join(d, 'mac_state.json'))
    r = fx.runs()['stpete/az6_cq']                    # 138696 / 93496 are 'alive' in the snapshot liveness
    assert r['run_alive'] is None and r['supervisor_alive'] is None
    assert r['remote']['error'] == 'mac_sync has not written mac_state.json'


def test_mirror_reached_twice_is_shown_once(fx):
    """E:/stpete-zero/runs and the C: junctions to it are both roots on the desktop."""
    make_mirror(fx)
    st = next(p for p in fx.cfg['projects'] if p['id'] == 'stpete')
    st['roots'] = st['roots'] * 2
    runs = fx.dash().discover(st)[0]
    assert sorted(r['id'] for r in runs) == ['az5', 'az6_cq']


def test_mirror_lag_is_not_stale(fx):
    """Files arrive every ~10 min: a quiet stretch that long past a local run's limit is fine."""
    make_mirror(fx, checked_ago=9 * 60)
    local = fx.runs()['stpete/az5']['stale_after_s']
    r = fx.runs()['stpete/az6_cq']
    assert r['stale_after_s'] == local + MIRROR_LAG_S
    assert r['run_alive'] and r['remote']['error'] is None          # a 9-minute-old pass still counts


def test_mirror_last_pass_from_marker(fx):
    """No 'checked' in mac_state.json: the MAC_SYNC marker's last pass (local time) is used."""
    d = make_mirror(fx)
    with open(os.path.join(d, 'mac_state.json'), encoding='utf-8') as f:
        ms = json.load(f)
    ms.pop('checked')
    with open(os.path.join(d, 'mac_state.json'), 'w', encoding='utf-8') as f:
        json.dump(ms, f)
    r = fx.runs()['stpete/az6_cq']
    assert r['remote']['probe_age_s'] is not None
