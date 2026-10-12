"""Pause / Resume / Pause all / Resume all on a temporary copy of the fixtures."""
import glob
import json
import os
import re

import pytest

from zerodash.control import ControlError, write_pause_local
from zerodash.model import NotFound

HAND_PAUSED = ('glassroad/grz2p_cq', 'caverna/cvz2p', 'dominion/all1', 'dominion/rand6')


def control_files(fx):
    """Every PAUSE / STOP / STOP.old under the copy -> {relative path: content}."""
    out = {}
    for name in ('PAUSE', 'STOP', 'STOP.old'):
        for p in glob.glob(os.path.join(fx.root, 'fixtures', '*', 'runs', '*', name)):
            with open(p, encoding='utf-8') as f:
                out[os.path.relpath(p, fx.root).replace('\\', '/')] = f.read()
    return out


def pause_of(fx, key):
    p, r = key.split('/')
    try:
        return json.loads(fx.read(p, r, 'PAUSE'))
    except FileNotFoundError:
        return None


def test_pause_and_resume_one_run(fx):
    d = fx.dash()
    res = d.pause('glassroad', 'grz2p_cq_s512', by='zerodash 100.64.0.7')
    assert res['ok'] and res['changed'] and res['run']['state'] == 'pausing'
    assert pause_of(fx, 'glassroad/grz2p_cq_s512') == {'time': fx.now, 'by': 'zerodash 100.64.0.7'}
    assert not glob.glob(fx.path('glassroad', 'grz2p_cq_s512', 'PAUSE.*'))          # no temp file left
    # pausing again keeps the original pause
    res = d.pause('glassroad', 'grz2p_cq_s512', by='zerodash 100.64.0.9')
    assert not res['changed'] and pause_of(fx, 'glassroad/grz2p_cq_s512')['by'] == 'zerodash 100.64.0.7'
    res = d.resume('glassroad', 'grz2p_cq_s512')
    assert res['changed'] and pause_of(fx, 'glassroad/grz2p_cq_s512') is None
    assert res['run']['state'] == 'running'
    assert not d.resume('glassroad', 'grz2p_cq_s512')['changed']


def test_resume_removes_a_hand_pause(fx):
    res = fx.dash().resume('glassroad', 'grz2p_cq')
    assert res['changed'] and pause_of(fx, 'glassroad/grz2p_cq') is None
    assert res['run']['state'] == 'resuming'                  # its supervisor (alive) relaunches it


def test_stop_only_project_is_never_written(fx):
    d = fx.dash()
    with pytest.raises(ControlError, match='STOP only'):
        d.pause('stpete', 'az6_cq', by='zerodash x')
    assert not os.path.exists(fx.path('stpete', 'az6_cq', 'PAUSE'))


def test_unknown_runs(fx):
    d = fx.dash()
    for pid, rid in (('glassroad', 'nope'), ('nope', 'grz2p_cq'), ('glassroad', '..'), ('glassroad', '../caverna')):
        with pytest.raises(NotFound):
            d.pause(pid, rid, by='x')


def test_per_run_pause_flag_flip(fx):
    """St. Pete's PAUSE support goes live per run (or per project) with a config flag."""
    st = next(p for p in fx.cfg['projects'] if p['id'] == 'stpete')
    st['runs'] = {'az6_cq': {'pause': True}}
    runs = fx.runs()
    assert runs['stpete/az6_cq']['can_pause'] and not runs['stpete/az5']['pause_supported']
    assert fx.dash().pause('stpete', 'az6_cq', by='zerodash x')['changed']


def test_pause_all_then_resume_all(fx):
    before = control_files(fx)
    d = fx.dash()
    res = d.pause_all()
    assert res['ok'] and re.match(r'^\d{8}-\d{6}-[0-9a-f]{4}$', res['id'])
    assert res['by'] == 'zerodash pause-all ' + res['id']
    assert sorted(x['key'] for x in res['paused']) == ['caverna/cvz2p_cq', 'glassroad/grz2p_cq_s512']
    assert [x['key'] for x in res['skipped']] == ['stpete/az6_cq']           # STOP only
    assert not os.path.exists(fx.path('stpete', 'az6_cq', 'PAUSE'))
    assert not os.path.exists(fx.path('stpete', 'az5', 'PAUSE'))             # stopped: not touched
    for key in ('caverna/cvz2p_cq', 'glassroad/grz2p_cq_s512'):
        assert pause_of(fx, key) == {'time': fx.now, 'by': res['by']}
    after = control_files(fx)
    for k, v in before.items():                    # hand pauses, STOP, STOP.old untouched
        assert after[k] == v, k
    runs = fx.runs()
    assert runs['caverna/cvz2p_cq']['state'] == 'pausing' and runs['caverna/cvz2p_cq']['pause']['kind'] == 'pause-all'
    snap = d.snapshot()
    assert snap['pause_all'] == {'ids': [res['id']], 'runs': ['caverna/cvz2p_cq', 'glassroad/grz2p_cq_s512']}

    res2 = d.resume_all()
    assert res2['ok'] and sorted(x['key'] for x in res2['resumed']) == ['caverna/cvz2p_cq', 'glassroad/grz2p_cq_s512']
    assert sorted(x['key'] for x in res2['kept']) == sorted(HAND_PAUSED)
    assert control_files(fx) == before             # exactly the state before pause-all
    assert d.snapshot()['pause_all'] == {'ids': [], 'runs': []}


def test_resume_all_leaves_hand_pauses_alone(fx):
    d = fx.dash()
    d.pause_all()
    # meanwhile: one run paused by hand (a fresh PAUSE by another tool), one by a zerodash Pause button
    os.remove(fx.path('caverna', 'cvz2p_cq', 'PAUSE'))
    fx.write_json('caverna', 'cvz2p_cq', 'PAUSE', {'time': fx.now + 60, 'by': 'tools/pause_run.py'})
    os.remove(fx.path('glassroad', 'grz2p_cq_s512', 'PAUSE'))
    d.pause('glassroad', 'grz2p_cq_s512', by='zerodash 100.64.0.7')
    res = d.resume_all()
    assert res['resumed'] == []
    assert pause_of(fx, 'caverna/cvz2p_cq')['by'] == 'tools/pause_run.py'
    assert pause_of(fx, 'glassroad/grz2p_cq_s512')['by'] == 'zerodash 100.64.0.7'
    for key in HAND_PAUSED:
        assert pause_of(fx, key) is not None, key


def test_resume_all_takes_any_pause_all(fx):
    """Two pause-alls (e.g. one before a reboot, one after): Resume all clears both."""
    fx.write_json('glassroad', 'grz2p_cq', 'PAUSE', {'time': fx.now - 99, 'by': 'zerodash pause-all 20261011-080000-abcd'})
    d = fx.dash()
    d.pause_all()
    res = d.resume_all()
    assert 'glassroad/grz2p_cq' in [x['key'] for x in res['resumed']]
    assert len(res['resumed']) == 3


def test_pause_all_skips_runs_already_paused_and_inactive(fx):
    d = fx.dash(alive={})                        # nothing alive: nothing to pause
    res = d.pause_all()
    assert res['paused'] == [] and res['skipped'] == []
    assert not os.path.exists(fx.path('caverna', 'cvz2p_cq', 'PAUSE'))


def test_pause_without_hard_links_falls_back(fx, monkeypatch):
    def nolink(a, b):
        raise OSError(1, 'hard links not supported')
    monkeypatch.setattr(os, 'link', nolink)
    d = fx.dir('glassroad', 'grz2p_cq_s512')
    assert write_pause_local(d, 'zerodash t', 1.0)
    assert not write_pause_local(d, 'zerodash t2', 2.0)
    assert json.loads(fx.read('glassroad', 'grz2p_cq_s512', 'PAUSE')) == {'time': 1.0, 'by': 'zerodash t'}
    assert not [f for f in os.listdir(d) if f.endswith('.tmp')]


def test_existing_pause_is_never_replaced_by_a_link(fx):
    d = fx.dir('caverna', 'cvz2p')
    before = fx.read('caverna', 'cvz2p', 'PAUSE')
    assert not write_pause_local(d, 'zerodash x', 5.0)
    assert fx.read('caverna', 'cvz2p', 'PAUSE') == before
