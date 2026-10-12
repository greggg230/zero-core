"""The Flask app: page, JSON shape of /api/state, the control endpoints."""
import json
import os

import pytest

from zerodash.app import create_app
from zerodash.machine import Machine

RUN_KEYS = {
    'key': str, 'project': str, 'id': str, 'name': str, 'state': str, 'label': str, 'severity': str,
    'detail': str, 'hidden': bool, 'pause_supported': bool, 'stop': bool, 'can_pause': bool, 'can_resume': bool,
    'iteration': dict, 'stale_after_s': int, 'source': str, 'crashes_1h': int,
}
STATES = {'running', 'pausing', 'paused', 'resuming', 'starting', 'restarting', 'stopping', 'stopped', 'crashed',
          'stale', 'unknown'}


class FakeMachine:
    def start(self):
        pass

    def snapshot(self):
        return {'gpu': {'error': 'nvidia-smi not found'}, 'cpu': {'percent': 12.5, 'cores': 8, 'mem_percent': 40.0,
                'mem_used_gb': 3.2, 'mem_total_gb': 8.0}, 'disks': [], 'mac': None, 'sampled': 1.0, 'age_s': 0.5}


@pytest.fixture
def client(fx):
    dash = fx.dash()
    dash.machine = FakeMachine()
    return create_app(dash).test_client()


def post(client, url, body={'confirm': True}):
    return client.post(url, data=json.dumps(body) if body is not None else None,
                       content_type='application/json' if body is not None else None)


def test_page_and_static(client):
    r = client.get('/')
    assert r.status_code == 200 and b'Zero dashboard' in r.data and b'static/app.js' in r.data
    for f in ('static/app.js', 'static/style.css'):
        assert client.get('/' + f).status_code == 200
    assert client.get('/healthz').get_json() == {'ok': True}


def test_state_shape(client, fx):
    j = client.get('/api/state').get_json()
    assert set(j) >= {'now', 'title', 'refresh_s', 'hide_after_days', 'projects', 'pause_all', 'machine'}
    assert j['now'] == fx.now and j['refresh_s'] == 15
    assert [p['id'] for p in j['projects']] == ['stpete', 'dominion', 'caverna', 'glassroad']
    links = {p['id']: p['dashboard'] for p in j['projects']}
    assert links == {'stpete': 'http://desktop-j05412i:5230', 'dominion': 'http://desktop-j05412i:5240',
                     'caverna': 'http://desktop-j05412i:5260', 'glassroad': 'http://desktop-j05412i:5280'}
    assert j['pause_all'] == {'ids': [], 'runs': []}
    assert j['machine']['cpu']['percent'] == 12.5
    for p in j['projects']:
        assert set(p) >= {'id', 'name', 'dashboard', 'pause_supported', 'pause_note', 'runs', 'errors', 'hidden_count'}
        assert p['errors'] == []
        for r in p['runs']:
            for k, t in RUN_KEYS.items():
                assert isinstance(r[k], t), (r['key'], k, r[k])
            assert r['state'] in STATES and r['key'] == f"{p['id']}/{r['id']}"
            assert set(r['iteration']) >= {'last', 'last_time', 'current', 'phase', 'phase_since', 'phase_detail',
                                           'phase_frac', 'live'}
            if r['eval']:
                assert set(r['eval']) >= {'key', 'label', 'iter', 'win', 'se', 'margin'}
            if r['elo']:
                assert set(r['elo']) >= {'iter', 'elo', 'se', 'best', 'series'}
                assert all(len(pt) == 2 for pt in r['elo']['series'])
            if r['throughput']:
                assert set(r['throughput']) == {'iter', 'games_per_h', 'leaves_per_s'}
            if r['pause']:
                assert set(r['pause']) == {'time', 'by', 'kind'}


def test_controls_need_a_confirmed_json_body(client, fx):
    url = '/api/run/glassroad/grz2p_cq_s512/pause'
    assert client.post(url).status_code == 400                                  # no body
    assert client.post(url, data='confirm=true', content_type='application/x-www-form-urlencoded').status_code == 400
    assert post(client, url, {'confirm': False}).status_code == 400
    assert post(client, url, {}).status_code == 400
    for u in ('/api/pause-all', '/api/resume-all'):
        assert client.post(u).status_code == 400
    assert not os.path.exists(fx.path('glassroad', 'grz2p_cq_s512', 'PAUSE'))
    assert client.get(url).status_code == 405


def test_pause_resume_endpoints(client, fx):
    r = post(client, '/api/run/glassroad/grz2p_cq_s512/pause')
    j = r.get_json()
    assert r.status_code == 200 and j['ok'] and j['changed'] and j['run']['state'] == 'pausing'
    with open(fx.path('glassroad', 'grz2p_cq_s512', 'PAUSE'), encoding='utf-8') as f:
        assert json.load(f)['by'] == 'zerodash 127.0.0.1'
    r = post(client, '/api/run/glassroad/grz2p_cq_s512/resume')
    assert r.status_code == 200 and r.get_json()['changed']
    assert not os.path.exists(fx.path('glassroad', 'grz2p_cq_s512', 'PAUSE'))


def test_errors(client, fx):
    assert post(client, '/api/run/glassroad/nope/pause').status_code == 404
    assert post(client, '/api/run/glassroad/grz2p_cq/explode').status_code == 404
    assert post(client, '/api/run/glassroad/..%2F..%2Fcaverna%2Fruns%2Fcvz2p/resume').status_code == 404
    r = post(client, '/api/run/stpete/az6_cq/pause')
    assert r.status_code == 409 and 'STOP only' in r.get_json()['error']
    assert os.path.exists(fx.path('caverna', 'cvz2p', 'PAUSE'))


def test_pause_all_resume_all_endpoints(client, fx):
    j = post(client, '/api/pause-all').get_json()
    assert j['ok'] and len(j['paused']) == 2 and j['skipped'] == [{'key': 'stpete/az6_cq', 'why': 'no pause (STOP only)'}]
    st = client.get('/api/state').get_json()
    assert st['pause_all']['ids'] == [j['id']] and len(st['pause_all']['runs']) == 2
    j2 = post(client, '/api/resume-all').get_json()
    assert j2['ok'] and len(j2['resumed']) == 2 and len(j2['kept']) == 4
    assert client.get('/api/state').get_json()['pause_all']['runs'] == []


def test_machine_strip_degrades_without_tools(fx, monkeypatch):
    """No nvidia-smi, no psutil, a missing drive, no ssh: each part reports its error."""
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    cfg = dict(fx.cfg, disks=[{'path': 'Q:/nonexistent-drive/', 'label': 'Q:', 'warn_below_gb': 15}])
    m = Machine(cfg, ssh=None)
    m.psutil = None
    m.sample()
    m.sample_mac_now()
    s = m.snapshot()
    assert s['gpu'] == {'error': 'nvidia-smi not found'}
    assert s['cpu'] == {'error': 'psutil not installed'}
    assert 'error' in s['disks'][0] and s['mac']['error'] == 'no ssh'
    dash = fx.dash()
    dash.machine = m
    j = create_app(dash).test_client().get('/api/state').get_json()
    assert j['machine']['gpu']['error'] == 'nvidia-smi not found'
