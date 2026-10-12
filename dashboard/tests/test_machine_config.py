"""Machine strip parsing / sampling with the tools mocked, and projects.json loading."""
import json
import os

import pytest

from zerodash import DEFAULT_CONFIG, FIXTURES_CONFIG, config
from zerodash.machine import GPU_QUERY, Machine, parse_gpu, parse_uptime


def test_parse_gpu():
    assert parse_gpu('5727, 32607, 148.43, 85\n') == [
        {'mem_used_mb': 5727.0, 'mem_total_mb': 32607.0, 'power_w': 148.43, 'util_pct': 85.0}]
    two = parse_gpu('1, 2, [N/A], 3\n4, 5, 6.5, 7\n')
    assert len(two) == 2 and two[0]['power_w'] is None
    assert parse_gpu('') == []


def test_parse_uptime():
    assert parse_uptime('10:01  up 3 days,  2:03, 2 users, load averages: 2.31 2.10 1.95') == [2.31, 2.10, 1.95]
    assert parse_uptime(' 10:01:02 up 1 day, load average: 0.00, 0.01, 0.05') == [0.0, 0.01, 0.05]
    assert parse_uptime('10:01  up 3 days, 2 users, load averages: 2,31 2,10 1,95') == [2.31, 2.10, 1.95]
    assert parse_uptime('garbage') is None


class FakeSsh:
    def __init__(self, result):
        self.result, self.calls = result, []

    def run(self, host, command, timeout=None):
        self.calls.append((host, command))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_machine_sample_with_mocked_tools(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: 'C:/fake/' + name + '.exe')
    calls = []

    def runner(argv, timeout=10):
        calls.append(argv)
        return 0, '5727, 32607, 148.43, 85\n'
    cfg = {'disks': [{'path': str(tmp_path), 'label': 'T:', 'warn_below_gb': 10 ** 9}],
           'mac': {'ssh': 'greggkeithley@100.95.248.116', 'label': 'Mac'}}
    ssh = FakeSsh((0, '10:01  up 3 days, load averages: 2.31 2.10 1.95\n', ''))
    m = Machine(cfg, runner=runner, ssh=ssh, clock=lambda: 1000.0)
    m.sample()
    m.sample_mac_now()
    s = m.snapshot()
    assert calls == [['C:/fake/nvidia-smi.exe'] + GPU_QUERY]
    assert s['gpu']['gpus'][0]['util_pct'] == 85.0
    assert s['disks'][0]['warn'] and s['disks'][0]['free_gb'] > 0           # under the (huge) threshold
    assert s['mac']['load'] == [2.31, 2.10, 1.95] and ssh.calls == [('greggkeithley@100.95.248.116', 'uptime')]
    assert s['age_s'] == 0.0 and s['cpu']['cores'] >= 1
    # the Mac stops answering: the error, and the last load seen
    ssh.result = (255, '', 'ssh: connect to host 100.95.248.116 port 22: Operation timed out')
    m.sample_mac_now()
    mac = m.snapshot()['mac']
    assert mac['error'].endswith('Operation timed out') and mac['last_load'] == [2.31, 2.10, 1.95]


def test_nvidia_smi_failing(monkeypatch):
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, 'which', lambda name: 'nvidia-smi')

    def boom(argv, timeout=10):
        raise subprocess.TimeoutExpired(argv, timeout)
    assert 'error' in Machine({}, runner=boom).sample_gpu()
    assert Machine({}, runner=lambda a, timeout=10: (9, '')).sample_gpu() == {'error': 'nvidia-smi exited with status 9'}


def test_desktop_config_loads():
    cfg = config.load(DEFAULT_CONFIG)
    ids = [p['id'] for p in cfg['projects']]
    assert ids == ['stpete', 'dominion', 'caverna', 'glassroad']
    st = cfg['projects'][0]
    assert st['pause'] == 'auto' and st['roots'][0].replace('\\', '/') == 'C:/Users/Gregg Keithley/stpete-zero/runs'
    assert {p['id']: p['pause'] for p in cfg['projects'][1:]} == {'dominion': True, 'caverna': True, 'glassroad': True}
    assert {p['id']: p['dashboard'].rsplit(':', 1)[1] for p in cfg['projects']} == \
        {'stpete': '5230', 'dominion': '5240', 'caverna': '5260', 'glassroad': '5280'}
    assert cfg['mac']['ssh'] == 'greggkeithley@100.95.248.116'
    assert cfg['disks'][0]['warn_below_gb'] == 15


def test_fixtures_config_is_relative():
    cfg = config.load(FIXTURES_CONFIG)
    for p in cfg['projects']:
        assert os.path.isdir(p['roots'][0]), p['roots']


def test_remote_run_config(tmp_path):
    raw = {'mac': {'ssh': 'greggkeithley@100.95.248.116'}, 'projects': [{
        'id': 'stpete', 'roots': ['runs'], 'pause': 'auto',
        'remote_runs': [{'name': 'az6_cq', 'path': '~/stpete-zero/runs/az6_cq', 'mirror': 'mirror/az6_cq'}]}]}
    cfg = config.normalize(raw, str(tmp_path))
    rr = cfg['projects'][0]['remote_runs'][0]
    assert rr['id'] == 'az6_cq@mac' and rr['ssh'] == 'greggkeithley@100.95.248.116' and rr['pause'] is True
    assert rr['mirror'] == os.path.normpath(os.path.join(str(tmp_path), 'mirror/az6_cq'))
    bad = json.loads(json.dumps(raw))
    bad['projects'][0]['remote_runs'][0]['ssh'] = '-oProxyCommand=evil'
    with pytest.raises(config.ConfigError):
        config.normalize(bad, str(tmp_path))
    bad['projects'][0]['remote_runs'][0] = {'name': 'x'}
    with pytest.raises(config.ConfigError):
        config.normalize(bad, str(tmp_path))
    with pytest.raises(config.ConfigError):
        config.normalize({'projects': [{'id': 'a'}, {'id': 'a'}]}, str(tmp_path))
    with pytest.raises(config.ConfigError):
        config.normalize({'projects': [{'id': 'a', 'pause': 'sometimes'}]}, str(tmp_path))
