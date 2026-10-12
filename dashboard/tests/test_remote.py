"""Remote runs (St. Pete's az6 runs on the Mac) with SSH mocked.

`ShellSsh` stands in for `ssh host cmd`: it runs the command with a local POSIX sh (Git's sh
on Windows) and $HOME pointing at a temporary "Mac home", so the real probe / pause / resume
scripts and their quoting are exercised; the host's `ps` part is replaced by the processes the
test says are alive. Pure parsing tests need no shell.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

from conftest import posix_shell
from zerodash import remote
from zerodash.remote import RemoteRun, SshError, SshRunner, parse_etime, parse_probe, qpath

SH = posix_shell()
needs_sh = pytest.mark.skipif(SH is None, reason='no POSIX sh to stand in for the remote shell')
MAC = 'greggkeithley@100.95.248.116'


class ShellSsh:
    def __init__(self, home, alive=None, now=None):
        self.home = home.replace('\\', '/')
        self.alive = alive or {}          # pid -> creation time (host clock)
        self.now = now
        self.calls = []
        self.fail = None

    def run(self, host, command, timeout=None):
        self.calls.append((host, command))
        if self.fail:
            return 255, '', self.fail
        env = dict(os.environ, HOME=self.home)
        p = subprocess.run([SH, '-c', command], capture_output=True, text=True, env=env, timeout=30)
        out = p.stdout
        if '@@ps' in out:                 # the host's processes: as the test says
            now = self.now if self.now is not None else int(re.search(r'@@now (\d+)', out).group(1))
            ps = ''.join(f'{pid} {self.etime(now - ct)} python -m stpz.run\n' for pid, ct in self.alive.items())
            out = re.sub(r'@@ps\n.*?@@end\n', lambda m: '@@ps\n' + ps + '@@end\n', out, flags=re.S)
            if self.now is not None:
                out = re.sub(r'@@now \d+', f'@@now {int(self.now)}', out)
        return p.returncode, out, p.stderr

    @staticmethod
    def etime(s):
        s = int(round(s))
        d, s = divmod(s, 86400)
        return (f'{d}-' if d else '') + f'{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}'


def mac_home(fx, tmp_path, run=('glassroad', 'grz2p_cq_s512'), name='az6_cq'):
    """A fake Mac home with ~/stpete-zero/runs/<name> = a copy of a fixture run; the mirror is
    the fixture copy itself."""
    home = tmp_path / 'machome'
    dst = home / 'stpete-zero' / 'runs' / name
    shutil.copytree(fx.dir(*run), dst)
    return str(home), str(dst)


def remote_cfg(fx, mirror, pause=True, name='az6_cq'):
    st = next(p for p in fx.cfg['projects'] if p['id'] == 'stpete')
    st['remote_runs'] = [{'id': f'{name}@mac', 'name': name, 'label': 'Mac', 'ssh': MAC,
                          'path': f'~/stpete-zero/runs/{name}', 'mirror': mirror, 'pause': pause}]
    return fx.cfg


# ------------------------------------------------------------------ pure parts

def test_ssh_argv_is_batch_mode_with_timeout():
    argv = SshRunner(exe='ssh').argv(MAC, 'uptime')
    assert argv == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', MAC, 'uptime']


def test_ssh_runner_without_ssh(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    with pytest.raises(SshError, match='not installed'):
        SshRunner().run(MAC, 'uptime')


def test_quoting():
    assert qpath('~/stpete-zero/runs/az6_cq') == '"$HOME"/stpete-zero/runs/az6_cq'
    assert qpath("~/my runs/it's") == '"$HOME"/\'my runs/it\'"\'"\'s\''
    assert qpath('/Users/g/x y') == "'/Users/g/x y'"
    assert qpath('~') == '"$HOME"'


def test_parse_etime():
    assert parse_etime('05:07') == 307
    assert parse_etime('01:02:03') == 3723
    assert parse_etime('2-01:00:00') == 2 * 86400 + 3600
    assert parse_etime('bogus') is None


def test_parse_probe():
    out = ('@@now 1791824000\n@@file PAUSE\n{"time": 1.0, "by": "zerodash pause-all x"}\n@@end\n'
           '@@file STOP\n\n@@end\n@@suplog\n2026-10-12 08:40:23 supervisor 138696 starting: x\n@@end\n'
           '@@ps\n  93496    01:13:00 /usr/bin/python3 -m stpz.run\n138696 1-00:00:00 python tools/supervise.py\n@@end\n')
    p = parse_probe(out)
    assert p['now'] == 1791824000 and not p['nodir']
    assert json.loads(p['files']['PAUSE'])['by'] == 'zerodash pause-all x'
    assert p['files']['STOP'] == ''
    assert p['suplog'] == ['2026-10-12 08:40:23 supervisor 138696 starting: x']
    assert p['procs'] == {93496: 1791824000 - 4380, 138696: 1791824000 - 86400}
    assert parse_probe('@@nodir\n')['nodir']


def test_remote_liveness():
    lv = remote.RemoteLiveness({'procs': {10: 1000.0}})
    assert lv.alive(10, started=1001.5) and not lv.alive(10, started=1010) and not lv.alive(11)
    assert remote.RemoteLiveness(None).alive(10) is None


def test_unreachable_host_falls_back_to_the_mirror(fx, tmp_path):
    """No answer: control state from the mirror's files, liveness unknown, the error shown."""
    mirror = fx.dir('glassroad', 'grz2p_cq')          # has a PAUSE
    cfg = remote_cfg(fx, mirror)
    ssh = ShellSsh(str(tmp_path))
    ssh.fail = 'ssh: connect to host 100.95.248.116 port 22: Operation timed out'
    d = fx.dash(cfg=cfg, ssh=ssh)
    for rr in d.remote.values():
        rr.refresh()
    r = fx.runs(d)['stpete/az6_cq@mac']
    assert r['state'] == 'paused' and r['run_alive'] is None
    assert r['remote']['error'].endswith('Operation timed out') and r['remote']['host'] == MAC
    assert 'grz2p_cq' not in [k.split('/')[1] for k in fx.runs(d) if k.startswith('glassroad/')]  # mirror shown once
    with pytest.raises(SshError):
        d.resume('stpete', 'az6_cq@mac')
    assert os.path.exists(os.path.join(mirror, 'PAUSE'))       # the mirror is never written


# ------------------------------------------------------------------ through a real sh

@needs_sh
def test_probe_reads_the_host(fx, tmp_path):
    home, rdir = mac_home(fx, tmp_path)
    ssh = ShellSsh(home, alive={145772: 1791820497.7519355, 150264: 1791820497.9998765}, now=fx.now)
    rr = RemoteRun({'ssh': MAC, 'path': '~/stpete-zero/runs/az6_cq', 'label': 'Mac'}, ssh, clock=lambda: fx.now)
    p = rr.refresh()
    assert rr.error is None
    assert set(p['files']) == {'run_state.json', 'supervisor_state.json', 'progress.json'}
    assert json.loads(p['files']['progress.json'])['games'] == 2941
    assert p['suplog'][-1].endswith('started grz.az.run pid 150264')
    assert set(p['procs']) == {145772, 150264}
    host, cmd = ssh.calls[-1]
    assert host == MAC and 'cd -- "$HOME"/stpete-zero/runs/az6_cq' in cmd


@needs_sh
def test_missing_remote_dir(fx, tmp_path):
    rr = RemoteRun({'ssh': MAC, 'path': '~/nope/az6_cq'}, ShellSsh(str(tmp_path)), clock=lambda: fx.now)
    rr.refresh()
    assert 'no such directory' in rr.error


@needs_sh
def test_remote_run_state_from_the_probe(fx, tmp_path):
    """The mirror lags (no PAUSE in it), the host has one: the probe wins."""
    home, rdir = mac_home(fx, tmp_path)
    mirror = fx.dir('glassroad', 'grz2p_cq_s512')
    alive = {145772: 1791820497.7519355, 150264: 1791820497.9998765}
    ssh = ShellSsh(home, alive=alive, now=fx.now)
    d = fx.dash(cfg=remote_cfg(fx, mirror), ssh=ssh)
    rr = d.remote[('stpete', 'az6_cq@mac')]
    rr.refresh()
    r = fx.runs(d)['stpete/az6_cq@mac']
    assert r['state'] == 'running' and r['run_alive'] and r['supervisor_alive'] and r['can_pause']
    assert r['remote']['label'] == 'Mac' and r['remote']['error'] is None
    assert r['iteration']['phase_detail'] == '2,941 / 3,000 games'
    with open(os.path.join(rdir, 'PAUSE'), 'w', encoding='utf-8') as f:
        json.dump({'time': fx.now, 'by': 'tools/pause_run.py'}, f)
    rr.refresh()
    r = fx.runs(d)['stpete/az6_cq@mac']
    assert r['state'] == 'pausing' and r['pause']['by'] == 'tools/pause_run.py'
    ssh.alive = {145772: alive[145772]}                 # the run process exited: paused, supervisor waits
    rr.refresh()
    r = fx.runs(d)['stpete/az6_cq@mac']
    assert r['state'] == 'paused' and r['supervisor_alive']


@needs_sh
def test_remote_pause_resume_over_ssh(fx, tmp_path):
    home, rdir = mac_home(fx, tmp_path)
    mirror = fx.dir('glassroad', 'grz2p_cq_s512')
    ssh = ShellSsh(home, alive={145772: 1791820497.7519355, 150264: 1791820497.9998765}, now=fx.now)
    d = fx.dash(cfg=remote_cfg(fx, mirror), ssh=ssh)
    d.remote[('stpete', 'az6_cq@mac')].refresh()
    res = d.pause('stpete', 'az6_cq@mac', by='zerodash 100.64.0.7')
    assert res['changed'] and res['run']['state'] == 'pausing'
    with open(os.path.join(rdir, 'PAUSE'), encoding='utf-8') as f:
        assert json.load(f) == {'time': fx.now, 'by': 'zerodash 100.64.0.7'}
    assert not os.path.exists(os.path.join(mirror, 'PAUSE'))          # written on the host only
    assert not [f for f in os.listdir(rdir) if 'tmp' in f]
    assert not d.pause('stpete', 'az6_cq@mac', by='zerodash other')['changed']   # an existing PAUSE stays
    res = d.resume('stpete', 'az6_cq@mac')
    assert res['changed'] and not os.path.exists(os.path.join(rdir, 'PAUSE'))
    assert not d.resume('stpete', 'az6_cq@mac')['changed']


@needs_sh
def test_remote_stop_only_run_is_not_paused(fx, tmp_path):
    home, rdir = mac_home(fx, tmp_path)
    ssh = ShellSsh(home, alive={145772: 1791820497.7519355, 150264: 1791820497.9998765}, now=fx.now)
    d = fx.dash(cfg=remote_cfg(fx, fx.dir('glassroad', 'grz2p_cq_s512'), pause=False), ssh=ssh)
    d.remote[('stpete', 'az6_cq@mac')].refresh()
    res = d.pause_all()
    assert 'stpete/az6_cq@mac' in [x['key'] for x in res['skipped']]
    assert not os.path.exists(os.path.join(rdir, 'PAUSE'))


@needs_sh
def test_remote_pause_all_resume_all(fx, tmp_path):
    home, rdir = mac_home(fx, tmp_path)
    ssh = ShellSsh(home, alive={145772: 1791820497.7519355, 150264: 1791820497.9998765}, now=fx.now)
    d = fx.dash(cfg=remote_cfg(fx, fx.dir('glassroad', 'grz2p_cq_s512')), ssh=ssh)
    rr = d.remote[('stpete', 'az6_cq@mac')]
    rr.refresh()
    res = d.pause_all()
    assert 'stpete/az6_cq@mac' in [x['key'] for x in res['paused']]
    with open(os.path.join(rdir, 'PAUSE'), encoding='utf-8') as f:
        assert json.load(f)['by'] == res['by']
    res2 = d.resume_all()
    assert 'stpete/az6_cq@mac' in [x['key'] for x in res2['resumed']]
    assert not os.path.exists(os.path.join(rdir, 'PAUSE'))
    # a hand pause on the host survives Resume all
    with open(os.path.join(rdir, 'PAUSE'), 'w', encoding='utf-8') as f:
        f.write('{"time": 5, "by": "tools/pause_run.py"}')
    res3 = d.resume_all()
    assert {'key': 'stpete/az6_cq@mac', 'by': 'tools/pause_run.py'} in res3['kept']
    assert os.path.exists(os.path.join(rdir, 'PAUSE'))


@needs_sh
def test_remote_resume_all_keeps_a_pause_replaced_meanwhile(fx, tmp_path):
    """Resume all removes the host's PAUSE only if it still holds what was read."""
    home, rdir = mac_home(fx, tmp_path)
    rr = RemoteRun({'ssh': MAC, 'path': '~/stpete-zero/runs/az6_cq'}, ShellSsh(home), clock=lambda: fx.now)
    pa = '{"time": 1, "by": "zerodash pause-all 20261012-090000-abcd"}'
    assert rr.write_pause(pa)
    assert rr.read_pause() == pa
    with open(os.path.join(rdir, 'PAUSE'), 'w', encoding='utf-8') as f:
        f.write('{"time": 2, "by": "tools/pause_run.py"}')
    assert not rr.remove_pause(expect=pa)
    assert os.path.exists(os.path.join(rdir, 'PAUSE'))
    assert rr.remove_pause(expect='{"time": 2, "by": "tools/pause_run.py"}')
    assert rr.read_pause() is None


@needs_sh
def test_paths_with_spaces_and_quotes(fx, tmp_path):
    home, rdir = mac_home(fx, tmp_path, name="az6 it's")
    rr = RemoteRun({'ssh': MAC, 'path': "~/stpete-zero/runs/az6 it's"}, ShellSsh(home), clock=lambda: fx.now)
    assert rr.write_pause('{"time": 1, "by": "zerodash x"}')
    assert os.path.exists(os.path.join(rdir, 'PAUSE'))
    assert rr.refresh() and rr.error is None
