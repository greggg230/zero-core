"""No private details in tracked files (tools/privacy_check.py, AGENTS.md "Private details"), and the machine's
own settings file (zero.local_config, tools/local_env.sh) read the same way from Python and bash."""
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

from zero import local_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('privacy_check', os.path.join(ROOT, 'tools', 'privacy_check.py'))
PC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PC)


def rules_hit(text):
    return sorted({r for _, r, _ in PC.scan_text(text, PC.GENERIC)})


def test_the_tracked_files_carry_no_private_details():
    """The whole checkout, with this machine's local patterns when it has them (a fresh clone: the generic set)."""
    if not os.path.isdir(os.path.join(ROOT, '.git')) and not os.path.isfile(os.path.join(ROOT, '.git')):
        pytest.skip('not a git checkout')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'privacy_check.py'), '--root', ROOT],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert ', 0 finding(s)' in r.stdout


@pytest.mark.parametrize('text,rule', [
    ('ssh 100.|101.102.103', 'tailscale-ip'),
    ('host pc-a.tail0a1b2c.|ts.net', 'tailscale-name'),
    ('pool-m_|ab12cd34ef56gh78jk90mn12pq_000.npz', 'farm-machine-id'),
    ('"C:/|Users/Jane Doe/mtg-zero"', 'windows-home-path'),
    (r'"C:\\|Users\\jdoe\\runs"', 'windows-home-path'),
    ('cd /c/|Users/jdoe/x', 'windows-home-path'),
    ("workdir='/Users/|jdoe/mtgz-remote'", 'mac-home-path'),
    ('played by DESKTOP-|AB12CD3', 'default-pc-hostname'),
    ('worker up on Janes-|MacBook-Pro.local', 'default-mac-hostname'),
    ('ssh jdoe@|10.1.2.3', 'ssh-user-at-ip'),
])
def test_each_generic_rule_fires(text, rule):
    # ('|' splits each token so that this file itself passes the check; it is removed here)
    assert rule in rules_hit(text.replace('|', ''))


@pytest.mark.parametrize('text', [
    'C:/Users/<user>/mtg-zero', '/Users/<user>/x', r'C:\Users\...\x', '/Users/user/x', 'C:/Users/user/runs',
    '%USERPROFILE%', 'user@mac', 'ssh user@192.0.2.10', 'zerodash 192.0.2.7', 'E:/mtg-zero/runs',
    'm_00000000000000000000000000', 'm_TEST0000000000000000000001', 'version 100.2.3', 'http://localhost:5230',
    'Squirrel token', 'C:/Users/Public/x',
])
def test_placeholders_and_ordinary_text_pass(text):
    assert rules_hit(text) == []


def test_local_patterns(tmp_path, monkeypatch):
    f = tmp_path / 'p.txt'
    f.write_text('# a comment\n\nsecret_host_[0-9]+\n', encoding='utf-8')
    monkeypatch.setenv(PC.LOCAL_ENV, str(f))
    local = PC.load_local(PC.local_patterns_path())
    assert len(local) == 1
    assert [r for _, r, _ in PC.scan_text('on SECRET_HOST_7 today', local)] == ['local pattern 3']
    f.write_text('bad(\n', encoding='utf-8')
    with pytest.raises(ValueError):
        PC.load_local(str(f))
    assert PC.load_local(str(tmp_path / 'absent.txt')) == []


def test_the_check_reports_findings_and_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(PC.LOCAL_ENV, str(tmp_path / 'absent.txt'))
    (tmp_path / 'a.md').write_text('fine\nssh jdoe@' + '10.9.8.7\n', encoding='utf-8')
    (tmp_path / 'b.bin').write_bytes(b'\0\0 jdoe@' + b'10.9.8.7')
    n, findings = PC.scan(str(tmp_path), ['a.md', 'b.bin', 'gone.txt'], PC.GENERIC)
    assert n == 1 and [(p, line, r) for p, line, r, _ in findings] == [('a.md', 2, 'ssh-user-at-ip')]
    assert PC.main(['--root', str(tmp_path), 'a.md']) == 1
    (tmp_path / 'c.md').write_text('nothing private\n', encoding='utf-8')
    assert PC.main(['--root', str(tmp_path), 'c.md']) == 0


LOCAL_ENV = '''# comment
ZERO_BIG_DISK=E:
export MTGZ_MAC="user@mac host"
  MTGZ_MAC_HOME = '/Users/user'
not a line
1BAD=x
EMPTY=
'''


def test_local_config_file_and_environment(tmp_path, monkeypatch):
    f = tmp_path / 'local.env'
    f.write_text(LOCAL_ENV, encoding='utf-8')
    monkeypatch.setenv(local_config.ENV_FILE, str(f))
    for k in ('ZERO_BIG_DISK', 'MTGZ_MAC', 'MTGZ_MAC_HOME', 'EMPTY'):
        monkeypatch.delenv(k, raising=False)
    assert local_config.load(refresh=True) == {'ZERO_BIG_DISK': 'E:', 'MTGZ_MAC': 'user@mac host',
                                               'MTGZ_MAC_HOME': '/Users/user', 'EMPTY': ''}
    assert local_config.get('MTGZ_MAC') == 'user@mac host'
    assert local_config.get('EMPTY', 'd') == 'd' and local_config.get('NOPE') is None
    assert local_config.big_disk_path('cargo-target', 'x') == 'E:/cargo-target/x'
    monkeypatch.setenv('ZERO_BIG_DISK', 'F:/')                  # the environment wins
    assert local_config.big_disk_path('a/') == 'F:/a'
    monkeypatch.setenv(local_config.ENV_FILE, str(tmp_path / 'absent.env'))
    monkeypatch.delenv('ZERO_BIG_DISK')
    assert local_config.load(refresh=True) == {} and local_config.big_disk_path('a', default='d') == 'd'


def test_portable_path():
    home = os.path.expanduser('~')
    assert local_config.portable_path(os.path.join(home, 'mtg-zero', 'a.pt')) == '~/mtg-zero/a.pt'
    assert local_config.portable_path('E:/x/a.pt') == 'E:/x/a.pt'
    assert local_config.portable_path(None) is None


@pytest.mark.skipif(shutil.which('bash') is None, reason='no bash')
def test_bash_reads_the_file_like_python(tmp_path):
    f = tmp_path / 'local.env'
    f.write_text(LOCAL_ENV.replace('\n', '\r\n'), encoding='utf-8')         # (CRLF too)
    env = {k: v for k, v in os.environ.items() if k not in ('ZERO_BIG_DISK', 'MTGZ_MAC', 'MTGZ_MAC_HOME')}
    env.update(ZERO_LOCAL_CONFIG=str(f).replace('\\', '/'), MTGZ_MAC_HOME='/from/env')
    script = '. tools/local_env.sh; printf "%s|%s|%s\\n" "$ZERO_BIG_DISK" "$MTGZ_MAC" "$MTGZ_MAC_HOME"'
    r = subprocess.run(['bash', '-c', script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == 'E:|user@mac host|/from/env'


def test_markers_spare_synthetic_data_from_the_generic_rules_only(tmp_path):
    """`privacy-check: allow` on a line or `privacy-check: allow-file` in a file's first 5 lines (a redactor and its
    tests); the machine's local patterns still apply there."""
    tok = 'ssh jdoe@' + '10.9.8.7'
    (tmp_path / 'a.md').write_text(f'fine\n{tok}\n{tok}  # privacy-check: allow\n', encoding='utf-8')
    (tmp_path / 'b.rs').write_text(f'// privacy-check: allow-file\n{tok}\nsecret_host_7\n', encoding='utf-8')
    local = [('local pattern 1', PC.re.compile('secret_host_[0-9]+', PC.re.I))]
    n, findings = PC.scan(str(tmp_path), ['a.md', 'b.rs'], list(PC.GENERIC) + local)
    assert n == 2 and [(p, line, r) for p, line, r, _ in findings] == [('a.md', 2, 'ssh-user-at-ip'),
                                                                       ('b.rs', 3, 'local pattern 1')]
