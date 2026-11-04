"""tools/smoke_all.sh's game discovery (docs/az.md 29): from a git worktree, and from a zero-core
submodule inside a worktree of its superproject (how MTG Zero's tiered gate runs it), it finds the
game repos beside the main checkouts; finding none fails loudly instead of printing GREEN. Only
--dry-run and the no-game exit are exercised (nothing is built), in throwaway git repos."""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, 'tools', 'smoke_all.sh')
GAMES = ('glassroad-zero', 'caverna-zero', 'kingdombuilder-zero')


def _bash():
    if sys.platform == 'win32':   # never WSL's System32\bash.exe
        git_bash = r'C:\Program Files\Git\bin\bash.exe'
        return git_bash if os.path.exists(git_bash) else None
    return shutil.which('bash')


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None or shutil.which('git') is None, reason='needs bash and git')


def git(*args, cwd):
    subprocess.run(['git', '-c', 'protocol.file.allow=always', '-c', 'user.name=t', '-c', 'user.email=t@t',
                    '-c', 'init.defaultBranch=main', *args], cwd=cwd, check=True, capture_output=True)


def smoke(script, *args, **env):
    e = dict(os.environ)
    for k in ('SMOKE_GAMES_ROOT', 'SKIP_CORE', 'SKIP_GAMES'):
        e.pop(k, None)
    e.update(env)
    p = subprocess.run([BASH, script, *args], capture_output=True, text=True, env=e, timeout=120)
    return p.returncode, p.stdout + p.stderr


def found(out):
    return [os.path.basename(line.split('game: ', 1)[1].strip()) for line in out.splitlines()
            if line.strip().startswith('game: ')]


@pytest.fixture
def layout(tmp_path):
    """tmp/zero-core (a repo holding a copy of smoke_all.sh) and two of the three games beside it."""
    core = tmp_path / 'zero-core'
    (core / 'tools').mkdir(parents=True)
    shutil.copy(SCRIPT, core / 'tools' / 'smoke_all.sh')
    git('init', '-q', cwd=core)
    git('add', '-A', cwd=core)
    git('commit', '-q', '-m', 'core', cwd=core)
    for g in GAMES[:2]:
        (tmp_path / g / 'third_party' / 'zero-core').mkdir(parents=True)
    return tmp_path


def test_finds_the_games_beside_the_main_checkout(layout):
    """The old rule (../<game>), unchanged."""
    rc, out = smoke(str(layout / 'zero-core' / 'tools' / 'smoke_all.sh'), '--dry-run')
    assert rc == 0 and found(out) == list(GAMES[:2]), out


def test_finds_the_games_from_a_worktree(layout):
    wt = layout / 'zero-core-wt' / 'x'
    git('worktree', 'add', '-q', '--detach', str(wt), cwd=layout / 'zero-core')
    rc, out = smoke(str(wt / 'tools' / 'smoke_all.sh'), '--dry-run', '--with-this-core', '--build-only')
    assert rc == 0, out
    assert found(out) == list(GAMES[:2]), out
    assert 'missing: kingdombuilder-zero' in out and 'games: 2' in out, out


def test_finds_the_games_from_a_submodule_in_a_worktree(layout):
    """MTG Zero's gate: <superproject worktree>/third_party/zero-core/tools/smoke_all.sh."""
    sup = layout / 'mtg-zero'
    sup.mkdir()
    git('init', '-q', cwd=sup)
    git('submodule', 'add', '-q', str(layout / 'zero-core'), 'third_party/zero-core', cwd=sup)
    git('commit', '-q', '-m', 'pin', cwd=sup)
    wt = layout / 'mtg-zero-wt' / 'gate'
    git('worktree', 'add', '-q', '--detach', str(wt), cwd=sup)
    git('submodule', 'update', '-q', '--init', cwd=wt)
    rc, out = smoke(str(wt / 'third_party' / 'zero-core' / 'tools' / 'smoke_all.sh'), '--dry-run')
    assert rc == 0, out
    assert found(out) == list(GAMES[:2]), out


def test_no_game_fails_loudly(layout, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    script = str(layout / 'zero-core' / 'tools' / 'smoke_all.sh')
    rc, out = smoke(script, '--dry-run', SMOKE_GAMES_ROOT=str(empty))
    assert rc != 0 and 'no game repo found' in out and 'GREEN' not in out, out
    # a real run (core steps skipped so nothing builds): fails before running anything, never GREEN
    rc, out = smoke(script, '--with-this-core', '--build-only', SMOKE_GAMES_ROOT=str(empty), SKIP_CORE='1')
    assert rc == 1 and 'FAIL  games: no game repo found' in out and 'GREEN' not in out, out
    # a deliberate core-only run is fine
    rc, out = smoke(script, '--dry-run', '--core-only', SMOKE_GAMES_ROOT=str(empty))
    assert rc == 0 and 'games: none (--core-only)' in out, out
    rc, out = smoke(script, '--dry-run', SMOKE_GAMES_ROOT=str(empty), SKIP_GAMES='1')
    assert rc == 0, out


def test_explicit_games_are_taken_as_given(layout, tmp_path):
    script = str(layout / 'zero-core' / 'tools' / 'smoke_all.sh')
    rc, out = smoke(script, '--dry-run', '--build-only', str(layout / GAMES[1]))
    assert rc == 0 and found(out) == [GAMES[1]] and 'missing' not in out, out
    rc, out = smoke(script, '--dry-run', str(layout / GAMES[0]), str(tmp_path / 'no-such-game'))
    assert rc == 2 and 'no such game checkout' in out, out
    rc, out = smoke(script, '--dry-run', '--core-only', str(layout / GAMES[0]))
    assert rc == 2, out
