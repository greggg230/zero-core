"""The live-run fence (docs/az.md 51): a supervisor on this machine is recognised from its command line (any game's
tools/supervise.py, or -m zero.tools.supervise; not a tool that merely says --supervised), and the tests marked
`live_run_hazard` are skipped while one is live, with a reason tools can count - unless ZERO_ALLOW_LIVE_RUN_HAZARD=1.
No process is started here: ZERO_LIVE_RUNS stands for the look."""
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from zero import runctl  # noqa: E402


@pytest.mark.parametrize('cmd, run', [
    (['python', 'tools/supervise.py', 'E:/runs/pb2', '--preset', 'x'], 'E:/runs/pb2'),
    (['C:\\venv\\python.exe', '-u', 'C:\\repo\\tools\\supervise.py', 'runs/a'], 'runs/a'),
    (['python3', '-m', 'zero.tools.supervise', 'runs/toy'], 'runs/toy'),
    (['python', 'tools/supervise.py'], ''),
    (['python', 'tools/delverai_publish.py', 'E:/runs/pb2', '--supervised'], None),
    (['python', 'tools/run_yardsticks.py', 'watch', 'E:/runs/pb2', '--supervised'], None),
    (['python', '-c', 'import tools.supervise.py'], None),
    (['python', '-m', 'mtgz.az.run', '--run', 'tools/supervise.py'], None),
    (['python'], None),
])
def test_a_supervisor_is_known_by_its_program(cmd, run):
    assert runctl._supervisor_target(cmd) == run


def test_the_override_and_the_reason(monkeypatch):
    monkeypatch.delenv(runctl.HAZARD_ENV, raising=False)
    monkeypatch.setenv(runctl.FENCE_ENV, '1')                # (whatever this platform)
    monkeypatch.setenv(runctl.LIVE_ENV, '0')
    assert runctl.live_supervisors() == [] and runctl.live_run_fence_reason() is None
    monkeypatch.setenv(runctl.LIVE_ENV, '1')
    why = runctl.live_run_fence_reason()
    assert why.startswith(runctl.FENCE_PREFIX) and runctl.HAZARD_ENV in why
    monkeypatch.setenv(runctl.HAZARD_ENV, '1')
    assert runctl.live_run_fence_reason() is None
    monkeypatch.delenv(runctl.HAZARD_ENV)
    monkeypatch.setenv(runctl.FENCE_ENV, '0')
    assert runctl.live_run_fence_reason() is None
    monkeypatch.delenv(runctl.FENCE_ENV)
    monkeypatch.setattr(runctl.os, 'name', 'posix')          # the Mac: other projects' runs live there, no pid hazard
    assert runctl.live_run_fence_reason() is None


class Item:
    def __init__(self, marked):
        self.marked, self.added = marked, []

    def get_closest_marker(self, name):
        return object() if self.marked and name == 'live_run_hazard' else None

    def add_marker(self, m):
        self.added.append(m)


def test_only_the_marked_tests_are_fenced_and_only_beside_a_live_run(monkeypatch):
    monkeypatch.delenv(runctl.HAZARD_ENV, raising=False)
    monkeypatch.setenv(runctl.FENCE_ENV, '1')
    looked = []
    real = runctl.live_supervisors
    monkeypatch.setattr(runctl, 'live_supervisors', lambda: looked.append(1) or real())
    plain = [Item(False), Item(False)]
    monkeypatch.setenv(runctl.LIVE_ENV, '1')
    assert runctl.fence_items(None, plain) == 0 and not looked          # nothing marked: no look at all
    items = [Item(True), Item(False), Item(True)]
    assert runctl.fence_items(None, items) == 2 and looked
    assert [len(i.added) for i in items] == [1, 0, 1]
    assert items[0].added[0].kwargs['reason'].startswith(runctl.FENCE_PREFIX)
    monkeypatch.setenv(runctl.LIVE_ENV, '0')
    items = [Item(True)]
    assert runctl.fence_items(None, items) == 0 and not items[0].added


def test_the_hazard_tests_are_marked():
    """The supervisor-starting tests carry the mark (a new one should too: docs/az.md 51)."""
    import ast
    want = {'test_run_infra.py': {'test_the_orphan_sweep_spares_processes_older_than_the_launch',
                                  'test_the_supervisor_gives_up_on_the_same_crash_three_times'},
            'test_loop.py': {'test_supervisor_with_fake_run', 'test_supervisor_stall_watchdog'},
            'test_encoding_guard.py': {'test_run_resume_refuses_exits_78_and_the_supervisor_does_not_restart'}}
    for f, names in want.items():
        tree = ast.parse(open(os.path.join(ROOT, 'tests', f), encoding='utf-8').read())
        marked = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                  and any('live_run_hazard' in ast.unparse(d) for d in n.decorator_list)}
        assert names <= marked, (f, names - marked)
    tree = ast.parse(open(os.path.join(ROOT, 'tests', 'test_companion.py'), encoding='utf-8').read())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith('test_') and 'supervise.py' in ast.unparse(n):
            assert any('live_run_hazard' in ast.unparse(d) for d in n.decorator_list), n.name
