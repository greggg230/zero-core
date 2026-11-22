"""tools/pytest_shards.py (docs/az.md 75): the plan runs every test exactly once, the result is read from the shards'
junit files, a failure fails the run, a named file's failure is rerun alone, and the durations are recorded. The
end-to-end cases run real pytest processes on a few trivial tests in a temp directory."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('pytest_shards', os.path.join(ROOT, 'tools', 'pytest_shards.py'))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def suite(tmp_path, files):
    (tmp_path / 'tests').mkdir()
    for name, body in files.items():
        p = tmp_path / 'tests' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding='utf-8')
    return str(tmp_path)


def test_the_plan_runs_every_test_once(tmp_path):
    root = suite(tmp_path, {f'test_{c}.py': '#' * (2000 * (k + 1)) for k, c in enumerate('abcdefg')})
    (tmp_path / 'tests' / 'sub').mkdir()
    (tmp_path / 'tests' / 'sub' / 'test_deep.py').write_text('', encoding='utf-8')
    (tmp_path / 'tests' / 'conftest.py').write_text('', encoding='utf-8')
    files = S.test_files(root)
    assert 'tests/sub/test_deep.py' in files and 'tests/conftest.py' not in files and len(files) == 8
    parts = S.partition(files, 3, {}, root)
    assert len(parts) == 3 and sorted(f for p, _ in parts for f in p) == files and S.coverage(files, parts) is None
    # a measured file gets its weight: one far longer than the rest has a shard to itself
    assert (['tests/test_a.py'], []) in S.partition(files, 3, {'tests/test_a.py': 10_000.0}, root)
    # a file longer than a shard's share with measured tests is split by its tests, each test in exactly one shard
    w = {'tests/test_a.py': 900.0, **{f'tests/test_a.py::t{i}': 100.0 for i in range(9)}}
    split = S.partition(files, 3, w, root)
    assert S.coverage(files, split) is None
    named = [e for p, _ in split for e in p if '::' in e]
    owner = next(d for p, d in split if 'tests/test_a.py' in p)
    assert named and sorted(named) == sorted(owner)
    assert S.coverage(files, [(p, []) for p, _ in split])           # (the named tests would run twice)
    assert S.coverage(files, parts[:2])                             # (a shard's files would not run)


def test_the_junit_node_ids():
    get = {'classname': 'tests.sub.test_x.TestA', 'name': 'test_y[1]'}.get
    assert S.junit_node(type('T', (), {'get': staticmethod(get)})) == 'tests/sub/test_x.py::TestA::test_y[1]'


def test_a_pass_records_the_durations_and_says_so(tmp_path, capsys):
    root = suite(tmp_path, {'test_one.py': 'def test_a(): pass\ndef test_b(): pass\n',
                            'test_two.py': 'def test_c(tmp_path): assert tmp_path.exists()\n',
                            'test_three.py': 'import pytest\n@pytest.mark.skip\ndef test_d(): pass\ndef test_e(): pass\n'})
    d = str(tmp_path / 'd.json')
    base = str(tmp_path / 'base')
    rc = S.main(['--root', root, '--shards', '2', '--durations', d, '--basetemp', base, '--python', sys.executable])
    out = capsys.readouterr().out
    assert rc == 0 and out.strip().splitlines()[-1].startswith('pytest shards: PASSED - 2 shard(s), 4 passed, 1 skipped')
    seen = json.load(open(d, encoding='utf-8'))
    assert {'tests/test_one.py', 'tests/test_one.py::test_a', 'tests/test_two.py::test_c'} <= set(seen)
    assert os.path.isfile(os.path.join(base, 'shard-1.log')) and os.path.isfile(os.path.join(base, 'shard-2.xml'))
    # --plan runs nothing
    assert S.main(['--root', root, '--shards', '2', '--durations', d, '--plan']) == 0
    assert 'shard 2:' in capsys.readouterr().out


def test_a_failure_fails_the_run_and_is_named(tmp_path, capsys):
    root = suite(tmp_path, {'test_ok.py': 'def test_a(): pass\n', 'test_bad.py': 'def test_b(): assert 0\n'})
    d = str(tmp_path / 'd.json')
    rc = S.main(['--root', root, '--shards', '2', '--durations', d, '--basetemp', str(tmp_path / 'b')])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and last.startswith('pytest shards: FAILED') and 'tests/test_bad.py::test_b' in last
    assert 'tests/test_bad.py' not in json.load(open(d, encoding='utf-8'))      # (only a passed shard is recorded)
    # a setup error is never retried, even in a --retry-alone file
    (tmp_path / 'tests' / 'test_bad.py').write_text(
        'import pytest\n@pytest.fixture\ndef f(): raise RuntimeError\ndef test_b(f): pass\n', encoding='utf-8')
    rc = S.main(['--root', root, '--shards', '2', '--durations', d, '--basetemp', str(tmp_path / 'b2'),
                 '--retry-alone', 'tests/test_bad.py'])
    out = capsys.readouterr().out
    assert rc == 1 and 'retry alone' not in out and '1 error(s)' in out.strip().splitlines()[-1]


def test_no_test_is_not_a_pass(tmp_path, capsys):
    (tmp_path / 'tests').mkdir()
    assert S.main(['--root', str(tmp_path)]) == 1 and 'no tests/**/test_*.py' in capsys.readouterr().out
    (tmp_path / 'tests' / 'test_empty.py').write_text('X = 1\n', encoding='utf-8')
    rc = S.main(['--root', str(tmp_path), '--durations', str(tmp_path / 'd.json'), '--basetemp', str(tmp_path / 'b')])
    assert rc == 1 and capsys.readouterr().out.strip().splitlines()[-1].startswith('pytest shards: FAILED')


def test_a_named_files_failure_is_rerun_alone(tmp_path, capsys):
    """--retry-alone: a test that fails beside its file's other tests and passes alone is "flaky, passed alone"; one
    that fails alone too fails the run; a failure in any other file is not retried."""
    marker = str(tmp_path / 'ran').replace('\\', '/')
    flaky = ('import os\n'
             f'M = "{marker}"\n'
             'def test_first():\n    open(M, "w").close()\n'
             'def test_then():\n    seen = os.path.exists(M)\n    if seen: os.remove(M)\n    assert not seen\n')
    root = suite(tmp_path, {'test_flaky.py': flaky, 'test_ok.py': 'def test_a(): pass\n'})
    args = ['--root', root, '--shards', '2', '--durations', str(tmp_path / 'd.json')]
    rc = S.main([*args, '--basetemp', str(tmp_path / 'b1'), '--retry-alone', 'tests/test_flaky.py'])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and last.startswith('pytest shards: PASSED - 2 shard(s), 3 passed')
    assert 'flaky, passed alone: tests/test_flaky.py::test_then' in last
    rc = S.main([*args, '--basetemp', str(tmp_path / 'b2')])                    # not named: a failure
    assert rc == 1 and 'tests/test_flaky.py::test_then' in capsys.readouterr().out.strip().splitlines()[-1]
    (tmp_path / 'tests' / 'test_flaky.py').write_text('def test_then(): assert 0\n', encoding='utf-8')
    rc = S.main([*args, '--basetemp', str(tmp_path / 'b3'), '--retry-alone', 'tests/test_flaky.py'])
    assert rc == 1 and 'failed again alone: tests/test_flaky.py::test_then' in capsys.readouterr().out
