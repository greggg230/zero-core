"""pytest in parallel shards (docs/az.md 74): a repo's `python -m pytest tests` as N processes over disjoint parts of
the tests, run at once - for this repo and for every game repo (tools/smoke_all.sh runs each suite through it).

    python tools/pytest_shards.py                        # this checkout's tests/, 4 shards
    python tools/pytest_shards.py --shards 6 --retry-alone tests/test_remoteplay.py
    python tools/pytest_shards.py --root ../glassroad-zero -- -W ignore::pytest.PytestWarning
    python tools/pytest_shards.py --plan                 # print the shards, run nothing

The suite is mostly single-threaded tests (tests/conftest.py: torch at one thread), about 35 minutes of them in one
process on a busy machine; four shards take about a quarter of that where four cores are free.

* **Every test exactly once.** The test files (`tests/**/test_*.py` under --root) are split into --shards parts of about
  equal time: each file's seconds from earlier runs (the durations file), a file without one weighed by its size; a
  file longer than a shard's share whose tests were measured is split by its tests - its first part runs the file with
  the other parts' tests deselected (so a test added since, unmeasured, runs there), the others name their tests. The
  plan is checked before anything runs (`coverage`): every file is the entry of exactly one shard and every test named
  elsewhere is deselected where its file runs.
* **Each shard is its own process**, with its own --basetemp (under --basetemp, default a new temp dir; removed after a
  pass) and junit file, its output in <basetemp>/shard-K.log. Tests that share something outside their tmp_path (a
  fixed port, a file in the checkout) would collide between shards as they would not in one process: keep tests to
  their tmp_path.
* **The result is the shards' junit files**, not their output: the run passes when every shard exited 0 and ran at least
  one test, and at least one test passed overall. The last line says so:
  `pytest shards: PASSED - 4 shard(s), 812 passed, 59 skipped in 640 s` or
  `pytest shards: FAILED - ...` naming the failed tests; a failed shard's last lines are printed. Exit 0 / 1.
* **--retry-alone FILE** (may repeat): when a shard's ONLY failures are tests of these files (tests that time real
  threads and can fail on a loaded machine: AGENTS.md), each is run again alone once; the run passes only if every one
  passes alone, and the last line names them ("flaky, passed alone: ..."). Any other failure, or an error, fails.
* **The durations file**: --durations, else $ZERO_PYTEST_DURATIONS, else
  ~/.config/zero/pytest_durations/<repo name>.json (shared by every checkout of the repo on the machine); each passed
  shard's per-test seconds are merged into it, so the next run's shards are even.

Standard library only (it runs before anything is built or installed beyond pytest itself).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_files(root):
    """Every test file `pytest tests` collects under `root`: tests/**/test_*.py, as posix paths relative to it."""
    found = glob.glob(os.path.join(root, 'tests', '**', 'test_*.py'), recursive=True)
    return sorted(os.path.relpath(p, root).replace(os.sep, '/') for p in found)


def partition(files, n, weights, root):
    """`files` into `n` shards of about equal time -> [(entries, deselected)]: `entries` the pytest arguments (files, or
    node ids of a split file), `deselected` node ids passed as --deselect. `weights` = seconds from earlier runs (per
    file, and per test as "<file>::<test id>"); a file without one is weighed by its size (1 s per 2 kB). A file longer
    than a shard's share with measured tests is split by its tests. Then longest first, each unit to the lightest
    shard."""
    def w(f):
        if f in weights:
            return float(weights[f])
        try:
            return os.path.getsize(os.path.join(root, f)) / 2000.0
        except OSError:
            return 1.0
    target = sum(w(f) for f in files) / max(1, n)
    units = []                                  # (seconds, entries, deselected)
    for f in files:
        tests = {k: float(v) for k, v in weights.items() if k.startswith(f + '::')}
        k = min(len(tests), int(-(-w(f) // target)) if target > 0 else 1)
        if k < 2:
            units.append((w(f), [f], []))
            continue
        parts = [[0.0, []] for _ in range(k)]
        for t in sorted(tests, key=lambda t: (-tests[t], t)):
            p = min(parts, key=lambda p: p[0])
            p[0] += tests[t]
            p[1].append(t)
        rest = [t for p in parts[1:] for t in p[1]]
        units.append((parts[0][0] + max(0.0, w(f) - sum(tests.values())), [f], sorted(rest)))
        units += [(p[0], sorted(p[1]), []) for p in parts[1:] if p[1]]
    bins = [[0.0, [], []] for _ in range(n)]
    for sec, entries, desel in sorted(units, key=lambda u: (-u[0], u[1])):
        b = min(bins, key=lambda b: b[0])
        b[0] += max(sec, 0.01)          # (tests measured at 0 s still spread over the shards)
        b[1] += entries
        b[2] += desel
    return [(b[1], b[2]) for b in bins if b[1]]


def coverage(files, shards):
    """None when `shards` run every test of `files` exactly once, else why not."""
    whole = [e for entries, _ in shards for e in entries if '::' not in e]
    if sorted(whole) != sorted(files):
        return f'files as entries: {sorted(set(files) ^ set(whole)) or "a file twice"}'
    owner = {e: k for k, (entries, _) in enumerate(shards) for e in entries if '::' not in e}
    named = [e for entries, _ in shards for e in entries if '::' in e]
    desel = {d: k for k, (_, ds) in enumerate(shards) for d in ds}
    if len(named) != len(set(named)) or sorted(named) != sorted(desel):
        return 'a split file\'s named tests and its deselected ones differ'
    for t in named:
        f = t.split('::')[0]
        if f not in owner or desel[t] != owner[f]:
            return f'{t}: not deselected in the shard that runs {f}'
    return None


def repo_name(root):
    """The repo's name for its durations file: the main checkout's directory (a worktree's too), a submodule's own
    name; `root`'s directory name outside git."""
    p = subprocess.run(['git', '-C', root, 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                       capture_output=True, text=True)
    common = p.stdout.strip()
    if p.returncode or not common:
        return os.path.basename(os.path.abspath(root))
    common = os.path.normpath(common)
    return os.path.basename(os.path.dirname(common)) if os.path.basename(common) == '.git' else os.path.basename(common)


def durations_path(root, arg=None):
    return arg or os.environ.get('ZERO_PYTEST_DURATIONS') or os.path.join(
        os.path.expanduser('~'), '.config', 'zero', 'pytest_durations', repo_name(root) + '.json')


def read_durations(path):
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def junit_node(tc):
    """A junit testcase's pytest node id ("tests/test_x.py::Class::test[param]") from its classname ("tests.test_x" or
    "tests.test_x.Class") and name."""
    parts = (tc.get('classname') or '').split('.')
    k = next((i for i in range(len(parts)) if parts[i].startswith('test_')), len(parts) - 1)
    return '::'.join(['/'.join(parts[:k + 1]) + '.py', *parts[k + 1:], tc.get('name') or ''])


def read_junit(path):
    """A shard's junit file -> {'tests', 'passed', 'failed', 'errors', 'skipped', 'failed_ids', 'error_ids', 'seconds':
    {node id: s}}, or None when it is missing or unreadable (the shard died before pytest wrote it)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    out = {'tests': 0, 'failed': 0, 'errors': 0, 'skipped': 0, 'failed_ids': [], 'error_ids': [], 'seconds': {}}
    for tc in root.iter('testcase'):
        node = junit_node(tc)
        out['tests'] += 1
        out['seconds'][node] = out['seconds'].get(node, 0.0) + float(tc.get('time') or 0)
        kinds = {child.tag for child in tc}
        if 'error' in kinds:
            out['errors'] += 1
            out['error_ids'].append(node)
        elif 'failure' in kinds:
            out['failed'] += 1
            out['failed_ids'].append(node)
        elif 'skipped' in kinds:
            out['skipped'] += 1
    out['passed'] = out['tests'] - out['failed'] - out['errors'] - out['skipped']
    return out


def record_durations(path, seconds):
    """Merge a passed shard's per-test seconds into the durations file (each touched file's total = the sum of its
    tests' latest seconds), under a lock file. Best effort: any failure changes nothing."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lock = path + '.lock'
        for _ in range(100):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                time.sleep(0.1)
        else:
            return
        try:
            d = read_durations(path)
            d.update({k: round(v, 2) for k, v in seconds.items()})
            for f in {n.split('::')[0] for n in seconds}:
                d[f] = round(sum(v for k, v in d.items() if k.startswith(f + '::')), 2)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(d, fh, indent=1, sort_keys=True)
            os.replace(tmp, path)
        finally:
            os.close(fd)
            os.remove(lock)
    except OSError:
        pass


def shard_argv(python, entries, desel, base, xml, extra):
    return [python, '-m', 'pytest', *entries, '-q', '-p', 'no:cacheprovider', *extra,
            *[f'--deselect={d}' for d in desel], f'--basetemp={base}', f'--junitxml={xml}']


def tail(path, n=40):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return ''.join(f.readlines()[-n:])
    except OSError:
        return ''


def run_shards(root, shards, base, python, extra):
    """Every shard at once -> [{'k', 'rc', 'log', 'junit' (read_junit's, or None)}]."""
    procs = []
    for k, (entries, desel) in enumerate(shards, 1):
        log = os.path.join(base, f'shard-{k}.log')
        xml = os.path.join(base, f'shard-{k}.xml')
        argv = shard_argv(python, entries, desel, os.path.join(base, f'shard-{k}'), xml, extra)
        fh = open(log, 'w', encoding='utf-8', errors='replace')
        fh.write('$ ' + ' '.join(argv) + '\n\n')
        fh.flush()
        procs.append((k, subprocess.Popen(argv, cwd=root, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL),
                      fh, log, xml))
    out = []
    for k, p, fh, log, xml in procs:
        rc = p.wait()
        fh.close()
        out.append({'k': k, 'rc': rc, 'log': log, 'junit': read_junit(xml)})
    return out


def retry_alone(root, ids, base, python, extra):
    """Each test of `ids` alone, once -> the ids that failed again."""
    again = []
    for j, nid in enumerate(ids, 1):
        log = os.path.join(base, f'retry-{j}.log')
        argv = [python, '-m', 'pytest', nid, '-q', '-p', 'no:cacheprovider', *extra,
                f'--basetemp={os.path.join(base, f"retry-{j}")}']
        with open(log, 'w', encoding='utf-8', errors='replace') as fh:
            rc = subprocess.run(argv, cwd=root, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        print(f'  retry alone: {nid}: {"passed" if rc == 0 else f"FAILED again (exit {rc}; {log})"}', flush=True)
        if rc != 0:
            again.append(nid)
    return again


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        k = argv.index('--')
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=HERE, help='the repo whose tests/ to run (default: this checkout)')
    ap.add_argument('--shards', type=int, default=4, help='how many pytest processes at once (default 4)')
    ap.add_argument('--durations', default=None, help='the durations file (default: $ZERO_PYTEST_DURATIONS, else '
                    '~/.config/zero/pytest_durations/<repo name>.json)')
    ap.add_argument('--basetemp', default=None, help='the directory for the shards\' --basetemp, junit files and logs '
                    '(default: a new temp dir; removed after a pass unless --keep)')
    ap.add_argument('--retry-alone', action='append', default=[], metavar='FILE', help='a test file whose failures, when '
                    'they are a shard\'s only ones, are each run again alone once (may repeat)')
    ap.add_argument('--python', default=sys.executable, help='the interpreter for pytest (default: this one)')
    ap.add_argument('--keep', action='store_true', help='keep the basetemp directory after a pass')
    ap.add_argument('--plan', action='store_true', help='print the shards and run nothing')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    files = test_files(root)
    if not files:
        print(f'pytest shards: FAILED - no tests/**/test_*.py under {root}')
        return 1
    dpath = durations_path(root, a.durations)
    weights = read_durations(dpath)
    shards = partition(files, max(1, a.shards), weights, root)
    why = coverage(files, shards)
    if why:
        print(f'pytest shards: FAILED - internal: the shards do not run every test exactly once ({why})')
        return 1
    measured = sum(1 for f in files if f in weights)
    print(f'pytest shards: {len(files)} file(s) of {root} in {len(shards)} shard(s) ({measured} with measured seconds: '
          f'{dpath})', flush=True)
    if a.plan:
        for k, (entries, desel) in enumerate(shards, 1):
            est = sum(float(weights.get(e, 0)) for e in entries) - sum(float(weights.get(d, 0)) for d in desel)
            print(f'  shard {k}: {len(entries)} entr(ies), {len(desel)} deselected, ~{est:.0f} measured s: '
                  f'{" ".join(entries)}')
        return 0
    made = a.basetemp is None
    base = os.path.abspath(a.basetemp) if a.basetemp else tempfile.mkdtemp(prefix='pytest-shards-')
    os.makedirs(base, exist_ok=True)
    t0 = time.time()
    results = run_shards(root, shards, base, a.python, extra)
    tot = {'passed': 0, 'skipped': 0, 'failed': 0, 'errors': 0}
    problems, flaky = [], []
    retry_files = {f.replace(os.sep, '/') for f in a.retry_alone}
    for r in results:
        j = r['junit']
        if j is None:
            problems.append(f'shard {r["k"]}: no junit file (exit {r["rc"]}; {r["log"]})')
            print(f'  shard {r["k"]}: exit {r["rc"]}, no junit file\n{tail(r["log"])}', flush=True)
            continue
        for key in tot:
            tot[key] += j[key]
        line = f'{j["passed"]} passed, {j["skipped"]} skipped, {j["failed"]} failed, {j["errors"]} error(s)'
        print(f'  shard {r["k"]}: exit {r["rc"]}, {line}', flush=True)
        if j['tests'] == 0:
            problems.append(f'shard {r["k"]}: ran no test (exit {r["rc"]}; {r["log"]})')
        elif r['rc'] == 0 and not j['failed'] and not j['errors']:
            record_durations(dpath, j['seconds'])
        elif (j['failed'] and not j['errors'] and retry_files
              and all(i.split('::')[0] in retry_files for i in j['failed_ids'])):
            again = retry_alone(root, j['failed_ids'], base, a.python, extra)
            if again:
                problems.append(f'shard {r["k"]}: failed again alone: {", ".join(again)}')
            else:
                flaky += j['failed_ids']
                tot['failed'] -= j['failed']
                tot['passed'] += j['failed']
        else:
            bad = j['failed_ids'] + j['error_ids']
            problems.append(f'shard {r["k"]}: ' + (', '.join(bad[:20]) if bad else f'exit {r["rc"]}') + f' ({r["log"]})')
            print(tail(r['log']), flush=True)
    if not problems and tot['passed'] <= 0:
        problems.append('no test passed')
    secs = time.time() - t0
    counts = f'{len(shards)} shard(s), {tot["passed"]} passed, {tot["skipped"]} skipped'
    note = f'; flaky, passed alone: {", ".join(flaky)}' if flaky else ''
    if problems:
        print(f'pytest shards: FAILED - {counts}, {tot["failed"]} failed, {tot["errors"]} error(s) in {secs:.0f} s - '
              f'{"; ".join(problems)}{note}')
        return 1
    if made and not a.keep:
        shutil.rmtree(base, ignore_errors=True)
    print(f'pytest shards: PASSED - {counts} in {secs:.0f} s{note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
