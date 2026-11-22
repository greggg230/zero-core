"""tools/check_checkpoint_compat.py's digests at once (docs/az.md 75.1): every target's self-play and the match as its
own process, both sides together, give exactly the digests of one process per side (--jobs 1) - on the toy game, this
checkout against itself."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, 'tools', 'check_checkpoint_compat.py')
spec = importlib.util.spec_from_file_location('check_checkpoint_compat', TOOL)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def run(*extra):
    py = os.path.join(ROOT, 'python')
    p = subprocess.run([sys.executable, TOOL, '--old', py, '--new', py, '--package', 'zero', '--net', 'random',
                        '--games', '6', '--sims', '8', *extra], capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    return p.returncode, [ln.split(' (')[0] if ln[:4] in ('old:', 'new:') else ln for ln in lines], p.stdout + p.stderr


def test_the_phases_are_each_target_then_the_match():
    assert C.phases(['visits', 'cq'], 16) == [['--targets', 'visits', '--match-games', '0'],
                                              ['--targets', 'cq', '--match-games', '0'],
                                              ['--targets', '', '--match-games', '16']]
    assert C.phases(['cq'], 0) == [['--targets', 'cq', '--match-games', '0']]


def test_the_merge_is_one_sides_output_and_refuses_processes_that_disagree():
    base = {'root': 'r', 'engine': 'e', 'game': 'toy', 'spec': '', 'evaluator': 'random'}
    parts = [dict(base, targets={'visits': {'samples': 3}}, seconds=2.0),
             dict(base, targets={'cq': {'samples': 4}}, seconds=5.0),
             dict(base, targets={}, match={'games': 2}, seconds=1.0)]
    out = C.merge(parts)
    assert list(out['targets']) == ['visits', 'cq'] and out['match'] == {'games': 2} and out['seconds'] == 5.0
    with pytest.raises(SystemExit):
        C.merge([parts[0], dict(parts[1], engine='another build')])


def test_at_once_gives_the_digests_of_one_process_per_side():
    rc1, one, text1 = run('--match-games', '4', '--jobs', '1')
    rc0, par, text0 = run('--match-games', '4')
    assert rc1 == 0 and rc0 == 0, text1 + text0
    assert one == par                       # every line but the seconds: targets, samples, digests, the match
    res = json.loads(par[-1])
    assert res['identical'] is True and res['differences'] == []
    assert sum('samples, records' in ln for ln in par) == 4 and sum(ln.strip().startswith('vs random') for ln in par) == 2
    # one target and no match: still both ways the same
    rc1, one, _ = run('--targets', 'cq', '--match-games', '0', '--jobs', '1')
    rc0, par, _ = run('--targets', 'cq', '--match-games', '0', '--jobs', '2')
    assert rc1 == 0 and rc0 == 0 and one == par and not any('visits:' in ln for ln in par)
