"""Remote self-play with variable-length rows (R13, docs/az.md 43.4 / 43.8): a worker with a token network plays
toy_tail on its own driver and sends v2 shards whose rows carry their tails and overrides, which the run trains on;
a worker whose checkout reports no capability 'tails' (the tests' `no_tails`: an older checkout) is refused and this
machine plays every game."""
import numpy as np

from zero import data
from zero.run import read_metrics

from test_loop import run_loop
from test_remoteplay import check_quota, pair_env, remote_shards

TAIL = ['--game', 'toy_tail', '--spec', '{"items0": 6, "max_moves": 24}', '--arch', 'tokens', '--shard-sparse', '1',
        '--conc', '32', '--games', '32', '--sims', '8', '--sims-fast', '4', '--p-full', '0.5', '--yardstick', 'random']


def test_a_remote_worker_carries_the_tails(tmp_path):
    run = tmp_path / 'tails'
    p = run_loop(run, '--iters', '2', *TAIL, '--selfplay-remote', 'local', env=pair_env(tmp_path), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' not in log, log[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 32)
    assert sum(r['selfplay']['remote']['games'] for r in m[1:]) > 0, [r['selfplay']['remote'] for r in m[1:]]
    shards = remote_shards(run)
    assert shards
    toks = 0
    for path in shards:
        s = data.read_shard(path, dense=False)
        assert s['meta']['x_version'] == 2 and s['meta']['t_segs'] == [[0, 'items', 16, 4], [1, 'marks', 4, 0]]
        toks += int(np.asarray(s['t_ptr'])[-1])
    assert toks > 0
    # the run trained on them: its window's tails and overrides
    assert all(r['train']['tail']['tokens'] > 0 and r['train']['tail']['overrides'] > 0 for r in m[1:]), \
        [r['train'].get('tail') for r in m[1:]]


def test_a_worker_without_the_tails_capability_is_refused(tmp_path):
    run = tmp_path / 'old'
    p = run_loop(run, '--iters', '2', *TAIL, '--selfplay-remote', 'local', env=pair_env(tmp_path, {'no_tails': 1}),
                 timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and "capability 'tails'" in log, log[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 32)
    assert all(r['selfplay']['remote']['games'] == 0 for r in m[1:]) and not remote_shards(run)
