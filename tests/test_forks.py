"""Branched self-play games and archive starts (zero/forks.py, docs/az.md 24, docs/search.md 2.11) on the
toy: the bindings (rows' branch / start, stats, the settings refused), the run settings, a rolling toy run
with both on (the iteration's games exactly its quota, branches and archive starts among them, the
metrics), the archive file's checks, and remote self-play with the remote 'local' (the workers branch and
start from the shipped archive; an older checkout refused by its hello)."""
import json
import os

import numpy as np
import pytest

from zero import forks as F
from zero import games as G
from zero.run import DEFAULTS, read_metrics

from test_loop import run_loop
from test_remoteplay import PAIR, check_quota, pair_env


def seat(sims=8):
    return G.nn_seat('m', sims, selfplay=True, temp_moves=4)


def play(drv, skew=False):
    """Every batch answered (uniform priors, a deterministic value) -> rows with their own samples."""
    rows = []
    while True:
        lb = drv.next_batch()
        if lb is None:
            break
        n = len(lb.player)
        off = np.asarray(lb.opt_off)
        pri = np.concatenate([np.full(off[b + 1] - off[b], 1.0 / max(1, off[b + 1] - off[b]), np.float32)
                              for b in range(n)]) if n else np.zeros(0, np.float32)
        v = np.zeros((n, 8), np.float32)
        v[:, 0], v[:, 1] = 0.5, 0.5
        drv.submit(lb.id, pri, v, np.zeros((n, 8), np.float32))
        rows.extend(split(drv))
    rows.extend(split(drv))
    return rows


def split(drv):
    from zero.workqueue import split_games
    return split_games(drv.take_results(), drv.take_samples())


def test_the_bindings_branch_and_start_games():
    specs = [('rich', {'food': [20, 20]}), ('poor', json.dumps({'food': [0, 3]}))]
    d = G.Driver('toy', [[seat(), seat()]] * 40, spec='{}', concurrency=16, seed=5, record=True, threads=2,
                 branch={'p': 0.5, 'moves': 2, 'temp': 2.0}, starts={'p': 0.3, 'specs': specs})
    rows = play(d)
    br = [r for r, _ in rows if 'branch' in r]
    main = [r for r, _ in rows if 'branch' not in r]
    assert len(main) == 40 and 8 <= len(br) <= 32, (len(main), len(br))
    assert all(set(r['branch']) == {'at', 'of', 'decision', 'step', 'action', 'moves', 'played', 'same_first', 'start'}
               for r in br)
    assert all(1 <= r['branch']['moves'] <= 2 and r['branch']['at'] < r['branch']['of'] for r in br)
    st = [r['start'] for r in main if 'start' in r]
    assert 4 <= len(st) <= 22 and set(st) <= {0, 1}, st
    assert all('start' not in r for r in br), 'a branch names its game\'s entry in branch.start'
    s = d.stats()
    assert s['branches'] == len(br) and s['starts'] == len(st)
    assert s['dev_moves'] == sum(r['branch']['played'] for r in br)
    # the branch's rows carry its own outcome
    for r, sd in rows:
        if sd is not None:
            z = np.asarray(sd['z'])[:, :2]
            exp = np.array([[r['result'][(p + j) % 2] for j in range(2)] for p in sd['player']])
            assert np.allclose(z, exp), r
    assert F.setups([r for r, _ in rows]) == 40
    # off: the rows have neither key, the stats zeros
    d0 = G.Driver('toy', [[seat(), seat()]] * 8, spec='{}', concurrency=8, seed=5, record=True, threads=2)
    rows0 = play(d0)
    assert all('branch' not in r and 'start' not in r for r, _ in rows0)
    assert d0.stats()['branches'] == d0.stats()['starts'] == 0


@pytest.mark.parametrize('kw,msg', [
    ({'branch': {'p': 0.0}}, 'branch p'),
    ({'branch': {'p': 0.5, 'moves': 0}}, 'moves 0'),
    ({'branch': {'p': 0.5, 'temp': -1}}, 'temp'),
    ({'branch': {'p': 0.5, 'nope': 1}}, 'unknown key'),
    ({'branch': {'moves': 2}}, 'p (the share'),
    ({'starts': {'p': 0.5, 'specs': []}}, 'no entries'),
    ({'starts': {'p': 0.5, 'specs': [('a', {'food': [1, 1, 1]})]}}, 'does not start'),
    ({'starts': {'p': 0.5, 'specs': [('a', {'nope': 1})]}}, 'entry "a"'),
])
def test_the_bindings_refuse_bad_settings(kw, msg):
    with pytest.raises(Exception) as e:
        G.Driver('toy', [[seat(), seat()]], spec='{}', concurrency=2, seed=1, record=True, **kw)
    assert msg in str(e.value), str(e.value)
    with pytest.raises(Exception) as e:                    # a match records nothing: never branches
        G.Driver('toy', [[seat(), seat()]], spec='{}', concurrency=2, seed=1, record=False,
                 branch={'p': 0.5})
    assert 'never branches' in str(e.value)


def toy_archive(path, spec='{}', entries=None, game='toy'):
    entries = entries or [{'id': f'food{k}', 'kind': 'rich' if k > 5 else 'poor', 'patch': {'food': [k, k]}}
                          for k in (0, 3, 9, 12)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'format': F.FORMAT, 'game': game, 'base_spec': spec, 'entries': entries, 'built': 'test'}, f)
    return str(path)


def test_settings_and_the_archive_file_are_checked(tmp_path):
    assert DEFAULTS['selfplay_branch'] == 0.0 and DEFAULTS['selfplay_archive'] == ''
    for args, msg in [(['--selfplay-branch', '1.5', '--rolling', '1'], 'selfplay_branch 1.5'),
                      (['--selfplay-branch', '0.2'], 'needs rolling self-play'),
                      (['--selfplay-branch', '0.2', '--rolling', '1', '--selfplay-branch-moves', '0'], 'selfplay_branch_moves'),
                      (['--selfplay-branch', '0.2', '--rolling', '1', '--selfplay-branch-temp', '0'], 'selfplay_branch_temp'),
                      (['--selfplay-archive', 'x.json', '--selfplay-archive-p', '0'], 'selfplay_archive_p'),
                      (['--selfplay-branch', '0.2', '--rolling', '1', '--selfplay-league', '0.2'], 'self-play league')]:
        p = run_loop(tmp_path / f'r{len(msg)}', '--iters', '1', *args)
        assert p.returncode != 0 and msg in p.stderr, (args, p.stderr[-500:])
    good = toy_archive(tmp_path / 'a.json')
    A = F.load_archive(good, 'toy', '{}')
    assert A['n'] == 4 and A['kinds'] == ['poor', 'poor', 'rich', 'rich'] and len(A['sha']) == 16
    assert json.loads(A['specs'][2][1]) == {'food': [9, 9]}
    for bad, msg in [(toy_archive(tmp_path / 'g.json', game='mtg'), "built for game 'mtg'"),
                     (toy_archive(tmp_path / 's.json', spec={'food': [1, 1]}), 'another spec'),
                     (toy_archive(tmp_path / 'd.json', entries=[{'id': 'a', 'patch': {'food': [1, 1]}}] * 2), 'duplicate'),
                     (toy_archive(tmp_path / 'e.json', entries=[{'id': 'a', 'patch': {}}]), 'entry 0')]:
        with pytest.raises(SystemExit) as e:
            F.load_archive(bad, 'toy', '{}')
        assert msg in str(e.value), str(e.value)


def test_a_rolling_run_branches_and_starts_from_its_archive(tmp_path):
    """Both on in a rolling toy run: each iteration exactly its quota of games, branches and archive
    starts among them at about their shares, their rows in the shards, the metrics and the log line."""
    arch = toy_archive(tmp_path / 'arch.json')
    run = tmp_path / 'fk'
    p = run_loop(run, '--iters', '3', '--rolling', '1', '--games', '96', '--conc', '32', '--selfplay-branch', '0.3',
                 '--selfplay-branch-moves', '3', '--selfplay-archive', arch, '--selfplay-archive-p', '0.25')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    tot = {'game': 0, 'branch': 0, 'archive': 0}
    for r in m[1:]:
        sp = r['selfplay']
        fk = sp['forks']
        assert sp['games'] == 96 and sum(fk['games'].values()) == 96, (sp['games'], fk)
        assert sum(fk['samples'].values()) == sp['samples'], (fk['samples'], sp['samples'])
        for k in tot:
            tot[k] += fk['games'][k]
        assert fk['archive_sha'] == F.load_archive(arch, 'toy', '{}')['sha']
    n = sum(tot.values())
    # branches: 0.3 of the non-branch games spawn one -> ~23% of all; archive: 0.25 of the non-branch games
    assert 0.12 < tot['branch'] / n < 0.36, tot
    assert 0.10 < tot['archive'] / (tot['game'] + tot['archive']) < 0.42, tot
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'self-play forks (zero/forks.py): branch 0.3' in log and 'forks: ' in log


def test_off_by_default_no_forks_metrics(tmp_path):
    run = tmp_path / 'off'
    p = run_loop(run, '--iters', '1', '--rolling', '1')
    assert p.returncode == 0, p.stderr[-2000:]
    assert all('forks' not in r['selfplay'] for r in read_metrics(str(run))[1:])


def test_remote_workers_branch_and_start_from_the_shipped_archive(tmp_path):
    """Remote self-play (the remote 'local'): the archive is shipped into the mirror, the workers branch and
    start from it as this machine does (their hello agrees), the shared quota stays exact with branches
    counted as games, and the remote's tally joins the iteration's `forks`."""
    arch = toy_archive(tmp_path / 'arch.json')
    run = tmp_path / 'rf'
    p = run_loop(run, '--iters', '3', *PAIR, '--selfplay-remote', 'local', '--selfplay-branch', '0.3',
                 '--selfplay-archive', arch, '--selfplay-archive-p', '0.2', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 192)
    assert os.path.exists(tmp_path / 'mirror')
    mirrors = [os.path.join(dp, 'archive.json') for dp, _, fs in os.walk(tmp_path / 'mirror') if 'archive.json' in fs]
    assert mirrors, 'the archive was shipped into the mirror'
    for r in m[1:]:
        fk = r['selfplay']['forks']
        assert sum(fk['games'].values()) == 192, fk
        if r['selfplay']['remote']['accepted']:
            assert sum(fk['remote']['games'].values()) == r['selfplay']['remote']['accepted'], (fk, r['selfplay']['remote'])
    rem = [r['selfplay']['forks'].get('remote') for r in m[1:]]
    assert any(x and x['games']['branch'] > 0 for x in rem) and any(x and x['games']['archive'] > 0 for x in rem), rem
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')


def test_an_older_worker_is_refused(tmp_path):
    """A worker whose checkout reports no forks (the tests' no_forks: an older checkout) is refused: the
    run plays its iterations on this machine alone, and says why."""
    run = tmp_path / 'old'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--selfplay-branch', '0.3',
                 env=pair_env(tmp_path, {'no_forks': 1}))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and 'branched games / archive starts' in log, log[-3000:]
    check_quota(read_metrics(str(run)), 192)


def test_differ_leaves_the_games_line_at_once():
    d = G.Driver('toy', [[seat(), seat()]] * 30, spec='{}', concurrency=16, seed=9, record=True, threads=2,
                 branch={'p': 1.0, 'moves': 1, 'differ': True})
    br = [r['branch'] for r, _ in play(d) if 'branch' in r]
    assert len(br) >= 25 and not any(b['same_first'] for b in br), br
