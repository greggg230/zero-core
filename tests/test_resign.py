"""Self-play resignation with a no-resign share (zero/resign.py, docs/az.md 27, docs/search.md 3.14) on the toy:
the bindings (rows' `resign`, the resigned games' z and records, the no-resign draw, the settings refused),
the run settings, a rolling toy run (exact quotas with resigned games among them, the metrics), off by default,
and remote self-play with the remote 'local' (the workers resign as this machine, the shared quota and the local
linger stay exact, the remote's tally joins the metrics; an older checkout refused by its hello)."""
import numpy as np
import pytest

from zero import games as G
from zero import resign as RS
from zero.games import engine
from zero.run import DEFAULTS, read_metrics

from test_loop import run_loop
from test_remoteplay import PAIR, check_quota, pair_env

BASE = 6 + 3 + 4 + 12 + 6 + 13 * 7            # ToyGame's first player block (toy.rs `encode`)


def seat(sims=8, **kw):
    return G.nn_seat('m', sims, selfplay=True, temp_moves=4, **kw)


def play(drv):
    """Every batch answered - uniform priors, the decider's value from its score lead and the round (a lead late
    in the game is decisive; a function of the row alone) -> rows with their own samples."""
    from zero.workqueue import split_games
    rows = []
    while True:
        lb = drv.next_batch()
        if lb is None:
            break
        n = len(lb.player)
        off = np.asarray(lb.opt_off)
        pri = np.concatenate([np.full(off[b + 1] - off[b], 1.0 / max(1, off[b + 1] - off[b]), np.float32)
                              for b in range(n)]) if n else np.zeros(0, np.float32)
        x = np.asarray(lb.x).reshape(n, -1)
        lead = (x[:, BASE + 14] - x[:, BASE + 16 + 14]) * 20.0
        rnd = np.argmax(x[:, :6] > 0, axis=1) + 1.0
        w = 1.0 / (1.0 + np.exp(-1.5 * lead * rnd / 6.0))
        v = np.zeros((n, 8), np.float32)
        v[:, 0], v[:, 1] = w, 1.0 - w
        drv.submit(lb.id, pri, v, np.zeros((n, 8), np.float32))
        rows.extend(split_games(drv.take_results(), drv.take_samples()))
    rows.extend(split_games(drv.take_results(), drv.take_samples()))
    return sorted(rows, key=lambda rs: rs[0]['gid'])


def driver(s, games=40, seed=5, **kw):
    return G.Driver('toy', [[s, s]] * games, spec='{}', concurrency=16, seed=seed, record=True, threads=2, **kw)


def test_the_bindings_resign_games():
    on = play(driver(seat(resign_t=0.05, resign_k=4, resign_noresign=0.3)))
    dry = play(driver(seat(resign_t=0.05, resign_k=4, resign_noresign=1.0)))
    off = play(driver(seat()))
    assert len(on) == len(dry) == len(off) == 40
    assert all('resign' not in r for r, _ in off), 'off: no key'
    keys = {'resigned', 'noresign', 'winner', 'decision', 'step', 'records'}
    res = 0
    for (r, sd), (r0, sd0), (rf, _) in zip(on, dry, off):
        x, x0 = r['resign'], r0['resign']
        assert set(x) == keys and x['noresign'] == engine().noresign_pick(5, r['gid'], 0.3)
        # the dry run is the game of before (no stream moves), noting where it would have resigned
        assert (r0['result'], r0['steps'], r0['samples']) == (rf['result'], rf['steps'], rf['samples'])
        assert x0['noresign'] and not x0['resigned']
        if not x['resigned']:
            assert (r['result'], r['steps'], r['samples']) == (r0['result'], r0['steps'], r0['samples'])
            continue
        res += 1
        w = x['winner']
        assert (x['winner'], x['decision'], x['step'], x['records']) == (x0['winner'], x0['decision'], x0['step'], x0['records'])
        assert r['over'] and r['result'] == [1.0 * (s == w) for s in range(2)]
        assert r['samples'] == x['records'] == len(sd['player']) and r['steps'] == x['step'] < r0['steps']
        # its records: the twin's first ones, z the resignation's outcome relative to each decider
        n = len(sd['player'])
        assert np.array_equal(sd['x'], sd0['x'][:n]) and np.array_equal(sd['q'], sd0['q'][:n])
        z = np.asarray(sd['z'])[:, :2]
        exp = np.array([[float((p + j) % 2 == w) for j in range(2)] for p in sd['player']])
        assert np.array_equal(z, exp)
    assert 5 <= res <= 35, res


def test_noresign_pick_is_a_hash_of_seed_and_gid():
    e = engine()
    assert not e.noresign_pick(1, 2, 0.0) and e.noresign_pick(1, 2, 1.0)
    picks = [e.noresign_pick(7, g, 0.1) for g in range(10_000)]
    assert 900 < sum(picks) < 1100
    assert picks == [e.noresign_pick(7, g, 0.1) for g in range(10_000)]
    assert picks != [e.noresign_pick(8, g, 0.1) for g in range(10_000)]


@pytest.mark.parametrize('kw,msg', [
    ({'resign_t': 0.5}, 'resign_t 0.5'),
    ({'resign_t': 0.05, 'resign_k': 0}, 'resign_k 0'),
    ({'resign_t': 0.05, 'resign_noresign': 1.5}, 'resign_noresign 1.5'),
])
def test_bad_settings_are_refused(kw, msg):
    with pytest.raises(ValueError) as e:
        seat(**kw)
    assert msg in str(e.value), str(e.value)
    with pytest.raises(Exception) as e:                    # the driver checks a seat dict too
        driver(dict(seat(), **dict({'resign_t': 0.05}, **kw)), games=2)
    assert msg in str(e.value), str(e.value)


def test_where_resignation_is_refused():
    with pytest.raises(ValueError) as e:
        G.nn_seat('m', 8, resign_t=0.05)                    # an evaluation seat
    assert 'self-play seat only' in str(e.value)
    s = seat(resign_t=0.05)
    for kw, table, msg in [({'record': False}, [[s, s]], 'never resigns'),
                           ({}, [[dict(s, record=False), dict(s, record=False)]], 'never resigns'),
                           ({}, [[s, s, s]], 'two-player'),
                           ({}, [[s, dict(s, key='o', resign_k=3)]], 'differ')]:
        with pytest.raises(Exception) as e:
            G.Driver('toy', table, spec='{}', concurrency=2, seed=1, **dict({'record': True}, **kw))
        assert msg in str(e.value), (msg, str(e.value))


def test_run_settings_are_checked(tmp_path):
    assert (DEFAULTS['selfplay_resign_t'], DEFAULTS['selfplay_resign_k'], DEFAULTS['selfplay_resign_noresign']) == (0.0, 5, 0.1)
    assert RS.seat_keys(DEFAULTS) == {} and RS.fingerprint(DEFAULTS) == ''
    for args, msg in [(['--selfplay-resign-t', '0.6'], 'selfplay_resign_t 0.6'),
                      (['--selfplay-resign-t', '0.05', '--selfplay-resign-k', '0'], 'selfplay_resign_k 0'),
                      (['--selfplay-resign-t', '0.05', '--selfplay-resign-noresign', '1.5'], 'selfplay_resign_noresign 1.5'),
                      (['--selfplay-resign-t', '0.05', '--selfplay-league', '0.2'], 'self-play league'),
                      (['--selfplay-resign-t', '0.05', '--players', '3'], 'two-player')]:
        p = run_loop(tmp_path / f'r{len(msg)}{len(args)}', '--iters', '1', *args)
        assert p.returncode != 0 and msg in p.stderr, (args, p.stderr[-500:])


def check_resign(m, games, remote=False):
    """Every iteration's `resign`: its games the iteration's, resigned and no-resign games among them."""
    tot = RS.new_tally()
    for r in m[1:]:
        sp, rs = r['selfplay'], r['selfplay']['resign']
        assert rs['games'] == games, (rs, sp['games'])
        assert rs['resigned'] + rs['noresign'] <= games and 0 <= rs['wrong'] <= rs['would'] <= rs['noresign']
        for k in ('resigned', 'noresign', 'would', 'wrong'):
            tot[k] += rs[k]
        if remote and sp['remote']['accepted']:
            assert rs['remote']['games'] == sp['remote']['accepted'], (rs['remote'], sp['remote'])
    assert tot['resigned'] > 0 and tot['noresign'] > 0 and tot['would'] > 0, tot
    return tot


def test_a_rolling_run_resigns_with_exact_quotas(tmp_path):
    run = tmp_path / 'rs'
    p = run_loop(run, '--iters', '3', '--rolling', '1', '--games', '96', '--conc', '32', '--selfplay-resign-t', '0.3',
                 '--selfplay-resign-k', '2', '--selfplay-resign-noresign', '0.3')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    for r in m[1:]:
        assert r['selfplay']['games'] == 96, r['selfplay']['games']
    check_resign(m, 96)
    assert 'resign: ' in (run / 'log.txt').read_text(encoding='utf-8')
    # off by default: no metrics
    run0 = tmp_path / 'off'
    p = run_loop(run0, '--iters', '1', '--rolling', '1')
    assert p.returncode == 0, p.stderr[-2000:]
    assert all('resign' not in r['selfplay'] for r in read_metrics(str(run0))[1:])


def test_remote_workers_resign_and_the_quotas_stay_exact(tmp_path):
    """Remote self-play (the remote 'local') with the local linger: the workers resign as this machine (their
    hello agrees), every iteration exactly its quota with resigned games counted as games, the remote's tally in
    the iteration's `resign`."""
    run = tmp_path / 'rr'
    p = run_loop(run, '--iters', '3', *PAIR, '--selfplay-remote', 'local', '--selfplay-local-linger', '1',
                 '--selfplay-resign-t', '0.3', '--selfplay-resign-k', '2', '--selfplay-resign-noresign', '0.3',
                 env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 192)
    check_resign(m, 192, remote=True)
    assert any(r['selfplay']['resign'].get('remote', {}).get('resigned', 0) > 0 for r in m[1:]), \
        [r['selfplay']['resign'].get('remote') for r in m[1:]]
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')


def test_an_older_worker_is_refused(tmp_path):
    run = tmp_path / 'old'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--selfplay-resign-t', '0.3',
                 env=pair_env(tmp_path, {'no_resign': 1}))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and 'self-play resignation' in log, log[-3000:]
    check_quota(read_metrics(str(run)), 192)
