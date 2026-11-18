"""The playout cap's full searches by the prior's uncertainty (zero/budget.py, docs/az.md 60, docs/search.md 3.21) on
the toy: the seat dict (off: the seat of before), the settings refused, the bindings (a floor at p_full is the coin;
the rule's share matches p_full with p_full_u_mean measured; the counters), the run settings, a rolling toy run with
the rule (its `budget` metrics), and remote self-play (the workers draw as this machine; an older checkout refused by
its hello)."""
import pytest

from zero import budget as BU
from zero import games as G
from zero.run import DEFAULTS, read_metrics

from test_loop import run_loop
from test_remoteplay import PAIR, check_quota, pair_env
from test_resign import driver, play

RULE = dict(p_full_rule='prior_uncertainty', p_full_floor=0.05, p_full_shape='linear', p_full_u_mean=0.4)


def seat(**kw):
    return G.nn_seat('m', 8, selfplay=True, temp_moves=4, sims_fast=2, p_full=0.25, **kw)


def run_stats(s, games=40):
    d = driver(s, games=games)
    rows = play(d)
    return rows, d.stats()


def test_the_seat_dict():
    assert seat() == seat(p_full_rule='random', p_full_floor=0.3, p_full_shape='square', p_full_u_mean=None)
    assert not set(BU.KEYS) & set(seat())
    s = seat(**RULE)
    assert {k: s[k] for k in BU.KEYS} == RULE
    assert BU.seat_keys(DEFAULTS) == {} and BU.fingerprint(DEFAULTS) == ''
    cfg = dict(DEFAULTS, **RULE)
    assert BU.seat_keys(cfg) == RULE and BU.fingerprint(cfg) == 'prior_uncertainty floor=0.05 shape=linear u_mean=0.4'
    # a stale mean is said loudly (the coin never drifts)
    assert BU.drift({'f_mean': 0.26, 'g_mean': 0.41}, cfg) == '' and BU.drift({'f_mean': 0.4, 'g_mean': 0.9}, DEFAULTS) == ''
    assert 'p_full_u_mean' in BU.drift({'f_mean': 0.31, 'g_mean': 0.5}, cfg)


@pytest.mark.parametrize('kw,msg', [
    ({'p_full_rule': 'coin'}, 'p_full_rule'),
    (dict(RULE, p_full_floor=0.5), 'p_full_floor 0.5'),
    (dict(RULE, p_full_shape='cubic'), 'p_full_shape'),
    (dict(RULE, p_full_u_mean=None), 'p_full_u_mean'),
    (dict(RULE, p_full_u_mean=0.0), 'p_full_u_mean'),
    (dict(RULE, root_search='gumbel', policy_target='cq'), 'gumbel'),
])
def test_bad_settings_are_refused(kw, msg):
    with pytest.raises(ValueError) as e:
        seat(**kw)
    assert msg in str(e.value), str(e.value)
    with pytest.raises(ValueError) as e:
        G.nn_seat('m', 8, selfplay=True, **RULE)          # no sims_fast: no playout cap
    assert 'sims_fast' in str(e.value)


def test_the_driver_checks_a_seat_dict_too():
    for d, msg in [(dict(seat(), p_full_floor=0.1), 'p_full_floor: a setting of p_full_rule'),
                   (dict(seat(), p_full_rule='prior_uncertainty'), 'p_full_u_mean'),
                   (dict(seat(**RULE), p_full_floor=0.9), 'p_full_floor 0.9')]:
        with pytest.raises(Exception) as e:
            driver(d, games=2)
        assert msg in str(e.value), (msg, str(e.value))


def test_the_bindings():
    # the coin: its counters (u's mean) and the stats keys; a floor at p_full plays the coin's games exactly
    rows0, st0 = run_stats(seat())
    b0 = BU.summary(st0)
    assert b0['decisions'] > 200 and abs(b0['f_mean'] - 0.25) < 1e-6 and 0.0 < b0['g_mean'] < 1.0
    assert abs(b0['sims_per_decision'] - (2 + 6 * b0['full_share'])) < 1e-3
    rows1, st1 = run_stats(seat(**dict(RULE, p_full_floor=0.25)))
    assert [(r['gid'], r['steps'], r['samples'], r['result']) for r, _ in rows0] == \
           [(r['gid'], r['steps'], r['samples'], r['result']) for r, _ in rows1]
    assert {k: st0[k] for k in BU.CAP} == {k: st1[k] for k in BU.CAP}
    # the rule with u's measured mean: the mean chance and the drawn share near p_full
    _, st2 = run_stats(seat(**dict(RULE, p_full_u_mean=b0['g_mean'])))
    b2 = BU.summary(st2)
    assert abs(b2['f_mean'] - 0.25) < 0.04 and abs(b2['full_share'] - b2['f_mean']) < 0.05, b2
    assert 'iter' not in BU.fmt(b2, dict(DEFAULTS, **RULE)) and 'p_full_u_mean' in BU.fmt(b2, dict(DEFAULTS, **RULE))
    # a driver without a playout cap reports none
    _, st3 = run_stats(G.nn_seat('m', 8, selfplay=True, temp_moves=4), games=4)
    assert 'cap_decisions' not in st3 and BU.summary(st3) is None


def test_run_settings_are_checked(tmp_path):
    assert (DEFAULTS['p_full_rule'], DEFAULTS['p_full_floor'], DEFAULTS['p_full_shape'], DEFAULTS['p_full_u_mean']) == \
        ('random', 0.05, 'linear', 0.0)
    for args, msg in [(['--p-full-rule', 'prior_uncertainty'], 'p_full_u_mean'),
                      (['--p-full-rule', 'prior_uncertainty', '--p-full-u-mean', '0.3', '--p-full-floor', '0.9'],
                       'p_full_floor 0.9'),
                      (['--p-full-rule', 'prior_uncertainty', '--p-full-u-mean', '0.3', '--selfplay-root-search',
                        'gumbel', '--policy-target', 'cq'], 'gumbel')]:
        p = run_loop(tmp_path / f'r{len(msg)}{len(args)}', '--iters', '1', *args)
        assert p.returncode != 0 and msg in p.stderr, (args, p.stderr[-500:])


def test_a_rolling_run_with_the_rule(tmp_path):
    run = tmp_path / 'pu'
    p = run_loop(run, '--iters', '2', '--rolling', '1', '--games', '48', '--conc', '16', '--p-full-rule',
                 'prior_uncertainty', '--p-full-u-mean', '0.5', '--p-full-floor', '0.1')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    for r in m[1:]:
        b = r['selfplay']['budget']
        assert r['selfplay']['games'] == 48 and b['decisions'] > 0 and 0.1 <= b['f_mean'] <= 1.0, b
    assert 'budget: ' in (run / 'log.txt').read_text(encoding='utf-8')


def test_remote_workers_draw_as_this_machine(tmp_path):
    run = tmp_path / 'rb'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--p-full-rule', 'prior_uncertainty',
                 '--p-full-u-mean', '0.5', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    check_quota(read_metrics(str(run)), 192)
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')


def test_an_older_worker_is_refused(tmp_path):
    run = tmp_path / 'old'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--p-full-rule', 'prior_uncertainty',
                 '--p-full-u-mean', '0.5', env=pair_env(tmp_path, {'no_budget': 1}))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and 'playout cap' in log, log[-3000:]
    check_quota(read_metrics(str(run)), 192)
