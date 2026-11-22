"""Near-tie branches (`selfplay_branch_rule` 'neartie', zero/forks.py, docs/az.md 72, docs/search.md 2.11.1) on the
toy: the bindings (the branch dict, rows' and stats' new keys, refusals), the run settings and their checks, the
driver kwargs / fingerprint / description, the tally and its summary, a rolling toy run and remote self-play with
the rule on (the uniform rule's fingerprint and tally keys unchanged)."""
import pytest

from zero import forks as F
from zero import games as G
from zero.run import DEFAULTS, read_metrics

from test_forks import play
from test_loop import run_loop
from test_remoteplay import PAIR, check_quota, pair_env

LOOSE = {'eps': 0.5, 'n_min': 1, 'p_min': 0.0, 'q_lo': 0.0, 'q_hi': 1.0}


def seat(sims=16):
    return G.nn_seat('m', sims, selfplay=True, temp_moves=4)


def test_the_bindings_play_neartie_branches():
    d = G.Driver('toy', [[seat(), seat()]] * 30, spec='{}', concurrency=16, seed=5, record=True, threads=2,
                 branch={'p': 0.5, 'rule': 'neartie', **LOOSE})
    rows = play(d)
    main = [r for r, _ in rows if 'branch' not in r]
    br = [r for r, _ in rows if 'branch' in r]
    assert len(main) == 30 and 5 <= len(br) <= 25, (len(main), len(br))
    assert all(set(r['branch']) == {'at', 'of', 'decision', 'step', 'action', 'moves', 'played', 'same_first', 'start',
                                    'alt', 'gap', 'tag', 'merged'} for r in br)
    for r in br:
        b = r['branch']
        assert b['alt'] != b['action'] and b['moves'] == b['played'] == 1 and not b['same_first'], b
        assert b['merged'] is None and 0.0 <= b['gap'] <= 0.5 and b['tag'] == 0, b
        assert 'neartie' not in r
    # every game counts its eligible decisions; the stats sum them
    assert all(isinstance(r['neartie'], int) and r['neartie'] > 0 for r in main)
    s = d.stats()
    assert s['neartie_seen'] == sum(r['neartie'] for r in main) and s['merges'] == s['merge_decisions'] == 0
    assert s['branches'] == len(br) == s['dev_moves']
    assert F.setups([r for r, _ in rows]) == 30
    # the uniform rule's rows and stats have none of the new keys
    d0 = G.Driver('toy', [[seat(), seat()]] * 10, spec='{}', concurrency=8, seed=5, record=True, threads=2,
                  branch={'p': 1.0, 'moves': 1})
    rows0 = play(d0)
    assert all('neartie' not in r and 'alt' not in r.get('branch', {}) for r, _ in rows0)
    assert 'neartie_seen' not in d0.stats()


@pytest.mark.parametrize('kw,msg', [
    ({'p': 0.5, 'rule': 'neartie', 'moves': 1}, "uniform rule's settings"),
    ({'p': 0.5, 'rule': 'neartie', 'temp': 2.0}, "uniform rule's settings"),
    ({'p': 0.5, 'eps': 0.1}, 'near-tie rule'),
    ({'p': 0.5, 'rule': 'best'}, "'uniform' or 'neartie'"),
    ({'p': 0.5, 'rule': 'neartie', 'n_min': 0}, 'n_min'),
    ({'p': 0.5, 'rule': 'neartie', 'q_lo': 0.9, 'q_hi': 0.1}, 'q_lo'),
])
def test_the_bindings_refuse_bad_neartie_settings(kw, msg):
    with pytest.raises(Exception) as e:
        G.Driver('toy', [[seat(), seat()]], spec='{}', concurrency=2, seed=1, record=True, branch=kw)
    assert msg in str(e.value), str(e.value)


def test_run_settings_kwargs_and_fingerprint():
    for k, v in F.NEARTIE.items():
        assert DEFAULTS[k] == v, k
    assert DEFAULTS['selfplay_branch_rule'] == 'uniform'
    base = dict(DEFAULTS, rolling=1, game='toy', spec='{}')
    F.check_settings(dict(base, selfplay_branch=0.2, selfplay_branch_rule='neartie'))
    for patch, msg in [({'selfplay_branch_rule': 'nearest'}, 'selfplay_branch_rule'),
                       ({'selfplay_branch_eps': -0.1}, 'selfplay_branch_eps'),
                       ({'selfplay_branch_n_min': 0}, 'selfplay_branch_n_min'),
                       ({'selfplay_branch_n_min': 2.5}, 'selfplay_branch_n_min'),
                       ({'selfplay_branch_p_min': 1.5}, 'selfplay_branch_p_min'),
                       ({'selfplay_branch_q_lo': 0.95}, 'selfplay_branch_q_lo'),
                       ({'selfplay_branch_rule': 'neartie', 'selfplay_branch_moves': 2}, "uniform rule's settings"),
                       ({'selfplay_branch_rule': 'neartie', 'selfplay_branch_differ': 1}, "uniform rule's settings"),
                       ({'selfplay_branch_eps': 0.05}, 'settings of selfplay_branch_rule neartie')]:
        with pytest.raises(SystemExit) as e:
            F.check_settings(dict(base, selfplay_branch=0.2, **patch))
        assert msg in str(e.value), (patch, str(e.value))
    # off (selfplay_branch 0): the rule's settings are only range-checked
    F.check_settings(dict(base, selfplay_branch_eps=0.05))
    fk = F.Forks('.', dict(base, selfplay_branch=0.2, selfplay_branch_rule='neartie', selfplay_branch_n_min=8))
    assert fk.kwargs() == {'branch': {'p': 0.2, 'rule': 'neartie', 'eps': 0.02, 'n_min': 8, 'p_min': 0.02, 'q_lo': 0.1,
                                      'q_hi': 0.9}}
    assert F.fingerprint(fk) == 'branch p=0.2 rule=neartie eps=0.02 n_min=8 p_min=0.02 q_lo=0.1 q_hi=0.9'
    assert 'near-tie' in fk.describe()
    # the uniform rule: the kwargs and fingerprint of before (an older worker still agrees)
    fu = F.Forks('.', dict(base, selfplay_branch=0.2))
    assert fu.kwargs() == {'branch': {'p': 0.2, 'moves': 3, 'temp': 2.0, 'differ': False}}
    assert F.fingerprint(fu) == 'branch p=0.2 moves=3 temp=2.0'


def test_the_tally_and_its_summary():
    rows = [{'samples': 10, 'decisions': 12, 'neartie': 3}, {'samples': 8, 'decisions': 9, 'neartie': 0},
            {'samples': 2, 'decisions': 9, 'branch': {'at': 1, 'of': 3, 'moves': 1, 'played': 1, 'same_first': False,
                                                      'alt': 2, 'gap': 0.01, 'tag': 7, 'merged': 2}},
            {'samples': 5, 'decisions': 11, 'branch': {'at': 0, 'of': 3, 'moves': 1, 'played': 1, 'same_first': False,
                                                       'alt': 1, 'gap': 0.03, 'tag': 4, 'merged': None}}]
    acc = F.tally(rows, F.new_tally())
    nt = acc['neartie']
    assert (nt['games'], nt['seen'], nt['branches'], nt['merges'], nt['merge_decisions'], nt['merge_samples']) == (2, 3, 2, 1, 2, 2)
    assert nt['tags'] == {'7': [1, 1], '4': [1, 0]}
    both = F.merge(acc, F.tally(rows[2:3], F.new_tally()))
    assert both['neartie']['merges'] == 2 and both['neartie']['tags']['7'] == [2, 2]
    # an older remote's tally without the key merges as zeros
    old = F.new_tally()
    del old['neartie']
    assert F.merge(acc, old)['neartie']['branches'] == 2
    s = F.summary(acc)
    assert s['neartie']['seen_per_game'] == 1.5 and s['neartie']['merge_share'] == 0.5
    assert s['neartie']['decisions_to_merge'] == 2.0 and abs(s['neartie']['gap'] - 0.02) < 1e-9
    assert 'near-tie 1.5 eligible/game, 1 of 2 branches merged' in F.fmt(s)
    # the uniform rule's summary has no near-tie part
    u = F.summary(F.tally([{'samples': 3, 'decisions': 4}, {'samples': 1, 'decisions': 2, 'branch': {
        'at': 0, 'of': 2, 'moves': 2, 'played': 2, 'same_first': True}}], F.new_tally()))
    assert 'neartie' not in u and 'near-tie' not in F.fmt(u)


def test_a_rolling_run_with_neartie_branches(tmp_path):
    run = tmp_path / 'nt'
    p = run_loop(run, '--iters', '2', '--rolling', '1', '--games', '64', '--conc', '24', '--selfplay-branch', '0.5',
                 '--selfplay-branch-rule', 'neartie', '--selfplay-branch-eps', '0.5', '--selfplay-branch-n-min', '1',
                 '--selfplay-branch-q-lo', '0', '--selfplay-branch-q-hi', '1')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    nb = 0
    for r in m[1:]:
        fk = r['selfplay']['forks']
        assert r['selfplay']['games'] == 64 and sum(fk['games'].values()) == 64, fk
        assert fk['neartie']['seen_per_game'] > 0, fk
        nb += fk['games']['branch']
        assert fk['neartie']['branches'] == fk['games']['branch'] and fk['neartie']['merges'] == 0
    assert nb > 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'near-tie decisions' in log and 'near-tie ' in log and 'eligible/game' in log


def test_remote_workers_play_neartie_branches(tmp_path):
    run = tmp_path / 'rnt'
    p = run_loop(run, '--iters', '2', *PAIR, '--selfplay-remote', 'local', '--selfplay-branch', '0.4',
                 '--selfplay-branch-rule', 'neartie', '--selfplay-branch-eps', '0.5', '--selfplay-branch-n-min', '1',
                 env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 192)
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')
    rem = [r['selfplay']['forks'].get('remote') for r in m[1:]]
    assert any(x and x['games']['branch'] > 0 for x in rem), rem
    assert any(r['selfplay']['forks'].get('neartie', {}).get('branches', 0) > 0 for r in m[1:])
