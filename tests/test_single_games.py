"""Single games (docs/az.md 82; zero-farm PROTOCOL 22): a game id is one game, or one whole contest of several (a
best-of-three match); every count of single games is exact or absent, never an estimate - `zero.games.single_games`
(a chained contest's episodes, else the game package's `id_games`), the farm pool's ack and metrics, a farm
evaluation's ack, the run's own metrics rows, and the own-machines report's `single_games` / `uncounted`."""
import json

import pytest

from zero import farm_eval as FE, farm_pool as FP, farm_report as fr, games as G, package
from zero.infer import drive
from zero.workqueue import split_games

from test_farm_pool import FARM, Hashed, farm_games, header, pool, records_file, wait  # noqa: F401  (pool: a fixture)
from test_farm_report import NOW, ey, jsonl, mrow


def acks(fake):
    return [b for m, p, b in fake.calls if m == 'POST' and p == f'/v1/admin/run/{FARM}/ack']


def ack_counts(fake):
    out = {}
    for b in acks(fake):
        for rid, n in b.get('single_games') or []:
            assert rid not in out or out[rid] == n
            out[rid] = n
    return out


# ------------------------------------------------------------------ the rule

def test_a_chained_contest_counts_its_episodes_and_a_plain_game_what_the_package_declares(monkeypatch):
    assert G.single_games('toy', '', {'gid': 0}) == 1
    eps = [{'over': True}, {'over': True}, {'over': True}]
    assert G.single_games('toy_match', '', {'gid': 0, 'episodes': eps}) == 3
    assert G.single_games('toy_match', '', {'gid': 0, 'episodes': eps[:2]}) == 2
    # a chained contest's row without its episodes (an untraced farm eval row): not known, never the package's number
    assert G.single_games('toy_match', '', {'gid': 0}) is None
    assert G.single_games('toy_match', '', {'gid': 0, 'episodes': []}) is None
    # episodes on a game that does not chain: not known
    assert G.single_games('toy', '', {'gid': 0, 'episodes': eps}) is None
    assert G.single_games_sum('toy_match', '', [{'episodes': eps}, {'episodes': eps[:2]}]) == 5
    assert G.single_games_sum('toy_match', '', [{'episodes': eps}, {}]) is None
    pkg = package.active()
    # a game the package does not list, a spec it cannot count, a contest of a fixed number of games an id
    monkeypatch.setattr(pkg, 'id_games', {})
    assert G.single_games('toy', '', {'gid': 0}) is None
    monkeypatch.setattr(pkg, 'id_games', {'toy': lambda spec: None if 'match' in spec else 2})
    assert G.single_games('toy', '', {}) == 2 and pkg.id_games_of('toy', '{"match": 1}') is None
    for bad in (0, True, 1.5):
        monkeypatch.setattr(pkg, 'id_games', {'toy': bad})
        with pytest.raises(ValueError, match='id_games'):
            G.single_games('toy', '', {})


# ------------------------------------------------------------------ the farm pool

def test_the_pool_acks_each_results_single_games_and_counts_the_accepted_ones(pool):
    from zero.remoteplay import RemotePlay
    fake, R, _ = pool
    seed = FP.coordinator_seed(FARM)
    games = farm_games(20, seed)
    by = {gid: t for gid, t, _ in games}
    order = [gid for gid, _, _ in games]
    ga, gb = [g for g in order if g < 12], [g for g in order if g >= 12]
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        fake.add('m_a', [[0, 12]], records_file(header(seed), [by[g] for g in ga]))                 # result 1
        tam = json.loads(by[order[0]])
        tam['steps'][2] = 999
        fake.add('m_bad', [[100, 101]], records_file(header(seed), [json.dumps(dict(tam, gid=100))]))   # 2: refused
        fake.add('m_b', [[12, max(order) + 1]], records_file(header(seed), [by[g] for g in gb]))    # result 3
        # 4: a result credited ids its file does not hold - no count for it (never a partial one)
        fake.add('m_c', [[200, 202]], records_file(header(seed), []))
        wait(lambda: fake.acked == 4, 'acked')
        assert ack_counts(fake) == {1: len(ga), 3: len(gb)}
        assert all(set(b) <= {'cursor', 'single_games'} for b in acks(fake))
        wait(lambda: rp.need(1) == 24 - 20, 'the games accepted')
        pr = rp.end(1)['pools'][f'pool:{FARM}']
        assert pr['games'] == 20 and pr['single_games'] == 20 and pr['single_uncounted'] == 0
        assert pr['single_machines'] == {'m_a': len(ga), 'm_b': len(gb)} == pr['machines']
    finally:
        rp.close()


def match_games(n, seed, sims=8):
    """`farm_games` for the chained toy match: [(gid, record text, its contest row)]."""
    seat = G.nn_seat('m', sims, selfplay=True)
    drv = G.Driver('toy_match', [[seat, seat]], spec='', concurrency=6, seed=seed, record=True, threads=2, groups=1,
                   rolling=True, aux=True)
    drv.set_trace(True)
    out = []
    while len(out) < n:
        drv.resume(n - len(out), 0)
        drive(drv, {'m': Hashed()})
        rows, texts, s = drv.take_game_records()
        for (row, _), t in zip(split_games(rows, s), texts):
            out.append((row['gid'], t, row))
    return out[:n]


@pytest.mark.parametrize('rolling', ['', 'auto'])
def test_the_pool_counts_the_games_of_matches(pool, monkeypatch, rolling):
    """A chained game (the toy's best of three): an id is a match, the ack and the metrics carry its games - the
    episodes its record lists, which the replay checked."""
    from zero.remoteplay import RemotePlay
    fake, R, _ = pool
    info = G.info('toy_match')
    R.cfg.update(game='toy_match', games=12, pool_rolling=rolling)
    R.info = info
    monkeypatch.setattr(FP, 'play_settings',
                        lambda worker, build, cfg, d, log=None: {'game': 'toy_match', 'spec': '', 'players': [2]})
    monkeypatch.setattr(FP.Pool, '_check_rolling', lambda self: None)
    seed = FP.coordinator_seed(FARM)
    games = match_games(12, seed)
    eps = {gid: len(row['episodes']) for gid, _, row in games}
    assert all(2 <= n <= 3 for n in eps.values())
    h = dict(header(seed), game='toy_match', features=info['num_features'], actions=info['num_actions'])
    ids = sorted(eps)
    a, b = ids[:7], ids[7:]
    text = {gid: t for gid, t, _ in games}
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    rp.pools[f'pool:{FARM}'].st['build_game'] = 'toy'      # (the worker builds' game: a slug)
    try:
        rp.begin(1)
        try:
            wait(lambda: 0 in fake.ckpts, 'gen 0 registered', t=20)
        except AssertionError:
            raise AssertionError(' | '.join(R.lines[-10:]))
        fake.add('m_a', [[g, g + 1] for g in a], records_file(h, [text[g] for g in a]))
        fake.add('m_b', [[g, g + 1] for g in b], records_file(h, [text[g] for g in b]))
        wait(lambda: fake.acked == 2, 'acked')
        assert ack_counts(fake) == {1: sum(eps[g] for g in a), 2: sum(eps[g] for g in b)}
        wait(lambda: rp.need(1) == 0, 'the quota met')
        pr = rp.end(1)['pools'][f'pool:{FARM}']
        assert pr['games'] == 12 and pr['single_games'] == sum(eps.values()) > 12 and pr['single_uncounted'] == 0
        assert pr['single_machines'] == {'m_a': sum(eps[g] for g in a), 'm_b': sum(eps[g] for g in b)}
        if rolling:
            assert {m: (c['games'], c['single_games'], c['single_uncounted']) for m, c in pr['by_machine'].items()} == \
                {'m_a': (7, sum(eps[g] for g in a), 0), 'm_b': (5, sum(eps[g] for g in b), 0)}
    finally:
        rp.close()


def test_a_spec_the_package_cannot_count_is_acked_without_a_count(pool, monkeypatch):
    from zero.remoteplay import RemotePlay
    fake, R, _ = pool
    monkeypatch.setattr(package.active(), 'id_games', {})
    seed = FP.coordinator_seed(FARM)
    games = farm_games(6, seed)
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        fake.add('m_a', [[0, max(g for g, _, _ in games) + 1]], records_file(header(seed), [t for _, t, _ in games]))
        wait(lambda: fake.acked == 1, 'acked')
        assert acks(fake)[-1] == {'cursor': 1}, 'the ack of before'
        wait(lambda: rp.need(1) == 24 - 6, 'accepted')
        pr = rp.end(1)['pools'][f'pool:{FARM}']
        assert (pr['games'], pr['single_games'], pr['single_uncounted'], pr['single_machines']) == (6, 0, 6, {})
    finally:
        rp.close()


# ------------------------------------------------------------------ farm evaluations

def test_a_farm_evaluations_single_games():
    rows = [{'gid': g, 'result': [1.0, 0.0]} for g in range(4)]
    r = {'id': 9, 'games': 4}
    assert FE.credited_single_games('toy', '', r, rows) == 4
    assert FE.credited_single_games('toy', '', r, rows + rows[:1]) == 4, 'a second copy of an id counts once'
    assert FE.credited_single_games('toy', '', r, rows[:3]) is None, 'a credited id is missing'
    assert FE.credited_single_games('toy', '', {'id': 9, 'games': 0}, []) is None
    # a chained contest: an untraced row does not say how many games its match took
    assert FE.credited_single_games('toy_match', '', r, rows) is None
    # a traced row's record lists them - when its digest is the record's
    traced = []
    for g, k in enumerate((2, 3, 3, 2)):
        rec = {'gid': g, 'episodes': [{'over': True}] * k}
        traced.append({'gid': g, 'rec': rec, 'digest': FE.trace_digest(rec)})
    assert FE.credited_single_games('toy_match', '', r, traced) == 10
    traced[1] = dict(traced[1], digest='0' * 64)
    assert FE.credited_single_games('toy_match', '', r, traced) is None
    assert FE.ack_body(7, {}) == {'cursor': 7} == FE.ack_body(7, {3: None, None: 4})
    assert FE.ack_body(7, {5: 12, 3: 4, 4: None}) == {'cursor': 7, 'single_games': [[3, 4], [5, 12]]}


# ------------------------------------------------------------------ the own-machines report

def test_the_report_splits_single_games_from_uncounted_ids(tmp_path):
    root = tmp_path / 'runs'
    t = NOW - 600
    rows = [
        # an old row (no counts): 100 self-play ids and 10 eval ids, all uncounted
        mrow(1, t - 7200, 100, eval_yard=ey(10)),
        # matches: 40 ids = 100 games here; a remote's 20 ids carry no count; eval 8 matches = 19 games; an extra
        # evaluation's games are never counted
        mrow(2, t, 40, remotes={'mac': {'games': 20, 'games_played': 20, 'seconds': 50.0, 'running_s': 50.0}},
             eval_yard=dict(ey(8), single_games=19), extras={'x': {'games': 6}}),
    ]
    rows[1]['selfplay']['single_games'] = 100
    jsonl(str(root / 'alpha' / 'metrics.jsonl'), rows, mtime=NOW - 60)
    with open(tmp_path / 'cfg.json', 'w', encoding='utf-8') as f:
        json.dump({'fuzz': None, 'vast_ledger': ''}, f)
    cfg = fr.load_config(str(tmp_path / 'cfg.json'))
    body, _ = fr.build_report(cfg, [str(root)], NOW, fuzz=False)
    s = {x['id']: x for x in body['sources']}
    local = s[cfg['local']['id']]
    assert local['selfplay_games'] == {'1h': 40, '24h': 140} and local['eval_games'] == {'1h': 14, '24h': 24}
    assert local['single_games'] == {'1h': 100 + 19, '24h': 100 + 19}
    assert local['uncounted'] == {'1h': 6, '24h': 6 + 100 + 10}
    mac = s['mac']
    assert mac['selfplay_games'] == {'1h': 20, '24h': 20}
    assert mac['single_games'] == {'1h': 0, '24h': 0} and mac['uncounted'] == {'1h': 20, '24h': 20}
    # a count that its ids cannot hold (fewer games than ids) is not one
    assert fr.single_of({'games': 5, 'single_games': 4}) is None and fr.single_of({'games': 5, 'single_games': 5}) == 5
    assert fr.single_of({'games': 5}) is None and fr.single_of({'games': 0, 'single_games': 0}) is None


# ------------------------------------------------------------------ the run's own rows

def test_a_runs_rows_record_their_single_games(tmp_path):
    from test_loop import run_loop
    from zero.run import read_metrics
    for game, kw in (('toy', []), ('toy_match', ['--yardstick', 'heuristic'])):
        run = tmp_path / game
        p = run_loop(run, '--iters', '2', '--game', game, *kw)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        m = read_metrics(str(run))
        assert len(m) >= 2
        seen = 0
        for r in m:
            sp = r.get('selfplay')
            if sp and sp.get('games'):
                seen += 1
                if game == 'toy':
                    assert sp['single_games'] == sp['games']
                else:
                    assert sp['single_games'] == sp['episodes']['games'] > sp['games']
            for k in ('eval_yard', 'eval_random'):
                e = r.get(k)
                if e and e.get('games'):
                    assert e['single_games'] == (e['games'] if game == 'toy' else e['per_game']['games'])
        assert seen
