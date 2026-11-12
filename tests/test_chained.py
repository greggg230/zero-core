"""Chained episodes (docs/search.md 1.3, 2.14; docs/az.md 42): the toy `toy_match` (a best of three toy games, each
game its own episode) through the bindings, the Rust and the Python drivers, the loop (self-play's per-game labels, the
evaluation's match and per-game numbers), h2h, resignation's tally, replays, remote self-play (the hello's fingerprint:
an older worker refused) and remote evaluation (its result's marker). Games that do not chain: no row gains a key."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from zero import games as G
from zero import resign as RS
from zero.games import engine
from zero.infer import UniformEvaluator, drive
from zero.play import NNSeat, PyDriver
from zero.run import read_metrics
from zero.workqueue import split_games

from test_loop import run_loop
from test_remoteplay import check_quota, pair_env

TOY_F = engine().game_info_of('toy')['num_features']


def play_rows(drv):
    """Every batch answered by the uniform evaluator -> rows with their own samples (split_games' contract)."""
    rows = []
    ev = UniformEvaluator()

    def on_batch(lb):
        rows.extend(split_games(drv.take_results(), drv.take_samples()))
    drive(drv, {'m': ev, 'o': ev}, on_batch=on_batch)
    rows.extend(split_games(drv.take_results(), drv.take_samples()))
    return sorted(rows, key=lambda rs: rs[0]['gid'])


def check_contest(r, s):
    """A contest row: the match's rules, its sums, every sample labelled with its own game's outcome (z, sc)."""
    eps = r['episodes']
    assert r['over'] and 2 <= len(eps) <= 3
    wins = [0, 0]
    for k, e in enumerate(eps):
        assert e['over'] and not e['aborted']
        for p in (0, 1):
            wins[p] += e['result'][p] > 0.75
        assert (max(wins) >= 2 or k == 2) == (k == len(eps) - 1), (k, len(eps), wins)
    want = [1.0, 0.0] if wins[0] > wins[1] else [0.0, 1.0] if wins[1] > wins[0] else [0.5, 0.5]
    assert r['result'] == want and r['scores'] == [float(w) for w in wins]
    assert r['samples'] == sum(e['samples'] for e in eps) == (0 if s is None else len(s['player']))
    assert r['decisions'] == sum(e['decisions'] for e in eps) and r['steps'] == sum(e['steps'] for e in eps)
    a = 0
    for e in eps:
        b = a + e['samples']
        for i in range(a, b):
            p = int(s['player'][i])
            for j in range(2):
                assert s['z'][i, j] == pytest.approx(e['result'][(p + j) % 2])
                assert s['sc'][i, j] == pytest.approx(e['margins'][(p + j) % 2])
        a = b


def test_the_bindings_play_a_match_by_episodes():
    E = engine()
    g = E.Game('toy_match', '{}', 2, 3)
    assert g.chained and g.can_concede and not E.Game('toy', '{}', 2, 3).chained
    with pytest.raises(ValueError, match='not chain'):
        E.Game('toy', '{}', 2, 3).next_episode()
    with pytest.raises(ValueError, match='has not ended'):
        g.next_episode()
    nxt, res = g.next_episode(conceded=0)                 # seat 0 concedes game 1: seat 1 leads 1-0
    assert res is None and nxt.scores() == [0.0, 1.0] and nxt.to_move == 0 and not nxt.can_concede
    with pytest.raises(ValueError, match='no game is in progress'):
        nxt.next_episode(conceded=0)
    rng = np.random.default_rng(1)
    for seed in range(20):
        g, games = E.Game('toy_match', '{}', 2, seed), 1
        while True:
            if g.over:
                nxt, res = g.next_episode()
                if nxt is None:
                    assert max(g.scores()) == 2 or games == 3
                    w = g.scores()
                    assert res == ([1.0, 0.0] if w[0] > w[1] else [0.0, 1.0] if w[1] > w[0] else [0.5, 0.5])
                    break
                g, games = nxt, games + 1
                continue
            legal = g.legal()
            g.step(legal[rng.integers(len(legal))])
    assert G.chained('toy_match', '{}') and not G.chained('toy', '{}')
    assert G.episodes_tag('toy_match', '') == 'chained' and G.episodes_tag('toy', '') == ''


def test_the_rust_driver_hands_out_contests_with_their_games_samples():
    seat = G.nn_seat('m', 12, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=4)
    rows = play_rows(G.Driver('toy_match', [[seat, seat]] * 24, concurrency=12, seed=3, record=True, threads=2))
    assert len(rows) == 24
    for r, s in rows:
        check_contest(r, s)
        # a start choice (the match block's flag) is the first decision of a later game: never in game 1's rows
        k1 = r['episodes'][0]['samples']
        assert not np.any(s['x'][:k1, TOY_F + 4] > 0.5)
    assert sum(int(np.sum(s['x'][:, TOY_F + 4] > 0.5)) for _, s in rows) > 0
    # a game that does not chain: rows of before, no key added
    plain = play_rows(G.Driver('toy', [[seat, seat]] * 4, concurrency=4, seed=3, record=True, threads=2))
    assert all('episodes' not in r for r, _ in plain)


def test_the_python_driver_chains_the_same_way():
    seat = NNSeat('m', 12, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=4)
    drv = PyDriver('toy_match', 12, lambda g: [seat, seat], concurrency=6, seed=2, record=True)
    rows = play_rows(drv)
    assert len(rows) == 12
    for r, s in rows:
        check_contest(r, s)
    plain = play_rows(PyDriver('toy', 3, lambda g: [seat, seat], concurrency=3, seed=2, record=True))
    assert all('episodes' not in r for r, _ in plain)


def test_h2h_and_the_resign_tally_read_the_games():
    from zero.h2h import summary
    seat = G.nn_seat('m', 8)
    rows = [r for r, _ in play_rows(G.Driver('toy_match', [[seat, {'bot': 'random'}], [{'bot': 'random'}, seat]] * 8,
                                             concurrency=16, seed=4, threads=2))]
    s = summary(rows)
    pg = s['per_game']
    assert s['games'] == 16 and pg['games'] == sum(len(r['episodes']) for r in rows)
    assert pg['score'] == pytest.approx(sum(e['result'][r['gid'] % 2] for r in rows for e in r['episodes']))
    # resignation counts the games of a contest (each with its own `resign`), never the contest
    fake = [{'gid': 0, 'over': True, 'result': [1.0, 0.0], 'samples': 9, 'decisions': 20, 'episodes': [
        {'over': True, 'result': [1.0, 0.0], 'samples': 4, 'decisions': 9,
         'resign': {'resigned': True, 'noresign': False, 'winner': 0, 'decision': 9, 'step': 30, 'records': 4}},
        {'over': True, 'result': [0.0, 1.0], 'samples': 5, 'decisions': 11,
         'resign': {'resigned': False, 'noresign': False, 'winner': None, 'decision': 0, 'step': 0, 'records': 0}}]}]
    acc = RS.tally(fake, RS.new_tally())
    assert acc['games'] == 2 and acc['resigned'] == 1 and acc['resigned_samples'] == 4


@pytest.fixture(scope='module')
def match_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('runs') / 'match'
    p = run_loop(run, '--iters', '2', '--game', 'toy_match', '--yardstick', 'heuristic')
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return run


def test_the_loop_trains_on_games_and_evaluates_matches(match_run):
    """Self-play's rows are contests with their games' samples (selfplay.episodes); the evaluation's rows are matches
    (eval_yard's win rate: the match win rate) with per-game numbers beside them; the Elo log names both."""
    m = read_metrics(str(match_run))
    for r in m[1:]:
        sp, ep = r['selfplay'], r['selfplay']['episodes']
        assert sp['games'] == 24 and ep['contests'] == 24 and 48 <= ep['games'] <= 72
        assert sp['samples'] == pytest.approx(ep['samples_per_game'] * ep['games'], abs=0.06 * ep['games'])
    for r in m:
        for k in ('eval_yard', 'eval_random'):
            e = r[k]
            assert e['games'] == (8 if k == 'eval_yard' else 4) and 2 * e['games'] <= e['per_game']['games'] <= 3 * e['games']
    shards = [os.path.join(dp, f) for dp, _, fs in os.walk(match_run / 'data') for f in fs if f.endswith('.npz')]
    n = 0
    for f in shards:
        with np.load(f) as z:
            n += len(z['player'])
            assert set(np.unique(z['z'][:, :2])) <= {0.0, 0.5, 1.0}
    assert n == sum(r['selfplay']['samples'] for r in m[1:])
    log = (match_run / 'log.txt').read_text(encoding='utf-8')
    assert 'episodes:' in log and '(games ' in log


def test_a_replay_holds_the_whole_match(match_run):
    from zero import replay
    reps = replay.record(str(match_run / 'models' / 'iter_0002.pt'), opp='random', games=2, sims=8, chance_k=0, seed=1,
                         log=lambda *x: None)
    for rep in reps:
        assert 2 <= len(rep['games']) <= 3 and rep['over']
        assert rep['states'][rep['final']]['over'] and max(rep['scores']) == 2 or len(rep['games']) == 3
        assert any(s['choosing'] is not None for s in rep['states']), 'the start choice between games is in the replay'


def test_remote_workers_and_an_older_one(tmp_path):
    """Remote self-play of a chained game: the workers' contests count in the shared quota with their games' samples;
    a worker whose checkout reports no chained episodes (the tests' no_episodes: an older checkout, which would play a
    whole match as one episode) is refused by its hello and the run plays on alone."""
    pair = ['--conc', '64', '--games', '48', '--sims', '16', '--sims-fast', '8', '--p-full', '0.5', '--game', 'toy_match']
    run = tmp_path / 'rsp'
    p = run_loop(run, '--iters', '2', *pair, '--selfplay-remote', 'local', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    check_quota(m, 48)
    assert 'REFUSED' not in (run / 'log.txt').read_text(encoding='utf-8')
    old = tmp_path / 'old'
    p = run_loop(old, '--iters', '2', *pair, '--selfplay-remote', 'local', env=pair_env(tmp_path, {'no_episodes': 1}))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    log = (old / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED' in log and 'one episode' in log, log[-3000:]
    check_quota(read_metrics(str(old)), 48)


def test_a_remote_evaluation_must_report_the_episodes(tmp_path):
    from zero import package, remote
    cfg = {'game': 'toy_match', 'spec': '', 'players': [2], 'eval_select': 'visits'}
    assert remote.eval_episodes(cfg) == 'chained' and remote.eval_episodes(dict(cfg, game='toy')) == ''
    ev = remote.RemoteEval.__new__(remote.RemoteEval)      # (no thread: the result's check alone)
    ev.R = SimpleNamespace(run=str(tmp_path), cfg=cfg, log=lambda *a: None)
    ev.name, ev.module, ev.rdir = 'local', 'zero.remote', str(tmp_path / 'mirror')
    ev._ship = lambda it: None
    out = {'iter': 1, 'eval_yard': {'games': 2, 'win': 0.5}}
    ev.t = SimpleNamespace(run_module=lambda *a, **k: SimpleNamespace(returncode=0, stderr='',
                                                                      stdout=remote.RESULT + json.dumps(out)))
    with pytest.raises(RuntimeError, match='one episode'):
        ev._evaluate(1)
    assert package.active() is not None


def test_the_delver_stack_on_a_chained_game(tmp_path):
    """What MTG Zero's Delver runs use, on the toy match (each with its own metrics): rolling self-play, sparse batches,
    the shard writer thread, sparse shards and window, information-set merging with root-view values, value-only samples
    (next_q), the root's belief (an auxiliary head), resignation (each game resigned, the match going on), the eval cache,
    paired Elo games - every iteration exactly its quota of contests, every shard's rows the contests' games' samples."""
    run = tmp_path / 'stack'
    p = run_loop(run, '--iters', '2', '--game', 'toy_match', '--rolling', '1', '--games', '24', '--sparse', '1',
                 '--shard-writer-thread', '1', '--shard-sparse', '1', '--window-sparse', '1', '--selfplay-infoset-merge', '1',
                 '--selfplay-leaf-value-view', 'root', '--selfplay-view-samples', '0.5', '--aux-next-card-w', '0.5',
                 '--selfplay-belief-tilt', '1', '--selfplay-resign-t', '0.3', '--selfplay-resign-k', '2',
                 '--eval-cache', '4096', '--elo-paired', '1', '--elo-games', '4', timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    for r in m[1:]:
        sp = r['selfplay']
        assert sp['games'] == 24 and sp['episodes']['contests'] == 24, sp.get('episodes')
        assert sp['samples'] > 0 and sp['view']['samples'] > 0 and sp['merge']['links'] > 0
        assert sp['resign']['games'] == sp['episodes']['games'], (sp['resign'], sp['episodes'])
    assert sum(r['selfplay']['episodes']['resigned'] for r in m[1:]) > 0
    n = 0
    for dp, _, fs in os.walk(run / 'data'):
        for f in fs:
            if f.endswith('.npz'):
                with np.load(os.path.join(dp, f)) as z:
                    n += len(z['player'])
    assert n == sum(r['selfplay']['samples'] for r in m[1:])
    for r in m:
        assert r['eval_yard']['per_game']['games'] >= 2 * r['eval_yard']['games']


def test_the_league_and_a_remote_evaluation_on_a_chained_game(tmp_path):
    """The self-play league (an older checkpoint in a share of the contests, its seat unrecorded) and remote evaluation
    (the remote 'local': its results carry `episodes`, recorded with their per-game numbers)."""
    run = tmp_path / 'lg'
    p = run_loop(run, '--iters', '3', '--game', 'toy_match', '--games', '16', '--conc', '16', '--selfplay-league', '0.25',
                 '--selfplay-league-pool', '1,2', '--eval-games', '4', '--eval-random-games', '0', '--elo-games', '2',
                 timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = read_metrics(str(run))
    assert all(r['selfplay']['games'] == 16 and r['selfplay']['episodes']['contests'] == 16 for r in m[1:])
    assert any(r['selfplay'].get('league', {}).get('games', 0) > 0 for r in m[2:]), [r['selfplay'].get('league') for r in m]
    rem = tmp_path / 'rem'
    p = run_loop(rem, '--iters', '1', '--game', 'toy_match', '--eval-remote', 'local',
                 env={'ZERO_REMOTE_WORKDIR': str(tmp_path / 'mirror')}, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    ev = [json.loads(line) for line in open(rem / 'evals.jsonl')]
    assert sorted(e['iter'] for e in ev) == [0, 1]
    assert all(e['eval_yard']['games'] == 8 and e['eval_yard']['per_game']['games'] >= 16 for e in ev)
