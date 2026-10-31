"""Self-play exploration (zero/explore.py, docs/search.md 3.9, docs/az.md 21) on the toy's two test tags
(`fish`: placing on Fishing, this decision only; `timber`: on Timber, the rest of the round): the
bindings (the game's tags, a name it does not declare refused, the rows' exploration, the samples'
`masked`), the run settings and their checks, the tallies, the window's `masked` and the margin loss
that skips it, the run loop end to end (metrics in the log line, masked rows in the shards, the run of
before with the settings off) and remote self-play (the worker explores and reports; a checkout
without exploration is refused). The trigger, the gate, the windows, one episode a game, the masking
and the own stream are the Rust driver's tests (driver::tests::*explor*)."""
import json
import os

import numpy as np
import pytest
import torch

from zero import explore as X, games as G
from zero.data import ShardWriter, Window, read_shard, shard_paths
from zero.run import read_metrics

from test_loop import run_loop
from test_remoteplay import env_for
from test_select_scale import _training_files

TOY_TAGS = [('fish', 0, 0.5), ('timber', 1, 0.25)]


def seat():
    return G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10, view_samples=1.0)


def uniform(lb):
    """Uniform priors, a value by a hash of the features (the Rust tests' evaluator)."""
    a = lb.a
    off = a['opt_off']
    pri = np.concatenate([np.full(int(off[r + 1] - off[r]), 1.0 / max(1, int(off[r + 1] - off[r])), np.float32)
                          for r in range(len(off) - 1)]) if off[-1] else np.zeros(0, np.float32)
    b = len(a['player'])
    w = 1 / (1 + np.exp(-(a['x'] * ((np.arange(a['x'].shape[1]) * 7 % 13) - 6.0)).sum(1) * 0.1))
    v = np.zeros((b, 8), np.float32)
    v[:, 0], v[:, 1] = w, 1 - w
    return pri, v, np.zeros((b, 8), np.float32)


def play(explore=None, games=24, seed=5):
    d = G.Driver('toy', [[seat(), seat()]] * games, concurrency=games, seed=seed, record=True, threads=2,
                 **({'explore': explore} if explore is not None else {}))
    rows, parts = [], []
    while True:
        lb = d.next_batch()
        if lb is None:
            break
        d.submit(lb.id, *uniform(lb))
        rows += d.take_results()
        s = d.take_samples()
        if s is not None:
            parts.append(s)
    rows += d.take_results()
    s = d.take_samples()
    if s is not None:
        parts.append(s)
    return rows, parts


def test_the_toys_tags_and_unknown_names_are_refused():
    assert [tuple(t) for t in G.info('toy')['explore_tags']] == TOY_TAGS
    assert G.info('toy_team')['explore_tags'] == []
    g = G.engine().Game('toy', None, 2, 3)
    t = g.explore_tags()
    assert len(t) == len(g.legal()) and set(t) <= {0, 1, 2} and 1 in t and 2 in t
    with pytest.raises(ValueError, match='declares no habit tag "trees"'):
        play({'tags': [{'tag': 'trees'}]})
    with pytest.raises(ValueError, match='unknown setting "gate"'):
        play({'gate': 0.1, 'tags': [{'tag': 'fish'}]})
    with pytest.raises(ValueError, match='no tags'):
        play({'tags': []})
    with pytest.raises(ValueError, match='records nothing'):
        G.Driver('toy', [[seat(), seat()]] * 2, record=False, explore={'tags': [{'tag': 'fish'}]})


def test_rows_and_samples():
    rows, parts = play()
    assert all('explore' not in r for r in rows) and all('masked' not in s for s in parts)
    x = {'games': 1.0, 'q_margin': 1e9, 'mask_z': True, 'tags': [{'tag': 'fish', 'p': 1.0}, {'tag': 'timber'}]}
    rows, parts = play(x)
    eps = [r['explore']['episode'] for r in rows if r['explore'].get('episode')]
    assert len(eps) > 10 and all(r['explore']['eligible'] for r in rows)
    assert {e['end'] for e in eps} <= {'decision', 'window', 'game', 'gate'}
    assert all(e['tag'] in (1, 2) and e['overridden'] >= 1 for e in eps)
    masked = np.concatenate([s['masked'] for s in parts])
    z, q = (np.concatenate([s[k] for s in parts]) for k in ('z', 'q'))
    sc = np.concatenate([s['sc'] for s in parts])
    assert masked.sum() == sum(r['explore']['masked'] for r in rows) > 0
    assert np.array_equal(z[masked == 1], q[masked == 1]) and not sc[masked == 1].any()
    assert sum(r['samples'] for r in rows) == len(masked)
    # mask_z off: the same games, no key, nothing masked or dropped
    rows2, parts2 = play(dict(x, mask_z=False))
    assert [r['steps'] for r in rows2] == [r['steps'] for r in rows]
    assert all('masked' not in s for s in parts2)
    assert all(r['explore']['masked'] == r['explore']['dropped'] == 0 for r in rows2)
    # the tallies
    d = X.driver_dict({'selfplay_explore': 'fish,timber', 'selfplay_explore_games': 1.0}, G.info('toy'))
    acc = X.tally(rows, X.new_tally(G.info('toy'), d))
    m = X.summary(X.merge(acc, X.tally(rows2, X.new_tally(G.info('toy'), d))))
    assert m['games'] == 2 * len(rows) and m['episodes'] == 2 * len(eps)
    assert m['masked'] == masked.sum() and 0 < m['masked_share'] < 1
    t = m['tags']['fish']
    assert t['episodes'] == 2 * sum(e['tag'] == 1 for e in eps) and t['opportunities'] >= t['episodes']
    assert sum(t['turns'].values()) == t['episodes'] and t['q_gap'] is not None


def test_settings():
    info = G.info('toy')
    assert X.driver_dict({}, info) is None and X.driver_dict({'selfplay_explore': ''}, info) is None
    d = X.driver_dict({'selfplay_explore': 'all'}, info)
    assert d == {'games': 0.25, 'q_margin': 0.1, 'mask_z': True,
                 'tags': [{'tag': 'fish', 'p': 0.5, 'window': 0}, {'tag': 'timber', 'p': 0.25, 'window': 1}]}
    d = X.driver_dict({'selfplay_explore': 'timber', 'selfplay_explore_p': '0.3', 'selfplay_explore_window': 'timber=2',
                       'selfplay_explore_games': 1.0, 'selfplay_explore_q_margin': 0.05, 'selfplay_explore_mask_z': 0},
                      info)
    assert d == {'games': 1.0, 'q_margin': 0.05, 'mask_z': False, 'tags': [{'tag': 'timber', 'p': 0.3, 'window': 2}]}
    for cfg, msg in [({'selfplay_explore': 'trees'}, "declares no habit tag \\['trees'\\]"),
                     ({'selfplay_explore': 'fish,fish'}, 'a tag twice'),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_games': 0}, 'selfplay_explore_games 0'),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_q_margin': -1}, 'q_margin -1'),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_p': 'timber=0.5'}, "'timber' is not an explored"),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_p': '2'}, 'fish 2.0'),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_window': '-1'}, 'fish -1'),
                     ({'selfplay_explore': 'fish', 'selfplay_explore_mask_z': 2}, 'mask_z 2')]:
        with pytest.raises(SystemExit, match=msg):
            X.driver_dict(cfg, info)
    with pytest.raises(SystemExit, match='declares no habit tags'):
        X.driver_dict({'selfplay_explore': 'fish'}, G.info('toy_team'))
    with pytest.raises(SystemExit, match='single-game runs only'):
        X.check({'selfplay_explore': 'fish', 'resolver': 'toy'}, info)


def test_window_masked_and_the_margin_loss(tmp_path):
    x = {'games': 1.0, 'q_margin': 1e9, 'mask_z': True, 'tags': [{'tag': 'fish', 'p': 1.0}]}
    _, parts = play(x)
    _, plain = play()
    for name, ps in (('a', parts), ('b', plain)):
        w = ShardWriter(str(tmp_path / name), 'c00', rows=1 << 20, meta={'game': 'toy'})
        for s in ps:
            w.add(s)
        w.close()
    pa, pb = shard_paths(str(tmp_path / 'a')), shard_paths(str(tmp_path / 'b'))
    assert 'masked' in read_shard(pa[0]) and 'masked' not in read_shard(pb[0])
    F = G.info('toy')['num_features']
    A = G.info('toy')['num_actions']
    Wb = Window(pb, F, A)
    assert Wb.masked is None and 'masked' not in Wb.batch(np.arange(8))
    W = Window(pa + pb, F, A)
    n = len(read_shard(pa[0])['player'])
    assert W.masked is not None and W.masked[:n].sum() > 0 and not W.masked[n:].any()
    b = W.batch(np.arange(W.N))
    assert torch.equal(b['masked'], torch.from_numpy(W.masked.astype(np.float32)))
    m = b['masked'].bool()
    assert torch.equal(b['z'][m], b['q'][m]) and not b['sc'][m].any()


def test_run_loop_explores_and_off_is_the_run_of_before(tmp_path):
    """A toy-smoke run exploring both tags: the selfplay metrics' `explore` (and in the log line), the
    shards' masked rows, a trained network; the settings at their defaults, given explicitly: the
    run of before (the same checkpoints)."""
    run = tmp_path / 'x'
    p = run_loop(run, '--iters', '2', '--selfplay-explore', 'fish,timber', '--selfplay-explore-games', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    ms = [m for m in read_metrics(str(run)) if 'selfplay' in m]
    assert len(ms) == 2
    for m in ms:
        e = m['selfplay']['explore']
        assert e['games'] == m['selfplay']['games'] and e['episodes'] > 0 and e['masked'] > 0
        assert set(e['tags']) == {'fish', 'timber'}
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert '"explore": {"games"' in log
    cfg = json.load(open(run / 'config.json'))
    assert cfg['selfplay_explore'] == 'fish,timber' and cfg['selfplay_explore_games'] == 1
    shards = shard_paths(str(run / 'data' / 'iter_0001'))
    assert sum(int(read_shard(s)['masked'].sum()) for s in shards) == ms[0]['selfplay']['explore']['masked']
    # off, explicitly at the defaults: the same files as a run that never heard of it
    a, b = tmp_path / 'a', tmp_path / 'b'
    pa = run_loop(a, '--iters', '1')
    pb = run_loop(b, '--iters', '1', '--selfplay-explore', '', '--selfplay-explore-games', '0.25',
                  '--selfplay-explore-q-margin', '0.1', '--selfplay-explore-mask-z', '1')
    assert pa.returncode == 0 and pb.returncode == 0, pa.stdout + pa.stderr + pb.stdout + pb.stderr
    fa, fb = _training_files(a), _training_files(b)
    assert len(fa) >= 3 and fa == fb
    assert all('explore' not in m['selfplay'] for m in read_metrics(str(b)) if 'selfplay' in m)
    # a name the game does not declare: refused at the start
    p = run_loop(tmp_path / 'bad', '--iters', '1', '--selfplay-explore', 'trees')
    assert p.returncode != 0 and "declares no habit tag ['trees']" in p.stdout + p.stderr


def test_remote_selfplay_explores_and_refuses_a_checkout_without_it(tmp_path):
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '2', '--conc', '64', '--games', '600', '--selfplay-remote', 'local',
                 '--selfplay-explore', 'fish,timber', env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))[-1]['selfplay']
    assert m['remote']['games'] > 0 and 'explore' not in m['remote']
    assert m['explore']['games'] == m['games'] + m['remote']['games']
    rs = [s for s in shard_paths(str(run / 'data' / 'iter_0002')) if os.path.basename(s).startswith('m')]
    assert rs and all('masked' in read_shard(s) for s in rs)
    run2 = tmp_path / 'sp2'
    p = run_loop(run2, '--iters', '2', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-explore', 'fish', env=env_for(tmp_path / 'm2', {'no_explore': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    for x in read_metrics(str(run2))[1:]:
        assert x['selfplay']['games'] == 200 and x['selfplay']['remote']['games'] == 0
    assert 'REFUSED: its checkout explores as None' in (run2 / 'log.txt').read_text(encoding='utf-8')
