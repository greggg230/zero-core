"""Strength measurement (docs/az.md 20; MTG Zero's Delver line, 2026-10-30): paired games (the driver's
`paired`), zero.h2h's pairs, per-side simulations and sequential tests (SPRT, target se), the Elo pool's
fixed anchors (calibration, the fit with them fixed, the sha256 guard, a remote mirror), elo_paired, the
held-out losses (zero.heldout) - and each one off = the run of before."""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from zero import elo, games as G, h2h
from zero.run import read_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']
INFO = G.info('toy')


def run_loop(run, *extra, env=None, timeout=300):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, ZERO_INFLIGHT='1', **(env or {}))
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, env=e)


def nets(tmp_path, n=2):
    import torch
    from zero.net import Net, save
    out = []
    for k in range(n):
        torch.manual_seed(k)
        out.append(str(tmp_path / f'm{k}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), out[-1], {'iter': k})
    return out


# ------------------------------------------------------------------ the driver's paired games

def test_paired_driver_pairs_share_their_streams():
    """Games 2k and 2k + 1 share every stream: two seats that search alike, swapped, play one game twice."""
    from zero.infer import UniformEvaluator
    seat = lambda k: G.nn_seat(k, 8)                         # noqa: E731
    seats = [[seat('a'), seat('b')] if g % 2 == 0 else [seat('b'), seat('a')] for g in range(16)]

    def play(paired):
        kw = {'paired': True} if paired else {}
        d = G.Driver('toy', seats, concurrency=16, seed=11, threads=2, move_log=500, **kw)
        from zero.infer import drive
        drive(d, {'a': UniformEvaluator(), 'b': UniformEvaluator()})
        return sorted(d.take_results(), key=lambda r: r['gid'])
    p, u = play(True), play(False)
    key = lambda r: (r['steps'], r['scores'], r['moves'])    # noqa: E731
    assert all(key(p[2 * k]) == key(p[2 * k + 1]) for k in range(8))
    assert any(key(u[2 * k]) != key(u[2 * k + 1]) for k in range(8))


# ------------------------------------------------------------------ zero.h2h

def test_h2h_paired_pairs_sims_and_the_defaults(tmp_path):
    pa, pb = nets(tmp_path)
    plain = h2h.match(pa, pb, games=20, sims=8, chance_k=0, threads=2)
    assert 'pairs' not in plain and 'paired' not in plain and 'sims_a' not in plain
    assert plain['se'] == round(math.sqrt(plain['a_score'] * (1 - plain['a_score']) / 20), 4)
    r = h2h.match(pa, pb, games=20, sims=8, chance_k=0, threads=2, paired=True, rows=True)
    ps = r['pairs']
    assert r['paired'] and ps['n'] == 10 and sum(ps['pentanomial']) == 10
    assert ps['both_won'] + ps['split'] + ps['both_lost'] + ps['with_draw'] == 10
    sc = [(r['rows'][2 * k]['result'][0] + r['rows'][2 * k + 1]['result'][1]) / 2 for k in range(10)]
    assert r['a_score'] == ps['score'] == round(sum(sc) / 10, 4)
    assert r['se'] == ps['se'] and r['se_games'] == round(math.sqrt(r['a_score'] * (1 - r['a_score']) / 20), 4)
    # the same network on both sides, searching alike: every pair is a split, no variance left
    same = h2h.match(pa, pa, games=12, sims=8, chance_k=0, threads=2, paired=True)
    assert same['a_score'] == 0.5 and same['se'] == 0 and same['pairs']['split'] + same['pairs']['with_draw'] == 6
    # per-side simulations: named, and the stronger search shows
    s = h2h.match(pa, pa, games=24, sims=4, chance_k=0, threads=2, paired=True, sims_a=48)
    assert (s['sims_a'], s['sims_b']) == (48, 4) and s['a_score'] > 0.5
    with pytest.raises(ValueError):
        h2h.match(pa, pb, games=7, sims=4, paired=True)


def test_sprt_llr_and_bounds():
    lo, hi = h2h.sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(math.log(0.05 / 0.95)) and hi == pytest.approx(math.log(0.95 / 0.05))
    obs = [1, 0, 1, 1, 0, 1, 0.5, 1]
    n, mean = len(obs), sum(obs) / len(obs)
    var = sum((x - mean) ** 2 for x in obs) / n
    s0, s1 = h2h.expected_score(0), h2h.expected_score(20)
    assert s0 == 0.5 and s1 == pytest.approx(1 / (1 + 10 ** (-20 / 400)))
    assert h2h.sprt_llr(obs, 0, 20) == pytest.approx(n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var))
    assert h2h.sprt_llr([0.5] * 10, 0, 20) < 0 < h2h.sprt_llr([1.0] * 9 + [0.5], 0, 20)


def test_h2h_sequential(tmp_path, capsys):
    pa, pb = nets(tmp_path)
    one = h2h.match(pa, pb, games=16, sims=8, chance_k=0, threads=2, seed=3, paired=True)
    # SPRT: a lopsided match is decided in its first batch, whose games are the plain match's
    r = h2h.sequential(pa, pb, batch_games=16, max_games=64, sprt=(0, 100), sims=8, chance_k=0, threads=2, seed=3,
                       paired=True)
    assert r['batches'][0]['a_score'] == one['a_score'] and r['batches'][0]['games'] == 16
    assert r['sprt']['decision'] in ('H0', 'H1') and r['stop'].startswith('sprt') and r['games'] == 16
    # nothing to decide (the same network): until max_games, every batch a new seed, the gids unique
    same = h2h.sequential(pa, pa, batch_games=8, max_games=24, sprt=(0, 5), sims=8, chance_k=0, threads=2, rows=True)
    assert same['games'] == 24 and len(same['batches']) == 3 and same['stop'] in ('max_games', 'sprt H0')
    assert sorted(x['gid'] for x in same['rows']) == list(range(24))
    t = h2h.sequential(pa, pb, batch_games=8, max_games=40, target_se=0.5, sims=8, chance_k=0, threads=2)
    assert t['stop'] == 'target_se' and t['games'] == 8 and 'sprt' not in t
    # the command line: flags, a progress line per batch on stderr
    h2h.main([pa, pb, '--games', '8', '--sims', '4', '--chance-k', '0', '--threads', '2', '--paired', '--sprt', '0,200',
              '--max-games', '16', '--a-sims', '6'])
    cap = capsys.readouterr()
    out = json.loads(cap.out.strip().splitlines()[-1])
    assert out['paired'] and out['sprt']['elo1'] == 200 and out['sims_a'] == 6 and '# batch ' in cap.err
    for bad in (['--games', '7', '--paired'], ['--sprt', '20,0'], ['--max-games', '10'], ['--a-sims', '0']):
        with pytest.raises(SystemExit):
            h2h.main([pa, pb, '--sims', '4'] + bad)


# ------------------------------------------------------------------ Elo: the fit with fixed anchors

def test_fit_fixed_is_fit_with_one_fixed_and_recovers_ratings():
    rng = np.random.default_rng(0)
    true = {'a1': 1000.0, 'a2': 1150.0, 'x': 1080.0, 'y': 1220.0, 'z': 950.0}
    res = {'pairs': {}, 'anchor': 'a1'}
    p = lambda a, b: 1 / (1 + 10 ** (-(true[a] - true[b]) / 400))       # noqa: E731
    for a, b in [('a1', 'a2'), ('x', 'a1'), ('x', 'a2'), ('y', 'a2'), ('y', 'x'), ('z', 'a1'), ('z', 'x')]:
        n = 4000
        elo.add(res, a, b, float(rng.binomial(n, p(a, b))), n)
    plain = elo.fit(res, anchor='a1')
    one = elo.fit_fixed(res, {'a1': 1000.0})
    for k in plain:
        assert one[k][0] == pytest.approx(plain[k][0], abs=1e-6) and one[k][1] == pytest.approx(plain[k][1], abs=1e-6)
    two = elo.fit_fixed(res, {'a1': 1000.0, 'a2': 1150.0})
    assert two['a2'] == (1150.0, 0.0, 12000)
    for k in ('x', 'y', 'z'):
        assert abs(two[k][0] - true[k]) < 3 * two[k][1] + 1, (k, two[k])
        assert two[k][1] <= one[k][1] + 1e-9        # fixing another anchor never widens an se
    # only some pairs (the calibration's): the others' games are not used
    cal = elo.fit_fixed(res, {'a1': 1000.0}, pairs=['a1|a2'])
    assert set(cal) == {'a1', 'a2'} and abs(cal['a2'][0] - 1150) < 20
    assert elo.fit_fixed(res, {}) == {}


def test_parse_anchors():
    assert elo.parse_anchors('') == [] and elo.parse_anchors(None) == []
    got = elo.parse_anchors('E:/x/r/models/iter_0192.pt, b160=C:/Users/G K/a.pt')
    assert got == [('iter_0192', 'E:/x/r/models/iter_0192.pt'), ('b160', 'C:/Users/G K/a.pt')]
    assert elo.parse_anchors(['a=x.pt', 'y.pt']) == [('a', 'x.pt'), ('y', 'y.pt')]
    with pytest.raises(ValueError):
        elo.parse_anchors('a=x.pt,a=y.pt')
    assert elo.anchor_name('b160') == 'anchor:b160' and elo.is_anchor('anchor:b160')


# ------------------------------------------------------------------ the run: anchors, elo_paired, held-out

@pytest.fixture(scope='module')
def base_run(tmp_path_factory):
    d = tmp_path_factory.mktemp('measure')
    run = d / 'base'
    p = run_loop(run, '--iters', '3')
    assert p.returncode == 0, p.stdout + p.stderr
    shutil.copyfile(run / 'models' / 'iter_0001.pt', d / 'a1.pt')
    shutil.copyfile(run / 'models' / 'iter_0003.pt', d / 'a3.pt')
    from zero import heldout
    heldout.make([str(run / 'data' / 'iter_0003')], str(d / 'ho.npz'), rows=120, seed=1)
    return d


def anchored_flags(d, *more):
    return ['--eval-games', '0', '--eval-random-games', '0', '--elo-anchors', f'a1={d / "a1.pt"},{d / "a3.pt"}',
            '--elo-anchor-games', '6', '--elo-anchor-calib-games', '10', '--heldout-shards', str(d / 'ho.npz'), *more]


def test_run_with_anchors_paired_elo_and_heldout(base_run, tmp_path):
    d = base_run
    run = tmp_path / 'anch'
    p = run_loop(run, '--iters', '3', '--elo-paired', '1', *anchored_flags(d))
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.load(open(run / 'elo' / 'results.json'))
    an = res['anchors']
    assert an['anchor:a1']['elo'] == 1000.0 and 'calibrated' in an['anchor:a3']['how']
    assert an['anchor:a3']['sha256'] and an['anchor:a3']['file'] == 'a3.pt'
    assert res['pairs']['anchor:a1|anchor:a3'][1] == 10                       # the calibration
    for k in (1, 2, 3):                                                       # every evaluated checkpoint
        for a in ('anchor:a1', 'anchor:a3'):
            assert res['pairs'][f'{a}|iter_{k:04d}'][1] == 6
    ar = json.load(open(run / 'elo' / 'anchored.json'))
    assert ar['ratings']['anchor:a1'] == {'elo': 1000.0, 'se': 0.0, 'games': 28}
    assert ar['ratings']['anchor:a3']['elo'] == an['anchor:a3']['elo'] and ar['ratings']['anchor:a3']['se'] == 0
    m = read_metrics(str(run))
    assert m[3]['elo_anchored']['elo'] == ar['ratings']['iter_0003']['elo']
    assert all('heldout' in r and r['heldout']['decision']['rows'] == r['heldout']['rows'] for r in m)
    assert len({r['heldout']['set'] for r in m}) == 1
    # heuristic games dropped (eval_games 0): every checkpoint still evaluated, Elo matches played
    assert all(r['eval_yard']['games'] == 0 for r in m) and m[3].get('elo')
    log = open(run / 'log.txt', encoding='utf-8').read()
    assert 'elo anchor anchor:a3 fixed at' in log and 'iter 3 elo anchored' in log and 'iter 3 heldout (set' in log
    # another file under a fixed anchor's label: refused (exit 78), never a moved scale
    shutil.copyfile(d / 'a1.pt', tmp_path / 'other.pt')
    shutil.copyfile(run / 'models' / 'iter_0002.pt', d / 'a1_moved.pt')
    p = run_loop(run, '--iters', '4', *anchored_flags(d)[:-6],
                 '--elo-anchors', f'a1={d / "a1_moved.pt"},{d / "a3.pt"}', '--elo-paired', '1',
                 '--heldout-shards', str(d / 'ho.npz'))
    assert p.returncode == 78 and 'not the network it was fixed with' in p.stdout + p.stderr + open(
        run / 'log.txt', encoding='utf-8').read()


def test_anchor_and_heldout_refusals_and_settings(base_run, tmp_path):
    d = base_run
    p = run_loop(tmp_path / 'r1', '--iters', '1', '--elo-anchors', str(d / 'missing.pt'))
    assert p.returncode == 78, p.stdout + p.stderr
    p = run_loop(tmp_path / 'r2', '--iters', '1', '--heldout-shards', str(d / 'missing.npz'))
    assert p.returncode == 78, p.stdout + p.stderr
    p = run_loop(tmp_path / 'r3', '--iters', '1', '--elo-paired', '1', '--elo-games', '5')
    assert p.returncode != 0 and 'must be even' in p.stdout + p.stderr
    p = run_loop(tmp_path / 'r4', '--iters', '1', '--elo-anchors', 'a=x.pt,a=y.pt')
    assert p.returncode != 0 and 'distinct' in p.stdout + p.stderr


def test_anchors_on_a_local_remote(base_run, tmp_path):
    """eval_remote: the anchors are shipped to the mirror under their labels, the mirror's config names
    them there, and its anchored ratings come back (evals.jsonl, anchored.json, the log)."""
    d = base_run
    run = tmp_path / 'rem'
    mirror = tmp_path / 'mirror'
    p = run_loop(run, '--iters', '2', '--eval-remote', 'local', *anchored_flags(d), env={'ZERO_REMOTE_WORKDIR': str(mirror)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert sorted(os.listdir(mirror / 'rem' / 'anchors')) == ['a1.pt', 'a3.pt']
    mc = json.load(open(mirror / 'rem' / 'config.json'))
    assert mc['elo_anchors'].split(',')[0] == 'a1=' + str(mirror / 'rem').replace(os.sep, '/') + '/anchors/a1.pt' or \
        mc['elo_anchors'].split(',')[0].startswith('a1=')
    ev = {e['iter']: e for e in (json.loads(x) for x in open(run / 'evals.jsonl'))}
    assert ev[2]['elo_anchored']['se'] > 0
    ar = json.load(open(run / 'elo' / 'anchored.json'))
    assert ar['ratings']['iter_0002']['elo'] == ev[2]['elo_anchored']['elo']
    m = read_metrics(str(run))
    assert all('heldout' in r for r in m)                                     # computed here, not remotely
    from zero import dashboard
    assert dashboard._metrics(str(run))[2]['elo_anchored'] == ev[2]['elo_anchored']
    assert 'iter 2 elo anchored' in open(run / 'log.txt', encoding='utf-8').read()


def test_defaults_add_nothing(base_run):
    """Off (the default): no anchored ratings, no held-out losses, the pool's files as before."""
    run = base_run / 'base'
    assert not os.path.exists(run / 'elo' / 'anchored.json')
    res = json.load(open(run / 'elo' / 'results.json'))
    assert 'anchors' not in res and not any('anchor:' in k for k in res['pairs'])
    assert not any('heldout' in r or 'elo_anchored' in r for r in read_metrics(str(run)))


# ------------------------------------------------------------------ zero.heldout

def test_heldout_make_losses_and_guards(base_run, tmp_path):
    from zero import heldout
    from zero.data import Window, read_shard
    from zero.distill import outputs
    from zero.net import load
    d = base_run
    s = read_shard(str(d / 'ho.npz'))
    assert s['meta']['heldout']['from'] == 3 and s['meta']['heldout']['rows'] == len(s['player']) <= 120
    H = heldout.HeldOut([str(d / 'ho.npz')], game='toy')
    m, _ = load(str(d / 'base' / 'models' / 'iter_0003.pt'))
    r = H.evaluate(m)
    dec = r['decision']
    assert r['rows'] == dec['rows'] and dec['decisions'] <= dec['rows'] and r['set'] == heldout.set_name([str(d / 'ho.npz')])
    # the numbers: by hand from distill.outputs
    W = Window([str(d / 'ho.npz')], H.features, H.actions)
    O = outputs(m, W)
    z = W.z / W.z.sum(1, keepdims=True)
    vce = -(z[:, :2] * np.log(np.maximum(O['v'][:, :2], 1e-12))).sum(1)
    assert dec['value_ce'] == pytest.approx(float(vce.mean()), abs=1e-5)
    # the same rows each time (a frozen set), another set for another seed
    assert heldout.HeldOut([str(d / 'ho.npz')]).evaluate(m) == dict(r, seconds=heldout.HeldOut(
        [str(d / 'ho.npz')]).evaluate(m)['seconds'])
    with pytest.raises(ValueError):
        H.check(features=H.features + 1)
    with pytest.raises(ValueError):
        heldout.HeldOut([str(d / 'ho.npz')], game='caverna')
    # value-only rows are reported apart (a shard with a row without options)
    v = dict(s)
    meta = v.pop('meta')
    lens = np.diff(v['opt_off'])
    keep = np.ones(len(v['opt_action']), bool)
    keep[v['opt_off'][0]:v['opt_off'][1]] = False             # row 0 loses its options
    v['opt_action'], v['pi'] = v['opt_action'][keep], v['pi'][keep]
    lens[0] = 0
    v['opt_off'] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    from zero.data import write_shard
    write_shard(str(tmp_path / 'view.npz'), v, meta)
    rv = heldout.HeldOut([str(tmp_path / 'view.npz')]).evaluate(m)
    assert rv['view']['rows'] == 1 and rv['decision']['rows'] == rv['rows'] - 1 and 'policy_ce' not in rv['view']
    # the command line
    out = subprocess.run([sys.executable, '-m', 'zero.heldout', 'eval', '--shards', str(d / 'ho.npz'), '--ckpt',
                          str(d / 'base' / 'models' / 'iter_0003.pt')], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=PYPATH))
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got['iter'] == 3 and got['decision']['value_ce'] == dec['value_ce']
