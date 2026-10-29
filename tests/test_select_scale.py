"""The 'cq' move rule's own completed-Q transform (docs/az.md 5.4, docs/search.md 3.3): run settings
eval_select_c_visit / eval_select_c_scale and selfplay_select_c_visit / selfplay_select_c_scale
(unset = cq_c_visit / cq_c_scale, the training target's), seat keys select_c_visit / select_c_scale,
h2h's per-side --a-cq-c-scale / --b-cq-c-scale. Unset or given equal to the target's: the seats,
runs and pools of before; set otherwise: the moves change, the training targets never do."""
import glob
import hashlib
import json
import math
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from zero import elo, games as G, mcts, parity, protocol
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, check_eval_settings, read_metrics

from test_loop import run_loop
from test_policy_target import parity_eval
from test_select import _Rand, _ckpts, _pairs, _play

INFO = G.info('toy')


# ------------------------------------------------------------------ settings, seats, names

def test_settings_seats_and_names():
    # unset (None) follows the target's transform; set, the rule's own
    assert mcts.select_settings({}) == ('visits', 0, 50.0, 1.0)
    c = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'cq_c_scale': 0.3}
    assert mcts.select_settings(c) == ('cq', 4, 50.0, 0.3)
    assert mcts.select_settings(dict(c, eval_select_c_scale=1)) == ('cq', 4, 50.0, 1.0)
    assert mcts.select_settings(dict(c, eval_select_c_visit=25, eval_select_c_scale=None)) == ('cq', 4, 25.0, 0.3)
    assert mcts.select_settings(dict(c, selfplay_select='cq', selfplay_select_c_scale=2.0), 'selfplay_select') == \
        ('cq', 0, 50.0, 2.0)
    # the pool is named after the rule's transform: a training-only change of cq_c_scale keeps the
    # pool once the selection scale is pinned; unset, it follows (today's behaviour)
    assert elo.eval_tag(c) == 'cq-n4-v50-s0.3'
    assert elo.eval_tag(dict(c, eval_select_c_scale=1.0)) == 'cq-n4'
    assert elo.eval_tag(dict(c, cq_c_scale=1.0, eval_select_c_scale=0.3)) == 'cq-n4-v50-s0.3'
    assert elo.eval_tag(dict(c, cq_c_scale=0.7, eval_select_c_scale=1.0)) == elo.eval_tag(dict(c, cq_c_scale=1.0))
    # validated like the guard: numbers >= 0, finite, the rule 'cq' only
    assert DEFAULTS['eval_select_c_scale'] is None and DEFAULTS['selfplay_select_c_visit'] is None
    for bad in ({'eval_select_c_scale': 0.3}, {'selfplay_select_c_visit': 25.0},
                {'eval_select': 'cq', 'eval_select_c_scale': -1.0}, {'eval_select': 'cq', 'eval_select_c_scale': math.nan},
                {'eval_select': 'cq', 'eval_select_c_visit': True}, {'eval_select': 'cq', 'eval_select_c_scale': '0.3'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, eval_select='cq', eval_select_c_scale=0.3, eval_select_c_visit=50,
                             selfplay_select='cq', selfplay_select_c_scale=0.0))
    with pytest.raises(ValueError):
        G.nn_seat('m', select_c_scale=0.3)                    # 'visits' has no transform
    with pytest.raises(ValueError):
        NNSeat('m', select='cq', select_c_scale=-0.1)
    # seat dicts depend on the rule's effective transform only: given equal to the target's, or
    # unset, the dict of before key for key; an evaluation seat (no 'cq' target) carries the rule's
    # transform as cq_c_*; a 'cq' target with another rule transform adds select_c_*
    for kw in ({}, {'policy_target': 'cq'}):
        base = G.nn_seat('m', 8, select='cq', select_min_visits=2, cq_c_scale=0.3, **kw)
        assert G.nn_seat('m', 8, select='cq', select_min_visits=2, cq_c_scale=0.3, select_c_scale=0.3,
                         select_c_visit=50, **kw) == base
        assert list(G.nn_seat('m', 8, select='cq', select_min_visits=2, cq_c_scale=0.3, select_c_scale=0.3, **kw)) == \
            list(base)
    assert G.nn_seat('m', 8, select='cq', cq_c_scale=1.0, select_c_scale=0.3) == G.nn_seat('m', 8, select='cq', cq_c_scale=0.3)
    s = G.nn_seat('m', 8, policy_target='cq', select='cq', cq_c_scale=1.0, select_c_scale=0.3)
    assert (s['cq_c_scale'], s['select_c_visit'], s['select_c_scale']) == (1.0, 50.0, 0.3)
    assert list(s)[-2:] == ['select_c_visit', 'select_c_scale']
    seat = NNSeat('m', select='cq', select_min_visits=4, cq_c_scale=1.0, select_c_scale=0.3)
    assert seat.select_transform == (50.0, 0.3) and seat.select_tag == 'cq-n4-v50-s0.3'
    assert NNSeat('m', select='cq', cq_c_scale=0.3).select_tag == 'cq-v50-s0.3'
    # the Rust driver reads the keys, and rejects them on a 'visits' seat (as the guard)
    with pytest.raises(ValueError, match='select_c_scale'):
        G.Driver('toy', [[dict(G.nn_seat('m', 4), select_c_scale=0.3)] * 2], concurrency=1)
    with pytest.raises(ValueError, match='transform'):
        G.Driver('toy', [[dict(G.nn_seat('m', 4, select='cq'), select_c_scale=-1.0)] * 2], concurrency=1)


# ------------------------------------------------------------------ the drivers: moves vs targets

def _games(rows):
    return sorted(((r['gid'], r['steps'], tuple(r['scores'])) for r in rows), key=lambda x: x[0])


def test_rust_driver_eval_seats_move_by_the_rules_transform():
    """An evaluation match whose seats carry select_c_scale 0.3 next to cq_c_scale 1.0 plays the
    games of cq_c_scale 0.3 (one knob, as before) - and not those of 1.0."""
    def rows(**kw):
        s = dict(G.nn_seat('m', 12, select='cq'), **kw)
        drv = G.Driver('toy', [[s, s]] * 30, concurrency=30, seed=3, threads=1, groups=1)
        return _games(_play(drv, _Rand()))
    own = rows(policy_target='cq', cq_c_scale=1.0, select_c_scale=0.3)       # a 'cq' target at 1.0 beside
    assert own == rows(cq_c_scale=0.3)
    assert own == rows(cq_c_scale=0.7, select_c_scale=0.3, select_c_visit=50.0)
    assert own != rows(cq_c_scale=1.0), 'the rule transform changed no game'


def _target_scale_spread(sa, sb, se, ratio):
    """ln pi_s = ln P0 + s (c_visit + max N) completedQ - ln Z_s: per record, (ln pi_A - ln pi_E)
    - ratio (ln pi_B - ln pi_E) is one constant over the options when A's target scale = ratio x
    B's and E's is 0 (the floored prior). -> (records checked, the largest spread)."""
    checked, worst = 0, 0.0
    off = sa['opt_off']
    pa, pb, pe = (np.asarray(s['pi'], np.float64) for s in (sa, sb, se))
    for lo, hi in zip(off[:-1].tolist(), off[1:].tolist()):
        a, b, e = pa[lo:hi], pb[lo:hi], pe[lo:hi]
        ok = (a > 1e-30) & (b > 1e-30) & (e > 1e-30)
        if ok.sum() < 2:
            continue
        d = (np.log(a[ok]) - np.log(e[ok])) - ratio * (np.log(b[ok]) - np.log(e[ok]))
        checked += 1
        worst = max(worst, float(d.max() - d.min()))
    return checked, worst


def _selfplay_check(play):
    """`play(target_scale, rule_scale)` -> (game rows, samples). A: target 1.0 + rule 0.3, B: target
    0.3 + rule 0.3, E: target 0 + rule 0.3, C: target 1.0 + rule 1.0."""
    (ra, sa), (rb, sb), (re, se), (rc, _) = play(1.0, 0.3), play(0.3, 0.3), play(0.0, 0.3), play(1.0, 1.0)
    assert ra == rb == re, 'the same rule must play the same games whatever the target'
    assert ra != rc, 'the rule transform changed no game'
    for k in protocol.SAMPLE_KEYS:
        if k != 'pi':
            assert np.array_equal(sa[k], sb[k]) and np.array_equal(sa[k], se[k]), k
    assert not np.array_equal(sa['pi'], sb['pi'])
    n, worst = _target_scale_spread(sa, sb, se, 1.0 / 0.3)
    assert n > len(sa['opt_off']) // 2 and worst < 1e-3, (n, worst)


def test_rust_driver_selfplay_targets_ignore_the_rules_transform():
    """Self-play through the bindings: the moves after temp_moves follow select_c_scale, the 'cq'
    targets cq_c_scale alone (A's targets are the c_scale 1.0 targets of B's roots)."""
    ev = parity_eval(INFO)

    def play(target, rule):
        kw = dict(select_c_scale=rule) if rule != target else {}
        seat = G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=4, policy_target='cq',
                         cq_c_scale=target, select='cq', **kw)
        drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=8, threads=2, seed=5, record=True, max_steps=5000)
        from zero.infer import drive
        parts = []
        drive(drv, {'m': ev}, on_batch=lambda lb: parts.append(drv.take_samples()))
        parts.append(drv.take_samples())
        return _games(drv.take_results()), protocol.concat_samples(parts)
    _selfplay_check(play)


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_python_driver_selfplay_targets_ignore_the_rules_transform(impl):
    """The same in the Python driver ('py': the reference formula; 'rs': the Rust tree's
    root_select and root_cq_target)."""
    ev = parity_eval(INFO)

    def play(target, rule):
        seat = NNSeat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=4, impl=impl, policy_target='cq',
                      cq_c_scale=target, select='cq', select_c_scale=rule)
        drv = PyDriver('toy', 8, lambda g: [seat, seat], concurrency=8, seed=3, record=True)
        from zero.infer import drive
        drive(drv, {'m': ev})
        return _games(drv.take_results()), drv.take_samples()
    _selfplay_check(play)


def test_rust_and_python_seats_agree_at_a_selection_scale():
    """Rust vs Python at the live rule's shape (cq, guard 4, select_c_scale 0.3 beside cq_c_scale
    1.0): the Python formula on Rust roots = the Rust driver's choice (parity.select_formula's
    rules include it), and in games a Rust-tree seat and a Python-tree seat score ~50%."""
    rule = ('cq', 4, 50.0, 0.3)
    r = parity.select_formula('toy', n=8, rules=(rule, ('cq', 4, 50.0, 1.0)))
    assert r['checked'] == 8 * 4 * 2 and r['agree'] + r['near_ties'] == r['checked'] and not r['mismatches']
    rs = NNSeat('m', 16, impl='rs', select='cq', select_min_visits=4, cq_c_scale=1.0, select_c_scale=0.3)
    py = NNSeat('m', 16, impl='py', select='cq', select_min_visits=4, cq_c_scale=1.0, select_c_scale=0.3)
    n = 300
    drv = PyDriver('toy', n, lambda g: [rs, py] if g % 2 == 0 else [py, rs], concurrency=100, seed=11)
    rows = _play(drv, _Rand(3))
    sc = [r['result'][r['gid'] % 2] for r in rows]
    w = sum(sc) / len(sc)
    se = math.sqrt(max(w * (1 - w), 1e-9) / len(sc))
    assert abs(w - 0.5) < 3.5 * se + 0.02, (w, se)
    # the 'rs' seat's choice is the Rust driver's: same games as a 'cq_c_scale 0.3' seat
    def games(**kw):
        s = NNSeat('m', 12, impl='rs', select='cq', select_min_visits=4, **kw)
        d = PyDriver('toy', 20, lambda g: [s, s], concurrency=20, seed=5)
        return _games(_play(d, _Rand()))
    assert games(cq_c_scale=1.0, select_c_scale=0.3) == games(cq_c_scale=0.3)


# ------------------------------------------------------------------ toy runs

def _digest(p):
    if p.endswith('.pt'):
        # torch.save's zip: the members' bytes (the archive folder name holds the writer's temp name)
        z = zipfile.ZipFile(p)
        return hashlib.sha256(b''.join(n.split('/', 1)[1].encode() + z.read(n) for n in sorted(z.namelist()))).hexdigest()
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _training_files(run):
    """{relative path: digest} of a run's shards, checkpoints and optimizer state."""
    out = {}
    for p in glob.glob(os.path.join(str(run), '**', '*'), recursive=True):
        if p.endswith(('.npz', '.pt')) and os.path.basename(p) != 'best.pt':
            out[os.path.relpath(p, str(run)).replace(os.sep, '/')] = _digest(p)
    return out


TIMING = ('seconds', 'time', 'phases', 'resumed_seconds')


def _no_timing(m):
    if isinstance(m, dict):
        return {k: _no_timing(v) for k, v in m.items() if k not in TIMING and not k.endswith('_s')
                and not k.endswith('_per_s') and k != 'games_per_h'}
    if isinstance(m, list):
        return [_no_timing(v) for v in m]
    return m


def _runs(tmp_path, specs):
    """Toy-smoke runs side by side: {name: extra args} -> {name: run dir} (each asserted to succeed)."""
    def one(item):
        name, extra = item
        run = tmp_path / name
        p = run_loop(run, '--iters', '2', '--policy-target', 'cq', *extra)
        assert p.returncode == 0, name + '\n' + p.stdout + p.stderr
        return name, run
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(ex.map(one, specs.items()))


def test_runs_unset_equal_and_selection_only(tmp_path):
    """Toy runs (2 iterations, 'cq' targets):
    * unset vs given equal to cq_c_scale (0.5, both rules): every shard, checkpoint and Elo file
      byte for byte, the metrics equal (timing aside), the same pool cq-n2-v50-s0.5;
    * the evaluation's selection scale alone (eval_select_c_scale 0.3 vs unset at cq_c_scale 1.0):
      every shard and checkpoint byte for byte (training untouched), the evaluations in another pool;
    * the training scale alone with the selection pinned (cq_c_scale 0.3, eval_select_c_scale 1.0):
      the pool of cq_c_scale 1.0 ('cq'), other shards (the targets changed);
    * self-play's selection scale (selfplay_select_c_scale 0.3 at cq_c_scale 1.0 vs cq_c_scale 0.3):
      the first self-play's games are the same (same moves), its targets are not."""
    rule = ['--eval-select', 'cq', '--eval-select-min-visits', '2', '--selfplay-select', 'cq']
    r = _runs(tmp_path, {
        'unset': ['--cq-c-scale', '0.5', *rule],
        'equal': ['--cq-c-scale', '0.5', *rule, '--eval-select-c-scale', '0.5', '--eval-select-c-visit', '50',
                  '--selfplay-select-c-scale', '0.5', '--selfplay-select-c-visit', '50'],
        'base': ['--eval-select', 'cq'],
        'sel03': ['--eval-select', 'cq', '--eval-select-c-scale', '0.3'],
        'train03': ['--eval-select', 'cq', '--cq-c-scale', '0.3', '--eval-select-c-scale', '1.0'],
        'sp_own': ['--selfplay-select', 'cq', '--selfplay-select-c-scale', '0.3'],
        'sp_one': ['--selfplay-select', 'cq', '--cq-c-scale', '0.3'],
    })
    # unset == explicitly equal
    a, b = _training_files(r['unset']), _training_files(r['equal'])
    assert len(a) >= 6 and a == b, sorted(a)
    for f in ('results.json', 'ratings.json'):
        ja = json.load(open(r['unset'] / 'elo' / 'cq-n2-v50-s0.5' / f))
        jb = json.load(open(r['equal'] / 'elo' / 'cq-n2-v50-s0.5' / f))
        ja.pop('updated', None), jb.pop('updated', None)
        assert ja == jb, f
    assert open(r['unset'] / 'models' / 'index.json', 'rb').read() == open(r['equal'] / 'models' / 'index.json', 'rb').read()
    assert [_no_timing(m) for m in read_metrics(str(r['unset']))] == [_no_timing(m) for m in read_metrics(str(r['equal']))]
    assert all(m['eval_yard']['select'] == 'cq-n2-v50-s0.5' for m in read_metrics(str(r['equal'])))
    ca, cb = (json.load(open(r[k] / 'config.json')) for k in ('unset', 'equal'))
    assert {k for k in ca if ca[k] != cb[k]} == {'eval_select_c_scale', 'eval_select_c_visit', 'selfplay_select_c_scale',
                                                  'selfplay_select_c_visit'}
    assert ca['eval_select_c_scale'] is None and cb['eval_select_c_scale'] == 0.5
    # the evaluation's selection scale changes the evaluations only
    assert _training_files(r['base']) == _training_files(r['sel03'])
    assert os.path.isdir(r['base'] / 'elo' / 'cq') and os.path.isdir(r['sel03'] / 'elo' / 'cq-v50-s0.3')
    assert not os.path.exists(r['sel03'] / 'elo' / 'cq')
    assert {m['eval_yard']['select'] for m in read_metrics(str(r['sel03']))} == {'cq-v50-s0.3'}
    # the training scale alone, selection pinned: the same pool name, other training data
    assert elo.active_pool(str(r['train03']))[0] == elo.active_pool(str(r['base']))[0] == 'cq'
    assert os.path.isdir(r['train03'] / 'elo' / 'cq')
    tb, tt = _training_files(r['base']), _training_files(r['train03'])
    assert tb['models/iter_0000.pt'] == tt['models/iter_0000.pt'] and tb != tt
    # self-play's selection scale: the first self-play (the same initial network) plays the same
    # games at rule scale 0.3 whatever the target's scale; the targets differ
    za, zb = (np.load(glob.glob(str(r[k] / 'data' / 'iter_0001' / '*.npz'))[0]) for k in ('sp_own', 'sp_one'))
    for k in za.files:
        if k != 'pi':
            assert np.array_equal(za[k], zb[k]), k
    assert not np.array_equal(za['pi'], zb['pi'])
    assert {m['eval_yard'].get('select', 'visits') for m in read_metrics(str(r['sp_own']))} == {'visits'}
    # zerodash names the pools the same way
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard'))
    from zerodash import runfiles
    for k in ('unset', 'equal', 'base', 'sel03', 'train03'):
        assert runfiles.elo_pool(str(r[k])) == elo.active_pool(str(r[k]))[1], k


def test_run_cli_unsets_with_none(tmp_path):
    """'none' on the command line unsets a saved selection scale (back to following cq_c_scale)."""
    from zero.run import parse, resolve_config
    saved = dict(DEFAULTS, eval_select='cq', eval_select_c_scale=0.3)
    cfg = resolve_config(parse(['--run', str(tmp_path / 'x'), '--eval-select-c-scale', 'none']), saved)
    assert cfg['eval_select_c_scale'] is None
    cfg = resolve_config(parse(['--run', str(tmp_path / 'x'), '--eval-select-c-scale', '0.25']), saved)
    assert cfg['eval_select_c_scale'] == 0.25
    assert resolve_config(parse(['--run', str(tmp_path / 'x')]), saved)['eval_select_c_scale'] == 0.3
    with pytest.raises(SystemExit):
        parse(['--run', 'x', '--eval-select-c-scale', 'half'])


# ------------------------------------------------------------------ h2h and replays

def test_h2h_per_side_scales(tmp_path, capsys):
    """One checkpoint, two selection scales: --a-cq-c-scale / --b-cq-c-scale (with --select, or over
    the run's own rule); a run's eval_select_c_scale is its h2h default."""
    from zero import h2h
    a, b = _ckpts(tmp_path, {'eval_select': 'cq', 'eval_select_min_visits': 4, 'cq_c_scale': 1.0,
                             'eval_select_c_scale': 0.3})
    assert h2h.run_rule(a) == {'select': 'cq', 'select_min_visits': 4, 'cq_c_visit': 50.0, 'cq_c_scale': 0.3}
    r = h2h.match(a, b, games=8, sims=8, chance_k=0, threads=1)
    assert (r['select_a'], r['select_b']) == ('cq-n4-v50-s0.3', 'cq-n4-v50-s0.3')

    def cli(*args):
        h2h.main([a, a, '--games', '8', '--sims', '8', '--chance-k', '0', '--threads', '1', *args])
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    out = cli('--select', 'cq', '--select-min-visits', '4', '--a-cq-c-scale', '0.3', '--b-cq-c-scale', '1.0')
    assert (out['select_a'], out['select_b']) == ('cq-n4-v50-s0.3', 'cq-n4')
    out = cli('--select', 'cq', '--cq-c-scale', '0.5', '--b-cq-c-scale', '2')      # the side's flag wins
    assert (out['select_a'], out['select_b']) == ('cq-v50-s0.5', 'cq-v50-s2')
    out = cli('--select', 'cq')                                                     # as before: 50 / 1
    assert (out['select_a'], out['select_b']) == ('cq', 'cq')
    out = cli('--b-cq-c-scale', '1.0')                                              # over the run's rule
    assert (out['select_a'], out['select_b']) == ('cq-n4-v50-s0.3', 'cq-n4')
    out = cli('--a-select', 'visits', '--cq-c-scale', '0.7')                        # both-sides: the cq side
    assert (out['select_a'], out['select_b']) == ('visits', 'cq-n4-v50-s0.7')
    with pytest.raises(SystemExit):
        cli('--a-select', 'visits', '--a-cq-c-scale', '0.3')
    with pytest.raises(SystemExit):
        cli('--select', 'visits', '--cq-c-scale', '0.3')
    # the seat dicts: the same rule given as the run's or on the command line plays the same games
    s1 = h2h.match(a, a, games=8, sims=8, chance_k=0, threads=1, seat_a=None, seat_b=None, rows=True)
    s2 = h2h.match(a, a, games=8, sims=8, chance_k=0, threads=1, seat_a=h2h.side_keys(a, 'cq', 4, None, 0.3),
                   seat_b=h2h.side_keys(a, None, 0, None, 0.3), rows=True)
    assert [(x['steps'], x['scores']) for x in s1['rows']] == [(x['steps'], x['scores']) for x in s2['rows']]


def test_replays_follow_the_runs_rule_transform(tmp_path):
    from zero import replay
    a, _ = _ckpts(tmp_path, {'eval_select': 'cq', 'cq_c_scale': 0.3, 'eval_select_c_scale': 1.0, 'spec': ''})
    reps = replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None)
    assert reps[0]['seats'][0]['select'] == 'cq'
    reps = replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None,
                         select=(None, 0, None, 0.5))
    assert reps[0]['seats'][0]['select'] == 'cq-v50-s0.5'
    with pytest.raises(ValueError):
        replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None,
                      select=('visits', 0, None, 0.5))
