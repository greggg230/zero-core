"""Heavy decisions in self-play (zero/heavy.py, docs/search.md 3.11, docs/az.md 23) on the toy: the
settings refused where they would do nothing or not be heavy, the self-play seat's keys and the
evaluation seats unchanged, the fingerprint, the probe's binding, a run's metrics and config.json, remote
self-play's tallies and a worker whose checkout searches otherwise refused. The toy reveals its round
cards only as a new round begins, which the probe leaves to time: its decisions are probed and never
heavy (the Rust tests' test game `peek` and MTG Zero's Delver mirror have heavy ones)."""
import json

import pytest

from zero import games as G, heavy as HV
from zero.run import DEFAULTS, Run, read_metrics

from test_loop import run_loop
from test_remoteplay import TOY, env_for


def cfg_of(**kw):
    c = dict(DEFAULTS, game='toy', sims=8, sims_fast=4)
    c.update(kw)
    return c


@pytest.mark.parametrize('kw, msg', [
    (dict(selfplay_sims_heavy_fast=8), 'without selfplay_sims_heavy'),
    (dict(selfplay_chance_k_heavy=2), 'without selfplay_sims_heavy'),
    (dict(selfplay_root_min_visits_heavy=4), 'without selfplay_sims_heavy'),
    (dict(selfplay_heavy_prior_max=0.05), 'without selfplay_sims_heavy'),
    (dict(selfplay_heavy_depth=1), 'without selfplay_sims_heavy'),
    (dict(selfplay_sims_heavy=4, selfplay_sims_heavy_fast=4), 'at least the one it replaces'),
    (dict(selfplay_sims_heavy=32), 'set selfplay_sims_heavy_fast'),
    (dict(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=2), 'at least the one it replaces'),
    (dict(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=8, selfplay_chance_k_heavy=0), 'would do nothing'),
    (dict(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=8, selfplay_heavy_prior_max=0), 'in \\(0, 1\\]'),
    (dict(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=8, selfplay_heavy_depth=5), '1..4'),
    (dict(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=8, selfplay_root_search='gumbel'), 'exclusive'),
    (dict(selfplay_sims_heavy=-1), '>= 0'),
])
def test_settings_that_would_do_nothing_are_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        HV.settings(cfg_of(**kw))


def test_the_seats(tmp_path):
    off = Run(str(tmp_path / 'off'), cfg_of())
    on = Run(str(tmp_path / 'on'), cfg_of(selfplay_sims_heavy=32, selfplay_sims_heavy_fast=8, selfplay_chance_k_heavy=4,
                                          selfplay_root_min_visits_heavy=2, selfplay_heavy_prior_max=0.05))
    assert off.heavy is None
    assert on.heavy == {'sims_heavy': 32, 'sims_heavy_fast': 8, 'chance_k_heavy': 4, 'root_min_visits_heavy': 2,
                        'heavy_prior_max': 0.05, 'heavy_depth': 2, 'heavy_horizon': 12, 'heavy_lines': 64}
    sp_off, sp_on = off.nn_seat('m', 8, selfplay=True), on.nn_seat('m', 8, selfplay=True)
    assert not any(k.startswith('sims_heavy') or 'heavy' in k for k in sp_off)
    assert dict(sp_on) == dict(sp_off, sims_heavy=32, sims_heavy_fast=8, chance_k_heavy=4, root_min_visits_heavy=2,
                               heavy_prior_max=0.05, heavy_depth=2, heavy_horizon=12, heavy_lines=64)
    # every evaluation seat is the seat of before, key for key
    assert on.nn_seat('m', 8) == off.nn_seat('m', 8) and on.nn_seat('o', 32) == off.nn_seat('o', 32)
    assert HV.fingerprint(on.cfg) == ('sims_heavy=32,sims_heavy_fast=8,chance_k_heavy=4,root_min_visits_heavy=2,'
                                      'heavy_prior_max=0.05,heavy_depth=2,heavy_horizon=12,heavy_lines=64')
    assert HV.fingerprint(off.cfg) == ''
    with pytest.raises(ValueError, match="self-play seat's setting"):
        G.nn_seat('m', 8, sims_heavy=32)
    with pytest.raises(ValueError, match='without sims_heavy'):
        G.nn_seat('m', 8, selfplay=True, chance_k_heavy=2)
    # the Rust driver refuses what the seat dict would do nothing with (an evaluation seat's keys)
    with pytest.raises(ValueError, match='self-play seat'):
        G.Driver('toy', [[dict(G.nn_seat('m', 8), sims_heavy=32)] * 2], concurrency=1, seed=1)


def test_the_probe_binding_and_the_heavy_floor():
    E = G.engine()
    g = E.Game('toy', None, 2, 5)
    while len(g.legal()) < 2:
        g.step(g.legal()[0])
    o = g.heavy_probe(seed=3)
    assert set(o) == {'reveal', 'probed', 'lines', 'steps', 'exhausted'}
    assert len(o['reveal']) == len(g.legal()) and not any(o['reveal']) and o['probed'] >= 1 and o['lines'] >= 1
    assert g.heavy_probe(seed=3, cand=[False] * len(g.legal()))['probed'] == 0
    with pytest.raises(ValueError, match='candidates'):
        g.heavy_probe(cand=[True])
    with pytest.raises(ValueError, match='heavy_depth'):
        g.heavy_probe(depth=0)
    # the toy has no passive option; every key reveals (the default)
    assert g.passive_action() is None and g.reveals(123, 0)
    assert (E.heavy_floor(8, 6, 256), E.heavy_floor(8, 6, 64), E.heavy_floor(8, 1, 256)) == (8, 5, 0)


def heavy_rows(run):
    return [m['selfplay'] for m in read_metrics(str(run)) if 'selfplay' in m]


def test_a_run_with_heavy_decisions_and_one_without(tmp_path):
    run = tmp_path / 'h'
    p = run_loop(run, '--iters', '2', '--selfplay-sims-heavy', '32', '--selfplay-sims-heavy-fast', '8',
                 '--selfplay-chance-k-heavy', '2', '--selfplay-root-min-visits-heavy', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    sp = heavy_rows(run)
    assert len(sp) == 2
    for s in sp:
        hv = s['heavy']
        assert hv['games'] == s['games'] == 24
        # the toy: every searched decision probed, none heavy (its revelations come with its rounds)
        assert hv['probed_share'] == 1.0 and hv['per_game'] == 0 and hv['sims_per_game'] == 0
        assert hv['sims_all_per_game'] > 0 and hv['probe_lines'] >= 1 and hv['probe_ms'] >= 0
    c = json.load(open(run / 'config.json'))
    assert (c['selfplay_sims_heavy'], c['selfplay_sims_heavy_fast'], c['selfplay_chance_k_heavy'],
            c['selfplay_root_min_visits_heavy'], c['selfplay_heavy_prior_max']) == (32, 8, 2, 2, 1.0)
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'heavy decisions: 32 simulations (full draws) / 8 (fast draws), chance_k 2, root floor 2' in log
    assert '"heavy": {"games": 24' in log
    # off: the metrics of before (no `heavy`)
    run0 = tmp_path / 'o'
    p = run_loop(run0, '--iters', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    assert all('heavy' not in s for s in heavy_rows(run0))
    # a setting that would do nothing is refused at the start
    p = run_loop(tmp_path / 'x', '--iters', '1', '--selfplay-chance-k-heavy', '2')
    assert p.returncode != 0 and 'without selfplay_sims_heavy' in p.stdout + p.stderr


def test_remote_selfplay_tallies_and_refuses_a_checkout_without_heavy_decisions(tmp_path):
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '3', *TOY, '--selfplay-remote', 'local', '--selfplay-sims-heavy', '16',
                 '--selfplay-sims-heavy-fast', '8', env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    remote = 0
    for x in read_metrics(str(run))[1:]:
        sp = x['selfplay']
        assert sp['heavy']['games'] == sp['games'] + sp['remote']['games'], (sp['heavy'], sp['games'], sp['remote'])
        remote += sp['remote']['games']
    assert remote > 0
    assert json.load(open(tmp_path / 'mirror' / 'sp' / 'config.json'))['selfplay_sims_heavy'] == 16
    # a worker whose checkout has no heavy decisions (its hello reports none) is killed at its hello
    run = tmp_path / 'sp2'
    p = run_loop(run, '--iters', '2', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-sims-heavy', '16', '--selfplay-sims-heavy-fast', '8',
                 env=env_for(tmp_path, {'no_heavy': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    for x in read_metrics(str(run))[1:]:
        assert x['selfplay']['games'] == 200 and x['selfplay']['remote']['games'] == 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "REFUSED: its checkout searches heavy decisions as None, the run as 'sims_heavy=16" in log
