"""Per-seat evaluation settings (docs/az.md 68): zero.seat_settings - one mapping of zero.h2h's per-side options to run
settings, used by zero.h2h (local) and the farm's eval job seats (`{"net": gen, "set": {...}}`) - and zero.farm_eval's
seats with a set.

* The keys: zero.seat_settings.SEAT_KEYS == zero_worker settings::SEAT_READS == what Run.nn_seat reads for an evaluation
  seat (plus eval_sims).
* Drift: every per-side option of zero.h2h is mapped (SIDE_OPTIONS, whose run settings are seat keys) or local-only
  (LOCAL_ONLY); zero.farm_h2h takes exactly the mapped ones; a per-side option zero.h2h gains fails here until it is
  one or the other.
* The meaning: side_set per option; zero.h2h's seats (h2h.sides_of) are the farm's seats of the same sets
  (farm_eval.seat_dicts) key for key; a seat without a set is the seat of before.
* Python matches: a set restating the settings plays the same games (the rows differ only in their seats, written as
  the worker writes them: keys sorted); a set that changes a seat's search plays other games; two seats of one network.
"""
import argparse
import json
import os
import re

import pytest

from zero import farm_eval as FE, h2h, run as R, seat_settings as SS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rs_list(name):
    with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'settings.rs'), encoding='utf-8') as f:
        m = re.search(r'pub const ' + name + r': &\[&str\] = &\[(.*?)\];', f.read(), re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


class _Track(dict):
    def __init__(self, d):
        super().__init__(d)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, d=None):
        self.read.add(k)
        return super().get(k, d)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)


def test_seat_keys_are_what_an_evaluation_seat_reads():
    seen = set()
    base = dict(R.DEFAULTS, game='toy', eval_spec='{}')
    for extra in ({}, {'eval_value_choice_worlds': 4, 'eval_value_choice_temp': 0.5}, {'eval_select': 'cq'},
                  {'eval_belief_tilt': 1.0}, {'eval_root_search': 'gumbel'}):
        c = _Track(dict(base, **extra))
        FE.seat_dicts(c, [{'net': 1}, {'net': 1}])
        seen |= c.read
    assert seen == set(SS.SEAT_KEYS), (sorted(seen - set(SS.SEAT_KEYS)), sorted(set(SS.SEAT_KEYS) - seen))
    assert list(SS.SEAT_KEYS) == _rs_list('SEAT_READS')
    assert set(SS.SEAT_KEYS) <= set(_rs_list('EVAL_READS'))


def _dests(ap):
    return {x.dest for x in ap._actions}


def test_every_per_side_option_of_zero_h2h_is_mapped_or_local_only():
    """Like crates/worker/src/run_settings.json for self-play: a per-side option zero.h2h gains (--a-x / --b-x) must be
    mapped to run settings (SIDE_OPTIONS: then a farm seat can play it) or declared local-only (LOCAL_ONLY)."""
    d = _dests(h2h.parser())
    per = {x[2:] for x in d if x.startswith('a_')}
    assert per == {x[2:] for x in d if x.startswith('b_')}
    known = set(SS.SIDE_OPTIONS) | set(SS.LOCAL_ONLY)
    assert per == known, (f'per-side options of zero.h2h neither mapped (zero.seat_settings.SIDE_OPTIONS) nor local-only '
                          f'(LOCAL_ONLY): {sorted(per - known)}; mapped but gone: {sorted(known - per)}')
    both = {x for x in d if f'a_{x}' in d} - {'sims'}        # (--sims: the match's own simulations, the settings' base)
    assert both == set(SS.BOTH_OPTIONS) | set(SS.LOCAL_ONLY), both
    for k, keys in SS.SIDE_OPTIONS.items():
        assert keys and set(keys) <= set(SS.SEAT_KEYS), k
    assert not set(SS.SIDE_OPTIONS) & set(SS.LOCAL_ONLY)


def test_farm_head_to_heads_take_the_mapped_options_only():
    from zero import farm_h2h as FH
    d = _dests(FH.parser())
    for k in SS.SIDE_OPTIONS:
        assert {f'a_{k}', f'b_{k}'} <= d, k
    for k in SS.BOTH_OPTIONS:
        assert k in d, k
    for k in SS.LOCAL_ONLY:
        assert not {f'a_{k}', f'b_{k}', k} & d, k


CFG = dict(R.DEFAULTS, game='toy', eval_spec='{}', eval_sims=12, conc=4, threads=1, groups=1, eval_select='cq',
           eval_select_min_visits=4, eval_select_c_scale=0.3, cq_c_scale=0.3, eval_leaf_value_view='root',
           eval_belief_tilt=0.0, belief_mix=0.25)


def ns(argv=(), local=True):
    ap = argparse.ArgumentParser()
    SS.add_side_args(ap, local=local)
    return ap.parse_args(list(argv))


def test_side_set_means_what_zero_h2h_means():
    s = lambda *argv, side='a', cfg=CFG: SS.side_set(ns(argv), side, cfg)
    assert s() == {} and s(side='b') == {}
    assert s('--a-sims', '40') == {'eval_sims': 40} and s('--a-sims', '40', side='b') == {}
    assert s('--a-infoset-merge', '1') == {'eval_infoset_merge': 1}
    assert s('--infoset-merge', '1', side='b') == {'eval_infoset_merge': 1}
    with pytest.raises(ValueError, match='leaf-value-view'):
        s('--a-infoset-merge', '1', '--a-leaf-value-view', 'mover')
    # the rule replaced: its guard 0 and transform 50 / 1 unless given; 'visits' drops the 'cq' keys
    assert s('--a-select', 'cq') == {'eval_select_min_visits': 0, 'eval_select_c_visit': 50.0, 'eval_select_c_scale': 1.0}
    assert s('--a-select', 'visits') == {'eval_select': 'visits', 'eval_select_min_visits': 0, 'eval_select_c_scale': None}
    assert s('--a-select', 'cq', '--a-select-min-visits', '2', '--a-cq-c-scale', '0.5') == {
        'eval_select_min_visits': 2, 'eval_select_c_visit': 50.0, 'eval_select_c_scale': 0.5}
    # a guard or a transform alone: of the side's own rule
    assert s('--a-select-min-visits', '1') == {'eval_select_min_visits': 1}
    assert s('--a-cq-c-visit', '20') == {'eval_select_c_visit': 20.0}
    with pytest.raises(ValueError, match="plays 'visits'"):
        s('--a-select', 'visits', '--a-cq-c-scale', '0.5')
    assert s('--select', 'visits', '--cq-c-scale', '0.5', side='b') != {}           # (both-sides flags skip 'visits')
    with pytest.raises(ValueError, match="no side plays 'cq'"):
        SS.sets_of(ns(['--select', 'visits', '--cq-c-scale', '0.5']), CFG)
    # the root belief: a belief its run evaluates without takes clip 20, mix 0, full searches unless given
    # (CFG: belief_clip 20 and belief_fast 0 already, belief_mix 0.25 - self-play's)
    assert s('--a-belief-tilt', '1') == {'eval_belief_tilt': 1.0, 'belief_mix': 0.0}
    assert s('--a-belief-tilt', '1', '--a-belief-mix', '0.25') == {'eval_belief_tilt': 1.0}
    assert s('--a-belief-tilt', '1', cfg=dict(CFG, belief_clip=10.0, belief_fast=1)) == {
        'eval_belief_tilt': 1.0, 'belief_clip': 20.0, 'belief_mix': 0.0, 'belief_fast': 0}
    on = dict(CFG, eval_belief_tilt=1.0)
    assert s('--a-belief-tilt', '0', cfg=on) == {'eval_belief_tilt': 0.0}
    assert s('--a-belief-mix', '0.1', cfg=on) == {'belief_mix': 0.1}
    assert s('--a-belief-tilts', '{"next_card": 0.5, "x": 0}') == {'eval_belief_tilts': {'next_card': 0.5}, 'belief_mix': 0.0}
    with pytest.raises(ValueError, match='without a root belief'):
        s('--a-belief-clip', '5')
    # the root search
    assert s('--a-root-search', 'gumbel', '--gumbel-m', '4', '--a-gumbel-scale', '1') == {
        'eval_root_search': 'gumbel', 'gumbel_m': 4, 'eval_gumbel_scale': 1.0}
    assert s('--gumbel-m', '4') == {}                                                   # (a PUCT side: skipped)
    for argv in (('--a-gumbel-scale', '1'), ('--a-gumbel-m', '4')):
        with pytest.raises(ValueError, match='PUCT'):
            s(*argv)
    # every set's keys are seat settings
    for argv in (('--a-select', 'visits'), ('--a-belief-tilt', '1'), ('--a-root-search', 'gumbel'), ('--a-sims', '3')):
        assert set(s(*argv)) <= set(SS.SEAT_KEYS)


def _run(tmp_path, cfg):
    d = tmp_path / 'runs' / 'r'
    os.makedirs(d / 'models', exist_ok=True)
    with open(d / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    return str(d / 'models' / 'iter_0001.pt')


@pytest.mark.parametrize('argv', [
    [], ['--a-infoset-merge', '1', '--b-infoset-merge', '0'], ['--a-sims', '24', '--b-sims', '6'],
    ['--a-select', 'visits'], ['--select', 'cq', '--a-cq-c-scale', '0.5'], ['--a-root-search', 'gumbel', '--gumbel-m', '4'],
    ['--a-belief-tilt', '1', '--b-belief-tilt', '0'], ['--b-leaf-value-view', 'mover', '--a-select-min-visits', '2'],
])
def test_zero_h2h_seats_are_the_farm_seats_of_the_same_sets(tmp_path, argv):
    """One checkpoint on both sides: zero.h2h's seats (each side's run settings and its options, h2h.sides_of) are the
    seats an eval job of the run's settings plays with the sets zero.farm_h2h gives the same options (farm_eval)."""
    ck = _run(tmp_path, {k: CFG[k] for k in CFG if k not in ('conc', 'threads', 'groups')})
    a = h2h.parser().parse_args([ck, ck, '--sims', '12', '--chance-k', '0'] + argv)
    local = h2h.sides_of(a)
    cfg = h2h.side_settings(ck, 12, 1.0, 0)
    sets = SS.sets_of(a, cfg)
    seats = [SS.net_seat(1, sets[0]), SS.net_seat(1, sets[1])]
    farm = FE.seat_dicts(cfg, seats)
    for s in farm:
        s.pop('key')
    assert local == farm
    if not argv:
        assert seats == [{'net': 1}, {'net': 1}]                      # (no option: the seats of before, no set)


def test_seats_with_a_set():
    seats = FE.check_seats([{'set': {'eval_sims': 8, 'eval_belief_tilts': {'z': 1, 'a': 0.5}}, 'net': 2}, {'net': 1}])
    assert json.dumps(seats, separators=(',', ':')) == ('[{"net":2,"set":{"eval_belief_tilts":{"a":0.5,"z":1},'
                                                        '"eval_sims":8}},{"net":1}]')
    for bad in ({'net': 1, 'set': {}}, {'bot': 'heuristic', 'set': {'eval_sims': 8}}, {'net': 1, 'set': {'conc': 2}},
                {'net': 1, 'set': {'lookahead': 'plies=2'}}, {'net': 1, 'x': 1}):
        with pytest.raises(ValueError):
            FE.check_seats([bad, {'net': 1}])
    st = [{'net': 1, 'set': {'eval_infoset_merge': 1, 'eval_sims': 20}}, {'net': 1}]
    got = FE.seat_dicts(CFG, st)
    want = [R.Run.nn_seat(FE._Stub(dict(CFG, eval_infoset_merge=1, eval_sims=20)), 'm', 20),
            R.Run.nn_seat(FE._Stub(CFG), 'o', 12)]
    assert got == want and got[0]['infoset_merge'] and 'infoset_merge' not in got[1]
    assert FE.seat_tags(CFG, st)[0].endswith('-ism') and not FE.seat_tags(CFG, st)[1].endswith('-ism')


def _rows(cfg, seats, games=8):
    from zero.infer import UniformEvaluator, drive
    drv = FE.match_driver(cfg, seats, games, 7, paired=True)
    drive(drv, {'m': UniformEvaluator(), 'o': UniformEvaluator()})
    return {r['gid']: FE.row_json(FE.row(r, seats, True)) for r in drv.take_results()}


def test_python_matches_play_each_seats_settings():
    cfg = dict(CFG, eval_leaf_value_view='mover', eval_sims=10)
    plain = _rows(cfg, [{'net': 1}, {'net': 2}])
    same = _rows(cfg, [{'net': 1, 'set': {'eval_sims': 10, 'eval_select': 'cq'}}, {'net': 2}])
    st = '{"net":1,"set":{"eval_select":"cq","eval_sims":10}}'
    for g, line in same.items():
        assert st in line and line.replace(st, '{"net":1}') == plain[g], g
    other = _rows(cfg, [{'net': 1, 'set': {'eval_sims': 40}}, {'net': 1, 'set': {
        'eval_select': 'visits', 'eval_select_min_visits': 0, 'eval_select_c_scale': None}}])
    res = lambda m: {g: json.loads(x)['result'] + [json.loads(x)['rounds']] for g, x in m.items()}
    assert res(other) != res(plain)
    rows = [json.loads(x) for x in other.values()]
    s0 = {'net': 1, 'set': {'eval_sims': 40}}
    FE.a_seat_of(rows, s0)                                          # (A: seats[0] as a whole, its set telling it apart)
    with pytest.raises(ValueError):
        FE.a_seat_of(rows, {'net': 1})
    assert FE.summary(rows, True, s0)['games'] == 8


def test_the_eval_gates_seat_cases_are_valid_seats():
    """tools/worker_eval_gate.py --seat-cases: each case's seats are two network seats with sets of seat settings that
    Run.nn_seat accepts, A and B different."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import worker_eval_gate as WG
    for case in WG.SEAT_CASES:
        seats, _ = WG.case_seats(CFG, [{'net': 1}, {'net': 1}], case)
        assert FE.check_seats(seats) == seats and seats[0] != seats[1], case
        d = FE.seat_dicts(CFG, seats)
        assert d[0] != d[1] or case == 'sims', case
