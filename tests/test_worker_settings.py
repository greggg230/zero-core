"""The native worker's check-config table (crates/worker/src/run_settings.json, docs/az.md 42) classifies
exactly zero-core's run settings: a setting added to zero.run.DEFAULTS without a class there fails here,
so no future setting is ignored by a native worker in silence (zero-farm DESIGN.md 2.1 point 4)."""
import json
import os

from zero import run as R

TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crates', 'worker', 'src',
                     'run_settings.json')
CLASSES = {'play', 'trainer', 'python_only', 'param'}


def table():
    with open(TABLE, encoding='utf-8') as f:
        return json.load(f)


def test_every_run_setting_is_classified():
    keys = table()['keys']
    core = set(R._CORE_DEFAULTS)
    missing = sorted(core - set(keys))
    stale = sorted(set(keys) - core)
    assert not missing, (f'run settings without a class in {TABLE}: {missing} - classify each (play / trainer / '
                         f'python_only / param) and, if play, make the native worker read it (docs/az.md 42)')
    assert not stale, f'{TABLE} classifies settings zero.run no longer has: {stale}'


def test_classes_are_well_formed():
    t = table()
    for k, e in t['keys'].items():
        assert e['class'] in CLASSES, (k, e)
        assert e.get('why'), k
        if e['class'] == 'param':
            assert t['keys'][e['of']]['class'] == 'python_only', (k, e)
        if e['class'] == 'python_only':
            # its off value is DEFAULTS' (a run with the feature off passes check-config)
            assert R._CORE_DEFAULTS[k] == e['off'], (k, R._CORE_DEFAULTS[k], e['off'])
    pats = {(p['prefix'], p['suffix']): p['class'] for p in t['patterns']}
    assert pats == {('aux_', '_w'): 'play', ('reader_', ''): 'trainer'}


def test_package_settings_follow_the_patterns():
    """A game package's aux_<target>_w and reader_<target> settings (zero.run.apply_package) follow the table's
    patterns; its own settings (GamePackage.settings, docs/az.md 40) the worker binary classifies
    (zero_worker::Worker::package_settings)."""
    from zero import package
    pkg = package.active()
    extra = set(pkg.aux_settings()) | set(pkg.reader_settings())
    for k in extra:
        assert (k.startswith('aux_') and k.endswith('_w')) or k.startswith('reader_'), k


def _eval_reads():
    """zero_worker::settings::EVAL_READS (crates/worker/src/settings.rs)."""
    import re
    src = os.path.join(os.path.dirname(TABLE), 'settings.rs')
    with open(src, encoding='utf-8') as f:
        m = re.search(r'pub const EVAL_READS: &\[&str\] = &\[(.*?)\];', f.read(), re.S)
    return set(re.findall(r'"([^"]+)"', m.group(1)))


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


def test_eval_jobs_read_what_an_evaluation_match_reads(monkeypatch):
    """The settings a native eval job reads (EVAL_READS; check-config --eval: every other setting is unused by
    it) are exactly those Run.play_match's driver reads - Run.nn_seat(key, eval_sims) and Run.driver of a
    match (zero.farm_eval.match_driver) -, over the variants that read conditional ones (value choices)."""
    from zero import farm_eval, games as G
    seen = set()

    class Captured(Exception):
        pass

    def fake_driver(game, seats, spec=None, **kw):
        raise Captured()

    monkeypatch.setattr(G, 'Driver', fake_driver)
    base = dict(R.DEFAULTS, game='toy', eval_spec='{}')
    for extra in ({}, {'eval_value_choice_worlds': 4, 'eval_value_choice_temp': 0.5}, {'eval_select': 'cq'},
                  {'eval_belief_tilt': 1.0}):
        c = _Track(dict(base, **extra))
        try:
            farm_eval.match_driver(c, [{'net': 1}, {'bot': 'heuristic'}], 4, 7, paired=True)
        except Captured:
            pass
        seen |= c.read
    want = _eval_reads()
    assert seen == want, (f'read by Python but not in EVAL_READS: {sorted(seen - want)}; in EVAL_READS but not read: '
                          f'{sorted(want - seen)}')
