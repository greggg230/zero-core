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
    """A game package adds only aux_<target>_w and reader_<target> settings (zero.run.apply_package); the
    table's patterns cover those for the game's targets."""
    from zero import package
    pkg = package.active()
    extra = set(pkg.aux_settings()) | set(pkg.reader_settings())
    for k in extra:
        assert (k.startswith('aux_') and k.endswith('_w')) or k.startswith('reader_'), k
