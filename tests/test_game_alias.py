"""Game aliases (docs/az.md 73): a game NAME whose meaning changed. The stand-in: `toy_match` is "revision 1" of its
name, and a record naming `toy_match` without a layout record is the game `toy` (what the name meant before)."""
import argparse
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from zero import games as G, net as N, package, run as R  # noqa: E402

ALIASES = {'toy_match': {'layout': 1, 'before': 'toy'}}


@pytest.fixture
def aliased(monkeypatch):
    monkeypatch.setattr(package.CORE, 'game_aliases', dict(ALIASES))
    return package.CORE


def test_the_rule(aliased):
    r = package.resolve_game
    assert r('toy_match', 1) == 'toy_match'
    assert r('toy_match', None) == 'toy'             # written before the change: the old game
    assert r('toy', None) == 'toy'                   # the old game by its own name
    assert r('toy_team', None) == 'toy_team'         # a name that never changed
    for name, lay in (('toy_match', 2), ('toy', 1), ('toy_team', 1), ('toy_match', True), ('toy_match', '1')):
        with pytest.raises(package.GameAliasError):
            r(name, lay)
    assert package.layout_stamp('toy_match') == {'game_layout': 1}
    assert package.layout_stamp('toy') == {} and package.layout_stamp('toy_team') == {}
    assert package.record_game({'game': 'toy_match'}) == 'toy'
    assert package.record_game({'game': 'toy_match', 'game_layout': 1}) == 'toy_match'
    assert package.record_game({}) is None and package.record_game({}, default='toy_match') == 'toy'
    # resolving a resolved name with its own stamp gives it back (what is written reads back as itself)
    for g in ('toy', 'toy_match', 'toy_team'):
        assert package.record_game(package.set_game({}, g)) == g
    d = package.set_game({'game_layout': 1}, 'toy')    # (a run's settings hold the key: None for a game without one)
    assert d == {'game': 'toy', 'game_layout': None}


def test_the_table_is_checked():
    for bad in ({'x': {'layout': 0, 'before': 'y'}}, {'x': {'layout': 1}}, {'x': {'layout': 1, 'before': 'x'}},
                {'x': {'layout': 1, 'before': 'y'}, 'y': {'layout': 1, 'before': 'z'}},
                {'x': {'layout': 1, 'before': 'y', 'other': 1}}):
        with pytest.raises(ValueError, match='game_aliases'):
            package.GamePackage('t', 'T', ROOT, game_aliases=bad)
    assert package.CORE.game_aliases == {}           # the core's toy games: no alias, nothing written


def resolve(saved, **given):
    ns = argparse.Namespace(**{k: None for k in R.DEFAULTS})
    for k, v in given.items():
        setattr(ns, k, v)
    return R.resolve_config(ns, dict(saved))


def test_a_run_keeps_the_game_it_was_made_with(aliased):
    # a first start: the name's current revision, recorded
    c = resolve({}, game='toy_match')
    assert (c['game'], c['game_layout']) == ('toy_match', 1)
    # resumed: as recorded (the written config reads back as itself)
    c2 = resolve(json.loads(json.dumps(c)))
    assert (c2['game'], c2['game_layout']) == ('toy_match', 1)
    # a config of before the change (the name, no layout record): the old game, and it stays that across resumes
    old = {'game': 'toy_match', 'iters': 3}
    c3 = resolve(old)
    assert (c3['game'], c3['game_layout']) == ('toy', None)
    assert resolve(c3)['game'] == 'toy'
    # the same name given again on the command line changes nothing
    assert resolve(old, game='toy_match')['game'] == 'toy'
    # moving it takes an explicit layout record (then the encoding guard decides on its checkpoints)
    assert resolve(old, game_layout=1.0)['game'] == 'toy_match'
    with pytest.raises(SystemExit, match='game_layout 2'):
        resolve({'game': 'toy_match', 'game_layout': 2})
    # another game given now: its current meaning
    assert resolve(c3, game='toy_match')['game'] == 'toy_match'
    # a game whose name never changed: no layout record, its configs exactly as before
    c4 = resolve({}, game='toy_team')
    assert (c4['game'], c4['game_layout']) == ('toy_team', None)


def test_presets_and_defaults_cannot_set_the_layout_record(monkeypatch):
    p = package.GamePackage('t', 'T', ROOT, presets={'x': {'game': 'toy', 'game_layout': 1}})
    with pytest.raises(ValueError, match='game_layout'):
        R.apply_package(p)
    with pytest.raises(ValueError, match='game_layout'):
        R.apply_package(package.GamePackage('t', 'T', ROOT, defaults={'game_layout': 1}))
    R.apply_package(package.active())


def test_a_checkpoint_of_before_the_change_loads_as_the_old_game(aliased, tmp_path):
    old = N.make_net(G.info('toy'), 'toy')
    assert 'game_layout' not in old.config         # (an unaliased game: the config of before)
    # a checkpoint written before the change recorded the name, no layout record: the old game's network
    old.config['game'] = 'toy_match'
    p = tmp_path / 'old.pt'
    N.save(old, str(p))
    m, _ = N.load(str(p))
    assert m.config['game'] == 'toy' and 'game_layout' not in m.config
    N.check_checkpoint(str(p), 'toy')
    with pytest.raises(N.EncodingMismatch, match="game 'toy' -> 'toy_match'"):
        N.load(str(p), game='toy_match')
    # a network of the name's revision records it
    new = N.make_net(G.info('toy_match'), 'toy_match')
    assert new.config['game'] == 'toy_match' and new.config['game_layout'] == 1
    q = tmp_path / 'new.pt'
    N.save(new, str(q))
    m2, _ = N.load(str(q))
    assert m2.config['game'] == 'toy_match' and m2.config['game_layout'] == 1
    with pytest.raises(N.EncodingMismatch):
        N.load(str(q), game='toy')
    # a layout record this build has no revision for: refused, never guessed
    bad = torch.load(str(q), weights_only=False)
    bad['config']['game_layout'] = 5
    torch.save(bad, str(tmp_path / 'bad.pt'))
    with pytest.raises(N.EncodingMismatch, match='game_layout 5'):
        N.load(str(tmp_path / 'bad.pt'))


def test_fork_run_moves_a_network_onto_the_names_revision(aliased, tmp_path):
    from zero.tools import fork_run
    old = N.make_net(G.info('toy'), 'toy')
    old.config['game'] = 'toy_match'                 # (a checkpoint of before the change)
    src = tmp_path / 'src.pt'
    N.save(old, str(src))
    fork_run.new_run(str(src), str(tmp_path / 'fork'), 'toy_match')
    m, _ = N.load(str(tmp_path / 'fork' / 'models' / 'iter_0000.pt'))
    assert m.config['game'] == 'toy_match' and m.config['game_layout'] == 1


def test_records_resolve(aliased, tmp_path):
    from zero.tools.distill import _meta_game
    assert _meta_game({'game': 'toy_match'}, 'toy') == 'toy'
    assert _meta_game({'game': 'toy_match', 'game_layout': 1}, 'toy') == 'toy_match'
    assert 'game_layout 3' in _meta_game({'game': 'toy_match', 'game_layout': 3}, 'toy')


def test_the_worker_table_declares_the_layout_record_python_only():
    with open(os.path.join(ROOT, 'crates', 'worker', 'src', 'run_settings.json'), encoding='utf-8') as f:
        t = json.load(f)
    e = t['keys']['game_layout']
    assert e['class'] == 'python_only' and e['off'] is None and R.DEFAULTS['game_layout'] is None
