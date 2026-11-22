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


# ---- readers of a saved config (docs/az.md 73.2): a remote's mirror, a remote evaluation's mirror, an eval job

def saved_config(**over):
    """A run's settings as an old run's config.json holds them: the aliased name, no layout record (`game_layout`
    null: the key of a run of before the change), the remote settings off as a mirror has them."""
    c = resolve({}, game='toy', preset='toy-smoke')
    c.update(eval_remote='', selfplay_remote='', resolver='', resolver_cfg={}, resolver_init='', infer_device='cpu',
             game='toy_match', game_layout=None)
    c.update(over)
    return c


def test_saved_configs_resolve_as_a_resumed_run_does(aliased):
    r = package.resolve_saved_games
    assert r({'game': 'toy_match', 'game_layout': None}) == {'game': 'toy', 'game_layout': None}
    assert r({'game': 'toy_match'}) == {'game': 'toy'}
    assert r({'game': 'toy_match', 'game_layout': 1}) == {'game': 'toy_match', 'game_layout': 1}
    # what the run writes reads back as itself (resolve_run_game's config of a resumed run)
    for c in (resolve({'game': 'toy_match'}), resolve({}, game='toy_match'), resolve({}, game='toy_team')):
        assert r(json.loads(json.dumps(c))) == c
    # a co-training resolver: its record in resolver_cfg
    c = r({'game': 'toy_team', 'resolver': 'toy_match', 'resolver_cfg': {'conc': 4}})
    assert (c['resolver'], c['resolver_cfg']) == ('toy', {'conc': 4})
    c = r({'game': 'toy_team', 'resolver': 'toy_match', 'resolver_cfg': {'game_layout': 1}})
    assert (c['resolver'], c['resolver_cfg']) == ('toy_match', {'game_layout': 1})
    with pytest.raises(package.GameAliasError, match='game_layout 2'):
        r({'game': 'toy_match', 'game_layout': 2}, 'x')
    with pytest.raises(package.GameAliasError, match='resolver'):
        r({'game': 'toy', 'resolver': 'toy_match', 'resolver_cfg': {'game_layout': 7}})


def play_mirror(rdir, cfg, ckpt_game, games=4):
    """zero-core's remote self-play worker (`remoteplay.Worker`) in this process on the mirror `rdir` (config.json =
    `cfg`, a fresh network of `ckpt_game` as iteration 0), as tools/worker_parity.py drives it: a phase of iteration
    1, every finished game accepted, `games` games -> (exit code, its events, the shards' metas, the worker)."""
    import threading
    from zero import remoteplay as RP
    from zero.data import read_shard
    os.makedirs(rdir / 'models')
    os.makedirs(rdir / 'data')
    m = N.make_net(G.info(ckpt_game), ckpt_game, hidden=16, blocks=1)
    N.save(m, str(rdir / 'models' / 'iter_0000.pt'))
    (rdir / 'config.json').write_text(json.dumps(dict(cfg, conc=games, threads=1, groups=1)), encoding='utf-8')
    (rdir / 'selfplay.json').write_text(json.dumps({'mode': 'selfplay', 'locks': [], 'nice': 0, 'run_id': 't',
                                                    'mirror': 't', 'watchdog_s': 1e9, 'linger': False, 'test': {}}),
                                        encoding='utf-8')
    events = []

    class W(RP.Worker):
        def _stdin(self):
            threading.Event().wait()

        def _topup(self, ph):
            if ph['ending']:
                return
            for _ in range(max(0, min(games - self.enq, int(self.cfg['conc']) - self.outstanding()))):
                self.drv.enqueue(f'{self.sid}:{self.enq}', self.spec)
                self.enq += 1

        def emit(self, ev, **kw):
            events.append(dict(kw, ev=ev))
            if ev == 'fin':
                self.ctl.put({'cmd': 'accept', 'iter': kw['iter'], 'n': kw['n']})
                if self.taken >= games and not getattr(self, 'end_sent', False):
                    self.end_sent = True
                    self.ctl.put({'cmd': 'end', 'iter': kw['iter']})
            elif ev == 'end':
                self.ctl.put({'cmd': 'quit'})

    w = W(str(rdir), 0)
    w.ctl.put({'cmd': 'phase', 'iter': 1, 'model': 'models/iter_0000.pt'})
    code = w.main()
    metas = [read_shard(str(p), head_only=True)['meta'] for p in sorted((rdir / 'data').rglob('*.npz'))]
    return code, events, metas, w


def test_a_remote_worker_on_a_mirror_of_before_the_change_plays_the_old_game(aliased, tmp_path):
    """MTG Zero's farm release, gate 2 (2026-11-22): a mirror config copied as it is from a run of before the change
    (the aliased name, `game_layout` null) is the old game - the worker plays it and loads that game's checkpoint (it
    played the name's revision and refused the checkpoint as a network of another encoding)."""
    from zero.runctl import EXIT_REFUSED
    code, events, metas, w = play_mirror(tmp_path / 'old', saved_config(), 'toy')
    assert code == 0, events
    assert (w.cfg['game'], w.cfg['game_layout']) == ('toy', None)
    assert metas and all(m['game'] == 'toy' and 'game_layout' not in m for m in metas)
    # the name's revision with its record: that game; a checkpoint of the old game refused there
    code, events, metas, w = play_mirror(tmp_path / 'new', saved_config(game_layout=1), 'toy_match')
    assert code == 0, events
    assert metas and all((m['game'], m['game_layout']) == ('toy_match', 1) for m in metas)
    code, events, _, _ = play_mirror(tmp_path / 'cross', saved_config(game_layout=1), 'toy')
    assert code == EXIT_REFUSED and [e['kind'] for e in events if e['ev'] == 'error'] == ['encoding']
    # a layout record this checkout has no game for: refused as another encoding (this is not the run's checkout)
    code, events, _, _ = play_mirror(tmp_path / 'bad', saved_config(game_layout=4), 'toy')
    err = [e for e in events if e['ev'] == 'error']
    assert code == EXIT_REFUSED and err[0]['kind'] == 'encoding' and 'game_layout 4' in err[0]['msg']


class _Made(Exception):
    pass


def test_a_remote_evaluation_reads_its_mirror_resolved(aliased, tmp_path, monkeypatch):
    """`python -m <pkg>.az.remote eval|evalnet MIRROR ITER` (zero/remote.py): the mirror's config resolved before
    the Run (or the co-training run) that evaluates is made; a bad layout record refused as another encoding."""
    from zero import cotrain, remote
    made = []

    def make_run(rdir, cfg):
        made.append(cfg)
        raise _Made

    rdir = tmp_path / 'm'
    os.makedirs(rdir)
    for cfg, want in ((saved_config(), ('toy', None)), (saved_config(game_layout=1), ('toy_match', 1))):
        (rdir / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        with pytest.raises(_Made):
            remote.evaluate_mirror(str(rdir), 1, make_run=make_run)
        assert (made[-1]['game'], made[-1]['game_layout']) == want

    class CoRun:
        def __init__(self, run, cfg):
            made.append(cfg)
            raise _Made

    monkeypatch.setattr(cotrain, 'CoRun', CoRun)
    (rdir / 'config.json').write_text(json.dumps(saved_config(game='toy_team', resolver='toy_match',
                                                              resolver_cfg={'conc': 4})), encoding='utf-8')
    with pytest.raises(_Made):
        remote.evaluate_net_mirror(str(rdir), 1)
    assert made[-1]['resolver'] == 'toy' and 'game_layout' not in made[-1]['resolver_cfg']
    (rdir / 'config.json').write_text(json.dumps(saved_config(game_layout=4)), encoding='utf-8')
    with pytest.raises(N.EncodingMismatch, match='game_layout 4'):
        remote.evaluate_mirror(str(rdir), 1, make_run=make_run)


def test_an_eval_jobs_settings_resolve(aliased, monkeypatch):
    """zero.farm_eval.match_driver (the Python port of the native eval job; tools/worker_eval_gate.py's Python
    side): the job's settings' game resolved as the native worker resolves it."""
    from zero import farm_eval as FE
    played = []
    real = G.Driver

    def driver(game, *a, **kw):
        played.append(game)
        return real(game, *a, **kw)

    monkeypatch.setattr(G, 'Driver', driver)
    seats = [{'bot': 'random'}, {'bot': 'random'}]
    for over, want in (({}, 'toy'), ({'game_layout': 1}, 'toy_match')):
        cfg = saved_config(eval_spec='{}', **over)
        FE.match_driver(cfg, seats, 2, 7)
        assert played[-1] == want and cfg['game'] == 'toy_match'      # (the caller's dict as it was)
    with pytest.raises(package.GameAliasError):
        FE.match_driver(saved_config(eval_spec='{}', game_layout=3), seats, 2, 7)
