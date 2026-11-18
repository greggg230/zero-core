"""Every other game recorded (zero/gamerec.py, docs/az.md 59.3-59.4): zero.h2h's matches and sequential batches while a
thread records (`gamerec.recording`), the run's evaluations with `game_records` (the yardstick and random games, the
Elo matches, the anchors' games and calibrations, the reference games - and a remote evaluation's, copied back from its
mirror): the same games, results and Elo files with the records on and off; every game a record naming both players
in their seats that replays to its result; every records file names the engine that played it (`build_info`,
`engine_id`: the Rust inputs only; 'dirty' with an uncommitted change). In-process (no run or worker process)."""
import gzip
import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from zero import elo, gamerec as GR, games as G, remote
from zero.net import Net, save
from zero.run import DEFAULTS, PRESETS, Run, model_path

INFO = G.info('toy')


def _ckpts(d, n=2):
    import torch
    os.makedirs(os.path.join(d, 'models'), exist_ok=True)
    out = []
    for k in range(n):
        torch.manual_seed(k)
        out.append(os.path.join(d, 'models', f'iter_{k:04d}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), out[-1], {'iter': k})
    return out


def replays(rec, spec):
    E = G.engine()
    g = E.Game('toy', spec, rec['n'], rec['seed'])
    for a in GR.actions_of(rec):
        assert a in g.legal()
        g.step(int(a))
    return g.to_move is None, (list(g.result()) if g.to_move is None else None)


def read_all(records):
    """Every records file under `records` that is not self-play -> [(kind, sub, builds, specs, recs)]."""
    out = []
    for kind, sub, path in GR.match_files(records):
        b = []
        specs, recs = GR.read_file(path, b)
        out.append((kind, sub, b, specs, recs))
    return out


def check_replays(files):
    n = 0
    for kind, sub, builds, specs, recs in files:
        assert builds and all(x.get('commit') for x in builds), (kind, builds)
        for r in recs:
            assert r['kind'] == kind and r['spec'] in specs and len(r['players']) == r['n']
            over, res = replays(r, specs[r['spec']])
            assert over == r['over']
            if over:
                assert [round(x, 5) for x in res[:r['n']]] == [round(x, 5) for x in r['result']], (kind, r['gid'])
            n += 1
    return n


def strip(o):
    return {k: v for k, v in o.items() if k not in ('seconds',)}


def test_h2h_records_its_games_and_plays_the_same_ones(tmp_path):
    from zero import h2h
    a, b = _ckpts(str(tmp_path / 'runs' / 'r'))
    kw = dict(games=8, sims=8, chance_k=0, threads=1, rows=True, seed=3)
    plain = h2h.match(a, b, **kw)
    sink = GR.MatchSink(str(tmp_path / 'rec'), 'h2h', labels={'a': 'A', 'b': None}, meta={'why': 'test'})
    with GR.recording(sink):
        assert GR.active() is sink
        got = h2h.match(a, b, **kw)
    assert GR.active() is None
    assert strip(got) == strip(plain)              # (the rows too: their logs taken out)
    files = read_all(str(tmp_path / 'rec'))
    assert [(k, s) for k, s, *_ in files] == [('h2h', 'misc')] and sink.stats['games'] == 8
    assert check_replays(files) == 8
    for r in files[0][4]:
        me = r['gid'] % 2                          # A in seat gid % 2
        assert r['players'][me]['label'] == 'A' and r['players'][1 - me]['label'] == 'r/iter_0001'
        assert r['players'][me]['ckpt'] == 'r/iter_0000' and r['players'][me]['sha'] == GR.file_sha(a)
        assert r['players'][me]['sims'] == 8 and r['meta']['why'] == 'test' and r['meta']['driver_seed'] == 3
    # sequential: each batch a file, its batch and game-id offset in the records' meta, the same games
    plain = h2h.sequential(a, b, batch_games=4, max_games=8, target_se=1e-9, sims=8, chance_k=0, threads=1, seed=5,
                           rows=True)
    with GR.recording(GR.MatchSink(str(tmp_path / 'rec2'), 'h2h')):
        got = h2h.sequential(a, b, batch_games=4, max_games=8, target_se=1e-9, sims=8, chance_k=0, threads=1, seed=5,
                             rows=True)
    assert strip(got) == strip(plain)
    files = read_all(str(tmp_path / 'rec2'))
    assert len(files) == len(got['batches']) and check_replays(files) == got['games']
    assert sorted((f[4][0]['meta']['batch'], f[4][0]['meta']['gid_offset']) for f in files) == [
        (k, 4 * k) for k in range(len(files))]


def test_a_match_sink_writes_whole_files_and_readers_skip_temp_files(tmp_path):
    s = GR.MatchSink(str(tmp_path), 'cells', it=7, labels={'a': 'x'})
    assert s.dir().endswith(os.path.join('cells', 'iter_0007'))
    open(os.path.join(tmp_path, 'stray.jsonl.gz.123.tmp'), 'w').close()
    os.makedirs(os.path.join(tmp_path, 'iter_0003'))             # (self-play's: never a kind)
    assert GR.match_files(str(tmp_path)) == [] and s.add([]) is None
    for bad in ('spec', 'build', 'iter_0001', 'Bad'):
        with pytest.raises(ValueError):
            GR.MatchSink(str(tmp_path), bad)
    c = s.with_meta(pair=['a', 'b'])
    assert c.dir() == s.dir() and c.meta == {'pair': ['a', 'b']} and c.stats is s.stats
    with pytest.raises(RuntimeError, match='no game log'):
        s.add_rows([{'gid': 0}], lambda g: [{}, {}])


def test_the_runs_evaluations_are_recorded_and_unchanged(tmp_path):
    """game_records on and off: the same evaluation metrics and Elo pool; every game recorded under its kind and
    iteration with both players in their seats (the yardstick, random, Elo, anchor, calibration and reference
    games)."""
    out = {}
    for on in (0, 1):
        run = tmp_path / f'r{on}' / 'runs' / 'r'
        cks = _ckpts(str(run), 3)
        anc = _ckpts(str(tmp_path / f'r{on}' / 'anc'), 2)
        cfg = dict(DEFAULTS, **PRESETS['toy-smoke'])
        cfg.update(game='toy', yardstick='heuristic', eval_games=4, eval_random_games=2, elo_games=2, ref_games=2,
                   eval_sims=8, conc=8, threads=1, game_records=on, elo_offsets='1',
                   elo_anchors=f'a0={anc[0]},a1={anc[1]}', elo_anchor_games=2, elo_anchor_calib_games=2)
        R = Run(str(run), cfg)
        m = {'iter': 2}
        R.evaluate(2, m)
        R.elo_update(2, m)
        res = json.load(open(os.path.join(elo.pool_dir(str(run)), 'results.json'), encoding='utf-8'))
        out[on] = (run, {k: m[k] for k in ('eval_yard', 'eval_random', 'elo_anchored')},
                   {k: v for k, v in res.items() if k != 'anchors'},
                   {n: {k: v for k, v in a.items() if k != 'path'} for n, a in res['anchors'].items()})
    assert out[0][1:3] == out[1][1:3] and out[0][3].keys() == out[1][3].keys()
    assert not os.path.exists(os.path.join(out[0][0], 'records'))
    files = read_all(os.path.join(out[1][0], 'records'))
    kinds = {}
    for kind, sub, _, _, recs in files:
        assert sub == 'iter_0002'
        kinds[kind] = kinds.get(kind, 0) + len(recs)
    # eval: 4 yardstick + 2 random; elo: checkpoints 1..2 against their offset 1 (2 games each); ref: random vs
    # heuristic; anchor: a1's calibration against a0, then every checkpoint 1..2 against both anchors
    assert kinds == {'eval': 6, 'elo': 4, 'ref': 2, 'anchor': 2 + 2 * 2 * 2}, kinds
    assert check_replays(files) == 22
    by = {(k, r['gid'], tuple(p['label'] for p in r['players'])) for k, _, _, _, rs in files for r in rs}
    assert ('eval', 0, ('iter_0002', 'heuristic')) in by and ('eval', 1, ('heuristic', 'iter_0002')) in by
    assert ('ref', 0, ('random', 'heuristic')) in by
    assert ('elo', 0, ('iter_0002', 'iter_0001')) in by and ('elo', 1, ('iter_0001', 'iter_0002')) in by
    assert any(k == 'anchor' and set(lb) == {'iter_0001', 'anchor:a0'} for k, _, lb in by)
    assert any(k == 'anchor' and set(lb) == {'anchor:a1', 'anchor:a0'} for k, _, lb in by)      # (the calibration)
    for kind, _, _, _, recs in files:
        for r in recs:
            for p in r['players']:
                assert p['ckpt'] is None or p['sha'], p   # (a network: its file's sha; a bot: none)


def test_a_remote_evaluation_brings_its_records_back(tmp_path, monkeypatch):
    cfg = dict(DEFAULTS, **PRESETS['toy-smoke'])
    cfg.update(game='toy', yardstick='random', eval_games=4, eval_random_games=0, elo_games=2, ref_games=0,
               eval_sims=8, conc=8, threads=1, eval_remote='local', game_records=1)
    run = tmp_path / 'runs' / 'r'
    (run / 'models').mkdir(parents=True)
    R = Run(str(run), cfg)
    for k in (0, 1):
        save(R.new_model(), model_path(str(run), k), {'iter': k})
    monkeypatch.setenv('ZERO_REMOTE_WORKDIR', str(tmp_path / 'remote'))
    ev = remote.RemoteEval.__new__(remote.RemoteEval)
    ev.R, ev.name, ev.spec = R, 'local', remote.remote_spec('local')
    ev.t = remote.Transport(ev.spec)
    ev.rdir = ev.spec['workdir'].rstrip('/') + '/r'
    ev.module, ev.shipped, ev.anchors_shipped, ev.lock, ev.queued = 'zero.remote', None, set(), threading.Lock(), set()

    def run_module(module, args, timeout):
        m = remote.evaluate_mirror(args[1], int(args[2]))
        return SimpleNamespace(returncode=0, stdout=remote.RESULT + json.dumps(m) + '\n', stderr='')
    ev.t.run_module = run_module
    ev._evaluate(1)
    files = read_all(str(run / 'records'))
    assert sorted(k for k, *_ in files) == ['elo', 'eval'] and check_replays(files) == 6
    assert all(r['src'] == 'remote' for *_, recs in files for r in recs)
    row = [json.loads(x) for x in open(run / 'evals.jsonl', encoding='utf-8')][-1]
    assert row['records'] == 2
    ev._evaluate(1)                                # (again: its new games' file; those already here not copied twice)
    mirror = os.path.join(ev.rdir, 'records')
    assert len(read_all(str(run / 'records'))) == len(read_all(mirror)) == 3


def _git(d, *a):
    subprocess.run(['git', '-C', str(d), *a], check=True, capture_output=True)


def test_engine_id_names_the_rust_inputs_only(tmp_path):
    d = tmp_path / 'g'
    (d / 'engine').mkdir(parents=True)
    (d / 'tools').mkdir()
    _git(d, 'init', '-q')
    _git(d, 'config', 'user.email', 'x@example.com')
    _git(d, 'config', 'user.name', 'x')
    (d / 'engine' / 'a.rs').write_text('fn a() {}\n')
    (d / 'tools' / 't.py').write_text('x = 1\n')
    _git(d, 'add', '-A')
    _git(d, 'commit', '-qm', 'one')
    e1 = GR.engine_id(str(d), ('engine',))
    assert e1 and len(e1) == 16 and GR.engine_id(str(d), None) is None
    (d / 'tools' / 't.py').write_text('x = 2\n')
    _git(d, 'commit', '-qam', 'tools only')
    assert GR.engine_id(str(d), ('engine',)) == e1               # (a commit outside the engine keeps it)
    (d / 'engine' / 'a.rs').write_text('fn a() { }\n')
    assert GR.engine_id(str(d), ('engine',)) == 'dirty'          # (an uncommitted engine change)
    _git(d, 'commit', '-qam', 'engine')
    e2 = GR.engine_id(str(d), ('engine',))
    assert e2 not in (e1, 'dirty')
    b = GR.build_info()
    assert set(b) == {'pkg', 'commit', 'core', 'engine', 'module'} and b['core']


def test_self_play_records_files_name_their_engine(tmp_path):
    w = GR.Records(str(tmp_path))
    rec = {'v': 1, 'gid': 0, 'seed': 1, 'spec': GR.spec_key('{}'), 'n': 2, 'actions': None, 'steps': [], 'n_actions': 0}
    w.add(1, 'local', [(rec, '{}')])
    w.add(1, 'w0', [(rec, '{}')], build={'engine': 'e', 'commit': 'c'})
    done = w.close(1, {'local': 1, 'w0': 1})
    assert done['games'] == 2
    fb = {}
    specs, recs = GR.read_iteration(str(tmp_path), 1, fb)
    assert len(recs) == 2 and all(len(v) == 1 for v in fb.values())
    got = {k.split('-')[0]: v[0] for k, v in fb.items()}
    assert got['w0'] == {'engine': 'e', 'commit': 'c'} and got['local'] == GR.build_info()
    d = GR.read_done(str(tmp_path), 1)
    assert {v['source']: v['build'] for v in d['files'].values()}['w0']['engine'] == 'e'
    # a file before the build lines (zero-core 5c2d3f39): its records, no build
    p = os.path.join(GR.iter_dir(str(tmp_path), 2), 'old.jsonl.gz')
    os.makedirs(os.path.dirname(p))
    with gzip.open(p, 'wt', encoding='utf-8') as f:
        f.write(json.dumps({'v': 1, 'kind': 'spec', 'spec': rec['spec'], 'json': '{}'}) + '\n' + json.dumps(rec) + '\n')
    b = []
    assert len(GR.read_file(p, b)[1]) == 1 and b == []


def test_a_subclass_with_its_own_matches_is_called_as_before(tmp_path):
    """Co-training's primary overrides play_match / bots_match (its matches are the resolver's episodes): with
    game_records on they get the calls of before - no `rec` - and record nothing."""
    calls = []

    class Primary(Run):
        def play_match(self, it, opp, n_games, sims, seed, opp_model=None):
            calls.append((opp, n_games))
            return [(1.0, 1.0, 3, True)] * n_games

        def bots_match(self, a, b, n_games, seed):
            calls.append((a, b))
            return n_games / 2

    run = tmp_path / 'runs' / 'r'
    _ckpts(str(run), 2)
    cfg = dict(DEFAULTS, **PRESETS['toy-smoke'])
    cfg.update(game='toy', yardstick='heuristic', eval_games=2, eval_random_games=2, elo_games=2, ref_games=2,
               game_records=1)
    R = Primary(str(run), cfg)
    m = {'iter': 1}
    R.evaluate(1, m)
    R.elo_update(1, m)
    assert ('heuristic', 2) in calls and ('random', 'heuristic') in calls and ('nn', 2) in calls
    assert not os.path.exists(run / 'records')
    assert Run(str(run), cfg).rec_kw('play_match', rec=1) == {'rec': 1}
    assert Run(str(run), dict(cfg, game_records=0)).rec_kw('play_match', rec=1) == {}
