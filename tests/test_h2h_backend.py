"""zero.h2h's backends (docs/az.md 69): --backend auto | farm | local, the game package's `h2h_backend` hook (its own
options, its plan before any game, its batches), the one log line, the JSON line of a match played elsewhere (the keys
and numbers of the local match of the same games, plus `backend` and the backend's own), sequential matches through
`batch_loop` (the local schedule), --records from the backend's traced records. A fake backend plays the toy's games
here, through zero.h2h.match itself, so its line can be compared with the local one number for number."""
import gzip
import json
import os
import struct

import pytest

from zero import gamerec as GR, games as G, h2h, package

from test_gamerec_all import _ckpts, replays


class _Grab:
    """A recording sink that keeps each game's log (zero.h2h._play's records) - the toy's stand-in for a traced record."""

    def __init__(self):
        self.labels, self.logs = {}, {}

    def add_rows(self, rows, players_of, meta=None):
        for r in rows:
            self.logs[int(r['gid'])] = dict(r.pop('log'), n=r['n'], over=r['over'], aborted=r['aborted'],
                                            result=list(r['result']), rounds=r.get('rounds'))


def _rec(gid, lg):
    """A game log -> a traced record's form (zero_search::record::game_json: steps, result as float32 bits)."""
    steps = list(memoryview(bytes(lg['actions'])).cast('H'))
    return {'gid': gid, 'seed': int(lg['seed']), 'n': int(lg['n']), 'steps': steps, 'over': bool(lg['over']),
            'aborted': bool(lg['aborted']), 'resigned': None, 'rounds': lg['rounds'], 'setup': lg.get('setup') or '',
            'result': [struct.unpack('<I', struct.pack('<f', float(x)))[0] for x in lg['result']]}


class _Runner:
    def __init__(self, a, sides):
        self.a, self.sides, self.batches = a, sides, []

    def play(self, k, n, seed):
        self.batches.append((k, n, seed))
        grab = _Grab()
        with GR.recording(grab):
            r = h2h.match(self.a.a, self.a.b, n, self.a.sims, self.a.chance_k, self.a.c_puct, seed, 1, spec=self.a.spec,
                          seat_a=self.sides[0], seat_b=self.sides[1], paired=self.a.paired, sims_a=self.a.a_sims,
                          sims_b=self.a.b_sims, rows=True)
        return [dict(x, rec=_rec(x['gid'], grab.logs[x['gid']])) for x in r['rows']]

    def info(self):
        return {'campaign': 'fake', 'batches_played': len(self.batches)}


class _Backend:
    name = 'farm'

    def __init__(self, ok=True, why='the fake backend'):
        self.ok, self.why, self.asked, self.runner = ok, why, [], None

    def add_args(self, ap):
        ap.add_argument('--fake-min', type=int, default=0)

    def plan(self, a, ask, sides, seq):
        self.asked.append((ask, sides, seq, a.fake_min))
        if not self.ok:
            return False, self.why, None
        self.runner = _Runner(a, sides)
        return True, self.why, self.runner


@pytest.fixture
def backend(monkeypatch):
    def use(be):
        monkeypatch.setattr(package.active(), 'h2h_backend', be)
        return be
    return use


def _line(capsys):
    cap = capsys.readouterr()
    lines = [x for x in cap.err.splitlines() if x.startswith('h2h: backend')]
    return json.loads(cap.out.strip().splitlines()[-1]), lines


BASE = ['--games', '8', '--sims', '4', '--chance-k', '0', '--threads', '1', '--paired']


def test_without_a_backend_the_match_is_local_and_says_so(tmp_path, capsys, backend):
    backend(None)
    a, b = _ckpts(str(tmp_path / 'r'))
    capsys.readouterr()
    h2h.main([a, b] + BASE)
    out, lines = _line(capsys)
    assert lines == ['h2h: backend local - Zero Core has no farm backend (GamePackage.h2h_backend)']
    want = h2h.match(a, b, 8, 4, 0, seed=0, threads=1, paired=True)
    assert out.pop('backend') == 'local' and list(out) == list(want)
    assert {k: v for k, v in out.items() if k != 'seconds'} == {k: v for k, v in want.items() if k != 'seconds'}
    h2h.main([a, b] + BASE + ['--backend', 'local'])
    out2, lines = _line(capsys)
    assert lines == ['h2h: backend local - asked (--backend local)'] and out2['a_score'] == out['a_score']
    with pytest.raises(SystemExit) as e:
        h2h.main([a, b] + BASE + ['--backend', 'farm'])
    assert e.value.code == 2 and 'h2h: backend farm refused - Zero Core has no farm backend' in capsys.readouterr().err


def test_a_backend_prints_the_local_line_of_the_same_games(tmp_path, capsys, backend):
    be = backend(_Backend(why='campaign fake'))
    a, b = _ckpts(str(tmp_path / 'r'))
    side = ['--a-sims', '6', '--b-sims', '3']
    capsys.readouterr()
    h2h.main([a, b] + BASE + side + ['--fake-min', '7', '--records', str(tmp_path / 'rec'), '--label-a', 'A'])
    out, lines = _line(capsys)
    assert lines == ['h2h: backend farm - campaign fake']
    ask, sides, seq, fmin = be.asked[0]
    assert (ask, seq, fmin) == ('auto', False, 7) and be.runner.batches == [(0, 8, 0)]
    # the sides it was asked about are the local match's seats (zero.h2h.sides_of)
    assert sides == h2h.sides_of(h2h.parser().parse_args([a, b] + BASE + side))
    h2h.main([a, b] + BASE + side + ['--backend', 'local'])
    loc, _ = _line(capsys)
    assert (out.pop('backend'), loc.pop('backend')) == ('farm', 'local')
    assert (out.pop('campaign'), out.pop('batches_played')) == ('fake', 1)
    assert list(out) == list(loc) and 'sims_a' in out and out['paired'] is True
    assert {k: v for k, v in out.items() if k != 'seconds'} == {k: v for k, v in loc.items() if k != 'seconds'}
    # the records: one file of the 8 games in the local records' format, A and B in their seats, each replaying
    (path,) = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp_path / 'rec') for f in fs]
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        lines = [json.loads(x) for x in f]
    assert lines[0]['kind'] == 'build' and lines[1]['kind'] == 'spec'
    recs = [x for x in lines if x.get('kind') == 'h2h']
    assert sorted(r['gid'] for r in recs) == list(range(8))
    for r in recs:
        assert r['src'] == 'farm' and r['meta'] == {'tool': 'h2h', 'driver_seed': 0, 'paired': True, 'backend': 'farm'}
        assert r['players'][r['gid'] % 2]['label'] == 'A' and r['players'][r['gid'] % 2]['sims'] == 6
        assert r['players'][1 - r['gid'] % 2]['sims'] == 3
        over, res = replays(r, lines[1]['json'])
        assert over == r['over'] and (not over or res == r['result'])


def test_a_sequential_match_runs_the_local_schedule_through_the_backend(tmp_path, capsys, backend):
    be = backend(_Backend())
    a, b = _ckpts(str(tmp_path / 'r'))
    seq = ['--sprt', '0,400', '--batch-games', '4', '--max-games', '12', '--seed', '3']
    capsys.readouterr()
    h2h.main([a, b] + BASE + seq)
    out, _ = _line(capsys)
    assert be.asked[0][2] is True
    n = len(out['batches'])
    assert be.runner.batches == [(k, 4, 3 + 7919 * k) for k in range(n)]
    h2h.main([a, b] + BASE + seq + ['--backend', 'local'])
    loc, _ = _line(capsys)
    for k in ('backend', 'campaign', 'batches_played', 'seconds'):
        out.pop(k, None)
        loc.pop(k, None)
    assert out == loc and list(out) == list(loc) and out['sprt']['decision'] in ('H0', 'H1', 'undecided')


def test_auto_plays_here_when_the_backend_cannot_and_farm_is_refused(tmp_path, capsys, backend):
    be = backend(_Backend(ok=False, why='--lookahead is local-only'))
    a, b = _ckpts(str(tmp_path / 'r'))
    capsys.readouterr()
    h2h.main([a, b] + BASE)
    out, lines = _line(capsys)
    assert lines == ['h2h: backend local - --lookahead is local-only'] and out['backend'] == 'local'
    with pytest.raises(SystemExit) as e:
        h2h.main([a, b] + BASE + ['--backend', 'farm'])
    assert e.value.code == 2 and 'h2h: backend farm refused - --lookahead is local-only' in capsys.readouterr().err
    assert [x[0] for x in be.asked] == ['auto', 'farm']
    # --backend local never asks it
    h2h.main([a, b] + BASE + ['--backend', 'local'])
    assert len(be.asked) == 2


def test_the_sequential_schedule_is_one_function():
    """batch_loop: the batches, ids, stop rules of zero.h2h.sequential on given rows."""
    rows = {0: [{'gid': g, 'result': [1.0, 0.0] if g % 2 == 0 else [0.0, 1.0], 'scores': [1, 0], 'over': True}
                for g in range(4)]}
    calls = []

    def play(k, n, off):
        calls.append((k, n, off))
        return rows[0][:n]
    bounds = h2h.seq_bounds((0, 20), 0.05, 0.05, None, True, 4, 40)
    rs, batches, stop, llr = h2h.batch_loop(play, 4, 40, True, bounds)
    assert stop == 'sprt H1' and [r['gid'] for r in rs] == list(range(len(rs))) and calls[0] == (0, 4, 0)
    assert all(c[2] == 4 * c[0] for c in calls) and batches[-1]['llr'] == round(llr, 3)
    for bad in (dict(sprt=None, target_se=None), dict(sprt=(0, 20), batch_games=3), dict(sprt=(20, 0)),
                dict(sprt=(0, 20), max_games=2)):
        kw = dict(dict(sprt=None, target_se=None, paired=True, batch_games=4, max_games=40), **bad)
        with pytest.raises(ValueError):
            h2h.seq_bounds(kw['sprt'], 0.05, 0.05, kw['target_se'], kw['paired'], kw['batch_games'], kw['max_games'])
