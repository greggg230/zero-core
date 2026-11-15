"""Traced eval rows (the eval setting `eval_trace`, docs/az.md 50): zero.farm_eval's desktop side - a record's digest
equals the worker's (the SHA-256 of serde_json's text of the record, which `take_game_records` returns), the games'
digest is independent of how games are cut into leases, a traced row replays through the engine from its game's seed,
and a tampered row (a move, the result, the seed, the digest) is refused; a Python match refuses the setting."""
import hashlib
import json

import pytest

from zero import _engine as E, farm_eval as FE, games as G
from zero.infer import drive
from zero.run import DEFAULTS

SEED = 31


class Uniform:
    def __call__(self, lb):
        import numpy as np
        cnt = np.diff(lb.opt_off)
        pri = np.repeat((1.0 / np.maximum(cnt, 1)).astype(np.float32), cnt)
        v = np.full((lb.B, 8), 0.5, np.float32)
        return pri, v, np.zeros((lb.B, 8), np.float32)


def traced_rows(n, seed=SEED):
    """n paired games of a traced rolling toy driver (as the worker's eval job plays them) -> rows with digest + rec,
    and each record's text as the engine wrote it."""
    seat = G.nn_seat('m', 8)
    drv = G.Driver('toy', [[seat, seat]], spec='', concurrency=4, seed=seed, threads=2, groups=1, rolling=True,
                   paired=True)
    drv.set_trace(True)
    out = []
    while len(out) < n:
        drv.resume(n - len(out), 0)
        drive(drv, {'m': Uniform()})
        rows, texts, _ = drv.take_game_records()
        for r, t in zip(rows, texts):
            row = FE.row(r, [{'net': 1}, {'net': 2}], True)
            row['digest'] = hashlib.sha256(t.encode()).hexdigest()[:16]     # (the worker's: serde_json's text)
            row['rec'] = json.loads(t)
            out.append((row, t))
    return out[:n]


def test_digest_is_the_workers_and_rows_replay():
    got = traced_rows(8)
    for row, text in got:
        assert FE.trace_digest(row['rec']) == row['digest'], 'Python\'s canonical text is serde_json\'s'
        assert json.dumps(row['rec'], sort_keys=True, separators=(',', ':'), ensure_ascii=False) == text
        info = FE.replay_trace(row, 'toy', '{}', E, SEED, paired=True)
        assert info['gid'] == row['gid'] and info['rows'] == 0
    rows = [r for r, _ in got]
    assert FE.games_digest(rows) == FE.games_digest(list(reversed(rows)))
    assert FE.games_digest(rows[:4]) != FE.games_digest(rows)


def test_tampered_rows_are_refused():
    row, _ = traced_rows(2)[0]
    bad = json.loads(json.dumps(row))
    bad['digest'] = '0' * 16
    with pytest.raises(FE.TraceMismatch, match='digest'):
        FE.check_digest(bad)
    bad = json.loads(json.dumps(row))
    bad['rec']['steps'][0] = int(bad['rec']['steps'][0]) + 1000
    bad['digest'] = FE.trace_digest(bad['rec'])
    with pytest.raises(FE.TraceMismatch, match='replay refuses'):
        FE.replay_trace(bad, 'toy', '{}', E, SEED)
    with pytest.raises(FE.TraceMismatch, match='seed'):
        FE.replay_trace(row, 'toy', '{}', E, SEED + 1)
    bad = json.loads(json.dumps(row))
    bad['result'] = [1 - x for x in bad['result']]
    with pytest.raises(FE.TraceMismatch, match='result'):
        FE.replay_trace(bad, 'toy', '{}', E, SEED)
    bad = json.loads(json.dumps(row))
    del bad['rec']
    with pytest.raises(FE.TraceMismatch, match='not a traced row'):
        FE.check_digest(bad)


def test_a_python_match_refuses_eval_trace():
    cfg = dict(DEFAULTS, game='toy', eval_spec='{}', eval_trace=1)
    with pytest.raises(ValueError, match='eval_trace'):
        FE.match_driver(cfg, [{'net': 1}, {'net': 2}], 4, 7)


def test_self_play_records_replay_with_their_samples():
    """A play job with search samples (the farm's self-play records): replay_trace checks the samples too (samples=True,
    aux as the file says); the eval rule (no samples) refuses such a record."""
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=2, p_full=0.5)
    drv = G.Driver('toy', [[seat, seat]], spec='', concurrency=4, seed=SEED, record=True, threads=2, groups=1,
                   rolling=True, aux=True)
    drv.set_trace(True)
    drv.resume(4, 0)
    drive(drv, {'m': Uniform()})
    rows, texts, _ = drv.take_game_records()
    assert texts
    for r, t in zip(rows, texts):
        rec = json.loads(t)
        assert rec['samples']['step'], 'self-play rows'
        row = FE.row(r, [{'net': 1}, {'net': 1}], False)
        row.update(digest=FE.trace_digest(rec), rec=rec)
        info = FE.replay_trace(row, 'toy', '{}', E, SEED, paired=False, samples=True, aux=True)
        assert info['rows'] == len(rec['samples']['step'])
        with pytest.raises(FE.TraceMismatch, match='has samples'):
            FE.replay_trace(row, 'toy', '{}', E, SEED, paired=False)
