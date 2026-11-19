"""Eval job games as the native worker plays them, for the farm tests' fake coordinator and the worker stub's `eval`
(zero-core has no toy worker binary): games [lo, hi) of an eval batch - its settings, seats, base seed, pairing - by a
rolling traced driver (game g: the seats in `seats` order for even g, swapped for odd g; with `paired` games 2k and
2k + 1 share the deal), each network seat's evaluator the generation's ONNX graph (zero.onnx_eval.OnnxEvaluator). The
driver starts at game 0 (the binding has no first game id), so games below `lo` are played and dropped: the same
games, as a game's seed and seats depend on its id alone. -> the traced rows (zero.farm_eval.row + digest + rec)."""
import hashlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def play_traced(settings, seats, lo, hi, seed, paired, models):
    """`models`: {gen: path of its model.onnx}."""
    from zero import farm_eval as FE, games as G
    from zero.infer import drive
    from zero.onnx_eval import OnnxEvaluator
    sd = FE.seat_dicts(settings, seats)
    evs = {}
    for k, (seat, d) in enumerate(zip(seats, sd)):
        if 'net' in seat:
            evs[d['key']] = OnnxEvaluator(models[seat['net']], threads=1)
    drv = G.Driver(settings['game'], [[sd[0], sd[1]], [sd[1], sd[0]]], spec=settings['eval_spec'], concurrency=4,
                   seed=int(seed), threads=1, groups=1, max_steps=int(settings.get('max_steps', 100000)), rolling=True,
                   paired=bool(paired))
    drv.set_trace(True)
    out = {}
    played = 0
    while any(g not in out for g in range(lo, hi)):
        drv.resume(max(1, hi - played), 0)
        drive(drv, evs)
        rows, texts, _ = drv.take_game_records()
        for r, t in zip(rows, texts):
            played += 1
            g = int(r['gid'])
            if lo <= g < hi:
                row = FE.row(r, seats, bool(paired))
                row['digest'] = hashlib.sha256(t.encode()).hexdigest()[:16]
                row['rec'] = json.loads(t)
                out[g] = row
    return [out[g] for g in range(lo, hi)]
