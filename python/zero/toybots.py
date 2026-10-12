"""Bots for the toy game (Mini Cavern, `crates/search/src/toy.rs`) that play through the
EXTERNAL seat path, the way the JS bot plays Caverna and Glass Road: they read the state JSON
the game exports and answer with a JSON list of action ids.

`heuristic_choice` is an exact port of `ToyGame::heuristic` (Rust): playing a seat with the
external `toyheur` must give the same games as the built-in bot `heuristic`
(tests/test_external.py)."""
import json
import random

S_TIMBER, S_QUARRY, S_FISH, S_START, S_FARM, S_TIMBER2, S_GRAIN = range(7)
CARD0 = 7
C_GRAIN, C_GROWTH, C_BUILD, C_URGENT, C_TRADE, C_FEAST = range(6)
M_NONE, M_HARVEST, M_FEED = range(3)
A_PLOW, A_SKIP, A_INC0, A_INC1, A_DONE = 13, 14, 15, 16, 17
ROUNDS = 6


def _can_build(p, k):
    k1 = k + 1
    return p['wood'] >= 2 * k1 and p['stone'] >= k1 and p['rooms'] + k < 4


def heuristic_choice(s):
    """The heuristic's action for the exported state dict `s` (= ToyGame::heuristic)."""
    legal = s['legal']
    has = legal.__contains__
    p = s['players'][s['cur']]
    marker = s['markers'][s['round'] - 1]
    adults = p['workers'] - p['newborn']
    sown = sum(1 for g in p['field'] if g > 0)
    need = adults if marker == M_FEED else 2 * adults + p['newborn']
    have = p['food'] + p['grain'] + (sown if marker == M_HARVEST else 0)
    sub = s['sub']
    if sub == 'plow':
        return A_PLOW if has(A_PLOW) else A_SKIP
    if sub in ('sow', 'build'):
        return A_INC0 if has(A_INC0) else A_DONE
    if sub == 'trade':
        if have + 2 * s['sub_a'] + s['sub_b'] < need + 1:
            if has(A_INC0):
                return A_INC0
            if has(A_INC1):
                return A_INC1
        return A_DONE
    if sub == 'feast':
        return A_INC0 if have + s['sub_k'] < need + 2 and has(A_INC0) else A_DONE
    goods = s['goods']
    empty = sum(1 for g in p['field'] if g == 0)
    can_plow = p['wood'] >= 1 and p['plowed'] < 4
    short = have < need
    spare = have >= need + 2
    last = s['round'] == ROUNDS
    picks = [
        (short and goods[S_FISH] >= 2, S_FISH),
        (short and p['grain'] >= 1, CARD0 + C_FEAST),
        (short and (p['stone'] >= 1 or p['wood'] >= 1), CARD0 + C_TRADE),
        (short and goods[S_FISH] >= 1, S_FISH),
        (short, S_START),
        (not last and spare and p['workers'] < p['rooms'] and p['workers'] < 4, CARD0 + C_GROWTH),
        (not last and spare and p['workers'] < 4, CARD0 + C_URGENT),
        (s['round'] <= 4 and _can_build(p, 0) and p['rooms'] <= p['workers'], CARD0 + C_BUILD),
        (p['grain'] >= 1 and (empty > 0 or can_plow), S_FARM),
        (last and goods[S_QUARRY] >= 1, S_QUARRY),
        (True, CARD0 + C_GRAIN),
        (True, S_GRAIN),
        (goods[S_FISH] >= 3, S_FISH),
        (goods[S_TIMBER] >= 3 or (p['wood'] < 2 and goods[S_TIMBER] >= 2), S_TIMBER),
        (goods[S_QUARRY] >= 2 or (p['stone'] < 1 and goods[S_QUARRY] >= 1), S_QUARRY),
    ] + [(True, x) for x in (S_FISH, S_TIMBER, S_QUARRY, S_TIMBER2, S_START, S_FARM)]
    for cond, space in picks:
        if cond and has(space):
            return space
    return legal[0]


def heuristic_external(lb):
    """External evaluator: one payload per request of an external LeafBatch."""
    return [json.dumps([heuristic_choice(json.loads(st))]) for st in lb.states]


class RandomExternal:
    """Uniformly random external player (seeded)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, lb):
        out = []
        for st in lb.states:
            legal = json.loads(st)['legal']
            out.append(json.dumps([self.rng.choice(legal)]))
        return out


class HeuristicBot:
    """The heuristic as a Python-driver bot (`play.PyDriver`): works on a `<package>._engine.Game`."""

    def __init__(self, rng=None):
        pass

    def choose(self, g):
        return heuristic_choice(json.loads(g.export_state()))
