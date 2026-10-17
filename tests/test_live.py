"""zero.live: batched single-game search (virtual loss) and whole-turn play."""
import random

import numpy as np

from zero import live, mcts
from zero.games import engine


class HashNet:
    """Deterministic stand-in network: every output a function of the leaf row only."""

    def __init__(self):
        self.calls = 0
        self.rows = 0

    def evaluate(self, x, n_players, opt_off, opt_action):
        self.calls += 1
        self.rows += x.shape[0]
        B = x.shape[0]
        raw = 1.0 + ((opt_action.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(28)).astype(np.float64) % 9
        pri = np.zeros_like(raw, dtype=np.float32)
        value = np.zeros((B, 8), np.float32)
        margin = np.zeros((B, 8), np.float32)
        w = np.arange(x.shape[1]) % 7 - 3.0
        for b in range(B):
            s, e = opt_off[b], opt_off[b + 1]
            pri[s:e] = raw[s:e] / raw[s:e].sum()
            v = 1.0 / (1.0 + np.exp(-(x[b] @ w) / 8.0))
            value[b, 0], value[b, 1] = v, 1.0 - v
        return pri, value, margin


def mid_game(seed, steps=5):
    g = engine().Game('toy', None, 2, seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if g.over:
            break
        mcts.transition(g, rng.choice(g.legal()))
    while not g.over and len(g.legal()) == 1:
        mcts.transition(g, g.legal()[0])
    return g


def test_batch_of_one_is_the_sequential_search():
    for seed in range(1, 5):
        g = mid_game(seed)
        if g.over:
            continue
        s = live.Searcher(HashNet(), sims=200, batch=1, c_puct=1.0, chance_k=2, seed=seed)
        tree_seed = int(np.random.default_rng(seed).integers(1 << 62))
        tb, st = s.search(g)
        ta = g.search_tree(True, tree_seed)
        net = HashNet()
        for _ in range(200):
            r = ta.simulate(g, 1.0, 2, 0.0)
            if r is None:
                continue
            off = np.array([0, len(r[2])], np.int32)
            p, v, m = net.evaluate(np.asarray(r[1], np.float32)[None], np.array([2], np.int32), off,
                                   np.asarray(r[2], np.int32))
            ta.expand(p.tolist(), v[0].tolist(), m[0].tolist(), 0.0)
        assert ta.root_visits == tb.root_visits, seed
        assert st['sims'] == 200


def test_batches_cut_network_calls_and_keep_the_visit_count():
    g = mid_game(3)
    net = HashNet()
    s = live.Searcher(net, sims=512, batch=16, c_puct=1.0, chance_k=2, vloss=0.0, seed=1)
    tree, st = s.search(g)
    assert st['sims'] == 512
    assert sum(tree.root_visits) == tree.root_tot == 511          # every simulation but the root's expansion
    assert net.calls < 512 / 4, net.calls                         # batches of several leaves
    q = [w / n for w, n in zip(tree.root_values, tree.root_visits) if n]
    assert all(-1e-9 <= x <= 1 + 1e-9 for x in q)                 # no virtual loss left
    assert live.best_action(tree) in g.legal()


def test_play_turn_plays_the_seat_until_another_seat_moves():
    g = mid_game(4)
    seat = g.to_move
    moves, stats = live.play_turn(g, live.Searcher(HashNet(), sims=64, batch=8, seed=2))
    assert moves == []                                            # the toy game has no external moves
    assert g.over or g.to_move != seat
    assert all(st['sims'] == 64 for st in stats)
