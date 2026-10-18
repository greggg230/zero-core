"""Live play: the network deciding in one real game (a web app's opponent), not self-play.

One game has no other games to batch its leaves with, so the search batches within its own
tree: `SearchTree.simulate_batch` walks up to `batch` simulations with virtual loss on their
paths (the walks of one batch spread out instead of all reaching the same leaf), the distinct
leaves go through the network in one call, `expand_batch` backs up their real values. batch 1
is exactly the sequential search; 16-32 cost little strength at ~1000 simulations and cut a
move from seconds (one leaf per network call) to a fraction of a second.

  move = search_move(game, evaluator, sims=1024, batch=16, c_puct=1.0, chance_k=2, margin_w=0.2)
  moves = play_turn(game, evaluator, ...)     # the seat's whole turn, as external-format moves

`evaluator` = an `infer.NetEvaluator` (or anything with its `evaluate(x, n_players, opt_off,
opt_action)`). The game must be the real game of the seat to move (hidden information is
determinized away by the search, `redet`).
"""
import time

import numpy as np

from . import mcts


class Searcher:
    """Batched MCTS settings of a live seat (a run's evaluation seat: c_puct, chance_k,
    margin_w, redet; sims / batch / vloss per move)."""

    def __init__(self, evaluator, sims=1024, batch=16, c_puct=1.0, chance_k=0, margin_w=0.0, vloss=0.0, redet=True,
                 seed=0):
        self.ev, self.sims, self.batch = evaluator, int(sims), max(1, int(batch))
        self.c_puct, self.chance_k, self.margin_w, self.vloss, self.redet = c_puct, chance_k, margin_w, vloss, redet
        self.rng = np.random.default_rng(seed)

    def evaluate(self, leaves, n_players=2):
        """[(decider, features, legal actions)] -> lists of priors, value and margin rows."""
        x = np.stack([lf[1] for lf in leaves]).astype(np.float32, copy=False)
        lens = [len(lf[2]) for lf in leaves]
        off = np.zeros(len(leaves) + 1, np.int32)
        off[1:] = np.cumsum(lens)
        acts = np.fromiter((a for lf in leaves for a in lf[2]), np.int32, count=int(off[-1]))
        pri, v, m = self.ev.evaluate(x, np.full(len(leaves), n_players, np.int32), off, acts)
        return ([pri[off[b]:off[b + 1]].tolist() for b in range(len(leaves))], v.tolist(), m.tolist())

    def search(self, game, tree=None):
        """`sims` simulations for the pending decision of `game` -> (tree, stats). `tree`: a
        tree of the same seat advanced to this decision (reused: its visits count toward
        nothing, the new simulations are added)."""
        if tree is None:
            tree = game.search_tree(self.redet, int(self.rng.integers(1 << 62)))
        else:
            tree.set_base(game)
        done = calls = leaves_n = 0
        t0 = time.perf_counter()
        while done < self.sims:
            leaves, d = tree.simulate_batch(game, min(self.batch, self.sims - done), self.c_puct, self.chance_k,
                                            self.margin_w, self.vloss)
            done += d
            if leaves:
                p, v, m = self.evaluate(leaves, game.n_players)
                tree.expand_batch(p, v, m, self.margin_w)
                calls += 1
                leaves_n += len(leaves)
        return tree, {'sims': done, 'net_calls': calls, 'leaves': leaves_n, 'seconds': time.perf_counter() - t0}


def best_action(tree):
    """The most visited root option (ties: the higher prior), as the runs' evaluation seats."""
    acts, nv, p0 = tree.root_acts, tree.root_visits, tree.root_priors0
    i = max(range(len(nv)), key=lambda k: (nv[k], p0[k]))
    return acts[i]


def search_move(game, searcher):
    """-> (action, tree, stats) for the pending decision of `game` (more than one option)."""
    tree, st = searcher.search(game)
    return best_action(tree), tree, st


def play_turn(game, searcher, max_steps=200):
    """Play the seat to move until another seat is to move or the game is over, on `game`
    itself. -> (external-format moves (JSON strings) in order, per-decision stats). Single-
    option decisions are taken without a search; the tree is reused between the seat's
    decisions (advance)."""
    seat = game.to_move
    moves, stats, tree = [], [], None
    for _ in range(max_steps):
        if game.over or game.to_move != seat:
            break
        legal = game.legal()
        if len(legal) == 1:
            a = legal[0]
            tree = None
        else:
            tree, st = searcher.search(game, tree)
            a = best_action(tree)
            rw = tree.root_win()
            st['options'] = len(legal)
            st['win'] = None if rw is None else float(rw[seat])      # the search's P(win) of the seat
            stats.append(st)
        keys, done = game.step_external(a)
        moves.extend(done)
        if tree is not None and not game.over and game.to_move == seat:
            if not tree.advance(a, list(keys), game):
                tree = None
        else:
            tree = None
    return moves, stats


__all__ = ['Searcher', 'best_action', 'search_move', 'play_turn', 'mcts']
