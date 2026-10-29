"""The Python reference driver: many games side by side in one process, every network seat
searching with the reference MCTS (`mcts.py`) - or, for mixed parity matches, with the Rust
tree through the bindings - speaking the same leaf-batch protocol as the Rust driver
(`protocol.py`), so `infer.drive` runs either.

Per decision (docs/search.md 3.4): playout-cap randomisation (a `p_full` share of decisions
gets `sims` and becomes a training sample, the rest `sims_fast`), root Dirichlet noise on full
self-play searches, temperature sampling by visit count for the first `temp_moves` searched
decisions of a game, otherwise the seat's move rule (`select`: argmax (N, P0) by default, or the
completed-Q logits' argmax 'cq', docs/search.md 3.3), tree reuse when the same seat decides again,
the opt-in root minimum visits of full searches (`root_min_visits`, docs/search.md 3.5: the
visit-based outputs without the forced walks, the 'cq' ones with them). Recorded decisions carry
the seat's policy target: the visit distribution ('visits') or the completed-Q target ('cq',
docs/search.md 3.1). Root-view leaf values (`leaf_value_view` 'root', docs/search.md 3.6): another
seat's leaf adds a value-only row of the tree's viewer right after its own row in the batch; the
value-only samples (`view_samples`): a share of the recorded decisions (`view_pick`, the Rust
driver's hash) also record every other recording seat's view of the real state, no options, q
= that seat's next search's root value ('next_q') or z.
"""
import random

import numpy as np

from . import mcts
from .protocol import LeafBatch, MAX_SEATS, concat_samples


POLICY_TARGETS = ('visits', 'cq')
M64 = (1 << 64) - 1


def split64(x):
    """splitmix64's finaliser (the Rust `rng::split64`)."""
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def view_pick(seed, gid, decision, share):
    """Whether recorded decision `decision` (the game's searched decisions before it) of game `gid` of
    a driver seeded `seed` records its value-only samples at share `share` (the Rust
    `driver::view_pick`, the same arithmetic: a hash, no random stream)."""
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    h = split64((seed & M64) ^ split64((((gid & 0xFFFFFFFF) << 32) | (decision & 0xFFFFFFFF)) ^ 0x76696577))
    return (h >> 11) * (1.0 / (1 << 53)) < share


class NNSeat:
    """Search settings of a network seat. impl: 'py' (reference tree) | 'rs' (Rust tree).
    policy_target: training target of recorded decisions, 'visits' | 'cq' (completed Q with
    cq_c_visit, cq_c_scale). select, select_min_visits: the rule of the moves it does not sample
    ('visits' | 'cq'; docs/search.md 3.3) and its completed-Q transform select_c_visit,
    select_c_scale (None: the target's cq_c_visit, cq_c_scale). record: whether its full searches
    are recorded (False: a league opponent, docs/az.md 14; the Rust seat key `record`). root_min_visits,
    root_min_visits_max_actions: the root minimum visits of its full searches (0 = off; docs/search.md
    3.5). leaf_value_view: 'mover' | 'root' (whose view a leaf's value comes from), view_samples: the
    share of its recorded decisions that also record value-only samples (0 = off), view_sample_target:
    their q, 'next_q' | 'z' (docs/search.md 3.6)."""

    def __init__(self, key='m', sims=64, c_puct=1.0, selfplay=False, alpha=0.3, eps=0.25, temp_moves=20,
                 sims_fast=None, p_full=1.0, chance_k=0, margin_w=0.0, redet=True, impl='py',
                 policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0, select='visits', select_min_visits=0,
                 select_c_visit=None, select_c_scale=None, record=True, root_min_visits=0,
                 root_min_visits_max_actions=4, leaf_value_view='mover', view_samples=0.0,
                 view_sample_target='next_q'):
        if policy_target not in POLICY_TARGETS:
            raise ValueError(f'policy_target {policy_target!r}: one of {POLICY_TARGETS}')
        mcts.check_select(select, select_min_visits)
        mcts.check_select_transform(select, select_c_visit, select_c_scale)
        mcts.check_root_min(root_min_visits, root_min_visits_max_actions)
        mcts.check_leaf_view(leaf_value_view, view_samples, view_sample_target)
        self.key, self.sims, self.c_puct, self.selfplay = key, sims, c_puct, selfplay
        self.alpha, self.eps, self.temp_moves = alpha, eps, temp_moves
        self.sims_fast, self.p_full, self.chance_k, self.margin_w = sims_fast, p_full, chance_k, margin_w
        self.redet, self.impl = redet, impl
        self.policy_target, self.cq_c_visit, self.cq_c_scale = policy_target, cq_c_visit, cq_c_scale
        self.select, self.select_min_visits = select, select_min_visits
        self.select_c_visit, self.select_c_scale = select_c_visit, select_c_scale
        self.record = record
        self.root_min_visits, self.root_min_visits_max_actions = root_min_visits, root_min_visits_max_actions
        self.leaf_value_view, self.view_samples = leaf_value_view, float(view_samples)
        self.view_sample_target = view_sample_target

    @property
    def select_transform(self):
        """The rule's completed-Q transform (c_visit, c_scale): its own, else the target's."""
        return mcts.select_transform(self.cq_c_visit, self.cq_c_scale, self.select_c_visit, self.select_c_scale)

    @property
    def select_tag(self):
        return mcts.select_tag(self.select, self.select_min_visits, *self.select_transform)


class PySearch:
    def __init__(self, game, seat, rng):
        self.t = mcts.Tree(game, seat.redet, rng)
        self.pending = None

    def set_base(self, game, rng):
        self.t.set_base(game, rng)

    @property
    def expanded(self):
        return self.t.root.P is not None

    @property
    def redet(self):
        return self.t.redet

    def simulate(self, game, rng, seat):
        """-> None (a terminal, backed up) or (decider, features, options, the viewer's features when
        the leaf takes a root-view value, else None)."""
        r = mcts.simulate(self.t, rng, seat.c_puct, seat.chance_k, seat.margin_w)
        if r is None:
            return None
        node, leaf, path = r
        v = self.t.viewer
        vx = leaf.encode(v) if seat.leaf_value_view == 'root' and node.player != v else None
        self.pending = (node, path, None if vx is None else v)
        return node.player, leaf.encode(node.player), node.acts, vx

    def expand(self, priors, value, margin, seat):
        node, path, vp = self.pending
        self.pending = None
        mcts.expand_and_backup(node, path, priors, value, margin, self.t.n, seat.margin_w, self.t.team, self.t,
                               value_player=vp)
        return node is self.t.root

    def add_noise(self, seat, np_rng):
        mcts.add_noise(self.t.root, seat.alpha, seat.eps, np_rng)

    def set_root_min(self, k):
        self.t.set_root_min(k)

    def stats(self):
        """acts, visit counts, P0, root win vector - without the forced walks (root minimum visits)."""
        r = self.t.root
        if self.t.root_min:
            return r.acts, mcts.root_visits_pruned(self.t), r.P0, mcts.root_win_pruned(self.t)
        return r.acts, r.Nv, r.P0, mcts.root_win(r)

    def target(self, seat):
        """The seat's policy target at the root (the visits target without the forced walks)."""
        if seat.policy_target == 'cq':
            return mcts.cq_target(self.t.root, seat.cq_c_visit, seat.cq_c_scale)
        if self.t.root_min:
            nv = mcts.root_visits_pruned(self.t)
            tot = sum(nv)
            return [n / tot for n in nv] if tot else list(self.t.root.P0)
        return mcts.visits_target(self.t.root)

    def choose(self, seat):
        """The root option the seat's move rule takes (the reference formula, `mcts.select_node`)."""
        return mcts.select_node(self.t.root, seat.select, seat.select_min_visits, *seat.select_transform)

    def advance(self, a, key, game):
        child = mcts.advance(self.t.root, a, key, game.to_move, game.legal())
        if child is None:
            return False
        self.t.root = child
        self.t.set_root_min(0)
        return True


class RsSearch:
    """The Rust tree (`<package>._engine.SearchTree`) behind the same interface."""

    def __init__(self, game, seat, rng):
        self.t = game.search_tree(seat.redet, rng.getrandbits(63))

    def set_base(self, game, rng):
        self.t.set_base(game)

    @property
    def expanded(self):
        return self.t.root_expanded

    @property
    def redet(self):
        return self.t.redet

    def simulate(self, game, rng, seat):
        if seat.leaf_value_view == 'root':
            return self.t.simulate(game, seat.c_puct, seat.chance_k, seat.margin_w, root_view=True)
        r = self.t.simulate(game, seat.c_puct, seat.chance_k, seat.margin_w)
        if r is None:
            return None
        return r[0], r[1], r[2], None

    def expand(self, priors, value, margin, seat):
        was = self.t.root_expanded
        self.t.expand(list(map(float, priors)), list(map(float, value)), list(map(float, margin)), seat.margin_w)
        return not was

    def add_noise(self, seat, np_rng):
        self.t.add_noise(seat.alpha, seat.eps, int(np_rng.integers(1 << 62)))

    def set_root_min(self, k):
        self.t.set_root_min(k)

    def stats(self):
        if self.t.root_min:
            return self.t.root_acts, self.t.root_visits_pruned, self.t.root_priors0, self.t.root_win_pruned()
        return self.t.root_acts, self.t.root_visits, self.t.root_priors0, self.t.root_win()

    def target(self, seat):
        if seat.policy_target == 'cq':
            return self.t.root_cq_target(seat.cq_c_visit, seat.cq_c_scale)
        nv = self.t.root_visits_pruned if self.t.root_min else self.t.root_visits
        tot = sum(nv)
        return [n / tot for n in nv] if tot else list(self.t.root_priors0)

    def choose(self, seat):
        """The root option the seat's move rule takes: the Rust driver's own code (`root_select`)."""
        return self.t.root_select(seat.select, seat.select_min_visits, *seat.select_transform)

    def advance(self, a, key, game):
        if game.over:
            return False
        return self.t.advance(a, list(key), game)


SEARCH = {'py': PySearch, 'rs': RsSearch}


class RandomBot:
    def __init__(self, rng):
        self.rng = rng

    def choose(self, g):
        return self.rng.choice(g.legal())


def make_bot(game, name, rng):
    if name == 'random':
        return RandomBot(rng)
    if game == 'toy' and name == 'heuristic':
        from .toybots import HeuristicBot
        return HeuristicBot(rng)
    raise ValueError(f'the Python driver has no bot {name!r} for {game}')


class Slot:
    __slots__ = ('gid', 'rng', 'np_rng', 'search_rng', 'game', 'n', 'seats', 'bots', 'trees', 'sims', 'target',
                 'full', 'records', 'decisions', 'steps', 'view_wait')

    def __init__(self, gid, game, spec, seats, rng):
        from .games import engine
        self.gid = gid
        self.rng = rng
        self.np_rng = np.random.default_rng(rng.randrange(1 << 30))
        self.search_rng = random.Random(rng.randrange(1 << 30))
        self.game = engine().Game(game, spec, len(seats), rng.getrandbits(63))
        self.n = len(seats)
        self.seats = seats
        self.bots = {i: make_bot(game, s, random.Random(rng.randrange(1 << 30)))
                     for i, s in enumerate(seats) if isinstance(s, str)}
        self.trees = {}
        self.sims = 0
        self.target = None
        self.full = True
        self.records = []
        self.decisions = 0
        self.steps = 0
        self.view_wait = []                 # value-only records waiting for (record index, observer)'s next search


class PyDriver:
    """seat_fn(game_index) -> list of seats (NNSeat or a bot name), one per player."""

    def __init__(self, game, n_games, seat_fn, spec=None, concurrency=64, seed=0, record=False, max_steps=100_000):
        from .games import spec_json
        self.game = game
        self.seed = seed
        self.spec = spec_json(spec)
        self.n_games = n_games
        self.seat_fn = seat_fn
        self.concurrency = concurrency
        self.rng = random.Random(seed)
        self.record = record
        self.max_steps = max_steps
        self.active = []
        self.next_gid = 0
        self.queue = []
        self.outstanding = {}
        self.next_bid = 0
        self.results = []
        self.samples = []
        self.sims = 0
        self._refill()

    # ---- protocol ---------------------------------------------------------------------------
    @property
    def finished(self):
        return not self.active and not self.queue and not self.outstanding

    def next_batch(self):
        while not self.queue:
            if self.outstanding:
                raise RuntimeError('submit the outstanding batches before asking for more')
            if not self.active:
                return None
            self._round()
        return self.queue.pop(0)

    def submit(self, bid, priors, value, margin):
        lb, items = self.outstanding.pop(bid)
        off = lb.opt_off
        for (slot, seat, tree, b, vb) in items:
            # (a root-view leaf: the value of its value-only row vb, the priors of its own row b)
            is_root = tree.expand(priors[off[b]:off[b + 1]].tolist(), value[vb], margin[vb], seat)
            if is_root and seat.selfplay and slot.full:
                tree.add_noise(seat, slot.np_rng)
            slot.sims += 1
        self.sims += len(items)

    def take_results(self):
        r, self.results = self.results, []
        return r

    def take_samples(self):
        s, self.samples = self.samples, []
        return concat_samples(s)

    # ---- internals --------------------------------------------------------------------------
    def _refill(self):
        while len(self.active) < self.concurrency and self.next_gid < self.n_games:
            seats = self.seat_fn(self.next_gid)
            self.active.append(Slot(self.next_gid, self.game, self.spec, seats, random.Random(self.rng.randrange(1 << 30))))
            self.next_gid += 1

    def _round(self):
        need = {}
        finished = []
        for slot in self.active:
            while True:
                g = slot.game
                if g.over or slot.steps >= self.max_steps:
                    finished.append(slot)
                    break
                legal = g.legal()
                if len(legal) == 1:
                    self._apply(slot, legal[0])
                    continue
                p = g.to_move
                seat = slot.seats[p]
                if not isinstance(seat, NNSeat):
                    self._apply(slot, slot.bots[p].choose(g))
                    continue
                if slot.target is None:
                    self._start(slot, seat, p)
                tree = slot.trees[p]
                if slot.sims >= slot.target:
                    self._decide(slot, seat, tree, p)
                    continue
                r = tree.simulate(g, slot.search_rng, seat)
                if r is None:
                    slot.sims += 1
                    continue
                player, x, acts, vx = r
                need.setdefault(seat.key, []).append((slot, seat, tree, player, x, acts, vx))
                break
        for slot in finished:
            self.active.remove(slot)
            self._finish(slot)
        self._refill()
        for key, items in need.items():
            # rows: each leaf's, then (root-view values) its value-only row of the tree's viewer
            rows, refs = [], []
            for it in items:
                slot, seat, tree, player, x, acts, vx = it
                b = len(rows)
                rows.append((x, acts, player, slot))
                if vx is not None:
                    rows.append((vx, [], tree.t.viewer, slot))
                refs.append((slot, seat, tree, b, len(rows) - 1))
            off = np.zeros(len(rows) + 1, np.int32)
            off[1:] = np.cumsum([len(r[1]) for r in rows])
            a = {'x': np.ascontiguousarray(np.stack([r[0] for r in rows]).astype(np.float32)),
                 'opt_off': off,
                 'opt_action': np.asarray([x for r in rows for x in r[1]], np.int32),
                 'player': np.asarray([r[2] for r in rows], np.int32),
                 'n_players': np.asarray([r[3].n for r in rows], np.int32),
                 'gid': np.asarray([r[3].gid for r in rows], np.int64)}
            lb = LeafBatch(self.next_bid, key, 'nn', a)
            self.outstanding[self.next_bid] = (lb, refs)
            self.next_bid += 1
            self.queue.append(lb)

    def _start(self, slot, seat, p):
        slot.full = seat.sims_fast is None or slot.rng.random() < seat.p_full
        slot.target = seat.sims if slot.full else seat.sims_fast
        tree = slot.trees.get(p)
        if tree is None:
            tree = slot.trees[p] = SEARCH[seat.impl](slot.game, seat, slot.search_rng)
        else:
            tree.set_base(slot.game, slot.search_rng)
            if seat.selfplay and slot.full and tree.expanded:
                tree.add_noise(seat, slot.np_rng)
        # the root minimum visits of this search (docs/search.md 3.5; 0 = off)
        tree.set_root_min(mcts.root_floor(seat.root_min_visits, seat.root_min_visits_max_actions, slot.full,
                                          len(slot.game.legal()), slot.target))

    def _decide(self, slot, seat, tree, p):
        acts, Nv, P0, rw = tree.stats()
        if slot.view_wait:
            # value-only records waiting for this seat's next search take its root value ('next_q')
            keep = []
            for i, o in slot.view_wait:
                if o == p:
                    r = slot.records[i]
                    slot.records[i] = r[:4] + (None if rw is None else list(rw),)
                else:
                    keep.append((i, o))
            slot.view_wait = keep
        tot = sum(Nv)
        if seat.selfplay and slot.decisions < seat.temp_moves and tot > 0:
            r = slot.rng.random() * tot
            i = len(Nv) - 1
            for k, n in enumerate(Nv):
                r -= n
                if r < 0:
                    i = k
                    break
        elif seat.select == 'visits':
            i = max(range(len(Nv)), key=lambda k: (Nv[k], P0[k]))
        else:
            i = tree.choose(seat)
        if self.record and seat.selfplay and seat.record and slot.full:
            pi = tree.target(seat)
            slot.records.append((slot.game.encode(p), list(acts), pi, p, rw))
            if seat.view_samples > 0 and view_pick(self.seed, slot.gid, slot.decisions, seat.view_samples):
                for o in range(slot.n):
                    so = slot.seats[o]
                    if o == p or not (isinstance(so, NNSeat) and so.selfplay and so.record):
                        continue
                    if seat.view_sample_target == 'next_q':
                        slot.view_wait.append((len(slot.records), o))
                    slot.records.append((slot.game.encode(o), [], [], o, None))
        slot.decisions += 1
        self._apply(slot, acts[i])

    def _apply(self, slot, a):
        g = slot.game
        p = g.to_move
        key, legal = mcts.transition(g, a)
        slot.steps += 1
        for q in list(slot.trees):
            tree = slot.trees[q]
            if not (q == p and not g.over and g.to_move == p and tree.redet and tree.advance(a, key, g)):
                del slot.trees[q]
        slot.sims = 0
        slot.target = None

    def _finish(self, slot):
        g = slot.game
        n = slot.n
        over = g.over
        if over and g.deferred:
            raise RuntimeError(f'game {slot.gid}: a deferred terminal (the Python reference driver does not park games; '
                               f'the Rust driver does, docs/search.md 2.8)')
        res = list(g.result()) if over else [1.0 / n] * n
        self.results.append({'gid': slot.gid, 'n': n, 'over': over, 'result': res, 'scores': g.scores(),
                             'steps': slot.steps, 'decisions': slot.decisions, 'ext_moves': 0,
                             'rounds': g.rounds, 'samples': len(slot.records) if self.record and over else 0})
        if self.record and over and slot.records:
            margins = [g.margin(i) for i in range(n)]
            self.samples.append(_sample_arrays(slot.records, res, margins, n))


def _sample_arrays(records, res, margins, n):
    N = len(records)
    z = np.zeros((N, MAX_SEATS), np.float32)
    q = np.zeros((N, MAX_SEATS), np.float32)
    sc = np.zeros((N, MAX_SEATS), np.float32)
    off = np.zeros(N + 1, np.int64)
    acts, pis = [], []
    for k, (_, a, pi, mover, rq) in enumerate(records):
        acts.extend(a)
        pis.extend(pi)
        off[k + 1] = len(acts)
        for j in range(n):
            s = (mover + j) % n
            z[k, j] = res[s]
            q[k, j] = rq[s] if rq is not None else res[s]
            sc[k, j] = margins[s]
    return {'x': np.stack([r[0] for r in records]).astype(np.float32),
            'player': np.asarray([r[3] for r in records], np.int32),
            'n_players': np.full(N, n, np.int32), 'opt_off': off,
            'opt_action': np.asarray(acts, np.int32), 'pi': np.asarray(pis, np.float32),
            'z': z, 'q': q, 'sc': sc, 'net': np.zeros(N, np.uint32)}
