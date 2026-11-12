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
= that seat's next search's root value ('next_q') or z. The root's belief (`belief_tilt` > 0,
docs/search.md 3.8): a search that takes it (full searches; every one with `belief_fast`) sends one
`ROW_BELIEF` row of the root's features - after the fresh root's own leaf row, or alone (a reused
root: its simulations wait) - and deals its worlds by the batch's auxiliary answer for it (`submit`'s
`aux`) from then on.
= that seat's next search's root value ('next_q') or z. The Gumbel root search (`root_search`
'gumbel', docs/search.md 3.7): every search of the seat has Gumbel noise (a self-play seat's first
temp_moves decisions; an evaluation seat's at `gumbel_scale`, every decision) and sequential halving
at the root instead of Dirichlet noise, temperature and the move rule; its move is the halving's
winner, its root value the improved policy's (`mcts.gumbel_choose`, `gumbel_q`).

Variable-length rows (a game with tail segments or pointer overrides, docs/search.md 2.15; `mcts.rows_of`): every
row - a leaf's own, a value-only row, a root belief row, a record - is `encode_row` of its state with its tail and,
for a row with options, its overrides against them (`mcts.leaf_row`; the Rust tree's seats: `SearchTree.simulate(...,
rows=True)`, the same rows); batches and samples carry them as the Rust driver's (`t_off` / `t_seg` / `t_slot` /
`t_val`, `ov_off` / `ov_opt` / `ov_tok`; `row_arrays`), and a row past the game's `max_row_tokens` aborts its game
(no samples, `abort_reason` 'max_row_tokens', counted in `aborted_row_tokens`, one stderr line) - never a truncated
row. A game without them: every path of before.
"""
import random
import sys

import numpy as np

from . import mcts
from .protocol import LeafBatch, MAX_SEATS, ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows, concat_samples


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
    their q, 'next_q' | 'z' (docs/search.md 3.6). belief_tilt (0 = off), belief_clip (None: 20),
    belief_mix, belief_fast: the root's belief of its searches (docs/search.md 3.8). root_search: 'puct' |
    'gumbel' (docs/search.md 3.7), gumbel_m the options sequential halving considers, gumbel_scale its
    noise's scale, gumbel_c_visit / gumbel_c_scale sigma's transform (None: cq_c_visit / cq_c_scale).
    cq_prior_w: beta, the prior's exponent in the recorded 'cq' target (docs/search.md 3.13; 1 = off).
    infoset_merge: information-set merging (docs/search.md 3.16): the Rust tree only (impl 'rs'; the reference
    tree has no merged mode), root-view leaf values."""

    def __init__(self, key='m', sims=64, c_puct=1.0, selfplay=False, alpha=0.3, eps=0.25, temp_moves=20,
                 sims_fast=None, p_full=1.0, chance_k=0, margin_w=0.0, redet=True, impl='py',
                 policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0, select='visits', select_min_visits=0,
                 select_c_visit=None, select_c_scale=None, record=True, root_min_visits=0,
                 root_min_visits_max_actions=4, leaf_value_view='mover', view_samples=0.0,
                 view_sample_target='next_q', belief_tilt=0.0, belief_clip=None, belief_mix=0.0, belief_fast=False,
                 belief_tilts=None, root_search='puct', gumbel_m=16, gumbel_scale=1.0, gumbel_c_visit=None, gumbel_c_scale=None,
                 cq_prior_w=1.0, infoset_merge=False):
        if policy_target not in POLICY_TARGETS:
            raise ValueError(f'policy_target {policy_target!r}: one of {POLICY_TARGETS}')
        mcts.check_cq_prior_w(cq_prior_w, policy_target)
        mcts.check_select(select, select_min_visits)
        mcts.check_select_transform(select, select_c_visit, select_c_scale)
        mcts.check_root_min(root_min_visits, root_min_visits_max_actions)
        mcts.check_leaf_view(leaf_value_view, view_samples, view_sample_target)
        mcts.check_belief(belief_tilt, belief_clip, belief_mix)
        mcts.check_belief_tilts(belief_tilts)
        mcts.check_root_search(root_search, gumbel_m, gumbel_scale, gumbel_c_visit, gumbel_c_scale)
        if root_search == 'gumbel' and root_min_visits:
            raise ValueError("root_min_visits with root_search 'gumbel': exclusive (docs/search.md 3.7)")
        mcts.check_merge(infoset_merge, leaf_value_view, selfplay)
        if infoset_merge and impl != 'rs':
            raise ValueError("infoset_merge: the Rust tree only (impl 'rs'): the Python reference tree has no merged "
                             "mode (docs/search.md 3.16)")
        self.infoset_merge = bool(infoset_merge)
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
        self.belief_tilt, self.belief_mix, self.belief_fast = float(belief_tilt), float(belief_mix), bool(belief_fast)
        self.belief_clip = float(mcts.BELIEF_CLIP if belief_clip is None else belief_clip)
        # per-target tilts {target: alpha > 0} (docs/search.md 3.8.1; {} = none)
        self.belief_tilts = {k: float(v) for k, v in sorted((belief_tilts or {}).items()) if v}
        self.root_search, self.gumbel_m, self.gumbel_scale = root_search, int(gumbel_m), float(gumbel_scale)
        self.gumbel_c_visit = float(cq_c_visit if gumbel_c_visit is None else gumbel_c_visit)
        self.gumbel_c_scale = float(cq_c_scale if gumbel_c_scale is None else gumbel_c_scale)
        self.cq_prior_w = float(cq_prior_w)

    def takes_belief(self, full):
        """Whether its search of a decision takes the root's belief (`full`: a full search)."""
        return (self.belief_tilt > 0 or bool(self.belief_tilts)) and (full or self.belief_fast)

    @property
    def gumbel(self):
        return self.root_search == 'gumbel'

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

    def simulate(self, game, rng, seat, rk=None):
        """-> None (a terminal, backed up) or (decider, features, options, the viewer's features when
        the leaf takes a root-view value, else None, rows): `rows` None for a game without variable-length rows
        (`rk` None), else (tail, ov, the value-only row's tail), `mcts.leaf_row`'s (docs/search.md 2.15)."""
        r = mcts.simulate(self.t, rng, seat.c_puct, seat.chance_k, seat.margin_w)
        if r is None:
            return None
        node, leaf, path = r
        v = self.t.viewer
        if rk is None:
            vx = leaf.encode(v) if seat.leaf_value_view == 'root' and node.player != v else None
            self.pending = (node, path, None if vx is None else v)
            return node.player, leaf.encode(node.player), node.acts, vx, None
        vrow = seat.leaf_value_view == 'root' and node.player != v
        vx, vt, _ = mcts.leaf_row(leaf, v, rk) if vrow else (None, None, None)
        self.pending = (node, path, v if vrow else None)
        x, t, ov = mcts.leaf_row(leaf, node.player, rk, opts=bool(node.acts))
        return node.player, x, node.acts, vx, (t, ov, vt)

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

    @property
    def has_belief(self):
        return self.t.belief is not None

    def set_belief(self, pred, seat, game, rng):
        b = (np.asarray(pred, np.float32), seat.belief_tilt, seat.belief_clip)
        self.t.set_belief(b + ((seat.belief_tilts,) if seat.belief_tilts else ()), seat.belief_mix, game, rng)

    def set_gumbel(self, m, budget, noise, c_visit, c_scale):
        self.t.set_gumbel(m, budget, noise, c_visit, c_scale)

    def gumbel_choose(self):
        return mcts.gumbel_choose(self.t)

    def stats(self):
        """acts, visit counts, P0, root win vector - without the forced walks (root minimum visits); a
        Gumbel search's root value is its own (`mcts.gumbel_q`)."""
        r = self.t.root
        if self.t.gumbel is not None:
            return r.acts, r.Nv, r.P0, mcts.gumbel_q(self.t)
        if self.t.root_min:
            return r.acts, mcts.root_visits_pruned(self.t), r.P0, mcts.root_win_pruned(self.t)
        return r.acts, r.Nv, r.P0, mcts.root_win(r)

    def target(self, seat):
        """The seat's policy target at the root (the visits target without the forced walks)."""
        if seat.policy_target == 'cq':
            return mcts.cq_target(self.t.root, seat.cq_c_visit, seat.cq_c_scale, seat.cq_prior_w)
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
        self.t.belief, self.t.belief_mix = None, 0.0          # (a belief is one decision's)
        self.t.gumbel = None
        return True


class RsSearch:
    """The Rust tree (`<package>._engine.SearchTree`) behind the same interface."""

    def __init__(self, game, seat, rng):
        self.t = game.search_tree(seat.redet, rng.getrandbits(63))
        if getattr(seat, 'infoset_merge', False):
            self.t.set_merge(True)                 # (docs/search.md 3.16: a fresh tree)

    def set_base(self, game, rng):
        self.t.set_base(game)

    @property
    def expanded(self):
        return self.t.root_expanded

    @property
    def redet(self):
        return self.t.redet

    def simulate(self, game, rng, seat, rk=None):
        if rk is not None:
            # (variable-length rows, docs/search.md 2.15: the tree's leaf rows with their tails and overrides)
            return self.t.simulate(game, seat.c_puct, seat.chance_k, seat.margin_w,
                                   root_view=seat.leaf_value_view == 'root', rows=True)
        if seat.leaf_value_view == 'root':
            r = self.t.simulate(game, seat.c_puct, seat.chance_k, seat.margin_w, root_view=True)
            return None if r is None else tuple(r) + (None,)
        r = self.t.simulate(game, seat.c_puct, seat.chance_k, seat.margin_w)
        if r is None:
            return None
        return r[0], r[1], r[2], None, None

    def expand(self, priors, value, margin, seat):
        was = self.t.root_expanded
        self.t.expand(list(map(float, priors)), list(map(float, value)), list(map(float, margin)), seat.margin_w)
        return not was

    def add_noise(self, seat, np_rng):
        self.t.add_noise(seat.alpha, seat.eps, int(np_rng.integers(1 << 62)))

    def set_root_min(self, k):
        self.t.set_root_min(k)

    @property
    def has_belief(self):
        return self.t.belief is not None

    def set_belief(self, pred, seat, game, rng):
        self.t.set_belief(np.asarray(pred, np.float32), game, seat.belief_tilt, seat.belief_clip, seat.belief_mix,
                          seat.belief_tilts or None)

    def set_gumbel(self, m, budget, noise, c_visit, c_scale):
        self.t.set_gumbel(m, budget, [float(x) for x in noise], c_visit, c_scale)

    def gumbel_choose(self):
        return self.t.gumbel_choose()

    def stats(self):
        if self.t.gumbel_on:
            return self.t.root_acts, self.t.root_visits, self.t.root_priors0, self.t.gumbel_q()
        if self.t.root_min:
            return self.t.root_acts, self.t.root_visits_pruned, self.t.root_priors0, self.t.root_win_pruned()
        return self.t.root_acts, self.t.root_visits, self.t.root_priors0, self.t.root_win()

    def target(self, seat):
        if seat.policy_target == 'cq':
            return self.t.root_cq_target(seat.cq_c_visit, seat.cq_c_scale, seat.cq_prior_w)
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
                 'full', 'records', 'decisions', 'steps', 'view_wait', 'belief', 'abort', 'chain')

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
        self.belief = False                 # the current search takes the root's belief
        self.abort = None                   # the driver aborted the game ('max_row_tokens', docs/search.md 2.15)
        # a chained game's contest so far (docs/search.md 2.14): its episodes, their samples, its counts, its result
        self.chain = {'episodes': [], 'samples': [], 'steps': 0, 'decisions': 0, 'result': None} if getattr(
            self.game, 'chained', False) else None


class PyDriver:
    """seat_fn(game_index) -> list of seats (NNSeat or a bot name), one per player."""

    def __init__(self, game, n_games, seat_fn, spec=None, concurrency=64, seed=0, record=False, max_steps=100_000):
        from . import package
        from .games import info, spec_json
        self.game = game
        inf = info(game)
        # the game's variable-length rows (docs/search.md 2.15; None: none, every path of before), checked against
        # the package's layout as the Rust driver's (R15)
        self.rk = mcts.rows_of(inf)
        package.check_rows(game, inf)
        self.aborted_row_tokens = 0         # games aborted by the row limit
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
        self.belief_rows = 0                # root belief rows sent (docs/search.md 3.8)
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

    def submit(self, bid, priors, value, margin, aux=None):
        """A network batch's answer; `aux` [belief rows, aux width]: the auxiliary outputs of its root
        belief rows (required when it has some: the Rust driver's `submit_aux`)."""
        lb, items = self.outstanding[bid]
        brows = belief_rows(lb.extra)
        if len(brows):
            from .games import info
            w = info(self.game)['aux_width']
            if aux is None or np.shape(aux) != (len(brows), w):
                raise ValueError(f'batch {bid} has {len(brows)} root belief rows (ROW_BELIEF: a seat\'s belief_tilt > 0): '
                                 f'its evaluator must answer their auxiliary outputs ([{len(brows)}, {w}], got '
                                 f'{None if aux is None else np.shape(aux)}; a network without the auxiliary head cannot '
                                 f'serve belief_tilt)')
            aux = np.asarray(aux, np.float32)
            if not np.all(np.isfinite(aux)) or np.any(aux < 0):
                raise ValueError(f'batch {bid}: a root belief row\'s auxiliary output is not finite and >= 0')
        elif aux is not None and len(aux):
            raise ValueError(f'batch {bid} has no root belief rows: {len(aux)} auxiliary rows given')
        del self.outstanding[bid]
        at = {int(r): k for k, r in enumerate(brows)}
        off = lb.opt_off
        n = 0
        for (slot, seat, tree, b, vb, bb) in items:
            if b is not None:
                # (a root-view leaf: the value of its value-only row vb, the priors of its own row b)
                is_root = tree.expand(priors[off[b]:off[b + 1]].tolist(), value[vb], margin[vb], seat)
                if is_root and seat.selfplay and slot.full and not seat.gumbel:
                    tree.add_noise(seat, slot.np_rng)
                slot.sims += 1
                n += 1
            if bb is not None:
                # the search's root belief (docs/search.md 3.8): its worlds are dealt by it from now on
                tree.set_belief(aux[at[bb]], seat, slot.game, slot.search_rng)
        self.sims += n

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
                if g.over or slot.steps >= self.max_steps or slot.abort:
                    if self._episode_end(slot):
                        continue                    # (a chained game's next episode, docs/search.md 2.14)
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
                if self._simulate(slot, seat, tree, p, g, need):
                    break
        for slot in finished:
            self.active.remove(slot)
            self._finish(slot)
        self._refill()
        self._leaf_batches(need)

    def _simulate(self, slot, seat, tree, p, g, need):
        """One step of seat p's search into `need`: its root belief row alone (a reused root waiting for
        its belief, docs/search.md 3.8), or its next leaf (with the root belief row after a fresh root's
        own leaf) -> True when the slot waits for a batch, False after a terminal walk."""
        bx = bt = None
        rk = self.rk
        if slot.belief and not tree.has_belief:
            bx, bt, _ = mcts.leaf_row(g, p, rk)     # the root's features: the decider's own view
            if rk is not None and self._over(slot, 'belief', bt):
                return False
            if tree.expanded:
                need.setdefault(seat.key, []).append((slot, seat, tree, p, None, None, None, bx, None, bt))
                self.belief_rows += 1
                return True
        r = tree.simulate(g, slot.search_rng, seat, rk)
        if r is None:
            if bx is not None:
                raise RuntimeError("internal: a fresh root's first walk did not stop at the root")
            slot.sims += 1
            return False
        player, x, acts, vx, rows = r
        if rows is not None and (self._over(slot, 'leaf', rows[0]) or self._over(slot, 'value', rows[2])):
            return False
        need.setdefault(seat.key, []).append((slot, seat, tree, player, x, acts, vx, bx, rows, bt))
        self.belief_rows += bx is not None
        return True

    def _over(self, slot, kind, tail):
        """A row of the slot's game (`kind`, its `tail`) past the game's row limit (docs/search.md 2.15): the game is
        aborted - it ends at the round's top, no samples, counted and logged - as the Rust driver's; -> whether."""
        cap = self.rk.over(tail)
        if cap is None:
            return False
        self._abort(slot, f'its {kind} row has capacity {cap} > {self.rk.limit}')
        return True

    def _abort(self, slot, why):
        """Abort the slot's game for the row limit (docs/search.md 2.15; `why`: the row, for the log line)."""
        print(f'max_row_tokens: game {slot.gid} (driver seed {self.seed}) aborted: {why}', file=sys.stderr, flush=True)
        slot.abort = 'max_row_tokens'
        self.aborted_row_tokens += 1

    def _leaf_batches(self, need):
        """Queue one network batch per key of `need` ({key: [(slot, seat, tree, player, x, acts, vx, bx, rows,
        bt)]}): each leaf's row, then (a root-view value) its value-only row of the tree's viewer, then (a
        fresh root's belief) its root belief row; x None: a root belief row alone. `rows` / `bt`: the leaf's rows'
        and the belief row's tails and overrides (a game with variable-length rows, docs/search.md 2.15)."""
        rk = self.rk
        no_ov = (np.zeros(0, np.int32), np.zeros(0, np.int64)) if rk is not None and rk.pointers else None
        for key, items in need.items():
            # rows: each leaf's, then (root-view values) its value-only row of the tree's viewer, then
            # (the root's belief) the root's belief row - each with its (tail, overrides)
            rows, refs, tails = [], [], []
            for it in items:
                slot, seat, tree, player, x, acts, vx, bx, lr, bt = it
                b = vb = bb = None
                if x is not None:
                    b = len(rows)
                    rows.append((x, acts, player, slot, ROW_LEAF))
                    tails.append((None, None) if lr is None else (lr[0], lr[1]))
                    if vx is not None:
                        rows.append((vx, [], tree.t.viewer, slot, ROW_VALUE))
                        tails.append((None, None) if lr is None else (lr[2], no_ov))
                    vb = len(rows) - 1
                if bx is not None:
                    bb = len(rows)
                    rows.append((bx, [], player, slot, ROW_BELIEF))
                    tails.append((bt, no_ov))
                refs.append((slot, seat, tree, b, vb, bb))
            off = np.zeros(len(rows) + 1, np.int32)
            off[1:] = np.cumsum([len(r[1]) for r in rows])
            a = {'x': np.ascontiguousarray(np.stack([r[0] for r in rows]).astype(np.float32)),
                 'opt_off': off,
                 'opt_action': np.asarray([x for r in rows for x in r[1]], np.int32),
                 'player': np.asarray([r[2] for r in rows], np.int32),
                 'n_players': np.asarray([r[3].n for r in rows], np.int32),
                 'gid': np.asarray([r[3].gid for r in rows], np.int64),
                 'extra': np.asarray([r[4] for r in rows], np.uint8)}
            if rk is not None:
                a.update(row_arrays(tails, rk))
            lb = LeafBatch(self.next_bid, key, 'nn', a, tails=rk.widths) if rk is not None else \
                LeafBatch(self.next_bid, key, 'nn', a)
            self.outstanding[self.next_bid] = (lb, refs)
            self.next_bid += 1
            self.queue.append(lb)

    def _start(self, slot, seat, p):
        if seat.gumbel and self.record and seat.selfplay and seat.record and seat.policy_target == 'visits':
            raise ValueError("root_search 'gumbel' records the policy_target 'cq' (its visit counts are sequential "
                             "halving's schedule, not a policy)")
        slot.full = seat.sims_fast is None or slot.rng.random() < seat.p_full
        slot.target = seat.sims if slot.full else seat.sims_fast
        slot.belief = seat.takes_belief(slot.full)          # (docs/search.md 3.8)
        tree = slot.trees.get(p)
        if tree is None:
            tree = slot.trees[p] = SEARCH[seat.impl](slot.game, seat, slot.search_rng)
        else:
            tree.set_base(slot.game, slot.search_rng)
            if seat.selfplay and slot.full and tree.expanded and not seat.gumbel:
                tree.add_noise(seat, slot.np_rng)
        # the root minimum visits of this search (docs/search.md 3.5; 0 = off)
        tree.set_root_min(mcts.root_floor(seat.root_min_visits, seat.root_min_visits_max_actions, slot.full,
                                          len(slot.game.legal()), slot.target))
        if seat.gumbel:
            # the Gumbel root search (docs/search.md 3.7): every search of the seat; its noise (drawn from
            # the slot's noise stream) replaces Dirichlet noise and temperature; the schedule is that of the
            # search's simulations (a fresh root's own evaluation takes the first)
            sc = mcts.gumbel_noise_scale(seat.selfplay, slot.decisions, seat.temp_moves, seat.gumbel_scale)
            k = len(slot.game.legal())
            noise = list(sc * slot.np_rng.gumbel(size=k)) if sc > 0 else []
            tree.set_gumbel(seat.gumbel_m, slot.target, noise, seat.gumbel_c_visit, seat.gumbel_c_scale)

    def _decide(self, slot, seat, tree, p):
        acts, Nv, P0, rw = tree.stats()
        if slot.view_wait:
            # value-only records waiting for this seat's next search take its root value ('next_q')
            keep = []
            for i, o in slot.view_wait:
                if o == p:
                    r = slot.records[i]
                    slot.records[i] = r[:4] + (None if rw is None else list(rw),) + r[5:]
                else:
                    keep.append((i, o))
            slot.view_wait = keep
        tot = sum(Nv)
        if seat.gumbel:
            i = tree.gumbel_choose()                   # the Gumbel search's move: its noise sampled
        elif seat.selfplay and slot.decisions < seat.temp_moves and tot > 0:
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
            rk = self.rk
            if rk is None:
                slot.records.append((slot.game.encode(p), list(acts), pi, p, rw))
            else:
                # (variable-length rows, docs/search.md 2.15: the record's tail and overrides; past the limit the
                # game is aborted, as the Rust driver's records)
                x, t, ov = mcts.leaf_row(slot.game, p, rk, opts=True)
                if self._over(slot, 'record', t):
                    return
                slot.records.append((x, list(acts), pi, p, rw, t, ov))
            if seat.view_samples > 0 and view_pick(self.seed, slot.gid, slot.decisions, seat.view_samples):
                for o in range(slot.n):
                    so = slot.seats[o]
                    if o == p or not (isinstance(so, NNSeat) and so.selfplay and so.record):
                        continue
                    if rk is None:
                        rec = (slot.game.encode(o), [], [], o, None)
                    else:
                        x, t, ov = mcts.leaf_row(slot.game, o, rk)
                        if self._over(slot, 'record', t):
                            return
                        rec = (x, [], [], o, None, t, ov)
                    if seat.view_sample_target == 'next_q':
                        slot.view_wait.append((len(slot.records), o))
                    slot.records.append(rec)
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

    def _episode_end(self, slot):
        """A chained game's episode ended (docs/search.md 2.14; the Rust driver's `finish_episode`): its records
        labelled with its outcome into the contest's samples, its row into the contest's episodes; -> True when the slot
        goes on with the next episode (`next_episode`'s state: fresh trees and records, decisions and moves counted from
        0), False when the contest is over or the episode was cut / aborted (`_finish` hands out the contest's row). A
        game that does not chain: False, nothing done."""
        ch = slot.chain
        if ch is None:
            return False
        g = slot.game
        n = slot.n
        # (an episode the driver aborted for the row limit, docs/search.md 2.15: aborted, no samples, the contest ends)
        over = g.over and not g.aborted and not slot.abort
        res = list(g.result()) if over else [1.0 / n] * n
        margins = [g.margin(i) for i in range(n)] if over else [0.0] * n
        k = len(slot.records) if self.record and over else 0
        if k:
            # (each record's tail and overrides with it, as `_finish`'s samples: docs/search.md 2.15)
            ch['samples'].append(_sample_arrays(slot.records, res, margins, n, self.rk))
        ch['episodes'].append({'over': over, 'aborted': bool(g.aborted) or slot.abort is not None, 'result': res,
                               'margins': margins,
                               'steps': slot.steps, 'decisions': slot.decisions, 'ext_moves': 0, 'rounds': g.rounds,
                               'samples': k})
        ch['steps'] += slot.steps
        ch['decisions'] += slot.decisions
        nxt, ch['result'] = g.next_episode() if over else (None, None)
        if nxt is None:
            return False
        slot.game = nxt
        slot.trees.clear()
        slot.sims = 0
        slot.target = None
        slot.full = True
        slot.belief = False
        slot.records = []
        slot.view_wait = []
        slot.decisions = 0
        slot.steps = 0
        return True

    def _finish(self, slot):
        g = slot.game
        n = slot.n
        ch = slot.chain
        if ch is not None:
            # a chained game's contest (docs/search.md 2.14): one row, the episodes' samples behind it (an episode the
            # driver aborted for the row limit ends the contest unfinished, docs/search.md 2.15)
            r = ch['result']
            self.results.append({'gid': slot.gid, 'n': n, 'over': r is not None,
                                 'aborted': bool(r is None and (g.aborted or slot.abort is not None)),
                                 'result': list(r) if r is not None else [1.0 / n] * n,
                                 'scores': g.scores(), 'steps': ch['steps'], 'decisions': ch['decisions'],
                                 'ext_moves': 0, 'rounds': g.rounds, 'samples': sum(e['samples'] for e in ch['episodes']),
                                 'episodes': ch['episodes']})
            if slot.abort:
                self.results[-1]['abort_reason'] = slot.abort
            self.samples.extend(ch['samples'])
            return
        # (an aborted game has no result: as unfinished; so has one the driver aborted for the row limit)
        over = g.over and not g.aborted and not slot.abort
        if over and g.deferred:
            raise RuntimeError(f'game {slot.gid}: a deferred terminal (the Python reference driver does not park games; '
                               f'the Rust driver does, docs/search.md 2.8)')
        res = list(g.result()) if over else [1.0 / n] * n
        self.results.append({'gid': slot.gid, 'n': n, 'over': over, 'aborted': bool(g.aborted) or slot.abort is not None,
                             'result': res, 'scores': g.scores(),
                             'steps': slot.steps, 'decisions': slot.decisions, 'ext_moves': 0,
                             'rounds': g.rounds, 'samples': len(slot.records) if self.record and over else 0})
        if slot.abort:
            self.results[-1]['abort_reason'] = slot.abort
        if self.record and over and slot.records:
            margins = [g.margin(i) for i in range(n)]
            self.samples.append(_sample_arrays(slot.records, res, margins, n, self.rk))


def row_arrays(rows, rk):
    """The batch / sample keys of rows' variable-length parts (docs/search.md 2.4, 2.15): {'t_off' int64 [N + 1],
    't_seg' uint16, 't_slot' uint32, 't_val' float32} with tail segments and {'ov_off' int64 [N + 1], 'ov_opt' int32,
    'ov_tok' int64} with pointer overrides - the Rust driver's arrays. `rows`: per row (tail, ov) as `mcts.leaf_row`
    gives them; `rk`: the game's `mcts.Rows`."""
    out = {}
    N = len(rows)
    if rk.widths:
        ts = [t for t, _ in rows]
        if any(t is None for t in ts):
            raise ValueError('row_arrays: a row without its tail (a game with tail segments)')
        off = np.zeros(N + 1, np.int64)
        off[1:] = np.cumsum([len(t['t_seg']) for t in ts])
        cat = lambda k, dt: np.concatenate([np.zeros(0, dt)] + [np.asarray(t[k], dt) for t in ts])  # noqa: E731
        out.update(t_off=off, t_seg=cat('t_seg', np.uint16), t_slot=cat('t_slot', np.uint32), t_val=cat('t_val', np.float32))
    if rk.pointers:
        os_ = [o for _, o in rows]
        if any(o is None for o in os_):
            raise ValueError('row_arrays: a row without its overrides (a game with pointer overrides)')
        off = np.zeros(N + 1, np.int64)
        off[1:] = np.cumsum([len(o[0]) for o in os_])
        out.update(ov_off=off, ov_opt=np.concatenate([np.zeros(0, np.int32)] + [np.asarray(o[0], np.int32) for o in os_]),
                   ov_tok=np.concatenate([np.zeros(0, np.int64)] + [np.asarray(o[1], np.int64) for o in os_]))
    return out


def _sample_arrays(records, res, margins, n, rk=None):
    N = len(records)
    z = np.zeros((N, MAX_SEATS), np.float32)
    q = np.zeros((N, MAX_SEATS), np.float32)
    sc = np.zeros((N, MAX_SEATS), np.float32)
    off = np.zeros(N + 1, np.int64)
    acts, pis = [], []
    for k, (_, a, pi, mover, rq, *_rows) in enumerate(records):
        acts.extend(a)
        pis.extend(pi)
        off[k + 1] = len(acts)
        for j in range(n):
            s = (mover + j) % n
            z[k, j] = res[s]
            q[k, j] = rq[s] if rq is not None else res[s]
            sc[k, j] = margins[s]
    out = {'x': np.stack([r[0] for r in records]).astype(np.float32),
           'player': np.asarray([r[3] for r in records], np.int32),
           'n_players': np.full(N, n, np.int32), 'opt_off': off,
           'opt_action': np.asarray(acts, np.int32), 'pi': np.asarray(pis, np.float32),
           'z': z, 'q': q, 'sc': sc, 'net': np.zeros(N, np.uint32)}
    if rk is not None:
        # (variable-length rows, docs/search.md 2.15: each record's tail and overrides, as the Rust driver's samples)
        out.update(row_arrays([(r[5], r[6]) for r in records], rk))
    return out
