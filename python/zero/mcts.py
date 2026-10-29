"""Reference MCTS (docs/search.md 3). The Rust search (`crates/search/src/tree.rs`)
implements these semantics; parity is checked statistically (`zero/parity.py`). Game access
only through the `<package>._engine.Game` API (clone, determinize, step, force_chance, legal,
to_move, over, result, margin, deferred, team_size).

Tree
* A **Node** is one decision as the search sees it: decider seat, legal options (fixed for
  the node), priors P (P0 = before root noise), visit counts N, value sums W (the decider's
  component of the utility vector), the network's utility vector `vec` (first-play urgency)
  and `wsum` (sum of backed-up win vectors: wsum / tot = the search's win estimate there).
* Children go through an **Edge** per (node, option): `kids` maps (chance key, decider,
  legal actions) -> Node, and `outs` lists the chance keys sampled through the edge (repeats
  kept). Keying by the next decision's signature keeps a node's options fixed even if hidden
  information differs between simulations.
* A transition = `step(a)` then auto-steps of every single-option decision; its chance key =
  the outcome keys of the whole transition.

Simulation (one network evaluation each, strictly sequential within a tree)
1. g = clone of the root game; with `redet` (default) `g.determinize(viewer, seed)` - a fresh
   world for everything hidden from the root decider, every simulation (`redet=False`: one
   world per decision, sampled when the search starts).
2. Walk: at an expanded node pick i = argmax Q_i + c_puct P_i sqrt(max(1, tot)) / (1 + N_i)
   with Q_i = W_i / N_i or the node's own utility for its decider (FPU) when N_i = 0; ties ->
   lowest index. If chance_k > 0 and the edge holds >= chance_k sampled keys, force one
   (uniform over the list) via `force_chance`; otherwise sample fresh and append the key (if
   non-empty). Terminal -> back up the result; else descend to (or create) the child.
3. Unexpanded node -> leaf: the network evaluates `g.encode(decider)`; the reply expands the
   node and is backed up: N_i += 1, W_i += util[decider of that node], tot += 1, wsum += win.
   util = team_size * win + margin_w * tanh(margin) (`team_size`: seats per team, 1 but for
   team games; win = the value / result, summing to 1).
4. A **deferred terminal** (`g.over and g.deferred`: the result comes from outside, docs/search.md
   3.2) is a node of no options of the tree's viewer, keyed like any child: unexpanded, it is a
   leaf the network evaluates (`g.encode(viewer)`, no priors) and it keeps its utility (`vec`)
   and win vector (`wsum`, tot 1); every later visit backs that value up again. `result()` is
   never called there.

Policy target of a recorded decision (docs/search.md 3.1): the root visit distribution, or
the completed-Q target of Gumbel AlphaZero (`cq_target`, `completed_q_policy`).

The move of a decision that is not sampled (docs/search.md 3.3; `select_index`): argmax (N, P0)
('visits', the default), or the argmax of the completed-Q target's logits ('cq',
`completed_q_logits`) among the options with at least `min_visits` visits. `select_tag` names a
rule (its evaluations' Elo pool, docs/az.md 5.4).
"""
import math

# Floor of a prior probability in `completed_q_policy` (float32 tiny, as the Rust tree and mctx).
PRIOR_FLOOR = 1.1754943508222875e-38


class Edge:
    __slots__ = ('kids', 'outs')

    def __init__(self):
        self.kids = {}
        self.outs = []


class Node:
    __slots__ = ('player', 'acts', 'P', 'P0', 'Nv', 'W', 'kids', 'tot', 'vec', 'wsum')

    def __init__(self, player, acts):
        self.player = player
        self.acts = acts
        self.P = None
        self.tot = 0

    def expand(self, priors, util, n):
        k = len(self.acts)
        self.P = list(priors)
        self.P0 = self.P
        self.Nv = [0] * k
        self.W = [0.0] * k
        self.kids = [None] * k
        self.vec = util
        self.wsum = [0.0] * n


class Tree:
    """The search for one decision of `viewer` (the decider at the root)."""
    __slots__ = ('base', 'root', 'viewer', 'redet', 'n', 'team')

    def __init__(self, game, redet=True, rng=None):
        self.viewer = game.to_move
        self.redet = redet
        self.n = game.n_players
        self.team = max(1, getattr(game, 'team_size', 1))
        self.root = Node(self.viewer, list(game.legal()))
        self.set_base(game, rng)

    def set_base(self, game, rng=None):
        if self.redet:
            self.base = game
        else:
            g = game.clone()
            g.determinize(self.viewer, rng.getrandbits(63))
            self.base = g


def transition(g, a, forced=None):
    """step(a), then auto-step single-option decisions -> (chance key tuple, legal actions of
    the pending decision or None when over)."""
    if forced:
        g.force_chance(list(forced))
    key = list(g.step(a))
    legal = None
    while not g.over:
        legal = g.legal()
        if len(legal) != 1:
            break
        key += g.step(legal[0])
        legal = None
    if forced:
        g.force_chance([])
    return tuple(key), legal


def select(node, c_puct):
    sq = c_puct * math.sqrt(node.tot if node.tot > 0 else 1)
    fpu = node.vec[node.player]
    P, Nv, W = node.P, node.Nv, node.W
    best_i, best = 0, -1e30
    for i in range(len(P)):
        n = Nv[i]
        u = (W[i] / n if n else fpu) + sq * P[i] / (1 + n)
        if u > best:
            best_i, best = i, u
    return best_i


def backup(path, util, win):
    for node, i in path:
        node.Nv[i] += 1
        node.W[i] += util[node.player]
        node.tot += 1
        ws = node.wsum
        for j in range(len(ws)):
            ws[j] += win[j]


def root_win(root):
    """The search's average win vector at the root (absolute seats), None before any backup."""
    if not root.tot:
        return None
    return [w / root.tot for w in root.wsum]


def leaf_values(value_rel, margin_rel, player, n, margin_w, team=1):
    """Network output (relative to the decider) -> (util, win) over absolute seats (util = team
    x win + margin_w tanh(margin))."""
    win = [0.0] * n
    util = [0.0] * n
    for j in range(n):
        s = (player + j) % n
        w = float(value_rel[j])
        win[s] = w
        util[s] = team * w + margin_w * math.tanh(float(margin_rel[j])) if margin_w else team * w
    return util, win


def terminal_values(g, margin_w, team=1):
    win = list(g.result())
    if not margin_w and team == 1:
        return win, win
    if not margin_w:
        return [team * w for w in win], win
    return [team * win[i] + margin_w * math.tanh(g.margin(i)) for i in range(len(win))], win


def simulate(tree, rng, c_puct, chance_k=0, margin_w=0.0):
    """One simulation -> (leaf node, leaf game, path) when the leaf needs a network
    evaluation, or None after backing up a terminal state."""
    g = tree.base.clone()
    if tree.redet:
        g.determinize(tree.viewer, rng.getrandbits(63))
        if g.legal() != tree.root.acts or g.to_move != tree.root.player:
            raise RuntimeError(f"determinize() changed the root decision: {tree.root.acts} -> {g.legal()}")
    node = tree.root
    path = []
    while True:
        if node.P is None:
            return node, g, path
        i = select(node, c_puct)
        path.append((node, i))
        edge = node.kids[i]
        if edge is None:
            edge = node.kids[i] = Edge()
        forced = None
        if chance_k and len(edge.outs) >= chance_k:
            forced = edge.outs[rng.randrange(len(edge.outs))]
        key, legal = transition(g, node.acts[i], forced)
        if key and forced is None:
            edge.outs.append(key)
        if g.over and g.deferred:
            # a deferred terminal: evaluated once by the network, then its value backed up
            sk = (key, tree.viewer, ())
            child = edge.kids.get(sk)
            if child is None:
                child = edge.kids[sk] = Node(tree.viewer, [])
            if child.P is None:
                return child, g, path
            backup(path, child.vec, child.wsum)
            return None
        if g.over:
            backup(path, *terminal_values(g, margin_w, tree.team))
            return None
        sk = (key, g.to_move, tuple(legal))
        child = edge.kids.get(sk)
        if child is None:
            child = edge.kids[sk] = Node(g.to_move, list(legal))
        node = child


def expand_and_backup(node, path, priors, value_rel, margin_rel, n, margin_w=0.0, team=1):
    util, win = leaf_values(value_rel, margin_rel, node.player, n, margin_w, team)
    node.expand(priors, util, n)
    if not node.acts:                       # a deferred terminal keeps its win vector
        node.wsum = list(win)
        node.tot = 1
    backup(path, util, win)


def add_noise(node, alpha, eps, np_rng):
    k = len(node.acts)
    if k < 2:
        return
    noise = np_rng.dirichlet([alpha] * k)
    node.P = [(1 - eps) * p + eps * x for p, x in zip(node.P0, noise)]


def advance(root, a, key, player, legal):
    """The subtree reached by real action a with the real chance key and next decision, or
    None."""
    if root is None or root.P is None:
        return None
    try:
        i = root.acts.index(a)
    except ValueError:
        return None
    edge = root.kids[i]
    if edge is None:
        return None
    return edge.kids.get((tuple(key), player, tuple(legal)))


def completed_q_logits(P0, Nv, W, v, c_visit=50.0, c_scale=1.0):
    """The logits of `completed_q_policy`: ln max(P0, PRIOR_FLOOR) + (c_visit + max N) c_scale
    completedQ per option (the Rust tree's `completed_q_logits`, f64)."""
    k = len(P0)
    sum_n = 0.0
    max_n = 0
    sum_p = 0.0
    sum_pq = 0.0
    for i in range(k):
        if Nv[i] > 0:
            p = max(float(P0[i]), PRIOR_FLOOR)
            sum_n += Nv[i]
            max_n = max(max_n, Nv[i])
            sum_p += p
            sum_pq += p * (W[i] / Nv[i])
    v_mix = (v + sum_n * (sum_pq / sum_p)) / (1.0 + sum_n) if sum_n > 0 else v
    scale = (c_visit + max_n) * c_scale
    return [math.log(max(float(P0[i]), PRIOR_FLOOR)) + scale * (W[i] / Nv[i] if Nv[i] > 0 else v_mix)
            for i in range(k)]


def completed_q_policy(P0, Nv, W, v, c_visit=50.0, c_scale=1.0):
    """Gumbel AlphaZero's policy-improvement target (Danihelka et al. 2022; mctx
    `qtransform_completed_by_mix_value` without per-node min-max rescaling):
    softmax(ln P0 + (c_visit + max N) * c_scale * completedQ), completedQ = W / N for visited
    options and v_mix = (v + sum N * sum_{N>0} P0 Q / sum_{N>0} P0) / (1 + sum N) for unvisited
    ones (their eq. 33; v = the network's value of the node for its decider). Q is the
    decider's utility (P(win) in [0, 1] with margin_w 0). Same arithmetic as the Rust tree's
    `completed_q_policy` (f64)."""
    lg = completed_q_logits(P0, Nv, W, v, c_visit, c_scale)
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def cq_target(node, c_visit=50.0, c_scale=1.0):
    """The completed-Q target at an expanded node (its P0, N, W and its decider's network
    utility `vec[player]`)."""
    return completed_q_policy(node.P0, node.Nv, node.W, node.vec[node.player], c_visit, c_scale)


def visits_target(node):
    """The visit distribution at an expanded node (P0 if nothing was visited)."""
    tot = sum(node.Nv)
    return [n / tot for n in node.Nv] if tot else list(node.P0)


# ------------------------------------------------------------------ the move rule (docs/search.md 3.3)

SELECT_RULES = ('visits', 'cq')


def check_select(rule, min_visits=0, what='select'):
    """A move rule's settings ('visits' | 'cq', the guard `min_visits` >= 0, cq's only) -> None,
    else ValueError (`what`: the setting's name in the message)."""
    if rule not in SELECT_RULES:
        raise ValueError(f'{what} {rule!r}: one of {SELECT_RULES}')
    if isinstance(min_visits, bool) or not isinstance(min_visits, int) or min_visits < 0:
        raise ValueError(f'{what}_min_visits {min_visits!r}: an integer >= 0')
    if rule == 'visits' and min_visits:
        raise ValueError(f"{what}_min_visits {min_visits}: a guard of {what} 'cq' ('visits' has none)")


def select_index(P0, Nv, W, v, rule='visits', min_visits=0, c_visit=50.0, c_scale=1.0):
    """The chosen option of a searched decision that is not sampled (the Rust driver's
    `Select::choose`): 'visits' = argmax (N, P0); 'cq' = argmax of `completed_q_logits` among the
    options with at least min(min_visits, max N) visits (0: every option, an unvisited one at
    v_mix). The first maximum on exact ties; 0 without options. `v`: the network's utility of the
    node for its decider."""
    k = len(Nv)
    if not k:
        return 0
    if rule == 'visits':
        return max(range(k), key=lambda i: (Nv[i], P0[i]))
    if rule != 'cq':
        raise ValueError(f'select {rule!r}: one of {SELECT_RULES}')
    lg = completed_q_logits(P0, Nv, W, v, c_visit, c_scale)
    m = min(min_visits, max(Nv))
    best = None
    for i in range(k):
        if Nv[i] >= m and (best is None or lg[i] > lg[best]):
            best = i
    return 0 if best is None else best


def select_node(node, rule='visits', min_visits=0, c_visit=50.0, c_scale=1.0):
    """`select_index` at an expanded node (its P0, N, W and its decider's network utility)."""
    return select_index(node.P0, node.Nv, node.W, node.vec[node.player], rule, min_visits, c_visit, c_scale)


def select_tag(rule='visits', min_visits=0, c_visit=50.0, c_scale=1.0):
    """The name of a move rule: 'visits', 'cq', 'cq-n2' (the guard), 'cq-v25-s0.5' (a completed-Q
    transform other than c_visit 50, c_scale 1). Evaluations record it (absent = 'visits') and
    each rule has its own Elo pool (docs/az.md 5.4)."""
    if rule == 'visits':
        return 'visits'
    t = rule
    if min_visits:
        t += f'-n{int(min_visits)}'
    if (float(c_visit), float(c_scale)) != (50.0, 1.0):
        t += f'-v{float(c_visit):g}-s{float(c_scale):g}'
    return t
