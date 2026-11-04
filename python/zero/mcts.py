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
`completed_q_logits`) among the options with at least `min_visits` visits, with the rule's own
completed-Q transform where set (`select_transform`; else the target's). `select_tag` names a
rule (its evaluations' Elo pool, docs/az.md 5.4), `select_settings` reads one from run settings.

Root-view leaf values (docs/search.md 3.6; `leaf_value_view` 'root', off by default): a leaf
decided by another seat than the tree's viewer takes its priors from `encode(decider)` and its
value from `encode(viewer)` (`expand_and_backup(..., value_player=viewer)`): the root decider's
information, not the guess at the other seats' hidden cards the sampled world holds.

The root's belief (docs/search.md 3.8; `Tree.set_belief`, off by default): a search may hold the
network's prediction of the hidden cards for its viewer (`Tree.belief` = (predictions, tilt, clip),
computed once per decision from the root's features); every simulation then deals its world with
the game's `determinize_with(viewer, seed, predictions, tilt, clip)` - but a `belief_mix` share, one
draw of the rng each, with `determinize` as before. No belief: the search of before.

Root minimum visits (docs/search.md 3.5; `Tree.set_root_min`, off by default): every root option
of a search gets at least k visits. A walk keeps PUCT's root option unless that option already has
k visits and another has fewer: then it takes the least-visited root option (the lowest index on
ties), a forced walk. The tree counts the forced walks per root option and sums their backed-up
win vectors: the visit-based outputs leave them out (`root_visits_pruned`, `root_win_pruned`), the
Q-based ones (Q, the 'cq' target and rule) use them. `root_floor` decides which searches take it
(the Rust driver's `driver::root_floor`).

Gumbel root search (docs/search.md 3.7; `Tree.set_gumbel`, off by default): Gumbel AlphaZero's root
(Danihelka et al. 2022; mctx's arithmetic). Root walk j of a search may only take an option whose
walks of THIS search equal `gumbel_sequence(min(m, options), budget)[j]` (sequential halving), the
highest `g + completed_q_logits` among them (g: the search's Gumbel noise). The move is
`gumbel_choose` (that score among the options this search walked most), the root value `gumbel_q`
(the improved policy's value over this search's walked options). PUCT below the root.
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


class GumbelRoot:
    """The Gumbel root search of a tree's current search (`Tree.set_gumbel`): the schedule `seq`, the
    noise `g` per root option, sigma's transform, this search's walks `n` and backed-up win vector
    sums `win` per root option, `walks` = sum(n)."""
    __slots__ = ('seq', 'g', 'c_visit', 'c_scale', 'n', 'win', 'walks')

    def __init__(self, seq, g, c_visit, c_scale, k, players):
        self.seq, self.g, self.c_visit, self.c_scale = seq, g, float(c_visit), float(c_scale)
        self.n = [0] * k
        self.win = [[0.0] * players for _ in range(k)]
        self.walks = 0


class Tree:
    """The search for one decision of `viewer` (the decider at the root). `root_min`: the root
    minimum visits of the current search (`set_root_min`; 0 = off), `forced` the forced walks per
    root option, `forced_win` / `forced_n` their backed-up win vectors' sum and number,
    `walk_forced` the root option of the walk in flight if it is forced (None). `gumbel`: the Gumbel
    root search of the current search (`set_gumbel`; None = PUCT at the root)."""
    __slots__ = ('base', 'root', 'viewer', 'redet', 'n', 'team', 'root_min', 'forced', 'forced_win', 'forced_n',
                 'walk_forced', 'belief', 'belief_mix', 'gumbel')

    def __init__(self, game, redet=True, rng=None):
        self.viewer = game.to_move
        self.redet = redet
        self.n = game.n_players
        self.team = max(1, getattr(game, 'team_size', 1))
        self.root = Node(self.viewer, list(game.legal()))
        self.belief, self.belief_mix = None, 0.0
        self.gumbel = None
        self.set_root_min(0)
        self.set_base(game, rng)

    def set_belief(self, belief, mix=0.0, game=None, rng=None):
        """The root's belief for the search starting now (docs/search.md 3.8; the Rust tree's
        `set_belief`): `belief` = (predictions float32 [aux width], tilt > 0, clip) or None (uniform
        worlds), `mix` the share of simulations dealt uniformly; without redet the one world of the
        decision is dealt again from `game`. `advance` clears it."""
        self.belief = belief
        self.belief_mix = float(mix) if belief is not None else 0.0
        if not self.redet:
            self.set_base(game, rng)

    def set_root_min(self, k):
        """Root minimum visits for the search starting now (0 = off); clears the forced walks'
        accounting (the Rust tree's `set_root_min`; a new root after `advance` needs it again)."""
        self.root_min = int(k)
        self.forced = [0] * len(self.root.acts) if k else []
        self.forced_win = [0.0] * self.n
        self.forced_n = 0
        self.walk_forced = None

    def set_gumbel(self, m, budget, noise=None, c_visit=50.0, c_scale=1.0):
        """The Gumbel root search for the search starting now (the Rust tree's `set_gumbel`): sequential
        halving over min(m, options) root options, the schedule of `budget` root walks (the driver: the
        search's simulations; a fresh root's walks follow its first budget - 1), `noise` the Gumbel noise per
        root option (None / []: none), sigma's transform (c_visit, c_scale); the floor goes off."""
        k = len(self.root.acts)
        if noise and len(noise) != k:
            raise ValueError(f'set_gumbel: {len(noise)} noise values for {k} root options')
        self.set_root_min(0)
        g = [float(x) for x in noise] if noise else [0.0] * k
        self.gumbel = GumbelRoot(gumbel_sequence(min(int(m), k), int(budget)), g, c_visit, c_scale, k, self.n)

    def set_base(self, game, rng=None):
        if self.redet:
            self.base = game
        else:
            g = game.clone()
            deal(g, self, rng)
            self.base = g


def deal(g, tree, rng):
    """Deal world `g` for the tree's viewer (docs/search.md 3.8): `determinize_with` the tree's belief,
    but a `belief_mix` share (one `rng.random()` draw when 0 < mix) and every world without a belief
    `determinize` (one `rng.getrandbits(63)` seed either way)."""
    b = tree.belief
    if b is not None and not (tree.belief_mix > 0 and rng.random() < tree.belief_mix):
        g.determinize_with(tree.viewer, rng.getrandbits(63), b[0], b[1], b[2])
    else:
        g.determinize(tree.viewer, rng.getrandbits(63))


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


def backup(path, util, win, tree=None):
    """N_i += 1, W_i += util[decider], tot += 1, wsum += win along the path; with `tree`, the
    backup of its forced walk in flight is also counted as forced (root minimum visits), and a Gumbel
    search counts its own root walks (`GumbelRoot.n`, `win`)."""
    if tree is not None and tree.walk_forced is not None:
        i = tree.walk_forced
        tree.walk_forced = None
        tree.forced[i] += 1
        tree.forced_n += 1
        fw = tree.forced_win
        for j in range(len(fw)):
            fw[j] += win[j]
    if tree is not None and tree.gumbel is not None and path and path[0][0] is tree.root:
        gr, i = tree.gumbel, path[0][1]
        gr.n[i] += 1
        gr.walks += 1
        gw = gr.win[i]
        for j in range(len(gw)):
            gw[j] += win[j]
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


def root_win_pruned(tree):
    """`root_win` without the current search's forced walks (= root_win when nothing was forced;
    None when no other backup went through the root)."""
    r = tree.root
    if not tree.forced_n:
        return root_win(r)
    t = r.tot - tree.forced_n
    if t <= 0:
        return None
    return [(w - f) / t for w, f in zip(r.wsum, tree.forced_win)]


def root_visits_pruned(tree):
    """The root's visit counts without the current search's forced walks (N - forced; the visit
    counts when the floor is off)."""
    nv = list(tree.root.Nv)
    for i, f in enumerate(tree.forced):
        nv[i] -= f
    return nv


def force_root(tree, i0):
    """The root option a walk takes under the floor given PUCT's choice i0: i0 unless it has at
    least root_min visits and another root option has fewer - then the least-visited one (the
    lowest index on ties), recorded as the walk's forced option."""
    nv = tree.root.Nv
    k = tree.root_min
    if nv[i0] < k:
        return i0
    m = min(nv)
    if m >= k:
        return i0
    j = nv.index(m)
    tree.walk_forced = j
    return j


def root_floor(k, max_actions, full, m, target):
    """The root minimum visits a search takes (the Rust driver's `root_floor`, docs/search.md
    3.5): k for a full search (a self-play seat's recorded ones, every search of an evaluation seat;
    never a fast or minor one) of m options, 2 <= m <= max_actions, with 2 m k <= target simulations;
    else 0."""
    if not k or not full or m < 2 or m > max_actions or 2 * m * k > target:
        return 0
    return int(k)


def check_root_min(k=0, max_actions=4, what='root_min_visits'):
    """The root minimum visits' settings (`what`: the setting's name; k an integer >= 0, max_actions
    an integer >= 2) -> None, else ValueError."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f'{what} {k!r}: an integer >= 0 (0 = off)')
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError(f'root_min_visits_max_actions {max_actions!r}: an integer >= 2')


def forced_tag(k=0, max_actions=4):
    """The suffix of a move rule's name for the root minimum visits of its searches: '' (off),
    '-f4' (k 4, the default max_actions 4), '-f4a6' (another max_actions) - an evaluation's moves
    change with it, so its Elo pool does (docs/az.md 5.4)."""
    if not k:
        return ''
    return f'-f{int(k)}' + (f'a{int(max_actions)}' if int(max_actions) != 4 else '')


LEAF_VIEWS = ('mover', 'root')
VIEW_TARGETS = ('next_q', 'z')
BELIEF_CLIP = 20.0                       # the Rust driver's default `belief_clip` (driver::BELIEF_CLIP)


def check_belief(tilt=0.0, clip=None, mix=0.0, what='belief_tilt'):
    """The root's belief settings (docs/search.md 3.8): tilt a finite number >= 0 (0 = off), clip
    None (the default 20) or a finite number >= 1, mix a share in [0, 1) (1 would deal every world
    uniformly: the tilt a no-op) -> None, else ValueError (`what`: the tilt's setting name)."""
    def num(x):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
    if not num(tilt) or tilt < 0:
        raise ValueError(f'{what} {tilt!r}: a finite number >= 0 (0 = off)')
    if clip is not None and (not num(clip) or clip < 1):
        raise ValueError(f'belief_clip {clip!r}: a finite number >= 1 (every weight in [1/clip, clip])')
    if not num(mix) or not 0.0 <= float(mix) < 1.0:
        raise ValueError(f'belief_mix {mix!r}: a share in [0, 1) of the simulations dealt uniformly (1 would make '
                         f'{what} a no-op: set it to 0)')


def belief_settings(cfg, pre='eval'):
    """(tilt, clip, mix, fast) of the root's belief of a run's seats `pre` ('eval' | 'selfplay') from
    its settings `cfg` (missing keys at their defaults: <pre>_belief_tilt 0 = off, belief_clip 20,
    belief_mix 0, belief_fast 0; docs/az.md 18)."""
    return (float(cfg.get(pre + '_belief_tilt', 0.0) or 0.0), float(cfg.get('belief_clip', BELIEF_CLIP) or BELIEF_CLIP),
            float(cfg.get('belief_mix', 0.0) or 0.0), bool(int(cfg.get('belief_fast', 0) or 0)))


def belief_tag(tilt=0.0, clip=None, mix=0.0, fast=False):
    """The suffix of a move rule's name for its searches' root belief (docs/search.md 3.8): '' (off),
    '-b1' (tilt 1, the default clip 20, no uniform share, full searches), '-b0.5c10m0.25f' (tilt 0.5,
    clip 10, mix 0.25, fast searches too) - an evaluation's moves change with it, so its Elo pool does
    (docs/az.md 5.4); a remote evaluation on a checkout without it reports another rule and is refused."""
    if not tilt:
        return ''
    t = f'-b{float(tilt):g}'
    if clip is not None and float(clip) != BELIEF_CLIP:
        t += f'c{float(clip):g}'
    if mix:
        t += f'm{float(mix):g}'
    return t + ('f' if fast else '')


def check_leaf_view(view='mover', share=0.0, target='next_q', what='leaf_value_view'):
    """Root-view settings (docs/search.md 3.6): `view` 'mover' | 'root' (`what`: its setting's name in
    the message), the value-only samples' share in [0, 1] and their q target 'next_q' | 'z' -> None,
    else ValueError."""
    if view not in LEAF_VIEWS:
        raise ValueError(f'{what} {view!r}: one of {LEAF_VIEWS}')
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0.0 <= float(share) <= 1.0:
        raise ValueError(f'view_samples {share!r}: a share in [0, 1] (0 = off)')
    if target not in VIEW_TARGETS:
        raise ValueError(f'view_sample_target {target!r}: one of {VIEW_TARGETS}')


def check_merge(on=False, view='mover', selfplay=False, what='infoset_merge'):
    """Information-set merging (docs/search.md 3.16): 0 / 1 (False / True); on needs root-view leaf values (an
    other seat's merged node is valued from the root decider's view, the same in every world of its observation)
    -> None, else ValueError. Evaluation and self-play seats alike (`selfplay`: kept for the callers; a self-play
    seat's records, root values and resignation read the root, the same kind of statistics merged or not). The
    Python reference tree (this module) has no merged mode: the Rust tree only (`play.NNSeat(impl='rs')`, the Rust
    driver)."""
    del selfplay
    if isinstance(on, float) or on not in (0, 1):
        raise ValueError(f'{what} {on!r}: 0 or 1')
    if on and view != 'root':
        raise ValueError(f"{what}: needs leaf_value_view 'root' (a merged node of another seat is valued from the root "
                         f"decider's view; docs/search.md 3.16)")


def merge_tag(on=False):
    """The suffix of a move rule's name for its searches' information-set merging (docs/search.md 3.16): '' (off),
    '-ism' (on) - an evaluation's moves change with it, so its Elo pool does (docs/az.md 5.4)."""
    return '-ism' if on else ''


def view_tag(view='mover'):
    """The suffix of a move rule's name for its searches' leaf values: '' ('mover'), '-rv' ('root')
    - an evaluation's moves change with it, so its Elo pool does (docs/az.md 5.4)."""
    return '-rv' if view == 'root' else ''


def leaf_values(value_rel, margin_rel, player, n, margin_w, team=1):
    """Network output (relative to `player`: the decider, or the viewer of a value-only row) ->
    (util, win) over absolute seats (util = team x win + margin_w tanh(margin))."""
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
    tree.walk_forced = None
    g = tree.base.clone()
    if tree.redet:
        deal(g, tree, rng)
        if g.legal() != tree.root.acts or g.to_move != tree.root.player:
            raise RuntimeError(f"determinize() changed the root decision: {tree.root.acts} -> {g.legal()}")
    node = tree.root
    path = []
    while True:
        if node.P is None:
            return node, g, path
        if tree.gumbel is not None and node is tree.root:
            i = gumbel_select(tree)             # sequential halving at the root (docs/search.md 3.7)
        else:
            i = select(node, c_puct)
            if tree.root_min and node is tree.root:
                i = force_root(tree, i)
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
            backup(path, child.vec, child.wsum, tree)
            return None
        if g.over:
            backup(path, *terminal_values(g, margin_w, tree.team), tree)
            return None
        sk = (key, g.to_move, tuple(legal))
        child = edge.kids.get(sk)
        if child is None:
            child = edge.kids[sk] = Node(g.to_move, list(legal))
        node = child


def expand_and_backup(node, path, priors, value_rel, margin_rel, n, margin_w=0.0, team=1, tree=None,
                      value_player=None):
    """Expand the leaf `node` with its evaluation and back it up along `path` (`tree`: the tree of
    the walk, for the root minimum visits' accounting of a forced walk). `value_player`: the seat
    the value / margin are relative to (None: the leaf's decider; root-view values: the viewer)."""
    vp = node.player if value_player is None else value_player
    util, win = leaf_values(value_rel, margin_rel, vp, n, margin_w, team)
    node.expand(priors, util, n)
    if not node.acts:                       # a deferred terminal keeps its win vector
        node.wsum = list(win)
        node.tot = 1
    backup(path, util, win, tree)


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


def completed_q_policy(P0, Nv, W, v, c_visit=50.0, c_scale=1.0, prior_w=1.0):
    """Gumbel AlphaZero's policy-improvement target (Danihelka et al. 2022; mctx
    `qtransform_completed_by_mix_value` without per-node min-max rescaling):
    softmax(ln P0 + (c_visit + max N) * c_scale * completedQ), completedQ = W / N for visited
    options and v_mix = (v + sum N * sum_{N>0} P0 Q / sum_{N>0} P0) / (1 + sum N) for unvisited
    ones (their eq. 33; v = the network's value of the node for its decider). Q is the
    decider's utility (P(win) in [0, 1] with margin_w 0). Same arithmetic as the Rust tree's
    `completed_q_policy` (f64). prior_w = beta, the prior's exponent (docs/search.md 3.13):
    softmax(beta ln P0 + sigma completedQ), the logits of `completed_q_logits` plus (beta - 1)
    ln max(P0, PRIOR_FLOOR) (v_mix keeps its P0 weights; the Rust `completed_q_policy_w`); 1.0 =
    the target of before, bit for bit."""
    lg = completed_q_logits(P0, Nv, W, v, c_visit, c_scale)
    if prior_w != 1.0:
        lg = [x + (prior_w - 1.0) * math.log(max(float(p), PRIOR_FLOOR)) for x, p in zip(lg, P0)]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def cq_target(node, c_visit=50.0, c_scale=1.0, prior_w=1.0):
    """The completed-Q target at an expanded node (its P0, N, W and its decider's network
    utility `vec[player]`), with the prior exponent `prior_w` (docs/search.md 3.13)."""
    return completed_q_policy(node.P0, node.Nv, node.W, node.vec[node.player], c_visit, c_scale, prior_w)


def check_cq_prior_w(prior_w=1.0, policy_target='cq', what='cq_prior_w'):
    """The prior's exponent beta in the 'cq' target (docs/search.md 3.13): a number in (0, 1] (1 = off),
    and != 1 only with policy_target 'cq' -> None, else ValueError."""
    if isinstance(prior_w, bool) or not isinstance(prior_w, (int, float)) or not 0.0 < prior_w <= 1.0:
        raise ValueError(f"{what} {prior_w!r}: the prior's exponent in the 'cq' target, in (0, 1] (1 = off)")
    if prior_w != 1.0 and policy_target != 'cq':
        raise ValueError(f"{what} {prior_w!r}: a setting of policy_target 'cq' ({policy_target!r} has no prior term)")


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


def check_select_transform(rule, c_visit=None, c_scale=None, what='select'):
    """A move rule's own completed-Q transform (`<what>_c_visit`, `<what>_c_scale`: None = unset,
    the 'cq' target's cq_c_visit / cq_c_scale; else a finite number >= 0, of the rule 'cq' only)
    -> None, else ValueError (`what`: the rule's setting name in the message)."""
    for name, x in (('c_visit', c_visit), ('c_scale', c_scale)):
        if x is None:
            continue
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            raise ValueError(f'{what}_{name} {x!r}: a finite number >= 0 (unset: cq_{name})')
        if rule == 'visits':
            raise ValueError(f"{what}_{name} {x!r}: a setting of {what} 'cq' ('visits' has no completed-Q transform)")


def select_transform(c_visit=50.0, c_scale=1.0, sel_c_visit=None, sel_c_scale=None):
    """The 'cq' rule's completed-Q transform (c_visit, c_scale) as floats: its own `sel_c_visit` /
    `sel_c_scale` where set, else the target's `c_visit` / `c_scale` (the cq_c_* settings)."""
    return (float(c_visit if sel_c_visit is None else sel_c_visit),
            float(c_scale if sel_c_scale is None else sel_c_scale))


def select_settings(cfg, pre='eval_select'):
    """(rule, min_visits, c_visit, c_scale) of a run's move rule `pre` ('eval_select' |
    'selfplay_select') from its settings `cfg` (a config dict, missing keys at their defaults): the
    transform is `<pre>_c_visit` / `<pre>_c_scale` where set, else cq_c_visit / cq_c_scale (docs/az.md
    5.4)."""
    cv, cs = select_transform(cfg.get('cq_c_visit', 50.0), cfg.get('cq_c_scale', 1.0), cfg.get(pre + '_c_visit'),
                              cfg.get(pre + '_c_scale'))
    return cfg.get(pre, 'visits'), int(cfg.get(pre + '_min_visits', 0)), cv, cs


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
    transform other than c_visit 50, c_scale 1: the rule's own, `select_transform`). Evaluations
    record it (absent = 'visits') and each rule has its own Elo pool (docs/az.md 5.4), so the
    pool follows the rule's transform only, never the training target's."""
    if rule == 'visits':
        return 'visits'
    t = rule
    if min_visits:
        t += f'-n{int(min_visits)}'
    if (float(c_visit), float(c_scale)) != (50.0, 1.0):
        t += f'-v{float(c_visit):g}-s{float(c_scale):g}'
    return t


# ------------------------------------------------------------------ the Gumbel root search (docs/search.md 3.7)

ROOT_SEARCHES = ('puct', 'gumbel')


def gumbel_sequence(m, n):
    """Sequential halving's schedule (mctx `get_sequence_of_considered_visits`; the Rust
    `tree::gumbel_sequence`): per root walk of a search of n root walks over m considered options, the
    visits (of that search) an option needs to be taken."""
    if m <= 1:
        return list(range(n))
    log2max = int(math.ceil(math.log2(m)))
    seq, visits, nc = [], [0] * m, m
    while len(seq) < n:
        extra = max(1, n // (log2max * nc))
        for _ in range(extra):
            seq.extend(visits[:nc])
            for i in range(nc):
                visits[i] += 1
        nc = max(2, nc // 2)
    return seq[:n]


def _first_max(score, ok):
    best = None
    for i, x in enumerate(score):
        if ok(i) and (best is None or x > score[best]):
            best = i
    return best


def gumbel_scores(tree):
    """g + completed_q_logits per root option (every visit of the root: a reused root's too)."""
    r, gr = tree.root, tree.gumbel
    lg = completed_q_logits(r.P0, r.Nv, r.W, r.vec[r.player], gr.c_visit, gr.c_scale)
    return [x + g for x, g in zip(lg, gr.g)]


def gumbel_select(tree):
    """The root option of a Gumbel search's next walk: among the options whose walks of this search
    equal the schedule's entry (past it: the most walked ones), the highest score (first maximum)."""
    gr = tree.gumbel
    want = gr.seq[gr.walks] if gr.walks < len(gr.seq) else max(gr.n)
    sc = gumbel_scores(tree)
    i = _first_max(sc, lambda k: gr.n[k] == want)
    return _first_max(sc, lambda k: True) if i is None else i


def gumbel_choose(tree):
    """The Gumbel search's move (the Rust tree's `gumbel_choose`): among the root options this search
    walked most, the highest g + ln P0 + sigma(completedQ) (first maximum); 0 without statistics."""
    gr = tree.gumbel
    if tree.root.P is None:
        return 0
    top = max(gr.n)
    i = _first_max(gumbel_scores(tree), lambda k: gr.n[k] == top)
    return 0 if i is None else i


def gumbel_q(tree):
    """The Gumbel search's root value (absolute seats; the Rust tree's `gumbel_q`): sum pi'_i Wv_i /
    sum pi'_i over the root options this search walked, pi' = the completed-Q policy with the
    search's transform, Wv_i = this search's mean backed-up win vector through option i; None when it
    walked none."""
    gr, r = tree.gumbel, tree.root
    if not gr.walks or r.P is None:
        return None
    lg = completed_q_logits(r.P0, r.Nv, r.W, r.vec[r.player], gr.c_visit, gr.c_scale)
    m = max(lg)
    tot, q = 0.0, [0.0] * len(gr.win[0])
    for i, x in enumerate(lg):
        if not gr.n[i]:
            continue
        p = math.exp(x - m)
        tot += p
        for j in range(len(q)):
            q[j] += p * gr.win[i][j] / gr.n[i]
    return [x / tot for x in q]


def check_root_search(name='puct', m=16, scale=1.0, c_visit=None, c_scale=None, what='root_search'):
    """The root search's settings ('puct' | 'gumbel'; gumbel_m an integer >= 1; the noise scale and
    sigma's transform finite numbers >= 0, None = unset) -> None, else ValueError (`what`: the
    setting's name in the message)."""
    if name not in ROOT_SEARCHES:
        raise ValueError(f'{what} {name!r}: one of {ROOT_SEARCHES}')
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError(f'gumbel_m {m!r}: an integer >= 1')
    for k, x in (('gumbel_scale', scale), ('gumbel_c_visit', c_visit), ('gumbel_c_scale', c_scale)):
        if x is None:
            continue
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            raise ValueError(f'{k} {x!r}: a finite number >= 0')


def gumbel_noise_scale(selfplay, decisions, temp_moves, scale):
    """The noise scale of a Gumbel search: a self-play seat's first `temp_moves` searched decisions of a
    game get `scale`, later ones none (the Gumbel noise is the temperature); an evaluation seat's
    every decision `scale` (the Rust driver's `worker::start`)."""
    return 0.0 if selfplay and decisions >= temp_moves else float(scale)


def gumbel_tag(m=16, c_visit=50.0, c_scale=1.0, scale=0.0):
    """The name of a Gumbel search's move rule: 'gumbel-m16' (+ '-v50-s0.3' for a transform other than
    50 / 1, + '-g1' with noise): an evaluation's moves are its, so its Elo pool is (docs/az.md 5.4)."""
    t = f'gumbel-m{int(m)}'
    if (float(c_visit), float(c_scale)) != (50.0, 1.0):
        t += f'-v{float(c_visit):g}-s{float(c_scale):g}'
    if float(scale):
        t += f'-g{float(scale):g}'
    return t
