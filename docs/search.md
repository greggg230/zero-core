# Search: the `SearchGame` trait, the driver protocol, MCTS semantics

Zero Core's game-agnostic AlphaZero search: crate `crates/search` (`zero_search`), the PyO3
classes `crates/py` (`zero_py`, behind every game's Python module `<pkg>._engine`), the Python
reference in `zero/mcts.py` + `zero/play.py`. Moved here unchanged (apart from names) from
Glass Road Zero and Caverna Zero, whose copies were identical: Caverna Zero had ported it from
Dominion Zero's `engine/src/search` (batched multi-threaded driver, rolling self-play) with St.
Pete Zero's fixed-width-feature network interface, plus **external seats** for players outside
the AZ action layer (the web apps' JS bots). How a game plugs in: docs/new_game.md; each game
repo documents its own integration (Caverna Zero docs/search.md 8, Glass Road Zero
docs/search.md 8-9: hidden hands, simultaneous choices, revelation keys).

| File | What |
|---|---|
| `crates/search/src/game.rs` | `SearchGame` trait, `Bot` trait, `RandomBot`, `MAX_SEATS` = 8 |
| `crates/search/src/tree.rs` | One search tree: PUCT, chance children, determinization, noise, reuse, `compact`, completed-Q targets |
| `crates/search/src/driver.rs`, `driver/worker.rs` | Batched multi-threaded driver: groups, chunks, parallel batch assembly, playout cap, temperature, records, rolling phases, arena keys, external seats |
| `crates/search/src/toy.rs` | "Mini Cavern", the toy game (section 5) + its heuristic bot |
| `crates/search/src/belief.rs` | Belief particles: another seat's hidden choice sampled with the network's policy (section 10.3) |
| `crates/search/src/secret.rs` | "Secret hands", a tiny hidden simultaneous-choice game: the test bed of section 10 (`zero._engine` game "secret") |
| `crates/search/src/dynapi.rs` | Object-safe wrappers (`DynGame`, `DynTree`, `DynDriver`) so the bindings hold any game |
| `crates/search/src/rng.rs`, `noise.rs` | xoshiro256++ / splitmix64, Gamma / Dirichlet |
| `crates/py/src/lib.rs` | `zero_py`: `Game`, `SearchTree`, `SearchDriver`, `game_names()`, `game_info_of()`, the `engine_module!` macro (section 4) |
| `crates/toy_py/src/lib.rs` | the core's own module `zero._engine` (the toy only) |
| `python/zero/mcts.py`, `play.py` | Python reference MCTS and driver (`PyDriver`, seats searching with the Python or the Rust tree) |
| `python/zero/protocol.py` | Leaf-batch protocol: `LeafBatch`, validators, `MAX_SEATS` |
| `python/zero/parity.py` | Rust-vs-Python statistical parity (section 6) |

Build: `bash tools/build_py.sh` (the toy module: release cdylib copied to
`python/zero/_engine.pyd`; target dir `$CARGO_TARGET_DIR` or `E:/cargo-target/zero-core/<branch>`
on Windows). A game builds its own `<pkg>._engine` with its `tools/build_py.sh` (docs/pinning.md).
Tests: `cargo test` (the `zero_search` crate: 42 tests, < 1 s in release), `python -m pytest
tests` (57 tests on the toy and the secret game, ~1 min on the loaded desktop).

## 1. The trait (final version)

```rust
pub const MAX_SEATS: usize = 8;              // value / margin vectors (Glass Road: 1-4 players)

pub trait SearchGame: Clone + Send + Sync + 'static {
    type Spec: Clone + Send + Sync + 'static; // settings shared by a driver's games (JSON)
    const NAME: &'static str;
    const NUM_ACTIONS: usize;                // global action ids are < NUM_ACTIONS (u16)
    const NUM_FEATURES: usize;               // fixed-width f32 feature vector
    const MIN_PLAYERS: usize;
    const MAX_PLAYERS: usize;

    fn parse_spec(json: &str) -> Result<Self::Spec, String>;
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String>;
    fn num_players(&self) -> usize;
    fn to_move(&self) -> Option<usize>;                            // None iff over
    fn legal_actions(&self, out: &mut Vec<u16>);                    // deterministic order, no duplicates
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String>; // pushes chance outcome keys
    fn force_chance(&mut self, forced: &[u64]) {}                   // queue outcomes (chance_k); &[] clears
    fn result(&self) -> [f32; MAX_SEATS];                          // per seat, sums to 1, ties split
    fn margin(&self, seat: usize) -> f32;                           // aux target: (score - best other), scaled O(1)
    fn score(&self, seat: usize) -> f32;                            // reports only
    fn determinize(&mut self, viewer: usize, rng: &mut Rng);        // resample everything hidden from viewer
    fn encode(&self, viewer: usize, out: &mut [f32]);               // out is zeroed by the caller
    fn rounds(&self) -> u32 { 0 }                                   // reports
    fn export_state(&self) -> String { .. }                         // external seats: the state (JSON)
    fn apply_external(&mut self, payload: &str, chance: &mut Vec<u64>) -> Result<(), String> { Err(..) }
    fn bot_names() -> &'static [&'static str] { &[] }               // built-in bots ('random' is generic)
    fn make_bot(spec: &Self::Spec, name: &str, seed: u64) -> Option<Box<dyn Bot<Self>>> { None }
    // optional hooks of the opponent models and analysis options (section 10), all defaulted:
    const DECISION_CLASSES: &'static [&'static str] = &[];         // per-class seat options
    fn decision_class(&self) -> usize { 0 }
    const HAS_ORACLE: bool = false;                                 // analysis: the true hidden information
    fn determinize_oracle(&mut self, viewer: usize, rng: &mut Rng) { unreachable!() }
    const HAS_PSAMPLE: bool = false;                                // policy-sampled simultaneous choices
    fn policy_sampled(&self) -> bool { false }
    fn uncommit_simultaneous(&mut self, viewer: usize) {}
    const HAS_BELIEF: bool = false;                                 // belief prior
    fn belief_start(&self) -> Option<u64> { None }
    fn belief_particle(&self, chooser: usize) -> Option<u64> { None }
    fn belief_target(&self, viewer: usize) -> Option<(usize, u64)> { None }
    fn belief_consistent(&self, viewer: usize, particle: u64) -> bool { false }
    fn determinize_particle(&mut self, viewer: usize, particle: u64, rng: &mut Rng) { unreachable!() }
}
pub trait Bot<G>: Send { fn choose(&mut self, g: &G, legal: &[u16]) -> u16; }
```

Changes from the original sketch (Caverna Zero DESIGN.md 5), and why:

* `Spec` + `parse_spec` + `new_game`: the driver creates its games (seeded from the driver
  seed and the game index); the spec is a JSON string so Python passes any game's settings.
* `step` returns `Result`: an illegal move is an engine inconsistency the driver reports
  (game index + driver seed) instead of panicking.
* `force_chance`: needed for `chance_k` (replaying a sampled outcome in another determinized
  world). Semantics: the queued keys are consumed in order by the next chance events; a key
  impossible in this world falls back to the world's own outcome; `step` always reports the
  **actual** outcome (children are keyed by it). Games without it just never force.
* `result()` returns a fixed `[f32; MAX_SEATS]` (no allocation on the hot path).
* `score`, `rounds`: reports. `export_state` / `apply_external`: external seats (section 2.7).
* `bot_names` / `make_bot`: game-specific built-in bots (the toy heuristic; Glass Road could
  add a Rust port of the JS bot later, Glass Road Zero DESIGN.md 6).
* NUM_ACTIONS / NUM_FEATURES / players stay associated constants (one Rust type per game; the
  bindings expose them through `game_info_of`).

Contract details a game must honour (the toy's tests check each):

* `legal_actions` never empty while not over; a decision may be single-option (the driver and
  the search auto-step it, never ask the network).
* Chance keys are canonical: equal keys iff equal outcomes (the toy and Caverna: `(round,
  card, marker)` packed into a u64 per reveal; Glass Road: Glass Road Zero docs/search.md 9.2).
* **Everything a transition reveals that the decider at the root could not predict must be
  keyed**, not only dice-like chance: children are keyed by (key, next decider, next legal
  actions), so two worlds that reveal different facts but lead to the same next signature
  would otherwise share a node (an information-set conflation; Glass Road: which seats
  followed a played card, a face-down card at its reveal - Glass Road Zero docs/search.md 9.2). `force_chance` may
  fail to force such a key (fall back) but must still consume it, so later queued keys stay
  aligned with their events.
* `determinize` changes only hidden information, never a revealed fact and never the viewer's
  current legal actions (the search checks this every simulation and stops with an error
  otherwise). It must also reseed any internal randomness used for future chance events
  (the toy pre-samples them in the state; Caverna's round cards / markers likewise; Glass
  Road: the unseen part of every building stack, and `state.rng` solo). The search only ever
  calls it with `viewer` = the root decider (= `to_move()`); other viewers are used by tests
  (features invariant under `determinize(v)` for every `v`) and analysis, so they must be
  valid too (Glass Road: a later simultaneous chooser's partial response is hidden from an
  earlier one).
* `encode` must not read hidden information: it is applied to determinized worlds.
* Cost model: per simulation the search clones the root state once, determinizes once, and
  per decision on the path calls `step` and `legal_actions` (cache legality inside the game
  if it is expensive - Caverna's exact-legality layer).

## 2. The driver protocol (Python <-> Rust)

```python
from zero import games as G       # in a game repo: from <pkg>.az import games as G
drv = G.Driver('toy', seats, spec=None, concurrency=256, seed=0, record=False, threads=4,
               groups=2, max_steps=100_000, rolling=False, net=0)
lb = drv.next_batch()            # LeafBatch (kind 'nn' | 'ext') or None when every game finished
drv.submit(lb.id, priors, value, margin)        # network batch
drv.submit_external(lb.id, payloads)            # external batch: one str per request
drv.take_results(); drv.take_samples(); drv.stats(); drv.in_flight()
drv.resume(quota, net); drv.end_phase(); drv.paused; drv.finished; drv.keys
```

`infer.drive(drv, evaluators, externals, check=..., on_batch=...)` is the loop:
network batches go to `evaluators[key](lb)`, external batches to `externals[key](lb)`.
`play.PyDriver` (the Python reference) has the same interface.

### 2.1 Seats

One list of seat dicts per game (players = its length):
`{'bot': 'random' | 'heuristic' | ...}`, `{'external': '<key>'}`, or a network seat
`{'key': 'm', 'sims', 'c_puct', 'selfplay', 'alpha', 'eps', 'temp_moves', 'sims_fast',
'p_full', 'chance_k', 'margin_w', 'redet', 'policy_target', 'cq_c_visit', 'cq_c_scale'}`
(`games.nn_seat`; the last three only matter for recorded decisions, section 3.1), plus the
opponent-model and analysis options of section 10 (`psample`, `belief_mix`, `belief_temp`,
`belief_particles`, `oracle_hands`, `oracle_p`, `class_sims`, `mask_features`; all off by
default, refused by games without the hooks). Evaluator keys: every distinct
network `key` is one evaluator (arena / Elo matches: `'m'` vs `'o'`), every distinct external
key one external player. A key cannot be both.

### 2.2 Network batches

| field | dtype / shape | content |
|---|---|---|
| `x` | f32 [B, F] | `encode(decider)` of the (determinized) leaf state |
| `opt_off` | i32 [B+1] | leaf b's legal actions = `opt_action[opt_off[b]:opt_off[b+1]]` (>= 2) |
| `opt_action` | i32 [sum] | global action ids in the game's order |
| `player`, `n_players`, `gid` | i32, i32, i64 [B] | decider (absolute), players, game index |

Answer: `priors` f32 [sum] (softmax over each leaf's legal actions), `value` f32 [B, 8] =
P(win) per seat **relative to the decider** (column j = seat (player + j) % n; columns >= n
are 0), `margin` f32 [B, 8] (same order). `NetEvaluator` does the gather + segment softmax on
the device.

### 2.3 Rules of the exchange

* `next_batch` advances every active game until its search needs ONE network evaluation, an
  external seat is to move, or the game ends: one leaf per game per batch, no virtual loss, no
  in-tree batching (the mtg_bot batch-32 lesson). Bots and single-option decisions are played
  inside the driver.
* The games are split into `groups`; each group's batches (one per key) must be answered
  before that group advances again; with 2 groups the search threads work on group B while
  Python evaluates group A. `next_batch` releases the GIL while waiting; groups are served in
  a fixed round-robin order.
* Batch assembly: every group is split into chunks (<= 4 x threads); the last chunk's worker
  lays out the batches and queues one fill job per chunk at the front of the queue, so all
  threads copy rows in parallel (fixed-width rows: one memcpy per chunk).
* Reproducible: every game's setup, search, noise and bot streams derive from (seed, gid)
  (one bot seed per seat whatever the seat kind); results do not depend on `threads` /
  `groups` (tested).
* Errors (an illegal move, a determinization that changed the root decision, a bad external
  payload, a worker panic) stop the driver: `next_batch` raises with the game index and seed.

### 2.4 Training samples (`take_samples`)

One row per recorded root decision (self-play network seat, full search), from the real
state: `x` [N, F], `player`, `n_players`, `opt_off` (i64 [N+1]), `opt_action`, `pi` (the
seat's policy target over the options: the visit distribution, or the completed-Q target with
`policy_target='cq'`, section 3.1), `z` [N, 8] (final `result()` relative to the decider), `q`
[N, 8] (root search win estimate `wsum / tot`, relative; = z if no backups), `sc` [N, 8]
(`margin()` per relative seat), `net` (network id of the search, rolling mode). A game's
samples are emitted when it ends.

### 2.5 Results (`take_results`)

`{gid, n, over, result, scores, steps, decisions, ext_moves, rounds, samples}`; a game cut at
`max_steps` has `over` False and result 1/n each. With `SearchDriver(..., trace=True)` each row
also has `seed` and `actions` (the game's `new_game` seed and real moves: a replay is
`Game(game, spec, n, seed)`, then per action `step(a)` and auto-steps of single-option
decisions; refused with external seats). `stats()` counts the options of section 10 too
(`psampled`, `belief_rows`, `belief_jobs`, `belief_targets`, `belief_searches`, `belief_missing`,
`belief_worlds`).

### 2.6 Rolling mode

`rolling=True`: game `gid` plays `seats[gid % len]` forever; play is divided into phases:
`resume(quota, net)` runs until `quota` more games have finished, then every group pauses
(batches assembled after the quota are dropped unevaluated; `next_batch` returns None once
all are paused); games in progress keep their state. A new `net` starts a new search epoch:
every game drops its trees and pending leaf and restarts its current decision's search from an
empty tree (same playout-cap draw), so no search mixes two networks. `end_phase()` ends the
phase now. Used by `run.py --rolling 1` (one persistent driver in the run process).

### 2.7 External seats

A seat whose moves come from outside the AZ action layer (the games' JS bots through the Node
bridge, which act on the engine's JS-level API). When an external seat is to move - whatever
the number of its legal AZ actions - the game's `export_state()` goes into an external batch
(`states`, `player`, `n_players`, `gid`); the caller answers each request with a payload
string; the driver applies it with `apply_external(payload)` (any number of moves: a JS
compound response, free actions ...) and continues from whatever state results: every search
tree of that game is dropped (the move was not an AZ action), and the real-move transitions
never auto-step an external seat's single-option decisions (the external player makes all its
decisions). External batches are batched across games like network batches, so a pool of JS
bot processes can answer a whole batch in parallel. The toy's `ext:toyheur`
(`zero/toybots.py`, a Python port of the toy heuristic reading the exported JSON) plays
exactly the games of the built-in Rust `heuristic` bot (tested), and is the toy proof run's
yardstick.

Each game repo registers its own external players (`GamePackage.externals`). Glass Road Zero's
`ext:jsbot` (`grz.az.games.JsBotExternal`) sends each request's state AND seat
(`lb.player`; several seats can be waiting on a multi-player prompt, and the bot server's
`chooseMove(state, p)` needs it) through `grz.jsbot`'s pool (`decide(state, p=, seed=)`, Glass
Road Zero DESIGN.md 6), the requests of a batch in parallel, and answers `{"p", "actions", "h"}`;
`apply_external` should check `p == to_move()`, apply the any-time actions then the response
through the JS-level API, and compare `h` (live parity). If the exported state has to hide
other seats' hidden cards from the bot (Glass Road Zero DESIGN.md 6), `h` is the hash of the bot's world, not
of ours: the check must then be skipped (or done on a projection both sides share). Caverna
Zero's `ext:jsbot` (`cvz.az.games.JsBotExternal`) sends a batch's states in one `decide_many`
call and answers `{"actions", "h"}`.

## 3. MCTS semantics (reference: `zero/mcts.py`)

* **Node** = one decision as the search sees it: decider, legal options (fixed for the node),
  priors P (P0 before root noise), visit counts N, value sums W (decider's component of the
  utility), `vec` = the network's utility vector at expansion (first-play urgency), `tot` =
  backups through the node, `wsum` = sum of backed-up win vectors (absolute seats).
* **Edge** (node, option): children keyed by (chance key of the transition, next decider,
  next legal actions) and `outs` = the chance keys sampled through the edge (repeats kept).
* **Transition** = `step(a)` then auto-step every single-option decision; its chance key =
  all outcome keys of the transition.
* **Simulation**: clone the root state; with `redet` (default) `determinize(viewer)` - a
  fresh world every simulation (`redet=0`: one world per decision); check the root decision
  is unchanged; walk: `i = argmax Q_i + c_puct P_i sqrt(max(tot, 1)) / (1 + N_i)`, Q_i =
  W_i / N_i or `vec[decider]` when N_i = 0, ties -> lowest index. Chance: if `chance_k > 0`
  and the edge holds >= chance_k sampled keys, force one of them (uniform over the list) via
  `force_chance`; otherwise sample fresh and append the key (if non-empty). Terminal ->
  back up `(util, win)` from `result()`; unexpanded node -> leaf (the network evaluates
  `encode(decider)`; the reply expands the node; the leaf itself gets no visit from its own
  evaluation).
* **Backup** along the path: N_i += 1, W_i += util[decider], tot += 1, wsum += win;
  util = win + margin_w * tanh(margin) (network margin, or `margin()` at terminals).
* **Per decision**: single-option decisions are not searched; playout cap (`p_full` share
  gets `sims` and is recorded, the rest `sims_fast`); root Dirichlet noise on full self-play
  searches (also on a reused, already expanded root); self-play samples moves by visit count
  for the first `temp_moves` searched decisions of the game, otherwise (and in evaluation)
  argmax (N, P0) with the first maximum on ties; tree reuse: after a real move the tree
  survives only for the seat that moved, if it decides again next, with `redet`, and if the
  child for (action, real chance key, next decision) exists.
* **Value target** (training): `(1 - q) z + q q_root`, `q = value_q * min(1, iter /
  value_q_warmup)`.
* **Policy target** of a recorded decision: the seat's `policy_target` (section 3.1).

The Rust tree stores everything in flat arrays (no allocation per node; children of an edge as
a linked list matched by a hash of (key, signature) then exactly); a reused subtree keeps the
old arrays until it holds > 65k nodes (`compact`).

### 3.1 Policy targets: visits or completed Q (`policy_target`)

The search (PUCT, noise, move choice, tree reuse) is the same either way; only the `pi` of a
recorded decision (self-play network seat, full search) differs:

* `'visits'` (default): `N_i / sum N` (P0 if nothing was visited). Bit-identical to the
  records before the setting existed (`driver::tests::visits_records_unchanged`: a digest of
  96baefd's toy records; the toy, toy with margin_w / chance_k and Glass Road self-play record
  digests were also compared once, identical).
* `'cq'`: the policy-improvement target of Gumbel AlphaZero (Danihelka et al. 2022, "Policy
  improvement by planning with Gumbel", ICLR; the formula as DeepMind's mctx
  `qtransform_completed_by_mix_value` implements it):

  ```
  pi'      = softmax(ln P0 + sigma(completedQ))
  sigma(q) = (cq_c_visit + max_b N_b) * cq_c_scale * q          (defaults 50, 1.0: the paper's Go / chess values)
  completedQ_a = Q_a = W_a / N_a                                  if N_a > 0
               = v_mix = (v + sum_b N_b * sum_{a:N_a>0} P0_a Q_a / sum_{a:N_a>0} P0_a) / (1 + sum_b N_b)   otherwise
  ```

  P0 = the network's noise-free prior at the root (the Dirichlet-noised P only steers the
  search), v = the network's value of the root for its decider (`vec[decider]`, stored at
  expansion; a reused root keeps the value it was expanded with), Q = the decider's mean
  backed-up utility per root option. Priors are floored at f32's smallest normal (as mctx) so
  a prior that underflowed to 0 stays finite. Computed in f64 (`tree::completed_q_policy`,
  `Tree::cq_target`; Python `mcts.completed_q_policy`, `mcts.cq_target`); no visits -> P0; a
  single option -> 1.

  **No min-max normalisation of Q.** The paper rescales Q to [0, 1] (Go / chess: by the
  known value bounds; mctx's generic default: by the min / max of the node's completed Q,
  with c_scale 0.1). Here values are already P(win) in [0, 1] (with `margin_w` = 0), so Q is
  used as is: sigma is in "P(win) units", so the target sharpens with how much better an
  option is, not with the spread of the options' values. A per-root min-max would stretch any
  spread - a 0.002 difference between two equally good options as much as a 0.3 one - to
  (50 + max N) * c_scale logits, i.e. a near-one-hot target on noise. With `margin_w` > 0 the
  utility leaves [0, 1] by up to `margin_w`; it is still used unnormalised.

  Settings: run setting / seat key `policy_target` ('visits' | 'cq'), `cq_c_visit`,
  `cq_c_scale` (`run.py` passes them to the self-play seat only; evaluation seats record
  nothing). Measured on Glass Road and the toy: docs/az.md 9.

## 4. Bindings and registering a game

A game's binding crate (`py/` in the game repo, a `cdylib` named `<pkg>_py`) is one macro call:

```rust
// glassroad-zero/py/src/lib.rs -> Python module grz._engine
zero_py::engine_module!(_engine, {
    "toy" => zero_search::toy::ToyGame,
    "glassroad" => grz_engine::az::GlassRoadAz,
});
```

with `zero_py`, `zero_search` (path dependencies into the pinned core,
`third_party/zero-core/crates/...`), the game's engine crate and
`pyo3 = { version = "0.23", features = ["extension-module"] }` as dependencies. That gives
Python `Game('glassroad', spec, players, seed)` (clone / legal / step / force_chance /
determinize / encode / result / margin / score / export_state / apply_external / search_tree),
`SearchDriver('glassroad', seats, spec=...)`, `game_names()`, `game_info_of('glassroad')` and
`MAX_SEATS`, and makes the Python reference MCTS, `PyDriver`, the parity check and `run.py
--game glassroad` work (the Python side finds the module through the game package,
`zero.package`). Keep `"toy"` registered: the shims' tests and tools use it. The game's engine
crate depends on `zero_search` for the trait.

## 5. The toy game: "Mini Cavern" (`crates/search/src/toy.rs`)

Built (for Caverna Zero) to exercise what a Rosenberg-style game needs: 2-4 players; worker placement over 6 rounds with
**multi-step turns** (Farm: plow? then a counts decision for sowing; Build, Trade, Feast:
counts decisions INC... DONE whose partial count is part of the state and the features, like
the AZ layers' partial compound responses); **chance revealed mid-game with hidden future
outcomes** (one round card per round from a hidden per-stage order - rounds 1-3 Grain /
Growth / Build, rounds 4-6 Urgent growth / Trade / Feast - and hidden harvest markers for
rounds 2-5: two full harvests, two feed-only rounds; `determinize` reshuffles the unrevealed
ones, `force_chance` replays a reveal); integer **scores with ties** (~5-10% of games).
Rules in the module docs. 18 actions (13 spaces + PLOW, SKIP, INC0, INC1, DONE), 195
features (viewer-relative seats). Bots: `random`, `heuristic` (food planning with a buffer,
growth only with food to spare, rooms to grow, farming, points in the last round: 96% vs
random). Chance keys `round << 16 | card << 8 | marker`.

## 6. Parity: Rust search vs Python reference

`python -m zero.parity positions` (also `tests/test_mcts.py`): 8-12 positions from random
play, both trees with the same deterministic stand-in evaluator (a fixed random linear
function of the features; weight scale 0.15 so the visit distributions are genuinely
stochastic), 96 simulations, 16-24 repetitions per position and tree. Measured in this repo
2026-10-11 in Glass Road Zero (12 positions, 96 simulations, 24 repetitions; Caverna Zero's
2026-10-10 figures in brackets):

| setting | TV(Rust, Python) | TV(Python, Python) floor | root win \|Rust - Python\| |
|---|---|---|---|
| chance_k 0 | 0.0056 (0.0041) | 0.0046 (0.0045) | 0.0006 (0.0004) |
| chance_k 2 | 0.0078 (0.0072) | 0.0060 (0.0061) | 0.0005 (0.0005) |
| Rust c_puct 1.3 (power check) | 0.046 (0.052) | 0.0046 (0.0045) | 0.002 (0.002) |

**Completed-Q targets** (section 3.1; `positions` also averages each tree's `cq` target per
position and compares the means the same way; `cq_formula` evaluates the Python formula on Rust
roots' P0 / N / W / network value). Measured 2026-10-11 (toy: 12 positions, 96 simulations, 24
repetitions; Glass Road: 10 positions, 64 simulations, 20 repetitions):

| setting | cq TV(Rust, Python) | cq TV(Python, Python) floor | visits TV (floor) |
|---|---|---|---|
| toy, chance_k 0, c_scale 1 | 0.0128 | 0.0114 | 0.0056 (0.0046) |
| toy, chance_k 0, c_scale 0.1 | 0.0049 | 0.0045 | |
| toy, chance_k 2, c_scale 1 | 0.0164 | 0.0184 | 0.0078 (0.0060) |
| toy, Rust c_puct 1.3 (power check) | 0.049 | 0.0114 | 0.046 (0.0046) |
| Glass Road, chance_k 0, c_scale 1 | 0.061 | 0.068 | 0.0066 (0.0050) |
| Glass Road, chance_k 2, c_scale 1 | 0.040 | 0.043 | 0.0043 (0.0046) |

Formula: max |Python - Rust| = 3e-8 on toy and Glass Road roots (the f32 rounding of the Rust
result). The c_scale 1 targets' noise floor is ~10x the visit distributions' (a sharp target
moves its mass between options whose Q are close), on Glass Road at 64 simulations most of all.
`tests/test_mcts.py` asserts both TVs, `tests/test_policy_target.py` the formula, the drivers
(Rust and Python: only `pi` changes) and the settings.

`python -m zero.parity h2h --model M --games 6000 --sims 32`: Rust-tree seat vs Python-tree
seat in `PyDriver`, the same network, seats alternating (docs/az.md 6.2): **49.94% +-1.04**
(95%) over 6000 games with chance_k 0 and 50.20% +-1.82 over 2000 games with chance_k 2 (the
toy proof run's iteration 4; the same figures as Caverna Zero's to the last digit, since the
code, the seeds and the reproduced network are the same), and 49.91% +-1.32 over 4000 games
with chance_k 0 at a fresh seed with the stronger iteration 12.

## 7. Throughput (desktop, 2026-10-11, other projects' training runs on the CPU and GPU)

| what | measured |
|---|---|
| Rust search, uniform stand-in evaluator, 4 threads (`bench search`) | 980k simulations/s, 1.65 us per simulation per thread, 1,100 toy games/s at 64/16 sims |
| CPU inference, 128 x 3 net (163k parameters), 1 torch thread (`bench infer`) | 39k / 78k / 108k leaves/s at batch 64 / 256 / 1024 |
| Self-play with the network on CPU (`bench selfplay`, iteration 24, 4 threads, 400 games) | 43k leaves/s, 208k games/h, inference 91% of the time (average batch 84) |
| Self-play in the proof run (docs/az.md 6) | median 62k leaves/s, ~290k games/h; inference-bound (search threads mostly wait) |
| CUDA inference | not measured here (the GPU belongs to Caverna Zero's live run); Caverna Zero measured 12k / 45k / 161k leaves/s at batch 64 / 256 / 1024 (eager) |

Torch must run with 1 intra-op thread on this machine while other jobs saturate the CPU:
with 2-4 threads the same forward passes were 10-1000x slower (each op waits for preempted
threads); `run.py`'s `torch_threads` default is 1.

## 8. Integrating a game

docs/new_game.md: implement `SearchGame` in the game's engine crate, register it in the game's
binding crate (section 4) and its Python game package (`zero.package.GamePackage`: presets,
external players, defaults), pin the core (docs/pinning.md). The game-specific analysis lives
with each game: Caverna Zero docs/search.md 8 (what its AZ layer provides), Glass Road Zero
docs/search.md 8-9 (chance and revelation keys, determinization and the viewer, hidden hands in
the tree, simultaneous choices, cost).

## 10. Opponent models and analysis options (network-seat options, all off by default)

Hidden simultaneous choices need more than a uniform determinization (Glass Road Zero
docs/hidden_info.md: its 5-of-15 hand choice `select`, sequentialised with seat 0 first, decided
self-play games; perfect knowledge of the opponent's hand was worth ~+100 Elo at 128
simulations). The core offers two opponent models and the analysis options that measured the
problem, each a network-seat option off by default, each behind trait hooks a game opts into
(`HAS_*` constants: a seat option for a game without the hooks is refused - it would silently
do nothing). With every option off the search is bit-identical to the plain one: tested
(the toy's golden records; `tree::tests::opponent_model_options_off_are_the_plain_search`;
Glass Road self-play digests - uniform / random / real network, visits and completed-Q targets,
`redet` on and off, `intro` both, and with every new option given explicitly off:
`tools/selfplay_digest.py --seat-opts`).

| seat option (`games.nn_seat` key) | trait hooks | what |
|---|---|---|
| `psample` | `HAS_PSAMPLE`, `policy_sampled`, `uncommit_simultaneous` | other seats' sequentialised simultaneous choices are drawn from the network's prior inside the search (10.2) |
| `belief_mix`, `belief_temp`, `belief_particles` | `HAS_BELIEF`, `belief_start`, `belief_particle`, `belief_target`, `belief_consistent`, `determinize_particle` | that share of the root worlds takes another seat's hidden choice from particles sampled with the network's own policy (10.3) |
| `oracle_hands`, `oracle_p` | `DECISION_CLASSES`, `decision_class`, `HAS_ORACLE`, `determinize_oracle` | analysis: root worlds keep the true hidden information at decisions of the given classes (10.1) |
| `class_sims` | `DECISION_CLASSES` | full-search simulations per decision class (10.1) |
| `mask_features` | - | input ablation: feature indices zeroed in every network input of the seat (10.1) |
| driver `trace` | - | results carry the game's seed and real moves (section 2.5) |

`run.py` settings `psample`, `belief_mix`, `belief_temp`, `belief_particles` (docs/az.md 1.2)
pass the opponent models to the run's self-play and evaluation seats alike (the agent's search),
so a training fork switches them on with `tools/fork_run.py --set psample=1 --set
belief_mix=...`. `zero.h2h` measures them: one checkpoint under two seat settings
(`--a-opts` / `--b-opts`), or against an external player (`ext:jsbot`), with A's score by seat,
the seat-adjusted Elo edge and the driver's counters.

### 10.1 Analysis options (ported from Glass Road Zero's branch `oracle`)

* **Decision classes**: `DECISION_CLASSES` names the game's classes, `decision_class()` gives the
  pending decision's (Glass Road: `play` = everything outside `select`, `select` = the hand
  picks). Seat keys taking class names: `oracle_hands`, `class_sims`.
* **Oracle** (`oracle_hands`: False / True / 'all' / a class name / 'select+play' / a list ->
  `TreeOpts::oracle`, a mask over classes; `oracle_p` in (0, 1]): at a root decision of a masked
  class the root world comes from `determinize_oracle` (the true hidden information; the game
  still reshuffles what nobody knows, e.g. Glass Road's building stacks) in an `oracle_p` share
  of the simulations, `determinize` in the rest. The network input stays honest (`encode` never
  reads hidden information). Analysis only: what is the hidden information worth?
* **`class_sims`** ({class: simulations}): full searches at decisions of that class use that
  many simulations (others `sims`).
* **`mask_features`**: the listed inputs are zeroed in every leaf, belief row and record of the
  seat (Glass Road: `[1697]` = `opp_selected`, the role-blind select before FEATURE_VERSION 1
  made it constant).
* **Traces** (`trace=True`): replayable games for offline probes (Glass Road
  `grz.az.handprobe`).

### 10.2 Policy-sampled simultaneous choices (`psample`)

A simultaneous choice that the game sequentialises (Glass Road `select`: seat 0 picks its hand,
then seat 1; each hand as 5 ascending atomic picks) puts the later chooser's decision below the
earlier chooser's edge in every tree. Two wrong opponent models follow: when the root decider
chooses first, the opponent's later choice is a PUCT node whose statistics are conditioned on
the decider's hidden choice in the same tree (in-tree clairvoyance: it counter-picks our actual
hand); when the root decider chooses second, `determinize` resamples the earlier choice
uniformly (a random opponent). With `psample`:

* **Walk**: at an expanded node whose decider is not the root decider and whose decision is
  `policy_sampled()`, the option is drawn from the node's P0 (the network's noise-free prior,
  evaluated from that seat's view - which never shows the root decider's hidden choice) instead
  of PUCT. N / W / backups as usual (reports, reuse); the root is never sampled; a `simulate_with`
  hook still has precedence. No extra network calls: those nodes are expanded as leaves anyway.
* **Root worlds**: after `determinize(viewer)`, `uncommit_simultaneous(viewer)` takes back the
  other seats' hidden commits of the same step (Glass Road: the earlier chooser's `select`
  pick), which are then made after the viewer's choice, sampled as above. Not in oracle worlds.
* So every search models the opponent's simultaneous choice the same way - a draw from the
  network's own (role-blind) policy, never seeing our choice - and best-responds to it. Inside
  the tree, later rounds' choices of the other seat are sampled whether it chooses first or
  second there; the root decider's own later choices there stay PUCT (they may see the other
  seat's sampled choice: a mild optimistic leak deep in the tree).
* **Contract**: `policy_sampled` depends only on the decision (the node); `uncommit_simultaneous`
  keeps the viewer's decider and legal actions (the root check of every simulation enforces it)
  and its features. It needs features that do not reveal whether the other seat has committed
  (Glass Road: `opp_selected` is always 0 since 2026-10-12): otherwise the network would see a
  different decision in the un-committed worlds.
* **Records**: the search changes, so the policy targets of the simultaneous choices change
  (best responses to the policy instead of to a clairvoyant / uniform opponent): a training
  fork, not a live switch.
* **Reference and parity**: `zero/mcts.py` (`Tree(psample=True)`) implements the same rule;
  `python -m zero.parity positions --psample` / `parity.positions(..., psample=True, poss=...)`.
  On the secret game's hand-choice positions (`tests/test_opponent_models.py`, 6 positions, 96
  simulations, 24 repetitions) and Glass Road's `select` positions of both choosers (its
  `tests/test_opponent_models.py`, 6 positions, 64 simulations, 16 repetitions, stand-in scale
  0.05), TV(Rust, Python) stays within 1.6x the Python-Python floor, and the check has power
  (Rust with `psample` vs without: > 3x the floor).

### 10.3 Belief prior (`belief_mix`, `belief_temp`, `belief_particles`)

`determinize` samples another seat's hidden choice uniformly among the ones consistent with what
the viewer observed, but the choice was made by a policy - in self-play, the network's own. The
belief prior samples it from that policy, restricted to the consistent choices, with a uniform
floor:

* **Snapshot** (driver): whenever a seat starts a hidden choice (`belief_start()` = Some(key) at
  its first decision of it; Glass Road: a seat's first `select` pick, key = the period), the
  driver keeps a copy of the real state per (chooser, key) - whatever the chooser's seat kind
  (network, bot, external: the JS bot's hands get a snapshot too). Only when some seat uses the
  belief.
* **Particles** (`zero_search::belief::BeliefJob`): when a belief seat's search starts at a
  decision whose `belief_target(seat)` is (q, key) and no particles are cached for it, the driver
  replays q's choice from the snapshot `belief_particles` (K) times, drawing every atomic option
  of q from the network's prior at q's state (encoded from q's view, the seat's `mask_features`
  applied), tempered p^(1/`belief_temp`), single-option decisions auto-stepped, until
  `belief_particle(q)` gives the finished choice. One level (one decision of q) per network
  batch: the distinct states of the level go into the seat's ordinary network batch as **belief
  rows** (the leaf layout; the evaluator cannot tell them apart; only their priors are used);
  identical prefixes are evaluated once, so a sharp policy costs few rows. The particles are
  cached per (seat, chooser, key) for the rest of the game; a rolling epoch drops pending belief
  rows (re-sampled with the new network) but keeps the cache.
* **Worlds** (`TreeOpts::belief_mix`, `Tree::set_belief`): at every search start (and a reused
  tree's restart) the particles `belief_consistent` with the root state are kept, repeats
  included, so a uniform draw among them samples the policy prior conditioned on the observations
  (exact restriction, as the consistent set is exact). Each root world: with probability
  `belief_mix` `determinize_particle` with a uniformly drawn consistent particle, else
  `determinize` (the uniform floor). No consistent particle: `determinize`. The oracle has
  precedence; `psample`'s un-commit applies after.
* **Counters** (`SearchDriver.stats()`): `belief_rows` (evaluated), `belief_jobs` (beliefs
  sampled), `belief_targets` (searches at a decision with a belief target), `belief_searches`
  (of those, with >= 1 consistent particle), `belief_missing`
  (targets without a snapshot: uniform worlds instead, never silently), `belief_worlds` (root
  worlds from a particle).
* **Cost**: K x (choice length) rows at most per belief, far fewer with identical prefixes
  (Glass Road, K = 64, 5 picks: measured in 10.5).
* **Approximations**: tempering per atomic choice is not the tempered joint distribution; K is
  finite (a choice the policy gives < 1/K is usually absent, which the uniform floor covers);
  the floor's weight is fixed (not re-weighted by how well each component predicted the
  observations).
* The Python reference driver (`play.PyDriver`) has no belief prior (nor the analysis options).

### 10.4 For a game: what to implement

* `psample`: `HAS_PSAMPLE = true`; `policy_sampled()` true at every atomic decision of a
  sequentialised simultaneous choice whose options do not depend on the other seats' hidden
  choices of the same step; `uncommit_simultaneous(viewer)` in a determinized world where the
  viewer is a later chooser: clear the earlier choosers' hidden commits and order the step so the
  viewer decides first (no-op otherwise). The features must not reveal commits.
* Belief: `HAS_BELIEF = true`; `belief_start` at the first decision of a hidden choice (a key the
  target can name: e.g. the round); `belief_particle(q)` = the finished choice as a u64 (a bit
  mask of cards); `belief_target(viewer)` = (q, key) while `determinize(viewer)` resamples that
  choice; `belief_consistent` = the exact consistency test (the true choice always passes);
  `determinize_particle` = `determinize` with the choice set to the particle.
* Tests as Glass Road's `engine/tests/az_opponent.rs`: un-committed worlds play on to the end and
  keep the viewer's options and features; the true choice is always consistent; particle worlds
  keep the viewer's view; and the Python parity with `psample` at the simultaneous decisions.
* The core's test bed is `zero_search::secret::SecretGame` (Rust tests in `secret.rs`,
  `belief.rs`, `tree/tests.rs`, `driver/tests.rs`; Python `tests/test_opponent_models.py`).
