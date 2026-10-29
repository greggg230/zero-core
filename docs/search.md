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
| `crates/search/src/toy_team.rs` | "Toy team", a two-team game with a deferred result decided by toy games (section 5.1) |
| `crates/search/src/dynapi.rs` | Object-safe wrappers (`DynGame`, `DynTree`, `DynDriver`) so the bindings hold any game |
| `crates/search/src/rng.rs`, `noise.rs` | xoshiro256++ / splitmix64, Gamma / Dirichlet |
| `crates/py/src/lib.rs` | `zero_py`: `Game`, `SearchTree`, `SearchDriver`, `game_names()`, `game_info_of()`, the `engine_module!` macro (section 4) |
| `crates/toy_py/src/lib.rs` | the core's own module `zero._engine` (`toy`, `toy_team`) |
| `python/zero/mcts.py`, `play.py` | Python reference MCTS and driver (`PyDriver`, seats searching with the Python or the Rust tree) |
| `python/zero/protocol.py` | Leaf-batch protocol: `LeafBatch`, validators, `MAX_SEATS` |
| `python/zero/parity.py` | Rust-vs-Python statistical parity (section 6) |

Build: `bash tools/build_py.sh` (the toy module: release cdylib copied to
`python/zero/_engine.pyd`; target dir `$CARGO_TARGET_DIR` or `E:/cargo-target/zero-core/<branch>`
on Windows). A game builds its own `<pkg>._engine` with its `tools/build_py.sh` (docs/pinning.md).
Tests: `cargo test` (the `zero_search` crate: 40 tests, ~5 s in release), `python -m pytest
tests` (101 tests on the toys, ~2.5 min on the loaded desktop).

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
    fn minor_decision(&self) -> bool { false }                      // the pending decision is minor (sims_minor)
    fn sims_divisor(&self) -> u32 { 1 }                             // the pending decision's simulations divided
    fn deferred(&self) -> bool { false }                            // over, the result from outside (1.1)
    fn deferred_payload(&self) -> Result<String, String> { Ok(String::new()) } // what decides it (JSON)
    fn resolve(&mut self, results: &str) -> Result<(), String> { Err(..) }     // the outside's results
    fn team_size(&self) -> usize { 1 }                              // seats per team (1.1)
    fn export_state(&self) -> String { .. }                         // external seats: the state (JSON)
    fn apply_external(&mut self, payload: &str, chance: &mut Vec<u64>) -> Result<(), String> { Err(..) }
    fn bot_names() -> &'static [&'static str] { &[] }               // built-in bots ('random' is generic)
    fn make_bot(spec: &Self::Spec, name: &str, seed: u64) -> Option<Box<dyn Bot<Self>>> { None }
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
* `minor_decision` (2026-10-22, MTG Zero M5b): the pending decision matters little (MTG: which
  mana source to tap next while paying). A self-play network seat with `sims_minor` (run
  setting, 0 = off) searches it with that many simulations and never records it (section 3).
  Games that do not implement it (every game but MTG) are unaffected, as are seats without the
  setting (`driver::tests::minor_decisions_get_their_own_budget_and_no_records`: the golden
  records are unchanged).
* `sims_divisor` (2026-10-27, MTG Zero M6 part 3): a lighter class of decisions. Every network
  seat (self-play or evaluation) searches the pending decision with its drawn budget (`sims`,
  `sims_fast` or `sims_minor`) divided by it, at least one simulation; the decision is recorded
  as usual (a full search stays a sample). MTG Zero's `team_draft` returns its spec's
  `build_sims_div` (2 in its run presets) during deckbuilding: picks at 64 / 16, deckbuilding
  at 32 / 8 (its DESIGN.md 2.3.9). Default 1: every game but `team_draft` is unaffected
  (`driver::tests::light_decisions_divide_the_simulations_and_are_recorded`: the golden records
  with divisor 1).
* `deferred`, `deferred_payload`, `resolve`, `team_size` (2026-10-27, MTG Zero M6 part 2:
  section 1.1). Games that do not implement them (every game but MTG Zero's `team_draft` and the
  core's `toy_team`) are unaffected: the golden records are unchanged, and a single-game run
  (toy-smoke, rolling and classic, with margin_w / chance_k / cq) produces the same metrics,
  checkpoints, optimizer state and sample shards byte for byte as zero-core 681ba8d.
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

### 1.1 Deferred results and teams (MTG Zero M6, 2026-10-27)

A game whose result is decided **outside** it: MTG Zero's `team_draft` (an event drafted and
built by two teams) ends when the decks are built, and its result is the outcome of the event's
`mtg` matches, played later by another network (co-training: docs/az.md 11). Such a game is
over (`to_move()` None) at a **deferred terminal**: `deferred()` is true until `resolve(results)`.

* `deferred_payload()` (JSON) says what must be played: by the run's convention
  `{"episodes": [<a spec of the resolver game>, ...], ...}` (the run's game package may read
  another format: `GamePackage.deferred`, docs/az.md 11). The driver calls it once, when it
  parks the game (an error there stops the driver, naming the game).
* `resolve(results)` takes the episodes' results (JSON, in the payload's order; the run sends
  `[{"result": [...], "scores": [...]}, ...]`) and sets the outcome; afterwards `deferred()` is
  false and `result()`, `margin()`, `score()` hold it. A malformed or impossible result is an
  error, never a default (the driver keeps the game parked and reports it).
* While deferred, **nothing reads `result()` or `margin()`**: the search evaluates the terminal
  with the network (section 3.2), the driver parks the game (2.8), the bindings refuse
  `Game.result()` / `margin()` (ValueError). A game should make them fail loudly there
  (`toy_team` and `team_draft` panic).
* `team_size()` = T: seats per team, seat `s` in team `s % (num_players() / T)` (MTG Zero's A1
  B1 A2 B2 ... seating: team `s % 2`). `result()[s]` is the team's outcome divided by T (the
  vector still sums to 1: the value head stays a softmax over seats), and the search's utility
  is `T x win[s]` (+ the margin term, section 3), so Q stays in [0, 1] and `c_puct`,
  `cq_c_scale` keep their meaning. Evaluation seats teams alternately with this rule
  (docs/az.md 11). Default 1: every seat alone, the utility as before (bit for bit).

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
drv.take_deferred(); drv.resolve(gid, results); drv.pending     # deferred games (2.8)
drv.enqueue(tag, spec); drv.queued                              # queued setups (2.9; queue_only=True)
```

`infer.drive(drv, evaluators, externals, check=..., on_batch=...)` is the loop:
network batches go to `evaluators[key](lb)`, external batches to `externals[key](lb)`.
`play.PyDriver` (the Python reference) has the same interface.

### 2.1 Seats

One list of seat dicts per game (players = its length):
`{'bot': 'random' | 'heuristic' | ...}`, `{'external': '<key>'}`, or a network seat
`{'key': 'm', 'sims', 'c_puct', 'selfplay', 'alpha', 'eps', 'temp_moves', 'sims_fast',
'sims_minor', 'p_full', 'chance_k', 'margin_w', 'redet', 'policy_target', 'cq_c_visit', 'cq_c_scale',
'select', 'select_min_visits'}` (`games.nn_seat`; `policy_target` only matters for recorded
decisions, section 3.1; `select` and its guard for the moves the seat does not sample, section
3.3; `cq_c_visit` / `cq_c_scale` are the completed-Q transform of both). Evaluator keys: every distinct
network `key` is one evaluator (arena / Elo matches: `'m'` vs `'o'`), every distinct external
key one external player. A key cannot be both.

### 2.2 Network batches

| field | dtype / shape | content |
|---|---|---|
| `x` | f32 [B, F] | `encode(decider)` of the (determinized) leaf state |
| `opt_off` | i32 [B+1] | leaf b's legal actions = `opt_action[opt_off[b]:opt_off[b+1]]` (>= 2; none at a deferred terminal, 3.2) |
| `opt_action` | i32 [sum] | global action ids in the game's order |
| `player`, `n_players`, `gid` | i32, i32, i64 [B] | decider (absolute), players, game index |

Answer: `priors` f32 [sum] (softmax over each leaf's legal actions), `value` f32 [B, 8] =
P(win) per seat **relative to the decider** (column j = seat (player + j) % n; columns >= n
are 0), `margin` f32 [B, 8] (same order). `NetEvaluator` does the gather + segment softmax on
the device. A row with an empty option segment is a deferred terminal's leaf: its value and
margin are used, it has no priors (`NetEvaluator`, the CUDA-graph path and `UniformEvaluator`
handle empty segments; `protocol.validate_batch` allows 0 or >= 2 options).

### 2.3 Rules of the exchange

* `next_batch` advances every active game until its search needs ONE network evaluation, an
  external seat is to move, or the game ends: one leaf per game per batch, no virtual loss, no
  in-tree batching (the mtg_bot batch-32 lesson). Bots and single-option decisions are played
  inside the driver.
* The games are split into `groups`; each group's batches (one per key) must be answered
  before that group advances again; with 2 groups the search threads work on group B while
  Python evaluates group A. `next_batch` releases the GIL while waiting; groups are served in
  a fixed round-robin order, except by a `queue_only` driver (2.9), which serves the first of its
  groups that is ready: a game whose step takes long (an engine's slow path) then holds up its
  own group only, not every game of the driver (docs/az.md 11.8). `stats()` reports the longest
  step of a game so far (`step_max_s`, `step_max_gid`) and the steps over 10 s (`slow_steps`).
  `in_flight()` never waits for a search thread (a chunk being searched counts its games as of
  its group's last dispatch).
* Batch assembly: every group is split into chunks (<= 4 x threads); the last chunk's worker
  lays out the batches and queues one fill job per chunk at the front of the queue, so all
  threads copy rows in parallel (fixed-width rows: one memcpy per chunk).
* Reproducible: every game's setup, search, noise and bot streams derive from (seed, gid)
  (one bot seed per seat whatever the seat kind); results do not depend on `threads` /
  `groups` (tested).
* Errors (an illegal move, a determinization that changed the root decision, a bad external
  payload, a worker panic) stop the driver: `next_batch` raises with the game index and seed.
  A panic inside a game (an engine bug) is caught per game ("game G (driver seed S): search
  worker panicked: <message>"; the panic's location is on stderr); recording an error wakes a
  caller already waiting in `next_batch`, which raises at once (it never waits forever).

### 2.4 Training samples (`take_samples`)

One row per recorded root decision (self-play network seat, full search), from the real
state: `x` [N, F], `player`, `n_players`, `opt_off` (i64 [N+1]), `opt_action`, `pi` (the
seat's policy target over the options: the visit distribution, or the completed-Q target with
`policy_target='cq'`, section 3.1), `z` [N, 8] (final `result()` relative to the decider), `q`
[N, 8] (root search win estimate `wsum / tot`, relative; = z if no backups), `sc` [N, 8]
(`margin()` per relative seat), `net` (network id of the search, rolling mode). A game's
samples are emitted when it ends.

### 2.5 Results (`take_results`)

`{gid, n, over, result, scores, steps, decisions, ext_moves, rounds, samples, tag}`; a game cut
at `max_steps` has `over` False and result 1/n each; `tag` is the tag of the queued setup the
game started from (2.9), else None. A deferred game's row comes when it is resolved (2.8).
With the driver's `move_log` N (a diagnostic, default 0 = off: nothing changes), a row also
has `moves`: its first N real moves as `(seat, action, legal actions)` (the external seats'
answers excluded; a resolved game's: none). `zero.h2h --move-log`.

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

### 2.8 Deferred games: parking and `resolve`

A game that ends at a deferred terminal (1.1) is **parked**: its training records (features
stored as float16: the shards store them so anyway, so nothing written changes; `Record` ->
`ParkedRecord`), root-search win estimates (`q`), steps and counters, and its slot starts the
next game (rolling) at once. `take_deferred()` -> `[(gid, payload)]` hands each parked game out
once; `resolve(gid, results)` calls the game's `resolve` and then emits its result row and
samples **exactly as a finished game's** (`z` from `result()`, `q` from the root values, `sc` from
`margin()`, relative to each record's decider; the shared `worker::emit`). `pending` counts parked
games; `parked_game(gid)` (Rust) shows one. Rolling quotas (2.6) count finished and **resolved**
games, never parked ones, so a phase ends when enough games have been resolved; parked games
survive phases. `finished` is false while games are parked. A game the driver did not park, a
result the game refuses (it stays parked) or a game still deferred after `resolve` is an error.
Stats: `parked`, `parked_bytes` (the records' memory), `deferred` (parked so far), `resolved`.
The Python reference driver (`play.PyDriver`) does not park: it raises at a deferred terminal.

### 2.9 Queued setups (`enqueue`, `queue_only`)

`enqueue(tag, spec)` queues a game setup: the next game a free slot starts takes that spec (JSON
or a dict; parsed at once, an invalid spec is an error at `enqueue`) instead of the driver's,
with the driver's seats and its (seed, gid) streams; its result row carries `tag`.
`queue_only=True` (rolling drivers only) makes slots start nothing but queued setups: with the
queue empty they stay idle, and `next_batch` returns None once nothing is left to play (the
phase has not ended: `enqueue` wakes the idle groups unless the phase ended). The groups share
the queued setups: a group dispatched starts its share of everything in flight or queued, never
fewer than the other groups' free slots cannot take (so the groups' search and inference overlap
when fewer setups than slots arrive at once - an evaluation's matches - and no slot idles while
setups wait); a setup's gid is its place in the queue. Co-training's
resolver driver is one (docs/az.md 11): it plays the episodes of parked games, tagged
`<gid>/<k>`, with the play network on every seat.

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
  util = T x win + margin_w * tanh(margin) (network margin, or `margin()` at terminals; T =
  `team_size`, 1 but in team games: section 1.1, `leaf_values_team`, `Tree::team`).
* **Per decision**: single-option decisions are not searched; playout cap (`p_full` share
  gets `sims` and is recorded, the rest `sims_fast`; a minor decision of a self-play seat with
  `sims_minor` gets `sims_minor` and is never recorded, the playout-cap draw being made all the
  same; the budget is then divided by the game's `sims_divisor`, at least one); root Dirichlet
  noise on full self-play
  searches (also on a reused, already expanded root); self-play samples moves by visit count
  for the first `temp_moves` searched decisions of the game, otherwise (and in evaluation) by
  the seat's move rule (section 3.3; the default argmax (N, P0) with the first maximum on
  ties); tree reuse: after a real move the tree
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

### 3.2 Deferred terminals in the tree

A walk whose transition ends the game at a deferred terminal (1.1) does not read a result:
it reaches a child node of **no options decided by the tree's viewer**, keyed like any child
(chance key of the transition, the viewer, no actions; it cannot meet a decision node, whose
options are never empty). Unexpanded, it is a **leaf** like any other (`Sim::Leaf { node, player:
viewer }`): the network evaluates `encode(viewer)` of the (determinized) terminal state - its own
estimate of the outcome, the quantity it learns from the resolved results - with an empty
option segment (no priors). `expand_leaf` stores the utility in `vec` and the win vector in
`wsum` (`tot` = 1); every later walk that reaches the node backs that stored value up
(`Sim::Terminal`) instead of evaluating again. Worlds that reach the terminal through the same
actions and keys share the node, as decisions do; the encoding is the viewer's knowledge, so it
is the same in all of them. The Python reference (`mcts.simulate`) does the same with
`Node(viewer, [])`; Rust and Python agree exactly where nothing is left to chance and
statistically elsewhere (section 6). `ToyTeam::result()` panics at the terminal, so the tests
prove the search never calls it.

### 3.3 The move rule: visits or completed Q (`select`; 2026-10-28)

A network seat's move that is not sampled - every move of an evaluation seat, and a self-play
seat's after its first `temp_moves` searched decisions of the game - is chosen from the root's
statistics by the seat's rule (`driver::Select`; the search and the recorded targets are the
same whatever the rule):

* `'visits'` (default): argmax (N, P0), the first maximum on exact ties - AlphaZero's rule and
  the only one before the setting existed, bit for bit (`driver::tests::select_visits_and_the_
  move_log_change_nothing`: the golden records and an evaluation match's digest, the same as on
  zero-core 3c26d9f; a toy-smoke run byte for byte, docs/az.md 5.4).
* `'cq'`: the argmax of the logits of the completed-Q target (section 3.1: `ln max(P0, floor)
  + (cq_c_visit + max N) cq_c_scale completedQ`; Gumbel AlphaZero's final move is the argmax of
  the same quantity over its considered actions) among the options with at least
  `select_min_visits` visits - `min(select_min_visits, max N)`, so the most visited option always
  qualifies; 0 (default) = every option, an unvisited one completed at v_mix - the first maximum
  on exact ties. With few simulations the prior's favourite collects most visits even when the
  search found it worse, and `visits` plays it anyway; `cq` lets the search's Q overrule it once
  the Q gap outweighs the log-prior gap (sigma is ~(50 + max N) logits per unit of P(win)). Its
  weak spot: a single-visit option with a lucky Q wins too - the guard (`select_min_visits` 2:
  options seen at least twice) is the remedy, measured in MTG Zero's ENGINE.md 57.
* No visits at all: both rules take the prior's argmax; no options: index 0.

Code: `Select::choose(p0, n, w, v)` (driver.rs; the driver's `decide` and the bindings'
`SearchTree.root_select(select, min_visits, c_visit, c_scale)` both call it), Python
`mcts.select_index` / `select_node` (the reference; `play.PyDriver`'s 'py' seats use it, its
'rs' seats the Rust tree's `root_select`), `mcts.select_tag` (the rule's name: 'visits', 'cq',
'cq-n2' with a guard, 'cq-v25-s0.5' with another transform). Seat keys `select` ('visits' |
'cq'), `select_min_visits` (cq's only: an error with 'visits'), and the rule's transform
`select_c_visit`, `select_c_scale` (absent: the seat's `cq_c_visit`, `cq_c_scale`, shared with the
cq target as in Gumbel AlphaZero; cq's only; the target never reads them - `Select::Cq` and
`PolicyTarget::CompletedQ` hold their own). Run settings: `eval_select` (every evaluation seat),
`selfplay_select` (self-play after `temp_moves`), their `*_min_visits` and their transforms
`*_c_visit` / `*_c_scale` (unset: `cq_c_*`; docs/az.md 1.2, 5.4, 12.5). Parity (section 6): the Python formula on Rust roots gives
the Rust driver's move on every root and rule checked.

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
random). Chance keys `round << 16 | card << 8 | marker`. Spec `{}` or `{"food": [a, b, ...]}`
(extra starting food per seat: `toy_team`'s matches; the default spec plays as before).

### 5.1 The toy team game: "Toy team" (`crates/search/src/toy_team.rs`, `toy_team`)

The core's test bed for deferred results, teams and co-training, shaped like MTG Zero's
`team_draft`: 2 or 4 seats, seat `s` in team `s % 2` (team size n / 2). Each of `rounds` rounds
(spec `{"rounds": 1-3}`, default 2) deals an offer of n + 1 cards from a shuffled deck of 16
(card c is worth c % 4 food), every seat picks one (round r starts at seat r % n), the last card
is discarded. Then the game is at its **deferred terminal**: the payload lists toy matches,
A player i (seat 2i) against B player (i + d) mod T for every rotation d (1v1: one match, 2v2:
four), each player with its picks' total as extra starting food (`{"episodes": [{"food": [a,
b]}, ...], "pairs": [...]}`); `resolve` takes their results (1/0, 0/1, 0.5/0.5 each): a team wins
with more match wins (`result` 1/T or 0 per seat, 1/2T each on a draw), margin = (team match wins
- opponent match wins) / matches. Hidden information: the undealt deck (`determinize`
reshuffles it); each later offer is a chance key `round << 32 | mask` (`force_chance` deals a
queued offer if its cards are undealt). `result()` / `margin()` panic before `resolve`. 16
actions (card ids), 64 features. Bots: `random`, `greedy` (the most food).

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

**Deferred leaves and teams** (`python -m zero.parity positions --game toy_team --players 4
--depth 0 8`; `tests/test_deferred.py`), 2026-10-27, 96 simulations: positions in the last round
(nothing left to chance) search identically in Rust and Python - the same visit counts, root
win (1e-9) and completed-Q targets (1e-6), with deferred leaves evaluated once per node (12 of
12 positions, 200 simulations) - and over the whole game the statistics sit at the noise floor:

| setting | TV(Rust, Python) | TV(Python, Python) floor | root win \|Rust - Python\| | cq TV (floor) |
|---|---|---|---|---|
| 2v2, 12 positions, 32 reps | 0.0066 | 0.0072 | 0.0006 | 0.037 (0.032) |
| 1v1, 12 positions, 32 reps | 0.0033 | 0.0034 | 0.0005 | 0.023 (0.031) |
| 2v2, Rust c_puct 1.3 (power check) | 0.029 | 0.0029 | 0.0017 | 0.037 (0.016) |

(Positions whose search is deterministic count as 0 for both; the others carry the noise.)

**The move rule** (section 3.3; `python -m zero.parity select`, `parity.select_formula`,
`tests/test_select.py`), 2026-10-28: on 8 toy positions searched by the Rust tree at 4, 8, 24
and 64 simulations, the Python `select_index` on each root's P0 / N / W / network value gives
the Rust driver's own move (`root_select`) for all 192 (position, simulations, rule) cases -
'visits', 'cq' with guards 0, 1, 2 and 4, and transforms (50, 1), (10, 3), (50, 0.1); no
near-ties - and the rules do differ on these roots (cq != visits, the guard != no guard). Since
2026-10-29 also guard 4 at (50, 0.3), the rule of MTG Zero's Delver line (224 cases), and
`tests/test_select_scale.py` plays it with its own transform beside a 1.0 target (docs/az.md
12.5).
In games (`python -m zero.parity h2h --select cq`, a toy-smoke checkpoint, 32 simulations,
Rust-tree seat vs Python-tree seat, both 'cq'): 52.1% +-2.0 (95%, 2000 games, seed 0), 49.7%
+-1.2 and 50.8% +-1.2 (6000 games each, seeds 1, 2) - 50.5% over the 14,000; the same seeds with
'visits': 49.7% +-1.2, 50.7% +-1.2. `tests/test_select.py` asserts both checks.

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
