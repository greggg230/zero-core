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
| `crates/search/src/tree.rs` | One search tree: PUCT, chance children, determinization, noise, reuse, `compact`, completed-Q targets, the root minimum visits (3.5), the Gumbel root search (3.7) |
| `crates/search/src/driver.rs`, `driver/worker.rs` | Batched multi-threaded driver: groups, chunks, parallel batch assembly, playout cap, temperature, records, rolling phases, arena keys, external seats |
| `crates/search/src/toy.rs` | "Mini Cavern", the toy game (section 5) + its heuristic bot |
| `crates/search/src/mull_game.rs` | Test games (cfg(test)): keep-or-mulligan (`Mull`, section 3.5) and its private-hand variant (`HiddenMull`, section 3.6) |
| `crates/search/src/lookahead.rs`, `peek_game.rs` | The exhaustive lookahead per information set (section 3.9: settings, the level-by-level state machine, its network rows, the budget's fallbacks) and its test game (cfg(test)) |
| `crates/search/src/toy_team.rs` | "Toy team", a two-team game with a deferred result decided by toy games (section 5.1) |
| `crates/search/src/dynapi.rs` | Object-safe wrappers (`DynGame`, `DynTree`, `DynDriver`, `DynLook`) so the bindings hold any game |
| `crates/search/src/rng.rs`, `noise.rs` | xoshiro256++ / splitmix64, Gamma / Dirichlet, Gumbel |
| `crates/py/src/lib.rs` | `zero_py`: `Game`, `SearchTree`, `SearchDriver`, `Lookahead` (3.9), `game_names()`, `game_info_of()`, `lookahead_settings()`, the `engine_module!` macro (section 4) |
| `crates/toy_py/src/lib.rs` | the core's own module `zero._engine` (`toy`, `toy_team`) |
| `python/zero/mcts.py`, `play.py` | Python reference MCTS and driver (`PyDriver`, seats searching with the Python or the Rust tree) |
| `python/zero/protocol.py` | Leaf-batch protocol: `LeafBatch`, validators, `MAX_SEATS` |
| `python/zero/parity.py` | Rust-vs-Python statistical parity (section 6) |
| `python/zero/lookahead.py` | The lookahead's settings and position API (section 3.9) |

Build: `bash tools/build_py.sh` (the toy module: release cdylib copied to
`python/zero/_engine.pyd`; target dir `$CARGO_TARGET_DIR` or `E:/cargo-target/zero-core/<branch>`
on Windows). A game builds its own `<pkg>._engine` with its `tools/build_py.sh` (docs/pinning.md).
Tests: `cargo test` (the `zero_search` crate: 88 tests, ~5 s in release), `python -m pytest
tests` (the toys, ~8 min on the loaded desktop).

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
    fn determinize_with(&mut self, viewer: usize, rng: &mut Rng, belief: &Belief) { determinize } // dealt by a belief (3.8)
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

### 1.2 Auxiliary targets (opt-in; MTG Zero's belief head, 2026-10-29)

A game may declare per-decision **auxiliary targets** (docs/az.md 15): `const AUX` names the
segments `(name, width)` of a label vector of `aux_width()` floats, and `aux_target(viewer, out)`
writes it for the pending decision from the **true** state. Unlike everything else the search
reads, it may hold what `viewer` cannot see - MTG Zero's `belief` is the opponent's actual hand by
card identity - because it is a training label, never an input: `encode` stays blind to it (a
game tests that worlds determinized for the decider encode alike while their targets differ: the
toy's `aux_target_is_the_hidden_next_card_and_never_an_input`, MTG Zero's
`the_belief_target_is_the_true_opponent_hand_and_never_an_input`). The toy's target `next_card` is
the next round's card, hidden until its round (a one-hot over the 6 cards; zero in the last round).
Default: no targets (`AUX` empty). The loss of each segment is the game package's (zero/package.py
`aux`). A driver created with `aux` records it (2.4); `Game.aux_target(viewer)` returns it.

## 2. The driver protocol (Python <-> Rust)

```python
from zero import games as G       # in a game repo: from <pkg>.az import games as G
drv = G.Driver('toy', seats, spec=None, concurrency=256, seed=0, record=False, threads=4,
               groups=2, max_steps=100_000, rolling=False, net=0)   # (+ queue_only, move_log, aux, paired, sparse: 2.9, 2.5, 1.2, 2.10, 2.2)
lb = drv.next_batch()            # LeafBatch (kind 'nn' | 'ext') or None when every game finished
drv.submit(lb.id, priors, value, margin[, aux])  # network batch (aux: its root belief rows', 3.8)
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
'select', 'select_min_visits', 'record', 'root_min_visits', 'root_min_visits_max_actions',
'leaf_value_view', 'view_samples', 'view_sample_target', 'belief_tilt', 'belief_clip', 'belief_mix', 'belief_fast',
'root_search', 'gumbel_m', 'gumbel_scale', 'gumbel_c_visit', 'gumbel_c_scale', 'lookahead'}`
(`games.nn_seat`; `policy_target` only matters for recorded decisions, section 3.1; `select` and its
guard for the moves the seat does not sample, section 3.3; `cq_c_visit` / `cq_c_scale` are the
completed-Q transform of both; `record` false - absent: true - a self-play seat whose full searches
are not recorded: it searches, samples its moves and draws its playout cap exactly as a recorded
seat, so its games are the same move for move, and emits no records; zero's self-play league,
docs/az.md 14, puts a past checkpoint there; `root_min_visits` the floor of its full searches' root
visits, section 3.5; `leaf_value_view` whose view its leaves' values come from, `view_samples` /
`view_sample_target` its value-only samples, section 3.6; `root_search` 'puct' | 'gumbel' and the
`gumbel_*` keys its root search, section 3.7; `belief_tilt`, `belief_clip`, `belief_mix`,
`belief_fast` the root's belief of its searches, section 3.8; `lookahead` a dict of the exhaustive
lookahead's settings that decides the seat's moves instead of MCTS, section 3.9). Evaluator keys: every distinct
network `key` is one evaluator (arena / Elo matches: `'m'` vs `'o'`), every distinct external
key one external player. A key cannot be both.

### 2.2 Network batches

| field | dtype / shape | content |
|---|---|---|
| `x` | f32 [B, F] | `encode(decider)` of the (determinized) leaf state |
| `opt_off` | i32 [B+1] | leaf b's legal actions = `opt_action[opt_off[b]:opt_off[b+1]]` (>= 2; none at a deferred terminal, 3.2) |
| `opt_action` | i32 [sum] | global action ids in the game's order |
| `player`, `n_players`, `gid` | i32, i32, i64 [B] | decider (absolute), players, game index |
| `extra` | u8 [B] | the row's kind: `ROW_LEAF` (0) a leaf's own row, else an extra row of the closest leaf row before it (same game): `ROW_VALUE` (1) its tree viewer's value-only row (no options; root-view leaf values, section 3.6); `ROW_BELIEF` (2) a search's root belief row (no options, `player` the root decider: the root's features, whose auxiliary output the driver reads; after the fresh root's own leaf row, or its game's only row - section 3.8). Evaluators may ignore it - but for `ROW_BELIEF` rows, whose `aux` the answer must carry |

Answer: `priors` f32 [sum] (softmax over each leaf's legal actions), `value` f32 [B, 8] =
P(win) per seat **relative to the decider** (column j = seat (player + j) % n; columns >= n
are 0), `margin` f32 [B, 8] (same order); with `ROW_BELIEF` rows also `aux` f32 [belief rows,
aux_width]: the network's auxiliary predictions of those rows (section 3.8). `NetEvaluator` does the
gather + segment softmax on the device. A row with an empty option segment is a deferred terminal's leaf: its value and
margin are used, it has no priors (`NetEvaluator`, the CUDA-graph path and `UniformEvaluator`
handle empty segments; `protocol.validate_batch` allows 0 or >= 2 options). A lookahead seat (section
3.9) puts many rows of one game in a batch, all `ROW_LEAF` (priors: rows with their >= 2 options;
values: value-only rows of the root decider, `player` its seat) but its root's `ROW_BELIEF`.

**Sparse batches** (`sparse=True`, `DriverCfg::sparse`; off by default: the batches of before). Instead
of `x` a network batch carries each row's non-zero features: `xs_off` i64 [B+1], `xs_idx` u32 [nnz] (the
feature indexes, ascending per row), `xs_val` f32 [nnz], `n_features`. The same rows (a chunk encodes each
into a zeroed scratch row and seals its non-zeros, -0.0 included), the same games. Why: an MTG row is
~150 non-zero floats of 37,620, and copying the dense rows - into the batch (fresh memory: page faults,
~70-120 us a row on the loaded desktop) and into the CUDA graphs' pinned buffers - bounded the lookahead's
batches of thousands of rows (section 3.9). `protocol.LeafBatch.x` densifies a sparse batch on first use
(an evaluator that reads `x` sees the same rows); `NetEvaluator`'s token CUDA graphs take it as it is
(`TokenGraphRunner.launch_sparse`: the presence flags read from the non-zeros, the graphs' input rows
zeroed on the device and the non-zeros scattered in - the same floats wherever the graphs read: the
answers are bit for bit those of the dense rows at the same batch shape). Measured (32 games of a 4-ply
lookahead against MCTS on MTG Zero's Delver mirror, the desktop shared): 165 s dense, 89 s sparse - the
batches' assembly 85 s -> 1.5 s -, the same games; 65 s with 4 driver groups.

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
samples are emitted when it ends. With value-only samples (section 3.6) a recorded decision's row
can be followed by rows of other seats' views of the same state with an empty option segment and no
`pi` entries. A driver created with `aux=True` (`DriverCfg::aux`; the game must
have targets, else the driver refuses) adds `aux` f32 [N, W]: each row's decider's
`aux_target` at that decision (section 1.2), computed when the decision is recorded; every other
array is the same with it on or off (tested: the golden records).

`take_records()` (2026-11-04; the run setting `shard_writer_thread`, docs/az.md 31) drains the same
records without the conversion: the driver's `Record`s move out as they are (no copy) as a
`SampleRecords` - `len()`, `nbytes` (what they hold: the bounded queue's measure), `split(sizes)`
(consecutive parts, every size > 0 and their sum `len()`: per finished game, its row's `samples`;
`zero/shardthread.py split_records` mirrors `zero/workqueue.py split_games`) and `arrays()`: the
dict `take_samples` would have given for those records, value for value and dtype for dtype (one
conversion for both, built and the records freed off the GIL), once - a second call, or one after
`split`, raises. A caller drains with one or the other at each point; `take_results` pairs with
either as before. So the thread that drives the search pays only the move, and a writer thread pays
the copy into the dense arrays (MTG: 37,620 floats a row).

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

**A phase's end depends on timing when batches are pipelined** (`zero.infer.drive` with two
batches in flight, the default `ZERO_INFLIGHT` 2; measured 2026-10-29, docs/az.md 14.5): when the
quota is met with the other group's batch already answered, that group was dispatched again and
its games decided moves from their old trees before the pause; when its batch was still in flight,
it pauses without that dispatch and the next phase searches those decisions again with the new
network. Which happens depends on whether the other group's batch was ready when polled, so a
rolling run's later iterations can differ between two runs of the same seed (zero-core edf4b89's
toy-smoke rolling run in the Delver main line's shape: 1 of 19 runs differed from its own reference
from iteration 2 on, `dropped_leaves` 11 vs 22);
`ZERO_INFLIGHT=1` (one batch at a time) is deterministic. A classic driver's games never depend on
it.

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

### 2.10 Paired games (`paired`; 2026-10-30)

`paired=True` (`DriverCfg::paired`; default False): games `2k` and `2k + 1` draw every random stream
from the pair's id `k` - the game's seed (the deal, every shuffle and chance outcome the game draws
from it), the driver's streams (playout cap, noise, search, the bots' seeds) - where each game draws
them from its own gid otherwise (`worker::init_slot`: `split64(seed ^ split64(k + 1))` for `k = gid /
2`). A table whose two games of a pair swap the seats (zero.h2h's and the run's matches: seat `g % 2`)
then plays each deal once from each seat: two sides that search alike play the same game twice
(tested: same moves, scores, results). The streams diverge only where the games do: the first move
that differs consumes the shared streams differently from then on. An odd count's last game plays
its pair's stream alone. Nothing else changes (no gid, seat or result row is different; rolling and
queued setups pair by gid too). Off: the streams of before, bit for bit (the golden match). The
Python reference driver has no pairing. Uses: docs/az.md 20.3 (zero.h2h `--paired`, the run setting
`elo_paired`).

### 2.11 Branched games and archive starts (`branch`, `starts`; MTG Zero's Delver mirror, 2026-11-03)

Two opt-in ways for a recording self-play driver to put positions into its samples that its own games
rarely reach (docs/az.md 24 has the why, the run settings and what they measured). Off (both `None`, the
defaults): every game, stream and sample of before - both read only their own streams, from (seed, the
game's stream id) and a salt of their own, never a draw of the game's other streams.

**Branched games** (`branch={'p', 'moves', 'temp', 'differ'}`, `DriverCfg::branch` / `BranchCfg`; KataGo's
side games): each game the driver starts (never a branch) draws, with chance `p`, a branch. Such a game
keeps a reservoir sample over its **eligible decisions** - a recording self-play network seat's searched
decision (two options or more) that is not minor (`SearchGame::minor_decision`): at the k-th one a copy of
the state is kept with chance 1/k, so the branch point is uniform over the game's eligible decisions (a
game whose length is not known in advance). The game plays and records exactly as without it. When the
game ends (its row and samples emitted), its slot plays the **branch** from the copy, with streams of its
own (from the game's branch stream) and fresh trees: its first `1..=moves` (uniform) non-minor network
decisions are **deviation moves** - the root's own evaluation alone (one simulation; none when a reused
root is expanded already; no Dirichlet noise, root floor, Gumbel schedule or belief), the move sampled
from the root's priors P0 (the network's policy) raised to 1/`temp`; with `differ` the first of them never
the game's own move there. Nothing is recorded until they are played (not the deviation decisions, nor a
minor decision between them): no state of the game before the branch point is recorded twice. Then
normal self-play to the end; the branch's records' z is its own outcome. It carries its game's searched
decisions, real moves (`max_steps` counts them) and move log up to the branch point, and its gid; its row
has `branch` = {at, of: the branch point's index among the game's eligible decisions and their count,
decision, step: the game's searched decisions and real moves before it, action: the game's own move there,
moves, played: deviation moves drawn and played, same_first: the first deviation was the game's own move,
start: the archive entry the game started from}; a branch never branches. A branch is a finished game:
rolling quotas count it (2.6), in a classic driver its branches play after the listed games; in a
`queue_only` driver it is the second row of its game's queued setup. A game at a deferred terminal (2.8)
that kept a branch point is an error. Stats: `branches` (started), `dev_moves`.

**Archive starts** (`starts={'p', 'specs': [(id, spec), ...]}`, `DriverCfg::starts` / `StartsCfg`): each game
the driver starts (never a branch) is started, with chance `p`, from an entry drawn uniformly: that entry's
spec instead of the driver's or its queued setup's (a self-play driver's setups are its own spec), the
game's seed drawn as for any other game (the spec decides what it seeds: MTG Zero's `start` block deals the
cards nobody knows from it). Every entry is parsed and started once when the driver is made: an entry the
game refuses is an error then. Its row has `start` = the entry's index. Stats: `starts`.

Both need a driver that records a self-play seat (an evaluation or a match is refused). Tests:
`driver::tests::branched_games_leave_their_games_alone_and_record_their_own_outcome` (with every search
full: a branch's records are exactly its searched decisions after the deviations; the games equal their
twins without branching, move for move and record for record), `branches_count_in_rolling_quotas_and_survive_the_phases`,
`archive_starts_play_their_entries_specs` (game g started from entry j is game g of a driver whose spec is
entry j's), `a_branch_of_an_archive_start_names_its_entry`, `differ_never_replays_the_games_own_move_first`,
`branch_and_starts_settings_are_checked`; tests/test_forks.py.

### 2.12 Sparse rows without full-row scans (`SearchGame::encode_sparse`; MTG Zero's Delver mirror, 2026-11-04)

No setting: the same batches, byte for byte, read with less work (docs/az.md 28 has the measurements).

**Why.** A sparse driver (2.2) read each network row by `encode` into a zeroed scratch row of `F` floats,
then `LeafBuf::seal` walked all `F` floats one by one for the non-zeros; with the evaluation cache (3.12)
`obs_hash` read the row a second time for the key, and a cached row was zeroed in full. An MTG Zero row is
~250 non-zero floats of 37,620 (82 of its 587 blocks of 64 hold any), and on 10k Delver leaf rows the scans
cost about as much as `encode` itself: `seal` ~11 us, `obs_hash` ~3 us, the zeroing ~2 us against `encode`
~13-15 us and ~14-18 us for the rest of a simulation.

**What.**

* `SearchGame::encode_sparse(viewer, scratch, idx, val)`: `encode(viewer)` as its non-zero (index, value)
  pairs - every feature whose f32 bits are not 0 (-0.0 included), indexes ascending - appended to `idx` /
  `val`; `scratch` is `F` zeros on entry and on return. The default is `encode` into `scratch`, then
  `game::sparse_scan`. A game may override it to emit its pairs without the dense row; its pairs must be
  exactly the default's (a test over its states: the driver's batches and cache keys depend on it).
* `game::sparse_scan(x, idx, val)`: `seal`'s scan, the same pairs: blocks of 32 zero floats skipped by an
  OR of their bits, a block's non-zero lanes taken from a bit mask (no branch per float), each zeroed again.
  `LeafBuf::seal` uses it for rows written into the scratch (the lookahead's, 3.9).
* `lookahead::obs_hash_sparse(idx, val, n)`: `obs_hash` of an `n`-float row given as those pairs - the same
  mixing of the same (pair index, bits), so `obs_hash_sparse` of a row's non-zeros == `obs_hash` of the row.
* The driver (`LeafBuf::encode_row` / `row_hash`): a sparse buffer's leaf rows, root-view value rows (3.6)
  and root belief rows (3.8) come from `encode_sparse`; the evaluation cache's key is `obs_hash_sparse` of the
  row's pairs (the same key as before: the same hits); a cached row's pairs are dropped (truncated) instead
  of zeroing the scratch. Dense drivers are unchanged (`encode` into the batch's `x`, `obs_hash` of the row).
  The lookahead's own observation key (3.9) still hashes its dense rows: unchanged.

**Tests.** `lookahead::tests::sparse_scan_and_obs_hash_sparse_match_the_dense_row` (lengths around both
block sizes, odd and even, empty to full rows, -0.0, NaN, subnormals: the old `seal` loop's pairs and the
dense hash); `driver::tests::sparse_rows_from_encode_sparse_are_the_dense_rows` (ToyGame with root-view
rows, HiddenMull with root-view and belief rows, eval_cache 0 and 4096: every batch's rows densified equal
the dense driver's, the same games, records, cache hits and rows); the golden digests (dense) unchanged.
MTG Zero checks its own states (ENGINE.md 89: 10k Delver leaf rows, pairs and keys equal).

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
  searches (also on a reused, already expanded root); the opt-in root minimum visits of full
  searches (section 3.5: forced walks, left out of the visit counts below); with the opt-in Gumbel
  root search (3.7) its Gumbel noise replaces the Dirichlet noise and the temperature, sequential
  halving replaces PUCT at the root and its move the rule below; self-play samples moves by visit count
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
old arrays until it holds > 65k nodes (`compact`). With information-set merging on (3.16, an evaluation seat's
`infoset_merge`) an edge's children are links to nodes merged by the root decider's observation.

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

### 3.5 Root minimum visits ("forced playouts"; `root_min_visits`, 2026-10-29)

**Why.** A near-one-hot prior starves an option of every visit at a small decision. MTG Zero's
Delver mirror (its ENGINE.md 64): at the mulligan (keep / mulligan) the prior is >= 0.98 on Keep for
89 of 100 one-land seven-card hands of delver2's iteration 21, and at 64 simulations Mulligan gets no
visit in 89 of the 100 - its Q is never estimated, and the completed-Q target (3.1) completes an unvisited
option at v_mix, about the visited option's Q, so its target stays at its prior and cannot recover.
Only root noise, when it happens to land there in a full self-play search, ever visits it. The same
starvation hits any small decision under a saturated prior (attack or not, pay for Daze, yes / no).

**The floor** (`Tree::set_root_min(k)`, Python `mcts.Tree.set_root_min`): every root option gets at
least k visits. A walk keeps PUCT's root option i unless N_i >= k and another root option has fewer
than k visits; then it takes the least-visited root option (the lowest index on ties) - a **forced
walk**. So the floor never overrules PUCT on an option that still lacks its k; the forced walks begin
once PUCT's favourite has k visits and stop when every option has k (a reused root's visits count).
Nothing changes below the root. The tree counts the forced walks per root option (`root_forced`) and
sums the win vectors they backed up; `reset` and `advance` turn the floor off (the driver sets each
search's own).

**Which searches** (`driver::root_floor(k, max_actions, full, m, target)`, Python `mcts.root_floor`,
the bindings' `root_floor`): a **full** search - a self-play seat's recorded searches (`sims`, the
`p_full` share; a league opponent's unrecorded seat too, docs/az.md 14, being the self-play seat with
`record` false: it plays as in its own self-play), every search of an evaluation seat - whose root has m options with 2 <= m <=
`root_min_visits_max_actions` (default 4), when 2 m k <= its simulations (after `sims_divisor`): the
floor never takes more than half the budget. Otherwise k = 0, the search of before. **Never a fast
(`sims_fast`) or minor (`sims_minor`) search**, as root noise: they record nothing, so the floor's Q
would feed no target, and it would take a large share of their budget (KataGo turns its forced
playouts off in its cheap searches for the same reason). With MTG Zero's Delver settings (64 / 16,
minor 4) k 4 or 8 applies to the full searches of roots of 2-4 options.

**What reads what.**

| output | with the floor |
|---|---|
| PUCT's Q; the 'cq' target (3.1); the 'cq' move rule (3.3) and its guard | every visit: a forced walk is a Q measurement. Every option of a forced root is visited, so the target is softmax(ln P0 + sigma(Q)) over measured Q's (no v_mix), well defined and summing to 1; a guard <= k admits every option |
| the 'visits' target N / sum N; temperature sampling (the first `temp_moves`); the 'visits' rule argmax (N, P0) | `root_visits_pruned` = N - forced: the visits PUCT chose. A move is sampled, or wins the count, only through PUCT's own visits - which include those PUCT gave it after the forced walks revealed a good Q. Nothing was visited but forced walks: P0 (target), the rule's prior argmax |
| the root value target q (2.4, `wsum / tot`) | `root_win_pruned`: without the forced walks' backups (they did not follow the search's policy) |

Why exact accounting rather than KataGo's pruning (Wu 2019, "Accelerating Self-Play Learning in Go",
3.1: forced playouts sqrt(k P N), pruned from the policy target as far as the child's PUCT value stays
below the best child's): the tree knows which walks it forced, so it removes exactly those, from the
value target too (KataGo's value target is the game's result). A forced option that turns out good is
not lost: PUCT then visits it on its own and those visits stay.

**Cost.** The forced walks come out of the same budget: at most (m - 1) k walks of a forced search
(a 2-option root at 64 simulations: 4 of 63 with k 4, 8 with k 8), none when PUCT already gives every
option k visits. Throughput (leaves per second) does not change; the visits the favourite gets do.

**Settings.** Seat keys `root_min_visits` (0 = off, the default) and `root_min_visits_max_actions`
(4); run settings `selfplay_root_min_visits` (self-play's seat), `eval_root_min_visits` (every
evaluation seat; its moves change, so its Elo pool's name gets `-f<k>`, `-f<k>a<m>` with another
max_actions: `mcts.forced_tag`, docs/az.md 5.4) and `root_min_visits_max_actions` (docs/az.md 1.2,
9.2). Off is the tree of before bit for bit (`set_root_min(0)`; the golden records and match, toy
runs byte for byte).

**Measured** on the keep-or-mulligan test game (`crates/search/src/mull_game.rs`: Keep's prior 0.999
whatever the hand, a mulligan worth +0.16 with a bad hand and -0.14 with a good one; 64 simulations;
`driver::tests::root_min_visits_*`): without the floor Mulligan gets 0 of 63 visits at every root and
its cq target stays at 0.001; with k 4 it gets exactly its 4 forced visits, and in self-play (160
games, cq target (50, 1)) the target is > 0.9 on Mulligan for 141 of 152 bad hands (the other 11:
four unlucky redraws hid its worth) and < 0.001 for all 168 good ones; the good hands' 'visits'
targets hold no forced visit (exactly 0 on Mulligan, unpruned 4 / 63) and their q is Keep's value to
1e-5; temperature sampling plays Mulligan with 16 of 152 bad hands (PUCT following the forced Q) and
0 of 168 good ones. Evaluation seats (200 games): the 'visits' rule never plays Mulligan, with or
without the floor; 'cq' with guard 4 plays it with 179 of 193 bad hands with k 4, never without, and
with 0 of 207 good hands either way. MTG Zero's Delver positions (its ENGINE.md 64): the floor
works as here, but there the network's own Q put Mulligan below Keep in every starved low-land hand
(-0.09 for 0 lands, -0.14 for 1 land), so the keep rates did not move - the keeps come from the value
head, not from the starvation; on general small decisions (2-4 options) k 4 moved the cq target (0.3)
by more than 0.1 at 11-13% of the roots.

Code: `tree.rs` (`set_root_min`, `force_root`, `root_forced`, `root_visits_pruned`,
`root_win_pruned`; `backup` counts a forced walk's backup), `driver.rs` (`root_floor`,
`NnSeat::root_min_visits`), `driver/worker.rs` (`start` sets each search's floor, `decide` reads the
pruned counts), `dynapi.rs` / the bindings (`SearchTree.set_root_min`, `root_min`, `root_forced`,
`root_visits_pruned`, `root_win_pruned`); Python `mcts.py` (`Tree.set_root_min`, `force_root`,
`root_visits_pruned`, `root_win_pruned`, `root_floor`, `check_root_min`, `forced_tag`; `backup` /
`expand_and_backup` take the tree for the accounting), `play.py` (`NNSeat`, `PyDriver`'s 'py' and 'rs'
seats), `games.nn_seat`. Tests: `tree::tests::root_min_visits_*`, `driver::tests::root_floor_*`,
`driver::tests::root_min_visits_*`, `tests/test_root_min_visits.py`.

### 3.6 Root-view leaf values (`leaf_value_view`, value-only samples; 2026-10-29)

**Why.** The search determinizes the hidden information from the root decider's view (a fresh
world every simulation) and, before this setting, valued every leaf with the network on
`encode(decider)` of the leaf - the view of **the player to move there**. At a leaf where another
seat is to move that value is wrong twice for the root decider: it is blind to the root decider's
own hidden cards (which the root decider knows), and it reads the other seat's **sampled** hidden
cards as known (a guess of the root decider's, not that seat's real hand). MTG Zero's Delver mirror
(its ENGINE.md 64.4, 67): after a kept landless hand, the next decider's view values the keeper at
0.51, the keeper's own view at 0.36; roughly every other leaf of a two-player search is another
seat's.

**The setting** (seat key `leaf_value_view`: `'mover'` default | `'root'`). With `'root'` a leaf whose
decider is not the tree's viewer is evaluated twice, in the **same batch**: its **priors** from its
own row `encode(decider)` (they model that seat, which cannot see our hidden cards either) and its
**value and margin** from a second, value-only row `encode(viewer)` of the same (sampled) state -
the root decider's information: its own cards as they are, everything it cannot see unknown. The
value is relative to the viewer and backed up as any (`leaf_values_team(..., viewer, ...)`); the
node's stored utility vector (`vec`, first-play urgency, v_mix of a cq target) is that one. A leaf of
the root decider, a deferred terminal (already `encode(viewer)`, 3.2) and a terminal are as before.
Tree reuse keeps a subtree only for the same seat, so a reused tree's values are its viewer's too.

**The batch** (2.2): the value-only row follows its leaf's row, with no options (an empty option
segment, as a deferred terminal's), `player` = the viewer, the same `gid`, and the row flag
`extra` = `ROW_VALUE` (1; a leaf's own row is `ROW_LEAF`, 0). The flags are generic: a row of kind > 0
belongs to the closest leaf row before it, so other features can add kinds (MTG Zero's belief
head, stage 2). An evaluator answers every row alike and can ignore the flags - `NetEvaluator`, the
CUDA-graph path and `UniformEvaluator` are unchanged; the driver takes the priors of the leaf's row
and the value of the row after it (`Pending::Leaf::vrow`). `stats()['view_rows']` counts the extra
rows (`leaves` still counts leaves).

**Leaks.** The value-only row is safe only if `encode(v)` of a state another seat decides holds
nothing of that seat's hidden cards - including its options (legal actions derived from its hand)
and anything a payment of its pins. The contract (section 1: `encode` reads no hidden information)
covers it; a game must test it for non-deciding viewers: the same encoding in every world that
deals the decider's hidden cards again, whatever the decider's options there (MTG Zero: its
ENGINE.md 67.2, tests/hidden.rs and tests/az.rs; the toy and `mull_game::HiddenMull` here).

**Value-only samples** (seat keys `view_samples`: a share in [0, 1], 0 = off; `view_sample_target`:
`'next_q'` default | `'z'`). A network trained on `encode(decider)` only is asked, by root-view
leaves, for the value of a state from a seat that does not decide it. To train that view, a share
of the recorded decisions (`driver::view_pick`: a hash of the driver seed, the game and the decision
index - no random stream changes; the Python driver's `play.view_pick` is the same arithmetic) also
records, for every other seat of the game that records (a self-play network seat with `record`), a
sample of the real state from that seat's view: `x` = `encode(other)`, `player` = the other seat, no
options and no policy target (an empty segment: the training's policy loss and entropy skip it, docs/az.md 16),
`z` = the result in its perspective, `sc` its margin. Its `q` (the value target's search part, the
run's `value_q` blend):

* `'next_q'`: the root win estimate of that seat's **own next search** in the game (full or fast; a
  minor search is too small; without its forced walks, 3.5), `z` if it never searches again. That
  seat only learns more until its next search, so that search's value is an estimate of the outcome
  given what it knows at the sample - the tower property: unbiased up to the search's own error, and
  of lower variance than z. A decider's sample's q is its own search's at the same state; this is the
  nearest search of the observer's.
* `'z'`: the outcome only (unbiased, noisier).
* Not offered: the decider's root q negated (zero-sum). It is computed from the decider's information
  set - with the observer's hand sampled, the decider's own known - so it takes the observer's
  private information out of the target and the decider's in: the bias the setting removes, the other
  way round. And not the observer's previous search's q: it predates what happened since.

With auxiliary targets on (`DriverCfg::aux`) a value-only record carries its observer's targets
(`aux_target(observer)`), so a game's records keep one width.

**Cost.** One extra network row per leaf of another seat: about half the leaves of a two-player
game. A batch-bound evaluator (CUDA, MPS: bigger batches nearly free) pays less than proportionally;
an inference-bound CPU pays it in full: MTG Zero's Delver network on the desktop CPU, 64 games, 4
torch threads: 1,313 leaves/s with mover-view values, 887 with root-view values (0.51 value rows per
leaf), the same 1,313 / 1,342 rows/s (its ENGINE.md 67.6). Value-only samples add `view_samples` x
(players - 1) samples per recorded decision: more rows in the window and, since training steps follow
the new samples, as many more training steps.

**Settings.** Run settings `selfplay_leaf_value_view` (the self-play seat; a league opponent's
unrecorded seat is the self-play seat and takes it too), `eval_leaf_value_view` (every evaluation
seat: its moves change, so its Elo pool's name gets `-rv`: `mcts.view_tag`, docs/az.md 5.4),
`selfplay_view_samples` and `view_sample_target` (docs/az.md 1.2, 16). Off (`mover`, 0) is the search
and the run of before bit for bit (the golden records and match; toy runs byte for byte).

**Measured** on the hidden-hand keep-or-mulligan test game (`mull_game::HiddenMull`: a seat sees its
own hand only; the stand-in network values the viewer's information; 200 games,
`driver::tests::root_view_values_read_the_root_deciders_hand`): after two simulations a bad hand's
Keep is worth 0.500 to the search with mover-view leaves (the other seat's sampled hand, average)
and exactly 0.350 with root-view leaves (the keeper's own hand, every world). MTG Zero's Delver
positions: its ENGINE.md 67.

Code: `driver.rs` (`LeafView`, `ViewTarget`, `view_pick`, `ROW_LEAF` / `ROW_VALUE`, `Batch::extra`,
`NnSeat::leaf_value_view` / `view_samples` / `view_target`), `driver/worker.rs` (`advance_slot`'s
value-only row, `apply_eval`'s `vrow`, `decide`'s value-only records and the `view_wait` fill),
`dynapi.rs` / the bindings (`SearchTree.simulate(root_view=True)` -> a 4th item, the viewer's
features or None; `SearchTree.viewer`; the module's `view_pick`); Python `mcts.py`
(`expand_and_backup(value_player=)`, `check_leaf_view`, `view_tag`), `play.py` (`NNSeat`,
`PyDriver._leaf_batches`, `view_pick`), `replay.py`, `parity.root_view`. Tests:
`driver::tests::leaf_view_*`, `root_view_*`, `value_only_*`, `tests/test_root_view.py`.

### 3.7 The Gumbel root search (`root_search`; 2026-10-29)

**Why.** At a few dozen simulations PUCT's root is locked in by the prior: it spends the walks where
the prior points and the move follows the visits. MTG Zero's Delver mirror (its ENGINE.md 64, 67): the
prior is >= 0.98 on Keep for 89 of 100 one-land hands and Mulligan often gets no visit at 64
simulations; an upkeep Brainstorm with prior 0.934 got 62 of 63 visits and Pass 1, also after
root-view leaf values (3.6) had raised Pass's Q to 0.638 against Brainstorm's ~0.65. The root minimum
visits (3.5) measure a starved option but leave PUCT's allocation and move alone; root-view values fix
the Q but not the visits. Gumbel AlphaZero's root (Danihelka et al. 2022, "Policy improvement by
planning with Gumbel", ICLR) is built for few simulations: it samples m candidate actions without
replacement (Gumbel-top-k of the logits), spreads the budget over them by sequential halving, and plays
the argmax of g + logits + sigma(completedQ) among the last survivors - a policy improvement in
expectation whatever the budget, and with <= m legal actions every option is considered.

**The algorithm** (`Tree::set_gumbel(m, budget, noise, c_visit, c_scale)`, Python
`mcts.Tree.set_gumbel`; the arithmetic of DeepMind's mctx `gumbel_muzero_policy` - `seq_halving.py`,
`action_selection.py`, checked against its source):

1. **Noise**: g_a = scale x Gumbel(0, 1) per root option, drawn once per search (`noise::gumbel`; 0
   without noise).
2. **Score**: `g_a + ln max(P0_a, floor) + sigma(completedQ_a)`, sigma(q) = (c_visit + max_b N_b) c_scale
   q - the logits of the 'cq' target and rule (`completed_q_logits`, 3.1, 3.3): completedQ is Q for a
   visited option, v_mix for an unvisited one, N every visit of the root (a reused root's too), no
   min-max normalisation (3.1). P0 is the noise-free prior (a Gumbel search has no Dirichlet noise).
3. **Sequential halving** (`tree::gumbel_sequence(min(m, options), budget)` = mctx
   `get_sequence_of_considered_visits`): root walk j may take only an option whose walks **of this
   search** equal `seq[j]`, the highest score among those (the first maximum on exact ties). The first
   min(m, k) walks take the unvisited options in score order - all share v_mix, so the top options by
   g + ln P0: the Gumbel-top-k sample -, then each round gives every survivor max(1, n / (ceil(log2 m)
   x survivors)) more walks and the survivors are halved (at least 2). 16 options at 64: 16 x 1, 8 x 2,
   4 x 4, 2 x 8 (final counts 15, 15, 7, 7, 3 x 4, 1 x 8); 2 options: 32 / 32; 4: 8 each, then 24 / 24. The
   schedule is that of the search's **simulations** (`budget` = the driver's target): a fresh root's own
   evaluation takes the first simulation, so its walks follow the schedule's first target - 1 entries
   (16 options at 64: 15, 14, 7, 7, ...; 2: 32 / 31) - the paper's allocation for the budget less one walk,
   where mctx's schedule of 63 walks would round its middle rounds down (19, 18, 5, 5, 2 x 4, 1 x 8). Past
   the schedule (a caller simulating more), the most walked options.
4. **Below the root**: PUCT, unchanged.
5. **The move** (`Tree::gumbel_choose`): among the options this search walked most (the last
   survivors; every option when it walked none), the highest score. With noise it is a sample of the
   improved policy pi' = softmax(ln P0 + sigma(completedQ)) over the considered options (the
   Gumbel-max trick; with every option considered and its Q exact, exactly pi'); without, its argmax.
6. **The root value** (`Tree::gumbel_q`): `sum_a pi'_a Wv_a / sum_a pi'_a` over the options this search
   walked, Wv_a = this search's mean backed-up win vector through a (`gumbel_win` / `gumbel_visits`).

The tree counts its own walks per root option (`gumbel_visits`, their win vectors `gumbel_win`) in
`backup`, apart from N, so a reused root's earlier visits stay in Q, sigma and the 'cq' target but not
in the schedule. `reset` and `advance` turn it off (the driver sets each search's own); `clear_gumbel`
is the tree of before bit for bit.

**The driver** (seat key `root_search` 'gumbel', `RootSearch::Gumbel { m, scale, c_visit, c_scale }`):
**every** search of the seat - full, fast and minor (`sims_fast`, `sims_minor`, after `sims_divisor`) -
is a Gumbel search, the schedule that of its simulations (`target`, step 3; a reused, expanded root
walks all of them). Interactions:

| with | what the Gumbel search does |
|---|---|
| root Dirichlet noise (`alpha`, `eps`) | replaced: no Dirichlet noise; the slot's noise stream draws the Gumbel noise |
| temperature (`temp_moves`) | replaced by the noise: a self-play seat's first `temp_moves` searched decisions of a game are searched with noise `scale` (the move a sample of pi'), later ones without (the argmax) - the role temperature had; an evaluation seat: `scale` on every decision (run default 0: deterministic) |
| the move rule (`select`, `select_min_visits`) | not used: the Gumbel move replaces it (an evaluation seat's sigma defaults to the rule's transform, below) |
| sigma | the seat's `gumbel_c_visit` / `gumbel_c_scale` (absent: `cq_c_visit` / `cq_c_scale`). `games.nn_seat` always writes it: a self-play seat recording 'cq' takes the target's transform - the search and its target share one sigma, as in the paper -, any other seat its move rule's (`select_c_*`, else `cq_c_*`) |
| the policy target | 'cq' (3.1), from every root visit as the paper's. A recording seat with 'visits' is refused (`Driver::new`, `PyDriver`): sequential halving's visit counts are its schedule (16 options at 64: eight options get 1 visit, two get 15, whatever their value), not a policy |
| the root value target q (2.4) | `gumbel_q`, not `wsum / tot`: the first round walks every considered option, so the plain mean mixes in what the halving ruled out - two options at 64 simulations get half the walks each (the mulligan test game: the root mean 0.07 below a good hand's Keep value, `gumbel_q` = Keep's to 1e-4). pi' is the policy the move samples, so its value is the outcome's expectation under the move played (after `temp_moves`, the argmax: Q of the move, `gumbel_q` slightly below) |
| value-only samples (`next_q`, 3.6) | the observer's next search's `gumbel_q` |
| fast and minor searches | Gumbel searches too, with the seat's m (they record nothing; the move is what matters, and a few walks is what Gumbel is for): a 16-simulation fast search over 16 options walks each once and takes the best score; a 4-simulation minor search the top 4 by g + ln P0 once each. Their noise follows the temperature rule |
| tree reuse | kept (the same seat deciding again): the reused root's visits count in Q and sigma; the schedule budgets this search's walks only, and its first round ranks the options this search has not walked by their score, so an option already measured competes with its Q |
| root minimum visits (3.5) | exclusive (`Driver::new` refuses both on one seat, the run settings one side): with k <= m options the halving gives each at least max(1, n / (ceil(log2 k) k)) walks (2 options at 64 simulations: 31; 3: 10; 4: 8) - more than the floor's 4 or 8 on the roots it applies to |
| root-view leaf values (3.6) | orthogonal: leaves are valued from the seat's view as before; the root's selection reads their Q |
| chance, determinization, `chance_k`, `redet` | unchanged: every simulation its own world, `chance_k` below the root; the root's noise is per search, not per world |
| a rolling driver's epoch restart (2.6) | the decision's search restarts from an empty tree with its noise kept (`Slot::gnoise`, as its playout-cap draw) |
| the self-play league (docs/az.md 14) | the league seat is the self-play seat with `record` false: a Gumbel search with self-play's noise |
| the root's belief (3.8) | orthogonal: a search that takes it deals its worlds by it; a reused root's belief row (the search waits for it) consumes no simulation, so the schedule is the same; a fresh root's rides with its own leaf (`driver::tests::gumbel_with_a_root_belief`, `tests/test_gumbel.py::test_gumbel_with_a_root_belief`) |
| external seats, deferred terminals, teams | unchanged (the root's Q is the decider's utility; deferred leaves as 3.2) |

**Why PUCT below the root.** Gumbel MuZero also replaces the interior selection with a deterministic
rule (the action maximising pi'(a) - N(a) / (1 + sum N), the completed-Q policy of each node). It is
not used here: the improvement guarantee is the root's, the interior rule changes every node of every
search (the opponent's decisions in determinized worlds included), and nothing measured here says the
interior is where PUCT fails - the starvation is at the root. It would be a separate seat flag
(`interior_search`), with its own A/B.

**Settings.** Seat keys `root_search` ('puct' default | 'gumbel'), `gumbel_m` (16, the paper's and
mctx's default), `gumbel_scale` (1: the paper's self-play), `gumbel_c_visit` / `gumbel_c_scale`
(absent: the seat's `cq_c_*`); a 'puct' seat with a `gumbel_*` key is an error. Run settings
`selfplay_root_search`, `eval_root_search`, `gumbel_m`, `selfplay_gumbel_scale` (1), `eval_gumbel_scale`
(0) (docs/az.md 1.2, 19). An evaluation's root search names its Elo pool (`mcts.gumbel_tag`:
`gumbel-m16`, `gumbel-m16-v50-s0.3` with a transform other than 50 / 1, `-g1` with noise, then `-rv`).
Off (`puct`) is the search of before bit for bit (the golden records and match; toy runs byte for byte).

**Measured** on the keep-or-mulligan test game (`mull_game::Mull`, Keep's prior 0.999 whatever the
hand; `tree::tests::gumbel_measures_the_starved_mulligan`, `driver::tests::gumbel_mulligan_selfplay_
and_evaluation`): every search walks both options 32 / 31 times. Self-play with sigma (50, 1.0), 160
games: the 'cq' target > 0.9 on Mulligan for 152 of 152 bad hands and < 0.001 for all 168 good ones,
the sampled move Mulligan with 151 of 152 bad hands and 0 of 168 good ones, a good hand's q Keep's
value to 1e-5 (PUCT: Mulligan never visited, never played, its target at its prior; the root minimum
visits k 4: 141 of 152 targets, 16 of 152 moves). Evaluation without noise: Mulligan with 193 of 193 bad
hands at sigma scale 1.0 - and with none at 0.3: (50 + 32) x 0.3 x 0.16 = 3.9 logits of Q gap cannot
beat ln 999 = 6.9 of prior. The Gumbel root measures every option; whether that changes the move is
sigma's scale against the prior's log-odds. MTG Zero's Delver positions: its ENGINE.md 68.

Code: `tree.rs` (`gumbel_sequence`, `GumbelRoot`, `set_gumbel`, `gumbel_select`, `gumbel_choose`,
`gumbel_q`, `gumbel_visits` / `gumbel_win` / `gumbel_noise` / `gumbel_schedule`; `backup` counts the
search's own walks), `noise.rs` (`gumbel`), `driver.rs` (`RootSearch`, `NnSeat::root_search`, the
checks in `Driver::new`, `Slot::gnoise`), `driver/worker.rs` (`start` sets each search's, `apply_eval`
and `start` skip Dirichlet noise, `decide` takes the move and q, `restart_search`), `dynapi.rs` / the
bindings (`SearchTree.set_gumbel`, `clear_gumbel`, `gumbel_on`, `gumbel_choose`, `gumbel_q`,
`gumbel_visits`, `gumbel_win`, `gumbel_noise`, `gumbel_schedule`; the module's `gumbel_sequence`);
Python `mcts.py` (`GumbelRoot`, `Tree.set_gumbel`, `gumbel_sequence`, `gumbel_scores`, `gumbel_select`,
`gumbel_choose`, `gumbel_q`, `check_root_search`, `gumbel_noise_scale`, `gumbel_tag`; `backup` counts the
walks), `play.py` (`NNSeat`, `PySearch` / `RsSearch`, `PyDriver._start` / `_decide`), `games.nn_seat`,
`run.py`, `elo.eval_tag`, `h2h.root_keys`, `replay._nn`, `parity.gumbel` / `gumbel_formula`. Tests:
`tree::tests::gumbel_*`, `driver::tests::gumbel_*`, `noise::tests::gumbel_moments_and_the_max_trick`,
`tests/test_gumbel.py`.

### 3.8 The root's belief: belief-weighted determinization (`belief_tilt`; 2026-10-29)

**Why.** Every simulation deals the hidden information again (`determinize`): uniformly over the
worlds consistent with what the root decider has seen, so a card the opponent would have played by
now is as likely in its hand as one it would hold. MTG Zero's belief head (an auxiliary target, docs/az.md
15; its ENGINE.md 66) predicts the opponent's hand from public information better than that deal (2.28
against 2.81 nats per card on held-out Delver positions; Force of Will held with open mana 2.7 times the
uniform deal's rate). Here the search deals its worlds from that prediction instead.

**The hook** (`game.rs`). `SearchGame::determinize_with(viewer, rng, &Belief)` - `determinize` with the
hidden cards drawn by weights from the belief, every constraint of `determinize` kept; default:
`determinize` (a game without it searches as before). `Belief { pred, tilt, clip }`: `pred` = the
network's predictions of the game's `AUX` targets for the root decider (`aux_width()` floats in `AUX`
order: a multinomial segment's distribution, `zero.net.aux_predict`), `tilt` alpha > 0, `clip` R >= 1.
`game::belief_weight(pred, uniform, tilt, clip)` = clip((max(pred, 0) / max(uniform, eps))^tilt, 1/R, R)
is the shared weight of one category of hidden cards: the belief's expected count of it among the cards
the deal draws against the uniform deal's. Tilt 1 deals the belief's expected counts to first order, 0
would be today's deal (never set: a tree without a belief deals uniformly), the clip keeps every card
possible (no world rarer than 1/R of today's odds per card).

**The tree** (`tree.rs`). `Tree::set_belief(Some(b), mix, live, rng)`: from now on every simulation deals
its world with `determinize_with`, a `mix` share in [0, 1) with `determinize` (one `rng.unit()` draw per
simulation when 0 < mix); without `redet` the one world of the decision is dealt again from `live` with
it. `None` (the default; `reset` and `advance` clear it: a belief is one decision's prediction) is the
tree of before bit for bit - no draw changes.

**The driver** (`driver.rs`, `driver/worker.rs`). A search takes the belief when its seat's `belief_tilt`
> 0 and it is a full search (`NnSeat::takes_belief`: every search of an evaluation seat, a self-play
seat's recorded ones; every one with `belief_fast`). It asks the network once, with the root's own
features - `encode(decider)` of the live decision: the decider's view, the same in every world - as a
`ROW_BELIEF` row (2.2):

* **a fresh root** (no tree, or a new search epoch): the row follows the root's own leaf row in the
  search's first batch (its first simulation's leaf is the root itself, whose features do not depend on
  the world it dealt) - no extra round trip;
* **a reused root** (tree reuse, 3): the row alone; the search's simulations wait for it (one batch round
  trip of that game; its group's other games go on).

The answer's auxiliary output of that row becomes the tree's belief (`set_belief`, the seat's tilt, clip
and mix). `Driver::submit_aux(id, priors, value, margin, aux)` (Python `drv.submit(id, priors, value,
margin, aux)`) takes `aux` = [belief rows x aux_width] in row order; a batch with belief rows answered
without them, with the wrong width or with a non-finite or negative value is an error - an evaluator that
cannot compute the belief never leaves the search dealing uniformly in silence. `NetEvaluator` answers
them from the auxiliary head (the head on those rows only, in the same forward pass; a fourth item of its
answer; such a batch skips a conv network's CUDA graphs, a token network's answer it: docs/az.md 10.6)
and raises `net.HeadMissing` for a network without one;
`UniformEvaluator` has none (the driver's error). `stats()['belief_rows']` counts the rows. **Rolling
epochs** (2.6): a belief asked of the old network is asked again of the new one (the restarted search's
tree has none); a phase's dropped batches drop a pending belief row too (asked again next phase).
**Root-view leaf values** (3.6) are independent: both extra kinds ride in the same batches (a fresh
root's leaf is the viewer's own: no value-only row there).

**Leaks.** The belief comes from the root decider's own features, never from the true state, and the game
maps it to weights from what the root decider knows (its own information, public facts, the hidden pool's
multiset as it knows it): worlds that differ only in the cards it cannot see must deal by the same
weights. A game tests it (MTG Zero: `deal_weights_ignore_the_opponents_hidden_cards` in its tests/hidden.rs,
`the_belief_weights_ignore_the_opponents_hidden_cards` in tests/az.rs, its ENGINE.md 69).

**Which searches, and why not fast ones by default.** Fast (`sims_fast`) and minor (`sims_minor`)
searches record nothing and are small: one more row is 1/16 or 1/4 of their leaves (1/64 of a full
search's), and a search too small to look deep gains little from better-dealt worlds. `belief_fast` turns
them on.

**Cost.** One extra network row per search that takes the belief - 1/sims of its leaves: +1.6% at 64
simulations, +3% at 32 - and the head on those rows only; a reused root waits one batch round trip. The
weighted deal is the game's (MTG Zero's: 6 us a call against `determinize`'s 1 us, nothing beside a leaf's
evaluation). Measured on MTG Zero's Delver network (docs/az.md 18.6): 3-7% of an evaluation seat's leaves
per second at 32 simulations, 1-3% of the self-play seat's.

**The test games' deals.** The toy (`toy.rs`): the next round's card - its `next_card` target - drawn
among the unrevealed cards of its stage with `belief_weight(p_c, 1 / cards, tilt, clip)`, the rest as
`determinize`. `mull_game::HiddenMull` (`AUX` "hand" [bad, good]): the other seat's hand good with
probability q w_good / (q w_good + (1 - q) w_bad), q the uniform deal's (1/2, 3/4 after its mulligan).

**Settings.** Seat keys `belief_tilt` (0 = off, the default), `belief_clip` (20), `belief_mix` (0; in
[0, 1): 1 would make the tilt a no-op and is refused), `belief_fast` (false); run settings
`selfplay_belief_tilt`, `eval_belief_tilt`, `belief_clip`, `belief_mix`, `belief_fast` (docs/az.md 18). An
evaluation's moves change with it: its Elo pool's name gets `mcts.belief_tag`'s suffix (`-b1`,
`-b0.5c10m0.25f`).

**Measured.** HiddenMull (`driver::tests::the_belief_deals_the_searches_worlds`, 300 games, 64
simulations): seat 1's root value after seat 0 kept is 0.5243 with uniform worlds, 0.5233 with a belief
equal to the uniform deal (0.5 / 0.5) and 0.4046 with "seat 0's hand is good with 0.9" - a shift of
-0.1197 against the analytic -0.3 x (0.9 - 0.5) x 63 / 64 = -0.1181. The toy (`tree::tests::
a_belief_deals_the_worlds_by_its_weights`): a sharp belief deals its card in > 98% of the worlds, the clip
2 and the uniform share 0.5 exactly their odds. Parity: section 6.

Code: `game.rs` (`Belief`, `belief_weight`, `SearchGame::determinize_with`), `tree.rs` (`deal`,
`set_belief`, `belief`, `belief_mix`), `driver.rs` (`ROW_BELIEF`, `NnSeat::belief_*`, `check_belief`,
`takes_belief`, `submit_aux`), `driver/worker.rs` (`start`, `advance_slot`'s belief row, `run_chunk`'s
`set_belief`, `restart_search`), `toy.rs`, `mull_game.rs`, `dynapi.rs` / the bindings (`Game.determinize_with`,
`SearchTree.set_belief` / `belief`, `SearchDriver.submit(aux=)`, `belief_weight`, `BELIEF_CLIP`); Python
`mcts.py` (`deal`, `Tree.set_belief`, `check_belief`, `belief_settings`, `belief_tag`), `play.py`
(`NNSeat`, `PyDriver._simulate`, `submit`), `protocol.py` (`ROW_BELIEF`, `belief_rows`, the validators),
`infer.py`, `net.py` (`HeadMissing`, `check_belief_head`, `forward(aux_rows=)`), `replay.py`, `parity.belief`.
Tests: `tree::tests::belief_*`, `a_belief_deals_*`, `no_belief_*`, `without_redet_*`,
`driver::tests::belief_*`, `one_belief_row_*`, `an_evaluator_without_the_belief_*`,
`the_belief_deals_the_searches_worlds`, `root_view_values_and_the_belief_*`, `rolling_phases_ask_the_belief_*`,
`tests/test_belief.py`.

#### 3.8.1 Per-target tilts (`belief_tilts`; MTG Zero's Edge of Eternities deck tilt, 2026-11-10)

A game whose belief predicts two hidden things - MTG Zero's opponent's hand (`belief`) and, with hidden decklists,
their whole unseen deck (`deck`) - may want each dealt by its own tilt, or one without the other. The seat key
`belief_tilts` ({target: alpha >= 0}; absent or {} = none, the seat of before) gives a target of the game's `AUX`
its own alpha:

* **The search**: a seat with a per-target tilt > 0 takes the root's belief exactly as one with `belief_tilt` > 0
  (`NnSeat::belief_on`, `takes_belief`: one `ROW_BELIEF` row per full search, or every search with `belief_fast`),
  even with `belief_tilt` 0. `belief_clip`, `belief_mix` and `belief_fast` apply to the whole deal as before.
* **The game**: `Belief::tilts` holds them in `AUX` order (0 for a target the seat does not name; empty when it
  names none - every `Belief` of before), `Belief::target_tilt(i)` reads one. What a target's tilt deals is the
  game's (`determinize_with`); `Belief::tilt` may now be 0 (the seat's `belief_tilt`). The toy and HiddenMull read
  their one target's tilt when set, else `tilt` (unchanged without per-target tilts).
* **Checks**: the driver refuses a name that is not one of the game's targets, a name twice, an alpha not finite
  or < 0 (`NnSeat::check_belief_tilts`), and per-target tilts on a lookahead seat (not built: its belief is
  `belief_tilt`'s). The bindings: `Game.determinize_with(..., tilts=None)`, `SearchTree.set_belief(..., tilts=None)`;
  the Python tree's belief takes them as a fourth item; `play.NNSeat(belief_tilts=)`.
* **Off**: no seat names one: no key in any seat dict, no row, no change (the golden records and match, the toy runs).

Tests: `driver::tests::per_target_tilts_take_the_belief_and_deal_by_it` (HiddenMull at `belief_tilt` 0 and `hand` 1:
one row per full search, the same shift of seat 1's root value as `belief_tilt` 1), `per_target_tilts_are_checked`;
`tests/test_belief_tilts.py`.

### 3.9 The exhaustive lookahead per information set (`lookahead`; MTG Zero's Delver mirror, 2026-11-01)

**Why.** The Delver Lab's x-ply lookahead (delver-lab `delverlab/xply_is.py`, the per-world tree
`delverlab/xply.py`, Python over the bindings) found lines the network's MCTS never visits - Mishra's
Bauble first at a turn-1 spot, with a card-dependent follow-up -, because it ignores the prior (every
option gets the same look), decides our moves per information set (the right value of information) and
searches every reply. But it walks every node in Python and values the leaves on the CPU (~1,500
positions/s): 5-30 s a 6-ply decision. Here the same lookahead is Rust inside the search crate, a seat
of the driver whose rows go through the same batches as MCTS leaves (one GPU batch for every game in
flight), and a position API for the lab.

**Semantics** (the lab's, ported exactly; `crates/search/src/lookahead.rs`):

* **Plies** are decisions with >= 2 options, of either seat: a move is `tree::transition` (single-option
  decisions stepped, as in the search). A **minor** decision (`SearchGame::minor_decision`: MTG's mana
  payments) takes the network's top prior (the mover's view; the first maximum) and counts no ply.
* **Horizon** (`horizon`): `plies` moves (default 4); 'turn' - a line ends at the first decision of the
  next turn (`rounds()` past the root's); 'next_turn' - at the first decision of the root decider's next
  turn (`SearchGame::active_seat`, a new hook: the seat whose turn it is; default None, the horizon
  refused). With a horizon a line is cut at `max_plies` moves (default 40; counted, `cut`). Checked per
  member in the lab's order: finished, the horizon, minor, the ply cap, no options, ours, the opponent's.
* **Branching**: every option (`topk` 0, the default), or the `topk` of highest prior (stable: ties in the
  options' order) below the root; `topk_root` caps the root too (the lab's information-set solver does,
  its per-world tree never does: the parity runs set it per mode); `topk_from_turn` every option before
  that turn.
* **Our decisions** (`mode` 'infoset', the default). All worlds are expanded in lockstep along the same
  action sequences. At each level - the members between two of our decisions - our members are grouped
  by observation: an exact hash of `encode(R)` (`obs_hash`, 128 bits of the float bits; the lab hashes
  the bytes with blake2b) and the legal options. A group takes ONE action, the one with the best average
  over its members weighted by their weights ("max of average"; a later action replaces the best only by
  more than 1e-12, the lab's tolerance), and every member gets its value under that choice. Our reveals
  (a draw, a look) change the encoding and split the groups: the value of information. `mode` 'world':
  every member its own group (each world a perfect-information tree: the lab's per-world upper bound).
* **The opponent** decides per world (`opp`): 'min' (alias 'best') its reply worst for us - each reply's
  member weighted 1 / replies in our later averages (we learn the reply, not why it was chosen); 'expect'
  (alias 'policy') the prior-weighted mean of its replies, the prior from its own view, member weights
  scaled by it; 'top' its top-prior reply only.
* **Leaves**: the network's value of a value-only row `encode(R)` (P(R wins): root-view leaves, as the
  evaluation seats of the Delver line); a finished game its `result()[R]` (a deferred terminal: the
  network's value). Margins are not used.
* **Worlds** (`worlds`, default 8): dealt once per decision from the root decider's view by the search's
  own deal (`tree::deal`): with the seat's `belief_tilt` > 0 one `ROW_BELIEF` row of the root first, then
  `determinize_with` that belief at the seat's tilt / clip, a `belief_mix` share `determinize`. The
  driver deals from the slot's search stream (reproducible from the driver's seed); the position API
  from explicit seeds (one per world), or takes the worlds given.
* **The decision**: the root option with the best mean over the worlds (the first in the root's order on
  ties: the lab sorts its options by value, stably).
* **The guarded player** (`guard` > 0, a driver seat): the seat's MCTS search decides as usual, and its
  move stands unless the lookahead's best option's mean exceeds the MCTS move's mean by more than
  `guard` (P(win), e.g. 0.05): the lookahead's max over many noisy leaf values is optimistic (the
  optimizer's curse: MTG Zero's teacher experiment, its ENGINE.md 80, found the 6-ply lookahead's
  predicted +2.4 pp realized at -1.0 where it disagreed with MCTS), large margins are not. Minor
  decisions are the MCTS search's alone. `guard_decisions` / `guard_overrides` count them.
* **Budget** (`budget` nodes per decision - the lab's count: children created -, 0 = none): over it the
  attempt is abandoned and the next stage runs on the same worlds: `fallback_topk` (default 4) below the
  root, then each of `fallback_plies` (default one stage at max(2, plies - 2), 4 for a horizon), then the
  final fallback `fallback` 'mcts' (the seat's own MCTS search of the decision: its `sims`, rule, belief,
  leaf view - an evaluation seat's settings) or 'prior' (the network's top prior). `LookStats` /
  `stats()['look']` count what each decision took (`by_stage`, `fallback_*`, `aborted_nodes`).

**Leaf values by observation** (`dedup`, default on): a leaf's value is the network's answer to its row
`encode(R)` alone, so leaves whose encodings are equal share one row - worlds the root decider cannot tell
apart after the same moves: ~70% of the leaves on turn 1 of the Delver mirror. Exact up to the network's
float rounding at another batch shape; `value_dups` counts them (`value_rows` the rows sent; the lab's
value rows = their sum). `dedup` 0 sends a row per leaf, as the lab.

**How it runs.** `lookahead::Look` is a state machine, level by level as the lab's recursion (its
members, groups and order are the lab's: the group, action, member order of the children; the minor
members then the opponent's children per advance round). It asks for network rows through a `RowSink`
(`Look::step` -> `Step::Rows(n)`, then `Look::answer`): the priors that choose what to expand or weigh -
minor decisions, the opponent's under 'top' / 'expect' or a cap, our groups under a cap - block their
level; leaf values never do and fill a round up to `rows` (default 512), a value round going out as soon
as `rows` of them wait (their games dropped once encoded: memory). Rows are encoded in place in the
sink's buffer (`push_row`; a duplicate observation's row popped again). Memory: one game per live member
(MTG: ~11 KB) - `peak_games` counts the most live at once; the node budget bounds it.

**The driver** (`driver.rs`, `driver/worker.rs`). Seat key `lookahead` (a dict of its settings; the Rust
`LookCfg::parse` checks them: an unknown key, a wrong type or a bad value is a ValueError, never
ignored; None = off). A lookahead seat is an evaluation seat (`selfplay` refused: no training records
yet). At each of its decisions with >= 2 options `look_step` creates the `Look` (the seat's belief
settings), steps it into the seat key's leaf buffer (`Pending::Look`: the game waits for its round's rows,
as a leaf), answers it from the group's evaluation (`run_chunk`), and plays its move (`slot.decisions`
counts it) - or, after the final fallback 'mcts', lets the seat's MCTS search the decision (`look_mcts`,
cleared by the move). A minor root decision takes the network's top prior (one row; `minor_root`).
Every lookahead row is a `ROW_LEAF` row (priors: its options; values: none) or the root's `ROW_BELIEF`:
the batches carry many rows per game, nothing else changes for the evaluator. A new search epoch
(rolling) or a phase's dropped batch restarts the decision's lookahead. Counters: `DriverStats::look`
(`LookTotals`), `stats()['look']` in Python - only for a driver that made lookahead decisions (the stats
of before otherwise).

**The evaluator.** `NetEvaluator(max_rows=n)` (opt-in; None: every batch whole, as before) evaluates a
batch of more than n rows in consecutive pieces, their answers concatenated: a lookahead's batches hold
thousands of rows, and the pieces keep a token network's CUDA graphs (docs/az.md 10.6) to a few shapes.

**The position API** (the lab's call). `<pkg>._engine.Lookahead(game, settings, worlds=None, seeds=None,
seed=0, belief_tilt=0, belief_clip=20, belief_mix=0, belief=None, lines=True)` with `next_batch()` /
`submit(priors, value, aux=None)` / `result()` / `line(world, action)` / `worlds()`;
`zero.lookahead.evaluate_position(game, ev, settings, ...)` drives it with an evaluator and returns per
root option its value in each world, mean +- sd, the modal principal line with labels (each world's line
replayed on its world: the exported state's option labels, `reveal(keys, player)` for what a move showed)
and the counters. `zero.lookahead.parse('plies=6,worlds=8,opp=best,budget=200000')` reads the
command-line form (`fallback_plies=4/2`), `settings(d)` the canonical dict, `tag(d)` a short name.

**zero.h2h**: `--a-lookahead` / `--b-lookahead` SETTINGS (`--lookahead` both): that side a lookahead seat
over its seat settings (its MCTS settings: the fallback; its belief: the worlds' deal); the output adds
`lookahead_a` / `lookahead_b`, `lookahead_tag_*`, `look` (the drivers' counters summed, per decision
means) and `max_rows` (default 1024 with a lookahead side). `--conc` sets the games in flight (default:
every game, at most 1024, as before).

**Parity with the lab** (MTG Zero's ENGINE.md 80.4): on the same dealt worlds, the same network (CPU),
both modes, the three opponent models, 4 and 6 plies, five positions - the values per world to 3.6e-7, the
choices, node / leaf / group / prior-row counts identical in 59 of 60 configurations (the 60th a root
near-tie, two options within ~1e-7, flipped by float rounding).

**Off** (the default: no seat key `lookahead`): the driver, the trees, the bindings' existing classes and
every evaluator of before, byte for byte (`check_checkpoint_compat`, toy runs; section 3.9's checks in
docs/az.md 21).

**Tests**: `lookahead::tests` (13: the test game `peek_game::Peek` - a hidden coin seat 0 may peek at -,
where looking is worth +0.32 per information set and -0.08 per world; the opponent models; the minor
payment; the horizons; the budget's stages to MCTS or the prior; the settings; top-k; reproducible deals
with the belief; `obs_hash`; a lookahead seat in the driver, reproducible across threads; its MCTS
fallback with the belief; sparse batches carrying the same rows and games as dense ones; the guarded
player's margin), `tests/test_lookahead.py` (10: settings, the seat key, the driver, the position API, the
protocol's refusals, h2h's flags, `max_rows`, sparse batches through the driver and through the token
graphs bit for bit).

**Measured** (MTG Zero's Delver mirror, its ENGINE.md 80): one position at a time on the GPU, 4 / 5 / 6
plies every option 0.19 / 0.37 / 0.80 s a decision (2.1k / 6.3k / 18k nodes; the lab's Python ~7-10 s for
a 6-ply turn-1 spot); against MCTS at 32 simulations (the same network, paired games) the pure lookahead
player loses (4 plies 0.32-0.39, 6 plies 0.37-0.47) and the guarded one at 5 pp is even (0.515 +- 0.015 over
300 pairs at 4 plies).

Code: `lookahead.rs` (`LookCfg`, `Look`, `RowSink`, `Answers`, `LookStats`, `obs_hash`), `peek_game.rs`,
`game.rs` (`SearchGame::active_seat`), `tree.rs` (`deal`), `driver.rs` (`NnSeat::look`, `Pending::Look`,
`LookTotals`), `driver/worker.rs` (`look_step`, `LeafSink`), `dynapi.rs` (`DynGame::lookahead`, `DynLook`,
`LookRows`), the bindings (`Lookahead`, `lookahead_settings`, seat key `lookahead`, `stats()['look']`);
Python `lookahead.py`, `infer.py` (`max_rows`), `h2h.py`.

### 3.12 The game's evaluation cache (`eval_cache`; MTG Zero's Delver mirror, 2026-11-03)

**Why.** A game's searches send the network the same rows again and again. Measured on MTG Zero's Delver
line (48 self-play games of its gen 960 at its exact self-play seat, 520,304 rows; its ENGINE.md / the probe
in docs/az.md 25): 33.8% of the rows repeat, bit for bit, a row the same game sent earlier - 62.6% of the
root-view value-only rows (3.6), 19.0% of the leaf rows, every root belief row (the root's own leaf row) -
and none repeat across games. The source is hidden information, not move order: each simulation deals the
other seat a new hand (`redet`), so its decision nodes differ (their option lists), while the root decider
sees one position - its own decisions repeat 34% (21% within a single search), the other seat's 3.8%.
Putting the permanents in a canonical order adds 0.6 points. The legal options never told two equal
feature rows apart.

**What it does.** A network seat with `eval_cache` N > 0 (`NnSeat::eval_cache`; the bindings' seat key
`eval_cache`, run setting `eval_cache`) keeps, per game in flight (`Slot::cache`), the answers to the rows
the game sent its network: the priors of the row's options, its value and margin per seat, by the key
(evaluator key, network id, `cache_key(obs_hash(features), options)` - the lookahead's 128-bit hash of the
float bits, 3.9, with the options in order; a value-only row has none; a sparse driver hashes the row's
non-zeros, `obs_hash_sparse`, the same key: 2.12). At a search leaf (`advance_slot`,
`Sim::Leaf`):

* each of the leaf's rows - its own row, and with `LeafView::Root` at another seat's leaf the root decider's
  value-only row - is encoded and looked up alone; a cached row is dropped from the batch (`LeafBuf::unrow`),
  a missing one is sent as before;
* every row cached: the leaf is expanded and backed up at once and the search goes on (a fresh root whose
  leaf was cached then sends its belief row alone, a reused root's way, 3.8);
* otherwise the leaf waits for its batch with what it took from the cache (`Pending::Leaf::part`, `LeafPart`:
  the rows it sent are consecutive from its `local`, the belief row after them); the answers to the rows it
  sent go into the cache when they come back.

The cache is emptied as a game starts (a branch too, 2.11), as a new search epoch begins (a rolling driver's new network: the
network id is in the key too, so another network's answer is never found) and when it holds N answers.
Each evaluator key has its own answers (a match's two networks never share one). The lookahead (3.9) keeps
its own per-round dedup and does not use the cache.

**What changes.** The answers a search reads are the network's answers to the same rows, computed in another
batch: equal up to the network's float rounding at another batch shape (an evaluator whose answer is a
function of the row alone gives the same games and records bit for bit: the tests). A leaf answered from
the cache takes no batch round, so a game moves faster through its rounds than the others: when games
finish changes, and with it the order of their records, a rolling phase's quota (which games it ends,
which are in flight at a network change) and a run's batch shapes. Off (0, the default), nothing changes:
no key, the rows, batches and games of before, byte for byte.

**Stats** (`Driver::stats`, the bindings' `stats()`): `cache_hits` (rows answered from a cache instead of
sent), `cache_leaves` (leaves expanded at once; not in `leaves`, which counts the leaves that sent rows),
`cache_partial` (leaves that sent one row and took the other from the cache), `cache_clears` (caches
emptied because full). Every row is sent or answered from the cache: rows sent + `cache_hits` = the rows
without the cache, for the same searches.

**Memory.** An answer is ~100 bytes plus its priors; N answers per game in flight at most.

**Tests** (`driver::tests`): `the_eval_cache_plays_the_same_games_with_fewer_rows` (ToyGame self-play, an
evaluator of the row alone: the same results and records - as a multiset - bit for bit, as many
simulations, rows sent + hits = rows without; dense and sparse batches; a cache of 3 emptied 7,143 times),
`the_eval_cache_keeps_each_networks_answers_apart` (a match of two keys bit for bit; rolling phases over a
network change: after it every cache holds the new network's answers alone),
`the_eval_cache_with_root_view_value_rows` (value-only rows cached, leaves answered in part, the same games),
`the_eval_cache_with_root_view_rows_and_the_belief` (HiddenMull with the root's belief: one belief row per
search either way, the same records).

Code: `driver.rs` (`NnSeat::eval_cache`, `Cached`, `LeafPart`, `cache_key`, `Slot::cache`, the stats),
`driver/worker.rs` (`advance_slot`'s cached leaf, `run_chunk`'s answers, `row_answer`, `cache_put`, the
emptying in `init_slot` / `restart_search`), the bindings (seat key `eval_cache`, the stats); Python
`games.nn_seat(eval_cache=)`, `run.py` (`eval_cache`, `Run.cache_metrics`).

### 3.13 A prior exponent in the 'cq' target (`cq_prior_w`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** The 'cq' target of 3.1 is `pi' ~ P0 exp(sigma completedQ)` with P0 the prior of the network that
plays. The policy fits its targets (MTG Zero's Delver line: train policy CE = the targets' entropy), so from
generation to generation `pi_{t+1} ~ pi_t exp(sigma Q_t)`, and after T generations `logit_T = ln pi_0 + sigma
sum_t Q_t`: **the policy's logits are the cumulative sum of every advantage a search ever measured, with no
forgetting.** Three consequences, all visible on the Delver line: a favourite whose Q sits a hair above v_mix
gains a little every generation, so the targets drift to one-hot (gen 859's 26,746 recorded decision rows:
max-share > 0.99 in 54.6%, > 0.999 in 38.7%; 52.2% with normalised entropy < 0.05); an option visited once on a
bad leaf loses sigma x delta logits at once and, once its prior is low enough that PUCT stops visiting it,
is never measured again (an unvisited option's completed Q is v_mix, about the favourite's: it neither gains
nor recovers); and the targets' entropy falls whatever the strength does (0.35 -> 0.20 over 1,000
generations while the anchored Elo was flat for hundreds at a time). Fable's review of 2026-11-04 (MTG Zero
hub notes, `E:/mtg-zero-scratch/fable-ideas/report.md` 2.1) ranked it first.

**What it does.** A 'cq' seat's prior exponent beta (`PolicyTarget::CompletedQ::prior_w`; the bindings'
seat key `cq_prior_w`, run setting `cq_prior_w`) records

```
pi' = softmax(beta ln max(P0, PRIOR_FLOOR) + sigma(completedQ))      beta in (0, 1]; 1 = the target of 3.1
```

i.e. `pi' ~ P0^beta exp(sigma completedQ)`. Fitted generation after generation, `logit_T = beta^T ln pi_0 +
sigma sum_t beta^(T-t) Q_t`: an exponentially weighted sum with a bounded fixed point `pi ~ exp(sigma Q /
(1 - beta))`, and the log-ratio of an option no search measures to the favourite shrinks by the factor beta
a generation, so within ~1/(1 - beta) generations its prior is high enough again for the root noise and
PUCT to visit it and its Q is measured afresh. At the Delver line's sigma (c_visit 50, c_scale 0.3, 64
simulations: (50 + max N) x 0.3 ~ 30): a systematic 1-pp Q gap is worth 0.3 logits a generation forever at
beta 1, 1.5 logits at the fixed point at beta 0.8 (82 / 18; a dead option relaxes in ~5 generations), 0.75 at
0.6 (68 / 32; ~2.5 generations), 0.6 at 0.5.

* **Only the prior's term.** v_mix keeps its P0 weights (the completed value of the unvisited options is the
  search's estimate of the node, not part of the target's regulariser) and sigma is unchanged. The logits are
  `completed_q_logits + (beta - 1) ln max(P0, PRIOR_FLOOR)` in f64, then the softmax (`tree::completed_q_policy_w`,
  `Tree::cq_target_w`); at beta 1 the term is skipped: `completed_q_policy` / `cq_target` are those functions at
  1.0, bit for bit.
* **Only the recorded target.** The worker's record (`decide`) is the one reader. The search (PUCT and its
  priors, root noise, temperature), the 'cq' move rule (`Select::Cq`, `completed_q_logits`, 3.3), the Gumbel
  root's scores, schedule and move (3.7), the root value q, z and the value-only samples are those of beta 1:
  for the same network a seat plays the same games at every beta, and two runs of an A/B diverge only through
  what their networks learn. (MTG Zero's ENGINE.md 59: a training A/B must never move the evaluation rule.)
* Python reference: `mcts.completed_q_policy(..., prior_w)` / `mcts.cq_target(..., prior_w)` (the same
  arithmetic), `play.NNSeat(cq_prior_w=)` for the Python driver; `mcts.check_cq_prior_w`.

**Settings.** Seat key `cq_prior_w` (absent = 1.0; refused outside (0, 1], and anything but 1.0 with
`policy_target` 'visits', which has no prior term to scale); `games.nn_seat(cq_prior_w=)` adds the key only
when != 1. Run setting `cq_prior_w` (default 1.0) goes to the self-play seat alone (`Run.nn_seat(selfplay=True)`,
so the local linger's and the remote's seats too; evaluation seats record nothing), checked by
`check_eval_settings` (also for `resolver_cfg`). Bindings: `SearchTree.root_cq_target(c_visit, c_scale,
prior_w=1.0)` and the module function `completed_q_policy(p0, n, w, v, c_visit=50, c_scale=1, prior_w=1)` (the
parity tests' handle on the Rust arithmetic).

**Off by default.** beta 1.0: no seat key, `PolicyTarget::parse` gives `prior_w` 1.0, and the records are those
of before byte for byte (`driver::tests::GOLDEN_CQ_RECORDS` / `GOLDEN_CQ_SKEWED_RECORDS`: digests of aae4ca2's
'cq' records, before the setting existed; `tools/check_checkpoint_compat.py --package zero --net random --game
toy` against aae4ca2: identical).

**Costs.** One `ln` and one multiply-add per option of each recorded decision. Nothing in the search.

**Remote self-play.** A remote worker builds its seat from the run's config.json with its own checkout: one
without this setting would record beta-1 targets in silence (docs/az.md 18.5's gap). MTG Zero's Mac checkout is
the run's commit (`RunCommitRemote`); any other remote must run a checkout with this core.

**Tests.** `tree::tests::completed_q_prior_w_by_hand` (50-digit references at beta 0.5 / 0.6 / 0.7 / 0.8;
nothing visited -> P0^beta normalised, 0.8 / 0.2 at 0.5 -> 2 / 1; beta 1 = `completed_q_policy` bit for bit;
`ln pi_beta - ln pi_1 = (beta - 1) ln P0 + const`; a zero prior; one option; the tree's target),
`driver::tests::cq_prior_w_changes_only_the_recorded_targets` (prior_w 1.0 reproduces aae4ca2's 'cq' record
digests; skewed priors, beta 0.8 and 0.5: the same games and every record field but pi equal, > 90% of the
targets differ, the log-ratio identity per root within 1e-3, normalised entropy up and max-share down at each
step; with the 'cq' move rule and with the Gumbel root, beta 0.6 plays the same games; beta 0, -0.5, 1.0001, 2,
NaN, inf refused, 'visits' takes 1.0 only), pytest `tests/test_cq_prior_w.py` (the hand-checked numbers in Python
and Rust; 40 random (P0, N, W, v, beta) Python = Rust within 1e-6; the Python formula on Rust roots at beta 1 /
0.8 / 0.5 / 0.2; the Rust driver and the Python driver (both trees): only pi changes; the seat key and its
refusals; the run setting reaching the self-play seat only).

**Measured.** The toy (skewed priors, c_scale 0.3, 24 games, 371 records): mean normalised entropy of the
targets 0.546 (beta 1) -> 0.576 (0.8) -> 0.605 (0.6) -> 0.619 (0.5), mean max-share 0.641 -> 0.624 -> 0.608 ->
0.600. MTG Zero's Delver line (its checkpoint 1000 and exact self-play seat, 24 games on the CPU, the same driver
seed: the same games at every beta, 1,475 recorded decisions): targets with max-share > 0.99 56.9% at beta 1 (the
prior itself: 51.9% - the search sharpens it every generation), 48.6% at 0.8, 38.2% at 0.6; details and the planned
A/B in docs/az.md 26.

Code: `tree.rs` (`completed_q_policy_w`, `Tree::cq_target_w`), `driver.rs` (`PolicyTarget::CompletedQ::prior_w`,
`PolicyTarget::with_prior_w`), `driver/worker.rs` (`decide`'s record), `dynapi.rs` (`root_cq_target`), the
bindings (seat key, `root_cq_target`, `completed_q_policy`); Python `mcts.py`, `games.py`, `play.py`, `run.py`,
`parity.py` (`cq_formula(prior_ws=)`).

### 3.14 Self-play resignation with a no-resign share (`resign_t`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** A third of a self-play game's searches can come after the game is decided. Measured on MTG Zero's
Delver line (gens 855 and 859, 958 whole games, 90,061 rows): 36.5% of the decision rows have the decider's
root win estimate outside (0.05, 0.95) and the outcome agrees with it in 98.9% of them. KataGo ends such games
early and keeps a share of games that never resign, to measure the rule and to keep showing the value head
decided positions. docs/az.md 27 has the run settings, the measurements and the recommended A/B.

**What it does.** A network seat with `resign_t` t > 0 (`NnSeat::resign_t`, `resign_k`, `resign_noresign`;
the bindings' seat keys of the same names, defaults 0 / 5 / 0.1; `games.nn_seat(resign_t=...)`) is a
**resigning seat**. A game with a resigning seat keeps one counter (`Slot::resign`, `ResignState`):

* **Which searches count**: a resigning seat's **full** searches - exactly the searches that make its
  training records (`slot.full`, after the deviation moves of a branch, 2.11). Fast searches (`sims_fast`),
  minor ones (`sims_minor`) and a branch's deviation evaluations never count, and never reset the count.
  The other seat's full searches count too when it is a resigning seat (a self-play mirror: both); a seat
  that does not resign (another network) neither counts nor resets.
* **Whose value**: the search's root win vector as its record stores it (`q`): `Tree::win(root)`, the root
  floor's `root_win_pruned` (3.5), the Gumbel root's `gumbel_q` (3.7) - by absolute seat. Its **favoured seat**
  is its argmax (the first maximum on ties); the search is **decided** when the favoured seat's estimate is
  >= 1 - t. In a two-player game that is the decider's own q <= t or >= 1 - t.
* **The count**: a decided search for the same favoured seat as the run's adds 1; decided for the other seat
  starts a new run at 1; a counted search that is not decided resets it to 0. The rule **fires** at the first
  search where the run reaches `resign_k`.
* **Resigning**: the firing search's decision is recorded as usual (its record and, with `view_samples`
  3.6, its value-only records), then the game ends there, before its move (`decide` returns; `advance_slot`
  finishes it). Its result is the favoured seat's win (1 for it, 0 for the other; never a draw); every record
  of the game is kept, `z` = that result relative to each record's decider, `q` the searches' own root values
  (a value-only record still waiting for its observer's next search gets `z`, as at any game end); `sc` =
  `margin()` and the row's `scores` = `score()` **of the position where it resigned** (the game was not
  played out). The game is `over`; rolling quotas, `queue_only` setups, the shared queue and the local linger
  count it like any finished game (it is one result row).
* **The no-resign share**: game `gid` of a driver seeded `seed` never resigns when `noresign_pick(seed, gid,
  resign_noresign)` (a hash of (seed, gid) with a salt of its own, `view_pick`'s arithmetic: no random stream
  draws it, so no game changes). It plays out as without resignation and notes the first time the rule fired
  - where it **would have** resigned. `resign_noresign` 1 is a dry run: no game resigns, every row reports.
* **Branches** (2.11): a game that resigns spawns its branch like one that plays out (the branch point is drawn
  over its eligible decisions up to the firing one). A branch counts its own searches from its start (after its
  deviation moves) with a fresh counter; it shares its game's gid and so its no-resign draw.

**Result row** (2.5): a game with a resigning seat has `resign` = {`resigned` (it ended by the rule),
`noresign` (a no-resign game), `winner` (the favoured seat when the rule fired - resigned there, or would
have; None: it never fired), `decision`, `step`, `records` (at the firing search: the game's searched
decisions with the firing one, real moves before it, records made with the firing decision's; 0 when it never
fired)}. A resigned game's `steps`, `decisions` and `samples` equal its `step`, `decision` and `records`; a
no-resign game's `samples - records` are the rows resigning would have saved. Games without a resigning seat
have no key.

**Refused** (`Driver::new` unless said): `resign_t` outside [0, 0.5), `resign_k` 0, `resign_noresign` outside
[0, 1]; a resigning seat that records no self-play (an evaluation seat, `record` false - a league opponent -,
a driver with `record` false); a game of other than two seats with a resigning seat; resigning seats of one
game with different settings (one counter per game); a game whose `team_size` is not 1 (when it starts); a
game with a resigning seat that reaches a deferred terminal (2.8; at that terminal - a deferred game that
resigns before reaching one is not caught here: the run settings refuse co-training, docs/az.md 27). The
Python reference driver (`zero/play.py`) has no resignation (its `NNSeat` refuses the keys).

**Off** (`resign_t` 0, the default): no key in the seat dict, no counter, no `resign` in a row - the driver of
before, byte for byte (the golden records). With it on, the searches, moves and records of every game up to
its firing search are those of the game without it (nothing reads a stream); a resigned game's records are
that game's first `records` records with `z` replaced, and nothing after.

**Tests** (`driver::tests`): `resignation_off_or_a_dry_run_plays_the_games_of_before` (off: the golden records;
a dry run at t 0.3, k 2: the golden records and results, every row reporting), `resigned_games_end_where_the_rule_fires_with_its_outcome`
(every game resigning vs the dry run, with an evaluator of the row alone: a resigned game is its dry twin up to
the firing search - the same x, pi, q, opts -, z the favoured seat's win, nothing after it, the firing search's
favoured estimate >= 1 - t; three seats: every search full, a playout cap - fast searches never count -, a
Gumbel root), `the_no_resign_share_is_a_hash_of_seed_and_gid` (the draw; a run at share 0.4 = the dry run's
games where it draws a no-resign game and the all-resigning run's elsewhere), `resigned_games_count_in_rolling_quotas`,
`branches_of_resigning_games`, `resign_settings_are_checked`; tests/test_resign.py (the bindings, the run
settings, the loop, remote self-play).

Code: `driver.rs` (`NnSeat::resign_t`, `resign_k`, `resign_noresign`, `NnSeat::check_resign`, `noresign_pick`,
`ResignState`, `ResignInfo`, `GameResult::resign`, `Slot::resign`, the checks in `Driver::new`),
`driver/worker.rs` (`resign_state`, `init_slot`, `decide`'s count, `advance_slot`'s end, `finish`, `emit`,
`spawn_branch`), the bindings (seat keys, the row's `resign`, `noresign_pick`); Python `games.nn_seat(resign_*)`,
`zero/resign.py`, `run.py`, `remoteplay.py`.

### 3.15 The tree's shape, and the information-set merging gate (`Tree::shape`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** A review of the Delver line (Fable, 2026-11-04) proposed merging the tree's nodes by the root decider's
observation (single-observer information-set MCTS inside the tree: every node keyed by `obs_hash(encode(viewer))`,
the other seat's nodes subset-armed over the union of their option lists, Cowling, Powley & Whitehouse 2012), from
the evaluation cache's measurement (3.12) that a third of a search's leaves re-expand an information set of the
root decider the tree already holds. Its gate before any build: a tree-shape probe of the run's 64-simulation
self-play search - build only if the mean leaf depth is < 2.5 and the main line's edge into the other seat's
decisions spawns > 8 child nodes ("fragments": the tree keys a child by chance key, decider and legal options,
so another dealt hand with other options is another child).

**The diagnostic** (read-only; no search change). `Tree::shape()` -> `Shape`: the nodes reachable from the root
(each once, at its shortest depth), the expanded ones per depth (root 0; in a fresh search each expanded node was
one walk's leaf), per number of other-seat nodes above them (`opp_hist`) and per decider changes above them
(`switch_hist`), the deepest node, every edge with a visit or a child (`EdgeShape`: depth, whether the viewer
decides its node, visits, child nodes, of them the other seat's and the expanded ones) and the main line (the most
visited option from the root on, then that edge's most backed-up child). Bindings: `SearchTree.shape()` (a dict),
`SearchTree.path_len` (the last simulation's walk length: its leaf's or terminal's depth). Each edge also counts
its other-seat children's distinct option lists and chance keys, the backups through them and those of the most
backed-up list (`opp_lists`, `opp_keys`, `opp_tot`, `opp_modal_tot`: the strategy-fusion sizing of 3.16).

**Measured** (desktop CPU at idle priority beside the live run; r3-2b-float550 gen 1000, its exact full self-play
search - c_puct 1, root noise, root belief tilt 1 / clip 20 / mix 0.25, root-view leaf values, redeterminized
worlds, chance_k 0 - on 50 positions of its network's own play (prior-sampled, 2,043 candidate decisions, PayMana
left out), 4 fresh searches each; `E:/mtg-zero-scratch/ideas-4/gate_probe.py`, its NOTES.md):

| | 64 sims | 32 sims (the evaluation seat's) | 16 sims |
|---|---:|---:|---:|
| mean walk depth (leaf or terminal; root's own evaluation left out) | 5.10 | 4.22 | 3.31 |
| walks ending at depth 1 / 2 | 11.4% / 14.7% | 16.7% / 18.4% | 25.2% / 21.8% |
| other-seat decisions above a new node: mean (share none / one) | 1.43 (40% / 28%) | 1.06 (49% / 27%) | 0.71 (63% / 23%) |
| new nodes whose root-decider observation the search held already: own / other seat / all | 11.6% / 48.4% / 29.6% | 8.4% / 44.1% / 25.5% | 4.7% / 36.5% / 18.7% |
| main line's first edge into the other seat: child nodes mean / median / p90 (> 8 in) | 6.1 / 3 / 12 (16%) | 4.4 / 3 / 10 (14%) | 3.2 / 2 / 7 (5%) |

**Decision: the gate fails; the merge was not built.** MTG walks are deep in decisions (the root decider's own
consecutive decisions - cast, target, pay, attack - each a level) and the main line fragments less than assumed;
the repeat share (30% of a fresh 64-simulation search's new nodes, ~18 walks of 63) is real but under the 36.5% the
proposal started from, i.e. ~1.4x the effective walks, less than the 2x of 128 simulations that measured "equal or
slightly better" (MTG Zero's runs). What the tree IS shallow in is the other seat's replies (68% of new nodes have
at most one above them at 64 simulations), and the strategy-fusion leak of the unmerged tree (the root decider's
next node split by what the other seat was dealt) cannot be measured by a shape probe - only by a paired match of
a built merge. The worked-out design (a two-level child identity: today's per-edge signature as a link, mapped to
a merged node by the observation at a new signature only; no depth in the key, a path check with alias nodes
against cycles; subset-armed PUCT with availability counts and mean priors; growth leaves for a world's new
options; `advance` by the real observation; `compact` once per node) and the h2h plan are in that NOTES.md. The
Python reference has no merged mode. (Built afterwards on the user's decision: 3.16.)

Tests: `tree::tests::shape_counts_every_walk_and_edge` (a fresh search's expanded nodes + terminal walks = its
simulations, the histograms and the edges' children add up, the main line follows the visits), pytest
`test_bindings.py::test_search_tree_api`.

### 3.16 Information-set merging by the root decider's observation (`infoset_merge`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** The tree keys a child by (chance key, decider, legal options) under its parent edge (section 3). When the
other seat's options depend on its hidden cards - MTG Zero's Delver mirror: Daze or Force of Will in the sampled
hand, or not - each dealt world with another option list gets its own node of the other seat, and the root decider's
next decision is split with them: one information set of the root decider becomes several nodes, each with its own
statistics, so PUCT can choose differently at one information set depending on what the other seat was dealt
(strategy fusion: the search values the root decider as if it knew the other hand two plies down). 3.15's gate
(build only if walks are shallow and the main line fragments > 8 ways) failed; the user decided to build it anyway
for the strategy-fusion question, to be answered by an evaluation-only paired match. Sizing (Step 0, below): at the
main line's first edge into the other seat, 65-72% of the run's searches see two or more option lists, ~28-31% of
the walks through it go to the less common lists, and the children differ by option list, not chance key (1.07-1.15
distinct keys per edge).

**What it does** (`tree.rs` module docs and `tree/merge.rs`; `Tree::set_merge`, off by default). Single-observer
information-set MCTS inside the tree (Cowling, Powley & Whitehouse 2012, "Information Set Monte Carlo Tree Search"):

* **Links.** An edge's children are links: the signature above and the node it leads to. A walk follows a known link
  without encoding anything (by the game contract the same path and chance keys give the root decider the same
  observation: revelations are keyed, section 1). At a new signature - exactly where the tree of before creates a
  node - it encodes the reached state from the viewer's view (`encode_sparse(viewer)`, the eval cache's key
  `obs_hash_sparse`, 2.12 / 3.12) and looks the observation up in the tree's map (key: the observation hash, the
  decider, and for the viewer's own nodes their options - equal features always had equal options, 3.12; a world
  where they differ stays apart): found, a new link to that node - the tree is a DAG; not found, a new node.
* **No depth in the key; a path check.** World-dependent auto-steps make depth world-dependent (the other seat with
  a single option is auto-stepped, one holding Daze decides: the root decider's next information set sits at depth 1
  in one world and 2 in the other - the strategy-fusion case itself), so the key holds no depth. A walk never holds a
  node twice: a link (known or new) that would lead to a node already on the walk's path leads to a fresh unmapped
  node instead (an alias; a known link is re-pointed to it), so path-based backup stays exact.
* **Subset-armed nodes of the other seats.** An other seat's node holds the union of its worlds' option lists
  (appended; indices never change). A walk takes only an option legal in its world: `argmax Q_a + c_puct P'_a
  sqrt(max(avail_a, 1)) / (1 + N_a)` over them, P' = the option's mean prior (over the network rows that offered it)
  renormalised over the available options, avail_a = the walks at the node in which a was legal (counted at
  selection), FPU the node's `vec`. The root decider's nodes keep PUCT (their options are fixed; `tot` counts the
  backups from every parent).
* **Growth leaves.** A walk reaching an expanded other-seat node with an option outside its union ends there:
  `Sim::Leaf` on the expanded node; the caller evaluates that world's row with its options (`Tree::leaf_acts`),
  `expand_leaf` appends the new options and folds the priors into the means, and the value is backed up along the
  path as any leaf's (the node itself gets no visit, as a leaf never does). A world whose options are all known walks
  on without a network row.
* **Values: root view only.** A merged node of another seat is valued from the root decider's view (3.6), the same in
  every world of its observation; the mover's view would be one world's opponent view (the driver refuses the key
  without `leaf_value_view` 'root').
* **Reuse, compaction, reset.** `Tree::advance_in(a, key, player, legal, live)` (the driver's tree reuse) finds the
  new root by the real signature, else by the real observation (the live game's `encode(viewer)`: no hidden
  information; an unknown real key - a revelation the search never sampled - is no longer a lost subtree), for the
  same viewer only; the map is then rebuilt from the nodes reachable from the new root (or `compact`, above 65k nodes,
  copies each reachable node once with its links, outs and side statistics). `reset` drops the merge state (the
  driver calls `set_merge` after each reset: `start`, `start_dev`, `restart_search`). The root (the viewer's node,
  on every walk's path) is never merged into; deferred terminals are not merged.
* **Unchanged:** the root's statistics are the same kind as before, so the Gumbel root (3.7), the root floor (3.5), the
  cq target, the move rules, root noise, the belief (3.8) and the eval cache (3.12: a growth leaf's rows are looked up
  like any) read and work as before; backup is path-based as before.

**The seat key** `infoset_merge` (bool; `NnSeat::infoset_merge`, the bindings' seat key, `games.nn_seat(
infoset_merge=)`), on evaluation and self-play seats alike (self-play: below). Refused without `leaf_value_view`
'root'. The Python reference `mcts.py` has no merged mode: `play.NNSeat(infoset_merge=True)` needs `impl='rs'`
(the Rust tree; `RsSearch` calls `set_merge`). Run settings `eval_infoset_merge` and `selfplay_infoset_merge`
(docs/az.md 30); h2h `--a-infoset-merge / --b-infoset-merge` (or `--infoset-merge`). Off (the default) none of the
merge runs: the tree of before, bit for bit (the golden records and matches).

**Driver touch-points** (`driver/worker.rs`): `set_merge(seat.infoset_merge)` after each reset; `tree.leaf_acts(node)`
in place of `tree.acts(node)` at the leaf's row (two places: with and without the eval cache) and in `apply_eval`'s
check; `apply` calls `advance_in` with the live game; the leaf's row of the viewer comes from the merged tree's
encoding when the walk encoded its leaf (`Tree::leaf_obs`, under Self-play below). Stats (`Driver::stats`, the
bindings' `stats()`): `merge_links` (new child signatures), `merge_merged` (of them, found by observation),
`merge_aliases`, `merge_growths` (growth leaves; in `leaves` too), `merge_shared` (leaf rows of the viewer taken from
the tree's encoding). Bindings: `SearchTree.set_merge` (a fresh tree only), `merge_on`, `merge_stats()`;
`simulate` gives a growth leaf's options and `expand` takes them; `advance` uses `advance_in`.

**Costs.** One `encode_sparse(viewer)` + `obs_hash_sparse` per new signature (about one per walk), a hash-map lookup,
a link per new signature (its key and options), and per stat 16 bytes of side statistics; nothing per known link. A
merged walk that reaches a known information set walks on, so a search's walks are deeper (more nodes per walk on
the path, PUCT at each). The driver used to encode the leaf's row of the viewer again (another seat's leaf's
value-only row, a viewer's leaf's own row); since the self-play build it takes the tree's encoding of it
(`Tree::leaf_obs`), so a merged walk's extra encodes are its merges and `advance_in`'s lookups. Merged walks send more
distinct rows per simulation (fewer re-expanded information sets for the evaluation cache to answer: measured under
Self-play).

**Tests** (`tree::tests`, `driver::tests`, pytest `tests/test_infoset_merge.py`). A new test game `counter_game`
(`#[cfg(test)]`): seat 1 holds hidden Daze / Force (1/2 each); seat 0 casts or holds for R rounds; a cast is answered
with ALLOW plus the counters held (auto-stepped when none) - the other seat's option list depends on its hidden hand;
spec `{"rounds", "forget"}` (forgetful features repeat observations along a line).

* `the_counter_game_merges_the_other_seats_options_and_the_next_round`: one seat-1 node under CAST over the union
  {ALLOW, DAZE, FORCE} (every seat-1 link leads to it), ALLOW available in every walk, each counter in 45-90% of
  them; seat 0's next round one node reached from the root's CAST (the auto-stepped world, depth 1) and seat 1's
  ALLOW (depth 2); the tree of before splits it into >= 3 nodes and holds more nodes; merges and growth leaves happen.
* `merged_trees_recount_exactly_with_chance_noise_floor_and_gumbel`: toy (chance keys, chance_k 0-2) and counter,
  PUCT / root noise / root floor / Gumbel root: N and W per edge, tot and wsum per node recounted bit for bit from
  the backed-up walks, no walk holds a node twice, N add up to tot and W to the decider's wins at every node,
  availability >= N and <= tot.
* `merged_compact_copies_each_node_once_and_searches_on_identically`: compact = the same search (a canonical dump
  through the links) and the same walks afterwards (the map rebuilt), toy and counter.
* `merged_advance_finds_the_root_by_signature_or_by_observation`: the reused subtree as it was, by signature and by
  an unknown real key's observation (one encode), refused for another viewer; the search goes on, compaction keeps it.
* `a_repeated_observation_on_the_walks_path_gets_a_fresh_node`: forgetful counter games alias, the recount stays
  exact, compact keeps it.
* `set_merge_takes_a_fresh_tree_and_reset_turns_it_off`.
* Driver: `infoset_merge_needs_root_view_values_and_an_evaluation_seat`, `infoset_merge_off_changes_nothing` (the
  golden match and records with the key given off), `infoset_merge_plays_counter_matches_that_merge_and_reproduce`
  (merged vs unmerged seats: merges, growths, aliases on forgetful games; the same games across thread counts and with
  the eval cache; tree reuse through `advance_in`), `infoset_merge_with_the_root_floor_gumbel_and_the_toy`,
  `infoset_merge_survives_rolling_network_changes` (restarts from fresh merged trees).
* pytest: the seat dict and the run setting (validated, '-ism' in the Elo pool's name), the tree bindings, the Rust
  driver's and the Python driver's ('rs') merged seats, h2h's flags, tags and `merge` counters.
* Self-play (`driver::tests`, 2026-11-04): `infoset_merge_needs_root_view_values` (evaluation and self-play seats),
  `infoset_merge_selfplay_records_are_valid_and_reproduce` (the counter game, a merged self-play seat on both sides
  with its playout cap, root noise, temperature and value-only samples 0 / 1: pi a distribution over the options, q in
  range, z the result from the record's player's view, every game over; the same rows across 1 / 2 / 4 threads and
  with the evaluation cache; merges, growth leaves, aliases (forgetful games) and shared rows happen; other records
  than unmerged self-play), `infoset_merge_selfplay_with_gumbel_the_root_floor_and_the_belief` (also the hidden-hand
  mulligan's root belief, and minor decisions with their own budget), `infoset_merge_shares_the_leaf_encoding_byte_
  for_byte` (every batch the same bytes with the reuse switched off - dense and sparse, eval cache 0 / 256, counter and
  toy -, the same games and records; 91-92% of the leaves take the tree's encoding),
  `infoset_merge_selfplay_resigns_and_branches` (a resigned game is its dry twin up to the firing search, z the
  resignation's; branches of resigning games reproducible across threads), `infoset_merge_selfplay_rolling_quotas_and_
  network_changes` (counter and toy, resignation on, the network changing every other phase: every phase its quota,
  every game counted once, restarts from fresh merged trees). pytest `tests/test_infoset_merge.py`: the run setting
  (validated; `Run.nn_seat` gives it to the self-play seat only; off = the seat dict of before), the bindings' merged
  self-play records (valid, the same across threads), a rolling toy run (exact quotas with resignation and value-only
  samples, the `selfplay.merge` metrics; refused without root-view values; no metrics when off), remote self-play (the
  workers merge: no refusal, exact quotas) and an older worker refused.

**Measured** (2026-11-04, the desktop CPU at idle priority beside the live A/B - timings loaded; r3-2b-float550 gen
1000 with its config; E:/mtg-zero-scratch/ideas-4b/, its NOTES.md):

* *Step 0, before the build* (`Tree::shape`'s per-edge split: `EdgeShape::opp_lists` / `opp_keys` / `opp_tot` /
  `opp_modal_tot`, the other-seat children's distinct option lists, chance keys, backups, and the backups of the most
  backed-up list; `step0.py`, the run's full self-play search, fresh trees): at the main line's first edge into the
  other seat, searches with two or more option lists 71.7% / 70.0% (64 / 32 simulations, 30 positions x 2) and
  65.4% / 65.4% (the gate's 50 positions x 4); distinct option lists among its children 4.3-7.7 (= the children: one
  chance key in > 90% of the searches); backups outside the most backed-up list 22-31% (pooled).
* *Merged against unmerged on the same positions and seeds* (`measure.py`: the run's root belief, root-view values,
  redet, chance_k 0; 64 simulations with root noise = self-play's full search, 32 without = the evaluation seat's;
  4 fresh searches per position; "uniform" = 20 positions drawn from 6 prior-played games, "contested" = the 24 of
  400 drawn decisions whose prior max is < 0.7):

  | off / on | 64 uniform | 32 uniform | 64 contested | 32 contested |
  |---|---:|---:|---:|---:|
  | mean walk depth (root's own evaluation left out) | 5.14 / **5.95** | 4.63 / **5.30** | 4.15 / **4.87** | 3.39 / **3.91** |
  | walks ending at depth <= 2 | 22.9% / 17.6% | 27.0% / 21.8% | 32.7% / 25.4% | 42.4% / 35.2% |
  | information sets expanded per search (per walk) | 41.6 / **55.5** (0.65 / 0.87) | 21.9 / **27.5** (0.69 / 0.86) | 43.5 / **55.9** (0.68 / 0.87) | 24.0 / **28.8** (0.75 / 0.90) |
  | leaves at an information set already expanded | 35.1% / 12.3% | 32.0% / 12.9% | 32.1% / 12.3% | 25.7% / 10.3% |
  | merges / growth leaves / aliases per search | - / 25.9 / 7.6 / 0.03 | - / 12.2 / 3.9 / 0.01 | - / 20.0 / 7.6 / 0.04 | - / 7.7 / 3.2 / 0.01 |
  | main line's first edge into the other seat: its other-seat children | 8.25 / **1.45** | 6.24 / **1.44** | 4.63 / **1.47** | 3.01 / **1.36** |
  | the evaluation rule's root pick (cq, guard 4, c_scale 0.3) as the unmerged search's | 100% | 100% | 90.6% | 89.6% |
  | the same pick across a mode's 4 repeats (off / on) | 100% / 100% | 100% / 100% | 85.4% / 77.8% | 81.2% / 79.9% |

  The merged tree's remaining repeats are its growth leaves (610 of 612 at 64 uniform). The root's visit
  distribution moves little where the root is easy: total variation merged vs unmerged 0.008 (64, noise; 0.117
  repeat to repeat) and 0.001 (32) on the uniform positions, whose roots average 2.25 options with 85% / 97.5% of the
  visits on one. So the merge changes the evaluation seat's move at roughly one contested decision in ten - about 1
  decision in 100 overall -, which is what the h2h measures. Wall time per walk in this lockstep Python harness (the
  CPU network included, loaded machine): 0.81 -> 1.10 ms (64 uniform), 0.88 -> 1.16 (32 uniform), 0.81 -> 0.90 and
  0.93 -> 0.87 (contested): longer walks and one viewer encode per new signature; not a driver measurement.
* *The MTG h2h, end to end* (2 paired games at 4 simulations, CPU, the command of docs/az.md 30): the sides named
  `cq-n4-v50-s0.3-rv-b1m0.25-ism` / `cq-n4-v50-s0.3-rv-b1m0.25` (the run's evaluation pool), `merge` {links 1,789,
  merged 264, aliases 2, growths 98}. MTG Zero e098aff with this core: the engine digest f6543eb2b1b8752f unchanged.

**Self-play** (2026-11-04, zero-core `feat/infoset-merge-selfplay`; built after the evaluation-only match came out
+9.3 +- 5.3 Elo at 2,400 paired games on MTG Zero's Delver mirror, above the +7 of its stop rule). `Driver::new` no
longer refuses the key on a self-play seat (root-view leaf values are still required); run setting
`selfplay_infoset_merge` (docs/az.md 30.1). Every search of the seat merges (full, fast, minor). What it composes with:

* *The recorded policy target* ('visits', 'cq' with `cq_prior_w`), *the root value q*, *value-only samples*
  (`view_samples` and their `next_q`), *self-play resignation* (3.14: the root win vector of its full searches) and
  *the move* (temperature, the move rule, the Gumbel root's choice) read the root only - its options' visits and
  values, `win(root)`, `root_win_pruned`, `gumbel_q` - and the root is the viewer's node, never merged into, whose
  statistics are the same kind as before. Merging changes what flows into them (the values of the subtrees below the
  root), which is the point; their formulas are untouched.
* *Root noise* (Dirichlet on a full search's root expansion or reused root; Gumbel noise), *the root floor* (3.5),
  *the root's belief* (3.8: the root's features, one row per search; a growth leaf is never the root), *the evaluation
  cache* (3.12: a growth leaf's row is keyed with its world's options), *playout-cap fast searches* and *minor searches*
  (a budget; the same tree), *tree reuse* (`advance_in`), *rolling restarts on a new network* (`restart_search`: reset
  + `set_merge`), *branched games and archive starts* (2.11: a deviation search is the root's own evaluation;
  `start_dev` calls `set_merge` after its reset; a branch starts with fresh trees): each tested below.
* *Remote self-play* (`zero/remoteplay.py`): a worker's hello reports `merge` ('ism' when its checkout's
  `selfplay_infoset_merge` is on); a worker that does not - an older checkout would search its games unmerged in
  silence - is refused (REFUSED in the run's log, killed, retried in 10 minutes), as resignation's and the belief's
  settings are. Remote evaluation needs no new guard: its result names the move rule as the remote's checkout computed
  it (`elo.eval_tag`, '-ism'), and another name is refused.
* *Refused or untested:* the Python reference tree (`mcts.py`; `play.NNSeat(impl='py')` refuses the key) and a seat
  without root-view values are refused. Not tested with merging: team games (`toy_team`), deferred terminals
  (co-training; a merged tree keeps deferred nodes unmerged), the self-play league (its seats are the self-play seat's).

*The leaf's row of the viewer from the tree's encoding* (the saving Costs named). The merged tree encodes the reached
state from the viewer's view at each new signature (its observation key), and nearly every leaf is reached through
one (a new node, an alias, a growth leaf). `Tree::leaf_obs(node)` returns those features (`encode_sparse(viewer)`'s
pairs) when the walk's last new signature encoded its leaf (`Merge::enc_node`: set by `merged_child` after an encode;
cleared at the walk's start, at every encode and by `advance_in`; a compacted tree starts without one), None otherwise
(off, the root's own evaluation, a known link). The driver takes the leaf's row of the viewer from it - the leaf's own
row at a viewer's leaf, the root-view value-only row at another seat's - instead of encoding the state again
(`LeafBuf::push_pairs`: a sparse batch appends the pairs, a dense one scatters them into zeros), the same row bit for
bit by `encode_sparse`'s contract (2.12). Off, `leaf_obs` is None: the path of before. Counted as `merge_shared`.
On the way: the worker flushed its merge counters only for a chunk that made a link, so aliases made through known
links went uncounted (`merge_aliases` differed across thread counts; the games never did) - fixed.

*Measured* (2026-11-04; MTG Zero's Delver mirror: the run's self-play seat and driver built by `zero.run` from gen
1000's config with eval_cache 8192 - full 64 / fast 16 at p_full 0.25 / minor 4 simulations, root noise, the root
belief, root-view values, value-only samples 0.5, 'cq' targets -, the gen-1000 network on the CPU, 24 games in flight,
2 threads x 2 groups, sparse batches; one process alternating off / on, 8 configurations of 12 timed games; the desktop
at idle priority beside the live run, so the timings are loaded; E:/mtg-zero-scratch/ideas-4c/throughput.py):

| per simulation | off | on |
|---|---:|---:|
| rows sent (leaf + value-only + belief) | 0.988 | **1.200 (+21.5%)** |
| leaf rows / value-only rows sent | 0.790 / 0.188 | 0.874 / 0.317 |
| rows answered by the evaluation cache | 0.503 | 0.293 |
| rows needed (sent + cached) | 1.490 | 1.493 |
| encodes (rows + the tree's observations) | 1.490 | 1.768 (+19%; 2.726 = +83% without the reuse) |
| links / merges / growth leaves / shared rows | - | 1.234 / 0.400 / 0.125 / 0.959 |
| search-thread time (driver `busy_s` / sims), median (min) of 4 | 99.0 (84.9) us | **140.1 (113.0) us (+42%, +33%)** |
| simulations per wall second, the CPU network included, median of 4 | 1,008 | 723 (-28%) |

A merged simulation needs the same rows, but fewer of them are answered by the evaluation cache - the repeats it used
to answer (re-expansions of a held information set) are merged away and the walk goes on to a new information set -
so it sends 21.5% more network rows: on a GPU-bound self-play, ~20% more evaluator time per game. The reuse of the
tree's encoding takes 0.96 encodes per simulation off (the merged search's extra encodes +19% instead of +83%); the
search threads' remaining extra time (longer walks, link lookups, subset-armed selection, more sent rows' bookkeeping)
was +33-42% under this load (the threads 5% busy, preempted by the live run: a noisy number). Every game finished
and every record checked valid (pi a distribution over two or more options, q in range, z the result from the
record's player's view): 634 + 318 value-only records per 12 games off, 607 + 301 on, the same in each repeat.

Training on merged searches is the self-play A/B (docs/az.md 30.1).

Code: `tree.rs` (`Tree::merge`, the simulate loop's merged branch, `expand`'s growth, `kids` / `kid_ids` /
`compact` dispatch), `tree/merge.rs` (`Merge`, `Link`, `MergeStats`, `set_merge`, `leaf_acts`, `leaf_obs`,
`merged_child`, `select_avail`, `grow`, `advance_in`, `compact_merged`, `avail`, `in_links`), `counter_game.rs`,
`driver.rs` (`NnSeat::infoset_merge`, the checks, the stats, `LeafBuf::push_pairs`), `driver/worker.rs` (`leaf_obs`,
`encode_leaf_row`), `dynapi.rs` and the bindings; Python `games.nn_seat`, `mcts.check_merge` / `merge_tag`,
`play.NNSeat` / `RsSearch`, `run.py` (`eval_infoset_merge`, `selfplay_infoset_merge`, `merge_metrics`), `remoteplay.py`
(`_merge_of`, the hello guard), `elo.eval_tag`, `h2h.py`.

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

**Root minimum visits** (section 3.5; `python -m zero.parity forced`, `parity.forced`,
`tests/test_root_min_visits.py`), 2026-10-29: 10 toy positions of 2-6 options, both trees with the
floor 4 at 96 simulations, 64 repetitions per position and tree, the stand-in evaluator with a lopsided
prior (`SkewEval`: 1 - eps on its favourite option; without the floor 60% of the options get no visit
at eps 0.001, 31% at 0.05). Every option of every search reached 4 visits in both trees, the forced
walks per search are the same (11.2 on average), and the statistics sit at the Python-Python floor:

| setting | visits TV rs-py (py-py) | pruned TV rs-py (py-py) | cq TV rs-py (py-py) | root win \|rs - py\| without the forced walks |
|---|---|---|---|---|
| eps 0.001, c_scale 1 | 0.00002 (0.00005) | 0.00002 (0.00006) | 0.0110 (0.0123) | 0.0008 |
| eps 0.05, c_scale 1 | 0.00035 (0.00026) | 0.00039 (0.00030) | 0.0150 (0.0126) | 0.0004 |
| eps 0.05, c_scale 0.3 | | | 0.0031 (0.0025) | |

Power check: a Rust floor of 5 against Python's 4 is seen at once (the test). The drivers (self-play,
'visits' targets, the floor 2 on toy roots of up to 12 options, 24 simulations, the lopsided prior):
the targets' mass off the prior's favourite is 0.0304 / 0.0305 / 0.0307 with the floor and 0.0296 /
0.0296 / 0.0288 without it for the Python driver's 'py' seats, its 'rs' seats and the Rust driver -
the forced walks pruned alike (kept, they would add at least 2 / 23).

**Root-view leaf values** (section 3.6; `python -m zero.parity root_view`, `parity.root_view`,
`tests/test_root_view.py`), 2026-10-29: 10 toy positions, 96 simulations, 24 repetitions per position
and tree, both trees with root-view values and the stand-in `LinearEval` (its value reads the row's
features, so the viewer's features give other values than the decider's): visits TV Rust-Python
0.0049 (Python-Python floor 0.0037), cq target TV 0.0162 (0.0088), root win |rs - py| 0.0004; the
positions where nothing is left to chance search identically. The setting matters: the Python
tree's root-view vs mover-view searches of the same positions differ by TV 0.096, and a Rust tree
searching mover-view against the Python root view is seen at once (TV 0.096, power check). The
drivers ('py' and 'rs' seats of `PyDriver`, the Rust driver): the same batch layout (a value-only
row after its leaf's, flagged `ROW_VALUE`), the same value-only samples and targets
(`test_value_only_samples_and_their_targets`).

**The root's belief** (section 3.8; `python -m zero.parity belief`, `parity.belief`,
`tests/test_belief.py`), 2026-10-29: 8 toy positions whose next round's card is hidden among 2-3 cards,
a belief of 0.9 on one of them (0.02 on the others), 96 simulations, 24 repetitions per position and
tree, both trees dealing by it (the Rust game's `determinize_with` either way: the trees' timing, mix
draws and statistics are what is compared): at tilt 1 visits TV Rust-Python 0.0040 (Python-Python
floor 0.0049), cq target TV 0.0081 (0.0177), root win |rs - py| 0.0004, the believed card in 96.6% of
the Python leaves' worlds (the deal's odds 96.9%); at tilt 0.5 with `belief_mix` 0.25: 0.0075 (0.0078),
0.022 (0.034), 0.0008, 73.8% (74.9%). The setting matters: the belief vs the uniform deal of the same
positions, TV 0.052 (tilt 1) and 0.027 (tilt 0.5, mix 0.25). The drivers (the Rust driver, `PyDriver`'s
'py' and 'rs' seats): every batch valid, one belief row per full search, answered by a network with the
head (`test_one_belief_row_per_full_search_answered_by_the_head`).

**The Gumbel root search** (section 3.7; `python -m zero.parity gumbel`, `parity.gumbel`,
`parity.gumbel_formula`, `tests/test_gumbel.py`), 2026-10-29: the schedule `gumbel_sequence` is mctx's
(`get_sequence_of_considered_visits` run from its source: every m < 40, n < 260) and the Rust tree's
(the bindings' `gumbel_sequence`); the Python `gumbel_choose`, `gumbel_q` and next root walk on the
statistics of Rust Gumbel roots (8 toy positions x 2, 5, 17, 64 simulations x m 1, 2, 4, 16) are the Rust
tree's own on all 256 checks, the root values to 0.0. In searches (10 toy positions of 3+ options, 64
simulations, 32 repetitions per position and tree, m 16, noise scale 1, sigma (50, 0.3); repetition r of
every tree with the same noise, the trees' own streams otherwise):

| | Rust-Python | Python-Python floor |
|---|---|---|
| mean visit distribution TV (= the searches' own walks: fresh roots) | 0.0096 | 0.0122 |
| mean completed-Q target TV | 0.0098 | 0.0141 |
| move distribution TV | 0.041 | 0.041 |
| repetitions whose moves agree | 0.92 | 0.92 |
| \|root value (gumbel_q) Rust - Python\| | 0.0005 | |

Every search walked exactly sims - 1 root walks over min(m, options) options in both trees. Power: a
Rust m of 2 against Python's 16 is seen at once (TV 0.285 against the floor 0.015; the test). The drivers (self-play, the 'cq' target,
24 simulations, a lopsided prior where PUCT starves every option but its favourite): the targets' mass
off the favourite is 0.562 / 0.558 / 0.594 with the Gumbel root for the Python driver's 'py' seats, its
'rs' seats and the Rust driver (whose games are seeded otherwise), 0.064 / 0.064 / 0.073 with PUCT.

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
