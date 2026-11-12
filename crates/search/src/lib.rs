//! Zero Core: game-agnostic AlphaZero search (docs/search.md).
//!
//! * `game`: the `SearchGame` trait every game implements (the toy game here; each game repo's
//!   AZ layer) and built-in bots.
//! * `tree`: PUCT MCTS with per-seat values, chance children keyed by chance outcomes
//!   (`chance_k`), a fresh determinization of hidden information per simulation, root noise,
//!   subtree reuse - the semantics of the Python reference `zero/mcts.py`.
//! * `driver`: the batched multi-threaded self-play / evaluation driver (leaf batches out,
//!   priors / values in; one leaf per game per batch - no virtual loss, no in-tree batching),
//!   playout-cap randomisation, temperature, training records, rolling mode, arena (several
//!   network keys), external seats.
//! * `toy`: "Mini Cavern", the toy game the stack is proven on; `toy_team`: a two-team game
//!   with a deferred result decided by toy matches (deferred terminals, teams, co-training); `toy_match`: a best of
//!   three toy games, each game its own episode (chained episodes, docs/search.md 1.3).
//! * `lookahead`: the exhaustive lookahead per information set (docs/search.md 3.9; a seat's opt-in
//!   `lookahead`), its rows batched through the driver like MCTS leaves.
//! * `tail`: variable-length rows (docs/search.md 1.4, 2.15): a row's tail tokens past the fixed-width head and
//!   its pointer overrides (`SearchGame::TAILS`, `encode_row`, `option_overrides`).
//! * `dynapi`: object-safe wrappers for the bindings (crate `zero_py`).
pub mod driver;
pub mod dynapi;
pub mod game;
pub mod lookahead;
pub mod noise;
pub mod rng;
pub mod tail;
pub mod toy;
pub mod toy_tail;
pub mod toy_match;
pub mod toy_team;
pub mod tree;

// a test game for the root minimum visits (docs/search.md 3.5)
#[cfg(test)]
pub(crate) mod mull_game;
// a test game for the lookahead (docs/search.md 3.9): looking is worth something per information set
#[cfg(test)]
pub(crate) mod peek_game;
// a test game for information-set merging (docs/search.md 3.16): the other seat's options depend on its hidden hand
#[cfg(test)]
pub(crate) mod counter_game;

pub use game::{belief_weight, team_of, Belief, Bot, Next, SearchGame, MAX_SEATS};
pub use tail::{pack_tok, unpack_tok, TailBuf, TailRow, TailSeg};
pub use rng::Rng;
