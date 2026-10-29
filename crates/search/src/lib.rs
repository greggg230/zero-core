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
//!   with a deferred result decided by toy matches (deferred terminals, teams, co-training).
//! * `dynapi`: object-safe wrappers for the bindings (crate `zero_py`).
pub mod driver;
pub mod dynapi;
pub mod game;
pub mod noise;
pub mod rng;
pub mod toy;
pub mod toy_team;
pub mod tree;

// a test game for the root minimum visits (docs/search.md 3.5)
#[cfg(test)]
pub(crate) mod mull_game;

pub use game::{team_of, Bot, SearchGame, MAX_SEATS};
pub use rng::Rng;
