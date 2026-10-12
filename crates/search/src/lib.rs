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
//! * `toy`: "Mini Cavern", the toy game the stack is proven on; `secret`: a tiny game with a
//!   hidden simultaneous hand choice (the opponent-model options' test bed, docs/search.md 10).
//! * `dynapi`: object-safe wrappers for the bindings (crate `zero_py`).
pub mod belief;
pub mod driver;
pub mod dynapi;
pub mod game;
pub mod noise;
pub mod rng;
pub mod secret;
pub mod toy;
pub mod tree;

pub use game::{Bot, SearchGame, MAX_SEATS};
pub use rng::Rng;
