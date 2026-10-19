//! The `SearchGame` trait: everything the search, the self-play driver and the bindings need
//! from a game (docs/search.md 1). The toy game (`toy.rs`) and each game's AZ decision layer
//! (Caverna Zero, Glass Road Zero) implement it.
use crate::rng::Rng;

/// Value / margin vectors are fixed arrays of this many seats (Caverna: up to 7 players; Glass
/// Road 1-4): the protocol and the network's value head always carry 8.
pub const MAX_SEATS: usize = 8;

/// A game the search can play (docs/search.md 1 has the full contract).
///
/// * **Decisions**: `to_move()` is the seat that must pick one of `legal_actions()` (global
///   action ids `< NUM_ACTIONS`, a deterministic order, no duplicates, never empty while the
///   game is not over). A turn may consist of several consecutive decisions by the same seat.
/// * **Chance**: `step` pushes one key per chance outcome that occurred during the step (the
///   order they occurred). Keys must be canonical: two outcomes are the same iff their keys
///   are equal. `force_chance(keys)` queues outcomes the next chance events must take when
///   possible (MCTS `chance_k`); an impossible forced outcome falls back to the game's own,
///   and the actual outcome is what `step` reports. `force_chance(&[])` clears the queue.
///   **Revelations count as chance**: whatever a step reveals that the searching seat could
///   not predict (Glass Road: which seats followed a played card, a face-down card at its
///   reveal) must be keyed too, or two worlds revealing different facts would share a search
///   node when they lead to the same next decision. Such keys cannot be forced (fall back)
///   but are consumed from the queue like any other (docs/search.md 1, 9.2).
/// * **Hidden information**: `determinize(viewer, rng)` resamples, in place, everything the
///   viewer cannot see, consistently with what it has seen. It must never change revealed
///   facts nor the viewer's own legal actions at the current decision. The search calls it
///   with `viewer` = the root decider (`to_move()`) only; other viewers must still be valid
///   (tests, analysis).
/// * **Values**: `result()` per seat (only meaningful once over) sums to 1 with ties split;
///   `margin(seat)` is the auxiliary target (score - best other score, scaled so typical
///   values are O(1)); `score` is for reports only.
/// * **Features**: `encode(viewer, out)` writes the network input (`NUM_FEATURES` f32, `out`
///   is zeroed by the caller) from the viewer's perspective; it must not read hidden
///   information (the state may be a determinized world).
/// * **External seats** (a player outside the AZ action layer, e.g. the JS bot through the
///   Node bridge): `export_state()` describes the state for that player, `apply_external`
///   applies its answer (any number of moves); afterwards the driver re-reads everything.
/// * **Auxiliary targets** (optional, KataGo-style; docs/search.md 2.4): `AUX` names the
///   segments of a per-sample vector of `aux_width()` floats that `aux_targets(seat, out)`
///   writes for a FINISHED game from `seat`'s perspective (the driver records it with every
///   sample of that seat when the driver's `aux` is on). Segment `("own", cells)`: final
///   ownership per board cell in [-1, 1] (+1 the viewer's, -1 another seat's, 0 nobody's; a
///   grid network predicts it per cell, so cells = the grid's h x w, row-major as its planes);
///   any other name (e.g. `("score", k)`): regression values scaled to O(1). Default: none.
pub trait SearchGame: Clone + Send + Sync + 'static {
    /// Settings shared by all games of one driver, parsed from a JSON string.
    type Spec: Clone + Send + Sync + 'static;
    const NAME: &'static str;
    const NUM_ACTIONS: usize;
    const NUM_FEATURES: usize;
    const MIN_PLAYERS: usize;
    const MAX_PLAYERS: usize;

    fn parse_spec(json: &str) -> Result<Self::Spec, String>;
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String>;

    fn num_players(&self) -> usize;
    /// The seat to decide, None iff the game is over.
    fn to_move(&self) -> Option<usize>;
    fn legal_actions(&self, out: &mut Vec<u16>);
    /// Apply a legal action; push the keys of the chance outcomes that occurred.
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String>;
    /// Queue chance outcomes for the next chance events (see the trait docs).
    fn force_chance(&mut self, forced: &[u64]) {
        let _ = forced;
    }
    fn result(&self) -> [f32; MAX_SEATS];
    fn margin(&self, seat: usize) -> f32;
    fn score(&self, seat: usize) -> f32;
    fn determinize(&mut self, viewer: usize, rng: &mut Rng);
    fn encode(&self, viewer: usize, out: &mut [f32]);
    /// Rounds played so far (reports).
    fn rounds(&self) -> u32 {
        0
    }
    /// State description for an external seat (JSON).
    fn export_state(&self) -> String {
        String::new()
    }
    /// Apply an external seat's answer to `export_state()`.
    fn apply_external(&mut self, payload: &str, chance: &mut Vec<u64>) -> Result<(), String> {
        let _ = (payload, chance);
        Err(format!("{}: external seats are not supported", Self::NAME))
    }
    /// Game-specific built-in bots (`random` is always available).
    fn bot_names() -> &'static [&'static str] {
        &[]
    }
    fn make_bot(spec: &Self::Spec, name: &str, seed: u64) -> Option<Box<dyn Bot<Self>>> {
        let _ = (spec, name, seed);
        None
    }
    /// Segments (name, width) of the auxiliary targets, in vector order (see the trait docs);
    /// empty: the game has none.
    const AUX: &'static [(&'static str, usize)] = &[];
    /// Width of the auxiliary target vector (the segments' sum).
    fn aux_width() -> usize {
        Self::AUX.iter().map(|s| s.1).sum()
    }
    /// The auxiliary targets of the finished game from `seat`'s perspective into `out`
    /// (`aux_width()` floats, zeroed by the caller).
    fn aux_targets(&self, seat: usize, out: &mut [f32]) {
        let _ = (seat, out);
    }
}

/// A built-in (Rust) player. `legal` = the decision's legal actions (>= 2 of them).
pub trait Bot<G>: Send {
    fn choose(&mut self, g: &G, legal: &[u16]) -> u16;
}

/// Uniform over the legal actions.
pub struct RandomBot {
    pub rng: Rng,
}

impl<G> Bot<G> for RandomBot {
    fn choose(&mut self, _g: &G, legal: &[u16]) -> u16 {
        legal[self.rng.below(legal.len() as u32) as usize]
    }
}

/// `make_bot` with the generic `random` bot added.
pub fn make_bot<G: SearchGame>(spec: &G::Spec, name: &str, seed: u64) -> Option<Box<dyn Bot<G>>> {
    if name == "random" {
        return Some(Box::new(RandomBot { rng: Rng::new(seed) }));
    }
    G::make_bot(spec, name, seed)
}

/// Bot names a game accepts.
pub fn bot_names<G: SearchGame>() -> Vec<&'static str> {
    let mut v = vec!["random"];
    v.extend_from_slice(G::bot_names());
    v
}
