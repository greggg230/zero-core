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
///   (tests, analysis). `determinize_oracle` (analysis only) keeps the game's hidden cards at
///   their true value; a seat's `oracle` mask selects the decision classes where its search
///   uses it (`Tree::oracle`).
/// * **Values**: `result()` per seat (only meaningful once over) sums to 1 with ties split;
///   `margin(seat)` is the auxiliary target (score - best other score, scaled so typical
///   values are O(1)); `score` is for reports only.
/// * **Features**: `encode(viewer, out)` writes the network input (`NUM_FEATURES` f32, `out`
///   is zeroed by the caller) from the viewer's perspective; it must not read hidden
///   information (the state may be a determinized world).
/// * **External seats** (a player outside the AZ action layer, e.g. the JS bot through the
///   Node bridge): `export_state()` describes the state for that player, `apply_external`
///   applies its answer (any number of moves); afterwards the driver re-reads everything.
/// * **Optional hooks, all defaulted** (docs/search.md 10; each behind a network-seat option
///   that is off by default, so a game without them behaves exactly as before):
///   decision classes (`DECISION_CLASSES`, `decision_class`: per-class seat options), the oracle
///   (`HAS_ORACLE`, `determinize_oracle`: analysis of what hidden information is worth),
///   policy-sampled simultaneous choices (`HAS_PSAMPLE`, `policy_sampled`,
///   `uncommit_simultaneous`: seat option `psample`) and the belief prior (`HAS_BELIEF`,
///   `belief_start`, `belief_particle`, `belief_target`, `belief_consistent`,
///   `determinize_particle`: seat options `belief_*`).
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

    // ------------------------------------------------------------ decision classes, oracle

    /// Names of the game's decision classes, for per-class options of network seats (oracle
    /// determinization, simulation counts; `NnSeat::oracle`, `NnSeat::class_sims`). Empty: one
    /// unnamed class 0.
    const DECISION_CLASSES: &'static [&'static str] = &[];
    /// The class of the pending decision (an index into `DECISION_CLASSES`; 0 without classes).
    fn decision_class(&self) -> usize {
        0
    }
    /// Does the game implement `determinize_oracle`?
    const HAS_ORACLE: bool = false;
    /// **Oracle determinization (analysis only: what is the hidden information worth?).** Like
    /// `determinize`, but the hidden information the game's oracle covers keeps its true value
    /// (Glass Road: the other seat's hidden cards; the building stacks are still reshuffled).
    /// The network input stays honest (`encode` never reads hidden information); only the
    /// search's worlds change. Never used for training data or real play. The driver calls it
    /// only for games with `HAS_ORACLE` (checked when a driver is built).
    fn determinize_oracle(&mut self, viewer: usize, rng: &mut Rng) {
        let _ = (viewer, rng);
        unreachable!("{}: no oracle determinization (HAS_ORACLE is false)", Self::NAME)
    }

    // ------------------------------------------------------------ policy-sampled simultaneous choices

    /// Does the game have sequentialised simultaneous choices (`policy_sampled`)? Seat option
    /// `psample` is refused for games without them (it would silently do nothing).
    const HAS_PSAMPLE: bool = false;
    /// **Policy-sampled simultaneous choices** (seat option `psample`, docs/search.md 10.2): is
    /// the pending decision part of a simultaneous choice that the game sequentialised (Glass
    /// Road: every `select` pick)? With `psample`, a search treats such decisions of seats other
    /// than its root decider as chance nodes: the option is drawn from the network's prior at
    /// the node (P0, never noised) instead of chosen by PUCT, so the opponent's simultaneous
    /// choice is modelled as its policy, never as a best response to our own hidden choice made
    /// earlier in the same tree (in-tree clairvoyance). Must depend only on the decision (its
    /// node), not on hidden information.
    fn policy_sampled(&self) -> bool {
        false
    }
    /// With `psample`, applied to every root world after `determinize(viewer)`: take back the
    /// hidden simultaneous choices that other seats committed before the viewer's pending one
    /// in the same step, and let them choose after it (the game's per-world order; Glass Road:
    /// the earlier chooser's `select` pick), so the viewer's search draws them from the policy
    /// after its own choice instead of from a uniform determinization. Must not change the
    /// viewer's decider / legal actions nor any seat's features (the choice was hidden). Default:
    /// nothing to take back.
    fn uncommit_simultaneous(&mut self, viewer: usize) {
        let _ = viewer;
    }

    // ------------------------------------------------------------ belief prior

    /// Does the game implement the belief hooks below? Seat option `belief_mix` > 0 is refused
    /// otherwise.
    const HAS_BELIEF: bool = false;
    /// **Belief prior** (seat options `belief_mix`, `belief_temp`, `belief_particles`;
    /// docs/search.md 10.3): Some(key) when the pending decision starts a hidden choice of the
    /// seat to move (one the other seats will not see: Glass Road, the first `select` pick of a
    /// period, key = the period). The driver keeps a copy of the real state here per (chooser,
    /// key): the root from which it later samples that choice with the network's own policy, as
    /// the chooser saw it (its features never show other seats' hidden information).
    fn belief_start(&self) -> Option<u64> {
        None
    }
    /// The completed hidden choice of `chooser` in a state reached from its `belief_start` (a
    /// particle, e.g. a card set as a bit mask), None while the chooser is still choosing (the
    /// driver keeps sampling its decisions from the policy).
    fn belief_particle(&self, chooser: usize) -> Option<u64> {
        let _ = chooser;
        None
    }
    /// The hidden choice of another seat that `determinize(viewer)` resamples at the pending
    /// decision and a belief can model: (chooser, key of its `belief_start`). None: determinize
    /// as usual (Glass Road: Some during a period's card play).
    fn belief_target(&self, viewer: usize) -> Option<(usize, u64)> {
        let _ = viewer;
        None
    }
    /// Is `particle` (the target chooser's hidden choice) consistent with everything `viewer`
    /// has observed (Glass Road: the cards the opponent played / followed are in it, none it
    /// failed to follow)?
    fn belief_consistent(&self, viewer: usize, particle: u64) -> bool {
        let _ = (viewer, particle);
        false
    }
    /// `determinize(viewer, rng)` with the target chooser's hidden choice set to `particle`
    /// (which `belief_consistent` accepted); everything else resampled as `determinize` does.
    fn determinize_particle(&mut self, viewer: usize, particle: u64, rng: &mut Rng) {
        let _ = (viewer, particle, rng);
        unreachable!("{}: no belief determinization (HAS_BELIEF is false)", Self::NAME)
    }
}

/// Number of decision classes of a game (at least 1: class 0 without names).
pub fn n_classes<G: SearchGame>() -> usize {
    G::DECISION_CLASSES.len().max(1)
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
