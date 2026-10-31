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
/// * **Deferred results** (docs/search.md 1.1): a game whose terminal result comes from outside
///   (MTG Zero's team draft: the matches of the drafted decks) is over (`to_move()` None) with
///   `deferred()` true until `resolve(results)`; the search evaluates such a terminal with the
///   network, the driver parks it until the results come (docs/search.md 2.8). `result()` and
///   `margin()` are never called on it. **Teams**: `team_size()` seats per team, seat `s` in
///   team `s % (num_players() / team_size())`; `result()` = the team's outcome / team_size.
/// * **Auxiliary targets** (optional; docs/search.md 1.2, docs/az.md 15): `AUX` names the
///   segments (name, width) of a per-decision label vector of `aux_width()` floats that
///   `aux_target(viewer, out)` writes for the pending decision from the TRUE state - it may
///   read what `viewer` cannot see (MTG Zero's belief target: the opponent's actual hand), since
///   it is a training label, never an input: `encode` stays blind to it. A driver created with
///   `aux` on records it with every sample (`Record::aux`); off (the default) nothing changes.
///   The loss of each segment is the game package's (zero/package.py `aux`).
/// * **Belief-weighted deals** (optional; docs/search.md 3.8): `determinize_with(viewer, rng,
///   belief)` is `determinize` with the hidden cards drawn by weights from the network's
///   prediction of the auxiliary targets for `viewer` (`Belief`); default: `determinize`.
/// * **Light decisions** (optional; docs/search.md 3.10): `LIGHT_CLASSES` names classes of
///   decisions and `light_class()` the pending decision's; a self-play seat with `sims_light`
///   searches the classes it selects with that budget (recorded or not, `light_record`).
/// * **Heavy decisions** (optional; docs/search.md 3.11): a self-play seat with `sims_heavy` probes
///   its decisions for options whose short line of play reveals hidden information to it
///   (`heavy::probe`); `passive_action` (the option that lets the game go on) and `reveals` (which
///   of `step`'s keys tell a viewer something) shape the probe's lines. Defaults: no passive option,
///   every key reveals.
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
    /// The pending decision is minor (MTG: which land to tap for mana): a self-play network
    /// seat with `sims_minor` searches it with that many simulations and never records it
    /// (docs/search.md 3). Default: no decision is minor.
    fn minor_decision(&self) -> bool {
        false
    }
    /// The pending decision is searched with the seat's simulations (`sims`, `sims_fast` or
    /// `sims_minor`, as drawn) divided by this, at least one: a lighter class of decisions,
    /// recorded as usual (MTG Zero's `team_draft`: deckbuilding at half a pick's simulations,
    /// its DESIGN.md 2.3.9). Every network seat, self-play or evaluation (docs/search.md 3).
    /// Default 1: every decision gets the seat's budget.
    fn sims_divisor(&self) -> u32 {
        1
    }
    /// Names of the game's classes of light decisions (docs/search.md 3.10; at most 64): the
    /// decisions a self-play seat with `sims_light` may search with a small budget, recorded or not
    /// (`light_record`), the seat choosing which classes (`light_classes`). Empty (the default): the
    /// game has none, and a seat asking for them is refused.
    const LIGHT_CLASSES: &'static [&'static str] = &[];
    /// The pending decision's light class (an index into `LIGHT_CLASSES`), None if it is not a light
    /// decision (MTG Zero: a priority decision with an empty stack outside the decider's own main
    /// phases, a "timing window", by step and whose turn). Read only for a self-play seat with
    /// `sims_light`, at the start of its search of a decision; never by evaluation seats. Default:
    /// None.
    fn light_class(&self) -> Option<usize> {
        None
    }
    /// The pending decision's passive option among `legal` (docs/search.md 3.11): the one that lets
    /// the game go on without the decider doing anything (MTG: pass priority, keep, no attackers, no
    /// blocks, decline, no more targets), None if it has none (a forced choice). Read only by the
    /// heavy decisions' reveal probe (`heavy::probe`): other seats take it on a probe's lines, the
    /// decider's own passive play continues with it, and the root's passive option is the probe's
    /// baseline, never probed. Default: None (no decision has one: a probe's lines then branch at
    /// every decision of the decider and take another seat's first option).
    fn passive_action(&self, legal: &[u16]) -> Option<u16> {
        let _ = legal;
        None
    }
    /// Whether chance key `key`, pushed by the `step` that led to this state in a world determinized
    /// for `viewer`, tells `viewer` something it did not know (docs/search.md 3.11): a draw or a look
    /// of its own, a hidden card made public, a die. Read only by the heavy decisions' reveal probe.
    /// Default: every key (docs/search.md 1: a transition keys what the decider could not predict;
    /// MTG Zero overrides it: its keys also carry single-option bookkeeping and other players'
    /// masked private facts).
    fn reveals(&self, key: u64, viewer: usize) -> bool {
        let _ = (key, viewer);
        true
    }
    /// The game is over (`to_move()` None) but its result comes from outside: episodes of
    /// another game decide it (docs/search.md 1.1). Until `resolve`, the search treats this
    /// terminal as a leaf the network evaluates and the driver parks the game; nothing calls
    /// `result()` or `margin()` on it. Default: never.
    fn deferred(&self) -> bool {
        false
    }
    /// What must be played to resolve the deferred terminal (JSON; by convention
    /// `{"episodes": [<a spec of the resolver game>, ...], ...}`, read by the run's game
    /// package, docs/az.md 11). Called once, when the driver parks the game.
    fn deferred_payload(&self) -> Result<String, String> {
        Ok(String::new())
    }
    /// The outcome of a deferred terminal from its episodes' results (JSON, in the payload's
    /// order; the run's convention `[{"result": [...], "scores": [...]}, ...]`). Afterwards
    /// `deferred()` is false and `result()` / `margin()` / `score()` hold the outcome. A
    /// malformed or impossible result is an error, never a default.
    fn resolve(&mut self, results: &str) -> Result<(), String> {
        let _ = results;
        Err(format!("{}: no deferred result to resolve", Self::NAME))
    }
    /// Seats per team (it divides `num_players()`; seat `s` is in team `s % (num_players() /
    /// team_size())`): `result()[s]` = the team's outcome / team_size, so the vector still sums
    /// to 1, and the search's utility is `team_size x result[s]` (+ the margin term): Q stays in
    /// [0, 1], `c_puct` and `cq_c_scale` keep their meaning (docs/search.md 3). Default 1.
    fn team_size(&self) -> usize {
        1
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
    /// The auxiliary targets of the pending decision for `viewer` (its decider) into `out`
    /// (`aux_width()` floats, zeroed by the caller), read from the true state: a training label
    /// that may hold what `viewer` cannot see (never fed to the network). A segment left all zero
    /// is "no target here" for a multinomial loss (docs/az.md 15.2).
    fn aux_target(&self, viewer: usize, out: &mut [f32]) {
        let _ = (viewer, out);
    }
    /// `determinize` dealt by a belief (docs/search.md 3.8): resample everything hidden from
    /// `viewer` as `determinize` does - the same constraints, the same reseeding - but draw the
    /// hidden cards the belief is about with weights from `belief` (the network's prediction of
    /// this game's auxiliary targets for the root decider, `Belief::pred`; `belief_weight` is the
    /// shared weight formula). The search calls it instead of `determinize` only while its tree
    /// holds a belief (seat key `belief_tilt` > 0); `pred` comes from the root decider's own
    /// features, never from the true state. Default: `determinize`, the belief ignored (a game
    /// without a weighted deal searches as before; a seat asking for one there pays the root's
    /// extra row for nothing: the driver refuses `belief_tilt` for a game without `AUX`).
    fn determinize_with(&mut self, viewer: usize, rng: &mut Rng, belief: &Belief) {
        let _ = belief;
        self.determinize(viewer, rng)
    }
}

/// The root's belief of a search (docs/search.md 3.8): the network's auxiliary predictions for the
/// root decider (`pred`, `SearchGame::aux_width()` floats in `AUX` order - a multinomial segment's
/// distribution over its columns, `zero.net.aux_predict`), computed once per decision from the
/// root's features (the decider's own view), and the seat's tilt and clip. `SearchGame::
/// determinize_with` deals the hidden cards it is about from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Belief {
    pub pred: Vec<f32>,
    /// alpha > 0 (seat key `belief_tilt`): the weights are the belief's ratio to the uniform deal
    /// to this power (1: the belief's expected counts, to first order; 0 would be today's deal and
    /// is never set - a tree without a belief deals as before).
    pub tilt: f64,
    /// R >= 1 (seat key `belief_clip`): every weight in [1/R, R], so no card is ever ruled out or
    /// forced.
    pub clip: f64,
}

/// The weight of one category of hidden cards in a weighted deal (docs/search.md 3.8): `pred` =
/// the belief's expected count of it among the cards the deal draws (already less what the viewer
/// knows), `uniform` = the uniform deal's expected count there; -> clip((max(pred, 0) / max(uniform,
/// eps))^tilt, 1/clip, clip). A category the belief puts at 0 gets 1/clip, never 0.
pub fn belief_weight(pred: f64, uniform: f64, tilt: f64, clip: f64) -> f64 {
    let r = pred.max(0.0) / uniform.max(1e-9);
    let w = if r > 0.0 { r.powf(tilt) } else { 0.0 };
    let lo = 1.0 / clip;
    if w.is_nan() {
        return lo;
    }
    w.clamp(lo, clip)
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

/// The team of `seat` in a game of `n` seats with `team_size` seats per team (the
/// `SearchGame::team_size` convention: seat `s` is in team `s % (n / team_size)`).
pub fn team_of(seat: usize, n: usize, team_size: usize) -> usize {
    seat % (n / team_size.max(1)).max(1)
}

/// Bot names a game accepts.
pub fn bot_names<G: SearchGame>() -> Vec<&'static str> {
    let mut v = vec!["random"];
    v.extend_from_slice(G::bot_names());
    v
}
