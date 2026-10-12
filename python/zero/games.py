"""Game adapters: the registered game types of the active game package's compiled module
(`zero.package.active().engine`, e.g. `grz._engine`; the core's own `zero._engine` has the toy
game only), seat specs, external players and a thin wrapper that turns the Rust driver's tuples
into `protocol.LeafBatch` objects.

Seats (one per player of a game), as strings in run settings:
  * 'nn' / an `NNSeat` dict  - a network seat (search settings, evaluator key)
  * 'random', 'heuristic'    - a built-in Rust bot (`SearchGame::bot_names`)
  * 'ext:<name>'             - an external player answered in Python (EXTERNALS), e.g. the
                               toy's 'ext:toyheur'; a game package adds its own (Caverna's and
                               Glass Road's 'ext:jsbot', the web app's bot through Node)
"""
import importlib
import json

from . import package
from .protocol import LeafBatch, MAX_SEATS  # noqa: F401  (MAX_SEATS re-exported)

_ENGINE = {}


def engine():
    """The active game package's compiled module (imported once per package)."""
    name = package.active().engine
    mod = _ENGINE.get(name)
    if mod is None:
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            raise ImportError(f'{name} is not built: run `bash tools/build_py.sh` in {package.active().root} '
                              f'({e})') from e
        _ENGINE[name] = mod
    return mod


def info(game):
    """{name, num_actions, num_features, min_players, max_players, bots, classes, oracle, psample,
    belief, max_seats} (classes: the decision classes of per-class seat options; oracle / psample
    / belief: the game implements those hooks, docs/search.md 10)."""
    return dict(engine().game_info_of(game))


def spec_json(spec):
    if spec is None or spec == '':
        return '{}'
    return spec if isinstance(spec, str) else json.dumps(spec)


# External players: name -> factory(seed) -> callable(LeafBatch) -> list of payload strings.
def _toyheur(seed=0):
    from .toybots import heuristic_external
    return heuristic_external


def _toyrandom(seed=0):
    from .toybots import RandomExternal
    return RandomExternal(seed)


EXTERNALS = {'toyheur': _toyheur, 'toyrandom': _toyrandom}


def register_external(name, factory):
    """Add an external player: factory(seed) -> callable(LeafBatch) -> one payload string per
    request (the game's `apply_external` format). A 'module:attr' string is imported when the
    player is first used (game packages register their players this way, so importing the
    package stays light)."""
    EXTERNALS[name] = factory


def apply_package(pkg):
    """The game package's external players (zero.package.register calls this)."""
    for name, factory in pkg.externals.items():
        register_external(name, factory)


def external(name, seed=0):
    if name not in EXTERNALS:
        raise ValueError(f'unknown external player {name!r} (registered: {sorted(EXTERNALS)})')
    f = EXTERNALS[name]
    if isinstance(f, str):
        f = EXTERNALS[name] = package.resolve(f)
    return f(seed)


def seat_dict(seat):
    """A seat spec string ('heuristic', 'random', 'ext:toyheur') -> driver seat dict."""
    if isinstance(seat, dict):
        return seat
    if seat.startswith('ext:'):
        return {'external': seat[4:]}
    return {'bot': seat}


def is_external(seat):
    return isinstance(seat, str) and seat.startswith('ext:')


def nn_seat(key='m', sims=64, c_puct=1.0, selfplay=False, alpha=0.3, eps=0.25, temp_moves=20, sims_fast=None,
            p_full=1.0, chance_k=0, margin_w=0.0, redet=True, policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0,
            oracle_hands=False, oracle_p=1.0, class_sims=None, mask_features=None, psample=False, belief_mix=0.0,
            belief_temp=1.0, belief_particles=64):
    """A network seat dict. policy_target: training target of its recorded decisions, 'visits'
    (the root visit distribution) | 'cq' (Gumbel AlphaZero's completed-Q target with cq_c_visit,
    cq_c_scale; docs/search.md 3.1). The target keys are only set when not the default.

    Opponent models (docs/search.md 10.2-10.3; set only when not the default, refused by games
    without the hooks):
      psample:      the game's sequentialised simultaneous choices (Glass Road: `select`) of the
                    other seats are drawn from the network's prior inside the search, and their
                    earlier hidden commits are taken back in the root worlds.
      belief_mix:   share of the search's root worlds whose other seat's hidden choice is a
                    particle sampled from the network's own policy at the start of that choice
                    (restricted to the consistent ones; the rest uniform); 0 = off.
      belief_temp, belief_particles: temperature of the particles' draws (p^(1/temp) per atomic
                    choice) and their number K.
    Analysis-only options (docs/search.md 10.1):
      oracle_hands: the search determinizes with the game's TRUE hidden cards at decisions of
                    the given classes (True / 'all', a class name, 'select+play', a list);
                    the network input stays honest. Glass Road classes: 'play' (every decision
                    outside select), 'select' (the select picks).
      oracle_p:     the share of the search's root worlds at those decisions that are true
                    (each simulation independently; the rest honest).
      class_sims:   {class name: simulations} for full searches at decisions of that class.
      mask_features: feature indices zeroed in every network input of the seat (ablation)."""
    d = dict(key=key, sims=sims, c_puct=c_puct, selfplay=selfplay, alpha=alpha, eps=eps, temp_moves=temp_moves,
             sims_fast=sims_fast, p_full=p_full, chance_k=chance_k, margin_w=margin_w, redet=redet)
    if policy_target != 'visits':
        d.update(policy_target=policy_target, cq_c_visit=float(cq_c_visit), cq_c_scale=float(cq_c_scale))
    if oracle_hands:
        d['oracle_hands'] = oracle_hands
        if oracle_p != 1.0:
            d['oracle_p'] = float(oracle_p)
    if class_sims:
        d['class_sims'] = {str(k): int(v) for k, v in class_sims.items()}
    if mask_features:
        d['mask_features'] = [int(i) for i in mask_features]
    if psample:
        d['psample'] = True
    if belief_mix:
        d.update(belief_mix=float(belief_mix), belief_temp=float(belief_temp), belief_particles=int(belief_particles))
    return d


class Driver:
    """`<package>._engine.SearchDriver` with LeafBatch objects (the Python driver has the same API)."""

    def __init__(self, game, seats, spec=None, **kw):
        self.game = game
        self.d = engine().SearchDriver(game, seats, spec=spec_json(spec), **kw)

    def next_batch(self):
        r = self.d.next_batch()
        return None if r is None else LeafBatch(*r)

    def poll_batch(self):
        r = self.d.poll_batch()
        return None if r is None else LeafBatch(*r)

    def submit(self, bid, priors, value, margin):
        self.d.submit(bid, priors, value, margin)

    def submit_external(self, bid, payloads):
        self.d.submit_external(bid, payloads)

    def __getattr__(self, k):
        return getattr(self.d, k)


apply_package(package.active())
