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
    """{name, num_actions, num_features, min_players, max_players, bots, max_seats}."""
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
            sims_minor=None, select='visits', select_min_visits=0, select_c_visit=None, select_c_scale=None,
            root_min_visits=0, root_min_visits_max_actions=4, leaf_value_view='mover', view_samples=0.0,
            view_sample_target='next_q', belief_tilt=0.0, belief_clip=None, belief_mix=0.0, belief_fast=False,
            root_search='puct', gumbel_m=16, gumbel_scale=1.0, gumbel_c_visit=None, gumbel_c_scale=None,
            eval_cache=0, resign_t=0.0, resign_k=5, resign_noresign=0.1, cq_prior_w=1.0):
    """A network seat dict. policy_target: training target of its recorded decisions, 'visits'
    (the root visit distribution) | 'cq' (Gumbel AlphaZero's completed-Q target with cq_c_visit,
    cq_c_scale; docs/search.md 3.1). sims_minor: simulations of the game's minor decisions
    (`SearchGame::minor_decision`) in self-play, never recorded (None: searched like the
    others). select, select_min_visits: the rule of the moves the seat does not sample ('visits'
    = argmax (N, P0) | 'cq' = the completed-Q logits' argmax among the options with >=
    select_min_visits visits; docs/search.md 3.3), its transform select_c_visit / select_c_scale
    (None: cq_c_visit / cq_c_scale). The target, minor and select keys are only set when not the
    default, and the dict depends on the rule's effective transform only: a seat without a 'cq'
    target carries it as cq_c_visit / cq_c_scale (nothing else reads them there), one with a 'cq'
    target adds the keys select_c_visit / select_c_scale only where the rule's transform differs
    from the target's - so a transform given equal to the target's is the seat of before, key for
    key. root_min_visits: the root minimum visits of its full searches on roots of up to
    root_min_visits_max_actions options (docs/search.md 3.5; 0 = off: no key, the seat of before).
    leaf_value_view: whose view its search leaves' values come from, 'mover' (no key, the seat of
    before) | 'root'; view_samples: the share of its recorded decisions that also record the other
    recording seats' value-only samples, with their q target view_sample_target (docs/search.md 3.6;
    0 = off: no keys). belief_tilt: the root's belief (docs/search.md 3.8; 0 = off: no keys, the seat
    of before) with its clip (None: the default 20), the uniform share belief_mix and belief_fast
    (fast and minor searches take it too). root_search: 'puct' (no key, the seat of before) | 'gumbel'
    (docs/search.md 3.7: Gumbel noise of scale gumbel_scale and sequential halving over gumbel_m options
    at the root; its move replaces temperature and the move rule). Sigma's transform gumbel_c_visit /
    gumbel_c_scale, None = the seat's: the 'cq' target's (cq_c_visit, cq_c_scale) when it records one -
    the target and the search share sigma - else its move rule's (select_c_* or cq_c_*); the dict always
    holds the resolved transform, so the Rust seat never falls back to its own defaults. eval_cache: the
    game's evaluation cache, at most this many answers per game (docs/search.md 3.12: a row whose features
    and options a game already sent is answered from the cache, not sent; 0 = off: no key, the seat of
    before). resign_t: self-play resignation (docs/search.md 3.14; a self-play seat that records): a game ends
    when resign_k consecutive full searches found one seat's root win estimate >= 1 - resign_t, scored as that
    seat's win; a share resign_noresign of the games never resigns (0 = off: no keys, the seat of before).
    cq_prior_w: beta, the prior's exponent in the recorded 'cq' target (docs/search.md 3.13:
    softmax(beta ln P0 + sigma completedQ), in (0, 1]; the moves never read it; 1 = off: no key, the seat
    of before)."""
    from .mcts import (check_belief, check_cq_prior_w, check_leaf_view, check_root_min, check_root_search,
                       check_select, check_select_transform, select_transform)
    check_select(select, select_min_visits)
    check_cq_prior_w(cq_prior_w, policy_target)
    check_root_search(root_search, gumbel_m, gumbel_scale, gumbel_c_visit, gumbel_c_scale)
    if root_search == 'gumbel' and root_min_visits:
        raise ValueError("root_min_visits with root_search 'gumbel': exclusive (sequential halving already visits "
                         "every considered option; docs/search.md 3.7)")
    check_select_transform(select, select_c_visit, select_c_scale)
    check_root_min(root_min_visits, root_min_visits_max_actions)
    check_leaf_view(leaf_value_view, view_samples, view_sample_target)
    check_belief(belief_tilt, belief_clip, belief_mix)
    if int(eval_cache) != eval_cache or eval_cache < 0:
        raise ValueError(f'eval_cache {eval_cache!r}: answers per game, an integer >= 0 (0 = off)')
    from .resign import check_values
    check_values(resign_t, resign_k, resign_noresign, what='resign')
    if resign_t and not selfplay:
        raise ValueError(f'resign_t {resign_t!r}: a self-play seat only (evaluations play their games out)')
    d = dict(key=key, sims=sims, c_puct=c_puct, selfplay=selfplay, alpha=alpha, eps=eps, temp_moves=temp_moves,
             sims_fast=sims_fast, p_full=p_full, chance_k=chance_k, margin_w=margin_w, redet=redet)
    sel = select_transform(cq_c_visit, cq_c_scale, select_c_visit, select_c_scale)
    if policy_target != 'visits':
        d.update(policy_target=policy_target, cq_c_visit=float(cq_c_visit), cq_c_scale=float(cq_c_scale))
    elif select != 'visits':
        d.update(cq_c_visit=sel[0], cq_c_scale=sel[1])
    if select != 'visits':
        d.update(select=select, select_min_visits=int(select_min_visits))
        if policy_target != 'visits' and sel != (float(cq_c_visit), float(cq_c_scale)):
            d.update(select_c_visit=sel[0], select_c_scale=sel[1])
    if sims_minor:
        d['sims_minor'] = int(sims_minor)
    if root_min_visits:
        d.update(root_min_visits=int(root_min_visits), root_min_visits_max_actions=int(root_min_visits_max_actions))
    if leaf_value_view != 'mover':
        d['leaf_value_view'] = leaf_value_view
    if view_samples:
        d.update(view_samples=float(view_samples), view_sample_target=view_sample_target)
    if belief_tilt:
        from .mcts import BELIEF_CLIP
        d.update(belief_tilt=float(belief_tilt), belief_clip=float(BELIEF_CLIP if belief_clip is None else belief_clip),
                 belief_mix=float(belief_mix), belief_fast=bool(belief_fast))
    if root_search == 'gumbel':
        cv, cs = (float(cq_c_visit), float(cq_c_scale)) if policy_target == 'cq' else sel
        d.update(root_search='gumbel', gumbel_m=int(gumbel_m), gumbel_scale=float(gumbel_scale),
                 gumbel_c_visit=float(cv if gumbel_c_visit is None else gumbel_c_visit),
                 gumbel_c_scale=float(cs if gumbel_c_scale is None else gumbel_c_scale))
    if eval_cache:
        d['eval_cache'] = int(eval_cache)
    if resign_t:
        d.update(resign_t=float(resign_t), resign_k=int(resign_k), resign_noresign=float(resign_noresign))
    if cq_prior_w != 1.0:
        d['cq_prior_w'] = float(cq_prior_w)
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

    def submit(self, bid, priors, value, margin, aux=None):
        """`aux`: the root belief rows' auxiliary outputs (docs/search.md 3.8), when the batch has some."""
        self.d.submit(bid, priors, value, margin, aux)

    def submit_external(self, bid, payloads):
        self.d.submit_external(bid, payloads)

    def __getattr__(self, k):
        return getattr(self.d, k)


apply_package(package.active())
