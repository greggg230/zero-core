"""The game package using the core in this process (docs/new_game.md).

Every process runs the core for exactly one game package: Glass Road Zero (`grz`), Caverna
Zero (`cvz`), ... or the core itself (`zero`, the toy game only). The game package registers
itself when it is imported (its `<pkg>/az/__init__.py` calls `register(GamePackage(...))`),
before any core module does its work; the core reads it through `active()`:

* `zero.games.engine()` imports the package's compiled module (`grz._engine`: the core's
  search driver + the package's games, built by its `tools/build_py.sh`);
* `zero.run` adds the package's presets and default overrides to `PRESETS` / `DEFAULTS`;
* `zero.games` adds its external players (`ext:<name>` seats, e.g. the JS bot);
* the tools (supervise, kill, pause, fork), the dashboard, `zero.targets` and `zero.h2h` take
  their names, ports, paths and per-game defaults from it.

Nothing is installed globally: the core is pinned per game repo (`third_party/zero-core`, a git
submodule) and put on `sys.path` by the game package's `__init__` (docs/pinning.md).
"""
import importlib
import os
import sys

CORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GamePackage:
    """What the core needs to know about a game package.

    name            Python package of the game ('grz'); its `az` subpackage holds the
                    command shims (`python -m grz.az.run`, ...)
    title           'Glass Road Zero' (dashboard, help texts)
    root            the game repo's root (runs/, tools/; the cwd of supervised runs)
    engine          the compiled module ('grz._engine')
    run_module      the loop's module for `python -m` ('grz.az.run')
    game            the default game for tools ('glassroad')
    specs           default spec per game ({'glassroad': '{"intro": false}'})
    dashboard_port  default port of `python -m <pkg>.az.dashboard`
    hub_id          the hub's project id (dashboard docs)
    env_prefix      environment variables of the tools ('GRZ' -> GRZ_SUPERVISE_WAIT, ...)
    defaults        overrides of zero.run.DEFAULTS ({'ext_groups': 8})
    presets         run presets added to zero.run.PRESETS ({'glassroad-2p': {...}})
    externals       external players: name -> factory(seed) or 'module:attr' (imported lazily)
    decision_kinds  game -> fn(sample dict) -> {kind: mask} or 'module:attr' (zero.targets)
    replay_url      the game app's base URL for `zero.replay --upload` (docs/replays.md)
    replay_hint     the external asked what it would do at each recorded network decision
                    ('ext:jsbot'; None: no hints)
    replay_opp      the default opponent of `zero.replay` ('ext:jsbot'; None: 'self')
    grids           game -> its board planes for arch 'grid' networks (net.GridNet):
                    {planes, h, w, offset, cell_action0, hex: 'odd-r' | None}
    remotes         name -> a machine for remote evaluation (run setting eval_remote; zero.remote):
                    {host, repo, python, workdir, env, overrides}
    tokens          game -> its object layout for arch 'tokens' networks (net.TokenNet: {vocab,
                    segments, bags}), or 'module:attr' (resolved when first used)
    eval_specs      game -> fn(spec) -> the spec of evaluation matches (a run's `eval_spec`
                    setting overrides it), or 'module:attr'; e.g. MTG: self-play on random
                    decks, evaluation on a fixed deck set. Default: the self-play spec
    deferred        game -> its deferred results' glue for co-training runs (docs/az.md 11), a
                    dict or 'module:attr': {'episodes': fn(payload) -> [spec JSON of the
                    resolver game, ...] (default: the payload's "episodes" list), 'results':
                    fn(payload, rows) -> the results JSON for `resolve` (rows: the episodes'
                    result rows in order; default `[{"result": [...], "scores": [...]}, ...]`)}.
                    Nothing in the core knows what an episode is.
    evals           game -> {name: fn(run, it) -> dict or None} or 'module:attr': yardsticks of a
                    checkpoint outside its matches (MTG Zero: the draft network's agreement with
                    held-out human picks), run on every evaluated iteration after the matches
                    (`run`: the `zero.run.Run`, `it`: the checkpoint) and logged in metrics.jsonl
                    under `extras.<name>` (None: skipped); measuring only, never a training
                    signal. By convention the dict holds `value` (the dashboard's series),
                    optional `label`, `pct` (a share: shown as a percentage) and `refs`
                    ({name: value}: reference lines, e.g. baselines on the same positions)
    """

    def __init__(self, name, title, root, engine=None, run_module=None, game='toy', specs=None,
                 dashboard_port=None, hub_id=None, env_prefix=None, defaults=None, presets=None,
                 externals=None, decision_kinds=None, replay_url=None, replay_hint=None, replay_opp=None,
                 grids=None, remotes=None, tokens=None, eval_specs=None, deferred=None, evals=None):
        self.name = name
        self.title = title
        self.root = os.path.abspath(root)
        self.engine = engine or f'{name}._engine'
        self.run_module = run_module or f'{name}.az.run'
        self.game = game
        self.specs = dict(specs or {})
        self.dashboard_port = dashboard_port
        self.hub_id = hub_id
        self.env_prefix = env_prefix or name.upper()
        self.defaults = dict(defaults or {})
        self.presets = dict(presets or {})
        self.externals = dict(externals or {})
        self.decision_kinds = dict(decision_kinds or {})
        self.replay_url = replay_url
        self.replay_hint = replay_hint
        self.replay_opp = replay_opp
        self.grids = dict(grids or {})
        self.remotes = dict(remotes or {})
        self.tokens = dict(tokens or {})
        self.eval_specs = dict(eval_specs or {})
        self.deferred = dict(deferred or {})
        self.evals = dict(evals or {})

    def deferred_of(self, game):
        """`game`'s deferred-result glue: (episodes(payload) -> [spec JSON], results(payload, rows)
        -> results JSON) - the package's, else the defaults (`default_episodes`,
        `default_results`)."""
        d = self.deferred.get(game) or {}
        if isinstance(d, str):
            d = self.deferred[game] = resolve(d)
        return (resolve(d.get('episodes')) or default_episodes, resolve(d.get('results')) or default_results)

    def evals_of(self, game):
        """`game`'s extra evaluations {name: fn(run, it) -> dict or None} (none: {})."""
        e = self.evals.get(game) or {}
        if isinstance(e, str):
            e = self.evals[game] = resolve(e)
        return {k: resolve(f) for k, f in e.items()}

    def tokens_of(self, game):
        """The object layout of `game` for arch 'tokens' (None if the package has none)."""
        t = self.tokens.get(game)
        if isinstance(t, str):
            t = self.tokens[game] = resolve(t)
        return t

    def eval_spec_of(self, game, spec):
        """The spec of `game`'s evaluation matches when self-play uses `spec` (eval_specs)."""
        f = self.eval_specs.get(game)
        if f is None:
            return spec
        if isinstance(f, str):
            f = self.eval_specs[game] = resolve(f)
        return f(spec)

    def spec_of(self, game):
        """The default spec of `game` ('' if none)."""
        return self.specs.get(game, '')

    def env(self, key, default=None):
        """`<PREFIX>_<key>` from the environment (e.g. GRZ_SUPERVISE_WAIT)."""
        return os.environ.get(f'{self.env_prefix}_{key}', default)

    def __repr__(self):
        return f'GamePackage({self.name!r}, root={self.root!r}, engine={self.engine!r})'


# The toy's objects for arch 'tokens' (crates/search/src/toy.rs `encode`, 195 features): the 13
# action spaces (7 floats each from feature 31: present, occupied, by whom, goods; space s is
# action s) and the 4 players (16 floats each from 122, the viewer first), the revealed round
# cards as a bag (6 from 25); the rest (round, players, start seat, markers, sub-decision) are
# globals.
TOY_TOKENS = {'vocab': 8,
              'segments': [{'name': 'spaces', 'offset': 31, 'count': 13, 'width': 7, 'present': 0, 'actions': [[0, 1]]},
                           {'name': 'players', 'offset': 122, 'count': 4, 'width': 16, 'present': 0}],
              'bags': [{'name': 'cards', 'offset': 25, 'count': 1, 'width': 6}]}

# the core's own package: the toy game, zero._engine (crates/toy_py, tools/build_py.sh)
CORE = GamePackage('zero', 'Zero Core', CORE_ROOT, engine='zero._engine', run_module='zero.run', game='toy',
                   dashboard_port=None, hub_id=None, env_prefix='ZERO', tokens={'toy': TOY_TOKENS})

def default_episodes(payload):
    """The resolver specs of a deferred terminal: its payload's "episodes" (each a JSON string,
    or an object serialised here). No episodes is an error."""
    import json
    try:
        v = json.loads(payload)
        eps = v['episodes']
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f'deferred payload without an "episodes" list ({e!r}): {payload[:300]}') from None
    if not isinstance(eps, list) or not eps:
        raise ValueError(f'deferred payload: "episodes" must be a non-empty list: {payload[:300]}')
    return [e if isinstance(e, str) else json.dumps(e) for e in eps]


def default_results(payload, rows):
    """The `resolve` JSON from the episodes' result rows (in the payload's order)."""
    import json
    return json.dumps([{'result': [float(x) for x in r['result']], 'scores': [float(x) for x in r['scores']]} for r in rows])


_ACTIVE = None


def register(pkg):
    """Make `pkg` this process's game package (once; registering the same package name again
    replaces it, e.g. a reload in tests). Applies its run presets / defaults and external
    players to the core modules already imported (later imports pick them up themselves)."""
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.name != pkg.name:
        raise RuntimeError(f'zero-core: game package {_ACTIVE.name!r} is already active in this process; '
                           f'cannot also use {pkg.name!r}')
    _ACTIVE = pkg
    run = sys.modules.get('zero.run')
    if run is not None:
        run.apply_package(pkg)
    games = sys.modules.get('zero.games')
    if games is not None:
        games.apply_package(pkg)
    return pkg


def active():
    """The registered game package, or the core's own (`CORE`: the toy game)."""
    return _ACTIVE if _ACTIVE is not None else CORE


def resolve(obj):
    """'module:attr' -> the object (imported now); anything else as it is."""
    if isinstance(obj, str) and ':' in obj:
        mod, _, attr = obj.partition(':')
        return getattr(importlib.import_module(mod), attr)
    return obj

