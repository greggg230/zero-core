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
                    under `extras.<name>` (None: skipped; an exception is logged and recorded
                    as {"error": ...}, the run goes on); measuring only, never a training
                    signal. By convention the dict holds `value` (the dashboard's series),
                    optional `label`, `pct` (a share: shown as a percentage) and `refs`
                    ({name: value}: reference lines, e.g. baselines on the same positions).
                    A remote evaluation (zero.remote, docs/az.md 5.2) runs them on the remote and
                    brings `extras` back into the run's evals.jsonl (docs/az.md 40)
    settings        {name: default}: run settings of the package's own (docs/az.md 40), e.g. an extra
                    evaluation's options or its hooks' (`selfplay_specs`) - zero.run adds them to
                    DEFAULTS (a flag each `--name`, saved in the run's config.json, presets and the
                    package's defaults may set them, shipped to a remote evaluation's mirror with the
                    rest); a name of the core's, an auxiliary target's or a reader's is refused. The
                    core never reads them: the package's hooks and evaluations do. Default none
    evals_every     {eval name: run setting}: an extra evaluation (`evals`) with a cadence of its own
                    (docs/az.md 40): it runs only on evaluated iterations that are multiples of the
                    setting's value, and never - silently: no log line, no `extras` key - while the
                    value is 0. The setting must be one of `settings` with an integer default.
                    Without an entry an extra evaluation runs on every evaluated iteration, as before
    aux             game -> its auxiliary training targets (docs/az.md 15): [{name, width, loss}, ...]
                    in the order of the game's `SearchGame::AUX` segments (the engine computes
                    them: a run checks the names and widths against it), loss 'multinomial' |
                    'poisson' | 'mse' (net.AUX_LOSSES). Each target gets the run setting
                    `aux_<name>_w` (default 0 = off: nothing recorded, no head, the run of before)
    replay_aux      game -> fn(predictions, state, seat, frame) -> {field: JSON} or 'module:attr':
                    what a replay frame of a network decision shows of a checkpoint's auxiliary
                    head (zero.replay; predictions {name: numpy [w]} = `net.aux_predict` at the
                    root, state = the frame's parsed export_state, seat = the decider). Default:
                    {'aux': {name: [[column, value], ...]}}, each target's 12 largest values
    readers         game -> its readers (docs/az.md 36): [{target, inputs, hidden, enter, belief}, ...],
                    each a small MLP of the network from the features `inputs` ([[offset, count],
                    ...]; a token network: bags and globals only) to the auxiliary target `target`
                    (one of `aux`), trained by its loss, its prediction also an input of the trunk
                    (enter 'trunk', the default; 'none': an auxiliary output only); hidden (default
                    64) its hidden width; belief (default false) true: a search's root belief rows
                    get the reader's prediction for the target instead of the head's. Each reader
                    gets the run setting `reader_<target>` (default 0 = off: the network of before)
    remote_eval_files
                    game -> [path, ...]: files a remote evaluation (zero.remote, docs/az.md 5.2) of the
                    game writes in its mirror that belong to the run - the package's remote side
                    writes them, `RemoteEval` copies each to the same path in the run after the
                    evaluation and before its result is recorded; a missing one fails the evaluation
                    (logged, retried). Paths are relative and '/'-separated, formatted with
                    `iter` = the evaluated iteration ('games/eval-iter_{iter:04d}.jsonl': MTG Zero's
                    game log of the evaluation's games). Default none: the copies of before
    selfplay_specs  game -> fn(run, it) -> spec JSON or None, or 'module:attr' (docs/az.md 40): the
                    spec of iteration `it`'s self-play (`run`: the `zero.run.Run`), asked at the start
                    of each local self-play phase; None = the run's spec (as before). A rolling driver
                    switches to a new spec between phases (`Driver.set_spec`: the games it starts
                    from then on; games in flight keep theirs), a phase's driver is made with it.
                    Refused with remote self-play (its workers play the run's spec). Default none:
                    the run's spec, every phase
    remote_eval_inputs
                    game -> fn(run, it) -> [path, ...] or 'module:attr' (docs/az.md 40): files of the run
                    a remote evaluation of iteration `it` needs (run-relative, '/'-separated), put at the
                    same paths in the mirror before it (`RemoteEval._ship`; each again when its size or
                    modification time changed). A missing file fails the evaluation (logged, retried).
                    Default none: the shipping of before
    """

    def __init__(self, name, title, root, engine=None, run_module=None, game='toy', specs=None,
                 dashboard_port=None, hub_id=None, env_prefix=None, defaults=None, presets=None,
                 externals=None, decision_kinds=None, replay_url=None, replay_hint=None, replay_opp=None,
                 grids=None, remotes=None, tokens=None, eval_specs=None, deferred=None, evals=None, aux=None,
                 replay_aux=None, readers=None, remote_eval_files=None, settings=None, selfplay_specs=None,
                 remote_eval_inputs=None, evals_every=None):
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
        self.aux = dict(aux or {})
        self.replay_aux = dict(replay_aux or {})
        self.readers = dict(readers or {})
        self.remote_eval_files = {g: list(ps) for g, ps in (remote_eval_files or {}).items()}
        for g, ps in self.remote_eval_files.items():
            for p in ps:
                if not isinstance(p, str) or not p or p.startswith('/') or '\\' in p or '..' in p.split('/'):
                    raise ValueError(f'{name}: remote_eval_files of {g!r}: {p!r} is not a relative /-separated path '
                                     f'inside the mirror')
        self.settings = dict(settings or {})
        self.selfplay_specs = dict(selfplay_specs or {})
        self.remote_eval_inputs = dict(remote_eval_inputs or {})
        self.evals_every = dict(evals_every or {})
        for ev, s in self.evals_every.items():
            d = self.settings.get(s)
            if s not in self.settings or not isinstance(d, int) or isinstance(d, bool) or d < 0:
                raise ValueError(f'{name}: evals_every of {ev!r}: {s!r} is not one of the package\'s settings with an '
                                 f'integer default >= 0')

    def eval_due(self, name, cfg, it):
        """Whether the extra evaluation `name` runs on the evaluated iteration `it` of a run with settings
        `cfg` (`evals_every`: a multiple of its setting's value, never at 0; without an entry: always)."""
        s = self.evals_every.get(name)
        if s is None:
            return True
        n = int(cfg.get(s, self.settings[s]) or 0)
        return n > 0 and int(it) % n == 0

    def selfplay_spec_fn(self, game):
        """`game`'s self-play spec hook fn(run, it) -> spec JSON or None (`selfplay_specs`), or None."""
        f = self.selfplay_specs.get(game)
        if isinstance(f, str):
            f = self.selfplay_specs[game] = resolve(f)
        return f

    def remote_eval_inputs_of(self, game, run, it):
        """The run's files a remote evaluation of `game`'s iteration `it` needs (`remote_eval_inputs`;
        `run`: the `zero.run.Run`): run-relative '/'-separated paths, checked; none: []."""
        f = self.remote_eval_inputs.get(game)
        if f is None:
            return []
        if isinstance(f, str):
            f = self.remote_eval_inputs[game] = resolve(f)
        out = list(f(run, it) or [])
        for p in out:
            if not isinstance(p, str) or not p or p.startswith('/') or '\\' in p or ':' in p or '..' in p.split('/'):
                raise ValueError(f'{self.name}: remote_eval_inputs of {game!r}: {p!r} is not a relative /-separated path '
                                 f'inside the run')
        return out

    def remote_eval_files_of(self, game, it):
        """The files a remote evaluation of `game`'s iteration `it` leaves in its mirror for the run
        (`remote_eval_files`, formatted with `iter`); none: []."""
        return [p.format(iter=int(it)) for p in self.remote_eval_files.get(game, [])]

    def aux_of(self, game):
        """`game`'s auxiliary targets [[name, width, loss], ...] (none: [])."""
        from .net import aux_layout
        return aux_layout(self.aux.get(game))

    def aux_settings(self):
        """The run settings of the package's auxiliary targets: {'aux_<name>_w': 0.0} for every
        target of every game (zero.run adds them to DEFAULTS)."""
        out = {}
        for targets in self.aux.values():
            for t in targets:
                name = t['name'] if isinstance(t, dict) else t[0]
                out[f'aux_{name}_w'] = 0.0
        return out

    def readers_of(self, game):
        """`game`'s readers in their normal form (`net.reader_layout`: each completed with its target's
        width and loss from `aux`; docs/az.md 36); none: []. A reader of a target the game does not
        declare: ValueError."""
        from .net import reader_layout
        targets = {n: (w, loss) for n, w, loss in self.aux_of(game)}
        out = []
        for r in self.readers.get(game) or []:
            if r['target'] not in targets:
                raise ValueError(f'{self.name}: the reader of {game!r} reads for {r["target"]!r}, not an auxiliary '
                                 f'target of the game ({sorted(targets)})')
            w, loss = targets[r['target']]
            out.append(dict(r, width=w, loss=loss))
        return reader_layout(out)

    def reader_settings(self):
        """The run settings of the package's readers: {'reader_<target>': 0} for every reader of every
        game (zero.run adds them to DEFAULTS)."""
        return {f'reader_{r["target"]}': 0 for rs in self.readers.values() for r in rs}

    def replay_aux_of(self, game):
        """`game`'s replay rendering of auxiliary predictions (`replay_aux`; None: zero.replay's default)."""
        f = self.replay_aux.get(game)
        if isinstance(f, str):
            f = self.replay_aux[game] = resolve(f)
        return f

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

# The toy's auxiliary target (crates/search/src/toy.rs `aux_target`, docs/az.md 15): the next round's
# card, hidden until its round - a one-hot over the 6 cards (none in the last round), learned as a
# distribution (the best from public information: uniform over the next stage's unrevealed cards).
TOY_AUX = [{'name': 'next_card', 'width': 6, 'loss': 'multinomial'}]

# The toy's reader (docs/az.md 36): next_card from the public information that determines its best
# prediction - the round (its one-hot, features 0-5) and the revealed round cards (25-30, the token
# layout's bag 'cards') - fed into the trunk.
TOY_READERS = [{'target': 'next_card', 'inputs': [[0, 6], [25, 6]], 'hidden': 32}]

# The variable-length rows' toy (crates/search/src/toy_tail.rs, docs/search.md 5.3; the varlen API's section 1.4,
# copied literally by the network's tests): 72 features - 8 globals, then segment `items` (4 head slots x 16 from
# feature 8: present, value / 9, marked, by the viewer, on the page, position / 8, the value one-hot over floats
# 6-15; TAKE j = action 4 + j on the first page), its items past the 4th as tail tokens; segment `marks` tail-only
# (width 4: present, position / 8, by the viewer, 0). Pointer overrides; a row has at most 64 tokens.
TOY_TAIL_TOKENS = {"vocab": 16, "name": "toy_tail", "version": 1, "option_pointers": True, "max_row_tokens": 64,
                   "segments": [
                       {"name": "items", "offset": 8, "count": 4, "width": 16, "present": 0, "id": {"at": 6, "radix": [10]},
                        "actions": [[4, 1]], "tail": True},
                       {"name": "marks", "offset": None, "count": 0, "width": 4, "present": 0, "id": None, "actions": [],
                        "tail": True}],
                   "bags": [],
                   "globals": [{"name": "globals", "offset": 0, "count": 8}]}
# the same without `marks` (the widen test adds it, gated)
TOY_TAIL_TOKENS_BASE = dict(TOY_TAIL_TOKENS, segments=TOY_TAIL_TOKENS['segments'][:1])

# the core's own package: the toy games, zero._engine (crates/toy_py, tools/build_py.sh)
CORE = GamePackage('zero', 'Zero Core', CORE_ROOT, engine='zero._engine', run_module='zero.run', game='toy',
                    dashboard_port=None, hub_id=None, env_prefix='ZERO',
                    tokens={'toy': TOY_TOKENS, 'toy_tail': TOY_TAIL_TOKENS},
                    aux={'toy': TOY_AUX, 'toy_match': TOY_AUX}, readers={'toy': TOY_READERS})
# (toy_match, the core's chained-episodes game, docs/az.md 42: the toy's features first, so the toy's target too)


class RowsMismatch(ValueError):
    """A game whose variable-length rows (`game_info()`: tails, option_pointers, max_row_tokens, head_tokens) differ
    from its package's token layout (R15, docs/az.md 43): a run refuses it as an encoding mismatch (exit 78)."""


def rows_problems(layout, info):
    """How a game's `games.info` disagrees with its token layout about variable-length rows: [] = they agree (or
    neither has them). Plain Python: the layout's tail segments (index, name, width, head slots), pointer overrides,
    row limit and head tokens (1 + every segment's head slots) against the game's."""
    segs = (layout or {}).get('segments', [])
    lay_tails = [[i, str(s['name']), int(s['width']), int(s.get('count') or 0)] for i, s in enumerate(segs) if s.get('tail')]
    game_tails = [[int(t['seg']), str(t['name']), int(t['width']), int(t['head'])] for t in info.get('tails') or []]
    out = []
    if lay_tails != game_tails:
        out.append(f'tail segments: the layout {lay_tails}, the game {game_tails}')
    lp, gp = bool((layout or {}).get('option_pointers')), bool(info.get('option_pointers'))
    if lp != gp:
        out.append(f'option_pointers: the layout {lp}, the game {gp}')
    if lay_tails or game_tails:
        lm, gm = (layout or {}).get('max_row_tokens'), info.get('max_row_tokens')
        if lm != gm:
            out.append(f'max_row_tokens: the layout {lm}, the game {gm}')
        lh = 1 + sum(int(s.get('count') or 0) for s in segs)
        gh = int(info.get('head_tokens') or 0)
        if gh != lh:
            out.append(f'head_tokens: the layout\'s 1 + Σ count = {lh}, the game\'s HEAD_TOKENS = {gh}'
                       + (' (a build without HEAD_TOKENS)' if not gh else ''))
    return out


def check_rows(game, info, pkg=None):
    """Refuse (RowsMismatch, R15) a game whose variable-length rows differ from the active package's token layout of
    it; nothing for a game without a layout or rows on either side."""
    pkg = pkg or active()
    lay = pkg.tokens_of(game)
    if lay is None:
        return                                      # (no token layout: an arch without one refuses the rows itself)
    probs = rows_problems(lay, info)
    if probs:
        raise RowsMismatch(f'game {game!r} of package {pkg.name!r}: its build and its token layout disagree about '
                           f'variable-length rows: ' + '; '.join(probs) + ' (rebuild the engine, or fix the layout)')

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
    eng = sys.modules.get(pkg.engine) if pkg.engine else None
    if eng is not None and hasattr(eng, 'game_info_of'):
        # R15 (docs/az.md 43): the engine is loaded already, so its games' variable-length rows are checked against
        # the layouts now (a light registration never imports it: zero.games and zero.run check them at use)
        names = set(eng.game_names())
        for g in list(pkg.tokens):
            if g in names:
                check_rows(g, dict(eng.game_info_of(g)), pkg)
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

