"""Co-training (docs/az.md 11): a **primary** game whose episodes end at deferred terminals (MTG
Zero: `team_draft`, an event drafted and built by teams) and a **resolver** game whose episodes
decide them (MTG Zero: `mtg`, the event's matches), one network each, in one run. Nothing here
names a game: the run's game package says how a deferred payload becomes resolver specs and how
their results go back (`GamePackage.deferred`, defaults in `zero.package`).

The run directory holds the primary as a single-game run does (`models/`, `data/`, `elo/`,
`metrics.jsonl`, `progress.json`: the dashboards and tools work unchanged) and the resolver in
`<run>/resolver/` with the same layout (iteration numbers shared; its resolved settings in
`resolver/config.json`). Iteration `i`:

* **Self-play**: two rolling drivers in this process - the primary's (net i-1, recording) on the
  main thread and the resolver's (net i-1, recording, `queue_only`) on a worker thread - joined
  by a `Coordinator`: each parked primary game's episodes go into the resolver's queue (tag
  `gid/k`), their result rows come back by tag, and once all are in the primary game is resolved
  (its samples emitted with the real outcome). The phase ends when `games` primary games have
  been resolved in it (exactly: further ready ones wait for the next phase); games still playing
  or waiting for episodes carry over in the drivers' memory. Backpressure: the primary waits while
  `pending_max` resolver episodes are queued or playing. Samples: `data/iter_i/` and
  `resolver/data/iter_i/`.
* **Training**: the primary net on its window, then the resolver's on its own; both checked by
  the non-finite guard (zero/finite.py: shards on write and load, every step, both checkpoints
  and optimizer states before either is written; `NonFinite` names the net; quarantine in
  `<run>/debug/` and `<run>/resolver/debug/`).
* **Evaluation**: the resolver as a single-game run (its yardstick, random, Elo between its
  checkpoints: `resolver/metrics.jsonl`, `resolver/elo/`); the primary by games against its
  yardstick, random and its earlier checkpoints, team seats alternating between the two players
  (`team_size`: seat s in team s % (n / team_size)), whose episodes a second resolver driver plays
  with the resolver's yardstick (every evaluated iteration: `eval_yard`, `eval_random`, the Elo
  matches) and with the resolver's network every `draft_eval_net_every` (`eval_yard_net`).
  `resolver_cfg['eval_remote']` (a remote of the game package, as a single-game run's
  `eval_remote`): the resolver's own evaluations run there (zero.remote, mirror `<run
  name>.resolver`; results in `resolver/evals.jsonl`, its Elo copied back to `resolver/elo/`)
  while the run trains; the primary's stay here, except, with `draft_eval_net_remote` (docs/az.md
  11.7), its resolver-network events (`eval_yard_net`): played on that remote from a mirror holding
  both networks' checkpoints (zero.remote's `evalnet`, `CoRun.eval_net` there), their results in the
  run's `evals.jsonl`.
* **Resume**: an iteration interrupted before both checkpoints is replayed from its start (both
  data directories cleared); one interrupted after them finishes its evaluations; the resolver's
  remote evaluations without a result are queued again. PAUSE / STOP / the supervisor / kill /
  fork work on the one process.
* `resolver_cfg['selfplay_remote']` (docs/az.md 11.6, zero/remoteplay.py, zero/workqueue.py):
  another machine plays a share of the queued episodes with the resolver's checkpoint i-1 - a second
  consumer of the coordinator's queue next to the resolver's driver here, each with its in-flight
  cap (its throughput's share of the iteration's matches); the coordinator keeps the local driver's
  room under its cap (`local_cap`) in reserve, so the remote takes only what is queued beyond it;
  their rows come back by tag, their samples as shards into `resolver/data/iter_i/`; episodes the
  remote loses (its session broke), gives back (it backed off) or does not finish in time are
  requeued here, and a duplicate row is dropped. The primary's own self-play (the drafting) stays
  here: the run's top-level `selfplay_remote` is refused.
"""
import glob
import json
import os
import threading
import time
from collections import deque

import numpy as np

from . import games as G
from . import package
from .data import ShardWriter, TargetStats
from .finite import NonFinite, check_state
from .infer import drive
from .net import (EncodingMismatch, check_checkpoint, check_encoding, layout_sans_identity, load, save, tokens_layout,
                  widen)
from .run import (DEFAULTS, PRESETS, Run, _fmt, _fmt_queue, _fmt_remote, _write_json, check_devices, check_eval_settings,
                  drain_remote_evals, last_rates, latest_iter, log, mark_phase_done, minor_metrics, model_path,
                  read_metrics, summarize)
from .workqueue import lag_hist

BIG = 1 << 62                                      # a resolver driver's quota: its phases end by end_phase()
CO_KEYS = ('resolver', 'resolver_cfg', 'resolver_init', 'resolver_conc', 'pending_max', 'draft_eval_net_every',
           'draft_eval_net_remote')


def resolver_config(cfg):
    """The resolver's settings: DEFAULTS (+ the package's) < its preset < `resolver_cfg`; its game is
    `cfg['resolver']`. Unknown or co-training keys in `resolver_cfg` are errors."""
    rc = dict(cfg.get('resolver_cfg') or {})
    bad = sorted(set(rc) - set(DEFAULTS))
    if bad:
        raise SystemExit(f'resolver_cfg: unknown settings {bad}')
    nested = sorted(k for k in CO_KEYS if k in rc)
    if nested:
        raise SystemExit(f'resolver_cfg: {nested} are the primary run\'s settings')
    preset = rc.get('preset') or ''
    if preset and preset not in PRESETS:
        raise SystemExit(f'resolver_cfg: unknown preset {preset!r} ({sorted(PRESETS)})')
    if rc.get('game') not in (None, cfg['resolver']):
        raise SystemExit(f"resolver_cfg: game {rc['game']!r} is not the resolver {cfg['resolver']!r}")
    out = dict(DEFAULTS)
    out.update(PRESETS.get(preset, {}))
    out.update(rc)
    out.update(game=cfg['resolver'], preset=preset, resolver='', resolver_cfg={}, resolver_init='')
    if out.get('policy_target', 'visits') not in ('visits', 'cq'):
        raise SystemExit(f"resolver_cfg: policy_target {out['policy_target']!r}: 'visits' or 'cq'")
    check_eval_settings(out, 'resolver_cfg: ')
    from .remote import known_remote
    from .remoteplay import remote_names
    for k in ('eval_remote', 'selfplay_remote'):
        # (selfplay_remote: a comma-separated list, each name a remote or a parametrised one, docs/az.md 34)
        for rem in (remote_names(out.get(k)) if k == 'selfplay_remote' else [out.get(k) or '']):
            if rem and not known_remote(rem):
                raise SystemExit(f'resolver_cfg: {k} {rem!r}: not in the game package remotes '
                                 f'({sorted(package.active().remotes)})')
    return out


class _Stop(BaseException):
    """Ends the resolver thread (pause, an error on the main thread)."""


class Coordinator:
    """Moves a primary driver's deferred episodes into a resolver driver's queue and the resolver's
    results back. Each driver is only ever touched by its own thread (a driver's methods must not
    run concurrently: `next_batch` holds it while it waits); the shared state is under `cv`.

    Primary thread: `take_events(drv)` (parked games -> queued episodes), `resolve_ready(drv)`.
    Resolver thread: `feed(drv)` (enqueue), `collect(drv)` (result rows by tag -> ready games).
    A remote consumer (zero.remoteplay, its threads): `take_for_remote(n)`, `collect_rows(rows)`,
    `requeue(items)`; with one, `local_cap` (the shared queue's cap of the local driver,
    zero/workqueue.py: kept up to date by the remote's manager) bounds the episodes outstanding on the
    local driver (queued in it or playing), and the coordinator keeps the local driver's room under
    it in reserve: a remote consumer takes only what is queued beyond that (the local driver never
    waits while a remote holds its share). A requeued tag may come back more than once (the remote
    finishes it after all; requeued k times, k + 1 copies can be in flight): `dupe_ok` counts each
    tag's spare copies (one per requeue), the first row fills the episode's slot and every later one
    is dropped against a spare (`dup_dropped`). A row for a filled slot (or an event already
    resolved) with no spare left is an error, as an unknown tag is."""

    def __init__(self, episodes, results):
        self.episodes_of, self.results_of = episodes, results
        self.cv = threading.Condition()
        self.events = {}                           # primary gid -> {payload, n, rows, t}
        self.to_queue = deque()                    # (tag, spec) for the resolver
        self.ready = deque()                       # (gid, results JSON, t ready, t parked)
        self.outstanding = 0                       # resolver episodes queued or playing
        self.done_rows = []                        # the resolver's result rows (taken by its owner)
        self.error = None
        self.stop = False                          # the resolver thread ends now
        self.closing = False                       # the primary's phase is over: the resolver's too
        self.snapshot = {}                         # the resolver driver's counters (its thread)
        self.resolved_phase = 0
        self.lat = []                              # per resolved game: (parked -> resolved s, ready -> resolved s)
        self.waited = 0.0                          # seconds the primary waited for backpressure
        self.local_cap = None                      # a remote consumer: the local driver's outstanding at most this
        self.local_out = 0                         # episodes fed to the local driver whose rows are not back
        self.util = [0.0, 0]                       # local_out / local_cap, sampled (the shared queue's measure)
        self.remote_rows = []                      # the remote's accepted rows (taken by the run)
        self.dupe_ok = {}                          # requeued tag -> its spare copies (one per requeue): dropped
        self.dup_dropped = 0
        self.ev_total = 0                          # events parked so far and their episodes (matches per event)
        self.ep_total = 0

    # ---- primary thread
    def take_events(self, drv):
        new = drv.take_deferred()
        if not new:
            return 0
        items = [(gid, payload, self.episodes_of(payload)) for gid, payload in new]
        now = time.time()
        with self.cv:
            for gid, payload, eps in items:
                if not eps:
                    raise ValueError(f'deferred game {gid}: no episodes in its payload {payload[:300]}')
                self.events[gid] = {'payload': payload, 'n': len(eps), 'rows': {}, 't': now}
                for k, spec in enumerate(eps):
                    self.to_queue.append((f'{gid}/{k}', spec))
                self.outstanding += len(eps)
                self.ev_total += 1
                self.ep_total += len(eps)
            self.cv.notify_all()
        return len(new)

    def resolve_ready(self, drv, limit=None):
        n = 0
        while limit is None or n < limit:
            with self.cv:
                if not self.ready:
                    break
                gid, res, t_ready, t_park = self.ready.popleft()
            drv.resolve(gid, res)
            now = time.time()
            self.lat.append((now - t_park, now - t_ready))
            n += 1
        self.resolved_phase += n
        return n

    # ---- resolver thread
    def feed(self, drv):
        with self.cv:
            if self.local_cap is None:
                items = list(self.to_queue)
                self.to_queue.clear()
            else:
                room = max(0, self.local_cap - self.local_out)
                items = [self.to_queue.popleft() for _ in range(min(room, len(self.to_queue)))]
            self.local_out += len(items)
        for tag, spec in items:
            drv.enqueue(tag, spec)
        return len(items)

    def collect(self, drv):
        rows = drv.take_results()
        if not rows:
            return 0
        with self.cv:
            self.local_out -= len(rows)
        return self._accept(rows, self.done_rows)

    def sample_util(self):
        """The local driver's outstanding over its cap, sampled (the resolver thread, twice a second)."""
        cap = self.local_cap
        if cap:
            with self.cv:
                self.util[0] += min(1.0, self.local_out / cap)
                self.util[1] += 1

    def take_util(self):
        with self.cv:
            u, self.util = self.util, [0.0, 0]
        return u[0] / u[1] if u[1] else None

    def _accept(self, rows, sink):
        """Result rows (any consumer's) into their events; an event with all its rows is ready."""
        done, kept = [], []
        with self.cv:
            for r in rows:
                tag = r.get('tag')
                g, _, k = (tag or '').partition('/')
                ev = self.events.get(int(g)) if g.isdigit() else None
                if self.dupe_ok.get(tag, 0) > 0 and (ev is None or (k.isdigit() and int(k) in ev['rows'])):
                    # a later copy of a requeued episode (its slot filled, or its event resolved): one spare used
                    self.dupe_ok[tag] -= 1
                    if not self.dupe_ok[tag]:
                        del self.dupe_ok[tag]
                    self.dup_dropped += 1
                    continue
                if ev is None or not k.isdigit():
                    raise RuntimeError(f'a resolver result row with an unknown tag {tag!r}: {r}')
                if int(k) in ev['rows']:
                    # (never counted twice: `outstanding` and the metrics would drift)
                    raise RuntimeError(f'a second resolver result row for {tag!r}, which was never requeued: {r}')
                ev['rows'][int(k)] = r
                self.outstanding -= 1
                kept.append(r)
                if len(ev['rows']) == ev['n']:
                    done.append((int(g), self.events.pop(int(g))))
            sink.extend(kept)
        out = [(g, self.results_of(ev['payload'], [ev['rows'][k] for k in range(ev['n'])]), time.time(), ev['t'])
               for g, ev in done]
        if out:
            with self.cv:
                self.ready.extend(out)
                self.cv.notify_all()
        return len(kept)

    def take_rows(self):
        with self.cv:
            rows, self.done_rows = self.done_rows, []
        return rows

    # ---- a remote consumer (zero.remoteplay's threads)
    def take_for_remote(self, n, skip=()):
        """Up to `n` queued episodes (tag, spec) for the remote, oldest first - beyond the local
        driver's room under its cap, which stays here for it. `skip`: tags this consumer must not
        get (a worker's own requeued ones: it was too slow with them or gave them back), left where
        they are for the other consumers."""
        with self.cv:
            reserve = 0 if self.local_cap is None else max(0, self.local_cap - self.local_out)
            k = min(n, max(0, len(self.to_queue) - reserve))
            if not skip:
                return [self.to_queue.popleft() for _ in range(k)]
            out, kept = [], []
            while self.to_queue and len(out) < k:
                item = self.to_queue.popleft()
                (kept if item[0] in skip else out).append(item)
            self.to_queue.extendleft(reversed(kept))
            return out

    def collect_rows(self, rows):
        """The remote's result rows -> how many were taken (the rest: duplicates dropped)."""
        return self._accept(rows, self.remote_rows)

    def requeue(self, items):
        """Episodes the remote lost or did not finish in time, back at the front of the queue: one
        more spare copy each (whichever copies finish after the first are dropped, however many
        times the episode was requeued)."""
        with self.cv:
            for tag, spec in reversed(items):
                self.to_queue.appendleft((tag, spec))
            for tag, _ in items:
                self.dupe_ok[tag] = self.dupe_ok.get(tag, 0) + 1
            self.cv.notify_all()

    def take_remote_rows(self):
        with self.cv:
            rows, self.remote_rows = self.remote_rows, []
        return rows


def _resolver_loop(co, drv, evs, ext, resume, on_batch):
    """The resolver thread: play queued episodes until the primary's phase is over (`co.closing`),
    then end the resolver's phase (games in flight are kept) and drain."""
    def pump(lb=None):
        if co.stop:
            raise _Stop()
        co.feed(drv)
        co.collect(drv)
        if on_batch is not None:
            on_batch(drv)
        if co.closing:
            drv.end_phase()
    try:
        if resume is not None:
            drv.resume(*resume)
        while True:
            pump()
            drive(drv, evs, ext, on_batch=pump)
            if co.closing:
                break
            with co.cv:                            # idle: nothing queued
                co.cv.wait_for(lambda: co.to_queue or co.closing or co.stop, timeout=0.5)
        pump()
    except _Stop:
        pass
    except BaseException as e:                     # noqa: BLE001 - re-raised on the main thread
        co.error = e
        with co.cv:
            co.cv.notify_all()


def run_pair(co, pdrv, pevs, rdrv, revs, pext=None, rext=None, check=None, quota=None, pending_max=0,
             on_primary=None, on_resolver=None, resume=None):
    """Play a primary driver on this thread and a resolver driver on a worker thread through `co`:
    a rolling primary until `quota` of its games have been resolved in this phase, a finite one
    until every game finished or was resolved. `resume`: (quota, net) of the resolver's phase."""
    co.closing = False
    co.stop = False
    co.resolved_phase = 0
    limit = (lambda: max(0, quota - co.resolved_phase)) if quota is not None else (lambda: None)
    done = (lambda: co.resolved_phase >= quota) if quota is not None else (lambda: False)

    def raise_error():
        if co.error is not None:
            raise co.error

    def pump(lb=None):
        raise_error()
        co.take_events(pdrv)
        co.resolve_ready(pdrv, limit())
        if on_primary is not None:
            on_primary(pdrv)
        t0 = time.time()
        while pending_max and co.outstanding >= pending_max and not done():
            raise_error()
            if check is not None:
                check()
            with co.cv:
                co.cv.wait_for(lambda: co.ready or co.error is not None, timeout=0.2)
            co.resolve_ready(pdrv, limit())
            if on_primary is not None:
                on_primary(pdrv)
        co.waited += time.time() - t0

    th = threading.Thread(target=_resolver_loop, args=(co, rdrv, revs, rext, resume, on_resolver),
                          name='zero-resolver', daemon=True)
    th.start()
    try:
        drive(pdrv, pevs, pext, check=check, on_batch=pump)
        if quota is None:                          # a finite primary: wait for the last resolutions
            while True:
                pump()
                if not pdrv.pending:
                    break
                if check is not None:
                    check()
                with co.cv:
                    co.cv.wait_for(lambda: co.ready or co.error is not None, timeout=0.5)
        pump()
    except BaseException:
        co.stop = True
        with co.cv:
            co.cv.notify_all()
        th.join()
        raise
    with co.cv:
        co.closing = True
        co.cv.notify_all()
    while th.is_alive():
        th.join(0.5)
        if check is not None and co.error is None:
            try:
                check()
            except BaseException:
                co.stop = True
                th.join()
                raise
    raise_error()


class PrimaryRun(Run):
    """The primary network's `Run`: its matches (evaluation, Elo, reference games) are games whose
    deferred episodes the resolver plays (`CoRun.event_match`)."""

    def __init__(self, run, cfg, pause, co):
        super().__init__(run, cfg, pause)
        self.co = co

    def play_match(self, it, opp, n_games, sims, seed, opp_model=None):
        return self.co.event_match(it, opp, n_games, sims, seed, opp_model)

    def play_matches(self, it, opps, n_games, sims, seed):
        return {j: self.co.event_match(it, 'nn', n_games, sims, seed + 101 * j, model_path(self.run, j)) for j in opps}

    def bots_match(self, a, b, n_games, seed):
        return sum(r[0] for r in self.co.event_match(None, b, n_games, None, seed, a=a))

    def evaluate(self, it, m):
        super().evaluate(it, m)
        if self.co.eval_net_due(it) and self.co.netremote is None:   # (with a remote: CoRun.finish_iter queues it)
            t0 = time.time()
            m['eval_yard_net'] = self.co.eval_net(it)
            self.log(f'iter {it} vs {self.yard} (episodes by the resolver net): {m["eval_yard_net"]}')
            m.setdefault('phases', {})['eval_net_s'] = time.time() - t0


class CoRun:
    """A co-training run (module docs). `loop()` replaces `Run.loop` when the setting `resolver` is
    set."""

    def __init__(self, run, cfg, pause=None):
        self.run, self.cfg, self.pw = run, cfg, pause
        if cfg.get('eval_remote'):
            raise SystemExit("eval_remote: a co-training run evaluates its primary network here (its matches' "
                             "episodes need the resolver's drivers); the resolver's evaluations go to a remote "
                             "with resolver_cfg's eval_remote")
        if cfg.get('selfplay_remote'):
            raise SystemExit("selfplay_remote: a co-training run's primary self-play (its parked games and their "
                             "coordinator) stays here; the resolver's episodes go to a remote with resolver_cfg's "
                             "selfplay_remote")
        from .remoteplay import REMOTES_FILE, remotes_file_in_use
        for d in (run, os.path.join(run, 'resolver')):
            if remotes_file_in_use(d):
                raise SystemExit(f"{os.path.join(d, REMOTES_FILE)}: a co-training run's resolver takes its remotes from "
                                 f"resolver_cfg's selfplay_remote (a comma-separated list), never from the file "
                                 f"(docs/az.md 34): remove it")
        if len(set(cfg['players'])) != 1:
            raise SystemExit(f"co-training: one player count for the primary game (players {cfg['players']})")
        self.rcfg = resolver_config(cfg)
        if len(set(self.rcfg['players'])) != 1:
            raise SystemExit(f"co-training: one player count for the resolver game (players {self.rcfg['players']})")
        self.P = PrimaryRun(run, cfg, pause, self)
        self.R = Run(os.path.join(run, 'resolver'), self.rcfg, pause)
        self.R.log = lambda msg: log(run, f'[resolver] {msg}')
        self.deferred = package.active().deferred_of(cfg['game'])
        n = cfg['players'][0]
        self.team = int(G.engine().Game(cfg['game'], G.spec_json(cfg['spec']), n, 0).team_size)
        if self.team < 1 or n % self.team:
            raise SystemExit(f"{cfg['game']}: team size {self.team} does not divide {n} players")
        self.nteams = n // self.team
        self.sp = None                             # the persistent self-play drivers
        # the primary's resolver-network events (eval_yard_net) on a remote (docs/az.md 11.7):
        # zero.remote.RemoteNetEval, set up by loop()
        self.netremote = None
        rem = cfg.get('draft_eval_net_remote') or ''
        if rem and rem != 'local' and rem not in package.active().remotes:
            raise SystemExit(f'draft_eval_net_remote {rem!r}: not in the game package remotes '
                             f'({sorted(package.active().remotes)})')

    def log(self, msg):
        log(self.run, msg)

    def check(self, where=''):
        if self.pw is not None:
            self.pw.check(where)

    # ---------------------------------------------------------------- checkpoints
    def latest(self):
        """The newest iteration with both checkpoints (None: none)."""
        a, b = latest_iter(self.run), latest_iter(self.R.run)
        if a is None or b is None:
            return None
        k = min(a, b)
        while k >= 0 and not (os.path.exists(model_path(self.run, k)) and os.path.exists(model_path(self.R.run, k))):
            k -= 1
        return k if k >= 0 else None

    def first_checkpoints(self):
        """No iteration has both checkpoints yet: iteration 0 of each network. A network's existing
        checkpoint 0 is kept, never overwritten (e.g. the primary forked into the run by
        tools/fork_run.py, the resolver brought in by resolver_init: docs/az.md 11); the missing one
        is created - the primary fresh, the resolver from resolver_init (else fresh). Refused (an
        inconsistent run directory: which checkpoint 0 to start from is not clear) when either
        network has checkpoints after 0, or when the resolver has a checkpoint 0 and resolver_init
        is set too."""
        P, R, c = self.P, self.R, self.cfg
        for X in (P, R):
            later = sorted(k for k in (int(os.path.basename(p)[5:9]) for p in
                                       glob.glob(os.path.join(X.run, 'models', 'iter_[0-9][0-9][0-9][0-9].pt'))) if k > 0)
            if later:
                raise SystemExit(f'co-training: {X.run} has checkpoints after iteration 0 ({later[:4]}) but no iteration '
                                 f'has both networks: not started (which checkpoint 0 to start from? '
                                 f'tools/fork_run.py --source copies a co-training run with both)')
        p0, r0 = model_path(P.run, 0), model_path(R.run, 0)
        if os.path.exists(r0) and c.get('resolver_init'):
            raise SystemExit(f'co-training: resolver_init {c["resolver_init"]} and a resolver checkpoint 0 already '
                             f'there ({r0}): not started (remove one of them)')
        if os.path.exists(p0):
            self.log(f'primary iteration 0 = the checkpoint already there, kept: {p0}')
        else:
            save(P.new_model(), p0, {'iter': 0})
        if os.path.exists(r0):
            self.log(f'resolver iteration 0 = the checkpoint already there, kept: {r0}')
        elif c.get('resolver_init'):
            self.init_resolver(c['resolver_init'])
        else:
            save(R.new_model(), r0, {'iter': 0})

    def init_resolver(self, path):
        """The resolver's iteration 0 from a checkpoint (a single-game run's works): widened to the
        game's features / actions if the game grew; its architecture is the checkpoint's. The
        checkpoint may be of an older encoding (the widen's purpose: loaded past the encoding
        guard); what is written is checked against this build's (docs/az.md 10.4)."""
        info = G.info(self.rcfg['game'])
        old, extra = load(path, 'cpu', allow_mismatch=True)
        game = old.config.get('game')
        if game not in (None, self.rcfg['game']):
            raise SystemExit(f"resolver_init {path}: a {game!r} network, the resolver is {self.rcfg['game']!r}")
        if old.config['features'] > info['num_features'] or old.config['actions'] > info['num_actions']:
            raise SystemExit(f"resolver_init {path}: {old.config['features']} features / {old.config['actions']} actions, "
                             f"the game has {info['num_features']} / {info['num_actions']}")
        grid = package.active().grids.get(self.rcfg['game']) if old.config.get('arch') == 'grid' else None
        # (a token network across a change of the game's object layout: widened segment by
        # segment, docs/az.md 10.3)
        layout = package.active().tokens_of(self.rcfg['game']) if old.config.get('arch') == 'tokens' else None
        moved = layout is not None and layout_sans_identity(
            tokens_layout(layout, info['num_features'], info['num_actions'])) != layout_sans_identity(old.config['layout'])
        new = old
        if old.config['features'] < info['num_features'] or old.config['actions'] < info['num_actions'] or moved:
            new = widen(old, info['num_features'], info['num_actions'], grid=grid, layout=layout)
        try:
            check_encoding(new.config, self.rcfg['game'])
        except EncodingMismatch as e:
            raise EncodingMismatch(f'resolver_init {path}, widened: {e}') from None
        save(new, model_path(self.R.run, 0), {'iter': 0, 'init_from': os.path.abspath(path), 'source_extra': extra})
        self.log(f'resolver iteration 0 = {path} ({old.config.get("arch", "mlp")}, {old.config["features"]} features, '
                 f'{old.config["actions"]} actions; extra {json.dumps(extra, default=str)[:200]})')

    # ---------------------------------------------------------------- matches of the primary
    def eval_net_due(self, it):
        """Iteration `it`'s evaluation includes the events played by the resolver's network
        (`eval_yard_net`): an evaluated iteration that is a multiple of draft_eval_net_every."""
        c = self.cfg
        k = int(c.get('draft_eval_net_every') or 0)
        evaluated = it % max(1, int(c['eval_every'])) == 0 or it == 0
        return bool(k and evaluated and it % k == 0 and c['eval_games'])

    def eval_net(self, it):
        """`eval_yard_net` of iteration `it`: the primary's checkpoint against its yardstick, each
        event's episodes played by the resolver's checkpoint of the same iteration at its evaluation
        settings. -> the summary (+ 'opp'). The same code here and on a remote (zero.remote's
        `evalnet`, docs/az.md 11.7)."""
        c, P = self.cfg, self.P
        s = summarize(self.event_match(it, P.yard, c['eval_games'], c['eval_sims'], seed=7919 * it + 3, play='net'))
        s['opp'] = P.yard
        # the move rules (recorded unless the default): the primary's seats', the resolver's
        for k, tag in (('select', P.eval_tag()), ('resolver_select', self.R.eval_tag())):
            if tag != 'visits':
                s[k] = tag
        return s

    def event_match(self, it, opp, n_games, sims, seed, opp_model=None, play='bot', a=None):
        """Primary checkpoint `it` (or the player `a`: a bot or 'ext:<name>') against `opp` ('nn' +
        opp_model, a bot or 'ext:<name>'), team seats alternating, each game's episodes played by
        the resolver's yardstick (`play` 'bot') or its network at checkpoint `it` ('net', evaluation
        settings). -> rows (score, margin, rounds, over) of the first player: its team's outcome, its
        team's score less the best other team's."""
        if n_games <= 0:
            return []
        P, R, c, rc = self.P, self.R, self.cfg, self.rcfg
        n, nt = c['players'][0], self.nteams
        if a is None:
            sa, evs = P.nn_seat('m', sims), {'m': P.evaluator(model_path(P.run, it))}
        else:
            sa, evs = G.seat_dict(a), {}
        so = P.opp_seat(opp, sims)
        if opp == 'nn':
            evs['o'] = P.evaluator(opp_model)
        pext = {s[4:]: G.external(s[4:], seed) for s in (a, opp) if G.is_external(s)}
        seats = [[sa if s % nt == g % nt else so for s in range(n)] for g in range(n_games)]
        pdrv = P.driver(seats, seed=seed, conc=min(c['conc'], n_games), groups=P.match_groups(pext))
        if play == 'net':
            rs, revs, rext = R.nn_seat('m', rc['eval_sims']), {'m': R.evaluator(model_path(R.run, it))}, {}
        else:
            rs, revs = G.seat_dict(R.yard), {}
            rext = {R.yard[4:]: G.external(R.yard[4:], seed + 1)} if G.is_external(R.yard) else {}
        rdrv = R.driver([[rs] * rc['players'][0]], seed=seed + 1, conc=rc['conc'], rolling=True, queue_only=True,
                        groups=R.match_groups(rext))
        co = Coordinator(*self.deferred)
        run_pair(co, pdrv, evs, rdrv, revs, pext=pext, rext=rext, check=lambda: self.check('evaluation'),
                 resume=(BIG, 0))
        for name, drv in (('primary', pdrv), ('resolver', rdrv)):
            self.log_slow(drv.stats(), {}, f'evaluation iter {it} vs {opp} ({name} driver)')
        rows = []
        for r in sorted(pdrv.take_results(), key=lambda r: r['gid']):
            me = r['gid'] % nt
            sc = r['scores']
            rows.append((sum(r['result'][s] for s in range(n) if s % nt == me),
                         sc[me] - max(sc[s] for s in range(n) if s % nt != me), r['rounds'], r['over']))
        return rows

    # ---------------------------------------------------------------- self-play
    def selfplay(self, it):
        P, R, c, rc = self.P, self.R, self.cfg, self.rcfg
        for X in (P, R):
            d = os.path.join(X.run, 'data', f'iter_{it:04d}')
            if os.path.exists(d) and not (X.rsp is not None and X.rsp.began(it)):
                X.drop_data(d, strict=True)        # an interrupted iteration is replayed
        t0 = time.time()
        rconc = c.get('resolver_conc') or rc['conc']
        rsp = R.rsp
        # backpressure: twice the local episodes in flight (and the remote's room, with a remote)
        pending_max = c.get('pending_max') or 2 * rconc + (rsp.capacity() if rsp is not None else 0)
        if self.sp is None:
            pseat = P.nn_seat('m', c['sims'], selfplay=True)
            rseat = R.nn_seat('m', rc['sims'], selfplay=True)
            pdrv = P.driver([[pseat] * p for p in c['players']], seed=c['seed'] * 7919 + it, record=True, rolling=True,
                            net=it - 1)
            rdrv = R.driver([[rseat] * rc['players'][0]], seed=rc['seed'] * 7919 + it + 104729, record=True,
                            rolling=True, net=it - 1, conc=rconc, queue_only=True)
            self.sp = {'p': pdrv, 'r': rdrv, 'pev': P.evaluator(model_path(P.run, it - 1)),
                       'rev': R.evaluator(model_path(R.run, it - 1)), 'co': Coordinator(*self.deferred)}
        else:
            for k, X in (('pev', P), ('rev', R)):
                m, _ = load(model_path(X.run, it - 1), 'cpu', game=X.cfg['game'])
                self.sp[k].set_network(m)
        sp = self.sp
        pdrv, rdrv, pev, rev, co = sp['p'], sp['r'], sp['pev'], sp['rev'], sp['co']
        pst0, rst0 = pdrv.stats(), rdrv.stats()
        pdrv.resume(c['games'], it - 1)
        writers, tstats = {}, {}
        for k, X in (('p', P), ('r', R)):
            # (`Run.shard_meta`: with the game's info, so a resolver game with variable-length rows writes format v2)
            meta = X.shard_meta(it, net='primary' if k == 'p' else 'resolver')
            writers[k] = ShardWriter(os.path.join(X.run, 'data', f'iter_{it:04d}'), 'r00', rows=X.cfg['shard_rows'],
                                     meta=meta, bad_dir=X.debug_dir())
            tstats[k] = TargetStats()
        events = []
        state = {'t': 0.0, 'parked_max': 0, 'tr': 0.0}
        stale = {'p': {}, 'r': {}}                 # rows by checkpoint lag (with a remote: in the metrics)
        prog = os.path.join(self.run, 'progress.json')
        co.take_rows()
        if rsp is None or not rsp.began(it):
            co.take_remote_rows()                  # (begun early: the rows of its head start are this phase's)
        co.lat = []
        co.waited = 0.0
        dup0 = co.dup_dropped
        if rsp is not None:
            # the remote takes queued episodes as it has room (the shared queue: the local driver's
            # share first, its cap in co.local_cap)
            rsp.begin(it, co)
            co.take_util()

        def add(k, s):
            try:
                writers[k].add(s)
            except NonFinite as e:
                raise NonFinite(f'{self.label(k)}: {e}') from e
            tstats[k].add(s)
            if s is not None:
                lag_hist(s['net'], it, stale[k])

        def on_primary(drv):
            add('p', drv.take_samples())
            events.extend(drv.take_results())
            now = time.time()
            if now - state['t'] > 2:
                state['t'] = now
                st = drv.stats()
                state['parked_max'] = max(state['parked_max'], st['parked_bytes'])
                r = co.snapshot
                p = {'iter': it, 'phase': 'selfplay', 'events_resolved': len(events), 'target': c['games'],
                     'games': len(events), 'drafting': drv.in_flight()['games'], 'pending_events': st['parked'],
                     'matches_queued': r.get('queued', 0) + len(co.to_queue),
                     'matches_in_flight': r.get('in_flight', 0), 'matches_done': r.get('done', 0), 'time': now}
                if rsp is not None:
                    p['matches_remote'] = rsp.progress()
                _write_json(prog, p)

        def on_resolver(drv):
            add('r', drv.take_samples())
            now = time.time()
            if now - state['tr'] > 0.5:
                state['tr'] = now
                st = drv.stats()
                co.snapshot = {'queued': st['queued'], 'in_flight': drv.in_flight()['games'],
                               'done': st['games_finished'] - rst0['games_finished']}
                co.sample_util()

        run_pair(co, pdrv, {'m': pev}, rdrv, {'m': rev}, check=lambda: self.check('self-play'), quota=c['games'],
                 pending_max=pending_max, on_primary=on_primary, on_resolver=on_resolver, resume=(BIG, it - 1))
        on_primary(pdrv)
        add('r', rdrv.take_samples())
        matches = co.take_rows()
        ws = {}
        for k in ('p', 'r'):
            try:
                ws[k] = writers[k].close()
            except NonFinite as e:
                raise NonFinite(f'{self.label(k)}: {e}') from e
        secs = time.time() - t0                    # this machine's phase (the remote's wait not counted)
        rem = None
        if rsp is not None:
            _write_json(prog, {'iter': it, 'phase': 'remote shards', 'time': time.time()})
            try:                                   # (PAUSE ends the wait, STOP bounds it: docs/az.md 47.4)
                rem = rsp.end(it, local={'units': len(matches), 'run_s': secs, 'util': co.take_util(),
                                         'leaves': rdrv.stats()['leaves'] - rst0['leaves']},
                              check=lambda: self.check('remote shards'))
            except NonFinite as e:
                raise NonFinite(f'{self.label("r")}: {e}') from e
        rrows = co.take_remote_rows()
        _write_json(prog, {'iter': it, 'phase': 'train', 'time': time.time()})
        pst, rst = pdrv.stats(), rdrv.stats()
        lat = np.array(co.lat) if co.lat else np.zeros((0, 2))

        def summary(rows, st, st0, ev, w, ts, X):
            fin = [r for r in rows if r['over']]
            leaves = st['leaves'] - st0['leaves']
            out = {'games': len(rows), 'unfinished': len(rows) - len(fin), 'samples': w['samples'], 'shards': w['shards'],
                   'bytes': w['bytes'], 'decisions_per_game': sum(r['decisions'] for r in rows) / max(1, len(rows)),
                   'steps_per_game': sum(r['steps'] for r in rows) / max(1, len(rows)),
                   'rounds': sum(r['rounds'] for r in fin) / max(1, len(fin)),
                   'avg_score': sum(sum(r['scores']) / r['n'] for r in fin) / max(1, len(fin)),
                   'leaves': leaves, 'leaves_per_s': leaves / max(secs, 1e-9),
                   'games_per_h': len(rows) / max(secs, 1e-9) * 3600, 'seconds': secs,
                   'infer_s': ev.seconds, 'avg_batch': ev.leaves / max(1, ev.calls),
                   **minor_metrics(st, st0, rows, exact=False),     # (rolling drivers: the count, az.md 47.9)
                   'target': ts.summary(X.cfg.get('policy_target', 'visits')),
                   'rolling': {'in_flight': 0, 'restarts': st['restarts'] - st0['restarts'],
                               'restart_sims': st['restart_sims'] - st0['restart_sims'],
                               'dropped_leaves': st['dropped_leaves'] - st0['dropped_leaves'],
                               'slow_steps': st.get('slow_steps', 0) - st0.get('slow_steps', 0)}}
            ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0
            return out
        mp = summary(events, pst, pst0, pev, ws['p'], tstats['p'], P)
        mr = summary(matches, rst, rst0, rev, ws['r'], tstats['r'], R)
        self.log_slow(pst, pst0, f'iter {it} self-play (primary driver)')
        self.log_slow(rst, rst0, f'iter {it} self-play (resolver driver)')
        mp['rolling']['in_flight'] = pdrv.in_flight()['games']
        mr['rolling']['in_flight'] = rdrv.in_flight()['games']
        nep = len(matches) + len(rrows)
        mp['cotrain'] = {'events': len(events), 'episodes': nep, 'episodes_per_h': nep / max(secs, 1e-9) * 3600,
                         'pending_events': pst['parked'], 'episodes_queued': rst['queued'] + len(co.to_queue),
                         'episodes_waiting': co.outstanding,
                         'parked_mb_max': round(max(state['parked_max'], pst['parked_bytes']) / 1e6, 3),
                         'event_latency_s': [round(float(lat[:, 0].mean()), 3), round(float(lat[:, 0].max()), 3)] if len(lat) else None,
                         'resolve_delay_ms': [round(float(lat[:, 1].mean()) * 1e3, 2), round(float(lat[:, 1].max()) * 1e3, 2)] if len(lat) else None,
                         'backpressure_s': round(co.waited, 2), 'pending_max': pending_max, 'resolver_conc': rconc}
        if rem is not None:
            from .remoteplay import staleness
            mp['cotrain']['episodes_remote'] = len(rrows)
            mp['cotrain']['duplicates_dropped'] = co.dup_dropped - dup0
            mp['cotrain']['staleness'] = staleness(stale['p'], {})
            qb = rem.pop('queue')
            cons = qb['consumers']
            cons.setdefault('local', {})['games'] = len(matches)
            for n, v in cons.items():
                if n != 'local':
                    v['games'] = v.get('units', 0)
            tot = sum(v.get('games', 0) for v in cons.values())
            for v in cons.values():
                v['share'] = round(v.get('games', 0) / max(1, tot), 4)
            rems = rem.pop('remotes', None)        # (a list of remotes: each one's share, docs/az.md 34)
            mr['remote'] = rem
            if rems is not None:
                mr['remotes'] = rems
            mr['queue'] = {'units': qb['units'], 'inflight': qb['inflight'], 'batch_inflight': qb['batch_inflight'],
                           'consumers': cons, 'rates': qb['rates'], 'local_cap': co.local_cap,
                           'staleness': staleness(stale['r'], qb['stale'])}
            R.log(f'iter {it} remote self-play: {_fmt_remote(rem)}')
            for name, r in (rems or {}).items():
                R.log(f'iter {it} remote self-play {name}: {_fmt_remote(r)}')
            R.log(f'iter {it} shared queue: {_fmt_queue(mr["queue"])}')
        return mp, mr

    def log_slow(self, st, st0, where):
        """Log the game steps (one game's search and moves between two of its leaves) that took
        over 10 s since `st0` (the driver's `slow_steps`; zero/search: SLOW_STEP_NS): an engine's
        slow path holds up its driver group that long (docs/az.md 11.8)."""
        n = st.get('slow_steps', 0) - st0.get('slow_steps', 0)
        if n > 0:
            self.log(f'{where}: {n} game step(s) took over 10 s (the longest so far {st.get("step_max_s", 0):.0f} s, '
                     f'game {st.get("step_max_gid")}): an engine slow path?')

    def label(self, k):
        return f"primary net ({self.cfg['game']})" if k == 'p' else f"resolver net ({self.rcfg['game']})"

    # ---------------------------------------------------------------- training
    def train(self, it, sp_p, sp_r):
        """Both networks, then the guard on both checkpoints and optimizer states, then both saves."""
        out = {}
        n_r = sp_r['samples'] + sp_r.get('remote', {}).get('samples', 0)
        for k, X, n in (('p', self.P, sp_p['samples']), ('r', self.R, n_r)):
            try:
                out[k] = X.train_model(it, n)
            except NonFinite as e:
                raise NonFinite(f'{self.label(k)}: {e}') from e
        for k, X in (('p', self.P), ('r', self.R)):
            model, ost, _ = out[k]
            try:
                check_state(ost, X.opt_path())
                check_state(model.state_dict(), model_path(X.run, it))
            except NonFinite as e:
                raise NonFinite(f'{self.label(k)}: {e}') from e
        self.R.save_trained(it, out['r'][0], out['r'][1])
        self.P.save_trained(it, out['p'][0], out['p'][1])
        return out['p'][2](), out['r'][2]()

    # ---------------------------------------------------------------- the loop
    def finish_iter(self, it, t0):
        """Both checkpoints exist: the resolver's evaluation (a single-game run's), then the
        primary's (its matches need the resolver's checkpoint), each from its partial stats."""
        for X in (self.R, self.P):
            if it in {m['iter'] for m in read_metrics(X.run)}:
                continue
            m = {'iter': it}
            if os.path.exists(X.partial_path(it)):
                with open(X.partial_path(it)) as f:
                    m = json.load(f)
            X.finish_iter(it, m, t0)
            if X is self.P and self.netremote is not None and self.eval_net_due(it):
                self.netremote.enqueue(it)         # eval_yard_net on the remote (its row: evals.jsonl)

    def loop(self):
        c, rc, run = self.cfg, self.rcfg, self.run
        P, R = self.P, self.R
        check_devices(c)                           # (both networks' devices: DeviceRefused, exit 78)
        check_devices(rc, 'resolver_cfg: ')
        os.makedirs(os.path.join(R.run, 'models'), exist_ok=True)
        _write_json(os.path.join(R.run, 'config.json'), rc, strict=True, indent=1)   # (atomically, as the run's)
        # variable-length rows (docs/az.md 43): both games' refusals at start (a draft layout, a stale build, an arch
        # or shard format that cannot carry them; RowsRefused, exit 78) - nothing for games without them
        for X in (P, R):
            X.check_rows()
        start = self.latest()
        if start is None:
            self.first_checkpoints()
            start = 0
        # both checkpoints of the start are networks of this build's encodings - a resume's, or an
        # iteration 0 kept from a fork (docs/az.md 10.4; EncodingMismatch: the run refuses to start)
        for X in (P, R):
            check_checkpoint(model_path(X.run, start), X.cfg['game'])
        self.log(f'co-training run {run} starting after iter {start}: primary {c["game"]} (team size {self.team}), '
                 f'resolver {rc["game"]}; resolver cfg {json.dumps(rc)}')
        if rc.get('eval_remote'):
            # the resolver's evaluations on a remote, as a single-game run's (zero.remote): queued
            # by its finish_iter, the missing ones again now (a restart)
            from .remote import RemoteEval
            R.remote = RemoteEval(R, rc['eval_remote'], mirror=os.path.basename(os.path.normpath(run)) + '.resolver')
            rdone = {m['iter'] for m in read_metrics(R.run)}
            R.remote.backfill(start if start in rdone else start - 1)
            R.log(f'evaluations on the remote {rc["eval_remote"]!r} ({R.remote.spec["host"]}, mirror {R.remote.rdir})')
        if c.get('draft_eval_net_remote'):
            # the primary's resolver-network events (eval_yard_net) on a remote (docs/az.md 11.7):
            # queued by finish_iter, the missing ones again now (a restart)
            from .remote import RemoteNetEval
            self.netremote = RemoteNetEval(self, c['draft_eval_net_remote'])
            pdone = {m['iter'] for m in read_metrics(run)}
            self.netremote.backfill(start if start in pdone else start - 1)
            self.log(f'eval_yard_net on the remote {c["draft_eval_net_remote"]!r} ({self.netremote.spec["host"]}, '
                     f'mirror {self.netremote.rdir})')
        if rc.get('selfplay_remote'):
            # a share of the resolver's queued episodes on a remote (zero.remoteplay, docs/az.md 11.6)
            from .remoteplay import RemotePlay
            R.rsp = RemotePlay(R, rc['selfplay_remote'], mode='matches',
                               mirror=os.path.basename(os.path.normpath(run)) + '.resolver')
            # the shared queue (zero/workqueue.py): the local resolver driver is a consumer with its
            # slots and today's queue in its driver at most (batch-bound on CUDA / MPS: its slots, never
            # that queue); an iteration's units are its events' matches; a restart's first caps follow
            # the last measured rates
            rconc = c.get('resolver_conc') or rc['conc']
            R.rsp.local(max(2, int(rc['groups'])), rconc + max(16, rconc // 4), device=rc['infer_device'], slots=rconc)
            R.rsp.events = int(c['games'])
            R.rsp.restore(last_rates(read_metrics(R.run)))
            if not R.rsp.multi:
                R.log(f'episodes also on the remote {rc["selfplay_remote"]!r} ({R.rsp.spec["host"]}, {R.rsp.procs} '
                      f'worker(s), mirror {R.rsp.rdir}){R.rsp.batch_note()}')
            else:                                  # a list of remotes (docs/az.md 34)
                R.log(f'episodes also on the remotes {R.rsp.describe()}{R.rsp.batch_note()}')
        try:
            self._iterations(start)
        finally:
            if R.rsp is not None:
                R.rsp.close()

    def _iterations(self, start):
        c, run = self.cfg, self.run
        P, R = self.P, self.R
        if any(start not in {m['iter'] for m in read_metrics(X.run)} for X in (P, R)):
            if os.path.exists(P.partial_path(start)) or os.path.exists(R.partial_path(start)):
                self.log(f'iter {start}: finishing its evaluations (interrupted)')
            self.finish_iter(start, time.time())
        stop = os.path.join(run, 'STOP')
        for it in range(start + 1, c['iters'] + 1):
            self.check('between iterations')
            if os.path.exists(stop):
                self.log('STOP file found; exiting')
                break
            t0 = time.time()
            # both phases finished before a PAUSE / crash in training (each data directory's DONE.json, verified,
            # docs/az.md 47.1): trained on as they are, never played again
            recs = [X.phase_done(it, quiet=True) for X in (P, R)]
            if all(r is not None for r in recs):
                (sp_p, sp_r), sp_s = (r['selfplay'] for r in recs), float(recs[0]['selfplay_s'])
                self.log(f'iter {it}: both self-play phases finished before the interruption (DONE.json in data/ and '
                         f'resolver/data/, as recorded): not played again, training on them')
                resumed = sp_s
            else:
                sp_p, sp_r = self.selfplay(it)
                sp_s, resumed = time.time() - t0, 0.0
                self.log(f'iter {it} selfplay: {_fmt(sp_p)}')
                self.log(f'[resolver] iter {it} selfplay: {_fmt(sp_r)}')
                if not (sp_r.get('remote') or {}).get('cut'):
                    for X, sp in ((R, sp_r), (P, sp_p)):
                        mark_phase_done(X, it, sp, sp_s)
            t1 = time.time()
            tr_p, tr_r = self.train(it, sp_p, sp_r)
            self.log(f'iter {it} train: {_fmt(tr_p)}')
            self.log(f'[resolver] iter {it} train: {_fmt(tr_r)}')
            t2 = time.time()
            for X, sp, tr in ((P, sp_p, tr_p), (R, sp_r, tr_r)):
                m = {'iter': it, 'selfplay': sp, 'train': tr, 'phases': {'selfplay_s': sp_s, 'train_s': t2 - t1},
                     'resumed_seconds': time.time() - t0 + resumed}
                if resumed:
                    m['selfplay_resumed'] = True
                _write_json(X.partial_path(it), m, strict=True)
            if R.rsp is not None and self.sp is not None and it < c['iters'] and not os.path.exists(stop) \
                    and not os.path.exists(os.path.join(run, 'PAUSE')):
                # both checkpoints are written: the remote goes on with the queued episodes with the
                # resolver's checkpoint it (iteration it + 1's) while the networks are evaluated here
                R.rsp.begin(it + 1, self.sp['co'])
            self.finish_iter(it, time.time())
        if R.rsp is not None:
            R.rsp.close()                          # no more self-play: the remote's workers quit
        waits = [q for q in (R.remote, self.netremote) if q is not None]
        if waits and not os.path.exists(stop):
            # (no time limit while the run goes on; STOP: the missing ones logged, the run ends aborted, az.md 47.3)
            drain_remote_evals(run, waits, self.check, self.log)
        _write_json(os.path.join(run, 'progress.json'), {'phase': 'idle', 'time': time.time()})
