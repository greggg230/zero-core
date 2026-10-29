"""AlphaZero loop: self-play -> train -> evaluate vs the yardstick -> Elo matches (docs/az.md).

  python -m <pkg>.az.run --run runs/toy1 --preset toy-1      (a game package: grz, cvz, ...)
  python -m zero.run --run runs/toy1 --game toy --iters 40 --games 1000 --threads 4   (the core)

Iteration i self-plays with models/iter_{i-1}.pt (the Rust driver in this process, CPU or
CUDA inference; samples streamed into shards under data/iter_NNNN/), trains on the newest
iterations' data (a RAM-bounded window), writes models/iter_i.pt (atomic), evaluates it
against the yardstick (the run's anchor at Elo 1000: a built-in bot, or an external player
such as 'ext:toyheur' / a game's 'ext:jsbot') and the random bot, plays Elo matches against
the checkpoints i-1, i-2, i-4 ... i-32, refits the ratings and appends a line to
metrics.jsonl. Resumes from the newest checkpoint. Settings: DEFAULTS (+ the game package's
overrides) < --preset (the core's toy presets + the game package's) < the run's config.json <
flags given on this command line (saved).

Control files in the run directory (zero/runctl.py): STOP = end after the current
iteration; PAUSE = stop at the next pause point (between batches / training steps) and exit
with status 75; tools/supervise.py relaunches the run when PAUSE is removed.

--rolling 1: one persistent driver keeps --conc games in flight across iterations; an
iteration's self-play ends when --games games have FINISHED (docs/search.md 2.6).

--resolver <game>: a co-training run (zero/cotrain.py, docs/az.md 11): this run's game ends at
deferred terminals whose results come from episodes of the resolver game, played by a second
network trained in the same run (<run>/resolver/); --resolver-cfg holds its settings.

--selfplay-remote <name>: another machine (a remote of the game package) self-plays too, with the
same checkpoint, while this one plays its quota; its shards come back into data/iter_i/
(zero/remoteplay.py, docs/az.md 5.3). A co-training run: the resolver's --resolver-cfg
'{"selfplay_remote": ...}' (the remote plays a share of the queued episodes, 11.6).
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import elo, games as G, package, runctl
from .data import BAD, ShardWriter, TargetStats, Window, shard_paths
from .finite import NonFinite, check_state, head
from .infer import NetEvaluator, drive, use_amp
from .net import load, make_net, save
from .protocol import MAX_SEATS
from .runctl import EXIT_PAUSED, Paused

DEFAULTS = dict(
    preset='',
    # game: a game registered in the package's compiled module ('toy', 'caverna', 'glassroad'),
    # its JSON spec, player counts to sample per self-play game
    game='toy', spec='', players=[2],
    # self-play: games per iteration (rolling: games FINISHED per iteration), games in
    # flight, search threads, batches in flight (groups), torch intra-op threads (1: on a
    # saturated CPU every torch op with > 1 thread waits for preempted threads - measured
    # 10-1000x slower on the shared desktop, docs/az.md 7)
    iters=40, games=1000, rolling=0, conc=256, threads=4, groups=2, torch_threads=1,
    # groups for matches with an external player: its batches are answered on threads while
    # the other groups search (infer.drive), so more groups hide more of a slow player's latency
    # but shrink the network batches. Caverna's JS bot is fast (~18 ms per decision) and CUDA
    # inference costs ~2.6 ms per call on the shared GPU whatever the batch: 200 games vs the JS
    # bot took 129 s inline, 110 s async with 2 groups, 128 s with 4, 186 s with 8 (docs/az.md 5).
    # A game package may override it (Glass Road's ~0.1 s bot: 8, GamePackage.defaults)
    ext_groups=2,
    # search: playout cap (a p_full share of decisions gets `sims` and becomes a sample, the
    # rest sims_fast), PUCT constant (values are P(win) in [0, 1]), root noise, temperature
    # moves, chance_k (sampled chance outcomes kept per edge; 0 = resample every visit),
    # margin_w (utility = P(win) + margin_w tanh(margin)), redet (fresh determinization every
    # simulation)
    sims=64, sims_fast=16, p_full=0.25, c_puct=1.0, alpha=0.3, eps=0.25, temp_moves=20,
    chance_k=0, margin_w=0.0, redet=1, max_steps=100_000,
    # simulations of the game's minor decisions in self-play (SearchGame::minor_decision, e.g.
    # MTG's choice of the land to tap), never recorded; 0 = searched like the others
    sims_minor=0,
    # policy target of the recorded (full-search) decisions: 'visits' (the root visit
    # distribution) | 'cq' (Gumbel AlphaZero's completed-Q target softmax(ln P0 + (cq_c_visit +
    # max N) cq_c_scale completedQ), docs/search.md 3.1); the search itself is the same
    policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0,
    # network (only when the run's first network is created): arch 'mlp' (hidden x blocks),
    # 'grid' (a conv tower of `ch` channels x blocks over the game package's board planes,
    # GamePackage.grids; net.GridNet) or 'tokens' (a transformer over the game package's
    # objects, GamePackage.tokens; net.TokenNet: width `hidden`, `blocks` layers, `heads`
    # attention heads, feed-forward width `ff`, 0 = 2 x hidden)
    hidden=256, blocks=4, arch='mlp', ch=64, heads=4, ff=0,
    # training: steps = new samples * reuse / batch (at least min_steps); AdamW; LR halving
    # every lr_halve_every iterations (floor lr_min), linear warmup while the optimizer state
    # is fresh
    batch=512, reuse=4.0, lr=1e-3, wd=1e-4, lr_halve_every=30, lr_min=1e-4, lr_warmup_steps=100, min_steps=50,
    # window: newest `window` iterations (growing 1 + iter // 2 with window_grow), more until
    # min_window_samples; data kept keep_data iterations; RAM budget; samples per shard
    window=8, window_grow=1, min_window_samples=20_000, keep_data=12, ram_budget_gb=4.0, shard_rows=65536,
    # value target = (1 - q) z + q root-search estimate, q = value_q * min(1, iter / warmup)
    value_q=0.5, value_q_warmup=5, margin_loss_w=0.5,
    # evaluation: the yardstick (Elo anchor = 1000): a built-in bot or 'ext:<name>'; 0 random games:
    # no match against the random bot (the reference games below still rate it, unless ref_games 0)
    yardstick='heuristic', eval_games=200, eval_sims=32, eval_random_games=50, eval_every=1, elo_games=100,
    ref_games=400,
    # the rule of the moves a network seat does not sample (docs/search.md 3.3): 'visits' = argmax
    # (N, P0), AlphaZero's, the default (today's moves, bit for bit) | 'cq' = argmax of the completed-Q
    # improved policy's logits (the 'cq' target's, with cq_c_visit and cq_c_scale) among the options
    # with at least *_min_visits visits (0: all). eval_select: every evaluation seat - the yardstick,
    # random and Elo matches here or on the eval_remote, co-training's evaluations of both networks
    # (resolver_cfg for the resolver's), and by default zero.h2h / zero.replay of the run's checkpoints;
    # the evaluations record it ('select', absent = 'visits') and each rule has its own Elo pool
    # (elo/ for 'visits', elo/<rule>/ for another: docs/az.md 5.4), so a run can change it at a restart.
    # selfplay_select: self-play's moves after its first temp_moves searched decisions of a game (the
    # recorded targets stay policy_target's)
    eval_select='visits', eval_select_min_visits=0, selfplay_select='visits', selfplay_select_min_visits=0,
    # the game spec of every match that records nothing (vs the yardstick, vs random, Elo, the
    # reference games); '' = the game package's evaluation spec for the self-play spec
    # (GamePackage.eval_specs; MTG: a fixed deck set, self-play random decks), else the spec
    eval_spec='',
    # cheaper Elo (CPU runs): each checkpoint plays the first `elo_opponents` of the checkpoints
    # 1, 2, 4 ... 32 evaluations back (x eval_every iterations; 0 = all), at elo_sims (0 = eval_sims);
    # elo_together = 1 plays an iteration's Elo matches in one driver (larger network batches, other
    # seeds). elo_offsets ('1,2,4,8'; '' = the evaluations back above): each evaluated checkpoint plays
    # the checkpoints that many ITERATIONS back instead, whatever eval_every (every iteration writes
    # one; elo_opponents is not used): the ones never evaluated are rated through these games only
    # and are never models/best.pt (docs/az.md 5)
    elo_opponents=0, elo_sims=0, elo_together=0, elo_offsets='',
    # evaluation on another machine (a name in the game package's `remotes`, 'local' for tests;
    # '' = here): the loop queues each evaluated checkpoint and keeps training (zero.remote)
    eval_remote='',
    # self-play on another machine too (a name in the game package's `remotes`, 'local' for tests;
    # '' = off; zero.remoteplay, docs/az.md 5.3): it plays with the same checkpoint while this one
    # plays its quota, its shards join data/iter_i/; worker processes there (0 = the remote's
    # `procs`), seconds the end of a phase waits for its last shards, and (co-training: the
    # resolver's setting) seconds of a worker's running time after which an episode it has not
    # finished is requeued here
    selfplay_remote='', selfplay_remote_procs=0, selfplay_remote_wait=30.0, selfplay_remote_timeout=3600.0,
    infer_device='cpu', train_device='cpu', seed=0,
    # network batches evaluated concurrently on this many Python threads (CPU: small networks
    # whose ops cannot keep the cores busy; give the driver as many `groups`); 1 = inline
    infer_workers=1,
    # co-training (zero/cotrain.py, docs/az.md 11): the resolver game whose episodes decide this
    # game's deferred terminals ('' = a single-game run: the loop of this module), its settings
    # (a dict of these same settings - game, spec, network, search, training, evaluation; 'preset'
    # names one - saved nested in config.json), the checkpoint its network starts from ('' =
    # fresh; a single-game run's checkpoint works), its episodes in flight in self-play (0 = its
    # `conc`), the backpressure (resolver episodes queued or playing at which this game's
    # self-play waits; 0 = 2 x in flight), and how often this game's evaluation also plays its
    # episodes with the resolver's network (evaluated iterations that are multiples of it; the
    # others play them with the resolver's yardstick; 0 = never), and where those events are played
    # (draft_eval_net_remote: a name in the game package's `remotes`, 'local' for tests; '' = here;
    # zero.remote's `evalnet`, docs/az.md 11.7)
    resolver='', resolver_cfg={}, resolver_init='', resolver_conc=0, pending_max=0, draft_eval_net_every=4,
    draft_eval_net_remote='',
)

PRESETS = {
    # end-to-end check in about a minute (tests, smoke runs)
    'toy-smoke': dict(iters=2, games=48, conc=48, sims=8, sims_fast=4, hidden=32, blocks=1, batch=64,
                      eval_games=16, eval_sims=4, eval_random_games=8, elo_games=8, ref_games=16,
                      min_window_samples=0, threads=2, min_steps=10, lr_warmup_steps=0),
    # the toy proof run (docs/az.md 6): the yardstick is the toy heuristic played through the
    # EXTERNAL seat path (Python, zero/toybots.py), as the JS bot is for Caverna and Glass Road
    'toy-1': dict(iters=24, games=600, conc=300, sims=64, sims_fast=16, p_full=0.25, temp_moves=16,
                  hidden=128, blocks=3, batch=512, reuse=4.0, lr=1e-3, lr_warmup_steps=100,
                  window=6, window_grow=1, min_window_samples=10_000, yardstick='ext:toyheur',
                  eval_games=400, eval_sims=32, eval_random_games=100, elo_games=100, ref_games=1000, threads=4),
    # co-training on the toys (docs/az.md 11): 2v2 toy_team events (games = events resolved per
    # iteration) whose four matches are toy games played by the toy network, both trained here
    'toy-cotrain-smoke': dict(game='toy_team', players=[4], yardstick='greedy', iters=2, games=8, conc=16, sims=8,
                              sims_fast=4, hidden=32, blocks=1, batch=64, eval_games=8, eval_sims=4,
                              eval_random_games=4, elo_games=4, ref_games=8, min_window_samples=0, threads=2,
                              min_steps=10, lr_warmup_steps=0, resolver='toy', resolver_conc=16,
                              resolver_cfg={'preset': 'toy-smoke', 'yardstick': 'heuristic', 'conc': 16,
                                            'eval_games': 8, 'eval_random_games': 4, 'elo_games': 4, 'ref_games': 8}),
}
# The core's toy presets; a game package adds its own (GamePackage.presets: caverna-smoke /
# caverna-2p / caverna-2p-cq, glassroad-smoke / glassroad-2p / glassroad-2p-cq) and may override
# DEFAULTS (GamePackage.defaults). Both dicts are updated in place, so `from zero.run import
# DEFAULTS` sees the package's view.
_CORE_DEFAULTS = dict(DEFAULTS)
_CORE_PRESETS = dict(PRESETS)


def apply_package(pkg):
    """The game package's run presets and DEFAULTS overrides (zero.package.register calls this;
    so does importing this module). Unknown setting names are an error."""
    bad = sorted(set(pkg.defaults) - set(_CORE_DEFAULTS))
    if bad:
        raise ValueError(f'{pkg.name}: unknown run settings in defaults: {bad}')
    for name, p in pkg.presets.items():
        bad = sorted(set(p) - set(_CORE_DEFAULTS))
        if bad:
            raise ValueError(f'{pkg.name}: preset {name!r} has unknown settings {bad}')
    DEFAULTS.clear()
    DEFAULTS.update(_CORE_DEFAULTS)
    DEFAULTS.update(pkg.defaults)
    PRESETS.clear()
    PRESETS.update(_CORE_PRESETS)
    PRESETS.update(pkg.presets)


apply_package(package.active())


def utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                          # noqa: BLE001
            pass


def below_normal():
    """Long jobs share the desktop: yield the CPU."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass


def log(run, msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    with open(os.path.join(run, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def model_path(run, i):
    return os.path.join(run, 'models', f'iter_{i:04d}.pt')


def latest_iter(run):
    its = [int(os.path.basename(p)[5:9]) for p in glob.glob(os.path.join(run, 'models', 'iter_[0-9][0-9][0-9][0-9].pt'))]
    return max(its) if its else None


def read_metrics(run):
    p = os.path.join(run, 'metrics.jsonl')
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_metrics(run, m):
    with open(os.path.join(run, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(m) + '\n')


def lr_at(cfg, it):
    return max(cfg['lr_min'], cfg['lr'] * 0.5 ** ((it - 1) // cfg['lr_halve_every']))


def value_q_at(cfg, it):
    w = cfg['value_q_warmup']
    return cfg['value_q'] * (min(1.0, it / w) if w else 1.0)


def window_iters(cfg, it):
    return min(cfg['window'], 1 + it // 2) if cfg['window_grow'] else cfg['window']


def summarize(rows):
    """rows: (score, margin, rounds, over) of the evaluated player."""
    n = len(rows)
    if not n:
        return {'win': 0.0, 'se': 0.0, 'margin': 0.0, 'games': 0, 'score': 0.0}
    w = sum(r[0] for r in rows) / n
    return {'win': round(w, 4), 'se': round(math.sqrt(max(w * (1 - w), 1e-9) / n), 4),
            'margin': round(sum(r[1] for r in rows) / n, 2), 'games': n, 'score': sum(r[0] for r in rows),
            'rounds': round(sum(r[2] for r in rows) / n, 2), 'unfinished': sum(1 for r in rows if not r[3])}


class Run:
    def __init__(self, run, cfg, pause=None):
        self.run, self.cfg, self.pw = run, cfg, pause
        self.info = G.info(cfg['game'])
        self.yard = cfg['yardstick']
        self.remote = None                         # zero.remote.RemoteEval when eval_remote is set
        self.rsp = None                            # zero.remoteplay.RemotePlay when selfplay_remote is set
        self.roll = None                           # rolling driver + its evaluator
        self.roll_ev = None

    def check(self, where=''):
        if self.pw is not None:
            self.pw.check(where)

    def log(self, msg):
        log(self.run, msg)

    # ---------------------------------------------------------------- models
    def new_model(self):
        c = self.cfg
        torch.manual_seed(c['seed'])
        return make_net(self.info, c['game'], c.get('arch', 'mlp'), c['hidden'], c['blocks'], c['ch'], c.get('heads', 4),
                        c.get('ff', 0))

    def evaluator(self, path):
        m, _ = load(path, self.cfg['infer_device'])
        return NetEvaluator(m, self.cfg['infer_device'], workers=self.cfg.get('infer_workers', 1))

    def nn_seat(self, key, sims, selfplay=False):
        """A network seat at the run's settings: self-play's, or every evaluation's (no noise, no
        temperature; the move rule eval_select)."""
        c = self.cfg
        if selfplay:
            return G.nn_seat(key, sims, c['c_puct'], True, c['alpha'], c['eps'], c['temp_moves'], c['sims_fast'],
                             c['p_full'], c['chance_k'], c['margin_w'], bool(c['redet']),
                             c.get('policy_target', 'visits'), c.get('cq_c_visit', 50.0), c.get('cq_c_scale', 1.0),
                             sims_minor=c.get('sims_minor', 0) or None, select=c.get('selfplay_select', 'visits'),
                             select_min_visits=c.get('selfplay_select_min_visits', 0))
        return G.nn_seat(key, sims, c['c_puct'], chance_k=c['chance_k'], margin_w=c['margin_w'], redet=bool(c['redet']),
                         cq_c_visit=c.get('cq_c_visit', 50.0), cq_c_scale=c.get('cq_c_scale', 1.0),
                         select=c.get('eval_select', 'visits'), select_min_visits=c.get('eval_select_min_visits', 0))

    def eval_tag(self):
        """The name of the evaluation seats' move rule ('visits', 'cq', 'cq-n2' ...): the evaluations'
        'select' and their Elo pool (docs/az.md 5.4)."""
        return elo.eval_tag(self.cfg)

    def driver(self, seats, seed, record=False, conc=None, rolling=False, net=0, groups=None, queue_only=False):
        """A driver for self-play (`record`: the run's spec) or a match (eval_spec, if set).
        `queue_only`: a rolling driver that plays only queued setups (co-training)."""
        c = self.cfg
        spec = c['spec'] if record else (c.get('eval_spec') or package.active().eval_spec_of(c['game'], c['spec']))
        kw = {'queue_only': True} if queue_only else {}
        return G.Driver(c['game'], seats, spec=spec, concurrency=conc or c['conc'], seed=seed, record=record,
                        threads=c['threads'], groups=groups or c['groups'], max_steps=c['max_steps'], rolling=rolling,
                        net=net, **kw)

    def match_groups(self, ext):
        return self.cfg.get('ext_groups', DEFAULTS['ext_groups']) if ext else None

    # ---------------------------------------------------------------- self-play
    def selfplay(self, it):
        c = self.cfg
        d = os.path.join(self.run, 'data', f'iter_{it:04d}')
        if os.path.exists(d) and not (self.rsp is not None and self.rsp.began(it)):
            self.drop_data(d, strict=True)         # an interrupted iteration is replayed
        t0 = time.time()
        seat = self.nn_seat('m', c['sims'], selfplay=True)
        rng = np.random.default_rng(c['seed'] * 1_000_003 + it)
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'], 'iter': it}
        if c['rolling']:
            if self.roll is None:
                table = [[seat] * p for p in c['players']]
                self.roll = self.driver(table, seed=c['seed'] * 7919 + it, record=True, rolling=True, net=it - 1)
                self.roll_ev = self.evaluator(model_path(self.run, it - 1))
            else:
                m, _ = load(model_path(self.run, it - 1), 'cpu')
                self.roll_ev.set_weights(m.state_dict())
            drv, ev = self.roll, self.roll_ev
            st0 = drv.stats()
            drv.resume(c['games'], it - 1)
            prefix = 'r00'
        else:
            counts = rng.choice(c['players'], size=c['games'])
            drv = self.driver([[seat] * int(p) for p in counts], seed=c['seed'] * 1_000_003 + it, record=True)
            ev = self.evaluator(model_path(self.run, it - 1))
            st0 = None
            prefix = 'c00'
        writer = ShardWriter(d, prefix, rows=c['shard_rows'], meta=meta, bad_dir=self.debug_dir())
        results = []
        state = {'n': 0, 't': time.time()}
        prog = os.path.join(self.run, 'progress.json')
        tstats = TargetStats()
        if self.rsp is not None:
            self.rsp.begin(it)                     # the remote self-plays with the same checkpoint

        def take():
            s = drv.take_samples()
            tstats.add(s)
            writer.add(s)

        def on_batch(lb):
            state['n'] += 1
            if state['n'] % 32 == 0:
                take()
                results.extend(drv.take_results())
            if time.time() - state['t'] > 2:
                state['t'] = time.time()
                p = {'iter': it, 'phase': 'selfplay', 'games': len(results), 'target': c['games'], 'time': time.time()}
                if self.rsp is not None:
                    p['remote'] = self.rsp.progress()
                _write_json(prog, p)
        drive(drv, {'m': ev}, check=lambda: self.check('self-play'), on_batch=on_batch)
        take()
        results.extend(drv.take_results())
        ws = writer.close()
        secs = time.time() - t0                    # this machine's phase (the remote's wait not counted)
        rem = None
        if self.rsp is not None:
            _write_json(prog, {'iter': it, 'phase': 'remote shards', 'time': time.time()})
            rem = self.rsp.end(it)                 # at most selfplay_remote_wait s
        _write_json(prog, {'iter': it, 'phase': 'train', 'time': time.time()})
        st = drv.stats()
        fin = [r for r in results if r['over']]
        leaves = st['leaves'] - (st0['leaves'] if st0 else 0)
        out = {'games': len(results), 'unfinished': len(results) - len(fin), 'samples': ws['samples'],
               'shards': ws['shards'], 'bytes': ws['bytes'],
               'decisions_per_game': sum(r['decisions'] for r in results) / max(1, len(results)),
               'steps_per_game': sum(r['steps'] for r in results) / max(1, len(results)),
               'rounds': sum(r['rounds'] for r in fin) / max(1, len(fin)),
               'avg_score': sum(sum(r['scores']) / r['n'] for r in fin) / max(1, len(fin)),
               'leaves': leaves, 'leaves_per_s': leaves / max(secs, 1e-9),
               'games_per_h': len(results) / max(secs, 1e-9) * 3600, 'seconds': secs,
               'infer_s': ev.seconds, 'avg_batch': ev.leaves / max(1, ev.calls),
               'minor_per_game': (st.get('minor', 0) - (st0.get('minor', 0) if st0 else 0)) / max(1, len(results)),
               'target': tstats.summary(c.get('policy_target', 'visits'))}
        ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0
        if c['rolling']:
            inf = drv.in_flight()
            out['rolling'] = {'in_flight': inf['games'], 'restarts': st['restarts'] - st0['restarts'],
                              'restart_sims': st['restart_sims'] - st0['restart_sims'],
                              'dropped_leaves': st['dropped_leaves'] - st0['dropped_leaves']}
        if rem is not None:
            out['remote'] = rem
            self.log(f'iter {it} remote self-play: {_fmt_remote(rem)}')
        return out

    # ---------------------------------------------------------------- training
    def window_shards(self, it):
        dirs = sorted(glob.glob(os.path.join(self.run, 'data', 'iter_*')), reverse=True)
        dirs = [d for d in dirs if int(os.path.basename(d)[5:9]) <= it]
        out, n = [], 0
        for k, d in enumerate(dirs):
            if k >= window_iters(self.cfg, it) and n >= self.cfg['min_window_samples']:
                break
            for p in shard_paths(d):
                out.append(p)
                with np.load(p) as z:
                    n += int(json.loads(str(z['meta']))['rows'])
        return out

    def train(self, it, new_samples):
        """Train checkpoint it - 1 on the window into checkpoint it (and the optimizer state)."""
        model, ost, stats = self.train_model(it, new_samples)
        self.save_trained(it, model, ost)
        return stats()

    def train_model(self, it, new_samples):
        """`train` without saving -> (model, optimizer state, stats(): the metrics dict, timed when
        called). Co-training trains both networks before either is saved (zero/cotrain.py)."""
        c = self.cfg
        dev = c['train_device']
        if dev == 'cuda' and not torch.cuda.is_available():
            dev = 'cpu'
        t0 = time.time()
        model, _ = load(model_path(self.run, it - 1), dev)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['wd'])
        opt_path = os.path.join(self.run, 'models', 'opt_latest.pt')
        fresh = True
        if os.path.exists(opt_path):
            try:
                o = torch.load(opt_path, map_location=dev, weights_only=False)
                if o.get('iter') == it - 1:
                    opt.load_state_dict(o['state'])
                    fresh = False
            except Exception as e:                 # noqa: BLE001
                self.log(f'iter {it}: optimizer state not loaded ({e}); fresh optimizer')
        lr = lr_at(c, it)
        paths = self.window_shards(it)
        bpr = model.features * 2 + 8 * 16 + 3 * MAX_SEATS * 4
        budget = int(c['ram_budget_gb'] * 1e9 / bpr)
        t1 = time.time()
        W = Window(paths, model.features, model.config['actions'], budget_rows=budget, seed=it)
        load_s = time.time() - t1
        steps = max(c['min_steps'], math.ceil(new_samples * c['reuse'] / c['batch']))
        vq = value_q_at(c, it)
        mw = c['margin_loss_w']
        amp = use_amp(model, dev)
        rng = np.random.default_rng(c['seed'] * 31 + it)
        acc = {'policy': 0.0, 'value': 0.0, 'margin': 0.0, 'entropy': 0.0, 'value_acc': 0.0}
        last = min(steps, 200)
        S = model.seats
        for step in range(steps):
            self.check('training')
            g_lr = lr * min(1.0, (step + 1) / c['lr_warmup_steps']) if fresh and c['lr_warmup_steps'] else lr
            for g in opt.param_groups:
                g['lr'] = g_lr
            idx = rng.integers(W.N, size=c['batch'])
            b = W.batch(idx, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                plog, vlog, mp = model(b['x'], b['n_players'])
            plog, vlog, mp = plog.float(), vlog.float(), mp.float()
            logp = F.log_softmax(plog.masked_fill(~b['mask'], -1e9), dim=1)
            lp = -(b['pi'] * logp).sum(1).mean()
            zt = ((1 - vq) * b['z'] + vq * b['q'])[:, :S]
            lv = -(zt * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1).mean()
            present = (torch.arange(S, device=b['x'].device)[None, :] < b['n_players'][:, None]).float()
            lm = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
            loss = lp + lv + mw * lm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # the non-finite guard (docs/az.md 4.1): one device sync per step, before the step
            if not bool(torch.isfinite(torch.stack([lp.detach(), lv.detach(), lm.detach(), gnorm.float()])).all()):
                with torch.no_grad():
                    per = {'policy': -(b['pi'] * logp).sum(1),
                           'value': -(zt * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1),
                           'margin': (((mp - b['sc'][:, :S]) ** 2) * present).sum(1) / 2}
                self.nonfinite_step(it, step, steps, b, idx, W, model, {'policy': lp, 'value': lv, 'margin': lm},
                                    gnorm, per, dict(lr=g_lr, value_q=vq, margin_loss_w=mw, amp=amp, device=dev))
            opt.step()
            if step >= steps - last:
                with torch.no_grad():
                    p = logp.exp()
                    ent = -(p * logp.clamp(min=-30)).sum(1).mean()
                    vacc = (vlog.argmax(1) == b['z'][:, :S].argmax(1)).float().mean()
                acc['policy'] += lp.item() / last
                acc['value'] += lv.item() / last
                acc['margin'] += lm.item() / last
                acc['entropy'] += ent.item() / last
                acc['value_acc'] += vacc.item() / last
        model.eval()
        ost = opt.state_dict()

        def stats():
            secs = time.time() - t0
            return dict({k: round(v, 5) for k, v in acc.items()}, steps=steps, window_samples=W.N, window_loaded=W.loaded,
                        window_shards=len(paths), window_mb=round(W.nbytes / 1e6, 1), load_s=round(load_s, 2), lr=lr,
                        value_q=vq, seconds=secs, samples_per_s=steps * c['batch'] / max(secs - load_s, 1e-9), device=dev)
        return model, ost, stats

    def opt_path(self):
        return os.path.join(self.run, 'models', 'opt_latest.pt')

    def save_trained(self, it, model, ost):
        """Checkpoint `it` and the optimizer state (atomic; NonFinite before either is written)."""
        opt_path = self.opt_path()
        check_state(ost, opt_path)                 # NonFinite before either file is written
        save(model, model_path(self.run, it), {'iter': it})
        tmp = opt_path + '.tmp'
        torch.save({'iter': it, 'state': ost}, tmp)
        os.replace(tmp, opt_path)

    def debug_dir(self):
        return os.path.join(self.run, 'debug')

    def nonfinite_step(self, it, step, steps, b, idx, W, model, terms, gnorm, per, info):
        """Training step `step` of iteration `it` produced a NaN / inf loss term or gradient norm:
        save the batch (the tensors fed to the model, where its rows came from, the per-sample
        losses, the weights before the step) to <run>/debug/nonfinite_iter_NNNN_step_SSSS.pt and
        raise NonFinite before optimizer.step()."""
        terms = {k: float(v) for k, v in terms.items()}
        gn = float(gnorm)
        bad = [k for k, v in terms.items() if not math.isfinite(v)] + ([] if math.isfinite(gn) else ['grad_norm'])
        per = {k: v.detach().float().cpu() for k, v in per.items()}
        rows = np.flatnonzero(~torch.isfinite(torch.stack(list(per.values()))).all(0).numpy())
        ins = [k for k in ('x', 'pi', 'z', 'q', 'sc') if not bool(torch.isfinite(b[k]).all())]
        wbad = [n for n, t in model.state_dict().items() if t.is_floating_point() and not bool(torch.isfinite(t).all())]
        src = W.provenance(idx[rows] if len(rows) else idx)
        prov = '; '.join(f'{os.path.relpath(p, self.run)} rows {head(r, 8)}' for p, r in src[:6])
        if len(src) > 6:
            prov += f'; ... (+{len(src) - 6} more shards)'
        path = os.path.join(self.debug_dir(), f'nonfinite_iter_{it:04d}_step_{step + 1:04d}.pt')
        os.makedirs(self.debug_dir(), exist_ok=True)
        tmp = path + f'.{os.getpid()}.tmp'
        torch.save({'iter': it, 'step': step + 1, 'steps': steps, 'terms': terms, 'grad_norm': gn, 'nonfinite': bad,
                    'batch': {k: v.detach().cpu() for k, v in b.items()}, 'window_rows': idx, 'shards': W.paths,
                    'shard_of_row': np.searchsorted(W.starts, idx, side='right') - 1, 'row_in_shard': W.src_row[idx],
                    'bad_batch_rows': rows, 'per_sample': per, 'model_config': model.config,
                    'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, **info}, tmp)
        os.replace(tmp, path)
        what = (f'{len(rows)} of {len(idx)} batch rows have a non-finite per-sample loss, from: {prov}' if len(rows)
                else f'every per-sample loss is finite (the gradient is not); the batch came from: {prov}')
        raise NonFinite(
            f'iter {it} training step {step + 1} of {steps}: non-finite {", ".join(bad)} before optimizer.step() '
            f'(policy {terms["policy"]:.5g}, value {terms["value"]:.5g}, margin {terms["margin"]:.5g}, '
            f'grad norm {gn:.5g}); batch inputs {"non-finite: " + ", ".join(ins) if ins else "finite"}; weights '
            f'{"non-finite: " + head(wbad, 6) if wbad else "finite"}; {what}. Batch, per-sample losses and '
            f'weights saved to {path}')

    def drop_data(self, d, strict=False):
        """Delete data directory `d`, keeping any quarantined shard in it (*.bad.npz, moved to
        <run>/debug/)."""
        for p in glob.glob(os.path.join(d, '*' + BAD)):
            os.makedirs(self.debug_dir(), exist_ok=True)
            os.replace(p, os.path.join(self.debug_dir(), f'{os.path.basename(d)}_{os.path.basename(p)}'))
        shutil.rmtree(d, ignore_errors=not strict)

    # ---------------------------------------------------------------- evaluation
    def opp_seat(self, opp, sims):
        if opp == 'nn':
            return self.nn_seat('o', sims)
        return G.seat_dict(opp)

    def play_match(self, it, opp, n_games, sims, seed, opp_model=None):
        """Checkpoint `it` against `opp` ('nn' + opp_model, a bot name or 'ext:<name>'), seats
        alternating. -> rows (score, margin, rounds, over) of the checkpoint."""
        if n_games <= 0:
            return []
        P = 2
        a = self.nn_seat('m', sims)
        o = self.opp_seat(opp, sims)
        seats = [[a if s == g % P else o for s in range(P)] for g in range(n_games)]
        evs = {'m': self.evaluator(model_path(self.run, it))}
        if opp == 'nn':
            evs['o'] = self.evaluator(opp_model)
        ext = {}
        if G.is_external(opp):
            ext[opp[4:]] = G.external(opp[4:], seed)
        drv = self.driver(seats, seed=seed, conc=min(self.cfg['conc'], n_games), groups=self.match_groups(ext))
        drive(drv, evs, ext, check=lambda: self.check('evaluation'))
        rows = []
        for r in sorted(drv.take_results(), key=lambda r: r['gid']):
            me = r['gid'] % P
            sc = r['scores']
            rows.append((r['result'][me], sc[me] - max(sc[j] for j in range(len(sc)) if j != me), r['rounds'], r['over']))
        return rows

    def play_matches(self, it, opps, n_games, sims, seed):
        """Checkpoint `it` against each of the checkpoints `opps` (n_games each, seats alternating
        within each match) in ONE driver: every opponent is its own evaluator key, so the
        checkpoint's network batches hold the leaves of all the matches. -> {opp: rows} as
        play_match."""
        P = 2
        a = self.nn_seat('m', sims)
        evs = {'m': self.evaluator(model_path(self.run, it))}
        seats = []
        for t, j in enumerate(opps):
            o = self.nn_seat(f'o{t}', sims)
            evs[f'o{t}'] = self.evaluator(model_path(self.run, j))
            seats += [[a if s == g % P else o for s in range(P)] for g in range(n_games)]
        if not seats:
            return {}
        drv = self.driver(seats, seed=seed, conc=min(self.cfg['conc'], len(seats)))
        drive(drv, evs, check=lambda: self.check('evaluation'))
        out = {j: [] for j in opps}
        for r in sorted(drv.take_results(), key=lambda r: r['gid']):
            t, g = divmod(r['gid'], n_games)
            me = g % P
            sc = r['scores']
            out[opps[t]].append((r['result'][me], sc[me] - max(sc[q] for q in range(len(sc)) if q != me), r['rounds'],
                                 r['over']))
        return out

    def bots_match(self, a, b, n_games, seed):
        """Two non-network players (Elo reference games). -> score of a."""
        seats = [[G.seat_dict(a), G.seat_dict(b)] if g % 2 == 0 else [G.seat_dict(b), G.seat_dict(a)] for g in range(n_games)]
        ext = {}
        for s in (a, b):
            if G.is_external(s):
                ext[s[4:]] = G.external(s[4:], seed)
        drv = self.driver(seats, seed=seed, conc=min(self.cfg['conc'], n_games), groups=self.match_groups(ext))
        drive(drv, {}, ext, check=lambda: self.check('reference games'))
        return sum(r['result'][r['gid'] % 2] for r in drv.take_results())

    def evaluate(self, it, m):
        c = self.cfg
        t0 = time.time()
        tag = self.eval_tag()                      # the move rule, recorded unless the default
        m['eval_yard'] = summarize(self.play_match(it, self.yard, c['eval_games'], c['eval_sims'], seed=7919 * it + 1))
        m['eval_yard']['opp'] = self.yard
        if tag != 'visits':
            m['eval_yard']['select'] = tag
        self.log(f'iter {it} vs {self.yard}: {m["eval_yard"]}')
        if c['eval_random_games'] and self.yard != 'random':
            m['eval_random'] = summarize(self.play_match(it, 'random', c['eval_random_games'], c['eval_sims'], seed=7919 * it + 2))
            if tag != 'visits':
                m['eval_random']['select'] = tag
            self.log(f'iter {it} vs random: {m["eval_random"]}')
        self.extra_evals(it, m)
        m.setdefault('phases', {})['eval_s'] = time.time() - t0

    def extra_evals(self, it, m):
        """The game package's other yardsticks of checkpoint `it` (GamePackage.evals): measured
        and logged under m['extras'][name], never fed back. One that raises does not stop the
        run (a yardstick, not the loop): its error is logged (EXTRA EVAL FAILED) and recorded as
        m['extras'][name] = {'error': ...}."""
        for name, fn in package.active().evals_of(self.cfg['game']).items():
            t0 = time.time()
            try:
                r = fn(self, it)
            except Paused:
                raise
            except Exception as e:                 # noqa: BLE001 - recorded, loudly
                m.setdefault('extras', {})[name] = {'error': f'{type(e).__name__}: {e}'[:500]}
                self.log(f'iter {it} {name}: EXTRA EVAL FAILED: {type(e).__name__}: {e}')
                continue
            if r is None:
                self.log(f'iter {it} {name}: skipped')
                continue
            m.setdefault('extras', {})[name] = dict(r, seconds=round(time.time() - t0, 2))
            self.log(f'iter {it} {name}: {_fmt(r)}')

    def elo_update(self, latest, current=None):
        """The Elo pool of the evaluation rule (docs/az.md 5.4: elo/ for 'visits', elo/<rule>/ for
        another, which holds only that rule's evaluations and plays its Elo matches from the
        iteration it began at, `since`): the evaluations not in it yet, the reference and Elo
        matches it lacks, the refit. -> (ratings, best)."""
        c = self.cfg
        t0 = time.time()
        tag = self.eval_tag()
        res = elo.load_results(self.run, anchor=self.yard, tag=tag)
        if tag != 'visits':
            res.setdefault('select', tag)
            res.setdefault('since', latest)
        if self.yard != 'random' and not elo.has(res, 'random', self.yard) and c['ref_games']:
            s = self.bots_match('random', self.yard, c['ref_games'], seed=17)
            elo.add(res, 'random', self.yard, s, c['ref_games'])
            self.log(f'reference: random vs {self.yard}: {s:.1f}/{c["ref_games"]}')
        logged = res.setdefault('logged', [])
        for m in read_metrics(self.run) + ([current] if current else []):
            name = elo.ckpt_name(m['iter'])
            if name in logged:
                continue
            if m.get('eval_yard') and m['eval_yard'].get('select', 'visits') != tag:
                continue                           # another rule's evaluation: never in this pool
            for k, opp in (('eval_yard', self.yard), ('eval_random', 'random')):
                e = m.get(k)
                if e and e.get('games'):
                    elo.add(res, name, e.get('opp', opp), e['score'], e['games'])
            if m.get('eval_yard'):
                logged.append(name)
                if 'since' in res:
                    res['since'] = min(res['since'], m['iter'])
        elo.save_results(self.run, res, tag=tag)
        e = max(1, int(c['eval_every']))
        since = res.get('since', 0)
        todo = [(k, j) for k in range(e, latest + 1, e) if k >= since
                for j in elo.opponents(k, e, c.get('elo_offsets', ''), c.get('elo_opponents', 0))
                if j >= 0 and os.path.exists(model_path(self.run, j))
                and os.path.exists(model_path(self.run, k))
                and not elo.has(res, elo.ckpt_name(k), elo.ckpt_name(j))]
        sims = c.get('elo_sims') or c['eval_sims']

        def record(k, j, rows):
            sc = sum(r[0] for r in rows)
            elo.add(res, elo.ckpt_name(k), elo.ckpt_name(j), sc, len(rows))
            elo.save_results(self.run, res, tag=tag)
            self.log(f'elo match {elo.ckpt_name(k)} vs {elo.ckpt_name(j)}: {sc:.1f}/{len(rows)}')
        if c.get('elo_together'):
            for k in sorted({k for k, _ in todo}):
                js = [j for kk, j in todo if kk == k]
                for j, rows in self.play_matches(k, js, c['elo_games'], sims, seed=31 + 7919 * k).items():
                    record(k, j, rows)
        else:
            for k, j in todo:
                record(k, j, self.play_match(k, 'nn', c['elo_games'], sims, seed=31 + 7919 * k + 101 * j,
                                             opp_model=model_path(self.run, j)))
        ratings = elo.fit(res, anchor=self.yard)
        elo.write_ratings(self.run, ratings, self.yard, tag=tag)
        # raw offsets rate checkpoints that never played the yardstick: never models/best.pt
        best = elo.tag_checkpoints(self.run, ratings, latest, self.yard, tag=tag,
                                   evaluated=set(logged) if c.get('elo_offsets') else None)
        if current is not None:
            current.setdefault('phases', {})['elo_s'] = time.time() - t0
        return ratings, best

    # ---------------------------------------------------------------- the loop
    def partial_path(self, it):
        return os.path.join(self.run, 'models', f'iter_{it:04d}.json')

    def finish_iter(self, it, m, t0):
        """The checkpoint exists: evaluation, Elo, metrics."""
        c = self.cfg
        _write_json(os.path.join(self.run, 'progress.json'), {'iter': it, 'phase': 'eval', 'time': time.time()})
        evaluated = it % max(1, int(c['eval_every'])) == 0 or it == 0
        if self.remote is None:
            if evaluated:
                self.evaluate(it, m)
            ratings, best = self.elo_update(it, m)
            me = ratings.get(elo.ckpt_name(it))
            if me:
                m['elo'] = elo.row(me, self.eval_tag())
                self.log(f'iter {it} elo {me[0]:.0f} +-{1.96 * me[1]:.0f}{elo.pool_note(self.eval_tag())}; '
                         + (f'best {best["file"]} {best["elo"]:.0f}' if best else 'no best yet'))
        m['seconds'] = time.time() - t0 + m.get('resumed_seconds', 0)
        m['time'] = time.time()
        append_metrics(self.run, m)
        if self.remote is not None and evaluated:
            self.remote.enqueue(it)
        try:
            os.remove(self.partial_path(it))
        except OSError:
            pass
        self.prune(it)

    def prune(self, it):
        for d in glob.glob(os.path.join(self.run, 'data', 'iter_*')):
            if int(os.path.basename(d)[5:9]) <= it - self.cfg['keep_data']:
                self.drop_data(d)                  # never <run>/debug/ (quarantined shards, debug batches)

    def loop(self):
        c = self.cfg
        run = self.run
        start = latest_iter(run)
        if start is None:
            save(self.new_model(), model_path(run, 0), {'iter': 0})
            start = 0
        done = {m['iter'] for m in read_metrics(run)}
        self.log(f'run {run} starting after iter {start}; cfg {json.dumps(c)}')
        if c.get('eval_remote'):
            from .remote import RemoteEval
            self.remote = RemoteEval(self, c['eval_remote'])
            self.remote.backfill(start if start in done else start - 1)
            self.log(f'evaluations on the remote {c["eval_remote"]!r} ({self.remote.spec["host"]})')
        if c.get('selfplay_remote'):
            from .remoteplay import RemotePlay
            self.rsp = RemotePlay(self, c['selfplay_remote'])
            self.log(f'self-play also on the remote {c["selfplay_remote"]!r} ({self.rsp.spec["host"]}, '
                     f'{self.rsp.procs} worker(s), mirror {self.rsp.rdir})')
        try:
            self._iterations(start, done)
        finally:
            if self.rsp is not None:
                self.rsp.close()

    def early_remote(self, nxt):
        """Checkpoint nxt - 1 is written: the remote starts iteration nxt's self-play with it now,
        while this machine evaluates (docs/az.md 5.3: on-policy, the checkpoint this machine's
        phase nxt will use). Not after the last iteration, nor when STOP or PAUSE is there."""
        if self.rsp is not None and nxt <= self.cfg['iters'] and not runctl.stop_requested(self.run) \
                and not runctl.pause_requested(self.run):
            self.rsp.begin(nxt)

    def _iterations(self, start, done):
        c = self.cfg
        run = self.run
        if start not in done:                      # iteration 0, or a checkpoint whose evaluation was interrupted
            t0 = time.time()
            m = {'iter': start}
            if os.path.exists(self.partial_path(start)):
                with open(self.partial_path(start)) as f:
                    m = json.load(f)
                self.log(f'iter {start}: finishing its evaluation (interrupted)')
            self.finish_iter(start, m, t0)
        stop = os.path.join(run, 'STOP')
        for it in range(start + 1, c['iters'] + 1):
            self.check('between iterations')
            if os.path.exists(stop):
                self.log('STOP file found; exiting')
                break
            t0 = time.time()
            sp = self.selfplay(it)
            self.log(f'iter {it} selfplay: {_fmt(sp)}')
            t1 = time.time()
            tr = self.train(it, sp['samples'] + sp.get('remote', {}).get('samples', 0))
            self.log(f'iter {it} train: {_fmt(tr)}')
            m = {'iter': it, 'selfplay': sp, 'train': tr, 'phases': {'selfplay_s': t1 - t0, 'train_s': time.time() - t1},
                 'resumed_seconds': 0}
            _write_json(self.partial_path(it), dict(m, resumed_seconds=time.time() - t0))
            self.early_remote(it + 1)
            self.finish_iter(it, m, t0)
        if self.rsp is not None:
            self.rsp.close()                       # no more self-play: the remote's workers quit
        if self.remote is not None and not os.path.exists(stop):
            _write_json(os.path.join(run, 'progress.json'), {'phase': 'remote evaluations', 'time': time.time()})
            self.remote.drain()
        _write_json(os.path.join(run, 'progress.json'), {'phase': 'idle', 'time': time.time()})


def _fmt(d):
    return json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})


def _fmt_remote(r):
    """The remote's share of a self-play phase in one log line."""
    s = (f"{r['workers']} worker(s), {r['games']} games, {r['samples']} samples in {r['shards']} shards "
         f"({r['bytes'] / 1e6:.1f} MB), {r['leaves_per_s']:.0f} leaves/s ({r['leaves_per_running_s']:.0f} running), "
         f"{r['seconds']:.0f}s ({r['backoff_s']:.0f}s backing off, started {r.get('lead_s', 0):.0f}s before this "
         f"machine's phase), waited {r['wait_s']:.1f}s")
    if 'matches' in r:
        s += (f"; matches {r['matches']} back, {r['assigned']} assigned, {r['outstanding']} outstanding, "
              f"{r['requeued']} + {r['requeued_timeout']} requeued, {r['duplicates_dropped']} duplicates dropped")
    for k in ('late', 'late_prev', 'rejected', 'quarantined', 'fetch_failed', 'session_errors'):
        if r.get(k):
            s += f', {k} {r[k]}'
    return s


def _write_json(path, obj):
    tmp = path + f'.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=f'{package.active().title} AlphaZero loop (docs/az.md)')
    ap.add_argument('--run', required=True)
    for k, v in DEFAULTS.items():
        flag = '--' + k.replace('_', '-')
        if isinstance(v, list):
            ap.add_argument(flag, type=int, nargs='+', default=None)
        elif isinstance(v, dict):
            ap.add_argument(flag, type=json.loads, default=None, help='JSON object')
        elif isinstance(v, bool):
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(v), default=None)
    return ap.parse_args(argv)


def resolve_config(args, saved):
    """DEFAULTS < preset < saved config < flags given now."""
    given = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}
    preset = given.get('preset') or saved.get('preset') or ''
    if preset and preset not in PRESETS:
        raise SystemExit(f'unknown preset {preset!r} ({sorted(PRESETS)})')
    cfg = dict(DEFAULTS)
    cfg.update(PRESETS.get(preset, {}))
    cfg.update(saved)
    cfg.update(given)
    cfg['preset'] = preset
    if cfg.get('policy_target', 'visits') not in ('visits', 'cq'):
        raise SystemExit(f"policy_target {cfg['policy_target']!r}: 'visits' or 'cq'")
    check_eval_settings(cfg)
    return cfg


def check_eval_settings(cfg, where=''):
    """The move-rule and Elo-offset settings (eval_select, selfplay_select and their guards,
    elo_offsets) -> None, else SystemExit (`where`: a prefix, e.g. 'resolver_cfg: ')."""
    from .mcts import check_select
    for k in ('eval_select', 'selfplay_select'):
        try:
            check_select(cfg.get(k, 'visits'), cfg.get(k + '_min_visits', 0), what=k)
        except ValueError as e:
            raise SystemExit(f'{where}{e}') from None
    try:
        elo.parse_offsets(cfg.get('elo_offsets', ''))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None


def main(argv=None):
    utf8_console()
    args = parse(argv)
    run = args.run
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    if runctl.pause_requested(run):
        log(run, f'PAUSE file present: not starting (exit status {EXIT_PAUSED}; remove it to resume: '
                 f'tools/pause_run.py --resume, or the dashboard)')
        runctl.write_state(run, 'run', state='paused', why='PAUSE file present at start')
        raise SystemExit(EXIT_PAUSED)
    cfg_path = os.path.join(run, 'config.json')
    saved = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
    cfg = resolve_config(args, saved)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=1)
    torch.set_num_threads(max(1, int(cfg['torch_threads'])))
    below_normal()
    runctl.write_state(run, 'run', state='running')

    def on_pause():
        runctl.write_state(run, 'run', state='pausing', pause_requested=pw.requested, pause_noticed=pw.seen)
        log(run, f'PAUSE file noticed ({pw.seen - pw.requested:.1f}s after it was created): stopping at the next '
                 f'pause point')
    pw = runctl.PauseWatch(run, on_pause=on_pause)
    if cfg.get('resolver'):
        from .cotrain import CoRun
        R = CoRun(run, cfg, pw)
    else:
        R = Run(run, cfg, pw)
    try:
        R.loop()
    except Paused as e:
        pw.close()
        now = time.time()
        log(run, f'PAUSED ({e}); exit status {EXIT_PAUSED}: the supervisor relaunches the run when PAUSE is removed')
        runctl.write_state(run, 'run', state='paused', pause_requested=pw.requested, pause_noticed=pw.seen,
                           workers_stopped=now, exited=now)
        raise SystemExit(EXIT_PAUSED)
    except BaseException as e:
        pw.close()
        log(run, f'run aborted: {type(e).__name__}: {e}')
        runctl.write_state(run, 'run', state='aborted', error=f'{type(e).__name__}: {e}'[:500])
        raise
    pw.close()
    log(run, 'done')
    runctl.write_state(run, 'run', state='done')


def cli():
    """`python -m zero.run` / `python -m <pkg>.az.run`: main(); a pause exits at once with status
    75 (os._exit: no waiting for non-daemon threads)."""
    try:
        main()
    except SystemExit as e:
        if e.code == EXIT_PAUSED:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(EXIT_PAUSED)
        raise


if __name__ == '__main__':
    cli()
