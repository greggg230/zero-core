"""AlphaZero loop: self-play -> train -> evaluate vs the yardstick -> Elo matches (docs/az.md).

  python -m <pkg>.az.run --run runs/toy1 --preset toy-1      (a game package: grz, cvz, ...)
  python -m zero.run --run runs/toy1 --game toy --iters 40 --games 1000 --threads 4   (the core)

Iteration i self-plays with models/iter_{i-1}.pt (the Rust driver in this process, CPU or
CUDA inference; samples streamed into shards under data/iter_NNNN/), trains on the newest
iterations' data (a RAM-bounded window), writes models/iter_i.pt (atomic), evaluates it
against the yardstick (the run's anchor at Elo 1000: a built-in bot, or an external player
such as 'ext:toyheur' / a game's 'ext:jsbot') and the random bot, plays Elo matches against
the checkpoints i-1, i-2, i-4 ... i-32, refits the ratings and appends a line to
metrics.jsonl. Resumes from the newest checkpoint - unless its network was built for another
encoding than this build's game feeds it: then the run refuses to start (zero.net's encoding
guard, exit status 78 = runctl.EXIT_REFUSED, not restarted by the supervisor; docs/az.md 10.4;
every checkpoint the run loads is checked so). Settings: DEFAULTS (+ the game package's
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

--selfplay-league 0.25: a quarter of the self-play games put a past checkpoint of the run (the pool
--selfplay-league-pool, raw offsets back) in one seat; only the current network's decisions are
recorded (zero/league.py, docs/az.md 14).

--aux-<name>-w 0.5: an auxiliary training target of the game (the game package's `aux`, e.g. MTG Zero's
`belief`, the opponent's hand): self-play records it with every sample, the network gets an auxiliary
head (added to a checkpoint without one, everything else as it was) and training adds its loss x the
weight (zero/auxtarget.py, docs/az.md 15). 0 (the default): the run of before.

--selfplay-belief-tilt 1 / --eval-belief-tilt 1: the root's belief (docs/search.md 3.8, docs/az.md 18):
every full search of that seat asks the network's auxiliary head once for the root's prediction of the
hidden cards and deals its simulations' worlds by it (the game's `determinize_with`; belief_clip,
belief_mix, belief_fast). Needs a checkpoint with the head (aux_<name>_w > 0 in its history): one
without it is refused (net.HeadMissing, exit 78). 0 (the default): the run of before.

--selfplay-remote <name>: another machine (a remote of the game package) self-plays too, with the
same checkpoint: both machines pull the iteration's --games (the total) from one shared queue, each
with at most its throughput's share in flight; its shards come back into data/iter_i/
(zero/remoteplay.py, zero/workqueue.py, docs/az.md 5.3). A co-training run: the resolver's
--resolver-cfg '{"selfplay_remote": ...}' (the remote plays a share of the queued episodes, 11.6).
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import auxtarget as AUX, elo, games as G, heavy as HV, league as L, light as LT, package, runctl
from .data import BAD, ShardWriter, TargetStats, Window, WindowCache, batches, shard_paths
from .finite import NonFinite, check_state, head
from .infer import NetEvaluator, drive, use_amp
from .net import EncodingMismatch, add_aux, aux_layout, check_belief_head, check_checkpoint, load, make_net, save
from .protocol import MAX_SEATS
from .runctl import EXIT_PAUSED, EXIT_REFUSED, Paused

BIG = 1 << 62                                      # a rolling driver's quota when its phases end by end_phase()

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
    # light decisions in self-play (zero/light.py, docs/search.md 3.10, docs/az.md 22): the game's
    # classes of decisions that rarely matter (SearchGame::LIGHT_CLASSES; MTG Zero: its timing
    # windows), selfplay_light_classes of them ('all', or comma-separated names / globs, '-' removes:
    # 'all,-opp_end'), searched with selfplay_sims_light simulations (a full draw) and
    # selfplay_sims_light_fast (a fast draw; 0 = selfplay_sims_light); selfplay_light_record 0 = never
    # recorded (a minor decision's treatment), 1 = full draws recorded with the light search's targets.
    # Self-play seats only (evaluation unchanged); 0 = off: nothing changes. Single-game runs only
    selfplay_sims_light=0, selfplay_sims_light_fast=0, selfplay_light_record=0, selfplay_light_classes='all',
    # heavy decisions in self-play (zero/heavy.py, docs/search.md 3.11, docs/az.md 23): a decision (not
    # minor, not light) where an option of prior <= selfplay_heavy_prior_max leads within a short line
    # (selfplay_heavy_depth deviations, selfplay_heavy_horizon steps, selfplay_heavy_lines lines) to a
    # revelation of hidden information to the decider is searched with selfplay_sims_heavy simulations
    # (a full draw) / selfplay_sims_heavy_fast (a fast draw; required with sims_fast), chance_k
    # selfplay_chance_k_heavy (-1 = chance_k) and a root floor of selfplay_root_min_visits_heavy visits
    # per option (0 = none). Self-play seats only (evaluation unchanged); 0 = off: nothing changes.
    # Single-game runs only
    selfplay_sims_heavy=0, selfplay_sims_heavy_fast=0, selfplay_chance_k_heavy=-1, selfplay_root_min_visits_heavy=0,
    selfplay_heavy_prior_max=1.0, selfplay_heavy_depth=2, selfplay_heavy_horizon=12, selfplay_heavy_lines=64,
    # ... and how a heavy search samples the decider's own revelations: 'sample' (independent worlds) |
    # 'stratified' (their equally likely units without replacement across its simulations; chance_k 0)
    selfplay_chance_mode_heavy='sample',
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
    # min_window_samples; data kept keep_data iterations; RAM budget; samples per shard;
    # window_cache 1 = the window's shards stay in RAM between iterations and each training loads
    # only the new ones (zero/data.py WindowCache: the same window; while it fits the budget)
    window=8, window_grow=1, min_window_samples=20_000, keep_data=12, ram_budget_gb=4.0, shard_rows=65536,
    window_cache=0,
    # train_prefetch n > 0 = training batches built n steps ahead on a background thread while the
    # device trains (zero/data.py batches: the same batches in the same order)
    train_prefetch=0,
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
    # the 'cq' rule's own completed-Q transform (docs/az.md 5.4): eval_select_c_visit / _c_scale for
    # every evaluation seat, selfplay_select_c_visit / _c_scale for self-play's moves after temp_moves.
    # None (unset, the default) = cq_c_visit / cq_c_scale, the training target's (one knob for both,
    # as before these settings); a number = the rule's own, so that a training A/B of cq_c_scale
    # leaves the evaluation's moves (and its Elo pool, named after the rule's transform) alone. The
    # recorded targets never read them; only with the rule 'cq'. On the command line 'none' unsets one.
    eval_select_c_visit=None, eval_select_c_scale=None, selfplay_select_c_visit=None, selfplay_select_c_scale=None,
    # root minimum visits ("forced playouts", docs/search.md 3.5, docs/az.md 5.5): a full search
    # (self-play: the recorded ones, sims; evaluation: every search) whose root has 2 ..
    # root_min_visits_max_actions options gives each at least k visits when its budget allows (2 x
    # options x k <= its simulations), so a near-one-hot prior cannot starve an option of the Q its
    # completed-Q target and rule need. The forced walks come out of the same budget; the visit
    # counts (the 'visits' target, temperature sampling, the 'visits' rule) and the root value
    # target leave them out. selfplay_root_min_visits: self-play's seat; eval_root_min_visits: every
    # evaluation seat (its moves change: the Elo pool's name gets '-f<k>'). 0 = off (the default:
    # today's searches, bit for bit)
    selfplay_root_min_visits=0, eval_root_min_visits=0, root_min_visits_max_actions=4,
    # root-view leaf values (docs/search.md 3.6, docs/az.md 16): 'mover' = a search leaf's value
    # comes from the network on its decider's view (the default: today's searches, bit for bit);
    # 'root' = from the root decider's view of the same (sampled) state - its own hidden cards as
    # they are, the other seats' unknown - in a second, value-only row of the same batch (the priors
    # stay the decider's). selfplay_leaf_value_view: self-play's seat; eval_leaf_value_view: every
    # evaluation seat (its moves change: the Elo pool's name gets '-rv'). selfplay_view_samples: the
    # share of self-play's recorded decisions that also record value-only samples - the real state
    # from each other recording seat's view, in its perspective, no options (no policy loss) - so
    # the value head learns the views root-view leaves ask it for; view_sample_target: their value
    # target's q, 'next_q' (that seat's next search's root value, z if none) | 'z'. 0 = off
    selfplay_leaf_value_view='mover', eval_leaf_value_view='mover', selfplay_view_samples=0.0,
    view_sample_target='next_q',
    # the root's belief (docs/search.md 3.8, docs/az.md 18): a full search of the seat asks the network's
    # auxiliary head once for the root's prediction of the hidden cards (one extra row: the root's own
    # features, the decider's view) and deals every simulation's world by it - the game's
    # `determinize_with`, weights clip((belief / uniform)^tilt, 1/belief_clip, belief_clip) - but a
    # belief_mix share dealt uniformly; belief_fast 1: fast and minor searches too. selfplay_belief_tilt:
    # self-play's seat (a league opponent's too); eval_belief_tilt: every evaluation seat (its moves
    # change: the Elo pool's name gets '-b<tilt>...'). Needs the auxiliary head (a checkpoint without it
    # is refused: net.HeadMissing). 0 = off (the default: today's searches, bit for bit)
    selfplay_belief_tilt=0.0, eval_belief_tilt=0.0, belief_clip=20.0, belief_mix=0.0, belief_fast=0,
    # the root search (docs/search.md 3.7, docs/az.md 19): 'puct' (the default: today's searches, bit
    # for bit) | 'gumbel' = Gumbel AlphaZero's root (Danihelka et al. 2022) - Gumbel noise per root
    # option and sequential halving over the gumbel_m options of highest noise + log-prior, the
    # completed-Q sigma of the 'cq' target; the move is the halving's winner (it replaces root Dirichlet
    # noise, temperature sampling and the move rule *_select; PUCT stays below the root). Every search
    # of the seat is one (full, fast, minor). selfplay_root_search: self-play's seat (needs
    # policy_target 'cq': the target shares its sigma, cq_c_visit / cq_c_scale); its first temp_moves
    # searched decisions of a game get noise of scale selfplay_gumbel_scale (1 = sampling from the
    # improved policy, the paper's), later ones none. eval_root_search: every evaluation seat (sigma =
    # the evaluation rule's transform, eval_select_c_* else cq_c_*; noise eval_gumbel_scale, 0 =
    # deterministic, the default) - its moves change, so its Elo pool is 'gumbel-m<m>...'. Exclusive with
    # the root minimum visits of the same side (sequential halving already visits every considered option)
    selfplay_root_search='puct', eval_root_search='puct', gumbel_m=16, selfplay_gumbel_scale=1.0,
    eval_gumbel_scale=0.0,
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
    # fixed anchors (docs/az.md 20.1): frozen checkpoints ('path' or 'label=path', comma-separated; '' =
    # off: nothing changes) every evaluated checkpoint of the pool plays elo_anchor_games games against
    # (at elo_sims; those of iteration >= elo_anchor_from, 0 = all of the pool's, the older ones caught up
    # first). Their ratings are FIXED: the first anchor's at elo_anchor_elo, each later one's once, from
    # elo_anchor_calib_games games against each anchor before it, kept in the pool's results.json
    # ('anchors': with each file's sha256 - another file under the label is refused). A Bradley-Terry fit
    # over the pool's whole results with them fixed rates every player: 'elo_anchored' in the metrics (and
    # evals.jsonl), the pool's anchored.json, a log line. elo_paired 1: every Elo match (offsets, anchors,
    # calibration) is played as paired games (docs/az.md 20.3: games 2k and 2k + 1 share their deal with
    # the seats swapped; even game counts). Single-game runs only
    elo_anchors='', elo_anchor_games=100, elo_anchor_elo=1000.0, elo_anchor_calib_games=200, elo_anchor_from=0,
    elo_paired=0,
    # held-out losses (docs/az.md 20.2, zero/heldout.py): the shards of a frozen set (comma-separated
    # files or directories; '' = off, nothing changes) on which every evaluated checkpoint's value and
    # policy losses are measured ('heldout' in the metrics, a log line), on heldout_device ('' = the
    # train_device); computed here, also with eval_remote
    heldout_shards='', heldout_device='',
    # evaluation on another machine (a name in the game package's `remotes`, 'local' for tests;
    # '' = here): the loop queues each evaluated checkpoint and keeps training (zero.remote)
    eval_remote='',
    # self-play on another machine too (a name in the game package's `remotes`, 'local' for tests;
    # '' = off; zero.remoteplay, docs/az.md 5.3): both machines pull the iteration's games (`games`
    # in total) from one shared queue with the same checkpoint, its shards join data/iter_i/; worker
    # processes there (0 = the remote's `procs`), seconds the end of a phase waits for its last
    # shards, and (co-training: the resolver's setting) seconds of a worker's running time after
    # which an episode it has not finished is requeued here at the latest
    selfplay_remote='', selfplay_remote_procs=0, selfplay_remote_wait=30.0, selfplay_remote_timeout=3600.0,
    # the shared queue (zero/workqueue.py): each machine's in-flight cap in iterations of its own
    # throughput (its measured share of the iteration's games or matches x this: 1 = a game it starts
    # finishes within about one iteration); co-training: a worker backing off this many seconds
    # gives its outstanding matches back (requeued here), and a match outstanding on a worker longer
    # than this many of its expected latencies is requeued (at least 120 s, at most
    # selfplay_remote_timeout; 0 = the timeout only); linger 1 = the workers keep playing with their
    # checkpoint while the run trains (those games count for the next iteration, one checkpoint stale)
    selfplay_remote_inflight=1.0, selfplay_remote_giveback=30.0, selfplay_remote_overdue=3.0,
    selfplay_remote_linger=0,
    # ... except a batch-bound machine's (its network batches grow with its games in flight, so a
    # smaller cap only starves it: 'auto' = the ones inferring on CUDA / MPS; 'local' | 'remote' |
    # 'all' | 'none'): its slots while its games take at most this many iterations (its throughput
    # share x this; 1 = as the others)
    selfplay_remote_batch='auto', selfplay_remote_batch_inflight=2.0,
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
    # the self-play league (zero/league.py, docs/az.md 14): this share of the self-play games puts a
    # past checkpoint of the run in one seat (0 = off: every game the current network's mirror, as
    # before); the pool it is picked from (raw iteration offsets back from the network that
    # self-plays, those that exist, or 'all'), past checkpoints per iteration (one evaluator key
    # each) and how they are picked ('uniform' | 'pfsp': weighted by how often each still beats the
    # current network). Only the current network's decisions become samples. Single-game runs only
    selfplay_league=0.0, selfplay_league_pool=L.DEFAULT_POOL, selfplay_league_slots=1, selfplay_league_pick='uniform',
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
    so does importing this module), and the weight setting of each of its auxiliary targets
    (`aux_<name>_w`, default 0 = off; docs/az.md 15). Unknown setting names are an error."""
    known = dict(_CORE_DEFAULTS, **pkg.aux_settings())
    bad = sorted(set(pkg.defaults) - set(known))
    if bad:
        raise ValueError(f'{pkg.name}: unknown run settings in defaults: {bad}')
    for name, p in pkg.presets.items():
        bad = sorted(set(p) - set(known))
        if bad:
            raise ValueError(f'{pkg.name}: preset {name!r} has unknown settings {bad}')
    DEFAULTS.clear()
    DEFAULTS.update(known)
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


def opt_state_mismatch(model, state, params=None):
    """An optimizer state dict (AdamW's) that does not fit `model` (or the parameter list `params`)
    -> why (the first parameter whose moments have another shape, or the parameter count), else
    None. torch's own load_state_dict checks the counts only."""
    params = list(model.parameters()) if params is None else list(params)
    groups = state.get('param_groups') or []
    n = sum(len(g.get('params', [])) for g in groups)
    if n != len(params):
        return f'{n} parameters, the network has {len(params)}'
    ids = [i for g in groups for i in g.get('params', [])]
    for i, p in zip(ids, params):
        for k, v in (state.get('state', {}).get(i) or {}).items():
            if torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != tuple(p.shape):
                return f'{k} of parameter {i}: {tuple(v.shape)}, the parameter {tuple(p.shape)}'
    return None


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
        self.wcache = None                         # zero.data.WindowCache with window_cache
        # the self-play league (zero/league.py): None when selfplay_league is 0 (nothing changes)
        self.league = L.League(self) if L.enabled(cfg) else None
        self.roll_kinds = None                     # the rolling driver's table: each entry's league game
        self.roll_evs = None                       # ... and its evaluators by key (m, L0, ...)
        # auxiliary targets (docs/az.md 15): the game's layout [[name, width, loss], ...] when one of
        # their weights is on, else [] (off: nothing recorded, built or trained)
        self.aux_w, self.aux = aux_settings(cfg, self.info)
        self.heldout = None                        # zero.heldout.HeldOut with heldout_shards (loaded once)
        self.anchor_sha = {}                       # anchor path -> ((size, mtime), sha256)
        # light decisions in self-play (zero/light.py): the resolved settings, None when off
        self.light = LT.check(cfg, self.info)
        # heavy decisions in self-play (zero/heavy.py): the resolved settings, None when off
        self.heavy = HV.check(cfg)

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
                        c.get('ff', 0), aux=self.aux or None)

    def evaluator(self, path):
        m, _ = load(path, self.cfg['infer_device'], game=self.cfg['game'])
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
                             select_min_visits=c.get('selfplay_select_min_visits', 0),
                             select_c_visit=c.get('selfplay_select_c_visit'), select_c_scale=c.get('selfplay_select_c_scale'),
                             root_min_visits=c.get('selfplay_root_min_visits', 0),
                             root_min_visits_max_actions=c.get('root_min_visits_max_actions', 4),
                             leaf_value_view=c.get('selfplay_leaf_value_view', 'mover'),
                             view_samples=c.get('selfplay_view_samples', 0.0),
                             view_sample_target=c.get('view_sample_target', 'next_q'),
                             root_search=c.get('selfplay_root_search', 'puct'), gumbel_m=c.get('gumbel_m', 16),
                             gumbel_scale=c.get('selfplay_gumbel_scale', 1.0), **belief_keys(c, 'selfplay'),
                             **LT.seat_keys(c, getattr(self, 'info', None)), **HV.seat_keys(c))
        return G.nn_seat(key, sims, c['c_puct'], chance_k=c['chance_k'], margin_w=c['margin_w'], redet=bool(c['redet']),
                         cq_c_visit=c.get('cq_c_visit', 50.0), cq_c_scale=c.get('cq_c_scale', 1.0),
                         select=c.get('eval_select', 'visits'), select_min_visits=c.get('eval_select_min_visits', 0),
                         select_c_visit=c.get('eval_select_c_visit'), select_c_scale=c.get('eval_select_c_scale'),
                         root_min_visits=c.get('eval_root_min_visits', 0),
                         root_min_visits_max_actions=c.get('root_min_visits_max_actions', 4),
                         leaf_value_view=c.get('eval_leaf_value_view', 'mover'),
                         root_search=c.get('eval_root_search', 'puct'), gumbel_m=c.get('gumbel_m', 16),
                         gumbel_scale=c.get('eval_gumbel_scale', 0.0), **belief_keys(c, 'eval'))

    def check_belief_heads(self, path):
        """With a root belief on (selfplay_ / eval_belief_tilt > 0, docs/az.md 18) the checkpoint at
        `path` must have the auxiliary head for the game's targets: else net.HeadMissing (the run is
        refused, exit 78) - never a search dealing uniformly in silence."""
        c = self.cfg
        on = [k for k in ('selfplay_belief_tilt', 'eval_belief_tilt') if float(c.get(k, 0.0) or 0.0) > 0]
        if on:
            m, _ = load(path, 'cpu', game=c['game'])
            check_belief_head(m, c['game'], f'{" / ".join(on)} (checkpoint {path})')

    def eval_tag(self):
        """The name of the evaluation seats' move rule ('visits', 'cq', 'cq-n2' ...): the evaluations'
        'select' and their Elo pool (docs/az.md 5.4)."""
        return elo.eval_tag(self.cfg)

    def driver(self, seats, seed, record=False, conc=None, rolling=False, net=0, groups=None, queue_only=False,
               paired=False):
        """A driver for self-play (`record`: the run's spec) or a match (eval_spec, if set).
        `queue_only`: a rolling driver that plays only queued setups (co-training). `paired`: games 2k
        and 2k + 1 share their streams (the driver's `paired`, docs/az.md 20.3)."""
        c = self.cfg
        spec = c['spec'] if record else (c.get('eval_spec') or package.active().eval_spec_of(c['game'], c['spec']))
        kw = {'queue_only': True} if queue_only else {}
        if paired:
            kw['paired'] = True
        if record and getattr(self, 'aux', None):
            # self-play records the auxiliary targets (docs/az.md 15; getattr: a caller's stand-in Run
            # with only a cfg, e.g. MTG Zero's test of its evaluation specs, has none)
            kw['aux'] = True
        return G.Driver(c['game'], seats, spec=spec, concurrency=conc or c['conc'], seed=seed, record=record,
                        threads=c['threads'], groups=groups or c['groups'], max_steps=c['max_steps'], rolling=rolling,
                        net=net, **kw)

    def match_groups(self, ext):
        return self.cfg.get('ext_groups', DEFAULTS['ext_groups']) if ext else None

    # ---------------------------------------------------------------- self-play
    def selfplay(self, it):
        if self.rsp is not None:
            return self.selfplay_shared(it)        # with a remote: the shared queue (docs/az.md 5.3)
        c = self.cfg
        d = os.path.join(self.run, 'data', f'iter_{it:04d}')
        if os.path.exists(d):
            self.drop_data(d, strict=True)         # an interrupted iteration is replayed
        t0 = time.time()
        seat = self.nn_seat('m', c['sims'], selfplay=True)
        rng = np.random.default_rng(c['seed'] * 1_000_003 + it)
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'], 'iter': it}
        lg = self.league
        if c['rolling']:
            if self.roll is None:
                if lg is not None:
                    table, self.roll_kinds = L.rolling_table(c, seat)
                else:
                    table = [[seat] * p for p in c['players']]
                self.roll = self.driver(table, seed=c['seed'] * 7919 + it, record=True, rolling=True, net=it - 1)
                self.roll_ev = self.evaluator(model_path(self.run, it - 1))
                self.roll_evs = {'m': self.roll_ev}
            else:
                m, _ = load(model_path(self.run, it - 1), 'cpu', game=c['game'])
                self.roll_ev.set_weights(m.state_dict())
            drv, ev = self.roll, self.roll_ev
            evs, kinds = self.roll_evs, self.roll_kinds
            st0 = drv.stats()
            drv.resume(c['games'], it - 1)
            prefix = 'r00'
        else:
            counts = rng.choice(c['players'], size=c['games'])
            if lg is not None:
                games, kinds = L.table(c, seat, counts)
            else:
                games, kinds = [[seat] * int(p) for p in counts], None
            drv = self.driver(games, seed=c['seed'] * 1_000_003 + it, record=True)
            ev = self.evaluator(model_path(self.run, it - 1))
            evs = {'m': ev}
            st0 = None
            prefix = 'c00'
        if lg is not None:
            plan = lg.plan(it)
            lg.evaluators(it, evs)                 # the slots' past checkpoints (guarded)
        writer = ShardWriter(d, prefix, rows=c['shard_rows'], meta=meta, bad_dir=self.debug_dir())
        results = []
        state = {'n': 0, 't': time.time()}
        prog = os.path.join(self.run, 'progress.json')
        tstats = TargetStats()

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
                _write_json(prog, p)
        drive(drv, evs, check=lambda: self.check('self-play'), on_batch=on_batch)
        take()
        results.extend(drv.take_results())
        ws = writer.close()
        secs = time.time() - t0
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
        self.view_metrics(out, st, st0, leaves)
        self.belief_metrics(out, st, st0, results)
        self.light_metrics(out, results)
        self.heavy_metrics(out, results)
        ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0
        if lg is not None:
            out['league'] = self.league_metrics(it, plan, L.tally(L.League.attribute(results, kinds, plan['slots'])),
                                                None, len(results), evs)
        if c['rolling']:
            inf = drv.in_flight()
            out['rolling'] = {'in_flight': inf['games'], 'restarts': st['restarts'] - st0['restarts'],
                              'restart_sims': st['restart_sims'] - st0['restart_sims'],
                              'dropped_leaves': st['dropped_leaves'] - st0['dropped_leaves']}
        return out

    def view_metrics(self, out, st, st0, leaves):
        """Self-play's root-view metrics (docs/az.md 16), only when a setting is on (the metrics of
        before otherwise): `view` = {rows: value-only rows sent with leaves, rows_per_leaf, samples:
        value-only samples recorded} by this machine's driver."""
        c = self.cfg
        if c.get('selfplay_leaf_value_view', 'mover') == 'mover' and not c.get('selfplay_view_samples'):
            return
        d = lambda k: st.get(k, 0) - ((st0 or {}).get(k, 0))           # noqa: E731
        out['view'] = {'rows': d('view_rows'), 'rows_per_leaf': round(d('view_rows') / max(1, leaves), 4),
                       'samples': d('view_recorded')}

    def heavy_metrics(self, out, results, remote=None):
        """Self-play's heavy-decision metrics (zero/heavy.py), only when they are on (the metrics of
        before otherwise): `heavy` = the phase's finished games' heavy decisions per game, their share of
        the searched decisions, the probed share, recorded, the simulations spent on them and their share,
        the forced walks, the probe's cost - this machine's games and the remote's (`remote`: its tally)."""
        if not getattr(self, 'heavy', None):
            return
        out['heavy'] = HV.summary(HV.merge(HV.tally(results), remote))

    def light_metrics(self, out, results, remote=None):
        """Self-play's light-decision metrics (zero/light.py), only when they are on (the metrics of
        before otherwise): `light` = the phase's finished games' light decisions per game, their share
        of the searched decisions, recorded / unrecorded, simulations taken and saved - this machine's
        games and the remote's (`remote`: its tally, light.tally)."""
        if not getattr(self, 'light', None):
            return
        out['light'] = LT.summary(LT.merge(LT.tally(results), remote))

    def belief_metrics(self, out, st, st0, results):
        """Self-play's root-belief metrics (docs/az.md 18), only when it is on (the metrics of before
        otherwise): `belief` = {rows: root belief rows sent by this machine's driver, per_decision: per
        searched decision of its finished games}."""
        if not float(self.cfg.get('selfplay_belief_tilt', 0.0) or 0.0):
            return
        rows = st.get('belief_rows', 0) - ((st0 or {}).get('belief_rows', 0))
        dec = sum(r.get('decisions', 0) for r in results)
        out['belief'] = {'rows': rows, 'per_decision': round(rows / max(1, dec), 4)}

    def selfplay_shared(self, it):
        """Iteration `it`'s self-play with a remote (docs/az.md 5.3; zero/remoteplay.py,
        zero/workqueue.py): `games` finished games in total, this machine's and the remote's, every
        one searched with checkpoint it - 1. This machine's driver is a rolling `queue_only` one
        (whatever `rolling` says): a setup of the run's spec is queued for every game it may start
        under its cap - its throughput's share of the iteration x selfplay_remote_inflight (a
        batch-bound driver, CUDA / MPS: its slots while its games take at most
        selfplay_remote_batch_inflight iterations) - so its games finish within about an iteration.
        A finished game counts once the shared quota accepts it; the games beyond the quota (finished
        while it filled) wait, with their samples, for the next iteration, whose quota takes them
        first. The phase ends when the quota is met; the games in flight carry over."""
        from .workqueue import lag_hist, split_games
        from .remoteplay import staleness
        c = self.cfg
        rsp = self.rsp
        d = os.path.join(self.run, 'data', f'iter_{it:04d}')
        if os.path.exists(d) and not rsp.began(it):
            self.drop_data(d, strict=True)         # an interrupted iteration is replayed
        t0 = time.time()
        lg = self.league
        if self.roll is None:
            seat = self.nn_seat('m', c['sims'], selfplay=True)
            if lg is not None:
                table, self.roll_kinds = L.rolling_table(c, seat)
            else:
                table = [[seat] * p for p in c['players']]
            self.roll = self.driver(table, seed=c['seed'] * 7919 + it, record=True, rolling=True, net=it - 1,
                                    queue_only=True)
            self.roll_ev = self.evaluator(model_path(self.run, it - 1))
            self.roll_evs = {'m': self.roll_ev}
            self.held, self.enq, self.taken = deque(), 0, 0
        else:
            m, _ = load(model_path(self.run, it - 1), 'cpu', game=c['game'])
            self.roll_ev.set_weights(m.state_dict())
        drv, ev, evs = self.roll, self.roll_ev, self.roll_evs
        if lg is not None:
            plan = lg.plan(it)
            lg.evaluators(it, evs)                 # the slots' past checkpoints (guarded)
        st0 = drv.stats()
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'], 'iter': it}
        writer = ShardWriter(d, 'r00', rows=c['shard_rows'], meta=meta, bad_dir=self.debug_dir())
        tstats = TargetStats()
        spec = G.spec_json(c['spec'])
        rsp.begin(it)                              # (a no-op if the remote began it early)
        prog = os.path.join(self.run, 'progress.json')
        acc = {'results': [], 'stale': {}, 'finished': 0, 'util': [0.0, 0], 't': 0.0, 'league': {}}

        def accept():
            """The held games, oldest first, as many as the quota takes."""
            a = rsp.accept_local(it, len(self.held))
            for _ in range(a):
                row, sd = self.held.popleft()
                acc['results'].append(row)
                if lg is not None:
                    L.tally([row], acc['league'])
                if sd is not None:
                    tstats.add(sd)
                    writer.add(sd)
                    lag_hist(sd['net'], it, acc['stale'])

        def pump(lb=None):
            rows = drv.take_results()
            s = drv.take_samples()
            if rows and lg is not None:
                L.League.attribute(rows, self.roll_kinds, plan['slots'])   # (the slots' checkpoints now)
            if rows:
                self.taken += len(rows)
                acc['finished'] += len(rows)
                self.held.extend(split_games(rows, s))
            if self.held:
                accept()
            if rsp.need(it) <= 0:
                drv.end_phase()                    # the quota is met: the games in flight carry over
            else:
                cap = rsp.local_cap()
                for _ in range(max(0, cap - (self.enq - self.taken))):
                    drv.enqueue(f'L{self.enq}', spec)
                    self.enq += 1
                acc['util'][0] += min(1.0, (self.enq - self.taken) / max(1, cap))
                acc['util'][1] += 1
            now = time.time()
            if now - acc['t'] > 2:
                acc['t'] = now
                rp = rsp.progress()
                q = rp.get('quota') or {}
                _write_json(prog, {'iter': it, 'phase': 'selfplay', 'games': q.get('accepted', len(acc['results'])),
                                   'target': c['games'], 'local': len(acc['results']), 'remote': rp, 'time': now})

        accept()                                   # the last iteration's leftovers come first
        while rsp.need(it) > 0:
            self.check('self-play')
            if drv.paused:
                drv.resume(BIG, it - 1)            # (again if a lost worker's games reopened the quota)
            pump()
            drive(drv, evs, check=lambda: self.check('self-play'), on_batch=pump)
            pump()
            if rsp.need(it) > 0 and drv.in_flight()['games'] == 0 and not drv.queued:
                time.sleep(0.05)                   # nothing to play (a cap of 0): wait for the quota
        if not drv.paused:
            drv.end_phase()
            drive(drv, evs, check=lambda: self.check('self-play'), on_batch=lambda lb: None)
        rows = drv.take_results()                  # finished while the phase ended: the next iteration's
        if rows and lg is not None:
            L.League.attribute(rows, self.roll_kinds, plan['slots'])
        if rows:
            self.taken += len(rows)
            acc['finished'] += len(rows)
            self.held.extend(split_games(rows, drv.take_samples()))
        ws = writer.close()
        secs = time.time() - t0                    # this machine's phase (the remote's wait not counted)
        _write_json(prog, {'iter': it, 'phase': 'remote shards', 'time': time.time()})
        util = acc['util'][0] / acc['util'][1] if acc['util'][1] else None
        rem = rsp.end(it, local={'units': acc['finished'], 'run_s': secs, 'util': util,
                                 'leaves': drv.stats()['leaves'] - st0['leaves']})
        _write_json(prog, {'iter': it, 'phase': 'train', 'time': time.time()})
        results = acc['results']
        st = drv.stats()
        fin = [r for r in results if r['over']]
        leaves = st['leaves'] - st0['leaves']
        out = {'games': len(results), 'unfinished': len(results) - len(fin), 'samples': ws['samples'],
               'shards': ws['shards'], 'bytes': ws['bytes'],
               'decisions_per_game': sum(r['decisions'] for r in results) / max(1, len(results)),
               'steps_per_game': sum(r['steps'] for r in results) / max(1, len(results)),
               'rounds': sum(r['rounds'] for r in fin) / max(1, len(fin)),
               'avg_score': sum(sum(r['scores']) / r['n'] for r in fin) / max(1, len(fin)),
               'leaves': leaves, 'leaves_per_s': leaves / max(secs, 1e-9),
               'games_per_h': len(results) / max(secs, 1e-9) * 3600, 'seconds': secs,
               'infer_s': ev.seconds, 'avg_batch': ev.leaves / max(1, ev.calls),
               'minor_per_game': (st.get('minor', 0) - st0.get('minor', 0)) / max(1, len(results)),
               'target': tstats.summary(c.get('policy_target', 'visits'))}
        self.view_metrics(out, st, st0, leaves)
        self.belief_metrics(out, st, st0, results)
        self.light_metrics(out, results, rem.pop('light', None))
        self.heavy_metrics(out, results, rem.pop('heavy', None))
        ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0
        out['rolling'] = {'in_flight': drv.in_flight()['games'], 'restarts': st['restarts'] - st0['restarts'],
                          'restart_sims': st['restart_sims'] - st0['restart_sims'],
                          'dropped_leaves': st['dropped_leaves'] - st0['dropped_leaves']}
        qb = rem.pop('queue')
        total = len(results) + rem['games']
        cons = qb['consumers']
        cons.setdefault('local', {})['games'] = len(results)
        for n, v in cons.items():
            if n != 'local':
                v['games'] = (qb.get('quota') or {}).get('by', {}).get(n, 0)
        for v in cons.values():
            v['share'] = round(v.get('games', 0) / max(1, total), 4)
        rem_league = rem.pop('league', None)
        out['remote'] = rem
        if lg is not None:
            out['league'] = self.league_metrics(it, plan, acc['league'], rem_league, total, evs)
        out['queue'] = {'quota': c['games'], 'games': total, 'short': max(0, c['games'] - total),
                        'held': {'local': len(self.held), 'remote': qb.get('held_remote', 0)},
                        'inflight': qb['inflight'], 'batch_inflight': qb['batch_inflight'], 'units': qb['units'],
                        'consumers': cons, 'rates': qb['rates'], 'staleness': staleness(acc['stale'], qb['stale'])}
        self.log(f'iter {it} remote self-play: {_fmt_remote(rem)}')
        self.log(f'iter {it} shared queue: {_fmt_queue(out["queue"])}')
        return out

    def league_metrics(self, it, plan, local, remote, games, evs):
        """The iteration's `selfplay.league` (zero/league.py: the share, the current network's score
        against the past checkpoints, which ones), logged; the slots' inference time."""
        lg = L.metrics(plan, local, remote, games, self.league.pick)
        lg['infer_s'] = round(sum(evs[k].seconds for k in self.league.keys() if k in evs), 2)
        for k in self.league.keys():
            if k in evs:
                evs[k].seconds, evs[k].calls, evs[k].leaves = 0.0, 0, 0
        self.log(f'iter {it} league: {L.fmt(lg)}')
        return lg

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
        model, _ = load(model_path(self.run, it - 1), dev, game=c['game'])
        added = False
        if self.aux and aux_layout(model.config.get('aux')) != self.aux:
            if model.config.get('aux'):
                raise SystemExit(f'iter {it}: checkpoint {it - 1} has the auxiliary head {model.config["aux"]}, this run '
                                 f'trains {self.aux} (net.strip_aux, then fork the run)')
            # a checkpoint without the head (a fork of a run without it): a fresh head, the rest as it was
            model = add_aux(model, self.aux, seed=c['seed'] * 7919 + it)
            added = True
            self.log(f'iter {it}: auxiliary head {self.aux} added to checkpoint {it - 1} (its other outputs unchanged)')
        model.train()
        heads = [p for n, p in model.named_parameters() if n.startswith('aux_')]
        trunk = [p for n, p in model.named_parameters() if not n.startswith('aux_')]
        if heads:                                  # two groups: the network of before, its auxiliary head
            opt = torch.optim.AdamW([{'params': trunk}, {'params': heads}], lr=c['lr'], weight_decay=c['wd'])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['wd'])
        opt_path = os.path.join(self.run, 'models', 'opt_latest.pt')
        fresh = True
        if os.path.exists(opt_path):
            try:
                o = torch.load(opt_path, map_location=dev, weights_only=False)
                if o.get('iter') == it - 1:
                    groups = o['state'].get('param_groups') or []
                    if heads and len(groups) == 1 and len(groups[0].get('params', [])) == len(trunk):
                        # the state of the network before its head: kept for it, the head starts fresh
                        bad = opt_state_mismatch(model, o['state'], params=trunk)
                        if bad:
                            raise ValueError(f'it belongs to another network ({bad})')
                        opt = torch.optim.AdamW(trunk, lr=c['lr'], weight_decay=c['wd'])
                        opt.load_state_dict(o['state'])
                        opt.add_param_group({'params': heads})
                    else:
                        # (the state of this network: its moments as its parameters; another network's,
                        # e.g. of another encoding, would load and fail at the first step, docs/az.md 10.4)
                        bad = opt_state_mismatch(model, o['state'])
                        if bad:
                            raise ValueError(f'it belongs to another network ({bad})')
                        opt.load_state_dict(o['state'])
                    fresh = False
            except Exception as e:                 # noqa: BLE001
                self.log(f'iter {it}: optimizer state not loaded ({e}); fresh optimizer')
        lr = lr_at(c, it)
        paths = self.window_shards(it)
        aw = sum(w for _, w, _ in self.aux)
        bpr = model.features * 2 + 8 * 16 + 3 * MAX_SEATS * 4 + (2 * aw + 1 if aw else 0)
        budget = int(c['ram_budget_gb'] * 1e9 / bpr)
        t1 = time.time()
        if c.get('window_cache') and self.wcache is None:
            self.wcache = WindowCache()
        kw = {'aux_width': aw} if aw else {}
        W = Window(paths, model.features, model.config['actions'], budget_rows=budget, seed=it,
                   cache=self.wcache if c.get('window_cache') else None, **kw)
        load_s = time.time() - t1
        steps = max(c['min_steps'], math.ceil(new_samples * c['reuse'] / c['batch']))
        # value-only samples (no options, docs/az.md 16): no policy loss; the policy loss and the
        # entropy are means over the decisions of a batch. A window without them: the losses of before
        nview = int(np.count_nonzero(np.diff(W.opt_off) == 0))
        vq = value_q_at(c, it)
        mw = c['margin_loss_w']
        amp = use_amp(model, dev)
        rng = np.random.default_rng(c['seed'] * 31 + it)
        acc = {'policy': 0.0, 'value': 0.0, 'margin': 0.0, 'entropy': 0.0, 'value_acc': 0.0}
        if aw:
            for name, _, loss in self.aux:
                acc[f'aux_{name}'] = 0.0
                if loss == 'multinomial':
                    acc[f'aux_{name}_hit'] = 0.0
        if nview:
            acc.update(value_view=0.0, value_decision=0.0)
        last = min(steps, 200)
        S = model.seats
        feed = batches(W, rng, steps, c['batch'], dev, c.get('train_prefetch', 0))
        for step in range(steps):
            self.check('training')
            g_lr = lr * min(1.0, (step + 1) / c['lr_warmup_steps']) if fresh and c['lr_warmup_steps'] else lr
            for g in opt.param_groups:
                g['lr'] = g_lr
            idx, b = next(feed)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                if aw:
                    plog, vlog, mp, ap = model(b['x'], b['n_players'], aux=True)
                else:
                    plog, vlog, mp = model(b['x'], b['n_players'])
            plog, vlog, mp = plog.float(), vlog.float(), mp.float()
            logp = F.log_softmax(plog.masked_fill(~b['mask'], -1e9), dim=1)
            if nview:
                dec = b['mask'].any(1).float()                         # rows with options: decisions
                lp = (-(b['pi'] * logp).sum(1) * dec).sum() / dec.sum().clamp(min=1.0)
            else:
                lp = -(b['pi'] * logp).sum(1).mean()
            zt = ((1 - vq) * b['z'] + vq * b['q'])[:, :S]
            lv = -(zt * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1).mean()
            present = (torch.arange(S, device=b['x'].device)[None, :] < b['n_players'][:, None]).float()
            lm = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
            loss = lp + lv + mw * lm
            terms = [lp.detach(), lv.detach(), lm.detach()]
            if aw:
                la = AUX.losses(self.aux, ap.float(), b['aux'], b['aux_mask'])
                for name, v in la.items():
                    if self.aux_w[name] > 0:
                        loss = loss + self.aux_w[name] * v['loss']
                    terms.append(v['loss'].detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # the non-finite guard (docs/az.md 4.1): one device sync per step, before the step
            if not bool(torch.isfinite(torch.stack(terms + [gnorm.float()])).all()):
                with torch.no_grad():
                    per = {'policy': -(b['pi'] * logp).sum(1),
                           'value': -(zt * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1),
                           'margin': (((mp - b['sc'][:, :S]) ** 2) * present).sum(1) / 2}
                named = {'policy': lp, 'value': lv, 'margin': lm}
                if aw:
                    named.update({f'aux_{name}': v['loss'] for name, v in la.items()})
                self.nonfinite_step(it, step, steps, b, idx, W, model, named,
                                    gnorm, per, dict(lr=g_lr, value_q=vq, margin_loss_w=mw, amp=amp, device=dev))
            opt.step()
            if step >= steps - last:
                with torch.no_grad():
                    p = logp.exp()
                    if nview:
                        ent = (-(p * logp.clamp(min=-30)).sum(1) * dec).sum() / dec.sum().clamp(min=1.0)
                        lvr = -(zt * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1)
                        acc['value_view'] += ((lvr * (1 - dec)).sum() / (1 - dec).sum().clamp(min=1.0)).item() / last
                        acc['value_decision'] += ((lvr * dec).sum() / dec.sum().clamp(min=1.0)).item() / last
                    else:
                        ent = -(p * logp.clamp(min=-30)).sum(1).mean()
                    vacc = (vlog.argmax(1) == b['z'][:, :S].argmax(1)).float().mean()
                acc['policy'] += lp.item() / last
                acc['value'] += lv.item() / last
                acc['margin'] += lm.item() / last
                acc['entropy'] += ent.item() / last
                acc['value_acc'] += vacc.item() / last
                if aw:
                    for name, v in la.items():
                        acc[f'aux_{name}'] += v['loss'].item() / last
                        if 'hit' in v:
                            acc[f'aux_{name}_hit'] += v['hit'].item() / last
        feed.close()
        model.eval()
        ost = opt.state_dict()

        def stats():
            secs = time.time() - t0
            out = dict({k: round(v, 5) for k, v in acc.items()}, steps=steps, window_samples=W.N, window_loaded=W.loaded,
                       window_reused=W.reused,
                       window_shards=len(paths), window_mb=round(W.nbytes / 1e6, 1), load_s=round(load_s, 2), lr=lr,
                       value_q=vq, seconds=secs, samples_per_s=steps * c['batch'] / max(secs - load_s, 1e-9), device=dev)
            if nview:
                out['window_view_samples'] = nview
            if aw:
                out['aux'] = dict(AUX.window_stats(W, self.aux), weights=dict(self.aux_w))
                if added:
                    out['aux']['head_added'] = True
            return out
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

    def play_match(self, it, opp, n_games, sims, seed, opp_model=None, paired=False, model=None):
        """Checkpoint `it` against `opp` ('nn' + opp_model, a bot name or 'ext:<name>'), seats
        alternating. -> rows (score, margin, rounds, over) of the checkpoint. `paired`: paired games
        (docs/az.md 20.3); `model`: another network than checkpoint `it` (a path: an anchor's
        calibration, docs/az.md 20.1)."""
        if n_games <= 0:
            return []
        P = 2
        a = self.nn_seat('m', sims)
        o = self.opp_seat(opp, sims)
        seats = [[a if s == g % P else o for s in range(P)] for g in range(n_games)]
        evs = {'m': self.evaluator(model or model_path(self.run, it))}
        if opp == 'nn':
            evs['o'] = self.evaluator(opp_model)
        ext = {}
        if G.is_external(opp):
            ext[opp[4:]] = G.external(opp[4:], seed)
        drv = self.driver(seats, seed=seed, conc=min(self.cfg['conc'], n_games), groups=self.match_groups(ext),
                          paired=paired)
        drive(drv, evs, ext, check=lambda: self.check('evaluation'))
        rows = []
        for r in sorted(drv.take_results(), key=lambda r: r['gid']):
            me = r['gid'] % P
            sc = r['scores']
            rows.append((r['result'][me], sc[me] - max(sc[j] for j in range(len(sc)) if j != me), r['rounds'], r['over']))
        return rows

    def play_matches(self, it, opps, n_games, sims, seed, paired=False, paths=None):
        """Checkpoint `it` against each of the checkpoints `opps` (n_games each, seats alternating
        within each match) in ONE driver: every opponent is its own evaluator key, so the
        checkpoint's network batches hold the leaves of all the matches. -> {opp: rows} as
        play_match. `paired`: paired games (n_games even: a pair never spans two opponents);
        `paths`: {opp: its network's path} for opponents that are not checkpoints of the run (the
        anchors, docs/az.md 20.1)."""
        P = 2
        a = self.nn_seat('m', sims)
        evs = {'m': self.evaluator(model_path(self.run, it))}
        seats = []
        for t, j in enumerate(opps):
            o = self.nn_seat(f'o{t}', sims)
            evs[f'o{t}'] = self.evaluator(paths[j] if paths and j in paths else model_path(self.run, j))
            seats += [[a if s == g % P else o for s in range(P)] for g in range(n_games)]
        if not seats:
            return {}
        drv = self.driver(seats, seed=seed, conc=min(self.cfg['conc'], len(seats)), paired=paired)
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
        pk = {'paired': True} if c.get('elo_paired') else {}     # (docs/az.md 20.3; off: the calls of before)

        def record(k, j, rows):
            sc = sum(r[0] for r in rows)
            elo.add(res, elo.ckpt_name(k), elo.ckpt_name(j), sc, len(rows))
            elo.save_results(self.run, res, tag=tag)
            self.log(f'elo match {elo.ckpt_name(k)} vs {elo.ckpt_name(j)}: {sc:.1f}/{len(rows)}{elo.pool_note(tag)}')
        if c.get('elo_together'):
            for k in sorted({k for k, _ in todo}):
                js = [j for kk, j in todo if kk == k]
                for j, rows in self.play_matches(k, js, c['elo_games'], sims, seed=31 + 7919 * k, **pk).items():
                    record(k, j, rows)
        else:
            for k, j in todo:
                record(k, j, self.play_match(k, 'nn', c['elo_games'], sims, seed=31 + 7919 * k + 101 * j,
                                             opp_model=model_path(self.run, j), **pk))
        anchors = elo.parse_anchors(c.get('elo_anchors', ''))
        if anchors:
            self.anchor_matches(res, anchors, latest, tag, sims, pk)
        ratings = elo.fit(res, anchor=self.yard)
        # raw offsets, a rule's pool begun mid-run and a run back at a rule after another rate checkpoints
        # that never played the yardstick in this pool: flagged "evaluated": false in ratings.json and
        # models/index.json, never models/best.pt (None, a default run's pool: every rated one is evaluated)
        evaluated = set(logged) if c.get('elo_offsets') or tag != 'visits' or elo.other_pools(self.run, tag) else None
        elo.write_ratings(self.run, ratings, self.yard, tag=tag, evaluated=evaluated)
        best = elo.tag_checkpoints(self.run, ratings, latest, self.yard, tag=tag, evaluated=evaluated)
        if anchors:
            # the fit with the anchors fixed (docs/az.md 20.1): every player of the pool, the yardstick too
            fixed = {n: a['elo'] for n, a in res['anchors'].items() if a.get('elo') is not None}
            ar = elo.fit_fixed(res, fixed)
            elo.write_anchored(self.run, ar, res['anchors'], tag)
            me = ar.get(elo.ckpt_name(latest))
            if current is not None and me and me[2]:
                current['elo_anchored'] = elo.row(me, tag)
        if current is not None:
            current.setdefault('phases', {})['elo_s'] = time.time() - t0
        return ratings, best

    # ---------------------------------------------------------------- fixed anchors (docs/az.md 20.1)
    def anchor_file_sha(self, path):
        """sha256 of an anchor's file (cached by size and mtime)."""
        import hashlib
        st = os.stat(path)
        key = (st.st_size, st.st_mtime_ns)
        got = self.anchor_sha.get(path)
        if got and got[0] == key:
            return got[1]
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        self.anchor_sha[path] = (key, h.hexdigest())
        return h.hexdigest()

    def check_anchors(self):
        """Every anchor of elo_anchors exists and is a network of this build's encoding (with the
        auxiliary head when the evaluation's root belief needs it): else AnchorRefused / EncodingMismatch
        (the run is refused, not restarted)."""
        for label, path in elo.parse_anchors(self.cfg.get('elo_anchors', '')):
            if not os.path.exists(path):
                raise AnchorRefused(f'elo_anchors: the anchor {label!r} ({path}) does not exist')
            check_checkpoint(path, self.cfg['game'])
            if float(self.cfg.get('eval_belief_tilt', 0.0) or 0.0) > 0:
                m, _ = load(path, 'cpu', game=self.cfg['game'])
                check_belief_head(m, self.cfg['game'], f'eval_belief_tilt (the anchor {label!r}, {path})')

    def anchor_matches(self, res, anchors, latest, tag, sims, pk):
        """The anchors' part of `elo_update` (docs/az.md 20.1): each anchor's file checked against the
        sha256 it was fixed with; a new anchor fixed (the pool's first at elo_anchor_elo, a later one by
        elo_anchor_calib_games games against each anchor fixed before it, then a fit of those games
        alone with the earlier ones fixed); then the anchor games of every evaluated checkpoint
        (iteration >= the pool's `since` and elo_anchor_from) that lacks them, elo_anchor_games each, all
        of a checkpoint's anchors in one driver. Everything recorded in `res` (saved after each match)."""
        c = self.cfg
        known = res.setdefault('anchors', {})
        named = [(elo.anchor_name(lb), lb, path) for lb, path in anchors]
        for name, lb, path in named:
            if not os.path.exists(path):
                raise AnchorRefused(f'elo_anchors: the anchor {lb!r} ({path}) does not exist')
            sha = self.anchor_file_sha(path)
            if name in known and known[name].get('sha256') not in (None, sha):
                raise AnchorRefused(f'elo_anchors: the anchor {lb!r} ({path}, sha256 {sha[:12]}) is not the network '
                                    f'it was fixed with in this pool ({known[name]["sha256"][:12]}, '
                                    f'{known[name].get("file")}): another file under the same label would move the '
                                    f'scale - give it another label')
        save = lambda: elo.save_results(self.run, res, tag=tag)
        for i, (name, lb, path) in enumerate(named):
            if known.get(name, {}).get('elo') is not None:
                continue
            entry = {'label': lb, 'file': os.path.basename(path), 'sha256': self.anchor_file_sha(path), 'since': latest}
            fixed = {n: a['elo'] for n, a in known.items() if a.get('elo') is not None}
            if not fixed:
                entry.update(elo=float(c['elo_anchor_elo']), how='set (elo_anchor_elo)')
            else:
                ng = int(c.get('elo_anchor_calib_games', 0))
                if ng <= 0:
                    raise SystemExit(f'elo_anchors: the anchor {lb!r} is new and the pool has fixed anchors already: '
                                     f'it needs elo_anchor_calib_games > 0 to be fixed against them')
                others = [n for n in fixed]
                paths = {n: known[n].get('path') for n in others}
                for n, _, p in named:
                    paths[n] = p
                keys = []
                for j, other in enumerate(sorted(others)):
                    op = paths.get(other)
                    if not elo.has(res, name, other):
                        if not op or not os.path.exists(op):
                            raise AnchorRefused(f'elo_anchors: calibrating {lb!r} needs the fixed anchor {other!r}, '
                                                f'which is not among elo_anchors (its file is needed)')
                        rows = self.play_match(None, 'nn', ng, sims, seed=97 + 1013 * (i + 1) + 31 * j, opp_model=op,
                                               model=path, **pk)
                        sc = sum(r[0] for r in rows)
                        elo.add(res, name, other, sc, len(rows))
                        save()
                        self.log(f'elo anchor calibration {name} vs {other}: {sc:.1f}/{len(rows)}{elo.pool_note(tag)}')
                    keys.append(f'{min(name, other)}|{max(name, other)}')
                fit = elo.fit_fixed(res, fixed, pairs=keys)
                entry.update(elo=round(fit[name][0], 1), se=round(fit[name][1], 1),
                             how=f'calibrated: {ng} games against each of {sorted(others)}')
            entry['path'] = path
            known[name] = entry
            save()
            self.log(f'elo anchor {name} fixed at {entry["elo"]:.1f} ({entry["how"]}){elo.pool_note(tag)}')
        for name, lb, path in named:                # (the path it is played from now: a remote mirror's)
            known[name]['path'] = path
        e = max(1, int(c['eval_every']))
        since = max(res.get('since', 0), int(c.get('elo_anchor_from', 0) or 0))
        ng = int(c.get('elo_anchor_games', 0))
        if ng <= 0:
            return
        for k in range(e, latest + 1, e):
            if k < since or not os.path.exists(model_path(self.run, k)):
                continue
            todo = [(name, path) for name, _, path in named if not elo.has(res, elo.ckpt_name(k), name)]
            if not todo:
                continue
            got = self.play_matches(k, [n for n, _ in todo], ng, sims, seed=53 + 7919 * k, paths=dict(todo), **pk)
            for name, rows in got.items():
                sc = sum(r[0] for r in rows)
                elo.add(res, elo.ckpt_name(k), name, sc, len(rows))
                save()
                self.log(f'elo match {elo.ckpt_name(k)} vs {name}: {sc:.1f}/{len(rows)}{elo.pool_note(tag)}')

    def heldout_eval(self, it, m):
        """The held-out losses of checkpoint `it` (docs/az.md 20.2) -> m['heldout'] + a log line. A
        failure is logged (HELDOUT FAILED) and recorded, never the run's end (a yardstick)."""
        from .heldout import HeldOut, fmt
        c = self.cfg
        dev = c.get('heldout_device') or c['train_device']
        if dev == 'cuda' and not torch.cuda.is_available():
            dev = 'cpu'
        try:
            if self.heldout is None:
                self.heldout = HeldOut(c['heldout_shards'], game=c['game'])
            model, _ = load(model_path(self.run, it), dev, game=c['game'])
            m['heldout'] = self.heldout.evaluate(model, dev)
            del model
        except Paused:
            raise
        except Exception as e:                     # noqa: BLE001 - recorded, loudly
            m['heldout'] = {'error': f'{type(e).__name__}: {e}'[:500]}
            self.log(f'iter {it} heldout: HELDOUT FAILED: {type(e).__name__}: {e}')
            return
        self.log(f'iter {it} heldout {fmt(m["heldout"])}')

    # ---------------------------------------------------------------- the loop
    def partial_path(self, it):
        return os.path.join(self.run, 'models', f'iter_{it:04d}.json')

    def finish_iter(self, it, m, t0):
        """The checkpoint exists: evaluation, Elo, metrics."""
        c = self.cfg
        _write_json(os.path.join(self.run, 'progress.json'), {'iter': it, 'phase': 'eval', 'time': time.time()})
        evaluated = it % max(1, int(c['eval_every'])) == 0 or it == 0
        if evaluated and c.get('heldout_shards'):
            self.heldout_eval(it, m)               # (docs/az.md 20.2: here, also with eval_remote)
        if self.remote is None:
            if evaluated:
                self.evaluate(it, m)
            ratings, best = self.elo_update(it, m)
            me = ratings.get(elo.ckpt_name(it))
            if me:
                m['elo'] = elo.row(me, self.eval_tag())
                self.log(f'iter {it} elo {me[0]:.0f} +-{1.96 * me[1]:.0f}{elo.pool_note(self.eval_tag())}; '
                         + (f'best {best["file"]} {best["elo"]:.0f}' if best else 'no best yet'))
            if m.get('elo_anchored'):
                a = m['elo_anchored']
                self.log(f'iter {it} elo anchored {a["elo"]:.0f} +-{1.96 * a["se"]:.0f}{elo.pool_note(self.eval_tag())}')
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
        else:
            # a resume: the newest checkpoint is a network of this build's encoding, or the run
            # refuses to start (EncodingMismatch; exit status EXIT_REFUSED, docs/az.md 10.4)
            check_checkpoint(model_path(run, start), c['game'])
        # the root's belief needs the auxiliary head (HeadMissing: refused the same way, docs/az.md 18)
        self.check_belief_heads(model_path(run, start))
        # the anchors and the held-out set (docs/az.md 20): checked now, refused before any work
        self.check_anchors()
        if c.get('heldout_shards'):
            from .heldout import HeldOut
            try:
                self.heldout = HeldOut(c['heldout_shards'], game=c['game'], features=self.info['num_features'],
                                       actions=self.info['num_actions'])
            except (OSError, ValueError) as e:
                raise HeldOutRefused(f'heldout_shards: {e}') from None
        done = {m['iter'] for m in read_metrics(run)}
        self.log(f'run {run} starting after iter {start}; cfg {json.dumps(c)}')
        if self.light:
            self.log(LT.describe(self.light))
        if self.heavy:
            self.log(HV.describe(self.heavy))
        if c.get('eval_remote'):
            from .remote import RemoteEval
            self.remote = RemoteEval(self, c['eval_remote'])
            self.remote.backfill(start if start in done else start - 1)
            self.log(f'evaluations on the remote {c["eval_remote"]!r} ({self.remote.spec["host"]})')
        if c.get('selfplay_remote'):
            from .remoteplay import RemotePlay
            self.rsp = RemotePlay(self, c['selfplay_remote'])
            self.rsp.local(max(2, int(c['groups'])), int(c['conc']), device=c['infer_device'])
            self.rsp.restore(last_rates(read_metrics(run)))
            self.log(f'self-play also on the remote {c["selfplay_remote"]!r} ({self.rsp.spec["host"]}, '
                     f'{self.rsp.procs} worker(s), mirror {self.rsp.rdir}): {c["games"]} games per iteration in '
                     f'total, in flight per machine its throughput share x {self.rsp.inflight:g}'
                     + self.rsp.batch_note() + (', lingering while the run trains' if self.rsp.linger else ''))
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
            if self.league is not None:
                self.league.plan(nxt)              # (made here, on the main thread; shipped with the phase)
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


class AnchorRefused(EncodingMismatch):
    """An Elo anchor the run cannot use (docs/az.md 20.1): missing, or another file than the one it was
    fixed with. An EncodingMismatch, so the run is refused (exit 78) and not restarted."""


class HeldOutRefused(EncodingMismatch):
    """A held-out set the run cannot use (docs/az.md 20.2): missing shards, or another game or encoding.
    Refused like an EncodingMismatch (exit 78)."""


def aux_settings(cfg, info):
    """(weights {name: aux_<name>_w}, layout) of a run's auxiliary targets (docs/az.md 15): the game
    package's targets for the run's game; the layout [[name, width, loss], ...] when a weight is > 0,
    else []. On, the engine's targets must be the package's (names and widths in order: the engine
    computes them, `SearchGame::AUX`), else SystemExit."""
    targets = package.active().aux_of(cfg['game'])
    w = {name: float(cfg.get(f'aux_{name}_w') or 0.0) for name, _, _ in targets}
    if not any(v > 0 for v in w.values()):
        return w, []
    eng = [(str(n), int(k)) for n, k in info.get('aux_layout') or []]
    want = [(n, k) for n, k, _ in targets]
    if eng != want:
        raise SystemExit(f"auxiliary targets of {cfg['game']!r}: the game package declares {want}, this build's engine "
                         f"computes {eng} (SearchGame::AUX)")
    return w, targets


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
              f"{r['requeued']} + {r['requeued_timeout']} + {r.get('requeued_backoff', 0)} requeued (lost, overdue, "
              f"backing off), {r['duplicates_dropped']} duplicates dropped")
    if 'held' in r:
        s += f"; {r.get('games_played', 0)} played, {r['held']} held for the next iteration"
    for k in ('late', 'late_prev', 'rejected', 'quarantined', 'fetch_failed', 'session_errors', 'lost'):
        if r.get(k):
            s += f', {k} {r[k]}'
    if r.get('linger'):
        s += f", lingered {r['linger']['seconds']:.0f}s before it ({r['linger']['games']} games)"
    return s


def _fmt_queue(q):
    """The shared queue of a self-play phase in one log line (zero/workqueue.py)."""
    cons = q.get('consumers') or {}
    names = sorted(cons, key=lambda n: (n != 'local', n))
    s = ''
    if 'quota' in q:
        s = f"quota {q['quota']}: {q['games']} games (" + ', '.join(f"{n} {cons[n].get('games', 0)}" for n in names) + ')'
        if q.get('short'):
            s += f", SHORT {q['short']}"
        s += '; '
    s += ('caps ' + ', '.join(f"{n} {cons[n].get('cap')} -> {cons[n].get('cap_next')}" for n in names)
          + '; rates/s ' + ', '.join(f"{n} {cons[n].get('rate', 0):.3g}" for n in names))
    st = (q.get('staleness') or {}).get('summary')
    if st:
        s += f"; staleness: mean {st['mean']:.2f}, max {st['max']}, {100 * st['le1']:.1f}% <= 1 ({st['rows']} rows)"
    return s


def last_rates(rows):
    """The shared queue's consumer rates of the newest metrics row that has them (a restart's first
    caps follow them); a row of a run that had a remote before the shared queue (its
    `selfplay.remote`, no `queue`): this machine's games per second and each worker's share of the
    remote's games per running second, the leaves per game this machine's."""
    for m in reversed(rows):
        sp = m.get('selfplay') or {}
        q = sp.get('queue')
        if q and q.get('rates'):
            return q['rates']
        rem = sp.get('remote')
        if rem and rem.get('games') and rem.get('running_s') and sp.get('games') and sp.get('seconds'):
            out = {'local': {'rate': sp['games'] / sp['seconds']}}
            if sp.get('leaves'):
                out['_leaves_per_unit'] = sp['leaves'] / sp['games']
            w = max(1, int(rem.get('workers') or rem.get('procs') or 1))
            for k in range(w):
                out[f'w{k}'] = {'rate': rem['games'] / rem['running_s'] / w}
            return out
    return None


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


UNSET = 'none'


def _opt_float(s):
    """A setting whose default is None (unset): a number, or 'none' = unset (`UNSET`, given as
    such: it replaces a saved value)."""
    if s.strip().lower() in (UNSET, ''):
        return UNSET
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a number or 'none' (unset)") from None


def parse(argv=None):
    ap = argparse.ArgumentParser(description=f'{package.active().title} AlphaZero loop (docs/az.md)')
    ap.add_argument('--run', required=True)
    for k, v in DEFAULTS.items():
        flag = '--' + k.replace('_', '-')
        if v is None:
            ap.add_argument(flag, type=_opt_float, default=None, help="a number, or 'none' = unset")
        elif isinstance(v, list):
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
    given = {k: None if (DEFAULTS[k] is None and v == UNSET) else v for k, v in given.items()}
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
    L.check_settings(cfg)
    for k in package.active().aux_settings():
        w = cfg.get(k, 0.0)
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not w >= 0:
            raise SystemExit(f'{k} {w!r}: a number >= 0 (an auxiliary target weight; 0 = off, docs/az.md 15)')
    if isinstance(cfg.get('resolver_cfg'), dict) and L.enabled(cfg['resolver_cfg']):
        raise SystemExit('resolver_cfg: selfplay_league: single-game runs only (not a co-training resolver)')
    # light decisions (zero/light.py): classes the game declares, budgets that are light, no setting
    # that would do nothing; single-game runs only
    LT.check(cfg, G.info(cfg['game']))
    if cfg.get('resolver') and int(cfg.get('selfplay_sims_light', 0) or 0):
        raise SystemExit('selfplay_sims_light: single-game runs only (not a co-training run)')
    if isinstance(cfg.get('resolver_cfg'), dict) and int(cfg['resolver_cfg'].get('selfplay_sims_light', 0) or 0):
        raise SystemExit('resolver_cfg: selfplay_sims_light: single-game runs only (not a co-training resolver)')
    # heavy decisions (zero/heavy.py): budgets that are heavy, no setting that would do nothing;
    # single-game runs only
    HV.check(cfg)
    if cfg.get('resolver') and int(cfg.get('selfplay_sims_heavy', 0) or 0):
        raise SystemExit('selfplay_sims_heavy: single-game runs only (not a co-training run)')
    if isinstance(cfg.get('resolver_cfg'), dict) and int(cfg['resolver_cfg'].get('selfplay_sims_heavy', 0) or 0):
        raise SystemExit('resolver_cfg: selfplay_sims_heavy: single-game runs only (not a co-training resolver)')
    return cfg


def belief_keys(cfg, pre):
    """The root-belief keyword arguments of `games.nn_seat` for a run's seat `pre` ('selfplay' | 'eval')
    from its settings (`mcts.belief_settings`; {} when its tilt is 0: the seat of before)."""
    from .mcts import belief_settings
    tilt, clip, mix, fast = belief_settings(cfg, pre)
    return dict(belief_tilt=tilt, belief_clip=clip, belief_mix=mix, belief_fast=fast) if tilt else {}


def check_eval_settings(cfg, where=''):
    """The move-rule and Elo-offset settings (eval_select, selfplay_select, their guards and their
    completed-Q transforms, the root minimum visits, elo_offsets) -> None, else SystemExit (`where`:
    a prefix, e.g. 'resolver_cfg: ')."""
    from .mcts import belief_settings, check_belief, check_leaf_view, check_root_min, check_select, check_select_transform
    for k in ('eval_select', 'selfplay_select'):
        try:
            check_select(cfg.get(k, 'visits'), cfg.get(k + '_min_visits', 0), what=k)
            check_select_transform(cfg.get(k, 'visits'), cfg.get(k + '_c_visit'), cfg.get(k + '_c_scale'), what=k)
        except ValueError as e:
            raise SystemExit(f'{where}{e}') from None
    for k in ('selfplay_root_min_visits', 'eval_root_min_visits'):
        try:
            check_root_min(cfg.get(k, 0), cfg.get('root_min_visits_max_actions', 4), what=k)
        except ValueError as e:
            raise SystemExit(f'{where}{e}') from None
    try:
        check_leaf_view(cfg.get('selfplay_leaf_value_view', 'mover'), cfg.get('selfplay_view_samples', 0.0),
                        cfg.get('view_sample_target', 'next_q'), what='selfplay_leaf_value_view')
        check_leaf_view(cfg.get('eval_leaf_value_view', 'mover'), what='eval_leaf_value_view')
        for pre in ('selfplay', 'eval'):
            tilt, clip, mix, _ = belief_settings(cfg, pre)
            check_belief(tilt, clip, mix, what=f'{pre}_belief_tilt')
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    from .mcts import check_root_search
    for side, scale_d in (('selfplay', 1.0), ('eval', 0.0)):
        try:
            rs = cfg.get(f'{side}_root_search', 'puct')
            check_root_search(rs, cfg.get('gumbel_m', 16), cfg.get(f'{side}_gumbel_scale', scale_d),
                              what=f'{side}_root_search')
        except ValueError as e:
            raise SystemExit(f'{where}{e}') from None
        if rs == 'gumbel' and cfg.get(f'{side}_root_min_visits', 0):
            raise SystemExit(f"{where}{side}_root_min_visits with {side}_root_search 'gumbel': exclusive (sequential "
                             f"halving already visits every considered option; docs/search.md 3.7)")
    if cfg.get('selfplay_root_search', 'puct') == 'gumbel' and cfg.get('policy_target', 'visits') != 'cq':
        raise SystemExit(f"{where}selfplay_root_search 'gumbel' needs policy_target 'cq' (sequential halving's visit "
                         f"counts are its schedule, not a policy; docs/search.md 3.7)")
    try:
        elo.parse_offsets(cfg.get('elo_offsets', ''))
        anchors = elo.parse_anchors(cfg.get('elo_anchors', ''))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
    for k in ('elo_anchor_games', 'elo_anchor_calib_games', 'elo_anchor_from'):
        v = cfg.get(k, 0)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise SystemExit(f'{where}{k} {v!r}: an integer >= 0')
    if (anchors or cfg.get('elo_paired')) and (cfg.get('resolver') or where):
        raise SystemExit(f'{where}elo_anchors / elo_paired: single-game runs only (a co-training run plays its '
                         f'matches as events)')
    if cfg.get('elo_paired'):
        odd = [k for k in ('elo_games', 'elo_anchor_games', 'elo_anchor_calib_games') if int(cfg.get(k, 0)) % 2]
        if odd:
            raise SystemExit(f'{where}elo_paired: {", ".join(f"{k} {cfg[k]}" for k in odd)} must be even (a pair is '
                             f'two games: the same deal, the seats swapped)')


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
    except EncodingMismatch as e:
        # a network of another encoding than this build's (docs/az.md 10.4): no restart fixes it
        pw.close()
        log(run, f'run REFUSED (exit status {EXIT_REFUSED}: the supervisor does not restart it): {e}')
        runctl.write_state(run, 'run', state='refused', error=f'{type(e).__name__}: {e}'[:1000])
        print(f'{type(e).__name__}: {e}', file=sys.stderr, flush=True)
        raise SystemExit(EXIT_REFUSED)
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
