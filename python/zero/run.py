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

--reader-<target> 1: the game package's reader of that auxiliary target (docs/az.md 36; the toy's
--reader-next-card): a small MLP of the network reads declared public features, predicts the target
(trained by its loss: needs --aux-<target>-w > 0) and feeds the prediction into the trunk (added to a
checkpoint without it, every output unchanged). --reader-detach 0: the trunk's losses train it too.
0 (the default): the network of before.

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
--selfplay-remote a,b: several hosts, each its own workers and mirror; the run directory's
`selfplay_remotes` file (one name per line) replaces the list at every self-play phase - hosts join
and leave a live run - and alone turns remote self-play on; --selfplay-local 0: this machine plays
only while no remote worker takes work (docs/az.md 34).
"""
import argparse
import contextlib
import glob
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import auxtarget as AUX, elo, forks as FK, games as G, league as L, package, resign as RS, runctl
from .data import BAD, ShardWriter, TargetStats, Window, WindowCache, batches, shard_format, shard_paths
from .finite import NonFinite, check_state, head
from .infer import T_STEP, NetEvaluator, drive, use_amp
from .net import (EncodingMismatch, add_aux, add_reader, aux_layout, check_belief_head, check_checkpoint, load, make_net,
                  reader_layout, save)
from .protocol import MAX_SEATS
from .runctl import EXIT_PAUSED, EXIT_REFUSED, Paused
from .shardthread import MB, SampleClock, ThreadedShardWriter, split_records

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
    # policy target of the recorded (full-search) decisions: 'visits' (the root visit
    # distribution) | 'cq' (Gumbel AlphaZero's completed-Q target softmax(ln P0 + (cq_c_visit +
    # max N) cq_c_scale completedQ), docs/search.md 3.1); the search itself is the same
    policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0,
    # cq_prior_w = beta in (0, 1], the prior's exponent in the recorded 'cq' target softmax(beta ln P0 +
    # sigma completedQ) (docs/search.md 3.13, docs/az.md 26): beta < 1 gives the generations' policy a
    # bounded fixed point instead of a cumulative sum of advantages. Self-play's seat only (the moves,
    # the 'cq' rule, the Gumbel root and every evaluation never read it). 1 = off, as before
    cq_prior_w=1.0,
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
    # window_sparse 1 = the window keeps each row's features as their non-zeros (zero/data.py
    # SparseRows): the same window and batches, in a fraction of the RAM for sparse encodings
    # (MTG's ~0.7% non-zero); with window_cache the cache then keeps every shard whole, also when the
    # window is over budget (subsampled from the cache), so an iteration loads only its new shards.
    # The budget (ram_budget_gb) still counts dense rows: the same rows as without it
    window_sparse=0,
    # shard_sparse 1 = self-play writes each shard's features as their non-zeros (zero/data.py: x_ptr /
    # x_idx / x_val, meta x_format 'sparse'; every other array as before): the dense rows bit for bit in a
    # fraction of the disk and write time for sparse encodings (MTG's ~0.7% non-zero). Every reader takes
    # both formats, so a window may mix them (docs/az.md 32). 0 = dense shards, byte for byte as before
    shard_sparse=0,
    # train_prefetch n > 0 = training batches built n steps ahead on a background thread while the
    # device trains (zero/data.py batches: the same batches in the same order)
    train_prefetch=0,
    # train_static 1 = a token network trains through its static forward (TokenNet.forward_static, the
    # inference CUDA graphs' - every batch one chunk of its longest row's tokens, no host sync, a few
    # hundred kernels instead of eager `forward`'s per-segment packing and chunks of 64 rows) and, on
    # CUDA, a fused AdamW: the same losses and updates up to float32 rounding, ~3x faster a step on
    # MTG Zero's Delver network (docs/az.md 10.9). Another network is refused. 0 = as before
    train_static=0,
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
    # information-set merging (docs/search.md 3.16, docs/az.md 30): 1 = the seat's trees merge their nodes by the
    # root decider's observation (single-observer information-set MCTS inside the tree: the other seat's nodes
    # subset-armed over the union of its sampled worlds' options, the root decider's next decision one node whatever
    # the other seat was dealt). selfplay_infoset_merge: self-play's seat, every search of it (full, fast, minor;
    # its records, root values and resignation read the root as before; a remote worker whose checkout does not
    # merge is refused); needs selfplay_leaf_value_view 'root'. eval_infoset_merge: every evaluation seat; needs
    # eval_leaf_value_view 'root'; its moves change, so the Elo pool's name gets '-ism'. 0 = off (the default:
    # today's searches, bit for bit)
    selfplay_infoset_merge=0, eval_infoset_merge=0,
    # the root's belief (docs/search.md 3.8, docs/az.md 18): a full search of the seat asks the network's
    # auxiliary head once for the root's prediction of the hidden cards (one extra row: the root's own
    # features, the decider's view) and deals every simulation's world by it - the game's
    # `determinize_with`, weights clip((belief / uniform)^tilt, 1/belief_clip, belief_clip) - but a
    # belief_mix share dealt uniformly; belief_fast 1: fast and minor searches too. selfplay_belief_tilt:
    # self-play's seat (a league opponent's too); eval_belief_tilt: every evaluation seat (its moves
    # change: the Elo pool's name gets '-b<tilt>...'). Needs the auxiliary head (a checkpoint without it
    # is refused: net.HeadMissing). 0 = off (the default: today's searches, bit for bit)
    selfplay_belief_tilt=0.0, eval_belief_tilt=0.0, belief_clip=20.0, belief_mix=0.0, belief_fast=0,
    # per-target tilts of the root's belief (docs/search.md 3.8.1, docs/az.md 38): {target: alpha} - an auxiliary
    # target of the game the seat's deal tilts by its own alpha (MTG Zero: 'deck', its whole unseen deck); a seat
    # with one > 0 asks the belief even with its <pre>_belief_tilt 0 (belief_clip / _mix / _fast apply). {} = none
    # (the default: today's searches, bit for bit). The Elo pool's name gets '-t<target><alpha>'
    selfplay_belief_tilts={}, eval_belief_tilts={},
    # value choices (docs/search.md 3.17, docs/az.md 39): at a decision the game marks (`SearchGame::value_choice`:
    # MTG Zero's sealed deck choice) a network seat values each option by the network's value from its own view of
    # the state one decision on, averaged over <pre>_value_choice_worlds worlds (a one-ply lookahead whose rows go in
    # the batches), and takes the best mean (temperature 0) or draws by softmax(mean / <pre>_value_choice_temp);
    # never recorded (the network cannot see the options). The Elo pool's name gets '-vc<worlds>[t<temp>]'. 0 worlds
    # = off (the default: the decision searched as any other, today's searches bit for bit)
    selfplay_value_choice_worlds=0, selfplay_value_choice_temp=0.0, eval_value_choice_worlds=0, eval_value_choice_temp=0.0,
    # readers (docs/az.md 36): the game package's `readers`, each turned on by its run setting
    # `reader_<target>` (default 0 = off: the network of before) - a small branch of the network reading
    # declared features, trained by its auxiliary target's loss (aux_<target>_w > 0), its prediction an
    # input of the trunk. reader_detach 1: the prediction enters the trunk detached (only the auxiliary
    # loss trains the reader); 0: the trunk's losses train it too
    reader_detach=1,
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
    # A comma-separated list of names (or the run directory's `selfplay_remotes` file, one name per line,
    # re-read at every self-play phase: hosts join and leave a live run; the file alone turns remote
    # self-play on): every host its own workers, mirror and shards, one shared queue (docs/az.md 34)
    selfplay_remote='', selfplay_remote_procs=0, selfplay_remote_wait=30.0, selfplay_remote_timeout=3600.0,
    # selfplay_local 0 = while at least one remote worker takes work, this machine's driver takes no
    # games (the run's machine only trains: an actor / learner split); with none (all offline, backing
    # off, refused, or no host listed) it plays the quota itself, so the run never stalls. 1 = this
    # machine always takes its share, as before (docs/az.md 34)
    selfplay_local=1,
    # the shared queue (zero/workqueue.py): each machine's in-flight cap in iterations of its own
    # throughput (its measured share of the iteration's games or matches x this: 1 = a game it starts
    # finishes within about one iteration); co-training: a worker backing off this many seconds
    # gives its outstanding matches back (requeued here), and a match outstanding on a worker longer
    # than this many of its expected latencies is requeued (at least 120 s, at most
    # selfplay_remote_timeout; 0 = the timeout only); linger 1 = the workers keep playing with their
    # checkpoint while the run trains (those games count for the next iteration, one checkpoint stale)
    selfplay_remote_inflight=1.0, selfplay_remote_giveback=30.0, selfplay_remote_overdue=3.0,
    selfplay_remote_linger=0,
    # selfplay_local_linger 1 = this machine's self-play driver plays on too while the run trains (on a
    # thread, its network batches on a CUDA stream of their own beside the training's): new games up to
    # its cap, and the games in flight carried over finish with the checkpoint they began with; the games
    # finished then count for the next iteration (its quota takes them first), one checkpoint stale - as
    # a remote's lingering games. Needs selfplay_remote (the shared queue); single-game runs without the
    # self-play league. 0 = this machine's self-play waits while the run trains, as before
    selfplay_local_linger=0,
    # shard_writer_thread 1 = self-play's training records are converted and written into shards on a thread of
    # their own (zero/shardthread.py, docs/az.md 31): the thread that drives the search only takes them from the
    # driver (SearchDriver.take_records: moved, not copied) and queues them, at most shard_writer_queue_mb MB (it
    # waits while the queue is full; metrics selfplay.shard_writer). The same shards byte for byte, but the caller's
    # timing changes which games finish when (a rolling or shared quota's games). This machine's self-play, the
    # local linger and the remote workers (their config's); single-game runs. 0 = the samples taken, converted
    # and written between batches, as before
    shard_writer_thread=0, shard_writer_queue_mb=4096,
    # branched self-play games and archive starts (zero/forks.py, docs/az.md 24, docs/search.md 2.11):
    # selfplay_branch p = a share p of the games each spawn one branch game from a copy of their state at
    # a uniformly drawn decision of theirs: 1..selfplay_branch_moves deviation moves sampled from the
    # network's priors at temperature selfplay_branch_temp (never recorded), then normal self-play, its
    # rows' z its own outcome; a branch counts as one of the iteration's games. selfplay_archive = an
    # archive file ('zero-archive/1', built for the run's spec by the game's own tool): a share
    # selfplay_archive_p of the games start from one of its positions instead of a new deal. 0 / '' =
    # off: nothing changes. Single-game runs without the league; branching needs rolling self-play
    # (selfplay_branch_differ 1: a branch's first deviation never the game's own move there)
    selfplay_branch=0.0, selfplay_branch_moves=3, selfplay_branch_temp=2.0, selfplay_branch_differ=0, selfplay_archive='',
    selfplay_archive_p=0.1,
    # ... except a batch-bound machine's (its network batches grow with its games in flight, so a
    # smaller cap only starves it: 'auto' = the ones inferring on CUDA / MPS; 'local' | 'remote' |
    # 'all' | 'none'): its slots while its games take at most this many iterations (its throughput
    # share x this; 1 = as the others)
    selfplay_remote_batch='auto', selfplay_remote_batch_inflight=2.0,
    infer_device='cpu', train_device='cpu', seed=0,
    # network batches evaluated concurrently on this many Python threads (CPU: small networks
    # whose ops cannot keep the cores busy; give the driver as many `groups`); 1 = inline
    infer_workers=1,
    # sparse 1 = the run's drivers (self-play and matches) send sparse network batches (each row's
    # non-zero features, DriverCfg::sparse, docs/search.md 2.2): the same rows, games and samples; a
    # token network's CUDA graphs scatter them on the device instead of the host copying the dense
    # rows (docs/az.md 10.6), other evaluators densify them. 0 = dense batches, as before
    sparse=0,
    # eval_cache N > 0 = every network seat of the run (self-play and evaluation) keeps its game's network
    # answers, at most N per game, by a hash of each row's features and options: a row the game already
    # sent is answered from the cache instead of sent (docs/search.md 3.12, docs/az.md 25: the same
    # answers up to the network's float rounding at another batch shape). 0 = off, as before
    eval_cache=0,
    # self-play resignation with a no-resign share (zero/resign.py, docs/az.md 27, docs/search.md 3.14; KataGo's):
    # selfplay_resign_t t > 0 = a self-play game ends when selfplay_resign_k consecutive full searches of its
    # seats (the recorded ones, never fast or minor) found one seat's root win estimate >= 1 - t - the same seat
    # each time -, scored as that seat's win, every record kept (z the resignation's); a share
    # selfplay_resign_noresign of the games (a hash of seed and game: no stream moves) never resigns and reports
    # where it would have (the live wrong-resignation rate, `selfplay.resign`; 1 = a dry run). Evaluations never
    # resign. 0 = off: nothing changes. Two-player single-game runs without the league
    selfplay_resign_t=0.0, selfplay_resign_k=5, selfplay_resign_noresign=0.1,
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
    (`aux_<name>_w`, default 0 = off; docs/az.md 15), and the package's own run settings
    (`GamePackage.settings`, docs/az.md 40; a name the core already has is an error). Unknown setting
    names are an error."""
    known = dict(_CORE_DEFAULTS, **pkg.aux_settings(), **pkg.reader_settings())
    # the package's own run settings (GamePackage.settings, docs/az.md 40): new names only
    clash = sorted(set(getattr(pkg, 'settings', {})) & set(known))
    if clash:
        raise ValueError(f'{pkg.name}: the package\'s settings {clash} are run settings of the core already '
                         f'(names of the core\'s, an auxiliary target\'s or a reader\'s)')
    known.update(getattr(pkg, 'settings', {}))
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


def match_opt_groups(state, pgroups):
    """Which of the parameter groups `pgroups` (lists of parameters) an optimizer state dict's groups belong
    to, in the saved order -> [an index of `pgroups` per saved group], or None when a saved group fits none
    (docs/az.md 37: the network of a widened head has the groups of before and new ones, the order of
    history). Each saved group takes the first group not taken with as many parameters whose moments have
    their shapes (a parameter never stepped has none: its count decides)."""
    groups = state.get('param_groups') or []
    st = state.get('state', {})
    taken = []
    for g in groups:
        ids = list(g.get('params', []))
        for j, pg in enumerate(pgroups):
            if j in taken or len(pg) != len(ids):
                continue
            if all(tuple(v.shape) == tuple(p.shape) for i, p in zip(ids, pg) for v in (st.get(i) or {}).values()
                   if torch.is_tensor(v) and v.dim() > 0):
                taken.append(j)
                break
        else:
            return None
    return taken or None


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
        self.roll_spec = None                      # the rolling driver's spec (selfplay_specs, docs/az.md 40)
        self.wcache = None                         # zero.data.WindowCache with window_cache
        # the self-play league (zero/league.py): None when selfplay_league is 0 (nothing changes)
        self.league = L.League(self) if L.enabled(cfg) else None
        self.roll_kinds = None                     # the rolling driver's table: each entry's league game
        self.roll_evs = None                       # ... and its evaluators by key (m, L0, ...)
        # auxiliary targets (docs/az.md 15): the game's layout [[name, width, loss], ...] when one of
        # their weights is on, else [] (off: nothing recorded, built or trained)
        self.aux_w, self.aux = aux_settings(cfg, self.info)
        # readers (docs/az.md 36): the game's readers turned on (reader_<target> 1), [] when none is
        self.readers = reader_settings(cfg, self.aux_w)
        self.heldout = None                        # zero.heldout.HeldOut with heldout_shards (loaded once)
        self.anchor_sha = {}                       # anchor path -> ((size, mtime), sha256)
        self.linger = None                         # selfplay_local_linger: the lingering thread's state
        self.writers = []                          # shard_writer_thread: the self-play phase's writer threads
        # branched games and archive starts (zero/forks.py): the self-play driver's settings, None when off
        self.forks = FK.setup(run, cfg)

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
                        c.get('ff', 0), aux=self.aux or None, readers=self.readers or None)

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
                             gumbel_scale=c.get('selfplay_gumbel_scale', 1.0), eval_cache=c.get('eval_cache', 0),
                             cq_prior_w=c.get('cq_prior_w', 1.0),
                             infoset_merge=bool(c.get('selfplay_infoset_merge', 0)),
                             **belief_keys(c, 'selfplay'), **RS.seat_keys(c), **value_choice_keys(c, 'selfplay'))
        return G.nn_seat(key, sims, c['c_puct'], chance_k=c['chance_k'], margin_w=c['margin_w'], redet=bool(c['redet']),
                         cq_c_visit=c.get('cq_c_visit', 50.0), cq_c_scale=c.get('cq_c_scale', 1.0),
                         select=c.get('eval_select', 'visits'), select_min_visits=c.get('eval_select_min_visits', 0),
                         select_c_visit=c.get('eval_select_c_visit'), select_c_scale=c.get('eval_select_c_scale'),
                         root_min_visits=c.get('eval_root_min_visits', 0),
                         root_min_visits_max_actions=c.get('root_min_visits_max_actions', 4),
                         leaf_value_view=c.get('eval_leaf_value_view', 'mover'),
                         root_search=c.get('eval_root_search', 'puct'), gumbel_m=c.get('gumbel_m', 16),
                         gumbel_scale=c.get('eval_gumbel_scale', 0.0), eval_cache=c.get('eval_cache', 0),
                         infoset_merge=bool(c.get('eval_infoset_merge', 0)), **belief_keys(c, 'eval'),
                         **value_choice_keys(c, 'eval'))

    def check_belief_heads(self, path):
        """With a root belief on (selfplay_ / eval_belief_tilt > 0, docs/az.md 18) the checkpoint at
        `path` must have the auxiliary head for the game's targets: else net.HeadMissing (the run is
        refused, exit 78) - never a search dealing uniformly in silence."""
        from .mcts import belief_tilts
        c = self.cfg
        on = [k for k in ('selfplay_belief_tilt', 'eval_belief_tilt') if float(c.get(k, 0.0) or 0.0) > 0]
        on += [k + 's' for k in ('selfplay_belief_tilt', 'eval_belief_tilt') if belief_tilts(c, k.split('_')[0])]
        if on:
            m, _ = load(path, 'cpu', game=c['game'])
            check_belief_head(m, c['game'], f'{" / ".join(on)} (checkpoint {path})')

    def eval_tag(self):
        """The name of the evaluation seats' move rule ('visits', 'cq', 'cq-n2' ...): the evaluations'
        'select' and their Elo pool (docs/az.md 5.4)."""
        return elo.eval_tag(self.cfg)

    def driver(self, seats, seed, record=False, conc=None, rolling=False, net=0, groups=None, queue_only=False,
               paired=False, spec=None):
        """A driver for self-play (`record`: the run's spec) or a match (eval_spec, if set).
        `queue_only`: a rolling driver that plays only queued setups (co-training). `paired`: games 2k
        and 2k + 1 share their streams (the driver's `paired`, docs/az.md 20.3). `spec`: self-play's
        spec of the phase instead of the run's (`selfplay_spec`, docs/az.md 40)."""
        c = self.cfg
        if spec is not None and not record:
            raise ValueError('driver: a spec of its own is self-play\'s (record)')
        if spec is None:
            spec = c['spec'] if record else (c.get('eval_spec') or package.active().eval_spec_of(c['game'], c['spec']))
        kw = {'queue_only': True} if queue_only else {}
        if paired:
            kw['paired'] = True
        if c.get('sparse'):
            kw['sparse'] = True                    # sparse network batches (docs/search.md 2.2)
        if record and getattr(self, 'aux', None):
            # self-play records the auxiliary targets (docs/az.md 15; getattr: a caller's stand-in Run
            # with only a cfg, e.g. MTG Zero's test of its evaluation specs, has none)
            kw['aux'] = True
        if record and getattr(self, 'forks', None) is not None:
            kw.update(self.forks.kwargs())         # branched games, archive starts (zero/forks.py)
        return G.Driver(c['game'], seats, spec=spec, concurrency=conc or c['conc'], seed=seed, record=record,
                        threads=c['threads'], groups=groups or c['groups'], max_steps=c['max_steps'], rolling=rolling,
                        net=net, **kw)

    def match_groups(self, ext):
        return self.cfg.get('ext_groups', DEFAULTS['ext_groups']) if ext else None

    # ---------------------------------------------------------------- self-play
    def shard_thread(self, drv, writer, on_item=None):
        """shard_writer_thread (docs/az.md 31): the phase's ShardWriter on a writer thread of its own (a
        zero.shardthread.ThreadedShardWriter the phase feeds with `drv.take_records()`; `on_item(s)`: run
        there on each converted sample dict) - else None, the synchronous writer of before."""
        c = self.cfg
        if not c.get('shard_writer_thread'):
            return None
        if not hasattr(drv, 'take_records'):
            raise RuntimeError('shard_writer_thread: this engine module has no SearchDriver.take_records (built '
                               'from an older zero-core): rebuild it (tools/build_py.sh)')
        tw = ThreadedShardWriter(writer, max_bytes=float(c['shard_writer_queue_mb']) * MB, on_item=on_item)
        self.writers.append(tw)
        return tw

    def selfplay(self, it):
        """Iteration `it`'s self-play (`_selfplay`). An exception on the way out (PAUSE, an error) stops the
        phase's shard writer thread first (shard_writer_thread): the shard it is writing is completed, the
        rest dropped - the iteration is replayed - and no thread outlives the phase."""
        try:
            return self._selfplay(it)
        except BaseException:
            for tw in self.writers:
                tw.abort()
            raise
        finally:
            self.writers = []

    def selfplay_spec(self, it):
        """Iteration `it`'s self-play spec (docs/az.md 40): the game package's `selfplay_specs` hook's answer, else
        (no hook, or it answers None) the run's spec. A spec of the hook's must be a non-empty string."""
        f = package.active().selfplay_spec_fn(self.cfg['game'])
        s = f(self, it) if f is not None else None
        if s is None:
            return self.cfg['spec']
        if not isinstance(s, str) or not s:
            raise ValueError(f'the game package\'s selfplay_specs hook answered {s!r} for iteration {it}: a spec JSON or None')
        return s

    def _selfplay(self, it):
        if self.rsp is not None:
            # (remote self-play's workers play the run's spec: a phase's spec of the package's is refused)
            if self.selfplay_spec(it) != self.cfg['spec']:
                raise RuntimeError('selfplay_specs: the game package changed the self-play spec, which remote self-play '
                                   'does not follow (docs/az.md 40)')
            return self.selfplay_shared(it)        # with a remote: the shared queue (docs/az.md 5.3)
        c = self.cfg
        spec = self.selfplay_spec(it)
        d = os.path.join(self.run, 'data', f'iter_{it:04d}')
        if os.path.exists(d):
            self.drop_data(d, strict=True)         # an interrupted iteration is replayed
        t0 = time.time()
        seat = self.nn_seat('m', c['sims'], selfplay=True)
        rng = np.random.default_rng(c['seed'] * 1_000_003 + it)
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'], 'iter': it,
                **shard_format(c)}
        lg = self.league
        if c['rolling']:
            if self.roll is None:
                if lg is not None:
                    table, self.roll_kinds = L.rolling_table(c, seat)
                else:
                    table = [[seat] * p for p in c['players']]
                self.roll = self.driver(table, seed=c['seed'] * 7919 + it, record=True, rolling=True, net=it - 1,
                                        spec=spec if spec != c['spec'] else None)
                self.roll_spec = spec
                self.roll_ev = self.evaluator(model_path(self.run, it - 1))
                self.roll_evs = {'m': self.roll_ev}
            else:
                m, _ = load(model_path(self.run, it - 1), 'cpu', game=c['game'])
                self.roll_ev.set_network(m)
                if spec != self.roll_spec:
                    # (docs/az.md 40: the games the driver starts from now on play the phase's spec)
                    self.roll.set_spec(spec)
                    self.roll_spec = spec
                    self.log(f'iter {it} self-play: the rolling driver plays the game package\'s new spec from now on')
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
            drv = self.driver(games, seed=c['seed'] * 1_000_003 + it, record=True, spec=spec if spec != c['spec'] else None)
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
        # shard_writer_thread (docs/az.md 31): the records queued as they are, converted, tallied and written
        # on the writer thread; None = here, between batches, as before
        tw = self.shard_thread(drv, writer, on_item=tstats.add)
        clock = SampleClock() if tw is not None else contextlib.nullcontext()

        def take():
            if tw is not None:
                tw.add(drv.take_records())
                return
            s = drv.take_samples()
            tstats.add(s)
            writer.add(s)

        def on_batch(lb):
            state['n'] += 1
            if state['n'] % 32 == 0:
                with clock:
                    take()
                    results.extend(drv.take_results())
            if time.time() - state['t'] > 2:
                state['t'] = time.time()
                p = {'iter': it, 'phase': 'selfplay', 'games': len(results), 'target': c['games'], 'time': time.time()}
                _write_json(prog, p)
        drive(drv, evs, check=lambda: self.check('self-play'), on_batch=on_batch)
        with clock:
            take()
            results.extend(drv.take_results())
        ws = writer.close() if tw is None else tw.close()
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
        self.fork_metrics(it, out, results)
        self.cache_metrics(out, st, st0)
        self.merge_metrics(it, out, st, st0, leaves)
        self.resign_metrics(it, out, results)
        self.episode_metrics(it, out, results)
        ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0
        if lg is not None:
            out['league'] = self.league_metrics(it, plan, L.tally(L.League.attribute(results, kinds, plan['slots'])),
                                                None, len(results), evs)
        if c['rolling']:
            inf = drv.in_flight()
            out['rolling'] = {'in_flight': inf['games'], 'restarts': st['restarts'] - st0['restarts'],
                              'restart_sims': st['restart_sims'] - st0['restart_sims'],
                              'dropped_leaves': st['dropped_leaves'] - st0['dropped_leaves']}
        self.writer_metrics(it, out, tw, clock)
        return out

    def writer_metrics(self, it, out, tw, clock):
        """shard_writer_thread: the phase's `shard_writer` block (docs/az.md 31) - the caller's seconds in its
        sample handling (samples_s: taking the results and records, splitting, the quota, queueing, its waits
        for room included), its waits for room (waits, wait_s), at the end for the last shard (close_s), the
        writer thread's busy seconds, the queue's peak - and a log line. Nothing when off."""
        if tw is None:
            return
        out['shard_writer'] = w = dict(tw.metrics(), samples_s=round(clock.s, 3))
        self.log(f'iter {it} shard writer thread: caller {w["samples_s"]:.2f}s in the samples ({w["waits"]} waits for '
                 f'room, {w["wait_s"]:.2f}s; {w["close_s"]:.2f}s at the end), writer {w["writer_s"]:.2f}s, queue peak '
                 f'{w["queue_peak_mb"]:.0f} of {w["queue_mb"]:.0f} MB')

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

    def fork_metrics(self, it, out, results, remote=None):
        """Self-play's branched games and archive starts (zero/forks.py), only when one is on (the metrics
        of before otherwise): `forks` = the iteration's accepted games, samples and searched decisions by
        origin (game / branch / archive), this machine's and the remote's (`remote`: its tally), logged."""
        if self.forks is None:
            return
        acc = FK.tally(results, FK.new_tally(), self.forks.kinds())
        out['forks'] = FK.summary(FK.merge(acc, remote), self.forks)
        if remote:
            out['forks']['remote'] = {'games': dict(remote['games']), 'samples': dict(remote['samples'])}
        self.log(f'iter {it} forks: {FK.fmt(out["forks"])}')

    def belief_metrics(self, out, st, st0, results):
        """Self-play's root-belief metrics (docs/az.md 18), only when it is on (the metrics of before
        otherwise): `belief` = {rows: root belief rows sent by this machine's driver, per_decision: per
        searched decision of its finished games}."""
        from .mcts import belief_tilts
        if not float(self.cfg.get('selfplay_belief_tilt', 0.0) or 0.0) and not belief_tilts(self.cfg, 'selfplay'):
            return
        rows = st.get('belief_rows', 0) - ((st0 or {}).get('belief_rows', 0))
        dec = sum(r.get('decisions', 0) for r in results)
        out['belief'] = {'rows': rows, 'per_decision': round(rows / max(1, dec), 4)}

    def cache_metrics(self, out, st, st0):
        """Self-play's evaluation-cache metrics (docs/az.md 25), only when it is on (the metrics of before
        otherwise): `eval_cache` = {hits: network rows answered from a game's cache by this machine's
        driver, leaves: leaves expanded at once (every row cached), partial: leaves that sent one row and
        took the other from the cache, clears: caches emptied because full, share: hits / (hits + rows
        sent)}."""
        if not self.cfg.get('eval_cache'):
            return

        def d(k):
            return st.get(k, 0) - ((st0 or {}).get(k, 0))
        hits = d('cache_hits')
        sent = d('leaves') + d('view_rows') + d('belief_rows')
        out['eval_cache'] = {'hits': hits, 'leaves': d('cache_leaves'), 'partial': d('cache_partial'),
                             'clears': d('cache_clears'),
                             'share': round(hits / max(1, hits + sent), 4)}

    def merge_metrics(self, it, out, st, st0, leaves):
        """Self-play's information-set merging (docs/search.md 3.16, docs/az.md 30), only when it is on (the metrics
        of before otherwise): `merge` = this machine's driver's merged trees - links (new child signatures), merged
        (of them, found by their observation), aliases, growths (growth leaves), shared (leaf rows of the viewer taken
        from the tree's encoding, not encoded again), merged_per_leaf. An iteration whose leaves made no link is
        logged: a setting that did nothing."""
        if not int(self.cfg.get('selfplay_infoset_merge', 0) or 0):
            return

        def d(k):
            return st.get(k, 0) - ((st0 or {}).get(k, 0))
        out['merge'] = {'links': d('merge_links'), 'merged': d('merge_merged'), 'aliases': d('merge_aliases'),
                        'growths': d('merge_growths'), 'shared': d('merge_shared'),
                        'merged_per_leaf': round(d('merge_merged') / max(1, leaves), 4)}
        if leaves > 0 and out['merge']['links'] == 0:
            self.log(f'iter {it} WARNING: selfplay_infoset_merge is on but the merged trees made no link '
                     f'({leaves} leaves): nothing was merged (docs/search.md 3.16)')

    def resign_metrics(self, it, out, results, remote=None):
        """Self-play resignation (zero/resign.py, docs/az.md 27), only when it is on (the metrics of before
        otherwise): `resign` = the iteration's accepted games resigned, the no-resign games, where those would
        have resigned and how often wrongly, the rows a resignation saves - this machine's and the remote's
        (`remote`: its tally) -, logged."""
        if not RS.enabled(self.cfg):
            return
        acc = RS.merge(RS.tally(results, RS.new_tally()), remote)
        out['resign'] = RS.summary(acc)
        if remote:
            out['resign']['remote'] = {'games': remote['games'], 'resigned': remote['resigned'],
                                       'noresign': remote['noresign']}
        self.log(f'iter {it} resign: {RS.fmt(out["resign"])}')

    def episode_metrics(self, it, out, results):
        """A chained game's contests (docs/az.md 42), only when the games chain (the metrics of before otherwise):
        `episodes` = this machine's finished contests' games - how many, per contest, drawn, resigned, unfinished
        (cut or aborted) - logged. The rows' `games` count contests."""
        eps = [e for r in results for e in (r.get('episodes') or ())]
        if not eps:
            return
        n = sum(1 for r in results if r.get('episodes'))
        out['episodes'] = {'games': len(eps), 'contests': n, 'per_contest': round(len(eps) / max(1, n), 3),
                           'draws': sum(1 for e in eps if e['over'] and max(e['result']) < 1.0 - 1e-6),
                           'resigned': sum(1 for e in eps if (e.get('resign') or {}).get('resigned')),
                           'unfinished': sum(1 for e in eps if not e['over']),
                           'samples_per_game': round(sum(e['samples'] for e in eps) / len(eps), 1)}
        self.log(f"iter {it} episodes: {out['episodes']['games']} games in {n} contests "
                 f"({out['episodes']['per_contest']} each), {out['episodes']['draws']} drawn, "
                 f"{out['episodes']['resigned']} resigned, {out['episodes']['unfinished']} unfinished")

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
            self.roll_ev.set_network(m)
        drv, ev, evs = self.roll, self.roll_ev, self.roll_evs
        if lg is not None:
            plan = lg.plan(it)
            lg.evaluators(it, evs)                 # the slots' past checkpoints (guarded)
        st0 = drv.stats()
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'], 'iter': it,
                **shard_format(c)}
        writer = ShardWriter(d, 'r00', rows=c['shard_rows'], meta=meta, bad_dir=self.debug_dir())
        tstats = TargetStats()
        spec = G.spec_json(c['spec'])
        rsp.begin(it)                              # (a no-op if the remote began it early)
        prog = os.path.join(self.run, 'progress.json')
        acc = {'results': [], 'stale': {}, 'finished': 0, 'util': [0.0, 0], 't': 0.0, 'league': {}}

        def tally(sd):
            tstats.add(sd)
            lag_hist(sd['net'], it, acc['stale'])
        # shard_writer_thread (docs/az.md 31): the held games' records are queued as they are, converted, tallied
        # and written on the writer thread; None = here, between batches, as before
        tw = self.shard_thread(drv, writer, on_item=tally)
        clock = SampleClock() if tw is not None else contextlib.nullcontext()
        take, split = (drv.take_samples, split_games) if tw is None else (drv.take_records, split_records)

        def accept():
            """The held games, oldest first, as many as the quota takes."""
            a = rsp.accept_local(it, len(self.held))
            for _ in range(a):
                row, sd = self.held.popleft()
                acc['results'].append(row)
                if lg is not None:
                    L.tally([row], acc['league'])
                if sd is not None and tw is not None:
                    tw.add(sd)
                elif sd is not None:
                    tstats.add(sd)
                    writer.add(sd)
                    lag_hist(sd['net'], it, acc['stale'])

        def pump(lb=None):
            with clock:
                rows = drv.take_results()
                s = take()
                if rows and lg is not None:
                    L.League.attribute(rows, self.roll_kinds, plan['slots'])   # (the slots' checkpoints now)
                if rows:
                    self.taken += FK.setups(rows)      # (a branch is its game's second row: zero/forks.py)
                    acc['finished'] += len(rows)
                    self.held.extend(split(rows, s))
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
                rps = rp.pop('remotes', None)      # (a list of hosts: each host's, docs/az.md 34)
                _write_json(prog, {'iter': it, 'phase': 'selfplay', 'games': q.get('accepted', len(acc['results'])),
                                   'target': c['games'], 'local': len(acc['results']), 'remote': rp,
                                   **({'remotes': rps} if rps is not None else {}), 'time': now})

        with clock:
            accept()                               # the last iteration's leftovers come first
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
            self.taken += FK.setups(rows)
            acc['finished'] += len(rows)
            self.held.extend(split(rows, take()))
        ws = writer.close() if tw is None else tw.close()
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
        self.fork_metrics(it, out, results, rem.pop('forks', None))
        self.cache_metrics(out, st, st0)
        self.merge_metrics(it, out, st, st0, leaves)
        self.resign_metrics(it, out, results, rem.pop('resign', None))
        self.episode_metrics(it, out, results)
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
        rems = rem.pop('remotes', None)            # (a list of hosts: each host's share, docs/az.md 34)
        out['remote'] = rem
        if rems is not None:
            out['remotes'] = rems
        if lg is not None:
            out['league'] = self.league_metrics(it, plan, acc['league'], rem_league, total, evs)
        out['queue'] = {'quota': c['games'], 'games': total, 'short': max(0, c['games'] - total),
                        'held': {'local': len(self.held), 'remote': qb.get('held_remote', 0)},
                        'inflight': qb['inflight'], 'batch_inflight': qb['batch_inflight'], 'units': qb['units'],
                        'consumers': cons, 'rates': qb['rates'], 'staleness': staleness(acc['stale'], qb['stale'])}
        self.log(f'iter {it} remote self-play: {_fmt_remote(rem)}')
        for name, r in (rems or {}).items():
            self.log(f'iter {it} remote self-play {name}: {_fmt_remote(r)}')
        self.log(f'iter {it} shared queue: {_fmt_queue(out["queue"])}')
        self.writer_metrics(it, out, tw, clock)
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

    def linger_start(self, it):
        """selfplay_local_linger (docs/az.md 5.3): after iteration `it`'s self-play phase, this machine's
        rolling driver plays on with checkpoint it - 1 on a thread while the run trains - setups queued up
        to its cap as in the phase, the games in flight finishing with the network they began with; the
        finished games are held, with their samples, for iteration it + 1's quota (which takes the held
        games first): one checkpoint stale, as a remote's linger. Its network batches run on a CUDA stream
        of their own (the training's are on the default one). Never after the last iteration, nor with
        STOP or PAUSE there, nor more games held than an iteration's quota."""
        c = self.cfg
        if not c.get('selfplay_local_linger') or self.rsp is None or self.roll is None or it >= c['iters'] \
                or runctl.stop_requested(self.run) or runctl.pause_requested(self.run):
            return
        st = self.linger = {'stop': threading.Event(), 'games': 0, 'error': None, 't0': time.time(),
                            'leaves0': self.roll.stats()['leaves']}
        st['thread'] = threading.Thread(target=self._linger, args=(it, st), name='zero-local-linger', daemon=True)
        st['thread'].start()

    def _linger(self, it, st):
        from .workqueue import split_games
        c = self.cfg
        drv, evs = self.roll, self.roll_evs
        spec = G.spec_json(c['spec'])
        cap_held = int(c['games'])
        cuda = str(c['infer_device']).startswith('cuda') and torch.cuda.is_available()
        side = torch.cuda.Stream() if cuda else None
        # shard_writer_thread (docs/az.md 31): the held games keep their records as they are (the next phase's
        # writer thread converts them); None = their sample arrays taken here, as before
        on = bool(c.get('shard_writer_thread'))
        clock = st['clock'] = SampleClock() if on else contextlib.nullcontext()
        take, split = (drv.take_records, split_records) if on else (drv.take_samples, split_games)

        def pump(lb=None):
            with clock:
                rows = drv.take_results()
                s = take()
                if rows:
                    self.taken += FK.setups(rows)      # (a branch is its game's second row: zero/forks.py)
                    st['games'] += len(rows)
                    self.held.extend(split(rows, s))
            if st['stop'].is_set() or len(self.held) >= cap_held:
                drv.end_phase()
            else:
                cap = self.rsp.local_cap()
                for _ in range(max(0, cap - (self.enq - self.taken))):
                    drv.enqueue(f'L{self.enq}', spec)
                    self.enq += 1
        try:
            with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
                drv.resume(BIG, it - 1)            # the same network: the searches in flight go on
                while True:
                    pump()
                    drive(drv, evs, on_batch=pump)
                    if drv.paused:
                        break
                    if drv.in_flight()['games'] == 0 and not drv.queued:
                        time.sleep(0.05)           # nothing to play (a cap of 0)
                pump()
            if side is not None:
                side.synchronize()
        except BaseException as e:                 # noqa: BLE001  (raised by linger_stop on the main thread)
            st['error'] = e

    def linger_stop(self, quiet=False):
        """End the lingering self-play (linger_start): the driver paused, its finished games held -> its
        metrics {games, seconds, leaves, infer_s, held}, None if none ran. A failure on its thread is
        raised here (`quiet`: not - another exception is on its way)."""
        st = self.linger
        if st is None:
            return None
        self.linger = None
        st['stop'].set()
        st['thread'].join()
        if st['error'] is not None and not quiet:
            raise RuntimeError(f'local linger: {type(st["error"]).__name__}: {st["error"]}') from st['error']
        ev = self.roll_ev
        out = {'games': st['games'], 'seconds': round(time.time() - st['t0'], 2),
               'leaves': self.roll.stats()['leaves'] - st['leaves0'], 'infer_s': round(ev.seconds, 2),
               'held': len(self.held)}
        if isinstance(st.get('clock'), SampleClock):
            out['samples_s'] = round(st['clock'].s, 3)       # (shard_writer_thread: its caller's sample handling)
        ev.seconds, ev.calls, ev.leaves = 0.0, 0, 0         # (the next phase's own inference metrics)
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
        model, _ = load(model_path(self.run, it - 1), dev, game=c['game'])
        added = False
        # a head for some of the game's targets was widened to all of them by `load` (docs/az.md 37)
        widened = getattr(model, 'aux_widened', None)
        if widened:
            self.log(f'iter {it}: auxiliary head of checkpoint {it - 1} widened {[t[0] for t in widened]} -> '
                     f'{[t[0] for t in model.aux_layout]} (every output of before unchanged; docs/az.md 37)')
        if self.aux and aux_layout(model.config.get('aux')) != self.aux:
            if model.config.get('aux'):
                raise SystemExit(f'iter {it}: checkpoint {it - 1} has the auxiliary head {model.config["aux"]}, this run '
                                 f'trains {self.aux} (a head for some of the targets is widened exactly, docs/az.md 37; '
                                 f'a target removed, renamed or changed: net.strip_aux, then fork the run)')
            # a checkpoint without the head (a fork of a run without it): a fresh head, the rest as it was
            model = add_aux(model, self.aux, seed=c['seed'] * 7919 + it)
            added = True
            self.log(f'iter {it}: auxiliary head {self.aux} added to checkpoint {it - 1} (its other outputs unchanged)')
        readers = getattr(self, 'readers', None) or []
        r_added = False
        have = reader_layout(model.config.get('readers'))
        if have != readers:
            if have:
                raise SystemExit(f'iter {it}: checkpoint {it - 1} has the readers {have}, this run reads {readers or "none"} '
                                 f'(a reader is part of the network\'s input: keep it - reader_<target> 1, its declaration '
                                 f'unchanged - or fork from a checkpoint without it; docs/az.md 36)')
            # a checkpoint without readers (a fork of a run without them): fresh readers, the rest as it was
            model = add_reader(model, readers, seed=c['seed'] * 7919 + it + 1)
            r_added = True
            self.log(f'iter {it}: readers {[r["target"] for r in readers]} added to checkpoint {it - 1} (its outputs '
                     f'unchanged)')
        if readers:
            model.reader_detach = bool(c.get('reader_detach', 1))
        model.train()
        static = bool(c.get('train_static'))
        if static and not hasattr(model, 'forward_static'):
            raise SystemExit(f'iter {it}: train_static is for a token network (arch tokens); checkpoint {it - 1} '
                             f'is {model.config.get("arch", "mlp")!r} (set train_static 0)')
        # fused AdamW (train_static on CUDA): the same update, one kernel set for every parameter
        okw = {'fused': True} if static and str(dev).startswith('cuda') else {}
        heads = [p for n, p in model.named_parameters() if n.startswith('aux_head.')]
        rdrs = [p for n, p in model.named_parameters() if n.startswith('readers.')]
        trunk = [p for n, p in model.named_parameters() if not n.startswith('aux_') and not n.startswith('readers.')]
        # a widened head's later parts (docs/az.md 37): a group each, after the readers
        more = [[p for n, p in model.named_parameters() if n.startswith(f'aux_more.{k}.')]
                for k in range(len(getattr(model, 'aux_more', None) or []))]
        # parameter groups: the network of before, its auxiliary head, its readers (each when it has them)
        pgroups = [g for g in [trunk, heads, rdrs] + more if g]
        if len(pgroups) > 1:
            opt = torch.optim.AdamW([{'params': g} for g in pgroups], lr=c['lr'], weight_decay=c['wd'], **okw)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['wd'], **okw)
        opt_path = os.path.join(self.run, 'models', 'opt_latest.pt')
        fresh = True
        if os.path.exists(opt_path):
            try:
                o = torch.load(opt_path, map_location=dev, weights_only=False)
                if o.get('iter') == it - 1:
                    groups = o['state'].get('param_groups') or []
                    k = len(groups)
                    if more:
                        # a widened head (docs/az.md 37): each saved group is one part of this network (the
                        # trunk, the head of before, the readers, an earlier widen's part) - matched by its
                        # parameters' shapes, its moments kept; the parts it lacks start fresh
                        at = match_opt_groups(o['state'], pgroups)
                        if at is None:
                            raise ValueError(f'it belongs to another network ({opt_state_mismatch(model, o["state"])})')
                        ordered = [pgroups[j] for j in at]
                        opt = torch.optim.AdamW(ordered[0] if len(ordered) == 1 else [{'params': g} for g in ordered],
                                                lr=c['lr'], weight_decay=c['wd'], **okw)
                        opt.load_state_dict(o['state'])
                        for j, g in enumerate(pgroups):
                            if j not in at:
                                opt.add_param_group({'params': g})
                    elif 0 < k < len(pgroups) and [len(g.get('params', [])) for g in groups] == [len(g) for g in pgroups[:k]]:
                        # the state of the network before its head (and / or readers): kept for it, the new
                        # parts start fresh
                        bad = opt_state_mismatch(model, o['state'], params=[p for g in pgroups[:k] for p in g])
                        if bad:
                            raise ValueError(f'it belongs to another network ({bad})')
                        opt = torch.optim.AdamW(trunk if k == 1 else [{'params': g} for g in pgroups[:k]], lr=c['lr'],
                                                weight_decay=c['wd'], **okw)
                        opt.load_state_dict(o['state'])
                        for g in pgroups[k:]:
                            opt.add_param_group({'params': g})
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
        # a widened head (docs/az.md 37): the window's shards of its older, narrower layouts load too
        older = AUX.older_columns(self.aux, getattr(model, 'aux_older', None)) if aw else None
        bpr = (model.features * 2 + 8 * 16 + 3 * MAX_SEATS * 4 + (2 * aw + 1 if aw else 0)
               + (len(self.aux) if older is not None else 0))
        budget = int(c['ram_budget_gb'] * 1e9 / bpr)
        t1 = time.time()
        if c.get('window_cache') and self.wcache is None:
            self.wcache = WindowCache()
        kw = {'aux_width': aw} if aw else {}
        if older is not None:
            kw['aux_older'] = older
        if c.get('window_sparse'):
            kw['sparse'] = True
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
        # each reader's columns of the forward's auxiliary outputs (after the head's) and of the targets
        rcols, k0 = [], aw
        cols = {name: sl for name, _, sl in AUX.columns(self.aux)}
        tidx = {name: j for j, (name, _, _) in enumerate(self.aux)}
        for r in readers:
            rcols.append((r['target'], [[r['target'], r['width'], r['loss']]], slice(k0, k0 + r['width']),
                          cols[r['target']]))
            k0 += r['width']
            acc[f'reader_{r["target"]}'] = 0.0
            if r['loss'] == 'multinomial':
                acc[f'reader_{r["target"]}_hit'] = 0.0
        if nview:
            acc.update(value_view=0.0, value_decision=0.0)
        last = min(steps, 200)
        S = model.seats
        tokens = (np.asarray(model.present_index(), np.int64), T_STEP) if static else None
        # (a sparse window: the static training's batches travel as their non-zeros, dense on the device)
        feed = batches(W, rng, steps, c['batch'], dev, c.get('train_prefetch', 0), tokens=tokens,
                       coo=static and W.sparse)
        for step in range(steps):
            self.check('training')
            g_lr = lr * min(1.0, (step + 1) / c['lr_warmup_steps']) if fresh and c['lr_warmup_steps'] else lr
            for g in opt.param_groups:
                g['lr'] = g_lr
            idx, b = next(feed)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                if static:
                    o = model.forward_static(b['x'], b['n_players'], b['full'], b['ntok'], aux=bool(aw))
                    plog, vlog, mp = o[:3]
                    if aw:
                        ap = o[3]
                elif aw:
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
                # (a window with shards of an older layout: each target's rows, docs/az.md 37)
                tkw = {'tmask': b['aux_tmask']} if 'aux_tmask' in b else {}
                la = AUX.losses(self.aux, ap.float(), b['aux'], b['aux_mask'], **tkw)
                for name, v in la.items():
                    if self.aux_w[name] > 0:
                        loss = loss + self.aux_w[name] * v['loss']
                    terms.append(v['loss'].detach())
                # the readers: their targets' losses on their own outputs (docs/az.md 36)
                lr_ = {}
                for name, lay, rsl, tsl in rcols:
                    rkw = {'tmask': b['aux_tmask'][:, tidx[name]:tidx[name] + 1]} if tkw else {}
                    lr_[name] = AUX.losses(lay, ap[:, rsl].float(), b['aux'][:, tsl], b['aux_mask'], **rkw)[name]
                    loss = loss + self.aux_w[name] * lr_[name]['loss']
                    terms.append(lr_[name]['loss'].detach())
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
                    named.update({f'reader_{name}': v['loss'] for name, v in lr_.items()})
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
                    for name, v in lr_.items():
                        acc[f'reader_{name}'] += v['loss'].item() / last
                        if 'hit' in v:
                            acc[f'reader_{name}_hit'] += v['hit'].item() / last
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
                if widened:
                    out['aux']['head_widened'] = {'from': [t[0] for t in widened],
                                                  'added': [t[0] for t in self.aux if t not in widened]}
            if readers:
                out['readers'] = {'targets': [r['target'] for r in readers], 'detach': bool(model.reader_detach)}
                if r_added:
                    out['readers']['added'] = True
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

    def eval_chained(self):
        """The evaluation spec's games chain their episodes (docs/az.md 42: its rows are contests, and the evaluation
        collects their games too); False for every other game, whose evaluation calls are those of before."""
        got = getattr(self, '_eval_chained', None)
        if got is None:
            c = self.cfg
            spec = c.get('eval_spec') or package.active().eval_spec_of(c['game'], c['spec'])
            got = self._eval_chained = G.chained(c['game'], spec, (c.get('players') or [2])[0])
        return got

    def play_match(self, it, opp, n_games, sims, seed, opp_model=None, paired=False, model=None, games=None):
        """Checkpoint `it` against `opp` ('nn' + opp_model, a bot name or 'ext:<name>'), seats
        alternating. -> rows (score, margin, rounds, over) of the checkpoint. `paired`: paired games
        (docs/az.md 20.3); `model`: another network than checkpoint `it` (a path: an anchor's
        calibration, docs/az.md 20.1). A game that chains its episodes (docs/az.md 42) plays contests: a row is
        a contest (a match), and `games` (a list) receives the checkpoint's games (`games.episode_rows`)."""
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
            if games is not None:
                games.extend(G.episode_rows(r, me))
        return rows

    def play_matches(self, it, opps, n_games, sims, seed, paired=False, paths=None, games=None):
        """Checkpoint `it` against each of the checkpoints `opps` (n_games each, seats alternating
        within each match) in ONE driver: every opponent is its own evaluator key, so the
        checkpoint's network batches hold the leaves of all the matches. -> {opp: rows} as
        play_match. `paired`: paired games (n_games even: a pair never spans two opponents);
        `paths`: {opp: its network's path} for opponents that are not checkpoints of the run (the
        anchors, docs/az.md 20.1). `games` (a dict): {opp: the checkpoint's games} of a chained game's contests
        (docs/az.md 42)."""
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
            if games is not None:
                games.setdefault(opps[t], []).extend(G.episode_rows(r, me))
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
        pg = []
        gk = {'games': pg} if self.eval_chained() else {}       # (docs/az.md 42; else the calls of before)
        m['eval_yard'] = summarize(self.play_match(it, self.yard, c['eval_games'], c['eval_sims'], seed=7919 * it + 1,
                                                   **gk))
        m['eval_yard']['opp'] = self.yard
        if pg:
            # a chained game's contests (docs/az.md 42): the rows above are matches, these its games
            m['eval_yard']['per_game'] = G.summarize_games(pg)
        if tag != 'visits':
            m['eval_yard']['select'] = tag
        self.log(f'iter {it} vs {self.yard}: {m["eval_yard"]}')
        if c['eval_random_games'] and self.yard != 'random':
            pg = []
            gk = {'games': pg} if self.eval_chained() else {}
            m['eval_random'] = summarize(self.play_match(it, 'random', c['eval_random_games'], c['eval_sims'],
                                                         seed=7919 * it + 2, **gk))
            if pg:
                m['eval_random']['per_game'] = G.summarize_games(pg)
            if tag != 'visits':
                m['eval_random']['select'] = tag
            self.log(f'iter {it} vs random: {m["eval_random"]}')
        self.extra_evals(it, m)
        m.setdefault('phases', {})['eval_s'] = time.time() - t0

    def extra_evals(self, it, m):
        """The game package's other yardsticks of checkpoint `it` (GamePackage.evals): measured
        and logged under m['extras'][name], never fed back. One that raises does not stop the
        run (a yardstick, not the loop): its error is logged (EXTRA EVAL FAILED) and recorded as
        m['extras'][name] = {'error': ...}. One with a cadence of its own (GamePackage.evals_every,
        docs/az.md 40) runs only when it is due and is passed over silently otherwise."""
        pkg = package.active()
        for name, fn in pkg.evals_of(self.cfg['game']).items():
            if not pkg.eval_due(name, self.cfg, it):
                continue
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

        def record(k, j, rows, games=None):
            sc = sum(r[0] for r in rows)
            elo.add(res, elo.ckpt_name(k), elo.ckpt_name(j), sc, len(rows))
            elo.save_results(self.run, res, tag=tag)
            # (a chained game's contests, docs/az.md 42: the pool rates the matches; the games are logged beside them)
            pg = f' (games {sum(g[0] for g in games):.1f}/{len(games)})' if games else ''
            self.log(f'elo match {elo.ckpt_name(k)} vs {elo.ckpt_name(j)}: {sc:.1f}/{len(rows)}{pg}{elo.pool_note(tag)}')
        if c.get('elo_together'):
            for k in sorted({k for k, _ in todo}):
                js = [j for kk, j in todo if kk == k]
                gs = {}
                gk = {'games': gs} if self.eval_chained() else {}
                for j, rows in self.play_matches(k, js, c['elo_games'], sims, seed=31 + 7919 * k, **gk, **pk).items():
                    record(k, j, rows, gs.get(j))
        else:
            for k, j in todo:
                gs = []
                gk = {'games': gs} if self.eval_chained() else {}
                record(k, j, self.play_match(k, 'nn', c['elo_games'], sims, seed=31 + 7919 * k + 101 * j,
                                             opp_model=model_path(self.run, j), **gk, **pk), gs)
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
        if self.forks is not None:
            self.log(f'self-play forks (zero/forks.py): {self.forks.describe()}')
        if RS.enabled(c):
            self.log(f'self-play resignation (zero/resign.py): {RS.fingerprint(c)}')
        if c.get('eval_remote'):
            from .remote import RemoteEval
            self.remote = RemoteEval(self, c['eval_remote'])
            self.remote.backfill(start if start in done else start - 1)
            self.log(f'evaluations on the remote {c["eval_remote"]!r} ({self.remote.spec["host"]})')
        from .remoteplay import remotes_file_in_use
        if c.get('selfplay_remote') or remotes_file_in_use(run):
            # (a list of remotes, or the run directory's selfplay_remotes file - which alone turns remote
            # self-play on: docs/az.md 34)
            from .remoteplay import RemotePlay
            self.rsp = RemotePlay(self, c['selfplay_remote'])
            self.rsp.local(max(2, int(c['groups'])), int(c['conc']), device=c['infer_device'])
            self.rsp.restore(last_rates(read_metrics(run)))
            if not self.rsp.multi:
                self.log(f'self-play also on the remote {c["selfplay_remote"]!r} ({self.rsp.spec["host"]}, '
                         f'{self.rsp.procs} worker(s), mirror {self.rsp.rdir}): {c["games"]} games per iteration in '
                         f'total, in flight per machine its throughput share x {self.rsp.inflight:g}'
                         + self.rsp.batch_note() + (', lingering while the run trains' if self.rsp.linger else ''))
            else:
                src = (f'the file {os.path.join(run, "selfplay_remotes")}' if remotes_file_in_use(run)
                       else f'selfplay_remote {c["selfplay_remote"]!r}')
                self.log(f'self-play also on the remotes of {src}: {self.rsp.describe()}: {c["games"]} games per '
                         f'iteration in total, in flight per machine its throughput share x {self.rsp.inflight:g}'
                         + self.rsp.batch_note() + (', lingering while the run trains' if self.rsp.linger else '')
                         + ('' if self.rsp.local_play else '; this machine plays only while no remote worker takes '
                            'work (selfplay_local 0)'))
            if not self.rsp.local_play and not self.rsp.multi:
                self.log('self-play here only while no remote worker takes work (selfplay_local 0)')
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
            self.linger_start(it)                  # (selfplay_local_linger: self-play on while it trains)
            try:
                tr = self.train(it, sp['samples'] + sp.get('remote', {}).get('samples', 0))
            except BaseException:
                self.linger_stop(quiet=True)
                raise
            t2 = time.time()
            lg = self.linger_stop()
            self.log(f'iter {it} train: {_fmt(tr)}')
            m = {'iter': it, 'selfplay': sp, 'train': tr, 'phases': {'selfplay_s': t1 - t0, 'train_s': t2 - t1},
                 'resumed_seconds': 0}
            if lg is not None:
                m['linger'] = lg
                self.log(f'iter {it} local linger: {lg["games"]} games in {lg["seconds"]:.1f}s while training '
                         f'({lg["leaves"] / max(lg["seconds"], 1e-9):.0f} leaves/s; {lg["held"]} held for iter {it + 1})')
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


def reader_settings(cfg, aux_w):
    """The readers a run turns on (docs/az.md 36): the game package's readers whose `reader_<target>` is 1,
    in their normal form (`net.reader_layout`), [] when none is. A reader on needs its target's weight
    (`aux_<target>_w` > 0: its loss is what trains it, and only then does self-play record the target),
    else SystemExit."""
    out = [r for r in package.active().readers_of(cfg['game']) if int(cfg.get(f'reader_{r["target"]}') or 0)]
    for r in out:
        if not aux_w.get(r['target'], 0.0) > 0:
            raise SystemExit(f'reader_{r["target"]} 1 needs aux_{r["target"]}_w > 0: its target\'s auxiliary loss trains '
                             f'the reader (docs/az.md 36)')
    return out


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
    for k in list(package.active().reader_settings()) + ['reader_detach']:
        if cfg.get(k, 0) not in (0, 1):
            raise SystemExit(f'{k} {cfg.get(k)!r}: 0 or 1 (docs/az.md 36)')
    if isinstance(cfg.get('resolver_cfg'), dict) and L.enabled(cfg['resolver_cfg']):
        raise SystemExit('resolver_cfg: selfplay_league: single-game runs only (not a co-training resolver)')
    # branched games and archive starts (zero/forks.py): ranges, single-game runs, rolling self-play
    FK.check_settings(cfg)
    if isinstance(cfg.get('resolver_cfg'), dict) and FK.enabled(cfg['resolver_cfg']):
        raise SystemExit('resolver_cfg: selfplay_branch / selfplay_archive: single-game runs only (not a co-training resolver)')
    # self-play resignation (zero/resign.py): ranges, two-player single-game runs without the league
    RS.check_settings(cfg)
    if isinstance(cfg.get('resolver_cfg'), dict) and RS.enabled(cfg['resolver_cfg']):
        raise SystemExit('resolver_cfg: selfplay_resign_t: single-game runs only (not a co-training resolver)')
    for where, cc in (('', cfg), ('resolver_cfg: ', cfg.get('resolver_cfg') if isinstance(cfg.get('resolver_cfg'), dict)
                                    else {})):
        v = cc.get('selfplay_local', 1)
        if v not in (0, 1, True, False):
            raise SystemExit(f'{where}selfplay_local {v!r}: 0 or 1 (docs/az.md 34)')
        if not v and (where or cfg.get('resolver')):
            raise SystemExit(f'{where}selfplay_local 0: single-game runs only (docs/az.md 34)')
    if cfg.get('selfplay_local_linger'):
        why = ('needs selfplay_remote (the shared queue holds the games it finishes)' if not cfg.get('selfplay_remote')
               else 'single-game runs only' if cfg.get('resolver')
               else 'not with the self-play league (selfplay_league 0)' if L.enabled(cfg) else None)
        if why:
            raise SystemExit(f'selfplay_local_linger: {why}')
    if cfg.get('shard_writer_thread') or (isinstance(cfg.get('resolver_cfg'), dict)
                                          and cfg['resolver_cfg'].get('shard_writer_thread')):
        if cfg.get('resolver'):
            raise SystemExit('shard_writer_thread: single-game runs only (co-training writes its shards on the '
                             'driving thread, as before)')
        if not float(cfg.get('shard_writer_queue_mb', 0)) > 0:
            raise SystemExit(f"shard_writer_queue_mb {cfg.get('shard_writer_queue_mb')!r}: a number > 0 (MB)")
    return cfg


def belief_keys(cfg, pre):
    """The root-belief keyword arguments of `games.nn_seat` for a run's seat `pre` ('selfplay' | 'eval')
    from its settings (`mcts.belief_settings`; {} when its tilt is 0: the seat of before)."""
    from .mcts import belief_settings, belief_tilts
    tilt, clip, mix, fast = belief_settings(cfg, pre)
    tilts = belief_tilts(cfg, pre)
    out = dict(belief_tilt=tilt, belief_clip=clip, belief_mix=mix, belief_fast=fast) if tilt or tilts else {}
    if tilts:
        out['belief_tilts'] = tilts            # (per-target tilts, docs/search.md 3.8.1)
    return out


def value_choice_keys(cfg, pre):
    """The value-choice keyword arguments of `games.nn_seat` for a run's seat `pre` ('selfplay' | 'eval'; docs/search.md
    3.17; {} when its worlds are 0: the seat of before)."""
    w = int(cfg.get(f'{pre}_value_choice_worlds', 0) or 0)
    return dict(value_choice_worlds=w, value_choice_temp=float(cfg.get(f'{pre}_value_choice_temp', 0.0) or 0.0)) if w else {}


def check_eval_settings(cfg, where=''):
    """The move-rule and Elo-offset settings (eval_select, selfplay_select, their guards and their
    completed-Q transforms, the root minimum visits, elo_offsets) -> None, else SystemExit (`where`:
    a prefix, e.g. 'resolver_cfg: ')."""
    from .mcts import (belief_settings, belief_tilts, check_belief, check_leaf_view, check_root_min, check_select,
                       check_select_transform)
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
        from .mcts import check_merge
        check_merge(cfg.get('eval_infoset_merge', 0), cfg.get('eval_leaf_value_view', 'mover'), what='eval_infoset_merge')
        check_merge(cfg.get('selfplay_infoset_merge', 0), cfg.get('selfplay_leaf_value_view', 'mover'),
                    what='selfplay_infoset_merge')
        for pre in ('selfplay', 'eval'):
            tilt, clip, mix, _ = belief_settings(cfg, pre)
            check_belief(tilt, clip, mix, what=f'{pre}_belief_tilt')
            belief_tilts(cfg, pre)
            from .mcts import check_value_choice
            check_value_choice(cfg.get(f'{pre}_value_choice_worlds', 0) or 0, cfg.get(f'{pre}_value_choice_temp', 0.0) or 0.0,
                               what=f'{pre}_value_choice')
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
    from .mcts import check_cq_prior_w
    try:
        check_cq_prior_w(cfg.get('cq_prior_w', 1.0), cfg.get('policy_target', 'visits'))
    except ValueError as e:
        raise SystemExit(f'{where}{e}') from None
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
