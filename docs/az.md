# The AlphaZero loop (`zero`)

Self-play -> training -> evaluation against a yardstick -> Elo. The search runs in Rust
(docs/search.md); the Python package `zero` (`python/zero/`) owns the network, inference,
training, the loop, Elo, the dashboard and the tools. Moved here from Glass Road Zero's
`grz/az` and Caverna Zero's `cvz/az` (copies of one stack: Caverna Zero had ported it from
Dominion Zero's `domz/az` and St. Pete Zero's `stpz` and made it game-agnostic); proven end to
end on the toy game (section 6) and in both games' runs. A game plugs in as a **game package**
(docs/new_game.md): its compiled module `<pkg>._engine`, its presets, its external players
(the web apps' JS bots, `ext:jsbot`, are the Elo-1000 yardsticks), its default spec and its
dashboard port, registered with `zero.package.register(GamePackage(...))` in
`<pkg>/az/__init__.py`. The game repo's `<pkg>/az/<module>.py` files are shims, so every
command keeps its name (`python -m grz.az.run`, `python -m cvz.az.dashboard`, ...).

| File (`python/zero/`) | What |
|---|---|
| `package.py` | The game package registry: `GamePackage` (name, root, engine module, run module, presets, DEFAULTS overrides, external players, specs, decision kinds, dashboard title / port, env prefix, deferred-result glue), `register`, `active` |
| `run.py` | The loop: `DEFAULTS`, `PRESETS` (the toy's + the package's), self-play (classic or rolling), training, evaluation, Elo, resume, STOP / PAUSE |
| `cotrain.py` | Co-training (section 11): a primary game whose deferred terminals the episodes of a resolver game decide, one network each in one run: `CoRun` (the loop), `Coordinator` (parked games -> queued episodes -> results by tag -> `resolve`), `run_pair` (two drivers on two threads), `PrimaryRun` (team events as matches) |
| `net.py` | `Net`: residual MLP with LayerNorm (arch 'mlp', the default); `GridNet` (arch 'grid'): residual 3x3 conv tower over the game's board planes (`GamePackage.grids`) with globals + pooled-board channel biases, one policy logit per cell from the tower; `TokenNet` (arch 'tokens', section 10): a transformer over the game's objects read through a layout descriptor (`GamePackage.tokens`), pointer logits for the objects' actions; all: policy logits over NUM_ACTIONS (masked to the legal actions by the caller), value = P(win) per relative seat (softmax over the seats present), auxiliary margin head; atomic `save`, `load` (by the checkpoint's arch), `widen`, `make_net` |
| `infer.py` | `NetEvaluator` (CPU / CUDA; gather + segment softmax on the device; `set_weights` in place; `workers`: eager batches on threads), `UniformEvaluator`, `drive()` (external batches answered on threads; several network batches in flight) |
| `data.py` | Sample shards (`ShardWriter`, atomic npz, optionally compressed - remote self-play's; non-finite ones quarantined as `*.bad.npz`), the RAM-bounded training `Window` (each shard subsampled as it is loaded; checks every shard; row provenance), `TargetStats` |
| `finite.py` | The non-finite guard (section 4.1): vectorised NaN / inf checks, `NonFinite` |
| `games.py` | `engine()` (the package's compiled module), registered games, seat specs, external players (`EXTERNALS`, `register_external`; a package registers its own, e.g. `ext:jsbot`), the `Driver` wrapper |
| `toybots.py` | The toy heuristic as an external player (`ext:toyheur`) |
| `elo.py` | Bradley-Terry Elo over all results, the run's yardstick = 1000; one pool per evaluation move rule (`pool_dir`, `eval_tag`, `active_pool`), the Elo opponents (`opponents`: evaluations back, or raw `elo_offsets`; section 5.4) |
| `runctl.py` | PAUSE / STOP files, exit status 75, run / supervisor state files, `PauseWatch` |
| `dashboard.py` | Flask dashboard (title, port and runs directory from the game package) |
| `bench.py` | Throughput measurements (`infer`: batch sizes, concurrent workers, a checkpoint, the operators and multiply-adds of a forward pass; `train`: seconds per training step; section 10.5) |
| `distill.py` | Distillation (section 13): a new network of another size or architecture trained on a run's shards to compute what a teacher computes (`new_student`, `outputs`, `distill`, `agreement`) |
| `h2h.py` | Checkpoint vs checkpoint, seats alternating (the readout of an A/B); each side's move rule its run's `eval_select` or `--select` / `--a-select` / `--b-select` (section 5.4); `--move-log` |
| `remote.py` | Remote evaluation (section 5.2): `RemoteEval` (the run's thread: queue, ship to the remote's mirror, run, results to `evals.jsonl`, Elo back), `RemoteNetEval` (a co-training run's `eval_yard_net`, 11.7), the remote side (`python -m <pkg>.az.remote eval MIRROR ITER`; `evalnet MIRROR ITER`; `selfplay MIRROR K`: remoteplay's worker), `Transport` (ssh / scp, or `local`) |
| `remoteplay.py` | Remote self-play (sections 5.3, 11.6): `RemotePlay` (the run's sessions with persistent workers on the remote, checkpoints out, compressed shards back into `data/iter_i/` with the non-finite guard on arrival; the shared queue's caps per consumer; single-game: the shared quota, games accepted by the run; co-training: a share of the resolver's queued episodes, rows back by tag, requeue, give-back on back-off), `Worker` (the remote side: a rolling `queue_only` driver across iterations, held games, back-off while an evaluation holds its lock, busy markers, linger) |
| `workqueue.py` | The shared work queue (section 5.3): `Shares` (each consumer's in-flight cap from its measured throughput, head start and the iteration's units; a batch-bound consumer's its slots within `batch_inflight` iterations), `split_games` (a driver's sample dict cut into its games), the staleness histogram |
| `targets.py` | Policy-target sharpness and quality: visits vs completed-Q targets (section 9) |
| `league.py` | The self-play league (section 14): the league games' kinds and seat tables (classic, rolling), the iteration's picks of past checkpoints (`League.plan`: uniform or pfsp), the slots' evaluators (guarded), the games' attribution and `selfplay.league` |
| `mcts.py`, `play.py`, `parity.py`, `protocol.py` | Reference search / driver, parity, the protocol (docs/search.md) |
| `tools/` (`supervise`, `kill_run`, `pause_run`, `fork_run`, `launch_hidden`, `distill`) | Unattended runs, control, forks (`fork_run --net`: a continuation with another network, section 13.3), distillation (section 13); each repo's `tools/<name>.py` activates its package and calls `main()` |

Repository tools (`tools/`): `build_engine.sh` (the PyO3 build every repo's `build_py.sh`
calls), `build_py.sh` (the core's toy module), `smoke_all.sh` (core tests + each game's
smoke), `selfplay_digest.py` + `check_checkpoint_compat.py` (old vs new checkout: golden
digests, checkpoint compatibility; docs/pinning.md).

## 1. Running

In a game repo (the core's own toy runs: `python -m zero.run`, `python tools/supervise.py` in
zero-core with `bash tools/build_py.sh` first):

```
bash tools/build_py.sh                                        # <pkg>/_engine.pyd for this checkout (inits the pinned core)
python -m <pkg>.az.run --run runs/toy1 --preset toy-1         # in the foreground
python -m <pkg>.az.dashboard --host 0.0.0.0                   # reads runs/* (--runs DIR elsewhere); port: the package's
```

Unattended (live runs from a pinned worktree, never from a checkout being edited; the
supervisor restarts a crashed run from its newest checkpoint, backs off after 5 crashes in an
hour, keeps Windows awake, waits while paused; the hidden console never flashes a window):

```
git -C ~/<game> worktree add --detach ~/<game>-wt/run-<name> <commit>
cd ~/<game>-wt/run-<name> && CARGO_TARGET_DIR=E:/cargo-target/<game>/run-<name> bash tools/build_py.sh
python tools/launch_hidden.py runs/toy1/supervisor_console.txt python tools/supervise.py runs/toy1 --preset toy-1
touch runs/toy1/STOP                     # clean stop after the current iteration (the supervisor exits too)
python tools/pause_run.py runs/toy1      # pause within seconds (exit 75; the supervisor waits); --resume, --status, --wait
python tools/kill_run.py runs/toy1       # hard stop: the supervisor, then the run (pids from the state files; --scan)
python tools/fork_run.py runs/toy1/models/iter_0010.pt runs/toy2          # new run from a checkpoint (widened; across a token layout change: 10.3)
python tools/fork_run.py --source runs/toy1 --dest runs/toy1b --iter 10 --set reuse=8   # continue at K
python tools/distill.py runs/toy1/models/iter_0010.pt big.pt --data runs/toy1/data --hidden 256 --blocks 6   # section 13
python tools/fork_run.py --source runs/toy1 --dest runs/toy1big --iter 10 --net big.pt  # continue at K with it
```

Settings: `run.DEFAULTS` (+ the game package's overrides) < `--preset` < the run's saved
`config.json` < flags given on this command line (saved). A preset therefore only fills
settings a run has not stored yet; a run directory keeps its settings across code versions
(every key is saved).

### 1.1 Presets

* `toy-smoke`: 2 tiny iterations, the end-to-end check in tests (~15 s).
* `toy-1`: the toy proof run (section 6): 24 iterations of 600 self-play games, 64 / 16 sims
  (p_full 0.25), 128 x 3 network, batch 512, reuse 4, window 6 (growing), value_q 0.5 (warm-up
  5), yardstick `ext:toyheur` (the toy heuristic through the external-seat path, as the JS bots
  play the games), 400 evaluation games at 32 sims, 100 against random, 100 games per Elo
  match, 4 search threads.
* `toy-cotrain-smoke`: co-training on the toys (section 11): 2v2 `toy_team` events (8 resolved
  per iteration, yardstick `greedy`) whose four matches each are toy games played by the toy
  network (`resolver` 'toy', its settings the `toy-smoke` preset with the `heuristic` yardstick),
  2 iterations (~30 s).
* Game presets come with the game package (`GamePackage.presets`): Caverna Zero
  `caverna-smoke`, `caverna-2p`, `caverna-2p-cq`; Glass Road Zero `glassroad-smoke`,
  `glassroad-2p`, `glassroad-2p-cq` (each repo's docs/az.md).

### 1.2 Settings (`python -m <pkg>.az.run --help`)

| Setting | Default | Meaning |
|---|---|---|
| `game`, `spec`, `players` | toy, '', [2] | registered game, its JSON spec, player counts sampled per self-play game |
| `iters`, `games` | 40, 1000 | iterations; self-play games per iteration (rolling: games FINISHED per iteration; with `selfplay_remote`: finished across every machine, section 5.3) |
| `rolling` | 0 | 1 = one persistent driver keeps `conc` games in flight across iterations (docs/search.md 2.6) |
| `conc`, `threads`, `groups`, `torch_threads` | 256, 4, 2, 1 | games in flight, search threads, batches in flight, torch intra-op threads (keep 1, section 7) |
| `ext_groups` | 2 (Glass Road Zero: 8) | groups for matches with an external player (section 5) |
| `sims`, `sims_fast`, `p_full` | 64, 16, 0.25 | playout-cap randomisation (only `p_full` decisions are training samples) |
| `sims_minor` | 0 | simulations of the game's minor decisions in self-play (`SearchGame::minor_decision`, MTG: the land to tap), never recorded; 0 = searched like the others (docs/search.md 1, 3) |
| `c_puct`, `alpha`, `eps`, `temp_moves` | 1.0, 0.3, 0.25, 20 | PUCT constant (values in [0, 1]), root Dirichlet noise, sampled moves per game |
| `chance_k`, `margin_w`, `redet`, `max_steps` | 0, 0, 1, 100000 | sampled chance outcomes kept per edge (0 = resample every visit), utility margin credit, fresh determinization per simulation, abandon a game after this many moves |
| `policy_target`, `cq_c_visit`, `cq_c_scale` | visits, 50, 1.0 | training target of the recorded decisions: the root visit distribution, or Gumbel AlphaZero's completed-Q target (docs/search.md 3.1, section 9); the search is the same |
| `hidden`, `blocks` | 256, 4 | network size (only when the run's first network is created) |
| `arch`, `ch` | 'mlp', 64 | 'grid': a `GridNet` of `ch` channels x `blocks` over the package's board grid (Kingdom Builder: 16 planes x 20 x 20; bfloat16 autocast on CUDA for inference and training; not in zero.webexport); 'tokens': a `TokenNet` of width `hidden`, `blocks` layers over the package's object layout (section 10; not in zero.webexport) |
| `heads`, `ff` | 4, 0 | arch 'tokens': attention heads, feed-forward width (0 = 2 x hidden) |
| `batch`, `reuse`, `lr`, `wd`, `min_steps` | 512, 4, 1e-3, 1e-4, 50 | steps = new samples x reuse / batch (at least min_steps); AdamW, grad-norm clip 5 |
| `lr_halve_every`, `lr_min`, `lr_warmup_steps` | 30, 1e-4, 100 | step decay; linear warmup while the optimizer state is fresh |
| `window`, `window_grow`, `min_window_samples` | 8, 1, 20000 | newest iterations trained on (1 + iter // 2 with window_grow), more until min samples |
| `keep_data`, `ram_budget_gb`, `shard_rows` | 12, 4, 65536 | data retention (iterations), RAM for the window (uniform subsample beyond), samples per shard |
| `value_q`, `value_q_warmup`, `margin_loss_w` | 0.5, 5, 0.5 | value target blend with the root search estimate; margin loss weight |
| `yardstick` | heuristic | the Elo anchor (1000): a built-in bot or `ext:<name>` (toy: `ext:toyheur`; Caverna, Glass Road: `ext:jsbot`) |
| `eval_games`, `eval_sims`, `eval_random_games`, `eval_every` | 200, 32, 50, 1 | evaluation vs the yardstick and vs random (`eval_random_games` 0: no random match; the reference games still rate the random bot unless `ref_games` is 0) |
| `elo_games`, `ref_games` | 100, 400 | games per Elo match (checkpoints 1, 2, 4 ... 32 evaluations back, i.e. x `eval_every` iterations; or `elo_offsets`); random vs yardstick reference games (once per Elo pool) |
| `elo_opponents`, `elo_sims`, `elo_together` | 0, 0, 0 | cheaper Elo (CPU runs): play only the first k of the offsets 1, 2, 4 ... 32 (0 = all); Elo matches at elo_sims (0 = eval_sims); 1 = an iteration's Elo matches in one driver (one evaluator key per opponent: the checkpoint's batches hold the leaves of every match; other seeds than separate matches) |
| `elo_offsets` | '' | raw Elo offsets ('1,2,4,8'): each evaluated checkpoint k plays the checkpoints k-1, k-2, k-4, k-8 (those that exist) whatever `eval_every`, instead of evaluations back; `elo_opponents` is then not used; checkpoints never evaluated are rated through these games only and are never `best.pt` (section 5.4). '' = the default above, unchanged |
| `eval_select`, `eval_select_min_visits` | visits, 0 | the move rule of every evaluation seat (docs/search.md 3.3): 'visits' = argmax (N, P0) (AlphaZero's; the default: today's moves bit for bit) or 'cq' = the argmax of the completed-Q target's logits (with `cq_c_visit`, `cq_c_scale`) among the options with at least `eval_select_min_visits` visits (0 = all). The yardstick, random and Elo matches (here or on the `eval_remote`), co-training's evaluations (the resolver's: `resolver_cfg`), and by default `zero.h2h` / `zero.replay` of the run's checkpoints. Recorded in the evaluations (`select`, absent = 'visits'); one Elo pool per rule (section 5.4) |
| `selfplay_select`, `selfplay_select_min_visits` | visits, 0 | the same rule for self-play's moves after its first `temp_moves` searched decisions of a game (remote self-play too); the recorded targets stay `policy_target`'s. For a forked A/B run (section 12.4) |
| `eval_select_c_visit`, `eval_select_c_scale` | unset (None) | the evaluation 'cq' rule's own completed-Q transform: unset = `cq_c_visit` / `cq_c_scale`, the training target's (one knob for both, the behaviour before these settings, bit for bit); a number >= 0 = the rule's own, so a training A/B of `cq_c_scale` no longer changes how the evaluations choose their moves. The rule's name and Elo pool carry the effective value (`cq-n4-v50-s0.3`, `cq-n4` at 50 / 1), so a run with the scale pinned keeps its pool when only `cq_c_scale` changes. The rule 'cq' only (an error with 'visits', as the guard); `none` on the command line unsets a saved value. Every evaluation path follows it (`Run.nn_seat`: yardstick, random, Elo, the `eval_remote`, co-training's evaluations, `eval_yard_net`), and `zero.h2h` / `zero.replay` of the run's checkpoints (sections 5.4, 12.5) |
| `selfplay_select_c_visit`, `selfplay_select_c_scale` | unset (None) | the same for `selfplay_select` 'cq' (self-play's moves after `temp_moves`, remote self-play too: its checkout must have the setting); the recorded targets keep `cq_c_visit` / `cq_c_scale` whatever these are |
| `selfplay_root_min_visits`, `eval_root_min_visits`, `root_min_visits_max_actions` | 0, 0, 4 | root minimum visits ("forced playouts", docs/search.md 3.5, section 9.2): a full search whose root has 2 .. `root_min_visits_max_actions` options gives each at least k visits when its budget allows (2 x options x k <= its simulations): the self-play seat's recorded searches (never its fast or minor ones) / every evaluation seat. The forced walks come out of the same budget and are left out of the visit counts (the 'visits' target, temperature sampling, the 'visits' rule) and of the root value target; Q, the 'cq' target and the 'cq' rule use them. The evaluation's floor names its Elo pool (`visits-f4`, `cq-n4-v50-s0.3-f4`; `-f4a6` with max_actions 6), self-play's does not. 0 = off: today's searches bit for bit |
| `eval_spec` | '' | the spec of every match that records nothing (evaluation, Elo, reference games); '' = the game package's evaluation spec of the self-play spec (`GamePackage.eval_specs`; MTG: the fixed evaluation deck set), else the spec itself |
| `eval_remote` | '' | the run's evaluations (yardstick, random, Elo) on another machine while it trains: a name in the game package's `remotes` (`local` = this machine, the tests; '' = here; section 5.2). A co-training run: its resolver's `resolver_cfg['eval_remote']` (section 11.5); its own is refused |
| `selfplay_remote` | '' | self-play on another machine too, with the same checkpoint: both machines pull the iteration's `games` (the total) from one shared queue (`zero/workqueue.py`), its compressed shards join `data/iter_i/` (a name in the game package's `remotes`; '' = off; section 5.3). A co-training run: its resolver's `resolver_cfg['selfplay_remote']`, a share of the queued episodes (11.6); its own is refused |
| `selfplay_remote_procs`, `selfplay_remote_wait` | 0, 30 | worker processes on the remote (0 = the remote's `selfplay.procs`, else 1); seconds the end of a phase waits for the remote's last shards (later ones are dropped) |
| `selfplay_remote_inflight` | 1.0 | each machine's in-flight cap, in iterations of its own measured throughput (its share of the iteration's games or matches x this; 5.3) |
| `selfplay_remote_batch`, `selfplay_remote_batch_inflight` | auto, 2.0 | the batch-bound consumers (`auto`: those whose inference is CUDA / MPS - their network batches grow with their games in flight; `local`, `remote`, `all`, `none`): such a consumer's cap is its slots while its games take at most `selfplay_remote_batch_inflight` iterations (its share x this; 1 = as the others; 5.3) |
| `selfplay_remote_linger` | 0 | 1 = the workers play on with their checkpoint between phases (while the run trains); those games count for the next iteration, one checkpoint stale (5.3) |
| `selfplay_remote_giveback`, `selfplay_remote_overdue`, `selfplay_remote_timeout` | 30, 3, 3600 | co-training (11.6): a worker backing off this many seconds gives its outstanding matches back; a match outstanding on a worker longer than this many of its expected latencies (at least 120 s; 0 = the timeout only) is requeued, and at the latest after `selfplay_remote_timeout` s of the worker's running time |
| `infer_device`, `train_device`, `seed` | cpu, cpu, 0 | inference / training device (`cuda` works; the GPU is shared), run seed |
| `infer_workers` | 1 | network batches evaluated concurrently on this many Python threads, with as many in flight (give the driver as many `groups`); CPU inference of a small network (section 10) |
| `resolver` | '' | co-training (section 11): the resolver game, whose episodes decide this game's deferred terminals ('' = a single-game run: this loop, line for line) |
| `resolver_cfg` | {} | the resolver's settings (a JSON object of these same settings: spec, players, network, search, training, evaluation; `preset` names one), saved nested in config.json; resolved into `<run>/resolver/config.json` |
| `resolver_init` | '' | the checkpoint the resolver's network starts from ('' = fresh); a single-game run's checkpoint works (widened if the game grew) |
| `resolver_conc`, `pending_max` | 0, 0 | resolver episodes in flight in self-play (0 = its `conc`); the backpressure: resolver episodes queued or playing at which the primary's self-play waits (0 = 2 x in flight) |
| `draft_eval_net_every` | 4 | the primary's evaluation also plays its episodes with the resolver's network on evaluated iterations that are multiples of it (`eval_yard_net`; 0 = never); the rest of its evaluation plays them with the resolver's yardstick |
| `draft_eval_net_remote` | '' | those events (`eval_yard_net`) on another machine while the run trains: a name in the game package's `remotes` (`local` = this machine, the tests; '' = here; section 11.7); their results in the run's `evals.jsonl` |
| `selfplay_league` | 0.0 | the self-play league (section 14): this share of the self-play games (exactly, over each driver's started games) puts a past checkpoint of the run in one seat; only the current network's decisions become samples. 0 = off (every game the current network's mirror, as before). Single-game runs (co-training refuses it) |
| `selfplay_league_pool` | '2,4,8,16,32' | the past checkpoints it may play: raw iteration offsets back from the network that self-plays (iteration i self-plays with i-1: offset 2 = checkpoint i-3), those that exist; 'all' = every earlier checkpoint (fictitious self-play) |
| `selfplay_league_slots`, `selfplay_league_pick` | 1, uniform | past checkpoints per iteration (one evaluator key each: 'L0', 'L1', ...), picked from the pool every iteration: 'uniform', or 'pfsp' (weight (1 - x)^2, x = the current networks' recent score against that checkpoint, from the metrics) |

## 2. Run control (`runctl.py`)

* **STOP** (file in the run dir): the run ends after the current iteration; the supervisor
  exits with it (a stale STOP is moved to STOP.old when a supervisor starts).
* **PAUSE**: a watch thread notices the file within 0.5 s; the run stops at its next pause
  point - after every self-play / evaluation batch (`drive(check=...)`) and every training
  step - writes `run_state.json` (state `paused`) and exits with status **75**. The
  supervisor then polls PAUSE every second and relaunches the run the moment it is removed
  (not a crash: no delay, no back-off); a run started while PAUSE exists exits 75 at once.
  Measured in `tests/test_loop.py`: paused mid self-play and exited in a few seconds.
* **Refused** (exit status **78**, `runctl.EXIT_REFUSED`): the run's checkpoint is a network of
  another encoding than this build's (the encoding guard, section 10.4); state `refused`, and the
  supervisor ends instead of restarting it.
* **Resume semantics**: an iteration interrupted before its checkpoint is replayed from the
  start (its data directory is cleared); one interrupted after the checkpoint was written
  (`models/iter_N.json` holds its self-play / training stats) finishes its evaluation and Elo
  matches on restart. Self-play games in progress are dropped (the toy: seconds of work).
* `run_state.json` / `supervisor_state.json` (pid + creation time, state, timings) feed
  `runctl.run_status` (running / stopping / pausing / paused / resuming / starting /
  restarting / stopped), shown on the dashboard with Pause / Resume buttons and by
  `tools/pause_run.py --status`.

## 3. Files in a run directory

| Path | Content |
|---|---|
| `config.json` | the resolved settings |
| `log.txt`, `stdout.txt` | the run's log; console output (UTF-8) under the supervisor |
| `metrics.jsonl` | one line per iteration: `selfplay` (games, samples, decisions / rounds per game, avg score, leaves/s, games/h, inference time, average batch; `rolling` stats; `target`: kind, mean options, normalised entropy `h_norm` = H(pi) / ln(options) and `max_share` of the recorded policy targets; `remote`: the remote's share with `selfplay_remote`, and `queue`: the shared queue's quota, caps, rates and staleness histograms, section 5.3; `league`: with `selfplay_league`, section 14), `train` (policy / value / margin loss, entropy, value accuracy, steps, window samples / MB / load time, lr, value_q, samples/s), `eval_yard` / `eval_random` (win, se, margin, games; `select`: the move rule unless 'visits'), `elo` (`select`: its pool unless 'visits'), `phases` (seconds per phase), `seconds` |
| `models/iter_NNNN.pt` | checkpoints (atomic); `best.pt` = highest rated (in the current rule's pool, among the checkpoints evaluated in it: section 5.4); `index.json` = every checkpoint's Elo in that pool (`select` unless 'visits'; `"evaluated": false` for one rated only through Elo matches: raw offsets, or a rule's pool begun mid-run); `opt_latest.pt` = optimizer state (with the iteration it belongs to); `iter_NNNN.json` = stats of an iteration whose evaluation has not finished |
| `data/iter_NNNN/*.npz` | sample shards (docs: `zero/data.py`): this machine's `c00_SSS` / `r00_SSS`, a remote's `m<KK>-<sid>_SSS` (compressed; section 5.3); the newest `keep_data` iterations are kept |
| `debug/` | only after a non-finite failure (section 4.1): `iter_NNNN_<task>_SSS.bad.npz` (a quarantined shard; a remote's: `iter_NNNN_m<KK>-<sid>_SSS.bad.npz`, `remote_*.bad.npz`, section 5.3) and `nonfinite_iter_NNNN_step_SSSS.pt` (a failing training batch); never rotated or cleared by the run |
| `elo/results.json`, `elo/ratings.json` | pair results (`{"a|b": [score of a, games]}`, anchor, `logged`: the checkpoints whose evaluations are in) and the current fit: the Elo pool of the move rule 'visits' (every run's pool from before `eval_select`) |
| `elo/<rule>/results.json`, `elo/<rule>/ratings.json` | the pool of another evaluation move rule ('cq', 'cq-n2', ...; section 5.4): the same files, plus `select` (the rule) and `since` (the first iteration it rated) |
| `evals.jsonl` | with `eval_remote` (section 5.2): one line per remote evaluation `{iter, eval_yard, eval_random, elo, time, host, seconds}`, merged into the iterations' rows by the dashboard; a co-training run's resolver: `resolver/evals.jsonl` (11.5); a co-training run's primary with `draft_eval_net_remote`: `{iter, eval_yard_net, time, host, seconds, remote_s}` (11.7) |
| `progress.json` | self-play progress of the iteration in flight (dashboard); a co-training run's also has `events_resolved`, `target`, `drafting`, `pending_events`, `matches_queued`, `matches_in_flight`, `matches_done` (section 11); with `selfplay_remote`: `remote` (single-game) or `matches_remote` (co-training), sections 5.3, 11.6 |
| `resolver/` | a co-training run's resolver network (section 11): the same layout (`config.json`, `metrics.jsonl`, `models/`, `data/`, `elo/`, `debug/`, `progress.json`), iteration numbers shared |
| `run_state.json`, `supervisor_state.json`, `supervisor.log`, `PAUSE`, `STOP` | run control |

## 4. Training

Each iteration trains the previous checkpoint on the window (newest iterations' shards,
loaded into RAM; uniformly subsampled per shard beyond `ram_budget_gb` - the rate from the shards'
row counts, each shard subsampled as it is read, so loading needs the window plus one shard:
before 2026-10-28 every shard was read first, a peak of the whole window's size) for
`max(min_steps, new samples x reuse / batch)` minibatch steps (uniform with replacement).
Loss = policy cross-entropy against the recorded policy target - the visit distribution, or
the completed-Q target with `policy_target='cq'` (section 9) - (log-softmax over the legal
actions only) + value cross-entropy against `(1 - q) z + q q_root` (softmax over the seats
present) + `margin_loss_w` x 0.5 x squared margin error over the seats present. AdamW, grad
clip 5, step-decay LR, warmup while the optimizer state is fresh. Feature-width changes are
handled by prefix reads (an older network reads the first columns) and `widen` (fork_run; a token
network across a change of its object layout: section 10.3).

### 4.1 The non-finite guard (`zero/finite.py`)

A NaN or inf anywhere between self-play and a checkpoint fails the run instead of poisoning it.
MTG Zero's run mtg4 (2026-10-27) took one NaN training step at iteration 30 (bfloat16 autocast,
clip 5; the trigger is unknown: that iteration's shard had been rotated out), saved a checkpoint
whose 93 float tensors were all NaN (iteration 29's were clean, max |w| 1.23) and ran on for 10
iterations of turn-capped self-play with NaN targets and all-draw Elo matches: nothing checked.

| Where | What is checked | On failure |
|---|---|---|
| `write_shard` (self-play) | `x`, `pi`, `z`, `q`, `sc` after the float16 cast (an encoder value above 65504 becomes inf) | the shard is written to `<run>/debug/iter_NNNN_<task>_SSS.bad.npz` (meta + `nonfinite`), not to `data/`; `NonFinite` names the arrays, rows (pi: the sample rows of its option entries), columns (x; z / q / sc: seats), NaN / +inf / -inf counts and the shard's meta |
| `Window` (training) | the same arrays of every shard it loads, before subsampling: shards written before the guard too | `NonFinite` naming the shard path and the same details; nothing is moved (move the shard out of `data/`, or rename it `*.bad.npz`, to train without it) |
| every training step | the policy, value and margin losses and the total gradient norm returned by `clip_grad_norm_`, before `optimizer.step()` | the batch is saved to `<run>/debug/nonfinite_iter_NNNN_step_SSSS.pt` (the tensors fed to the model, window rows, shard paths + rows of every batch row, per-sample losses, the weights before the step, lr, value_q, amp); `NonFinite` names the iteration, the step, the non-finite terms, whether the batch inputs and the weights are finite, and the shard rows of the batch rows whose per-sample loss is non-finite (all of the batch's shards when only the gradient is) |
| `net.save`, the optimizer state | every float tensor (one sync per device) | `NonFinite` before either file is written (the optimizer state is checked first) |

`NonFinite` is an uncaught exception: the run logs `run aborted: NonFinite: ...` to `log.txt`,
writes `run_state.json` state `aborted` and exits with status 1 (traceback on stderr). The
supervisor treats it as a crash: it restarts the run after 60 s (15 minutes once 5 crashes fall
within an hour), and the restart replays the failed iteration from its self-play (no checkpoint
was written), so a deterministic cause fails again, once per replayed iteration, until someone
looks. A quarantined shard and the debug batches stay in `debug/`: `keep_data` rotation and an
iteration replay never delete `debug/`, and they move a `*.bad.npz` found in a data directory
there instead of deleting it. The window never loads `*.bad.npz` (`shard_paths` skips it).

Cost (desktop, 2026-10-27): the shard check runs on the float16 bits in 1M-element chunks, about
6.5 GB/s on one core: 0.70 s for a real-sized MTG shard (65,536 rows x 33,750 features, 4.43 GB;
about 11 us per sample) on top of a 5.7 s write, and 0.72 s on top of a 3.3 s read (the Window
loads that shard in 6.5 s). The step check is one device sync: about 12 us on the CPU, about
0.1-0.2 ms on CUDA (+2.4% of a 9 ms MLP step on the shared GPU; less for larger networks). A
normal run is bit-identical (toy-smoke, 3 iterations: the same losses, checkpoints and optimizer
state as before the guard). Tests: `tests/test_nonfinite.py`.

## 5. Evaluation and Elo

Every `eval_every` iterations the checkpoint plays `eval_games` against the yardstick and
`eval_random_games` against the random bot (seats alternating, network at `eval_sims`); then
Elo matches against the checkpoints 1, 2, 4, 8, 16, 32 evaluations back (`elo_games` each);
the random bot is rated once against the yardstick (`ref_games`). All results are refit by
Bradley-Terry maximum likelihood (weak Gaussian prior) with the yardstick at 1000 after every
iteration; `models/index.json` and `best.pt` follow the fit. The network seats search with
`eval_sims`, no noise, no temperature, and choose their moves by `eval_select` (section 5.4).

Matches with an external player (the JS bot) run with `ext_groups` groups, and `infer.drive`
answers the external batches on worker threads while the driver searches and evaluates the
other groups (docs/search.md 2), with the same games as answering them inline (tested).
Measured in Caverna Zero 2026-10-12 with `runs/cvz2p` iteration 167, the run's evaluation settings (200 games
vs `ext:jsbot`, 64 sims, chance_k 2, CUDA inference on the GPU shared with two training runs),
3 search threads, 4 JS-bot processes:

| drive loop | groups | seconds | network calls | average batch | inference s |
|---|---|---|---|---|---|
| inline (before) | 2 | 129.2 | 20,445 | 57.6 | 53.0 |
| async | 2 | **110.2** | 20,445 | 57.6 | 57.2 |
| async | 4 | 127.7 | 38,556 | 30.6 | 105.2 |
| async | 8 | 186.0 | 67,858 | 17.4 | 178.6 |

Every configuration played the same 200 games (scores, steps and bot moves identical; 197/200
wins). Caverna's JS bot is fast (~18 ms per decision), and a CUDA call costs ~2.6 ms whatever
its batch on the shared GPU, so more groups cost more in inference calls than they hide in bot
latency: `ext_groups` 2, the core default. (Glass Road Zero's bot takes ~0.1 s per decision: there
8 groups cut 200 games from 12.4 to 3.75 min, so its package sets `ext_groups` 8.)


### 5.1 Dashboard (`python -m <pkg>.az.dashboard --host 0.0.0.0 [--port P]`)

One per game (hub projects `caverna_zero` :5260, `glassroad_zero` :5280: the package's
`dashboard_port`; title from the package; runs from `<repo>/runs` or `--runs`). Run selector; run state with Pause / Resume; status line
(self-play progress, last log line age, STOP); tiles (iteration, newest / best Elo - the best an
evaluated checkpoint's -, win rate vs the yardstick and vs random, self-play games, leaves/s);
charts: Elo by iteration (yardstick and random reference lines; the current evaluation rule's
pool, and the run's other pools as their own lines: section 5.4), win rate vs yardstick and
random, each checkpoint vs the previous
one, losses, self-play average score, leaves/s, minutes per iteration by phase; head-to-head
table of the newest checkpoint with the Elo prediction; iteration table; log and supervisor
tails. With no runs it shows how to start one. Endpoints: `GET /api/runs`,
`GET /api/run/<name>`, `GET /api/run/<name>/control`, `POST /api/run/<name>/pause|resume`
(JSON body required; only plain names of run directories under `--runs`). A co-training run
(section 11.3a): a tab per network (`/api/run/<name>` adds `resolver`: its config, metrics, Elo),
the co-training progress and tiles; a run whose metrics hold `extras` (`GamePackage.evals`):
an "Other yardsticks" chart. Single-game runs render as before. Remote evaluations (5.2): each
network's `evals.jsonl` is merged into its rows (`resolver/evals.jsonl` into the resolver's tab).

### 5.2 Remote evaluation (`zero/remote.py`, run setting `eval_remote`)

`eval_remote <name>` (a remote of the game package, `GamePackage.remotes`; `local` = this machine,
the tests) moves a run's evaluations - the yardstick and random matches, the package's other
yardsticks, the Elo matches and reference games - to another machine while the run trains
(Kingdom Builder Zero's and MTG Zero's runs evaluate on the MacBook this way).

* **Local side** (`RemoteEval`, a thread of the run process): `finish_iter` appends the
  iteration's metrics row without an evaluation and queues each evaluated iteration
  (`eval_every`); the thread ships what the evaluation needs to the remote's **mirror** of the
  run, `<workdir>/<run name>` (config.json: the run's settings with `eval_remote` '' and the
  remote's `overrides`; every checkpoint the Elo matches may play, each copied under a temporary
  name and renamed; for a new mirror the run's `elo/results.json` and `metrics.jsonl`, so the
  remote continues a run evaluated here so far), runs `python -m <pkg>.az.remote eval <mirror>
  <iter>` in the remote's `repo` (ssh, `BatchMode`), appends the result to the run's
  **`evals.jsonl`** (`{iter, eval_yard, eval_random, elo, time, host, seconds}`) and copies the
  mirror's `elo/results.json`, `elo/ratings.json` and `models/index.json` back: the remote owns
  the run's Elo while `eval_remote` is on (`models/best.pt` is not copied back).
* **Remote side** (`zero.remote.main`): `Run.evaluate` + `Run.elo_update` on the mirror (the
  run's own code), with the remote's `<workdir>/overrides.json` (if any) over the mirror's config;
  **one evaluation at a time per workdir** (the lock directory `<workdir>/.eval.lock`, freed when
  its owner is dead or after 4 hours); the result printed as one line `ZERO_REMOTE_RESULT {json}`.
  A checkpoint of another encoding than the remote checkout's is refused there (the encoding
  guard, 10.4): exit 78 and one line `ZERO_REMOTE_REFUSED {"error": ...}`, logged here in full.
* **A failure never blocks training**: an evaluation that fails (the remote asleep or
  unreachable, its checkout refused) is logged (`remote eval <name> iter N: ...; retry in 300s`)
  and retried every `RETRY_S` (300 s) on the thread; later iterations queue behind it.
* **Restart**: the evaluated iterations without a result (no `eval_yard` in metrics.jsonl, no row
  in evals.jsonl) are queued again. A run that has finished its iterations waits for its queue
  (progress phase `remote evaluations`, at most 6 hours) unless STOP exists.
* **On or off at a restart**: `--eval-remote <name>` (the evaluations made here so far are kept:
  a new mirror starts from them) or `--eval-remote ''` (the run evaluates here again, from its
  `elo/` as the remote last left it).
* **A remote**: `{host: 'user@host' | 'local', repo, python, workdir, env: {...}, overrides: {run
  settings}}`; a package may give a Mapping that resolves when the run starts (MTG Zero's
  `RunCommitRemote`: the remote's checkout of exactly the run's commit, checked before every
  evaluation).
* **The move rule** (section 5.4): the mirror's config.json carries `eval_select`, so the remote
  plays the run's rule with the run's code; its result's `eval_yard.select` must name the run's
  rule ('visits' when absent), else it is refused before anything is recorded (an older checkout
  that ignores the setting; logged and retried like any failure). The copies back are the pool's
  (`elo/` or `elo/<rule>/`); a rule's new pool on an existing mirror keeps the mirror's
  metrics.jsonl (only a mirror without one gets the run's). With `elo_offsets` every raw-offset
  opponent is shipped too.
* **Co-training runs**: the resolver's evaluation on a remote, section 11.5.

Tests: `tests/test_loop.py` (`test_remote_evaluation_local`, `..._continues_a_local_run`),
`tests/test_select.py` (the rule on a remote, a refused result, raw offsets shipped) and
`tests/test_cotrain.py` (11.5).

### 5.3 Remote self-play (`zero/remoteplay.py`, `zero/workqueue.py`, run setting `selfplay_remote`)

`selfplay_remote <name>` (a remote of the game package, the same `GamePackage.remotes` entry as
`eval_remote`; `local` = this machine, the tests) gives a run a second producer of each
iteration's samples: this machine and the remote's workers **pull the iteration's work from one
shared queue**, all searching with checkpoint i-1, and the remote's sample shards come back into
`data/iter_i/`. MTG Zero's MacBook self-plays 3-6 times as fast as the shared desktop (its
ENGINE.md 50.3, 52, 56). A co-training run gives its resolver's queued episodes to the remote
instead (11.6), through the same queue model. A worker whose checkout reads another encoding
than the run's checkpoints refuses them (the encoding guard, 10.4): the run logs `worker <k>
REFUSED the run's checkpoint ...` and plays on alone.

* **One pull model** (`zero/workqueue.py`, since zero-core's `feat/shared-queue`; it replaces the
  first design's "this machine's quota + whatever the remote finished"). An iteration needs `units`
  of work in total: a single-game run's **`games` - the finished games of the iteration across
  every machine** - or a co-training run's matches (its `games` events x their matches, 11.6). Each
  **consumer** - this machine's driver (`local`) and every remote worker (`w0`, `w1`, ...) - takes
  work as it has room under its **in-flight cap**, what it finishes in one iteration's running time:

  ```
  cap = clamp(ceil(inflight x rate x (lead + P)), lo, hi)      P = (units - sum(rate x lead)) / total rate
  ```

  `rate`: the consumer's measured throughput, units per second of its running time (back-offs
  excluded), an EWMA over its phases (weight 0.5), estimated as its leaves per running second over
  the iteration's leaves per finished unit - so a consumer whose games outlast a phase is measured
  too, slow, and capped at its floor - and, when it had less work than its cap (co-training's
  supply), scaled as if it had kept its cap busy (at most x4); `lead`: its head start (a remote's
  phase begins when the checkpoint is written, before this machine's: an EWMA of the seconds it
  played before this machine's phase; this machine's 0; co-training 0); `P`: the phase they all play
  together; `total rate`: over the consumers taking work now - **a worker backing off, offline or
  not yet up drops out, its cap is 0 and the others' shares grow at once** (this machine alone: the
  whole quota, up to its slots); `inflight`: the run setting `selfplay_remote_inflight` (default 1:
  one iteration of its own throughput, so a game it starts finishes within about one iteration);
  `lo` = 2 x its driver groups (a few games per group: at least 2), `hi` = its driver's slots
  (`conc`; a co-training worker: `conc` + `conc` / 4). The caps sum to `inflight x units` when no
  bound binds. **Bootstrap**: an unmeasured consumer's rate is proportional to its slots (`hi`,
  scaled by the measured consumers' rate per slot), and an unmeasured remote gets a quarter of that
  share (`workqueue.BOOT`) until its first measurement: its phase's end, or 10 s of its status lines
  in its first phase (`PROVISIONAL_S`, env `ZERO_RSP_PROVISIONAL_S`). A restart takes the rates and
  head starts from the run's last metrics row (`selfplay.queue.rates`; a row of the first design,
  `selfplay.remote` without `queue`: this machine's games per second and the remote's per worker).
* **Batch-bound consumers** (`selfplay_remote_batch`: `auto`, the default, = the consumers whose
  inference runs on CUDA / MPS - MTG Zero's desktop driver; `local`, `remote`, `all`, `none` choose
  explicitly): such a consumer's network batches hold its games' leaves and a network call costs
  about the same whatever its batch, so its leaves per second grow with its games in flight and its
  games take about as long at any in-flight. The cap above then sets its throughput, not its
  latency: a cap that starts small measures a small rate, which gives a small cap again, and it
  settles only where the consumer's games last one phase - a point that moves with the games'
  length. **MTG Zero's delver1 on 213a9c0** (its ENGINE.md 56): the desktop's cap wandered between 66
  and 131 of its 192 slots (2 groups: batches of half its in-flight) at 2.0-5.3k leaves/s - ~35
  leaves/s per game in flight at any in-flight - against 6.1-6.8k with all 192 before the queue; it
  made 8-17% of the 800 games. Its cap is therefore its **slots** while its games finish within
  `selfplay_remote_batch_inflight` (default 2) iterations:

  ```
  cap = clamp(ceil(max(inflight x share, min(slots, batch_inflight x share))), lo, hi)      share = rate x (lead + P)
  ```

  (by Little's law a consumer's in-flight over its rate is its games' latency, so `batch_inflight x
  share` is the in-flight at which they last `batch_inflight` horizons). `slots`: its driver's
  `conc`; a co-training consumer's `hi` adds a queue beyond them (a match waiting there only waits),
  which the batch rule never fills. A batch-bound consumer whose games outlast `batch_inflight`
  horizons even so falls to `lo` as any other (a GPU too slow for the run's iterations does not fill
  the data with stale rows), and a consumer that is not batch-bound keeps the formula above exactly
  (CPU inference: its rate is its cores', nearly whatever its in-flight; delver1's Mac workers lost
  ~13% of their leaves per second at 45% of their slots, 29.5k -> 25.5k, so more in flight would buy
  them little and cost the staleness of 80% of the data). Not a fixed floor (no guard against a
  consumer too slow for the iterations), nor a curve fitted to each consumer's latency against its
  in-flight across phases (it needs phases at different caps and is noisy when the games' length
  changes between iterations): the device says which kind a consumer is, `batch_inflight` bounds its
  staleness. **The cost**: a batch-bound consumer's rows lag by up to `batch_inflight` checkpoints
  (delver1's desktop at its 192 slots: games of ~1.15 phases, its rows' mean lag ~0.6 -> ~0.7); with
  it a small part of the data, the run's mean lag moves by hundredths. **Choosing `games`**: a
  consumer's latency in phases is its in-flight over its games per phase, so a batch-bound one keeps
  its slots while it plays at least `slots / batch_inflight` games an iteration: `games >= slots /
  (batch_inflight x its share of the throughput)` (delver1: 192 / (2 x ~0.2) ~ 480 - its 800 leave a
  margin of 1.7x; a remote's back-off only raises the desktop's share). Below that the desktop holds
  fewer than its slots (still `batch_inflight` x its share, not 1x).
* **Single-game: the shared quota.** Both sides play a rolling `queue_only` driver (docs/search.md
  2.9) - this machine's whatever `rolling` says - and queue one setup of the run's spec for every
  game they may still start under their cap. **A finished game counts only once the quota accepts
  it**: this machine's at once, a worker's in answer to its `fin` line (`accept`); the iteration
  ends when `games` are accepted, **exactly**. The games finishing beyond the quota (while it filled,
  or while the end reached a worker) **wait, with their samples, for the next iteration**, whose
  quota takes them first (the driver hands out each game's rows and samples together, so a game's
  samples are cut out of a batch with it: `workqueue.split_games`). A worker never holds a share of
  the quota: one that backs off, sleeps or dies leaves the rest to this machine, whose cap is then
  the whole quota. The quota met (by anyone, also during a worker's head start), the workers get
  `end` at once; the games in flight carry over on both machines. A worker that dies with accepted
  games whose samples are in no announced shard yet gives them back to the quota (`lost`) while
  this machine still plays; a shard lost after the phase (late, failed) leaves the iteration short
  by its games (`short`, logged). Training's steps follow the new samples of both (`new samples x
  reuse / batch`).
* **Sessions** (local side, `RemotePlay`: a manager thread, a fetch thread per worker, three pipe
  threads per session; the main thread only calls `begin(it)`, `accept_local` and `end(it)`): `procs`
  persistent worker processes on the remote, each a session `ssh host 'cd repo && <env> exec python
  -m <pkg>.az.remote selfplay MIRROR K'` (BatchMode, ConnectTimeout 20, ServerAliveInterval 15 x 4;
  'local': a subprocess). The worker's stdin is the control channel (JSON lines: `phase`, `end`,
  `cap`, `accept`, `enqueue`, `fetched`, `ping`, `quit`), its stdout its events (lines `ZERO_SP
  {json}`: `hello`, `status`, `fin`, `shard`, `row`, `end`, `error`). The mirror `<selfplay
  workdir>/<run name>` gets `config.json` (the run's settings + the remote's self-play `overrides`;
  shards of at most 8192 samples unless they set `shard_rows`) and `selfplay.json` (mode, locks,
  niceness, linger, the run's id) at each session start, and `models/iter_{i-1}.pt` at each phase
  start (a temporary name, then renamed). The manager recomputes the caps every tick (0.25 s) and
  sends a single-game worker its own (`cap`) when it changes. A session that fails is started again
  after `RETRY_S` (60 s) - immediately if it ran for more than 2 minutes - one that is silent for
  `STALL_S` (300 s) is killed; failures are logged the first three times in a row, then every 10
  minutes. The workers live as long as the run's process: PAUSE, STOP, an error and the end of the
  iterations close them (`quit`, stdin closed, killed after 10 s); a run killed hard closes their
  stdin by dying, and a worker that hears nothing for `watchdog_s` (180 s; the run pings every 20
  s) exits by itself.
* **A phase**: `begin(it)` -> each live worker gets `phase it` (after checkpoint i-1 is in the
  mirror) and, single-game, an `accept` for the games it held: its rolling driver - persistent
  across iterations, games in flight carry over a phase boundary, a new search epoch per checkpoint,
  every sample stamped with its search's `net` - resumes with the new weights. **The remote's phase
  i starts as soon as checkpoint i-1 is written** - right after training, before this machine
  evaluates it - and this machine's own phase i calls `begin(it)` again (a no-op; `lead_s` records
  the head start): the remote plays through the evaluation phase with the checkpoint of the next
  self-play, on-policy, and its games count for iteration i's quota; not after the last iteration,
  nor with STOP or PAUSE there, and a restart replays the iteration from an empty data directory.
  `end(it)` (or the quota met) -> `end it`: the worker ends its phase (`end_phase`; games in flight
  wait for the next phase), flushes its last shard and reports its counters. Shards are written by
  a background thread of the worker (`data/iter_i/m<KK>-<sid>_SSS.npz`, `sid` the session's id),
  **compressed** (zlib level 1, `data.savez_level`: MTG's sparse features ~58x), each announced on
  stdout as soon as it is written with the games whose samples it holds (whole games: a shard is the
  writer's whole buffer).
* **Linger** (`selfplay_remote_linger` 1, opt-in): after `end it` a worker closes the phase's
  accounting (its last shard, its `end` line) and plays on with checkpoint i-1 while the run trains
  and evaluates; the games it finishes then are held (co-training: their rows go to the coordinator
  at once, their samples are held) and count for iteration i+1, into whose shards their samples go:
  those rows are one checkpoint behind the iteration's (lag 1); new games start only while its held
  and outstanding games stay under its cap. The next `phase` ends the lingering at once.
* **Arrival**: fetch threads (one per worker) copy each announced shard of the running phase under a
  temporary name (`*.tmp.npz`: never a loadable shard name), check it - **the non-finite guard**
  (4.1) and the run's game / features / actions - and rename it into `data/iter_i/` beside the
  run's own `r00` shards, then tell the worker to delete its copy. Only while phase i accepts: a
  shard announced or still in flight after `end(it)`'s deadline is dropped (`late`, or `late_prev`
  when it was announced after the phase closed), never moved into another iteration. **Staleness:
  same phase only** - a shard of phase i holds the games the run accepted for phase i, searched
  with checkpoint i-1 and (carried over) earlier ones, exactly what the run's own rolling driver puts
  into `data/iter_i/`. Every arrived shard's `net` column goes into the iteration's **staleness
  histogram** (rows by `(i - 1) - net`). A **non-finite shard** is quarantined to
  `<run>/debug/iter_NNNN_<name>.bad.npz` and the run fails with NonFinite at the end of the phase
  (before training: the window never sees it), as for a local one; a worker whose own write check
  fails (its ShardWriter quarantines the shard in the mirror's `debug/`) reports it, the run fetches
  that copy into `<run>/debug/remote_*.bad.npz` and fails the same way. A shard of another game or
  encoder width (not the run's code) is rejected and logged.
* **Remote evaluation comes first**: a worker looks at its locks every 2 s - its remote's own
  `<workdir>/.eval.lock` (zero.remote's, 5.2) always, and the `locks` of the remote's `selfplay`
  dict (MTG Zero: the other games' evaluation locks on the Mac) - and while one is held by a live
  process (or is being created) it asks for no leaf batches, so its search and inference threads
  idle; control lines are still answered (an `end` ends the phase at once). With the remote's
  `selfplay.backoff` 'background' (macOS) it plays on instead under the Darwin background policy
  (`setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG)`: the lowest CPU band, the efficiency
  cores; the process sets and clears it itself) and leaves it when the lock is gone (MTG Zero
  measured both, its ENGINE.md 52.4); elsewhere 'background' is 'stop'. Its running time (`active_s`)
  stops meanwhile either way; `backoff_s` is reported per phase. **The queue gives its work back**:
  a worker whose status says it backs off is no consumer meanwhile (cap 0, the others' shares grow
  at once; single-game it holds no share of the quota anyway; co-training: its matches flow back
  after `selfplay_remote_giveback` s, 11.6). While a phase runs and it is not backing off, a worker
  holds a **busy marker** `<workdir>/.busy-<mirror>-w<K>` (an `owner` file "<pid> <time>" like the
  eval lock's), so that lower-priority jobs on the remote can yield to self-play (MTG Zero's fuzz
  farm and studies: `tools/mac_backoff.py`).
* **One worker per slot**: a worker takes `<mirror>/.w<K>.lock` (owner "<pid> <time> <run id>");
  a live previous worker of the same run (an orphan of a broken session) is stopped first
  (SIGTERM, then SIGKILL after 10 s), another run's is refused (two runs with the same mirror
  name). At each phase start a worker deletes its slot's leftover shards (late, dropped, or an
  earlier session's) and checkpoints older than i-3.
* **The remote spec** (`GamePackage.remotes[name]`, 5.2) may carry `selfplay`: `{workdir`
  (default the remote's `workdir`; `$ZERO_SELFPLAY_WORKDIR` on the run's machine overrides it),
  `procs` (default 1; the run setting `selfplay_remote_procs` overrides it), `nice` (the workers'
  niceness, default 0), `locks` (lock directories to back off for), `backoff` ('stop', the
  default, or 'background'), `env` (over the remote's),
  `overrides` (run settings for the workers: `conc` - their caps' ceiling -, `threads`, `groups`,
  `infer_workers`, `torch_threads`, `infer_device`, `shard_rows`)`}`. A spec that is a Mapping
  resolving at run start (MTG Zero's `RunCommitRemote`: the remote's checkout of exactly the run's
  commit) is read when the run starts, and the remote command checks it again (MTG Zero: `python -m
  mtgz.az.remote selfplay` refuses other code than the run's, as `eval` does).
* **Settings** (section 1.2): `selfplay_remote`, `selfplay_remote_procs` (0), `selfplay_remote_wait`
  (30 s: how long the end of a phase waits for the last shards), `selfplay_remote_inflight` (1.0),
  `selfplay_remote_batch` ('auto') and `selfplay_remote_batch_inflight` (2.0: a batch-bound
  consumer's games in iterations; 1 = as the others), `selfplay_remote_linger` (0), and co-training's `selfplay_remote_giveback` (30 s),
  `selfplay_remote_overdue` (3) and `selfplay_remote_timeout` (3,600 s, the ceiling). **An existing
  run's `games`** means the total of both machines from its first iteration on this code (the first
  design's runs played `games` here and the remote's extras on top): a run moving from that design
  sets `--games` to the total it wants (MTG Zero's delver1: its ENGINE.md 56); a run without a
  remote is unchanged.
* **Metrics**: `selfplay.games` stays this machine's (its accepted games), `selfplay.remote` = `{host,
  procs, workers` (live at the end), `in_phase, ended, games` (single-game: its accepted games whose
  samples arrived; co-training: rows), `games_played, accepted, held, lost, samples` (arrived),
  `samples_played, shards, bytes` (compressed, shipped), `leaves, leaves_per_s` (per worker: leaves /
  its phase's wall time, summed), `leaves_per_running_s` (back-offs excluded), `seconds, running_s,
  backoff_s, in_flight, late, late_prev, rejected, quarantined, fetch_failed, fetch_s, wait_s`
  (how long `end` waited), `session_errors, lead_s` (how long the remote's phase ran before this
  machine's), `caps` (each consumer's mean cap in the phase), `linger` (`{games, seconds}` of the
  lingering before it)`}` and **`selfplay.queue`** = `{quota, games` (all accepted), `short, held`
  (`{local, remote}`: waiting for the next iteration), `inflight, units, consumers` (per consumer:
  `games, share, units` (finished), `run_s, util, rate_measured, rate, cap` (the phase's mean),
  `cap_max, cap_next`), `batch_inflight`, `rates` (the model's state: each consumer's rate, lead, lo,
  hi, and `batch` true with its `slots` for a batch-bound one; the leaves per unit),
  `staleness` (`{local, remote, all}`: rows by checkpoint lag, `summary` = rows, mean, max and the
  share at lag <= 1)`}`. The log: `iter N remote self-play: 2 worker(s), 812 games, ...; 815 played,
  3 held for the next iteration` and `iter N shared queue: quota 600: 600 games (local 54, w0 271, w1
  275); caps local 56.0 -> 55, w0 273.0 -> 274, ...; rates/s local 0.223, w0 1.1, ...; staleness:
  mean 0.41, max 2, 99.6% <= 1 (52800 rows)`; at the start `self-play also on the remote 'mac' (...):
  800 games per iteration in total, in flight per machine its throughput share x 1; batch-bound
  (local): its slots while its games take at most 2 iterations`. progress.json during self-play: `games` (accepted of
  the quota), `target`, `local`, `remote` = `{host, workers, state` (playing / backing off / idle /
  connecting / offline), `games, in_flight, samples, shards, caps, quota, held}` (the dashboard's
  status line shows it); phase `remote shards` while `end` waits.
* **Measured** (the toy, desktop 2026-10-28, the remote 'local': a worker process on the same CPU,
  slower than the run's own driver; 192 games an iteration, 768 slots each, 64 / 32 simulations,
  half of the decisions full; `selfplay_remote_inflight` 1 against 1000, which caps every consumer
  at its slots - the first design's in-flight):

  | iteration | 2 | 3 | 4 | 5 | 6 |
  |---|---:|---:|---:|---:|---:|
  | **shared queue**: mean lag / max / rows at lag <= 1 | 0.29 / 1 / 100% | 0.34 / 2 / 99.2% | 0.37 / 2 / 98.3% | 0.39 / 2 / 99.2% | 0.44 / 2 / 98.6% |
  | its games here / on the worker; caps | 160 / 32; 164 / 29 | 136 / 56; 156 / 37 | 140 / 52; 149 / 44 | 118 / 74; 142 / 51 | 109 / 83; 131 / 62 |
  | **every slot full**: mean lag / max / rows at lag <= 1 | 0.84 / 1 / 100% | 1.63 / 2 / 21.8% | 1.92 / 3 / 37.6% | 2.30 / 4 / 32.2% | 2.89 / 5 / 24.5% |
  | its games here / on the worker | 192 / 0 | 192 / 0 | 56 / 136 | 0 / 192 | 0 / 192 |

  With every slot full the 1,536 games in flight are 8 iterations' worth: the rows drift to a mean
  of nearly 3 checkpoints behind, and the worker, whose phase starts early, fills the whole quota
  before this machine's phase begins. With the shared queue 98-100% of the rows are within one
  checkpoint (the few at lag 2 are games longer than a phase: `inflight` 1 sizes the mean latency
  to one iteration); the worker started at a quarter of its slot share and grows by its measured
  throughput (its caps 29 -> 62). A worker 10 ms per batch slower (its games longer than 7 phases)
  is measured at 3% of this machine's rate and capped at 5-6 games; this machine plays the quota.
  **Linger** (the same, training made ~70% of the iteration with `--min-steps 1500`): the worker's
  share grows from 30-70 to 54-95 of the 192 games, the self-play phases are ~25% shorter (0.7-1.0
  against 1.0-1.3 s), the rows' mean lag 0.54-0.67 against 0.28-0.44 (94-98% at lag <= 1): worth
  it when training is a large part of the iteration and the remote a large part of the self-play.
  **Batch-bound** (`tests/test_workqueue.py`'s simulation of delver1's consumers: the desktop's games
  180-200 s at any in-flight, each of two Mac workers 1.85 games/s at any in-flight, 800 games): the
  cap of 213a9c0 settles at 111-116 of the desktop's 192 slots whether it starts at 75 or at 181
  (where its games last one phase), at ~0.6 games/s; batch-bound it is at 192 one phase later and
  stays, at 0.96 games/s (1.6x), its games 1.14 phases long, the phase 191 -> 175 s, the workers'
  caps one phase of their throughput either way. The toy pair above at 256 slots each with
  `--selfplay-remote-batch local` (this machine's CPU driver taken for batch-bound): its cap is its
  256 slots from iteration 2, the rows' mean lag 0.41-0.65 (max 2, 91-100% within one) against the
  pair's 0.29-0.44 - what the rule costs a consumer that is not batch-bound (twice its latency for
  no throughput), which is why `auto` goes by the device.
* **A remote workdir must be an absolute path there**: a relative one (Git Bash turns
  `ZERO_SELFPLAY_WORKDIR=/Users/...` into `C:/Program Files/Git/Users/...` for a Windows process;
  set `MSYS2_ENV_CONV_EXCL=ZERO_SELFPLAY_WORKDIR`) is refused at start, before ssh or scp could
  create it under the remote's home.
* **Determinism**: none beyond the run's own. A worker's driver seed is `seed x 7919 + i0 +
  7,777,777 x (K + 1)` (i0: the iteration its session started in), so its games' setups and
  search streams are reproducible per session, but what it contributes to an iteration depends
  on timing (when its session started, its caps, back-offs, which games the quota accepted first),
  and a remote's arithmetic may differ from this machine's at float rounding. The data of a run with
  a remote depends on the remote's load; evaluations stay comparable (fixed pools, the same
  players). A run without a remote plays exactly as before (the shared queue is not in its path;
  tested byte for byte, 5.3's tests and `tests/test_loop.py`). A replay of a remote game: its sample
  rows carry `net`, its shard's meta the worker and session; the game itself is not recorded.
* **On or off at a restart**: `--selfplay-remote <name>` / `--selfplay-remote ''` (saved), with
  `--games` the total wanted per iteration.

Tests: `tests/test_remoteplay.py` (single-game: the shared quota exact with shards in `data/iter_i/`
and training's steps, the caps of a slow / fast pair following the model and the rows within one
checkpoint, a batch-bound driver at its slots (`selfplay_remote_batch` 'local'; a bad value
refused), a worker slower than a phase capped at its floor, the quota moving here while the worker
backs off, a dying worker's accepted games played here, linger, late shards dropped, an unreachable
remote, a non-finite shard quarantined, the back-off while `.eval.lock` is held, PAUSE mid-phase and
resume with no worker left; co-training 11.6, a batch-bound local resolver driver; the
coordinator's second consumer, `lock_held`, compressed shards), `tests/test_workqueue.py` (the caps'
model, a consumer dropping out, the rates' EWMA and utilisation, restore, the batch-bound rule -
delver1's desktop curve, the spiral of 213a9c0 against it, a batch-bound consumer too slow for the
iterations at its floor, the formula's bounds and `slots`, a back-off -, `split_games` on a sample
dict and on the driver, the staleness histogram, the coordinator's reserve and matches per event),
`tests/test_net_data.py` (the streamed window).

### 5.4 The evaluation move rule and its Elo pools; raw Elo offsets; no random games (2026-10-28)

**The rule** (`eval_select`, `eval_select_min_visits`; docs/search.md 3.3): how every evaluation
seat chooses its moves - 'visits' (argmax (N, P0), the default) or 'cq' (the completed-Q
target's argmax, optionally among the options with at least `eval_select_min_visits` visits).
`Run.nn_seat` builds every evaluation seat (the yardstick and random matches, the Elo matches -
both seats -, the reference games have none, co-training's primary and resolver evaluations,
`eval_yard_net`), so they all follow it, here and on an `eval_remote` (its config is the run's;
section 5.2). `zero.h2h` and `zero.replay` take a checkpoint's run's rule unless told otherwise
(`--select`, `--a-select` / `--b-select`, the transform `--cq-c-visit` / `--cq-c-scale` and h2h's
per-side `--a-cq-c-scale` / `--b-cq-c-scale`; replays record it in the seat, `select`).
`selfplay_select` is the same rule for self-play after `temp_moves` (off by default: a forked
run's A/B, section 12.4).

**The rule's own completed-Q transform** (`eval_select_c_visit` / `eval_select_c_scale`,
`selfplay_select_c_visit` / `selfplay_select_c_scale`; 2026-10-29, section 12.5). The 'cq' rule
ranks options by the 'cq' target's logits, whose transform is `(c_visit + max N) c_scale`. Unset
(the default), a rule uses the target's `cq_c_visit` / `cq_c_scale` - one knob for training and
move selection, as before these settings; set, the rule has its own and the targets keep theirs.
A run's evaluation rule is `mcts.select_settings(cfg, 'eval_select')` (rule, guard, effective
transform): `Run.nn_seat` builds every evaluation seat from it, `elo.eval_tag` names its pool
after it (`select_tag`: the effective transform, so `cq-n4-v50-s0.3` whether the 0.3 comes from
`cq_c_scale` or from `eval_select_c_scale`, and `cq-n4` at 50 / 1), and `zero.h2h` / `zero.replay`
read a checkpoint's run through it. Consequences: a run whose rule transform is pinned keeps its
pool when only its training `cq_c_scale` changes; a run with the setting unset changes pool with
`cq_c_scale` as before; pinning the scale at a restart to the value the run already selects with
(e.g. `--eval-select-c-scale 0.3` on a run at `cq_c_scale` 0.3) keeps the pool, the evaluations
and every file byte for byte, and pinning it to another value begins that value's pool (as any
rule change does: `since`, no replay). The seat keys (docs/search.md 3.3): an evaluation seat
has no 'cq' target, so it carries the rule's transform as `cq_c_visit` / `cq_c_scale` (the seat
dict of an equal effective rule is the same dict, key for key); a self-play seat with a 'cq'
target adds `select_c_visit` / `select_c_scale` where the rule's transform differs from the
target's. A remote evaluation on an older checkout computes the rule from `cq_c_scale`: when that
differs from the run's effective rule its result names another rule and is refused (5.2); a
remote self-play checkout without the setting would select at `cq_c_scale` (nothing to compare
there: update the remote's checkout with the pin).

**What "unset" should mean long-term.** Today unset follows `cq_c_scale`, and it stays so for
now (every existing run and config behaves bit for bit as before). Recommended later: unset = a
fixed transform (50 / 1, the paper's and the 'cq' rule's measured default, or whatever value the
selection-scale measurement settles on) instead of following the target. The reasons: the two
knobs answer different questions (how sharp the training target is; how far Q may overrule the
prior when choosing a move), they act separately (12.5: in MTG Zero's A/B the training scale 0.3 had
no measurable effect, while selecting at 0.3 instead of 1.0 was worth ~+100 Elo head to head), and following makes every training A/B of `cq_c_scale`
silently an A/B of the evaluation rule. The switch needs one migration: every run config with a
'cq' rule and a `cq_c_*` other than the new default gets its current effective transform written
in (`eval_select_c_scale` = its `cq_c_scale`, likewise `selfplay_*`), so no run changes moves or
pool; then the default flips. Until then: **pin the selection scale explicitly** on any run whose
`cq_c_scale` might be changed, and give both arms of a `cq_c_scale` A/B the same
`eval_select_c_scale`.

**Bookkeeping.** Every evaluation records its rule: `eval_yard`, `eval_random` and `elo` rows
carry `select` = the rule's name (`mcts.select_tag`: 'cq', 'cq-n2', 'cq-v25-s0.5' with another
completed-Q transform), absent = 'visits' - so a default run writes exactly what it wrote
before, and every older row reads as 'visits'. Co-training's `eval_yard_net` records `select`
(the primary's) and `resolver_select`.

**One Elo pool per rule.** A checkpoint that chooses its moves by another rule is another
player, so its games never enter another rule's pool: 'visits' keeps `elo/` (results.json,
ratings.json - every run's pool from before the setting), any other rule gets `elo/<rule>/`
with the same files plus `select` and `since`. A pool's evaluations are the metrics rows whose
`eval_yard.select` is its rule; its Elo matches are played with both seats at its rule; its
reference games (random vs the yardstick: bots only, rule-free) are played once in it. Chosen
over one file with rule-tagged player names refit per rule because a pool is then a plain run
pool (the fit, `logged`, the dashboards and the remote's copies work on it unchanged) and the
other pools' files are never written again - nothing to filter, nothing to migrate.

* **Changing the rule mid-run** (at a restart: `--eval-select cq`): the old pool's files stay
  byte for byte as they were (tested), the new pool starts at the first iteration it sees
  (`since`): its Elo matches are those of checkpoints evaluated from then on, against their
  usual opponents (older checkpoints, rated in the new pool through those games only) - no
  replay of the history. `models/index.json` and `best.pt` follow the current rule's pool (the
  best checkpoint *evaluated with the rule the run evaluates with*: the older opponents are listed
  `"evaluated": false` - in the pool's ratings.json too - and never best), and so do the
  dashboards (`elo.active_pool`: config.json's rule; zerodash's `runfiles.elo_pool`; both take
  their "best" among the evaluated checkpoints). zero's dashboard draws the run's other pools
  (`elo.other_pools`) as their own lines next to the current one, with a legend - one chart, never
  one fit: the lines share the yardstick's 1000 but are refit apart. The metrics rows' `elo` of
  the earlier iterations keep their old pool's ratings; Elo is comparable within one pool only.
  Back to a rule used before: its pool continues (and plays the Elo matches it lacks for the
  iterations evaluated meanwhile - k >= its `since` -, at the old rule); the checkpoints those
  matches rate were evaluated with the other rule, so this pool flags them too (`"evaluated":
  false` whenever the run has another pool) and none of them becomes `best.pt`.
* **On a remote**: the remote plays the run's rule and owns its pool (`elo/<rule>/` on the
  mirror, copied back); a result naming another rule is refused (5.2).

**Raw Elo offsets** (`elo_offsets`, e.g. '1,2,4,8'). By default an evaluated checkpoint k plays
k - off x `eval_every` for off in 1, 2, 4 ... 32 (the first `elo_opponents`), so with
`eval_every` 8 only k-8 and k-16, and a run whose `eval_every` changed skips the iterations in
between (MTG Zero's delver1: iteration 12, evaluated at `eval_every` 4, never played again at 8).
`elo_offsets` makes the offsets raw iterations: k plays k-1, k-2, k-4, k-8 (those that exist;
every iteration writes a checkpoint) whatever `eval_every`, and `elo_opponents` is not used.
The opponents that were never evaluated join the pool as players rated only through these games
(the fit needs no yardstick games for them; `logged` lists the evaluated ones only), are listed
in the pool's `ratings.json` and in `models/index.json` with `"evaluated": false`, and are never
`best.pt` (nor either dashboard's "best"); the remote ships them (5.2). '' (the default) keeps
the evaluations-back offsets exactly, and a default run's pool flags nothing (its ratings.json
and index.json as before the flag).

**No random games** (`eval_random_games` 0): no match against the random bot (no `eval_random`
row: the Elo code, the dashboards and the remote work without it). A run that turns it off at a
restart keeps its older rows: the dashboard's "vs random" tile then names the iteration of its
newest value. The reference games still rate the random bot against the yardstick (bots only,
`ref_games` once per pool; 0 = not at all).

Default runs are unchanged: toy-smoke (classic; rolling; margin_w + chance_k + cq targets with
`eval_every` 2) gives the same metrics (timing aside), checkpoints, optimizer states, shards, Elo
files and models/index.json byte for byte as zero-core 3c26d9f, with the settings left out or
given explicitly at their defaults (section 12.3). Tests: `tests/test_select.py`.

**Changing a live run's rule** (MTG Zero's delver1 at its re-pin, its ENGINE.md 57): stop it at
an iteration boundary, re-pin, restart with `--eval-select cq --eval-select-min-visits N` (and,
if wanted, `--eval-random-games 0 --elo-offsets 1,2,4,8`). The next evaluation (here or on the
`eval_remote`, whose checkout must have the rule: a remote without it is refused, 5.2) begins
the rule's pool `elo/<rule>/` at that iteration: 100-ish reference games once (`ref_games`), the
checkpoint's yardstick games, its Elo matches against its raw offsets; `elo/` and the metrics'
older `elo` rows are left as they were. A toy rehearsal of exactly this (a remotely evaluated run
re-pinned with those flags): section 12.3.

## 6. Learning: the toy proof run (Glass Road Zero 2026-10-11; Caverna Zero 2026-10-10)

### 6.1 The run

`toy1` (`--preset toy-1`, tabula rasa, CPU only: 4 search threads + 1 torch thread at
below-normal priority while Caverna Zero's, Dominion Zero's and St. Pete Zero's training runs
used the machine), under the supervisor from Glass Road Zero's `infra` worktree (commit 5187538),
run dir `~/glassroad-zero-wt/infra/runs/toy1` (gitignored). Yardstick: the toy heuristic
through the external-seat path (`ext:toyheur`); random vs the heuristic: 15/1000 (1.5%).

| Iteration | Self-play games (cum.) | vs heuristic (400 games) | margin | vs random (100) | Elo (final fit) | policy / value loss | self-play leaves/s |
|---|---|---|---|---|---|---|---|
| 0 (random net) | 0 | 3.8% +-1.9 | -16.9 | 57% | 334 +-72 |  |  |
| 1 | 600 | 41.5% +-4.8 | -3.1 | 99% | 905 +-31 | 1.618 / 0.601 | 62,960 |
| 2 | 1,200 | 76.9% +-4.1 | +2.4 | 100% | 1224 +-31 | 1.572 / 0.562 | 60,078 |
| 3 | 1,800 | 81.6% +-3.8 | +3.3 | 100% | 1295 +-33 | 1.560 / 0.564 | 63,511 |
| 4 | 2,400 | 97.1% +-1.6 | +4.9 | 100% | 1404 +-35 | 1.519 / 0.575 | 72,983 |
| 5 | 3,000 | 98.0% +-1.4 | +5.0 | 100% | 1552 +-40 | 1.505 / 0.576 | 71,015 |
| 6 | 3,600 | 97.6% +-1.5 | +4.5 | 100% | 1626 +-41 | 1.486 / 0.557 | 63,894 |
| 7 | 4,200 | 99.2% +-0.8 | +4.7 | 100% | 1777 +-44 | 1.438 / 0.563 | 65,320 |
| 8 | 4,800 | 99.8% +-0.5 | +5.0 | 100% | 1786 +-44 | 1.421 / 0.555 | 64,607 |
| 9 | 5,400 | 99.9% +-0.4 | +5.9 | 100% | 1878 +-46 | 1.383 / 0.549 | 51,865 |
| 10 | 6,000 | 99.6% +-0.6 | +5.2 | 100% | 1845 +-46 | 1.374 / 0.550 | 64,413 |
| 11 | 6,600 | 100.0% +-0.0 | +5.8 | 100% | 1904 +-46 | 1.336 / 0.544 | 51,581 |
| 12 | 7,200 | 99.6% +-0.6 | +5.5 | 100% | 1973 +-47 | 1.313 / 0.547 | 50,130 |
| 13 | 7,800 | 98.6% +-1.1 | +5.3 | 100% | 2010 +-47 | 1.280 / 0.540 | 53,826 |
| 14 | 8,400 | 99.0% +-1.0 | +5.3 | 100% | 2015 +-48 | 1.258 / 0.545 | 65,749 |
| 15 | 9,000 | 97.8% +-1.5 | +4.9 | 100% | 2053 +-48 | 1.239 / 0.540 | 66,292 |
| 16 | 9,600 | 99.4% +-0.8 | +5.1 | 100% | 2074 +-48 | 1.225 / 0.539 | 70,326 |
| 17 | 10,200 | 98.5% +-1.2 | +5.7 | 100% | 2102 +-49 | 1.200 / 0.521 | 49,286 |
| 18 | 10,800 | 99.0% +-1.0 | +5.0 | 100% | 2114 +-50 | 1.184 / 0.532 | 56,959 |
| 19 | 11,400 | 99.0% +-1.0 | +5.6 | 100% | 2138 +-50 | 1.164 / 0.528 | 59,994 |
| 20 | 12,000 | 98.0% +-1.4 | +6.2 | 100% | 2178 +-50 | 1.144 / 0.519 | 50,274 |
| 21 | 12,600 | 99.4% +-0.8 | +6.1 | 100% | 2196 +-52 | 1.133 / 0.524 | 50,566 |
| 22 | 13,200 | 98.6% +-1.1 | +6.3 | 100% | 2180 +-52 | 1.116 / 0.529 | 56,985 |
| 23 | 13,800 | 99.0% +-1.0 | +6.4 | 100% | 2189 +-53 | 1.088 / 0.521 | 60,943 |
| 24 | 14,400 | 99.0% +-1.0 | +6.4 | 100% | 2254 +-56 | 1.078 / 0.530 | 65,159 |

* **Same run as Caverna Zero's** (`~/caverna-zero/docs/az.md` 6, 2026-10-10): every
  evaluation against the heuristic and every training loss above equals its counterpart there
  (same code, same seeds: the loop is deterministic). Only games of the random bot differ
  (random vs the heuristic 15/1000 here, 23/1000 there; 99% vs 98% at iteration 1), because
  Caverna Zero's run predates its per-seat bot seeds (146d349), hence a slightly different Elo
  fit (2254 vs 2255 at iteration 24; random 269 vs 321). The carried-over stack is the proven
  one, not merely a similar one.
* **Run control, exercised on this run**: PAUSE (`tools/pause_run.py --wait`) during
  iteration 6's Elo matches - the run noticed it after 0.5 s, exited with status 75, the
  supervisor waited; `--resume` relaunched it, and it finished iteration 6's evaluation from
  `models/iter_0006.json` without replaying self-play or training. A hard kill
  (`tools/kill_run.py`) during iteration 9's self-play, then a new supervisor: iteration 9 was
  replayed from the start (its data directory cleared). A STOP file during iteration 11: the
  run ended after it (exit 0, "STOP file found"), the supervisor with it; the next supervisor
  moved the stale STOP to STOP.old and the run resumed after iteration 11.
* **Wall clock**: 24 iterations in 12.3 minutes of run time (13.1 minutes including the pause
  and the restarts; median 29 s per iteration: self-play 7 s, training 1 s, evaluation 4 s,
  Elo matches 15 s). 14,400 self-play games, ~290k games/h, ~62k leaves/s (median), inference
  ~91% of the self-play time (one torch thread; average batch ~110 leaves). About twice
  Caverna Zero's speed on the same machine the day before (less load).
* **Learning**: the random network loses 96% to the heuristic; one iteration (600 games)
  takes it to 41.5%, four iterations (2,400 games) past 97%, and from then on it wins 97-100%
  with a growing score margin (+4.5 to +6.4). Elo keeps rising after the heuristic is
  saturated (1404 at iteration 4, 1845 at 10, 2254 at 24) while the policy loss falls steadily
  (1.62 -> 1.08). Random rates 269.

### 6.2 Rust search vs Python search, head to head

`python -m zero.parity h2h --model <toy1 iteration N> --sims 32`: a seat searching with the
Rust tree (`<pkg>._engine.SearchTree`) against a seat searching with the Python reference tree
(`zero/mcts.py`), both in the Python driver with the same network and the same settings,
seats alternating:

| Network | Setting | Games | Rust-search seat's score | 95% interval |
|---|---|---|---|---|
| iteration 4 (97% vs the heuristic) | chance_k 0 (seed 1) | 6000 | 49.94% | +-1.04 |
| iteration 4 | chance_k 2 (seed 2) | 2000 | 50.20% | +-1.82 |
| iteration 12 (99.6%) | chance_k 0 (seed 7) | 4000 | 49.91% | +-1.32 |

All consistent with 50% (Dominion Zero's gate: 51.1% +-1.3). The first two rows are Caverna
Zero's measurement reproduced game for game (same network, seeds and code); the third is new.
Together with the root-statistics check (docs/search.md 6: TV distance at the
Python-vs-Python noise floor, a 30% c_puct change detected at 10x the floor) the Rust search
implements the reference semantics.

## 7. Throughput and the shared CPU

See docs/search.md 7 for the measured numbers. On the shared desktop (other projects' jobs
use every core at below-normal priority) **torch must use one intra-op thread**: a 128 x 3
forward pass at batch 512 took 7.9 ms with 1 thread, 96 ms with 2 and 879 ms with 4 (each op
waits for preempted threads); training slowed the same way. `torch_threads` defaults to 1;
the Rust search threads are independent (no barriers inside a group's work) and scale.
Toy self-play is inference-bound (~40-65k leaves/s on one torch thread, depending on the
load); a toy iteration of 600 games takes ~30-60 s including evaluation and Elo matches.

## 8. Not in the core yet (step 2 extension points)

From Dominion Zero (their code is in `~/dominion-zero`, not in the repos the core was extracted
from): worker processes for self-play / evaluation and resuming a crashed self-play phase from
its complete shards (`workers.py`; here everything runs in the run process: the Rust driver is
multi-threaded, a crash costs the iteration in progress, the supervisor restarts it), the
CUDA-graph evaluator / pinned staging buffers / two batches in flight (`infer.py`; eager CUDA
inference works here), rented-GPU experiments (Vast),
exploration games, behaviour cloning. Where they plug in:

* **Evaluator**: anything with `__call__(LeafBatch) -> (priors, value, margin)` and
  `set_weights(state_dict)` can replace `infer.NetEvaluator`; `Run.evaluator(path)` is the one
  place the loop builds it (a `zero.infer` factory keyed by a run setting is the natural
  switch, e.g. `infer_impl='cudagraph'`).
* **Remote / worker self-play**: in the core since 2026-10-28 (sections 5.3, 11.6: workers on
  another machine whose shards join `data/iter_NNNN/` under their own prefix). Local worker
  processes would plug in the same way: `Run.selfplay(it)` produces shards under
  `data/iter_NNNN/` through `ShardWriter` and returns the metrics dict; the window
  (`Run.window_shards`) reads every shard of an iteration directory.
* **Remote evaluation**: `Run.evaluate(it, m)` / `Run.elo_update` fill `m['eval_yard']`,
  `m['eval_random']` and the Elo results; an asynchronous evaluator can write the same keys
  later (Dominion's `--remote-eval`).
* **Cloud experiments**: forks (`zero.tools.fork_run`) + `zero.h2h` are the A/B primitives.
* New run settings go into `run.DEFAULTS` (saved in every run's config.json: a missing key in an
  old run's config takes the default, so defaults must keep old behaviour).

Browser players (Caverna Zero's `wasm/` + `web/`, docs/wasm.md and docs/web.md there): the
game-independent network moved into the core - crate `crates/webnet` (`zero_webnet`: the
`zero.net` forward pass over float16 `.bin` weights, deterministic, native = wasm32 bit for bit)
and `zero.webexport` (checkpoint -> `.bin`); Caverna re-exports them (`cvz_wasm::net`,
`tools/export_web.py`). Still Caverna's, for a later step: the single-game search jobs
(`wasm/src/search.rs`: choose / analyze / rate in slices over `zero_search::tree::Tree`), the JSON
command API and the wasm ABI (`api.rs`, `lib.rs`: `cvz_*` exports), the Web Worker / `neural.js`,
and the ladder tooling (`tools/ladder_calibrate.py`, `export_ladder.py`, `rated_js_check.py`).
They are coupled to Caverna's `WebAz` (a decision = several AZ picks, JS response -> AZ picks in
`decompose.rs`) and to the JS state format; generalising them means a `WebGame` trait (JS state
in, decision boundaries, picks -> JS response) that each game implements, with the jobs, API,
exports (renamed `zero_*`), worker and ladder calibration in the core.

## 9. Policy targets: visits vs completed Q

`policy_target='cq'` trains the policy head on Gumbel AlphaZero's completed-Q target instead of
the root visit distribution (docs/search.md 3.1: the formula, why Q is not min-max normalised,
parity). The search, the moves played, the value targets and every other field of a record are
the same; with the same seed both settings play the same self-play games.

Why: with values = P(win) in [0, 1] and c_puct 1, PUCT's visit counts at 128 simulations stay
close to the prior, so a visit-count target barely moves the policy. Both games plateaued with
nearly flat visit targets (Glass Road `grz2p`: H(pi) / ln(#options) 0.85, max share 0.47, policy
loss = policy entropy, flat; Caverna `cvz2p` likewise). `python -m <pkg>.az.targets sharpness`
and `quality` measure the targets of a checkpoint (each game's docs/az.md 9.2-9.3: Glass Road's
c_scale 1 target removes 73% of the prior's regret against a 16x larger search, visits 32%).
Results: Glass Road adopted cq after its A/B (56-65% head to head over iterations 36-45, Glass
Road Zero docs/az.md 9.6); Caverna's A/B (`cvz2p_vis` / `cvz2p_cq`) was running when the core
was extracted (Caverna Zero docs/az.md 9.4).

### 9.1 Toy proof (`toy-1`, 24 iterations; Glass Road Zero 2026-10-11)

Runs `toy1_cq` (c_scale 1) and `toy1_cq03` (c_scale 0.3) in a Glass Road Zero worktree's `runs/` (gitignored;
`--preset toy-1 --policy-target cq [--cq-c-scale 0.3] --threads 3|2`, CPU, below normal), against
the visits baseline `~/glassroad-zero/runs/toy1` (section 6; the visits loop on that branch
reproduces its iterations 1-2 exactly: same win rates, policy losses and sample counts). The
heuristic saturates early, so the yardstick is a head-to-head of each cq checkpoint against the
visits checkpoint of the same iteration (600 games, 32 sims, seats alternating; score of the cq
checkpoint, +-1 se, and the Elo difference):

| iteration | cq c_scale 1 vs visits | Elo | cq c_scale 0.3 vs visits | Elo | vs heuristic: visits / cq 1 / cq 0.3 |
|---|---|---|---|---|---|
| 1 | 41.8% +-2.0 | -57 | 54.3% +-2.0 | +30 | 41.5 / 32.4 / 39.4% |
| 2 | 10.0% +-1.2 | -382 | 18.3% +-1.6 | -261 | 76.9 / 51.1 / 59.4% |
| 4 | **87.1% +-1.4** | **+332** | 72.6% +-1.8 | +169 | 97.1 / 89.0 / 80.0% |
| 8 | 70.7% +-1.9 | +153 | 59.9% +-2.0 | +70 | 99.8 / 97.1 / 97.0% |
| 12 | 65.1% +-2.0 | +108 | 59.5% +-2.0 | +67 | 99.6 / 97.9 / 99.8% |
| 16 | 60.7% +-2.0 | +75 | 58.7% +-2.0 | +61 | 99.4 / 100 / 95.3% |
| 20 | 47.8% +-2.0 | -16 | 39.8% +-2.0 | -72 | 98.0 / 97.5 / 96.6% |
| 24 | 42.9% +-2.0 | -50 | 31.8% +-1.9 | -133 | 99.0 / 98.8 / 96.8% |

More matches (600 games each): cq-1 iteration 24 vs visits iteration 16 / 20: 68.4% / 55.3%
(+134 / +37); vs visits 24 at 128 sims 41.4% (-60); cq-1 24 vs cq-1 16: 75.8% (+198; visits 24
vs 16: 77.2%, +211); cq-1 24 vs cq-0.3 24: 66.4% (+119).

* **c_scale 1 learns much faster through iteration 16** (after a bad iteration 2, when the
  first informative values made the targets suddenly sharp: H/ln n 0.78 -> 0.27): +332 Elo
  over visits at iteration 4, still +75 at 16. **From iteration ~18 visits catches up and
  passes it**: at 24 cq-1 is ~2 iterations behind (between visits 20 and 24), while still
  improving as fast as visits (+198 vs +211 Elo from 16 to 24). So the gate "at least as fast
  as the visits baseline" holds for the first two thirds of the run, not at its end.
* **The policy collapses**: the completed-Q target is the prior times exp(sigma(Q)), so a sharp
  prior stays sharp; nothing restores entropy. Training policy entropy at iteration 24: visits
  1.08 nats, cq-1 0.60, cq-0.3 0.52 (recorded targets at the end: H/ln n ~0.15 / 0.21, max share
  ~0.90 / 0.86; the visit targets at iterations 1-2: 0.92 / 0.37 and 0.85 / 0.46). A near-deterministic prior concentrates PUCT's visits and
  the self-play data; the likely reason for the late slowdown. The paper avoids this by acting
  with Gumbel noise + sequential halving at the root, which this branch deliberately does not
  implement.
* **c_scale 0.3 is worse than 1** throughout (it collapses as well, only later, with smaller
  gains early).
* Cost: none (the target is computed once per recorded decision from the root statistics).

### 9.2 Root minimum visits: Q for every option of a small decision (MTG Zero's Delver mirror, 2026-10-29)

The completed-Q target improves an option's policy only through its Q, and an unvisited option has
none: it is completed at v_mix, close to the visited options' Q, so its target stays at its prior. A
prior that has saturated (MTG Zero's Delver mulligan: >= 0.98 on Keep for most hands, 0 visits for
Mulligan at 64 simulations) can then never move, whatever the option is worth - only root noise
reaches it, now and then. The **root minimum visits** (docs/search.md 3.5) give every option of a
small decision (2 .. `root_min_visits_max_actions` options, default 4) at least k visits in each full
search, when the budget allows (2 x options x k <= the search's simulations): the forced walks are Q
measurements for the 'cq' target and rule, and are left out of the visit counts (the 'visits' target,
temperature sampling, the 'visits' rule) and of the root value target.

**Settings** (section 1.2): `selfplay_root_min_visits` (the self-play seat: its recorded searches;
never the fast or minor ones), `eval_root_min_visits` (every evaluation seat; its Elo pool is named
after it, `visits-f4`, `cq-n4-v50-s0.3-f4`, `-f4a6` with max_actions 6: the evaluations' moves change),
`root_min_visits_max_actions`; all at 0 / 0 / 4 by default = the run of before, byte for byte. The
split follows `selfplay_select` / `eval_select`: a training A/B sets only `selfplay_root_min_visits`
and keeps both arms' evaluation rule (and pool) the same. Remote self-play and remote evaluation get
the seat keys like every other: the remote's checkout must have this core (an older one refuses the
unknown seat key - a visible error, never a silent off).

**Cost.** No extra simulations and no extra network calls: the forced walks are part of the search's
budget, so leaves per second do not change; what changes is how the budget of a forced search is
spent - at most (options - 1) x k of its walks (at 64 simulations on a 2-option root: 4 of 63 with k 4,
8 with k 8), and none where PUCT gives every option k visits anyway. MTG Zero measured the share of
the full searches' walks on its Delver line (its ENGINE.md 64).

**Measured** (MTG Zero ENGINE.md 64, delver2's iterations 21 and 29, CPU): the floor works as
designed - every option of the forced roots measured - but on the mulligan the network's own Q put
Mulligan below Keep in every starved low-land hand, so the keep rates did not move (0 lands 58%, 1
land 91%, at k 0, 4 and 8, under every move rule): those keeps come from the value head. On general
small decisions (2-4 options, mana payments aside) 62% of the roots leave an option without a visit at
k 0, and k 4 moved the cq target by more than 0.1 total variation at 11-13% of them and changed the
cq4 evaluation move at 3-4%. Whether that trains a better network is an A/B question (MTG Zero
ENGINE.md 64.6 has the flags).

Tests: section docs/search.md 3.5's, and `tests/test_root_min_visits.py` (settings and names; the
Python driver's 'py' and 'rs' seats and the Rust driver prune alike; toy runs: the settings at their
defaults byte for byte, self-play's floor changes the shards and not the pool, the evaluation's floor
changes the pool and not the training).

Checks (5ce50d4, on main f57ee57; CPU only): `cargo test --release --workspace` 59 + 3 passed (the
feature's 9 also in a debug build); `pytest tests` 228 passed, the 3 CUDA tests skipped; toy-smoke runs
byte for byte against f57ee57 (classic; rolling; margin_w 0.2 + chance_k 2 + cq targets + eval_every
2), the settings left out and given at their defaults (config.json gains the three keys); the golden
records and match unchanged; `tools/check_checkpoint_compat.py --package zero` (random, uniform, a toy
checkpoint) identical; Glass Road Zero, Caverna Zero (and its wasm crate) and Kingdom Builder Zero
build against it; MTG Zero: its `az` tests and pytest, and delver2's checkpoint plays identically on
its main's build and this one (its ENGINE.md 64.7). With feat/league (this branch is rebased on it):
the seat keys, `NnSeat`'s fields, the Python `NNSeat` and the driver tests met textually, both sides
kept; a league opponent's unrecorded seat is the self-play seat, so it takes the self-play floor too.

## 10. The token network (arch 'tokens', `net.TokenNet`; MTG Zero M5b, 2026-10-22)

A transformer over the game's objects for games whose state is a variable set of things (MTG:
permanents, hand cards, stack objects, cards of a searched library), read from the same
fixed-width features as the MLP: the `SearchGame` trait, the driver, the leaf batches and the
shards are unchanged. The game package describes where the objects are (`GamePackage.tokens`,
game -> layout; stored in the checkpoint's config):

```
{"vocab": V,                                   # identities: V embedding rows + 1 (none / unknown)
 "segments": [{"name", "offset", "count", "width",   # count object slots of width floats at offset
               "present": i | null,           # slot float i > 0.5 = the slot holds an object (null: always)
               "id": {"at": j, "radix": [16, 16]} | null,   # identity as mixed-radix one-hot digits
               "actions": [[base, stride], ...]}, ...],    # slot s scores actions base + s * stride
 "bags": [{"name", "offset", "count", "width"}]}  # count multisets of identities (width <= V counts)
```

Every feature outside the segments and bags is a global. The network:

* **Tokens**: the global token (a linear map of the globals, plus one linear map of the bags:
  each bag pooled as the count-weighted sum of the identity embeddings and its total), and one
  token per present slot = its identity's embedding (shared by every segment and the bags) + a
  linear map of the slot's floats (per segment) + a learned (segment, slot) embedding (slot
  orders mean something: timestamps, the top of the stack, whose library).
* **Packing**: only present slots are embedded and attended to. A batch's rows hold their
  tokens first (layout order, the global token at 0); rows are sorted by token count and run
  in chunks of 64, each padded to its own longest row (padding masked out of the attention). A
  row's outputs do not depend on the batch or the chunking (tested). MTG's positions hold 21
  tokens on average (p99 39, max 53) of the 153 slots.
* **Trunk**: `blocks` pre-norm blocks of width `hidden` (multi-head attention with `heads`
  heads, GELU feed-forward of width `ff`), a final LayerNorm.
* **Heads** read z = [the global token's output, the mean of the row's tokens]: value / margin
  as `Net`, generic policy logits for every action (`Linear(2 hidden, actions)`); a slot's own
  actions (`actions` pairs) are scored from its token conditioned on z (pointer logits:
  `Linear(gelu(W_k t + W_q z))`, one output per action pair of the segment) and replace the
  generic logits of those ids. An empty slot's actions keep the generic logits (they are not
  legal anyway). The policy starts uniform (both policy output layers at zero): with random
  ones the generic logit of a frequent action and a segment's pointer logits are a few random
  constants, and MTG's first token run (mtg2a) started with 55% of the prior on PASS where 25%
  is uniform; its self-play games ran into the turn cap (draws, no signal) and its first
  checkpoint lost 100/100 to the heuristic. The value head keeps its random init: a random value
  head can still steer the first searches (mtg2b, seed 0: the cq targets of its first games put
  49% on PASS against 36% uniform; seed 1 33% / 40%; the MLP's seed 0 25% / 40%), but a zero one
  is worse: every search is then a tie and the search's tie rule (the first maximum) plays the
  first option, PASS (mtg2n: 3 self-play games finished in 15 minutes). MTG's token run uses
  seed 1.
* **Actions by identity** (`id_actions`, 2026-10-27, MTG Zero M6 part 3): a segment with
  `id_actions: base` (and an identity) also scores the action base + each present slot's
  identity, one more pointer column per segment: in MTG Zero's team draft a card's pick, or
  putting it in the deck, is its identity wherever the card sits (layout `draft_v1`: the pack's
  and the pool's slots, base 0). Its range [base, base + vocab) may hold no slot action of the
  layout; two segments may share it (their identities are distinct within a row; were one
  repeated, one of its tokens would score it). Layouts without the key build exactly the
  network they did (no identity column; the normalised layout stored in a checkpoint keeps no
  `id_actions` key): MTG Zero's runs/mtg5 checkpoints load and play identically
  (`check_checkpoint_compat`, its ENGINE.md 45).
* `widen`: new actions get zero generic weights, new features (after the old ones) are new
  globals with zero weights; the layout stays. With the game's new layout (`widen(...,
  layout=)`, fork_run) a token network follows a layout that only adds (wider slots, bag rows,
  bags, globals; the same tokens) exactly: section 10.3. Not in `zero.webexport`. CUDA: float32 eager
  (no autocast, no CUDA graphs; not measured: the GPU was busy).

Tests: `tests/test_net_data.py` (masking of absent slots, chunking / batch independence, the
pointer logits land on the slot actions, identity digits, save / load / widen, a training step
reaches every part, layout validation, the toy's layout `package.TOY_TOKENS` through the
evaluator, actions by identity: the logits, the validation, save / load / widen),
`tests/test_loop.py` (the loop with arch 'tokens' on the toy).

### 10.1 CPU cost (MTG, 13,532 features; desktop 2026-10-22, the GPU busy with other runs)

Inference on real MTG positions (runs/mtg1 iteration 9 shards; `python -m <pkg>.az.bench infer
--arch tokens --shard ...`), leaves per second, one inference thread:

| network | parameters | 2 torch threads, batch 192 / 384 | 6 threads, 192 / 384 |
|---|---:|---:|---:|
| MLP 256 x 3 (mtg1's) | 4.19M | 11.9k / 11.8k | 24.0k / 19.5k |
| tokens 64 x 3 (ff 128) | 0.42M | 4.3k / 4.8k | 3.8k / 6.4k |
| tokens 96 x 3 (ff 192) | 0.76M | 3.2k / 3.2k | 3.2k / 4.3k |
| tokens 128 x 3 (ff 256) | 1.17M | 2.0k / 2.3k | 2.6k / 3.4k |

The token network does as much matrix work per leaf as the MLP at 64 x 3 (the MLP's cost is
its 13,532 x 256 input layer) but spends it in many small ops (attention over ~25 tokens per
row, layer norms, gathers): it does not scale with intra-op threads. Several batches evaluated
concurrently on Python threads with few intra-op threads each do (`infer_workers`; tokens 96 x 3,
batch 384, 8 batches each): 1 x 6 threads 4.7k leaves/s, 2 x 3 6.3k, 3 x 2 7.2k, 4 x 2 7.7k.
In MTG self-play (rolling, 64 / 16 sims, 8 search threads; another job using 6 cores):
the MLP 256 x 3 at 1 x 6 threads 9.0k leaves/s, tokens 96 x 3 at 1 x 6 threads 2.8k, at 4
workers x 2 threads (4 groups, 768 games in flight) 4.6k.

**The 13.5k-float feed is not the bottleneck**: the search threads were busy 14 s of 60 in
that self-play measurement (encoding included), the rest of the time they waited for the
network; the feature copy is one memcpy per chunk in the driver. A sparse feed (the present
slots only, ~5% of the floats) would save memory bandwidth and shard space (27 KB per sample)
but not inference time; not built.

### 10.2 First result: MTG Zero's run mtg2 against an MLP control (2026-10-22)

MTG Zero docs/ENGINE.md 22.4 has the details. Same settings (64 / 16 sims, cq targets, value_q
0.9, reuse 2, minor decisions at 4 sims, 600 games per iteration), CPU only on a machine whose
32 threads other runs kept busy: tokens 96 x 3 (0.76M parameters, `infer_workers` 4 x 2 torch
threads) against the MLP 256 x 3 (4.19M).

* **Per self-play game the token network is the stronger player**: head to head at 32 sims it
  scores 65.8% +-6.1 (iteration 3 vs 3, 60 games) and 57.0% +-5.0 (iteration 5 vs 5, 100
  games); its value head predicts unseen games better (67% vs 62% accuracy on the same held-out
  games at iteration 3) with the same policy cross-entropy per decision kind.
* **Per hour it is not**: 3.3-5.4k leaves/s in self-play against the MLP's 8.9-17.6k, and
  evaluation on tiny batches is slow (12 minutes for one 20-game Elo match between two token
  networks): 5 iterations in 3 h 35 min against the MLP's 10 in 3 h 4 min, and the MLP's
  iteration 10 beats the token network's iteration 5 63.5% +-4.8.
* Against MTG's rule-based yardstick both stay flat in these few iterations (tokens 2-6%, MLP
  5.5-17%): the yardstick and head to head disagree (non-transitive styles), so a token run
  needs Elo matches against fixed checkpoints too.
* Initialisation matters with completed-Q targets and argmax play: random policy output layers
  made one action dominate (55% PASS), a zero value head made every search a tie (ties play the
  first option); the policy output layers now start at zero and the value head stays random
  (section 10).

### 10.3 Widening across a layout change (`net.widen(..., layout=)`; MTG Zero's layout v5, 2026-10-29)

A game that needs more inputs can grow its object layout without a fresh start for its token
networks, as long as the new layout only **adds**: a checkpoint of the old layout is widened onto
the new one so that it computes exactly what it did. MTG Zero's layout v4 -> v5 (its ENGINE.md 60:
wider permanent and stack slots, two more bags, 64 more globals) is the first such change.

**What a new layout may change** (`token_feature_map(old_layout, F0, new_layout, F1)`, both
normalised; `widen` refuses anything else with a ValueError):

* **Segments**: the same names in the same order, the same slot counts, present flags,
  identities (`id`), actions and `id_actions` - so the same tokens, (segment, slot) positions and
  pointer columns. A slot may get **wider at its end**: its old floats are the first ones of the
  new slot. Segment offsets may move (everything after a wider block moves).
* **Bags**: every old bag by name, with the same width and its old rows first (new rows after
  them); new bags anywhere.
* **Globals**: with declared global blocks in the old layout (layout key `globals`, below), by
  block name (the old floats first in each block; new blocks anywhere); without them, the old
  globals are the new layout's first globals in feature order (new globals after the old ones in
  feature order, e.g. after a trailing block).

**What the widen does**: the new network is built from the new layout (same width, layers,
heads, vocabulary); every parameter whose shape does not depend on the inputs is copied (the
identity embedding, the (segment, slot) embeddings, the blocks, the heads, the pointer head);
`seg_in.k` gets the old columns first and zeros for the new ones; `bag` gets each old row's
(d + 1) columns at its new row and zeros elsewhere; `g` gets each old global's column at its new
position and zeros elsewhere; new actions get zero generic weights and bias (as before). So every
new input column has zero weight in its projection, and on any input of the new layout the network
computes what the old one computed on the old features of it: a token's input map adds 0 for the
new floats, a bag row's pool and total go through zero weights, a new global likewise, and the
attention sees the same tokens (present flags and identities are old floats). Exact up to float
rounding: a linear map over more (zero-weighted) columns may round its sum differently (measured
on MTG Zero's main-line checkpoint: max |delta| 5.7e-6 on the policy logits, 3e-7 on the value
probabilities; 95% of the rows bit for bit), and the new columns have no effect at all (the
widened network on the same input with the new floats zeroed or random: bit for bit identical).

**`globals`** (optional layout key): `[{"name", "offset", "count"}, ...]`, named blocks of
globals in feature order that together cover every feature outside the segments and bags
(validated). The network is the same with or without it (its globals are those features in
feature order); it is stored in the normalised layout only when set, so the layouts in older
checkpoints are unchanged. A layout that declares its blocks lets a later layout insert globals
between them (MTG Zero's v5 declares `globals`, `match`, `globals_v5`).

**Callers**: `tools/fork_run.py <src.pt> <new run>` passes the game package's layout
(`GamePackage.tokens`) for a token network: the new run's iteration 0 is the widened source (the
same layout: as before, new features are globals); the co-training `resolver_init` does the same.
A resumed run never widens: a checkpoint whose layout is not the game's is a fork into a new run
(the encoding guard refuses to resume it, 10.4; MTG Zero's ENGINE.md 60.5 for the procedure and what restarts: iteration numbers, Elo pool,
window, optimizer state). The MLP and GridNet keep their widen (appended features; grid planes).

Tests (`tests/test_net_data.py`): a wider synthetic layout (wider slots, a bag row, a new bag, new
globals, every block after moved) - the map, outputs equal on any input of the new layout
(random values in the new floats), save / load, gradients reach the new columns, a widen to the
same layout equals the old widen; declared global blocks (the same network as without them, a
block grown at its end and a new block between two: mapped by name; the validation); refusals
(a new segment, more slots, a narrower slot, another present flag or identity, segments
reordered, a bag gone, another vocabulary); `fork_run` across a layout change (the toy's layout
replaced). MTG Zero: the real-checkpoint check of its ENGINE.md 60.2 and its pytest
`test_a_v4_token_network_widens_onto_layout_v5_exactly`.

### 10.4 The encoding guard (`net.check_encoding`; MTG Zero's layout v5, 2026-10-29)

**Why**: a network fed another encoding than the one it was built for misreads it **without an
error**. A v4 TokenNet (33,652 features) loaded by MTG Zero's v5 build (37,620) takes the first
33,652 floats of every row (`x[:, :features]`) and reads its slots and globals where v4 had them;
an MLP does the same; a network wider than the rows fails only somewhere later. The one safe way
across an encoding change is a widen into a new run (10.3). So every load is checked.

**The check** (`net.encoding_mismatch(config, game)` -> the differences, `check_encoding` raises
`net.EncodingMismatch`, a ValueError): a checkpoint's `config` against what this build's game
reports - `games.info(game)` and the game package:

* the **game** (the config's `game` against the game the load site plays: the run's, h2h's);
  a game this build does not have is a mismatch too;
* the **feature and action counts** (`num_features`, `num_actions`);
* arch 'tokens', when the package has a layout for the game (`GamePackage.tokens`): the object
  layout in the parts the network reads - vocabulary, segments (every key: offsets, slot counts
  and widths, present flags, identities, actions, `id_actions`), bags - both sides normalised
  (`tokens_layout`); not the declared global blocks (they change nothing in the network, 10.3).
  And the layout's **identity** (layout keys `name` / `version`, optional; MTG Zero's files carry
  `version`): kept in the normalised layout since this change (only when set: the toy's configs
  are unchanged), compared when the checkpoint and the build both carry one. Checkpoints from
  before (MTG Zero's v4 runs and delver2's v5) carry none: judged by the structure, which already
  tells v4 from v5;
* arch 'grid', when the package has a grid for the game (`GamePackage.grids`): the board grid,
  both sides in normal form (`net.grid_spec`: an older config without `segments` is the same);
* a network without layout or grid metadata (arch 'mlp', older checkpoints, a game package
  without a layout): the game and the counts alone.

The message names both sides and the way out - MTG Zero's v4 main-line checkpoint on its v5
build: `.../iter_0137.pt: a network of another encoding than this build's (features 33652 ->
37620; token layout: segment tops: offset 17792 -> 19840; segment my_permanents: offset 18116 ->
20164, width 148 -> 172; segment opponent_permanents: offset 22852 -> 25668, width 148 -> 172; 8
more). The network: 'mtg': 33,652 features, 1,024 actions, token layout without a version (11
segments, 1 bag), arch 'tokens'. This build: 'mtg': 37,620 features, 1,024 actions, token layout
v5 (11 segments, 1 bag). Fed this build's inputs it would misread them without an error. Move a
run across an encoding change with python tools/fork_run.py <checkpoint> <new run> (an exact
widen when the change only adds inputs; docs/az.md 10.4), or use a build of the network's
encoding.`

**Where it runs** (every load path, for every game - a game package's own tools load through
`zero.net.load` too):

* `net.load(path, device, game=None, allow_mismatch=False)` checks before it builds the network
  (`game`: the game the caller plays; default the config's own). Every core load goes through it:
  the run's evaluators (self-play, the yardstick and random matches, the Elo matches, the
  reference games), the rolling drivers' new weights, training's checkpoint, co-training's both
  networks, the remote evaluation's mirror (`zero.remote` runs the run's own `Run.evaluate`),
  the remote self-play worker, `zero.h2h` (A against the game played, B against its own game,
  which must be A's), `zero.replay` (both checkpoints), `zero.targets`, `zero.bench`,
  `zero.parity`, `zero.webexport`, a game package's extra evaluations (MTG Zero's human picks).
  The dashboards load no networks.
* **Run resume** (`Run.loop`, `CoRun.loop`: both networks): the newest checkpoint is checked
  before anything starts. A mismatch ends the run with **exit status 78** (`runctl.EXIT_REFUSED`,
  sysexits EX_CONFIG): log line `run REFUSED ...`, `run_state.json` state `refused` (the
  dashboard: stopped, "Refused its checkpoint ..."), the message on stderr; the supervisor does
  **not** restart it (state `refused`: a restart would refuse again).
* **The optimizer state** (`opt_latest.pt`): loaded only if its moments have the shapes of the
  checked network's parameters (`run.opt_state_mismatch`; torch checks the counts only, and another
  network's state would fail at the first step); else logged, fresh optimizer.
* **The evaluator's last line** (`infer.NetEvaluator`): a leaf batch whose rows are not as wide
  as the network's input raises EncodingMismatch (one comparison per batch), whatever loaded the
  network.
* **Remote evaluation** (5.2): the remote's `python -m <pkg>.az.remote eval|evalnet` refuses with
  exit 78 and one line `ZERO_REMOTE_REFUSED {"error": ...}`; the run logs `remote eval <name> iter
  <i> REFUSED by the remote's build ...: <the message in full>` and retries every `RETRY_S`
  (the remote's checkout may be fixed: `mac.sh --eval-checkout`), as for any failure.
* **Remote self-play** (5.3): the worker refuses the phase's checkpoint (error kind `encoding`,
  exit 78, its own log `REFUSED the checkpoint`); the run logs `worker <k> REFUSED the run's
  checkpoint (its checkout <repo> on <host> is not the run's encoding?): <the message>` (the first
  three times, then every 10 minutes), restarts the worker after the retry delay, and plays on
  alone (this machine's share grows: no game is lost).

**The opt-outs** (`allow_mismatch=True`), only where a mismatch is the point:

* `tools/fork_run.py <src.pt> <new run> [--game G]`: loads the source past the guard (an older
  encoding is what it widens), widens it (10.3; a conv network now follows the game's grid too)
  and **checks the fork** against this build's encoding before writing it (`--game`: recorded in
  the fork's config; widen keeps the source's game). `fork_run --source` (a continuation) copies
  files and loads nothing.
* Co-training's `resolver_init` (11): the same - loaded past the guard, widened, the result
  checked before it is written.
* `tools/selfplay_digest.py` (`tools/check_checkpoint_compat.py`): an old-vs-new comparison may
  play a checkpoint on a build of another encoding on purpose (`net.load` and `NetEvaluator`
  with `allow_mismatch`, when the checkout has the argument).
* `net.widen` itself loads nothing; the widen tests load their test networks (of no game of this
  build) with the opt-out.

**What does not change**: a matching checkpoint loads, builds and plays exactly as before (the
check only reads the config); toy checkpoints' configs are unchanged; `widen` to the layout a
network has (up to its identity) is the old widen (it records the new layout's identity).

**Co-training's first start** (found by MTG Zero's td1 migration, its ENGINE.md 60.7): with no
iteration holding both checkpoints, `CoRun.loop` used to save a fresh primary over an existing
`models/iter_0000.pt` - a draft network forked into the run, with the play network to come from
`resolver_init`, became the preset's fresh network without a word. Now (`CoRun.first_checkpoints`)
an existing checkpoint 0 of either network is kept (logged: `primary iteration 0 = the checkpoint
already there, kept: ...`), the missing one is created (the primary fresh, the resolver from
`resolver_init` or fresh), and both are checked by the guard; refused (exit 1, nothing written)
when the resolver has a checkpoint 0 and `resolver_init` is set too, or when either network has
checkpoints after 0 but no iteration has both.

Tests (`tests/test_encoding_guard.py`; the "v4" stand-ins: the toy's encoding with the players'
slots one float narrower, 191 features, and an MLP of 185 features, against the toy build as
"v5"): the check (both sides named; the game, an unknown game, a moved segment at the same width,
declared blocks ignored, the version compared when both carry one, the counts alone for an MLP,
the grid and its older normal form); every load site refusing (`load`, `Run.evaluator`, h2h both
ways, replay both checkpoints, targets, bench, webexport, the evaluator's rows, also on its worker
threads); the optimizer state of another network; run resume (exit 78, nothing written, the
state, the supervisor not restarting a refusing run); fork_run widening a v4 token checkpoint
exactly (outputs equal on real toy rows at the map) and the fork resuming, an MLP fork, a fork the
widen cannot bring onto the encoding not written; co-training (resolver_init from a v4 checkpoint,
widened exactly; a v4 resolver refused at the resume; the first start keeping a forked primary bit
for bit with resolver_init, keeping a resolver checkpoint 0, refusing the ambiguous directories);
the remote evaluation's refusal (exit 78, the line, the lock released; the run's log in full, the
retry); a remote self-play worker refusing while the run plays every game; selfplay_digest playing
a narrower checkpoint (the opt-out).

**Checks** (7132cb8 on main f3f5535; desktop 2026-10-29, CPU only, MTG Zero's delver2 live on the
GPU and the Mac). Tier "core behaviour, byte-identical defaults for matching checkpoints":

* `cargo test --release`: zero_search 50 passed, zero_webnet 3; `pytest tests`: **202 passed** (9
  min 49 s; the 15 above among them).
* **Byte identity** against f3f5535 (its own worktree and target dir): toy-smoke runs of 3
  iterations - MLP vs `ext:toyheur`, tokens, and `cq` targets + `eval_select cq` guard 2 + rolling -
  on both checkouts: every sample shard, `config.json`, `elo/results.json`, `models/index.json`
  byte for byte, every checkpoint tensor for tensor with the same config and extra (the files
  differ only in torch's archive root, named after the writing pid), the metrics equal without
  their timings; `tools/check_checkpoint_compat.py --package zero` with a toy MLP checkpoint, a toy
  token checkpoint, `random` and `uniform`: identical.
* **The other games** build against it (build only, at their HEADs with this core in their
  `third_party/zero-core`; the PyO3 crate, every test compiled, the package imported with this
  core): Glass Road Zero c1f56dc 36 s, Caverna Zero a6fad73 55 s, Kingdom Builder Zero 4202a4a 85 s.
* **Every game's existing checkpoints** (each run's newest, read-only): the paused runs of the
  hub's manifest with the worktree each resumes from (its game code and built engine, this core's
  Python): Glass Road grz2p_rb (run-c1f56dc), Caverna cvz2p_cq (run-5170335), Kingdom Builder kbz2p
  (run-001dd88), kbz2p_pin1 (fixed-setup), kbz2p_vq9 (run-5c93167), kbz2p_grid and kbz2p_gridgold
  (run-8f9ff15), kbz2p_xstbig_aux_rm and _rm_deep (run-99993c0): **all accepted**. Every other run
  too: Glass Road's 7 and Caverna's 5 with their HEAD builds; Kingdom Builder's 22 with the build of
  their game (HEAD; the aux-targets line run-2ece130 for kingdombuilder-xs / -xst; `options` for
  -glso): all accepted - on KB's HEAD build its xs / xst / glso runs are refused as games it does
  not have, correctly (they were trained on branches). The five `aux` runs' networks do not build
  on main's core at all (GridNet has no `aux`: that line's core is unmerged), whatever the guard.
  Dominion Zero and St. Pete Zero are not on zero-core (their own `net.load`): not affected.
* MTG Zero (its ENGINE.md 61): every v5 run accepted, every v4 and older refused; the main line's
  v4 `ab2-cqscale03/iter_0137` refused at run resume (exit 78) and by the evaluator, delver2's v5
  `iter_0000` accepted by both.

### 10.5 What a bigger token network costs (MTG Zero's Delver positions; desktop CPU, 2026-10-29)

`python -m <pkg>.az.bench infer --arch tokens --hidden D --blocks L --heads H --ff FF --shard S ...
--batches 96,192,384 --workers W --torch-threads T --ops` (the bench's new options: real positions
from shards, concurrent batches on threads as `infer_workers` runs them, the operators and
multiply-adds of one forward pass). MTG Zero's layout v5 (37,620 features, 1,024 actions), rows of
its Delver mirror run's shards (runs/delver2 iterations 19 and 21: 17 tokens a row on average);
the desktop's i9-13900K at below-normal priority beside other runs, 5 s per cell, the mean of two
rounds; heads of 32 floats, feed-forward 2 x width:

| network | parameters | multiply-adds / leaf | kernels / forward (batch 96 / 192) | 2 workers x 3 threads, batch 192 | 1 x 3, batch 96 / 192 / 384 | 1 x 1, batch 192 |
|---|---:|---:|---:|---:|---:|---:|
| 128 x 4 (MTG Zero's) | 1.55M | 21.9M | 515 / 585 | 4,627 | 3,290 / 3,250 / 4,280 | 1,622 |
| 128 x 8 | 2.08M | 39.9M | 595 / 705 | 2,918 (1.59x) | 1,954 / 1,960 / 2,742 | 980 (1.66x) |
| 192 x 6 | 3.59M | 65.5M | 555 / 645 | 1,996 (2.32x) | 1,497 / 1,500 / 1,540 | 642 (2.53x) |
| 192 x 8 | 4.18M | 85.1M | 595 / 705 | 1,536 (3.01x) | 1,072 / 1,040 / 1,580 | 526 (3.09x) |
| 256 x 6 | 5.93M | 112.8M | 555 / 645 | 1,114 (4.16x) | 894 / 941 / 1,232 | 379 (4.28x) |
| 256 x 8 | 6.98M | 147.1M | 595 / 705 | 864 (5.36x) | 670 / 716 / 928 | 314 (5.16x) |

(leaves per second; in brackets the cost per leaf against 128 x 4.)

* **On a CPU the cost follows the arithmetic**: 2.3x per leaf for 192 x 6 (3.0x the multiply-adds),
  4.2x for 256 x 6 (5.1x), 5.4x for 256 x 8 (6.7x). At width 128 the small operators still take a
  share (10.1), so the ratios stay below the multiply-adds'; the bigger networks are arithmetic.
  Concurrent batches (`infer_workers`) help every size alike.
* **The operators barely change**: a forward pass dispatches 515 kernel-launching operators at
  128 x 4 (batch 96: two chunks of rows), 555 at 6 layers of any width, 595 at 8, and 14 host syncs
  (the packing's `nonzero` per segment, the chunks' lengths) in every size. Eager CUDA inference of
  this network is bound by its operators, its syncs and the per-row work (the 37,620-float rows'
  upload, the packing) rather than by its arithmetic: MTG Zero's desktop self-plays at 5-6k leaves/s
  at batches of 96 and of ~384 alike (its ENGINE.md 54.11, 62). So a wider network costs the GPU
  little and a deeper one a few percent per layer (+8% operators at 6 layers, +16% at 8); not
  measured on the GPU (busy with the runs). MTG Zero's throughput plan for a bigger network: its
  ENGINE.md 62.

## 11. Deferred results and co-training (`zero/cotrain.py`; MTG Zero M6 part 2, 2026-10-27)

MTG Zero's team cube draft (its DESIGN.md 2.3, 5.1) is a game whose result is decided outside
it: a `team_draft` event (two teams draft and build decks) ends at a **deferred terminal**, and
its outcome is that of the event's `mtg` matches, which are themselves the self-play of the
play network. So one run trains two networks: the **primary** (`team_draft`, the draft network)
and the **resolver** (`mtg`, the play network). The core knows neither: the trait hook
(`deferred`, `deferred_payload`, `resolve`, `team_size`; docs/search.md 1.1), the search's
deferred leaf (docs/search.md 3.2), the driver's parking and queued setups (docs/search.md
2.8-2.9), and this loop. The toys play both roles in the tests (`toy_team` events whose matches
are toy games; preset `toy-cotrain-smoke`).

### 11.1 Settings and directories

`--resolver <game>` turns a run into a co-training run; the resolver's settings are
`resolver_cfg` (a dict of the run settings; `DEFAULTS` + the package's < its `preset` < the dict;
co-training keys inside it are errors, as are unknown ones), saved nested in `config.json` and
resolved into `<run>/resolver/config.json`; `resolver_init`, `resolver_conc`, `pending_max`,
`draft_eval_net_every`, `draft_eval_net_remote` (section 1.2). The primary keeps the run directory as a single-game run
does (`models/`, `data/`, `elo/`, `metrics.jsonl`, `progress.json`: the dashboard and the tools
work unchanged); the resolver has `<run>/resolver/` with the same layout. One player count per
game (`players`). Its `iters`, `games` and `rolling` are not used; its `eval_remote` sends its
evaluations to a remote (11.5), its `selfplay_remote` a share of its episodes (11.6). The run's
own `eval_remote` and `selfplay_remote` (the primary's) are refused; `draft_eval_net_remote` sends
the primary's resolver-network events to a remote (11.7). Each network has its own evaluation
move rule (section 5.4): the run's `eval_select` for the primary's seats, `resolver_cfg`'s for the
resolver's (its own evaluations and the episodes of `eval_yard_net`), each with its Elo pool
(`elo/[<rule>/]`, `resolver/elo/[<rule>/]`); `eval_yard_net` records both (`select`,
`resolver_select`, absent = 'visits'). `selfplay_select` the same way for each network's self-play.

**The game package's glue** (`GamePackage.deferred`, game -> `{'episodes': fn(payload) ->
[resolver spec JSON, ...], 'results': fn(payload, rows) -> results JSON}`): the defaults read
the payload's `"episodes"` list and send `[{"result": [...], "scores": [...]}, ...]` in order
(an episode cut at `max_steps` arrives as a draw). An empty episode list is an error.

### 11.2 An iteration

* **Self-play** (`CoRun.selfplay`): two rolling drivers in the run process - the primary's
  (network i - 1, recording) on the main thread and the resolver's (network i - 1, recording,
  `queue_only`, `resolver_conc` games in flight) on a worker thread (`infer.drive` on each; a
  driver is only ever called from its own thread: its `next_batch` holds it while it waits), joined
  by a `Coordinator`. Each game the primary parks becomes queued episodes (tags `<gid>/<k>`);
  their result rows come back by tag; when all are in, the game is resolved (its samples, with
  the real outcome, go to `data/iter_i/`; the episodes' samples to `resolver/data/iter_i/`). The
  phase ends when `games` primary games have been **resolved** in it - exactly: ready ones
  beyond the quota wait for the next phase - and the resolver's phase ends with it
  (`end_phase`). Games still playing, parked, queued or ready carry over in memory; the next
  iteration's networks restart their searches (a new epoch in both drivers).
  **Backpressure**: while `pending_max` resolver episodes are queued or playing, the primary
  stops taking batches (its in-flight games wait) and resolves what comes back.
* **Training** (`CoRun.train`): the primary's network on its window, then the resolver's on its
  own (`Run.train_model`, each with its own settings), then the non-finite guard on both
  checkpoints and optimizer states, then both are saved (`Run.save_trained`): a NaN in either
  network writes neither.
* **Evaluation** (`CoRun.finish_iter`): the resolver as a single-game run (its yardstick,
  random, Elo between its checkpoints: `resolver/metrics.jsonl`, `resolver/elo/`; on a remote
  with `resolver_cfg['eval_remote']`, 11.5), then the
  primary (`PrimaryRun`): its matches are **events** (`CoRun.event_match`), team seats
  alternating between the two players (seat s in team s % (n / team_size): game g gives team g %
  teams to the first player), each event's episodes played by a second, non-recording resolver
  driver - with the resolver's **yardstick** on every seat (`eval_yard` vs the primary's
  yardstick, `eval_random`, the Elo matches against earlier primary checkpoints, the reference
  games random vs the yardstick: the primary's Elo pool in `elo/`, one play agent) and, on
  evaluated iterations that are multiples of `draft_eval_net_every`, with the resolver's
  **network** of the same iteration (`eval_yard_net`: in metrics.jsonl, not in the Elo fit, a
  different play agent). A row's score is its team's outcome, its margin the team's score less
  the best other team's.
* **progress.json** during self-play: `{iter, phase, events_resolved, target, drafting,
  pending_events, matches_queued, matches_in_flight, matches_done, games, time}` (`games` =
  `events_resolved`, for the dashboard).
* **metrics.jsonl**: the primary's rows as a single-game run's (`selfplay.games` = games
  resolved) plus `selfplay.cotrain` (`events`, `episodes` and `episodes_per_h`, `pending_events`,
  `episodes_queued`, `episodes_waiting`, `parked_mb_max`, `event_latency_s` [mean, max] from park
  to resolve, `resolve_delay_ms` [mean, max] from the last episode's result to `resolve`,
  `backpressure_s`, `pending_max`, `resolver_conc`) and `eval_yard_net`;
  `resolver/metrics.jsonl` as a single-game run's (`selfplay.games` = episodes finished).

### 11.3 Resume, control, tools

* **Resume**: the newest iteration with both checkpoints; an iteration interrupted before both
  is replayed from its start (both data directories cleared; games in flight are gone, and the
  episodes already played for them stay only in the replayed iteration's data if it completes);
  one interrupted after them finishes its evaluations from each network's
  `models/iter_NNNN.json` (tested: PAUSE mid self-play, then resume; an iteration with both
  checkpoints and no metrics).
* **PAUSE / STOP / the supervisor / kill_run** work on the one process (a PAUSE noticed on the
  main thread stops the resolver thread first). `fork_run --source` copies a co-training run with
  both networks (at an iteration where both have a checkpoint). **Bootstrap**: iteration 0 = a
  fresh primary network and a fresh resolver network, or `resolver_init`; a checkpoint 0 of either
  network already there (e.g. a primary forked into the run) is kept, never overwritten (10.4).
  Both networks' checkpoints are checked by the encoding guard at every start (10.4).
* **NonFinite names the network** (`primary net (<game>): ...`, `resolver net (<game>): ...`);
  quarantined shards and failing batches go to `<run>/debug/` or `<run>/resolver/debug/`.

### 11.3a The pieces MTG Zero's part 3 added (2026-10-27, MTG Zero M6 part 3, its ENGINE.md 45)

* **The game package's glue** in use: `GamePackage.deferred['team_draft']` (mtgz/az/drafted.py)
  maps an event's payload to `mtg` specs (the drafted decks, the rest of the pools as
  sideboards, the real first picks, what each player knows is elsewhere, the match seed) and the
  matches' rows back, refusing a malformed result (ValueError: the coordinator's thread raises
  it on the main thread, the run aborts).
* **`SearchGame::sims_divisor`** (docs/search.md 1): `team_draft`'s deckbuilding at half a pick's
  simulations.
* **`TokenNet` actions by identity** (layout key `id_actions`, section 10): a segment's slot
  scores the action base + its identity (the team draft's pick, or putting a card in the deck,
  is the card's identity wherever it sits); layouts and checkpoints without the key are
  unchanged (MTG Zero: runs/mtg5's checkpoints play identically, `check_checkpoint_compat`).
* **`GamePackage.evals`**: a checkpoint's other yardsticks (game -> {name: fn(run, it) -> dict
  or None}), run on every evaluated iteration after the matches (`Run.extra_evals`, also for a
  co-training run's primary) and logged under `extras.<name>` in metrics.jsonl; measuring only.
  MTG Zero: the draft network's top-1 agreement with held-out human picks.
* **The dashboard** (section 5.1): a co-training run shows a tab per network (the primary's
  metrics at the run root, the resolver's from `<run>/resolver/`, each with its Elo chart, win
  rates, head-to-head table, losses and throughput), the co-training progress line (events
  resolved of the target, drafting, waiting for their matches, matches queued, in flight and
  done), tiles for the other network's Elo, matches per hour and the event latency, the primary's
  `eval_yard_net` in its win-rate chart, and an "Other yardsticks" chart of `extras`. A
  single-game run renders exactly as before (checked: the same page DOM for a copy of MTG
  Zero's runs/mtg5 with the old and the new dashboard).

### 11.4 Measured (desktop, 2026-10-27, other projects' runs on the machine)

`toy-cotrain-smoke` with `--games 300 --conc 64 --sims 32 --sims-fast 8 --resolver-conc 128`
and the resolver at 32 / 8 simulations, 2 iterations:

| | CPU (both networks) | CUDA (both networks) |
|---|---|---|
| resolver (toy matches) leaves/s, iterations 1 / 2 | 134k / 140k | 30.5k / 27.5k |
| resolver alone (a single-game rolling run, same settings) | | 27.8k / 31.6k |
| primary leaves/s; its time waiting for backpressure | 9.5k / 8.6k; 3.2 of 3.9 s | 2.2k / 1.6k; 13.7 of 17.2 s |
| event latency, park -> resolve (mean / max) | 0.77 / 1.0 s | |
| coordinator delay, last result -> resolve (mean / max) | 0.11 / 2.4 ms | |
| parked memory (max) | 32 KB (70 events of 64 features) | |

* **GPU sharing**: the two drivers on two threads, each with its own evaluator on the same GPU,
  do not slow each other measurably on the toy (the resolver's leaves/s with the primary
  alongside are within the run-to-run spread of it alone; the toy's tiny network is per-call
  bound on CUDA, slower than the CPU). The primary is ~7% of the leaves and waits for the
  episodes most of the time - MTG's budget (its DESIGN.md 2.3.9) expects ~11-13%.
* **Parking**: records are float16 (2 bytes per feature): MTG's 4v4 event, ~185 records of 14,488
  features, ~5.4 MB parked. MTG Zero measures `team_draft`'s (its ENGINE.md 43).
* **The coordinator** costs nothing measurable: the delay between an event's last episode and
  its `resolve` is a fraction of a millisecond (the primary resolves after every batch).
* Single-game runs are untouched: toy-smoke (rolling, and classic with margin_w, chance_k and
  cq targets) gives the same metrics, checkpoints, optimizer states and shards byte for byte as
  zero-core 681ba8d; the golden digests (`check_checkpoint_compat` against 681ba8d, random and
  uniform evaluators: the toy, and Glass Road Zero c1f56dc built with each core) are identical;
  `smoke_all.sh --with-this-core` with Glass Road, Caverna, Kingdom Builder and MTG Zero: green
  (b890655).

Tests: `tests/test_deferred.py` (the toy team game through the bindings, deferred leaves Rust vs
Python exactly and statistically, team utilities, empty option segments in the evaluators, the
driver's parking, `resolve`, queued setups, `queue_only`, tags), `tests/test_cotrain.py` (two
iterations of `toy-cotrain-smoke`, PAUSE mid self-play and resume, resume after the
checkpoints, NonFinite in each network, `resolver_init` from a single-game checkpoint,
`fork_run`, settings checks), the crate's `toy_team`, `tree` and `driver` tests (deferred leaves
evaluated once and backed up, team utilities, parking and resolving with samples, rolling
quotas counting resolved games, queued setups, float16 rounding).

### 11.5 The resolver's evaluation on a remote (`resolver_cfg['eval_remote']`; MTG Zero, 2026-10-27)

The resolver's evaluation is a single-game run's (11.2), and so is its remote: `eval_remote` in
`resolver_cfg` (a remote of the game package, section 5.2; `local` in the tests) sends the
resolver's yardstick, random and Elo matches there while both networks train. MTG Zero: the
MacBook plays the `mtg` play network's evaluation of its 4v4 team-draft run (~9 min there against
~30-70 on the shared desktop; its ENGINE.md 48).

* **The setting** is the resolver's own `eval_remote`, saved in `<run>/resolver/config.json`: it
  means there what it means for a single-game run (no new run setting). The run's top-level
  `eval_remote` would be the primary's and is refused: the primary's matches are events whose
  episodes the resolver's drivers play in the run process. A name the package does not have is
  refused at start.
* **The mirror** is `<workdir>/<run name>.resolver`: a sibling of the single-game runs' mirrors
  in the remote's workdir, so the lock (one evaluation at a time per workdir: a co-training run's
  resolver and a single-game run on the same remote queue behind each other) and
  `<workdir>/overrides.json` apply unchanged. Its config.json is the resolver's settings, a
  single-game run's, so the remote side (`python -m <pkg>.az.remote eval`) is unchanged.
* **Results**: `resolver/evals.jsonl` (5.2's rows), the mirror's Elo copied back to
  `resolver/elo/` and `resolver/models/index.json`; `resolver/metrics.jsonl` rows carry no
  evaluation. Log lines: `[resolver] evaluations on the remote 'mac' (host, mirror ...)`,
  `[resolver] iter N vs heuristic (remote mac, 560s): ...`, `[resolver] iter N elo ... (remote mac)`.
* **The primary's evaluation stays here**, unchanged: its yardstick-played events every evaluated
  iteration and its resolver-network events (`eval_yard_net`) every `draft_eval_net_every`, which
  load the resolver's checkpoint of the iteration here (or on a remote of their own with
  `draft_eval_net_remote`, 11.7). They do not wait for the remote.
* **Resume**: the resolver's evaluated iterations without a result are queued again at start
  (`RemoteEval.backfill`); nothing else is replayed. **An unreachable remote** is retried every
  5 minutes on its thread: both networks keep training and the primary keeps being evaluated; at
  the end of its iterations the run waits for the queue (progress phase `remote evaluations`),
  where PAUSE stops it (exit 75; the restart queues what is missing again). The co-training wait
  checks PAUSE; a single-game run's does not (unchanged).
* **On or off for a run** at a restart: `--resolver-cfg` with the whole dict (a flag replaces the
  saved dict), e.g. `--resolver-cfg '{"preset": "mtg-4", "eval_every": 2, "eval_remote": ""}'`.
* **The dashboard** (5.1): the resolver's tab merges `resolver/evals.jsonl` into its rows as a
  single-game run's view does; its Elo, win rates and head-to-head come from the copied-back
  `resolver/elo/`.
* **The primary's resolver-network events** (`eval_yard_net`) are not this setting's: the run
  setting `draft_eval_net_remote` sends them to a remote (11.7).

Tests (`tests/test_cotrain.py`): two iterations of `toy-cotrain-smoke` with the resolver's
`eval_remote` 'local' (the resolver's evaluations in `resolver/evals.jsonl` from the mirror
`<run>.resolver`, its Elo copied back, no evaluation in its metrics rows, the primary evaluated
here with `eval_yard_net`, the dashboard's merge); a restart with two evaluations pending (both
queued again and evaluated once, nothing retrained, the primary not evaluated again); an
unreachable remote (ssh to a host that does not resolve, retried every second: both networks
train and the primary is evaluated through every iteration, the run waits for the queue at the
end, PAUSE exits 75 there; restarted with a reachable remote, the three missing evaluations land);
the run's own `eval_remote` and an unknown remote name refused.

### 11.6 The resolver's episodes on a remote (`resolver_cfg['selfplay_remote']`; MTG Zero, 2026-10-28)

The resolver's self-play is the coordinator's queued episodes (11.2), so its remote self-play is a
**second consumer of that queue** - the shared queue of 5.3 with matches for units:
`selfplay_remote` in `resolver_cfg` (5.3's setting, the resolver's own) starts remote workers in
mode `matches` - each a rolling `queue_only` driver of the resolver's seats, persistent across
iterations - that play episodes the run hands them with the resolver's checkpoint i-1. MTG Zero:
the MacBook plays a share of the 4v4 run's drafted matches with the play network (its ENGINE.md 52,
56).

* **The split** (zero-core's `feat/shared-queue`; the first design capped the local driver's queue
  at max(16, resolver_conc / 4) and let each worker take conc x 1.25 outstanding, so in MTG Zero's
  td1 three Mac workers drained the queue and the desktop played none of an iteration's 1,812
  matches): the local resolver driver and every worker are consumers with **5.3's caps** - units =
  the iteration's matches (its `games` events x the mean matches per event the coordinator has seen;
  before any event: the consumers' slots), no head start, a consumer that had less work than its cap
  measured as if it had kept it busy - and **the coordinator keeps the local driver's share in
  reserve**: `feed` keeps the local driver's outstanding episodes (queued in it or playing:
  `Coordinator.local_out`) at its cap (`local_cap`, which the remote's manager keeps up to date),
  and the remote's manager takes for a worker, up to its cap less its outstanding, only what is
  queued **beyond** the local driver's room. So the local driver never waits for work a remote holds,
  and each worker takes its share as it has room, in the queue's order. The local consumer's bounds:
  2 x the resolver's groups and resolver_conc + max(16, resolver_conc / 4) (its slots and the first
  design's queue; a batch-bound one - a CUDA resolver driver, 5.3 - is taken up to its slots, never
  into that queue, by the batch rule); a worker's: 2 x its groups and conc + conc / 4. Without a remote, `feed` takes
  everything as before. **Backpressure**: `pending_max` 0 means 2 x resolver_conc + the remote's
  room (procs x (conc + conc / 4)), as before. The remote's phase starts when both checkpoints are
  written (5.3): while the networks are evaluated here it finishes its outstanding episodes and
  takes what is queued (the primary drafts nothing then); the events those rows complete wait,
  ready, for the next phase, and its rows from the head start count for that phase.
* **Results**: a worker answers each finished episode with its result row (`row` events, by tag);
  the run puts it into the coordinator (`collect_rows`) as the local driver's rows go in: an event
  is resolved when all its episodes' rows are in, from either machine. Its samples come back as
  5.3's compressed shards into `resolver/data/iter_i/`. The quota and the events are untouched:
  an iteration still ends when exactly `games` primary games are resolved here; the remote's
  episodes in flight at a phase's end carry over, as the local driver's do.
* **Requeue - work flows back**: episodes a worker cannot deliver go back to the **front** of the
  queue, where the local driver (or another worker) takes them: all of a worker's outstanding
  episodes when its session ends (its stdout closed: every row it sent has been taken) or stalls
  (killed after `STALL_S`); **all of them once it has backed off (an evaluation's lock, 5.3) for
  `selfplay_remote_giveback` s** (default 30; meanwhile its cap is 0, so it takes nothing new and
  the local driver's share is all of it); and any episode outstanding longer than its worker's
  **overdue limit** - `selfplay_remote_overdue` (default 3) x the worker's expected latency (its cap
  over its measured rate), at least `OVERDUE_MIN_S` (120 s), at most `selfplay_remote_timeout` s
  (default 3600: the first design's only limit) - of that worker's **running time** (its `active_s`,
  which stops between phases and while it backs off, so a long evaluation phase here does not time
  episodes out). A requeued episode may still finish on the worker: whichever copy's row comes
  second is dropped (`Coordinator.dupe_ok`; `duplicates_dropped`), so an episode is counted once and
  played at worst twice (its second copy's samples are kept: valid self-play of the same
  checkpoint). A row of an unknown tag that was never requeued is still an error.
* **Linger** (`selfplay_remote_linger` 1 in `resolver_cfg`): the workers play their outstanding
  episodes on while the run trains; their rows go into the coordinator at once (their events wait,
  ready, for the next phase), their samples into the next iteration's shards (5.3).
* **The primary stays here**: the run's top-level `selfplay_remote` (the primary's) is refused:
  its parked games live in the run process's driver, and their coordinator with them. Offloading
  the drafting would mean a whole co-training loop on the remote (both networks, both drivers, the
  coordinator), events resolved there and both kinds of shards shipped back: not a small change,
  not done.
* **Metrics**: the resolver's `selfplay.remote` (5.3's keys, plus `matches` (rows taken),
  `assigned`, `requeued` (lost sessions), `requeued_timeout` (overdue), `requeued_backoff` (given
  back), `duplicates_dropped`, `outstanding`) and `selfplay.queue` (5.3's `units, inflight,
  consumers, rates, staleness`, plus `local_cap`); its `selfplay.games` stays the local driver's; the
  primary's `selfplay.cotrain.episodes` counts both, `episodes_remote` the remote's share,
  `duplicates_dropped`, `staleness` (the primary's rows by checkpoint lag). progress.json:
  `matches_remote` = 5.3's `remote` + `outstanding`, `matches`. Log lines `[resolver] episodes also
  on the remote 'mac' (...)`, `[resolver] iter N remote self-play: ...; matches 412 back, 960
  assigned, 300 outstanding, 0 + 0 + 0 requeued (lost, overdue, backing off), 0 duplicates dropped`,
  `[resolver] iter N shared queue: caps local 146.0 -> 150, w0 144.0 -> 141, ...; rates/s ...;
  staleness: ...`, `remote self-play mac: worker 1: backing off for 31s (...eval.lock): its 140
  outstanding episodes requeued here (a second result is dropped)`.
* **On or off at a restart**: `--resolver-cfg` with the whole dict, e.g. `--resolver-cfg
  '{"preset": "mtg-4", "eval_every": 2, "eval_remote": "mac", "selfplay_remote": "mac"}'`.

Tests (`tests/test_remoteplay.py`): toy-cotrain-smoke with the resolver's `selfplay_remote`
'local' (exact quotas, the remote's rows and shards counted, the backpressure's room, both
consumers in the queue's metrics, the local driver playing every iteration), a remote with four
times the local driver's room (the local driver's share kept for it), a worker that backs off
mid-phase (its matches given back, quotas exact), a slow worker whose episodes time out and are
requeued (duplicates dropped, quotas exact), a worker that dies mid-phase every time (its episodes
requeued, the worker restarted), the settings refused; `tests/test_workqueue.py` (the coordinator's
reserve, matches per event).

### 11.7 The primary's resolver-network events on a remote (`draft_eval_net_remote`; MTG Zero, 2026-10-28)

`eval_yard_net` (11.2) - the primary's checkpoint against its yardstick with every event's
episodes played by the resolver's **network** of the same iteration - is the largest single
evaluation of a co-training run (MTG Zero's 4v4 run: 32 events, 384 play-network matches every 4th
iteration, 51-88 minutes on the desktop, its ENGINE.md 52.5 and 53), and its matches are played by
one driver whose last few matches each need tens of thousands of sequential network calls (11.8).
The run setting **`draft_eval_net_remote`** (a remote of the game package, 5.2; `local` in the
tests; '' = here, the default) plays it on another machine while both networks train. The rest
of the primary's evaluation (its yardstick- and random-played events, its Elo, the package's other
yardsticks) stays here; the resolver's own evaluation has its own setting (11.5).

* **Generic**: nothing game-specific - the remote rebuilds `CoRun` from a mirror of the run and
  calls `CoRun.eval_net(it)`, the function that plays the events here (`PrimaryRun.evaluate` calls
  it when the setting is off): the same drivers, coordinator, the package's deferred glue, seeds
  (`7919 * it + 3`) and settings (the remote's overrides aside).
* **Local side** (`zero.remote.RemoteNetEval`, a `RemoteEval` - its queue, thread, retries and
  drain): `CoRun.finish_iter` appends the primary's metrics row without `eval_yard_net` (and
  without `phases.eval_net_s`) and queues the iteration when its `eval_yard_net` is due
  (`CoRun.eval_net_due`: an evaluated iteration that is a multiple of `draft_eval_net_every`, with
  `eval_games`). The thread ships the **mirror** `<workdir>/<run name>`, laid out as the run:
  `config.json` (the run's settings with every remote setting off - `eval_remote`,
  `selfplay_remote`, `draft_eval_net_remote`, and the resolver's `eval_remote` and
  `selfplay_remote` - and the remote's `overrides` on both networks' settings: the top level and
  `resolver_cfg`), `models/iter_NNNN.pt` and `resolver/models/iter_NNNN.pt` of that iteration
  only (each under a temporary name, then renamed); runs `python -m <pkg>.az.remote evalnet
  <mirror> <iter>` in the remote's `repo`; appends `{iter, eval_yard_net, time, host, seconds,
  remote_s}` (`seconds`: from the queue, shipping included; `remote_s`: the events' own) to the
  run's **`evals.jsonl`** and logs `iter N vs <yardstick> (episodes by the resolver net; remote
  mac, 812s): {...}`. Nothing is copied back: `eval_yard_net` is not in the Elo fit.
* **Remote side** (`zero.remote.evaluate_net_mirror`): `CoRun(mirror, config)` (both networks'
  directories in the mirror; the remote's `<workdir>/overrides.json` on both networks' settings),
  `CoRun.eval_net(it)`, the result printed as `ZERO_REMOTE_RESULT {json}` and appended to the
  mirror's `metrics.jsonl`. **One evaluation at a time per workdir**: the same `.eval.lock` as
  `eval`, so the resolver's evaluation (11.5) and these events queue behind each other on the
  remote, and remote self-play there backs off while either runs (5.3). MTG Zero's commit check
  applies unchanged (`python -m mtgz.az.remote` checks the checkout before any subcommand).
* **A failure never blocks training**: an attempt that fails (the remote asleep, unreachable, its
  checkout refused) is logged (`remote eval mac iter 8: ...; retry in 300s`) and retried every
  `RETRY_S` on its thread; later iterations queue behind it. **Restart**: the due iterations up to
  the newest finished one without `eval_yard_net` in their metrics row or in `evals.jsonl` are
  queued again (`RemoteNetEval.backfill`); nothing is replayed. At the end of the iterations the
  run waits for both remote queues (progress phase `remote evaluations`; PAUSE ends the wait, the
  restart queues what is missing again).
* **The dashboard** (5.1) merges `eval_yard_net` from the root's `evals.jsonl` into the primary's
  rows (the win-rate chart's resolver-network line), as it merges a remote's `eval_yard`.
* **On or off at a restart**: `--draft-eval-net-remote mac` / `--draft-eval-net-remote ''`
  (saved in config.json; a top-level co-training setting, so no `--resolver-cfg` is needed; inside
  `resolver_cfg` it is refused, as is an unknown remote name at start). Turned off, the iterations
  evaluated remotely keep their `evals.jsonl` rows; nothing is played again here.
* **`ZERO_EVAL_WORKDIR`** (environment): another evaluation workdir on a named remote - its own
  mirrors and its own `.eval.lock` (an end-to-end test beside a live run's workdir); an absolute
  path on the remote (Git Bash rewrites `/Users/...` in a Windows process's environment:
  `MSYS2_ENV_CONV_EXCL=ZERO_EVAL_WORKDIR`). Remote self-play then backs off for that lock.

Tests (`tests/test_cotrain.py`, the remote 'local'): two iterations of `toy-cotrain-smoke` with
`draft_eval_net_remote` (iterations 0 and 2's `eval_yard_net` in `evals.jsonl`, none in the metrics
rows, the mirror holding the run's settings with every remote setting off and both networks'
checkpoints of those iterations only, the lock released, the dashboard's merge); a restart with
one missing (played again alone, nothing retrained); an unreachable remote (retried every second,
both networks trained and evaluated here through every iteration, PAUSE in the final wait; then
with a reachable remote both land); the setting refused inside `resolver_cfg` and with an unknown
name; `ZERO_EVAL_WORKDIR`. MTG Zero's `tests/test_mac_remote.py`: its `evalnet` refused at another
commit before anything is evaluated or locked; a team_draft mirror evaluated by the remote side.

### 11.8 One slow game step held up the whole driver (MTG Zero, 2026-10-28)

**The symptom** (MTG Zero's ENGINE.md 52.8, 53): `team_draft-smoke --iters 3 --games 8` (CPU) sat
~10 minutes in iteration 2's self-play - 6 of 8 events resolved, 16 matches in flight, the resolver
thread waiting in the driver's `next_batch`, the primary on backpressure, the process idle - and
then went on; the same at an older core, at the same point (both runs had played identical games
up to it). Separately, MTG Zero's 4v4 run td1 spent 51 and 88 minutes on its iterations 0 and 4's
`eval_yard_net` with its process mostly idle.

**Root cause, the stall.** A group's batch is assembled when every game of the group has made its
step (its search and moves up to its next leaf; docs/search.md 2.3), and `next_batch` served the
groups in **strict round-robin order**. So one game whose step takes long - an engine's slow path
(52.8's runs used engines that still had MTG Zero's exponential payment search, its ENGINE.md 51.3;
which step it was is not known) - stopped **every game of the driver**: its own group waits for it
by design, and the other
group, its batch answered and its next one ready, waited for its turn behind it. In co-training
the resolver's episode player is that driver: no episode finishes, the episodes outstanding stay
at `pending_max`, the primary waits on its backpressure, and the process idles (one search
thread in the slow step) until the step ends. Two things made it worse:
* `in_flight()` (co-training's progress snapshot, every half second on the resolver's thread)
  locked every chunk of the driver **holding the GIL** (the binding does not release it), so when
  it met the chunk of the slow game it blocked every Python thread of the process - the primary's
  backpressure loop and progress.json included - until the step ended;
* a `queue_only` driver's first group re-dispatched took the whole queue: an evaluation's matches,
  queued at once, started 383 / 1 in td1's `eval_yard_net` (768 slots, 384 matches), so its two
  groups' search and inference no longer overlapped.

What made the step slow in 52.8's runs could not be reproduced: 16 runs of the smoke since (69
iterations, ~2,400 drafted matches; 27 of the iterations on 692a52c's engine, the 52.8 runs') did
not idle; the slow-step counters below now say it when it happens.

**The fix** (zero-core 9714be3):
* A `queue_only` driver (co-training's episode players - the resolver's self-play driver,
  `event_match`'s, a remote matches worker's - whose results depend on timing anyway) serves the
  **first of its working groups that is ready**, in round-robin order from the cursor: a slow game
  holds up its own group only; the others play on (with `groups` 2, half the episodes in flight).
  Every other driver keeps the strict order (a run reproducible from its seed).
* `queue_share`: a group dispatched starts its share of everything in flight or queued, never fewer
  than the other groups' free slots cannot take (no free slot idles while setups wait; a setup's
  gid is still its place in the queue): the evaluation's 384 matches start 192 / 192.
* `in_flight()` try-locks each chunk; a chunk being searched counts its games as of its group's
  last dispatch (exact once the phase is paused).
* `DriverStats`: `step_max_s`, `step_max_gid` (the longest step of a game so far) and `slow_steps`
  (steps over 10 s); `CoRun` logs `iter N self-play (resolver driver): K game step(s) took over 10 s
  (the longest so far S s, game G): an engine slow path?` (and the same after each event match) and
  counts them in the rolling stats (`slow_steps`).
* Not changed: the step itself (a game's search is one thread's), and the end of a phase - a group
  in a slow step pauses when it ends - so a slow step still delays its own group's games and an
  iteration's end; it no longer stops everything else for its duration.

**Regression tests** (the crate's `driver` tests): `a_slow_step_holds_up_its_own_group_only` - a
`queue_only` driver of 2 groups x 4 with 8 queued toy games, one of which sleeps 1.5 s in one
step: the other group plays on to its last batch while it sleeps (in strict order it got one
batch: the test fails on the old code), the step counted (`step_max_s`, `step_max_gid`), and
`in_flight()` returns at once while a search thread sleeps; `queued_setups_are_shared_between_the_groups`
(8 setups for 2 x 8 slots: 4 / 4, was 7 / 1; the same games as with one group);
`queued_setups_fill_every_free_slot_when_the_queue_is_long`.

**The evaluation's time** (`eval_yard_net`; MTG Zero's ENGINE.md 53.2 measures td1's): its
events are drafted first (all at once, a few rows per network call), then their matches all start
in one `queue_only` driver, and the evaluation lasts as long as its **longest match**: one leaf per
game per call, so the last matches' tens of thousands of calls follow one another with a handful of
rows each. A call costs about the same whatever its rows on a GPU shared with training (MTG Zero's
token networks on the desktop's RTX 5090, td1 training beside: ~9-11 ms for 1-32 rows, 14 / 21 /
49 ms for 96 / 192 / 384; on 2 CPU threads 3.0 / 4.3 / 6.6 ms for 1 / 8 / 32 rows, 15 / 27 / 50 for
96 / 192 / 384), and the evaluating thread is busy with calls all the time (the search threads
idle: the process looks idle): the phase is latency-bound, not stalled, and the fixes above do not
shorten it (the 383 / 1 start cost nothing measurable there: one thread evaluates both groups'
batches in turn either way). The levers are the remote (11.7: its CPU), CPU inference for the
evaluation's small batches, and fewer or shorter calls.

## 12. The move rule: visits vs completed Q (MTG Zero's delver1, 2026-10-28)

### 12.1 The experiment

MTG Zero's Legacy Izzet Delver mirror run (runs/delver1; its ENGINE.md 54) evaluates at
`eval_sims` 32. An experiment (zero-core branch exp/select-rule 945d75e + 4a198cb, MTG Zero
tools/exp_select.py on branch exp/select-rule 102083d; not merged) measured each checkpoint
against the heuristic bot (the yardstick) with three rules for the evaluation seat, 200 games a
cell, paired deals:

| checkpoint | A: visits | B: cq | C: argmax Q, >= 2 visits | C: >= 3 visits |
|---|---|---|---|---|
| iter 12 | 42.0% | **57.0%** | 51.0% | 54.5% |
| iter 14 | 52.0% | **61.0%** | 45.0% | 49.5% |

Head to head (the same checkpoint, B vs A): B 62.5% at iter 12, 58% at iter 14 (+56 to +89 Elo).
At 32 simulations the prior's favourite collects most visits even where the search found it
worse, so argmax visits mostly plays the prior; B lets Q overrule it. The user's decision
(2026-10-28): B as an opt-in setting with today's rule the default everywhere; delver1's
evaluation switches to B at its next re-pin; self-play later, by a forked A/B run.

Then on the production path (`Run.evaluate` with `eval_select`, MTG Zero's tools/select_eval.py,
its ENGINE.md 57; 200 games a cell against the heuristic at `eval_sims` 32, paired deals):

| checkpoint | visits | cq | cq, >= 2 visits | cq, >= 4 visits |
|---|---|---|---|---|
| iter 12 | 46.0% | **61.5%** | - | - |
| iter 26 | 71.0% | 77.0% | **77.5%** | 76.5% |
| iter 27 | 71.0% | 73.0% | 73.5% | **76.5%** |

Paired against visits, iterations 26 + 27 (400 games): cq +4.0 +- 2.6 points, cq-n2 +4.5 +- 2.6,
cq-n4 +5.5 +- 2.5; between the guards within 1.5 +- 2.0. Head to head at iteration 27 (the same
checkpoint, cq vs visits, 200 games): 51.0 +- 3.5%. The gain shrinks as the network improves
(+15.5 points at iteration 12, +2 to +6 at 26-27); the recommended re-pin of delver1 uses the guard 4 (MTG Zero
ENGINE.md 57.4).

### 12.2 What was built

`driver::Select` (docs/search.md 3.3: 'visits' | 'cq' + the guard), seat keys `select` /
`select_min_visits`, `SearchTree.root_select`, the Python reference `mcts.select_index`; run
settings `eval_select`, `selfplay_select` and their guards (1.2); one Elo pool per rule, raw
Elo offsets, runs without random games (5.4); the rule on the remote (5.2), in co-training (11),
h2h and replays. The experiment's 'q' rule (C) lost to B and was not kept; its move log is
(`move_log`, docs/search.md 2.5).

### 12.3 Checks

* Rust: the golden records (`visits_records_unchanged`) and a new evaluation-match golden at
  the default rule, equal to zero-core 3c26d9f's (12492034373733022939, computed there with the
  same test code); new goldens for 'cq' (self-play records (329, 8716559848855127382), a cq vs
  visits match 10451580073337998459, a guarded cq-n2 vs cq-n2 match 2225760264326051811).
* Parity: docs/search.md 6 (the Python rule on Rust roots = the Rust driver's move on all 192
  cases checked; Rust-tree vs Python-tree 'cq' seats in games).
* Toy-smoke byte for byte against 3c26d9f (3 iterations each: classic; rolling; margin_w 0.2 +
  chance_k 2 + cq targets + eval_every 2), the new settings left out and given at their
  defaults: metrics.jsonl (timing fields aside), the optimizer state, every shard, elo/results.json,
  elo/ratings.json (its `updated` aside) and models/index.json identical; every checkpoint's zip
  members byte for byte (the files themselves differ in the archive's folder name only: torch.save
  takes it from the temporary file name, which holds the writer's pid). config.json gains the five
  new keys at their defaults. 85 existing run configs of the game repos (runs/*/config.json and
  resolvers') pass the new checks and resolve to the rule 'visits'.
* The same byte-for-byte comparison again after the "evaluated" flags (a167ee2; and on
  feat/select-rule-on-queue against 9453e18): a default run's ratings.json and index.json carry
  no flag; `tools/check_checkpoint_compat.py --package zero` (random, uniform and a checkpoint's
  evaluator: self-play records and results for both targets, the match vs random) identical.
* A toy rehearsal of delver1's re-pin: a run evaluated by the remote 'local' at the default rule
  (iterations 0-4, `eval_every` 2), restarted with `--eval-select cq --eval-select-min-visits 4
  --eval-random-games 0 --elo-offsets 1,2,4,8 --elo-games 25` (iterations 5-8): the run's and
  the mirror's `elo/` byte for byte as before, `elo/cq-n4/` since 6 with iterations 6 and 8
  logged and their raw-offset matches (k-1, k-2, k-4, k-8) of 25 games, the reference games once,
  the other checkpoints flagged, evals.jsonl rows tagged, no random games, both dashboards on
  the new pool (zero's with the 'visits' line).
* MTG Zero (its ENGINE.md 57): the production evaluation path (`Run.evaluate`) of delver1's
  checkpoints reproduces the experiment's direction (12.1); the guard's effect is measured there;
  its engine tests and pytest at the pin; the other games (Glass Road, Caverna, Kingdom Builder)
  build against the core.

### 12.4 A self-play A/B (not run)

`selfplay_select cq` changes self-play's moves after `temp_moves` only (the recorded targets
stay `policy_target`'s, the first `temp_moves` decisions are sampled as before). An A/B: fork a
run at an iteration (`tools/fork_run.py`), set `--selfplay-select cq` on the fork and keep the
parent as the control (same seed, same settings otherwise), evaluate both with the same
`eval_select` (both pools then compare: the same rule), and read it with an h2h between the
arms' checkpoints of equal iterations (`zero.h2h`, both sides at the same `--select`), as the
policy-target A/Bs (section 9) were read. Expected effects to watch: games follow Q earlier, so
self-play's value targets see more search-preferred lines; the visit-count policy targets do
not change meaning (a cq target is the natural pair, as Gumbel AlphaZero trains it).

### 12.5 The selection scale is its own setting (2026-10-29)

**Why.** MTG Zero's overnight A/B series (its `runs/ab/results.md`) forked the Delver main line
with one setting changed per experiment and read each with an h2h of the two arms' checkpoints at
their runs' own rules. Experiment 2, `--cq-c-scale 0.3` (the 'cq' training target's scale), won
64.2 +- 2.4% - but the 'cq' evaluation rule read the same `cq_c_scale`, so the fork *selected*
its moves at 0.3 (`cq-n4-v50-s0.3`) while the control selected at 1.0 (`cq-n4`). Re-run with both
sides at `--select visits` (400 games, the same checkpoints): 49.0 +- 2.5%, no training effect.
The 64% was the selection scale. One knob controlled both, so a training A/B silently changed the
evaluation rule; now they are two settings (1.2, 5.4).

**What was built.**

* Run settings `eval_select_c_visit`, `eval_select_c_scale`, `selfplay_select_c_visit`,
  `selfplay_select_c_scale` (default None = unset = follow `cq_c_visit` / `cq_c_scale`; validated
  by `check_eval_settings`, for `resolver_cfg` too; `none` on the command line unsets).
* `mcts.select_settings(cfg, pre)` (rule, guard, effective transform), `mcts.select_transform`,
  `mcts.check_select_transform`; `elo.eval_tag` and zerodash's `runfiles.elo_pool` name the pool
  after the effective transform.
* Seats: `games.nn_seat(..., select_c_visit, select_c_scale)` (the dict depends on the effective
  rule only, see 5.4), `play.NNSeat` the same (`select_transform`, `select_tag`; 'py' seats'
  formula and 'rs' seats' `root_select` take it); the bindings' seat keys `select_c_visit` /
  `select_c_scale` (absent: the seat's `cq_c_*`; an error on a 'visits' seat) feed `Select::Cq`,
  while `PolicyTarget::CompletedQ` keeps `cq_c_*` (the Rust driver already held the two apart).
* `zero.h2h`: a run's rule with its effective transform (`run_rule`), `--cq-c-visit` /
  `--cq-c-scale` for both sides (with `--select`: 50 / 1 when not given, as before; without it now
  over each 'cq' side's run rule - an error if no side plays 'cq'), per side `--a-cq-c-visit`,
  `--a-cq-c-scale`, `--b-cq-c-visit`, `--b-cq-c-scale` (over the both-sides flags; an error on a
  'visits' side). `python -m <pkg>.az.h2h K.pt K.pt --select cq --select-min-visits 4
  --a-cq-c-scale 0.3 --b-cq-c-scale 1.0` compares two selection scales on one checkpoint
  (`select_a` 'cq-n4-v50-s0.3', `select_b` 'cq-n4').
* `zero.replay`: the run's rule with its transform; `--cq-c-visit` / `--cq-c-scale` (with or
  without `--select`). `zero.parity h2h --select-c-visit / --select-c-scale`; parity's rules include
  ('cq', 4, 50, 0.3).

**Checks** (zero-core main 45885aa as the base; `tests/test_select_scale.py`, 10 tests):

* Defaults byte for byte against 45885aa: the three toy-smoke runs of 12.3 (classic; rolling;
  margin_w 0.2 + chance_k 2 + cq targets + `eval_every` 2), each with the settings left out and given
  explicitly as `none`: shards, checkpoints (by zip member), optimizer state, metrics (timing aside),
  Elo files and models/index.json identical; config.json gains the four keys (null). The same for
  a run *at* the 'cq' rule with a non-default scale (cq targets, `cq_c_scale` 0.3, `eval_select cq`
  guard 2, `selfplay_select cq`, raw Elo offsets, 4 iterations), unset and given equal to 0.3: 11
  files identical, pool `cq-n2-v50-s0.3` byte for byte (the live Delver line's shape).
  `tools/check_checkpoint_compat.py --package zero` (random, uniform, a toy checkpoint):
  identical.
* Unset vs given equal to `cq_c_scale` (0.5, both rules, a toy run): every shard, checkpoint and
  Elo file identical, the same pool; the seat dicts key for key.
* The selection scale changes selection only: a toy run with `--eval-select-c-scale 0.3` beside
  its control has identical shards and checkpoints, its evaluations in pool `cq-v50-s0.3`; a run
  with `--cq-c-scale 0.3 --eval-select-c-scale 1.0` has other shards (the targets) and the
  control's pool `cq`; `--selfplay-select-c-scale 0.3` at `cq_c_scale` 1.0 vs `--cq-c-scale 0.3`
  plays the same first self-play games with other targets. In the drivers (Rust driver through the
  bindings, the Python driver's 'py' and 'rs' seats, and a Rust unit test
  `select_transform_moves_only_the_moves_never_the_targets`): a self-play seat with target 1.0 +
  rule 0.3 plays the games of target 0.3 + rule 0.3 (every record field but pi equal) and its
  targets are the 1.0 targets of the same roots (checked against a target-0 run: per record,
  (ln pi_A - ln pi_E) - (ln pi_B - ln pi_E) / 0.3 is constant over the options to < 1e-3); rule
  1.0 plays other games (the check has power).
* Rust vs Python at the live rule's shape (cq, guard 4, 0.3 beside a 1.0 target): the Python
  formula on Rust roots = the Rust driver's choice on all cases (`parity.select_formula`, now 224
  with the added rule); a Rust-tree seat vs a Python-tree seat ~50% (300 games); an 'rs' seat with
  select_c_scale 0.3 plays exactly the games of one at `cq_c_scale` 0.3.
* h2h per side (both flags, the side's over the both-sides', over a run's rule, the errors),
  replays with the transform, the command line's `none`.

## 13. Distillation: a bigger network that starts where the run is (`zero/distill.py`, `tools/distill.py`; MTG Zero's Delver line, 2026-10-29)

**Why.** A run whose network has plateaued may be bound by the network's capacity (MTG Zero's Delver
mirror, runs/delver2: a 128 x 4 token network at ~86-90% against its heuristic and about +1 Elo an
iteration in 400-game head-to-heads; its ENGINE.md 62). A bigger network trained from scratch would
replay the whole run; a bigger network **distilled** from the run's checkpoint starts where the run is.
`distill` trains a fresh network of the new size - any width, depth or architecture of this build's
encoding - on the run's own sample shards to compute what the checkpoint computes, and the run then
continues from it (13.3).

### 13.1 The method

* **The student** (`new_student(teacher, arch, hidden, blocks, heads, ff, ch, seed, game)`): `make_net`
  for the teacher's game with this build's encoding (the feature / action counts, the game package's
  token layout - its version recorded - or board grid), seeded; a token network's policy starts
  uniform as always (10).
* **The positions**: a `data.Window` over the shards (RAM-bounded as the run's: each shard
  subsampled at one rate), every shard checked to be of this build's encoding (its meta's game,
  feature and action counts: the student reads this build's inputs). Held out: other shards (default
  the last one given), never trained on.
* **The teacher's answers** (`outputs(model, W)`): once per window row, as the search sees them - the
  policy as probabilities over the row's legal actions (one float per option, aligned with the
  window's `opt_action`; the evaluator's segment softmax: equal to `NetEvaluator`'s answers, tested),
  P(win) per seat present, the margin. `Taught(W, T)` adds their dense forms to the window's batches,
  so `data.batches` prefetches them as the run's training does.
* **The loss** (`losses`; w = `teacher_w`): policy (1 - w) CE(pi, p_s) + w CE(p_t, p_s) over the
  legal actions; value (1 - w) CE((1 - vq) z + vq q, v_s) + w CE(v_t, v_s) over the seats present;
  margin `margin_w` ((1 - w) |sc - m_s|^2 + w |m_t - m_s|^2) / 2. **w = 1 (the default) is pure
  distillation**: the student learns the teacher's function - its whole policy over the legal actions
  and its value estimate, where a shard's z is one game's 0 / 1 - so the warm start is the teacher,
  not a sibling trained on the same last few iterations; w = 0 is the run's own supervised loss on
  the shards (tested equal). AdamW, gradient-norm clip 5, the non-finite check before every step
  (4.1), a linear warm-up, then a cosine decay to `lr_min`.
* **Agreement** (`agreement(S, T, W)`, on held-out rows): KL(teacher || student) of the policies over
  the legal actions (nats; the mean and the 90th percentile over the decisions with at least 2 legal
  actions), the share of them whose argmax agrees (top-1), the squared and absolute differences of
  P(win) over the seats present, the margin's squared difference, and each network's losses against
  the shards' own targets (policy cross-entropy against pi, value cross-entropy against z, value
  accuracy). A network against itself: 0, 1.0, 0 (tested).

### 13.2 Why not a function-preserving widen

Net2Wider / Net2Deeper growth would give the bigger network the teacher's function exactly, with no
training at all. **Depth** is clean for this pre-norm transformer: a block whose attention output and
second feed-forward layer are zero adds nothing (x + 0 + 0), and its other weights can be random, so
it learns from its first step. **Width** is not: the blocks' LayerNorms average over the width, so new
residual channels must keep every token's mean and variance - only exact duplicates do ([h, h]: width
x 2 only, the heads doubled at the same head size, every reader of the stream halved) - and duplicated
units train in lockstep (with an even split their gradients and AdamW's updates stay identical for
ever; uneven splits drift apart only slowly, at second order), so the network keeps the old one's
capacity until noise, which breaks the exactness, separates them. Distillation has neither problem:
any size (192 is not 2 x 128), any architecture (an MLP into a token network, tested), a fresh network
that uses all of its capacity, and the same tool for every game. Its costs are a training phase (MTG
Zero's 192 x 6: an estimated 35-50 minutes on the desktop GPU, its ENGINE.md 62.6) and a fidelity short of
exact, which the held-out agreement and a head-to-head against the teacher measure before a run uses
the student.

### 13.3 The tool, and continuing a run from the student

```
python tools/distill.py TEACHER.pt OUT.pt --data runs/a/data [--heldout runs/a/data/iter_K/r00_000.npz] \
    --hidden 192 --blocks 6 --heads 6 --ff 384 [--arch tokens] --steps 12000 --batch 512 --lr 1e-3 \
    --warmup 500 --ram-budget-gb 24 --eval-every 1000 --save-every 2000 --keep-best --device cuda \
    [--resume | --init S.pt]
python tools/distill.py --compare OUT.pt TEACHER.pt --data runs/a/data/iter_K/r00_000.npz   # agreement only
python tools/fork_run.py --source runs/a --dest runs/a-big --iter K --net OUT.pt [--set ...]
```

* `distill.py` (its module docstring): the student's size (default the teacher's: `--hidden`,
  `--blocks`, `--heads`, `--ff`, `--arch`, `--ch`), the training (`--steps` or `--epochs`, `--batch`,
  `--lr`, `--lr-min`, `--warmup`, `--wd`, `--teacher-w`, `--value-q`, `--margin-w`), the window
  (`--ram-budget-gb`, `--heldout-rows`), held-out evaluations every `--eval-every` steps (with
  `--keep-best` also `OUT.best.pt`, the student at its lowest held-out KL), `OUT.pt.partial` (student,
  optimizer, step) every `--save-every` and `--resume` from it, `--init S.pt` (a further distillation
  from a student: its size, its weights), a JSON line
  per report on stdout and in `OUT.pt.log.jsonl`, below-normal priority. OUT.pt is a normal
  checkpoint, checked by the encoding guard before it is written; its `extra` records
  `distilled_from`, `teacher_sha256`, `teacher_extra` and `distill` (the settings, both networks'
  sizes, the shards, every held-out agreement).
* **`fork_run --source --net`**: a continuation (2) of the run at K with the student as its
  checkpoint K (checked by the guard for the run's game), **without the source's optimizer state**
  (it belongs to the source's network: the fork's first training starts a fresh AdamW with its
  warm-up), **K's Elo results dropped** from every pool (they rated the source's network;
  ratings.json is refit at the fork's next evaluation) and the network's size in the fork's
  config.json (`arch`, `hidden`, `blocks`, `heads`, `ff` / `ch`); the window's shards, the
  checkpoints before K and the metrics as in any continuation. Nothing in the run loop changed: a
  run resumes the student like any checkpoint (its size comes from its config), so an A/B of a
  bigger network is a fork with `--net` against the plain continuation of the source on one build
  (MTG Zero's ENGINE.md 62.6). `fork_run OUT.pt <new run>` makes it iteration 0 of a new run instead
  (no window, a new Elo pool).

### 13.4 Measured

* **Toy** (the tests' setting, CPU): a random 64 x 2 MLP teacher's answers on 3,471 toy positions
  (120 games), 1,145 positions of 40 other games held out; 300 steps of 128 rows. A 32 x 2 token
  student (another architecture): KL 0.077 -> 0.0087, top-1 0.12 -> 0.92, P(win) squared error 0.0109
  -> 0.0003; a 128 x 3 MLP student: KL 0.095 -> 0.0034, top-1 0.18 -> 0.94, 0.0136 -> 0.0002.
* **MTG Zero's delver2** (its ENGINE.md 62.5; CPU, 23 minutes): its iteration-21 checkpoint (128 x 4,
  1.55M parameters) into a 192 x 6 student (3.59M) on 79,427 rows of iterations 16-20, 2,500 steps of
  256 rows; held out 8,192 rows of iteration 21: KL 0.595 -> 0.067 (p90 1.06 -> 0.17), top-1 0.69 ->
  0.92, the P(win) difference 0.30 -> 0.053 (squared 0.122 -> 0.0075), the student's own policy
  cross-entropy against the shards 0.946 -> 0.414 (the teacher's 0.352), value cross-entropy 0.70 ->
  0.40 (0.377). The training KL ended at 0.026: on 79k rows the student fits its rows better than it
  generalises, so the real distillation takes the whole window and more steps. Head to head on CPU
  (48 games, 32 simulations, both sides at the line's 'cq' rule): the student **62.5% +- 7.0** against
  its teacher (parity within 1.8 se: not weaker), the fresh 192 x 6 network 0 / 24.

### 13.5 Tests and checks

`tests/test_distill.py` (8): `outputs` are the evaluator's answers (several batches, a ragged last
one); a network's agreement with itself is exact and another network's is not; a token and an MLP
student converge to an MLP teacher on held-out shards; `teacher_w` 0 is the run's supervised loss
and 1 the teacher's terms (the dense teacher policy sums to 1 over the legal actions only); the tool
writes a checkpoint the guard accepts, with its record (the held-out curve at steps 0 / 100 / 200)
and its log, resumes a partial checkpoint at its step, and `--compare` reproduces the last held-out
KL; shards of another encoding are refused; a toy run's checkpoint 2 distilled into a bigger student
on the run's own shards and forked with `--net` (the student bit for bit as checkpoint 2, no
optimizer state, the size in config.json, K's Elo pairs gone from the fork and kept in the source),
then one more iteration trains the student; a network of another game is refused by `--net`. MTG
Zero: `test_a_bigger_network_distilled_on_delver_shards_reads_layout_v5`.

**Checks** (c539d8f + 907bad6 on main edf4b89, and the GPU device fix after them; desktop 2026-10-29, CPU only - MTG Zero's round-2
A/B held the GPU and the Mac). Tier "opt-in core feature with byte-identical defaults": nothing on the
run path changed (`run.py`, `net.py`, `data.py`, `infer.py`, `cotrain.py`, the Rust crates untouched;
the diff is `distill.py`, `tools/distill.py`, `bench.py`, `fork_run.py`'s `--net`, the tests, these
docs).

* `cargo test --release`: zero_search 50 passed, zero_webnet 3; `pytest tests`: **208 passed** and the
  2 CUDA tests skipped (12 min 53 s; the 8 above among them). A fix found in review, after these: `outputs` sends
  the rows to the model's device and the tool moves a fresh (CPU-built) student to `--device` before
  its first held-out evaluation (on CUDA the step-0 evaluation would have met a CPU network);
  `tests/test_distill.py` again 8 passed. Run with `CUDA_VISIBLE_DEVICES=` (empty)
  those 2 fail - torch then sees the driver but no device - and they fail alike on edf4b89; with
  `CUDA_VISIBLE_DEVICES=-1` they skip.
* **Byte identity** against edf4b89 (its worktree zero-core-wt/ckpt-guard): `fork_run --source --iter 1
  --set reuse=8` (no `--net`) of a toy MLP run and of a toy token run: the two continuations identical
  file for file; `tools/check_checkpoint_compat.py --package zero` with a toy MLP checkpoint, a toy
  token checkpoint, `random` and `uniform`: identical. MTG Zero: its delver2 iteration-21 checkpoint
  on its e5483aa build against this core, identical (its ENGINE.md 62.7).
* **The other games** build against it (build only, at their HEADs with this core in their
  `third_party/zero-core`: the PyO3 crate, every test compiled, the package imported with this core and
  its `zero.distill`): Glass Road Zero c1f56dc 67 s, Caverna Zero a6fad73 107 s, Kingdom Builder Zero
  4202a4a 171 s.
* MTG Zero (ENGINE.md 62.7): `--test az` 27 passed, pytest 49 (the new distillation test among them),
  the CPU distillation and head-to-heads of 13.4.

## 14. The self-play league (`zero/league.py`; MTG Zero's Delver mirror, 2026-10-29)

### 14.1 Why

MTG Zero's Legacy Izzet Delver mirror (its main line `runs/delver2`) settled at about +1 Elo per
iteration (its ENGINE.md 63). In a mirror, pure self-play trains a network against its own current
habits only: what beats them is learned, the habits it exploited fade, and nothing keeps the policy
from narrowing or cycling. Fictitious self-play and AlphaStar's league answer it by playing part of
the games against past versions. `selfplay_league` is that in its simplest form: an opt-in share of
the self-play games puts a past checkpoint of the run in one seat.

### 14.2 What it does

* **Settings** (1.2): `selfplay_league` (share, 0 = off), `selfplay_league_pool` (raw iteration
  offsets back from the network that self-plays - iteration i self-plays with checkpoint i-1, so the
  default '2,4,8,16,32' offers i-3, i-5, i-9, i-17, i-33, those that exist -, or 'all': every
  earlier checkpoint), `selfplay_league_slots` (past checkpoints per iteration, default 1; each its
  own evaluator key 'L0', 'L1', ...), `selfplay_league_pick` ('uniform' | 'pfsp'). Validated at
  start (`league.check_settings`); a co-training run (`resolver`, or `resolver_cfg` holding the
  setting) is refused. The default offsets skip offset 1: checkpoint i-2 is one training iteration
  away from the network that self-plays, so its games would be nearly mirror games recorded on one
  side.
* **Which games** (`league.kinds`): game g is a league game when floor((g + 1) f) > floor(g f), so a
  driver's started games hold the share exactly (floor(n f) of n) and evenly spread; league game m
  plays slot m mod k, the past network in seat (m div k) mod players (seats alternate, so neither
  side gets the play / draw advantage), the other seats the current network. A classic driver gets
  one table per iteration; a rolling or `queue_only` driver's cyclic table (`games[gid % len]`, the
  Rust driver's) is one cycle of `league.cycle` entries (the share's denominator x slots x seats x
  player counts), built once - its keys stay, the weights behind them follow each iteration's picks.
  The shared queue (5.3) counts league games as any game: the quota stays exact; the share of the
  *accepted* games is the started share up to which games finish first (25.0 +- 2% in the toy runs).
* **The past network's seat** is the run's self-play seat on its key with `record` false (the Rust
  seat key, docs/search.md 2.1): it searches with the self-play settings (root noise, temperature
  moves, the playout cap, `selfplay_select`), so a past network plays as it did in its own
  self-play, and it records nothing.
* **Training targets: only the current network's decisions.** Its records are the league game's
  samples: `pi` its search's target, `q` its own root value, `z` the game's result from its seat (the
  driver's records are per decider). The past network's moves are not the current policy's: its
  visit / completed-Q targets are an older network's improvement of an older prior (training on them
  pulls the policy back toward the habits the league is meant to move away from), and its root
  values estimate the game from an older value head. AlphaStar trains its main agents only on their
  own decisions in league games, for the same reason. The cost: a league game gives about half the
  samples of a mirror game, so an iteration's samples fall by about f / 2 (f = 0.25: ~12.5%) and
  training's steps with them (`new samples x reuse / batch`).
* **The picks** (`League.plan(it)`, made once per iteration and cached; the same plan whether made
  for the remote's early phase while checkpoint it - 1 is evaluated or for this machine's phase, and
  after a restart): the candidates are the pool's checkpoints that exist (newest first); with no
  more candidates than slots every candidate plays (cycled over the slots), else `slots` are drawn
  without replacement, seeded by (run seed, it). 'uniform' weighs them equally; 'pfsp' (prioritized
  fictitious self-play, AlphaStar's f_hard) weighs candidate j by (1 - x_j)^2 (at least 1e-3), x_j =
  the current networks' score against j from the metrics rows of iterations <= it - 2 (each row
  weighted 0.7^age, a prior of 4 games at 0.5): checkpoints the network still loses to are played
  more. **No candidate yet** (a new run's first iterations, a pool reaching further back than the
  run): the slots hold checkpoint it - 1, i.e. mirror games recorded on one side (`self` in the
  metrics).
* **Evaluators**: each slot's checkpoint is loaded through `zero.net.load` with the run's game - the
  encoding guard (10.4) refuses a past network of another encoding - into the slot's existing
  evaluator when the network's config is the same (`set_weights`), else a new one.
* **Metrics** (`selfplay.league` in metrics.jsonl): `share` (league games / the iteration's games),
  `games`, `score` and `win` (+ `se`): the current network's score against the league (draws 0.5;
  more than two players: 1 - the past seat's result), `pool` (per checkpoint: games, score, win),
  `slots` ({key: checkpoint}), `candidates`, `pick` (and pfsp's `weights`), `local` / `remote` (the
  league games of each machine), `self`, `infer_s` (the slots' inference time). A finished game is
  attributed to the checkpoint in its slot when it finished (a rolling driver's game carried over a
  phase boundary searched with both). The log: `iter 4 league: 149 league games (24.8%), slots
  iter_0001 of 2 candidates (uniform); the current network scores 0.846 +- 0.030 against them
  (iter_0001 149 games 0.846)`.
* **Remote self-play** (5.3): the phase line names the slots' checkpoints (`league`: {key:
  `models/iter_j.pt`}), shipped into the mirror with checkpoint i-1 at every phase start; a worker's
  clean-up (checkpoints older than i-3) keeps the ones its phase names. The worker builds the run's
  league table from the mirror's config, points its slot evaluators at the named checkpoints
  (guarded), annotates its finished games with their slot's checkpoint, tallies the games the run
  accepts and reports the tallies with its phase's `end` (`selfplay.league.remote`; a lingering
  worker's games, held for the next phase, keep the checkpoint they played). A worker whose hello
  does not announce the league (an older checkout: it would play mirror games and record both seats
  of the league games) is killed and retried every 10 minutes (`worker k REFUSED: its checkout has no
  self-play league`); this machine plays every game meanwhile. MTG Zero's remote `mac` runs exactly
  the run's commit anyway (its RunCommitRemote).
* **fork_run** with `--set selfplay_league=...`: the continuation links every checkpoint its league
  may play (back to the pool's furthest offset, `league.fork_needs`; 'all': every one), not only the
  Elo matches'. A fork made without the setting and started with it plays only the candidates it has.
* **Off** (`selfplay_league` 0, the default): no League object, the seat dicts, tables, evaluators,
  phase lines and metrics of before (14.5).

### 14.3 Costs

* **Inference calls**: each slot is one more network batch per group and dispatch (its leaves are a
  share of the group's: f / 2 of the leaves split over the slots). On a CPU the extra call costs its
  overhead plus its leaves; on CUDA mostly the call. Measured on the desktop's CPU (other runs on the
  machine; rolling driver, 2 groups, 64 / 16 simulations):

  | | leaves/s | calls per network (current / each slot) | batch (current / slot) |
  |---|---:|---:|---:|
  | toy 32 x 1, 256 in flight, 4 threads: off | 179,949 / 208,544 | 7,040 / - | 128 / - |
  | toy, 0.25, one slot | 135,755 / 134,194 (-30%) | 7,051 / 7,051 | 112 / 16 |
  | toy, 0.25, two slots | 123,872 / 118,763 (-38%) | 7,048 / 7,048 | 112 / 8 |
  | toy, 0.5, one slot | 148,288 / 156,655 (-21%) | 7,075 / 7,075 | 95 / 33 |
  | MTG Delver token network 128 x 4, 96 in flight, 4 search + 2 torch threads: off | 1,824 / 2,034 / 1,757 / 1,330 (mean 1,736) | - | 48 / - |
  | MTG, 0.25, one slot | 1,642 / 1,344 / 1,770 / 1,296 (mean 1,513: -13%) | equal | 42 / 6 |
  | MTG, 0.25, two slots | 1,395 / 1,233 (mean 1,314: -24%) | equal | 42 / 3 |

  (Each phase 60-90 s after a warm-up; the toy's networks two checkpoints of a toy run, MTG Zero's
  delver2 iterations 23 and 11; the desktop shared with MTG Zero's A/B arms, so the MTG phases vary by
  +-20% between neighbours.) The toy's tiny network is the worst case: a call's overhead is most of
  its cost (a 16-leaf call ~0.8 ms against ~1.0 ms for 112 leaves), so a slot costs nearly a call.
  MTG's: a ~6-leaf call ~12 ms against ~46 ms for the current network's ~42 leaves - +26% inference
  per dispatch with one slot, +38% with two - and inference is not all of the loop: about -13% of the
  leaves per second with one slot. More slots cost more calls, a larger share does not (the same
  number of calls, larger league batches). CUDA (a batch-bound consumer): one more graph replay per
  dispatch, not measured here (CPU only).

* **Memory**: one more network per slot in every self-playing process (this machine's driver and
  each remote worker), inference only (no optimizer state; on CUDA its graph buffers too): MTG Zero's
  Delver token network is 6.2 MB on disk. A rolling driver keeps its slots' evaluators.
* **Transfer** (remote self-play): `slots` more checkpoints per iteration into the mirror - Delver:
  6.2 MB against the 6.2 MB checkpoint and ~50 MB of compressed shards coming back per iteration.
* **Samples**: ~f / 2 fewer per iteration (14.2), the same games.

### 14.4 Tests (`tests/test_league.py`, Rust `an_unrecorded_seat_plays_the_same_games_and_records_nothing`)

The kinds (exact, spread, slots and seats in turn) and the rolling cycle; the seat tables; the
settings' checks (shares, pools, slots, picks, co-training refused, the command line); off by
default (no League, the seat dict without `record`); through the bindings, half the games against a
past checkpoint: the league games' samples are the current network's decisions only (decider its
seat, z its result), and with the league seat recorded instead the same 24 games move for move whose
current-seat samples are exactly these; the attribution; a past checkpoint of another encoding
refused; the plans (the pool's candidates newest first, 'all', the `self` case, deterministic across
League objects, uniform about even over 390 iterations); pfsp's weights from the metrics (decay, the
prior, rows <= it - 2) and its picks (P 0.02 / 0.57 / 0.40 over 300 seeds); toy runs with the league
(classic: exactly 12 of 48 games every iteration; rolling with two slots), their metrics, candidates
and shard samples; a mirror run with no league in its metrics or log; the shared queue with the
remote 'local' and linger: the quota exact, league games on both machines, the share 18-32%, the
mirror after the last phase = checkpoints >= 1 and the phase's slot (iter_0000 only because it is
named); a worker without the league refused while this machine plays every game; fork_run linking
the league's checkpoints and the fork playing its league. In Rust: the same games with the league
seat recorded or not, its records exactly the m seat's, z from the m seat, and an unrecorded seat
alone recording nothing.

### 14.5 Checks

Tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only
(`CUDA_VISIBLE_DEVICES=`, below-normal priority), MTG Zero's A/B round live on the GPU and the Mac.
First on main edf4b89 (code 8295152), then again after the rebase onto main 437cef0 (feat/bignet's
distillation: nothing on the self-play path), code 5099f00 + these docs:

* `cargo test --release`: zero_search **51 passed** (50 + the unrecorded seat), zero_webnet 3 (8295152).
  On 5099f00 50 passed and `queued_setups_are_shared_between_the_groups` failed once: a test whose
  first assertion depends on which group's first search finishes first (its own comment), failing 1 of
  60 runs on this branch and 4 of 60 on 437cef0 interleaved with it (2 of 30 and 0 of 30 before), with
  pytest and the identity runs on the machine; nothing in it records.
* `pytest tests`: 8295152 **213 passed, 2 failed** (13 min 47 s), 5099f00 **220 passed, 3 failed** (21
  min; main's distillation tests among them). Two failures are CUDA-only tests
  (`test_net_data.py::test_graph_runner_matches_eager_and_follows_new_weights`,
  `test_nonfinite.py::test_normal_run_has_no_debug_dir[cuda]`) that fail the same way on edf4b89 with
  `CUDA_VISIBLE_DEVICES=` (they do not skip without a visible device); the third,
  `test_remoteplay.py::test_cotrain_backoff_gives_matches_back` (a co-training worker backing off in
  time to give its matches back), failed under the load of 180 concurrent Rust test runs and passed 3
  of 3 alone (the league is refused in co-training; matches mode never builds it).
  `tests/test_league.py` 13 passed on both.
* **Byte identity** against edf4b89 and again against 437cef0 (each its own worktree and target dir),
  the settings left out and given at their defaults: toy-smoke runs - MLP vs `ext:toyheur` and tokens
  (classic, 3 iterations), `cq` targets + `eval_select cq` guard 2 (rolling, 3), and the Delver main
  line's shape (rolling, `cq` targets at scale 0.3, `eval_select cq` guard 4 at 0.3, raw Elo offsets,
  no random games, value_q 0.6, window_cache, train_prefetch; 4 iterations): every shard,
  `elo/results.json`, `ratings.json` (`updated` aside), `models/index.json` byte for byte, every
  checkpoint and the optimizer state tensor for tensor (the files differ in torch's archive root
  only), the metrics equal without their timings, config.json = the base's + the four keys. The
  rolling runs needed care: a rolling phase's end depends on timing with two batches in flight
  (docs/search.md 2.6) - edf4b89 itself reproduced its own reference in 18 of 19 runs of the Delver
  shape (one differed from iteration 2 on), 8295152 in 10 of 12 (6 of 6 interleaved with the base's);
  with `ZERO_INFLIGHT=1` (deterministic) both rolling configurations are identical in 2 runs each of
  the base, unset and the defaults (11 and 13 files, the metrics), and against 437cef0 the rolling
  ones were compared that way.
* `tools/check_checkpoint_compat.py --package zero` (random, uniform, a toy MLP and a toy token
  checkpoint): identical, against edf4b89 and against 437cef0.
* The other games build against it (build only, their HEADs with this core in `third_party/zero-core`;
  the PyO3 crate, every test compiled, the package imported): Glass Road Zero c1f56dc 45 / 36 s,
  Caverna Zero a6fad73 70 / 59 s, Kingdom Builder Zero 4202a4a 106 / 94 s (before / after the rebase).
* MTG Zero (branch `league`, its ENGINE.md 63): `cargo test --release -p engine --test az` 27 passed,
  `pytest tests` 48 passed; delver-smoke with the league on, classic and with the remote 'local'
  (rolling, linger).

### 14.6 An A/B of the league

Fork the main line at checkpoint K with the league on (`tools/fork_run.py --source <main> --dest
<fork> --iter K --set selfplay_league=0.25`, so the fork has the league's checkpoints), run the fork
and the control K -> K + N with the same build and flags (the fork adding `--selfplay-league 0.25`),
read it with an h2h of the two K + N checkpoints at the same move rule (as the policy-target and
selection A/Bs, 9 and 12.5). Watch `selfplay.league.win` (the network's score against its own past:
a mirror run's policy that drifts in a cycle shows it falling against some checkpoints) and
`train.entropy`. MTG Zero's flags: its ENGINE.md 63.
