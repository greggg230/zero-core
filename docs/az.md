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
| `net.py` | `Net`: residual MLP with LayerNorm (arch 'mlp', the default); `GridNet` (arch 'grid'): residual 3x3 conv tower over the game's board planes (`GamePackage.grids`) with globals + pooled-board channel biases, one policy logit per cell from the tower; `TokenNet` (arch 'tokens', section 10): a transformer over the game's objects read through a layout descriptor (`GamePackage.tokens`), pointer logits for the objects' actions, `forward_static` (static shapes for the CUDA graphs, 10.6); all: policy logits over NUM_ACTIONS (masked to the legal actions by the caller), value = P(win) per relative seat (softmax over the seats present), auxiliary margin head; atomic `save`, `load` (by the checkpoint's arch), `widen`, `make_net` |
| `infer.py` | `NetEvaluator` (CPU / CUDA; gather + segment softmax on the device; `set_weights` in place; `workers`: eager batches on threads; `graphs`: the conv network's `GraphRunner`, the token network's opt-in `TokenGraphRunner` - env ZERO_TOKEN_GRAPHS=1, section 10.6 -, `graph_calls`), `UniformEvaluator`, `drive()` (external batches answered on threads; several network batches in flight) |
| `data.py` | Sample shards (`ShardWriter`, atomic npz, optionally compressed - remote self-play's; the features dense or, with `shard_sparse`, as their non-zeros - section 32; non-finite ones quarantined as `*.bad.npz`), the RAM-bounded training `Window` (each shard subsampled as it is loaded; checks every shard; row provenance), `TargetStats` |
| `finite.py` | The non-finite guard (section 4.1): vectorised NaN / inf checks, `NonFinite` |
| `games.py` | `engine()` (the package's compiled module), registered games, seat specs, external players (`EXTERNALS`, `register_external`; a package registers its own, e.g. `ext:jsbot`), the `Driver` wrapper |
| `toybots.py` | The toy heuristic as an external player (`ext:toyheur`) |
| `elo.py` | Bradley-Terry Elo over all results, the run's yardstick = 1000; one pool per evaluation move rule (`pool_dir`, `eval_tag`, `active_pool`), the Elo opponents (`opponents`: evaluations back, or raw `elo_offsets`; section 5.4); fixed anchors (`parse_anchors`, `fit_fixed`: the fit with the anchors held at their ratings, `anchored.json`; section 20.1) |
| `runctl.py` | PAUSE / STOP files, exit status 75, run / supervisor state files, `PauseWatch` |
| `dashboard.py` | Flask dashboard (title, port and runs directory from the game package) |
| `bench.py` | Throughput measurements (`infer`: batch sizes, concurrent workers, a checkpoint, the operators and multiply-adds of a forward pass; `train`: seconds per training step; section 10.5; `--graphs`: 10.6) |
| `distill.py` | Distillation (section 13): a new network of another size or architecture trained on a run's shards to compute what a teacher computes (`new_student`, `outputs`, `distill`, `agreement`); the teacher's auxiliary head and value-only rows (13.6) |
| `h2h.py` | Checkpoint vs checkpoint, seats alternating (the readout of an A/B); each side's move rule its run's `eval_select` or `--select` / `--a-select` / `--b-select` (section 5.4); `--move-log`; paired games, per-side simulations, the sequential tests (`--paired`, `--a-sims` / `--b-sims`, `--sprt`, `--target-se`: `pair_stats`, `sprt_llr`, `sequential`; section 20.3); each side's CUDA graphs (`--graphs`, `--a-graphs` / `--b-graphs`, section 10.6) |
| `heldout.py` | Held-out losses (section 20.2): a frozen set of sample rows (`make`), a checkpoint's value cross-entropy / accuracy / Brier against z and policy cross-entropy / top-1 against the recorded targets, decision and value-only rows apart (`HeldOut`, `losses`); `python -m <pkg>.az.heldout make | eval` |
| `remote.py` | Remote evaluation (section 5.2): `RemoteEval` (the run's thread: queue, ship to the remote's mirror, run, results to `evals.jsonl`, Elo back), `RemoteNetEval` (a co-training run's `eval_yard_net`, 11.7), the remote side (`python -m <pkg>.az.remote eval MIRROR ITER`; `evalnet MIRROR ITER`; `selfplay MIRROR K`: remoteplay's worker), `Transport` (ssh / scp, or `local`) |
| `remoteplay.py` | Remote self-play (sections 5.3, 11.6): `RemotePlay` (the run's sessions with persistent workers on the remote, checkpoints out, compressed shards back into `data/iter_i/` with the non-finite guard on arrival; the shared queue's caps per consumer; single-game: the shared quota, games accepted by the run; co-training: a share of the resolver's queued episodes, rows back by tag, requeue, give-back on back-off), `Worker` (the remote side: a rolling `queue_only` driver across iterations, held games, back-off while an evaluation holds its lock, busy markers, linger) |
| `workqueue.py` | The shared work queue (section 5.3): `Shares` (each consumer's in-flight cap from its measured throughput, head start and the iteration's units; a batch-bound consumer's its slots within `batch_inflight` iterations), `split_games` (a driver's sample dict cut into its games), the staleness histogram |
| `targets.py` | Policy-target sharpness and quality: visits vs completed-Q targets (section 9) |
| `league.py` | The self-play league (section 14): the league games' kinds and seat tables (classic, rolling), the iteration's picks of past checkpoints (`League.plan`: uniform or pfsp), the slots' evaluators (guarded), the games' attribution and `selfplay.league` |
| `forks.py` | Branched self-play games and archive starts (section 24): the settings, the archive file (`zero-archive/1`: its entries' patches over the run's spec), the self-play driver's `branch` / `starts`, the remote hello's fingerprint, the tallies and `selfplay.forks` |
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
| `cq_prior_w` | 1.0 | beta in (0, 1]: the prior's exponent in self-play's recorded 'cq' targets, softmax(beta ln P0 + sigma completedQ) (docs/search.md 3.13): < 1 turns the generations' cumulative sum of advantages into a bounded fixed point exp(sigma Q / (1 - beta)), so an option no search visits recovers and is re-measured; the self-play seat only, never its moves (PUCT, temperature, `selfplay_select`, the Gumbel root) or an evaluation; needs `policy_target` 'cq' (26) |
| `hidden`, `blocks` | 256, 4 | network size (only when the run's first network is created) |
| `arch`, `ch` | 'mlp', 64 | 'grid': a `GridNet` of `ch` channels x `blocks` over the package's board grid (Kingdom Builder: 16 planes x 20 x 20; bfloat16 autocast on CUDA for inference and training; not in zero.webexport); 'tokens': a `TokenNet` of width `hidden`, `blocks` layers over the package's object layout (section 10; not in zero.webexport) |
| `heads`, `ff` | 4, 0 | arch 'tokens': attention heads, feed-forward width (0 = 2 x hidden) |
| `batch`, `reuse`, `lr`, `wd`, `min_steps` | 512, 4, 1e-3, 1e-4, 50 | steps = new samples x reuse / batch (at least min_steps); AdamW, grad-norm clip 5 |
| `lr_halve_every`, `lr_min`, `lr_warmup_steps` | 30, 1e-4, 100 | step decay; linear warmup while the optimizer state is fresh |
| `window`, `window_grow`, `min_window_samples` | 8, 1, 20000 | newest iterations trained on (1 + iter // 2 with window_grow), more until min samples |
| `keep_data`, `ram_budget_gb`, `shard_rows` | 12, 4, 65536 | data retention (iterations), RAM for the window (uniform subsample beyond), samples per shard |
| `window_cache`, `window_sparse` | 0, 0 | 1 = the window's shards stay in RAM between iterations (zero/data.py WindowCache); 1 = the window keeps each row's features as their non-zeros - the same window and batches in a fraction of the RAM for a sparse encoding, and with `window_cache` every shard cached whole also over budget, so an iteration loads only its new shards (10.8). The budget still counts dense rows |
| `shard_sparse` | 0 | 1 = self-play writes each shard's features as their non-zeros (`x_ptr` / `x_idx` / `x_val`, meta `x_format` 'sparse'; every other array as before; this machine's shards and a remote's): the dense rows bit for bit in ~1/25 of the disk on MTG Zero's Delver line, written ~4-5x faster with the float32 rows never buffered. Every reader takes both formats, so a window may mix them; an older checkout cannot read a sparse shard (`data.convert_shard` makes it dense again) (32) |
| `train_static` | 0 | 1 = a token network trains through its static forward (`TokenNet.forward_static`, the inference graphs': each batch one chunk of its longest row's tokens, no host sync) with a fused AdamW on CUDA, and a sparse window's batches travel as their non-zeros (dense on the device): the same batches and losses, the updates up to float32 rounding; ~2.8x a step on MTG Zero's Delver network (10.9). Another network is refused |
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
| `selfplay_leaf_value_view`, `eval_leaf_value_view` | mover, mover | whose view a search leaf's value comes from (docs/search.md 3.6, section 16): 'mover' = the network on the leaf's decider's view (today's searches bit for bit); 'root' = a leaf of another seat than the root decider also sends a value-only row of the root decider's view of the same sampled state (same batch, flagged `ROW_VALUE`), whose value and margin it takes (the priors stay the decider's). The self-play seat (remote self-play too: its checkout must have this core) / every evaluation seat; the evaluation's names its Elo pool (`-rv`), self-play's does not |
| `eval_infoset_merge` | 0 | 1 = every evaluation seat's trees merge their nodes by the root decider's observation (single-observer information-set MCTS inside the tree: the other seat's nodes subset-armed over the union of its sampled worlds' options, the root decider's next decision one node whatever the other seat was dealt; docs/search.md 3.16). Needs `eval_leaf_value_view` 'root'; names the Elo pool (`-ism`); h2h `--a-infoset-merge / --b-infoset-merge`. Self-play: `selfplay_infoset_merge`. 0 = off (nothing changes) (30) |
| `selfplay_infoset_merge` | 0 | 1 = self-play's seat merges its trees' nodes by the root decider's observation, every search of it (full, fast, minor; docs/search.md 3.16). Needs `selfplay_leaf_value_view` 'root'; the records, root values, value-only samples and resignation read the root as before; the leaf's viewer row is taken from the merged tree's encoding; remote self-play refuses a worker whose checkout does not merge; metrics `selfplay.merge`. 0 = off (nothing changes) (30.1) |
| `selfplay_belief_tilt`, `eval_belief_tilt` | 0.0, 0.0 | the root's belief (docs/search.md 3.8, section 18): a full search of the seat asks the network's auxiliary head once for the root's prediction of the hidden cards (one `ROW_BELIEF` row of the root's features) and deals every simulation's world by it (the game's `determinize_with`, weights clip((belief / uniform)^tilt, 1/belief_clip, belief_clip)). The self-play seat (a league opponent's too; remote self-play: a checkout without it is refused at its hello, 18.5) / every evaluation seat (its Elo pool's name gets `-b<tilt>`). Needs a checkpoint with the auxiliary head: one without is refused (`net.HeadMissing`, exit 78; 18.3). 0 = off |
| `belief_clip`, `belief_mix`, `belief_fast` | 20.0, 0.0, 0 | the belief's clip R (every weight in [1/R, R]), the share of a belief search's simulations still dealt uniformly ([0, 1)), and whether fast and minor searches take it too (section 18) |
| `selfplay_view_samples`, `view_sample_target` | 0.0, next_q | value-only samples (section 16): this share of self-play's recorded decisions also records the real state from each other recording seat's view - its perspective, no options (no policy loss), z its result - with q = that seat's next search's root value ('next_q'; z if none) or z ('z'). 0 = off |
| `selfplay_root_search`, `eval_root_search` | puct, puct | the root search (docs/search.md 3.7, section 19): 'puct' = today's searches bit for bit; 'gumbel' = Gumbel AlphaZero's root - Gumbel noise per root option, sequential halving over the `gumbel_m` options of highest noise + log-prior, the completed-Q sigma - whose move replaces root Dirichlet noise, temperature sampling and the move rule (`*_select`; PUCT stays below the root). Every search of the seat is one (full, fast, minor). The self-play seat (remote self-play and the league seat too: their checkouts must have this core) needs `policy_target` 'cq' (an error otherwise: sequential halving's counts are no policy), its sigma the target's (`cq_c_visit`, `cq_c_scale`); every evaluation seat: sigma the evaluation rule's transform (`eval_select_c_*`, else `cq_c_*`), and its root search names its Elo pool (`gumbel-m16`, `gumbel-m16-v50-s0.3`, `-g1` with noise, `-rv`), self-play's does not. Exclusive with the same side's `*_root_min_visits` (an error). `zero.h2h` / `zero.replay` follow the run's |
| `gumbel_m`, `selfplay_gumbel_scale`, `eval_gumbel_scale` | 16, 1.0, 0.0 | the Gumbel search's considered options (the paper's and mctx's 16: every option of a smaller root) and its noise scale: self-play's first `temp_moves` searched decisions of a game are searched with `selfplay_gumbel_scale` (1 = the move a sample of the improved policy, the paper's), later ones without - the role temperature had; the evaluation's on every decision (0 = deterministic, the paper's evaluation) |
| `eval_spec` | '' | the spec of every match that records nothing (evaluation, Elo, reference games); '' = the game package's evaluation spec of the self-play spec (`GamePackage.eval_specs`; MTG: the fixed evaluation deck set), else the spec itself |
| `eval_remote` | '' | the run's evaluations (yardstick, random, Elo) on another machine while it trains: a name in the game package's `remotes` (`local` = this machine, the tests; '' = here; section 5.2). A co-training run: its resolver's `resolver_cfg['eval_remote']` (section 11.5); its own is refused |
| `selfplay_remote` | '' | self-play on another machine too, with the same checkpoint: both machines pull the iteration's `games` (the total) from one shared queue (`zero/workqueue.py`), its compressed shards join `data/iter_i/` (a name in the game package's `remotes`; '' = off; section 5.3). A co-training run: its resolver's `resolver_cfg['selfplay_remote']`, a share of the queued episodes (11.6); its own is refused |
| `selfplay_remote_procs`, `selfplay_remote_wait` | 0, 30 | worker processes on the remote (0 = the remote's `selfplay.procs`, else 1); seconds the end of a phase waits for the remote's last shards (later ones are dropped) |
| `selfplay_remote_inflight` | 1.0 | each machine's in-flight cap, in iterations of its own measured throughput (its share of the iteration's games or matches x this; 5.3) |
| `selfplay_remote_batch`, `selfplay_remote_batch_inflight` | auto, 2.0 | the batch-bound consumers (`auto`: those whose inference is CUDA / MPS - their network batches grow with their games in flight; `local`, `remote`, `all`, `none`): such a consumer's cap is its slots while its games take at most `selfplay_remote_batch_inflight` iterations (its share x this; 1 = as the others; 5.3) |
| `selfplay_remote_linger` | 0 | 1 = the workers play on with their checkpoint between phases (while the run trains); those games count for the next iteration, one checkpoint stale (5.3) |
| `selfplay_local_linger` | 0 | 1 = this machine's self-play plays on too while the run trains (a thread; CUDA batches on their own stream): its finished games count for the next iteration, one checkpoint stale, as the remote's linger (5.3, 10.10). Needs `selfplay_remote`; single-game runs without the league |
| `shard_writer_thread`, `shard_writer_queue_mb` | 0, 4096 | 1 = self-play's training records are converted and written into shards on a writer thread of their own (`zero/shardthread.py`): the thread that drives the search only takes them (`SearchDriver.take_records`: moved, not copied), splits them per game and queues them, at most `shard_writer_queue_mb` MB (it waits while the queue is full, counted). The same shards byte for byte; the caller's timing changes which games finish when (a rolling or shared quota's). This machine's self-play (plain, rolling, the shared queue), the local linger and the remote's workers (their config's); metrics `selfplay.shard_writer`, `linger.samples_s`, `selfplay.remote.shard_writer`. Single-game runs (co-training refused). 0 = taken, converted and written between batches, as before (31) |
| `selfplay_remote_giveback`, `selfplay_remote_overdue`, `selfplay_remote_timeout` | 30, 3, 3600 | co-training (11.6): a worker backing off this many seconds gives its outstanding matches back; a match outstanding on a worker longer than this many of its expected latencies (at least 120 s; 0 = the timeout only) is requeued, and at the latest after `selfplay_remote_timeout` s of the worker's running time |
| `infer_device`, `train_device`, `seed` | cpu, cpu, 0 | inference / training device (`cuda` works; the GPU is shared), run seed |
| `infer_workers` | 1 | network batches evaluated concurrently on this many Python threads, with as many in flight (give the driver as many `groups`); CPU inference of a small network (section 10) |
| `sparse` | 0 | 1 = the run's drivers (self-play and matches) send sparse network batches (each row's non-zero features; docs/search.md 2.2): the same rows, games and samples; a token network's CUDA graphs scatter them on the device, other evaluators densify them (10.7) |
| `eval_cache` | 0 | N > 0 = every network seat (self-play and evaluation) keeps its game's network answers, at most N per game, by a hash of each row's features and options, and answers a repeated row from them instead of sending it (docs/search.md 3.12; the same answers up to float rounding at another batch shape) (25) |
| `resolver` | '' | co-training (section 11): the resolver game, whose episodes decide this game's deferred terminals ('' = a single-game run: this loop, line for line) |
| `resolver_cfg` | {} | the resolver's settings (a JSON object of these same settings: spec, players, network, search, training, evaluation; `preset` names one), saved nested in config.json; resolved into `<run>/resolver/config.json` |
| `resolver_init` | '' | the checkpoint the resolver's network starts from ('' = fresh); a single-game run's checkpoint works (widened if the game grew) |
| `resolver_conc`, `pending_max` | 0, 0 | resolver episodes in flight in self-play (0 = its `conc`); the backpressure: resolver episodes queued or playing at which the primary's self-play waits (0 = 2 x in flight) |
| `draft_eval_net_every` | 4 | the primary's evaluation also plays its episodes with the resolver's network on evaluated iterations that are multiples of it (`eval_yard_net`; 0 = never); the rest of its evaluation plays them with the resolver's yardstick |
| `draft_eval_net_remote` | '' | those events (`eval_yard_net`) on another machine while the run trains: a name in the game package's `remotes` (`local` = this machine, the tests; '' = here; section 11.7); their results in the run's `evals.jsonl` |
| `selfplay_league` | 0.0 | the self-play league (section 14): this share of the self-play games (exactly, over each driver's started games) puts a past checkpoint of the run in one seat; only the current network's decisions become samples. 0 = off (every game the current network's mirror, as before). Single-game runs (co-training refuses it) |
| `selfplay_league_pool` | '2,4,8,16,32' | the past checkpoints it may play: raw iteration offsets back from the network that self-plays (iteration i self-plays with i-1: offset 2 = checkpoint i-3), those that exist; 'all' = every earlier checkpoint (fictitious self-play) |
| `selfplay_league_slots`, `selfplay_league_pick` | 1, uniform | past checkpoints per iteration (one evaluator key each: 'L0', 'L1', ...), picked from the pool every iteration: 'uniform', or 'pfsp' (weight (1 - x)^2, x = the current networks' recent score against that checkpoint, from the metrics) |
| `selfplay_branch`, `selfplay_branch_moves`, `selfplay_branch_temp`, `selfplay_branch_differ` | 0.0, 3, 2.0, 0 | branched self-play games (section 24, docs/search.md 2.11): this share of the games each spawn one branch game from a copy of their state at a uniformly drawn decision of theirs - 1..`moves` deviation moves sampled from the network's priors at temperature `temp` (never recorded; `differ` 1: the first never the game's own move), then normal self-play, its rows' z its own outcome; a branch counts as one of the iteration's games. 0 = off (nothing changes); needs rolling self-play (`rolling` 1 or `selfplay_remote`); single-game runs without the league |
| `selfplay_archive`, `selfplay_archive_p` | '', 0.1 | archive starts (section 24): this share of the games start from a position of the archive file (format 'zero-archive/1', built for the run's spec by the game's own tool: MTG Zero's `tools/delver_archive.py`) instead of a new deal. '' = off |
| `selfplay_resign_t`, `selfplay_resign_k`, `selfplay_resign_noresign` | 0.0, 5, 0.1 | self-play resignation (section 27, docs/search.md 3.14): t > 0 = a self-play game ends when `k` consecutive full searches (the recorded ones; never fast or minor) found one seat's root win estimate >= 1 - t, the same seat each time, scored as that seat's win, every record kept (z the resignation's); this share of the games (a hash of seed and game) never resigns and reports where it would have (`selfplay.resign`: the live wrong-resignation rate; 1 = a dry run). Evaluations never resign. 0 = off (nothing changes); two-player single-game runs without the league |
| `aux_<name>_w` (one per target the game package declares: the toy's `aux_next_card_w`, MTG Zero's `aux_belief_w`) | 0.0 | an auxiliary training target (section 15): > 0 = self-play records the game's per-decision target with every sample (shard key `aux`), the network gets an auxiliary head (a new run's first network; added to a checkpoint without one at its next training, everything else exactly as it was) and training adds this weight x the target's loss. 0 = off: nothing recorded, built or trained, the run of before bit for bit (section 15.6) |

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
| `metrics.jsonl` | one line per iteration: `selfplay` (games, samples, decisions / rounds per game, avg score, leaves/s, games/h, inference time, average batch; `rolling` stats; `target`: kind, mean options, normalised entropy `h_norm` = H(pi) / ln(options) and `max_share` of the recorded policy targets; `remote`: the remote's share with `selfplay_remote`, and `queue`: the shared queue's quota, caps, rates and staleness histograms, section 5.3; `league`: with `selfplay_league`, section 14; `view`: with a root-view setting on, the value-only rows sent and samples recorded, section 16; `belief`: with `selfplay_belief_tilt` on, the root belief rows sent and their share of the searched decisions, section 18), `train` (policy / value / margin loss, entropy, value accuracy, steps, window samples / MB / load time, lr, value_q, samples/s; with an auxiliary target on, `aux_<name>` its loss, `aux_<name>_hit` a multinomial target's top-1 hit rate, and `aux`: rows with targets and their share, per multinomial target its mean total and non-empty share, the weights, `head_added` the iteration that added the head, section 15); with value-only samples in the window (section 16): `value_view` / `value_decision` (the value loss on each kind of row) and `window_view_samples`, `eval_yard` / `eval_random` (win, se, margin, games; `select`: the move rule unless 'visits'; with `eval_games` 0 `eval_yard` has 0 games: the iteration is still evaluated, section 20.1), `elo` (`select`: its pool unless 'visits'), `elo_anchored` (with `elo_anchors`: the rating of the fit with the anchors fixed, section 20.1), `heldout` (with `heldout_shards`: the held-out losses, section 20.2), `phases` (seconds per phase), `seconds` |
| `models/iter_NNNN.pt` | checkpoints (atomic); `best.pt` = highest rated (in the current rule's pool, among the checkpoints evaluated in it: section 5.4); `index.json` = every checkpoint's Elo in that pool (`select` unless 'visits'; `"evaluated": false` for one rated only through Elo matches: raw offsets, or a rule's pool begun mid-run); `opt_latest.pt` = optimizer state (with the iteration it belongs to); `iter_NNNN.json` = stats of an iteration whose evaluation has not finished |
| `data/iter_NNNN/*.npz` | sample shards (docs: `zero/data.py`; with an auxiliary target on, the key `aux`: section 15; with `shard_sparse` the features as `x_ptr` / `x_idx` / `x_val` instead of `x`: section 32): this machine's `c00_SSS` / `r00_SSS`, a remote's `m<KK>-<sid>_SSS` (compressed; section 5.3); the newest `keep_data` iterations are kept |
| `debug/` | only after a non-finite failure (section 4.1): `iter_NNNN_<task>_SSS.bad.npz` (a quarantined shard; a remote's: `iter_NNNN_m<KK>-<sid>_SSS.bad.npz`, `remote_*.bad.npz`, section 5.3) and `nonfinite_iter_NNNN_step_SSSS.pt` (a failing training batch); never rotated or cleared by the run |
| `elo/results.json`, `elo/ratings.json` | pair results (`{"a|b": [score of a, games]}`, anchor, `logged`: the checkpoints whose evaluations are in) and the current fit: the Elo pool of the move rule 'visits' (every run's pool from before `eval_select`) |
| `elo/<rule>/results.json`, `elo/<rule>/ratings.json` | the pool of another evaluation move rule ('cq', 'cq-n2', ...; section 5.4): the same files, plus `select` (the rule) and `since` (the first iteration it rated) |
| `elo/[<rule>/]anchored.json` | with `elo_anchors` (section 20.1): the anchors (label, file, sha256, fixed rating and how) and every player's rating in the fit with them fixed; the pool's `results.json` then also holds `anchors` and the anchors' games (`anchor:<label>` players) |
| `evals.jsonl` | with `eval_remote` (section 5.2): one line per remote evaluation `{iter, eval_yard, eval_random, elo, time, host, seconds}` (+ `elo_anchored` with `elo_anchors`), merged into the iterations' rows by the dashboard; a co-training run's resolver: `resolver/evals.jsonl` (11.5); a co-training run's primary with `draft_eval_net_remote`: `{iter, eval_yard_net, time, host, seconds, remote_s}` (11.7) |
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
actions only; a value-only sample, section 16, has none and is left out of its mean) + value cross-entropy against `(1 - q) z + q q_root` (softmax over the seats
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
* **Local linger** (`selfplay_local_linger` 1, opt-in; 10.10): the run's own driver does the same while
  the run trains - `Run.linger_start` after the phase (not after the last iteration, nor with STOP or
  PAUSE there), on a thread whose network batches run on a CUDA stream of their own beside the
  training's default one: setups queued up to the local cap as in the phase, the games in flight
  finishing with checkpoint i-1 (the same network id: nothing restarts), the finished ones held with
  their samples for iteration i+1's quota (which takes held games first: lag 1), never more held than
  an iteration's quota. `Run.linger_stop` after the training (and on any exception from it) ends the
  phase, answers its last batches and joins the thread; `linger` in the iteration's metrics row =
  `{games, seconds, leaves, infer_s, held}` (the lingering's inference is not counted in the next
  phase's `infer_s`), and a log line `iter i local linger: ...`. The rates and caps of the shared
  queue measure the phases only.
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
  (no autocast); CUDA graphs opt-in (`TokenGraphRunner`, ZERO_TOKEN_GRAPHS=1: section 10.6).

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
  ENGINE.md 62. Section 10.6 takes the operators and syncs out with CUDA graphs.

### 10.6 CUDA graphs of the token network (`infer.TokenGraphRunner`; MTG Zero's Delver line, 2026-10-31)

**Why.** The desktop's share of MTG Zero's Delver self-play and, above all, every head-to-head and
Elo evaluation it runs on the RTX 5090 are eager CUDA inference of the token network, bound by its
operators and host syncs (10.5): 445 kernels and 13 syncs a forward of 64 rows of the main line's
128 x 4 network, 585 and 15 at 192 rows; on the GPU shared with a training run every sync waits
behind the other process's work. A 2,400-game paired head-to-head at 32 simulations took ~80 minutes.
The conv network has had CUDA graphs since Kingdom Builder (`GraphRunner`, ZERO_CUDA_GRAPHS); the
token network's eager forward packs each batch by its data (`nonzero` per segment, the rows sorted
by token count, chunks as long as their longest row), which a graph cannot capture.

**Opt-in, off by default** (every default: the evaluator of before, byte for byte):

* `NetEvaluator(model, 'cuda', graphs=True)` (False: eager; None, the default: the env below);
* env `ZERO_TOKEN_GRAPHS=1`: every token network's evaluator of the process that has no `graphs`
  given - the run loop's self-play and evaluations (`Run.evaluator`, co-training's, the league's),
  `zero.h2h`, `zero.bench`, `zero.replay`, `zero.targets`, on CUDA (distillation and the held-out losses
  call the network directly: eager). A run gets it from its environment (`ZERO_TOKEN_GRAPHS=1 python
  tools/launch_hidden.py ... tools/supervise.py ...`: the supervisor passes its environment to the run;
  a remote's workers do not get it: the Mac runs on its CPU anyway). No run
  setting: `Run.evaluator` would pass one (`infer_graphs` in DEFAULTS), left for when run.py is not
  being changed by another branch;
* `ZERO_CUDA_GRAPHS=0` (the conv network's switch) turns the env off too: no CUDA graphs anywhere,
  said on stderr. An explicit `graphs=True` wins over both envs, as it always did for the conv
  network;
* `zero.h2h --graphs 0 | 1` (both sides), `--a-graphs` / `--b-graphs` (one side, over `--graphs`): given,
  the output names `graphs_a` / `graphs_b`, each side's share of batches through the graphs;
  `zero.bench infer | selfplay --graphs 0 | 1` (the output's `graphs`);
* **never a silent fallback**: asked for by the env on a CPU evaluator, or turned off by
  ZERO_CUDA_GRAPHS=0, is said on stderr (`zero.infer: ... eager inference`); `graphs=True` on a CPU
  evaluator, or for an MLP, is a ValueError; when on, each evaluator says `zero.infer: token network
  CUDA graphs on` (a run's console). Every batch goes through the graphs - root belief rows and
  value-only rows included -, and `NetEvaluator.graph_calls` counts them (`graph_calls == calls`).
* **float32** (`use_amp` stays the conv network's). `TokenGraphRunner(model, amp=True)` (bfloat16
  autocast) exists for the measurement below only; no switch reaches it.

**How** (`zero/infer.py`, `zero/net.py`):

* **The host packs the batch** (numpy, `token_presence`, `token_index`): each slot's present flag read
  from the features, each row's token count and its tokens' positions in the full token axis (the
  global token's 0 first, layout order, then padding). The batch is padded to B rows (a multiple of
  ROW_STEP = 64) x T tokens (the longest row's, a multiple of T_STEP = 16); one graph per (B, T),
  captured on first use (two warm-up runs on a side stream, then the capture; ~10 shapes in a
  200-game head-to-head).
* **`TokenNet.forward_static(x, n, full, ntok, aux)`**, what the graphs capture: every slot embedded,
  present or not (the per-segment input maps on all slots, every identity decoded by one gather of
  the digit floats per radix: 156 kernels whatever the batch, no sync), the tokens gathered by `full`,
  the whole batch one chunk of T tokens (padding masked out of the attention as in a chunk), then the
  trunk and heads of `forward` (`_trunk`, `_heads`: the chunk body and the heads moved out of `forward`
  unchanged - the same operations in the same order: `forward`'s outputs and gradients are bit for bit
  those of c15e3d0 on the Delver rows, CPU and CUDA). With the auxiliary head, the head's predictions
  of every row (`aux_predict` in the graph); after the replay the batch's root belief rows are
  gathered on the device and downloaded (no host sync). Value-only rows are rows with no options.
* **One upload, one download.** Pinned host buffers, a set per batch in flight (any number in flight:
  a set is reused after its `result`); the host copies the features outside the segments and only the
  present slots' floats (an absent slot's are never read: its token is not gathered; for the Delver
  rows 44% fewer bytes); the options' softmax on the CPU (numpy), as GraphRunner. One set of device
  inputs and one memory pool for every graph (they run in order on one stream). The graphs read the
  model's own parameters: `set_weights` (load_state_dict, in place) is seen by the next replay.

**Numerics** (graphs against eager on the main line's iteration 328 and 8,192 held-out rows,
`runs/heldout/r3-2b_iter0197fresh_8192.npz`; batches of 1-1,024 rows, a third value-only, 5% root
belief rows, 3 batches a size):

| output | max abs. difference | mean abs. difference |
|---|---:|---:|
| priors | 2.9e-6 | 8.7e-8 |
| value (P(win)) | 9.5e-7 | 1.3e-8 |
| margin | 2.4e-6 | 1.1e-7 |
| belief (the head's distribution) | 5.1e-7 | 3.9e-10 |

float32 rounding: the matrix products run at other shapes (one chunk of B x T rows against eager's
chunks of 64 rows each as long as its own longest row, the input maps over all slots) and cuBLAS picks
its kernels by shape; the same at every batch size (no growth with B or T), replays bit for bit
deterministic, a row alone and inside a batch within 1.3e-7. A fresh 192 x 6 network (6 heads, feed-
forward 384, the belief head: MTG Zero's bigger candidate, its ENGINE.md 62.3) on the same rows: priors
4.8e-7, value 1.8e-7, margin 5.4e-7. Not a concern for the search (a prior or value 1e-6 off moves a
visit only at a near tie, at 32-64 simulations); but the games are then not exactly the same games,
which is why the head-to-head below measures the equivalence.

**Throughput** (the desktop's RTX 5090 shared with the main line's training and self-play and another
session's head-to-head, 67-78% busy before any of these; eager and graphs interleaved so the
contention hits both alike). Inference alone (real rows as the search sends them - a third
value-only, 3% belief rows -, drive's pipeline of two batches in flight, 3 s a cell, the median of 3
rounds), leaves per second (ms a batch):

| rows a batch | eager | graphs | graphs / eager | graphs, bfloat16 |
|---:|---:|---:|---:|---:|
| 64 | 3,034 (21.1) | 21,790 (2.9) | 7.2x | 22,488 |
| 96 | 3,510 (27.4) | 26,008 (3.7) | 7.4x | 23,837 |
| 144 | 4,291 (33.6) | 27,054 (5.3) | 6.3x | 26,816 |
| 192 | 5,305 (36.2) | 25,680 (7.5) | 4.8x | 26,605 |
| 256 | 5,698 (44.9) | 27,071 (9.5) | 4.8x | 25,364 |
| 384 | 6,897 (55.7) | 27,580 (13.9) | 4.0x | 27,761 |
| 512 | 6,871 (74.5) | 26,690 (19.2) | 3.9x | 26,491 |

A head-to-head of iteration 328 against itself (`zero.h2h`'s seats: 200 games, 32 simulations,
`cq-n4-v50-s0.3-rv-b1m0.25` both sides as the A/B driver plays them, 8 search threads, 200 games in
flight: batches of 36 rows on average, 64.9k of them): **eager 1,753.7 s, graphs 261.3 s - 6.7x**
(843 against 5,681 leaves/s; an earlier version of the graphs, 443 kernels a replay and the whole
rows copied, 284.1 s); the same seed's games, A 0.505 in both. With the graphs the evaluators took
230 s of the 261 (3.5 ms a batch), the search threads were busy 11% of their time. The A/B driver's
shape, 2,400 paired games (1,024 in flight) with the graphs on both sides: **1,670.6 s (27.8 min)**
against ~80 minutes for an eager 2,400-game paired head-to-head that day (as reported; not
interleaved, and this one ran with this branch's gate compiling and testing on the CPU beside it); its
1,200 pairs all split (two identical, deterministic sides).

Self-play at the main line's settings (its config.json's self-play seat - 'cq' targets, root-view
leaves with value-only samples 0.5, the belief tilt 1 / mix 0.25, 64 / 16 simulations, minor 4 -, 192
games in flight, 8 threads, 2 groups, recording; batches of ~145 rows as its desktop share's; 60 s of
warm-up, then 120 s measured; eager, graphs, eager, graphs): **eager 2,195 and 2,004 leaves/s, graphs
6,457 and 8,028 - 3.4x** (2.9x and 4.0x a pair; the run's own desktop self-play measured 4.3k eager
leaves/s that day, its phases not overlapping another head-to-head). With the graphs the evaluator holds 72-76% of the time: the search
and the host's per-row work are now a quarter of it.

**Bfloat16** (the same graphs under autocast, measured only): the same leaves/s (the table's last
column: the graphs are not bound by arithmetic) and 10^4 x float32's differences (priors up to 0.036,
value up to 0.014, margin 0.021, belief 0.007). Not worth it; no switch reaches it.

**Where the graphs' time goes** (one batch, 64 / 192 / 512 rows of T 64; CUDA events and the host's
clock): the host's packing 0.2 / 0.3 / 1.9 ms, its copy of the rows into the pinned buffer 1.9 / 3.8 /
12.9 ms, the upload 0.6 / 1.7 / 4.4 ms, the replay 1.2 / 2.4 / 4.3 ms (156 kernels; 1.7 ms at 64
rows with the first version's 443), the download 0.04-0.13 ms. Pipelined (two batches in flight), a
batch costs about the larger of the host's part and the device's: **from ~192 rows the host's copy of
the 37,620-float (150 KB) rows bounds it, at ~27k leaves/s.** Copying only the features outside the
segments and the present slots (44% of the bytes) saves 10-26% of the copy against whole rows (fresh
batches: 0.89 against 1.20 ms at 36 rows, 3.07 / 3.38 at 144, 9.86 / 11.58 at 512; the fancy indexing
costs the rest), and the bags (19,456 of the floats, 0.4% nonzero) are copied whole. The rows are 0.7%
nonzero (~260 floats): the next step is a sparse feed from the driver (10.1: not built), which would
take the copy and most of the upload out; pinning the driver's own array in place
(`cudaHostRegister`, ~1 ms) was no faster (its upload 9-15 ms and the unregister 3-7 ms at 512 rows).

**Memory**: the graphs share one pool; a 200-game head-to-head's two evaluators peaked at 414 MiB of
the device (10 shapes each), one evaluator with shapes up to 1,024 x 80 at 914 MiB; pinned host
buffers of at least 256 rows x 150 KB (38 MB) a batch in flight and an evaluator.

**Equivalence** (the same checkpoint, A through the graphs, B eager: `zero.h2h A A --paired --games 400
--a-graphs 1 --b-graphs 0`, seed 11, the A/B driver's search otherwise: 32 simulations, chance_k 2,
`cq-n4-v50-s0.3-rv-b1m0.25` both sides; `graphs_a` 1.0, `graphs_b` 0.0): **A 0.4975 +- 0.0043** (the pair
scores' se; the games' binomial 0.025), Elo -1.7 +- 3.0, margin +0.03, none unfinished. 197 of the 200
deals were split - each seat won its deal under both inferences, as a mirror's paired games do when
the two players play the same moves -, 1 won twice by A and 2 by B (variance ratio 0.03): float32
rounding changed the outcome of 1.5% of the deals and favours neither side. (Unpaired, the same seed's
200-game matches with the graphs on both sides and eager on both sides scored A 0.505 alike.)

**Limits.** One graph per (B, T) shape: a batch of an unseen shape pays a capture (26-75 ms from 64 x
32 to 1,024 x 80; the process's first 0.7 s, the libraries' start; ~10 shapes in a match, 13 in the
numerics' 1-1,024 rows). Batches of fewer rows than ROW_STEP run as 64 (at 36 rows on average a
200-game match's padding is ~45% of the rows; the GPU does not care, the copy is of the real rows
only). `GraphRunner` (the conv network's) is unchanged: its two fixed host
slots would be overwritten by a third batch in flight (ZERO_INFLIGHT > 2), which the token runner's
per-launch buffers avoid. Eager `NetEvaluator.launch` counts its batch's seconds twice (in `evaluate`
and again in `launch`): an eager run's `selfplay.infer_s` is about twice the evaluator's time (the
main line: 408 s of a 240-s phase); not changed here (a metric of every eager run).

Tests (`tests/test_token_graphs.py`): `forward_static` against `forward` (both layouts, the auxiliary
head, longer rows, padding positions holding other slots, padding rows), the host's packing; off by
default and the switches (env, ZERO_CUDA_GRAPHS=0, a CPU evaluator said / refused, an MLP, explicit
False); zero.h2h's flags (given: `graphs_a` / `graphs_b`; not given: the match of before; a CPU side
refused); CUDA: the graphs against eager on batches of 1-130 rows with value-only and root belief
rows (2e-5), replays deterministic, new weights in place, an absent slot's NaN floats never read
(before and after other batches), four launches in flight, toy games with the belief and root-view
leaves through the graphs (every batch, every answer valid, drive's pipeline too).

Checks (tier "opt-in core feature with byte-identical defaults"; the desktop, 2026-10-31, on 3d5875e,
the code; this commit adds documentation only):

* **`TokenNet.forward` bit for bit c15e3d0's** (the refactor into `_global` / `_trunk` / `_heads`): its
  outputs (with the auxiliary head on a subset of rows, and without) at batches of 1-1,500 Delver
  rows and a training step's gradients (256 rows), CPU and CUDA, compared bit for bit (the CPU gradient
  of `pos` differs run to run in both checkouts: threaded accumulation).
* `pytest tests/test_token_graphs.py tests/test_net_data.py tests/test_belief.py tests/test_aux.py
  tests/test_measure.py` with CUDA: 79 passed, none skipped. The whole suite on the CPU (in MTG Zero's
  gate): 305 passed, 8 skipped (CUDA), 2 failed - `test_remoteplay.py`'s
  `test_cotrain_the_local_driver_is_never_starved` and `test_a_dying_worker_gives_its_accepted_games_back`,
  shares of games between the local driver and a worker; run side by side on c15e3d0 and here under
  the same load the first fails in both the same way (the local driver's share 23 / 158 and 21 / 158,
  the bar 15%) and the second passes in both: load, not this change.
* Toy-smoke token runs byte for byte against c15e3d0 (its own worktree and target, `ZERO_INFLIGHT=1`,
  the CPU, 4 iterations, seed 3): classic with `ext:toyheur`, and the Delver line's shape (rolling, 'cq'
  0.3, the cq2 rule, root-view leaves with value-only samples 0.5, the auxiliary head `aux_next_card_w`
  0.25 with the belief tilt 1 / mix 0.25, value_q 0.6, `elo_offsets` 1,2, no random games, margin_w
  0.2), each without the env and with `ZERO_TOKEN_GRAPHS=1` on the CPU (said: eager): every shard,
  checkpoint (zip members), optimizer state, Elo file and the metrics without timings identical. The
  Delver-shaped run on CUDA with the env: 4 iterations, every evaluator through the graphs (rolling
  `set_weights` included).
* `check_checkpoint_compat --package mtgz` against MTG Zero's main 1605436 (zero-core c15e3d0): a random
  network (the gate) and the main line's iteration 328 (the Delver spec, 8 games a target, 8 against
  random): identical (records visits 73693aedf6cb0b7f, cq 6dd36ebeb19d9fbf, results 38b68141fc4cff90).
* MTG Zero (branch `token-graphs`, its ENGINE.md 78): `tools/gate.py optin-core` twice on 23d4fd3 (this
  core at 3d5875e): core tests 92 passed, checkpoint_compat byte identity identical against main 1605436,
  `--test az` 39 passed, `--test hidden` 16 passed, Glass Road c1f56dc / Caverna a6fad73 / Kingdom Builder
  4202a4a build against this core - and the last step **FAILED both times** on its smoke_all's core
  pytest: the first run on the two `test_remoteplay.py` tests above (305 passed), the second on
  `test_cotrain_the_local_driver_is_never_starved` alone (306 passed), which fails the same way on
  c15e3d0 under the same load (above). Every other step passed both times.

### 10.7 Sparse batches in a run (run setting `sparse`; MTG Zero's Delver line, 2026-11-01)

**Why.** With the token network's CUDA graphs (10.6) a batch of the Delver line's self-play (~144 rows,
37,620 floats a row, ~0.7% of them non-zero) spent most of its time moving zeros: the driver assembling
the dense rows in fresh memory, the graphs' pinned copy (3.8 ms at 192 rows) and the upload (1.7 ms),
against a 2.4 ms replay. Its self-play process used under half a CPU core and waited on inference 90%
of the time. The sparse batches built for the lookahead (docs/search.md 2.2, `DriverCfg::sparse`,
`TokenGraphRunner.launch_sparse`) take both copies out.

**What.** Run setting `sparse` (default 0: the drivers of before, byte for byte): `Run.driver` passes
`sparse=True` to every driver the run builds - self-play (rolling or not), the yardstick / random / Elo
matches, co-training's and a remote's (from the mirror's config; a remote's `overrides` can turn it off -
MTG Zero's Mac does, its CPU evaluators would only densify). The rows, games and samples are the dense
ones' (the graphs' answers bit for bit at the same batch shape, 10.6 / search.md 2.2); only the batches'
transport changes.

**Equivalence.** `tests/test_loop.py::test_sparse_setting_plays_the_same_selfplay`: an iteration's
self-play of the toy run, its shards array for array the dense one's. On MTG Zero's Delver line
(iteration 489, the run's self-play seat, CUDA graphs, 24 games in flight, 8 threads, the first 24
finished games' samples and results hashed): dense d31a5acb09179104 twice, sparse d31a5acb09179104.

**Throughput** (the line's local self-play exactly as `Run.selfplay` drives it - its rolling driver,
seat and evaluator with ZERO_TOKEN_GRAPHS=1, samples taken every 32 batches -, from iteration 489; 90 s
of warm-up, then 150 s measured; one configuration a process, configurations interleaved, two or four
rounds; the RTX 5090 shared with another session's live run, so the absolute numbers are a loaded GPU's;
8 search threads):

| batches | groups | in flight | leaves/s (range) | x dense | rows a batch | CPU cores |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2 | 192 | 13,390 (12,792-13,989) | 1.00 | 144 | 0.40 |
| sparse | 2 | 192 | 24,996 (24,801-25,192) | 1.87 | 144 | 0.62 |
| sparse | 4 | 192 | 20,037 (19,867-20,207) | 1.50 | 72 | 0.52 |
| sparse | 2 | 384 | 35,610 (34,723-37,636) | 2.66 | 288 | 0.65 |
| sparse | 4 | 384 | 31,536 (31,214-31,859) | 2.36 | 144 | 0.58 |
| sparse | 2 | 768 | 44,278 (43,477-45,078) | 3.31 | 575 | 0.53 |
| sparse | 2 | 1,536 | 41,694 (40,614-42,775) | 3.11 | 1,143 | 0.55 |

Dense batches grow their host copy with the batch (10.6: flat at ~27k leaves/s from ~192 rows), so more
games in flight did not pay; sparse ones do, up to ~768 in flight (batches of ~575 rows) on this GPU.
More groups only split the same games into smaller batches. The process stays under a core: the GPU (and
its other tenant) is the limit now, not the host.

### 10.8 A sparse training window (run setting `window_sparse`; MTG Zero's Delver line, 2026-11-01)

**Why.** With faster self-play (10.7) the Delver line's training became the larger part of a
generation: ~100 s, of which 40-54 s was loading the window. The window's shards (iteration 489: 73
shards, 864k rows - 7 local shards of 29.8 GB stored plain, 66 remote ones of 0.5 GB compressed, 36 GB
inflated) exceed the 34 GB budget, so the window is a subsample (438k rows) - and a subsampled window
cannot use the WindowCache (window_cache was on and reused nothing): every iteration read and inflated
every shard again. The rows are 252 non-zeros of 37,620 floats on average.

**What** (zero/data.py; opt-in, the default the window of before):

* `SparseRows`: features as CSR rows (int64 offsets, uint16 column indexes - int32 past 65,536 features -,
  float16 values). Non-zero is decided on the float16 bits, so a -0.0 is kept: `dense()` gives the dense
  rows bit for bit, and float32 rows directly (exact) for a batch.
* `Window(sparse=True)` (run setting `window_sparse`): each shard's features converted to SparseRows as
  it loads - in chunks of 1,024 rows on the window's threads (numpy releases the GIL), read in place
  (np.memmap of the npz member) when the shard is stored uncompressed, since np.load copies a member
  through a Python loop that holds the GIL (two 4 GB local shards could not load in parallel). The
  non-finite guard of `x` reads the non-zeros (a NaN / inf is never zero bits); a bad shard is described
  from its dense rows as before.
* With a WindowCache every shard is cached whole (as SparseRows), also over budget: the window draws the
  same picks as loading (the same random stream), cuts the other arrays from the cached parts
  (`_subset_part`) and reads each row's features from its cached shard by `src_row` when a batch is built.
  An iteration loads only its new shards. The cache's key includes the representation, so turning the
  setting on or off mid-run loads afresh.
* The budget (`ram_budget_gb`) still counts dense rows: the same rows as without the setting. A bigger
  budget would now cost little RAM - and change the window (more rows kept): a training change, not part
  of this setting.

**Equivalence.** `tests/test_net_data.py::test_sparse_window_is_the_same_window` (mostly-zero features
with -0.0s, auxiliary targets, plain and compressed shards, networks narrower and wider than the shards,
over budget, a cache over sliding windows, several chunks on threads): every array, the features bit for
bit and the batches. `tests/test_loop.py::test_window_cache_trains_the_same_network`: training with
window_sparse (with and without the cache and prefetch, and over budget from the cache twice) gives the
network of the dense window, weight for weight. On the Delver line's shards: the dense and sparse windows
of iteration 489 at a 40k-row budget, and the cached and uncached sparse windows at the full budget -
every array equal, 20 batches of 512 bit for bit.

**Measured** (the Delver line's windows, budget 438,636 rows, 8 threads, the desktop shared with a live
run):

| | dense, cache over budget (before) | sparse + cache |
|---|---:|---:|
| a window, cold (the process's first) | 54.1 s (iteration 489) | 29.8 s (iteration 488) |
| the next iteration's window | the same: nothing reused (the run's log: 32-61 s) | 6.6 s (iteration 489: 8 new shards, 2 of them local) |
| the process's RAM after it | 34 GB of window + the rest | 4.3 GB in all (the cache 2.8 GB of it) |
| a batch of 512 rows | - | 19-21 ms |

A warm window is its new shards (3.3 s: a 65k-row local shard read in place and converted on 8
threads 1.8 s, on one 5.0 s) and the window's assembly from the cache (1.1 s).

### 10.9 The static training step (run setting `train_static`; MTG Zero's Delver line, 2026-11-01)

**Why.** With sparse self-play batches and a sparse window (10.7, 10.8) a Delver generation was ~183 s:
self-play ~133 s and training ~49 s (542 steps of 512 rows, ~91 ms a step; the window 3 s). The step was
the eager `TokenNet.forward` (10.5): the batch packed by its data (`nonzero` per segment, its rows sorted
by token count, chunks of 64 rows each as long as its longest row: ~20 host syncs and ~1,200 kernels a
forward of 512 rows) and its backward, then a foreach AdamW. A micro-benchmark on the line's iteration
580 (512 held-out rows, the RTX 5090 shared with the live run; the run's own loss replaced by a simpler
one of the same outputs):

| the step | ms |
|---|---:|
| eager forward, foreach AdamW (the run's) | 88.5 |
| eager forward, fused AdamW | 71.1 |
| `forward_static` (10.6), fused AdamW | 29.2 |
| `forward_static`, fused AdamW, bfloat16 autocast (measured only) | 20.2 |

**What** (opt-in, `train_static` 0 the training of before line for line):

* `Run.train_model` trains a token network through `TokenNet.forward_static` - every slot embedded, each
  row's tokens gathered by `full`, the whole batch one chunk of its longest row's tokens (rounded up to
  `infer.T_STEP`), no host sync - and, on CUDA, with `torch.optim.AdamW(..., fused=True)` (the same
  update, one kernel set for all parameters; its optimizer state loads from a foreach one's and back).
  Another network (MLP, GridNet) is refused (`SystemExit`: set train_static 0).
* The batches carry each row's tokens (`Window.batch(..., tokens=(present_index, T_STEP))`: `full` [B, T]
  and `ntok` [B], `infer.token_presence` / `token_index`'s, built on the prefetch thread).
* **A sparse window's batches travel as their non-zeros** (`Window.batch(..., coo=True)`,
  `Window.features_coo`; `data.dense_coo`): ~130k (row, column, float16 value) triples a batch of 512
  Delver rows (~2.5 MB) instead of 77 MB of float32 rows, scattered into zeroed rows on the device - the
  dense rows bit for bit (float16 values are exact in float32; a -0.0 is kept) - and the tokens read from
  the non-zeros (a flag present when its value > 0.5, as on the dense rows). With the dense rows the
  feed cost the step ~15 ms of densifying and ~10 ms of tokens on the prefetch thread and a ~10 ms
  pageable upload on the training thread; as non-zeros a batch takes ~10 ms on the prefetch thread and
  ~1 ms to upload (fed: 6.2-6.9 ms a batch with prefetch 2, the step no longer waits for it).

**Equivalence.** `forward_static`'s outputs are `forward`'s up to float32 rounding (10.6: other matrix
shapes, cuBLAS picks its kernels by shape). One step's loss and gradients on the toy token run: equal to
1e-7 of each gradient's scale (`tests/test_loop.py::test_train_static_trains_the_same_network_up_to_rounding`).
Over a training AdamW's per-parameter normalisation amplifies that rounding where a gradient is tiny: the
toy's trained networks end within ~2% of the largest weight change (the test's bar 5%), CPU (foreach) and
CUDA (fused) alike; with a sparse window (cached or not, prefetch on) the same network as the dense
window's, weight for weight on the CPU. On the Delver line (one generation's training, iteration 585 from
checkpoint 584 and its optimizer state, the same window, 542 steps, in one process): the training's
logged losses (means of the last 200 steps)

| | policy | value | margin | entropy | belief |
|---|---:|---:|---:|---:|---:|
| eager | 0.22216 | 0.37837 | 0.17771 | 0.22322 | 2.50654 |
| train_static | 0.22216 | 0.37837 | 0.17770 | 0.22322 | 2.50654 |

and the two networks 0.016 of the update's norm apart (update norms 0.70996 / 0.70983, distance 0.01140).

**Measured** (the Delver line's training on the desktop, the GPU shared with the live run's self-play
throughout, so absolute numbers vary with its phase; same process, same window):

| the training (542 steps) | ms a step |
|---|---:|
| eager (the run's; its log 91-96 ms on an unshared GPU) | 113.9 |
| train_static, the dense rows fed | 60.6 |
| train_static, the non-zeros fed | 40.4 (32.8 in a quieter minute) |

A step is now bound by the device's float32 arithmetic (~23 ms of kernels, ~1,200 of them; the matrix
products are float32 SIMT kernels, no tensor cores): the same batches with their tokens cut to 32 (a
timing only) take 17.8 ms, so about half of it is padding - a row has 15 tokens on average (90% at most
20) and every row is run as long as the batch's longest (64-80). Not done here: bucketing a batch's rows
by token count (two static chunks), bfloat16 (19.4 ms; a numerics change: its own decision), the step as
one CUDA graph, the non-finite guard's per-step sync made asynchronous.

Tests: `tests/test_loop.py::test_train_static_trains_the_same_network_up_to_rounding` (above; a step's
gradients, the trained network CPU and CUDA, sparse windows cached or not with prefetch, the batches'
tokens against `infer`'s, the COO batch against the dense one bit for bit, an MLP refused) and
`test_window_cache_trains_the_same_network` (the default path, unchanged).

### 10.10 Local linger: this machine's self-play while the run trains (run setting `selfplay_local_linger`; MTG Zero's Delver line, 2026-11-01)

**Why.** With a remote (5.3), the remote's workers play on while the run trains (`selfplay_remote_linger`)
but this machine's driver stopped for the whole training: on the Delver line the desktop makes ~75% of a
generation's games (~9 a second) and idled ~49 s a generation (~19 s with `train_static`).

**What** (opt-in, `selfplay_local_linger` 0 the loop of before; 5.3's "Local linger" has the protocol):
after the phase, `Run.linger_start` resumes the rolling driver with the same network id (the searches in
flight go on with checkpoint i-1, nothing restarts) on a thread and keeps queueing setups up to the local
cap; the finished games wait, with their samples, for the next iteration's quota, which takes them first
(lag 1, as a remote's lingering games). `Run.linger_stop` after the training ends the phase, answers the
last batches and joins. Needs `selfplay_remote` (the shared queue keeps the held games and the quota) and
a single-game run without the league (refused otherwise, at the start).

**Two CUDA consequences.** The thread's network batches run on a CUDA stream of its own (`torch.cuda.Stream()`,
non-blocking with the legacy default stream the training uses): the inference kernels interleave with the
training's instead of queueing behind a whole step. And **a CUDA graph capture is thread-local now**
(`infer.CAPTURE_MODE = 'thread_local'`, both graph runners): torch's default ('global') fails any other
thread's sync or allocation while a stream captures. Measured (`capture_stress.py`, the line's network:
one thread trains with the static step - a sync and allocator churn every step - while another evaluates
52 new (rows, tokens) shapes through the graphs on its own stream): 'global' failed at once (the training
thread `cudaErrorStreamCaptureUnsupported`, the capture `cudaErrorStreamCaptureInvalidated`);
'thread_local' three times clean (245-286 training steps alongside, every answer within 9.8e-7 of the
eager forward). One thread: the same as 'global'.

**Evaluators on other threads.** `infer_workers` > 1 evaluates on worker threads whose current stream is
the default one: correct, but queued behind the training's kernels.

**On the Delver line** (a 3-iteration GPU smoke forked from r3-2b-late550 at 589: 400 games an iteration,
384 in flight, `train_static` 1, the desktop alone - MTG Zero's 'local' remote refuses an unpinned
checkout -, the GPU shared with the live run): 80 and 57 games during trainings of 31.3 s (the cold window)
and 12.9 s, at 18.4k / 31.4k leaves/s against ~38k in the phases; none after the last iteration; quotas
exact. The training beside it took 68-78 ms a step against 50 ms alone (the GPU shared three ways). The
desktop's samples one checkpoint behind: 67% and 54% (+6% two behind) at this small quota (400 games with
384 in flight: most games span phases); the live line without the setting (1,600 games, 768 in flight):
~35% one behind, from the games in flight carried over.

Tests: `tests/test_remoteplay.py::test_local_linger_plays_while_the_run_trains` (a toy run with the
remote 'local': linger games every iteration but the last, quotas exact, the held games' samples lag 1
in the next iteration, refused without a remote) and the remote linger's test (unchanged).

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
Zero's 192 x 6: an estimated 40-55 minutes on the desktop GPU for its main line's window, its ENGINE.md 62.6) and a fidelity short of
exact, which the held-out agreement and a head-to-head against the teacher measure before a run uses
the student.

### 13.3 The tool, and continuing a run from the student

```
python tools/distill.py TEACHER.pt OUT.pt --data runs/a/data [--heldout runs/a/data/iter_K/r00_000.npz] \
    --hidden 192 --blocks 6 --heads 6 --ff 384 [--arch tokens] --steps 12000 --batch 512 --lr 1e-3 \
    --warmup 500 --ram-budget-gb 24 --eval-every 1000 --save-every 2000 --keep-best --device cuda \
    [--aux-w 1 --aux-teacher-w W] [--resume | --init S.pt]
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
* **A teacher with an auxiliary head** (15: MTG Zero's belief) gets a student with the same head,
  distilled too (`--aux-w`, its term's weight, default 1; `--aux-teacher-w`, the blend of the
  teacher's predictions with the shards' own targets, default `--teacher-w`); **shards with value-only
  rows** (16) train the policy on their decision rows only, as the run does; the held-out agreement
  reports both kinds of row and the head. Both in 13.6; a teacher without a head on shards without
  value-only rows distils as before, bit for bit.
* **`fork_run --source --net`**: a continuation (2) of the run at K with the student as its
  checkpoint K (checked by the guard for the run's game, and its auxiliary head against the fork's
  settings - with a root belief on, the head for the game's targets: `check_heads`, 13.6 - before
  anything is copied), **without the source's optimizer state**
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

`tests/test_distill.py` (8; 7 more for 13.6): `outputs` are the evaluator's answers (several batches, a ragged last
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

### 13.6 Auxiliary heads and value-only rows (MTG Zero's main line `r3-2b-beliefsearch`, 2026-10-29)

**Why.** The tool predates two things MTG Zero's main line now has (its ENGINE.md 62.9): an
**auxiliary head** (15: the belief over the opponent's hand, `aux_belief_w` 0.25) that its search
reads (18: `selfplay_belief_tilt` / `eval_belief_tilt` 1, which refuse a checkpoint without the head,
exit 78), and **value-only rows** (16: `selfplay_view_samples` 0.5, a third of the window's rows). A
student distilled before had no head - the main line refuses it - and its policy terms were means over
every row of a batch, the value-only rows' zeros included (a policy weight two thirds of the value's).

**What it does** (`zero/distill.py`, `tools/distill.py`, `tools/fork_run.py`):

* **The student gets the teacher's head** (`new_student(..., aux='teacher')`, the default; `aux=None`:
  none): the same targets, so a run trains it and a search reads it as it did the teacher's; built
  after every other parameter (15.2), so the rest is drawn as for a student without one (tested).
  The tool checks the teacher's head and the student's against the game's targets (names and widths,
  `net.check_belief_head`) before it trains and before it writes; `--init` from a student without the
  head gives it a fresh one (`net.add_aux`), one with another head is refused.
* **The teacher's answers** include the head's predictions (`outputs`: 'aux' [N, W] by
  `net.aux_predict` - a multinomial's distribution, a poisson's rates, an mse's values - equal to
  `NetEvaluator.aux`'s answers, tested; the other answers equal the network's without its head) and the
  window carries the shards' own targets (`Window(aux_width=W)`); `Taught` batches add `taux`.
* **The loss** adds per target `aux_w` x ((1 - wa) L(the shards' target) + wa L(the teacher's
  prediction)), L the target's own loss (`auxtarget.losses`; for a multinomial the cross-entropy to a
  distribution = the KL to it + its entropy), wa = `aux_teacher_w`. The shards' term counts the rows with
  a target (the run's rule); the teacher's term every row except those whose shard says a multinomial's
  segment is empty (an empty hand: n x p = 0 whatever p, so the teacher's p there was never trained and
  means nothing) - including the rows of shards recorded without targets, where the teacher's prediction
  is the only label. `teacher_w` 0 with `aux_w` = the run's `aux_<name>_w` is `run.train_model`'s loss
  line for line (tested). Training reports `aux_<name>_t`, `aux_<name>_d` and a multinomial's
  `aux_<name>_kl` (to the teacher). A student with a head and no teacher's predictions to learn is
  refused (a head trained on nothing would be a silent no-op).
* **The defaults, and why.** `aux_teacher_w` = `teacher_w` (1: pure distillation of every head): the
  teacher's distribution estimates the hand's composition given the public state - the mean of which
  the one recorded hand is a sample -, a warm start whose searches deal their worlds as the teacher's
  do lets the head-to-head gate compare the networks rather than two beliefs, and the continued run
  trains the head on the true hands anyway (`aux_belief_w`, with its whole window). `aux_w` 1: every
  output's term is then a KL to the teacher (the policy's over the legal actions, the value's, the
  head's), equally weighted; AdamW scales the head's own parameters alike whatever the weight, so the
  weight sets the trunk's balance only. Measured on the main line (the CPU evidence below, 2,000 steps
  into 64 x 2, held-out rows of another iteration):

  | `aux_w` / `aux_teacher_w` | policy KL (decisions) | P(win) sq. error, decision / view rows | belief KL to the teacher | belief CE vs the true hands (the teacher 2.649) |
  |---|---:|---|---:|---:|
  | **1 / 1 (the defaults)** | **0.099** | **0.0106 / 0.0097** | **0.026** | 2.663 |
  | 0.25 / 1 (the run's weight) | 0.098 | 0.0117 / 0.0113 | 0.036 | 2.672 |
  | 1 / 0.5 | 0.104 | 0.0115 / 0.0107 | 0.054 | 2.631 |
  | 1 / 0 (the true hands only) | 0.116 | 0.0131 / 0.0121 | 0.252 | 2.735 |

  The run's weight 0.25 keeps the policy no closer (0.098 vs 0.099) and the value and belief further
  off; the true hands cost policy and value fidelity (their one-sample noise reaches the trunk) and on
  73k rows overfit (their CE rose after step 1,000); the 0.5 blend's head reads the true hands slightly
  better than the teacher's (2.631) at twice the KL - a choice for a student meant to improve on its
  teacher's belief, not for a warm start.
* **Value-only rows** (`losses(view=True)`): the policy terms and the teacher's entropy are means over
  each batch's decision rows (rows with options), the value, margin and auxiliary terms over every row -
  `run.train_model`'s mask; `distill` decides from the window (`np.diff(W.opt_off) == 0`, `view_rows`),
  as the run does. Their value and margin distillation is the teacher's answers on the viewer's row.
* **Agreement**: with value-only rows in the held-out window also 'decision' and 'view' - each kind's
  rows, P(win) squared and absolute difference, margin squared difference, each network's value
  cross-entropy and accuracy against z -; with a head 'aux' per target - a multinomial's KL(teacher ||
  student) over the teacher's rows (mean, p90, per kind of row), top-1 agreement, and each network's
  cross-entropy and `hit` against the shards' targets (the run's `train.aux_<name>` / `_hit`); another
  loss: the squared difference and each one's loss. The keys of before are unchanged (the policy over
  the decisions with 2+ options, P(win) over every row). `--compare` reports the head when both networks
  have the same one. `--keep-best` still keeps the lowest held-out policy KL.
* **RAM**: the window's budget counts 6 W + 1 bytes more a row with a head - the targets and their mask
  as the run's window holds them, the teacher's predictions in float32 (MTG's 1,024: +6.1 KB on a
  75.5 KB row, 8%).
* **`fork_run --net`** checks the network's head against the fork's settings (source config + `--set`)
  before anything is copied (`check_heads`): with a root belief on, the head for the game's targets
  (`net.HeadMissing`, the run's own refusal at its start); a head for other targets than the run trains,
  refused (its first training would refuse it); no head where the run only trains targets, accepted
  (the run adds one, 15.4). The encoding guard's refusal now also comes before the copy.
* **The start log** gains `train_view_rows`, `heldout_view_rows` (with value-only rows), `aux`,
  `train_aux_rows`, `heldout_aux_rows`, `aux_w`, `aux_teacher_w` (with a head); the record's settings
  gain `aux_w` and `aux_teacher_w`.

**Off: the distillation of before.** A teacher without a head on shards without value-only rows: the
forward pass without `aux`, the policy means `.mean()` as before, no new batch keys, terms or log keys.
Checked byte for byte (the record's `settings` aside, which gains the two flags): the tool on toy shards
(a token student with `--keep-best`, an MLP student at `teacher_w` 0.5, `--compare`, a toy run
distilled and continued with `fork_run --net` one iteration) on main 2f0a2e8 (its own worktree,
`E:/zero-scratch/distill2/base-2f0a2e8`) and on this branch: 16 checkpoints (configs and tensors), 9 shards and 4 logs
identical (the teacher's file sha256 aside: torch writes a new serialization id with every save); and
MTG Zero's delver2 iteration 71 (no head, no value-only rows) into a 32 x 1 token and a 32 x 1 MLP
student (a 0.3 GB budget: 3,971 of 8,202 rows), `--compare`: identical (its ENGINE.md 62.9).

**Tests** (`tests/test_distill.py`, 7 more, 15 in all): the toy's aux shards (targets, empty targets,
value-only rows); `outputs` with a head = `NetEvaluator.aux`, the rest = the network without it; the
student gets the head, built last; value-only rows: the policy terms over the decisions (the old ones
the diluted mean), the other terms unchanged; `teacher_w` 0 = `run.train_model`'s loss with the head and
value-only rows, `teacher_w` 1 the teacher's terms on the rows of `teacher_aux_rows`, the blend, the KL;
a token student converges to an MLP teacher's head and to both kinds of row, a head without the
teacher's predictions or the window's targets refused; the tool on a toy run with the target, root-view
leaves, value-only samples and the belief in self-play: the student has the head and passes
`check_belief_head`, its record and log, `fork_run --net` accepts it, the fork self-plays with the
belief and trains it (head kept), `--init` from a student without the head adds it, a network without
the head refused for the belief run (nothing copied) and accepted by a fork without it.

**Measured on MTG Zero** (its ENGINE.md 62.9, CPU): the main line's checkpoint 91 (128 x 4 with the
belief head) into a 64 x 2 student, 2,000 steps of 256 rows on 73,454 rows of iterations 89-90 (24,411
of them value-only), held out 4,096 rows of iteration 91 (1,312 value-only): policy KL 0.571 -> 0.099,
P(win) squared error 0.124 -> 0.0106 (decision rows) and 0.135 -> 0.0097 (value-only rows), the belief's
KL 4.29 -> 0.026 (its CE against the true hands 6.93 -> 2.663, the teacher's 2.649); `fork_run --net`
into a copy of the main line, two CPU iterations with the run's settings (belief tilt 1, mix 0.25,
root-view leaves, value-only samples 0.5, belief weight 0.25): self-play with the belief, the head
trained; the 192 x 6 student builds with the head (3,766,493 parameters) and passes the guard and the
belief's check.

**Checks** (tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal; MTG Zero's round-3 Gumbel test live on the GPU and the Mac):
nothing on the run path changed (`run.py`, `net.py`, `data.py`, `infer.py`, the Rust crates untouched;
the diff: `distill.py`, `tools/distill.py`, `tools/fork_run.py`, the tests, these docs). `cargo build
--release --workspace` (26 s, its own target dir); `pytest tests` **291 passed**, 3 skipped (the CUDA
tests; 12 min 38 s; the 15 of `test_distill.py` among them); the goldens above. MTG Zero (its ENGINE.md
62.9): `tools/gate.py python` passed (pytest 59, the new Delver distillation test among them; the
run-loop smoke).

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

## 15. Auxiliary training targets (`zero/auxtarget.py`; MTG Zero's belief head, 2026-10-29)

### 15.1 Why

A network trained on self-play learns what wins; whether its trunk also tracks what it cannot see is
another question. MTG Zero's Delver mirror (its ENGINE.md 66) wants a head that predicts the
opponent's hidden hand from public information - "they left blue open and passed: likely Daze or
Force of Will" - first as a measurement (does the trunk carry it?) and as an auxiliary loss that may
shape the trunk (KataGo's ownership and score targets are the precedent), later as the prior of the
search's determinization (stage 2, its ENGINE.md 66.8: designed, not built). The core part is
generic: a game declares per-decision labels computed from the TRUE state, the driver records them,
the network predicts them, training adds their loss. An unmerged branch had end-of-game targets
(`aux-targets`: Kingdom Builder's ownership and score); these are per decision and may be hidden
information.

### 15.2 What it does

* **The game** (docs/search.md 1.2): `SearchGame::AUX` = segments `(name, width)`,
  `aux_target(viewer, out)` = the label vector of the pending decision for its decider, from the true
  state (a label, never an input: `encode` stays blind; each game tests that worlds determinized for
  the decider encode alike while their targets differ). The toy: `next_card`, the next round's hidden
  card (one-hot over 6; none in the last round).
* **The game package** (zero/package.py `aux`): `{game: [{name, width, loss}]}` in the engine's
  segment order; loss 'multinomial' | 'poisson' | 'mse' (`net.AUX_LOSSES`). A run checks the names and
  widths against the engine's (`aux_layout` of `game_info_of`). Each target gets the run setting
  `aux_<name>_w` (DEFAULTS, default 0.0); `replay_aux` renders a checkpoint's head in replays (15.5).
* **Recording**: a weight > 0 creates every self-play driver with `aux=True` (this machine's, rolling,
  the shared queue's, remote workers': `Run.driver(record=True)`); each recorded decision gets its
  decider's target (`Record::aux`, also through parked deferred games), `take_samples()['aux']` f32
  [N, W], shards `aux` float16 [N, W] (the non-finite guard checks it). A remote shard without targets
  in a run that wants them is accepted and logged (its checkout predates the run's targets).
* **Windows**: `Window(aux_width=W)` loads `aux` and `aux_mask` per row; a shard without targets (a
  fork's older window, a run before the targets) gives zeros with mask 0, so its rows still train the
  policy / value / margin and the auxiliary losses skip them. The window cache keys its shards by the
  width too. Budget: 2 W + 1 more bytes per row (MTG Zero's 1,024-wide belief: +2.7% of a Delver row).
* **The head** (zero/net.py): config key `aux` = `[[name, width, loss], ...]`; ONE MLP from the trunk's
  pooled state - where value and margin read it: `Net`'s h, `GridNet`'s e, `TokenNet`'s z (the global
  token and the tokens' mean) - to `head` (128) to W, built after every other parameter (a network
  with the head initialises the rest exactly as one without), its last layer zero. Only
  `forward(x, n, aux=True)` returns it (a fourth output: the raw values), so self-play's and the
  evaluations' inference - eager, CUDA graphs, remote - is the network of before. `aux_predict` turns
  the raw values into predictions: 'multinomial' a softmax over the segment, 'poisson' exp (rates),
  'mse' as is.
* **Losses** (zero/auxtarget.py), per row, averaged over the rows that have a target: 'multinomial' =
  cross-entropy against the counts normalised to sum 1 (-sum_k t_k / n log p_k): the head learns the
  composition, and n x p is the expected count given the total - right when the total is public (a
  hand's size), and what stage 2 needs (an urn's weights). A row whose segment is all zero has no
  target (an empty hand). 'poisson' = mean_k exp(o_k) - t_k o_k (counts without a known total); 'mse'.
  The training loss adds `aux_<name>_w` x each; metrics `train.aux_<name>`, `aux_<name>_hit` (a
  multinomial target's most likely column is in the target), `train.aux` (rows with targets, their
  share, the mean total and non-empty share, the weights, `head_added`); the non-finite guard checks
  the auxiliary losses too.
* **Inference API** (zero/infer.py): `NetEvaluator.aux(x, n_players)` -> `{name: predictions [B, w]}`,
  one forward pass of the given rows (a decision's root); `{}` without a head. The search's own
  batches never compute the head (an additive method: `evaluate`, `launch` / `result` and the leaf
  batch protocol are unchanged). `set_weights` takes a state with or without the head either way (a
  rolling evaluator built from a checkpoint without the head, a run that turned it on later).

### 15.3 Off by default

Every weight 0 (the default): no driver records, no network has the head, `Window` loads nothing more,
the optimizer and the losses are the code of before line for line; config.json gains one key per
declared target (0.0). Checked byte for byte (15.6).

### 15.4 Carry-over: a checkpoint without the head

`net.add_aux(model, layout, seed)` returns a copy with a fresh head (seeded; the global torch RNG
untouched) and every other parameter and buffer exactly the checkpoint's, so its policy, value and
margin are unchanged (tested on the three archs: outputs equal, states equal); a network with that
head is returned as is, one with another head refused (`strip_aux` first). The run does it itself: a
run with a weight on whose checkpoint has no head - a fork of a run without one, `tools/fork_run.py
--source <run> --dest <fork> --iter K --set aux_<name>_w=0.5` - adds the head at its first training
(logged; `train.aux.head_added`), keeps the optimizer's AdamW moments for the trunk (the saved state
of one parameter group fits the trunk's parameters: it is loaded, and the head joins as a second
group with fresh moments) and trains on the window whose older shards have no targets (masked). Every
later checkpoint has the head and two parameter groups. The encoding guard (10.4) ignores the head (it
reads no input): such a checkpoint loads, resumes and is checked like any. A new run with a weight on
builds its first network with the head.

### 15.5 Replays (`zero.replay`, format 'zero-replay/1' extended)

A network seat whose checkpoint has a head lists its targets (`seats[i].aux`) and each of its
decisions' frames gets the game package's `replay_aux(predictions, state, seat, frame)` fields - the
head at the decision's root for the decider: one forward pass of the root's features after its search
(the search's evaluations do not compute the head, and a reused tree's root was evaluated during an
earlier search, so the root's own pass is the one that exists for every decision) - by default
`aux: {name: [[column, value], ...]}` (each target's 12 largest). MTG Zero renders `belief: {for, n,
cards: [[name, expected count], ...]}` (its ENGINE.md 66.5). A checkpoint without a head: the frames of
before. Readers of the format that do not know the fields ignore them.

### 15.6 Tests and checks

Rust: `aux_target_is_the_hidden_next_card_and_never_an_input` (the toy's target is its hidden card;
determinized worlds encode alike, their targets their own), `aux_records_the_decider_targets_and_
changes_nothing_else` (the golden records with aux on and off; each record's target its decision's,
replayed move for move; a game without targets refuses aux). `tests/test_aux.py` (18): the package's
declaration and setting, off by default, a negative weight refused; the binding's target and
determinize; the driver's records (every other array identical); shards and windows (old shards masked,
the cache, a wrong width refused); the head on the three archs (built last, outputs unchanged, last layer
zero), `add_aux` / `strip_aux` exact; a checkpoint with the head through the encoding guard; the losses
and predictions; `NetEvaluator.aux` and `set_weights` either way; a toy run with the target on (shards,
heads from iteration 0, two parameter groups, the loss below ln 6), a run off (nothing), a fork of it
with the target on (the head added at its first training, the trunk's optimizer state kept, the window
partly masked), replays with and without a head.

Checks (tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal; MTG Zero's A/B round live on the GPU and the Mac), on main
b486d36: `cargo test --release --workspace` 61 + 3 passed; `pytest tests` 246 passed, 2 skipped (the
CUDA graph-runner test deselected: it fails without a visible device, as on the base); byte identity
against b486d36 (its own worktree and target dir), the settings left out and given at their default,
`ZERO_INFLIGHT=1`: toy-smoke MLP vs `ext:toyheur`, tokens (classic, 3 iterations), cq targets +
`eval_select cq` guard 2 (rolling, 3), the Delver line's shape (rolling, cq 0.3, guard 4 at 0.3, raw Elo
offsets, no random games, value_q 0.6, window_cache, train_prefetch; 4) - every shard,
`elo/results.json`, `ratings.json` (`updated` aside), `models/index.json`, every checkpoint and
optimizer state tensor for tensor, the metrics without timings, config.json = the base's +
`aux_next_card_w` 0.0; `check_checkpoint_compat --package zero` (random, uniform, a toy MLP and a toy
token checkpoint) identical; Glass Road Zero c1f56dc, Caverna Zero a6fad73 and Kingdom Builder Zero
4202a4a build against it (build_py, every test compiled, the package imported). MTG Zero (branch
`belief`, its ENGINE.md 66.9): its suites, behaviour digests unchanged, `check_checkpoint_compat
--package mtgz` with delver2's iteration 41 identical.

### 15.7 What it measured on MTG Zero (its ENGINE.md 66.6)

On 80 held-out Delver games (CPU), a head on delver2's FROZEN trunk predicts the opponent's hand at
2.60 nats per card against 2.81 for the uniform deal the search uses today and 2.71 for a static
per-card bias (count-matching); over that bias (a residual head) 2.44. After 400 joint training steps
with the belief loss a fresh head reads 2.28 from the trunk against 2.54 after the same steps without
it: the auxiliary loss shapes the trunk, at +0.008 nats of held-out policy loss. Stage 2 - dealing the
search's determinizations from the head's prediction (a `SearchGame::determinize_with` hook, flagged
belief rows in the leaf batch) - is designed there (66.8), not built.

## 16. Root-view leaf values (MTG Zero's Delver mirror, 2026-10-29)

### 16.1 Why

The search values a leaf with the network on the view of the player to move there (docs/search.md
3): at a leaf where the opponent decides, the root player's value is read from a view that cannot see
the root player's hidden cards and that treats the opponent's sampled ones as known. MTG Zero
measured the bias on its Delver mirror (its ENGINE.md 64.4): a kept landless hand on the draw is
worth 0.36 to the keeper's own view and 0.51 to the next decider's, and every other leaf of a
two-player search is the opponent's.

### 16.2 What it does (docs/search.md 3.6)

* **Root-view values** (`selfplay_leaf_value_view` / `eval_leaf_value_view` = 'root'): a leaf of
  another seat than the tree's root decider takes its priors from its own row, `encode(decider)`, and
  its value and margin from a second, value-only row `encode(viewer)` in the same batch (flagged
  `ROW_VALUE` in the batch's `extra`). Evaluators are unchanged.
* **Value-only samples** (`selfplay_view_samples` = a share of the recorded decisions,
  `view_sample_target` = 'next_q' | 'z'): the real state from each other recording seat's view, in its
  perspective, with an empty option segment (no policy target); z its result, q its own next search's
  root value (or z). They teach the value head the views root-view leaves ask it for: before them it
  only ever saw `encode(decider)`.
* **Training** (section 4): a window holding value-only rows takes the policy loss and the policy
  entropy over the decision rows of each batch only (a mean over the batch's rows with options);
  the value and margin losses over every row. Metrics `train.value_view` / `value_decision` (the value
  loss on each kind of row, the last 200 steps) and `window_view_samples`; self-play's `view` (`rows`:
  value-only rows sent with leaves, `rows_per_leaf`, `samples`: value-only samples recorded). A window
  without value-only rows trains as before, bit for bit.
* **Evaluation and tools**: `eval_leaf_value_view` names the Elo pool (`visits-rv`,
  `cq-n4-v50-s0.3-rv`; after a floor: `-f4-rv`); `zero.h2h` and `zero.replay` follow a run's
  `eval_leaf_value_view` (`zero.h2h --a-leaf-value-view / --b-leaf-value-view mover | root`
  override it per side: the same checkpoint with both views is the cheapest test of the search alone).

### 16.3 Off by default

All four settings at their defaults (`mover`, `mover`, 0, `next_q`) are the run of before, byte for
byte: no seat key is added (`games.nn_seat`), the batches carry `extra` all zero, no row is added, no
sample recorded, the losses are computed as before. config.json gains the four keys.

### 16.4 Costs

* **Inference**: one extra row per leaf of another seat (about half the leaves of a two-player
  game). On a CPU (inference-bound) leaves/s drop by a third (MTG Zero's Delver network, 64 games: 1,313
  -> 887 leaves/s, the same rows/s); a GPU or MPS evaluator, whose cost grows less than its batch, pays
  less. Not measured on the GPU (A/B round 2 holds it).
* **Samples**: `selfplay_view_samples` x (players - 1) extra samples per recorded decision - as many
  more rows in the shards and the window (`ram_budget_gb` holds fewer iterations) and, since the
  training's steps follow the new samples (`new samples x reuse / batch`), as many more steps.
* **Remote self-play / evaluation**: the seat keys `leaf_value_view`, `view_samples`,
  `view_sample_target` go to the remote's driver: its checkout must have this core (an older one
  refuses an unknown seat key - a visible error, never a silent off).

### 16.5 Tests and checks

Rust: `leaf_view_settings_and_view_pick`, `root_view_off_and_value_samples_leave_the_searches_alone`
(the golden records and match with the settings at their defaults; value-only samples change nothing
but add records), `root_view_values_read_the_root_deciders_hand` (`mull_game::HiddenMull`, a
private-hand keep-or-mulligan: the batch layout, the value from the keeper's own hand in every world),
`value_only_samples_and_their_targets`, `value_only_samples_carry_the_observers_auxiliary_targets`.
`tests/test_root_view.py` (12): settings and names (the pool, zerodash, h2h), `view_pick` = the
Rust hash, Rust vs Python trees at the noise floor with a power check, the binding's 4-tuple, the
drivers' ('py', 'rs', Rust) samples and targets and flags, toy runs (below).

Checks (tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal; A/B round 2 live on the GPU and the Mac), on main a560448:
`cargo test --release --workspace` 65 + 3 passed and `queued_setups_are_shared_between_the_groups`
failed under the machine's load (timing-dependent: it failed 2 of 3 times on the base b486d36 under the
same load, 14.5; alone it passes 8 of 8); the feature's 5 Rust tests in a debug build too; `pytest tests`
258 passed, 3 skipped (CUDA); `python -m zero.parity root_view` (docs/search.md 6); toy-smoke runs
byte for byte against a560448 (its own worktree and target, `ZERO_INFLIGHT=1`), the settings left out
and given at their defaults - classic, rolling, margin_w 0.2 + chance_k 2 + cq targets + eval_every 2:
every shard, checkpoint (zip members), optimizer state, Elo file (`updated` aside), models/index.json,
the metrics without timings; `check_checkpoint_compat --package zero` (random, uniform, a toy
checkpoint) identical; Glass Road Zero c1f56dc, Caverna Zero a6fad73 and Kingdom Builder Zero 4202a4a
build against it. MTG Zero (branch `root-view-values`, its ENGINE.md 67): its suites, behaviour
digests unchanged, `check_checkpoint_compat --package mtgz` with delver2's iteration 41 identical.

### 16.6 What it measured on MTG Zero (its ENGINE.md 67)

On delver2's networks (CPU): the two views of the same real self-play states disagree by 0.077 on
average (p90 0.21) but neither is biased (the decider's own value minus the other view's: -0.002 over
13,442 pairs of 192 games), and the value head is about as calibrated on the never-trained other view
(log loss 0.367, ECE 0.014) as on the decider's (0.358, 0.025): the distribution shift is real but
small. A 300-step fine-tune with value-only samples improves the other view's held-out log loss
0.349 -> 0.330, a control without them to 0.334. In searches the views differ more (opponent leaves
of mulligan searches: 0.11 on average, 73% by more than 0.05); root-view values bring a 0-land keep's
Q on the draw down to the mulligan's (0.485 -> 0.378 against 0.37 after the fine-tune) but the keep
rates barely move (the prior holds the visits), and a replay's upkeep Pass the opponent's view valued
at 0.51 is worth 0.64 with root-view leaves - the move (Brainstorm, prior 0.93) stays.

## 17. Tooling for the tiered verification gates (MTG Zero's `tools/gate.py`, 2026-10-29)

MTG Zero's policy (its ENGINE.md 70, memory `feedback_tiered_gates`) tiers its verification gates
by change type instead of running the full search smoke before every merge; two zero-core-side
pieces support it directly:

**`tools/smoke_all.sh --build-only`**: per game, run `tools/build_py.sh` only - no `cargo test`,
`pytest` or smoke training run. For the tier "an opt-in core feature with byte-identical
defaults" the question is narrower than the full cross-game smoke answers: does the OTHER games'
build still compile against this core commit, nothing about their behaviour (the feature is off
by default there; MTG Zero's own suite already covers whether it changed anything). Combine with
`--with-this-core` to build the games against THIS checkout's HEAD rather than their own pin.
Default game discovery (no explicit paths) now also looks for `../kingdombuilder-zero`, alongside
`../glassroad-zero` and `../caverna-zero` - it was missing from the list before this. (From a worktree or a
`third_party/zero-core` submodule that discovery found no game and printed GREEN anyway until 29.)

**De-flaked `crates/search/src/driver/tests.rs::queued_setups_are_shared_between_the_groups`**:
asserted the exact POSITION of each group's dispatch size in the combined `sizes` vector it
collects across every `next_batch` call (`sizes[..2]`, `sizes[2..4]`). Reproduced reliably under
real thread contention (ten copies of the test binary running at once, roughly 1 run in 300): the
two groups' batches are reported in THREAD-SCHEDULING order (which group's worker happens to call
back first), not in the order `queue_share`'s fairness logic (the thing under test) actually
assigns work - under load one group can report several batches before the other's first. Fixed by
asserting each group's OWN sequence of distinct sizes (deduplicated: a size repeats across many
calls while that wave of games plays out its turns) is `[1, 4]`, which does not depend on how the
two groups' reports interleave; verified against the original (still reproduced the failure) and
the fix (480/480 passed) under the same contention. The sibling test
`queued_setups_fill_every_free_slot_when_the_queue_is_long` and the rest of the module (36 tests)
were unaffected and still pass.

**De-flaked `tests/test_remoteplay.py::test_pause_mid_phase_then_resume`**: waited for
`progress.json`'s `remote.games > 50` inside a flat 180 s window, which occasionally saw
`{'phase': 'idle'}` (a different phase's progress shape, no `remote` key) for long enough under
load to miss the window - not because anything was stuck, but because a loaded machine is simply
slower. The new `wait_for_remote_progress` polls with a stall-based deadline instead of a flat
one (it keeps extending as long as the remote's game count keeps climbing, and gives up only once
that count has been flat for `stall_s`, or a hard ceiling `max_s` is hit), resets what counts as
"progress" on an iteration boundary (so a stale high-water mark from an already-finished phase
cannot short-circuit the wait for a new one), tolerates a torn read of `progress.json` (keeps the
last good reading instead of resetting to `{}`), and fails immediately with the run's own output
if the subprocess has already exited, rather than spinning out the rest of the window on a run
that is never coming back.

## 18. The root's belief: belief-weighted determinization (MTG Zero's Delver mirror, 2026-10-29)

### 18.1 Why

The search deals the hidden information uniformly (docs/search.md 3): every simulation a fresh world,
every world consistent with what the root decider has seen equally likely. An auxiliary head (section
15) can know better: MTG Zero's belief head predicts the opponent's hand from public information at 2.28
nats per card against the uniform deal's 2.81 on held-out Delver positions, and reads "they passed with
blue open: Force of Will" (its ENGINE.md 66). The root's belief deals the search's worlds from that
prediction - the design of MTG Zero's ENGINE.md 66.8, built here and in its ENGINE.md 69.

### 18.2 What it does (docs/search.md 3.8)

* **Settings**: `selfplay_belief_tilt` (the self-play seat - a league opponent's too, being the
  self-play seat), `eval_belief_tilt` (every evaluation seat), `belief_clip` (R, default 20),
  `belief_mix` (the share of a belief search's simulations still dealt uniformly, in [0, 1)),
  `belief_fast` (0: full searches only - every evaluation search, self-play's recorded ones; 1: fast and
  minor searches too). Seat keys `belief_tilt` / `belief_clip` / `belief_mix` / `belief_fast`, only when
  the tilt is > 0 (`games.nn_seat`, `run.belief_keys`).
* **One row per search**: a search that takes it sends the root's own features (the decider's view) as a
  `ROW_BELIEF` row - with the fresh root's first leaf row, or alone for a reused root (its simulations
  wait one round trip) -, the evaluator answers that row's auxiliary predictions (`aux`, the fourth item
  of `NetEvaluator`'s answer: the head on those rows only), and every simulation of the search deals its
  world with the game's `determinize_with` at the seat's tilt and clip. A belief is one decision's: a
  reused tree's (and a new network's search, a rolling epoch's) asks again.
* **The game's weights** (its `determinize_with`): MTG Zero, its ENGINE.md 69.2: per card identity of the
  opponent's hidden pool, the belief's expected count among the hand's unknown cards against the uniform
  deal's, to the tilt, clipped; the hand's unknown slots drawn one at a time by those weights; every
  constraint of `determinize` kept. A game without one ignores the belief (the default `determinize_with`);
  the driver refuses `belief_tilt` for a game without `AUX` (nothing to predict).
* **Evaluation and tools**: `eval_belief_tilt` names the Elo pool (`mcts.belief_tag`: `visits-b1`,
  `cq-n4-v50-s0.3-b0.5m0.25`, after a view: `-rv-b1`); `zero.h2h` and `zero.replay` follow a run's
  evaluation belief (its config.json), `zero.h2h --a-belief-tilt / --b-belief-tilt` (0 = uniform) and
  `--*-belief-clip / -mix / -fast` override it per side: `A A --a-belief-tilt 1 --b-belief-tilt 0` is
  the evaluation-only A/B on one checkpoint. Self-play's metrics gain `selfplay.belief` (`rows`: belief
  rows its driver sent, `per_decision`) when its tilt is on.

### 18.3 Off by default; a network without the head is refused

With every tilt 0 (the default) nothing changes: no seat key, no row, no draw (`Tree::set_belief` is
never called; the golden records and match, toy runs byte for byte: 18.7). config.json gains the five
keys.

**A checkpoint without the auxiliary head cannot serve a tilt > 0, and the run refuses it** - it does not
fall back to the uniform deal with a warning. The choice, against MTG Zero's rule that a step which
silently does nothing is a bug (memory `feedback_silent_noop_pipelines`): a run (or an A/B arm, or an h2h
side) asked for belief-weighted searches that quietly dealt uniformly would report a result - "the belief
does not help" - for an experiment that never happened, the false success the rule exists to prevent, and a
warning in a log of thousands of lines is no defence (the shards, metrics and Elo would look the same).
The cost of refusing is small and visible: the run stops at start with the reason. So:

* the run checks its newest checkpoint at start (`Run.check_belief_heads`: `net.check_belief_head`, the
  head for exactly the game's targets) and refuses with `net.HeadMissing` - an `EncodingMismatch`, so the
  refusal paths of the encoding guard (section 10.4) apply: exit status 78, state `refused`, the
  supervisor does not restart it; a fresh run gets the head from `aux_<name>_w` > 0 (an untrained head
  predicts uniformly over the identities: the clip makes that deal about uniform);
* a league opponent's checkpoint (`league.load_into`) is checked the same way when the self-play tilt is on;
* `NetEvaluator` raises `HeadMissing` for a batch with belief rows and no head (every other path: h2h,
  replays, remote evaluation, a remote worker); `zero.h2h` checks both sides before the first game;
* the driver refuses an answer without the belief rows' `aux` (`UniformEvaluator`, a custom evaluator).

### 18.4 Interactions

* **Tree reuse**: kept; the belief is recomputed per decision (`Tree::advance` clears it; the reused root
  sends its row alone and waits). The reused subtree's statistics were dealt by the previous decision's
  belief - as a reused tree's always came from an earlier information set.
* **Fast and minor searches**: no belief unless `belief_fast` (they record nothing; one row is 1/16 or 1/4
  of their leaves).
* **Root-view leaf values** (section 16): independent; both extra row kinds share the batches.
* **Root minimum visits, the move rules, cq targets**: untouched (the belief changes the worlds only).
* **The league** (section 14): the opponent's seat is the self-play seat and takes the belief; its pool's
  checkpoints need the head (refused otherwise): a `selfplay_league_pool` reaching back before the head
  was trained must be narrowed.
* **Rolling self-play**: a new network restarts every search and its belief; a phase's end drops a pending
  belief row as it drops leaves (asked again next phase).
* **Remote self-play** (section 5.3): 18.5. **Remote evaluation** (5.2): the mirror's config.json carries
  `eval_belief_tilt`; the result's rule name must carry the belief tag, so an older checkout (no tag) is
  refused before anything is recorded, as for the move rule; its checkpoints need the head there too.
* **Co-training**: the resolver's own settings (`resolver_cfg`) as any.

### 18.5 Remote self-play: an older checkout is refused, not silent

A remote worker builds its seat from the run's config.json with its own checkout's code: an older
checkout would read no belief setting and deal uniformly, its shards indistinguishable. So the worker's
hello reports its self-play seat's belief keys (`remoteplay._belief_of`), and the run compares them with
its own: a worker without them (an older checkout) or reading them otherwise is killed at its hello and
retried every 10 minutes (log `worker K REFUSED: its checkout plays the root belief as ...`); this machine
plays every game meanwhile (`tests/test_belief.py::test_remote_selfplay_refuses_a_checkout_without_the_belief`).
MTG Zero's Mac checkout is the run's commit anyway (its `RunCommitRemote`). (The same gap exists for any
seat setting a remote worker reads from config.json - root-view values', section 16.4, say an older
checkout "refuses an unknown seat key", which holds for shipped seat dicts, not for config.json; the
commit-pinned checkout is what protects them.)

### 18.6 Costs

* **Inference**: one extra row per search that takes the belief (1/sims of its leaves: +3% at 32
  simulations, +1.6% at 64) and the head on those rows; a reused root's search waits one batch round trip
  for it. MTG Zero's Delver network on the desktop CPU (64 games, 4 torch + 4 search threads, the arms
  interleaved over 4 rounds on a shared machine, its ENGINE.md 69.6): evaluation seats at 32 simulations
  1,402 -> 1,304 leaves/s, the paired ratio 0.929 +- 0.019 (0.032 belief rows per leaf: the rows and the
  reused roots' waits, ~3% each); the line's self-play seat (64 / 16, p_full 0.25, minor 4) 0.974 +- 0.034
  (0.0094 rows per leaf - only the full searches take it).
* **The deal**: the game's (MTG Zero: the weights and weighted draws over <= 64 hidden cards, 6 us a call
  against `determinize`'s 1 us - every simulation deals, nothing beside its leaf's evaluation).
* **Training**: unchanged (stage 1's target and head, section 15).

### 18.7 Tests and checks

Rust (`zero_search`, 11 new): `belief_weight_formula`, `a_belief_deals_the_worlds_by_its_weights` (the
toy's next card: uniform 1/k, a sharp belief, the clip, the uniform share), `no_belief_is_the_tree_of_before_and_advance_clears_it`,
`without_redet_the_belief_deals_the_decisions_one_world_again`, `belief_off_is_the_search_of_before` (the
golden records and match with the settings given), `belief_settings_are_checked`,
`one_belief_row_per_search_that_takes_it` (the layout; rows = full searches; `belief_fast`; evaluation
seats), `an_evaluator_without_the_belief_stops_the_driver`, `the_belief_deals_the_searches_worlds`
(HiddenMull: the analytic shift of seat 1's root value, with and without redet),
`root_view_values_and_the_belief_share_the_batches`, `rolling_phases_ask_the_belief_again_of_a_new_network`.
`tests/test_belief.py` (11): settings and names (the pool), the toy's deal at its odds, Rust vs Python trees
at the noise floor with a power check (`parity.belief`), the three drivers' rows answered by the head,
`HeadMissing` and the drivers' refusal, the Python tree's uniform share, `zero.h2h`'s sides and flags, toy
runs (off byte for byte, self-play and evaluation on, a run without the head refused with exit 78), remote
self-play refusing a checkout without it.

Checks (tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal; A/B round 3 live on the GPU and the Mac), on main 3b5134d:
`cargo test --release --workspace` 77 + 3 passed; `pytest tests` 268 passed, 2 skipped (the CUDA graph-runner
test deselected: no visible device), and one `test_a_dying_worker_gives_its_accepted_games_back` failed once
under the shared machine's load (30 games of the dying worker not played again: a timing race of the quota) - it
passes alone on this branch and on 3b5134d, and `tests/test_remoteplay.py` passes whole (27);
`python -m zero.parity belief` (docs/search.md 6: Rust-Python at the noise floor, the belief's effect 10x it);
toy-smoke runs byte for byte against 3b5134d (its own worktree and target, `ZERO_INFLIGHT=1`), the settings left
out and given at their defaults - classic MLP against `ext:toyheur`, classic tokens, rolling with cq targets and
`eval_select cq` guard 2, the Delver line's shape (rolling, cq 0.3, guard 4 at 0.3, raw Elo offsets, no random
games, value_q 0.6, window_cache, train_prefetch, tokens, margin_w 0.2, chance_k 2): every shard, checkpoint (zip
members), optimizer state, Elo file (`updated` aside), models/index.json, the metrics without timings identical
(12-14 files a run), config.json = the base's + the five keys; `check_checkpoint_compat --package zero` (random,
uniform, a toy MLP and a toy token checkpoint): identical; `smoke_all.sh --with-this-core --build-only`: Glass
Road Zero c1f56dc (47 s), Caverna Zero a6fad73 (54 s) and Kingdom Builder Zero 4202a4a (78 s) build against it.
MTG Zero (branch `belief-search`, its ENGINE.md 69): its suites, behaviour digests unchanged,
`check_checkpoint_compat --package mtgz` identical, the weighted-deal fuzz.

### 18.8 What it measured on MTG Zero (its ENGINE.md 69)

On the Delver mirror with stage 1's two networks (desktop CPU, 2026-10-29): jt, delver2's iteration 41
fine-tuned jointly with the head, and ho, the head fitted on the frozen iteration 41.

* **The deal against the true hand** (2,412 positions of a network's games against itself, 200 worlds each):
  the cards misplaced by the worlds' mean hand 2.84 uniform -> 2.64 / 2.54 / 2.52 at tilt 0.5 / 1 / 2; the
  cross-entropy per true card 2.78 -> 2.62 / 2.68 / 3.04 (tilt 2 overconfident: 97 distinct hands of 200
  instead of 151). Mostly base rates the uniform deal gets wrong: the opponent holds Force of Will in 60% of
  the positions, the uniform deal gives it one in 30%, tilt 1 in 54%; discrimination within a stratum is
  modest.
* **Decisions**: at 64 simulations the move of 40 threat-casting decisions into open mana changed at 1; the
  threat's Q fell where the head expected a counter (Murktide Regent into an expected Daze 0.58 -> 0.47).
* **Head to head** (the same checkpoint, A belief-weighted against B uniform, 32 simulations, 300 games an
  arm, the line's rule): a null arm (both uniform) 0.517 +- 0.029; jt at tilt 1 0.567 under mover-view and
  under root-view leaf values, tilt 1 with mix 0.25 0.512 / 0.557, tilt 0.5 0.517 - jt's five belief arms
  pooled 0.544 +- 0.013 (+3.4 se, about +30 Elo), the settings not separable at this size; ho at tilt 1 0.500.
* **Cost**: 3-7% of an evaluation's leaves per second, 1-3% of self-play's (section 18.6).

The A/B (its ENGINE.md 69.9): evaluation only first (the same checkpoint both sides, the round's h2h), then a
self-play arm (`selfplay_belief_tilt=1`, `eval_belief_tilt=1`, `belief_mix=0.25`, `aux_belief_w=0.25`) once
the main line has a trained head.

## 19. The Gumbel root search (MTG Zero's Delver mirror, 2026-10-29)

### 19.1 Why

At 64 simulations PUCT's root follows the prior: MTG Zero's Delver mirror (its ENGINE.md 64, 67) put
>= 0.98 on Keep in 89 of 100 one-land hands with Mulligan often unvisited, and an upkeep Brainstorm with
prior 0.934 took 62 of 63 visits - Pass 1, also once root-view leaf values (section 16) had raised
Pass's Q to within 0.014 of Brainstorm's. The root minimum visits (section 9.2) and root-view values
measure or correct a starved option's Q but leave the allocation and the move to PUCT and the prior.
Gumbel AlphaZero (Danihelka et al. 2022) replaces the root itself: candidates sampled by Gumbel-top-k,
the budget spread by sequential halving, the move the argmax of noise + log-prior + sigma(completed
Q) - a policy improvement whatever the budget. The completed-Q target (section 9) and the 'cq' move
rule (section 12) were already its other two pieces.

### 19.2 What it does (docs/search.md 3.7)

* **Settings** (section 1.2): `selfplay_root_search` / `eval_root_search` ('puct' | 'gumbel'),
  `gumbel_m` (16), `selfplay_gumbel_scale` (1), `eval_gumbel_scale` (0). Seat keys `root_search`,
  `gumbel_m`, `gumbel_scale`, `gumbel_c_visit`, `gumbel_c_scale` (`games.nn_seat` writes them only for a
  Gumbel seat, with sigma resolved).
* **Self-play**: every search of the seat (full, fast, minor) is a Gumbel search. Its noise replaces
  the root Dirichlet noise and the temperature: the first `temp_moves` searched decisions of a game are
  searched with noise (`selfplay_gumbel_scale`; the move a sample of the improved policy), later ones
  without (the argmax). `selfplay_select` is not used. The recorded target must be 'cq'
  (`policy_target`; checked at the start, `check_eval_settings`) and shares the search's sigma
  (`cq_c_visit`, `cq_c_scale`). The root value target's search part q is the improved policy's value
  (`gumbel_q`), not the root's mean backup (docs/search.md 3.7: sequential halving walks every
  considered option in its first round). Value-only samples' `next_q` is the observer's next
  search's `gumbel_q`. The league seat (section 14) is the self-play seat: Gumbel too.
* **Evaluation**: every evaluation seat (yardstick, random, Elo, the `eval_remote`, co-training's) with
  `eval_root_search` 'gumbel' plays the Gumbel move with noise `eval_gumbel_scale` (0: deterministic,
  the paper's evaluation; a positive scale samples), sigma the evaluation rule's transform
  (`eval_select_c_*`, else `cq_c_*`); `eval_select` and its guard are not used. The Elo pool is the
  search's (`mcts.gumbel_tag`: `gumbel-m16-v50-s0.3` at Delver's 0.3; zerodash's `elo_pool` alike).
* **Tools**: `zero.h2h` gives each side its run's `eval_root_search` (from config.json, whatever
  `--select` says, so the A/B driver's fixed `--select cq ...` flags leave a Gumbel-evaluated fork's side
  Gumbel), `--a-root-search` / `--b-root-search` override it, `--gumbel-m` and `--a-gumbel-scale` /
  `--b-gumbel-scale` set its parameters; a Gumbel side's sigma is its 'cq' rule's transform (the
  `--*-cq-c-*` flags or its run's). `zero.replay` seats follow the run's root search.
* **Exclusive** with the same side's root minimum visits (`*_root_min_visits` > 0: an error).
* **With the root's belief** (section 18): both on one seat work together - the belief deals the
  Gumbel search's worlds, a reused root's belief row costs it no simulation (tested: below); a
  Gumbel-evaluated run's Elo pool carries both (`gumbel-m16-v50-s0.3-b1`).

### 19.3 Off by default

`puct` everywhere (the defaults) is the run of before, byte for byte: no seat key is added, no noise is
drawn from another stream, the tree's Gumbel state is never set (the golden records and match; toy-smoke
runs below). config.json gains the five keys.

### 19.4 Costs

* **Search**: none measured. The root's selection computes the completed-Q logits once per walk (O(options)
  at the root only). MTG Zero's Delver self-play on the desktop CPU (64 games of the line's seat, 4 torch +
  4 search threads, 120 s each): PUCT 2,042 / 2,065 leaves/s, Gumbel 2,098; on the final build, at another
  hour's background load, PUCT 1,747, Gumbel 1,794 (its ENGINE.md 68.5; fewer walks end in a terminal, the
  search's own time per leaf 63 vs 68 us).
* **Training**: the same rows and steps (the target was 'cq' already).
* **Remote self-play / evaluation**: the seat keys go to the remote's driver: its checkout must have
  this core (an older one refuses `root_search` - a visible error, never a silent off): MTG Zero's
  `mac.sh --eval-checkout` before such a run is pinned.

### 19.5 Tests and checks

Rust (`cargo test`, 88 tests): `tree::tests::gumbel_sequence_is_mctxs`, `gumbel_root_walks_follow_
sequential_halving` (toy roots of 3-12 options, m 4 / 16: the schedule's counts, the Gumbel-top-k,
the move and q formulas), `gumbel_off_changes_no_walk_and_reuse_budgets_this_search`,
`gumbel_measures_the_starved_mulligan`; `driver::tests::gumbel_settings_are_checked`,
`gumbel_selfplay_is_reproducible_and_its_records_valid`, `gumbel_noise_only_within_temp_moves`,
`gumbel_mulligan_selfplay_and_evaluation`, `gumbel_with_root_view_leaves_and_rolling_restarts`,
`gumbel_with_a_root_belief` (the belief on every search, fresh and reused roots: records valid, the same
across threads, other than without it);
`noise::tests::gumbel_moments_and_the_max_trick`. `tests/test_gumbel.py` (15): settings, seats and names
(the pool, zerodash, h2h, replay), the schedule = the Rust tree's, the Python formulas on Rust roots
exactly, the trees at the noise floor with a power check, the Python tree's halving and reuse, the
drivers ('py', 'rs', Rust) alike, the 'visits' target refused, noise and temperature, toy runs, the
Gumbel root with the root's belief in the three drivers (a network with the auxiliary head).

Checks (tier "opt-in core feature with byte-identical defaults"; desktop 2026-10-29, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal; A/B round 3 live on the GPU and the Mac), on main 353335c (rebased
after the root's belief; first checked on 3b5134d / 2c04cd9 alike): `cargo test --release` 88 passed;
`pytest tests` 283 passed, 3 skipped (CUDA), and `test_remoteplay.py::test_a_dying_worker_gives_its_accepted_games_back` failed under the CPU's load (a remote-play queue's timing, the Gumbel search off there; the same on 3b5134d's run) - alone it passes on this branch and on 353335c; `python -m zero.parity gumbel` (docs/search.md 6);
toy-smoke runs byte for byte against 353335c (its own worktree and target, `ZERO_INFLIGHT=1`, 3
iterations, seed 3) - classic, rolling with 'cq' targets and the cq4 rule, margin_w 0.2 + chance_k 2 + cq
+ eval_every 2, and the new settings given at their defaults: every shard, checkpoint (zip members),
optimizer state, Elo file (`updated` aside), models/index.json and the metrics without timings;
`check_checkpoint_compat --package zero` (random, uniform, a toy checkpoint) identical; Glass Road Zero
c1f56dc, Caverna Zero a6fad73 and Kingdom Builder Zero 4202a4a build against it (`smoke_all.sh
--with-this-core --build-only`). MTG Zero (branch `gumbel-root`, its ENGINE.md 68): its suites, behaviour
digests unchanged, `check_checkpoint_compat --package mtgz` with delver2's iteration 51 identical.

### 19.6 What it measured on MTG Zero (its ENGINE.md 68)

On delver2's iteration 51 (CPU): the Gumbel root walks the mulligan's Mulligan 31-32 times per search
(PUCT: 0-1 on the draw) and the replay's upkeep Pass 31 of 63 times (PUCT: 1) - and neither move
changes: measured, Mulligan's Q is far below Keep's where the line keeps (one land on the draw: 0.36
against 0.50) and Pass's 0.03 below Brainstorm's, so the keep rates stay 58% (no land) / 91% (one land)
under every mode and view, and Brainstorm is cast 16 of 16. A Gumbel search's plain root mean of a
kept hand is 0.435 against Keep's Q 0.518; `gumbel_q` is 0.518 (PUCT's root value 0.519): the reason q
is the improved policy's value. Head to head on the same checkpoint (seats alternating, the Delver spec,
side B PUCT with the line's rule cq4 s0.3): 32 simulations 0.468 +- 0.020 (600 games), 64 simulations
0.502 +- 0.020 (600), both sides with root-view leaves at 32 0.520 +- 0.029 (300) - no search-only gain
either way on a network trained on PUCT's searches; whether training on Gumbel's targets and moves helps
is the A/B's question (its ENGINE.md 68.7). Self-play records are valid and as cheap (1,794 against
1,747 leaves/s on the final build).

## 20. Strength measurement: fixed anchors, held-out losses, paired and sequential head-to-heads (MTG Zero's Delver mirror, 2026-10-30)

A Fable review of MTG Zero's Delver main line (`runs/r3-2b-beliefsearch`, paused at iteration 196; its
ENGINE.md 73) found that the line's yardsticks could not see its progress:

* **The Elo pool is too coarse.** Every 8 iterations each checkpoint plays the checkpoints 1, 2, 4, 8
  iterations back (`elo_offsets`), 25 games each at 32 simulations: +-50-115 Elo per rating, and no
  fixed reference (each rating is a chain of noisy links to the yardstick). The pooled pairs give 0.481
  +- 0.015, -1.9 +- 2.3 Elo per iteration: "plateau" and "+1 Elo a generation" look alike.
* **The yardstick is saturated** (92-98% wins against the heuristic): its games carry almost no
  information, and the user stopped testing against it.
* **Head-to-head verdicts are too noisy.** `zero.h2h` (600 games at 32 simulations, se 0.020)
  alternates the seats but does not pair the games: each game's deal comes from its own gid, so the
  opening hands' variance of a mirror stays in the score. Two features were adopted at 0.538 and 0.534,
  just under the 2-se bar (0.541), and a null cannot be told from +10 Elo.

This section adds four opt-in measurements; every one off is the run of before, byte for byte (20.6). What
they measured on the Delver line: 20.7.

### 20.1 Fixed anchors in the Elo pool (`elo_anchors`)

* **Settings** (run.py DEFAULTS): `elo_anchors` - frozen checkpoints, comma-separated, each `path` or
  `label=path` (label default: the file's stem); '' = off. `elo_anchor_games` (100): games per anchor
  for every evaluated checkpoint. `elo_anchor_elo` (1000): the first anchor's fixed rating.
  `elo_anchor_calib_games` (200): games of a later anchor against each anchor fixed before it.
  `elo_anchor_from` (0): only checkpoints from this iteration on play the anchors (0: every evaluated
  checkpoint of the pool, from its `since`). `elo_paired` (0): paired Elo matches (20.3). Single-game
  runs only (a co-training run plays its matches as events: refused).
* **Players**: an anchor is the pool's player `anchor:<label>` (never a checkpoint's name, even when it
  is a copy of one: it is a frozen network, not the run's), played with the run's evaluation seat (the
  pool's move rule, `elo_sims`), so it must be of the run's encoding and, with a root belief, have the
  auxiliary head (checked at the start: refused, exit 78).
* **Fixing them** (`Run.anchor_matches`, in each `elo_update`): the pool's first anchor is fixed at
  `elo_anchor_elo`; each later anchor plays `elo_anchor_calib_games` against every anchor already fixed
  and is fixed at the rating of a fit of those games alone with the earlier ones held. The values, the
  file's sha256, its name, the iteration and how it was fixed are kept in the pool's `results.json`
  (`anchors`) and never refit: an anchor's games against later checkpoints never move the scale.
  Another file under a fixed label is refused (`AnchorRefused`, exit 78: a moved scale in silence would
  be worse than no anchor); a new label is a new anchor, calibrated against the old ones.
* **Anchor games**: every evaluated checkpoint (`eval_every` multiples, iteration >= the pool's `since`
  and `elo_anchor_from`) that has not played an anchor plays it, all of its anchors in one driver
  (`play_matches` with the anchors' paths; seed 53 + 7919 k), so switching anchors on for a running
  pool first catches its evaluated checkpoints up (the first evaluation after the switch is that much
  longer). Recorded in `results.json` like any Elo match (`elo match iter_0200 vs anchor:a192:
  52.0/100` in the log).
* **The fits**: the pool's own fit (`elo.fit`, the yardstick at 1000, `ratings.json`, `models/index.json`,
  `best.pt`) is computed as before over all the pool's games, the anchors' included (free players
  there, listed with the references). The anchored fit, `elo.fit_fixed(res, fixed)`, is the
  Bradley-Terry maximum likelihood over the pool's WHOLE results file with the anchors held at their
  values and every other player free (the yardstick too); its weak Gaussian prior is centred on the
  anchors' mean. -> `anchored.json` (the anchors and every player's rating), `elo_anchored` {elo, se} in
  the iteration's metrics (and `evals.jsonl` from a remote), a log line `iter 200 elo anchored 1640 +-38`
  (+- 1.96 se). A checkpoint's anchored se comes from its anchor games (200 games near 50%: ~25 Elo)
  and shrinks as later checkpoints play it (the offsets); none of it is a chain to a saturated
  yardstick. With one anchor `fit_fixed` is `fit` (tested to 1e-6).
* **Remote evaluation** (`eval_remote`, 5.2): `RemoteEval` puts each anchor once a session to
  `<mirror>/anchors/<label>.pt` and writes the mirror's config with `label=<that path>`, so the players
  and the sha256 are the same on both machines; the mirror's `anchored.json` and `elo_anchored` come back
  with the pool's files. The anchors' calibration and games happen on the remote, as the pool's other
  matches.
* **Dropping the yardstick's games**: `--eval-games 0` (the run's `eval_random_games` 0 already). The
  iteration's `eval_yard` then has 0 games - it still marks the checkpoint evaluated (the pool's
  `logged`, the remote's `done_iters`) and every Elo match (offsets, anchors) is played: tested. A pool
  whose older iterations played the yardstick keeps its link to it (the ordinary fit); a fresh run
  without yardstick games has no ordinary rating link (its `elo` se is the prior's), and the anchored fit
  does not need one.
* **Trade-offs**: a later anchor's calibration error (~174 / sqrt(calib games / 4) Elo: 200 games ~25)
  is frozen with it: a constant offset of the ratings that lean on that anchor, never a trend. Anchors
  near the line's strength give the most information per game (an anchor scored 90% against teaches
  little); add a stronger one later (a new label), it is calibrated against the others.

### 20.2 Held-out losses (`heldout_shards`, `zero/heldout.py`)

The cheapest learning signal: a checkpoint's losses on a FIXED set of sample rows no checkpoint trains
on. `python -m <pkg>.az.heldout make --from runs/x/data/iter_0191 --rows 8192 --out set.npz` draws a
uniform random subset (fixed seed) of an iteration's self-play rows and writes it as one compressed
shard (its meta: the source, rows, seed). Iteration i's rows train checkpoints i .. i + window - 1:
they are held out for every other checkpoint, the run's later ones included.

* **Metrics** (`heldout.losses`), per kind of row: **decision** rows (recorded searches) and **view** rows
  (value-only samples, section 16): `value_ce` (cross-entropy of the value head against the game's
  outcome z: the training loss with value_q 0), `value_acc` (rows with one winner: the most likely seat
  won), `value_brier` (squared error of P(win)); the decisions also `policy_ce` (against the recorded
  target, decisions with >= 2 options) and `policy_top1` (argmax agreement). `set` names the set (a hash
  of its shards' names, sizes and rows): numbers are compared across checkpoints on one set (paired
  rows: their own difficulty cancels), never across sets.
* **Run settings**: `heldout_shards` (files or directories, comma-separated; '' = off) and
  `heldout_device` ('' = `train_device`): every evaluated iteration gets `heldout` in its metrics row and
  a log line (`iter 200 heldout (set 463ad8eefa5c): decision value_ce 0.3812 acc 0.8102 policy_ce ...`),
  computed in the run process right after training (also with `eval_remote`). The set is loaded once at
  the start (8,192 MTG rows ~0.6 GB of RAM) and must be of the run's game and encoding (refused
  otherwise, exit 78); a failure later is logged (`HELDOUT FAILED`), never the run's end.
* **Reading it**: `value_ce` against z is the clean signal (z is the truth). `policy_ce` against a frozen
  target set measures agreement with the search that recorded it: it falls towards the checkpoint that
  generated the set and may rise again after it with no loss of strength - read it as drift, not
  strength. A checkpoint that trained on the set is contaminated (its value loss falls sharply on rows it
  saw: below).
* **Cost**: 2.4 s per checkpoint on the desktop CPU for 8,192 MTG rows (128 x 4 tokens net).

### 20.3 Paired games and sequential head-to-heads (`zero.h2h`, `elo_paired`)

* **Paired games** (docs/search.md 2.10): the driver's `paired` gives games 2k and 2k + 1 every random
  stream of the pair k (the deal, shuffles and chance, the driver's streams); the match tables swap the
  seats between them, so each deal is played once from each seat by each side. `zero.h2h --paired`
  (`match(..., paired=True)`): `pairs` {n, both_won, split, both_lost, with_draw, pentanomial (counts of
  a pair's points 0, 0.5, 1, 1.5, 2), score, se}; `se` is then the se of the pair scores, `se_games` the
  games' binomial se (the unpaired formula), `variance_ratio` = var(pair score) / (var(game) / 2), the
  share of the variance the pairing left (1: none removed; below 1: the deal's variance is gone). The
  A/B driver's rule `score >= 0.5 + 2 se` reads the paired se unchanged. The run setting `elo_paired` 1
  plays the run's Elo matches (offsets, anchors, calibration) paired (game counts even; a pair never
  spans two opponents). The Bradley-Terry fit treats paired games as independent: pairing makes a pair's
  two results negatively correlated, so its se is then conservative.
* **Sequential tests** (`zero.h2h.sequential`): `--sprt E0,E1` plays batches of `--batch-games` (default
  `--games`) until the log-likelihood ratio of H1 (A is E1 Elo better) against H0 (E0) leaves Wald's
  bounds (ln(beta / (1 - alpha)), ln((1 - beta) / alpha)), `--alpha` / `--beta` 0.05, or `--max-games`
  (default 4 x `--games`) are played: decision 'H1' | 'H0' | 'undecided'. The LLR is the generalized
  SPRT's normal approximation over the per-pair scores (or per-game scores unpaired): n (s1 - s0)
  (2 mean - s0 - s1) / (2 var), s the logistic expected scores - Fishtest's pentanomial GSPRT, the
  standard for paired, seat-swapped games. `--target-se S`: batches until `se` <= S. Batch b is seeded
  seed + 7919 b (batch 0 is the plain match), gids continue across batches, the output is over every
  game plus `batches` (cumulative score, se, llr), `stop`, `sprt`; a progress line per batch goes to
  stderr.
* **Why an SPRT** rather than "extend by 600 games when within 1 se of the bar": the fixed bar's error
  rates are known only for one look (a second look inflates the false positives, by an amount nobody
  states), and 600 games cannot resolve the effects that matter (20.7: a +20 Elo fork passes the
  600-game bar about a quarter of the time unpaired, half the time paired). The SPRT states its error
  rates (alpha, beta at E0, E1; checking after batches only makes it slightly conservative), stops early
  on clear results (a lopsided fork is decided in its first batch) and spends its games where the effect
  is small. `--target-se` is there for a plain measurement (an Elo with a stated precision), not for a
  verdict.

### 20.4 Per-side simulations

`zero.h2h --a-sims` / `--b-sims` (default `--sims`), `match(..., sims_a=, sims_b=)` / `sequential(...)`:
each side searches with its own budget (`A A --a-sims 128 --b-sims 32`: what more search buys one
checkpoint); the output names `sims_a`, `sims_b` (MTG Zero's quick checks monkeypatched `games.nn_seat`
for it before).

### 20.5 The A/B driver (MTG Zero's `runs/ab-r3/ab_driver.py`)

Its `h2h()` builds `python -m mtgz.az.h2h A B --games N --sims 32 --spec S --device cuda --threads 8
--seed 7 --select cq ...` and adopts the fork at `score >= 0.5 + 2 se` with `score = a_score`, `se =
se`. With `--paired` alone that rule reads the paired se; with `--sprt` the verdict is `sprt.decision ==
'H1'` (the driver's own rule should then be replaced by it). MTG Zero's ENGINE.md 73.6 gives the exact
lines. h2h's `--chance-k` defaults to 2 where a run's evaluation plays the run's `chance_k` (MTG Zero's
Delver line: 0) - pass it to measure what the run plays (and, on MTG Zero, to avoid the engine panics
of 20.7).

### 20.6 Off by default; tests and checks

Off (`elo_anchors` '', `elo_paired` 0, `heldout_shards` '', the driver's `paired` False, h2h without the
new flags): the runs of before, byte for byte - no seat, stream or file changes; config.json gains the
eight keys; zero.h2h's output gains `elo_se` only.

Tests: Rust `driver::tests::paired_games_share_their_streams` (two seats that search alike, swapped:
each pair one game twice; unpaired no; pair k = the unpaired game k's streams; an odd last game; the
golden match unchanged). `tests/test_measure.py` (11): the binding's `paired`; h2h's pairs, se and
`se_games`, the same network's pairs all split with se 0, per-side sims, odd games refused; the SPRT's
LLR and bounds; `sequential` (decided in the first batch = the plain match; undecided to `max_games`
with unique gids; `target_se`; the command line and its errors); `fit_fixed` = `fit` with one anchor,
recovers ratings with two, the calibration's pairs only; `parse_anchors`; a toy run with two anchors,
paired Elo matches, the yardstick's games dropped and a held-out set (calibration, anchor games of every
evaluated checkpoint, `anchored.json` = the metrics, every iteration's `heldout`, the log lines) and then
another file under a fixed label refused (78); missing anchors / sets refused (78), odd paired games and
duplicate labels refused; anchors on a `local` remote (shipped under their labels, the mirror's config,
`evals.jsonl`'s `elo_anchored` = `anchored.json`, the dashboard's merge); the defaults add nothing;
`heldout.make` / `losses` against `distill.outputs` by hand, the encoding and game guards, value-only
rows apart, the command line.

Checks: see 20.8.

### 20.7 What it measured on MTG Zero's Delver line (its ENGINE.md 73)

* **Held-out losses** (set: 8,192 rows of `runs/r3-2b-beliefsearch/data/iter_0191`, generated by
  checkpoint 190; 2.4 s a checkpoint on the CPU): the decision rows' value CE falls 0.393
  (checkpoint 83) -> 0.384 (188), -0.009 per 100 iterations over the checkpoints that never saw the
  rows, a checkpoint-to-checkpoint scatter of ~0.003; the value-only rows' does not fall at all. The
  six checkpoints that trained on the rows fall to 0.319 (196): the value head memorizes the outcomes of
  the games it trained on (0.065 nats in six iterations, against 0.009 of real progress in a hundred) -
  a set is only valid for checkpoints that never trained on it. Policy CE falls 0.522 -> 0.360 towards
  checkpoint 190, the search that recorded the targets: agreement, not strength.
* **Paired games** (checkpoints 196 vs 188, 32 simulations, the line's evaluation search, `chance_k` 0;
  five paired batches of 200 games, four on the Mac, one on the desktop): 73% of the 500 pairs split
  (each side won the same seat's deal), and **var(pair score) = 0.53 x var(game) / 2** (95% bootstrap
  0.46-0.61): a paired h2h of N games is as precise as an unpaired one of 1.9 N; 600 games: se 0.020 ->
  0.015, the 2-se bar 0.541 -> 0.530. Simulated with the four Mac batches' 0.54 (8,000 tests a row), the
  old readout (600 unpaired games, the 2-se bar) adopts a real +20 Elo fork 28% of the time and a null
  2%; paired + SPRT(0, 20) with batches of 600 and at most 2,400 games adopts +20 90% of the time (1,343
  games on average), a null 2% (1,354 games, 9% undecided), +30 100% (841 games).
* **Engine panics** (MTG Zero's, not this section's): three of three 600-game h2h runs of the line's
  checkpoints 196 and 188 at h2h's default `chance_k` 2 died on a game's panic in a search
  (deterministic: one reproduces on the desktop), six 200-game batches at `chance_k` 0 (the run's
  evaluation) none. The driver fails a whole match on one game's panic, so a sequential test multiplies
  the exposure: pass the run's `--chance-k` (its ENGINE.md 73.8).
* **Fixed anchors** are not live yet: the main line's relaunch with them (and the held-out set, no
  yardstick games, paired Elo matches) is its ENGINE.md 73.7; checked end to end on the Mac with a
  smoke run (20.8).

### 20.8 Checks

Tier "opt-in core feature with byte-identical defaults" (desktop 2026-10-30, CPU only,
`CUDA_VISIBLE_DEVICES=-1`, below normal priority; the orchestrator's checks on the GPU, the Mac at
nice 19 for the measurements), on ec57c15 (the code; this commit adds documentation only):

* `cargo test --release`: 89 passed (zero_search; webnet 3).
* `pytest tests`: 302 passed, 3 skipped (CUDA), 13:54 on the loaded CPU.
* Toy-smoke runs byte for byte against 85459f5 (its own worktree and target, `ZERO_INFLIGHT=1`, 3-4
  iterations, seed 3), four configurations - classic with `ext:toyheur`, tokens with `elo_together`,
  rolling with 'cq' targets and the cq2 rule, the Delver line's shape (rolling, 'cq' 0.3, the cq4 rule,
  `elo_offsets` 1,2, no random games, value_q 0.6, window cache, prefetch, tokens, margin_w 0.2,
  chance_k 2) - each without the new settings and with all eight given at their defaults: every
  shard, checkpoint (zip members), optimizer state, Elo file (`updated` aside), models/index.json and
  the metrics without timings identical; config.json only gains the eight keys.
* `check_checkpoint_compat --package zero` (random, uniform, a toy checkpoint): identical.
* MTG Zero (branch `measure`, its ENGINE.md 73.9): `tools/gate.py optin-core` PASSED twice (on main
  e08bade and after the rebase onto eedbf14: core tests 92, checkpoint_compat byte identity, `--test az`,
  `--test hidden`, Glass Road c1f56dc / Caverna a6fad73 / Kingdom Builder 4202a4a build against this
  core: GREEN), `check_checkpoint_compat` with the Delver line's checkpoint 196 identical, pytest 62
  passed; the anchors end to end on the Mac (`eval_remote mac`).

## 21. A lookahead player: the exhaustive lookahead per information set as a seat (MTG Zero's Delver mirror, 2026-11-01)

The design, the semantics and the API are docs/search.md 3.9; this section is the run-level view: how
a match fields a lookahead seat, what it costs on the desktop, what it measured.

**A match.** `zero.h2h A B --a-lookahead SETTINGS` (or `--b-lookahead`, `--lookahead` for both) makes a side
a lookahead seat: its seat dict is its checkpoint's run's evaluation seat (`run_rule`: rule, leaf view,
root belief) with the key `lookahead` added - the belief deals the lookahead's worlds, the MCTS settings
play its final fallback. `--conc` (games in flight; default every game, at most 1024) and `--groups`
(default 2) shape the driver; `--max-rows` (default 1024 with a lookahead side) the evaluators' pieces
(`NetEvaluator(max_rows=)`: a token network's CUDA graphs answer them two in flight). The output's
`look` sums the lookahead counters (decisions, nodes, leaf / prior / value rows, the shared values,
rounds, fallbacks per stage, the most live member games), `perf` where the time went (the drivers'
search, assembly and wait seconds; each evaluator's seconds, calls, rows, pieces).

**The host's copies.** The lookahead's batches hold up to `rows` per game in flight - thousands of
37,620-float rows for MTG, ~150 of them non-zero - and the host's copies of the dense rows bound them (the
CUDA graphs' own limit from ~192 rows, 10.6): the driver assembling a batch into fresh memory (page
faults: ~70-120 us a row on the loaded desktop, 59% of a 4-ply match's wall time waiting for it) and the
graphs' pinned copy. Three opt-in remedies: **sparse batches** (`sparse=True`, docs/search.md 2.2: each
row's non-zeros; `TokenGraphRunner.launch_sparse` builds the rows on the device; h2h `--sparse`, on with a
lookahead side) - a 32-game 4-ply match 165 s -> 89 s, its batches' assembly 85 s -> 1.5 s, the same
games; **4 driver groups** (h2h `--groups 4`: 89 s -> 65 s); and for dense token batches of >= 512 rows
the graphs' host copy split over 4 threads (`COPY_PAR_ROWS`, `COPY_THREADS`; the same bytes, answers bit
for bit; MCTS batches of tens to a few hundred rows unaffected). MCTS self-play and evaluation could take
sparse batches too (not measured here).

**Measured** (MTG Zero's Delver mirror, its ENGINE.md 80; the same network both sides, MCTS at 32 simulations
with the run's evaluation rule, paired games): the pure lookahead player loses - 4 plies 0.315 +- 0.030 (best
reply) and 0.390 +- 0.034 (policy-weighted) over 100 pairs, 6 plies 0.470 +- 0.046 and 0.370 +- 0.045 over 50 -:
its max over many noisy leaf values is optimistic (another session's teacher experiment saw the same where
the lookahead disagreed with MCTS). The guarded player (`guard` 0.05, 16 worlds; docs/search.md 3.9) changes
~2% of the MCTS moves and is even: 0.515 +- 0.015 over 300 pairs at 4 plies, 0.460 +- 0.035 over 50 at 6.
Throughput with 64 games in flight on the desktop's RTX 5090: 4 / 5 / 6 plies 104 / 65 / 40 lookahead
decisions a second (83k / 143k / 242k nodes, 29k / 48k / 79k network rows a second; one game alone 17 / 12.5
/ 5.6); a 6-ply match ~9 s a game with 64 in flight.

**Checks** (tier "opt-in core feature with byte-identical defaults"): cargo test 102 (13 lookahead tests) +
webnet 3; pytest 316 passed / 9 skipped (CPU; the CUDA tests separately); check_checkpoint_compat `--package
zero` identical; toy-smoke runs byte for byte against c6b959b in two configurations (classic; rolling with
'cq' 0.3 and the cq2 rule); MTG Zero's `tools/gate.py optin-core` PASSED (its ENGINE.md 80.8: digests
unchanged, check_checkpoint_compat with the Delver line's iteration 475 identical, Glass Road / Caverna /
Kingdom Builder build against this core).

## 24. Branched self-play games and archive starts (`zero/forks.py`; MTG Zero's Delver mirror, 2026-11-03)

(Sections 22 and 23 are taken by open branches: `feat/heavy-decisions`, `feat/light-decisions`, `feat/temp-late`.)

### 24.1 Why

MTG Zero's Delver mirror review 5 (its hub notes, `delver_review_5_2026-11-03.md` section 5, step 2): the
measured bottleneck is **which positions the value head sees** - its own sharp, self-confirming lines, one
outcome per ~90 rows. At 71 decisions from human games that a strong player calls mistakes, three
generations re-searched at 1,024 simulations rate the played move as their own with a gap of 0.000 at 62:
the value head rates the lines it never plays like the lines it plays. Its exploration A/B (section 21 of
`feat/explore`: gated single deviations, targets unchanged) and late sampling re-weighted options inside
the games; neither put *new positions with real outcomes* into the data. KataGo's answer is side games
(a share of games branch at a random move into a game of their own); Go-Exploit's is starting games from
an archive of states. This section is both, as opt-in run settings.

### 24.2 What it does

* **Branched games** (`selfplay_branch` = p; driver: docs/search.md 2.11). A share p of the games each
  spawn one branch from a uniformly drawn decision of theirs (a recording seat's searched decision, not
  minor). When the game ends, its slot plays the branch: 1..`selfplay_branch_moves` (uniform) deviation
  moves sampled from the network's priors at temperature `selfplay_branch_temp` - the root's evaluation
  alone, no search: **they are recorded nowhere** (no policy target, no value target) - then normal
  self-play to the end; every row of the branch gets **the branch's own z**. The game itself is untouched:
  its moves, its rows, its streams (a branch reads only its own stream), so none of its rows - the branch
  point's included - is recorded twice. `selfplay_branch_differ` 1: the first deviation is never the
  game's own move at the branch point (without it, a sharp policy at temperature 2 often samples the
  game's own move first, and a one-move branch then re-plays its game's line from there: 24.6).
* **No row needs a masked value target.** The exploration feature (`feat/explore`) masks z := q on the rows
  before its forced deviation, because there the deviation is inside the game whose outcome those rows
  get. Here the deviation is in the branch only, and recorded nowhere; the game keeps its own outcome; the
  branch's rows all come after its deviations, and an archive start's rows after its start. So there is
  nothing to mask, and the shards carry no new column.
* **Archive starts** (`selfplay_archive` = a file, `selfplay_archive_p` = the share). A game starts from an
  archived position instead of a new deal: an entry's `patch` merged into the run's spec (MTG Zero: its
  `start` block - the source game's seed and root actions, the library orders nobody knows dealt again
  from the game's own seed: MTG Zero's ENGINE.md 76, 87). The file, `zero-archive/1` = {format, game,
  base_spec, entries: [{id, kind, patch, ...}], ...}, is the game's own tool's (MTG Zero's
  `tools/delver_archive.py` replays and checks every position and lists what it skips); `zero/forks.py`
  refuses another game, a `base_spec` other than the run's, duplicate ids, an empty or malformed entry,
  and the driver starts every entry once when it is made. Its sha256 (of the entries) is logged at the
  run's start and in every iteration's `selfplay.forks.archive_sha`.

**Quota and compute.** A branch is a finished game of its own: it counts toward the iteration's `games`
(local, remote, shared queue alike), so an arm with branching plays the same number of games an iteration
as one without - but a branch starts mid-game and an archive start at turn 3-15, so the arm plays fewer
searched decisions and records fewer rows an iteration (24.6: MTG Zero's numbers). An archive start is one
of the iteration's games. The shared queue (5.3) and the local linger count setups by the games that
started from one (`forks.setups`: a branch is the second row of its game's setup), else they would queue
one more setup per branch, every iteration.

**Metrics** (`selfplay.forks`, only when one is on): the iteration's accepted games, samples and searched
decisions by origin (game / branch / archive), their shares, samples per game, the branches' means
(deviation moves drawn and played, `same_first`, the branch point's place in its game `at_frac`, of
archive games), archive starts by kind, the archive's sha, and the remote's own counts (`remote`); one log
line per iteration (`iter i forks: ...`).

**Remote self-play** (5.3): the workers branch and start from the archive exactly as this machine - their
config.json is the run's, and the archive file is shipped into the mirror as `archive.json` (the config's
path rewritten to it) at each session start. A worker's hello reports its `forks.fingerprint` (the branch
settings, the archive's p, entry count and sha); a worker whose checkout plays them otherwise (an older
checkout reports none) is refused, retried in 10 minutes, and the run plays on alone - never silently
without them. Their accepted games' tallies come back in their end lines.

### 24.3 Off by default

`selfplay_branch` 0 and `selfplay_archive` '' (the defaults): the driver gets neither (`DriverCfg::branch`
and `starts` None), no metric, no new key in a result row or a stats dict but the three zero counters,
the run of before byte for byte. Refused: a share outside [0, 1] / (0, 1], moves < 1, temperature <= 0,
`selfplay_branch` without rolling self-play (a classic iteration's list of games would get its branches
on top of its quota), co-training runs (and a `resolver_cfg` with either), the self-play league.

### 24.4 Costs

A game that drew a branch clones its state at the reservoir's picks (H(n) ~ ln n + 0.58 clones over n
eligible decisions: ~6 for MTG's ~300) and holds one copy until it ends; a deviation move costs one network
row. The archive's entries are parsed and started once per driver (MTG Zero's 321 entries: a fraction of a
second). Nothing else: the games, search and training of before.

### 24.5 Tests and checks

`cargo test` 108 (the driver's 6 tests of docs/search.md 2.11); `tests/test_forks.py` (15: the bindings -
rows, stats, the settings refused, `differ` -, the run settings, the archive file's checks, a rolling toy
run with both on: every iteration exactly its quota with branches and archive starts among them, the
metrics; off has no metrics; remote self-play with the remote 'local': the shipped archive, exact shared
quotas with branches counted as games, the remote's tally; an older worker refused by its hello). The full
pytest (CPU, beside the live Delver line): 334 passed, 9 skipped, 1 failed - `test_cotrain_pause_mid_iteration_replays_it`,
whose run hung on a print: the new settings pushed the co-training run's start-up lines past a Windows
pipe's 4 KB buffer, which the test read only at the end (the exploration branch met the same: its fix,
`feat/explore` 7985add - the PAUSE tests' output to a file -, cherry-picked here); the PAUSE tests then pass.

### 24.6 What it measured on MTG Zero's Delver mirror (its ENGINE.md 87)

2026-11-03, the desktop's CPU at below-normal priority beside the live Delver line (MTG Zero's ENGINE.md 87
has the tables). **The archive** (its `tools/delver_archive.py`): 321 positions - 76 canary-library spots,
242 Play-tab positions (turns 3-8 of 50 human games), 3 Play reports - every one rebuilt through MTG
Zero's `start` block equal to its source replay; 4 Play games skipped and listed (played on an older
engine build). **At the driver** (MTG Zero's `tools/forks_check.py`, its line's self-play seat at 16
simulations, checkpoint 808): branches 21.1% +- 3.5 of 128 games (p 0.2), deviation moves 1/2/3 drawn
11/3/13 and all played, every branch row's z its own outcome, none beyond its searched decisions after the
deviations; archive starts 10.9% +- 2.7 (p 0.1). **Without `differ` the first deviation was the game's own
move in 79% of the branches** (the network's top move at temperature 2), and those branches' rows re-played
their games' states; with it 0%. A branch records ~39 rows against a game's ~63, an archive game ~53: at
p 0.2 / 0.1 the arm records ~8% fewer rows and searches ~8% fewer decisions per iteration at the same games.
With a batch-invariant evaluator every game equals its twin without branching (44 of 44, 87 of 87); with
the CPU token network the twins differ (two runs without branching are identical, 32 of 32): its answers
depend on a batch's other rows (padding), which the branches change. **End to end** (`mtgz.az.run`,
the Delver preset with a 32 x 1 token network on CPU, the line's flags - sparse batches and window, window
cache, train_static, local and remote linger -, both features on, remote self-play on the remote 'local'
with one worker): 4 iterations of 192 games, every one exactly its quota (this machine 115 / 93 / 73 / 58,
the worker 77 / 99 / 119 / 134), the archive shipped into the mirror byte for byte, no worker refused, the
iterations' `forks` = their games and samples (local + the remote's tally); 77% games, 15% branches, 8%
archive starts over the run (branches lag their games by up to a phase: 17.6% per game that can branch).

**Off**: MTG Zero's `check_checkpoint_compat` against main 37d6574 (zero-core cd2f43f) identical - random
network on the cube spec, and checkpoint 808 on the Delver spec at 16 simulations (1,161 samples, records
cc72319a5d5ca31d / 5a26e15c983461e8); its engine digest f6543eb2b1b8752f unchanged.

## 25. The game's evaluation cache (`eval_cache`; MTG Zero's Delver mirror, 2026-11-03)

The semantics are docs/search.md 3.12; this section is the run-level view.

**Why.** A third of the network rows of MTG Zero's Delver self-play repeat a row the same game sent earlier
(measured below), and self-play's time is mostly the network's: the Delver line's gen 960 spent 96 of its
128 s of self-play in the evaluator. A row the game already sent need not be sent again.

**The setting.** `eval_cache` N (default 0 = off): every network seat of the run - self-play's
(`Run.nn_seat(selfplay=True)`, also the remote's and the local linger's) and the evaluations' - keeps its
game's answers, at most N per game in flight, and answers a repeated row from them. Off, the seat dict has
no key and the run plays, records and trains as before, byte for byte. With it on, the answers are the
network's answers to the same rows computed in another batch (equal up to its float rounding at another
batch shape); games finish in another order, so a run's batches, shards and rolling phases differ from the
run without it at the float-rounding level, never in what the search computes.

**Metrics.** Self-play's `eval_cache` (when on): `hits` (rows answered from a cache by this machine's
driver), `leaves` (leaves expanded at once), `partial` (leaves that sent one row and took the other from the
cache), `clears` (caches emptied because full), `share` = hits / (hits + rows sent).

**Costs.** Per row, one 128-bit hash of its features (`obs_hash` skips all-zero blocks of 64 floats) and a
hash-map lookup, on the driver's search threads. Memory: N answers per game in flight (~100 bytes each plus
the priors): N 8,192 at 768 games in flight is at most ~0.7 GB.

**Measured.**

* *How many rows repeat* (2026-11-03; MTG Zero's Delver line r3-2b-float550, gen 960, its self-play seat
  exactly - 64 / 16 simulations, p_full 0.25, 4 minor, root noise, root belief, root-view leaf values,
  redeterminized worlds - 48 games on the CPU, every leaf batch's rows hashed before the network answered:
  E:/mtg-zero-scratch/state-dedup/): 520,304 rows; repeats of a row from earlier in the same game 33.8% (one
  search 25.2%): value-only 62.6%, leaf 19.0%, belief 100%; none across games. By decider: the root decider's
  own decisions 34.0%, the other seat's 3.8% - hidden information (each simulation deals the other seat a new
  hand). Equal features always had equal options. The permanents in a canonical order (cross-references
  relabelled) add 0.6 points.
* *Throughput* (2026-11-03, the desktop's RTX 5090 alone - the run stopped -, MTG Zero main 37d6574 on this core,
  E:/mtg-zero-scratch/state-dedup/gpu_timing.py): the Delver line's self-play seat with its gen 1000, a rolling
  driver as the run's local one (768 games in flight, 8 threads, 2 groups, sparse batches, token CUDA graphs),
  256 games to warm up, then 640 timed; off / on alternately, each pair on one seed:

  | eval_cache | games/s | simulations/s | rows sent/s | rows cached | leaves at once / in part | avg batch |
  |---|---:|---:|---:|---:|---:|---:|
  | 0 | 10.67, 10.44 | 73.7k, 72.6k | 108.7k, 107.1k | - | - | 577 |
  | 8192 | **14.29, 14.00** | 99.5k, 97.8k | 97.2k, 95.4k | 34.0%, 34.1% | 0.81M / 1.38M per 640 games | 465 |

  +34% games a second (10.55 -> 14.15); the cache was emptied ~0.3 times a game (some games pass 8,192 answers).
* *The toy* (`driver::tests`): ToyGame self-play at 32 simulations 21,630 rows -> 15,750 sent (27% cached);
  with root-view values 33,871 -> 27,631 (value-only 11,446 -> 10,657).

**Checks.** cargo test 106 + webnet 3 on the desktop (the four eval-cache tests, docs/search.md 3.12; the
driver's golden records and matches unchanged with the cache off); on the MacBook the ten golden tests fail
alike on main 96f1566 and on this branch, with the same digests (the goldens are the desktop's floats);
pytest `test_loop.py::test_eval_cache_setting`.

## 26. A prior exponent in the 'cq' target (`cq_prior_w`; MTG Zero's Delver mirror, 2026-11-04)

The semantics are docs/search.md 3.13; this section is the run-level view.

### 26.1 Why

With `policy_target` 'cq' the recorded target is `softmax(ln P0 + sigma completedQ)` with P0 the playing
network's own prior, and the network fits its targets: generation after generation the policy's logits add up
every advantage a search measured and forget none. On MTG Zero's Delver line the targets have drifted to one-hot
(gen 859: 54.6% of the recorded decisions with max-share > 0.99, 38.7% > 0.999; the targets' mean normalised
entropy 0.35 -> 0.20 between gens 40 and 1096) while the habits human players beat it with stayed flat for
hundreds of generations: an option PUCT stopped visiting is never measured again. A prior exponent beta < 1
gives the generations' policy a bounded fixed point `exp(sigma Q / (1 - beta))` and lets a dead option's prior
recover by the factor beta a generation until the search measures it again (Fable's review of 2026-11-04, its
rank 1: `E:/mtg-zero-scratch/fable-ideas/report.md` 2.1).

### 26.2 The setting

`cq_prior_w` = beta (default 1.0 = off): self-play's recorded 'cq' targets are `softmax(beta ln P0 + sigma
completedQ)`. It goes to the self-play seat alone (`Run.nn_seat(selfplay=True)`: this machine's, the local
linger's and the remote workers'), never to an evaluation, and inside the seat to the recorded target alone: the
moves (PUCT, root noise, temperature, `selfplay_select`, the Gumbel root), q and z are those of beta 1. It needs
`policy_target` 'cq' and a value in (0, 1] (`resolve_config` / `check_eval_settings` exit otherwise, also for
`resolver_cfg`). Off, the seat dict has no key and the run records and trains as before, byte for byte.

**What changes with it on.** Only the `pi` of the recorded rows, so a run's first generation with it plays
exactly the games it would have played; from the next one on its network differs (it trained on softer targets)
and with it everything the network touches. The evaluation seats' rule and Elo pool are unchanged, so an A/B's
arms share them. The readouts that show it: self-play's `target` (`h_norm` up, `max_share` down, from the first
generation) and `train.entropy` (the policy's own entropy, a generation later). A remote worker must run a
checkout with this core (docs/search.md 3.13: an older one would record beta-1 targets in silence; MTG Zero's Mac
runs the run's commit).

**Costs.** None measurable: one `ln` per option of each recorded decision.

### 26.3 Measured: the targets at beta 1, 0.8 and 0.6 (MTG Zero's Delver line, CPU, 2026-11-04)

The Delver line's checkpoint 1000 (r3-2b-float550) with its exact self-play seat (config.json: 64 / 16
simulations, p_full 0.25, 4 minor, root noise, temp_moves 60, cq c_scale 0.3, root belief, root-view leaf values,
value-only samples 0.5, plus the run's eval_cache 8192), 24 games on the desktop's CPU at idle priority beside the
live run (2 search + 2 torch threads, ~210 s per beta), recorded (`take_samples`), once per beta with the same
driver seed; an evaluator wrapper kept every leaf row's priors, so each recorded decision's P0 is known (all
1,475). Script and data: `E:/mtg-zero-scratch/ideas-1/` (`cq_prior_probe.py`, `probe_24g.json`,
`probe_24g_beta*.npz`, `analyze.py`).

* **The games are identical** at the three betas: every result and every sample field but `pi` bit for bit
  (2,235 samples, 1,475 decisions with 2+ options; 167,483 rows sent, 83,569 answered from the cache, each run).
* **pi_beta = pi_1 P0^(beta - 1), normalised**: rebuilt offline from the beta-1 targets and the captured priors it
  matches the driver's targets at 0.8 and 0.6 to 5e-8 (only the prior's term differs, as designed).

| recorded targets (1,475 decisions) | P0 (the prior) | beta 1.0 (today) | 0.9 * | 0.8 | 0.7 * | 0.6 | 0.5 * |
|---|---:|---:|---:|---:|---:|---:|---:|
| max-share > 0.9 | 75.1% | 76.7% | 74.5% | 71.4% | 69.1% | 65.4% | 59.7% |
| max-share > 0.99 | 51.9% | **56.9%** | 53.2% | **48.6%** | 43.6% | **38.2%** | 31.3% |
| max-share > 0.999 | 34.6% | 40.3% | 36.0% | 31.3% | 28.0% | 24.3% | 20.3% |
| mean max-share | 0.916 | 0.919 | | 0.902 | | 0.876 | |
| mean normalised entropy | 0.214 | 0.202 | 0.221 | 0.244 | 0.273 | 0.307 | 0.350 |
| normalised entropy < 0.05 | 49.5% | 54.7% | 50.4% | 45.4% | 40.7% | 34.8% | 29.7% |
| KL(pi \|\| P0), mean (median) | 0 | 0.021 (0.00004) | 0.025 | 0.033 (0.003) | 0.046 | 0.069 (0.020) | 0.109 |
| argmax pi = argmax P0 | | 97.8% | | 97.2% | | 96.4% | |

\* rebuilt offline from the beta-1 targets and P0 (exact up to f32 rounding: see the check above).

The beta-1 row reproduces the line's own readouts (gen 859's offline screen: 75.7% / 54.6% / 38.7%, entropy 0.20;
`metrics.jsonl` max-share 0.92). **At beta 1 the target is sharper than the prior** (56.9% vs 51.9% above 0.99):
the ratchet, measured within one generation. beta 0.9 still sharpens (53.2%); **0.8 is about where the target
stops sharpening the prior** (48.6%, mean max-share 0.902 vs the prior's 0.916); 0.6 softens it clearly.

*Projected, not measured* (`analyze.py`, assuming the network fits its targets exactly and every option's
completed Q stays what this search measured - an option whose prior rises would in fact be visited and its Q
re-measured, so the projection overstates the softening of options no search visited): the share above 0.99
after 1 / 5 / 10 generations and at the fixed point is 57 / 65 / 69% at beta 1 (still climbing), 49 / 34 / 30 /
31% at 0.8, 38 / 20 / 20 / 20% at 0.6; at the fixed point the mean entropy is 0.52 at 0.8 and 0.62 at 0.6. The
relaxation is fast (most of it within ~5 generations at 0.8), well inside a 40-generation A/B.

### 26.4 The A/B (planned, not run)

Recommended first arm: **beta 0.8** - it passes the review's kill test (a) (fewer than 50% of the targets above
0.99 at the chosen beta: 48.6%), it is the first beta at which the targets stop out-sharpening the prior, and
its fixed point keeps a 1-pp systematic Q gap at ~1.5 logits (docs/search.md 3.13) while the self-play policy
softens least. 0.6 is the second arm if 0.8's lever proves too weak. The fork + control pattern of section 24
(MTG Zero's `E:/mtg-zero-scratch/forked-games/launch.sh`), arms in turn, both from the line's checkpoint K,
40 generations each at the line's flags: the fork `--cq-prior-w 0.8` on a build pinning this core, the control
the line continued on its own pin. Endpoints at equal generations: self-play `target.h_norm` / `max_share`
and `train.entropy` (the lever), paired GPU h2h fork N vs control N at K+20 and K+40 (1,200 games, 32 sims,
the evaluation seat), the blunder sidecar's flat fingerprints (`fp_daze_tapout`, `fp_fow_pitch_fow`,
`fp_fetch_now`, and `tam_flip_clean_pre`), the held-out canary half, fresh-row value CE at K+40. Stop rules:
the fork's `target.h_norm` at K+5 less than 0.04 above the control's (no compounding beyond the first
generation's +0.043) = too weak, rerun at 0.6; h2h <= -20 Elo at K+20 = too soft; entropy up and no fingerprint
moved >= 5 points by K+40 with the h2h within noise = the ratchet was not the freeze (stop); a fingerprint moved
>= 5 points with the h2h within noise = adopt (a relaunch flag; the evaluation pools are unchanged). The
plan in full: `E:/mtg-zero-scratch/ideas-1/AB_PLAN.md`.

### 26.5 Checks

cargo test 114 + webnet 3 on the desktop (`tree::tests::completed_q_prior_w_by_hand`,
`driver::tests::cq_prior_w_changes_only_the_recorded_targets`: prior_w 1.0 reproduces aae4ca2's 'cq' record
digests; the driver's other golden records and matches unchanged); pytest `tests/test_cq_prior_w.py` (7),
`test_policy_target.py` + `test_mcts.py` (17), `test_gumbel.py` + `test_bindings.py` (26);
`tools/check_checkpoint_compat.py --package zero --net random --game toy` against aae4ca2: identical; the toy
loop (`--preset toy-smoke --policy-target cq --cq-prior-w 0.6` vs 1.0): the first iteration's 385 samples
alike, target `h_norm` 0.900 vs 0.878, then the networks diverge; MTG Zero main 619f1f2 builds against this
core and plays the identical games of 26.3, and `check_checkpoint_compat.py --package mtgz` with the Delver
line's checkpoint 1000 and its spec (8 games, 32 sims, 'visits' and 'cq' records, a match vs random), 619f1f2 on
aae4ca2 vs 619f1f2 on this core: identical.

## 27. Self-play resignation with a no-resign share (`selfplay_resign_t`; MTG Zero's Delver mirror, 2026-11-04)

The driver's semantics are docs/search.md 3.14; this section is the run-level view.

### 27.1 Why

MTG Zero's Delver line is data-rate-limited (its value head improves with games and with nothing else tried)
and, since the evaluation cache, its self-play phase is bound by the CPU's searches (Fable's review of
2026-11-04, `E:/mtg-zero-scratch/fable-ideas/report.md` 1.1, 2.2). A third of those searches come after the
game is decided: 36.5% of its recorded decision rows have the decider's root win estimate outside
(0.05, 0.95), and the outcome agrees in 98.9% of them (27.6). Their policy targets come from searches with
no gradient (utility = P(win) only) and their value targets are the ones the value head already predicts.
Ending a game once the search has called it decided for a while buys games per hour at the same compute.
KataGo's form: resign after a run of decided searches, and keep a share of games that never resign - they
keep the value head seeing decided positions and are the live measurement of how often the rule is wrong.

### 27.2 What it does

* **The settings.** `selfplay_resign_t` t (0 = off), `selfplay_resign_k` k, `selfplay_resign_noresign` (the
  no-resign share) become the self-play seat's keys `resign_t`, `resign_k`, `resign_noresign`
  (`Run.nn_seat(selfplay=True)`, `resign.seat_keys`): this machine's rolling or classic driver, the shared
  queue's driver, the local linger's (the same driver) and every remote worker's (they build the seat from
  the run's config.json) - one rule everywhere.
* **The rule** (docs/search.md 3.14): a game ends when k consecutive **full** searches of its self-play seat
  (the searches that make its records; fast and minor searches neither count nor reset) found **one seat's
  root win estimate >= 1 - t** - the record's q, by absolute seat; the same favoured seat each time. It is
  scored as that seat's win before the firing decision's move; every record so far is kept with z = that
  outcome. A share of the games - `noresign_pick(seed, gid, share)`, a hash, so no game changes - never
  resigns: each reports where the rule would have fired and whether the favoured seat then won.
* **Who never resigns**: evaluation seats (the evaluations, Elo matches, anchors, `zero.h2h`, a game's Play
  tab - `games.nn_seat` refuses `resign_t` on a seat that is not self-play, the driver on one that records
  nothing), and every tool that builds its own seats (MTG Zero's blunder sidecar plays full games).
* **Quotas**: a resigned game is one finished game - one result row - for the rolling quota, the shared
  queue's quota (5.3), the local linger's held games and a remote's `fin` / accept: an iteration plays its
  `games` games exactly as before, each shorter. Branched games (24) compose: a resigned game spawns its
  branch, a branch resigns by its own count.
* **Metrics** (`selfplay.resign`, only when on; the iteration's accepted games, this machine's and the
  remote's): `games`, `resigned` / `resigned_share`, `noresign`, `would` (no-resign games where the rule
  fired) / `would_share`, `wrong` (of those, the favoured seat did not win: a loss or a draw) / `wrong_rate` -
  **the live wrong-resignation rate** -, `rows_saved` and `decisions_saved` (the share of the no-resign games'
  samples and searched decisions after the point where they would have resigned: what resigning saves per
  game), `samples_per_game` {resigned, noresign}, `remote` {games, resigned, noresign}. A log line per
  iteration (`iter i resign: ...`) and the settings at the run's start (`self-play resignation
  (zero/resign.py): t=... k=... noresign=...`).
* **Remote self-play** (5.3): the workers resign exactly as this machine. A worker's hello reports its
  `resign.fingerprint` (t, k, share); one whose checkout plays otherwise - an older checkout reports nothing -
  is refused, retried in 10 minutes, and the run plays on alone: never a worker playing every game out in
  silence. Its accepted games' tally comes back in its end line and joins the iteration's metrics.

### 27.3 Off by default

`selfplay_resign_t` 0 (the default): the seat dict has no key, no row a `resign` field, no metric, no log line:
the run of before byte for byte (27.5). `selfplay_resign_noresign` 1 is a **dry run**: no game resigns, the
games are those of the run without it (nothing reads a stream) and every row reports where it would have -
the rule measured on the live run at no cost. Refused at the start (`resolve_config`): t outside [0, 0.5), k < 1, a
share outside [0, 1]; co-training (`resolver`, and a `resolver_cfg` with it: deferred terminals take their
outcome from outside), the self-play league (its opponent seat copies the self-play seat; one counter per
game), `players` other than 2.

### 27.4 Costs, and what changes in the data

* **Compute**: a few comparisons per full search. Memory: none.
* **The rows**: a resigned game records only its rows up to the firing search, so the window holds fewer
  decided-position rows - the no-resign games' and the rows before each resignation remain. That is the
  intent (they are the rows the value head already predicts at 99% and whose policy targets carry no
  gradient), and the risk: a value head trained on fewer decided positions may drift in them, which shows up
  first as a rising `wrong_rate`.
* **A wrong resignation** gives every row of its game the wrong z (the rule's call is not the played-out
  outcome). In self-play both seats are the network, so a wrong resignation is its own misjudgement of a
  position it then would have lost; the no-resign games measure the rate every iteration.
* **The margin and score of a resigned game** are the position's where it resigned: its records' `sc`
  (`margin()`, the margin head's target at `margin_loss_w`) and its row's `scores` (`avg_score`) are
  mid-game values, and `rounds`, `steps_per_game`, `decisions_per_game` fall with the resignations.
* **Throughput**: an iteration of the same `games` searches fewer decisions. `decisions_saved` is the share
  of a game's searched decisions that resigning saves (averaged over the no-resign games, those that never
  fired counting 0), so a game costs about 1 - (1 - the no-resign share) x `decisions_saved` of one played out,
  and its rows fall by about (1 - the share) x `rows_saved` - per iteration in the metrics.

### 27.5 Tests and checks

`cargo test` 118 + webnet 3 (the driver's six resignation tests, docs/search.md 3.14; every golden record and
match unchanged); `tools/check_checkpoint_compat.py` main (aae4ca2) vs this branch on the toy, random network:
identical (records, results, the match); `tests/test_resign.py` (10: the bindings - a resigned game is its dry
twin up to the firing search with the resignation's z, the draw, the settings refused -, the run settings, a
rolling toy run of 3 x 96 games with exact quotas and the metrics, off has none, remote self-play with the
remote 'local' and the local linger - exact shared quotas with resigned games counted, the remote's tally in the
metrics -, an older worker refused by its hello); `test_bindings.py` + `test_forks.py` 27 passed; `-k 'refused
or older or eval_cache or local_linger'` 16 passed. MTG Zero main 619f1f2 builds against this branch.

### 27.6 What it measured on MTG Zero's Delver mirror

**Offline, the exact rule** (2026-11-04, numpy on the desktop's CPU: `E:/mtg-zero-scratch/ideas-2/offline_resign.py`,
`offline_grid.py`, on the review-5 offline screens' rows of gens 855 and 859 - 522 + 436 whole games, 90,061 rows,
60,232 decision rows; the run's seat records its full searches only, so the decision rows are exactly the
searches the rule counts). Per game, the rule is replayed over its decision rows in order (both seats, the
row's q by absolute seat, the favoured seat the first maximum); a resignation keeps the firing decision's row
and its value-only rows; wrong = the favoured seat did not win (no draws occurred).

| t | k | games resigned | rows saved | wrong (of resigned) | 855 / 859 wrong | games per hour at share 0.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 3 | 76.7% | 21.9% | 2 / 735 = 0.27 +- 0.19% | 0.51 / 0.00% | x1.25 |
| 0.02 | 2 | 87.9% | 28.9% | 7 / 842 = 0.83 +- 0.31% | 0.89 / 0.77% | x1.35 |
| **0.02** | **3** | **82.6%** | **26.4%** | **7 / 791 = 0.88 +- 0.33%** | 0.95 / 0.81% | **x1.31** |
| 0.02 | 5 | 73.6% | 22.1% | 2 / 705 = 0.28 +- 0.20% | 0.27 / 0.30% | x1.25 |
| 0.03 | 3 | 86.1% | 29.8% | 10 / 825 = 1.21 +- 0.38% | 1.13 / 1.31% | x1.37 |
| 0.05 | 3 | 90.6% | 35.2% | 21 / 868 = 2.42 +- 0.52% | 2.75 / 2.02% | x1.46 |
| 0.05 | 5 | 80.9% | 29.7% | 13 / 775 = 1.68 +- 0.46% | 2.14 / 1.13% | x1.37 |
| 0.05 | 8 | 71.2% | 24.3% | 7 / 682 = 1.03 +- 0.39% | 1.09 / 0.95% | x1.28 |
| 0.10 | 5 | 88.9% | 38.9% | 28 / 852 = 3.29 +- 0.61% | 3.85 / 2.60% | x1.54 |

(Rows saved = decision rows saved within 0.2 points; games per hour = 1 / (1 - 0.9 x decisions saved), if a
game's time scales with its searched decisions - decided positions are not cheaper than average, Fable's 2.2.)
**The threshold matters more than the count**: at ~29% saved, t 0.02 / k 2 is wrong half as often as t 0.05 /
k 5. Fable's numbers for t 0.05 / k 5 (32.2% / 31.4% saved, 8 / 5 wrong) counted any recorded row, value-only
rows included (their q is the observer's next search); counting searches only, as the driver does, gives
30.5% / 28.8% saved and 9 / 4 wrong.

**The driver on MTG** (2026-11-04, `E:/mtg-zero-scratch/ideas-2/mtg_smoke.py`: MTG Zero main 619f1f2 built on this
branch, the Delver line's exact self-play seat from checkpoint 1000's config.json - 64 / 16 simulations, p_full
0.25, 4 minor, the cq target, root-view values, the root belief, value-only samples 0.5 - plus `eval_cache` 8192
and resignation; 32 games a run on the desktop's CPU at idle priority beside the live line, CUDA hidden). For
every game the rule replayed over its own records' decision rows (`offline_resign.fire`) fires exactly where the
driver resigned - and in a no-resign game where it said it would have, and never where it never fired -; every
resigned game's z is the favoured seat's win and its samples its `records`; every game's no-resign flag is
`noresign_pick`'s: **0 mismatches in 96 games**.

| run | resigned | no-resign games that would have / wrongly | never fired | seconds |
|---|---:|---:|---:|---:|
| t 0.05, k 5, share 0.25 | 20 | 4 / 0 | 8 | 137 |
| t 0.02, k 3, share 0.25 | 21 | 5 / 0 | 6 | 107 |
| t 0.05, k 5, share 1 (dry run: every game played out) | 0 | 24 / 1 | 8 | 163 |

The dry run's 24 firings would have saved 26.5% of its rows and 26.7% of its searched decisions (offline at
t 0.05, k 5 on gens 855 / 859: 29.7%); its one wrong call favoured seat 1 at decision 122 of 272 and seat 0 won.
137 s against 163 s for the same 32 games at share 0.25 (-16%; the CPU beside the live line, a run's tail
its slowest game's: indicative only, not a throughput measurement). Resigned games averaged 13.1 turns and
62 samples, played-out ones 81 samples.

### 27.7 The recommended A/B and its kill rule

Not launched. **Fork**: the line's checkpoint and settings plus `selfplay_resign_t 0.02 selfplay_resign_k 3
selfplay_resign_noresign 0.1` and `games 2000` (x1.25: a generation then takes about the control's self-play
time and records about its rows - offline 26% fewer rows per game x 0.9 resigning -, so the two arms compare at
equal compute per generation and the fork has seen 1.25x the games); **control**: the line as is (1,600 games).
Both from the same checkpoint, 40 generations (the `E:/mtg-zero-scratch/forked-games/launch.sh` pattern). The
fork's remote workers (the Mac) must run this zero-core: an older checkout is refused by its hello, never
silently playing its games out.
A cheaper first step, if the line is relaunched on this core anyway: a **dry run** (`selfplay_resign_noresign
1` with t 0.02, k 3) for a few generations - the line's games unchanged - measures the rule on the live network
(`selfplay.resign.would_share`, `wrong_rate`, `rows_saved`) before any arm resigns.

Readouts per generation: `selfplay.resign` (wrong_rate, would_share, rows_saved, decisions_saved,
resigned_share), `selfplay.seconds` and `samples` (fork ~ control), fresh-row value CE, the blunder
fingerprints, then the paired GPU h2h at +20 / +40 against the control (within noise or better: resignation
paid for itself; the data-rate argument predicts +).

**Kill rule**: the wrong-resignation rate of the no-resign games (`wrong / would`), pooled over the last 10
generations (~165 would-have-resigned games a generation at 2,000 games x 0.1 x ~0.83: se ~0.25% at 1%),
must stay **below 3%**; stop the fork when it reaches 3%, or when a single generation is above 8% (a gross
failure: ~4 se above 3% at ~165 games), or when the h2h at +40 is -20 or worse. A rate that creeps up generation after generation
while still under 3% is the value head drifting in the positions it no longer trains on: raise the no-resign
share to 0.2 or lower t to 0.01 before it reaches the line.

## 28. Sparse rows without the full-row scans (no setting; MTG Zero's Delver mirror, 2026-11-04)

The driver side is docs/search.md 2.12; this section is the why and what it measured.

**Why.** With the evaluation cache (25) on, the Delver line's self-play phase no longer waits on the network
alone (MTG Zero's r3-2b-float550, gens 1048-1112: the evaluator busy 62-70 s of a 92-102 s phase), and a
search thread spends most of a simulation on its network rows. Measured on 10k Delver leaf rows (below):
per row `encode` ~13-22 us, `LeafBuf::seal`'s float-by-float scan ~11-14 us, `obs_hash` (the cache's key)
~3-4.6 us, a cached row's full zeroing ~2 us - against ~14-22 us for the rest of a simulation (the deal, the
walk, the engine's steps, the expansion and backup). At the run's mix of 1.48 rows encoded per simulation
(0.98 sent, 0.50 answered from the cache) the scans, the key and the zeroing were ~17 us of a simulation's
~50 us of measured work, `encode` ~20 us.

**What changes.** Nothing a run sees: the same batches byte for byte, the same cache keys and hits, so there
is no setting (it would switch between two ways of computing the same bytes). A sparse driver (run setting
`sparse`, search.md 2.2) reads its rows with `SearchGame::encode_sparse` (default: `encode` + the
block-skipping `game::sparse_scan`) and keys its cache by `obs_hash_sparse` of the pairs; a dense driver is
unchanged. A game may override `encode_sparse` to emit its pairs without the dense row (search.md 2.12 has the
contract); MTG Zero does not (ENGINE.md 89: what is left of the scan after this, ~3 us a row, does not pay for
rewriting its encoder).

**Costs.** None: the same memory (a scratch row per chunk buffer, as before), the same work elsewhere.

**Measured** (2026-11-04, the desktop beside the live Delver run - CPU timings inflated by it, so the
comparisons are within one process or interleaved A / B on one seed):

* *Per row* (`rowcost`, E:/mtg-zero-scratch/ideas-3/rowcost: 15 Delver mirror games searched at the line's
  64 / 16 simulations with priors 0.85 on the heuristic bot's move, every 16th leaf kept - 10,001 rows, both rows
  of a root-view leaf, ~252 non-zeros of 37,620 a row in ~82 of 587 blocks of 64; mimalloc, as MTG Zero's module):
  the old scan 10.7-12.3 us -> `sparse_scan` 3.0-3.9 us; `obs_hash` 3.0-3.5 us -> `obs_hash_sparse` 0.65-0.88 us;
  a whole row as the driver reads it, old -> new, interleaved per row: **sent 21.0 -> 11.8 us, cached 12.4 -> 11.4
  us**; pairs and keys equal on every row. At the run's 0.98 sent and 0.50 cached rows per simulation: -9.5 us
  of thread time a simulation (-23% of the ~41 us measured in the same process).
* *The driver* (MTG Zero's self-play seat - eval_cache 8192, root-view rows, the root's belief, sparse batches -
  with the line's gen 1000 on the CPU, 24 games in flight, 2 threads, 2 groups, one seed; the old build (zero-core
  aae4ca2) and the new one run at the same time, so under the same load; E:/mtg-zero-scratch/ideas-3/ab_conc.sh):
  the search threads' time per simulation (`busy_s` / sims) **103.4 -> 87.0 us and 93.8 -> 80.8 us (-16%,
  -14%)**, the same 62,578 rows sent and simulations in the timed window. (Run one after the other: 101.1 /
  104.6 old, 93.3 / 68.9 new - the load moved between runs.) The absolute numbers are the loaded desktop's.
* *Byte identity on MTG* (E:/mtg-zero-scratch/ideas-3/digest_ab.py: the same seat, 6 games, answered by a
  function of each row's non-zeros - no network, whose CPU floats are not reproducible run to run): the old and
  the new build give the same 29,114 batches (sparse arrays, options, kinds, players, gids), samples and results,
  digest for digest, and the same counters (25,428 cache hits, 14,533 value-only rows, 536 belief rows).

**Checks.** cargo test 114 (the golden digests unchanged; `lookahead::tests::sparse_scan_and_obs_hash_sparse_
match_the_dense_row`, `driver::tests::sparse_rows_from_encode_sparse_are_the_dense_rows`), pytest -k "sparse or
cache" 6 passed / 1 skipped (CUDA); MTG Zero (ENGINE.md 89): its engine test
`encode_sparse_is_the_rows_non_zeros_with_the_same_key` (cube and Delver mirror states, both viewers) and its re-pin
gate (the engine digest f6543eb2b1b8752f unchanged, the 1-config Delver search smoke); `tools/smoke_all.sh
--with-this-core --build-only` with Glass Road, Caverna and Kingdom Builder: all three build (the trait method has
a default).

## 29. `smoke_all.sh` finds the games from a worktree or a submodule, and no game is a failure (2026-11-04)

**The bug.** With no game arguments `tools/smoke_all.sh` looked for the game repos only beside the checkout
it ran from (`../glassroad-zero`, `../caverna-zero`, `../kingdombuilder-zero`). From a git worktree
(`zero-core-wt/<name>`) or a zero-core submodule (a game's or MTG Zero's `third_party/zero-core`, in a
worktree or not) there is nothing there, so it found no game, ran the core's own steps and printed GREEN:
"do the other games still build against this core?" was silently never asked. MTG Zero's tiered gate runs it
exactly that way - `bash third_party/zero-core/tools/smoke_all.sh --with-this-core --build-only` from the
checkout under gate, always a worktree, as its `optin-core` tier's step "build-only: the other games" - so
that step built no game whenever it was run without `--other-games` (its GREEN on 619f1f2, 2026-11-03, was
the core's cargo test, build and pytest alone; MTG Zero's ENGINE.md 90). Found 2026-11-04 when the same
command with the three repos passed explicitly built all three. A pipeline that processed 0 of N must fail
loudly.

**What it does now.**

* **Discovery** (no game arguments): `../<game>`; beside the main checkout when this is a worktree (the
  directory holding `git rev-parse --git-common-dir`); for a submodule, beside the superproject's checkout
  (`--show-superproject-working-tree`) and beside the superproject's main checkout. The first hit per game
  wins; `SMOKE_GAMES_ROOT=<dir>` looks in `<dir>` only.
* **No game is a failure**: found none and given none, it fails before running anything - exit 1, a
  `FAIL  games: no game repo found ...` summary line naming where it looked, never GREEN. A deliberate
  core-only run says so: `--core-only` (or `SKIP_GAMES=1`). A game it cannot find is a `missing:` summary
  line, and the summary counts what it checked (`games: 3 of 3 checked (glassroad-zero caverna-zero
  kingdombuilder-zero)`).
* **Explicit game arguments**: unchanged for game checkouts. A path that does not exist exits 2 (it used to
  become an empty entry that was skipped silently), an unknown option exits 2 (it used to be taken as a
  path), and a directory that is not a game repo is a FAIL (was a `skip`).
* **`--with-this-core`**: a scratch worktree whose zero-core cannot be moved to this core's HEAD is a FAIL
  (it used to build the game's own pin and report it as built against this core); a note says when the core
  checkout has uncommitted changes, which the games' builds do not contain.
* **`--dry-run`**: prints the core steps, each game and where it was found, and exits (1 when no game is
  found).

**Tests and checks.** `tests/test_smoke_all.py` (throwaway git repos, `--dry-run` and the no-game exit only,
nothing built): the games found beside the main checkout, from a worktree and from a submodule inside a
worktree of its superproject, the no-game failure and `--core-only`, explicit paths - 5 passed; the worktree,
submodule, no-game and explicit-path cases fail against the old script. On the desktop: the old script (aa10733) from `zero-core-wt/smoke-fix` and from an MTG Zero worktree's
`third_party/zero-core`, `SKIP_CORE=1 ... --with-this-core --build-only`, printed an empty summary and GREEN
(exit 0); the new one (7d3dbd8) finds `C:/Users/Gregg Keithley/{glassroad,caverna,kingdombuilder}-zero` from
both, and with `SMOKE_GAMES_ROOT` pointing at a missing directory exits 1 without GREEN (dry run and real
run). MTG Zero's fixed gate step (its ENGINE.md 90: the three repos passed explicitly, an "ok <game>: build"
line required for each), run for real against 7d3dbd8 at IDLE priority with 2 build jobs: PASS in 159 s,
Glass Road 41 s, Caverna 43 s, Kingdom Builder 66 s, `games: 3 of 3 checked`; its summary is the old
script's explicit-path summary line for line plus that count line.

## 30. Information-set merging (`eval_infoset_merge`, `selfplay_infoset_merge` in 30.1; MTG Zero's Delver mirror, 2026-11-04)

The semantics are docs/search.md 3.16; this section is the run-level view.

**Why.** With hidden cards, the other seat's options differ between the worlds a search deals (Daze in the sampled
hand or not), and the tree of before keys the other seat's decision by its option list - so the root decider's next
decision is split into one node per option list above it, each with its own statistics: the search can play one
information set of the root decider differently depending on what the other seat was dealt (strategy fusion). In the
Delver line's searches two thirds of the main lines see the other seat with two or more option lists (Step 0 in
3.16). Merging the tree's nodes by the root decider's observation (single-observer ISMCTS) removes the split. Its gate
(3.15) failed on depth and fragment counts; the user decided to build it and let an evaluation-only paired match
decide.

**The setting.** `eval_infoset_merge` 0 | 1 (default 0 = off): every evaluation seat's trees merge their nodes by the
root decider's observation (`games.nn_seat(infoset_merge=True)`, the seat key `infoset_merge`). Needs
`eval_leaf_value_view` 'root' (validated: a merged node of the other seat is valued from the root decider's view).
The evaluation's moves change with it, so its Elo pool's name gets `-ism` (`mcts.merge_tag`, after the view's and
the belief's suffixes: `cq-n4-v50-s0.3-rv-b1m0.25-ism`). Off, the seat dict has no key and every search is the one
of before, bit for bit. Self-play: 30.1.

**h2h.** `--a-infoset-merge / --b-infoset-merge 0 | 1` (or `--infoset-merge` for both; default: each side's run's
`eval_infoset_merge`). A merged side needs root-view leaf values (its run's `eval_leaf_value_view` 'root' or
`--*-leaf-value-view root`), else the match is refused before any game. The output names the side (`select_a`:
`...-ism`) and adds `merge`: the merged trees' links, merges, aliases and growth leaves summed over the drivers (a
merged side whose trees made no link is reported on stderr: a setting that did nothing).

**Costs.** One sparse encode of the viewer's row and a hash per new child signature (about one per walk), a link per
signature, 16 bytes of side statistics per option of an expanded node; known links cost nothing. Measured below.

**Measured** (MTG, CPU, the desktop beside the live A/B; docs/search.md 3.16 has the tables): the merged tree's walks
are 14-17% deeper, a search expands 20-33% more distinct information sets (0.86-0.90 per walk against 0.65-0.75),
the share of leaves re-expanding a held information set falls from 26-35% to 10-13% (the rest: growth leaves), and the
main line's edge into the other seat holds ~1.4 other-seat nodes instead of 3-8. The evaluation rule's root pick
changes in ~10% of contested decisions (prior max < 0.7; 8% of all decisions) and in none of 160 searches at
uniformly drawn positions: an effect on ~1% of the evaluation seat's moves, if any.

**The evaluation-only h2h** (the GPU's, after the A/B; from an MTG Zero checkout whose zero-core has this branch -
`C:/Users/Gregg Keithley/mtg-zero-wt/infoset-build` is one, built): the same checkpoint on both sides, the run's
evaluation seat (32 simulations, cq rule guard 4 c_scale 0.3, root-view values, the root's belief tilt 1 / clip 20 /
mix 0.25), side A merged:

```
python -m mtgz.az.h2h CK CK --paired --games 1200 --sims 32 --chance-k 0 --device cuda --seed 0 \
    --select cq --select-min-visits 4 --cq-c-scale 0.3 \
    --a-leaf-value-view root --b-leaf-value-view root \
    --a-belief-tilt 1 --b-belief-tilt 1 --a-belief-mix 0.25 --b-belief-mix 0.25 \
    --a-infoset-merge 1 --b-infoset-merge 0
```

(With CK = `E:/mtg-zero/runs/r3-2b-float550/models/iter_NNNN.pt` the explicit rule flags are what the run's
config.json gives anyway - `--a-infoset-merge 1 --b-infoset-merge 0` alone would do; a checkpoint copied elsewhere
has no config.json two levels up, hence the flags.) The output must name `select_a` `cq-n4-v50-s0.3-rv-b1m0.25-ism`
and `select_b` `cq-n4-v50-s0.3-rv-b1m0.25`, and carry `merge` with links > 0. **Stop rule** (Fable's): `elo_diff`
within +-7 -> stop (the merge is null at play); above +7 -> build self-play support and run a self-play A/B; below -7
-> drop it. At 1,200 paired games `elo_se` is ~7-8, so a result near the boundary can be extended with the same seed
schedule (`--sprt -7,7 --batch-games 600 --max-games 2400`, or `--target-se 5`).

**Checks.** cargo test (zero_search 134, webnet 3): the golden records and matches unchanged with the setting off;
the merge's tree and driver tests (docs/search.md 3.16). pytest `tests/test_infoset_merge.py` (4) and `-k "h2h or
settings_seats or search_tree_api or tree_bindings or eval_tag or names"` (22). MTG Zero e098aff with this core
(`ZERO_CORE_UNPINNED=1`): the engine digest f6543eb2b1b8752f unchanged, and the h2h command above end to end (2 paired
games at 4 simulations on the CPU: the names and `merge` counters as stated).

### 30.1 Self-play (`selfplay_infoset_merge`, 2026-11-04)

**Why.** The evaluation-only h2h of 30 (one checkpoint on both sides, 2,400 paired games at the evaluation seat)
came out **+9.3 +- 5.3 Elo** for the merged side - above the stop rule's +7, so (Fable's rule) self-play support and a
self-play A/B. Training on merged searches changes the targets the network learns from: the 'cq' policy target and the
root value q of a root whose subtrees no longer split the root decider's next information set by what the other seat
was dealt. Whether the network learns better from them is the A/B's question, not this build's.

**What it does.** `selfplay_infoset_merge` 0 | 1 (default 0): `Run.nn_seat(selfplay=True)` gives the self-play seat
`infoset_merge=True`, every search of it merged (full, fast, minor). Needs `selfplay_leaf_value_view` 'root'
(validated: `check_eval_settings`). What it composes with and why (the recorded targets, q, value-only samples and
resignation read the root only; root noise, the belief, the evaluation cache, the playout cap, minor searches, reuse,
rolling restarts, branched games and archive starts), the remote guard and what is refused: docs/search.md 3.16
(Self-play). The leaf's row of the viewer comes from the merged tree's encoding (`Tree::leaf_obs`: one encode fewer
per leaf, the same bytes). Metrics `selfplay.merge` (this machine's driver, only when on): `links`, `merged`,
`aliases`, `growths`, `shared`, `merged_per_leaf`; an iteration whose leaves made no link logs a WARNING (a setting
that did nothing). Remote self-play: a worker whose checkout does not report merging in its hello is REFUSED, killed
and retried in 10 minutes (as 27's resignation); the remote's own counters are not in `selfplay.merge`.

**Off by default.** 0: `Run.nn_seat` gives the seat dict of before, key for key (pytest), and the driver runs the path
of before (the golden records and matches; `Tree::leaf_obs` is None).

**Costs** (measured on MTG Zero's Delver mirror, CPU, docs/search.md 3.16 Self-play has the table): a merged
self-play simulation sends **21.5% more network rows** (1.200 against 0.988: the same rows are needed, 1.49 a
simulation, but the evaluation cache answers 0.29 instead of 0.50 - its repeats are merged away), so a GPU-bound
self-play spends ~20% more evaluator time per game; the search threads' time per simulation was +33-42% on the loaded
desktop (113 / 140 us against 85 / 99 us, min / median of 4). Encodes per simulation +19% with the tree's encoding
reused for the leaf's viewer row (0.96 of 1.23 tree encodes), +83% without. Simulations per wall second with the
network on the CPU: 723 against 1,008 (median).

**The self-play A/B** (proposed; the main session's, on the GPU). As the three-arm A/B of 2026-11-04 (MTG Zero
E:/mtg-zero-scratch/ab-ideas/RESULTS.md): one fork point K of the Delver line, the arms in turn, 40 generations each,
both on the same checkout (MTG Zero `feat/infoset-selfplay`, which pins this core: off, the line's searches bit for
bit), every other flag the line's, `eval_infoset_merge` 0 in both (the in-run Elo pools stay comparable).

* *Arms:* control (the line) and ism (`selfplay_infoset_merge 1`; the line already has `selfplay_leaf_value_view`
  root). The same games per generation (1,600): equal data, the ism arm slower per generation (above). Remote self-play
  only if the Mac's checkout has this core (else its workers are REFUSED and the arm plays its quota locally); the
  simplest symmetric choice is the same remote setting in both arms.
* *Primary endpoint - what the network learned:* paired GPU h2h ism@K+40 vs control@K+40, 1,200 paired games (seed 0),
  32 simulations, the evaluation seat, both sides UNMERGED (`--a-infoset-merge 0 --b-infoset-merge 0`); a readout at
  K+20 likewise.
* *Secondary:* the deployable pair, ism@K+40 merged vs control@K+40 unmerged (`--a-infoset-merge 1 --b-infoset-merge
  0`; the evaluation-only +9 plus the training effect); both merged; the fresh-row value CE on the control's rows (the
  neutral set) at K+37; the 'cq' targets' h_norm / max_share and the training entropy; `selfplay.merge` per
  generation (merged_per_leaf, never the WARNING); min per generation, rows per second, the evaluator's busy share; the
  blunder suite, fingerprints and canary at the usual generations.
* *Stop rule:* at K+20 the primary below -14 Elo (~2 SE) -> stop the ism arm. At K+40: primary > +7 -> extend both
  arms to K+80 and confirm with the same seed schedule (`--sprt -7,7 --batch-games 600 --max-games 2400`), adopt if it
  holds; within +-7 -> do not adopt self-play merging (it costs ~20% of a generation's evaluator time for no measured
  gain; the evaluation-only merge is decided on its own, e.g. by extending its h2h); below -7 -> drop it. (The last
  A/B's lesson: 40 generations resolve only effects of ~+-15 Elo; the line gains ~+23 Elo per 200 generations.)

**Checks.** cargo test 139 + webnet 3 (the golden digests unchanged) with the self-play driver tests of docs/search.md
3.16; pytest `tests/test_infoset_merge.py` (8: settings, seats, bindings self-play, a rolling toy run with exact
quotas and the metrics, remote workers merging, an older worker refused) and `tests/test_resign.py`,
`tests/test_forks.py`, `tests/test_bindings.py` (37); MTG Zero `feat/infoset-selfplay` (its ENGINE.md 92): the engine
digest f6543eb2b1b8752f, its optin-core and repin gates.

## 31. Self-play's shards written on a thread of their own (`shard_writer_thread`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** One Python thread drives a self-play driver (`zero.infer.drive`: `next_batch` -> the evaluator -> `submit`)
and, between batches, also handled the training samples: `take_samples` (a copy of every record into one dense
float32 array - MTG Zero's rows are 37,620 floats, ~9 GB of shards a generation), `split_games`, `TargetStats`, and
`ShardWriter.add`, whose every `shard_rows`-th add concatenated, cast to float16, checked and wrote an .npz shard
synchronously. Meanwhile the network and the search threads waited. MTG Zero's thread benchmark on the GPU
(E:/mtg-zero-scratch/ideas-3/bench_threads_ab.json: the Delver line's self-play seat and driver, 8 threads x 2
groups, 768 games in flight, 640 games timed) did **13.7 games/s without the sample handling and 10.7 with the
run's** (-22%); the evaluator's busy share fell 0.92 -> 0.71 (the live run's self-play phases show ~66-71%). Of
the pump's 15.4 s of the caller's time in that configuration, 11.6 s were the shard writes and ~3 s `take_samples`'
copy (the same take every 64 batches costs 2.3-3.0 s without any writing: 6-7% of the caller's time).

**What it does** (`zero/shardthread.py`; docs/search.md 2.4 for the binding). With `shard_writer_thread` 1:

* **The caller only takes.** `SearchDriver.take_records()` moves the driver's records out as they are (no copy:
  a `SampleRecords`), `split_records` cuts them per finished game (the records move, as `split_games` slices a dict:
  the shared queue's held games keep their records), and `ThreadedShardWriter.add` queues them. What stays on the
  caller: `take_results` (the finished games' rows: the quota needs them now), the move, the split (moves in Rust),
  the queueing. Measured below, per 1,000 rows.
* **The writer thread** ('zero-shard-writer', one per self-play phase) converts each item - `SampleRecords.arrays()`
  is `take_samples`' conversion (one function for both), built and the records freed off the GIL; the features cast
  to float16, the dtype a dense shard stores (the same values: the same bytes; half the memory while the writer
  buffers a shard) - runs the phase's tallies on it (`TargetStats`; the shared queue's staleness histogram), and
  calls the wrapped `zero.data.ShardWriter`'s `add`, then its `close`, in the order the items came. Same records,
  same order, same writer: **the shards are the synchronous writer's byte for byte** - rows, order, file names,
  meta, quarantined `*.bad.npz`. `ShardWriter` itself is unchanged (wrapped, not edited; a writer that keeps its
  rows' non-zeros - an attribute `sparse` true, as MTG Zero's parallel sparse-shards work proposes - gets the float32
  rows and casts them itself).
* **Bounded.** At most `shard_writer_queue_mb` MB of items wait (`SampleRecords.nbytes`: MTG ~150 KB a row; the
  default 4096 holds ~27k MTG rows, more than the GPU's self-play makes while one 65,536-row shard is written);
  when full, `add` blocks (counted: `waits`, `wait_s`), and an item larger than the bound passes alone. The writer
  buffers at most a shard, in float16; the synchronous writer buffered it in float32 and concatenated it in float32
  (a peak of ~2 float32 shards), so the threaded writer with its default queue stays within what the synchronous one
  used.
* **Errors** on the writer thread (`NonFinite` - the shard quarantined exactly as before -, an `OSError`) are
  re-raised in the caller by its next `add` or by `close`: the exception itself, never lost; the run fails as it
  did (state `aborted`, "run aborted: NonFinite: ..." in its log).
* **Every shard on disk before training reads the window.** The phase's `close` waits for the queue, the last shard
  (`ShardWriter.close`, on the writer thread) and the thread before self-play returns: this machine's phase end, the
  shared queue's (its `samples` = what was written), so before `linger_start` (the local linger takes records and
  holds them; the next phase's writer converts them) and before training. **PAUSE or an error mid-phase:**
  `Run.selfplay` aborts the phase's writer first - it finishes the shard it is writing (completed and renamed: never
  a half-written shard, nor its temporary file; the test fails without the abort) and drops the rest; the iteration
  is replayed, as always (its data directory cleared). STOP is read between iterations: the phase closes normally.
* **Remote workers** (`selfplay_remote`): a worker reads the setting from the run's config; with it, it takes
  records and writes on a bounded `ThreadedShardWriter` instead of the unbounded `_BgWriter` (its shards were already
  written on a thread: the take's copy and the bound are what change) and announces each shard with its games as
  before. A worker on an older checkout ignores the setting: its shards are the same. An engine module built
  before `take_records` (a checkout updated but not rebuilt) is refused with that reason, here and on a worker.

**Metrics** (only with it on): `selfplay.shard_writer` = `samples_s` (the caller's seconds in its sample handling:
taking the results and records, splitting, the quota's accounting, queueing, waits included), `waits` / `wait_s`
(for room), `close_s` (the phase's end waiting for the last shard), `writer_s` (the writer thread's busy seconds),
`items`, `rows`, `queue_peak_mb`, `queue_mb`, and a log line per iteration; `linger.samples_s` (the linger thread's
take and split); `selfplay.remote.shard_writer` (the workers', summed).

**Off by default.** 0: `take_samples`, `split_games` and the synchronous `ShardWriter` between batches (the remote's
`_BgWriter`), exactly the loop of before; no new metrics key. Turn it on with `--shard-writer-thread 1` (and
`--shard-writer-queue-mb N`). Although the shards stay byte-identical, **the caller's timing changes which games
finish when** - a rolling driver's quota and the shared queue take the games that finished first - hence opt-in.
Single-game runs: a co-training run is refused (its writers stay on the driving thread).

**Costs.** One thread per phase: the conversion and the writes still take a core, now beside the search instead of
between its batches (memory bandwidth, and the GIL in short pieces: numpy's casts, copies and CRCs release it,
`np.savez`'s chunk copies do not). Memory: the queue's bound (4 GB by default) beside a shard's float16 buffer. The
binding: `SearchDriver.take_records` and `SampleRecords` (additive; `take_samples` now frees its records off the GIL
too - the same arrays).

**Measured** (2026-11-04, the desktop's CPU only - the GPU is the live run's -, IDLE priority beside the live
Delver run and other jobs: relative numbers). MTG Zero's thread bench with the 't' configurations added
(E:/mtg-zero-scratch/shard-thread/bench_threads.py, bench_cpu.log / .json): gen 1000's self-play seat and driver
(eval_cache 8192, sparse batches, the belief recorded), the network on the CPU, 48 games in flight, 4 threads x 2
groups, the same seed in every configuration (the same 32 timed games, 2,849 rows, 3 shards of 1,024 rows), 'p' =
the run's synchronous pump (take_results + take_samples after every batch, split_games, TargetStats, ShardWriter),
't' = this setting's (take_results + take_records, split_records, the queue), interleaved p, t, p, t:

| the caller's seconds | p | t | p | t |
|---|---|---|---|---|
| take (take_results + take_samples / take_records) | 0.35 | 0.035 | 0.45 | 0.021 |
| split per game | 0.002 | 0.002 | 0.004 | 0.002 |
| add (the writes / the queueing) | 1.16 | 0.001 | 0.80 | 0.001 |
| in the phase, per 1,000 rows | **529 ms** | **13 ms** | **442 ms** | **8 ms** |
| close (the phase end: the last shard / the writer's drain) | 0.20 | 0.23 | 0.21 | 0.11 |
| the writer thread's busy seconds | - | 1.54 | - | 0.94 |

So the caller's share of the sample handling inside the phase fell ~40-55x: take_samples' copy alone was 120-160 us
a row on this loaded machine (take_records + take_results: 7-12 us), the writes ~0.3-0.4 ms a row; the queue never
filled (peak 20 MB). The phase end waits for the last shard in both (a training needs it). Games per second are not
comparable on the CPU: the network there is the bottleneck (the caller's sample handling was < 1% of its time) and
the load moved 0.14-0.21 games/s between configurations. On the GPU the same per-row costs were the 22% above (the
caller's 15.4 s of 59.7 in 'p'); the GPU run (`CONFIGS=8x2,8x2p,8x2t,16x3,16x3p,16x3t bash
E:/mtg-zero-scratch/shard-thread/bench_threads.sh`, the main session's) measures what it gives back.

**Tests and checks** (`tests/test_shard_thread.py`, 17): `take_records` + `split` per game + `arrays()` =
`take_samples` + `split_games`, array for array and dtype for dtype (and the binding's errors); the threaded writer's
shards = the synchronous writer's byte for byte, file for file (per-game records and dicts, plain and compressed
shards, the `on_write` calls, a queue bound smaller than an item so that every add waits); whole takes with
`TargetStats` on the writer thread (the same summary); a writer keeping its rows' non-zeros gets float32 rows;
`NonFinite` and an `OSError` re-raised in the caller (the same quarantined shard); a full queue blocks the caller and
a writer error frees it; `abort` completes the shard being written; an iteration's shards with the setting off and on
byte for byte (several shards; the same samples, games and target metrics; no writer thread left; no metrics key
off); a rolling run's quotas exact; the shared queue with the remote 'local' and the local linger (quotas exact, the
linger's held records counted one checkpoint stale, the workers' writer threads, every shard holding the rows its
metrics report); PAUSE mid-phase with the writer inside a shard (exit 75, no temporary file - fails without the
abort -, the resume replays and finishes); the loop's `NonFinite` from the writer thread; co-training refused. Off:
the full pytest (383 passed, 9 skipped, with 16 of these; the 17th alone), cargo test 139 + webnet 3 (the golden
records), and MTG Zero's gates (its ENGINE.md 93: `check_checkpoint_compat` byte identity - take_samples now goes
through the shared conversion -, the engine digest, the other games built against this core).

## 32. Sparse sample shards (`shard_sparse`; MTG Zero's Delver mirror, 2026-11-04)

**Why.** A shard stored each row's features dense as float16: for MTG Zero's Delver line 37,620 floats (75 KB) a
row, ~9.6 GB per generation (iteration 1289: 124,581 rows in two shards, 9,632 MB - the `bytes` of its self-play
metrics), though a row has ~242 non-zeros (0.64%). The window has kept only the non-zeros in RAM since 10.8
(`window_sparse`), so the dense files were overhead everywhere else: disk (E: has run short: a forked A/B arm needs
~50 GB of shards of its own), the write on the run's caller thread (part of the 22% of self-play throughput lost to
sample handling, MTG Zero's E:/mtg-zero-scratch/ideas-3/bench_threads_ab.json), RAM (the writer buffers the driver's
float32 rows until a shard is full - 9.5 GB for a 65k-row MTG shard - then concatenates and casts them at the
flush: a peak of ~20 GB), and the copies from the remote.

**What** (zero/data.py, zero/finite.py; opt-in, the default the shards of before):

* *The format* (version 1): the features as `x_ptr` int64 [N+1], `x_idx` uint16 [nnz] (int32 past 65,536 columns),
  `x_val` float16 [nnz] instead of `x` - row r's non-zeros at `x_ptr[r]:x_ptr[r + 1]`, columns ascending, non-zero
  decided on the float16 bits (a -0.0 is kept, a value whose float16 is +0.0 is not: the dense rows come back bit for
  bit) - and in the meta `x_format` 'sparse', `x_version` 1, `x_cols` (the dense width), `x_nnz`. Every other array
  (`player`, `n_players`, `opt_off`, `opt_action`, `pi`, `z`, `q`, `sc`, `net`, `aux`) is the dense shard's, in the
  same order. A shard is sparse when it has `x_val`; a newer `x_version` is refused by every reader (ValueError
  naming it). Plain or compressed npz as before (local plain, the remote's zlib 1).
* *Writing.* `data.shard_format(cfg)` gives the meta keys that ask for it (`{'x_format': 'sparse'}`, `{}` off);
  `write_shard` writes sparse when its meta asks, and a `ShardWriter` whose meta asks converts each added sample dict
  as it arrives (`SparseRows.from_dense` of the driver's float32 rows: their non-zero bits, then those values cast)
  and never holds or builds the dense rows - its buffer is the non-zeros and the other arrays. ShardWriter's
  interface is unchanged (constructor, `add`, `close` and its `{'samples', 'bytes', 'shards'}`, `on_write`), so a
  wrapper calling it on another thread works on either format; `bytes` is the files as written. The writers that
  ask: the run's self-play (one-shot, rolling and the shared queue with a remote), co-training's two networks, and
  a remote self-play worker (the run's settings reach it through its mirror's config.json; an older remote checkout
  writes dense shards, accepted as any).
* *Reading* - every reader takes both formats, and a window mixes them (the generations before the setting beside
  the ones after):
  * `read_shard(path)`: the arrays and meta, a sparse shard's features densified as `x` (all its rows - tools,
    tests); `read_shard(path, dense=False)`: as stored (`x_ptr` / `x_idx` / `x_val`).
  * `shard_features(path, rows, F)`: some rows of either, cut / zero-padded to F (a dense shard's read in place
    when stored plain); `stored_rows(p, F)`: a sparse shard's SparseRows at a network width.
  * `Window`: a dense window scatters a sparse shard's rows into its own (subsampled or whole, the cache too); a
    sparse window (`window_sparse`, with or without the cache) takes the stored non-zeros as its SparseRows, cut to
    the network's width, the column dtype `from_dense` would give - the same window either way, so `batch`,
    `features_coo` (train_static's COO path), `batches` / prefetch, `WindowCache` and the budget (dense rows, from
    the meta's `rows`) are unchanged.
  * The non-finite guard (`finite.shard_problems`): a sparse shard's `x_val`, described by the rows and columns of
    the dense rows (the same message); the writer quarantines a bad sparse shard as `*.bad.npz` in its format.
  * `heldout` (Window, meta), `distill` / `tools/distill.py` (Window, meta), remote self-play's fetch check (the
    arrays as stored: no 0.6 GB dense copy per arriving MTG shard) and its test helper `_corrupt` (keeps the
    format), `bench --shard` (`shard_features`), `run.window_shards` and `shard_rows` (the meta), `fork_run` (hard
    links / copies the files, never reads them).
* `data.convert_shard(src, dst, sparse=True | False)`: one shard rewritten in the other format offline, every other
  array as stored (a dense source read in place in chunks, never its dense rows in RAM): an old generation shrunk for
  good, or a sparse shard made readable by an older checkout.

**Off by default.** `shard_sparse` 0: the shards of before - the same arrays in the same order and the same meta (no
`x_*` key; `compact` now builds its dict from `_compact_rest`, the same dict). On: `--shard-sparse 1` (or in a
preset); it takes effect at the next self-play phase, and the window reads the dense generations already on disk
beside the new sparse ones. Nothing about the samples, the window, the batches or the training changes.

**Costs.** The conversion of each added sample dict on the writer's thread (~63 us a MTG row: 4.1 s for a 65k-row
shard, spread over the phase) against the dense flush's concatenation, cast and 5 GB write. An older zero-core
cannot read a sparse shard (KeyError 'x'): a fork onto an older pin, tools of pinned run worktrees and scripts that
read the `x` member need `convert_shard(..., sparse=False)` or this reader (MTG Zero's readout scripts: its ENGINE.md
94). The auxiliary targets stay dense: on MTG they are now 134 MB of the 200 MB shard (zlib 1 makes them 1.4 MB;
the next candidate, not part of this setting).

**Tests and checks** (pytest; the whole suite below):

* `tests/test_net_data.py::test_sparse_shards_round_trip` - the same driver-format samples (float32, ~5% non-zero,
  -0.0s, values whose float16 is +0.0 and -0.0, an all-zero row, value-only rows, auxiliary targets) through a dense
  and a sparse ShardWriter, plain and compressed, two shards each: every array and the features bit for bit, the
  meta plus the `x_*` keys, the writers' counts; the stored arrays (dtypes, ascending columns, -0.0 kept); the dense
  shard's members and meta exactly those of before; `shard_rows`, `shard_features` (rows, slices, cut / padded);
  `convert_shard` both ways equal to the writers' shards; an unknown `x_format` and a newer `x_version` refused by
  every reader.
* `test_sparse_shards_are_the_same_window` - windows over sparse shards and over a mix of dense and sparse ones are
  the windows of the dense shards of the same samples (every array, the features bit for bit, the batches - COO too
  -, the bytes and reuse): dense and sparse windows, narrower / wider networks, over budget, auxiliary targets with
  one shard without, and both caches over sliding windows.
* `tests/test_loop.py::test_shard_sparse_writes_the_same_samples` - a toy run's self-play (one-shot and rolling)
  with the setting on writes the shards of the setting off, read back array for array, smaller; off, the dense
  members and meta of before. `test_sparse_shards_train_the_same_network` - the smoke run's shards converted to
  sparse, and a mix of dense and sparse generations, train the network of the dense shards weight for weight
  (dense / sparse window, cache, prefetch, over budget). `test_loop_with_sparse_shards` - `--shard-sparse 1` end to
  end.
* `tests/test_nonfinite.py::test_sparse_shards_are_guarded_alike` - a float16 overflow quarantined with the dense
  shard's message, the bad shard sparse; a NaN in a sparse shard refused by both windows, naming its row and column.
* `tests/test_remoteplay.py::test_sparse_shards_from_both_machines` (the remote 'local' worker writes sparse shards,
  each checked and counted on arrival; the steps follow both machines' samples) and
  `test_nonfinite_remote_shard_is_quarantined[0|1]` (a damaged sparse remote shard quarantined).
* `tests/test_measure.py::test_heldout_on_sparse_shards` (a held-out set made from sparse shards is the dense
  shards' set; a sparse set gives the same losses) and `tests/test_distill.py::test_the_tool_reads_sparse_shards`
  (`--compare` on sparse shards = on dense ones).
* The whole suite on 3f830c5: 377 passed, 9 skipped (once standalone with one known timing flake in
  `test_remoteplay.py::test_a_dying_worker_gives_its_accepted_games_back` that passed alone; once in MTG Zero's
  optin-core gate with none). MTG Zero `feat/sparse-shards` (its ENGINE.md 94): its own test on Delver rows, the
  optin-core gate (checkpoint_compat identical, the three other games built against this core) and the repin gate
  (engine digest f6543eb2b1b8752f).
* A test merge with `feat/shard-writer-thread` 57fd981 (the writer thread that wraps ShardWriter, section 31; `git
  merge-tree`: no conflict): an iteration's toy self-play, one-shot and rolling, writes the same samples with
  `shard_writer_thread` x `shard_sparse` in all four combinations, and both branches' shard tests pass on it. With
  both on, the writer thread casts each dict's features to float16 before `add`, which then converts that (the same
  non-zeros; `from_dense` would take the float32 rows directly).

**Measured** (2026-11-04, the desktop beside the live Delver line, IDLE priority; MTG Zero's r3-2b-float550,
iteration 1289's shards copied to E:/mtg-zero-scratch/sparse-shards/real, converted there, deleted after;
measure.py / measure2.py / measure.json):

| | dense | sparse |
|---|---:|---:|
| a local 65,566-row shard (the features 4,933 MB, `aux` 134 MB, the rest 2 MB) | 5,069 MB | **200 MB (25.3x)**: the features 63.9 MB (15.85 M non-zeros, 242 a row, 0.64%: 77x), `aux` 134 MB |
| the same, zlib level 1 | - | 11.0 MB (461x; 3.4 s to convert against 3.1 s plain) |
| a remote shard (the Mac's, zlib 1; 6,673-7,405 rows) | 6.8-7.6 MB | **1.11-1.25 MB (6.1x)** |
| a generation (iteration 1289: 124,581 local + 21,450 remote rows) | 9,632 + 22 MB | ~380 + 3.6 MB |
| writing (ShardWriter, the driver's float32 rows in dicts of 512 rows) | 0.32-0.38 ms a row at the flush (4,096 rows 1.31-1.47 s, 8,192 rows 3.12 s): ~21-25 s for a 65k-row shard | 4.1 s of `add` (the conversion) + 0.5 s at the flush for the 65,566-row shard: ~0.07 ms a row |
| the writer's buffer before the flush / its peak (tracemalloc; 8,192 rows) | 1,187 / 2,533 MB (~9.5 / ~20 GB for 65k rows) | 39 / 163 MB (~0.3 GB for 65k rows: the float32 `aux` most of it) |
| a remote shard written at zlib 1 (on the desktop) | 1.77-2.00 s | 0.44-0.56 s |
| a shard into a sparse window (whole, the cache's way; 8 threads) | 1.77 s (read in place + converted) | 0.36 s (zlib 1: 0.53 s) |
| 4,096 rows of it into a dense window | - | 0.46 s |
| a remote shard read (`read_shard`, densified) | 0.79-0.93 s | 0.18-0.21 s |

Every one of the 65,566 rows came back bit for bit (0 mismatched values over 2.47 G) with every other array equal;
the ShardWriter's own sparse shard of those rows (from float32 dicts) was `convert_shard`'s, array for array; the
remote shards' round trips too. (The timings are the loaded desktop's: the dense writer's full 65k-row flush could
not run beside the live run - it needs ~20 GB - so its row is the 4,096- and 8,192-row flushes, linear in rows.)
