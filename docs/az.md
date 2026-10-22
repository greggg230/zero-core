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
| `package.py` | The game package registry: `GamePackage` (name, root, engine module, run module, presets, DEFAULTS overrides, external players, specs, decision kinds, dashboard title / port, env prefix), `register`, `active` |
| `run.py` | The loop: `DEFAULTS`, `PRESETS` (the toy's + the package's), self-play (classic or rolling), training, evaluation, Elo, resume, STOP / PAUSE |
| `net.py` | `Net`: residual MLP with LayerNorm (arch 'mlp', the default); `GridNet` (arch 'grid'): residual 3x3 conv tower over the game's board planes (`GamePackage.grids`) with globals + pooled-board channel biases, one policy logit per cell from the tower; `TokenNet` (arch 'tokens', section 10): a transformer over the game's objects read through a layout descriptor (`GamePackage.tokens`), pointer logits for the objects' actions; all: policy logits over NUM_ACTIONS (masked to the legal actions by the caller), value = P(win) per relative seat (softmax over the seats present), auxiliary margin head; atomic `save`, `load` (by the checkpoint's arch), `widen`, `make_net` |
| `infer.py` | `NetEvaluator` (CPU / CUDA; gather + segment softmax on the device; `set_weights` in place; `workers`: eager batches on threads), `UniformEvaluator`, `drive()` (external batches answered on threads; several network batches in flight) |
| `data.py` | Sample shards (`ShardWriter`, atomic npz), the RAM-bounded training `Window`, `TargetStats` |
| `games.py` | `engine()` (the package's compiled module), registered games, seat specs, external players (`EXTERNALS`, `register_external`; a package registers its own, e.g. `ext:jsbot`), the `Driver` wrapper |
| `toybots.py` | The toy heuristic as an external player (`ext:toyheur`) |
| `elo.py` | Bradley-Terry Elo over all results, the run's yardstick = 1000 |
| `runctl.py` | PAUSE / STOP files, exit status 75, run / supervisor state files, `PauseWatch` |
| `dashboard.py` | Flask dashboard (title, port and runs directory from the game package) |
| `bench.py` | Throughput measurements |
| `h2h.py` | Checkpoint vs checkpoint, seats alternating (the readout of an A/B) |
| `targets.py` | Policy-target sharpness and quality: visits vs completed-Q targets (section 9) |
| `mcts.py`, `play.py`, `parity.py`, `protocol.py` | Reference search / driver, parity, the protocol (docs/search.md) |
| `tools/` (`supervise`, `kill_run`, `pause_run`, `fork_run`, `launch_hidden`) | Unattended runs, control, forks; each repo's `tools/<name>.py` activates its package and calls `main()` |

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
python tools/fork_run.py runs/toy1/models/iter_0010.pt runs/toy2          # new run from a checkpoint (widened)
python tools/fork_run.py --source runs/toy1 --dest runs/toy1b --iter 10 --set reuse=8   # continue at K
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
* Game presets come with the game package (`GamePackage.presets`): Caverna Zero
  `caverna-smoke`, `caverna-2p`, `caverna-2p-cq`; Glass Road Zero `glassroad-smoke`,
  `glassroad-2p`, `glassroad-2p-cq` (each repo's docs/az.md).

### 1.2 Settings (`python -m <pkg>.az.run --help`)

| Setting | Default | Meaning |
|---|---|---|
| `game`, `spec`, `players` | toy, '', [2] | registered game, its JSON spec, player counts sampled per self-play game |
| `iters`, `games` | 40, 1000 | iterations; self-play games per iteration (rolling: games FINISHED per iteration) |
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
| `eval_games`, `eval_sims`, `eval_random_games`, `eval_every` | 200, 32, 50, 1 | evaluation vs the yardstick and vs random |
| `elo_games`, `ref_games` | 100, 400 | games per Elo match (checkpoints 1, 2, 4 ... 32 iterations back); random vs yardstick reference games |
| `elo_opponents`, `elo_sims`, `elo_together` | 0, 0, 0 | cheaper Elo (CPU runs): play only the first k of the offsets 1, 2, 4 ... 32 (0 = all); Elo matches at elo_sims (0 = eval_sims); 1 = an iteration's Elo matches in one driver (one evaluator key per opponent: the checkpoint's batches hold the leaves of every match; other seeds than separate matches) |
| `eval_spec` | '' | the spec of every match that records nothing (evaluation, Elo, reference games); '' = the game package's evaluation spec of the self-play spec (`GamePackage.eval_specs`; MTG: the fixed evaluation deck set), else the spec itself |
| `infer_device`, `train_device`, `seed` | cpu, cpu, 0 | inference / training device (`cuda` works; the GPU is shared), run seed |
| `infer_workers` | 1 | network batches evaluated concurrently on this many Python threads, with as many in flight (give the driver as many `groups`); CPU inference of a small network (section 10) |

## 2. Run control (`runctl.py`)

* **STOP** (file in the run dir): the run ends after the current iteration; the supervisor
  exits with it (a stale STOP is moved to STOP.old when a supervisor starts).
* **PAUSE**: a watch thread notices the file within 0.5 s; the run stops at its next pause
  point - after every self-play / evaluation batch (`drive(check=...)`) and every training
  step - writes `run_state.json` (state `paused`) and exits with status **75**. The
  supervisor then polls PAUSE every second and relaunches the run the moment it is removed
  (not a crash: no delay, no back-off); a run started while PAUSE exists exits 75 at once.
  Measured in `tests/test_loop.py`: paused mid self-play and exited in a few seconds.
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
| `metrics.jsonl` | one line per iteration: `selfplay` (games, samples, decisions / rounds per game, avg score, leaves/s, games/h, inference time, average batch; `rolling` stats; `target`: kind, mean options, normalised entropy `h_norm` = H(pi) / ln(options) and `max_share` of the recorded policy targets), `train` (policy / value / margin loss, entropy, value accuracy, steps, window samples / MB / load time, lr, value_q, samples/s), `eval_yard` / `eval_random` (win, se, margin, games), `elo`, `phases` (seconds per phase), `seconds` |
| `models/iter_NNNN.pt` | checkpoints (atomic); `best.pt` = highest rated; `index.json` = every checkpoint's Elo; `opt_latest.pt` = optimizer state (with the iteration it belongs to); `iter_NNNN.json` = stats of an iteration whose evaluation has not finished |
| `data/iter_NNNN/*.npz` | sample shards (docs: `zero/data.py`); the newest `keep_data` iterations are kept |
| `elo/results.json`, `elo/ratings.json` | pair results (`{"a|b": [score of a, games]}`, anchor) and the current fit |
| `progress.json` | self-play progress of the iteration in flight (dashboard) |
| `run_state.json`, `supervisor_state.json`, `supervisor.log`, `PAUSE`, `STOP` | run control |

## 4. Training

Each iteration trains the previous checkpoint on the window (newest iterations' shards,
loaded into RAM; uniformly subsampled per shard beyond `ram_budget_gb`) for
`max(min_steps, new samples x reuse / batch)` minibatch steps (uniform with replacement).
Loss = policy cross-entropy against the recorded policy target - the visit distribution, or
the completed-Q target with `policy_target='cq'` (section 9) - (log-softmax over the legal
actions only) + value cross-entropy against `(1 - q) z + q q_root` (softmax over the seats
present) + `margin_loss_w` x 0.5 x squared margin error over the seats present. AdamW, grad
clip 5, step-decay LR, warmup while the optimizer state is fresh. Feature-width changes are
handled by prefix reads (an older network reads the first columns) and `widen` (fork_run).

## 5. Evaluation and Elo

Every `eval_every` iterations the checkpoint plays `eval_games` against the yardstick and
`eval_random_games` against the random bot (seats alternating, network at `eval_sims`); then
Elo matches against the checkpoints 1, 2, 4, 8, 16, 32 evaluations back (`elo_games` each);
the random bot is rated once against the yardstick (`ref_games`). All results are refit by
Bradley-Terry maximum likelihood (weak Gaussian prior) with the yardstick at 1000 after every
iteration; `models/index.json` and `best.pt` follow the fit.

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
(self-play progress, last log line age, STOP); tiles (iteration, newest / best Elo, win rate vs
the yardstick and vs random, self-play games, leaves/s); charts: Elo by iteration (yardstick
and random reference lines), win rate vs yardstick and random, each checkpoint vs the previous
one, losses, self-play average score, leaves/s, minutes per iteration by phase; head-to-head
table of the newest checkpoint with the Elo prediction; iteration table; log and supervisor
tails. With no runs it shows how to start one. Endpoints: `GET /api/runs`,
`GET /api/run/<name>`, `GET /api/run/<name>/control`, `POST /api/run/<name>/pause|resume`
(JSON body required; only plain names of run directories under `--runs`).

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
inference works here), the remote (Mac) workers and evaluator, rented-GPU experiments (Vast),
exploration games, behaviour cloning. Where they plug in:

* **Evaluator**: anything with `__call__(LeafBatch) -> (priors, value, margin)` and
  `set_weights(state_dict)` can replace `infer.NetEvaluator`; `Run.evaluator(path)` is the one
  place the loop builds it (a `zero.infer` factory keyed by a run setting is the natural
  switch, e.g. `infer_impl='cudagraph'`).
* **Remote / worker self-play**: `Run.selfplay(it)` produces shards under `data/iter_NNNN/`
  through `ShardWriter` and returns the metrics dict; a remote or multi-process implementation
  only has to write the same shards (any prefix: `c00`, `r00`, a host name) and the same
  metrics keys. The window (`Run.window_shards`) reads every shard of an iteration directory.
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
* `widen`: new actions get zero generic weights, new features (after the old ones) are new
  globals with zero weights; the layout stays. Not in `zero.webexport`. CUDA: float32 eager
  (no autocast, no CUDA graphs; not measured: the GPU was busy).

Tests: `tests/test_net_data.py` (masking of absent slots, chunking / batch independence, the
pointer logits land on the slot actions, identity digits, save / load / widen, a training step
reaches every part, layout validation, the toy's layout `package.TOY_TOKENS` through the
evaluator), `tests/test_loop.py` (the loop with arch 'tokens' on the toy).

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
