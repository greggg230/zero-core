# Cloud task: cross-game Zero dashboard (zerodash, port 5200)

Start a Claude Code cloud session on `greggg230/zero-core` (outcome branch `dashboard`) with this prompt:

---

Build the **cross-game Zero dashboard** (`zerodash`) for the user's AlphaZero training projects. Read `DESIGN.md` in this repo first: section 3 is your spec, sections 1 and 4 give context. Work on branch `dashboard` and push it (never push to `main`; no pull request). Everything goes under `dashboard/`: package `dashboard/zerodash/`, started on the user's Windows desktop as `cd dashboard && python -m zerodash --host 0.0.0.0 --port 5200` (that exact command is registered in the user's hub), tests under `dashboard/tests/`.

## Inputs you have (the dashboard will run on the desktop; you are in the cloud and can't reach it)
- `dashboard/fixtures/<project>/runs/<run>/`: real snapshots of run files from all four projects (St. Pete `az5`, `az6_cq`; Dominion `all1`, `rand6`; Caverna `cvz2p`, `cvz2p_cq`; Glass Road `grz2p_cq`, `grz2p_cq_s512`): `config.json`, tails of `metrics.jsonl` / `log.txt` / `supervisor.log` / `stdout.txt`, `progress.json`, `run_state.json`, `supervisor_state.json`, `PAUSE` files where present, `elo/`, `models/index.json`, `models_listing.json` (file names only). The formats differ between projects: derive state detection from them and from `dashboard/fixtures/reference_code/<project>/` (each game's own dashboard, runctl / pause tools and supervisor).
- Build `dashboard/projects.json` for the desktop with these real roots (Windows paths): St. Pete `C:/Users/Gregg Keithley/stpete-zero/runs` (its `az6_vis` / `az6_cq` entries are directory junctions to `E:/stpete-zero/runs/...` - never delete them, never rm -rf), Dominion `C:/Users/Gregg Keithley/dominion-zero/runs`, Caverna `C:/Users/Gregg Keithley/caverna-zero/runs`, Glass Road `C:/Users/Gregg Keithley/glassroad-zero/runs`; plus a fixtures config for tests/dev.

## What it shows (phone-first: the user mostly opens it on a phone over Tailscale)
- One card per project (St. Pete Zero, Dominion Zero, Caverna Zero, Glass Road Zero) listing its runs: name, state (running / pausing / paused / stopping / stopped / crashed or restart-looping / stale = supervisor alive but no progress for a long time), current iteration + phase + age of the last finished iteration, the latest yardstick evaluation (win %, margin), latest Elo + a small sparkline of Elo by iteration, self-play throughput, who paused it and since when. Hide old finished runs by default (e.g. not modified in 3 days and no live supervisor) with a "show all" toggle. Each project card links to the game's own dashboard at `http://desktop-j05412i:<port>` (St. Pete 5230, Dominion 5240, Caverna 5260, Glass Road 5280).
- Machine strip: GPU memory / power / utilisation (`nvidia-smi --query-gpu=memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader,nounits`), CPU load (psutil), free disk on C: and E: (warn under 15 GB on C:), and the user's Mac load (`ssh -o BatchMode=yes -o ConnectTimeout=5 greggkeithley@100.95.248.116 uptime`, cached ~60 s, run in the background, never blocking a page). All of these must degrade gracefully when the tool is missing (as in the cloud): mock them in tests.
- Auto-refresh every ~15 s via a JSON API.

## Controls
- Pause / Resume per run, **Pause all** and **Resume all**, each with a confirm step. PAUSE contract (DESIGN.md 3, and see `reference_code/glassroad/grz/az/runctl.py`): write `<run>/PAUSE` = `{"time": <epoch>, "by": "zerodash ..."}`; pause-all uses `"by": "zerodash pause-all <id>"`; **Resume all removes only PAUSE files a pause-all wrote** (a run paused by hand or another tool stays paused); per-run Resume removes that run's PAUSE. Never write or remove STOP files (stopping a run stays manual).
- Pause support is a per-project/per-run capability in the config: Glass Road, Caverna, Dominion support PAUSE; St. Pete runs currently honour only STOP (PAUSE support is being added and will go live when its `az6` runs restart) - show "no pause (STOP only)" for St. Pete and make it a one-line config flip later.
- Remote runs: St. Pete's az6 runs will move to the user's Mac. Let a run be remote in `projects.json` (`"ssh": "greggkeithley@100.95.248.116", "path": "~/stpete-zero/runs/<run>"`) with a local mirror directory for reading (a separate sync copies metrics/log/progress there), and pause/resume executed over SSH. Implement and unit-test that path with SSH mocked.

## Quality bar
- Tests (pytest) for: state detection on every fixture run, pause / resume / pause-all / resume-all semantics on temporary copies of the fixtures (incl. "resume-all leaves hand pauses alone"), the remote path with SSH mocked, the API JSON shape.
- Screenshots: run it on the fixtures config and take Playwright screenshots at phone width (390 px) and desktop width, light and dark; commit them under `dashboard/docs/` so the orchestrator can look without running it.
- Plain HTML/CSS/vanilla JS, Flask, no build step. Windows-compatible paths and file operations (the desktop is Windows 11, Python 3.13 with flask + psutil installed system-wide; don't require new packages beyond those).
- Commit messages end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: files, how state detection works per project, test results, screenshot paths, the exact launch command, and anything the orchestrator must do on the desktop (e.g. verifying against live runs, the St. Pete pause flip, the Mac sync path).
