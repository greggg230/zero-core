"""zerodash - one page for every AlphaZero project's training runs (DESIGN.md 3).

    cd dashboard && python -m zerodash --host 0.0.0.0 --port 5200      (hub id zero_dash)
    python -m zerodash --fixtures                                      (the snapshot in fixtures/)

It reads run files only (config.json, metrics.jsonl, log.txt, supervisor.log, progress
files, run_state.json / supervisor_state.json, PAUSE / STOP, elo/ratings.json) and never
imports from the game repos. The one thing it writes is the PAUSE file of the PAUSE contract
(`<run>/PAUSE` = {"time": <epoch>, "by": "<who>"}); it never writes or removes STOP.

Modules: config (projects.json), runfiles (reading run files), procs (is a process alive),
state (one state per run), summary (what a run card shows), remote (runs on another host
over SSH), control (pause / resume / pause-all / resume-all), machine (GPU / CPU / disks /
Mac strip), app (Flask: page + JSON API), fixtures (the frozen snapshot for dev and tests).
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(PACKAGE_DIR)
# The machine's own projects file stays out of the repository (it names the machine's paths and hosts; AGENTS.md
# "Private details"): $ZERODASH_CONFIG, else ~/.config/zero/projects.json. projects.example.json shows the format.
DEFAULT_CONFIG = os.path.expanduser(os.environ.get('ZERODASH_CONFIG') or os.path.join('~', '.config', 'zero',
                                                                                     'projects.json'))
EXAMPLE_CONFIG = os.path.join(DASHBOARD_DIR, 'projects.example.json')
FIXTURES_CONFIG = os.path.join(DASHBOARD_DIR, 'projects.fixtures.json')
FIXTURES_DIR = os.path.join(DASHBOARD_DIR, 'fixtures')
