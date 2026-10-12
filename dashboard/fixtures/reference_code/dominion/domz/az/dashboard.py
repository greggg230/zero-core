"""Training dashboard: python -m domz.az.dashboard [--host 0.0.0.0] [--port 5240] [--runs DIR]

Reads <runs>/*/metrics.jsonl, config.json, log.txt, supervisor.log, elo/ and the self-play
progress files of the iteration in flight. Binds 0.0.0.0 so it is reachable over Tailscale
(hub project dominion_zero, port 5240). Charts are drawn at the element's real pixel width
(redrawn on resize), so text is never stretched on phones.

Run control (domz/az/runctl.py, docs/az.md 1.4): each run's state (running / pausing /
paused / resuming / stopped ...) and Pause / Resume buttons: POST /api/run/<name>/pause
creates <run>/PAUSE, POST /api/run/<name>/resume removes it (JSON body required; only run
directories under --runs); GET /api/run/<name>/control is the state alone.
"""
import argparse
import glob
import json
import os
import re
import time

from flask import Flask, jsonify, request, Response

from . import runctl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, 'runs')
EXPS = os.path.join(ROOT, 'experiments')         # tools/experiment.py's experiment dirs (docs/az.md 10)
app = Flask(__name__)
app.json.sort_keys = False          # fixed kingdoms in the run's order


def _metrics(run):
    rows = []
    p = os.path.join(RUNS, run, 'metrics.jsonl')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass                      # a line being written right now
    return rows


def _tail(path, n):
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        return f.read().decode('utf-8', 'replace').splitlines()[-n:]


def _progress(d, cfg):
    """Self-play progress of the newest data dir (workers.write_progress files)."""
    root = cfg.get('data_dir') or os.path.join(d, 'data')
    dirs = sorted(glob.glob(os.path.join(root, 'iter_*')))
    if not dirs:
        return None
    files = glob.glob(os.path.join(dirs[-1], 'progress_*.json'))
    if not files:
        return None
    tot = {'iter': int(os.path.basename(dirs[-1])[5:9]), 'games_done': 0, 'games': 0, 'samples': 0,
           'positions': 0, 'started': None, 'time': 0, 'done': True, 'remote_games_done': 0, 'remote_games': 0}
    for p in files:
        try:
            j = json.load(open(p))
        except (OSError, ValueError):
            continue
        for k in ('games_done', 'games', 'samples', 'positions'):
            tot[k] += j.get(k, 0)
        if j.get('remote'):                       # mirrored from the remote host (docs/az.md 9)
            tot['remote'] = j['remote']
            tot['remote_games_done'] += j.get('games_done', 0)
            tot['remote_games'] += j.get('games', 0)
        if j.get('rolling'):                      # the persistent worker (docs/az.md 2.9)
            tot['rolling'] = True
            tot['in_flight'] = j.get('in_flight')
        tot['started'] = min(tot['started'] or j['started'], j['started'])
        tot['time'] = max(tot['time'], j['time'])
        tot['done'] = tot['done'] and bool(j.get('done'))
    return tot


RUN_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


def _run_dir(name):
    """The directory of a known run (a plain name of a run directory under RUNS), else None:
    no separators, no '..', no other locations."""
    if not RUN_NAME.match(name or ''):
        return None
    try:                                   # exactly a listed name (Windows also opens 'all1.' / 'ALL1' as all1)
        if name not in os.listdir(RUNS):
            return None
    except OSError:
        return None
    d = os.path.join(RUNS, name)
    if os.path.dirname(os.path.abspath(d)) != os.path.abspath(RUNS) or not runctl.is_run_dir(d):
        return None
    return d


def _control(d):
    st = runctl.run_status(d)
    st['now'] = time.time()
    return st


@app.route('/api/run/<name>/control')
def control(name):
    d = _run_dir(name)
    if d is None:
        return jsonify({'error': f'no run named {name!r}'}), 404
    return jsonify(_control(d))


@app.route('/api/run/<name>/<action>', methods=['POST'])
def control_action(name, action):
    """Pause / resume: a JSON body is required (a plain cross-site form cannot send one)."""
    d = _run_dir(name)
    if d is None or action not in ('pause', 'resume'):
        return jsonify({'error': f'no run named {name!r} or no action {action!r}'}), 404
    if not isinstance(request.get_json(silent=True), dict):
        return jsonify({'error': 'send a JSON body'}), 400
    if action == 'pause':
        changed = runctl.request_pause(d, by=f'dashboard {request.remote_addr}')
    else:
        changed = runctl.request_resume(d)
    return jsonify({'ok': True, 'changed': changed, 'control': _control(d)})


@app.route('/api/runs')
def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, '*'))):
        if os.path.exists(os.path.join(d, 'metrics.jsonl')):
            out.append({'name': os.path.basename(d), 'mtime': os.path.getmtime(os.path.join(d, 'metrics.jsonl'))})
    out.sort(key=lambda r: -r['mtime'])
    return jsonify(out)


KINGDOM_CATS = ('normal', 'forced', 'unforced')


def kingdom_series(rows):
    """The kingdom cards the self-play players gained / bought, per iteration (docs/az.md 2.10,
    `selfplay.strategy.<category>.kingdom_gained / kingdom_bought`): {'bins': the histograms'
    bin labels, 'iters': [{'iter', 'explore' (the iteration explored), 'cats': {category:
    {'gained': {...}, 'bought': {...}}}}]}. Iterations without the stats (older builds) are
    left out."""
    out, bins = [], None
    for m in rows:
        st = (m.get('selfplay') or {}).get('strategy') or {}
        cats = {}
        for c in KINGDOM_CATS:
            d = st.get(c) or {}
            if isinstance(d.get('kingdom_gained'), dict):
                cats[c] = {k: d[k] for k in ('kingdom_gained', 'kingdom_bought') if isinstance(d.get(k), dict)}
                cats[c] = {k[8:]: v for k, v in cats[c].items()}            # 'gained' / 'bought'
        if not cats or m.get('iter') is None:
            continue
        bins = st.get('kingdom_bins') or bins
        out.append({'iter': m['iter'], 'explore': bool(st.get('explore_games')), 'cats': cats})
    out.sort(key=lambda p: p['iter'])
    return {'bins': bins, 'iters': out}


@app.route('/api/run/<name>/kingdom')
def run_kingdom(name):
    """The kingdom-card series alone (the run page's comparison with another run)."""
    d = _run_dir(name)
    if d is None:
        return jsonify({'error': f'no run named {name!r}'}), 404
    return jsonify({'run': name, 'kingdom': kingdom_series(_metrics(name))})


@app.route('/api/run/<name>')
def run(name):
    d = os.path.join(RUNS, os.path.basename(name))
    cfg = {}
    if os.path.exists(os.path.join(d, 'config.json')):
        cfg = json.load(open(os.path.join(d, 'config.json')))
    log = _tail(os.path.join(d, 'log.txt'), 60)
    ratings = {}
    rp = os.path.join(d, 'elo', 'ratings.json')
    if os.path.exists(rp):
        ratings = json.load(open(rp))
    pairs, failed_pairs = {}, {}
    pp = os.path.join(d, 'elo', 'results.json')
    if os.path.exists(pp):
        res = json.load(open(pp))
        pairs, failed_pairs = res.get('pairs', {}), res.get('failed', {})
    lp = os.path.join(d, 'log.txt')
    status = {'log_age_s': time.time() - os.path.getmtime(lp) if os.path.exists(lp) else None,
              'progress': _progress(d, cfg), 'now': time.time()}
    remote = None
    rp = os.path.join(d, 'remote', 'state.json')
    if os.path.exists(rp):
        try:
            remote = json.load(open(rp, encoding='utf-8'))
        except (OSError, ValueError):
            remote = None
    metrics = _metrics(os.path.basename(name))
    return jsonify({'config': cfg, 'metrics': metrics, 'kingdom': kingdom_series(metrics), 'log': log, 'ratings': ratings,
                    'pairs': pairs, 'failed_pairs': failed_pairs, 'supervisor': _tail(os.path.join(d, 'supervisor.log'), 12), 'status': status,
                    'stopping': os.path.exists(os.path.join(d, 'STOP')), 'remote': remote,
                    'control': _control(d) if runctl.is_run_dir(d) else None})


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


EXP_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')


@app.route('/api/experiments')
def experiments():
    from . import expview
    return jsonify(expview.list_experiments(EXPS))


@app.route('/api/experiment/<name>')
def experiment(name):
    from . import expview
    if not EXP_NAME.match(name or '') or name not in os.listdir(EXPS):
        return jsonify({'error': f'no experiment named {name!r}'}), 404
    x = expview.experiment(EXPS, RUNS, name)
    if x is None:
        return jsonify({'error': f'no experiment named {name!r}'}), 404
    x['alerts'] = expview.ledger_alerts(EXPS)
    return jsonify(x)


@app.route('/experiments')
def experiments_page():
    return Response(EXP_PAGE, mimetype='text/html')


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dominion Zero</title>
<style>
:root{color-scheme:light;--bg:#f4f3ef;--surface-1:#fcfcfb;--line:#e4e2dc;--grid:#ecebe6;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#8a8984;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;--ref:#a7a59e;--good:#1a7f37;--warn:#9a6700;--act-pause:#c2410c;--act-resume:#1d63c4;--on-act:#fff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;--ref:#6b6a65;--good:#4ac26b;--warn:#d4a72c;--act-pause:#c2410c;--act-resume:#1d63c4;--on-act:#fff}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;--ref:#6b6a65;--good:#4ac26b;--warn:#d4a72c;--act-pause:#c2410c;--act-resume:#1d63c4;--on-act:#fff}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text-primary);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:0}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-secondary);margin:22px 0 8px;font-weight:600}
.sub{color:var(--text-secondary);font-size:13px;margin-top:2px}
.status{font-size:13px;color:var(--text-secondary);margin:-6px 0 12px}
.status b{color:var(--text-primary);font-weight:600}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:14px}
select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface-1);color:var(--text-primary);max-width:100%}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:10px 12px;min-width:0}
.tile .l{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-secondary)}
.tile .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.15}
.tile .n{font-size:12.5px;color:var(--muted)}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:14px;min-width:0}
.card h2{font-size:14px;margin:0}
.card .cap{font-size:12.5px;color:var(--text-secondary);margin:2px 0 6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-bottom:14px}
.grid2 .card{margin-bottom:0}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--text-secondary);margin:2px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px}
.legend i.bar{height:10px;width:10px;border-radius:2px}
svg{display:block;width:100%;overflow:visible;touch-action:pan-y}
svg text{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 6px 18px rgba(0,0,0,.18);z-index:9;color:var(--text-primary);max-width:260px}
.tip b{font-variant-numeric:tabular-nums}
.tip i{display:inline-block;width:10px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-secondary);text-align:right;padding:4px 6px;font-weight:600;white-space:nowrap}
td{text-align:right;padding:4px 6px;border-top:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.up{color:var(--good)} td.down{color:var(--warn)}
.tbl{overflow-x:auto}
pre{font-size:11.5px;white-space:pre-wrap;word-break:break-word;margin:0;color:var(--text-secondary);max-height:260px;overflow:auto}
details summary{cursor:pointer;font-weight:600;font-size:14px}
.bar{height:6px;border-radius:3px;background:var(--grid);overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:var(--series-1)}
.warn{color:var(--warn)} .rem{font-size:13px;color:var(--text-secondary);margin:-6px 0 12px}
.rem b{color:var(--text-primary);font-weight:600} .rem .warn b{color:var(--warn)}
.ctl{display:flex;align-items:center;justify-content:space-between;gap:10px 14px;flex-wrap:wrap;background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:0 0 12px}
.ctl-state{display:flex;align-items:center;gap:6px 10px;flex-wrap:wrap;min-width:0;flex:1 1 240px}
.ctl-state b{font-size:16px;font-weight:650}
.ctl-detail{font-size:13px;color:var(--text-secondary);flex-basis:100%;margin-left:20px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;background:var(--muted)}
.dot.running{background:var(--good)} .dot.paused{background:var(--act-resume)}
.dot.pausing,.dot.resuming,.dot.starting,.dot.restarting,.dot.stopping{background:var(--warn);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.3}}
@media (prefers-reduced-motion:reduce){.dot{animation:none!important}}
.hdr-r{display:flex;align-items:center;gap:8px 14px;flex-wrap:wrap}
.navlink{font-size:14px;font-weight:600;color:var(--series-1);text-decoration:none;padding:10px 4px;min-height:44px;display:inline-flex;align-items:center}
.navlink:hover{text-decoration:underline}
.btn{font:inherit;font-weight:650;font-size:15px;min-height:44px;min-width:128px;padding:10px 20px;border-radius:10px;border:0;cursor:pointer;color:var(--on-act);background:var(--act-resume);touch-action:manipulation}
.btn.pause{background:var(--act-pause)}
.btn:disabled{opacity:.55;cursor:default}
.btn:focus-visible{outline:3px solid var(--series-1);outline-offset:2px}
@media (max-width:520px){.btn{flex:1 1 100%}}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:0 0 10px;background:var(--surface-1)}
.seg button{font:inherit;font-size:14px;font-weight:600;min-height:44px;padding:8px 16px;border:0;background:transparent;color:var(--text-secondary);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--series-1);color:var(--on-act)}
.seg button:focus-visible{outline:3px solid var(--series-1);outline-offset:-3px}
.kgbar{display:flex;align-items:center;justify-content:space-between;gap:4px 14px;flex-wrap:wrap}
.kgbar .seg{margin:0}
.kgbar select{min-height:44px}
:root{--ord-0:var(--ref);--ord-1:#86b6ef;--ord-2:#5598e7;--ord-3:#2a78d6;--ord-4:#1c5cab;--ord-5:#104281}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--ord-1:#184f95;--ord-2:#256abf;--ord-3:#3987e5;--ord-4:#6da7ec;--ord-5:#9ec5f4}}
:root[data-theme="dark"]{--ord-1:#184f95;--ord-2:#256abf;--ord-3:#3987e5;--ord-4:#6da7ec;--ord-5:#9ec5f4}
</style></head><body><div class="wrap">
<header><div><h1>Dominion Zero</h1><div class="sub" id="sub">AlphaZero training for Dominion</div></div>
<div class="hdr-r"><a class="navlink" id="explink" href="experiments" hidden>Experiments</a><label class="sub">Run <select id="runsel"></select></label></div></header>
<div class="ctl" id="ctl" hidden><div class="ctl-state"><span class="dot" id="ctl-dot"></span><b id="ctl-label"></b><span class="ctl-detail" id="ctl-detail"></span></div>
<button type="button" class="btn" id="ctl-btn" hidden></button></div>
<div class="status" id="status"></div>
<div class="rem" id="remote" hidden></div>
<div class="tiles" id="tiles"></div>

<h3>Strength</h3>
<div class="card"><h2>Elo by iteration</h2>
<div class="cap">Big Money Ultimate (BMU) is fixed at 1000; the heuristic bot and the random bot are rated once against it (lines). Each checkpoint plays rating matches against the checkpoints 1, 2, 4, 8, 16 and 32 iterations before it, plus its evaluation games against the bots; all games are refit together after every iteration. Band = 95% interval.</div>
<svg id="elo" style="height:300px" role="img" aria-label="Elo by iteration"></svg></div>
<div class="card"><h2>Win rate against the bots (random kingdoms)</h2>
<div class="cap">Network searching with the eval sims against each bot, seats alternated, kingdoms drawn like self-play's. Hover for the 95% interval and game count.</div>
<svg id="win" style="height:300px" role="img" aria-label="Win rate against the bots by iteration"></svg></div>
<div class="card" id="kcard" hidden><h2>Fixed benchmark kingdoms</h2>
<div class="cap" id="kcap">The rulebook's named kingdoms, the same every iteration (not part of the Elo fit). Average win rate over all of them.</div>
<svg id="kwin" style="height:240px" role="img" aria-label="Win rate on the fixed kingdoms by iteration"></svg>
<div class="tbl" id="ktable" style="margin-top:8px"></div></div>
<div class="grid2">
<div class="card"><h2>VP margin against the bots</h2><div class="cap">Network's final VP minus the bot's, average over the evaluation games (a learning signal while the win rate is still 0)</div><svg id="mgn" style="height:180px" role="img" aria-label="VP margin by iteration"></svg></div>
<div class="card"><h2>Each checkpoint against the one before it</h2><div class="cap">Iteration N vs N&minus;1, both searching with the eval sims. Above 50% = the newer network is stronger. Band = 95% interval.</div><svg id="prev" style="height:180px" role="img" aria-label="Win rate against the previous iteration"></svg></div>
</div>
<div class="card" id="refcard" hidden><h2 id="ref-t">Against the reference network</h2>
<div class="cap">This run's checkpoints against a fixed network from another run (each side with its own run's search settings). Band = 95% interval.</div>
<svg id="refc" style="height:220px" role="img" aria-label="Win rate against the reference network"></svg></div>
<div class="card"><h2 id="h2h-t">Newest checkpoint: head to head</h2>
<div class="cap">Every opponent the newest rated checkpoint has played. "Elo predicts" is the win rate the fitted ratings expect, as a consistency check.</div>
<div class="tbl" id="h2h"></div></div>

<h3>Games</h3>
<div class="grid2">
<div class="card"><h2>Game length</h2><div class="cap">Rounds per game (turns per player), self-play and evaluation games against BMU</div><svg id="turns" style="height:180px" role="img" aria-label="Rounds per game"></svg></div>
<div class="card"><h2>Turn-cap hit rate</h2><div class="cap">Share of games stopped at the turn cap instead of ending on Provinces or piles</div><svg id="capped" style="height:180px" role="img" aria-label="Turn-cap hit rate"></svg></div>
<div class="card"><h2>Decisions per game</h2><div class="cap">Multi-option decisions searched per self-play game (both seats)</div><svg id="dpg" style="height:180px" role="img" aria-label="Decisions per game"></svg></div>
<div class="card"><h2>Self-play average final VP</h2><div class="cap">Per player, per game</div><svg id="sc" style="height:180px" role="img" aria-label="Average VP"></svg></div>
</div>

<div id="kgsec">
<h3>Kingdom cards</h3>
<div class="card"><div class="kgbar"><div class="seg" role="group" aria-label="Count the kingdom cards"><button type="button" id="km-gained" aria-pressed="true">Gained</button><button type="button" id="km-bought" aria-pressed="false">Bought</button></div>
<label class="sub">Compare with <select id="kgcmp"><option value="">no other run</option></select></label></div>
<div class="cap" id="kgcap" style="margin-top:8px">Kingdom cards each self-play player acquired during the game: <b>gained</b> = bought or gained any other way (Workshop, Messenger ...), <b>bought</b> = the purchases among them. A card trashed later still counts; the starting cards never do. Kingdom card = a card of the game's kingdom piles (incl. split / mixed piles such as Knights or Castles, the Bane) or one from the Black Market deck - not Copper ... Province, Curse, Potion, Platinum, Colony, Ruins, Shelters, Spoils, Heirlooms, Prizes, Horses, Spirits or other non-supply cards. Iterations recorded before these stats existed are not shown. "Compare with" adds another run's normal games as a dashed line named after it.</div></div>
<div class="grid2">
<div class="card"><h2>Distinct kingdom cards per player</h2><div class="cap">Different kingdom cards a player took in a game, average (two different Knights count twice)</div><svg id="kgd" style="height:200px" role="img" aria-label="Distinct kingdom cards per player by iteration"></svg></div>
<div class="card"><h2>Kingdom cards per player</h2><div class="cap">Every copy counts: 2 Villages + 1 Cellar = 3; average per player and game</div><svg id="kgt" style="height:200px" role="img" aria-label="Kingdom cards per player by iteration"></svg></div>
</div>
<div class="card"><h2>Players by number of distinct kingdom cards</h2><div class="cap" id="kgmix-cap">Share of the players of normal games that took 0, 1, 2, 3, 4 or 5+ different kingdom cards</div><svg id="kgmix" style="height:220px" role="img" aria-label="Distribution of distinct kingdom cards per player by iteration"></svg></div>
<div class="grid2">
<div class="card"><h2 id="kgh1-t">Distinct kingdom cards gained, newest iteration</h2><div class="cap" id="kgh1-cap">Share of the players of normal games</div><svg id="kgh1" style="height:180px" role="img" aria-label="Histogram of distinct kingdom cards per player"></svg></div>
<div class="card"><h2 id="kgh2-t">Kingdom cards gained (every copy), newest iteration</h2><div class="cap" id="kgh2-cap">Share of the players of normal games</div><svg id="kgh2" style="height:180px" role="img" aria-label="Histogram of kingdom cards per player"></svg></div>
</div>
</div>

<h3>Training</h3>
<div class="grid2">
<div class="card"><h2>Policy loss</h2><div class="cap">Cross-entropy against the search's visit counts (last 200 steps of the iteration)</div><svg id="lp" style="height:170px" role="img" aria-label="Policy loss"></svg></div>
<div class="card"><h2>Value loss</h2><div class="cap">Cross-entropy against the value target (result blended with the root search estimate)</div><svg id="lv" style="height:170px" role="img" aria-label="Value loss"></svg></div>
<div class="card"><h2>Margin loss</h2><div class="cap">Squared error of the predicted VP margin (VP difference / 40 for 2 players); small while games end at the cap with few VP</div><svg id="lm" style="height:170px" role="img" aria-label="Margin loss"></svg></div>
<div class="card" id="auxcard" hidden><h2>Auxiliary losses</h2><div class="cap">Poisson NLL of the "where is this game heading" heads (docs/az.md 2.13): cards of each pile every seat gains from the position to the end, per pile and seat; rounds left, per sample</div><svg id="laux" style="height:170px" role="img" aria-label="Auxiliary losses"></svg></div>
<div class="card"><h2>Training samples</h2><div class="cap">New samples per iteration, and the window trained on (all of it, or the part that fit the RAM budget)</div><svg id="samples" style="height:170px" role="img" aria-label="Samples"></svg></div>
</div>

<h3>Throughput</h3>
<div class="grid2">
<div class="card"><h2>Self-play games per hour</h2><div class="cap">Throughput of the self-play phase</div><svg id="gph" style="height:170px" role="img" aria-label="Games per hour"></svg></div>
<div class="card"><h2>Network evaluations per second</h2><div class="cap">Search leaves evaluated during self-play (all machines over the phase; the desktop over its own part)</div><svg id="pps" style="height:170px" role="img" aria-label="Leaves per second"></svg></div>
<div class="card"><h2>Average batch</h2><div class="cap">Leaves per network call in the desktop's self-play; the line is a full batch (games in flight / groups). Below it: the tail of a self-play task</div><svg id="abatch" style="height:170px" role="img" aria-label="Average batch size"></svg></div>
<div class="card" id="rollcard"><h2>Rolling self-play</h2><div class="cap">Share of an iteration's samples searched by an earlier network (games span iterations), and the search work redone because every search in progress restarts with the new network</div><svg id="stale" style="height:170px" role="img" aria-label="Stale samples and restart cost"></svg></div>
<div class="card" id="loopcard"><h2>Loop guard</h2><div class="cap">Longest turn of the iteration's self-play games, in steps (every decision of the turn, all players): the engine's loop guard stops a turn at 1,000 and ends the game at 2,000 (docs/engine.md 2). Rolling: age of the oldest game still in progress at the end of the phase, in iterations - a game that never ends would grow without bound</div><svg id="loops" style="height:150px" role="img" aria-label="Longest turn per iteration"></svg><svg id="ages" style="height:130px" role="img" aria-label="Oldest game in progress"></svg></div>
<div class="card"><h2>Minutes per iteration</h2><div class="cap">Where an iteration's time goes</div><svg id="mins" style="height:170px" role="img" aria-label="Minutes per iteration by phase"></svg></div>
<div class="card"><h2>Training speed</h2><div class="cap">Samples per second through the optimizer (minibatch steps x batch size / training time)</div><svg id="tps" style="height:170px" role="img" aria-label="Training samples per second"></svg></div>
</div>

<div class="card"><details><summary>Table</summary><div class="tbl" id="table"></div></details></div>
<div class="card"><details><summary>Log tail</summary><pre id="log"></pre></details></div>
<div class="card"><details><summary>Supervisor</summary><pre id="sup"></pre></details></div>
</div><div class="tip" id="tip" hidden></div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let RUN=null, DATA=null;
function ns(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e}
const pct=x=>(100*x).toFixed(1)+'%';
function big(v){const a=Math.abs(v);if(a===0)return '0';return a>=1e6?(v/1e6).toFixed(a>=1e7?0:1)+'M':a>=1e4?Math.round(v/1e3)+'k':a>=1e3?(v/1e3).toFixed(1)+'k':a>=100?Math.round(v).toString():a>=10?v.toFixed(1):a>=1?v.toFixed(2):v.toFixed(3)}
function niceStep(r,n){const raw=r/n,p=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/p;return (f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*p}
// nice ticks for [lo, hi]; eLo / eHi widen a free end of the range to a whole step
function ticksFor(lo,hi,n,eLo,eHi){const s=niceStep(hi-lo||1,n);if(eLo)lo=Math.floor(lo/s+1e-9)*s;if(eHi)hi=Math.ceil(hi/s-1e-9)*s;
  const out=[];for(let v=Math.ceil(lo/s-1e-9)*s;v<=hi+1e-9*s;v+=s)out.push(+v.toFixed(10));return {ticks:out,lo,hi}}
function width(svg){const r=svg.getBoundingClientRect().width;return Math.max(240,r||svg.parentNode.clientWidth||600)}
// series: [{label, color (css var), pts:[{x,y,lo?,hi?,tip?}]}]; opt: {y0,y1,fmt,refs,tipx}
function chart(svg,series,opt){
  opt=opt||{};svg.innerHTML='';
  let leg=svg.previousElementSibling&&svg.previousElementSibling.classList.contains('legend')?svg.previousElementSibling:null;
  series=series.filter(s=>s.pts.length);
  if(series.length>1){if(!leg){leg=document.createElement('div');leg.className='legend';svg.parentNode.insertBefore(leg,svg)}
    leg.innerHTML=series.map(s=>`<span><i style="background:${css(s.color)}"></i>${s.label}</span>`).join('')}else if(leg)leg.innerHTML='';
  const W=width(svg),H=parseFloat(svg.style.height)||160;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  if(!series.length){const t=ns('text',{x:W/2,y:H/2,'text-anchor':'middle'});t.textContent='No data yet';svg.appendChild(t);return}
  const all=series.flatMap(s=>s.pts),xs=all.map(p=>p.x);let x0=Math.min(...xs),x1=Math.max(...xs);if(x1===x0){x0-=.5;x1+=.5}
  const endLab=series.length>1&&series.length<=4&&W>=480;
  let fmt=opt.fmt||(v=>big(v));
  let ylo=opt.y0,yhi=opt.y1;
  if(ylo==null||yhi==null){const ys=all.flatMap(p=>[p.y,p.lo??p.y,p.hi??p.y]).concat((opt.refs||[]).filter(r=>r.fit).map(r=>r.y));let lo=Math.min(...ys),hi=Math.max(...ys);if(hi-lo<1e-9){const d=Math.abs(hi)*.1||1;lo-=d;hi+=d}const pad=(hi-lo)*.08;ylo=ylo??lo-pad;yhi=yhi??hi+pad}
  const tk=ticksFor(ylo,yhi,4,opt.y0==null,opt.y1==null);ylo=tk.lo;yhi=tk.hi;
  if(!opt.fmt){const st=tk.ticks.length>1?Math.abs(tk.ticks[1]-tk.ticks[0]):1,dec=Math.min(4,(String(+st.toPrecision(6)).split('.')[1]||'').length);fmt=v=>Math.abs(v)>=1e4?big(v):v.toFixed(dec)}
  const tl=tk.ticks;
  const L=Math.max(30,8+6.2*Math.max(...tl.map(v=>fmt(v).length))),R=endLab?78:12,T=10,B=24;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-ylo)/(yhi-ylo))*(H-T-B);
  for(const v of tl){svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:css('--grid'),'stroke-width':1}));
    const t=ns('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'});t.textContent=fmt(v);svg.appendChild(t)}
  const nt=Math.min(8,Math.floor((W-L-R)/(opt.xfmt?54:36)));const step=Math.max(1,Math.ceil((x1-x0)/Math.max(1,nt)));
  if(opt.xfmt){for(const v of ticksFor(x0,x1,Math.max(1,nt)).ticks){if(v<x0-1e-9||v>x1+1e-9)continue;const t=ns('text',{x:X(v),y:H-6,'text-anchor':'middle'});t.textContent=opt.xfmt(v);svg.appendChild(t)}}
  else for(let v=Math.ceil(x0);v<=x1;v+=step){const t=ns('text',{x:X(v),y:H-6,'text-anchor':'middle'});t.textContent=v;svg.appendChild(t)}
  for(const r of (opt.refs||[])){if(r.y<ylo||r.y>yhi)continue;svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(r.y),y2:Y(r.y),stroke:css('--ref'),'stroke-width':1,'stroke-dasharray':'4 3'}));
    const t=ns('text',r.left?{x:L+4,y:Y(r.y)-4,'text-anchor':'start'}:{x:W-R-4,y:Y(r.y)-4,'text-anchor':'end'});t.textContent=r.label;svg.appendChild(t)}
  for(const s of series){const col=css(s.color),pts=s.pts;
    if(pts[0].lo!=null&&s.band!==false){let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.hi));for(let i=pts.length-1;i>=0;i--)d+='L'+X(pts[i].x)+','+Y(pts[i].lo);svg.appendChild(ns('path',{d:d+'Z',fill:col,opacity:series.length>1?.08:.12}))}
    let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.y));
    const pa={d,fill:'none',stroke:col,'stroke-width':s.strong?3:2,'stroke-linejoin':'round','stroke-linecap':'round'};if(s.dash)pa['stroke-dasharray']=s.dash;
    svg.appendChild(ns('path',pa));
    if(pts.length<=40)for(const p of pts)svg.appendChild(ns('circle',{cx:X(p.x),cy:Y(p.y),r:pts.length<=12?3:2,fill:col}));
    const last=pts[pts.length-1];svg.appendChild(ns('circle',{cx:X(last.x),cy:Y(last.y),r:4.5,fill:col,stroke:css('--surface-1'),'stroke-width':2}))}
  if(endLab){const labs=series.map(s=>({s,y:Y(s.pts[s.pts.length-1].y)})).sort((a,b)=>a.y-b.y);for(let i=1;i<labs.length;i++)labs[i].y=Math.max(labs[i].y,labs[i-1].y+13);
    for(const l of labs){const t=ns('text',{x:W-R+8,y:l.y+4,'text-anchor':'start'});t.textContent=l.s.label;svg.appendChild(t)}}
  const cross=ns('line',{y1:T,y2:H-B,stroke:css('--ref'),'stroke-width':1,visibility:'hidden'});svg.appendChild(cross);
  const dots=series.map(s=>{const c=ns('circle',{r:5,fill:css(s.color),stroke:css('--surface-1'),'stroke-width':2,visibility:'hidden'});svg.appendChild(c);return c});
  const hit=ns('rect',{x:L,y:0,width:W-L-R,height:H-B,fill:'transparent'});svg.appendChild(hit);
  const tip=$('tip');
  const show=ev=>{const b=svg.getBoundingClientRect();const fx=(ev.clientX-b.left)/b.width*W;let bx=xs[0];for(const x of xs)if(Math.abs(X(x)-fx)<Math.abs(X(bx)-fx))bx=x;
    cross.setAttribute('x1',X(bx));cross.setAttribute('x2',X(bx));cross.setAttribute('visibility','visible');
    let h=`${opt.xlabel||'Iteration'} ${opt.xfmt?opt.xfmt(bx):bx}`;
    series.forEach((s,i)=>{const p=opt.nearest?s.pts.reduce((b,q)=>Math.abs(X(q.x)-X(bx))<Math.abs(X(b.x)-X(bx))?q:b,s.pts[0]):s.pts.find(p=>p.x===bx);
      if(!p||(opt.nearest&&Math.abs(X(p.x)-X(bx))>Math.max(24,(W-L-R)/12))){dots[i].setAttribute('visibility','hidden');return}
      dots[i].setAttribute('cx',X(p.x));dots[i].setAttribute('cy',Y(p.y));dots[i].setAttribute('visibility','visible');
      h+=`<br>${series.length>1?`<i style="background:${css(s.color)}"></i>${s.label}: `:''}<b>${(opt.tipfmt||fmt)(p.y)}</b>${p.tip?' '+p.tip:''}`});
    if(opt.nearest)h=h.replace(/^[^<]*/,'');
    tip.hidden=false;tip.innerHTML=h;tip.style.left=Math.min(ev.clientX+14,innerWidth-270)+'px';tip.style.top=(ev.clientY-10)+'px'};
  hit.addEventListener('pointermove',show);hit.addEventListener('pointerdown',show);
  hit.addEventListener('pointerleave',()=>{tip.hidden=true;cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'))});
}
// stacked bars: keys [{key,label,color}], rows [{x, values:{key:v}, tip?}];
// opt: {unit, ymax (fixed top), yfmt (axis + values), xlabel (tooltip head, default Iteration), total (false: no total line), legend (false: none)}
function bars(svg,keys,rows,opt){
  opt=opt||{};svg.innerHTML='';
  let leg=svg.previousElementSibling&&svg.previousElementSibling.classList.contains('legend')?svg.previousElementSibling:null;
  if(!leg){leg=document.createElement('div');leg.className='legend';svg.parentNode.insertBefore(leg,svg)}
  leg.innerHTML=opt.legend===false?'':keys.map(k=>`<span><i class="bar" style="background:${css(k.color)}"></i>${k.label}</span>`).join('');
  const W=width(svg),H=parseFloat(svg.style.height)||160;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  if(!rows.length){const t=ns('text',{x:W/2,y:H/2,'text-anchor':'middle'});t.textContent='No data yet';svg.appendChild(t);return}
  const tot=rows.map(r=>keys.reduce((s,k)=>s+(r.values[k.key]||0),0));
  const tk=opt.ymax!=null?ticksFor(0,opt.ymax,4,false,false):ticksFor(0,Math.max(...tot)*1.05||1,4,false,true);
  const yf=opt.yfmt||big,vf=opt.yfmt||(v=>v.toFixed(1));
  const L=Math.max(30,8+6.2*Math.max(...tk.ticks.map(v=>yf(v).length))),R=12,T=10,B=24,yhi=tk.hi;
  const Y=v=>T+(1-v/yhi)*(H-T-B);const n=rows.length,slot=(W-L-R)/n,bw=Math.max(2,Math.min(28,slot-2));
  for(const v of tk.ticks){svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:css('--grid'),'stroke-width':1}));const t=ns('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'});t.textContent=yf(v);svg.appendChild(t)}
  const every=Math.max(1,Math.ceil(n/Math.floor((W-L-R)/36)));
  rows.forEach((r,i)=>{const cx=L+slot*(i+.5);let acc=0;
    keys.forEach(k=>{const v=r.values[k.key]||0;if(v<=0)return;const y1=Y(acc),y2=Y(acc+v);svg.appendChild(ns('rect',{x:cx-bw/2,y:y2+1,width:bw,height:Math.max(0,y1-y2-1),fill:css(k.color),rx:1.5}));acc+=v});
    if(i%every===0){const t=ns('text',{x:cx,y:H-6,'text-anchor':'middle'});t.textContent=r.x;svg.appendChild(t)}
    const hit=ns('rect',{x:cx-slot/2,y:T,width:slot,height:H-T-B,fill:'transparent'});
    const show=ev=>{const tip=$('tip');tip.hidden=false;tip.innerHTML=`${opt.xlabel??'Iteration'} ${r.x}`+keys.map(k=>`<br>${keys.length>1?`<i style="background:${css(k.color)}"></i>${k.label}: `:''}<b>${vf(r.values[k.key]||0)}</b>`).join('')+(opt.total===false?'':`<br>total <b>${tot[i].toFixed(1)}</b> ${opt.unit||''}`)+(r.tip?'<br>'+r.tip:'');tip.style.left=Math.min(ev.clientX+14,innerWidth-270)+'px';tip.style.top=(ev.clientY-10)+'px'};
    hit.addEventListener('pointermove',show);hit.addEventListener('pointerdown',show);hit.addEventListener('pointerleave',()=>$('tip').hidden=true);svg.appendChild(hit)});
}
const hostName=h=>!h?'':/^local:/.test(h)?'local test host':/100\.95\.248\.116|^mac$/i.test(h)?'the Mac':h.replace(/^.*@/,'');
// remote worker host (docs/az.md 9): its state, pending evaluations, loud fallbacks
function renderRemote(M,cfg){
  const el=$('remote'),R=DATA.remote;if(!cfg.remote){el.hidden=true;return}
  const out=[],now=Date.now()/1000;
  const host=hostName(R&&R.host||cfg.remote);
  const pw=R&&R.power;let head=`<b>Remote host:</b> ${host}`;
  if(pw)head+=pw.ac?` · on AC power`:` · <span class="warn"><b>on battery</b> (${pw.percent}%)${cfg.remote_allow_battery?'':': no new jobs'}</span>`;
  if(R&&R.updated)head+=` · updated ${ago(now-R.updated)} ago`;
  const jobs=R&&R.jobs?Object.entries(R.jobs):[];
  for(const [n,j] of jobs){if(!['running','starting','fetching'].includes(j.state))continue;const p=j.progress||{};
    const g=Object.values(p).filter(v=>v&&typeof v==='object'&&v.games!=null);
    const prog=p.tasks?`${p.tasks_done}/${p.tasks} parts`:g.length?`${g.reduce((a,v)=>a+(v.games_done||0),0).toLocaleString()} of ${g.reduce((a,v)=>a+v.games,0).toLocaleString()} games`:'';
    head+=` · ${n.startsWith('eval')?'evaluating':'self-play of'} iteration ${+n.slice(-4)}: ${j.state}${prog?' ('+prog+')':''}, ${ago(now-j.started)}`}
  out.push(head+'.');
  const pend=M.filter(m=>m.eval_pending);
  if(pend.length)out.push(`Evaluation pending on ${host}: iteration${pend.length>1?'s':''} ${pend.map(m=>m.iter).join(', ')} (since ${ago(now-pend[0].eval_pending.since)})${pend[0].eval_pending.waiting?' - waiting for '+pend[0].eval_pending.waiting:''}.`);
  const fb=M.filter(m=>(m.eval_remote&&m.eval_remote.fallback)||(m.selfplay&&m.selfplay.remote&&m.selfplay.remote.fallback)).slice(-3);
  for(const m of fb){const parts=[];if(m.eval_remote&&m.eval_remote.fallback)parts.push(`evaluation ran on the desktop (${m.eval_remote.reason})`);
    if(m.selfplay&&m.selfplay.remote&&m.selfplay.remote.fallback)parts.push(`the desktop took over the remote self-play games (${m.selfplay.remote.fallback})`);
    out.push(`<span class="warn"><b>Iteration ${m.iter}:</b> ${parts.join('; ').replace(/</g,'&lt;')}.</span>`)}
  const ev=(R&&R.events||[]).filter(e=>e.loud&&now-e.time<6*3600).slice(-2);
  for(const e of ev)out.push(`<span class="warn"><b>${ago(now-e.time)} ago:</b> ${e.msg.replace(/</g,'&lt;')}</span>`);
  el.innerHTML=out.join('<br>');el.hidden=false}
function tile(l,v,n){return `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n||'&nbsp;'}</div></div>`}
// kingdom cards the self-play players gained / bought (docs/az.md 2.10; DATA.kingdom = dashboard.kingdom_series)
let KMODE='gained',KCMP='',KCMPDATA=null,RUNS=[];
try{KMODE=localStorage.getItem('domz-kmode')||'gained';KCMP=localStorage.getItem('domz-kcmp')||''}catch(e){}
const KCATS=[['normal','normal games','--series-1'],['forced','forced seats','--series-2'],['unforced','their opponents','--series-3']];
function setKMode(m){KMODE=m;try{localStorage.setItem('domz-kmode',m)}catch(e){}if(DATA)renderKingdom()}
$('km-gained').onclick=()=>setKMode('gained');$('km-bought').onclick=()=>setKMode('bought');
async function loadCmp(){
  if(!KCMP||KCMP===RUN){KCMPDATA=null;return}
  try{const r=await fetch('api/run/'+encodeURIComponent(KCMP)+'/kingdom');KCMPDATA=r.ok?await r.json():null}catch(e){KCMPDATA=null}}
$('kgcmp').onchange=async e=>{KCMP=e.target.value;try{localStorage.setItem('domz-kcmp',KCMP)}catch(er){}await loadCmp();if(DATA)renderKingdom()};
function renderKingdom(){
  const cfg=DATA.config||{},K=DATA.kingdom||{iters:[]},its=K.iters||[];
  $('kgsec').hidden=cfg.game!=='dominion';if(cfg.game!=='dominion')return;
  const word=KMODE==='bought'?'bought':'gained';
  $('km-gained').setAttribute('aria-pressed',KMODE!=='bought');$('km-bought').setAttribute('aria-pressed',KMODE==='bought');
  const sel=$('kgcmp'),others=RUNS.filter(n=>n!==RUN);
  sel.innerHTML='<option value="">no other run</option>'+others.map(n=>`<option${n===KCMP?' selected':''}>${n.replace(/</g,'&lt;')}</option>`).join('');
  const v=(it,c)=>it.cats[c]&&it.cats[c][KMODE];
  const cmp=KCMPDATA&&KCMPDATA.run===KCMP&&KCMP!==RUN?KCMPDATA.kingdom.iters||[]:[];
  for(const [id,key] of [['kgd','distinct'],['kgt','total']]){
    const ser=KCATS.map(([c,l,col])=>({label:l,color:col,pts:its.filter(it=>v(it,c)).map(it=>({x:it.iter,y:v(it,c)[key],tip:`(${v(it,c).seats.toLocaleString()} players)`}))}));
    if(cmp.length)ser.push({label:KCMP,color:'--series-7',dash:'6 4',pts:cmp.filter(it=>v(it,'normal')).map(it=>({x:it.iter,y:v(it,'normal')[key],tip:`(${v(it,'normal').seats.toLocaleString()} players)`}))});
    chart($(id),ser,{y0:0,tipfmt:x=>x.toFixed(2)+' cards '+word})}
  // distribution of the distinct count over iterations (normal games): 0 / 1 / 2 / 3 / 4 / 5+
  const DK=['0','1','2','3','4','5+'].map((l,i)=>({key:'d'+i,label:l,color:'--ord-'+i}));
  const share=h=>{const n=h.reduce((a,b)=>a+b,0)||1,o={};for(let i=0;i<6;i++)o['d'+i]=(i<5?h[i]||0:h.slice(5).reduce((a,b)=>a+b,0))/n;return o};
  const nm=its.filter(it=>v(it,'normal'));
  $('kgmix-cap').textContent=`Share of the players of normal games that ${word} 0, 1, 2, 3, 4 or 5+ different kingdom cards in a game, by iteration`;
  bars($('kgmix'),DK,nm.map(it=>({x:it.iter,values:share(v(it,'normal').distinct_hist),tip:`${v(it,'normal').seats.toLocaleString()} players`})),{ymax:1,yfmt:x=>Math.round(100*x)+'%',total:false});
  // histograms of the newest iteration (normal games)
  const last=nm[nm.length-1],B=K.bins||{};
  for(const [id,key,lab,title] of [['kgh1','distinct_hist','distinct',w=>`Distinct kingdom cards ${w}`],['kgh2','total_hist','total',w=>`Kingdom cards ${w} (every copy)`]]){
    $(id+'-t').textContent=`${title(word)}, ${last?'iteration '+last.iter:'newest iteration'}`;
    const h=last?v(last,'normal')[key]:[],n=h.reduce((a,b)=>a+b,0)||1,labels=B[lab]||[];
    $(id+'-cap').textContent=last?`Share of the ${v(last,'normal').seats.toLocaleString()} players of normal games, by number of ${lab==='distinct'?'different kingdom cards':'kingdom cards (every copy)'}`:'Share of the players of normal games';
    bars($(id),[{key:'s',label:'players',color:'--series-1'}],h.map((c,i)=>({x:labels[i]??i,values:{s:c/n},tip:`${c.toLocaleString()} players`})),
      {ymax:Math.min(1,Math.max(.1,Math.ceil(Math.max(0,...h)/n*10+1e-9)/10)),yfmt:x=>Math.round(100*x)+'%',total:false,legend:false,xlabel:lab==='distinct'?'Distinct cards':'Cards'})}
}
const winPt=(x,e)=>({x,y:e.win,lo:Math.max(0,e.win-1.96*e.se),hi:Math.min(1,e.win+1.96*e.se),tip:`(±${(196*e.se).toFixed(1)}, ${e.games} games)`});
const ago=s=>s<90?Math.round(s)+' s':s<5400?Math.round(s/60)+' min':(s/3600).toFixed(1)+' h';
// run control (domz/az/runctl.py): state + Pause / Resume (PAUSE file in the run dir)
const CTL={running:['Running','pause'],stopping:['Stopping after this iteration','pause'],pausing:['Pausing','resume'],
  paused:['Paused','resume'],resuming:['Resuming',null],starting:['Starting','pause'],restarting:['Restarting after a crash','pause'],stopped:['Stopped',null]};
const clock=t=>new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
let CTLDATA=null,FAST=null,FAST_UNTIL=0;
function renderCtl(c){
  CTLDATA=c;const el=$('ctl');
  if(!c||!c.state){el.hidden=true;return}
  el.hidden=false;
  const [label,act]=CTL[c.state]||[c.state,null],now=c.now||Date.now()/1000,r=c.run||{};
  $('ctl-dot').className='dot '+c.state;
  let lab=label,det=c.detail||'';
  if(c.state==='paused'&&c.paused_since)lab+=` since ${clock(c.paused_since)} (${ago(now-c.paused_since)} ago)`;
  else if(c.state==='pausing'&&c.paused_since)lab+=` (${ago(Math.max(0,now-c.paused_since))})`;
  if(c.state==='paused')det=(r.state==='paused'&&r.pause_requested&&r.exited?
    `GPU released ${Math.max(0,(r.workers_stopped||r.exited)-r.pause_requested).toFixed(1)} s after the request. `:'GPU released. ')+det;
  $('ctl-label').textContent=lab;$('ctl-detail').textContent=det;$('ctl-detail').hidden=!det;
  const b=$('ctl-btn');
  if(!act||(act==='resume'&&!c.pause)){b.hidden=true}
  else{b.hidden=false;b.disabled=false;b.dataset.act=act;b.className='btn '+act;b.textContent=act==='pause'?'Pause':'Resume';
    b.setAttribute('aria-label',(act==='pause'?'Pause run ':'Resume run ')+RUN)}
  if(['pausing','resuming','starting'].includes(c.state))fastPoll(60);
}
// while the state changes: poll the control endpoint every 1.5 s (the page refreshes every 30 s)
function fastPoll(sec){FAST_UNTIL=Math.max(FAST_UNTIL,Date.now()+sec*1000);if(FAST)return;
  FAST=setInterval(async()=>{if(Date.now()>FAST_UNTIL||!RUN){clearInterval(FAST);FAST=null;return}
    try{const r=await fetch('api/run/'+encodeURIComponent(RUN)+'/control');if(r.ok){const c=await r.json();const was=CTLDATA&&CTLDATA.state;renderCtl(c);
      if(was!==c.state&&['running','paused','stopped'].includes(c.state)){FAST_UNTIL=Date.now()+3000;load()}}}catch(e){}},1500)}
$('ctl-btn').onclick=async()=>{
  const b=$('ctl-btn'),act=b.dataset.act,run=RUN;if(!act||!run)return;
  if(act==='pause'&&!confirm(`Pause ${run}?\n\nSelf-play stops within seconds and the GPU (compute and memory) is released. Games in progress are dropped; finished games are kept. Resume continues from the newest checkpoint.`))return;
  b.disabled=true;
  try{const r=await fetch('api/run/'+encodeURIComponent(run)+'/'+act,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:act})});
    const j=await r.json();if(!r.ok){alert(j.error||('HTTP '+r.status));b.disabled=false;return}
    if(run===RUN)renderCtl(j.control)}
  catch(e){alert('Request failed: '+e);b.disabled=false;return}
  fastPoll(90)};
function render(){
  const M=DATA.metrics, cfg=DATA.config;
  renderCtl(DATA.control);
  const R=(DATA.ratings&&DATA.ratings.ratings)||{};
  const BOTS={bmu:'BMU',random:'Random',smithy:'Smithy-BM',heuristic:'Heuristic'};
  const nm=i=>'iter_'+String(i).padStart(4,'0');
  const P=DATA.pairs||{};
  const rec=(a,b)=>{const k=a<b?a+'|'+b:b+'|'+a, p=P[k]; if(!p||!p[1])return null; const s=a<b?p[0]:p[1]-p[0], w=s/p[1]; return {win:w, n:p[1], se:Math.sqrt(Math.max(w*(1-w),1e-6)/p[1])}};
  const sp=M.filter(m=>m.selfplay), tr=M.filter(m=>m.train);
  const eloPts=Object.keys(R).filter(k=>k.startsWith('iter_')).map(k=>({x:+k.slice(5),y:R[k].elo,lo:R[k].elo-1.96*R[k].se,hi:R[k].elo+1.96*R[k].se,tip:`±${Math.round(1.96*R[k].se)} · ${R[k].games.toLocaleString()} rated games`})).sort((a,b)=>a.x-b.x);
  const lastElo=eloPts[eloPts.length-1], bestElo=eloPts.reduce((b,p)=>!b||p.y>b.y?p:b,null);
  // header + status
  const gname=cfg.game==='dominion'?`Dominion (${cfg.sets})`:(cfg.toy_random?'toy game, random kingdoms':'toy game');
  const drv=cfg.driver==='rust'?`Rust driver ${cfg.rust_threads} threads, ${Number(cfg.rust_conc).toLocaleString()} games in flight${cfg.rolling?' (rolling: kept across iterations)':''}`:`${cfg.workers} workers`;
  $('sub').textContent=[`Run ${RUN}`,cfg.preset?`preset ${cfg.preset}`:'',gname,cfg.players?cfg.players.join('/')+'-player':'',`net ${cfg.d}x${cfg.layers}`,
    `${Number(cfg.games||0).toLocaleString()} ${cfg.rolling?'finished ':''}games/iteration`,`self-play ${cfg.sims}/${cfg.sims_fast} sims`,`eval ${cfg.eval_sims} sims`,cfg.turn_cap?`turn cap ${cfg.turn_cap}`:'',drv].filter(Boolean).join(' · ');
  const st=DATA.status||{}, pr=st.progress;let s='';
  const lastLine=(DATA.log||[]).slice(-1)[0]||'', ended=/ done$|STOP file found|run aborted/.test(lastLine);
  if(ended)s+=/aborted/.test(lastLine)?`<b>Aborted</b> (${lastLine.replace(/^\S+ /,'').slice(0,160)}). `:`<b>Stopped</b>${DATA.stopping?' (STOP file)':' (finished)'}. `;
  else if(DATA.stopping)s+='<b>Stopping</b> after this iteration (STOP file). ';
  if(st.log_age_s!=null){const live=ended||st.log_age_s<(cfg.phase_timeout||3600)+600;s+=`Last log line ${ago(st.log_age_s)} ago${live?'':' (the run looks stopped)'}. `}
  const halted=DATA.control&&['paused','pausing','stopped'].includes(DATA.control.state);
  if(pr&&!pr.done&&!halted){const el=(pr.time-pr.started),rate=pr.games_done/Math.max(el,1);s+=`<b>Self-play of iteration ${pr.iter}</b>: ${pr.games_done.toLocaleString()} of ${pr.games.toLocaleString()} games (${pct(pr.games_done/Math.max(1,pr.games))}), ${big(pr.samples)} samples, ${ago(el)} in`+(rate>0&&pr.games_done<pr.games?`, ~${ago((pr.games-pr.games_done)/rate)} to go`:'')+'.'+
    (pr.remote_games?` ${pr.remote_games_done.toLocaleString()} of ${pr.remote_games.toLocaleString()} on ${hostName(pr.remote)}.`:'')+
    (pr.rolling&&pr.in_flight!=null?` ${pr.in_flight.toLocaleString()} games in flight (rolling).`:'')+
    `<div class="bar"><div style="width:${(100*pr.games_done/Math.max(1,pr.games)).toFixed(1)}%"></div></div>`}
  // loop guard (docs/engine.md 2) and evaluation parts that timed out: loud
  const esc=x=>String(x).replace(/</g,'&lt;');
  const lg=M.filter(m=>(m.selfplay&&(m.selfplay.loop_stopped||m.selfplay.loop_ended))||['eval_bmu','eval_random','eval_heuristic'].some(k=>m[k]&&(m[k].loop_stopped||m[k].loop_ended))).slice(-3);
  for(const m of lg){const sp1=m.selfplay||{},ee=['eval_bmu','eval_random','eval_heuristic'].filter(k=>m[k]&&(m[k].loop_stopped||m[k].loop_ended));
    s+=`<div class="warn"><b>Loop guard, iteration ${m.iter}:</b> ${sp1.loop_stopped||0} self-play game(s) had a turn stopped, ${sp1.loop_ended||0} ended by the guard (longest turn ${sp1.max_turn_steps||'?'} steps)${ee.length?'; evaluation: '+ee.map(k=>`${k.slice(5)} ${m[k].loop_stopped||0}/${m[k].loop_ended||0}`).join(', '):''} - see the log (LOOP GUARD).</div>`}
  const ef=M.filter(m=>m.eval_failed).slice(-3);
  for(const m of ef)s+=`<div class="warn"><b>Evaluation parts that timed out, iteration ${m.iter}:</b> ${esc(Object.entries(m.eval_failed).map(([k,v])=>k+' ('+v+')').join(', '))}.</div>`;
  const efe=Object.keys((DATA.failed_pairs)||{});
  if(efe.length)s+=`<div class="warn"><b>Elo matches that timed out</b> (not retried): ${esc(efe.slice(-6).join(', '))}${efe.length>6?' ...':''}.</div>`;
  $('status').innerHTML=s;
  renderRemote(M,cfg);
  // tiles
  const ev=M.filter(m=>m.eval_bmu), lb=ev[ev.length-1], lh=M.filter(m=>m.eval_heuristic).pop(), lk=M.filter(m=>m.eval_kingdoms).pop();
  const games=sp.reduce((a,m)=>a+m.selfplay.games,0), t0=M.length?M[0].time:0, tl=M.length?M[M.length-1].time:0;
  const ltr=tr[tr.length-1];
  const wt=(e)=>e?`±${(196*e.se).toFixed(1)} pts · margin ${e.margin>0?'+':''}${e.margin}`:'';
  $('tiles').innerHTML=tile('Iteration',M.length?M[M.length-1].iter:'–',cfg.iters?`of ${cfg.iters}`:'')+
    tile('Elo (newest)',lastElo?Math.round(lastElo.y):'–',lastElo?`±${Math.round(lastElo.hi-lastElo.y)} · best ${Math.round(bestElo.y)} (iteration ${bestElo.x})`:'')+
    tile('vs BMU',lb?pct(lb.eval_bmu.win):'–',wt(lb&&lb.eval_bmu))+
    tile('vs heuristic',lh?pct(lh.eval_heuristic.win):'–',lh?wt(lh.eval_heuristic)+(R.heuristic?` · bot Elo ${Math.round(R.heuristic.elo)}`:''):'')+
    (lk?tile('Fixed kingdoms',pct((lk.eval_kingdoms.bmu||Object.values(lk.eval_kingdoms)[0])._all.win),`vs ${lk.eval_kingdoms.bmu?'BMU':Object.keys(lk.eval_kingdoms)[0]} · ${Object.keys(Object.values(lk.eval_kingdoms)[0]).length-1} kingdoms`):'')+
    tile('Self-play games',big(games),`${((tl-t0)/3600).toFixed(1)} h elapsed`)+
    (ltr?tile('Training window',big(ltr.train.window_loaded??ltr.train.window_samples),ltr.train.window_loaded!=null?`samples in RAM (${(ltr.train.window_ram_gb||0).toFixed(2)} GB)${ltr.train.window_frac<1?` = ${pct(ltr.train.window_frac)} of ${big(ltr.train.window_samples)}`:''}`:'samples'):'');
  // Elo
  const eloRefs=[{y:1000,label:'BMU 1000',fit:true}];
  for(const b of ['heuristic','random','smithy'])if(R[b])eloRefs.push({y:R[b].elo,label:BOTS[b]+' '+Math.round(R[b].elo),left:true,fit:b!=='random'||eloPts.some(p=>p.lo<R[b].elo+300)});
  chart($('elo'),[{label:'Elo',color:'--series-1',pts:eloPts}],{refs:eloRefs,fmt:v=>Math.round(v).toLocaleString(),xlabel:'Iteration'});
  // win rates vs bots
  const wser=[['eval_bmu','vs BMU','--series-1'],['eval_heuristic','vs Heuristic','--series-2'],['eval_random','vs Random','--series-3']]
    .map(([k,l,c])=>({label:l,color:c,band:k==='eval_bmu',pts:M.filter(m=>m[k]).map(m=>winPt(m.iter,m[k]))}));
  const p100=v=>Math.round(v*100)+'%';
  chart($('win'),wser,{y0:0,y1:1,fmt:p100,tipfmt:pct,refs:[{y:.5,label:'50%',left:true}]});
  // fixed kingdoms
  const KM=M.filter(m=>m.eval_kingdoms);$('kcard').hidden=!KM.length;
  if(KM.length){const opps=Object.keys(KM[KM.length-1].eval_kingdoms);
    chart($('kwin'),opps.map((o,i)=>({label:'vs '+(BOTS[o]||o),color:['--series-1','--series-2','--series-3'][i%3],band:i===0,pts:KM.filter(m=>m.eval_kingdoms[o]).map(m=>winPt(m.iter,m.eval_kingdoms[o]._all))})),{y0:0,y1:1,fmt:p100,tipfmt:pct,refs:[{y:.5,label:'50%',left:true}]});
    const last=KM[KM.length-1],prev=KM.length>1?KM[KM.length-2]:null;const names=Object.keys(last.eval_kingdoms[opps[0]]).filter(k=>k!=='_all');
    const per=last.eval_kingdoms[opps[0]][names[0]];$('kcap').textContent=`The rulebook's named kingdoms, the same every iteration (not part of the Elo fit): ${per.games} games per kingdom against each bot, ${last.eval_kingdoms[opps[0]]._all.games} in all. Table: iteration ${last.iter}${prev?`, change since iteration ${prev.iter}`:''}.`;
    const cell=(o,k)=>{const e=last.eval_kingdoms[o][k];if(!e)return '<td></td><td></td>';const pe=prev&&prev.eval_kingdoms[o]&&prev.eval_kingdoms[o][k];const d=pe?100*(e.win-pe.win):null;
      return `<td>${pct(e.win)}</td><td class="${d==null?'':d>=0?'up':'down'}">${d==null?'':(d>=0?'+':'')+d.toFixed(0)}</td>`};
    $('ktable').innerHTML='<table><tr><th>Kingdom</th>'+opps.map(o=>`<th>vs ${BOTS[o]||o}</th><th>&Delta;</th>`).join('')+'<th>Margin vs '+(BOTS[opps[0]]||opps[0])+'</th><th>Rounds</th></tr>'+
      names.concat(['_all']).map(k=>`<tr${k==='_all'?' style="font-weight:600"':''}><td>${k==='_all'?'All kingdoms':k}</td>`+opps.map(o=>cell(o,k)).join('')+`<td>${(e=>e.margin>0?'+'+e.margin:e.margin)(last.eval_kingdoms[opps[0]][k])}</td><td>${last.eval_kingdoms[opps[0]][k].turns??''}</td></tr>`).join('')+'</table>'}
  // margins, vs previous
  chart($('mgn'),[['eval_bmu','vs BMU','--series-1'],['eval_heuristic','vs Heuristic','--series-2']].map(([k,l,c])=>({label:l,color:c,pts:M.filter(m=>m[k]).map(m=>({x:m.iter,y:m[k].margin}))})),{refs:[{y:0,label:'Even',left:true,fit:true}],tipfmt:v=>(v>0?'+':'')+v.toFixed(1)+' VP'});
  const prevPts=eloPts.map(p=>({x:p.x,r:rec(nm(p.x),nm(p.x-1))})).filter(p=>p.r).map(p=>({x:p.x,y:p.r.win,lo:Math.max(0,p.r.win-1.96*p.r.se),hi:Math.min(1,p.r.win+1.96*p.r.se),tip:`(±${(196*p.r.se).toFixed(1)}, ${p.r.n} games)`}));
  chart($('prev'),[{label:'vs previous',color:'--series-1',pts:prevPts}],{y0:0,y1:1,fmt:p100,tipfmt:pct,refs:[{y:.5,label:'Even',left:true}]});
  const refPts=M.filter(m=>m.eval_ref).map(m=>winPt(m.iter,m.eval_ref));$('refcard').hidden=!refPts.length;
  if(refPts.length){const rname=M.filter(m=>m.eval_ref).pop().eval_ref.ref||'';$('ref-t').textContent='Against the reference network ('+rname.replace(/^.*runs[\/]/,'').replace(/[\/]models[\/]/,' ').replace('.pt','')+')';
    chart($('refc'),[{label:'vs reference',color:'--series-1',pts:refPts}],{y0:0,y1:1,fmt:p100,tipfmt:pct,refs:[{y:.5,label:'Even',left:true}]})}
  const newest=lastElo?lastElo.x:null;
  if(newest!==null&&R[nm(newest)]){
    const me=nm(newest), opp=Object.keys(P).flatMap(k=>{const [a,b]=k.split('|');return a===me?[b]:b===me?[a]:[]});
    opp.sort((a,b)=>a==='bmu'?-1:b==='bmu'?1:b.localeCompare(a));
    const label=o=>o==='bmu'?'Big Money Ultimate':o.startsWith('iter_')?'Iteration '+(+o.slice(5)):(BOTS[o]||o);
    const pred=o=>R[o]&&R[me]?1/(1+Math.pow(10,-(R[me].elo-R[o].elo)/400)):null;
    $('h2h-t').textContent=`Newest checkpoint (iteration ${newest}, Elo ${Math.round(R[me].elo)}): head to head`;
    $('h2h').innerHTML='<table><tr><th>Opponent</th><th>Opp. Elo</th><th>Games</th><th>Wins</th><th>±95%</th><th>Elo predicts</th></tr>'+
      opp.map(o=>{const r=rec(me,o), q=pred(o);return `<tr><td>${label(o)}</td><td>${R[o]?Math.round(R[o].elo):''}</td><td>${r.n}</td><td>${pct(r.win)}</td><td>${(196*r.se).toFixed(1)}</td><td>${q==null?'':pct(q)}</td></tr>`}).join('')+'</table>'}
  // games
  chart($('turns'),[{label:'self-play',color:'--series-1',pts:sp.filter(m=>m.selfplay.turns_per_game!=null).map(m=>({x:m.iter,y:m.selfplay.turns_per_game}))},
    {label:'vs BMU',color:'--series-2',pts:M.filter(m=>m.eval_bmu&&m.eval_bmu.turns!=null).map(m=>({x:m.iter,y:m.eval_bmu.turns}))}],
    {refs:cfg.turn_cap?[{y:cfg.turn_cap,label:'cap '+cfg.turn_cap,left:true,fit:true}]:[],tipfmt:v=>v.toFixed(1)+' rounds'});
  chart($('capped'),[{label:'self-play',color:'--series-1',pts:sp.filter(m=>m.selfplay.capped_rate!=null).map(m=>({x:m.iter,y:m.selfplay.capped_rate}))},
    {label:'vs BMU',color:'--series-2',pts:M.filter(m=>m.eval_bmu&&m.eval_bmu.capped!=null).map(m=>({x:m.iter,y:m.eval_bmu.capped}))}],{y0:0,y1:1,fmt:p100,tipfmt:pct});
  chart($('dpg'),[{label:'decisions',color:'--series-1',pts:sp.map(m=>({x:m.iter,y:m.selfplay.decisions_per_game}))}],{tipfmt:v=>v.toFixed(1)+' decisions'});
  chart($('sc'),[{label:'VP',color:'--series-1',pts:sp.map(m=>({x:m.iter,y:m.selfplay.avg_vp}))}],{tipfmt:v=>v.toFixed(1)+' VP'});
  renderKingdom();
  // training
  const l3=v=>v.toFixed(3);
  chart($('lp'),[{label:'policy',color:'--series-1',pts:tr.map(m=>({x:m.iter,y:m.train.policy}))}],{tipfmt:l3});
  chart($('lv'),[{label:'value',color:'--series-1',pts:tr.map(m=>({x:m.iter,y:m.train.value}))}],{tipfmt:l3});
  chart($('lm'),[{label:'margin',color:'--series-1',pts:tr.map(m=>({x:m.iter,y:m.train.margin}))}],{tipfmt:v=>v.toFixed(4)});
  const ta=tr.filter(m=>m.train.aux_gain!=null);$('auxcard').hidden=!ta.length;
  if(ta.length)chart($('laux'),[{label:'gains per pile',color:'--series-1',pts:ta.map(m=>({x:m.iter,y:m.train.aux_gain}))},
    {label:'rounds left',color:'--series-2',pts:ta.map(m=>({x:m.iter,y:m.train.aux_turns}))}],{tipfmt:l3});
  chart($('samples'),[{label:'new',color:'--series-1',pts:sp.map(m=>({x:m.iter,y:m.selfplay.samples,tip:m.selfplay.bytes_per_sample?`(${Math.round(m.selfplay.bytes_per_sample)} B each on disk)`:''}))},
    {label:'window',color:'--series-2',pts:tr.map(m=>({x:m.iter,y:m.train.window_samples}))},
    {label:'in RAM',color:'--series-3',pts:tr.filter(m=>m.train.window_loaded!=null).map(m=>({x:m.iter,y:m.train.window_loaded,tip:`(${(m.train.window_ram_gb||0).toFixed(2)} GB)`}))}],{y0:0,tipfmt:v=>big(v)});
  // throughput
  const spr=sp.filter(m=>m.selfplay.remote&&m.selfplay.remote.games_per_s);
  chart($('gph'),[{label:spr.length?'all':'games/h',color:'--series-1',pts:sp.map(m=>({x:m.iter,y:m.selfplay.games/m.selfplay.seconds*3600}))},
    {label:'desktop',color:'--series-2',pts:spr.map(m=>({x:m.iter,y:m.selfplay.remote.local_games_per_s*3600}))},
    {label:'remote host',color:'--series-3',pts:spr.map(m=>({x:m.iter,y:m.selfplay.remote.games_per_s*3600,tip:`(${pct(m.selfplay.remote.share)} of the games)`}))}],{y0:0,tipfmt:v=>Math.round(v).toLocaleString()+' games/h'});
  const spl=sp.filter(m=>m.selfplay.local_positions_per_s!=null);
  chart($('pps'),[{label:'leaves/s',color:'--series-1',pts:sp.map(m=>({x:m.iter,y:m.selfplay.positions_per_s||0}))}].concat(spl.length?
    [{label:'desktop',color:'--series-2',pts:spl.map(m=>({x:m.iter,y:m.selfplay.local_positions_per_s}))}]:[]),{y0:0,tipfmt:v=>Math.round(v).toLocaleString()+' leaves/s'});
  const full=cfg.driver==='rust'?cfg.rust_conc/Math.max(1,cfg.rust_groups||2):null;
  chart($('abatch'),[{label:'leaves per call',color:'--series-1',pts:sp.filter(m=>m.selfplay.avg_batch!=null||(m.selfplay.batches&&!m.selfplay.remote)).map(m=>({x:m.iter,
    y:m.selfplay.avg_batch!=null?m.selfplay.avg_batch:m.selfplay.positions/Math.max(1,m.selfplay.batches)}))}],
    {y0:0,refs:full?[{y:full,label:'full '+Math.round(full).toLocaleString(),left:true,fit:true}]:[],tipfmt:v=>Math.round(v).toLocaleString()+' leaves'});
  const rl=sp.filter(m=>m.selfplay.rolling);
  $('rollcard').hidden=!rl.length;
  if(rl.length)chart($('stale'),[{label:'searched by an earlier network',color:'--series-1',pts:rl.map(m=>({x:m.iter,y:m.selfplay.stale_frac||0,
      tip:Object.keys(m.selfplay.rolling.nets||{}).length?`(networks ${Object.entries(m.selfplay.rolling.nets).map(([k,v])=>k+': '+big(v)).join(', ')})`:''}))},
    {label:'search redone after the swap',color:'--series-2',pts:rl.map(m=>({x:m.iter,y:m.selfplay.restart_frac||0}))}],{y0:0,fmt:p100,tipfmt:pct});
  const lt=sp.filter(m=>m.selfplay.max_turn_steps!=null);
  $('loopcard').hidden=!lt.length;
  if(lt.length){chart($('loops'),[{label:'longest turn',color:'--series-1',pts:lt.map(m=>({x:m.iter,y:m.selfplay.max_turn_steps,
      tip:(m.selfplay.loop_stopped||m.selfplay.loop_ended)?`(guard: ${m.selfplay.loop_stopped||0} stopped, ${m.selfplay.loop_ended||0} ended)`:''}))}],
      {y0:0,refs:[{y:1000,label:'stop rule 1,000',left:true,fit:false}],tipfmt:v=>Math.round(v).toLocaleString()+' steps'});
    const ag=lt.filter(m=>m.selfplay.rolling&&m.selfplay.rolling.oldest_age_games!=null);
    $('ages').style.display=ag.length?'':'none';
    if(ag.length)chart($('ages'),[{label:'oldest game in progress',color:'--series-2',pts:ag.map(m=>({x:m.iter,y:m.selfplay.rolling.oldest_age_games/Math.max(1,cfg.games),
        tip:`(${big(m.selfplay.rolling.oldest_age_games)} games started since; longest in progress ${m.selfplay.rolling.in_flight_max_steps} steps, its turn ${m.selfplay.rolling.in_flight_max_turn_steps})`}))}],
      {y0:0,tipfmt:v=>v.toFixed(2)+' iterations'})}
  bars($('mins'),[{key:'sp',label:'self-play',color:'--series-1'},{key:'tr',label:'training',color:'--series-2'},{key:'ev',label:'evaluation + Elo',color:'--series-3'}],
    M.filter(m=>m.seconds&&m.selfplay).map(m=>({x:m.iter,values:{sp:m.selfplay.seconds/60,tr:(m.train?m.train.seconds+(m.train.window_load_s||0):0)/60,ev:Math.max(0,m.seconds-m.selfplay.seconds-(m.train?m.train.seconds+(m.train.window_load_s||0):0))/60}})),{unit:'min'});
  chart($('tps'),[{label:'samples/s',color:'--series-1',pts:tr.filter(m=>m.train.samples_per_s||m.train.seconds).map(m=>({x:m.iter,y:m.train.samples_per_s||m.train.steps*(m.train.batch||cfg.batch)/m.train.seconds,tip:m.train.data_wait_s!=null?`(waited ${m.train.data_wait_s}s for data)`:''}))}],{y0:0,tipfmt:v=>Math.round(v).toLocaleString()+' samples/s'});
  // table + logs
  const f1=v=>v==null?'':v.toFixed(1);
  const remCell=m=>{const r=[],e=m.eval_remote,sr=m.selfplay&&m.selfplay.remote;
    if(m.eval_pending)r.push('eval pending');else if(e)r.push(e.fallback?`<span class="warn">eval local</span>`:'eval '+hostName(e.host));
    if(sr)r.push(sr.fallback?`<span class="warn">${pct(sr.share)} games, taken over</span>`:`${pct(sr.share)} games`);return r.join(' · ')};
  const pend=m=>m.eval_pending?'<span class="warn">pending</span>':'';
  const KI=new Map(((DATA.kingdom||{}).iters||[]).filter(it=>it.cats.normal&&it.cats.normal.gained).map(it=>[it.iter,it.cats.normal.gained]));
  $('table').innerHTML='<table><tr><th>Iter</th><th>Elo</th><th>vs prev</th><th>vs BMU</th><th>vs heur.</th><th>vs random</th><th>Fixed vs BMU</th><th>Margin BMU</th><th>Policy</th><th>Value</th><th>Margin loss</th><th>Games</th><th>Samples</th><th>Rounds</th><th>Capped</th><th>Decisions</th><th>Avg VP</th>'+(KI.size?'<th>Kingdom gained</th>':'')+'<th>Batch</th>'+(cfg.rolling?'<th>Stale</th>':'')+'<th>Minutes</th>'+(cfg.remote?'<th>Remote</th>':'')+'</tr>'+
    M.slice().reverse().map(m=>{const r=R[nm(m.iter)],p=m.iter>0?rec(nm(m.iter),nm(m.iter-1)):null,s=m.selfplay,t=m.train;
      return `<tr><td>${m.iter}</td><td>${r?Math.round(r.elo)+' ±'+Math.round(1.96*r.se):''}</td><td>${p?pct(p.win):''}</td><td>${m.eval_bmu?pct(m.eval_bmu.win):pend(m)}</td><td>${m.eval_heuristic?pct(m.eval_heuristic.win):''}</td><td>${m.eval_random?pct(m.eval_random.win):''}</td><td>${m.eval_kingdoms&&m.eval_kingdoms.bmu?pct(m.eval_kingdoms.bmu._all.win):''}</td><td>${m.eval_bmu?m.eval_bmu.margin:''}</td><td>${t?t.policy.toFixed(3):''}</td><td>${t?t.value.toFixed(3):''}</td><td>${t?t.margin.toFixed(4):''}</td><td>${s?s.games.toLocaleString():''}</td><td>${s?s.samples.toLocaleString():''}</td><td>${s?f1(s.turns_per_game):''}</td><td>${s&&s.capped_rate!=null?pct(s.capped_rate):''}</td><td>${s?f1(s.decisions_per_game):''}</td><td>${s?f1(s.avg_vp):''}</td>${KI.size?`<td>${(k=>k?k.distinct.toFixed(2)+' / '+k.total.toFixed(2):'')(KI.get(m.iter))}</td>`:''}<td>${s&&s.avg_batch!=null?Math.round(s.avg_batch).toLocaleString():''}</td>${cfg.rolling?`<td>${s&&s.stale_frac!=null?pct(s.stale_frac):''}</td>`:''}<td>${m.seconds?(m.seconds/60).toFixed(1):''}</td>${cfg.remote?`<td>${remCell(m)}</td>`:''}</tr>`}).join('')+'</table>';
  $('log').textContent=DATA.log.map(l=>l.length>400?l.slice(0,400)+'…':l).join('\n');
  $('sup').textContent=(DATA.supervisor||[]).join('\n')||'No supervisor log (run started directly).';
}
async function load(){
  const runs=await (await fetch('api/runs')).json();
  const sel=$('runsel');
  if(!RUN&&runs.length)RUN=runs[0].name;
  sel.innerHTML=runs.map(r=>`<option${r.name===RUN?' selected':''}>${r.name}</option>`).join('');
  RUNS=runs.map(r=>r.name).sort();
  if(!RUN)return;
  const [d]=await Promise.all([fetch('api/run/'+encodeURIComponent(RUN)).then(r=>r.json()),loadCmp()]);
  DATA=d;
  render();
}
$('runsel').onchange=e=>{RUN=e.target.value;load()};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>DATA&&render());
let rsz=null,lastW=innerWidth;addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;clearTimeout(rsz);rsz=setTimeout(()=>DATA&&render(),150)});
addEventListener('scroll',()=>{$('tip').hidden=true},{passive:true});
load();setInterval(load,30000);
fetch('api/experiments').then(r=>r.json()).then(x=>{if(x.length)$('explink').hidden=false}).catch(()=>{});
</script></body></html>'''


def _between(text, a, b):
    i = text.index(a) + len(a)
    return text[i:text.index(b, i)]


STYLE = _between(PAGE, '<style>', '</style>')
COMMON_JS = PAGE[PAGE.index('<script>') + 8:PAGE.index('const hostName=')]
from .expdash import EXP_PAGE_TEMPLATE  # noqa: E402
EXP_PAGE = EXP_PAGE_TEMPLATE.replace('/*STYLE*/', STYLE).replace('/*COMMON_JS*/', COMMON_JS)


def main():
    global RUNS, EXPS
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5240)
    ap.add_argument('--runs', default=RUNS, help='directory holding the run dirs')
    ap.add_argument('--experiments', default=EXPS, help='directory holding the experiment dirs')
    a = ap.parse_args()
    RUNS = os.path.abspath(a.runs)
    EXPS = os.path.abspath(a.experiments)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
