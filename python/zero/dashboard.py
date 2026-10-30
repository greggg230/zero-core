"""Training dashboard: python -m <pkg>.az.dashboard [--host 0.0.0.0] [--port PORT] [--runs DIR]

Reads <runs>/*/metrics.jsonl, config.json, log.txt, supervisor.log, progress.json and elo/, plus
deep_evals.jsonl when a run has one: occasional strength points with a long search, written by a job
outside the loop (one JSON row per point: iter, sims, opponent, games, win, se, margin; a later row
for the same iter and sims replaces an earlier one), charted per sims value.
Binds 0.0.0.0 by default so it is reachable over Tailscale (the game package's hub project and
port: Caverna Zero caverna_zero :5260, Glass Road Zero glassroad_zero :5280; `zero.package`).
The runs directory defaults to <game repo>/runs. Works with no runs yet (shows how to start
one). Charts are drawn at the element's real pixel width (redrawn on resize), so text is never
stretched on phones.

Co-training runs (zero/cotrain.py, docs/az.md 11: a `resolver/` directory in the run): the
run's own metrics are the primary network's, `<run>/resolver/` holds the resolver network's with
the same layout; the page shows a tab per network (each with its Elo, win rates, losses,
throughput and head-to-head table), the co-training progress (games resolved, in play and parked,
episodes queued, in flight and done) and throughput. Each network's remote evaluations
(`evals.jsonl` in its directory, zero.remote: a single-game run's `eval_remote`, a co-training
run's `resolver_cfg['eval_remote']` -> `resolver/evals.jsonl`, its `draft_eval_net_remote` ->
the root's `evals.jsonl`, eval_yard_net) are merged into its rows. Other
yardsticks of a checkpoint (`GamePackage.evals`, metrics `extras`) are charted when a run has
them. Single-game runs render as before.

Run control (zero/runctl.py): each run's state and Pause / Resume buttons: POST
/api/run/<name>/pause creates <run>/PAUSE, POST /api/run/<name>/resume removes it (JSON body
required; only plain names of run directories under --runs); GET /api/run/<name>/control is
the state alone.
"""
import argparse
import glob
import html
import json
import os
import re
import time

from flask import Flask, Response, jsonify, request

from . import elo, package, runctl

app = Flask(__name__)
app.config['RUNS'] = os.path.join(package.active().root, 'runs')
RUN_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


def runs_dir():
    return app.config['RUNS']


def _json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _jsonl(p):
    rows = []
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass                      # a line being written right now
    return rows


def _metrics(d):
    """metrics.jsonl, with the remote evaluations (evals.jsonl, zero.remote) merged into their
    iterations' rows (`d`: a run directory or a co-training run's resolver/; a co-training run's
    root: its remote eval_yard_net, draft_eval_net_remote)."""
    rows = _jsonl(os.path.join(d, 'metrics.jsonl'))
    evals = {e['iter']: e for e in _jsonl(os.path.join(d, 'evals.jsonl')) if 'iter' in e}
    if evals:
        for m in rows:
            e = evals.get(m.get('iter'))
            if e:
                for k in ('eval_yard', 'eval_random', 'elo', 'eval_yard_net', 'elo_anchored'):
                    if k in e:
                        m[k] = e[k]
    return rows


def _deep(d):
    """deep_evals.jsonl: the run's long-search strength points, by iteration (the last row of each
    iter and sims)."""
    rows = {}
    for e in _jsonl(os.path.join(d, 'deep_evals.jsonl')):
        if isinstance(e.get('iter'), int) and isinstance(e.get('sims'), int) and isinstance(e.get('win'), (int, float)):
            rows[(e['iter'], e['sims'])] = e
    return [rows[k] for k in sorted(rows)]


def _tail(path, n):
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        return f.read().decode('utf-8', 'replace').splitlines()[-n:]


def _run_dir(name):
    """The directory of a known run (a plain name of a run directory under the runs dir), else
    None: no separators, no '..', no other locations."""
    if not RUN_NAME.match(name or ''):
        return None
    try:
        if name not in os.listdir(runs_dir()):
            return None
    except OSError:
        return None
    d = os.path.join(runs_dir(), name)
    if os.path.dirname(os.path.abspath(d)) != os.path.abspath(runs_dir()) or not runctl.is_run_dir(d):
        return None
    return d


def _control(d):
    st = runctl.run_status(d)
    st['now'] = time.time()
    return st


@app.route('/api/runs')
def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir(), '*'))):
        for f in ('metrics.jsonl', 'config.json'):
            p = os.path.join(d, f)
            if os.path.exists(p):
                out.append({'name': os.path.basename(d), 'mtime': os.path.getmtime(p)})
                break
    out.sort(key=lambda r: -r['mtime'])
    return jsonify(out)


def _net(d):
    """One network's view of a run directory: its settings, metrics, long-search points and Elo
    (the pool of its evaluation move rule, docs/az.md 5.4: elo/ or elo/<rule>/)."""
    tag, pool = elo.active_pool(d)
    res = _json(os.path.join(pool, 'results.json'), {}) or {}
    # the run's other pools (another rule it evaluated with before): their checkpoints' ratings, drawn
    # as their own lines - never refit with this pool's
    others = {}
    for t in elo.other_pools(d, tag):
        r = (_json(os.path.join(elo.pool_dir(d, t), 'ratings.json'), {}) or {}).get('ratings') or {}
        others[t] = {k: {'elo': v.get('elo'), 'se': v.get('se')} for k, v in r.items()
                     if k.startswith('iter_') and isinstance(v, dict) and v.get('elo') is not None}
    return {'config': _json(os.path.join(d, 'config.json'), {}), 'metrics': _metrics(d), 'deep': _deep(d),
            'ratings': _json(os.path.join(pool, 'ratings.json'), {}), 'pairs': res.get('pairs', {}),
            'anchor': res.get('anchor'), 'select': tag, 'since': res.get('since'), 'other_pools': others}


def _resolver_dir(d):
    """A co-training run's resolver directory (zero/cotrain.py), else None."""
    r = os.path.join(d, 'resolver')
    return r if any(os.path.exists(os.path.join(r, f)) for f in ('config.json', 'metrics.jsonl')) else None


@app.route('/api/run/<name>')
def run(name):
    d = _run_dir(name)
    if d is None:
        return jsonify({'error': f'no run named {name!r}'}), 404
    lp = os.path.join(d, 'log.txt')
    out = dict(_net(d), **{
        'log': _tail(lp, 60),
        'supervisor': _tail(os.path.join(d, 'supervisor.log'), 12),
        'progress': _json(os.path.join(d, 'progress.json')),
        'status': {'log_age_s': time.time() - os.path.getmtime(lp) if os.path.exists(lp) else None, 'now': time.time()},
        'stopping': os.path.exists(os.path.join(d, 'STOP')),
        'control': _control(d),
    })
    rd = _resolver_dir(d)
    if rd is not None:
        out['resolver'] = _net(rd)
    return jsonify(out)


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


@app.route('/')
def index():
    pkg = package.active()
    game = pkg.title[:-5] if pkg.title.endswith(' Zero') else pkg.title
    return Response(PAGE.replace('__TITLE__', html.escape(pkg.title)).replace('__GAME__', html.escape(game)),
                    mimetype='text/html')


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--bg:#f4f3ef;--surface-1:#fcfcfb;--line:#e4e2dc;--grid:#ecebe6;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#8a8984;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1a9c74;--ref:#a7a59e;--good:#1a7f37;--warn:#b25e00;--act-pause:#b3412a;--act-resume:#1d6fd1;--on-act:#fff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--series-3:#2fb88a;--ref:#6b6a65;--good:#4ac26b;--warn:#e0a040;--act-pause:#c9553c;--act-resume:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--series-3:#2fb88a;--ref:#6b6a65;--good:#4ac26b;--warn:#e0a040;--act-pause:#c9553c;--act-resume:#3987e5}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text-primary);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:0}
.sub{color:var(--text-secondary);font-size:13px;margin-top:2px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:14px}
select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface-1);color:var(--text-primary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.tile .l{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-secondary)}
.tile .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.15}
.tile .n{font-size:12.5px;color:var(--muted)}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:14px}
.card h2{font-size:14px;margin:0}
.card .cap{font-size:12.5px;color:var(--text-secondary);margin:2px 0 6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-bottom:14px}
.grid2 .card{margin-bottom:0}
svg{display:block;width:100%;overflow:visible}
svg text{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--text-secondary);margin-bottom:4px}
.legend i{display:inline-block;width:10px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 6px 18px rgba(0,0,0,.18);z-index:9;color:var(--text-primary)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-secondary);text-align:right;padding:4px 6px;font-weight:600}
td{text-align:right;padding:4px 6px;border-top:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.tbl{overflow-x:auto}
pre{font-size:11.5px;white-space:pre-wrap;word-break:break-word;margin:0;color:var(--text-secondary);max-height:260px;overflow:auto}
code{font-size:12.5px;background:var(--grid);padding:1px 5px;border-radius:5px;word-break:break-all}
details summary{cursor:pointer;font-weight:600;font-size:14px}
.status{font-size:13px;color:var(--text-secondary);margin:-4px 0 12px}
.ctl{display:flex;align-items:center;justify-content:space-between;gap:10px 14px;flex-wrap:wrap;background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:0 0 12px}
.ctl-state{display:flex;align-items:center;gap:6px 10px;flex-wrap:wrap;min-width:0;flex:1 1 240px}
.ctl-state b{font-size:16px;font-weight:650}
.ctl-detail{font-size:13px;color:var(--text-secondary);flex-basis:100%;margin-left:20px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;background:var(--muted)}
.dot.running{background:var(--good)} .dot.paused{background:var(--act-resume)}
.dot.pausing,.dot.resuming,.dot.starting,.dot.restarting,.dot.stopping{background:var(--warn);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.3}}
@media (prefers-reduced-motion:reduce){.dot{animation:none!important}}
.btn{font:inherit;font-weight:650;font-size:15px;min-height:44px;min-width:128px;padding:10px 20px;border-radius:10px;border:0;cursor:pointer;color:var(--on-act);background:var(--act-resume);touch-action:manipulation}
.btn.pause{background:var(--act-pause)}
.btn:disabled{opacity:.55;cursor:default}
@media (max-width:520px){.btn{flex:1 1 100%}}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}
.tab{font:inherit;font-size:14px;font-weight:600;min-height:40px;padding:8px 14px;border-radius:10px;border:1px solid var(--line);background:var(--surface-1);color:var(--text-secondary);cursor:pointer;touch-action:manipulation}
.tab.on{background:var(--text-primary);color:var(--bg);border-color:var(--text-primary)}
.empty{padding:28px 18px;text-align:center}
.empty p{margin:6px 0;color:var(--text-secondary)}
</style></head><body><div class="wrap">
<header><div><h1>__TITLE__</h1><div class="sub" id="sub">AlphaZero training for __GAME__</div></div>
<label class="sub" id="runlab" hidden>Run <select id="runsel"></select></label></header>
<div class="card empty" id="empty" hidden><h2>No runs yet</h2>
<p>Start one (from the repository root):</p>
<p><code>python tools/launch_hidden.py runs/toy1/supervisor_console.txt python tools/supervise.py runs/toy1 --preset toy-1</code></p>
<p>Runs appear here as soon as they write their config.</p></div>
<div id="main" hidden>
<div class="ctl" id="ctl" hidden><div class="ctl-state"><span class="dot" id="ctl-dot"></span><b id="ctl-label"></b><span class="ctl-detail" id="ctl-detail"></span></div>
<button type="button" class="btn" id="ctl-btn" hidden></button></div>
<div class="tabs" id="tabs" role="tablist" hidden></div>
<div class="status" id="status"></div>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Elo by iteration</h2>
<div class="cap" id="elo-cap"></div>
<div class="legend" id="elo-leg"></div>
<svg id="elo" style="height:300px" role="img" aria-label="Elo by iteration"></svg></div>
<div class="card"><h2 id="win-t">Win rate against the yardstick</h2>
<div class="cap">Each checkpoint against the yardstick and the random bot, seats alternating, the network searching with the eval sims. Band = 95% interval.</div>
<div class="legend" id="win-leg"></div>
<svg id="win" style="height:260px" role="img" aria-label="Win rate by iteration"></svg></div>
<div class="card" id="deep-card" hidden><h2 id="deep-t">Win rate with a long search</h2>
<div class="cap" id="deep-cap"></div><div class="legend" id="deep-leg"></div>
<svg id="deep" style="height:240px" role="img" aria-label="Win rate with a long search by iteration"></svg></div>
<div class="card" id="extra-card" hidden><h2>Other yardsticks</h2>
<div class="cap" id="extra-cap"></div><div class="legend" id="extra-leg"></div>
<svg id="extra" style="height:220px" role="img" aria-label="Other yardsticks by iteration"></svg></div>
<div class="card"><h2>Each checkpoint against the one before it</h2>
<div class="cap">Each rated checkpoint vs the previous rated one (N&minus;1, or N&minus;eval_every when evaluation runs every few iterations), both networks searching with the eval sims, seats alternated. Above 50%: the newer network is stronger. Band = 95% interval.</div>
<svg id="prev" style="height:220px" role="img" aria-label="Win rate against the previous iteration"></svg></div>
<div class="card"><h2 id="h2h-t">Newest checkpoint: head to head</h2>
<div class="cap">Every opponent the newest rated checkpoint has played. "Elo predicts" is the win rate the fitted ratings expect, as a consistency check.</div>
<div class="tbl" id="h2h"></div></div>
<div class="grid2">
<div class="card"><h2>Losses</h2><div class="cap">Policy cross-entropy against the visit counts, value cross-entropy against the blended target, margin regression</div><div class="legend" id="loss-leg"></div><svg id="loss" style="height:180px"></svg></div>
<div class="card"><h2>Self-play average final score</h2><div class="cap">Per player, per game</div><svg id="sc" style="height:180px"></svg></div>
<div class="card"><h2>Self-play throughput</h2><div class="cap">Network leaves evaluated per second during self-play</div><svg id="lps" style="height:180px"></svg></div>
<div class="card"><h2>Minutes per iteration</h2><div class="cap">Self-play, training, evaluation and Elo matches</div><div class="legend" id="min-leg"></div><svg id="mins" style="height:180px"></svg></div>
</div>
<div class="card"><details><summary>Table</summary><div class="tbl" id="table"></div></details></div>
<div class="card"><details><summary>Log tail</summary><pre id="log"></pre></details></div>
<div class="card"><details><summary>Supervisor log</summary><pre id="sup"></pre></details></div>
</div></div><div class="tip" id="tip" hidden></div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let RUN=null, DATA=null, VIEW='primary';
function ns(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e}
const pct=x=>(100*x).toFixed(1)+'%';
const ago=s=>s<90?Math.round(s)+' s':s<5400?Math.round(s/60)+' min':(s/3600).toFixed(1)+' h';
function autoFmt(v,range){return range>=40?Math.round(v).toLocaleString():range>=4?v.toFixed(1):range>=.4?v.toFixed(2):v.toFixed(3)}
function legend(el,series){el.innerHTML=series.map(s=>`<span><i style="background:${css(s.color)}"></i>${s.label}</span>`).join('')}
// series: [{pts:[{x,y,lo?,hi?,...}], color, label, tip}]
function chart(svg,series,opt){
  svg.innerHTML='';
  const W=Math.max(260,svg.clientWidth||1000),H=parseFloat(svg.style.height)||160,L=48,R=12,T=10,B=24;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const all=series.flatMap(s=>s.pts);
  if(!all.length){const t=ns('text',{x:W/2,y:H/2,'text-anchor':'middle'});t.textContent='No data yet';svg.appendChild(t);return}
  const xs=all.map(p=>p.x),x0=Math.min(opt.x0??Infinity,...xs),x1=Math.max(opt.x1??-Infinity,...xs,x0+1);
  let y0=opt.y0,y1=opt.y1;
  if(y0==null||y1==null){const ys=all.flatMap(p=>p.lo!=null?[p.lo,p.hi]:[p.y]).concat((opt.refs||[]).map(r=>r.y));const lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo)*.12||Math.abs(hi)*.1||1;y0=y0??lo-pad;y1=y1??hi+pad}
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:css('--grid'),'stroke-width':1}));
    const t=ns('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'});t.textContent=opt.fmt?opt.fmt(v):autoFmt(v,y1-y0);svg.appendChild(t)}
  const nt=Math.min(8,x1-x0);for(let i=0;i<=nt;i++){const v=Math.round(x0+(x1-x0)*i/nt);const t=ns('text',{x:X(v),y:H-6,'text-anchor':'middle'});t.textContent=v;svg.appendChild(t)}
  for(const r of (opt.refs||[])){svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(r.y),y2:Y(r.y),stroke:css('--ref'),'stroke-width':1,'stroke-dasharray':'4 3'}));
    const t=ns('text',{x:L+4,y:Y(r.y)-4,'text-anchor':'start'});t.textContent=r.label;svg.appendChild(t)}
  for(const s of series){
    const pts=s.pts;if(!pts.length)continue;const col=css(s.color);
    if(pts.length===1&&pts[0].lo!=null)svg.appendChild(ns('line',{x1:X(pts[0].x),x2:X(pts[0].x),y1:Y(pts[0].lo),y2:Y(pts[0].hi),stroke:col,'stroke-width':2,opacity:.35}));
    else if(pts[0].lo!=null){let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.hi));for(let i=pts.length-1;i>=0;i--)d+='L'+X(pts[i].x)+','+Y(pts[i].lo);svg.appendChild(ns('path',{d:d+'Z',fill:col,opacity:.12}))}
    let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.y));
    svg.appendChild(ns('path',{d,fill:'none',stroke:col,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'}));
    const last=pts[pts.length-1];svg.appendChild(ns('circle',{cx:X(last.x),cy:Y(last.y),r:4,fill:col,stroke:css('--surface-1'),'stroke-width':2}));
  }
  const cross=ns('line',{y1:T,y2:H-B,stroke:css('--ref'),'stroke-width':1,visibility:'hidden'});svg.appendChild(cross);
  const hit=ns('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'});svg.appendChild(hit);
  const tip=$('tip');const xsU=[...new Set(xs)].sort((a,b)=>a-b);
  hit.addEventListener('pointermove',ev=>{const b=svg.getBoundingClientRect();const fx=(ev.clientX-b.left)/b.width*W;let bx=xsU[0];for(const x of xsU)if(Math.abs(X(x)-fx)<Math.abs(X(bx)-fx))bx=x;
    cross.setAttribute('x1',X(bx));cross.setAttribute('x2',X(bx));cross.setAttribute('visibility','visible');
    const lines=series.map(s=>{const p=s.pts.find(q=>q.x===bx);return p?`${series.length>1?s.label+': ':''}<b>${s.tip?s.tip(p):autoFmt(p.y,y1-y0)}</b>`:null}).filter(Boolean);
    tip.hidden=false;tip.innerHTML=`Iteration ${bx}<br>`+lines.join('<br>');tip.style.left=Math.min(ev.clientX+14,innerWidth-220)+'px';tip.style.top=(ev.clientY-10)+'px'});
  hit.addEventListener('pointerleave',()=>{tip.hidden=true;cross.setAttribute('visibility','hidden')});
}
function tile(l,v,n){return `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n||'&nbsp;'}</div></div>`}
const winPts=(M,k)=>M.filter(m=>m[k]&&m[k].games).map(m=>({x:m.iter,y:m[k].win,lo:Math.max(0,m[k].win-1.96*m[k].se),hi:Math.min(1,m[k].win+1.96*m[k].se),e:m[k]}));
const CTL={running:['Running','pause'],stopping:['Stopping after this iteration','pause'],pausing:['Pausing','resume'],
  paused:['Paused','resume'],resuming:['Resuming',null],starting:['Starting','pause'],restarting:['Restarting after a crash','pause'],stopped:['Stopped',null]};
function renderCtl(c){
  const el=$('ctl');if(!c||!c.state){el.hidden=true;return}
  el.hidden=false;const [label,act]=CTL[c.state]||[c.state,null];
  $('ctl-dot').className='dot '+c.state;
  let lab=label;if(c.state==='paused'&&c.paused_since)lab+=` (${ago((c.now||Date.now()/1000)-c.paused_since)})`;
  $('ctl-label').textContent=lab;$('ctl-detail').textContent=c.detail||'';$('ctl-detail').hidden=!c.detail;
  const b=$('ctl-btn');
  if(!act||(act==='resume'&&!c.pause)){b.hidden=true}
  else{b.hidden=false;b.disabled=false;b.dataset.act=act;b.className='btn '+act;b.textContent=act==='pause'?'Pause':'Resume'}
}
$('ctl-btn').onclick=async()=>{
  const b=$('ctl-btn'),act=b.dataset.act,run=RUN;if(!act||!run)return;
  if(act==='pause'&&!confirm(`Pause ${run}?\n\nThe run stops within seconds; games in progress are dropped. Resume continues from the newest checkpoint.`))return;
  b.disabled=true;
  try{const r=await fetch('api/run/'+encodeURIComponent(run)+'/'+act,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:act})});
    const j=await r.json();if(!r.ok){alert(j.error||('HTTP '+r.status));b.disabled=false;return}
    renderCtl(j.control);setTimeout(load,2000)}catch(e){alert('Request failed: '+e);b.disabled=false}};
function renderTabs(){
  const tb=$('tabs'),co=!!DATA.resolver;tb.hidden=!co;if(!co){tb.innerHTML='';return}
  const pg=(DATA.config||{}).game||'primary',rg=(DATA.resolver.config||{}).game||'resolver';
  tb.innerHTML=[['primary',`${pg} network`],['resolver',`${rg} network (resolver)`]].map(([k,l])=>`<button type="button" role="tab" aria-selected="${VIEW===k}" class="tab${VIEW===k?' on':''}" data-v="${k}">${l}</button>`).join('');
  tb.querySelectorAll('button').forEach(b=>b.onclick=()=>{VIEW=b.dataset.v;render()});
}
const newestElo=V=>{const R=(V&&V.ratings&&V.ratings.ratings)||{};const k=Object.keys(R).filter(k=>k.startsWith('iter_')).sort().pop();return k?{x:+k.slice(5),y:R[k].elo,se:R[k].se}:null};
function render(){
  const co=!!DATA.resolver,rv=co&&VIEW==='resolver';
  const V=rv?Object.assign({},DATA,DATA.resolver):DATA;
  const M=V.metrics,cfg=V.config;
  renderCtl(DATA.control);
  renderTabs();
  const yard=V.anchor||cfg.yardstick||'yardstick';
  const R=(V.ratings&&V.ratings.ratings)||{};
  const sp=M.filter(m=>m.selfplay),tr=M.filter(m=>m.train);
  const games=sp.reduce((s,m)=>s+m.selfplay.games,0);
  const t0=M.length?M[0].time:0,tl=M.length?M[M.length-1].time:0;
  const eloPts=Object.keys(R).filter(k=>k.startsWith('iter_')).map(k=>({x:+k.slice(5),y:R[k].elo,lo:R[k].elo-1.96*R[k].se,hi:R[k].elo+1.96*R[k].se,r:R[k]})).sort((a,b)=>a.x-b.x);
  // the best: a checkpoint evaluated in this pool (one rated through Elo matches only is "evaluated": false)
  const bestElo=eloPts.filter(p=>p.r.evaluated!==false).reduce((b,p)=>!b||p.y>b.y?p:b,null),lastElo=eloPts[eloPts.length-1];
  const wy=winPts(M,'eval_yard'),wr=winPts(M,'eval_random');const ly=wy[wy.length-1],lr=wr[wr.length-1];
  const P=V.pairs||{},nm=i=>'iter_'+String(i).padStart(4,'0');
  const rec=(a,b)=>{const k=a<b?a+'|'+b:b+'|'+a,p=P[k];if(!p||!p[1])return null;const s=a<b?p[0]:p[1]-p[0],w=s/p[1];return {win:w,n:p[1],se:Math.sqrt(Math.max(w*(1-w),1e-6)/p[1])}};
  const prevPts=eloPts.map((p,i)=>i?{x:p.x,px:eloPts[i-1].x,r:rec(nm(p.x),nm(eloPts[i-1].x))}:{x:p.x,r:null}).filter(p=>p.r).map(p=>({x:p.x,px:p.px,y:p.r.win,lo:Math.max(0,p.r.win-1.96*p.r.se),hi:Math.min(1,p.r.win+1.96*p.r.se),r:p.r}));
  const lsp=sp[sp.length-1];
  const D=V.deep||[],dsims=[...new Set(D.map(r=>r.sims))].sort((a,b)=>a-b);
  const dmain=dsims.reduce((b,s)=>D.filter(r=>r.sims===s).length>D.filter(r=>r.sims===b).length?s:b,dsims[0]);
  const dl=D.filter(r=>r.sims===dmain).pop();
  $('sub').textContent=`Run ${RUN} · ${cfg.game||''} ${(cfg.players||[]).join('/')}-player · self-play ${cfg.sims}/${cfg.sims_fast} sims${cfg.rolling?' (rolling)':''} · eval ${cfg.eval_sims} sims · yardstick ${yard} = Elo 1000`;
  const pr=DATA.progress,st=DATA.status;let s=[];
  const live=DATA.control&&['running','stopping'].includes(DATA.control.state);
  if(pr&&live&&pr.phase==='selfplay'&&pr.events_resolved!=null)s.push(`Iteration ${pr.iter}: self-play ${pr.events_resolved}/${pr.target} events resolved · ${pr.drafting} drafting · ${pr.pending_events} waiting for their matches · matches ${pr.matches_queued} queued, ${pr.matches_in_flight} in flight, ${pr.matches_done} done`);
  else if(pr&&live&&pr.phase==='selfplay')s.push(`Iteration ${pr.iter}: self-play ${pr.games}/${pr.target} games`);
  else if(pr&&live&&pr.phase==='train')s.push(`Iteration ${pr.iter}: training`);
  else if(pr&&live&&pr.phase==='eval')s.push(`Iteration ${pr.iter}: evaluation and Elo matches`);
  else if(pr&&live&&pr.phase==='remote shards')s.push(`Iteration ${pr.iter}: the remote's last shards`);
  const rp=pr&&live&&pr.phase==='selfplay'?(pr.remote||pr.matches_remote):null;
  if(rp)s.push(`remote ${rp.host}: ${rp.state}, ${rp.workers} worker(s)`+(rp.outstanding!=null?`, ${rp.matches} matches back, ${rp.outstanding} outstanding`:`, ${rp.games} games, ${rp.samples.toLocaleString()} samples back`));
  if(st.log_age_s!=null)s.push(`last log line ${ago(st.log_age_s)} ago`);
  if(DATA.stopping)s.push('STOP file: ending after this iteration');
  $('status').textContent=s.join(' · ');
  $('tiles').innerHTML=tile('Iteration',M.length?M[M.length-1].iter:'–',cfg.iters?`of ${cfg.iters}`:'')+
    tile('Elo (newest)',lastElo?Math.round(lastElo.y):'–',lastElo?`±${Math.round(1.96*lastElo.r.se)} · iteration ${lastElo.x}`:'')+
    tile('Best Elo',bestElo?Math.round(bestElo.y):'–',bestElo?`iteration ${bestElo.x} · ${yard} = 1000`:'')+
    tile('vs '+yard,ly?pct(ly.y):'–',ly?`±${(196*ly.e.se).toFixed(1)} pts · margin ${ly.e.margin>0?'+':''}${ly.e.margin}`:'')+
    tile('vs random',lr?pct(lr.y):'–',lr?`±${(196*lr.e.se).toFixed(1)} pts · ${lr.e.games} games`+(ly&&lr.x!==ly.x?` · iteration ${lr.x}`:''):'')+
    (dl?tile(`vs ${dl.opponent||yard} · ${dl.sims} sims`,pct(dl.win),`±${(196*dl.se).toFixed(1)} pts · iteration ${dl.iter}`):'')+
    tile('Self-play games',games.toLocaleString(),`${((tl-t0)/3600).toFixed(1)} h elapsed`)+
    tile('Leaves / s',lsp?Math.round(lsp.selfplay.leaves_per_s).toLocaleString():'–',lsp?`${Math.round(lsp.selfplay.games_per_h).toLocaleString()} games/h`:'');
  if(co){
    const oe=newestElo(rv?DATA:DATA.resolver),og=((rv?DATA.config:DATA.resolver.config)||{}).game||'';
    const ct=(rv?DATA.metrics:M).filter(m=>m.selfplay&&m.selfplay.cotrain).pop(),c=ct&&ct.selfplay.cotrain;
    $('tiles').innerHTML+=tile(`${og} network Elo`,oe?Math.round(oe.y):'–',oe?`±${Math.round(1.96*oe.se)} · iteration ${oe.x}`:'the other tab')+
      (c?tile('Matches / h',Math.round(c.episodes_per_h).toLocaleString(),`${c.episodes.toLocaleString()} matches, ${c.events} events · iteration ${ct.iter}`)+
        tile('Event latency',c.event_latency_s?ago(c.event_latency_s[0]):'–',`park to resolve, mean · ${c.pending_events} pending · parked ${c.parked_mb_max} MB max`):'');
  }
  const refs=[{y:1000,label:yard+' 1000'}];
  if(R.random&&yard!=='random')refs.push({y:R.random.elo,label:'random '+Math.round(R.random.elo)});
  const offs=cfg.elo_offsets?`the checkpoints ${cfg.elo_offsets.split(',').join(', ')} iterations before it (those never evaluated are rated through these games only)`:`the checkpoints 1, 2, 4, 8, 16 and 32 evaluations before it`;
  const OP=V.other_pools||{},opn=Object.keys(OP);
  const pool=(V.select&&V.select!=='visits'?` Moves by the rule ${V.select} (its own Elo pool, from iteration ${V.since!=null?V.since:'?'}).`:'')+
    (opn.length?` The other lines: the run's pools of the rules it evaluated with before (${opn.join(', ')}); Elo compares within one pool only.`:'');
  $('elo-cap').textContent=`The yardstick (${yard}) is fixed at 1000. Each evaluated checkpoint plays the yardstick, ${cfg.eval_random_games===0?'':'the random bot, '}${offs}; all games are refit together after every iteration (Bradley-Terry). Band = 95% interval.${pool}`;
  const es=[{pts:eloPts,color:'--series-1',label:opn.length?`Elo (${V.select||'visits'})`:'Elo',tip:p=>`${Math.round(p.y)} ±${Math.round(1.96*p.r.se)} (${p.r.games} games)${p.r.evaluated===false?' · not evaluated: Elo matches only':''}`}];
  opn.forEach((t,i)=>es.push({pts:Object.keys(OP[t]).map(k=>({x:+k.slice(5),y:OP[t][k].elo,r:OP[t][k]})).sort((a,b)=>a.x-b.x),color:i?'--series-3':'--series-2',label:`Elo (${t})`,tip:p=>`${Math.round(p.y)} ±${Math.round(1.96*p.r.se)} (pool ${t})`}));
  legend($('elo-leg'),opn.length?es:[]);
  chart($('elo'),es,{refs});
  $('win-t').textContent='Win rate against '+yard+(wr.length?' and the random bot':'');
  const ws=[{pts:wy,color:'--series-1',label:yard,tip:p=>`${pct(p.y)} ±${(196*p.e.se).toFixed(1)} · margin ${p.e.margin}`},{pts:wr,color:'--series-2',label:'random',tip:p=>`${pct(p.y)} ±${(196*p.e.se).toFixed(1)}`}];
  const wn=winPts(M,'eval_yard_net');
  if(wn.length)ws.push({pts:wn,color:'--series-3',label:`${yard} (episodes played by the resolver network)`,tip:p=>`${pct(p.y)} ±${(196*p.e.se).toFixed(1)} · margin ${p.e.margin}`});
  legend($('win-leg'),ws);chart($('win'),ws,{y0:0,y1:1,fmt:v=>Math.round(v*100)+'%',refs:[{y:.5,label:'even'}]});
  $('deep-card').hidden=!D.length;
  if(D.length){
    const dopp=[...new Set(D.map(r=>r.opponent||yard))].join(', '),cols=['--series-1','--series-2','--series-3'];
    const ds=dsims.map((n,i)=>({pts:D.filter(r=>r.sims===n).map(r=>({x:r.iter,y:r.win,lo:Math.max(0,r.win-1.96*r.se),hi:Math.min(1,r.win+1.96*r.se),r})),color:cols[i%3],label:n+' sims',
      tip:p=>`${pct(p.y)} ±${(196*p.r.se).toFixed(1)} · ${p.r.games} games · margin ${p.r.margin>0?'+':''}${p.r.margin}`}));
    $('deep-t').textContent=`Win rate against ${dopp} with a long search`;
    $('deep-cap').textContent=`Occasional checkpoints against ${dopp}, the network searching with ${dsims.join(' or ')} sims (the charts above use the eval sims, ${cfg.eval_sims}), on the same boards each time. Band = 95% interval.`;
    legend($('deep-leg'),ds);chart($('deep'),ds,{fmt:v=>Math.round(v*100)+'%',x1:M.length?M[M.length-1].iter:null});
  }
  const X=M.filter(m=>m.extras),xn=[...new Set(X.flatMap(m=>Object.keys(m.extras)))];
  $('extra-card').hidden=!xn.length;
  if(xn.length){
    const last=X[X.length-1].extras,cols=['--series-1','--series-2','--series-3'],pc=k=>!!(last[k]&&last[k].pct);
    const xs=xn.map((k,i)=>({pts:X.filter(m=>m.extras[k]&&typeof m.extras[k].value==='number').map(m=>({x:m.iter,y:m.extras[k].value,e:m.extras[k]})),color:cols[i%3],label:(last[k]&&last[k].label)||k,
      tip:p=>(p.e.pct?pct(p.y):autoFmt(p.y,1))+(p.e.n?` · ${p.e.n.toLocaleString()} positions`:'')}));
    const refs=xn.flatMap(k=>Object.entries((last[k]&&last[k].refs)||{}).map(([n,v])=>({y:v,label:`${n} ${pc(k)?pct(v):autoFmt(v,1)}`})));
    $('extra-cap').textContent=(xn.map(k=>(last[k]&&last[k].caption)||'').filter(Boolean).join(' ')||'Measured on every evaluated checkpoint; never a training signal. Dashed lines: references on the same positions.')+
      xn.filter(k=>last[k]&&last[k].error).map(k=>` ${k} failed at iteration ${X[X.length-1].iter}: ${last[k].error}`).join('');
    legend($('extra-leg'),xs);chart($('extra'),xs,xn.every(pc)?{y0:0,y1:1,fmt:v=>Math.round(v*100)+'%',refs}:{refs});
  }
  chart($('prev'),[{pts:prevPts,color:'--series-3',label:'vs previous',tip:p=>`vs iter ${p.px}: ${pct(p.y)} ±${(196*p.r.se).toFixed(1)} (${p.r.n} games)`}],{y0:0,y1:1,fmt:v=>Math.round(v*100)+'%',refs:[{y:.5,label:'even'}]});
  const newest=lastElo?lastElo.x:null;
  if(newest!==null){
    const me=nm(newest),opp=Object.keys(P).flatMap(k=>{const [a,b]=k.split('|');return a===me?[b]:b===me?[a]:[]});
    opp.sort((a,b)=>a===yard?-1:b===yard?1:b.localeCompare(a));
    const label=o=>o.startsWith('iter_')?'Iteration '+(+o.slice(5)):o;
    const pred=o=>R[o]&&R[me]?1/(1+Math.pow(10,-(R[me].elo-R[o].elo)/400)):null;
    $('h2h-t').textContent=`Newest checkpoint (iteration ${newest}, Elo ${Math.round(R[me].elo)}): head to head`;
    $('h2h').innerHTML='<table><tr><th>Opponent</th><th>Opponent Elo</th><th>Games</th><th>Newest wins</th><th>±95%</th><th>Elo predicts</th></tr>'+
      opp.map(o=>{const r=rec(me,o),q=pred(o);return `<tr><td>${label(o)}</td><td>${R[o]?Math.round(R[o].elo):''}</td><td>${r.n}</td><td>${pct(r.win)}</td><td>${(196*r.se).toFixed(1)}</td><td>${q==null?'':pct(q)}</td></tr>`}).join('')+'</table>';
  } else $('h2h').innerHTML='<div class="cap">No rated checkpoint yet.</div>';
  const ls=[{pts:tr.map(m=>({x:m.iter,y:m.train.policy})),color:'--series-1',label:'policy'},{pts:tr.map(m=>({x:m.iter,y:m.train.value})),color:'--series-2',label:'value'},{pts:tr.map(m=>({x:m.iter,y:m.train.margin})),color:'--series-3',label:'margin'}];
  legend($('loss-leg'),ls);chart($('loss'),ls,{y0:0});
  chart($('sc'),[{pts:sp.map(m=>({x:m.iter,y:m.selfplay.avg_score})),color:'--series-1',label:'score'}],{});
  chart($('lps'),[{pts:sp.map(m=>({x:m.iter,y:m.selfplay.leaves_per_s})),color:'--series-1',label:'leaves/s',tip:p=>Math.round(p.y).toLocaleString()}],{y0:0});
  const ph=M.filter(m=>m.phases),mk=(k,c,l)=>({pts:ph.filter(m=>m.phases[k]!=null).map(m=>({x:m.iter,y:m.phases[k]/60})),color:c,label:l,tip:p=>p.y.toFixed(2)+' min'});
  const ms=[mk('selfplay_s','--series-1','self-play'),mk('train_s','--series-2','training'),mk('eval_s','--series-3','evaluation'),mk('elo_s','--ref','Elo')];
  legend($('min-leg'),ms);chart($('mins'),ms,{y0:0});
  const er=i=>R[nm(i)]?Math.round(R[nm(i)].elo)+' ±'+Math.round(1.96*R[nm(i)].se):'';
  $('table').innerHTML='<table><tr><th>Iter</th><th>Elo</th><th>vs '+yard+'</th><th>vs random</th><th>vs previous</th><th>Policy</th><th>Value</th><th>Games</th><th>Samples</th><th>Leaves/s</th><th>Minutes</th></tr>'+
    M.slice().reverse().map(m=>{const pv=m.iter>0?rec(nm(m.iter),nm(m.iter-1)):null;return `<tr><td>${m.iter}</td><td>${er(m.iter)}</td><td>${m.eval_yard?pct(m.eval_yard.win):''}</td><td>${m.eval_random?pct(m.eval_random.win):''}</td><td>${pv?pct(pv.win):''}</td><td>${m.train?m.train.policy.toFixed(3):''}</td><td>${m.train?m.train.value.toFixed(3):''}</td><td>${m.selfplay?m.selfplay.games:''}</td><td>${m.selfplay?m.selfplay.samples.toLocaleString():''}</td><td>${m.selfplay?Math.round(m.selfplay.leaves_per_s).toLocaleString():''}</td><td>${m.seconds?(m.seconds/60).toFixed(1):''}</td></tr>`}).join('')+'</table>';
  $('log').textContent=DATA.log.map(l=>l.length>400?l.slice(0,400)+'…':l).join('\n');
  $('sup').textContent=(DATA.supervisor||[]).join('\n')||'No supervisor log (run started directly).';
}
async function load(){
  let runs=[];try{runs=await (await fetch('api/runs')).json()}catch(e){}
  $('empty').hidden=runs.length>0;$('main').hidden=!runs.length;$('runlab').hidden=!runs.length;
  if(!runs.length)return;
  if(!RUN||!runs.some(r=>r.name===RUN))RUN=runs[0].name;
  $('runsel').innerHTML=runs.map(r=>`<option${r.name===RUN?' selected':''}>${r.name}</option>`).join('');
  const r=await fetch('api/run/'+encodeURIComponent(RUN));if(!r.ok)return;
  DATA=await r.json();render();
}
$('runsel').onchange=e=>{RUN=e.target.value;load()};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>DATA&&render());
let rsz=null;addEventListener('resize',()=>{clearTimeout(rsz);rsz=setTimeout(()=>DATA&&render(),150)});
load();setInterval(load,30000);
</script></body></html>'''


def main(argv=None):
    pkg = package.active()
    ap = argparse.ArgumentParser(description=f'{pkg.title} training dashboard')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=pkg.dashboard_port, required=pkg.dashboard_port is None,
                    help=f'default {pkg.dashboard_port} (the hub registry\'s port for {pkg.hub_id})' if pkg.dashboard_port
                    else 'required (the core has no registered port)')
    ap.add_argument('--runs', default=None, help='runs directory (default: <repo>/runs)')
    a = ap.parse_args(argv)
    if a.runs:
        app.config['RUNS'] = os.path.abspath(a.runs)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
