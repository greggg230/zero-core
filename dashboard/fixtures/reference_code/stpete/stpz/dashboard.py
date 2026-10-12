"""Training dashboard: python -m stpz.dashboard [--port 5230]

Reads runs/*/metrics.jsonl and log.txt. Binds 0.0.0.0 so it is reachable over Tailscale.
"""
import argparse
import glob
import json
import os

from flask import Flask, jsonify, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
app = Flask(__name__)


def _metrics(run):
    rows = []
    p = os.path.join(RUNS, run, 'metrics.jsonl')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@app.route('/api/runs')
def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, '*'))):
        if os.path.exists(os.path.join(d, 'metrics.jsonl')):
            out.append({'name': os.path.basename(d), 'mtime': os.path.getmtime(os.path.join(d, 'metrics.jsonl'))})
    out.sort(key=lambda r: -r['mtime'])
    return jsonify(out)


@app.route('/api/run/<name>')
def run(name):
    d = os.path.join(RUNS, os.path.basename(name))
    cfg = {}
    if os.path.exists(os.path.join(d, 'config.json')):
        cfg = json.load(open(os.path.join(d, 'config.json')))
    log = []
    if os.path.exists(os.path.join(d, 'log.txt')):
        log = open(os.path.join(d, 'log.txt'), encoding='utf-8').read().splitlines()[-40:]
    ratings = {}
    rp = os.path.join(d, 'elo', 'ratings.json')
    if os.path.exists(rp):
        ratings = json.load(open(rp))
    pairs = {}
    pp = os.path.join(d, 'elo', 'results.json')
    if os.path.exists(pp):
        pairs = json.load(open(pp)).get('pairs', {})
    return jsonify({'config': cfg, 'metrics': _metrics(os.path.basename(name)), 'log': log, 'ratings': ratings,
                    'pairs': pairs,
                    'stopping': os.path.exists(os.path.join(d, 'STOP'))})


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>St. Pete Zero</title>
<style>
:root{color-scheme:light;--bg:#f4f3ef;--surface-1:#fcfcfb;--line:#e4e2dc;--grid:#ecebe6;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#8a8984;
--series-1:#2a78d6;--series-2:#eb6834;--ref:#a7a59e;--good:#1a7f37}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--ref:#6b6a65;--good:#4ac26b}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface-1:#1a1a19;--line:#2c2c2a;--grid:#262624;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#8d8c86;--series-1:#3987e5;--series-2:#d95926;--ref:#6b6a65;--good:#4ac26b}
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
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.grid2 .card{margin-bottom:0}
svg{display:block;width:100%;overflow:visible}
svg text{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 6px 18px rgba(0,0,0,.18);z-index:9;color:var(--text-primary)}
.tip b{font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-secondary);text-align:right;padding:4px 6px;font-weight:600}
td{text-align:right;padding:4px 6px;border-top:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.tbl{overflow-x:auto}
pre{font-size:11.5px;white-space:pre-wrap;word-break:break-word;margin:0;color:var(--text-secondary);max-height:260px;overflow:auto}
details summary{cursor:pointer;font-weight:600;font-size:14px}
</style></head><body><div class="wrap">
<header><div><h1>St. Pete Zero</h1><div class="sub" id="sub">AlphaZero training for St. Petersburg</div></div>
<label class="sub">Run <select id="runsel"></select></label></header>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Elo by iteration</h2>
<div class="cap">The web app's Shrewd AI is fixed at 1000. Each checkpoint plays 200-game matches against the checkpoints 1, 2, 4, 8, 16 and 32 iterations before it, plus its 800 games against Shrewd; all games are refit together after every iteration. Band = 95% interval.</div>
<svg id="elo" viewBox="0 0 1000 300" preserveAspectRatio="none" style="height:300px" role="img" aria-label="Elo by iteration"></svg></div>
<div class="card" id="refcard" hidden><h2 id="ref-t">Against the reference network</h2>
<div class="cap">This run's checkpoints against a fixed network from another run (each side with its own run's search settings), 400 games per point. Above 50% means this run has pulled ahead. Iteration 0 is the reference's own weights, so it shows the effect of the search settings alone. Band = 95% interval.</div>
<svg id="refc" viewBox="0 0 1000 260" preserveAspectRatio="none" style="height:260px" role="img" aria-label="Win rate against the reference network"></svg></div>
<div class="card"><h2>Each checkpoint against the one before it</h2>
<div class="cap">Iteration N vs iteration N&minus;1: 200 games, both networks searching with the eval sims, seats alternated. Above 50% means the newer network is stronger. Band = 95% interval.</div>
<svg id="prev" viewBox="0 0 1000 260" preserveAspectRatio="none" style="height:260px" role="img" aria-label="Win rate against the previous iteration"></svg></div>
<div class="card"><h2 id="h2h-t">Newest checkpoint: head to head</h2>
<div class="cap">Every opponent the newest rated checkpoint has played. "Elo predicts" is the win rate the fitted ratings expect, as a consistency check.</div>
<div class="tbl" id="h2h"></div></div>
<div class="card"><h2>Win rate against the web app's Shrewd AI</h2>
<div class="cap">2-player, 800 games per point, network searches with the eval sims shown in the config. Band = 95% interval. Reference lines: Shrewd vs itself, Casual vs Shrewd.</div>
<svg id="win" viewBox="0 0 1000 300" preserveAspectRatio="none" style="height:300px" role="img" aria-label="Win rate by iteration"></svg></div>
<div class="grid2" id="mp" style="margin-bottom:14px">
<div class="card"><h2>3-player: share of wins against two Shrewds</h2><div class="cap">400 games per point; a player as good as Shrewd wins a third</div><svg id="w3" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
<div class="card"><h2>4-player: share of wins against three Shrewds</h2><div class="cap">400 games per point; a player as good as Shrewd wins a quarter</div><svg id="w4" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
</div>
<div class="grid2">
<div class="card"><h2>Policy loss</h2><div class="cap">Cross-entropy against the search's policy targets (visit counts, or completed Q)</div><svg id="lp" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
<div class="card"><h2>Value loss</h2><div class="cap">Cross-entropy against game outcome</div><svg id="lv" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
<div class="card"><h2>Self-play average final score</h2><div class="cap">Per player, per game</div><svg id="sc" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
<div class="card"><h2>Self-play games per hour</h2><div class="cap">Throughput of the self-play phase</div><svg id="gph" viewBox="0 0 500 160" preserveAspectRatio="none" style="height:160px"></svg></div>
</div>
<div class="card" style="margin-top:14px"><details><summary>Table</summary><div class="tbl" id="table"></div></details></div>
<div class="card"><details><summary>Log tail</summary><pre id="log"></pre></details></div>
</div><div class="tip" id="tip" hidden></div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let RUN=null, DATA=null;
function ns(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e}
function fmtPct(x){return (100*x).toFixed(1)+'%'}
function autoFmt(v,range){return range>=40?Math.round(v).toLocaleString():range>=4?v.toFixed(1):range>=.4?v.toFixed(2):v.toFixed(3)}
function chart(svg,pts,opt){
  svg.innerHTML='';
  // draw in real pixels so text is never stretched (phones)
  const W=Math.max(260,svg.clientWidth||1000),H=parseFloat(svg.style.height)||160,L=44,R=12,T=10,B=24;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  if(!pts.length){const t=ns('text',{x:W/2,y:H/2,'text-anchor':'middle'});t.textContent='No data yet';svg.appendChild(t);return}
  const xs=pts.map(p=>p.x),x0=Math.min(...xs),x1=Math.max(...xs,x0+1);
  let y0=opt.y0,y1=opt.y1;
  if(y0==null||y1==null){const ys=pts.map(p=>p.y);const lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo)*.12||Math.abs(hi)*.1||1;y0=y0??lo-pad;y1=y1??hi+pad}
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  const g=css('--grid');
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:g,'stroke-width':1,'vector-effect':'non-scaling-stroke'}));
    const t=ns('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'});t.textContent=opt.fmt?opt.fmt(v):autoFmt(v,y1-y0);svg.appendChild(t)}
  const nt=Math.min(8,x1-x0);for(let i=0;i<=nt;i++){const v=Math.round(x0+(x1-x0)*i/nt);const t=ns('text',{x:X(v),y:H-6,'text-anchor':'middle'});t.textContent=v;svg.appendChild(t)}
  for(const r of (opt.refs||[])){svg.appendChild(ns('line',{x1:L,x2:W-R,y1:Y(r.y),y2:Y(r.y),stroke:css('--ref'),'stroke-width':1,'vector-effect':'non-scaling-stroke'}));
    const t=ns('text',r.left?{x:L+4,y:Y(r.y)-4,'text-anchor':'start'}:{x:W-R-4,y:Y(r.y)-4,'text-anchor':'end'});t.textContent=r.label;svg.appendChild(t)}
  const col=css(opt.color||'--series-1');
  if(pts[0].lo!=null){let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.hi));for(let i=pts.length-1;i>=0;i--)d+='L'+X(pts[i].x)+','+Y(pts[i].lo);svg.appendChild(ns('path',{d:d+'Z',fill:col,opacity:.12}))}
  let d='';pts.forEach((p,i)=>d+=(i?'L':'M')+X(p.x)+','+Y(p.y));
  svg.appendChild(ns('path',{d,fill:'none',stroke:col,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round','vector-effect':'non-scaling-stroke'}));
  const last=pts[pts.length-1];
  svg.appendChild(ns('circle',{cx:X(last.x),cy:Y(last.y),r:4.5,fill:col,stroke:css('--surface-1'),'stroke-width':2,'vector-effect':'non-scaling-stroke'}));
  const cross=ns('line',{y1:T,y2:H-B,stroke:css('--ref'),'stroke-width':1,'vector-effect':'non-scaling-stroke',visibility:'hidden'});svg.appendChild(cross);
  const dot=ns('circle',{r:5,fill:col,stroke:css('--surface-1'),'stroke-width':2,visibility:'hidden','vector-effect':'non-scaling-stroke'});svg.appendChild(dot);
  const hit=ns('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'});svg.appendChild(hit);
  const tip=$('tip');
  hit.addEventListener('pointermove',ev=>{const b=svg.getBoundingClientRect();const fx=(ev.clientX-b.left)/b.width*W;let best=pts[0];for(const p of pts)if(Math.abs(X(p.x)-fx)<Math.abs(X(best.x)-fx))best=p;
    cross.setAttribute('x1',X(best.x));cross.setAttribute('x2',X(best.x));cross.setAttribute('visibility','visible');dot.setAttribute('cx',X(best.x));dot.setAttribute('cy',Y(best.y));dot.setAttribute('visibility','visible');
    tip.hidden=false;tip.innerHTML=opt.tip(best);tip.style.left=Math.min(ev.clientX+14,innerWidth-200)+'px';tip.style.top=(ev.clientY-10)+'px'});
  hit.addEventListener('pointerleave',()=>{tip.hidden=true;cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden')});
}
function tile(l,v,n){return `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n||'&nbsp;'}</div></div>`}
function render(){
  const M=DATA.metrics, cfg=DATA.config;
  const ev=M.filter(m=>m.eval_shrewd);
  const win=ev.map(m=>({x:m.iter,y:m.eval_shrewd.win,lo:Math.max(0,m.eval_shrewd.win-1.96*m.eval_shrewd.se),hi:Math.min(1,m.eval_shrewd.win+1.96*m.eval_shrewd.se),m}));
  const tr=M.filter(m=>m.train), sp=M.filter(m=>m.selfplay);
  const games=sp.reduce((s,m)=>s+m.selfplay.games,0);
  const t0=M.length?M[0].time:0, tl=M.length?M[M.length-1].time:0;
  const best=win.reduce((b,p)=>!b||p.y>b.y?p:b,null), last=win[win.length-1];
  const R=(DATA.ratings&&DATA.ratings.ratings)||{};
  const eloPts=Object.keys(R).filter(k=>k.startsWith('iter_')).map(k=>({x:+k.slice(5),y:R[k].elo,lo:R[k].elo-1.96*R[k].se,hi:R[k].elo+1.96*R[k].se,r:R[k]})).sort((a,b)=>a.x-b.x);
  const bestElo=eloPts.reduce((b,p)=>!b||p.y>b.y?p:b,null);
  const P=DATA.pairs||{}, nm=i=>'iter_'+String(i).padStart(4,'0');
  const rec=(a,b)=>{const k=a<b?a+'|'+b:b+'|'+a, p=P[k]; if(!p||!p[1])return null; const s=a<b?p[0]:p[1]-p[0], w=s/p[1]; return {win:w, n:p[1], se:Math.sqrt(Math.max(w*(1-w),1e-6)/p[1])}};
  const prevPts=eloPts.map(p=>({x:p.x,r:rec(nm(p.x),nm(p.x-1))})).filter(p=>p.r).map(p=>({x:p.x,y:p.r.win,lo:Math.max(0,p.r.win-1.96*p.r.se),hi:Math.min(1,p.r.win+1.96*p.r.se),r:p.r}));
  const lastPrev=prevPts[prevPts.length-1];
  const extras=[cfg.value_q?`value target ${Math.round(100*(1-cfg.value_q))}% result / ${Math.round(100*cfg.value_q)}% search`:'',cfg.margin_w?`margin credit ${cfg.margin_w}`:'',cfg.chance_k?`${cfg.chance_k} draws kept per refill`:'',cfg.policy_target==='cq'?`completed-Q policy targets (c_scale ${cfg.cq_c_scale})`:''].filter(Boolean).join(' · ');
  $('sub').textContent=`Run ${RUN} · ${cfg.players?cfg.players.join('/')+'-player':''} · self-play ${cfg.sims}/${cfg.sims_fast} sims · eval ${cfg.eval_sims} sims${extras?' · '+extras:''}${DATA.stopping?' · stopping after this iteration':''}`;
  $('tiles').innerHTML=tile('Iteration',M.length?M[M.length-1].iter:'–',cfg.iters?`of ${cfg.iters}`:'')+
    tile('Latest vs Shrewd',last?fmtPct(last.y):'–',last?`±${(196*last.m.eval_shrewd.se).toFixed(1)} pts · margin ${last.m.eval_shrewd.margin>0?'+':''}${last.m.eval_shrewd.margin}`:'')+
    tile('Newest vs previous',lastPrev?fmtPct(lastPrev.y):'–',lastPrev?`iteration ${lastPrev.x} vs ${lastPrev.x-1} · ±${(196*lastPrev.r.se).toFixed(1)} pts · ${lastPrev.r.n} games`:'')+
    tile('Best Elo',bestElo?Math.round(bestElo.y):'–',bestElo?`iteration ${bestElo.x} · ±${Math.round(1.96*bestElo.r.se)} · Shrewd = 1000`:'')+
    tile('Self-play games',games.toLocaleString(),`${((tl-t0)/3600).toFixed(1)} h elapsed`);
  const eloRefs=[{y:1000,label:'Shrewd 1000'}];
  if(R.casual)eloRefs.push({y:R.casual.elo,label:'Casual '+Math.round(R.casual.elo)});
  const ys=eloPts.flatMap(p=>[p.lo,p.hi]).concat(eloRefs.map(r=>r.y));
  chart($('elo'),eloPts,{y0:ys.length?Math.floor((Math.min(...ys)-50)/100)*100:0,y1:ys.length?Math.ceil((Math.max(...ys)+50)/100)*100:2000,refs:eloRefs,
    tip:p=>`Iteration ${p.x}<br>Elo <b>${Math.round(p.y)}</b> ±${Math.round(1.96*p.r.se)}<br>${p.r.games.toLocaleString()} rated games`});
  const refPts=M.filter(m=>m.eval_ref).map(m=>({x:m.iter,y:m.eval_ref.win,lo:Math.max(0,m.eval_ref.win-1.96*m.eval_ref.se),hi:Math.min(1,m.eval_ref.win+1.96*m.eval_ref.se),m:m.eval_ref}));
  $('refcard').hidden=!refPts.length;
  if(refPts.length){
    const rname=refPts[refPts.length-1].m.ref||'';
    $('ref-t').textContent='Against the reference network ('+rname.replace(/^.*runs[\/]/,'').replace(/[\/]models[\/]/,' ').replace('.pt','')+')';
    const rys=refPts.flatMap(p=>[p.lo,p.hi]);
    chart($('refc'),refPts,{y0:Math.max(0,Math.floor(Math.min(.3,...rys)*10)/10),y1:Math.min(1,Math.ceil(Math.max(.7,...rys)*10)/10),fmt:v=>Math.round(v*100)+'%',refs:[{y:.5,label:'Even',left:true}],
      tip:p=>`Iteration ${p.x}<br>Win <b>${fmtPct(p.y)}</b> (±${(196*p.m.se).toFixed(1)})<br>Margin <b>${p.m.margin>0?'+':''}${p.m.margin}</b> · ${p.m.games} games`});
  }
  const pys=prevPts.flatMap(p=>[p.lo,p.hi]);
  chart($('prev'),prevPts,{y0:Math.max(0,Math.floor(Math.min(.2,...pys)*5)/5),y1:1,fmt:v=>Math.round(v*100)+'%',refs:[{y:.5,label:'Even',left:true}],
    tip:p=>`Iteration ${p.x} vs ${p.x-1}<br>Win <b>${fmtPct(p.y)}</b> (±${(196*p.r.se).toFixed(1)})<br>${p.r.n} games`});
  const newest=eloPts.length?eloPts[eloPts.length-1].x:null;
  if(newest!==null){
    const me=nm(newest), opp=Object.keys(P).flatMap(k=>{const [a,b]=k.split('|');return a===me?[b]:b===me?[a]:[]});
    opp.sort((a,b)=>a==='shrewd'?-1:b==='shrewd'?1:b.localeCompare(a));
    const label=o=>o==='shrewd'?'Shrewd (web app AI)':o.startsWith('iter_')?'Iteration '+(+o.slice(5)):o;
    const pred=o=>R[o]&&R[me]?1/(1+Math.pow(10,-(R[me].elo-R[o].elo)/400)):null;
    $('h2h-t').textContent=`Newest checkpoint (iteration ${newest}, Elo ${Math.round(R[me].elo)}): head to head`;
    $('h2h').innerHTML='<table><tr><th>Opponent</th><th>Opponent Elo</th><th>Games</th><th>Newest wins</th><th>±95%</th><th>Elo predicts</th></tr>'+
      opp.map(o=>{const r=rec(me,o), q=pred(o);return `<tr><td>${label(o)}</td><td>${R[o]?Math.round(R[o].elo):''}</td><td>${r.n}</td><td>${fmtPct(r.win)}</td><td>${(196*r.se).toFixed(1)}</td><td>${q==null?'':fmtPct(q)}</td></tr>`}).join('')+'</table>';
  }
  chart($('win'),win,{y0:0,y1:1,fmt:v=>Math.round(v*100)+'%',refs:[{y:.494,label:'Shrewd vs Shrewd 49%'},{y:.292,label:'Casual 29%'}],
    tip:p=>`Iteration ${p.x}<br>Win <b>${fmtPct(p.y)}</b> (±${(196*p.m.eval_shrewd.se).toFixed(1)})<br>Score margin <b>${p.m.eval_shrewd.margin>0?'+':''}${p.m.eval_shrewd.margin}</b>`});
  $('mp').hidden=!(cfg.players||[]).some(n=>n>2);
  for(const n of [3,4]){const k='eval_shrewd_'+n+'p';chart($('w'+n),M.filter(m=>m[k]).map(m=>({x:m.iter,y:m[k].win,lo:Math.max(0,m[k].win-1.96*m[k].se),hi:Math.min(1,m[k].win+1.96*m[k].se),m:m[k]})),{y0:0,y1:1,fmt:v=>Math.round(v*100)+'%',refs:[{y:1/n,label:'Shrewd level '+Math.round(100/n)+'%'}],
    tip:p=>`Iteration ${p.x}<br>Win share <b>${fmtPct(p.y)}</b> (±${(196*p.m.se).toFixed(1)})<br>Margin over best opponent <b>${p.m.margin>0?'+':''}${p.m.margin}</b>`})}
  chart($('lp'),tr.map(m=>({x:m.iter,y:m.train.policy,t:m.selfplay&&m.selfplay.target})),{tip:p=>`Iteration ${p.x}<br><b>${p.y.toFixed(3)}</b>`+(p.t?`<br>${p.t.kind} targets: H/ln n ${p.t.h_norm.toFixed(2)} · max share ${p.t.max_share.toFixed(2)}`:'')});
  chart($('lv'),tr.map(m=>({x:m.iter,y:m.train.value})),{tip:p=>`Iteration ${p.x}<br><b>${p.y.toFixed(3)}</b>`});
  chart($('sc'),sp.map(m=>({x:m.iter,y:m.selfplay.avg_score})),{tip:p=>`Iteration ${p.x}<br><b>${p.y.toFixed(1)}</b> points`});
  chart($('gph'),sp.map(m=>({x:m.iter,y:m.selfplay.games/m.selfplay.seconds*3600})),{y0:0,tip:p=>`Iteration ${p.x}<br><b>${Math.round(p.y).toLocaleString()}</b> games/h`});
  $('table').innerHTML='<table><tr><th>Iter</th><th>Elo</th><th>Win vs previous</th><th>Win vs Shrewd</th><th>Margin</th><th>Policy loss</th><th>Value loss</th><th>Self-play games</th><th>Samples</th><th>Avg score</th><th>Rounds</th><th>Minutes</th></tr>'+
    M.slice().reverse().map(m=>`<tr><td>${m.iter}</td><td>${R['iter_'+String(m.iter).padStart(4,'0')]?Math.round(R['iter_'+String(m.iter).padStart(4,'0')].elo)+' ±'+Math.round(1.96*R['iter_'+String(m.iter).padStart(4,'0')].se):''}</td><td>${(r=>r?fmtPct(r.win):'')(m.iter>0?rec(nm(m.iter),nm(m.iter-1)):null)}</td><td>${m.eval_shrewd?fmtPct(m.eval_shrewd.win):''}</td><td>${m.eval_shrewd?m.eval_shrewd.margin:''}</td><td>${m.train?m.train.policy.toFixed(3):''}</td><td>${m.train?m.train.value.toFixed(3):''}</td><td>${m.selfplay?m.selfplay.games:''}</td><td>${m.selfplay?m.selfplay.samples.toLocaleString():''}</td><td>${m.selfplay?m.selfplay.avg_score.toFixed(1):''}</td><td>${m.selfplay?m.selfplay.rounds.toFixed(2):''}</td><td>${m.seconds?(m.seconds/60).toFixed(1):''}</td></tr>`).join('')+'</table>';
  $('log').textContent=DATA.log.map(l=>l.length>400?l.slice(0,400)+'…':l).join('\n');
}
async function load(){
  const runs=await (await fetch('api/runs')).json();
  const sel=$('runsel');
  if(!RUN&&runs.length)RUN=runs[0].name;
  sel.innerHTML=runs.map(r=>`<option${r.name===RUN?' selected':''}>${r.name}</option>`).join('');
  if(!RUN)return;
  DATA=await (await fetch('api/run/'+encodeURIComponent(RUN))).json();
  render();
}
$('runsel').onchange=e=>{RUN=e.target.value;load()};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>DATA&&render());
let rsz=null;addEventListener('resize',()=>{clearTimeout(rsz);rsz=setTimeout(()=>DATA&&render(),150)});
load();setInterval(load,30000);
</script></body></html>'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5230)
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, threaded=True)
