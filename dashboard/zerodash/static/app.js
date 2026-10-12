'use strict';
// zerodash page: renders /api/state every refresh_s seconds; Pause / Resume / Pause all /
// Resume all go through a confirm sheet and POST {"confirm": true}.
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
let DATA = null, FETCHED = 0, TIMER = null, BUSY = false, FAILS = 0;
let SHOW_ALL = false;
const OPEN = new Set();          // ended runs whose full card was opened (tap the row)
const ENDED = new Set(['stopped', 'crashed']);
try { SHOW_ALL = localStorage.getItem('zerodash.showAll') === '1'; } catch (e) { /* private mode */ }

// ---------------------------------------------------------------- formatting
const nowS = () => DATA ? DATA.now + (Date.now() - FETCHED) / 1000 : Date.now() / 1000;
function ago(t) {
  if (t == null) return '?';
  const s = Math.max(0, nowS() - t);
  if (s < 90) return Math.round(s) + ' s';
  if (s < 5400) return Math.round(s / 60) + ' min';
  if (s < 172800) return (s / 3600).toFixed(1) + ' h';
  return (s / 86400).toFixed(1) + ' days';
}
const clock = t => { const d = new Date(t * 1000), n = new Date(nowS() * 1000);
  const hm = d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
  return d.toDateString() === n.toDateString() ? hm : d.toLocaleDateString([], {month: 'short', day: 'numeric'}) + ' ' + hm; };
function compact(v) {
  if (v == null || !isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
  if (a >= 1e4) return Math.round(v / 1e3) + 'k';
  if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  return Math.round(v).toLocaleString();
}
const pct = x => (100 * x).toFixed(1) + '%';
const signed = x => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toLocaleString(undefined, {maximumFractionDigits: 2});
const PHASE = {selfplay: 'self-play', train: 'training', eval: 'evaluation'};

const ICON = {
  running: '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>',
  paused: '<svg viewBox="0 0 16 16"><rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/></svg>',
  stopped: '<svg viewBox="0 0 16 16"><rect x="3" y="3" width="10" height="10" rx="1.5"/></svg>',
  crashed: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5"/><path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" stroke="var(--surface-1)" stroke-width="1.8" fill="none" stroke-linecap="round"/></svg>',
  warn: '<svg viewBox="0 0 16 16"><path d="M8 1.5l7 12.5H1z"/><path d="M8 6v4" stroke="var(--surface-1)" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="12" r="1" fill="var(--surface-1)" stroke="none"/></svg>',
  busy: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5"/></svg>',
  unknown: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5"/><path d="M6.2 6.3a1.9 1.9 0 1 1 2.6 1.7c-.6.3-.8.6-.8 1.3" stroke="var(--surface-1)" stroke-width="1.6" fill="none" stroke-linecap="round"/><circle cx="8" cy="11.6" r=".9" fill="var(--surface-1)" stroke="none"/></svg>',
};
const STATE_ICON = {running: 'running', paused: 'paused', pausing: 'paused', stopped: 'stopped', crashed: 'crashed',
  stale: 'warn', restarting: 'warn', resuming: 'busy', starting: 'busy', stopping: 'busy', unknown: 'unknown'};
const PULSE = new Set(['pausing', 'resuming', 'starting', 'stopping', 'restarting']);
function pill(r) {
  return `<span class="pill ${r.severity}${PULSE.has(r.state) ? ' pulse' : ''}" title="${esc(r.detail || r.label)}">${ICON[STATE_ICON[r.state] || 'unknown']}${esc(r.label)}</span>`;
}

// ---------------------------------------------------------------- machine strip
function mtile(label, value, note, opts = {}) {
  const meter = opts.frac != null ? `<div class="meter"><i class="${opts.warn ? 'warn' : ''}" style="width:${Math.max(0, Math.min(100, 100 * opts.frac)).toFixed(1)}%"></i></div>` : '';
  return `<div class="mtile${opts.warn ? ' warn' : ''}"><div class="l"><span>${esc(label)}</span>${opts.right ? `<span>${opts.right}</span>` : ''}</div>` +
    `<div class="v">${value}</div>${meter}<div class="n${opts.err ? ' err' : ''}">${note || '&nbsp;'}</div></div>`;
}
function renderMachine(m) {
  const el = $('machine');
  if (!m) { el.hidden = true; return; }
  el.hidden = false;
  const out = [];
  const g = m.gpu;
  if (g && g.gpus) {
    g.gpus.forEach((x, i) => {
      const used = x.mem_used_mb / 1024, tot = x.mem_total_mb / 1024;
      out.push(mtile('GPU' + (g.gpus.length > 1 ? ' ' + i : ''),
        `${x.util_pct != null ? Math.round(x.util_pct) + '%' : '–'} <small>${x.power_w != null ? Math.round(x.power_w) + ' W' : ''}</small>`,
        `mem ${used.toFixed(1)} / ${tot.toFixed(0)} GB`, {frac: used / tot, warn: used / tot > 0.92}));
    });
  } else out.push(mtile('GPU', '–', esc(g ? g.error : 'not sampled yet'), {err: !!g}));
  const c = m.cpu;
  if (c && !c.error) out.push(mtile('CPU', Math.round(c.percent) + '%', `RAM ${Math.round(c.mem_used_gb)} / ${Math.round(c.mem_total_gb)} GB`,
    {frac: c.percent / 100, warn: c.mem_percent > 92}));
  else out.push(mtile('CPU', '–', esc(c ? c.error : 'not sampled yet'), {err: !!c}));
  for (const d of m.disks || []) {
    if (d.error) { out.push(mtile(d.label + ' free', '–', esc(d.error), {err: true})); continue; }
    out.push(mtile(d.label + ' free', `${d.free_gb < 100 ? d.free_gb.toFixed(1) : Math.round(d.free_gb)} <small>GB</small>`,
      d.warn ? `under ${d.warn_below_gb} GB!` : `of ${Math.round(d.total_gb)} GB`, {frac: 1 - d.free_gb / d.total_gb, warn: d.warn}));
  }
  const mac = m.mac;
  if (mac) {
    if (mac.load) out.push(mtile(mac.label + ' load 1/5/15', `${mac.load[0].toFixed(2)} <small>${mac.load[1].toFixed(1)} ${mac.load[2].toFixed(1)}</small>`, `${ago(mac.time)} ago`));
    else out.push(mtile(mac.label, 'offline', esc((mac.last_load ? `load ${mac.last_load[0].toFixed(2)} ${ago(mac.last_time)} ago · ` : '') + (mac.error || '')), {err: true}));
  }
  el.innerHTML = out.join('');
}

// ---------------------------------------------------------------- sparkline
function spark(series, run) {
  if (!series || series.length < 2) return '';
  return `<svg class="spark" data-key="${esc(run.key)}" role="img" aria-label="Elo from ${Math.round(series[0][1])} (iteration ${series[0][0]}) to ${Math.round(series[series.length - 1][1])} (iteration ${series[series.length - 1][0]})"></svg>`;
}
function drawSparks() {
  const byKey = {};
  for (const p of DATA.projects) for (const r of p.runs) byKey[r.key] = r;
  document.querySelectorAll('svg.spark').forEach(svg => {
    const r = byKey[svg.dataset.key]; if (!r || !r.elo) return;
    const pts = r.elo.series, W = Math.max(60, svg.clientWidth || 100), H = 26, P = 4;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
    // at least 150 Elo tall: refit noise of a flat run must not look like a trend
    const mid = (Math.min(...ys) + Math.max(...ys)) / 2, half = Math.max(75, (Math.max(...ys) - Math.min(...ys)) / 2);
    const y0 = mid - half, y1 = mid + half;
    const X = x => P + (x - x0) / (x1 - x0) * (W - 2 * P), Y = y => H - 3 - (y - y0) / (y1 - y0) * (H - 6);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join('');
    const last = pts[pts.length - 1];
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = `<path d="${d}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
      `<line class="x" y1="0" y2="${H}" stroke="var(--neutral)" stroke-width="1" visibility="hidden"/>` +
      `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="3.5" fill="var(--series-1)" stroke="var(--surface-2)" stroke-width="2"/>`;
    const tip = $('tip'), cross = svg.querySelector('line.x');
    const move = ev => {
      const b = svg.getBoundingClientRect(), fx = (ev.clientX - b.left) / b.width * W;
      let best = pts[0]; for (const p of pts) if (Math.abs(X(p[0]) - fx) < Math.abs(X(best[0]) - fx)) best = p;
      cross.setAttribute('x1', X(best[0])); cross.setAttribute('x2', X(best[0])); cross.setAttribute('visibility', 'visible');
      tip.hidden = false; tip.innerHTML = `Iteration ${best[0]} · Elo <b>${Math.round(best[1])}</b>`;
      tip.style.left = Math.min(ev.clientX + 12, innerWidth - 170) + 'px'; tip.style.top = (ev.clientY - 34) + 'px';
    };
    svg.onpointermove = move; svg.onpointerdown = move;
    svg.onpointerleave = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); };
  });
}

// ---------------------------------------------------------------- runs
function stat(label, value, note, extra = '') {
  return `<div class="stat"><div class="l">${label}</div><div class="v">${value}</div>${extra}<div class="n">${note || '&nbsp;'}</div></div>`;
}
function endedHTML(p, r) {
  const it = r.iteration || {};
  const when = it.last != null ? `it ${it.last} · ${ago(it.last_time)} ago` : (r.modified ? ago(r.modified) + ' ago' : '');
  const why = r.state === 'crashed' ? `<div class="line detail warnline">${esc(r.detail)}</div>` : '';
  const elo = r.elo ? ` · Elo ${Math.round(r.elo.elo).toLocaleString()}` : '';
  return `<div class="run ended${r.hidden ? ' hidden-run' : ''}"><button type="button" class="rrow" data-open="${esc(r.key)}" aria-expanded="false">` +
    `<span class="rname">${esc(r.name)}</span>${pill(r)}<span class="spacer"></span><span class="when">${when}${elo}</span><span class="chev" aria-hidden="true">›</span></button>${why}</div>`;
}
function runHTML(p, r) {
  if (ENDED.has(r.state) && !OPEN.has(r.key) && !r.can_resume) return endedHTML(p, r);
  const it = r.iteration || {};
  const lines = [];
  const live = ['running', 'stale', 'stopping', 'pausing', 'starting'].includes(r.state);
  if (live && it.current != null) {
    let s = `Iteration <b>${it.current}</b>`;
    if (it.phase) s += ` · ${PHASE[it.phase] || esc(it.phase)}`;
    if (it.phase_detail) s += ` <b>${esc(it.phase_detail)}</b>`;
    if (it.phase_since) s += ` · for ${ago(it.phase_since)}`;
    lines.push(`<div class="line">${s}</div>`);
    if (it.phase_frac != null) lines.push(`<div class="bar" role="progressbar" aria-valuenow="${Math.round(100 * it.phase_frac)}" aria-valuemin="0" aria-valuemax="100"><i style="width:${(100 * it.phase_frac).toFixed(1)}%"></i></div>`);
  }
  if (r.pause) {
    const who = r.pause.kind === 'pause-all' ? 'Pause all' : esc(r.pause.by || 'unknown');
    const m = !live && it.paused_in ? /^PAUSED \(([^):]*)/.exec(it.paused_in) : null;
    const where = m ? ` (${esc(m[1])})` : '';
    lines.push(`<div class="line">Paused by <b>${who}</b> · ${r.pause.time ? `<b>${ago(r.pause.time)}</b> (since ${clock(r.pause.time)})` : '?'}${!live && it.current != null ? ` · in iteration ${it.current}${where}` : ''}</div>`);
  }
  if (it.last != null) lines.push(`<div class="line">Iteration ${it.last} finished <b>${ago(it.last_time)} ago</b>${it.last_seconds ? ` · took ${Math.round(it.last_seconds / 60)} min` : ''}</div>`);
  else lines.push('<div class="line">No finished iteration yet</div>');
  if (r.detail && !['running'].includes(r.state)) lines.push(`<div class="line detail${['crashed', 'stale', 'restarting'].includes(r.state) ? ' warnline' : ''}">${esc(r.detail)}</div>`);
  if (r.remote) {
    const rm = r.remote, mirror = rm.kind === 'mirror';
    const path = String(rm.path || '?').replace(/^\/(Users|home)\/[^/]+\//, '~/');
    lines.push(`<div class="line detail">Runs on the ${esc(rm.label || 'remote host')} in ${esc(path)} · ${mirror ? 'mirrored by mac_sync' : 'over SSH'}` +
      (rm.error ? ` · <span class="warnline">${esc(rm.error)}</span>` : rm.probe_age_s != null ? ` · ${mirror ? 'synced' : 'checked'} ${ago(nowS() - rm.probe_age_s)} ago` : ' · not checked yet') + '</div>');
  }
  // stats
  const e = r.eval, el = r.elo, tp = r.throughput;
  let ev = stat('vs yardstick', '<span class="na">–</span>', '');
  if (e && e.win != null) {
    const pend = e.pending ? ` · it ${e.pending.iter} pending` : '';
    ev = stat('vs ' + esc(e.label), pct(e.win), `±${(196 * (e.se || 0)).toFixed(1)}${e.margin != null ? ' · margin ' + signed(Math.round(e.margin * 10) / 10) : ''} · it ${e.iter}${pend}`);
  } else if (e && e.pending) ev = stat('vs yardstick', '<span class="na">pending</span>', `it ${e.pending.iter} on ${esc(e.pending.host || 'remote')}`);
  const eloS = el ? stat('Elo', Math.round(el.elo).toLocaleString(), `${el.se != null ? '±' + Math.round(1.96 * el.se) + ' · ' : ''}it ${el.iter}${el.best && el.best.iter !== el.iter ? ' · best ' + Math.round(el.best.elo) : ''}`, spark(el.series, r))
    : stat('Elo', '<span class="na">–</span>', '');
  const tpS = tp ? stat('Self-play', `${compact(tp.games_per_h)} <small class="muted" style="font-size:12px;font-weight:500">g/h</small>`, `${tp.leaves_per_s ? compact(tp.leaves_per_s) + ' leaves/s · ' : ''}it ${tp.iter}`)
    : stat('Self-play', '<span class="na">–</span>', '');
  let btn = '';
  if (r.can_resume) btn = `<button type="button" class="btn small resume ctl" data-act="resume" data-p="${esc(p.id)}" data-r="${esc(r.id)}">Resume</button>`;
  else if (r.can_pause) btn = `<button type="button" class="btn small pause ctl" data-act="pause" data-p="${esc(p.id)}" data-r="${esc(r.id)}">Pause</button>`;
  const badges = (r.remote ? `<span class="chip" title="${esc(r.remote.host || '')}">${esc(r.remote.label || 'remote')}</span>` : '') +
    (!r.pause_supported && !p.pause_note ? `<span class="chip" title="${esc(p.run_pause_note || 'no pause (STOP only)')}">STOP only</span>` : '') +
    (r.hidden ? '<span class="chip">old</span>' : '');
  const fold = OPEN.has(r.key) ? `<button type="button" class="fold" data-open="${esc(r.key)}" aria-expanded="true" title="Fold">‹</button>` : '';
  return `<div class="run${r.hidden ? ' hidden-run' : ''}"><div class="rhead"><span class="rname" title="${esc(r.name)}">${esc(r.name)}</span>${badges}${pill(r)}<span class="spacer"></span>${btn}${fold}</div>` +
    lines.join('') + `<div class="stats">${ev}${eloS}${tpS}</div></div>`;
}
function projectHTML(p) {
  const runs = p.runs.filter(r => SHOW_ALL || !r.hidden);
  const head = `<div class="phead"><div class="pmeta"><h2>${esc(p.name)}</h2>` +
    (p.pause_note ? `<span class="chip" title="Its runs honour only STOP files: no Pause button, Pause all skips them">${esc(p.pause_note)}</span>` : '') +
    `</div>${p.dashboard ? `<a class="open" href="${esc(p.dashboard)}" target="_blank" rel="noopener">Dashboard ↗</a>` : ''}</div>`;
  const errs = (p.errors || []).map(e => `<div class="perr">${esc(e)}</div>`).join('');
  let body = runs.map(r => runHTML(p, r)).join('');
  if (!runs.length) body = `<div class="empty">${p.runs.length ? 'No active runs' : 'No runs found'}${p.hidden_count && !SHOW_ALL ? ` · ${p.hidden_count} older run${p.hidden_count > 1 ? 's' : ''} hidden` : ''}.</div>`;
  else if (!SHOW_ALL && p.hidden_count) body += `<div class="empty small">${p.hidden_count} older run${p.hidden_count > 1 ? 's' : ''} hidden (Show all runs).</div>`;
  return `<section class="project" aria-label="${esc(p.name)}">${head}${errs}${body}</section>`;
}
function render() {
  if (!DATA) return;
  document.title = DATA.title || 'Zero dashboard';
  $('title').textContent = DATA.title || 'Zero dashboard';
  renderMachine(DATA.machine);
  const all = DATA.projects.flatMap(p => p.runs);
  const hidden = all.filter(r => r.hidden).length;
  $('hidden-count').textContent = hidden ? `(${hidden} older hidden)` : '';
  $('show-all').checked = SHOW_ALL;
  $('projects').innerHTML = DATA.projects.map(projectHTML).join('');
  const pausable = all.filter(r => r.can_pause), paRuns = DATA.pause_all.runs;
  $('pause-all').disabled = BUSY || !pausable.length;
  $('resume-all').disabled = BUSY || !paRuns.length;
  $('resume-all').textContent = paRuns.length ? `Resume all (${paRuns.length})` : 'Resume all';
  $('pa-note').textContent = paRuns.length ? `${paRuns.length} run${paRuns.length > 1 ? 's' : ''} paused by Pause all` : '';
  drawSparks();
  tick();
}
function tick() {
  if (!DATA) return;
  const all = DATA.projects.flatMap(p => p.runs).filter(r => !r.hidden);
  const live = all.filter(r => ['running', 'stale', 'stopping', 'pausing', 'starting', 'restarting', 'resuming'].includes(r.state)).length;
  const paused = all.filter(r => r.state === 'paused').length;
  const age = Math.round((Date.now() - FETCHED) / 1000);
  $('sub').innerHTML = `${live} running · ${paused} paused · updated ${age < 2 ? 'just now' : age + ' s ago'}` +
    (FAILS ? ` · <span class="offline">offline (${FAILS} failed refresh${FAILS > 1 ? 'es' : ''})</span>` : '');
}

// ---------------------------------------------------------------- data + controls
async function load() {
  try {
    const r = await fetch('api/state', {cache: 'no-store'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    DATA = await r.json(); FETCHED = Date.now(); FAILS = 0;
    render();
  } catch (e) { FAILS++; tick(); }
}
function schedule() {
  clearInterval(TIMER);
  TIMER = setInterval(() => { if (!document.hidden) load(); }, 1000 * ((DATA && DATA.refresh_s) || 15));
}
let toastT = null;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 5000); }

function confirmSheet(title, html, okLabel, okClass) {
  return new Promise(resolve => {
    $('confirm-title').textContent = title; $('confirm-body').innerHTML = html;
    const ok = $('confirm-ok'); ok.textContent = okLabel; ok.className = 'btn ' + okClass;
    $('confirm').hidden = false; ok.focus();
    const done = v => { $('confirm').hidden = true; ok.onclick = $('confirm-cancel').onclick = $('confirm').onclick = document.onkeydown = null; resolve(v); };
    ok.onclick = () => done(true); $('confirm-cancel').onclick = () => done(false);
    $('confirm').onclick = ev => { if (ev.target === $('confirm')) done(false); };
    document.onkeydown = ev => { if (ev.key === 'Escape') done(false); };
  });
}
async function post(url) {
  BUSY = true; render();
  try {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({confirm: true})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.ok === false) toast((j.error || (j.errors && j.errors.map(e => e.key + ': ' + e.error).join('; ')) || 'HTTP ' + r.status));
    return j;
  } catch (e) { toast('Request failed: ' + e); return null; } finally { BUSY = false; await load(); setTimeout(load, 2500); }
}
const listRuns = keys => `<ul>${keys.map(k => `<li>${esc(k)}</li>`).join('')}</ul>`;

document.addEventListener('click', ev => {
  const t = ev.target.closest('[data-open]'); if (!t) return;
  const k = t.dataset.open; OPEN.has(k) ? OPEN.delete(k) : OPEN.add(k); render();
});
document.addEventListener('click', async ev => {
  const b = ev.target.closest('button.ctl'); if (!b || !DATA) return;
  const p = DATA.projects.find(x => x.id === b.dataset.p), r = p && p.runs.find(x => x.id === b.dataset.r); if (!r) return;
  const pause = b.dataset.act === 'pause';
  const ok = pause
    ? await confirmSheet(`Pause ${r.name}?`, `<p>${esc(p.name)}: writes <code>PAUSE</code>. The run stops within seconds at its next pause point (games in progress are dropped); Resume continues from its newest checkpoint.</p>`, 'Pause', 'pause')
    : await confirmSheet(`Resume ${r.name}?`, `<p>${esc(p.name)}: removes <code>PAUSE</code>${r.pause && r.pause.by ? ` (written by <b>${esc(r.pause.by)}</b>)` : ''}; its supervisor relaunches the run.</p>`, 'Resume', 'resume');
  if (!ok) return;
  const j = await post(`api/run/${encodeURIComponent(p.id)}/${encodeURIComponent(r.id)}/${pause ? 'pause' : 'resume'}`);
  if (j && j.ok) toast(`${r.name}: ${pause ? (j.changed ? 'PAUSE written' : 'already paused') : (j.changed ? 'PAUSE removed' : 'was not paused')}`);
});
$('pause-all').onclick = async () => {
  const all = DATA.projects.flatMap(p => p.runs.map(r => ({p, r})));
  const go = all.filter(x => x.r.can_pause).map(x => x.r.key);
  const skip = all.filter(x => !x.r.pause_supported && ['running', 'stale', 'starting', 'restarting', 'resuming', 'stopping'].includes(x.r.state)).map(x => x.r.key);
  const ok = await confirmSheet(`Pause ${go.length} run${go.length === 1 ? '' : 's'}?`,
    `<p>Writes <code>PAUSE</code> (by “zerodash pause-all …”) into:</p>${listRuns(go)}` +
    (skip.length ? `<p>Not paused (STOP only): ${esc(skip.join(', '))}</p>` : '') +
    '<p>Resume all later removes exactly these PAUSE files; runs paused any other way stay paused.</p>', 'Pause all', 'pause');
  if (!ok) return;
  const j = await post('api/pause-all');
  if (j && j.ok) toast(`Paused ${j.paused.length} run${j.paused.length === 1 ? '' : 's'}` + (j.skipped.length ? ` · ${j.skipped.length} skipped (STOP only)` : ''));
};
$('resume-all').onclick = async () => {
  const keys = DATA.pause_all.runs;
  const others = DATA.projects.flatMap(p => p.runs).filter(r => r.pause && r.pause.kind !== 'pause-all').map(r => r.key);
  const ok = await confirmSheet(`Resume ${keys.length} run${keys.length === 1 ? '' : 's'}?`,
    `<p>Removes the PAUSE files Pause all wrote:</p>${listRuns(keys)}` +
    (others.length ? `<p>Stay paused (paused another way): ${esc(others.join(', '))}</p>` : ''), 'Resume all', 'resume');
  if (!ok) return;
  const j = await post('api/resume-all');
  if (j && j.ok) toast(`Resumed ${j.resumed.length} run${j.resumed.length === 1 ? '' : 's'}` + (j.kept.length ? ` · ${j.kept.length} left paused` : ''));
};
$('show-all').onchange = ev => { SHOW_ALL = ev.target.checked; try { localStorage.setItem('zerodash.showAll', SHOW_ALL ? '1' : '0'); } catch (e) { /* */ } render(); };
document.addEventListener('visibilitychange', () => { if (!document.hidden) load(); });
let rsz = null; addEventListener('resize', () => { clearTimeout(rsz); rsz = setTimeout(() => DATA && drawSparks(), 150); });
setInterval(tick, 1000);
load().then(schedule);
