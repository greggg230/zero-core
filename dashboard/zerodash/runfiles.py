"""Reading a run directory's files - every format of the four projects.

* metrics.jsonl: one JSON row per finished iteration ({"iter", "time", "seconds", "selfplay",
  "train", "eval_<opp>", "elo", ...}); a half-written last line is skipped.
* log.txt: "HH:MM:SS message" lines; supervisor.log: "YYYY-MM-DD HH:MM:SS message" lines.
* progress.json (Glass Road, Caverna): {"iter", "phase", "games", "target", "time"};
  Dominion writes <data_dir>/iter_NNNN/progress_*.json per worker instead.
* run_state.json / supervisor_state.json (runctl: Glass Road, Caverna, Dominion).
* PAUSE: JSON {"time", "by"} (an older or hand-made one may be empty: its mtime is the time).
* elo/ratings.json: {"anchor", "anchor_elo", "updated", "ratings": {name: {elo, se, games}}} (a run
  whose eval_select is not 'visits': elo/<rule>/ratings.json, its rule's pool; `elo_pool`).

Everything returns None / [] for a missing or unreadable file: a run being written must never
break the page. Parsed metrics and ratings are cached by (size, mtime).
"""
import datetime
import glob
import json
import os
import re
import threading
import time

MAX_TAIL = 2_000_000           # bytes of metrics.jsonl parsed (the newest rows)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_text(path, limit=65536):
    try:
        with open(path, 'rb') as f:
            return f.read(limit).decode('utf-8', 'replace')
    except OSError:
        return None


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def tail_lines(path, n=60, max_bytes=65536):
    """The last `n` lines of a text file (complete lines only when the read starts mid-file)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
    except OSError:
        return []
    lines = data.decode('utf-8', 'replace').splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return [ln for ln in lines if ln.strip()][-n:]


def is_run_dir(d):
    return os.path.isdir(d) and any(os.path.isfile(os.path.join(d, f)) for f in ('config.json', 'metrics.jsonl'))


# ------------------------------------------------------------------ cached parses

class _Cache:
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def get(self, path, build):
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_size, st.st_mtime_ns)
        with self._lock:
            hit = self._d.get(path)
            if hit and hit[0] == key:
                return hit[1]
        val = build(path)
        with self._lock:
            self._d[path] = (key, val)
        return val


_metrics_cache = _Cache()
_ratings_cache = _Cache()

KEEP_SELFPLAY = ('games', 'games_per_h', 'games_per_hour', 'leaves_per_s', 'positions_per_s', 'seconds', 'samples')


def _summarize_metrics(path):
    """The parts of metrics.jsonl the overview needs: the newest row, the newest row with each
    evaluation, with self-play stats and with an Elo, and the recent iteration durations."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - MAX_TAIL)
            f.seek(start)
            data = f.read()
    except OSError:
        return None
    rows = []
    lines = data.decode('utf-8', 'replace').splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue                              # being written right now
        if isinstance(r, dict) and isinstance(r.get('iter'), int):
            rows.append(r)
    if not rows:
        return {'rows': 0}
    out = {'rows': len(rows), 'first_iter': rows[0]['iter'],
           'last': {'iter': rows[-1]['iter'], 'time': rows[-1].get('time'), 'seconds': rows[-1].get('seconds')},
           'evals': {}, 'selfplay': None, 'elo': None, 'pending': None,
           'seconds': [r['seconds'] for r in rows[-8:] if isinstance(r.get('seconds'), (int, float))]}
    for r in rows:
        for k, v in r.items():
            if k.startswith('eval_') and isinstance(v, dict) and 'win' in v:
                out['evals'][k] = {'iter': r['iter'], 'time': r.get('time'), **{
                    kk: v.get(kk) for kk in ('win', 'se', 'margin', 'games', 'opp') if kk in v}}
        if isinstance(r.get('selfplay'), dict):
            sp = r['selfplay']
            out['selfplay'] = {'iter': r['iter'], **{k: sp[k] for k in KEEP_SELFPLAY if k in sp}}
        if isinstance(r.get('elo'), dict) and r['elo'].get('elo') is not None:
            out['elo'] = {'iter': r['iter'], 'elo': r['elo']['elo'], 'se': r['elo'].get('se')}
        if isinstance(r.get('eval_pending'), dict):
            out['pending'] = {'iter': r['iter'], **r['eval_pending']}
        elif any(k.startswith('eval_') and k != 'eval_pending' for k in r):
            if out['pending'] and out['pending']['iter'] <= r['iter']:
                out['pending'] = None
    return out


def metrics_summary(run_dir):
    return _metrics_cache.get(os.path.join(run_dir, 'metrics.jsonl'), _summarize_metrics)


def _parse_ratings(path):
    j = read_json(path)
    if not isinstance(j, dict):
        return None
    rr = j.get('ratings') if isinstance(j.get('ratings'), dict) else {}
    iters = []
    for k, v in rr.items():
        m = re.match(r'^iter_(\d+)$', k)
        if m and isinstance(v, dict) and v.get('elo') is not None:
            iters.append((int(m.group(1)), float(v['elo']), v.get('se'), v.get('games')))
    iters.sort()
    return {'anchor': j.get('anchor'), 'anchor_elo': j.get('anchor_elo'), 'updated': j.get('updated'),
            'iters': iters, 'refs': {k: v.get('elo') for k, v in rr.items()
                                     if not k.startswith('iter_') and isinstance(v, dict)}}


def elo_pool(run_dir):
    """The Elo pool directory of a run's evaluation move rule (zero-core's `zero.elo.active_pool`,
    docs/az.md 5.4): elo/ for eval_select 'visits' (and runs without it), elo/<rule>/ otherwise
    ('cq', 'cq-n2', 'cq-v25-s0.5': guard, then a completed-Q transform other than 50 / 1)."""
    c = read_json(os.path.join(run_dir, 'config.json'))
    c = c if isinstance(c, dict) else {}
    rule = c.get('eval_select') or 'visits'
    if rule == 'visits':
        return os.path.join(run_dir, 'elo')
    tag = rule
    if c.get('eval_select_min_visits'):
        tag += f"-n{int(c['eval_select_min_visits'])}"
    cv, cs = float(c.get('cq_c_visit', 50.0)), float(c.get('cq_c_scale', 1.0))
    if (cv, cs) != (50.0, 1.0):
        tag += f'-v{cv:g}-s{cs:g}'
    return os.path.join(run_dir, 'elo', tag)


def ratings(run_dir):
    return _ratings_cache.get(os.path.join(elo_pool(run_dir), 'ratings.json'), _parse_ratings)


# ------------------------------------------------------------------ supervisor.log

SUP_LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$')


def local_epoch(stamp):
    """'YYYY-MM-DD HH:MM:SS' in this machine's local time -> epoch (None if malformed)."""
    try:
        return time.mktime(time.strptime(stamp, '%Y-%m-%d %H:%M:%S'))
    except (ValueError, OverflowError):
        return None


def supervisor_events(lines):
    """supervisor.log lines (all four projects' tools/supervise.py) -> a list of events
    {'t', 'kind', ...}: sup_start(pid) / run_start(pid) / run_exit(code) / done / crash(n,
    wait) / paused / resumed / other."""
    ev = []
    for line in lines:
        m = SUP_LINE.match(line)
        if not m:
            continue
        t, msg = local_epoch(m.group(1)), m.group(2)
        e = {'t': t, 'kind': 'other', 'msg': msg}
        if (x := re.match(r'supervisor (\d+) starting', msg)):
            e.update(kind='sup_start', pid=int(x.group(1)))
        elif (x := re.match(r'started \S+ pid (\d+)', msg)):
            e.update(kind='run_start', pid=int(x.group(1)))
        elif (x := re.match(r'\S+ exited with code (-?\d+)', msg)):
            e.update(kind='run_exit', code=int(x.group(1)))
        elif (x := re.match(r'crash #(\d+) in the last hour; restarting in ([\d.]+)s', msg)):
            e.update(kind='crash', n=int(x.group(1)), wait=float(x.group(2)))
        elif 'supervisor done' in msg or 'not restarting' in msg:
            e.update(kind='done')
        elif msg.startswith('run paused') or msg.startswith('PAUSE file present'):
            e.update(kind='paused')
        elif msg.startswith('PAUSE removed'):
            e.update(kind='resumed')
        ev.append(e)
    return ev


def supervisor_info(events):
    """The newest supervisor's pid and start, its newest run process (None once it exited),
    and how often the run crashed in the hour before the last event."""
    info = {'sup_pid': None, 'sup_t': None, 'run_pid': None, 'run_t': None, 'last': None,
            'crashes_1h': 0, 'last_exit': None}
    for e in events:
        k = e['kind']
        if k == 'sup_start':
            info.update(sup_pid=e['pid'], sup_t=e['t'], run_pid=None, run_t=None)
        elif k == 'run_start':
            info.update(run_pid=e['pid'], run_t=e['t'])
        elif k == 'run_exit':
            info.update(run_pid=None, run_t=None, last_exit=e['code'])
        if k != 'other':
            info['last'] = e
    crashes = [e for e in events if e['kind'] == 'crash']
    if crashes and info['last'] and crashes[-1]['t'] and info['last']['t']:
        info['crashes_1h'] = sum(1 for e in crashes if e['t'] and info['last']['t'] - e['t'] < 3600)
    return info


# ------------------------------------------------------------------ log.txt

LOG_LINE = re.compile(r'^(\d\d):(\d\d):(\d\d) (.*)$')
ITER_EVENT = re.compile(r'^iter (\d+) (selfplay|train|vs|elo|evaluation|random buys|targets|board)\b')


def log_line_time(hh, mm, ss, ref):
    """A log line's clock time -> epoch, on the day of `ref` (the file's mtime) or the day
    before when that would be in the future."""
    if ref is None:
        return None
    d = datetime.datetime.fromtimestamp(ref)
    t = d.replace(hour=int(hh), minute=int(mm), second=int(ss), microsecond=0)
    if t.timestamp() > ref + 60:
        t -= datetime.timedelta(days=1)
    return t.timestamp()


def log_events(lines, ref):
    """log.txt lines -> [{'t', 'msg', 'iter', 'what'}] (iter / what for 'iter N <what>' lines)."""
    out = []
    for line in lines:
        m = LOG_LINE.match(line)
        if not m:
            continue
        msg = m.group(4)
        e = {'t': log_line_time(m.group(1), m.group(2), m.group(3), ref), 'msg': msg, 'iter': None, 'what': None}
        if (x := ITER_EVENT.match(msg)):
            e['iter'], e['what'] = int(x.group(1)), x.group(2)
        elif (x := re.search(r'starting after iter (\d+)', msg)):
            e['iter'], e['what'] = int(x.group(1)), 'start'
        out.append(e)
    return out


def log_flags(events):
    """What the end of log.txt says about how the run process ended (if it did)."""
    tail = events[-6:]
    msgs = [e['msg'] for e in tail]
    last = msgs[-1] if msgs else ''
    return {
        'done': last == 'done' or last.endswith(' done') and last.startswith('run '),
        'stop_found': any('STOP file found' in m for m in msgs[-3:]),
        'aborted': any(m.startswith('run aborted') for m in msgs[-3:]),
        'traceback': any('Traceback' in m or 'Error:' in m[:40] for m in msgs[-2:]),
        'paused': next((m for m in reversed(msgs[-3:]) if m.startswith('PAUSED')), None),
        'last': last,
    }


# ------------------------------------------------------------------ progress

def progress_file(run_dir):
    p = read_json(os.path.join(run_dir, 'progress.json'))
    return p if isinstance(p, dict) else None


def dominion_progress(run_dir, cfg):
    """Self-play progress of the newest data dir (Dominion: data/iter_NNNN/progress_*.json,
    one per worker) -> {'iter', 'games_done', 'games', 'done', 'time', 'mtime'} or None."""
    root = (cfg or {}).get('data_dir') or os.path.join(run_dir, 'data')
    if not os.path.isabs(root):
        root = os.path.join(run_dir, root)
    try:
        dirs = sorted(d for d in os.listdir(root) if re.match(r'^iter_\d+$', d))
    except OSError:
        return None
    if not dirs:
        return None
    files = glob.glob(os.path.join(root, dirs[-1], 'progress_*.json'))
    if not files:
        return None
    tot = {'iter': int(dirs[-1][5:]), 'games_done': 0, 'games': 0, 'done': True, 'time': 0.0, 'mtime': 0.0}
    for f in files:
        j = read_json(f)
        if not isinstance(j, dict):
            continue
        tot['games_done'] += j.get('games_done', 0) or 0
        tot['games'] += j.get('games', 0) or 0
        tot['done'] = tot['done'] and bool(j.get('done'))
        tot['time'] = max(tot['time'], j.get('time') or 0)
        tot['mtime'] = max(tot['mtime'], mtime(f) or 0)
    return tot


def parse_pause(run_dir):
    """<run>/PAUSE -> {'time', 'by', 'raw'} (None without one)."""
    p = os.path.join(run_dir, 'PAUSE')
    raw = read_text(p, 4096)
    if raw is None:
        return None
    return pause_from_text(raw, mtime(p))


def pause_from_text(raw, mt=None):
    try:
        j = json.loads(raw) if raw.strip() else {}
    except ValueError:
        j = {}
    if not isinstance(j, dict):
        j = {}
    t = j.get('time')
    return {'time': float(t) if isinstance(t, (int, float)) else mt, 'by': str(j.get('by') or ''), 'raw': raw}
