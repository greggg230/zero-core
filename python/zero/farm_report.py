"""python -m zero.farm_report: what the owner's own machines contributed, for zero-farm's admin page
(docs/az.md 44.3).

zero-farm's coordinator counts the work its agents do. The owner's own machines - this desktop playing
its runs, the MacBook, rented Vast boxes, all driven through zero-core's SSH remotes (zero.remote,
zero.remoteplay), never the farm agent - are invisible to it. This reads what the runs on this machine
recorded and what the Mac's fuzz farm logged, and POSTs one report to `<url>/v1/admin/external`:

    {"reporter": "desktop", "t": <unix s>, "sources": [{"id", "name", "kind": "local" | "ssh" | "vast",
      "hardware", "online", "working_on": ["run:<id>", ...], "selfplay_games": {"1h", "24h"},
      "eval_games": {"1h", "24h"}, "fuzz_seeds": {"1h", "24h"}, "busy_h_24h"[, "error"][, "fuzz_state"]}]}

Read only: nothing under a run directory is written, the Mac is only read (one ssh command).

**Runs.** Every directory directly under a run root (the config's `roots`, `--roots`, or the default:
~/<game>/runs and E:/<game>/runs of mtg-zero and the other zero games, those that exist) whose
metrics.jsonl, evals.jsonl, run_state.json, progress.json or supervisor_state.json (or resolver/'s
metrics / evals, a co-training run) changed in the last 24 h. A directory is taken once, by its real
path (~/mtg-zero/runs/<run> are junctions to E:/mtg-zero/runs/<run>). A run's id is its directory's
name; when two roots hold different runs of one name, the first root in the configured order keeps
the bare name and a later one gets `<name>@<the root's parent directory's name>` (`base1@dominion-zero`),
or `<name>@<the root's path>` when that is taken too.

**Time.** Every metrics.jsonl row (one per iteration: zero.run.Run.finish_iter, co-training's too)
holds `time`, the unix time it was appended - after the iteration's self-play, training and local
evaluation - with `seconds` (the iteration's wall time) and `phases` ({selfplay_s, train_s, eval_s,
elo_s}); every evals.jsonl row (zero.remote.RemoteEval) holds `time` (appended here when the remote's
result came back), `host` (the remote's name) and `seconds`. A row without `time` (none since the core
extraction) is given the next row's `time` minus the next row's `seconds` (the last row: the file's
mtime). The files are read from their ends, back to 26 h ago. A forked run (tools/fork_run.py) copies
its source's rows: a row is counted once, by (iteration, `time`), for the run whose directory is the
oldest (its creation time; a fork's directory is created when it forks), i.e. the run that played it.

**Self-play games** (per metrics row): this machine's `selfplay.games` and each remote's
`selfplay.remotes.<name>.games` (a list of hosts, docs/az.md 34) or `selfplay.remote.games` under
`selfplay.remote.host` (one host) - the games the iteration's shared quota accepted from each, so a
game counts once, in the iteration that took it (a lingering game in the next one). They count at the
end of the iteration's self-play phase, `time - seconds + phases.selfplay_s`. A co-training run's
two rows (the primary's and resolver/'s) both count.

**Eval games** (per evaluation): the `games` of every top-level `eval_*` record (eval_yard,
eval_random, eval_yard_net) and of every `extras.<name>` that records `games`, plus the checkpoint's
Elo matches from the run's Elo pools (elo/results.json, elo/<rule>/results.json: the pairs
`iter_k|iter_j`, j < k, and `iter_k|anchor:<label>`; the pools' yardstick and 'random' pairs are left
to the eval_* records, which count those games). Local (eval_remote off): the metrics row that holds `eval_yard`,
this machine's; remote: the evals.jsonl row, its `host`'s. They count at the row's `time`. Not counted:
an extra yardstick that records no `games` (MTG Zero's real_decks records its 2 x n games as `n`),
anchor calibration and the one-off reference match.

**Sources.** A remote's recorded name maps to a source through the config's `aliases` (an old row's
host 'user@host' -> 'mac'); `local` and `gpu:N` (`local_names`: this machine's other GPU) are this
machine; `vast` and `vast:<box>` (`vast_names`) are Vast boxes, described by mtg-zero tools/vast.py's
ledger (~/.config/vastai/mtgz_boxes.json, read only: cpu, threads, ram_gb, state, deadline; it records
no GPU: `vast_gpu`, the GPU vast.py rents, RTX 5090); any other name is an ssh machine (`sources` gives
its name and hardware). A run that lives on a box (vast.py run-push / run-start: the run directory's
ON_VAST.json, renamed ON_VAST_<home time>.json by run-home, {box, since}) plays its "local" games on
that box from `since` to its homing. Listed: this machine, the config's `sources`, the ledger's boxes
that are running, and any source with work in the last 24 h.

**busy_h_24h**: the hours of the last 24 h covered by the union of the source's work intervals -
this machine (or a box running the run): every iteration, `[time - seconds, time]` (self-play,
training, local evaluation); a remote's self-play: `[end of self-play - running_s, end of self-play]`
(its running time in the phase, back-offs excluded) and its linger after the phase, `[end, end +
linger.seconds]`; a remote evaluation: its evals row's `[time - seconds, time]` (shipping the
checkpoints included); the fuzz farm: each chunk's running seconds, ending at its log line. A union:
two runs at once, or self-play while the fuzz farm runs, count once, and it is at most 24.

**The fuzz farm** (mtg-zero tools/fuzz_farm.py, LaunchAgent com.mtgzero.fuzzfarm on the Mac): one
`ssh -o BatchMode=yes -o ConnectTimeout=10 <host> '"<python>" - "<farm>" 3600 86400'` whose stdin is
FUZZ_SCRIPT, a read-only parser of `<farm>/logs/farm.log`'s chunk lines (`<Mac local time> <mode>
[<first>+<n>][ (bisect)]: rc <rc>, <done> <unit> in <s> s`, one per finished chunk) of the last 24 h
and of status.json's state. A seed is one of a chunk's `done` units (a fuzzed game, match or draft
event, or a smoke seed); the seeds per window are the chunks' sums. (tools/fuzz_farm_status.sh prints
totals, not windows, and writes a mark file here.) The Mac unreachable: its source is `online: false`
with the `error` and `fuzz_seeds` null; the report still goes out.

**POST** with `Authorization: Bearer <admin token>` (env ZEROFARM_ADMIN_TOKEN, else
~/.config/zero-farm/admin_token; never printed or logged); a 5xx or a network error is retried after
2, 5 and 10 s, a 4xx is not. Exit 0 posted, 1 not, 2 a usage, config or token error. Each run appends
one line to ~/.config/zero-farm/farm_report.log (`--log`; '' = none; never with --dry-run).

    python -m zero.farm_report [--config F] [--roots DIR ...] [--url U] [--no-fuzz] [--dry-run] [-v]
    python -m zero.farm_report --install-task [--task-root DIR]    # every 5 min, no window
    python -m zero.farm_report --uninstall-task

The config (~/.config/zero-farm/farm_report.json, optional; `--config`) overrides DEFAULTS' keys (its
dicts key by key; `"fuzz": null` turns the Mac probe off). `--install-task` registers the Task
Scheduler task TASK_NAME with PowerShell's Register-ScheduledTask: the current user, logged on
(interactive, never SYSTEM: the Store Python behind the venvs needs the user's profile), every 5 min,
`pythonw.exe -m zero.farm_report` (no console window; its ssh gets CREATE_NO_WINDOW) in the working
directory `--task-root` (default: the directory holding this `zero` package), one instance at a time,
4 min at most; the run's --config / --roots / --url / --no-fuzz / --log are passed on.

Stdlib only (no torch, no numpy), so the task imports nothing heavy.
"""
import argparse
import fnmatch
import glob
import http.client
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = '/v1/admin/external'
DEFAULT_URL = 'https://farm.greggg230.com'
TOKEN_ENV = 'ZEROFARM_ADMIN_TOKEN'
CONFIG_DIR = os.path.join('~', '.config', 'zero-farm')
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, 'farm_report.json')
TOKEN_FILE = os.path.join(CONFIG_DIR, 'admin_token')
DEFAULT_LOG = os.path.join(CONFIG_DIR, 'farm_report.log')
LOG_MAX = 512 * 1024
TASK_NAME = 'ZeroFarmReport'
TASK_EVERY_MIN = 5
TASK_LIMIT_MIN = 4
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # holds the `zero` package
WINDOWS = (('1h', 3600), ('24h', 86400))
DAY = 86400
SLACK_S = 2 * 3600                 # rows read back to 26 h ago (a remote's linger outlasts its row)
FUTURE_S = 600                     # another machine's clock a little ahead
RETRY_S = (2, 5, 10)
USER_AGENT = 'zero-farm-report/1'
RESULT = 'FARM_REPORT '
RUN_FILES = ('metrics.jsonl', 'evals.jsonl', 'run_state.json', 'progress.json', 'supervisor_state.json',
             os.path.join('resolver', 'metrics.jsonl'), os.path.join('resolver', 'evals.jsonl'))
PARTS = ('', 'resolver')           # a co-training run's resolver keeps its own metrics, evals and Elo
GAMES = ('mtg-zero', 'caverna-zero', 'glassroad-zero', 'kingdombuilder-zero', 'dominion-zero', 'stpete-zero')
ON_VAST = re.compile(r'^ON_VAST(?:_(\d+))?\.json$')
CKPT = re.compile(r'^iter_(\d+)$')
ANCHOR_PREFIX = 'anchor:'

DEFAULTS = {
    'reporter': 'desktop',
    'url': DEFAULT_URL,
    'roots': None,                 # None: default_roots()
    'local': {'id': 'desktop', 'name': 'Desktop', 'hardware': 'i9-13900K / RTX 5090'},
    'sources': {'mac': {'name': 'MacBook Pro', 'kind': 'ssh', 'hardware': 'M4 Max 16-core / 64 GB'}},
    'aliases': {'greggkeithley@100.95.248.116': 'mac'},
    'local_names': ['local', 'gpu:*'],
    'vast_names': ['vast', 'vast:*'],
    'vast_ledger': os.path.join('~', '.config', 'vastai', 'mtgz_boxes.json'),
    'vast_gpu': 'RTX 5090',
    'fuzz': {'source': 'mac', 'host': 'greggkeithley@100.95.248.116', 'ssh': 'ssh',
             'python': '$HOME/.venvs/dominion-zero/bin/python', 'farm': '$HOME/mtgz-fuzzfarm', 'timeout': 90},
}

# Run on the Mac (python - FARM WINDOW_S...), read only: the farm log's finished chunks of the widest
# window, as [time, mode, done, running s]; status.json's state. One line: FARM_REPORT {json}.
FUZZ_SCRIPT = r'''
import json, os, re, sys, time
farm = os.path.expanduser(os.path.expandvars(sys.argv[1]))
wins = [int(x) for x in sys.argv[2:]] or [3600, 86400]
now = time.time()
since = now - max(wins) - 7200
out = {'now': now, 'chunks': [], 'truncated': False}
rx = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+) \[(\d+)\+(\d+)\][^:]*: rc (-?\d+), (\d+) (\S+) in (\d+(?:\.\d+)?) s')
log = os.path.join(farm, 'logs', 'farm.log')
try:
    size = os.path.getsize(log)
    start = max(0, size - (16 << 20))
    with open(log, 'rb') as f:
        f.seek(start)
        lines = f.read().decode('utf-8', 'replace').splitlines()
    if start:
        lines = lines[1:]
    first = None
    for ln in lines:
        m = rx.match(ln)
        if not m:
            continue
        t = time.mktime(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'))
        if first is None:
            first = t
        if t >= since:
            out['chunks'].append([t, m.group(2), int(m.group(6)), float(m.group(8))])
    out['truncated'] = bool(start) and first is not None and first > since
except OSError as e:
    out['log_error'] = str(e)[:200]
try:
    with open(os.path.join(farm, 'status.json'), encoding='utf-8') as f:
        s = json.load(f)
    out.update(state=s.get('state'), updated=s.get('updated'), sha=(s.get('sha') or '')[:12])
except (OSError, ValueError) as e:
    out['status_error'] = str(e)[:200]
print('FARM_REPORT ' + json.dumps(out))
'''


# ------------------------------------------------------------------ small helpers

def _num(x):
    """A finite number (not a bool) as a float, else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _no_window():
    """CREATE_NO_WINDOW for a child of a process without a console window (pythonw, the task): a console
    child (ssh, powershell) would get a visible console of its own, stealing focus. 0 otherwise."""
    if os.name != 'nt':
        return 0
    try:
        import ctypes
        return 0 if ctypes.windll.kernel32.GetConsoleWindow() else subprocess.CREATE_NO_WINDOW
    except (AttributeError, OSError):
        return subprocess.CREATE_NO_WINDOW


def run_cmd(argv, input=None, timeout=None):
    """argv run to completion -> (returncode, stdout, stderr); TimeoutExpired / OSError propagate."""
    kw = {'input': input} if input is not None else {'stdin': subprocess.DEVNULL}
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                       creationflags=_no_window(), **kw)
    return p.returncode, p.stdout or '', p.stderr or ''


def _say(msg, err=False):
    """print, to stderr when `err`; a no-op without the stream (pythonw)."""
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _log(path, line):
    """One line appended to the log `path` ('' / None: none), the file kept under LOG_MAX."""
    if not path:
        return
    p = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        if os.path.exists(p) and os.path.getsize(p) > LOG_MAX:
            with open(p, 'rb') as f:
                f.seek(-LOG_MAX // 2, 2)
                keep = f.read()
            with open(p, 'wb') as f:
                f.write(keep[keep.find(b'\n') + 1:])
        with open(p, 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + line + '\n')
    except OSError:
        pass


def _redact(text, token):
    return text.replace(token, '***') if token else text


# ------------------------------------------------------------------ config

def default_roots(home=None, drives=('E:',)):
    """~/<game>/runs and <drive>/<game>/runs of the zero games, those that exist, mtg-zero's first."""
    home = home or os.path.expanduser('~')
    out = []
    for g in GAMES:
        for base in (home, *(d + os.sep for d in drives)):
            p = os.path.join(base, g, 'runs')
            if os.path.isdir(p):
                out.append(p)
    return out


def load_config(path=None):
    """DEFAULTS overridden by the JSON file `path` (None: DEFAULT_CONFIG, optional; a path given must
    exist). A dict-valued key merges key by key; null for 'fuzz' turns the Mac probe off."""
    cfg = json.loads(json.dumps(DEFAULTS))
    p = os.path.expanduser(path or DEFAULT_CONFIG)
    if not os.path.exists(p):
        if path:
            raise OSError(f'no config file {p}')
        return cfg
    with open(p, encoding='utf-8') as f:
        got = json.load(f)
    if not isinstance(got, dict):
        raise ValueError('the config must be a JSON object')
    for k, v in got.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = dict(cfg[k], **v)
        else:
            cfg[k] = v
    return cfg


def load_token(environ, path):
    """The admin token: env ZEROFARM_ADMIN_TOKEN, else the file `path`; None if neither has one."""
    t = (environ.get(TOKEN_ENV) or '').strip()
    if t:
        return t
    try:
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            t = f.read().strip()
    except OSError:
        return None
    return t or None


# ------------------------------------------------------------------ runs

class RunDir:
    def __init__(self, root, name, path, real, mtime, born=0.0):
        self.root, self.name, self.path, self.real, self.mtime, self.born = root, name, path, real, mtime, born
        self.id = name
        self.elo = {}                  # part -> {iter: Elo games} (loaded when needed)
        self.claimed = set()           # (part, iter) whose Elo games are counted
        self.boxes = None

    def box_at(self, t):
        """The Vast box that ran this run at time t (ON_VAST*.json), or None."""
        if self.boxes is None:
            self.boxes = box_periods(self.path)
        for since, until, box in self.boxes:
            if since <= t <= until:
                return box
        return None

    def elo_games(self, part, it):
        """Checkpoint `it`'s Elo games in this part's pools, once per (part, iteration)."""
        if not isinstance(it, int) or (part, it) in self.claimed:
            return 0
        self.claimed.add((part, it))
        if part not in self.elo:
            self.elo[part] = elo_games_by_iter(os.path.join(self.path, part) if part else self.path)
        return self.elo[part].get(it, 0)


def _newest(path):
    best = None
    for f in RUN_FILES:
        try:
            m = os.stat(os.path.join(path, f)).st_mtime
        except OSError:
            continue
        best = m if best is None else max(best, m)
    return best


def _born(path):
    """The directory's creation time (st_birthtime; Windows' st_ctime), 0.0 where the OS keeps none."""
    try:
        st = os.stat(path)
    except OSError:
        return 0.0
    b = getattr(st, 'st_birthtime', None)
    if b is None and os.name == 'nt':
        b = st.st_ctime
    return float(b or 0.0)


def find_runs(roots, now, active_s=DAY):
    """The run directories under `roots` (in order) changed within `active_s` of `now`, each real
    directory once, with their ids (the module's docstring: Runs)."""
    seen, runs = set(), []
    for root in roots:
        try:
            entries = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                if not e.is_dir():
                    continue
            except OSError:
                continue
            real = os.path.normcase(os.path.realpath(e.path))
            if real in seen:
                continue
            mt = _newest(e.path)
            if mt is None:
                continue
            seen.add(real)
            if mt < now - active_s:
                continue
            runs.append(RunDir(root, e.name, e.path, real, mt, _born(e.path)))
    taken = set()
    for r in runs:
        rid = r.name
        if rid in taken:
            rid = f'{r.name}@{os.path.basename(os.path.dirname(os.path.abspath(r.root)))}'
        if rid in taken:
            rid = f'{r.name}@{os.path.abspath(r.root)}'.replace(os.sep, '/')
        taken.add(rid)
        r.id = rid
    return runs


def box_periods(run):
    """[(since, until, source id)] of the Vast boxes that ran `run` (mtg-zero tools/vast.py: ON_VAST.json
    while it lives on a box, ON_VAST_<home time>.json once home)."""
    out = []
    try:
        names = os.listdir(run)
    except OSError:
        return out
    for n in names:
        m = ON_VAST.match(n)
        if not m:
            continue
        try:
            with open(os.path.join(run, n), encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        since, box = _num(d.get('since')), d.get('box')
        if since is None or not isinstance(box, str) or not box:
            continue
        until = float(m.group(1)) if m.group(1) else math.inf
        out.append((since, until, box_id(box)))
    return out


def box_id(box):
    """A vast.py box name -> its source id (the remote names: 'vast' = the box 'vast', 'vast:<box>')."""
    return 'vast' if box == 'vast' else f'vast:{box}'


def elo_games_by_iter(d):
    """{k: games} of the Elo pools under `d` (elo/results.json, elo/<rule>/results.json): the pairs
    iter_k|iter_j (j < k) and iter_k|anchor:<label>."""
    out = {}
    for p in [os.path.join(d, 'elo', 'results.json')] + sorted(glob.glob(os.path.join(d, 'elo', '*', 'results.json'))):
        try:
            with open(p, encoding='utf-8') as f:
                pairs = json.load(f).get('pairs') or {}
        except (OSError, ValueError, AttributeError):
            continue
        for key, v in pairs.items():
            a, _, b = str(key).partition('|')
            n = _num(v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None
            if not n:
                continue
            ma, mb = CKPT.match(a), CKPT.match(b)
            if ma and mb:
                k = max(int(ma.group(1)), int(mb.group(1)))
            elif ma and b.startswith(ANCHOR_PREFIX):
                k = int(ma.group(1))
            elif mb and a.startswith(ANCHOR_PREFIX):
                k = int(mb.group(1))
            else:
                continue
            out[k] = out.get(k, 0) + int(n)
    return out


def _lines_backwards(path, chunk=1 << 20):
    """The non-empty lines of `path` (bytes), the last first."""
    with open(path, 'rb') as f:
        f.seek(0, 2)
        pos, buf = f.tell(), b''
        while pos > 0:
            n = min(chunk, pos)
            pos -= n
            f.seek(pos)
            buf = f.read(n) + buf
            lines = buf.split(b'\n')
            buf = lines[0]
            for ln in reversed(lines[1:]):
                if ln.strip():
                    yield ln
        if buf.strip():
            yield buf


def timed_rows(path, stop_before):
    """(row, time, derived) of a jsonl file, the newest first, until a row older than `stop_before`
    (that one included). A row without `time`: the next row's time - its seconds (the newest: the
    file's mtime). Unreadable lines (a row being written) are skipped."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return
    nxt = None
    for raw in _lines_backwards(path):
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        t, derived = _num(row.get('time')), False
        if t is None:
            derived = True
            t = mt if nxt is None else nxt[0] - (nxt[1] or 0.0)
        yield row, t, derived
        nxt = (t, _num(row.get('seconds')))
        if t < stop_before:
            return


def eval_games(row):
    """The games a metrics / evals row records as evaluations: every top-level eval_* record's `games`
    and every extras.<name>.games."""
    n = 0
    for k, v in row.items():
        if k.startswith('eval_') and isinstance(v, dict):
            n += _num(v.get('games')) or 0
    ex = row.get('extras')
    if isinstance(ex, dict):
        for v in ex.values():
            if isinstance(v, dict):
                n += _num(v.get('games')) or 0
    return n


def remote_records(sp):
    """[(remote name, its record)] of a self-play record: selfplay.remotes (a list of hosts), else
    selfplay.remote under its host."""
    rems = sp.get('remotes')
    if isinstance(rems, dict) and rems:
        return [(str(n), r) for n, r in rems.items() if isinstance(r, dict)]
    r = sp.get('remote')
    if not isinstance(r, dict):
        return []
    h = r.get('host')
    if isinstance(h, list):                # (several hosts recorded without their shares: kept together)
        h = h[0] if len(h) == 1 else ','.join(str(x) for x in h)
    return [(str(h or 'remote'), r)]


# ------------------------------------------------------------------ the report

class Report:
    """Per source: games per window and kind, work intervals, runs."""

    def __init__(self, cfg, now):
        self.cfg, self.now = cfg, now
        self.local = cfg['local']['id']
        self.acc = {}
        self.seen = set()
        self.stats = []                    # (run id, part, file, rows read, rows counted, derived times)

    def _a(self, sid):
        return self.acc.setdefault(sid, {'selfplay': {w: 0 for w, _ in WINDOWS}, 'eval': {w: 0 for w, _ in WINDOWS},
                                         'fuzz': {w: 0 for w, _ in WINDOWS}, 'busy': [], 'runs': set()})

    def source_of(self, name):
        name = self.cfg.get('aliases', {}).get(name, name)
        if any(fnmatch.fnmatchcase(name, p) for p in self.cfg.get('local_names') or ()):
            return self.local
        return name

    def kind_of(self, sid):
        if sid == self.local:
            return 'local'
        if any(fnmatch.fnmatchcase(sid, p) for p in self.cfg.get('vast_names') or ()):
            return 'vast'
        return (self.cfg.get('sources', {}).get(sid) or {}).get('kind') or 'ssh'

    def games(self, sid, kind, n, t, run_id):
        n = _num(n) or 0
        if n <= 0 or not (self.now - DAY < t <= self.now + FUTURE_S):
            return
        a = self._a(sid)
        for w, s in WINDOWS:
            if t > self.now - s:
                a[kind][w] += n
        if run_id:
            a['runs'].add(run_id)

    def busy(self, sid, t0, t1, run_id):
        if t1 <= t0 or t1 <= self.now - DAY or t0 > self.now + FUTURE_S:
            return
        a = self._a(sid)
        a['busy'].append((t0, t1))
        if run_id:
            a['runs'].add(run_id)

    # ---- one run
    def add_run(self, run):
        stop = self.now - DAY - SLACK_S
        for part in PARTS:
            d = os.path.join(run.path, part) if part else run.path
            self._evals(run, part, os.path.join(d, 'evals.jsonl'), stop)
            self._metrics(run, part, os.path.join(d, 'metrics.jsonl'), stop)

    def _evals(self, run, part, path, stop):
        n = used = derived = 0
        for row, t, der in timed_rows(path, stop):
            n += 1
            key = ('e', part, row.get('iter'), row.get('host'), round(t, 3))
            if key in self.seen:
                continue
            self.seen.add(key)
            used += 1
            derived += der
            src = self.source_of(str(row.get('host') or 'remote'))
            secs = _num(row.get('seconds')) or 0.0
            self.busy(src, t - secs, t, run.id)
            g = eval_games(row)
            if isinstance(row.get('eval_yard'), dict):
                g += run.elo_games(part, row.get('iter'))
            self.games(src, 'eval', g, t, run.id)
        if n:
            self.stats.append((run.id, part, 'evals.jsonl', n, used, derived))

    def _metrics(self, run, part, path, stop):
        n = used = derived = 0
        for m, t, der in timed_rows(path, stop):
            n += 1
            key = ('m', part, m.get('iter'), round(t, 3))
            if key in self.seen:
                continue
            self.seen.add(key)
            used += 1
            derived += der
            ph = m.get('phases') if isinstance(m.get('phases'), dict) else {}
            secs = _num(m.get('seconds'))
            if secs is None:
                secs = sum(_num(v) or 0.0 for v in ph.values())
            start = t - secs
            sp_s = _num(ph.get('selfplay_s'))
            sp_end = min(t, start + sp_s) if sp_s is not None else t
            local = run.box_at(t) or self.local
            self.busy(local, start, t, run.id)
            sp = m.get('selfplay')
            if isinstance(sp, dict):
                self.games(local, 'selfplay', sp.get('games'), sp_end, run.id)
                for name, r in remote_records(sp):
                    src = self.source_of(name)
                    self.games(src, 'selfplay', r.get('games'), sp_end, run.id)
                    rs = _num(r.get('running_s'))
                    rs = _num(r.get('seconds')) if rs is None else rs
                    self.busy(src, sp_end - (rs or 0.0), sp_end, run.id)
                    lg = r.get('linger')
                    if isinstance(lg, dict) and _num(lg.get('seconds')):
                        self.busy(src, sp_end, sp_end + lg['seconds'], run.id)
            g = eval_games(m)
            if isinstance(m.get('eval_yard'), dict):
                g += run.elo_games(part, m.get('iter'))
            self.games(local, 'eval', g, t, run.id)
        if n:
            self.stats.append((run.id, part, 'metrics.jsonl', n, used, derived))

    # ---- the fuzz farm
    def add_fuzz(self, sid, got):
        for c in got.get('chunks') or ():
            try:
                t, _mode, done, secs = float(c[0]), c[1], float(c[2]), float(c[3])
            except (TypeError, ValueError, IndexError):
                continue
            self.games(sid, 'fuzz', done, t, None)
            self.busy(sid, t - secs, t, None)

    def busy_h(self, sid):
        lo, hi = self.now - DAY, self.now
        iv = sorted((max(a, lo), min(b, hi)) for a, b in self._a(sid)['busy'] if b > lo and a < hi)
        total, cur = 0.0, None
        for a, b in iv:
            if b <= a:
                continue
            if cur is None or a > cur[1]:
                if cur is not None:
                    total += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur is not None:
            total += cur[1] - cur[0]
        return round(total / 3600, 2)


def fuzz_probe(fz, runner, windows=WINDOWS):
    """The Mac's fuzz farm read over ssh -> {online, error?, chunks, state, ...}."""
    cmd = [fz.get('ssh') or 'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', fz['host'],
           f'"{fz["python"]}" - "{fz["farm"]}" ' + ' '.join(str(s) for _, s in windows)]
    try:
        rc, out, err = runner(cmd, input=FUZZ_SCRIPT, timeout=fz.get('timeout') or 90)
    except subprocess.TimeoutExpired:
        return {'online': False, 'error': f'ssh {fz["host"]}: no answer in {fz.get("timeout") or 90} s'}
    except OSError as e:
        return {'online': False, 'error': f'ssh: {e}'[:300]}
    lines = [ln for ln in (out or '').splitlines() if ln.startswith(RESULT)]
    if rc != 0 or not lines:
        tail = [ln for ln in ((err or '') + '\n' + (out or '')).splitlines() if ln.strip()][-1:]
        what = 'unreachable' if rc == 255 else f'exit {rc}'
        return {'online': rc != 255, 'error': f'ssh {fz["host"]} {what}: {" ".join(tail)}'[:300]}
    try:
        got = json.loads(lines[-1][len(RESULT):])
    except ValueError as e:
        return {'online': True, 'error': f'fuzz farm status unreadable: {e}'[:300]}
    got['online'] = True
    errs = [got[k] for k in ('log_error', 'status_error') if got.get(k)]
    if got.get('truncated'):
        errs.append('farm.log holds more than 16 MB of the last 24 h: the seeds are undercounted')
    if errs:
        got['error'] = '; '.join(errs)[:300]
    return got


def load_ledger(path):
    try:
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _vast_box(sid):
    return 'vast' if sid == 'vast' else sid.split(':', 1)[1] if ':' in sid else sid


def build_report(cfg, roots, now, runner=None, fuzz=True):
    """-> (the body to POST, the Report)."""
    rep = Report(cfg, now)
    runs = find_runs(roots, now)
    # the oldest directory first: a row a fork copied is counted for the run that played it
    for _, r in sorted(enumerate(runs), key=lambda x: (x[1].born, x[0])):
        rep.add_run(r)
    fz = cfg.get('fuzz') if fuzz else None
    probe = None
    if fz and fz.get('host'):
        probe = fuzz_probe(fz, runner or run_cmd)
        if probe.get('chunks') is not None:
            rep.add_fuzz(fz['source'], probe)
    ledger = load_ledger(cfg.get('vast_ledger') or '')
    ids = set(rep.acc) | {rep.local} | set(cfg.get('sources') or {})
    if fz and fz.get('source'):
        ids.add(fz['source'])
    for name, e in ledger.items():
        if isinstance(e, dict) and e.get('state') in ('running', 'starting'):
            ids.add(box_id(name))
    order = {'local': 0, 'ssh': 1, 'vast': 2}
    sources = []
    for sid in sorted(ids, key=lambda s: (order.get(rep.kind_of(s), 3), s)):
        kind = rep.kind_of(sid)
        a = rep._a(sid)
        conf = cfg['local'] if sid == rep.local else (cfg.get('sources') or {}).get(sid) or {}
        d = {'id': sid, 'name': conf.get('name') or sid, 'kind': kind, 'hardware': conf.get('hardware'),
             'online': True if sid == rep.local else None}
        if kind == 'vast':
            e = ledger.get(_vast_box(sid))
            box = _vast_box(sid)
            if isinstance(e, dict):
                th = _num(e.get('threads'))
                hw = [str(e.get('cpu') or 'cpu ?') + (f' ({th:g} threads)' if th else ''),
                      str(e.get('gpu') or e.get('gpu_name') or cfg.get('vast_gpu') or 'GPU ?')]
                if _num(e.get('ram_gb')):
                    hw.append(f'{e["ram_gb"]:g} GB')
                d['hardware'] = conf.get('hardware') or ' / '.join(hw)
                d['name'] = conf.get('name') or f'Vast box {box}' + (f' ({e["geo"]})' if e.get('geo') else '')
                dl = _num(e.get('deadline'))
                d['online'] = e.get('state') in ('running', 'starting') and (dl is None or dl > now)
            else:
                d['hardware'] = conf.get('hardware') or f'Vast.ai box {box}'
                d['name'] = conf.get('name') or f'Vast box {box}'
        d['working_on'] = [f'run:{r}' for r in sorted(a['runs'])]
        d['selfplay_games'] = {w: int(round(v)) for w, v in a['selfplay'].items()}
        d['eval_games'] = {w: int(round(v)) for w, v in a['eval'].items()}
        d['fuzz_seeds'] = {w: int(round(v)) for w, v in a['fuzz'].items()}
        if fz and sid == fz.get('source') and probe is not None:
            d['online'] = probe['online']
            if probe.get('chunks') is None:
                d['fuzz_seeds'] = {w: None for w, _ in WINDOWS}       # unknown, not 0
            if probe.get('state'):
                d['fuzz_state'] = probe['state']
            if probe.get('error'):
                d['error'] = probe['error']
        d['busy_h_24h'] = rep.busy_h(sid)
        sources.append(d)
    body = {'reporter': cfg.get('reporter') or 'desktop', 't': int(now), 'sources': sources}
    rep.runs = runs
    rep.probe = probe
    return body, rep


# ------------------------------------------------------------------ POST

def http_post(url, data, headers, timeout):
    """-> (status, body text); a network error raises (OSError / http.client.HTTPException)."""
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(4000).decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, (e.read(4000) or b'').decode('utf-8', 'replace')


def post(url, body, token, transport=None, sleep=time.sleep, retries=RETRY_S, timeout=30):
    """POST `body` as JSON with the bearer token -> (ok, message). 5xx and network errors are retried
    after each of `retries` seconds; anything else that is not 2xx is final. The message never holds
    the token."""
    transport = transport or http_post
    data = json.dumps(body).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}', 'User-Agent': USER_AGENT}
    last = ''
    for attempt in range(len(retries) + 1):
        try:
            status, text = transport(url, data, headers, timeout)
        except (OSError, http.client.HTTPException) as e:
            status, text = None, f'{type(e).__name__}: {e}'
        if status is not None and 200 <= status < 300:
            return True, f'HTTP {status}' + (f' after {attempt} retries' if attempt else '')
        last = (f'HTTP {status}: ' if status is not None else '') + ' '.join(str(text).split())[:300]
        if status is not None and status < 500:
            break                                  # a 4xx: the same request would be refused again
        if attempt < len(retries):
            sleep(retries[attempt])
    return False, _redact(last, token)


# ------------------------------------------------------------------ the scheduled task

def _ps_quote(s):
    return "'" + str(s).replace("'", "''") + "'"


def pythonw_path(exe=None):
    """pythonw.exe beside the interpreter (no console window), else the interpreter itself."""
    exe = exe or sys.executable
    w = os.path.join(os.path.dirname(exe), 'pythonw.exe')
    return w if os.path.isfile(w) else exe


def task_install_argv(pythonw, python_dir, args, name=TASK_NAME):
    """The PowerShell command that registers the task (the module's docstring)."""
    argline = subprocess.list2cmdline(['-m', 'zero.farm_report', *args])
    script = '; '.join([
        "$ErrorActionPreference = 'Stop'",
        f'$a = New-ScheduledTaskAction -Execute {_ps_quote(pythonw)} -Argument {_ps_quote(argline)} '
        f'-WorkingDirectory {_ps_quote(python_dir)}',
        f'$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes {TASK_EVERY_MIN})',
        '$p = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) '
        '-LogonType Interactive -RunLevel Limited',
        f'$s = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes '
        f'{TASK_LIMIT_MIN}) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries',
        f'Register-ScheduledTask -TaskName {_ps_quote(name)} -Description '
        f'{_ps_quote("zero.farm_report: the own machines' contributions to zero-farm (zero-core docs/az.md 44.3)")} '
        '-Action $a -Trigger $t -Principal $p -Settings $s -Force | Out-Null',
        f"Write-Output ('registered ' + {_ps_quote(name)})",
    ])
    return ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script]


def task_uninstall_argv(name=TASK_NAME):
    script = (f"$ErrorActionPreference = 'Stop'; Unregister-ScheduledTask -TaskName {_ps_quote(name)} -Confirm:$false; "
              f"Write-Output ('removed ' + {_ps_quote(name)})")
    return ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script]


def task_args(a):
    """The options of this invocation the task passes on (paths made absolute)."""
    out = []
    if a.config:
        out += ['--config', os.path.abspath(os.path.expanduser(a.config))]
    if a.roots:
        out += ['--roots', *[os.path.abspath(os.path.expanduser(r)) for r in a.roots]]
    if a.url:
        out += ['--url', a.url]
    if a.no_fuzz:
        out.append('--no-fuzz')
    if a.log != DEFAULT_LOG:
        out += ['--log', a.log]
    return out


def task(a, runner):
    if a.install_task:
        root = os.path.abspath(a.task_root or PYTHON_DIR)
        if not os.path.isfile(os.path.join(root, 'zero', 'farm_report.py')):
            _say(f'farm_report: {root} holds no zero/farm_report.py (--task-root: the directory holding the zero '
                 f'package)', err=True)
            return 2
        argv = task_install_argv(pythonw_path(), root, task_args(a))
    else:
        argv = task_uninstall_argv()
    rc, out, err = runner(argv, timeout=120)
    for ln in (out + err).strip().splitlines():
        _say(ln, err=rc != 0)
    return 0 if rc == 0 else 1


# ------------------------------------------------------------------ main

def parse_args(argv):
    ap = argparse.ArgumentParser(prog='python -m zero.farm_report', allow_abbrev=False,
                                 description="The own machines' contributions, POSTed to zero-farm's admin page.")
    ap.add_argument('--config', help=f'the config file (default {DEFAULT_CONFIG}, optional)')
    ap.add_argument('--roots', nargs='+', metavar='DIR', help="run roots (instead of the config's)")
    ap.add_argument('--url', help=f'the coordinator (default: the config\'s, {DEFAULT_URL})')
    ap.add_argument('--token-file', default=TOKEN_FILE, help=f'the admin token file ({TOKEN_FILE}; env {TOKEN_ENV} first)')
    ap.add_argument('--no-fuzz', action='store_true', help="don't read the Mac's fuzz farm")
    ap.add_argument('--dry-run', action='store_true', help='print the report, post nothing')
    ap.add_argument('--log', default=DEFAULT_LOG, help=f"one line per run ({DEFAULT_LOG}; '' = none)")
    ap.add_argument('-v', '--verbose', action='store_true', help='the runs and rows read, on stderr')
    ap.add_argument('--install-task', action='store_true', help=f'register the Task Scheduler task {TASK_NAME}')
    ap.add_argument('--uninstall-task', action='store_true', help=f'remove the task {TASK_NAME}')
    ap.add_argument('--task-root', help='the directory holding the zero package, the task\'s working directory '
                                        '(default: this one\'s)')
    return ap.parse_args(argv)


def main(argv=None, *, now=None, runner=None, transport=None, sleep=time.sleep, environ=None):
    a = parse_args(argv)
    runner = runner or run_cmd
    if a.install_task or a.uninstall_task:
        return task(a, runner)
    environ = os.environ if environ is None else environ
    try:
        cfg = load_config(a.config)
    except (OSError, ValueError) as e:
        _say(f'farm_report: config: {e}', err=True)
        return 2
    token = None
    if not a.dry_run:
        token = load_token(environ, a.token_file)
        if not token:
            _say(f'farm_report: no admin token (env {TOKEN_ENV} or {a.token_file})', err=True)
            _log(a.log, 'FAILED: no admin token')
            return 2
    roots = a.roots or cfg.get('roots') or default_roots()
    now = time.time() if now is None else now
    body, rep = build_report(cfg, roots, now, runner=runner, fuzz=not a.no_fuzz)
    if a.verbose:
        for r in rep.runs:
            _say(f'run {r.id}: {r.path} (changed {(now - r.mtime) / 3600:.1f} h ago)', err=True)
        for rid, part, f, n, used, der in rep.stats:
            _say(f'  {rid} {part + "/" if part else ""}{f}: {n} rows read, {used} not in an older run'
                 + (f', {der} timed from their neighbours' if der else ''), err=True)
        if rep.probe is not None:
            _say(f'fuzz farm: online {rep.probe.get("online")}, {len(rep.probe.get("chunks") or [])} chunks'
                 + (f', {rep.probe["error"]}' if rep.probe.get('error') else ''), err=True)
    if a.dry_run:
        _say(json.dumps(body, indent=1))
        return 0
    url = (a.url or cfg.get('url') or DEFAULT_URL).rstrip('/') + ENDPOINT
    ok, msg = post(url, body, token, transport=transport, sleep=sleep)
    off = [s['id'] for s in body['sources'] if s.get('online') is False]
    line = (f'{"posted" if ok else "FAILED to post"} {len(body["sources"])} sources from {len(rep.runs)} runs to {url}: '
            f'{msg}' + (f' (offline: {", ".join(off)})' if off else ''))
    line = _redact(line, token)
    _say(line, err=not ok)
    _log(a.log, line)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
