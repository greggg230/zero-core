"""Game records (docs/az.md 59): every finished self-play game of a run, as what replays it, kept per iteration outside
the training window - for a database of games (MTG Zero's `tools/gamedb.py`), never for training.

On with the run setting `game_records` (default 0: nothing is written, nothing changes). Then every self-play game an
iteration's quota accepts - this machine's driver's, a remote host's workers' (zero/remoteplay.py: the worker sends the
records of the games the run accepted), the farm pool's (zero/farm_pool.py: the farm's records already hold every move)
- is one JSON line in

    <run>/records/iter_NNNN/<source>-<pid>.jsonl.gz

(`source`: the shared queue's consumer - 'local', 'w0' or '<host>/w0' as a file-safe name, 'pool-<farm run>'). The
records are read from the driver's game log (`SearchDriver.set_game_log`, docs/search.md 2.20: the game's seed and every
real move the driver keeps anyway), so the games, their search and their samples are those of a run without records:
shards, checkpoints and every digest stay byte-identical. A line:

    {"v": 1, "it": accepting iteration, "src": source, "gid", "seed": the game's seed (new_game's), "spec": spec key,
     "n": players, "actions": base64 of the real moves as little-endian u16 (null when a step is not an action; then
     "steps": the trace record's list), "n_actions", "setup": the game's record_setup (MTG Zero: the decks and the
     metagame's pairing), "over", "aborted", "resigned": the winner seat of a resigned game or null, "abort_reason"
     (only when set), "result": [per seat], "decisions" (null: the farm's), "rounds", "nets": [lo, hi] the networks
     its samples were searched with (null: none), "machine" (the farm's: a short hash of the machine id), "t"}

and a spec line `{"v": 1, "kind": "spec", "spec": key, "json": the spec's JSON}` comes before the first game of that
spec in each file (`key` = sha256 of the JSON, 16 hex digits). A replay is `new_game(json, n, seed)` + the actions.

`records/iter_NNNN/DONE.json` (written when the iteration's self-play phase ends, before its data directory's DONE.json)
lists the files of the process that played the phase - the only ones a reader takes: a file of an interrupted attempt
(another pid) is not listed - with each source's records, the games each source's quota accepted (`accepted`; a gap
is a source that sent none: a worker on an older checkout) and the seconds spent recording.

Every other game (docs/az.md 59.3): a record of the same format plus `kind`, `it` (null for a standalone tool's),
`players` (per seat: `net_player` / `bot_player`) and `meta`, in `<records>/<kind>/iter_NNNN/` (`misc/`), one whole file
per match or batch (`MatchSink`) - the run's evaluations with `game_records` ('eval', 'elo', 'anchor', 'ref'), zero.h2h
while a thread records (`recording`), a game repo's own tools. Every records file starts with a build line (docs/az.md
59.4): `{"v": 1, "kind": "build", "build": build_info()}` - the commits and the engine id the records replay on.
"""
import base64
import contextlib
import glob
import gzip
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import zlib

FORMAT = 'zero-gamerec/1'
DIR = 'records'
DONE = 'DONE.json'
LINE_KINDS = ('spec', 'build')             # a file's non-game lines (`kind`); a game's own `kind` is never one of them


_KEYS = {}


def spec_key(spec):
    """A spec's key: sha256 of its canonical JSON (sorted keys, no spaces), 16 hex digits - one key for one spec value,
    whichever text carried it."""
    k = _KEYS.get(spec)
    if k is None:
        canon = json.dumps(json.loads(spec), sort_keys=True, separators=(',', ':'))
        k = hashlib.sha256(canon.encode('utf-8')).hexdigest()[:16]
        if len(_KEYS) < 256:
            _KEYS[spec] = k
    return k


def iter_dir(run, it):
    return os.path.join(run, DIR, f'iter_{int(it):04d}')


def safe(name):
    """A source name as a file name part ('host/w0' -> 'host_w0', 'pool:x' -> 'pool-x')."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(name).replace('pool:', 'pool-'))


def machine_tag(machine):
    """A farm machine id as the records keep it: a short hash (never the id itself)."""
    return hashlib.sha256(str(machine).encode('utf-8')).hexdigest()[:8] if machine else None


def from_row(row, it, src):
    """A driver's result row with its game log (`row['log']`) -> (record dict, spec JSON); None without a log."""
    lg = row.get('log')
    if not lg:
        return None
    rs = row.get('resign') or {}
    rec = {'v': 1, 'it': int(it), 'src': src, 'gid': int(row['gid']), 'seed': int(lg['seed']),
           'spec': spec_key(lg['spec']), 'n': int(row['n'])}
    if lg.get('actions') is not None:
        rec['actions'] = base64.b64encode(bytes(lg['actions'])).decode('ascii')
    else:
        rec['actions'] = None
        rec['steps'] = json.loads(lg['steps'])
    rec.update({'n_actions': int(lg['n_actions']), 'setup': lg.get('setup') or '', 'over': bool(row['over']),
                'aborted': bool(row['aborted']), 'resigned': rs.get('winner') if rs.get('resigned') else None,
                'result': [float(x) for x in row['result']], 'decisions': row.get('decisions'),
                'rounds': row.get('rounds'), 'nets': lg.get('nets'), 't': round(time.time(), 1)})
    if row.get('abort_reason'):
        rec['abort_reason'] = row['abort_reason']
    if row.get('branch') is not None:
        rec['branch'] = branch_block(row['branch'], lg.get('branch_from'))
    return rec, lg['spec']


def from_farm(rec, spec, it, src, machine=None):
    """A farm record (zero_search::record::game_json's object, docs/search.md 2.17) of spec `spec` -> (record dict,
    spec JSON): its moves and outcome, without its samples (their networks kept as `nets`)."""
    import struct
    steps = rec['steps']
    if all(isinstance(a, int) for a in steps):
        acts = base64.b64encode(struct.pack(f'<{len(steps)}H', *steps)).decode('ascii')
        extra = {}
    else:
        acts, extra = None, {'steps': steps}
    nets = (rec.get('samples') or {}).get('net') or []
    out = {'v': 1, 'it': int(it), 'src': src, 'gid': int(rec['gid']), 'seed': int(rec['seed']), 'spec': spec_key(spec),
           'n': int(rec['n']), 'actions': acts, **extra, 'n_actions': len(steps), 'setup': rec.get('setup') or '',
           'over': bool(rec['over']), 'aborted': bool(rec['aborted']), 'resigned': rec.get('resigned'),
           'result': [struct.unpack('<f', struct.pack('<I', int(b)))[0] for b in rec['result']],
           'decisions': None, 'rounds': rec.get('rounds'), 'nets': [min(nets), max(nets)] if nets else None,
           'machine': machine_tag(machine), 't': round(time.time(), 1)}
    if rec.get('abort_reason'):
        out['abort_reason'] = rec['abort_reason']
    if rec.get('branch') is not None:
        out['branch'] = branch_block(rec['branch'], rec['branch'].get('from'))
    return out, spec


def branch_block(b, frm):
    """A branch game's `branch` block in a game record (docs/az.md 72.7): its actions start with its game's first
    `from` real moves (the same gid and seed), then its own; `merged` (null, or its decisions after the deviation): a
    near-tie branch that reached one of its game's later states and ended there - its `result` is its game's and its
    last state is a game in progress. The other keys: where it branched (`at` of `of` eligible decisions, the game's
    `decision` / `step` / `action` there), its deviation moves, the near-tie rule's runner-up `alt`, gap and tag."""
    out = {'from': None if frm is None else int(frm)}
    for k in ('at', 'of', 'decision', 'step', 'action', 'moves', 'played', 'same_first', 'alt', 'gap', 'tag', 'merged'):
        if k in b:
            out[k] = b[k]
    return out


def actions_of(rec):
    """A record's real moves: [int] (actions), or the trace list when a step is not an action."""
    if rec.get('actions') is not None:
        b = base64.b64decode(rec['actions'])
        return list(memoryview(b).cast('H')) if len(b) else []
    return list(rec.get('steps') or [])


# ---------------------------------------------------------------- the engine that played (docs/az.md 59.4)
CORE_ENGINE_PATHS = ('crates', 'Cargo.toml', 'Cargo.lock')
_BUILD = {}


def _git(root, *args):
    try:
        r = subprocess.run(['git', '-C', root, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def core_root():
    """This zero-core checkout (python/zero/gamerec.py -> the repository's root)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def engine_id(root, paths, core=None):
    """The identity of the engine a checkout builds (docs/az.md 59.4): sha256[:16] over the tracked files' blob ids under
    the game package's `engine_paths` at HEAD of `root` and under zero-core's crates (CORE_ENGINE_PATHS) at HEAD of
    `core` - every Rust input of the game's engine, nothing else, so a Python or docs commit keeps it and any engine
    change moves it. 'dirty' when one of those files has an uncommitted change (the module may be built from it: no
    commit names that engine); None without git or without `paths`."""
    if not paths:
        return None
    core = core or core_root()
    a = _git(root, 'ls-tree', '-r', 'HEAD', '--', *paths)
    b = _git(core, 'ls-tree', '-r', 'HEAD', '--', *CORE_ENGINE_PATHS)
    if a is None or b is None:
        return None
    da = _git(root, 'status', '--porcelain', '--untracked-files=no', '--', *paths)
    db = _git(core, 'status', '--porcelain', '--untracked-files=no', '--', *CORE_ENGINE_PATHS)
    if da or db:
        return 'dirty'
    return hashlib.sha256((a + '\n--zero-core--\n' + b).encode('utf-8')).hexdigest()[:16]


def build_info(refresh=False):
    """This process's engine (docs/az.md 59.4): {'pkg': the game package, 'commit': its checkout's HEAD, 'core':
    zero-core's HEAD, 'engine': `engine_id` (the game package's `engine_paths`), 'module': sha256[:24] of the engine
    module's file} - what every game record file says it was played by. Cached (one git call per process)."""
    if _BUILD and not refresh:
        return dict(_BUILD)
    from . import game_failure as GF
    from . import package
    try:
        pkg = package.active()
    except Exception:                              # noqa: BLE001 - (no package)
        pkg = None
    root = pkg.root if pkg is not None else None
    commit, module = GF.local_ids()
    core = GF.head_commit(core_root())
    paths = getattr(pkg, 'engine_paths', None)
    _BUILD.clear()
    _BUILD.update({'pkg': getattr(pkg, 'name', None), 'commit': commit, 'core': core,
                   'engine': engine_id(root, paths) if root else None, 'module': module})
    return dict(_BUILD)


# ---------------------------------------------------------------- every other game (docs/az.md 59.3)
def ckpt_id(path):
    """A checkpoint's id in the records: '<run>/iter_NNNN' for a run's checkpoint (runs/<run>/models/iter_NNNN.pt),
    else its file name without the extension (an anchor's, a frozen reference's). `sha` (`file_sha`) is the identity
    across machines (a remote mirror's anchor file has another name)."""
    p = os.path.abspath(str(path)).replace('\\', '/')
    m = re.search(r'/([^/]+)/models/(iter_\d+)\.pt$', p)
    return f'{m.group(1)}/{m.group(2)}' if m else os.path.splitext(os.path.basename(p))[0]


_SHA = {}


def file_sha(path):
    """sha256[:12] of a file (cached by size and modification time), None when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if k not in _SHA:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        _SHA[k] = h.hexdigest()[:12]
    return _SHA[k]


def net_player(path, label=None, seat=None, rule=None):
    """A network player of a record: {'ckpt': ckpt_id, 'sha', 'label' (the caller's name for it: 'iter_0120', 'gen0',
    an anchor's), 'sims', 'rule' (its move rule's tag)}."""
    cid = ckpt_id(path)
    out = {'ckpt': cid, 'sha': file_sha(path), 'label': str(label) if label is not None else cid}
    if seat:
        out['sims'] = seat.get('sims')
    if rule is not None:
        out['rule'] = rule
    return out


def bot_player(name, label=None):
    """A non-network player of a record (a built-in bot or 'ext:<name>')."""
    return {'ckpt': None, 'bot': str(name), 'label': str(label) if label is not None else str(name)}


def _check_kind(kind):
    if not isinstance(kind, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', kind) or kind in LINE_KINDS:
        raise ValueError(f'game records: kind {kind!r} (a lower-case name, not {LINE_KINDS})')
    if re.fullmatch(r'iter_\d+', kind):
        raise ValueError(f'game records: kind {kind!r} looks like a self-play iteration directory')


def from_match_row(row, kind, players, it=None, src='local', meta=None):
    """A driver's result row with its game log (`SearchDriver.set_game_log`) of a game that is not self-play -> (record,
    spec JSON); None without a log. The record is the self-play record's (module docs) with `kind`, `it` (the
    iteration it belongs to - the checkpoint evaluated -, None for a standalone tool's), `players` (per seat, in the
    game's seat order: `net_player` / `bot_player`) and `meta` (the caller's: the pairing, the reference, the match's
    driver seed ...)."""
    lg = row.get('log')
    if not lg:
        return None
    rs = row.get('resign') or {}
    rec = {'v': 1, 'kind': kind, 'it': None if it is None else int(it), 'src': src, 'gid': int(row['gid']),
           'seed': int(lg['seed']), 'spec': spec_key(lg['spec']), 'n': int(row['n'])}
    if lg.get('actions') is not None:
        rec['actions'] = base64.b64encode(bytes(lg['actions'])).decode('ascii')
    else:
        rec['actions'] = None
        rec['steps'] = json.loads(lg['steps'])
    rec.update({'n_actions': int(lg['n_actions']), 'setup': lg.get('setup') or '', 'over': bool(row['over']),
                'aborted': bool(row['aborted']), 'resigned': rs.get('winner') if rs.get('resigned') else None,
                'result': [float(x) for x in row['result']], 'decisions': row.get('decisions'),
                'rounds': row.get('rounds'), 'players': list(players), 'meta': dict(meta or {}),
                't': round(time.time(), 1)})
    if row.get('abort_reason'):
        rec['abort_reason'] = row['abort_reason']
    return rec, lg['spec']


def from_trace(rec, spec, kind, players, it=None, src='farm', meta=None):
    """A traced game record (`zero_search::record::game_json`: a farm play job's `rec`, docs/az.md 50) of spec `spec`
    -> (record, spec JSON) as `from_match_row`'s."""
    out, spec = from_farm(rec, spec, 0, src)
    out.pop('machine', None)
    out.update(kind=kind, it=None if it is None else int(it), players=list(players), meta=dict(meta or {}))
    return out, spec


_SEQ = itertools.count(1)                  # (a file name's sequence number: unique in the process)


def _sub(it):
    return f'iter_{int(it):04d}' if it is not None else 'misc'


class MatchSink:
    """Where the games of one kind that are not self-play go (docs/az.md 59.3): `<records>/<kind>/iter_NNNN/` (`it`: the
    iteration they belong to) or `<records>/<kind>/misc/` (a standalone tool's), one gzip'd JSON-lines file per `add`
    (a match, a batch), written whole - a temp file renamed - so a reader takes every `*.jsonl.gz` there as complete:
    a build line (`build_info`, or `build`), the spec lines, then the records. `labels`: the callers' names for the
    sides ({'a': ..., 'b': ...}: zero.h2h's A and B). `meta`: merged into every record's `meta`."""

    def __init__(self, records, kind, it=None, src='local', meta=None, labels=None, build=None, log=None):
        _check_kind(kind)
        self.records = records
        self.kind = kind
        self.it = it
        self.src = src
        self.meta = dict(meta or {})
        self.labels = dict(labels or {})
        self.build = build
        self.log = log or (lambda m: None)
        self.pid = os.getpid()
        self.stats = {'files': 0, 'games': 0, 'bytes': 0, 'seconds': 0.0}
        self.paths = []

    def dir(self):
        return os.path.join(self.records, self.kind, _sub(self.it))

    def with_meta(self, **meta):
        """A sink writing where this one does with more `meta` in its records (a match of a measurement: its pairing,
        its chooser); its files count in this one's `stats` and `paths`."""
        c = MatchSink(self.records, self.kind, self.it, self.src, dict(self.meta, **meta), self.labels, self.build,
                      self.log)
        c.stats, c.paths = self.stats, self.paths
        return c

    def add(self, items, meta=None):
        """[(record, spec JSON)] -> the file written (None for no items)."""
        if not items:
            return None
        t0 = time.perf_counter()
        d = self.dir()
        os.makedirs(d, exist_ok=True)
        name = f'{safe(self.src)}-{self.pid}-{int(time.time() * 1000)}-{next(_SEQ)}.jsonl.gz'
        out = [json.dumps({'v': 1, 'kind': 'build', 'build': self.build or build_info()}, separators=(',', ':'))]
        seen = set()
        for rec, spec in items:
            if rec['spec'] not in seen:
                out.append(json.dumps({'v': 1, 'kind': 'spec', 'spec': rec['spec'], 'json': spec}, separators=(',', ':')))
                seen.add(rec['spec'])
            if self.meta or meta:
                rec = dict(rec, meta=dict(self.meta, **(meta or {}), **rec.get('meta', {})))
            out.append(json.dumps(rec, separators=(',', ':')))
        path = os.path.join(d, name)
        tmp = path + f'.{self.pid}.tmp'
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=6) as f:
            f.write('\n'.join(out) + '\n')
        os.replace(tmp, path)
        self.stats['files'] += 1
        self.stats['games'] += len(items)
        self.stats['bytes'] += os.path.getsize(path)
        self.stats['seconds'] += time.perf_counter() - t0
        self.paths.append(path)
        return path

    def add_rows(self, rows, players_of, meta=None):
        """A driver's result rows (with their logs, `set_game_log`): each a record with players_of(gid) -> [player per
        seat]. The rows' `log` is taken out (the caller's rows are those of a driver without the log). -> the file."""
        items = []
        for r in rows:
            x = from_match_row(r, self.kind, players_of(int(r['gid'])), self.it, self.src)
            r.pop('log', None)
            if x is not None:
                items.append(x)
        if len(items) != len(rows):
            # (a driver without the log on: never a silent gap)
            raise RuntimeError(f'game records ({self.kind}): {len(rows) - len(items)} of {len(rows)} rows carry no game '
                               f'log (set_game_log was not on)')
        return self.add(items, meta)


_TL = threading.local()


@contextlib.contextmanager
def recording(sink):
    """The games this thread plays through zero.h2h (`match`, `sequential`) while the block runs are written to `sink`
    (a MatchSink; None: nothing - the block runs as without). -> the sink."""
    if sink is None:
        yield None
        return
    st = getattr(_TL, 'stack', None)
    if st is None:
        st = _TL.stack = []
    st.append(sink)
    try:
        yield sink
    finally:
        st.pop()


def active():
    """The innermost `recording` sink of this thread, or None."""
    st = getattr(_TL, 'stack', None)
    return st[-1] if st else None


def match_dirs(records):
    """The directories of the records under `records` that are not self-play: [(kind, sub, path)], sorted."""
    out = []
    if not os.path.isdir(records):
        return out
    for kind in sorted(os.listdir(records)):
        kd = os.path.join(records, kind)
        if not os.path.isdir(kd) or re.fullmatch(r'iter_\d+', kind):
            continue
        for sub in sorted(os.listdir(kd)):
            if os.path.isdir(os.path.join(kd, sub)):
                out.append((kind, sub, os.path.join(kd, sub)))
    return out


def match_files(records):
    """Every complete file of records that are not self-play under `records` -> [(kind, sub, path)] (temp files of a
    write in progress left out)."""
    return [(k, s, p) for k, s, d in match_dirs(records) for p in sorted(glob.glob(os.path.join(d, '*.jsonl.gz')))]


class Records:
    """A run's game records writer (module docs): thread-safe; one gzip'd file per (iteration, source) of this process."""

    def __init__(self, run, log=None):
        self.run = run
        self.log = log or (lambda m: None)
        self.mu = threading.Lock()
        self.files = {}                    # (it, source) -> {'f', 'path', 'specs', 'games', 'build'}
        self.secs = {}                     # it -> seconds spent here
        self.pid = os.getpid()

    def _file(self, it, src, build=None):
        k = (int(it), src)
        f = self.files.get(k)
        if f is None:
            d = iter_dir(self.run, it)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f'{safe(src)}-{self.pid}.jsonl.gz')
            f = self.files[k] = {'f': gzip.open(path, 'at', encoding='utf-8', compresslevel=6), 'path': path,
                                 'specs': set(), 'games': 0, 'build': build or build_info()}
            # (the engine that played the file's games, docs/az.md 59.4: its first line)
            f['f'].write(json.dumps({'v': 1, 'kind': 'build', 'build': f['build']}, separators=(',', ':')) + '\n')
        return f

    def add(self, it, src, items, build=None):
        """Records of iteration `it` from source `src`: [(record dict, spec JSON)] (from_row / from_farm). `build`: the
        engine that played them (`build_info` of the process that did: a remote worker's, the farm pool's), None: this
        process's."""
        if not items:
            return
        t0 = time.perf_counter()
        with self.mu:
            f = self._file(it, src, build)
            out = []
            for rec, spec in items:
                k = rec['spec']
                if k not in f['specs']:
                    out.append(json.dumps({'v': 1, 'kind': 'spec', 'spec': k, 'json': spec}, separators=(',', ':')))
                    f['specs'].add(k)
                rec = dict(rec, it=int(it), src=src)
                out.append(json.dumps(rec, separators=(',', ':')))
                f['games'] += 1
            f['f'].write('\n'.join(out) + '\n')
            self.secs[int(it)] = self.secs.get(int(it), 0.0) + time.perf_counter() - t0

    def add_rows(self, it, src, rows):
        """Accepted result rows (with their logs) of source `src` into iteration `it`; rows without a log are skipped."""
        self.add(it, src, [x for x in (from_row(r, it, src) for r in rows) if x is not None])

    def close(self, it, accepted=None):
        """Iteration `it`'s self-play phase is over: its files closed and DONE.json written (module docs). `accepted`:
        {source: games the quota accepted from it}. -> the summary (the iteration's `records` metrics)."""
        it = int(it)
        t0 = time.perf_counter()
        with self.mu:
            mine = {k: v for k, v in self.files.items() if k[0] == it}
            for k, v in mine.items():
                v['f'].close()
                del self.files[k]
            files = {os.path.basename(v['path']): {'source': k[1], 'games': v['games'], 'build': v['build'],
                                                     'bytes': os.path.getsize(v['path'])} for k, v in mine.items()}
            secs = self.secs.pop(it, 0.0) + time.perf_counter() - t0
        by = {}
        for v in files.values():
            by[v['source']] = by.get(v['source'], 0) + v['games']
        out = {'format': FORMAT, 'it': it, 'pid': self.pid, 'files': files, 'games': sum(by.values()),
               'bytes': sum(v['bytes'] for v in files.values()), 'by_source': by, 'seconds': round(secs, 3),
               'time': time.time()}
        if accepted is not None:
            acc = {str(k): int(v) for k, v in accepted.items() if int(v)}
            out['accepted'] = acc
            miss = {k: v - by.get(k, 0) for k, v in acc.items() if v > by.get(k, 0)}
            if miss:
                out['missing'] = miss
                self.log(f'iter {it} game records: {sum(miss.values())} accepted games without a record {miss} '
                         f'(a source that sends none: an older checkout, docs/az.md 59)')
        d = iter_dir(self.run, it)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, DONE + f'.{self.pid}.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, os.path.join(d, DONE))
        return {k: out[k] for k in ('games', 'bytes', 'seconds', 'by_source') if k in out} | (
            {'missing': out['missing']} if 'missing' in out else {})

    def abort(self):
        """Close every open file (an error or PAUSE on the way out); no DONE.json is written for them."""
        with self.mu:
            for v in self.files.values():
                try:
                    v['f'].close()
                except OSError:
                    pass
            self.files.clear()


def read_done(run, it):
    """Iteration `it`'s DONE.json, or None (no complete iteration of records)."""
    p = os.path.join(iter_dir(run, it), DONE)
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def done_iterations(run):
    """The iterations with a DONE.json, in order."""
    out = []
    for name in os.listdir(os.path.join(run, DIR)) if os.path.isdir(os.path.join(run, DIR)) else []:
        m = re.fullmatch(r'iter_(\d+)', name)
        if m and os.path.exists(os.path.join(run, DIR, name, DONE)):
            out.append(int(m.group(1)))
    return sorted(out)


def read_file(path, builds=None):
    """A records file -> (specs {key: JSON}, [record dicts]). A truncated gzip (a process that died writing it) yields
    the lines before the cut. `builds` (a list): receives the file's build lines (docs/az.md 59.4; none in a file
    written before them)."""
    specs, recs = {}, []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.endswith('\n'):
                    break
                o = json.loads(line)
                k = o.get('kind')
                if k == 'spec':
                    specs[o['spec']] = o['json']
                elif k == 'build':
                    if builds is not None:
                        builds.append(o.get('build') or {})
                else:
                    recs.append(o)
    except (EOFError, zlib.error, gzip.BadGzipFile):
        pass                               # (cut short: the lines before the cut)
    return specs, recs


def read_iteration(run, it, builds=None):
    """Iteration `it`'s records as DONE.json lists them -> (specs, records). None when it is not complete. `builds` (a
    dict): {file name: [its build lines]}."""
    done = read_done(run, it)
    if done is None:
        return None
    specs, recs = {}, []
    for name in sorted(done.get('files') or {}):
        b = []
        s, r = read_file(os.path.join(iter_dir(run, it), name), b)
        if builds is not None:
            builds[name] = b
        specs.update(s)
        recs += r
    return specs, recs


def drop(run, it):
    """Delete iteration `it`'s records (gamedb prune: after ingest)."""
    shutil.rmtree(iter_dir(run, it), ignore_errors=True)
