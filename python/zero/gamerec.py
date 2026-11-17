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
"""
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import threading
import time
import zlib

FORMAT = 'zero-gamerec/1'
DIR = 'records'
DONE = 'DONE.json'


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
    return out, spec


def actions_of(rec):
    """A record's real moves: [int] (actions), or the trace list when a step is not an action."""
    if rec.get('actions') is not None:
        b = base64.b64decode(rec['actions'])
        return list(memoryview(b).cast('H')) if len(b) else []
    return list(rec.get('steps') or [])


class Records:
    """A run's game records writer (module docs): thread-safe; one gzip'd file per (iteration, source) of this process."""

    def __init__(self, run, log=None):
        self.run = run
        self.log = log or (lambda m: None)
        self.mu = threading.Lock()
        self.files = {}                    # (it, source) -> {'f', 'path', 'specs', 'games'}
        self.secs = {}                     # it -> seconds spent here
        self.pid = os.getpid()

    def _file(self, it, src):
        k = (int(it), src)
        f = self.files.get(k)
        if f is None:
            d = iter_dir(self.run, it)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f'{safe(src)}-{self.pid}.jsonl.gz')
            f = self.files[k] = {'f': gzip.open(path, 'at', encoding='utf-8', compresslevel=6), 'path': path,
                                 'specs': set(), 'games': 0}
        return f

    def add(self, it, src, items):
        """Records of iteration `it` from source `src`: [(record dict, spec JSON)] (from_row / from_farm)."""
        if not items:
            return
        t0 = time.perf_counter()
        with self.mu:
            f = self._file(it, src)
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
            files = {os.path.basename(v['path']): {'source': k[1], 'games': v['games'],
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


def read_file(path):
    """A records file -> (specs {key: JSON}, [record dicts]). A truncated gzip (a process that died writing it) yields
    the lines before the cut."""
    specs, recs = {}, []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.endswith('\n'):
                    break
                o = json.loads(line)
                if o.get('kind') == 'spec':
                    specs[o['spec']] = o['json']
                else:
                    recs.append(o)
    except (EOFError, zlib.error, gzip.BadGzipFile):
        pass                               # (cut short: the lines before the cut)
    return specs, recs


def read_iteration(run, it):
    """Iteration `it`'s records as DONE.json lists them -> (specs, records). None when it is not complete."""
    done = read_done(run, it)
    if done is None:
        return None
    specs, recs = {}, []
    for name in sorted(done.get('files') or {}):
        s, r = read_file(os.path.join(iter_dir(run, it), name))
        specs.update(s)
        recs += r
    return specs, recs


def drop(run, it):
    """Delete iteration `it`'s records (gamedb prune: after ingest)."""
    shutil.rmtree(iter_dir(run, it), ignore_errors=True)
