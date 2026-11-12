"""Signed definitions on zero-farm (zero-farm PROTOCOL.md 7, rev 6; docs/az.md 44.3): what a run or a campaign lets
friends' PCs download and run, signed on the desktop with the data key (the coordinator never holds it), and the
desktop's coordinator client the farm tools share.

A definition (`definition/1`, the bytes of `definition.json`, signed as `definition.json.minisig`):

  {"definition": 1, "kind": "run" | "campaign", "id": <slug>, "version": n, "created": t, "expires": t | null,
   "game": "mtg", "builds": [<worker build_id>, ...], "runtime": <runtime build_id> | null,
   "checkpoints": {"<gen>": {"model_sha256", "model_size", "reference_sha256", "reference_size"}},
   "jobs": {"selfplay": {"settings"} | null, "eval": {"settings", "seats", "paired", ...} | null,
            "fuzz": {"mode", "env"} | null}}

The job templates are exact: an agent builds its job line from them plus only the per-lease fields (self-play /
eval `gid_base`, `games`, `seed`; fuzz `first`, `count`), so a tool puts the run's / campaign's job object equal to
its template (`zero.farm_h2h`'s eval job is the template itself).

* `version` is strictly increasing per (kind, id): `next_version` takes the larger of the coordinator's current one
  (`GET /v1/admin/run/{id}` or `GET /v1/admin/campaign/{id}`: its `definition`) and the newest one kept here, plus 1.
* Every definition is kept, as written and signed, in `<defs>/<kind>/<id>/<version>/definition.json` (`defs_root`:
  env ZEROFARM_DEFS_DIR, else ~/.zero-farm/definitions); its bytes are never re-serialised.
* `farm-release define <definition.json> --key <data key> --publish --url <url>` signs, uploads and registers it
  (`FarmRelease`; the executable: env ZEROFARM_FARM_RELEASE, else E:/cargo-target/zero-farm/main/release/
  farm-release.exe; a `.py` path runs with this Python - the tests' stub). The admin token goes to it in the
  environment (ZEROFARM_ADMIN_TOKEN), never on its command line, and is never printed.
* The data key: env ZEROFARM_DATA_KEY, else ~/.config/zero-farm/data.key. The admin token: env
  ZEROFARM_ADMIN_TOKEN, else the file ~/.config/zero-farm/admin_token.
* After publishing, the coordinator's answer must name this version as current (when it names a definition at all:
  a coordinator before rev 6 names none, and `farm-release define --publish` fails against it anyway).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = 'https://farm.greggg230.com'
FARM_RELEASE = 'E:/cargo-target/zero-farm/main/release/farm-release.exe'
SLUG_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,62}$')
BUILD_RE = re.compile(r'^[0-9a-f]{24}$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
GEN_RE = re.compile(r'^(0|[1-9][0-9]{0,8})$')
KINDS = ('run', 'campaign')
JOB_KINDS = ('selfplay', 'eval', 'fuzz')
MULTIPART = 90 << 20                    # bigger artifacts go up in parts (a Worker request's limit ~100 MB)
PART = 32 << 20


# ---------------------------------------------------------------- paths and secrets

def config_dir():
    return os.path.join(os.path.expanduser('~'), '.config', 'zero-farm')


def token_path():
    return os.path.join(config_dir(), 'admin_token')


def data_key_path():
    return os.environ.get('ZEROFARM_DATA_KEY') or os.path.join(config_dir(), 'data.key')


def farm_release_path():
    return os.environ.get('ZEROFARM_FARM_RELEASE') or FARM_RELEASE


def defs_root():
    return os.environ.get('ZEROFARM_DEFS_DIR') or os.path.join(os.path.expanduser('~'), '.zero-farm', 'definitions')


def read_token():
    """The admin token (env ZEROFARM_ADMIN_TOKEN, else ~/.config/zero-farm/admin_token); never printed."""
    t = os.environ.get('ZEROFARM_ADMIN_TOKEN')
    if t:
        return t.strip()
    p = token_path()
    try:
        with open(p, encoding='utf-8') as f:
            t = f.read().strip()
    except OSError:
        raise SystemExit(f'no admin token: {p} (or env ZEROFARM_ADMIN_TOKEN)') from None
    if not t:
        raise SystemExit(f'{p} is empty')
    return t


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- the coordinator's desktop API

class FarmError(RuntimeError):
    def __init__(self, status, code, message, body=None):
        super().__init__(f'{status} {code}: {message}')
        self.status, self.code, self.body = status, code, body or {}


class Client:
    """The coordinator's desktop API (bearer = the admin token, never printed). `transport(method, url, headers,
    data) -> (status, bytes)` (tests pass a fake coordinator); 5xx and network errors are retried with backoff."""

    def __init__(self, url=URL, token=None, transport=None, tries=6):
        self.url = url.rstrip('/')
        self._token = token if token is not None else read_token()
        self.transport = transport or self._urllib
        self.tries = tries

    def __repr__(self):
        return f'Client({self.url!r})'

    @property
    def token(self):
        return self._token

    @staticmethod
    def _urllib(method, url, headers, data):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def call(self, method, path, body=None, data=None, headers=None, raw=False):
        h = {'authorization': f'Bearer {self._token}', 'user-agent': 'zero-farm-desktop'}
        if body is not None:
            data = json.dumps(body).encode()
            h['content-type'] = 'application/json'
        h.update(headers or {})
        delay = 1.0
        for k in range(self.tries):
            try:
                status, out = self.transport(method, self.url + path, h, data)
            except (OSError, urllib.error.URLError) as e:
                status, out = None, str(e).encode()
            if status is not None and status < 500:
                break
            if k == self.tries - 1:
                raise FarmError(status or 0, 'unreachable', f'{method} {path}: {out[:300]!r}')
            time.sleep(delay)
            delay = min(delay * 2, 60)
        if status >= 400:
            try:
                e = json.loads(out)
            except ValueError:
                e = {'error': 'http', 'message': out[:300].decode('utf-8', 'replace')}
            raise FarmError(status, e.get('error'), e.get('message'), e)
        return out if raw else (json.loads(out) if out else None)

    def get_or_none(self, path):
        """GET `path`, None on 404."""
        try:
            return self.call('GET', path)
        except FarmError as e:
            if e.status == 404:
                return None
            raise

    def put_artifact(self, key, path, sha):
        size = os.path.getsize(path)
        if size <= MULTIPART:
            with open(path, 'rb') as f:
                return self.call('PUT', f'/v1/admin/artifact/{key}', data=f.read(),
                                 headers={'x-sha256': sha, 'content-type': 'application/octet-stream'})
        up = self.call('POST', f'/v1/admin/artifact/{key}?uploads', data=b'')['upload_id']
        parts = []
        with open(path, 'rb') as f:
            n = 1
            while True:
                blk = f.read(PART)
                if not blk:
                    break
                r = self.call('PUT', f'/v1/admin/artifact/{key}?upload_id={up}&part={n}', data=blk,
                              headers={'content-type': 'application/octet-stream'})
                parts.append({'part': n, 'etag': r['etag']})
                n += 1
        return self.call('POST', f'/v1/admin/artifact/{key}?upload_id={up}', body={'parts': parts})


# ---------------------------------------------------------------- definition/1 documents

def checkpoint_entry(model, reference=None):
    """A checkpoint's entry from its local files: the model's (and reference set's) sha256 and size."""
    out = {'model_sha256': file_sha256(model), 'model_size': os.path.getsize(model),
           'reference_sha256': None, 'reference_size': None}
    if reference:
        out.update(reference_sha256=file_sha256(reference), reference_size=os.path.getsize(reference))
    return out


# zero-farm's agent/common/src/definition.rs (farm-release define and the agent check exactly this): the keys a job
# template may have, per kind - never a per-lease field (gid_base, games, seed / first, count: the coordinator's) - and
# nothing path-like anywhere in it.
TEMPLATE_KEYS = {'selfplay': ('settings',), 'eval': ('settings', 'seats', 'paired'), 'fuzz': ('mode', 'env', 'announce_result')}
MAX_ID = 1 << 53
MAX_CHECKPOINT_FILE_BYTES = 1 << 30


def looks_like_path(s):
    """A string a worker could take for a file path (definition.rs `looks_like_path`): a separator, `..`, or a drive
    (`C:` at its start). Job templates carry none: paths on a job line are only ever the agent's own."""
    return '\\' in s or '/' in s or '..' in s or (len(s) >= 2 and s[0].isascii() and s[0].isalpha() and s[1] == ':')


def check_no_paths(v, at, depth=0):
    """Every string and object key in `v` free of path-like text (definition.rs `check_no_paths`)."""
    if depth > 32:
        raise ValueError(f'{at}: nested too deeply')
    if isinstance(v, str):
        if looks_like_path(v):
            raise ValueError(f'{at}: {v[:60]!r} looks like a file path (a job template carries none)')
    elif isinstance(v, list):
        for i, x in enumerate(v):
            check_no_paths(x, f'{at}[{i}]', depth + 1)
    elif isinstance(v, dict):
        for k, x in v.items():
            if looks_like_path(str(k)):
                raise ValueError(f'{at}: key {str(k)[:60]!r} looks like a file path')
            check_no_paths(x, f'{at}.{k}', depth + 1)


def _is_name(s, n, extra):
    return isinstance(s, str) and 0 < len(s) <= n and all((c.isascii() and c.isalnum()) or c in extra for c in s)


def seat_gens(seats):
    """An eval template's seats (2 to 16, each {"net": gen} or {"bot": a built-in bot's name}) -> the gens named."""
    if not isinstance(seats, list) or not 2 <= len(seats) <= 16:
        raise ValueError('seats: a list of 2 to 16 seats')
    gens = []
    for i, s in enumerate(seats):
        if not isinstance(s, dict) or ('net' in s) == ('bot' in s):
            raise ValueError(f'seat {i}: exactly one of net and bot')
        if 'net' in s:
            g = s['net']
            if isinstance(g, bool) or not isinstance(g, int) or g < 0:
                raise ValueError(f'seat {i}: net is a generation number')
            if g not in gens:
                gens.append(g)
        elif not _is_name(s['bot'], 64, '_.:-') or s['bot'].startswith('ext:'):
            raise ValueError(f'seat {i}: bot {s["bot"]!r} is not a built-in bot name')
    return gens


def _check_job(kind, job):
    """A job template (definition.rs `check_template`): only its kind's keys, the right types, no path-like text."""
    if job is None:
        return
    if not isinstance(job, dict):
        raise ValueError(f'jobs.{kind}: an object or null')
    extra = sorted(set(job) - set(TEMPLATE_KEYS[kind]))
    if extra:
        raise ValueError(f'jobs.{kind}: {extra} not allowed (a {kind} template has only {TEMPLATE_KEYS[kind]}; the '
                         f'per-lease fields are the coordinator\'s)')
    if kind in ('selfplay', 'eval') and not isinstance(job.get('settings'), dict):
        raise ValueError(f'jobs.{kind}: settings (an object)')
    if kind == 'eval':
        seat_gens(job.get('seats'))
        if not isinstance(job.get('paired'), bool):
            raise ValueError('jobs.eval: paired (true or false)')
    if kind == 'fuzz':
        if not _is_name(job.get('mode'), 128, '_.-'):
            raise ValueError(f'jobs.fuzz.mode {job.get("mode")!r}: a mode name')
        env = job.get('env', {})
        if not isinstance(env, dict):
            raise ValueError('jobs.fuzz.env: an object')
        for k, v in env.items():
            if not _is_name(k, 64, '_') or not isinstance(v, str) or len(v.encode()) > 4096:
                raise ValueError(f'jobs.fuzz.env: {k!r}={v!r}')
        if 'announce_result' in job and not isinstance(job['announce_result'], bool):
            raise ValueError('jobs.fuzz.announce_result: true or false')
    check_no_paths(job, f'jobs.{kind}')


def validate(doc):
    """A definition/1 document's checks (what an agent and the coordinator refuse) -> doc."""
    if not isinstance(doc, dict) or doc.get('definition') != 1:
        raise ValueError('definition: 1')
    if doc.get('kind') not in KINDS:
        raise ValueError(f'kind {doc.get("kind")!r}: one of {KINDS}')
    if not isinstance(doc.get('id'), str) or not SLUG_RE.match(doc['id']):
        raise ValueError(f'id {doc.get("id")!r}: [a-z0-9][a-z0-9-]{{0,62}}')
    v = doc.get('version')
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_ID:
        raise ValueError(f'version {v!r}: an integer >= 1')
    c = doc.get('created')
    if isinstance(c, bool) or not isinstance(c, int) or c <= 0:
        raise ValueError('created: integer Unix seconds')
    e = doc.get('expires')
    if e is not None and (isinstance(e, bool) or not isinstance(e, int) or e <= doc['created']):
        raise ValueError(f'expires {e!r}: null or integer Unix seconds after created')
    if not isinstance(doc.get('game'), str) or not SLUG_RE.match(doc['game']):
        raise ValueError(f'game {doc.get("game")!r}: a slug - the worker builds\' game ("mtg"), not a run\'s settings '
                         f'game ("mtg_eoe")')
    b = doc.get('builds')
    if (not isinstance(b, list) or not 1 <= len(b) <= 64 or len(set(b)) != len(b)
            or not all(isinstance(x, str) and BUILD_RE.match(x) for x in b)):
        raise ValueError(f'builds {b!r}: 1 to 64 distinct worker build_ids (24 hex)')
    rt = doc.get('runtime')
    if rt is not None and not (isinstance(rt, str) and BUILD_RE.match(rt)):
        raise ValueError(f'runtime {rt!r}: a runtime build_id or null')
    cks = doc.get('checkpoints')
    if not isinstance(cks, dict):
        raise ValueError('checkpoints: an object')
    for g, c in cks.items():
        if not GEN_RE.match(g) or not isinstance(c, dict):
            raise ValueError(f'checkpoints[{g!r}]')
        if set(c) != {'model_sha256', 'model_size', 'reference_sha256', 'reference_size'}:
            raise ValueError(f'checkpoints[{g}]: model_sha256, model_size, reference_sha256, reference_size')
        size_ok = lambda n: not isinstance(n, bool) and isinstance(n, int) and 0 < n <= MAX_CHECKPOINT_FILE_BYTES
        if not SHA_RE.match(str(c['model_sha256'])) or not size_ok(c['model_size']):
            raise ValueError(f'checkpoints[{g}]: the model\'s sha256 and size (1 B to 1 GiB)')
        if (c['reference_sha256'] is None) != (c['reference_size'] is None):
            raise ValueError(f'checkpoints[{g}]: the reference set\'s sha256 and size, both or neither')
        if c['reference_sha256'] is not None and (not SHA_RE.match(str(c['reference_sha256'])) or not size_ok(c['reference_size'])):
            raise ValueError(f'checkpoints[{g}]: the reference set\'s sha256 and size (1 B to 1 GiB)')
    jobs = doc.get('jobs')
    if not isinstance(jobs, dict) or set(jobs) != set(JOB_KINDS):
        raise ValueError(f'jobs: exactly {JOB_KINDS} (null when unused)')
    for k in JOB_KINDS:
        _check_job(k, jobs[k])
    on = [k for k in JOB_KINDS if jobs[k] is not None]
    if not on:
        raise ValueError('jobs: at least one job kind')
    if doc['kind'] == 'campaign' and (on != ['fuzz'] or cks):
        raise ValueError('a campaign\'s definition: jobs.fuzz only, no checkpoints')
    if doc['kind'] == 'run' and 'fuzz' in on:
        raise ValueError('a run\'s definition: no fuzz job')
    if jobs['eval'] is not None:
        nets = {str(g) for g in seat_gens(jobs['eval']['seats'])}
        if not nets <= set(cks):
            raise ValueError(f'jobs.eval names gens {sorted(nets - set(cks))} the definition has no checkpoint for')
    return doc


def definition(kind, id, version, game, builds, runtime=None, checkpoints=None, jobs=None, created=None, expires=None):
    """A definition/1 document (its keys in the protocol's order), checked (`validate`)."""
    jobs = dict(jobs or {})
    unknown = set(jobs) - set(JOB_KINDS)
    if unknown:
        raise ValueError(f'jobs: unknown kinds {sorted(unknown)}')
    created = int(time.time()) if created is None else int(created)
    doc = {'definition': 1, 'kind': kind, 'id': id, 'version': int(version), 'created': created,
           'expires': None if expires is None else int(expires), 'game': game, 'builds': list(builds),
           'runtime': runtime, 'checkpoints': {str(g): dict(c) for g, c in sorted((checkpoints or {}).items(), key=lambda x: int(x[0]))},
           'jobs': {k: jobs.get(k) for k in JOB_KINDS}}
    return validate(doc)


def expires_at(created, days):
    """`expires` for a definition made at `created` that lasts `days` (None or 0: never)."""
    return None if not days else int(created) + int(round(float(days) * 86400))


def encode(doc):
    """The bytes of definition.json (written once, signed as they are; never re-serialised)."""
    return (json.dumps(doc, indent=1, ensure_ascii=True) + '\n').encode('ascii')


CONTENT = ('game', 'builds', 'runtime', 'checkpoints', 'jobs')


def same_content(a, b):
    """Two definitions allow the same builds, runtime, checkpoints and jobs (version and times aside)."""
    return a is not None and b is not None and all(a.get(k) == b.get(k) for k in CONTENT)


def r2_key(kind, id, version):
    return f'defs/{kind}/{id}/{int(version)}/definition.json'


def build_manifest(client, build_id):
    """A registered build's signed manifest (`GET /v1/admin/artifact/builds/<id>/manifest.json`), checked against its
    id (sha256[:24] of the bytes: the coordinator cannot answer another) -> the manifest dict."""
    if not BUILD_RE.match(build_id or ''):
        raise SystemExit(f'build {build_id!r}: a build_id (24 hex)')
    b = client.call('GET', f'/v1/admin/artifact/builds/{build_id}/manifest.json', raw=True)
    if hashlib.sha256(b).hexdigest()[:24] != build_id:
        raise SystemExit(f'build {build_id}: the coordinator\'s manifest.json is not that build\'s (sha256)')
    return json.loads(b)


def worker_game(client, build_id, runtime):
    """The game (repo) of a worker build - a definition's `game` (PROTOCOL.md 7: "mtg" for any MTG Zero run, as the
    build's manifest says, not a run's settings game such as "mtg_eoe") - checked: a worker build, and `runtime` the
    runtime pack it requires (when it requires one)."""
    m = build_manifest(client, build_id)
    if m.get('kind') != 'worker':
        raise SystemExit(f'build {build_id} is a {m.get("kind")!r} build, not a worker')
    need = m.get('requires_runtime')
    if need and need != runtime:
        raise SystemExit(f'worker build {build_id} requires runtime {need}, not {runtime}')
    if not m.get('game'):
        raise SystemExit(f'worker build {build_id}: its manifest names no game')
    return m['game']


# ---------------------------------------------------------------- versions

def current_of(answer):
    """The current definition a coordinator answer (`GET /v1/admin/run|campaign/{id}`) names: {"version", "sha256"?,
    "key"?} or None (none, or a coordinator before rev 6)."""
    if not isinstance(answer, dict):
        return None
    d = answer.get('definition')
    if isinstance(d, dict) and isinstance(d.get('version'), int):
        return d
    v = answer.get('definition_version')
    if isinstance(v, int) and not isinstance(v, bool):
        return {'version': v, 'sha256': answer.get('definition_sha256')}
    return None


def admin_path(kind, id):
    return f'/v1/admin/{kind}/{id}'


def remote_current(client, kind, id):
    """(the coordinator's current definition or None, its answer or None when the run / campaign doesn't exist)."""
    ans = client.get_or_none(admin_path(kind, id))
    return current_of(ans), ans


def local_dir(kind, id, root=None):
    return os.path.join(root or defs_root(), kind, id)


def local_versions(kind, id, root=None):
    d = local_dir(kind, id, root)
    try:
        names = os.listdir(d)
    except OSError:
        return []
    return sorted(int(n) for n in names if GEN_RE.match(n) and os.path.exists(os.path.join(d, n, 'definition.json')))


def load_local(kind, id, version=None, root=None):
    """A kept definition (the newest without `version`) -> (doc, path) or (None, None)."""
    vs = local_versions(kind, id, root)
    if version is None:
        if not vs:
            return None, None
        version = vs[-1]
    p = os.path.join(local_dir(kind, id, root), str(int(version)), 'definition.json')
    if not os.path.exists(p):
        return None, None
    with open(p, 'rb') as f:
        return json.loads(f.read()), p


def next_version(client, kind, id, root=None):
    """The next version of (kind, id): the larger of the coordinator's current and the newest kept here, plus 1."""
    cur, _ = remote_current(client, kind, id)
    remote = cur['version'] if cur else 0
    local = max(local_versions(kind, id, root), default=0)
    return max(remote, local) + 1


def write(doc, root=None):
    """Keep `doc` as <defs>/<kind>/<id>/<version>/definition.json (never overwritten with other bytes) -> its path."""
    validate(doc)
    d = os.path.join(local_dir(doc['kind'], doc['id'], root), str(doc['version']))
    p = os.path.join(d, 'definition.json')
    b = encode(doc)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            if f.read() != b:
                raise SystemExit(f'{p} exists with other bytes: version {doc["version"]} of {doc["kind"]} {doc["id"]} '
                                 f'was made before (a version is never reused)')
        return p
    os.makedirs(d, exist_ok=True)
    tmp = p + f'.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(b)
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------- signing and publishing

def exe_argv(path, *args):
    """The argv that runs the executable `path` (a `.py` path: with this Python - the tests' stand-ins)."""
    if path.lower().endswith('.py'):
        return [sys.executable, path, *args]
    return [path, *args]


def define_result(out):
    """farm-release define's result: its one JSON line ({kind, id, version, key, sha256, signature, published}), the
    last line of `out` that is a JSON object with a version - or None."""
    for ln in reversed((out or '').splitlines()):
        ln = ln.strip()
        if ln.startswith('{'):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if isinstance(d, dict) and 'version' in d:
                return d
    return None


class FarmRelease:
    """zero-farm's `farm-release` (the desktop's signing tool). `path`: the executable (default
    `farm_release_path()`); a `.py` path runs with this Python (a stand-in, e.g. the tests' stub)."""

    def __init__(self, path=None):
        self.path = path or farm_release_path()

    def argv(self, *args):
        return exe_argv(self.path, *args)

    def define(self, path, key, url=None, token=None, publish=True):
        """`farm-release define <path> --key <key> [--publish --url <url>]`, the admin token in the environment
        (ZEROFARM_ADMIN_TOKEN), never in the arguments -> its output (the token scrubbed)."""
        if not os.path.exists(self.path):
            raise SystemExit(f'farm-release not found: {self.path} (--farm-release, or env ZEROFARM_FARM_RELEASE)')
        if not os.path.exists(key):
            raise SystemExit(f'the data key not found: {key} (--key, or env ZEROFARM_DATA_KEY)')
        args = ['define', os.path.abspath(path), '--key', key]
        env = dict(os.environ)
        env.pop('ZEROFARM_ADMIN_TOKEN', None)
        if publish:
            if not url or not token:
                raise ValueError('define --publish: the url and the admin token')
            args += ['--publish', '--url', url]
            env['ZEROFARM_ADMIN_TOKEN'] = token
        p = subprocess.run(self.argv(*args), capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        out = (p.stdout or '') + (p.stderr or '')
        if token:
            out = out.replace(token, '<admin token>')
        if p.returncode != 0:
            raise SystemExit(f'farm-release define {path} failed (exit {p.returncode}):\n'
                             f'{out.strip()}')
        return out.strip()


def prepare(client, kind, id, game, builds, runtime=None, checkpoints=None, jobs=None, expires_days=None, root=None,
            now=None):
    """A new version of (kind, id)'s definition, written to the local archive -> {"kind", "id", "version", "path",
    "sha256", "key", "created", "expires"}."""
    version = next_version(client, kind, id, root)
    created = int(time.time()) if now is None else int(now)
    doc = definition(kind, id, version, game, builds, runtime, checkpoints, jobs, created, expires_at(created, expires_days))
    p = write(doc, root)
    return {'kind': kind, 'id': id, 'version': version, 'path': p, 'sha256': file_sha256(p), 'key': r2_key(kind, id, version),
            'created': created, 'expires': doc['expires']}


def publish(client, info, key=None, farm_release=None, log=None):
    """Sign, upload and register the prepared definition `info` (`prepare`) with `farm-release define --publish`;
    then the coordinator must name it as current (when it names a definition) -> info with "published"."""
    log = log or (lambda m: print('# ' + m, file=sys.stderr, flush=True))
    fr = farm_release if isinstance(farm_release, FarmRelease) else FarmRelease(farm_release)
    out = fr.define(info['path'], key or data_key_path(), client.url, client.token, publish=True)
    res = define_result(out)
    want = {'kind': info['kind'], 'id': info['id'], 'version': info['version'], 'key': info['key'], 'sha256': info['sha256'],
            'published': True}
    if res is None or any(res.get(k) != v for k, v in want.items()):
        raise SystemExit(f'farm-release define: its result {res} is not {want} (its output: {out[-500:]!r})')
    if not os.path.exists(info['path'] + '.minisig'):
        raise SystemExit(f'farm-release define left no signature next to {info["path"]}')
    cur, ans = remote_current(client, info['kind'], info['id'])
    if cur is None:
        log(f'note: the coordinator\'s answer for {info["kind"]} {info["id"]} names no definition (a coordinator before rev 6?)')
    elif cur['version'] != info['version'] or (cur.get('sha256') and cur['sha256'] != info['sha256']):
        raise SystemExit(f'{info["kind"]} {info["id"]}: the coordinator\'s current definition is version {cur["version"]} '
                         f'(sha256 {str(cur.get("sha256"))[:16]}), not the one just published (version {info["version"]}, '
                         f'{info["sha256"][:16]})')
    log(f'definition {info["kind"]}/{info["id"]} version {info["version"]} signed and published ({info["sha256"][:16]}, '
        f'expires {"never" if info["expires"] is None else time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(info["expires"]))})')
    mark = {'url': client.url, 'version': info['version'], 'sha256': info['sha256'], 't': int(time.time())}
    with open(published_path(info['path']), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(mark, f)
    return dict(info, published=True)


def published_path(path):
    """The marker `publish` leaves next to a definition it published (url, version, sha256, t)."""
    return os.path.join(os.path.dirname(path), 'published.json')


def published_to(path, url):
    """Was the kept definition at `path` published to the coordinator at `url` (by `publish`, these bytes)?"""
    try:
        with open(published_path(path), encoding='utf-8') as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    return m.get('url') == url.rstrip('/') and m.get('sha256') == file_sha256(path)
