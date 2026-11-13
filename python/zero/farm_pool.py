"""The farm pool (zero-farm DESIGN.md 7, docs/az.md 45): a run's self-play also on friends' idle PCs, through the
zero-farm coordinator - `selfplay_remote` names `pool:<farm run>` (alone or in a list with remote hosts).

Unlike a remote host it has no sessions and never holds a share of the iteration's quota: it is one thread of the
run process (`Pool`, owned by zero.remoteplay.RemotePlay) that

1. **publishes** checkpoint it - 1 when iteration it's self-play begins (RemotePlay.begin, right after the
   checkpoint is written): the checkpoint exported to zero-onnx/1 with its reference set (zero.onnx_export, checked
   through onnxruntime) under `<run>/pool/<farm run>/ckpt/<gen>/`, uploaded (`ckpt/<farm run>/<gen>/...`), a NEW
   signed definition version (zero.farm_def: the worker build, its runtime, the gen and up to `pool_max_stale` gens
   before it, the self-play template {settings: the worker build's own `check-config` play settings}) published
   with farm-release - and only then the gen registered current (`PUT .../checkpoint/<gen>`): farm leases switch
   only to gens a definition names. The run on the coordinator is created on the first publish (`selfplay`:
   quota `pool_quota` (default the run's `games`), slack, max_stale, games per lease, needs).
2. **ingests** records (zero-farm PROTOCOL.md 5 `records`, zero-core docs/search.md 2.17): it polls
   `GET .../results`, downloads each self-play chunk, checks its header (the run's game, features, actions, spec,
   auxiliary targets, base seed and worker build), replays every credited game through the engine
   (`<pkg>._engine.replay_rows`: the rows bit for bit as the worker's shards would hold them; replay is the
   validation: an illegal move, a wrong decider or option count, another result, margin or setup - and a game seed
   that is not its id's - refuses the whole record) and keeps the game, its rows compacted, until the quota takes
   it. A refused record is counted against its machine (`<run>/debug/pool/refused.jsonl`, the chunk kept under
   `<run>/debug/pool/<machine>/`).
3. **counts** accepted games toward the iteration's quota exactly as a remote worker's (RemotePlay.accept_pool:
   oldest first, as many as the quota still needs; the rest held for the next iteration), writes their rows as
   ordinary shards into `data/iter_i/` (`pool-<machine>_SSS.npz`, meta source 'pool', machine), and tells the
   coordinator its running total for the generation - accepted plus held - (`POST .../accept`) before it acks the
   results (`POST .../ack`).
4. **staleness**: a game whose oldest row was searched with network `net` is accepted into iteration i only while
   net >= i - 1 - `pool_max_stale` (default 1); older ones are dropped (counted). Every accepted row goes into the
   iteration's staleness histogram as a remote's do.

The run never waits for the pool: RemotePlay.end closes the pool's shards of the iteration with whatever it accepted.

Settings (zero.run DEFAULTS, docs/az.md 45): `pool_build` (the worker build id, required), `pool_runtime` (its runtime
build), `pool_worker` (that build's signed executable on this machine: its `check-config` picks the template's
settings), `pool_url` (default zero.farm_def.URL), `pool_max_stale`, `pool_quota`, `pool_slack`,
`pool_games_per_lease`, `pool_priority`, `pool_needs_ram_mb`, `pool_poll_s`, `pool_key` (the data key; default
zero.farm_def's), `pool_farm_release` (farm-release; default zero.farm_def's), `pool_expires_days`, `pool_min_agent`
(the farm run's `min_agent`, zero-farm PROTOCOL rev 9: its leases go only to agents at or above it; default 0.2.8, the
first agent that accepts a template naming a split or double-faced card; '' = none). The admin token as
the farm tools read it (zero.farm_def.read_token). State: `<run>/pool/<farm run>/state.json`.
"""
import gzip
import hashlib
import json
import os
import re
import threading
import time
import traceback
from collections import deque

from . import farm_def as FD

PREFIX = 'pool:'
# (as the coordinator checks a version: zero-farm coordinator util.ts)
SEMVER_RE = re.compile(r'^\d{1,6}\.\d{1,6}\.\d{1,6}(?:-[0-9A-Za-z.-]{1,64})?$')
PAGE = 200                          # results per GET


def is_pool(name):
    return str(name).startswith(PREFIX)


def coordinator_seed(run):
    """The farm run's base seed (zero-farm book.ts `baseSeed` without a job seed: 32-bit FNV-1a of its name)."""
    h = 0x811c9dc5
    for ch in run:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xffffffff
    return h


def read_records(body):
    """A records file's bytes (gzip'd JSON lines) -> (header, [game record text, ...])."""
    lines = gzip.decompress(body).decode('utf-8').splitlines()
    if not lines:
        raise ValueError('an empty records file')
    return json.loads(lines[0]), [ln for ln in lines[1:] if ln.strip()]


def _spec_value(s):
    return json.loads(s) if s else {}


def _in_ranges(g, ranges):
    return any(a <= g < b for a, b in ranges)


def play_settings(worker, build_id, cfg, d):
    """The self-play template's settings: the keys the worker build's own `check-config --json` calls `play` (what
    its self-play driver reads; never a trainer-only key or a path, which farm-release refuses in a template),
    checked again on their own, and giving the driver every setting gives (`driver-cfg`)."""
    from .farm_h2h import worker_json
    rc, info, text = worker_json(worker, 'info', '--ort-dll', 'none')
    if rc != 0 or not isinstance(info, dict):
        raise SystemExit(f'{worker} info: {text.strip()[-500:]}')
    if info.get('build_id') != build_id or 'selfplay' not in (info.get('kinds') or []):
        raise SystemExit(f'{worker} is build {info.get("build_id")} ({info.get("kinds")}), not the self-play worker build '
                         f'{build_id}: pool_worker must be that build\'s executable')
    os.makedirs(d, exist_ok=True)
    full = os.path.join(d, 'settings-run.json')
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    rc, out, text = worker_json(worker, 'check-config', full, '--json')
    if rc != 0 or not isinstance(out, dict) or out.get('failures'):
        raise SystemExit(f'the worker refuses the run\'s settings for self-play:\n{text.strip()[-3000:]}')
    status = {s['key']: s['status'] for s in out.get('settings') or []}
    play = {k: v for k, v in cfg.items() if status.get(k) == 'play'}
    path = os.path.join(d, 'settings.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(play, f)
    rc, out, text = worker_json(worker, 'check-config', path, '--json')
    if rc != 0 or not isinstance(out, dict) or out.get('failures') or \
            any(s['status'] != 'play' for s in out.get('settings') or []):
        raise SystemExit(f'the worker refuses the self-play job\'s own settings:\n{text.strip()[-3000:]}')
    a = worker_json(worker, 'driver-cfg', full, '--seed', '1', '--net', '0')
    b = worker_json(worker, 'driver-cfg', path, '--seed', '1', '--net', '0')
    if a[0] != 0 or b[0] != 0 or a[2] != b[2]:
        raise SystemExit('the play settings give another self-play driver than the run\'s settings:\n'
                         f'{a[2][-1500:]}\n---\n{b[2][-1500:]}')
    return play


class Pool:
    """The farm pool `name` ('pool:<farm run>') of RemotePlay `rp` (module docstring). `client`: a zero.farm_def
    Client (tests: a fake coordinator); `exporter(ckpt, gen, out_dir) -> {model, sha256, model_size, reference,
    reference_sha256, reference_size}` and `definer(client, info) -> info` (tests stand in for the ONNX export and
    farm-release)."""

    def __init__(self, rp, name, client=None, exporter=None, definer=None, settings_of=None):
        from .run import DEFAULTS
        self.rp, self.R, self.name = rp, rp.R, name
        self.farm = name[len(PREFIX):]
        if not FD.SLUG_RE.match(self.farm or ''):
            raise SystemExit(f'selfplay_remote {name!r}: the farm run must be a slug ([a-z0-9-], 1-63)')
        c = self.R.cfg

        def setting(k):
            v = c.get(k)
            return DEFAULTS.get(k) if v is None else v
        self.build = str(setting('pool_build') or '')
        if not self.build:
            raise SystemExit(f'selfplay_remote {name!r}: pool_build (the farm worker build) is not set')
        self.runtime = str(setting('pool_runtime') or '') or None
        self.worker = str(setting('pool_worker') or '')
        self.url = str(setting('pool_url') or FD.URL).rstrip('/')
        self.max_stale = int(setting('pool_max_stale'))
        self.quota = int(setting('pool_quota') or c['games'])
        self.slack = float(setting('pool_slack'))
        self.per_lease = setting('pool_games_per_lease')
        self.per_lease = 'auto' if str(self.per_lease) in ('', 'auto', '0') else int(self.per_lease)
        self.priority = float(setting('pool_priority'))
        self.needs_ram = int(setting('pool_needs_ram_mb') or 0)
        self.poll_s = float(setting('pool_poll_s'))
        self.live_s = float(setting('pool_live_s'))
        self.key = str(setting('pool_key') or '') or None
        self.farm_release = str(setting('pool_farm_release') or '') or None
        self.expires_days = setting('pool_expires_days')
        self.min_agent = str(setting('pool_min_agent') or '').strip() or None
        if self.min_agent and not SEMVER_RE.match(self.min_agent):
            raise SystemExit(f'selfplay_remote {name!r}: pool_min_agent {self.min_agent!r} is not a version like 0.2.8')
        self.client = client or FD.Client(self.url)
        self.exporter = exporter or self._export
        self.definer = definer or self._define
        self.settings_of = settings_of or (lambda: play_settings(self.worker, self.build, dict(c), os.path.join(self.dir, 'settings')))
        self.dir = os.path.join(self.R.run, 'pool', self.farm)
        self.debug = os.path.join(self.R.run, 'debug', 'pool')
        os.makedirs(self.dir, exist_ok=True)
        self.st = self._load()
        self.seed = coordinator_seed(self.farm)
        from . import games as G
        self.spec = G.spec_json(c['spec'])
        self.players = [int(p) for p in c['players']]
        self.aux = bool(getattr(self.R, 'aux', None))
        self.mu = threading.Lock()                 # the held games, the iteration's writers, the counters
        self.held = deque()                        # games replayed, waiting for a quota: dicts (below)
        self.writers = {}                          # (iteration, machine) -> ShardWriter
        self.cur = self._counters(None)
        self.want = None                           # the iteration whose checkpoint (it - 1) should be published
        self.t_game = None                         # when the last valid game arrived (`live`)
        self.closed = False
        self.error = None
        self.wake = threading.Event()
        self.th = threading.Thread(target=self._loop, name=f'zero-pool-{self.farm}', daemon=True)
        self.th.start()

    # ------------------------------------------------------------ state
    def _load(self):
        p = os.path.join(self.dir, 'state.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        return {'farm_run': self.farm, 'url': self.url, 'cursor': None, 'gens': {}, 'current_gen': None,
                'run_put': False, 'selfplay_put': False, 'definition': None, 'refused': {}, 'accepted_total': 0}

    def _save(self):
        p = os.path.join(self.dir, 'state.json')
        tmp = p + f'.{os.getpid()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.st, f, indent=1)
        os.replace(tmp, p)

    def log(self, msg):
        self.R.log(f'farm pool {self.farm}: {msg}')

    @staticmethod
    def _counters(it):
        return {'it': it, 'games': 0, 'rows': 0, 'shards': 0, 'records': 0, 'chunks': 0, 'refused': 0,
                'stale_dropped': 0, 'machines': {}, 'stale': {}, 'replay_s': 0.0}

    def live(self):
        """The farm is delivering: a valid game arrived within `pool_live_s` (selfplay_local 0: this machine plays no
        games meanwhile - RemotePlay._refresh; once the farm goes quiet for that long, it plays again)."""
        t = self.t_game
        return t is not None and time.time() - t < self.live_s

    def describe(self):
        return f'{self.name!r} (farm {self.url}, build {self.build}, max_stale {self.max_stale})'

    # ------------------------------------------------------------ the run's side (RemotePlay)
    def begin(self, it):
        """Iteration `it`'s self-play begins: checkpoint it - 1 is published on the pool's thread (never waited for)."""
        self.want = it
        self.wake.set()

    def end_phase(self, it):
        """Iteration `it`'s self-play ended (RemotePlay.end): its pool shards are closed -> its share (metrics)."""
        with self.mu:
            ws = [(k, w) for k, w in self.writers.items() if k[0] == it]
            for k, _ in ws:
                del self.writers[k]
            for _, w in ws:
                w.close()
            cur = self.cur if self.cur['it'] == it else self._counters(it)
            self.cur = self._counters(None)
            out = {'farm_run': self.farm, 'games': cur['games'], 'samples': cur['rows'], 'shards': cur['shards'],
                   'records': cur['records'], 'chunks': cur['chunks'], 'refused': cur['refused'],
                   'stale_dropped': cur['stale_dropped'], 'held': len(self.held), 'machines': dict(cur['machines']),
                   'replay_s': round(cur['replay_s'], 2), 'gen': self.st.get('current_gen'), 'stale': dict(cur['stale'])}
        if self.error:
            out['error'] = self.error
        return out

    def progress(self):
        with self.mu:
            return {'farm_run': self.farm, 'games': self.cur['games'], 'held': len(self.held),
                    'refused': self.cur['refused'], 'gen': self.st.get('current_gen'),
                    'error': self.error}

    def close(self):
        self.closed = True
        self.wake.set()
        with self.mu:
            for w in self.writers.values():
                w.close()
            self.writers.clear()

    # ------------------------------------------------------------ the thread
    def _loop(self):
        while not self.closed:
            try:
                if self.want is not None:
                    self._publish(self.want - 1)
                self._ingest()
                self.error = None
            except Exception as e:  # noqa: BLE001 - logged; the run never stops for the pool
                self.error = f'{type(e).__name__}: {str(e)[-300:]}'
                self.log(f'error (retried in {self.poll_s:g} s): {self.error}\n{traceback.format_exc()[-1500:]}')
            except SystemExit as e:             # (zero.farm_def's refusals)
                self.error = f'{str(e)[-500:]}'
                self.log(f'refused (retried in {self.poll_s:g} s): {self.error}')
            self.wake.wait(self.poll_s)
            self.wake.clear()

    # ---- publishing
    def _export(self, ckpt, gen, out_dir):
        from . import games as G
        from .farm_h2h import export, record_rows
        from .onnx_export import load_checkpoint
        m, _ = load_checkpoint(ckpt, self.R.cfg['game'])
        rows = record_rows(m, self.R.cfg['game'], G.spec_json(self.R.cfg['spec']))
        return export(ckpt, gen, out_dir, self.R.cfg['game'], [rows], f'{self.farm}/{gen}', checkpoint_iter=gen)

    def _define(self, client, info):
        return FD.publish(client, info, key=self.key, farm_release=self.farm_release, log=self.log)

    def _selfplay_body(self, play):
        sp = {'quota': self.quota, 'slack': self.slack, 'max_stale': self.max_stale, 'games_per_lease': self.per_lease,
              'gpu': 'any', 'job': {}}
        if self.needs_ram:
            sp['needs'] = {'ram_mb': self.needs_ram}
        return {'game': self.R.cfg['game'], 'build_id': self.build, 'runtime': self.runtime, 'priority': self.priority,
                'min_agent': self.min_agent, 'selfplay': sp, 'eval': None, 'state': 'active'}

    def _publish(self, gen):
        """Gen `gen` (checkpoint `gen`) exported, uploaded, signed into a new definition version and registered
        current - each step once (state.json)."""
        from .run import model_path
        if gen < 0 or str(gen) in self.st['gens'] and self.st['gens'][str(gen)].get('registered'):
            return
        st = self.st
        if st.get('play') is None:
            st['play'] = self.settings_of()
            self._save()
        if not st.get('build_game'):
            st['build_game'] = FD.worker_game(self.client, self.build, self.runtime) if self.runtime else self.R.cfg['game']
            self._save()
        if not st['run_put']:
            body = self._selfplay_body(st['play'])
            body['selfplay'] = None
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=body)
            st['run_put'] = True
            self._save()
            self.log(f'farm run {self.farm} created (worker build {self.build}, runtime {self.runtime}, '
                     f'min_agent {self.min_agent or "none"})')
        g = st['gens'].setdefault(str(gen), {})
        if not g.get('model'):
            ckpt = model_path(self.R.run, gen)
            t0 = time.time()
            ex = self.exporter(ckpt, gen, os.path.join(self.dir, 'ckpt', str(gen)))
            g.update({k: ex[k] for k in ('model', 'sha256', 'model_size', 'reference', 'reference_sha256', 'reference_size')})
            g['export_s'] = round(time.time() - t0, 1)
            self._save()
            self.log(f'gen {gen} exported ({g["export_s"]} s, {g["sha256"][:12]})')
        keys = {'model': f'ckpt/{self.farm}/{gen}/model.onnx', 'reference': f'ckpt/{self.farm}/{gen}/reference.npz'}
        if not g.get('uploaded'):
            self.client.put_artifact(keys['model'], g['model'], g['sha256'])
            self.client.put_artifact(keys['reference'], g['reference'], g['reference_sha256'])
            g['uploaded'] = True
            self._save()
        d = g.get('definition')
        if not d or not d.get('published'):
            if not d:
                gens = sorted((int(k) for k, v in st['gens'].items() if v.get('sha256')), reverse=True)
                keep = [x for x in gens if gen - self.max_stale <= x <= gen]
                cks = {str(x): {'model_sha256': st['gens'][str(x)]['sha256'], 'model_size': st['gens'][str(x)]['model_size'],
                                'reference_sha256': st['gens'][str(x)]['reference_sha256'],
                                'reference_size': st['gens'][str(x)]['reference_size']} for x in keep}
                d = FD.prepare(self.client, 'run', self.farm, game=st['build_game'], builds=[self.build], runtime=self.runtime,
                               checkpoints=cks, jobs={'selfplay': {'settings': st['play']}, 'eval': None, 'fuzz': None},
                               expires_days=self.expires_days, root=os.path.join(self.dir, 'definitions'))
                g['definition'] = d
                self._save()
            d = self.definer(self.client, d)
            g['definition'] = d
            st['definition'] = {k: d[k] for k in ('version', 'sha256', 'key')}
            self._save()
            self.log(f'definition version {d["version"]} signs gen {gen}')
        self.client.call('PUT', f'/v1/admin/run/{self.farm}/checkpoint/{gen}',
                         body={'model': keys['model'], 'sha256': g['sha256'], 'reference': keys['reference'],
                               'reference_sha256': g['reference_sha256']})
        g['registered'] = True
        st['current_gen'] = gen
        if not st['selfplay_put']:
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=self._selfplay_body(st['play']))
            st['selfplay_put'] = True
        self._save()
        self.log(f'gen {gen} registered current: farm self-play plays it')
        with self.mu:
            # (the new generation's count starts: games accepted from now on; the held ones count as held)
            st['accepted_total'] = 0

    # ---- ingesting
    def _ingest(self):
        if self.st.get('current_gen') is None:
            return
        q = f'/v1/admin/run/{self.farm}/results?limit={PAGE}' + (f'&after={self.st["cursor"]}' if self.st['cursor'] else '')
        ans = self.client.call('GET', q)
        res = ans.get('results') or []
        for r in res:
            if r.get('kind') != 'selfplay':
                continue
            self._chunk(r)
        self._drain()
        gen = self.st['current_gen']
        with self.mu:
            total = self.st.get('accepted_total', 0) + len(self.held)
        if res or self.st.get('accept_sent') != [gen, total]:
            # the running total for the generation, held games included, before the ack (PROTOCOL rev 2)
            self.client.call('POST', f'/v1/admin/run/{self.farm}/accept', body={'gen': gen, 'games': total})
            self.st['accept_sent'] = [gen, total]
        if res and ans.get('cursor') is not None:
            self.client.call('POST', f'/v1/admin/run/{self.farm}/ack', body={'cursor': ans['cursor']})
            self.st['cursor'] = ans['cursor']
        self._save()
        if len(res) >= PAGE:
            self.wake.set()                        # more pages waiting

    def _refuse(self, r, why, body=None, gid=None):
        m = r.get('machine_id') or '?'
        self.st['refused'][m] = self.st['refused'].get(m, 0) + 1
        with self.mu:
            self.cur['refused'] += 1
        os.makedirs(self.debug, exist_ok=True)
        if body is not None:
            d = os.path.join(self.debug, m)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, hashlib.sha256(r['key'].encode()).hexdigest()[:24] + '.rec.jsonl.gz')
            if not os.path.exists(p):
                with open(p, 'wb') as f:
                    f.write(body)
        with open(os.path.join(self.debug, 'refused.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': time.time(), 'machine': m, 'key': r.get('key'), 'lease': r.get('lease_id'),
                                'seq': r.get('seq'), 'gid': gid, 'why': why[:2000]}) + '\n')
        self.log(f'record REFUSED (machine {m}, {r.get("key")}{"" if gid is None else f", game {gid}"}): {why[:300]}')

    def _chunk(self, r):
        """One self-play result chunk: its credited games replayed and held (module docstring)."""
        from .data import compact, compact_sparse, shard_format
        from .finite import shard_problems
        E = self._engine()
        body = self.client.call('GET', f'/v1/admin/artifact/{r["key"]}', raw=True)
        if r.get('sha256') and hashlib.sha256(body).hexdigest() != r['sha256']:
            raise RuntimeError(f'result {r["key"]}: sha256 mismatch (download again)')
        with self.mu:
            self.cur['chunks'] += 1
        try:
            h, lines = read_records(body)
        except (OSError, ValueError, EOFError) as e:
            return self._refuse(r, f'not a records file: {e}', body)
        info = self.R.info
        want = {'records': 1, 'game': self.R.cfg['game'], 'features': info['num_features'],
                'actions': info['num_actions'], 'aux': self.aux, 'seed': self.seed, 'build_id': self.build}
        got = {k: h.get(k) for k in want}
        if got != want:
            return self._refuse(r, f'header {got}, the run wants {want}', body)
        try:
            if _spec_value(h.get('spec')) != _spec_value(self.spec):
                return self._refuse(r, 'the records were played with another spec', body)
        except ValueError:
            return self._refuse(r, 'the spec is not JSON', body)
        ids = [tuple(x) for x in (r.get('ids') or [])]
        sparse = shard_format(self.R.cfg).get('x_format') == 'sparse'
        machine = r.get('machine_id') or '?'
        keep = []
        for ln in lines:
            try:
                gid = int(json.loads(ln)['gid'])
            except (ValueError, KeyError, TypeError):
                return self._refuse(r, 'a game line without its gid', body)
            if not _in_ranges(gid, ids):
                continue                           # (not credited to this chunk: another copy counts)
            t0 = time.time()
            try:
                s, g = E.replay_rows(self.R.cfg['game'], self.spec, ln, self.aux)
            except ValueError as e:
                return self._refuse(r, f'replay: {e}', body, gid)
            dt = time.time() - t0
            if g['seed'] != E.game_seed_of(self.seed, gid):
                return self._refuse(r, f'game {gid}: seed {g["seed"]} is not its game id\'s', body, gid)
            if g['n'] != self.players[gid % len(self.players)]:
                return self._refuse(r, f'game {gid}: {g["n"]} players, the run plays {self.players[gid % len(self.players)]}',
                                    body, gid)
            net = None
            if s is not None:
                # the rows as a shard stores them, checked as write_shard checks them (the non-finite guard)
                if sparse:
                    s['xs'] = _sparse_rows(s)
                    del s['x']
                    c, _cols = compact_sparse(s)
                else:
                    s['x'] = s['x'].astype('float16')
                    c = compact(s)
                probs = shard_problems(c)
                if probs:
                    return self._refuse(r, f'game {gid}: non-finite rows: {"; ".join(probs)}', body, gid)
                net = int(s['net'].min())
            keep.append({'gid': gid, 'machine': machine, 'net': net if net is not None else r.get('gen'), 's': s,
                         'rows': g['rows'], 'replay_s': dt})
        with self.mu:
            self.held.extend(keep)
            self.cur['records'] += len(keep)
        if keep:
            self.t_game = time.time()

    def _engine(self):
        from . import games as G
        return G.engine()

    def _drain(self):
        """Held games, oldest first, into the open iteration's quota (RemotePlay.accept_pool); the stale ones dropped."""
        from .data import ShardWriter, shard_format
        from .workqueue import lag_hist
        it = self.rp.pool_iteration()
        if it is None:
            return
        with self.mu:
            if self.cur['it'] != it:
                self.cur = self._counters(it)
            while self.held:
                g = self.held[0]
                if g['net'] is not None and g['net'] < it - 1 - self.max_stale:
                    self.held.popleft()
                    self.cur['stale_dropped'] += 1
                    continue
                if not self.rp.accept_pool(it, self.name, 1):
                    break
                self.held.popleft()
                self.st['accepted_total'] = self.st.get('accepted_total', 0) + 1
                self.cur['games'] += 1
                self.cur['replay_s'] += g['replay_s']
                m = g['machine']
                self.cur['machines'][m] = self.cur['machines'].get(m, 0) + 1
                if g['s'] is None:
                    continue
                key = (it, m)
                w = self.writers.get(key)
                if w is None:
                    d = os.path.join(self.R.run, 'data', f'iter_{it:04d}')
                    meta = {'game': self.R.cfg['game'], 'features': self.R.info['num_features'],
                            'actions': self.R.info['num_actions'], 'iter': it, 'source': 'pool', 'machine': m,
                            'farm_run': self.farm, **shard_format(self.R.cfg)}
                    pre = 'pool-' + ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in m)

                    def wrote(*a, **k):
                        self.cur['shards'] += 1
                    w = ShardWriter(d, pre, rows=self.R.cfg['shard_rows'], meta=meta, bad_dir=self.R.debug_dir(),
                                    on_write=wrote)
                    self.writers[key] = w
                w.add(g['s'])
                self.cur['rows'] += g['rows']
                lag_hist(g['s']['net'], it, self.cur['stale'])


def _sparse_rows(s):
    """A sample dict's features as the shard writer's SparseRows (the float16 cast's non-zeros, bit for bit):
    a held game keeps them instead of its dense float32 rows."""
    from .data import SparseRows
    return SparseRows.from_dense(s['x'], s['x'].shape[1])
