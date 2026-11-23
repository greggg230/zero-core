"""A run's evaluation on the farm (docs/az.md 65): the run setting `eval_remote` = 'pool:<farm run>' plays the games of
the run's checkpoint evaluations - the anchors' games, the Elo matches, a yardstick against a built-in bot - on zero-farm
(friends' idle PCs and the owner's machines that run the farm agent) instead of on a remote machine over SSH, and
records them exactly where and how zero.remote's RemoteEval records a remote's evaluation.

Three parts, all in the run process (one thread, RemoteEval's: the iteration never waits for an evaluation):

1. `FarmEval` (a RemoteEval): its "remote" is a mirror on this machine, `<run>/eval_pool/<farm run>/mirror/<run name>`,
   shipped exactly as RemoteEval ships a remote's mirror (config.json, the Elo pool's results so far, the checkpoints
   and the anchors - hard links where the disk allows, copies otherwise: `LinkTransport`). It is evaluated in this
   process by zero.remote.evaluate_mirror with `FarmRun` - Run.evaluate + Run.elo_update, the run's own code - and the
   result recorded by RemoteEval._record: the evals.jsonl row (host 'pool:<farm run>', plus `eval_pool`: how its games
   were played), the Elo pool's files, models/index.json, best.pt, the game records (src 'farm').
2. `FarmRun` (a zero.run.Run): `play_match` and `play_matches` - every match the evaluation plays between two networks,
   or a network and a built-in bot - go to `FarmMatches`; the rows they return are those the local driver returns
   ((score, margin, rounds, over) of the evaluated network, in game order), and with game_records each game's record
   comes from its trace. Played here as before: a match against an external bot (`ext:`, a process of this machine) and
   the reference games between two bots (no network; once per Elo pool). A chained evaluation spec (docs/az.md 42) is
   refused when the run starts: farm eval rows carry no episodes.
3. `FarmMatches`: the farm run `<farm run>` (its own run on the coordinator, `selfplay` null; never the run's self-play
   pool run: that pool PUTs `eval: null` and signs definitions without an eval template, zero/farm_pool.py), created on
   first use with the pool's worker build, runtime, alternates, priority and min_agent. A match is ONE eval batch
   (zero-farm PROTOCOL 5.2): its networks registered as gens of the farm run (a run checkpoint's gen is its iteration,
   another network - an anchor - gets `ANCHOR_GEN0` + n; each exported once with zero.farm_h2h.export_cached, uploaded
   and registered), a new signed definition whose eval template is the batch's job (settings: the worker build's own
   `check-config --eval` play settings of the run's settings at the match's simulations, with `eval_trace` 1; seats
   [{"net": A}, {"net": B} | {"bot": name}]; paired), then the batch (its job = the template + the match's driver seed as
   the base seed). The coordinator holds one batch per run, so matches are played one after the other.

   Collecting: every eval chunk the coordinator lists is downloaded (sha256), its credited rows taken (ids, also,
   dup_ids: the first copy of a game wins, as zero.farm_h2h.pull) and each row checked before it counts - its digest
   against its record (zero.farm_eval.check_digest), its game id inside the batch, its seats the batch's for that id,
   its record's seed the game id's (`game_seed_of(seed, gid, paired)`: a worker cannot choose its games). When every
   game of the batch is in: a spot check, `eval_pool_spot` (5%, at least one) of the farm's games, chosen at random
   seeded by the batch's name, replayed through the engine from seed + moves (zero.farm_eval.replay_trace: every move
   legal, the record's result, rounds and setup). Any mismatch refuses the batch: logged (`refused.jsonl`), the batch
   dropped on the farm, and its games played here instead (or, with no deadline, the evaluation fails and is retried
   with a new batch).

   Why a rules replay and not a search replay: the network's arithmetic differs in its last bits across platforms and
   execution providers (x64 vs arm64, CPU vs CUDA; zero-farm PROTOCOL 11.4), so the same game searched on two machines
   may choose differently - neither is wrong. The engine's rules, though, replay exactly on every platform (the farm
   pool's platform check, docs/az.md 62, is made here too for each platform alternate, on the eval settings), so the
   spot replay is valid whoever played the game; it catches bad hardware, transport faults and broken builds, not a
   network choosing its moves differently (PROTOCOL 5.3).

   Deadline and fallback: the games of a batch the farm has not delivered `eval_pool_deadline_s` seconds after it was
   put are played here: the batch is dropped on the farm (`eval: null`: its leases are cancelled at their next beat),
   then the missing game ids, in whole pairs, are played by `pool_worker` (the same signed worker build, on the same
   graphs, the same base seed and game ids: the same games the farm would have played) with `eval_pool_threads`
   threads at below-normal priority. `eval_pool_deadline_s` 0: no deadline, the evaluation waits for the farm.

   The deals: a batch's game ids are the farm run's (the coordinator reserves them from its own counter), so a farm
   match plays other deals than a local driver of the same seed would (game g's seed is game_seed(seed, gid, paired));
   the same matches, the same seats and settings, equally random deals. (A remote's games were never bit-identical to
   this machine's either: other arithmetic.)

State: `<run>/eval_pool/<farm run>/`: state.json (the farm run's PUT state, the networks' gens and files, every batch:
name, key, seats, games, seed, ids, status, machines), batches/<name>.jsonl (the rows taken), ckpt/<gen>/ (the
exports), export/ (the weight-swap templates and reference positions), definitions/ (every definition version made),
settings/, local/ (the fallback's pieces), refused.jsonl, mirror/. Every step is idempotent: a restarted run resumes
the batch it was collecting (the same batch key). The admin token, the data key and farm-release are
zero.farm_def's (never printed).
"""
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time

from . import elo, farm_def as FD, local_config, package
from .remote import RemoteEval, Transport, evaluate_mirror

PREFIX = 'pool:'
ANCHOR_GEN0 = 900_000_000                  # the first gen of a network that is not a checkpoint of the run (an anchor)
PAGE = 500                                 # results per GET
LOCAL_CHUNK = 64                           # games per local fallback piece
ORT_NAMES = ('onnxruntime.dll', 'libonnxruntime.dylib', 'libonnxruntime.so')


class EvalPoolError(RuntimeError):
    """A farm evaluation that cannot be recorded (logged by RemoteEval and retried)."""


def is_pool(name):
    return str(name).startswith(PREFIX)


def _setting(cfg, k):
    from .run import DEFAULTS
    v = cfg.get(k)
    return DEFAULTS.get(k) if v is None else v


def ranges_of(gids):
    """Sorted ids -> [[lo, hi), ...]."""
    out = []
    for g in sorted(set(int(x) for x in gids)):
        if out and out[-1][1] == g:
            out[-1][1] = g + 1
        else:
            out.append([g, g + 1])
    return out


def ids_of(rs):
    return {g for lo, hi in (rs or []) for g in range(int(lo), int(hi))}


def local_ranges(gids, paired, chunk=LOCAL_CHUNK):
    """Missing ids -> contiguous pieces of at most `chunk` games (whole pairs when paired)."""
    out = []
    for lo, hi in ranges_of(gids):
        if paired:
            lo, hi = lo - lo % 2, hi + hi % 2
        while lo < hi:
            step = min(chunk, hi - lo)
            out.append([lo, lo + step])
            lo += step
    return out


def ort_library(worker, runtime):
    """The ONNX Runtime library the local fallback's worker loads: the runtime pack's next to the worker build
    (<builds>/<runtime>/onnxruntime.dll, the farm builds' layout) or in $ZERO_BIG_DISK/zero-farm-builds/<runtime>/ ->
    (path or None, the paths looked at)."""
    bases = [os.path.dirname(os.path.dirname(os.path.abspath(worker)))]
    big = local_config.big_disk_path('zero-farm-builds')
    if big:
        bases.append(big)
    looked = []
    for b in bases:
        for n in ORT_NAMES:
            p = os.path.join(b, runtime or '', n)
            looked.append(p)
            if runtime and os.path.exists(p):
                return p, looked
    return None, looked


def slug_part(s, n=40):
    return re.sub(r'[^a-z0-9-]+', '-', str(s).lower()).strip('-')[:n].strip('-')


class LinkTransport(Transport):
    """RemoteEval's 'local' transport (this machine) with the checkpoints and anchors (`*.pt`: never rewritten in place)
    hard-linked into the mirror instead of copied; a copy where the disk refuses a link. Every other file is copied
    as before (the mirror writes its own Elo pool and metrics)."""

    def put(self, src, dst):
        if str(src).endswith('.pt'):
            tmp = dst + '.tmp'
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
                os.link(src, tmp)
                os.replace(tmp, dst)
                return
            except OSError:
                pass
        super().put(src, dst)


# ------------------------------------------------------------------------------------------------ the farm's side
class FarmMatches:
    """The farm run `farm` of the run whose settings are `cfg` and directory `run_dir` (module docstring, 3). `client`:
    a zero.farm_def Client (tests: a fake coordinator); `exporter(ckpt, gen, out_dir) -> {model, sha256, model_size,
    reference, reference_sha256, reference_size}` and `definer(client, info) -> info` stand in for the ONNX export and
    farm-release in tests; `engine`: the game's engine module (default zero.games.engine()); `clock`, `sleep`: time."""

    def __init__(self, cfg, run_dir, farm, log, client=None, exporter=None, definer=None, engine=None, clock=None,
                 sleep=None):
        self.cfg = cfg
        self.farm = farm
        self.log_fn = log
        if not FD.SLUG_RE.match(farm or ''):
            raise SystemExit(f'eval_remote {PREFIX}{farm!r}: the farm run must be a slug ([a-z0-9-], 1-63)')
        s = lambda k: _setting(cfg, k)
        self.build = str(s('pool_build') or '')
        self.runtime = str(s('pool_runtime') or '') or None
        self.worker = str(s('pool_worker') or '')
        if not self.build or not self.worker:
            raise SystemExit(f'eval_remote {PREFIX}{farm}: pool_build (the farm worker build) and pool_worker (its '
                             f'executable here: check-config picks the eval template\'s settings, and it plays the '
                             f'games past the deadline) must be set')
        if not os.path.isfile(self.worker):
            raise SystemExit(f'eval_remote {PREFIX}{farm}: pool_worker {self.worker} does not exist')
        self.url = str(s('pool_url') or FD.URL).rstrip('/')
        self.key = str(s('pool_key') or '') or None
        self.farm_release = str(s('pool_farm_release') or '') or None
        self.expires_days = s('pool_expires_days')
        self.min_agent = str(s('pool_min_agent') or '').strip() or None
        if self.min_agent and not FD.SEMVER_RE.match(self.min_agent):
            raise SystemExit(f'eval_remote {PREFIX}{farm}: pool_min_agent {self.min_agent!r} is not a version like 0.2.8')
        from .farm_pool import alternate_ids
        self.alt_builds = alternate_ids(s('pool_alternates'))
        if any(not FD.BUILD_RE.match(b) for b in self.alt_builds) or len(self.alt_builds) > FD.MAX_ALTERNATES:
            raise SystemExit(f'eval_remote {PREFIX}{farm}: pool_alternates {self.alt_builds}: up to '
                             f'{FD.MAX_ALTERNATES} worker build ids (24 hex)')
        self.alt_vendor = str(s('pool_alt_vendor') or '').strip() or None
        self.priority = float(s('pool_priority'))
        self.needs_ram = int(s('pool_needs_ram_mb') or 0)
        self.export_graph = str(s('pool_export') or '')
        self.deadline_s = float(s('eval_pool_deadline_s') or 0)
        self.threads = max(1, int(s('eval_pool_threads') or 1))
        self.spot = float(s('eval_pool_spot') or 0)
        if not 0 <= self.spot <= 1:
            raise SystemExit(f'eval_pool_spot {self.spot}: a share of the farm\'s games, 0 to 1')
        gpl = s('eval_pool_games_per_lease')
        self.per_lease = 'auto' if str(gpl).strip() in ('', 'auto', '0') else int(gpl)
        self.poll_s = float(s('eval_pool_poll_s'))
        self.ort = None
        if self.deadline_s > 0:
            self.ort, looked = ort_library(self.worker, self.runtime)
            if self.ort is None:
                raise SystemExit(f'eval_remote {PREFIX}{farm}: eval_pool_deadline_s {self.deadline_s:g} plays the games '
                                 f'the farm has not delivered here with pool_worker, but no ONNX Runtime of the runtime '
                                 f'pack {self.runtime} is next to it (looked at {looked}); or set eval_pool_deadline_s 0')
        self.dir = os.path.join(run_dir, 'eval_pool', farm)
        os.makedirs(self.dir, exist_ok=True)
        self.client = client or FD.Client(self.url)
        self.exporter = exporter or self._export
        self.definer = definer or self._define
        self._engine = engine
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.st = self._load()
        self.play_cache = {}                       # (sims, the run's settings' sha) -> the template's settings
        self.sha_cache = {}                        # (path, size, mtime) -> sha256

    # ------------------------------------------------------------ state
    def _load(self):
        p = os.path.join(self.dir, 'state.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        return {'farm_run': self.farm, 'url': self.url, 'run_put': False, 'nets': {}, 'next_anchor': 0,
                'batches': {}, 'next_batch': 1, 'cursor': 0}

    def save(self):
        p = os.path.join(self.dir, 'state.json')
        tmp = p + f'.{os.getpid()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.st, f, indent=1)
        os.replace(tmp, p)

    def log(self, msg):
        self.log_fn(f'eval pool {self.farm}: {msg}')

    def describe(self):
        alts = f', alternates {",".join(self.alt_builds)}' if self.alt_builds else ''
        dl = f'deadline {self.deadline_s:g} s (then here, {self.threads} threads)' if self.deadline_s > 0 else 'no deadline'
        return (f"farm run {self.farm!r} (farm {self.url}, build {self.build}{alts}, min_agent {self.min_agent or 'none'}, "
                f'{dl}, spot {self.spot:g})')

    def engine(self):
        if self._engine is None:
            from . import games as G
            self._engine = G.engine()
        return self._engine

    # ------------------------------------------------------------ settings, networks, the farm run
    def play_settings(self, cfg, sims):
        """The eval template's settings of a match at `sims` simulations: the run's settings with eval_sims and
        eval_trace 1, eval_spec resolved, kept to what the worker build's `check-config --eval` calls play
        (zero.farm_h2h.play_settings: the seats checked against the full settings')."""
        from . import farm_eval as FE, farm_h2h as FH
        c = dict(cfg, eval_sims=int(sims), eval_trace=1)
        c = FE.resolve_eval_spec(c)
        key = (int(sims), hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest())
        if key not in self.play_cache:
            play = FH.play_settings(self.worker, self.build, c, os.path.join(self.dir, 'settings'), self.log)
            if not play.get('eval_trace'):
                raise SystemExit(f'worker build {self.build} does not trace eval games (eval_trace, docs/az.md 50): the '
                                 f'farm evaluation needs every game\'s record')
            self.play_cache[key] = play
        return self.play_cache[key]

    def ensure_run(self, settings):
        """The farm run, created once (`selfplay` and `eval` null): the worker build, its runtime, its alternates (each
        platform alternate's settings checked to replay exactly there, docs/az.md 62), priority, min_agent."""
        st = self.st
        if not st.get('build_game'):
            st['build_game'] = FD.worker_game(self.client, self.build, self.runtime) if self.runtime else self.cfg['game']
            self.save()
        if st.get('alternates') is None or st.get('alt_builds') != self.alt_builds:
            alternates, builds = FD.run_alternates(self.client, self.build, self.runtime, self.alt_builds, self.alt_vendor)
            for p in sorted({x['when'].get('platform') for x in alternates} - {None}):
                self.platform_check(settings, p)
            st.update(alternates=alternates, builds=builds, alt_builds=list(self.alt_builds))
            st['run_put'] = False
            self.save()
        if not st['run_put']:
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=self.run_body(None))
            st['run_put'] = True
            self.save()
            self.log(f'farm run {self.farm} put (worker build {self.build}, runtime {self.runtime}, '
                     f'{len(st["alternates"])} alternate(s), min_agent {self.min_agent or "none"})')

    def platform_check(self, settings, platform):
        """A platform alternate plays the batches only if their games replay exactly there (`check-config --eval
        --platform`, the game's libm rule; docs/az.md 62): SystemExit otherwise."""
        from .farm_h2h import worker_json
        d = os.path.join(self.dir, 'settings')
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'eval-{platform}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(settings, f)
        rc, out, text = worker_json(self.worker, 'check-config', path, '--json', '--eval', '--platform', platform)
        if not isinstance(out, dict) or out.get('platform') != platform:
            raise SystemExit(f'pool_worker {self.worker} cannot judge the eval settings for {platform} (a build before '
                             f'docs/az.md 62): no {platform} alternate without that check')
        if out.get('libm_refusals'):
            raise SystemExit(f'the eval games would replay differently on {platform} (its libm, docs/az.md 62): '
                             f'{"; ".join(out["libm_refusals"])} - leave its alternate out of pool_alternates')
        self.log(f'{platform}: the eval games replay exactly there (check-config --eval --platform {platform})')

    def run_body(self, ev, state='active'):
        b = {'game': self.st['build_game'], 'build_id': self.build, 'runtime': self.runtime, 'priority': self.priority,
             'min_agent': self.min_agent}
        if self.st.get('alternates'):
            b['alternates'] = self.st['alternates']
        b.update(selfplay=None, eval=ev, state=state)
        return b

    def file_sha(self, path):
        st = os.stat(path)
        k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
        if k not in self.sha_cache:
            self.sha_cache[k] = FD.file_sha256(path)
        return self.sha_cache[k]

    def net(self, path, label=None):
        """The farm run's gen of the network `path` (by the file's sha256): a run checkpoint's
        (`.../models/iter_NNNN.pt`) is its iteration, any other network's ANCHOR_GEN0 + n; exported, uploaded and
        registered once -> the gen."""
        sha = self.file_sha(path)
        nets = self.st['nets']
        n = nets.get(sha)
        if n is None:
            m = re.search(r'[\\/]models[\\/]iter_(\d+)\.pt$', str(path))
            used = {x['gen'] for x in nets.values()}
            if m and int(m.group(1)) < ANCHOR_GEN0 and int(m.group(1)) not in used:
                gen = int(m.group(1))
            else:
                gen = ANCHOR_GEN0 + self.st['next_anchor']
                self.st['next_anchor'] += 1
            n = nets[sha] = {'gen': gen, 'file': os.path.basename(path), 'label': label, 'checkpoint_sha256': sha}
            self.save()
        gen = n['gen']
        if not n.get('model'):
            t0 = self.clock()
            ex = self.exporter(path, gen, os.path.join(self.dir, 'ckpt', str(gen)))
            n.update({k: ex[k] for k in ('model', 'sha256', 'model_size', 'reference', 'reference_sha256',
                                         'reference_size')})
            n['export_s'] = round(self.clock() - t0, 1)
            self.save()
            self.log(f'gen {gen} ({os.path.basename(path)}{f", {label}" if label else ""}) exported in {n["export_s"]} s '
                     f'({n["sha256"][:12]})')
        if not n.get('registered'):
            keys = {'model': f'ckpt/{self.farm}/{gen}/model.onnx', 'reference': f'ckpt/{self.farm}/{gen}/reference.npz'}
            self.client.put_artifact(keys['model'], n['model'], n['sha256'])
            self.client.put_artifact(keys['reference'], n['reference'], n['reference_sha256'])
            self.client.call('PUT', f'/v1/admin/run/{self.farm}/checkpoint/{gen}',
                             body={'model': keys['model'], 'sha256': n['sha256'], 'reference': keys['reference'],
                                   'reference_sha256': n['reference_sha256']})
            n['registered'] = True
            self.save()
        return gen

    def gen_files(self, gen):
        return next(n for n in self.st['nets'].values() if n['gen'] == gen)

    def _export(self, ckpt, gen, out_dir):
        from . import games as G
        from .farm_h2h import export_cached
        cache = os.path.join(self.dir, 'export')
        self._seed_export_cache(cache)
        return export_cached(ckpt, gen, out_dir, self.cfg['game'], G.spec_json(self.cfg['spec']), cache,
                             f'{self.farm}/{gen}', checkpoint_iter=gen, graph=self.export_graph)

    def _seed_export_cache(self, cache):
        """A run with a farm pool has built the architecture's weight-swap template and recorded its reference
        positions already (`<run>/pool/<farm run>/export/`, zero.farm_pool): copied once into this farm run's own
        cache (never written there), so the first export here is a weight swap too, not a template build and a
        recording (minutes of CPU in the run process). A template of another fingerprint is built here as before."""
        if os.path.isdir(cache):
            return
        run_dir = os.path.dirname(os.path.dirname(self.dir))
        rem = self.cfg.get('selfplay_remote') or ''
        names = [x.strip() for x in (rem.split(',') if isinstance(rem, str) else rem) if str(x).strip()]
        import shutil
        for n in names:
            src = os.path.join(run_dir, 'pool', n[len(PREFIX):], 'export') if is_pool(n) else None
            if src and os.path.isdir(src):
                tmp = cache + f'.{os.getpid()}.tmp'
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(src, tmp)
                os.replace(tmp, cache)
                self.log(f'export cache seeded from the farm pool\'s ({src})')
                return

    def _define(self, client, info):
        return FD.publish(client, info, key=self.key, farm_release=self.farm_release, log=self.log)

    # ------------------------------------------------------------ a match = one batch
    def match(self, cfg, a, b, games, sims, seed, paired, label):
        """The match between network `a` (a path) and `b` (('net', path) or ('bot', name)) - `games` games at `sims`
        simulations, base seed `seed`, `paired` - played as one eval batch of the farm run (module docstring) -> (the
        batch's record, its rows sorted by game id: each {gid, pair, seats, result, margin, rounds, over, digest, rec,
        by: {machine, lease}})."""
        settings = self.play_settings(cfg, sims)
        self.ensure_run(settings)
        ga = self.net(a, label=None)
        seat_b = {'net': self.net(b[1])} if b[0] == 'net' else {'bot': str(b[1])}
        seats = [{'net': ga}, seat_b]
        sk = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        key = f'{json.dumps(seats, sort_keys=True)}|{int(games)}|{int(seed)}|{bool(paired)}|{sk}'
        bt = self.st['batches'].get(key)
        if bt is None or bt.get('status') == 'refused':
            name = f'b{self.st["next_batch"]}-{slug_part(label)}'.strip('-')[:63].strip('-')
            self.st['next_batch'] += 1
            bt = self.st['batches'][key] = {'name': name, 'key': key, 'seats': seats, 'games': int(games),
                                            'seed': int(seed), 'paired': bool(paired), 'sims': int(sims),
                                            'settings_sha': sk, 'label': label, 'status': 'new', 'created': self.clock(),
                                            'have': [], 'machines': {}, 'local': []}
            self.save()
        if bt['status'] != 'done':
            self._play(bt, settings)
        return bt, self.rows(bt)

    def template(self, bt, settings):
        """The definition's eval template - also the batch's job without its seed (PROTOCOL 7)."""
        return {'settings': settings, 'seats': bt['seats'], 'paired': bt['paired']}

    def gens(self, bt):
        return sorted({s['net'] for s in bt['seats'] if 'net' in s})

    def _play(self, bt, settings):
        if bt['status'] == 'new':
            self._submit(bt, settings)
        self._collect(bt, settings)

    def _submit(self, bt, settings):
        """The batch's signed definition (its gens and template), then the batch on the farm run."""
        gens = self.gens(bt)
        d = bt.get('definition')
        if not d or not d.get('published'):
            if not d:
                cks = {}
                for g in gens:
                    n = self.gen_files(g)
                    cks[str(g)] = {'model_sha256': n['sha256'], 'model_size': n['model_size'],
                                   'reference_sha256': n['reference_sha256'], 'reference_size': n['reference_size']}
                d = FD.prepare(self.client, 'run', self.farm, game=self.st['build_game'], builds=self.st['builds'],
                               runtime=self.runtime, checkpoints=cks,
                               jobs={'selfplay': None, 'eval': self.template(bt, settings), 'fuzz': None},
                               expires_days=self.expires_days, root=os.path.join(self.dir, 'definitions'))
                bt['definition'] = d
                self.save()
            d = self.definer(self.client, d)
            bt['definition'] = {k: d[k] for k in ('version', 'sha256', 'key', 'path', 'published') if k in d}
            self.save()
        job = dict(self.template(bt, settings), seed=int(bt['seed']))
        for tries in range(3):
            ev = {'batch': bt['name'], 'games': bt['games'], 'checkpoints': gens, 'games_per_lease': self.per_lease,
                  'gpu': 'any', 'needs': {'ram_mb': self.needs_ram} if self.needs_ram else None, 'job': job}
            try:
                self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=self.run_body(ev))
                break
            except FD.FarmError as e:
                # (a batch name the farm run used before - this state is newer than the farm run, e.g. a run
                # directory copied: a new name; a batch never changes, PROTOCOL 5.2)
                if e.status != 409 or tries == 2:
                    raise
                bt['name'] = f'b{self.st["next_batch"]}-{slug_part(bt["label"])}'.strip('-')[:63].strip('-')
                self.st['next_batch'] += 1
                self.save()
        es = (self.client.call('GET', f'/v1/admin/run/{self.farm}') or {}).get('eval_state') or {}
        if es.get('batch') != bt['name'] or 'lo' not in es or 'hi' not in es:
            raise EvalPoolError(f'batch {bt["name"]}: the coordinator\'s eval_state {es} does not name it with its ids')
        if int(es['hi']) - int(es['lo']) != bt['games']:
            raise EvalPoolError(f'batch {bt["name"]}: {bt["games"]} games, the coordinator reserved [{es["lo"]}, '
                                f'{es["hi"]})')
        bt.update(lo=int(es['lo']), hi=int(es['hi']), status='submitted', submitted=self.clock())
        self.save()
        vs = ' v '.join(f'gen {s["net"]}' if 'net' in s else f'bot {s["bot"]}' for s in bt['seats'])
        self.log(f'batch {bt["name"]}: {bt["games"]} {"paired " if bt["paired"] else ""}games, {vs}, ids '
                 f'[{bt["lo"]}, {bt["hi"]}), definition v{bt["definition"]["version"]}')

    def missing(self, bt):
        have = set(bt['have'])
        return [g for g in range(bt['lo'], bt['hi']) if g not in have]

    def _collect(self, bt, settings):
        last = None
        while True:
            self.pull()
            if bt.get('refused'):
                break
            miss = self.missing(bt)
            if not miss:
                break
            now = self.clock()
            if self.deadline_s > 0 and now >= bt['submitted'] + self.deadline_s:
                break
            line = f'{bt["games"] - len(miss)}/{bt["games"]} games in'
            if line != last and bt['have']:
                self.log(f'batch {bt["name"]}: {line} ({len(bt["machines"])} machine(s), {now - bt["submitted"]:.0f} s)')
            last = line
            self.sleep(self.poll_s)
        if bt.get('refused') or self.missing(bt):
            self._drop(bt)
            if not bt.get('refused'):
                self.pull()                        # (a chunk that came meanwhile)
        if not bt.get('refused'):
            miss = self.missing(bt)
            if miss:
                if self.deadline_s <= 0:
                    raise EvalPoolError(f'batch {bt["name"]}: games {ranges_of(miss)} missing')
                self.log(f'batch {bt["name"]}: deadline ({self.deadline_s:g} s): {len(miss)} game(s) the farm did not '
                         f'deliver ({ranges_of(miss)}) are played here')
                self.play_local(bt, settings, miss)
            self.spot_check(bt)                    # (a mismatch refuses the batch)
        if bt.get('refused'):
            self._refused(bt, settings)            # (its games played here, or EvalPoolError without a deadline)
        bt['status'] = 'done'
        bt['done'] = self.clock()
        self.save()
        nl = len(ids_of(bt['local']))
        self.log(f'batch {bt["name"]}: {bt["games"]} games in {bt["done"] - bt["submitted"]:.0f} s (farm '
                 f'{bt["games"] - nl} from {len(bt["machines"])} machine(s), here {nl}); spot check '
                 f'{bt.get("spot", {}).get("checked", 0)} replayed, all equal')

    def _drop(self, bt):
        """The batch off the farm (`eval: null`: its unissued ids dropped, its leases cancelled at their next beat),
        once."""
        if not bt.get('dropped'):
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=self.run_body(None))
            bt['dropped'] = self.clock()
            self.save()

    def _refused(self, bt, settings):
        """A refused batch (a row its record does not support): every game played here instead (the deadline's
        fallback), or, with no deadline, the evaluation fails (retried with a new batch)."""
        why = bt['refused']
        self._drop(bt)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'refused.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': self.clock(), 'batch': bt['name'], 'why': why}) + '\n')
        self.log(f'batch {bt["name"]} REFUSED: {why}')
        if self.deadline_s <= 0:
            bt['status'] = 'refused'
            self.save()
            raise EvalPoolError(f'batch {bt["name"]} refused (no deadline: no local fallback; a new batch on the next '
                                f'try): {why}')
        p = self.rows_path(bt)
        if os.path.exists(p):
            os.replace(p, p + f'.refused-{int(self.clock())}')
        bt.update(have=[], machines={}, local=[], refused_was=why, refused=None, spot={'checked': 0, 'of': 0})
        self.save()
        self.log(f'batch {bt["name"]}: its {bt["games"]} games are played here instead')
        self.play_local(bt, settings, list(range(bt['lo'], bt['hi'])))

    # ------------------------------------------------------------ results
    def rows_path(self, bt):
        return os.path.join(self.dir, 'batches', bt['name'] + '.jsonl')

    def rows(self, bt):
        """The batch's rows taken, by game id (the first copy wins), sorted."""
        out = {}
        p = self.rows_path(bt)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out.setdefault(int(r['gid']), r)
        return [out[g] for g in sorted(out)]

    def _append(self, bt, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(self.rows_path(bt)), exist_ok=True)
        with open(self.rows_path(bt), 'a', encoding='utf-8', newline='\n') as f:
            for r in rows:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')

    def check_row(self, bt, row):
        """A row counts only if its record supports it (module docstring) -> None, else why not."""
        from . import farm_eval as FE
        try:
            FE.check_digest(row)
        except FE.TraceMismatch as e:
            return str(e)
        g = int(row['gid'])
        if not bt['lo'] <= g < bt['hi']:
            return f'game {g} is outside the batch\'s ids [{bt["lo"]}, {bt["hi"]})'
        if row.get('seats') != FE.seats_of(bt['seats'], g):
            return f'game {g}: seats {row.get("seats")}, the batch\'s are {FE.seats_of(bt["seats"], g)}'
        want = self.engine().game_seed_of(int(bt['seed']), g, bool(bt['paired']))
        if row['rec'].get('seed') != want:
            return f'game {g}: its record\'s seed {row["rec"].get("seed")} is not its game id\'s ({want})'
        if not isinstance(row.get('result'), list) or len(row['result']) != len(bt['seats']):
            return f'game {g}: result {row.get("result")}'
        # (the row's outcome is its record's: the record is what the spot check replays)
        import struct
        rres = [struct.unpack('<f', struct.pack('<I', int(b) & 0xffffffff))[0] for b in row['rec'].get('result') or []]
        if bool(row['rec'].get('over')) != bool(row.get('over')) or \
                int(row['rec'].get('rounds', -1)) != int(row.get('rounds', -2)) or \
                (row['rec'].get('over') and [float(x) for x in row['result']] != rres):
            return (f'game {g}: row result/over/rounds {row.get("result")}/{row.get("over")}/{row.get("rounds")}, its '
                    f'record\'s {rres}/{row["rec"].get("over")}/{row["rec"].get("rounds")}')
        return None

    def _file(self, key, sha=None):
        d = os.path.join(self.dir, 'results')
        p = os.path.join(d, hashlib.sha256(key.encode()).hexdigest()[:24] + '.jsonl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                body = f.read()
        else:
            body = self.client.call('GET', f'/v1/admin/artifact/{key}', raw=True)
            if sha and hashlib.sha256(body).hexdigest() != sha:
                raise EvalPoolError(f'result {key}: sha256 mismatch (downloaded again on the next poll)')
            os.makedirs(d, exist_ok=True)
            tmp = p + f'.{os.getpid()}.tmp'
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, p)
        if sha and hashlib.sha256(body).hexdigest() != sha:
            os.remove(p)
            raise EvalPoolError(f'result {key}: sha256 mismatch')
        return body.decode('utf-8')

    def pull(self):
        """New result chunks of the farm run: each eval chunk of a batch of ours downloaded and its credited rows
        checked and taken (the first copy of a game wins); a row that fails its check refuses its batch. Acked once
        saved -> rows taken."""
        from . import farm_eval as FE
        by_name = {b['name']: b for b in self.st['batches'].values()}
        taken = 0
        while True:
            ans = self.client.call('GET', f'/v1/admin/run/{self.farm}/results?after={self.st.get("cursor", 0)}'
                                          f'&limit={PAGE}') or {}
            res = ans.get('results') or []
            counts = {}                            # result id -> its credited ids' single games (docs/az.md 82)
            for r in res:
                bt = by_name.get(r.get('batch'))
                if r.get('kind') != 'eval' or bt is None or bt.get('status') != 'submitted' or bt.get('refused') or \
                        bt.get('refused_was'):
                    continue                       # (another kind, a batch done or refused: nothing more counts)
                have = set(bt['have'])
                dup = ids_of(r.get('dup_ids'))
                parts = [(r['key'], r.get('sha256'), ids_of(r.get('ids')))]
                parts += [(x['key'], None, ids_of(x.get('ids'))) for x in r.get('also') or []]
                new = []
                cred = []
                machine = r.get('machine_id') or '?'
                for k, sha, ids in parts:
                    for row in FE.read_rows(self._file(k, sha)):
                        g = int(row['gid'])
                        if g in ids:
                            cred.append(row)
                        if g in dup or g in have or g not in ids:
                            continue
                        why = self.check_row(bt, row)
                        if why:
                            bt['refused'] = f'machine {machine}, lease {r.get("lease_id")}: {why}'
                            break
                        row['by'] = {'machine': machine, 'lease': r.get('lease_id')}
                        have.add(g)
                        new.append(row)
                    if bt.get('refused'):
                        break
                if bt.get('refused'):
                    continue
                try:
                    counts[r.get('id')] = FE.credited_single_games(self.cfg['game'], self._spec(bt), r, cred)
                except Exception:                  # noqa: BLE001 - a count is exact or absent, never the pull's failure
                    pass
                self._append(bt, new)
                bt['have'] = sorted(have)
                if new:
                    if not bt.get('first'):
                        bt['first'] = self.clock()
                    bt['machines'][machine] = bt['machines'].get(machine, 0) + len(new)
                taken += len(new)
            if res:
                self.st['cursor'] = ans.get('cursor', self.st.get('cursor', 0))
                self.save()
                self.client.call('POST', f'/v1/admin/run/{self.farm}/ack', body=FE.ack_body(self.st['cursor'], counts))
            if len(res) < PAGE:
                self.save()
                return taken

    # ------------------------------------------------------------ verification
    def spot_check(self, bt):
        """`eval_pool_spot` of the batch's farm games (at least one when there are any), a random choice seeded by
        the batch's name, replayed through the engine from seed + moves (zero.farm_eval.replay_trace); a mismatch
        refuses the batch."""
        from . import farm_eval as FE
        local = ids_of(bt['local'])
        rows = [r for r in self.rows(bt) if int(r['gid']) not in local]
        if not rows or self.spot <= 0:
            bt['spot'] = {'checked': 0, 'of': len(rows)}
            return
        k = min(len(rows), max(1, math.ceil(self.spot * len(rows))))
        rng = random.Random(int(hashlib.sha256(bt['name'].encode()).hexdigest()[:16], 16))
        pick = sorted(rng.sample(range(len(rows)), k))
        spec = None
        t0 = self.clock()
        for i in pick:
            r = rows[i]
            if spec is None:
                spec = self._spec(bt)
            try:
                FE.replay_trace(r, self.cfg['game'], spec, self.engine(), bt['seed'], bt['paired'])
            except FE.TraceMismatch as e:
                bt['refused'] = f'spot check: machine {r.get("by", {}).get("machine")}: {e}'
                self.save()
                return
        bt['spot'] = {'checked': k, 'of': len(rows), 'seconds': round(self.clock() - t0, 2)}
        self.save()

    def _spec(self, bt):
        """The batch's eval spec (its template's settings)."""
        d = bt.get('definition') or {}
        if d.get('path') and os.path.exists(d['path']):
            with open(d['path'], encoding='utf-8') as f:
                return json.load(f)['jobs']['eval']['settings']['eval_spec']
        from . import farm_eval as FE
        return FE.resolve_eval_spec(dict(self.cfg))['eval_spec']

    # ------------------------------------------------------------ the deadline's fallback
    def local_cmd(self, bt, lo, hi, out):
        cmd = FD.exe_argv(self.worker, 'eval', '--settings', os.path.join(out, 'settings.json'), '--seats',
                          json.dumps(bt['seats']), '--games', str(hi - lo), '--gid-base', str(lo), '--seed',
                          str(bt['seed']), '--threads', str(self.threads), '--out', out, '--prefix', f'l{lo}',
                          '--lease', f'local-{lo}', '--flush-games', str(LOCAL_CHUNK))
        if self.ort:
            cmd += ['--ort-dll', self.ort]
        if bt['paired']:
            cmd.append('--paired')
        for g in self.gens(bt):
            n = self.gen_files(g)
            cmd += ['--model', f'{g}={n["model"]}', '--reference', f'{g}={n["reference"]}', '--sha256', f'{g}={n["sha256"]}']
        return cmd

    def play_local(self, bt, settings, gids):
        """Games `gids` of the batch played here by pool_worker on the batch's graphs (`<worker> eval`: the same games
        the farm would have played), at below-normal priority, in pieces of LOCAL_CHUNK games; each piece's rows
        checked as the farm's. A piece that fails raises EvalPoolError (the evaluation is retried: the batch resumes
        and plays what is still missing)."""
        from . import farm_eval as FE
        have = set(bt['have'])
        for lo, hi in local_ranges(gids, bt['paired']):
            out = os.path.join(self.dir, 'local', f'{bt["name"]}-{lo}')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'settings.json'), 'w', encoding='utf-8') as f:
                json.dump(settings, f)
            kw = {}
            if os.name == 'nt':
                kw['creationflags'] = 0x08000000 | 0x00004000        # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
            else:
                kw['preexec_fn'] = lambda: os.nice(10)
            t0 = self.clock()
            with open(os.path.join(out, 'events.jsonl'), 'w', encoding='utf-8') as fo, \
                    open(os.path.join(out, 'worker.log'), 'w', encoding='utf-8') as fe:
                p = subprocess.run(self.local_cmd(bt, lo, hi, out), stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, **kw)
            evs = []
            with open(os.path.join(out, 'events.jsonl'), encoding='utf-8', errors='replace') as f:
                for line in f:
                    try:
                        evs.append(json.loads(line))
                    except ValueError:
                        pass
            errs = [e.get('message') for e in evs if e.get('ev') == 'error' and e.get('fatal')]
            new = []
            for e in evs:
                if e.get('ev') != 'evals':
                    continue
                with open(os.path.join(out, e['path']), encoding='utf-8') as f:
                    rows = FE.read_rows(f.read())
                for row in rows:
                    g = int(row['gid'])
                    if g in have or not lo <= g < hi:
                        continue
                    why = self.check_row(bt, row)
                    if why:
                        raise EvalPoolError(f'batch {bt["name"]}: the local worker\'s game {g}: {why}')
                    row['by'] = {'machine': 'local', 'lease': f'local-{lo}'}
                    have.add(g)
                    new.append(row)
            self._append(bt, new)
            bt['have'] = sorted(have)
            bt['local'] = ranges_of(list(ids_of(bt['local'])) + [r['gid'] for r in new])
            bt['local_s'] = round(bt.get('local_s', 0) + self.clock() - t0, 1)
            self.save()
            left = [g for g in range(lo, hi) if g not in have]
            if p.returncode != 0 or errs or left:
                with open(os.path.join(out, 'worker.log'), encoding='utf-8', errors='replace') as f:
                    tail = f.read()[-600:].strip()
                raise EvalPoolError(f'batch {bt["name"]}: the local worker played [{lo}, {hi}) with exit {p.returncode}'
                                    f'{f", {errs[-1]}" if errs else ""}; {len(left)} game(s) missing ({tail})')
            self.log(f'batch {bt["name"]}: [{lo}, {hi}) played here in {self.clock() - t0:.0f} s')


# ------------------------------------------------------------------------------------------------ the evaluating Run
def _row_of(r):
    """A farm row -> the local driver's row of the network in seat A ((score, margin, rounds, over); A sits in seat
    gid % 2: the batch's seats for an even game id, reversed for an odd one)."""
    me = int(r['gid']) % 2
    margin = float(r['margin']) if me == 0 else -float(r['margin'])
    return (float(r['result'][me]), margin, r['rounds'], bool(r['over']))


def _make_run_class():
    from .run import Run, model_path

    class FarmRun(Run):
        """Run.evaluate + Run.elo_update of a farm evaluation's mirror (module docstring, 2): its matches between
        networks, or a network and a built-in bot, played as eval batches of `matches` (a FarmMatches)."""

        def __init__(self, run, cfg, matches):
            super().__init__(run, cfg)
            self.matches = matches
            self.played = []                       # (batch record, rows) of this evaluation

        def rec_kw(self, fn, **kw):
            """game_records: this class's matches are recorded too (Run.rec_kw records only Run's own methods, not a
            subclass's that plays other games: these are the same games, from their traces)."""
            return kw if self.cfg.get('game_records') else {}

        def _records(self, bt, rows, rec, seat_a, path_a, seat_b, b):
            """game_records (docs/az.md 59.3): each game's record from its trace (src 'farm', or 'local' for the
            deadline's fallback), the players as a local match names them."""
            from . import gamerec as GR
            sink = self.match_sink(rec[0], rec[1])
            if sink is None:
                return
            rule = self.eval_tag()
            pa = GR.net_player(path_a, rec[2], seat_a, rule)
            pb = GR.net_player(b[1], rec[3], seat_b, rule) if b[0] == 'net' else GR.bot_player(b[1], rec[3])
            spec = self.matches._spec(bt)
            local = ids_of(bt['local'])
            items = []
            for r in rows:
                g = int(r['gid'])
                src = 'local' if g in local else 'farm'
                x, sp = GR.from_trace(r['rec'], spec, rec[0], [pa, pb] if g % 2 == 0 else [pb, pa], rec[1], src,
                                      {'driver_seed': int(bt['seed']), 'paired': bool(bt['paired']),
                                       'farm_run': self.matches.farm, 'batch': bt['name']})
                if src == 'farm':
                    x['machine'] = GR.machine_tag((r.get('by') or {}).get('machine'))
                items.append((x, sp))
            sink.add(items)

        def _label(self, rec, fallback):
            if not rec:
                return fallback
            b = rec[3] if not isinstance(rec[3], dict) else ''
            return f'{rec[0]}-{rec[2]}-{b}'.replace('iter_', 'i').replace('anchor:', '')

        def play_match(self, it, opp, n_games, sims, seed, opp_model=None, paired=False, model=None, games=None,
                       rec=None):
            from . import games as G
            if n_games <= 0:
                return []
            if games is not None or G.is_external(opp):
                # (an external bot is a process of this machine; a chained game's contests carry episodes: here)
                self.log(f'eval pool: {opp} played here (not a farm match)')
                return super().play_match(it, opp, n_games, sims, seed, opp_model=opp_model, paired=paired,
                                          model=model, games=games, rec=rec)
            a = model or model_path(self.run, it)
            b = ('net', opp_model) if opp == 'nn' else ('bot', opp)
            bt, rows = self.matches.match(self.cfg, a, b, n_games, sims, seed, paired,
                                          self._label(rec, f'm-{elo.ckpt_name(it) if it is not None else "x"}-{opp}'))
            self.played.append((bt, rows))
            if rec:
                self._records(bt, rows, rec, self.nn_seat('m', sims), a, self.opp_seat(opp, sims), b)
            return [_row_of(r) for r in rows]

        def play_matches(self, it, opps, n_games, sims, seed, paired=False, paths=None, games=None, rec=None):
            if games is not None:
                self.log('eval pool: a chained game\'s matches played here (not a farm match)')
                return super().play_matches(it, opps, n_games, sims, seed, paired=paired, paths=paths, games=games,
                                            rec=rec)
            out = {}
            a = model_path(self.run, it)
            for t, j in enumerate(opps):
                op = paths[j] if paths and j in paths else model_path(self.run, j)
                lb = (rec[3] or {}).get(j, j) if rec else j
                r1 = (rec[0], rec[1], rec[2], lb) if rec else None
                bt, rows = self.matches.match(self.cfg, a, ('net', op), n_games, sims, seed, paired,
                                              self._label(r1, f'm-{elo.ckpt_name(it)}-{j}'))
                self.played.append((bt, rows))
                if r1:
                    self._records(bt, rows, r1, self.nn_seat('m', sims), a, self.nn_seat(f'o{t}', sims), ('net', op))
                out[j] = [_row_of(r) for r in rows]
            return out

        def pool_summary(self):
            """How this evaluation's games were played (the evals.jsonl row's `eval_pool`): batches, games, the farm's
            and this machine's, how many machines, the spot check, seconds."""
            games = farm = local = checked = 0
            machines = set()
            local_s = 0.0
            for bt, rows in self.played:
                nl = len(ids_of(bt['local']))
                games += len(rows)
                local += nl
                farm += len(rows) - nl
                machines |= set(bt['machines'])
                checked += (bt.get('spot') or {}).get('checked', 0)
                local_s += bt.get('local_s', 0)
            return {'farm_run': self.matches.farm, 'batches': [bt['name'] for bt, _ in self.played], 'games': games,
                    'farm': farm, 'local': local, 'local_s': round(local_s, 1), 'machines': len(machines),
                    'spot': checked}

    return FarmRun


FarmRun = None


def farm_run_class():
    global FarmRun
    if FarmRun is None:
        FarmRun = _make_run_class()
    return FarmRun


# ------------------------------------------------------------------------------------------------ the run's side
class FarmEval(RemoteEval):
    """The run `R`'s evaluations on the farm run `name` ('pool:<farm run>'; module docstring, 1). `matches`: a
    FarmMatches (tests: one on a fake coordinator)."""

    def __init__(self, R, name, matches=None):
        farm = str(name)[len(PREFIX):]
        rem = R.cfg.get('selfplay_remote') or ''
        names = [x.strip() for x in (rem.split(',') if isinstance(rem, str) else rem) if str(x).strip()]
        if name in names:
            raise SystemExit(f'eval_remote {name!r} is the run\'s self-play pool: its evaluation needs a farm run of its '
                             f'own (the pool PUTs eval null and signs no eval template), e.g. {PREFIX}{farm}-eval')
        if R.eval_chained():
            raise SystemExit(f'eval_remote {name!r}: the evaluation spec chains its episodes (docs/az.md 42) and a farm '
                             f'eval row carries no episodes: evaluate this run here or on a remote')
        self.matches = matches or FarmMatches(R.cfg, R.run, farm, R.log)
        workdir = os.path.join(self.matches.dir, 'mirror').replace('\\', '/')
        spec = {'host': 'local', 'repo': package.active().root, 'python': sys.executable, 'workdir': workdir,
                'env': {}, 'overrides': {'infer_device': 'cpu', 'train_device': 'cpu'}}
        super().__init__(R, name, spec=spec, transport=LinkTransport(spec))

    def describe(self):
        return f'{self.name!r}: {self.matches.describe()}, mirror {self.rdir}'

    def _result(self, it):
        made = []

        def make(rdir, cfg):
            r = farm_run_class()(rdir, cfg, self.matches)
            made.append(r)
            return r
        m = evaluate_mirror(self.rdir, it, make_run=make, src='farm')
        m['caps'] = ['tails']                      # (the run's own code, in this process)
        if made:
            m['eval_pool'] = made[0].pool_summary()
        return m

    def _row_extras(self, m):
        return {'eval_pool': m['eval_pool']} if m.get('eval_pool') else {}
