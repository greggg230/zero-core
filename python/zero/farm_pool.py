"""The farm pool (zero-farm DESIGN.md 7, docs/az.md 45): a run's self-play also on friends' idle PCs, through the
zero-farm coordinator - `selfplay_remote` names `pool:<farm run>` (alone or in a list with remote hosts).

Unlike a remote host it has no sessions and never holds a share of the iteration's quota: it is one thread of the
run process (`Pool`, owned by zero.remoteplay.RemotePlay) that

1. **publishes** checkpoint it - 1 when iteration it's self-play begins (RemotePlay.begin, right after the
   checkpoint is written): the checkpoint exported to zero-onnx/1 (or /2, docs/az.md 46) with its reference set
   (zero.onnx_export, checked through onnxruntime; by weight swap into the architecture's template, on the run's kept
   reference positions: zero.farm_h2h.export_cached, docs/az.md 53) under `<run>/pool/<farm run>/ckpt/<gen>/`, uploaded
   (`ckpt/<farm run>/<gen>/...`), a NEW signed definition version (zero.farm_def: the worker build and its
   alternates' builds, its runtime, the gen and up to `pool_max_stale` gens before it, the self-play template
   {settings: the worker build's own `check-config` play settings}) published with farm-release - and only then the
   gen registered current (`PUT .../checkpoint/<gen>`): farm leases switch only to gens a definition names. The run
   on the coordinator is created on the first publish (`selfplay`: quota `pool_quota` (default the run's `games`),
   slack, max_stale, games per lease, needs).
2. **ingests** records (zero-farm PROTOCOL.md 5 `records`, zero-core docs/search.md 2.17): it polls
   `GET .../results`, downloads each self-play chunk, checks its header (the run's game, features, actions, spec,
   auxiliary targets, base seed and worker build - the default's or an alternate's), replays every credited game
   through the engine (`<pkg>._engine.replay_rows`: the rows bit for bit as the worker's shards would hold them;
   replay is the validation: an illegal move, a wrong decider or option count, another result, margin or setup - and
   a game seed that is not its id's - refuses the whole record) and keeps the game, its rows compacted, until the
   quota takes it. A refused record is counted against its machine (`<run>/debug/pool/refused.jsonl`, the chunk kept
   under `<run>/debug/pool/<machine>/`).
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
first agent that accepts a template naming a split or double-faced card; '' = none), `pool_alternates` and
`pool_alt_vendor` (below), `pool_gpu_games_per_lease` (the farm run's `selfplay.gpu_games_per_lease`, PROTOCOL rev
11, 9.4: the games of a GPU lease, 'auto' or a number; '' = the key left out, GPU leases sized as before). The admin
token as the farm tools read it (zero.farm_def.read_token). State: `<run>/pool/<farm run>/state.json`.

Run alternates (zero-farm PROTOCOL rev 11, 9.4): `pool_alternates` (worker build ids, comma-separated or a list, in
order of preference, at most 4; '' = none) names worker builds that are the same engine signed for another runtime pack
(the CUDA pack: offered only to PCs with an NVIDIA GPU that may run GPU work, agents from 0.2.10). On the first publish
of a pool process they are checked (zero.farm_def.run_alternates: a worker of the default build's game requiring a
runtime pack; `when.vendor` the pack's `gpu.vendor`, else `pool_alt_vendor`, '' = none), then sent in every PUT of the
farm run (`"alternates"`) and their builds added to every definition version's `builds` after the default one (its
`runtime` stays the default pack). A record whose header names an alternate's build is accepted like the default's
(the same engine; its manifest, next to the executable, gives the worker its build id). Changed on a resumed run, the
new alternates (or `pool_gpu_games_per_lease`) are PUT again once the next gen's definition (which lists their
builds) is registered (state `put_extras`: what the last self-play PUT carried of them). None (the default): no
`alternates` key and builds [pool_build] - every PUT body and definition exactly as before.

Rolling leases (zero-farm PROTOCOL rev 15, section 10; docs/az.md 54): `pool_rolling` ('auto', the default since
2026-11-17 - a deliberate change, docs/az.md 54.7; '' = off: every PUT, definition and staleness judgement exactly as
before rolling leases, and what a worker build without the feature `rolling` needs) is the farm run's `selfplay.rolling` - a JSON object with
the keys `inflight`, `gpu_inflight` (1 to 4,096 or 'auto'), `block`, `max_s`, `flush_s` (`rolling_setting`), or 'auto'
= {"inflight": "auto", "gpu_inflight": "auto"}. Set, the first publish of a pool process checks that the worker
build plays rolling jobs (`pool_worker`'s `info` lists the feature `rolling`; each alternate whose executable is the
default's - the same sha256 in its signed manifest - is that executable, another one is checked when it is next to
`pool_worker`'s build directory, else logged as unchecked) and refuses loudly otherwise; every self-play PUT carries
`rolling` (also PUT again on a resumed run whose setting changed: `put_extras`); and staleness is judged per row: a
held game's rows searched with a network older than i - 1 - `pool_max_stale` are dropped (`stale_rows_dropped`), the
rest of the game is kept, and the game is dropped (`stale_dropped`) only when none is left - a rolling lease lives
for hours, switching networks in place, so its games straddle generations.

Lock order (docs/az.md 55; the pb3 hang of 2026-11-16): `Pool.dmu` -> `RemotePlay.mu` -> `Pool.mu`, outermost first.
`Pool.mu` is innermost: a Pool never calls into its RemotePlay (`pool_iteration`, `accept_pool`, `add_failures`) while
it holds `Pool.mu`, and RemotePlay may call the Pool methods that take only `Pool.mu` (`close`, `begin`, `progress`)
with or without its own mu. `Pool.dmu` (the drain lock) is held across a whole `_drain`, so a game the quota accepted
is in its shard writer before `end_phase` (which takes `dmu` first) closes the iteration's writers: RemotePlay never
calls `end_phase` holding its mu. The methods RemotePlay calls under its mu that take no Pool lock: `live`, `describe`.
"""
import gzip
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
from collections import deque

from . import farm_def as FD
from . import gamerec as GR

PREFIX = 'pool:'
SEMVER_RE = FD.SEMVER_RE                    # (as the coordinator checks a version: zero-farm coordinator util.ts)
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


def header_failures(h):
    """A records file header's `failures` (games the worker dropped after a panic, docs/search.md 2.19; absent in
    files of older workers and of jobs without one) -> failure dicts, each with the worker's `build` (its build_id)."""
    fails = h.get('failures') if isinstance(h, dict) else None
    if not isinstance(fails, list):
        return []
    return [dict(f, build=h.get('build_id')) for f in fails if isinstance(f, dict)]


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


def alternate_ids(v):
    """`pool_alternates` -> its worker build ids: a comma- (or space-) separated string or a list; ''/None/[] = none."""
    if v is None:
        return []
    items = v.replace(',', ' ').split() if isinstance(v, str) else [str(x).strip() for x in v]
    return [x for x in items if x]


# a variant's form: 'fp16' = onnx_export.to_float16 of the gen's graph; 'matmul' = the gen's checkpoint exported as
# graph 'fast-matmul' (onnx_export.GRAPHS: the fast graph with its attention as MatMuls - the one zero-onnx/2 graph
# DirectML runs, docs/az.md 46.10), float32 - the AMD GPUs' form (zero-farm PROTOCOL rev 22: AMD GPUs through DirectML;
# their DirectML answers the float16 copy non-deterministically, 46.10); 'matmul-fp16' = that graph in float16 (DirectML
# on NVIDIA and Intel GPUs runs it deterministically)
VARIANT_FORMS = ('fp16', 'matmul-fp16', 'matmul')


def variant_setting(v):
    """The run setting `pool_variants` (docs/az.md 67, 46.10; zero-farm PROTOCOL rev 20, 14) -> [(name, form, vendor)]:
    '' / None -> []; 'name=form[@vendor][,...]' - each name `farm_def.VARIANT_RE`, distinct, at most `MAX_VARIANTS`; each
    form one of VARIANT_FORMS; `vendor` (optional, a profile vendor name `farm_def.VENDORS`) - the variant goes to the run's
    GPU alternates of that vendor; a variant without one to every other GPU alternate (the first such variant), None in
    the triple. At most one variant per vendor. ValueError otherwise."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return []
    out = []
    for item in str(v).replace(' ', '').split(','):
        name, _, rest = item.partition('=')
        form, _, vendor = rest.partition('@')
        if not FD.VARIANT_RE.match(name) or form not in VARIANT_FORMS or ('@' in rest and vendor not in FD.VENDORS):
            raise ValueError(f'variant {item!r}: name=form[@vendor], the name [a-z0-9][a-z0-9-]{{0,31}}, the form one of '
                             f'{VARIANT_FORMS}, the vendor one of {FD.VENDORS}')
        out.append((name, form, vendor or None))
    if len({n for n, _, _ in out}) != len(out) or len(out) > FD.MAX_VARIANTS:
        raise ValueError(f'variants {v!r}: distinct names, at most {FD.MAX_VARIANTS}')
    vendors = [x for _, _, x in out if x]
    if len(set(vendors)) != len(vendors):
        raise ValueError(f'variants {v!r}: at most one variant per vendor')
    return out


def variant_of(variants, vendor):
    """The variant name a GPU alternate of `vendor` plays (`variant_setting`'s list): the one for its vendor, else the
    first without a vendor, else None (the gen's default model)."""
    return next((n for n, _, x in variants if x == vendor), None) or next((n for n, _, x in variants if x is None), None)


def version_tuple(v):
    """'0.3.5' -> (0, 3, 5) (a pre-release's suffix ignored); None -> ()."""
    return tuple(int(x) for x in v.split('-')[0].split('.')) if v else ()


ROLLING_KEYS = ('inflight', 'gpu_inflight', 'block', 'max_s', 'flush_s')
_ROLLING_LIMITS = {'inflight': (1, 4096), 'gpu_inflight': (1, 4096), 'block': (1, 1 << 32), 'max_s': (600, 86400),
                   'flush_s': (10, 3600)}


def rolling_setting(v):
    """The run setting `pool_rolling` -> the farm run's `selfplay.rolling` (zero-farm PROTOCOL 10.1) or None (off):
    '' / None -> None; 'auto' -> {"inflight": "auto", "gpu_inflight": "auto"}; a dict (or its JSON text) with keys of
    ROLLING_KEYS: inflight / gpu_inflight 1 to 4,096 or 'auto', block >= 1, max_s 600 to 86,400, flush_s 10 to 3,600
    (missing keys: the coordinator's defaults). ValueError otherwise (an unknown key, a value out of range)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, str):
        if v.strip() == 'auto':
            return {'inflight': 'auto', 'gpu_inflight': 'auto'}
        try:
            v = json.loads(v)
        except ValueError:
            raise ValueError(f"rolling {v!r}: 'auto' or a JSON object of {ROLLING_KEYS}") from None
    if not isinstance(v, dict) or not v:
        raise ValueError(f"rolling {v!r}: 'auto' or a non-empty object of {ROLLING_KEYS}")
    bad = [k for k in v if k not in ROLLING_KEYS]
    if bad:
        raise ValueError(f'rolling: unknown key(s) {bad} (keys: {ROLLING_KEYS})')
    out = {}
    for k in ROLLING_KEYS:
        if k not in v:
            continue
        x = v[k]
        if k in ('inflight', 'gpu_inflight') and x == 'auto':
            out[k] = x
            continue
        lo, hi = _ROLLING_LIMITS[k]
        if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
            auto = " or 'auto'" if k in ('inflight', 'gpu_inflight') else ''
            raise ValueError(f'rolling: {k} {x!r}: an integer {lo} to {hi}{auto}')
        out[k] = x
    return out


def worker_features(worker):
    """The protocol features a worker executable announces (`<worker> info`'s `features`, zero-farm PROTOCOL 10.5;
    absent: none) and its build id."""
    from .farm_h2h import worker_json
    rc, info, text = worker_json(worker, 'info', '--ort-dll', 'none')
    if rc != 0 or not isinstance(info, dict):
        raise SystemExit(f'{worker} info: {text.strip()[-500:]}')
    return list(info.get('features') or []), info.get('build_id')


def _entry_sha(m):
    """A worker manifest's executable: its `entry` file's sha256 (None if it lists no such file)."""
    return next((f.get('sha256') for f in m.get('files') or [] if f.get('path') == m.get('entry')), None)


def keep_rows(s, rows, widths=None):
    """Rows `rows` (ascending indices) of a held game's sample dict: every per-row key, the options' arrays by
    `opt_off`, the sparse features (`xs`), the tail and override CSRs by their offsets (`widths`: the tail segments,
    games.tail_widths' form, needed with a tail) - protocol.take_rows for any set of rows."""
    import numpy as np
    from .data import SparseRows, _ranges
    from .protocol import ROW_SAMPLE_KEYS
    from .rows import Overrides, Tail
    rows = np.asarray(rows, np.int64)
    off = np.asarray(s['opt_off'], np.int64)
    lo, n = off[rows], off[rows + 1] - off[rows]
    pos = _ranges(lo, n)
    out = {}
    for k, v in s.items():
        if k == 'opt_off':
            out[k] = np.concatenate([np.zeros(1, np.int64), np.cumsum(n)]).astype(np.asarray(v).dtype)
        elif k in ('opt_action', 'pi'):
            out[k] = v[pos]
        elif isinstance(v, SparseRows):
            out[k] = v.take(rows)
        elif k not in ROW_SAMPLE_KEYS:
            out[k] = v[rows]
    if 'ov_off' in s:
        o = Overrides(np.asarray(s['ov_off'], np.int64), s['ov_opt'], s['ov_tok']).take(rows)
        out.update(ov_off=o.off, ov_opt=o.opt, ov_tok=o.tok)
    if 't_off' in s:
        if widths is None:
            raise ValueError('keep_rows: a sample dict with a tail needs its tail widths')
        t = Tail.from_arrays(s, widths).take(rows)
        out.update(t_off=t.off, t_seg=t.seg, t_slot=t.slot, t_val=t.val)
    return out


def play_settings(worker, build_id, cfg, d, log=None):
    """The self-play template's settings: the keys the worker build's own `check-config --json` calls `play` (what
    its self-play driver reads; never a trainer-only key or a path, which farm-release refuses in a template),
    checked again on their own, and giving the driver every setting gives (`driver-cfg`). A setting the build
    doesn't know but this zero-core classifies trainer-only is logged and left out (zero.farm_h2h.config_refused:
    a worker build older than the pool's own settings). A build announcing the feature `setting_classes` checks the
    settings with the trainer's declared classes - its own table and the game package's - and the template carries
    them, signed with it (zero.farm_h2h.with_classes, docs/az.md 61): a game package's new trainer-only setting needs no
    new worker build."""
    from .farm_h2h import CLASSES_KEY, _stderr_log, config_refused, log_declared, with_classes, worker_json
    log = log or _stderr_log
    rc, info, text = worker_json(worker, 'info', '--ort-dll', 'none')
    if rc != 0 or not isinstance(info, dict):
        raise SystemExit(f'{worker} info: {text.strip()[-500:]}')
    if info.get('build_id') != build_id or 'selfplay' not in (info.get('kinds') or []):
        raise SystemExit(f'{worker} is build {info.get("build_id")} ({info.get("kinds")}), not the self-play worker build '
                         f'{build_id}: pool_worker must be that build\'s executable')
    os.makedirs(d, exist_ok=True)
    full = os.path.join(d, 'settings-run.json')
    run = with_classes(info, cfg)
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(run, f)
    rc, out, text = worker_json(worker, 'check-config', full, '--json')
    if config_refused(rc, out, build_id, log):
        raise SystemExit(f'the worker refuses the run\'s settings for self-play:\n{text.strip()[-3000:]}')
    log_declared(out, build_id, log)
    status = {s['key']: s['status'] for s in out.get('settings') or []}
    play = {k: v for k, v in cfg.items() if status.get(k) == 'play' and k != CLASSES_KEY}
    if CLASSES_KEY in run:                     # (the declaration signed with the template: docs/az.md 61)
        play[CLASSES_KEY] = run[CLASSES_KEY]
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


def platform_check(worker, cfg, d, platform, log=None):
    """A worker of another platform may take the run's leases only if the run's settings replay exactly there (docs/az.md
    62, zero-farm PROTOCOL 11.4): the pool's own worker build (pool_worker, the default platform's) judges them as a
    `platform` worker would (`check-config --platform`, the game's libm rule) - a refusal, or a build too old to judge
    (its answer names no platform), stops the pool before any PUT names that platform's alternate."""
    from .farm_h2h import _stderr_log, with_classes, worker_json
    log = log or _stderr_log
    if not worker:
        raise SystemExit(f'a {platform} alternate needs pool_worker (the build\'s executable here) to check the run\'s '
                         f'settings for {platform}')
    rc, info, text = worker_json(worker, 'info', '--ort-dll', 'none')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'settings-{platform}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(with_classes(info if isinstance(info, dict) else {}, cfg), f)
    rc, out, text = worker_json(worker, 'check-config', path, '--json', '--platform', platform)
    if not isinstance(out, dict) or out.get('platform') != platform:
        raise SystemExit(f'pool_worker {worker} cannot judge the run\'s settings for {platform} (a build before zero-core '
                         f'docs/az.md 62): no {platform} alternate without that check')
    if out.get('libm_refusals'):
        raise SystemExit(f'the run\'s settings would replay differently on {platform} (the platform\'s libm, docs/az.md '
                         f'62): {"; ".join(out["libm_refusals"])} - leave its {platform} alternate out')
    log(f'{platform}: the run\'s settings replay exactly there (check-config --platform {platform})')


def export_breakdown(ex):
    """The "gen N exported" line's per-step timing (docs/az.md 53): ': load 0.1 s, swap 0.4 s, reference 0.5 s, check
    1.2 s (graph by weight swap; reference positions of gen 12)' from an exporter's `steps`, `graph` and `positions`;
    '' without steps (an exporter that reports none)."""
    st = ex.get('steps') or {}
    if not st:
        return ''
    how = {'template': "the architecture's template built first", 'swap': 'graph by weight swap'}.get(ex.get('graph'))
    tail = '; '.join(x for x in (how, None if ex.get('positions') is None else
                                 f'reference positions of gen {ex["positions"]}') if x)
    return ': ' + ', '.join(f'{k} {v:g} s' for k, v in st.items()) + (f' ({tail})' if tail else '')


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
        # run alternates (module docstring): their ids here, checked against their manifests on the first publish
        self.alt_builds = alternate_ids(setting('pool_alternates'))
        bad = [b for b in self.alt_builds if not FD.BUILD_RE.match(b)]
        if bad or len(self.alt_builds) > FD.MAX_ALTERNATES or len(set(self.alt_builds)) != len(self.alt_builds):
            raise SystemExit(f'selfplay_remote {name!r}: pool_alternates {self.alt_builds}: up to {FD.MAX_ALTERNATES} '
                             f'distinct worker build ids (24 hex)')
        self.alt_vendor = str(setting('pool_alt_vendor') or '').strip() or None
        if self.alt_vendor and self.alt_vendor not in FD.VENDORS:
            raise SystemExit(f'selfplay_remote {name!r}: pool_alt_vendor {self.alt_vendor!r}: one of {FD.VENDORS}')
        self.alternates = None                    # (PUT form, from run_alternates on the first publish)
        self.builds = None
        # model variants per gen (docs/az.md 67, 46.10; zero-farm PROTOCOL rev 20, 14): each gen's graph again in each
        # form, signed and registered beside the default; the run's GPU alternates name theirs (`model`, variant_of: by
        # vendor). A run with variants needs agents that read them: its min_agent at least farm_def.VARIANT_AGENT
        try:
            self.variants = variant_setting(setting('pool_variants'))
        except ValueError as e:
            raise SystemExit(f'selfplay_remote {name!r}: pool_variants: {e}') from None
        if self.variants and version_tuple(self.min_agent) < version_tuple(FD.VARIANT_AGENT):
            self.min_agent = FD.VARIANT_AGENT
        self.variant_maker = self._make_variant
        try:                                      # GPU leases' games (PROTOCOL rev 11, 9.4; None: the key left out)
            self.gpu_per_lease = FD.gpu_games_per_lease(setting('pool_gpu_games_per_lease'))
        except ValueError as e:
            raise SystemExit(f'selfplay_remote {name!r}: pool_{e}') from None
        try:                                      # rolling leases (PROTOCOL rev 15, 10; None: off, all as before)
            self.rolling = rolling_setting(setting('pool_rolling'))
        except ValueError as e:
            raise SystemExit(f'selfplay_remote {name!r}: pool_{e}') from None
        self.rolling_checked = False
        # the graph each gen is exported as (docs/az.md 63: '' the default graph, 'fast' the fast graph)
        self.export_graph = str(setting('pool_export') or '')
        try:
            from .onnx_export import graph_options
            graph_options(self.export_graph)
        except ValueError as e:
            raise SystemExit(f'selfplay_remote {name!r}: pool_export: {e}') from None
        self.client = client or FD.Client(self.url)
        self.exporter = exporter or self._export
        self.definer = definer or self._define
        self.settings_of = settings_of or (lambda spec=None: play_settings(
            self.worker, self.build, dict(c) if spec is None else dict(c, spec=spec),
            os.path.join(self.dir, 'settings'), self.log))
        self.phase_specs = {}                      # iteration -> its phase spec, None = the run's (docs/az.md 52)
        self.dir = os.path.join(self.R.run, 'pool', self.farm)
        self.debug = os.path.join(self.R.run, 'debug', 'pool')
        os.makedirs(self.dir, exist_ok=True)
        self.st = self._load()
        self.seed = coordinator_seed(self.farm)
        from . import games as G
        self.spec = G.spec_json(c['spec'])
        self.players = [int(p) for p in c['players']]
        self.aux = bool(getattr(self.R, 'aux', None))
        # (lock order, module docstring: dmu -> RemotePlay.mu -> mu; never a RemotePlay call while holding mu)
        self.mu = threading.Lock()                 # the held games, the iteration's writers, the counters
        self.dmu = threading.Lock()                # a whole _drain; end_phase waits for it (accepted games written)
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
                'stale_dropped': 0, 'stale_rows_dropped': 0, 'machines': {}, 'stale': {}, 'replay_s': 0.0}

    def live(self):
        """The farm is delivering: a valid game arrived within `pool_live_s` (selfplay_local 0: this machine plays no
        games meanwhile - RemotePlay._refresh; once the farm goes quiet for that long, it plays again)."""
        t = self.t_game
        return t is not None and time.time() - t < self.live_s

    def describe(self):
        alts = f', alternates {",".join(self.alt_builds)}' if self.alt_builds else ''
        roll = f', rolling {json.dumps(self.rolling)}' if self.rolling is not None else ''
        return f'{self.name!r} (farm {self.url}, build {self.build}{alts}, max_stale {self.max_stale}{roll})'

    # ------------------------------------------------------------ the run's side (RemotePlay)
    def begin(self, it, spec=None):
        """Iteration `it`'s self-play begins: checkpoint it - 1 is published on the pool's thread (never waited for).
        `spec`: the iteration's phase spec of the game package's (docs/az.md 52; None: the run's) - the definition that
        signs gen it - 1 carries it in its self-play template, and the records played with it are accepted."""
        with self.mu:
            self.phase_specs[it] = spec
        self.want = it
        self.wake.set()

    def _gen_spec(self, gen):
        """The spec gen `gen`'s leases play (the phase spec of iteration gen + 1, else the run's): JSON text."""
        with self.mu:
            s = self.phase_specs.get(gen + 1)
        return s if s is not None else self.R.cfg['spec']

    def _play_of(self, spec):
        """The template's play settings for `spec`: the run's settings (`settings_of()`, as before) for the run's spec,
        else the same with the spec replaced, checked by the worker build (`settings_of(spec)`)."""
        if spec == self.R.cfg['spec']:
            return self.settings_of()
        return self.settings_of(spec)

    def known_spec(self, s):
        """The spec of a record's header among the specs this pool's gens were defined with (docs/az.md 52), as the
        run's text -> the text, or None."""
        want = _spec_value(s)
        for x in [self.spec] + [g.get('spec') for g in self.st['gens'].values() if g.get('spec')]:
            if _spec_value(x) == want:
                return x
        return None

    def end_phase(self, it):
        """Iteration `it`'s self-play ended (RemotePlay.end, never under its mu): its pool shards are closed -> its share
        (metrics). Waits for a running `_drain` (dmu): every game the quota accepted is written first."""
        with self.dmu, self.mu:
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
            if self.rolling is not None:           # (per-row staleness; off: the metrics of before)
                out['stale_rows_dropped'] = cur['stale_rows_dropped']
        if self.error:
            out['error'] = self.error
        return out

    def progress(self):
        with self.mu:
            return {'farm_run': self.farm, 'games': self.cur['games'], 'held': len(self.held),
                    'refused': self.cur['refused'], 'gen': self.st.get('current_gen'),
                    'error': self.error}

    def close(self):
        """(RemotePlay may call it under its mu: takes only mu, never dmu.) A drain that is accepting meanwhile closes
        the writers it opens after this (`_drain`: `closed` read under mu)."""
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
        """Gen `gen`'s graph and reference set (zero.farm_h2h.export_cached, docs/az.md 53: the architecture's
        weight-swap template under <pool>/export/, the run's reference positions kept there)."""
        from . import games as G
        from .farm_h2h import export_cached
        return export_cached(ckpt, gen, out_dir, self.R.cfg['game'], G.spec_json(self.R.cfg['spec']),
                             os.path.join(self.dir, 'export'), f'{self.farm}/{gen}', checkpoint_iter=gen,
                             graph=self.export_graph)

    def _make_variant(self, form, src, dst, ckpt=None, gen=None):
        """Gen `gen`'s variant in form `form` (VARIANT_FORMS) written to `dst`: from its graph `src` ('fp16'), or from
        its checkpoint `ckpt` exported again as another graph ('matmul' / 'matmul-fp16': export_cached's graph
        'fast-matmul' under <dst's dir>/v-<form>/, its own weight-swap template and check against the reference set;
        'matmul-fp16' then float16)."""
        from .onnx_export import to_float16
        if form == 'fp16':
            to_float16(src, dst)
        elif form in ('matmul', 'matmul-fp16'):
            from . import games as G
            from .farm_h2h import export_cached
            ex = export_cached(ckpt, gen, os.path.join(os.path.dirname(dst), f'v-{form}'), self.R.cfg['game'],
                               G.spec_json(self.R.cfg['spec']), os.path.join(self.dir, 'export'), f'{self.farm}/{gen}',
                               checkpoint_iter=gen, graph='fast-matmul')
            if form == 'matmul':
                shutil.copyfile(ex['model'], dst)
            else:
                to_float16(ex['model'], dst)
        else:
            raise ValueError(f'variant form {form!r}')

    def _define(self, client, info):
        return FD.publish(client, info, key=self.key, farm_release=self.farm_release, log=self.log)

    def _selfplay_body(self, play):
        sp = {'quota': self.quota, 'slack': self.slack, 'max_stale': self.max_stale, 'games_per_lease': self.per_lease}
        if self.gpu_per_lease is not None:          # (PROTOCOL rev 11; None: the key is absent, the body of before)
            sp['gpu_games_per_lease'] = self.gpu_per_lease
        sp.update(gpu='any', job={})
        if self.needs_ram:
            sp['needs'] = {'ram_mb': self.needs_ram}
        if self.rolling is not None:               # (PROTOCOL rev 15; off: the key is absent, the body of before)
            sp['rolling'] = self.rolling
        b = {'game': self.R.cfg['game'], 'build_id': self.build, 'runtime': self.runtime, 'priority': self.priority,
             'min_agent': self.min_agent}
        if self.alternates:                         # (PROTOCOL rev 11; none: the key is absent, the body of before)
            b['alternates'] = self.alternates
        b.update(selfplay=sp, eval=None, state='active')
        return b

    def _put_extras(self):
        """The rev 11 fields this pool's PUTs carry beyond the body of before ({} with the defaults): a state's
        `put_extras` is what the last self-play PUT sent (absent: none of them)."""
        out = {}
        if self.variants:
            out['min_agent'] = self.min_agent       # (rev 20: at least VARIANT_AGENT, put again on a resumed run)
        if self.alternates:
            out['alternates'] = self.alternates
        if self.gpu_per_lease is not None:
            out['gpu_games_per_lease'] = self.gpu_per_lease
        if self.rolling is not None:
            out['rolling'] = self.rolling
        return out

    def _check_rolling(self):
        """`pool_rolling` set: the worker build must play rolling jobs (module docstring) - SystemExit otherwise (the
        pool's thread logs it as refused and tries again; nothing is published meanwhile)."""
        if not self.worker:
            raise SystemExit("pool_rolling is set but pool_worker (the build's executable here) is not: its features "
                             "cannot be checked")
        feats, bid = worker_features(self.worker)
        if bid != self.build or FD.FEATURE_ROLLING not in feats:
            raise SystemExit(f'pool_rolling is set but pool_worker {self.worker} (build {bid}) announces features '
                             f'{feats}, not {FD.FEATURE_ROLLING!r} of build {self.build}: a worker built before rolling '
                             f'leases (zero-farm PROTOCOL rev 15) - rebuild it, or unset pool_rolling')
        if self.alt_builds:
            base = _entry_sha(FD.build_manifest(self.client, self.build))
            here = os.path.dirname(os.path.dirname(os.path.abspath(self.worker)))
            for b in self.alt_builds:
                m = FD.build_manifest(self.client, b)
                sha = _entry_sha(m)
                if sha is not None and sha == base:
                    continue                       # (the same executable, signed for another pack)
                plat = m.get('platform') or FD.PLATFORM_DEFAULT
                if plat != FD.host_platform():     # (a farm release keeps a macOS build's executable here too)
                    self.log(f"pool_rolling: alternate {b} is a {plat} build, which this machine cannot run: its rolling "
                             f"support is unchecked (a worker without it plays its rolling leases as classic ones)")
                    continue
                exe = os.path.join(here, b, m.get('entry') or '')
                if m.get('entry') and os.path.isfile(exe) and FD.file_sha256(exe) == sha:
                    feats, abid = worker_features(exe)
                    if abid != b or FD.FEATURE_ROLLING not in feats:
                        raise SystemExit(f'pool_rolling is set but alternate {b} ({exe}) announces features {feats}, '
                                         f'not {FD.FEATURE_ROLLING!r}')
                    continue
                self.log(f"pool_rolling: alternate {b}'s executable is not the default build's and is not here: its "
                         f"rolling support is unchecked (a worker without it plays its rolling leases as classic ones)")
        self.log(f'pool_rolling {json.dumps(self.rolling)}: the worker build plays rolling jobs')

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
        # (docs/az.md 52: a gen whose iteration plays another spec than the last definition's gets a template of its own;
        # a pool of the run's spec only never comes here)
        gspec = self._gen_spec(gen)
        play = st['play']
        if 'spec' in play and _spec_value(play['spec']) != _spec_value(gspec):
            play = st['play'] = self._play_of(gspec)
            self._save()
            self.log(f'gen {gen}: the self-play template plays the game package\'s phase spec {hashlib.sha256(gspec.encode()).hexdigest()[:12]}')
        if not st.get('build_game'):
            st['build_game'] = FD.worker_game(self.client, self.build, self.runtime) if self.runtime else self.R.cfg['game']
            self._save()
        if self.alternates is None:
            # (none: [], [pool_build] without a call)
            alternates, builds = FD.run_alternates(self.client, self.build, self.runtime, self.alt_builds, self.alt_vendor)
            for p in sorted({x['when'].get('platform') for x in alternates} - {None}):
                platform_check(self.worker, self.R.cfg, os.path.join(self.dir, 'template'), p, self.log)
            if self.variants:
                # (rev 20, 14.3: each GPU alternate names its variant - its vendor's, else the first without a vendor
                # (variant_of); every other machine plays the default)
                gpu = [x for x in alternates if 'vendor' in x['when']]
                for x in gpu:
                    n = variant_of(self.variants, x['when']['vendor'])
                    if n:
                        x['model'] = n
                for n, _, _ in self.variants:
                    if not any(x.get('model') == n for x in gpu):
                        self.log(f'pool_variants {n}: no GPU alternate names it (each gen exported and registered '
                                 f'with it, played by no machine)')
            self.alternates, self.builds = alternates, builds
        if self.rolling is not None and not self.rolling_checked:
            self._check_rolling()
            self.rolling_checked = True
        if not st['run_put']:
            body = self._selfplay_body(st['play'])
            body['selfplay'] = None
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=body)
            st['run_put'] = True
            self._save()
            alts = ''.join(f', alternate {x["build_id"]} (runtime {x["runtime"]}, '
                           f'{" ".join(str(v) for v in x["when"].values())})' for x in self.alternates)
            self.log(f'farm run {self.farm} created (worker build {self.build}, runtime {self.runtime}{alts}, '
                     f'min_agent {self.min_agent or "none"})')
        g = st['gens'].setdefault(str(gen), {})
        if gspec != self.R.cfg['spec'] and g.get('spec') != gspec:
            g['spec'] = gspec                      # (the records of this gen's leases may carry it: known_spec)
            self._save()
        if not g.get('model'):
            ckpt = model_path(self.R.run, gen)
            t0 = time.time()
            ex = self.exporter(ckpt, gen, os.path.join(self.dir, 'ckpt', str(gen)))
            g.update({k: ex[k] for k in ('model', 'sha256', 'model_size', 'reference', 'reference_sha256', 'reference_size')})
            g['export_s'] = round(time.time() - t0, 1)
            if ex.get('steps'):
                g['export_steps'] = ex['steps']
            self._save()
            self.log(f'gen {gen} exported ({g["export_s"]} s, {g["sha256"][:12]}){export_breakdown(ex)}')
        if self.variants and not g.get('variants'):
            # (rev 20: the gen's variants beside its graph, model-<name>.onnx; the reference serves them all)
            t0 = time.time()
            ckpt_of_gen = model_path(self.R.run, gen)
            vs = {}
            for vname, form, _ in self.variants:
                vp = os.path.join(os.path.dirname(g['model']), f'model-{vname}.onnx')
                self.variant_maker(form, g['model'], vp, ckpt=ckpt_of_gen, gen=gen)
                vs[vname] = {'model': vp, 'sha256': FD.file_sha256(vp), 'model_size': os.path.getsize(vp)}
            g['variants'] = vs
            self._save()
            self.log(f'gen {gen} variants {", ".join(f"{k} ({v["sha256"][:12]})" for k, v in vs.items())} '
                     f'({round(time.time() - t0, 1)} s)')
        keys = {'model': f'ckpt/{self.farm}/{gen}/model.onnx', 'reference': f'ckpt/{self.farm}/{gen}/reference.npz'}
        vkeys = {k: f'ckpt/{self.farm}/{gen}/model-{k}.onnx' for k in g.get('variants') or {}}
        if not g.get('uploaded'):
            self.client.put_artifact(keys['model'], g['model'], g['sha256'])
            self.client.put_artifact(keys['reference'], g['reference'], g['reference_sha256'])
            g['uploaded'] = True
            self._save()
        if vkeys and not g.get('variants_uploaded'):
            for k, v in g['variants'].items():
                self.client.put_artifact(vkeys[k], v['model'], v['sha256'])
            g['variants_uploaded'] = True
            self._save()
        d = g.get('definition')
        if not d or not d.get('published'):
            if not d:
                gens = sorted((int(k) for k, v in st['gens'].items() if v.get('sha256')), reverse=True)
                keep = [x for x in gens if gen - self.max_stale <= x <= gen]
                cks = {str(x): {'model_sha256': st['gens'][str(x)]['sha256'], 'model_size': st['gens'][str(x)]['model_size'],
                                'reference_sha256': st['gens'][str(x)]['reference_sha256'],
                                'reference_size': st['gens'][str(x)]['reference_size']} for x in keep}
                for x in keep:                     # (rev 20, 14.1: each gen's variants as uploaded; none: no key)
                    vs = st['gens'][str(x)].get('variants') if st['gens'][str(x)].get('variants_uploaded') else None
                    if vs:
                        cks[str(x)]['variants'] = {k: {'model_sha256': v['sha256'], 'model_size': v['model_size']}
                                                   for k, v in vs.items()}
                d = FD.prepare(self.client, 'run', self.farm, game=st['build_game'], builds=self.builds, runtime=self.runtime,
                               checkpoints=cks, jobs={'selfplay': {'settings': st['play']}, 'eval': None, 'fuzz': None},
                               expires_days=self.expires_days, root=os.path.join(self.dir, 'definitions'))
                g['definition'] = d
                self._save()
            d = self.definer(self.client, d)
            g['definition'] = d
            st['definition'] = {k: d[k] for k in ('version', 'sha256', 'key')}
            self._save()
            self.log(f'definition version {d["version"]} signs gen {gen}')
        body = {'model': keys['model'], 'sha256': g['sha256'], 'reference': keys['reference'],
                'reference_sha256': g['reference_sha256']}
        if vkeys:                                  # (rev 20, 14.2: none - the body of before)
            body['variants'] = {k: {'model': vkeys[k], 'sha256': v['sha256']} for k, v in g['variants'].items()}
        self.client.call('PUT', f'/v1/admin/run/{self.farm}/checkpoint/{gen}', body=body)
        g['registered'] = True
        st['current_gen'] = gen
        # (alternates or pool_gpu_games_per_lease changed on a resumed run: PUT again, now that a definition lists the
        # alternates' builds; a state without `put_extras` sent neither)
        extras = self._put_extras()
        if not st['selfplay_put'] or st.get('put_extras', {}) != extras:
            self.client.call('PUT', f'/v1/admin/run/{self.farm}', body=self._selfplay_body(st['play']))
            st['selfplay_put'] = True
            if extras:
                st['put_extras'] = extras
            else:
                st.pop('put_extras', None)
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
        from .data import _rows_shard, compact, compact_rows, compact_sparse, shard_format
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
        if got['build_id'] in self.alt_builds:
            want['build_id'] = got['build_id']     # an alternate's worker: the same engine for another runtime pack
        if got != want:
            return self._refuse(r, f'header {got}, the run wants {want}', body)
        try:
            # (the run's spec, or a phase spec a gen of this pool was defined with: docs/az.md 52)
            rspec = self.known_spec(h.get('spec'))
            if rspec is None:
                return self._refuse(r, 'the records were played with another spec', body)
        except ValueError:
            return self._refuse(r, 'the spec is not JSON', body)
        fails = header_failures(h)
        if fails:
            # games the worker dropped after a panic (docs/search.md 2.19): to the run's game_failures.jsonl (source
            # 'pool': never the machine), with the worker's build
            self.rp.add_failures('pool', fails)
            self.log(f'{len(fails)} self-play game(s) dropped after a panic on a farm PC: to game_failures.jsonl')
        ids = [tuple(x) for x in (r.get('ids') or [])]
        gr = self.R.gamerecs() if hasattr(self.R, 'gamerecs') else None
        fmt = shard_format(self.R.cfg, self.R.info)     # (a game with tails / overrides: format v2, docs/az.md 43)
        sparse = fmt.get('x_format') == 'sparse'
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
                s, g = E.replay_rows(self.R.cfg['game'], rspec, ln, self.aux)
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
                    if _rows_shard(fmt):
                        c.update(compact_rows(s, fmt))      # (the tails' floats as stored: float16)
                else:
                    s['x'] = s['x'].astype('float16')
                    c = compact(s)
                probs = shard_problems(c)
                if probs:
                    return self._refuse(r, f'game {gid}: non-finite rows: {"; ".join(probs)}', body, gid)
                net = int(s['net'].min())
            item = {'gid': gid, 'machine': machine, 'net': net if net is not None else r.get('gen'), 's': s,
                    'rows': g['rows'], 'replay_s': dt}
            if gr is not None:
                # game_records (docs/az.md 59): the record's moves and outcome, written if the quota accepts the game
                item['rec'] = GR.from_farm(json.loads(ln), rspec, 0, self.name, machine)
            keep.append(item)
        with self.mu:
            self.held.extend(keep)
            self.cur['records'] += len(keep)
        if keep:
            self.t_game = time.time()

    def _engine(self):
        from . import games as G
        return G.engine()

    def _tail_widths(self):
        """The game's tail segments (None for a game without tails): `keep_rows`' widths."""
        from . import games as G
        return G.tail_widths(self.R.info or {})

    def _drain(self):
        """Held games, oldest first, into the open iteration's quota (RemotePlay.accept_pool); the stale ones dropped.
        Lock order (module docstring): RemotePlay is called with `mu` released - each game is chosen under `mu`, offered
        to the quota without it, then written under `mu` again; `dmu` is held throughout (`end_phase` waits for it).
        Only this thread takes games out of `held`, so the game offered is still its head when it is written."""
        with self.dmu:
            it = self.rp.pool_iteration()
            if it is None:
                return
            with self.mu:
                if self.cur['it'] != it:
                    self.cur = self._counters(it)
            while True:
                with self.mu:
                    g = None if self.closed else self._head(it)
                if g is None:
                    break
                if not self.rp.accept_pool(it, self.name, 1):     # (RemotePlay.mu: never under self.mu)
                    break
                with self.mu:
                    assert self.held and self.held[0] is g, 'a held game left the queue during its acceptance'
                    self.held.popleft()
                    self._accepted(it, g)
                    if self.closed:                # (closed meanwhile: the writers this drain opened are closed too)
                        for w in self.writers.values():
                            w.close()
                        self.writers.clear()
                        break

    def _head(self, it):
        """(mu held) The oldest held game that iteration `it` may take, the stale ones before it dropped (None: none).
        Rolling: a stale game's rows judged one by one - the head replaced by what is left of it."""
        import numpy as np
        lim = it - 1 - self.max_stale
        while self.held:
            g = self.held[0]
            if g['net'] is not None and g['net'] < lim:
                if self.rolling is not None and g['s'] is not None:
                    # (rolling: judged per row - the rows of a network too old dropped, the rest of the game kept)
                    nets = np.asarray(g['s']['net'])
                    ok = np.flatnonzero(nets >= lim)
                    self.cur['stale_rows_dropped'] += len(nets) - len(ok)
                    if len(ok):
                        s = keep_rows(g['s'], ok, self._tail_widths())
                        g = self.held[0] = dict(g, s=s, rows=len(ok), net=int(np.asarray(s['net']).min()))
                if g['net'] < lim:
                    self.held.popleft()
                    self.cur['stale_dropped'] += 1
                    continue
            return g
        return None

    def _accepted(self, it, g):
        """(mu held) Held game `g`, accepted by iteration `it`'s quota: counted and its rows into the machine's shards."""
        from .data import ShardWriter, shard_format
        from .workqueue import lag_hist
        self.st['accepted_total'] = self.st.get('accepted_total', 0) + 1
        self.cur['games'] += 1
        self.cur['replay_s'] += g['replay_s']
        m = g['machine']
        self.cur['machines'][m] = self.cur['machines'].get(m, 0) + 1
        if g.get('rec') is not None:
            # (game_records: Records' lock is innermost, below Pool.mu - it calls nothing back)
            # (the engine: this run's, which replayed the game and whose farm compatibility the worker build has -
            # its build_id beside it, docs/az.md 59.4)
            self.R.gamerecs().add(it, self.name, [g['rec']], build=dict(GR.build_info(), worker=self.build))
        if g['s'] is None:
            return
        key = (it, m)
        w = self.writers.get(key)
        if w is None:
            d = os.path.join(self.R.run, 'data', f'iter_{it:04d}')
            # (format v2's keys for a game with tails / overrides, docs/az.md 43: `shard_format` with the game's
            # info, as Run.shard_meta)
            meta = {'game': self.R.cfg['game'], 'features': self.R.info['num_features'],
                    'actions': self.R.info['num_actions'], 'iter': it, 'source': 'pool', 'machine': m,
                    'farm_run': self.farm, **shard_format(self.R.cfg, self.R.info)}
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
