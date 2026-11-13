"""A head-to-head of two checkpoints on zero-farm (friends' idle PCs; zero-farm PROTOCOL.md 5.2, docs/az.md 44.2):
the readout zero.h2h gives on this machine, played by the farm's native workers.

  python -m zero.farm_h2h --a runs/eoe1/models/iter_0080.pt --b runs/eoe1/models/iter_0070.pt --games 400 \
      --build <worker build_id> --runtime <runtime build_id> --worker <that build's mtgz-worker.exe>
      [--config <run>/config.json | --preset NAME] [--sims 32] [--name SLUG] [--games-per-lease auto] [--priority 1]
      [--needs-ram-mb 1500] [--reference-batches NPZ ...] [--url https://farm.greggg230.com] [--no-wait]
      [--key <data key>] [--farm-release <farm-release.exe>] [--expires-days 30] [--defs-dir DIR]
      [--alt <alternate worker build_id> ...] [--alt-vendor nvidia|amd|intel] [--min-agent 0.2.10]
      [--gpu-games-per-lease N|auto]
  python -m zero.farm_h2h --resume NAME        (re-attach: poll, download, print the readout)
  python -m zero.farm_h2h --cancel NAME        (pause the run: its leases are cancelled at their next beat)
  (python -m <pkg>.az.farm_h2h ...: the game package registered; or --package <pkg>.az [--root <checkout>])

What it does:
1. The settings: `--config` (a run's config.json) or `--preset`, else A's run's config.json (next to its
   models/), resolved as a run resolves them; `--sims` sets eval_sims; `eval_spec` resolved (the game
   package's eval spec of the run's spec, `Run.driver`'s rule: a worker cannot run the package's Python hook).
   Both seats play the run's evaluation seat (`Run.nn_seat(key, eval_sims)`), paired games: exactly the
   games of `Run.play_match` / an Elo match at those settings (zero.farm_eval). The job keeps only the settings
   the worker build's own `check-config --eval --json` calls `play` (`play_settings`: zero-farm PROTOCOL.md 7 -
   a signed template carries no trainer-only key and nothing path-like); `--worker` is that build's executable. A
   worker build older than a setting calls it "unknown": when this zero-core's crates/worker/src/run_settings.json
   classifies it trainer-only, that is logged and ignored (it never reaches the worker), any other failure refuses.
   The batch's base seed is the coordinator's, from the run's name (`coordinator_seed`): a template has no seed.
2. Both checkpoints exported to zero-onnx/1 or /2 (zero.onnx_export, 'packed') with their reference sets (256 rows
   of recorded batches - `--reference-batches`, else a short self-play of A recorded here, CPU - with torch's
   answers, checked through onnxruntime: <= 2e-5, else nothing is published).
3. The run `h2h-<slug>` (`PUT /v1/admin/run/h2h-<slug>`, selfplay null, the worker build and runtime); its signed
   definition (zero-farm PROTOCOL.md 7, zero.farm_def: builds [the worker build], the runtime, gens 1 and 2 with
   their graphs' and reference sets' sha256 and sizes, `jobs.eval` = the batch's job; the next version, signed with
   the data key and published by `farm-release define --publish`; it expires after `--expires-days`); the graphs
   and reference sets uploaded (`PUT /v1/admin/artifact/ckpt/<run>/<gen>/...`), registered as gens 1 (A) and 2 (B),
   then the eval batch: `checkpoints [1, 2]`, its job {settings, seats [{"net": 1}, {"net": 2}], paired} - the
   definition's template exactly (the coordinator issues leases only within it), `--games`.
   Run alternates (zero-farm PROTOCOL rev 11, 9.4; zero.farm_def.run_alternates): each `--alt` names a worker build
   that is the same engine signed for another runtime pack (the CUDA pack: offered only to PCs with an NVIDIA GPU that
   may run GPU work, agents from 0.2.10). Checked before the export (a worker of the build's game requiring a runtime
   pack); `when.vendor` is the pack's `gpu.vendor`, else `--alt-vendor` (for a pack without `gpu`; a different one is
   refused). They go into every PUT of the run (`"alternates"`, omitted without any: the body is then as before) and
   their builds into the definition's `builds` after the default one (its `runtime` stays the default pack); kept in
   state.json, so `--resume` and `--cancel` keep them (they are fixed when a head-to-head starts).
   `--min-agent` (PROTOCOL rev 9): the run's `min_agent` - its leases go only to agents at or above it (a canary-only
   head-to-head: 0.2.10); `--gpu-games-per-lease` (rev 11, 9.4): the eval batch's `gpu_games_per_lease` (the games
   of a GPU lease, `auto` or a number). Each goes into every PUT of the run only when given (otherwise the body is
   as before), kept in state.json like the alternates.
4. It polls `GET /v1/admin/run/<run>/results`, downloads each eval file (first copy of a game id wins;
   `dup_ids` skipped), acks what it saved, and prints a progress line; at the end the readout: A's score +- its
   standard error (pair-aware: the se of the pair scores), the Elo difference with its 95% interval, the games,
   the machines that played them, and the JSON line zero.h2h prints (`zero.h2h.summary`'s fields, plus run,
   machines, elo_95), so results compare directly with a local zero.h2h.

State: `~/.zero-farm/h2h/<run>/` (`--state-dir`): state.json (the run's body, the gens' files and hashes, the
definition's version and file, the results cursor), ckpt/<gen>/ (the exported graphs and reference sets), results/
(the downloaded files); the definition itself in the archive (~/.zero-farm/definitions/run/<run>/<version>/). Every
step is idempotent: `--resume` finishes whatever an interrupted start left. The admin token is read from
`%USERPROFILE%/.config/zero-farm/admin_token` (or env ZEROFARM_ADMIN_TOKEN) and never printed; farm-release gets
it in its environment, never on its command line.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

from .farm_def import (URL, Client, FarmError, file_sha256, read_token, token_path)  # noqa: F401  (the farm tools' client)
from . import farm_def as FD

SLUG_RE = FD.SLUG_RE
EXPIRES_DAYS = 30                       # a head-to-head's definition expires after this many days (--expires-days)


# ---------------------------------------------------------------- settings and names

def run_slug(a, b):
    """h2h-<run>-<A>v<B> from checkpoint paths runs/<run>/models/iter_0080.pt (iterations as numbers)."""
    def part(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r'^iter_0*(\d+)$', stem)
        return m.group(1) if m else stem
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a))))
    s = f'h2h-{run}-{part(a)}v{part(b)}'.lower()
    s = re.sub(r'[^a-z0-9-]+', '-', s).strip('-')[:63]
    return s


def settings_of(a, ckpt_a, game):
    """The eval job's settings (module docs, 1)."""
    from . import farm_eval as FE, run as R
    from .h2h import run_config
    ns = argparse.Namespace(**{k: None for k in R.DEFAULTS})
    if a.config:
        with open(a.config, encoding='utf-8') as f:
            saved = json.load(f)
    elif a.preset:
        saved = {'preset': a.preset}
    else:
        saved = run_config(ckpt_a)
        if not saved:
            raise SystemExit(f'no run config next to {ckpt_a} (runs/<run>/config.json): give --config or --preset')
    saved = dict(saved)
    if saved.get('game', game) != game:
        raise SystemExit(f'the settings play {saved.get("game")!r}, the checkpoints {game!r}')
    saved['game'] = game
    cfg = R.resolve_config(ns, saved)
    if a.sims is not None:
        cfg['eval_sims'] = int(a.sims)
    cfg = FE.resolve_eval_spec(cfg)
    FE.seat_dicts(cfg, [{'net': 1}, {'net': 2}])        # (Run.nn_seat's checks: refused here, not on a PC)
    return cfg


# ---------------------------------------------------------------- reference sets

VARLEN_KEYS = ('t_off', 't_seg', 't_slot', 't_val', 'ov_off', 'ov_opt', 'ov_tok')
MAX_RECORD_BATCHES = 4096


def record_rows(model, game, spec, rows=320, conc=32, sims=16, threads=4, seed=1):
    """`rows` real network batch rows (whole batches) of a short self-play of `model` (tools/onnx_record.py's
    seats: root belief rows when the network has the auxiliary head, root-view leaf values with view samples)
    -> onnx_parity.Rows with batch_off. A game with variable-length rows: the batches' tail tokens and overrides
    too, and the self-play goes on until the rows hold at least one tail token and one override of each kind the
    network takes (zero-onnx/2's self-test answers them, docs/az.md 46) - refused (RuntimeError) when
    `MAX_RECORD_BATCHES` batches hold none."""
    import numpy as np
    from . import games as G
    from .infer import NetEvaluator, drive
    from .onnx_parity import Rows
    ev = NetEvaluator(model, 'cpu')
    need_t = bool(getattr(model, 'tails', None))
    need_o = bool(getattr(model, 'option_pointers', False))
    tilt = 1.0 if getattr(model, 'aux_layout', None) else 0.0
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=max(1, sims // 2), p_full=0.25, leaf_value_view='root',
                     view_samples=0.5, belief_tilt=tilt)
    drv = G.Driver(game, [[seat, seat]] * (conc * 4), spec=spec, concurrency=conc, seed=seed, record=True,
                   threads=threads, groups=1, sparse=True)
    keep = []

    class Enough(Exception):
        pass

    def rec(lb):
        keep.append({k: np.array(lb.a[k]) for k in ('xs_off', 'xs_idx', 'xs_val', 'n_players', 'extra', 'opt_off',
                                                     'opt_action') + VARLEN_KEYS if k in lb.a})
        return ev(lb)

    def have(k):
        return sum(int(b[k][-1]) for b in keep if k in b)

    def check():
        if sum(len(b['n_players']) for b in keep) >= rows and len(keep) >= 2 \
                and (not need_t or have('t_off')) and (not need_o or have('ov_off')):
            raise Enough()
        if len(keep) >= MAX_RECORD_BATCHES:
            raise RuntimeError(f'record_rows: {len(keep)} batches with {have("t_off")} tail tokens and '
                               f'{have("ov_off")} overrides: the network\'s {"tails" if need_t else ""}'
                               f'{" and " if need_t and need_o else ""}{"overrides" if need_o else ""} would go untested')

    os.environ['ZERO_INFLIGHT'] = '1'
    try:
        drive(drv, {'m': rec}, check=check)
    except Enough:
        pass
    if (need_t and not have('t_off')) or (need_o and not have('ov_off')):
        raise RuntimeError(f'record_rows: the self-play ended after {len(keep)} batches with {have("t_off")} tail tokens '
                           f'and {have("ov_off")} overrides: the network\'s variable-length rows would go untested')
    n = [len(b['n_players']) for b in keep]
    xs_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['xs_off']) for b in keep]))])
    opt_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['opt_off']) for b in keep]))])
    cat = lambda k: np.concatenate([b[k] for b in keep])
    tail = ov = None
    if 't_off' in keep[0]:
        from .rows import Tail
        w = {int(x['seg']): (int(x['width']), int(x['head']), k) for k, x in enumerate(G.info(game)['tails'])}
        tail = Tail.concat([Tail.from_arrays(b, w) for b in keep])
    if 'ov_off' in keep[0]:
        from .rows import Overrides
        ov = Overrides.concat([Overrides.from_arrays(b) for b in keep])
    return Rows(xs_off, cat('xs_idx'), cat('xs_val'), cat('n_players'), opt_off, cat('opt_action'), cat('extra'),
                model.features, batch_off=np.concatenate([[0], np.cumsum(n)]).astype(np.int64), source='farm_h2h',
                tail=tail, ov=ov)


def export(ckpt, gen, out_dir, game, rows_list, ckpt_id, threads=4, checkpoint_iter=None):
    """`ckpt` -> <out_dir>/model.onnx + reference.npz, the reference checked through onnxruntime -> their
    {model, sha256, reference, reference_sha256, check} (local paths). `checkpoint_iter`: the graph's network id
    (its metadata `checkpoint_iter`: a self-play worker's rows carry it; default the checkpoint's iteration)."""
    from .onnx_export import REFERENCE_TOL, check_reference, export as onnx_export, load_checkpoint, write_reference
    os.makedirs(out_dir, exist_ok=True)
    m, info = load_checkpoint(ckpt, game)
    info['checkpoint'] = ckpt_id
    if checkpoint_iter is not None:
        info['checkpoint_iter'] = str(int(checkpoint_iter))
    model = os.path.join(out_dir, 'model.onnx')
    ref = os.path.join(out_dir, 'reference.npz')
    onnx_export(m, model, 'packed', info)
    write_reference(m, rows_list, ref, 256, info)
    chk = check_reference(model, ref, threads)
    if not chk['max_abs'] <= REFERENCE_TOL:
        raise SystemExit(f'gen {gen} ({ckpt}): the graph differs from torch by {chk["max_abs"]:g} on its reference '
                         f'set (> {REFERENCE_TOL:g}): not published')
    return {'model': model, 'sha256': file_sha256(model), 'model_size': os.path.getsize(model), 'reference': ref,
            'reference_sha256': file_sha256(ref), 'reference_size': os.path.getsize(ref), 'check': chk,
            'checkpoint': os.path.abspath(ckpt), 'checkpoint_sha256': info['checkpoint_sha256']}


# ---------------------------------------------------------------- state

def state_dir(a, run):
    return os.path.join(a.state_dir or os.path.join(os.path.expanduser('~'), '.zero-farm', 'h2h'), run)


def save_state(d, st):
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'state.json.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(d, 'state.json'))


def load_state(d):
    p = os.path.join(d, 'state.json')
    if not os.path.exists(p):
        raise SystemExit(f'no head-to-head here: {p}')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def eval_job(st):
    """The eval batch's job object - also, exactly, the signed definition's `jobs.eval` template (PROTOCOL.md 7: an
    agent builds its job line from the template - settings, seats, paired: nothing else - plus only the lease's
    `gid_base`, `games` and `seed`)."""
    return {'settings': st['settings'], 'seats': [{'net': 1}, {'net': 2}], 'paired': True}


def coordinator_seed(run):
    """The run's base seed as the coordinator derives it from the run's name (32-bit FNV-1a; zero-farm coordinator
    book.ts `baseSeed`): a rev 6 template carries no seed, so a head-to-head's games are fixed by its name (and
    settings and networks). Every lease of the run carries it; workers derive each game's seed from (seed, gid)."""
    h = 0x811c9dc5
    for ch in run:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xffffffff
    return h


def worker_json(worker, *args):
    """`<worker> args...` -> (exit status, its JSON output or None, its output text)."""
    import subprocess
    p = subprocess.run(FD.exe_argv(worker, *args), capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=600)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = None
    return p.returncode, out, (p.stdout or '') + (p.stderr or '')


# this zero-core's settings table (the one its worker builds embed: crates/worker/src/run_settings.json)
RUN_SETTINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'crates',
                            'worker', 'src', 'run_settings.json')


def load_run_settings(path=None):
    """This zero-core's settings table ({"keys": {key: {"class", ...}}, "patterns": [...]}), or None when it isn't
    there (no source checkout)."""
    try:
        with open(path or RUN_SETTINGS, encoding='utf-8') as f:
            t = json.load(f)
    except (OSError, ValueError):
        return None
    return t if isinstance(t, dict) and isinstance(t.get('keys'), dict) else None


def setting_class(key, table):
    """A run setting's class in a settings table: its own entry, else the first pattern (prefix, suffix) it matches,
    else None (a setting this zero-core doesn't know)."""
    e = table['keys'].get(key)
    if isinstance(e, dict):
        return e.get('class')
    for p in table.get('patterns') or []:
        if key.startswith(p.get('prefix', '')) and key.endswith(p.get('suffix', '')):
            return p.get('class')
    return None


def config_refused(rc, out, build_id, log, table=None):
    """Does a worker build's `check-config --json` answer (exit status `rc`, its JSON `out`) on a run's FULL settings
    refuse them? A failure whose key the worker calls "unknown" but this zero-core's settings table
    (`load_run_settings`, or `table`) classifies "trainer" is not a refusal: a worker build older than the setting
    doesn't know it, and a template carries only the keys the worker calls "play", so the worker never gets it -
    each is logged ("worker build X doesn't know trainer-only setting K: ignored"). Any other failure, a driver error,
    or a failed exit without failures refuses as before; without the table (no source checkout) every failure does."""
    if not isinstance(out, dict):
        return True
    fails = list(out.get('failures') or [])
    if not fails:
        return rc != 0
    if out.get('driver_error'):
        return True
    table = load_run_settings() if table is None else table
    if table is None:
        return True
    status = {s.get('key'): s.get('status') for s in out.get('settings') or [] if isinstance(s, dict)}
    if any(not (status.get(k) == 'unknown' and setting_class(k, table) == 'trainer') for k in fails):
        return True
    for k in fails:
        log(f'worker build {build_id} doesn\'t know trainer-only setting {k}: ignored')
    return False


def _stderr_log(m):
    print('# ' + m, file=sys.stderr, flush=True)


def play_settings(worker, build_id, cfg, d, log=None):
    """The eval template's settings: only those the worker build's own `check-config --eval` classifies `play` (what
    an evaluation match reads - zero_worker EVAL_READS; never a trainer-only key or a path, which farm-release refuses
    in a template), checked again by the worker on their own, and giving the seats every setting gives
    (`Run.nn_seat`). `worker` must be that build's executable (its `info` names `build_id` and the kind eval). A
    setting the build doesn't know but this zero-core classifies trainer-only is logged and left out
    (`config_refused`)."""
    from . import farm_eval as FE
    log = log or _stderr_log
    rc, info, text = worker_json(worker, 'info', '--ort-dll', 'none')
    if rc != 0 or not isinstance(info, dict):
        raise SystemExit(f'{worker} info: {text.strip()[-500:]}')
    if info.get('build_id') != build_id or 'eval' not in (info.get('kinds') or []):
        raise SystemExit(f'{worker} is build {info.get("build_id")} ({info.get("kinds")}), not the eval-capable worker '
                         f'build {build_id}: give --worker the signed build\'s executable')
    os.makedirs(d, exist_ok=True)
    full = os.path.join(d, 'settings-run.json')
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    rc, out, text = worker_json(worker, 'check-config', full, '--eval', '--json')
    if config_refused(rc, out, build_id, log):
        raise SystemExit(f'the worker refuses the settings for an eval job:\n{text.strip()[-3000:]}')
    status = {s['key']: s['status'] for s in out.get('settings') or []}
    play = {k: v for k, v in cfg.items() if status.get(k) == 'play'}
    missing = sorted(set(out.get('eval_reads') or []) - set(play))
    if missing:
        raise SystemExit(f'the settings lack {missing}, which an eval job reads')
    path = os.path.join(d, 'settings.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(play, f)
    rc, out, text = worker_json(worker, 'check-config', path, '--eval', '--json')
    if rc != 0 or not isinstance(out, dict) or out.get('failures') or \
            any(s['status'] != 'play' for s in out.get('settings') or []):
        raise SystemExit(f'the worker refuses the eval job\'s own settings:\n{text.strip()[-3000:]}')
    seats = [{'net': 1}, {'net': 2}]
    if FE.seat_dicts(play, seats) != FE.seat_dicts(cfg, seats):
        raise SystemExit('the play settings give other seats than the run\'s settings (EVAL_READS is not what '
                         'Run.nn_seat reads: tests/test_worker_settings.py)')
    return play


def run_body(st, eval_on=True):
    """The run's PUT body (the eval batch once the definition is published and the gens are registered); its
    `min_agent` (PROTOCOL rev 9), `alternates` (rev 11) and the batch's `gpu_games_per_lease` (rev 11) only when the
    head-to-head has them (else the keys are absent: the body of before)."""
    b = {'game': st['game'], 'build_id': st['build_id'], 'runtime': st['runtime'], 'priority': st['priority']}
    if st.get('min_agent'):
        b['min_agent'] = st['min_agent']
    if st.get('alternates'):
        b['alternates'] = st['alternates']
    b.update(selfplay=None, eval=None, state=st.get('run_state', 'active'))
    if eval_on:
        ev = {'batch': st['batch'], 'games': st['games'], 'checkpoints': [1, 2], 'games_per_lease': st['games_per_lease']}
        if st.get('gpu_games_per_lease') is not None:
            ev['gpu_games_per_lease'] = st['gpu_games_per_lease']
        ev.update(gpu='any', needs=st['needs'], job=eval_job(st))
        b['eval'] = ev
    return b


def definition_fields(st):
    """The run's definition (PROTOCOL.md 7): the worker build (then its alternates' builds, rev 11), its runtime, gens 1
    and 2 (each graph's and reference set's sha256 and size) and the eval template; no self-play, no fuzz."""
    cks = {}
    for gen in ('1', '2'):
        g = st['gens'][gen]
        cks[gen] = {'model_sha256': g['sha256'], 'model_size': g.get('model_size') or os.path.getsize(g['model']),
                    'reference_sha256': g['reference_sha256'],
                    'reference_size': g.get('reference_size') or os.path.getsize(g['reference'])}
    builds = [st['build_id']] + [x['build_id'] for x in st.get('alternates') or []]
    return {'game': st['build_game'], 'builds': builds, 'runtime': st['runtime'], 'checkpoints': cks,
            'jobs': {'selfplay': None, 'eval': eval_job(st), 'fuzz': None}}


def define(cl, d, st, log, opts):
    """The run's signed definition, made and published once (state `definition`): a new version (the coordinator's
    current + 1), kept in the definitions archive, signed with the data key by `farm-release define --publish`."""
    info = st.get('definition')
    if info and info.get('published'):
        return info
    if not st.get('build_game'):
        st['build_game'] = FD.worker_game(cl, st['build_id'], st['runtime'])
    if not info:
        info = FD.prepare(cl, 'run', st['run'], expires_days=opts.get('expires_days', EXPIRES_DAYS),
                          root=opts.get('defs_dir'), **definition_fields(st))
        st['definition'] = info
        save_state(d, st)
        log(f'definition run/{st["run"]} version {info["version"]}: {info["path"]}')
    info = FD.publish(cl, info, key=opts.get('key'), farm_release=opts.get('farm_release'), log=log)
    st['definition'] = info
    save_state(d, st)
    return info


def setup(cl, d, st, log, opts=None):
    """The coordinator's side of a new head-to-head, each step once (state flags): the run (no batch), its signed
    definition, the gens' files and registrations, then the eval batch, whose job is the definition's template."""
    run = st['run']
    opts = opts or {}
    if not st.get('run_put'):
        cl.call('PUT', f'/v1/admin/run/{run}', body=run_body(st, eval_on=False))
        st['run_put'] = True
        save_state(d, st)
        alts = ''.join(f', alternate {x["build_id"]} (runtime {x["runtime"]}, {x["when"]["vendor"]})'
                       for x in st.get('alternates') or [])
        ma = f', min_agent {st["min_agent"]}' if st.get('min_agent') else ''
        log(f'run {run} created (worker build {st["build_id"]}, runtime {st["runtime"]}{alts}{ma})')
    define(cl, d, st, log, opts)
    for gen in ('1', '2'):
        g = st['gens'][gen]
        if not g.get('registered'):
            keys = {'model': f'ckpt/{run}/{gen}/model.onnx', 'reference': f'ckpt/{run}/{gen}/reference.npz'}
            for k in ('model', 'reference'):
                cl.put_artifact(keys[k], g[k], g[f'{k}_sha256' if k == 'reference' else 'sha256'])
            cl.call('PUT', f'/v1/admin/run/{run}/checkpoint/{gen}',
                    body={'model': keys['model'], 'sha256': g['sha256'], 'reference': keys['reference'],
                          'reference_sha256': g['reference_sha256']})
            g['registered'] = True
            save_state(d, st)
            log(f'gen {gen} uploaded and registered ({os.path.basename(g["checkpoint"])}, {g["sha256"][:12]})')
    if not st.get('eval_put'):
        cl.call('PUT', f'/v1/admin/run/{run}', body=run_body(st))
        st['eval_put'] = True
        save_state(d, st)
        log(f'eval batch {st["batch"]}: {st["games"]} paired games')


# ---------------------------------------------------------------- results

def _file(cl, d, key, sha=None):
    """A result file's text (downloaded once, kept under results/; `sha`: its sha256, checked)."""
    p = os.path.join(d, 'results', hashlib.sha256(key.encode()).hexdigest()[:24] + '.jsonl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            body = f.read()
    else:
        body = cl.call('GET', f'/v1/admin/artifact/{key}', raw=True)
        tmp = p + f'.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            f.write(body)
        os.replace(tmp, p)
    if sha and hashlib.sha256(body).hexdigest() != sha:
        raise SystemExit(f'result {key}: sha256 mismatch')
    return body.decode('utf-8')


def pull(cl, d, st):
    """New result files: downloaded, checked, their credited games merged -> result entries taken. A result entry
    credits the rows of its file whose gid is in its `ids`, and the rows of earlier files of its lease that it
    completed (`also: [{seq, key, ids}]`: a pair counts only when both its games are in, from one lease); ids in
    `dup_ids` (credited before) and games already here are skipped: the first copy of a game wins."""
    from .farm_eval import read_rows, row_json
    run = st['run']
    os.makedirs(os.path.join(d, 'results'), exist_ok=True)
    rows_path = os.path.join(d, 'rows.jsonl')
    have = set(st.setdefault('have', []))
    machines = st.setdefault('machines', {})
    taken = 0
    rng = lambda rs: {g for lo, hi in (rs or []) for g in range(lo, hi)}
    while True:
        ans = cl.call('GET', f'/v1/admin/run/{run}/results?after={st.get("cursor", 0)}&limit=500')
        res = ans.get('results') or []
        for r in res:
            if r.get('kind') != 'eval' or (r.get('batch') not in (None, st['batch'])):
                continue
            dup = rng(r.get('dup_ids'))
            parts = [(_file(cl, d, r['key'], r.get('sha256')), rng(r.get('ids')))]
            for x in r.get('also') or []:
                parts.append((_file(cl, d, x['key']), rng(x.get('ids'))))
            new = []
            for text, ids in parts:
                for row in read_rows(text):
                    g = row['gid']
                    if g in dup or g in have or g not in ids:
                        continue
                    have.add(g)
                    new.append(row)
            if new:
                with open(rows_path, 'a', encoding='utf-8', newline='\n') as f:
                    for row in new:
                        f.write(row_json(row) + '\n')
                m = machines.setdefault(r.get('machine_id') or '?', {'games': 0})
                m['games'] += len(new)
            taken += 1
        st['have'] = sorted(have)
        if res:
            st['cursor'] = ans.get('cursor', st.get('cursor', 0))
            save_state(d, st)
            cl.call('POST', f'/v1/admin/run/{run}/ack', body={'cursor': st['cursor']})
        if len(res) < 500:
            return taken


def rows_of(d):
    from .farm_eval import read_rows
    p = os.path.join(d, 'rows.jsonl')
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return read_rows(f.read())


def machine_names(cl, st):
    try:
        ms = cl.call('GET', '/v1/admin/machines')
        ms = ms.get('machines', ms) if isinstance(ms, dict) else ms
        names = {m.get('machine_id'): m.get('name') for m in ms or [] if isinstance(m, dict)}
    except (FarmError, AttributeError, TypeError):
        names = {}
    return {(names.get(k) or k): v['games'] for k, v in sorted(st.get('machines', {}).items())}


def readout(cl, d, st):
    """A's score, its se, the Elo difference with its 95% interval, games, machines -> (lines, json dict)."""
    from . import farm_eval as FE
    rows = rows_of(d)
    if not rows:
        return ['no games yet'], None
    out = FE.summary(rows, True, {'net': 1})
    machines = machine_names(cl, st)
    full = dict({'a': st['a'], 'b': st['b'], 'game': st['game'], 'sims': st['settings']['eval_sims'],
                 'chance_k': st['settings']['chance_k']}, **out)
    full.update(select_a=st['select'], select_b=st['select'], paired=True, run=st['run'], machines=machines,
                farm=True)
    lines = [f'{st["run"]}: A = {os.path.basename(st["a"])}, B = {os.path.basename(st["b"])} ({st["game"]}, '
             f'eval_sims {full["sims"]}, {st["select"]})',
             f'A scores {out["a_score"]:.4f} +- {out["se"]:.4f} (pair se; per-game se {out.get("se_games")}) over '
             f'{out["games"]} games ({out["pairs"]["n"]} pairs: both won {out["pairs"]["both_won"]}, split '
             f'{out["pairs"]["split"]}, both lost {out["pairs"]["both_lost"]}, with a draw {out["pairs"]["with_draw"]})',
             f'Elo A - B: {out["elo_diff"]:+.1f} (95%: {out["elo_95"][0]:+.1f} .. {out["elo_95"][1]:+.1f}); mean margin '
             f'{out["a_margin"]:+.2f}; unfinished {out["unfinished"]}',
             f'machines: ' + ', '.join(f'{k} {v}' for k, v in sorted(machines.items(), key=lambda x: -x[1]))]
    return lines, full


def eval_state(cl, st):
    """The coordinator's progress of the batch (`GET /v1/admin/run/{run}`'s eval_state: done, leased, unissued),
    or None."""
    try:
        es = cl.call('GET', f'/v1/admin/run/{st["run"]}').get('eval_state')
    except (FarmError, AttributeError):
        return None
    return es if isinstance(es, dict) and es.get('batch') in (None, st['batch']) else None


def progress_line(d, st, t0, n0, es=None):
    from . import farm_eval as FE
    rows = rows_of(d)
    n = len(rows)
    # (games per minute since the batch was put: a re-attached watch counts the games already in too)
    rate = n / max(1e-9, (time.time() - st.get('created', t0)) / 60)
    if rows and len(rows) >= 2:
        try:
            s = FE.summary(rows, True, {'net': 1})
            sc = f'A {s["a_score"]:.3f} +- {s["se"]:.3f}, Elo {s["elo_diff"]:+.0f}'
        except (ZeroDivisionError, KeyError):
            sc = ''
    else:
        sc = ''
    farm = f' (farm: {es.get("done")} done, {es.get("leased")} leased, {es.get("unissued")} unissued)' if es else ''
    return f'# {n}/{st["games"]} games{farm}, {len(st.get("machines", {}))} machines, {rate:.1f} games/min {sc}'


def watch(cl, d, st, poll, log):
    t0, n0 = time.time(), len(rows_of(d))
    while True:
        pull(cl, d, st)
        line = progress_line(d, st, t0, n0, eval_state(cl, st))
        print('\r' + line + ' ' * 8, end='', file=sys.stderr, flush=True)
        if len(st.get('have', [])) >= st['games']:
            print(file=sys.stderr)
            return
        time.sleep(poll)


def finish_print(cl, d, st):
    lines, full = readout(cl, d, st)
    for ln in lines:
        print('# ' + ln, file=sys.stderr)
    if full is not None:
        print(json.dumps(full), flush=True)


# ---------------------------------------------------------------- main

def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--a')
    ap.add_argument('--b')
    ap.add_argument('--games', type=int)
    ap.add_argument('--config')
    ap.add_argument('--preset')
    ap.add_argument('--sims', type=int)
    ap.add_argument('--name', help='the run is h2h-<name> (default: h2h-<run>-<A>v<B>)')
    ap.add_argument('--build', help='the worker build_id (a registered release worker build)')
    ap.add_argument('--runtime', help='the runtime pack build_id the worker build requires')
    ap.add_argument('--games-per-lease', default='auto')
    ap.add_argument('--priority', type=float, default=1.0)
    ap.add_argument('--needs-threads', type=int, default=1)
    ap.add_argument('--needs-ram-mb', type=int, default=1500)
    ap.add_argument('--reference-batches', nargs='+')
    ap.add_argument('--worker', help="the worker build's executable (its signed build directory's): its check-config "
                    "--eval picks the eval job's settings")
    ap.add_argument('--resume', metavar='NAME')
    ap.add_argument('--cancel', metavar='NAME')
    ap.add_argument('--url', default=os.environ.get('FARM_URL', URL))
    ap.add_argument('--state-dir')
    ap.add_argument('--package', help='the game package to register (e.g. mtgz.az), with --root its checkout')
    ap.add_argument('--root')
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--no-wait', action='store_true')
    ap.add_argument('--threads', type=int, default=4, help='CPU threads of the export and the reference recording')
    ap.add_argument('--key', default=None, help='the data key that signs the run definition (default: env '
                    'ZEROFARM_DATA_KEY, else ~/.config/zero-farm/data.key)')
    ap.add_argument('--farm-release', default=None, help='zero-farm farm-release (default: env ZEROFARM_FARM_RELEASE, '
                    f'else {FD.FARM_RELEASE})')
    ap.add_argument('--expires-days', type=float, default=EXPIRES_DAYS,
                    help=f'the definition expires after this many days (default {EXPIRES_DAYS}; 0 = never)')
    ap.add_argument('--defs-dir', help='the definitions archive (default: env ZEROFARM_DEFS_DIR, else ~/.zero-farm/definitions)')
    ap.add_argument('--alt', action='append', metavar='BUILD_ID',
                    help='a run alternate (zero-farm PROTOCOL rev 11, 9.4): a worker build of the same engine signed '
                    'for another runtime pack, offered only to PCs with that pack\'s GPU vendor; repeatable, in order '
                    f'of preference, at most {FD.MAX_ALTERNATES}')
    ap.add_argument('--alt-vendor', choices=FD.VENDORS,
                    help='the GPU vendor of the alternates whose pack names none (no `gpu` in its manifest)')
    ap.add_argument('--min-agent', metavar='VERSION',
                    help="the run's min_agent (zero-farm PROTOCOL rev 9): its leases go only to agents at or above it "
                    '(e.g. 0.2.10: canary agents only); default none')
    ap.add_argument('--gpu-games-per-lease', metavar='N|auto',
                    help="the eval batch's gpu_games_per_lease (PROTOCOL rev 11, 9.4): the games of a GPU lease; "
                    'default: left out (GPU leases sized as before)')
    a = ap.parse_args(argv)
    if a.min_agent is not None and not FD.SEMVER_RE.match(a.min_agent):
        ap.error(f'--min-agent {a.min_agent!r}: a version like 0.2.10')
    try:
        gpu_per_lease = FD.gpu_games_per_lease(a.gpu_games_per_lease)
    except ValueError:
        ap.error(f'--gpu-games-per-lease {a.gpu_games_per_lease!r}: "auto" or a number of games, 1 to '
                 f'{FD.MAX_LEASE_GAMES:,}')
    opts = {'key': a.key, 'farm_release': a.farm_release, 'expires_days': a.expires_days, 'defs_dir': a.defs_dir}
    log = lambda m: print('# ' + m, file=sys.stderr, flush=True)
    if a.root:
        sys.path.insert(0, os.path.abspath(a.root))
    if a.package:
        __import__(a.package)
    if a.alt_vendor and not a.alt:
        ap.error('--alt-vendor: with --alt (the alternates it applies to)')
    if a.resume or a.cancel:
        if a.alt or a.min_agent is not None or a.gpu_games_per_lease is not None:
            ap.error('--alt, --alt-vendor, --min-agent, --gpu-games-per-lease: only when a head-to-head starts (--resume '
                     'and --cancel keep what it started with)')
        name = a.resume or a.cancel
        run = name if name.startswith('h2h-') else f'h2h-{name}'
        d = state_dir(a, run)
        st = load_state(d)
        cl = client or Client(st.get('url', a.url))
        if a.cancel:
            st['run_state'] = 'paused'
            cl.call('PUT', f'/v1/admin/run/{run}', body=run_body(st, eval_on=bool(st.get('eval_put'))))
            save_state(d, st)
            log(f'{run} paused (its leases are cancelled at their next beat); --resume {name} to read what came in')
            return 0
        setup(cl, d, st, log, opts)
        if not a.no_wait:
            watch(cl, d, st, a.poll, log)
        finish_print(cl, d, st)
        return 0
    if not (a.a and a.b and a.games):
        ap.error('--a, --b and --games (or --resume / --cancel NAME)')
    if a.games < 2 or a.games % 2:
        ap.error('--games: an even number (paired games: each deal from both seats)')
    if not (a.build and a.runtime and a.worker):
        ap.error('--build, --runtime and --worker: the worker build, the runtime pack it requires and the build '
                 'executable')
    # the worker build's manifest first (fail before the export): its game is the definition's, and it must require
    # this runtime; then the alternates' (each a worker of that game requiring its own pack)
    cl = client or Client(a.url)
    build_game = FD.worker_game(cl, a.build, a.runtime)
    alternates, _builds = FD.run_alternates(cl, a.build, a.runtime, a.alt, a.alt_vendor)
    import torch
    torch.set_num_threads(max(1, a.threads))
    from . import farm_eval as FE
    from .h2h import _load_pair, _tag
    ma, mb, game = _load_pair(a.a, a.b, 'cpu', None)
    cfg = settings_of(a, a.a, game)
    run = f'h2h-{a.name}' if a.name and not a.name.startswith('h2h-') else (a.name or run_slug(a.a, a.b))
    if not SLUG_RE.match(run):
        ap.error(f'run name {run!r}: [a-z0-9][a-z0-9-]{{0,62}}')
    d = state_dir(a, run)
    if os.path.exists(os.path.join(d, 'state.json')):
        raise SystemExit(f'{run} exists ({d}): --resume {run}, or another --name')
    run_cfg = cfg
    cfg = play_settings(a.worker, a.build, run_cfg, d, log)
    log(f'{run}: {game}, {a.games} paired games, eval_sims {cfg["eval_sims"]}, eval spec {cfg["eval_spec"][:80]}; '
        f'{len(cfg)} of the run\'s {len(run_cfg)} settings are the eval job\'s (the worker\'s check-config: play)')
    if a.reference_batches:
        from .onnx_parity import Rows
        rows_list = [Rows.load(p) for p in a.reference_batches]
    else:
        log('recording reference rows (a short self-play of A, CPU)')
        rows_list = [record_rows(ma, game, run_cfg['spec'], threads=a.threads)]
    gens = {}
    for gen, ck in (('1', a.a), ('2', a.b)):
        rid = f'{os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ck))))}/{os.path.basename(ck)}'
        gens[gen] = export(ck, gen, os.path.join(d, 'ckpt', gen), game, rows_list, rid, a.threads)
        log(f'gen {gen}: {ck} exported, reference max abs {gens[gen]["check"]["max_abs"]:.2e}')
    gpl = a.games_per_lease if a.games_per_lease == 'auto' else int(a.games_per_lease)
    st = {'run': run, 'url': a.url, 'a': os.path.abspath(a.a), 'b': os.path.abspath(a.b), 'game': game,
          'games': a.games, 'seed': coordinator_seed(run), 'build_id': a.build, 'runtime': a.runtime, 'priority': a.priority,
          'games_per_lease': gpl, 'needs': {'threads': a.needs_threads, 'ram_mb': a.needs_ram_mb}, 'batch': 'h2h',
          'settings': cfg, 'select': _tag(FE.seat_dicts(cfg, [{'net': 1}, {'net': 2}])[0]), 'gens': gens,
          'created': time.time(), 'cursor': 0, 'build_game': build_game, 'alternates': alternates,
          'min_agent': a.min_agent, 'gpu_games_per_lease': gpu_per_lease}
    save_state(d, st)
    setup(cl, d, st, log, opts)
    if a.no_wait:
        log(f'set up; read the results with: python -m zero.farm_h2h --resume {run}')
        return 0
    watch(cl, d, st, a.poll, log)
    finish_print(cl, d, st)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
