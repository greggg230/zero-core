"""A head-to-head of two checkpoints on zero-farm (friends' idle PCs; zero-farm PROTOCOL.md 5.2, docs/az.md 42.2):
the readout zero.h2h gives on this machine, played by the farm's native workers.

  python -m zero.farm_h2h --a runs/eoe1/models/iter_0080.pt --b runs/eoe1/models/iter_0070.pt --games 400 \
      --build <worker build_id> --runtime <runtime build_id> [--config <run>/config.json | --preset NAME]
      [--sims 32] [--name SLUG] [--seed 1] [--games-per-lease auto] [--priority 1] [--needs-ram-mb 1500]
      [--reference-batches NPZ ...] [--worker <mtgz-worker.exe>] [--url https://farm.greggg230.com] [--no-wait]
      [--key <data key>] [--farm-release <farm-release.exe>] [--expires-days 30] [--defs-dir DIR]
  python -m zero.farm_h2h --resume NAME        (re-attach: poll, download, print the readout)
  python -m zero.farm_h2h --cancel NAME        (pause the run: its leases are cancelled at their next beat)
  (python -m <pkg>.az.farm_h2h ...: the game package registered; or --package <pkg>.az [--root <checkout>])

What it does:
1. The settings: `--config` (a run's config.json) or `--preset`, else A's run's config.json (next to its
   models/), resolved as a run resolves them; `--sims` sets eval_sims; `eval_spec` resolved (the game
   package's eval spec of the run's spec, `Run.driver`'s rule: a worker cannot run the package's Python hook).
   Both seats play the run's evaluation seat (`Run.nn_seat(key, eval_sims)`), paired games: exactly the
   games of `Run.play_match` / an Elo match at those settings (zero.farm_eval).
2. Both checkpoints exported to zero-onnx/1 (zero.onnx_export, 'packed') with their reference sets (256 rows
   of recorded batches - `--reference-batches`, else a short self-play of A recorded here, CPU - with torch's
   answers, checked through onnxruntime: <= 2e-5, else nothing is published).
3. The run `h2h-<slug>` (`PUT /v1/admin/run/h2h-<slug>`, selfplay null, the worker build and runtime); its signed
   definition (zero-farm PROTOCOL.md 7, zero.farm_def: builds [the worker build], the runtime, gens 1 and 2 with
   their graphs' and reference sets' sha256 and sizes, `jobs.eval` = the batch's job; the next version, signed with
   the data key and published by `farm-release define --publish`; it expires after `--expires-days`); the graphs
   and reference sets uploaded (`PUT /v1/admin/artifact/ckpt/<run>/<gen>/...`), registered as gens 1 (A) and 2 (B),
   then the eval batch: `checkpoints [1, 2]`, its job {settings, seats [{"net": 1}, {"net": 2}], paired, seed} -
   the definition's template exactly (the coordinator issues leases only within it), `--games`.
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

def record_rows(model, game, spec, rows=320, conc=32, sims=16, threads=4, seed=1):
    """`rows` real network batch rows (whole batches) of a short self-play of `model` (tools/onnx_record.py's
    seats: root belief rows when the network has the auxiliary head, root-view leaf values with view samples)
    -> onnx_parity.Rows with batch_off."""
    import numpy as np
    from . import games as G
    from .infer import NetEvaluator, drive
    from .onnx_parity import Rows
    ev = NetEvaluator(model, 'cpu')
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
                                                     'opt_action')})
        return ev(lb)

    def check():
        if sum(len(b['n_players']) for b in keep) >= rows and len(keep) >= 2:
            raise Enough()

    os.environ['ZERO_INFLIGHT'] = '1'
    try:
        drive(drv, {'m': rec}, check=check)
    except Enough:
        pass
    n = [len(b['n_players']) for b in keep]
    xs_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['xs_off']) for b in keep]))])
    opt_off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(b['opt_off']) for b in keep]))])
    cat = lambda k: np.concatenate([b[k] for b in keep])
    return Rows(xs_off, cat('xs_idx'), cat('xs_val'), cat('n_players'), opt_off, cat('opt_action'), cat('extra'),
                model.features, batch_off=np.concatenate([[0], np.cumsum(n)]).astype(np.int64), source='farm_h2h')


def export(ckpt, gen, out_dir, game, rows_list, ckpt_id, threads=4):
    """`ckpt` -> <out_dir>/model.onnx + reference.npz, the reference checked through onnxruntime -> their
    {model, sha256, reference, reference_sha256, check} (local paths)."""
    from .onnx_export import REFERENCE_TOL, check_reference, export as onnx_export, load_checkpoint, write_reference
    os.makedirs(out_dir, exist_ok=True)
    m, info = load_checkpoint(ckpt, game)
    info['checkpoint'] = ckpt_id
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
    agent builds its job line from the template plus only `gid_base`, `games` and `seed`; `seed` here is the run's
    base seed, which the coordinator copies into every lease)."""
    return {'settings': st['settings'], 'seats': [{'net': 1}, {'net': 2}], 'paired': True, 'seed': st['seed']}


def run_body(st, eval_on=True):
    """The run's PUT body (the eval batch once the definition is published and the gens are registered)."""
    b = {'game': st['game'], 'build_id': st['build_id'], 'runtime': st['runtime'], 'priority': st['priority'],
         'selfplay': None, 'eval': None, 'state': st.get('run_state', 'active')}
    if eval_on:
        b['eval'] = {'batch': st['batch'], 'games': st['games'], 'checkpoints': [1, 2],
                     'games_per_lease': st['games_per_lease'], 'gpu': 'any', 'needs': st['needs'],
                     'job': eval_job(st)}
    return b


def definition_fields(st):
    """The run's definition (PROTOCOL.md 7): the worker build, its runtime, gens 1 and 2 (each graph's and reference
    set's sha256 and size) and the eval template; no self-play, no fuzz."""
    cks = {}
    for gen in ('1', '2'):
        g = st['gens'][gen]
        cks[gen] = {'model_sha256': g['sha256'], 'model_size': g.get('model_size') or os.path.getsize(g['model']),
                    'reference_sha256': g['reference_sha256'],
                    'reference_size': g.get('reference_size') or os.path.getsize(g['reference'])}
    return {'game': st['game'], 'builds': [st['build_id']], 'runtime': st['runtime'], 'checkpoints': cks,
            'jobs': {'selfplay': None, 'eval': eval_job(st), 'fuzz': None}}


def define(cl, d, st, log, opts):
    """The run's signed definition, made and published once (state `definition`): a new version (the coordinator's
    current + 1), kept in the definitions archive, signed with the data key by `farm-release define --publish`."""
    info = st.get('definition')
    if info and info.get('published'):
        return info
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
        log(f'run {run} created (worker build {st["build_id"]}, runtime {st["runtime"]})')
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
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--games-per-lease', default='auto')
    ap.add_argument('--priority', type=float, default=1.0)
    ap.add_argument('--needs-threads', type=int, default=1)
    ap.add_argument('--needs-ram-mb', type=int, default=1500)
    ap.add_argument('--reference-batches', nargs='+')
    ap.add_argument('--worker', help='a local worker exe: its check-config --eval checks the settings first')
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
    a = ap.parse_args(argv)
    opts = {'key': a.key, 'farm_release': a.farm_release, 'expires_days': a.expires_days, 'defs_dir': a.defs_dir}
    log = lambda m: print('# ' + m, file=sys.stderr, flush=True)
    if a.root:
        sys.path.insert(0, os.path.abspath(a.root))
    if a.package:
        __import__(a.package)
    if a.resume or a.cancel:
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
    if not (a.build and a.runtime):
        ap.error('--build and --runtime: the worker build and the runtime pack it requires')
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
    if a.worker:
        import subprocess
        os.makedirs(d, exist_ok=True)
        cp = os.path.join(d, 'settings.json')
        with open(cp, 'w', encoding='utf-8') as f:
            json.dump(cfg, f)
        p = subprocess.run([a.worker, 'check-config', cp, '--eval'], capture_output=True, text=True, encoding='utf-8')
        if p.returncode != 0:
            raise SystemExit(f'the worker refuses the settings:\n{p.stdout}{p.stderr}')
    log(f'{run}: {game}, {a.games} paired games, eval_sims {cfg["eval_sims"]}, eval spec {cfg["eval_spec"][:80]}')
    if a.reference_batches:
        from .onnx_parity import Rows
        rows_list = [Rows.load(p) for p in a.reference_batches]
    else:
        log('recording reference rows (a short self-play of A, CPU)')
        rows_list = [record_rows(ma, game, cfg['spec'], threads=a.threads)]
    gens = {}
    for gen, ck in (('1', a.a), ('2', a.b)):
        rid = f'{os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ck))))}/{os.path.basename(ck)}'
        gens[gen] = export(ck, gen, os.path.join(d, 'ckpt', gen), game, rows_list, rid, a.threads)
        log(f'gen {gen}: {ck} exported, reference max abs {gens[gen]["check"]["max_abs"]:.2e}')
    gpl = a.games_per_lease if a.games_per_lease == 'auto' else int(a.games_per_lease)
    st = {'run': run, 'url': a.url, 'a': os.path.abspath(a.a), 'b': os.path.abspath(a.b), 'game': game,
          'games': a.games, 'seed': a.seed, 'build_id': a.build, 'runtime': a.runtime, 'priority': a.priority,
          'games_per_lease': gpl, 'needs': {'threads': a.needs_threads, 'ram_mb': a.needs_ram_mb}, 'batch': 'h2h',
          'settings': cfg, 'select': _tag(FE.seat_dicts(cfg, [{'net': 1}, {'net': 2}])[0]), 'gens': gens,
          'created': time.time(), 'cursor': 0}
    save_state(d, st)
    cl = client or Client(a.url)
    setup(cl, d, st, log, opts)
    if a.no_wait:
        log(f'set up; read the results with: python -m zero.farm_h2h --resume {run}')
        return 0
    watch(cl, d, st, a.poll, log)
    finish_print(cl, d, st)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
