"""The torch worker's gate (docs/az.md 78.4; zero-farm's per-machine torch runtime): the torch worker's game records,
replayed by the trainer's path, against the rows the trainer's own local self-play produces for the same seed, game
ids and network - row for row, every array bit for bit.

    python tools/torch_worker_gate.py --root <game checkout> --package mtgz.az --worker <build>/worker_main.py \
        --settings <run config.json> --checkpoint <iter_N.pt> --out DIR [--games 24] [--device cpu|cuda]
        [--precision fp32|bf16] [--conc C] [--groups G] [--threads T] [--seed S] [--gid-base G] [--set key=JSON ...]
        [--reference reference.npz] [--json PATH] [--no-isolate]

1. The checkpoint is written as the gen's `torch-weights/1` variant (zero.torch_weights.export, the run's settings with
   --precision as infer_precision).
2. **The worker**: `<python> -I -X utf8 -u <worker> serve` (the build's entry, as zf-pyshim starts it) plays one
   classic job of `--games` games from `--gid-base` with `--seed` - the job line an agent sends - and writes records.
3. **The trainer**: in this process, the trainer's local self-play path on the same settings - `Run.driver` (rolling,
   queue_only, recording, the same seed, numbered from the same first gid), `NetEvaluator` at the precision (a token
   network's CUDA graphs on CUDA, as ZERO_TOKEN_GRAPHS=1 gives a run), `infer.drive` with a setup queued after every
   batch up to `conc` in flight (the cadence of `Run.selfplay_shared`'s pump and of the worker), `take_samples` cut per
   game (`workqueue.split_games`).
4. Every record is replayed by the pool driver's path (`<pkg>._engine.replay_rows` on the header's spec; the seed must be
   its gid's, `game_seed_of`; the header as zero.farm_pool checks it) and its rows compared with the trainer's rows of the
   same gid: every array, dtype, shape and bytes. The game ids must be the same set.

What it proves: with a batch order that is deterministic (groups 1, threads 1: one batch at a time in a fixed order)
the worker plays exactly the trainer's games - same seat, seeds, ids, evaluator arithmetic and records - and its records
replay into the trainer's rows. With groups > 1 or several search threads the batches' composition follows thread
timing, and a bf16 evaluator's answers depend on the batch's shape, so games may diverge after the first different
batch: the report then counts identical games, and every record must still replay (legal, its seed its gid's, rows
finite) - the gate passes only on the identical-games case unless `--allow-divergence` (a measurement, not a gate).

Prints one JSON summary; last line "torch worker gate: PASSED ..." or "FAILED ..."; exit 0 iff passed.
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_set(kvs):
    out = {}
    for kv in kvs or []:
        k, v = kv.split('=', 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def run_worker(python, worker, job, cwd, timeout=3600, env=None, isolate=True):
    """The worker's entry, fed one job line; -> (events, stderr tail, exit code)."""
    p = subprocess.Popen([python, *(['-I'] if isolate else []), '-X', 'utf8', '-u', worker, 'serve'], cwd=cwd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env)
    err = []
    threading.Thread(target=lambda: err.extend(p.stderr), daemon=True).start()
    evs = []
    first = p.stdout.readline()
    evs.append(json.loads(first))
    p.stdin.write(json.dumps(job) + '\n')
    p.stdin.flush()
    t0 = time.time()
    for ln in p.stdout:
        e = json.loads(ln)
        evs.append(e)
        if e.get('ev') in ('end',) or (e.get('ev') == 'error' and e.get('fatal')):
            break
        if time.time() - t0 > timeout:
            break
    try:
        p.stdin.close()
    except OSError:
        pass
    rc = p.wait(60)
    return evs, ''.join(err[-60:]), rc


def trainer_rows(cfg, ckpt, gen, games, seed, gid_base, device, rdir):
    """The trainer's local self-play (module docs, step 3) -> {gid: (row, samples dict or None)}."""
    from zero import games as G
    from zero.infer import NetEvaluator, drive
    from zero.net import load
    from zero.run import Run
    from zero.workqueue import split_games
    import torch
    torch.set_num_threads(max(1, int(cfg.get('torch_threads', 1) or 1)))   # (the run's, as its workers set it)
    R = Run(rdir, cfg)
    seat = R.nn_seat('m', cfg['sims'], selfplay=True)
    drv = R.driver([[seat] * int(p) for p in cfg['players']], seed=seed, record=True, rolling=True, net=gen, queue_only=True)
    if gid_base:
        drv.set_first_gid(gid_base)
    m, _ = load(ckpt, device, game=cfg['game'])
    graphs = True if str(device).startswith('cuda') and m.config.get('arch') == 'tokens' else None
    ev = NetEvaluator(m, device, graphs=graphs, workers=cfg.get('infer_workers', 1),
                      token_chunks=cfg.get('infer_token_chunks', 0), precision=cfg.get('infer_precision', 'fp32'))
    spec = G.spec_json(cfg['spec'])
    tws = G.tail_widths(G.info(cfg['game']))
    st = {'enq': 0, 'fin': 0}
    out = {}
    cap = int(cfg['conc'])

    def pump(lb=None):
        rows = drv.take_results()
        s = drv.take_samples()
        if rows:
            for r, sd in split_games(rows, s, tws):
                out[int(r['gid'])] = (r, sd)
            st['fin'] += sum(1 for r in rows if not r.get('branch'))
        room = min(max(0, cap - (st['enq'] - st['fin'])), games - st['enq'])
        for _ in range(room):
            drv.enqueue(f'T:{st["enq"]}', spec)
            st['enq'] += 1
    drv.resume(1 << 62, gen)
    pump()
    while st['fin'] < games:
        drive(drv, {'m': ev}, on_batch=pump)
        pump()
        if st['fin'] < games and drv.in_flight()['games'] == 0 and not drv.queued:
            time.sleep(0.01)
    return out


# a record carries the search's outputs at the shard's own precision (zero_search::record: pi and q as float16 bits), so
# the trainer's rows are compared as its shard stores them
F16_KEYS = ('pi', 'q')


def same(a, b):
    """A replayed game's rows `a` and the trainer's `b` (or None) equal array for array - dtype, shape, bytes; `b`'s pi / q
    rounded to float16 as a shard stores them -> (ok, why)."""
    import numpy as np
    if a is None or b is None:
        return (a is None and b is None), 'one side has no rows'
    if set(a) != set(b):
        return False, f'keys {sorted(set(a) ^ set(b))}'
    for k in sorted(a):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if k in F16_KEYS:
            y = y.astype(np.float16).astype(y.dtype)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False, f'{k}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}'
    return True, ''


def replay_files(out_dir, game, aux, seed):
    """Every record of the worker's files replayed -> ({gid: samples}, header problems, the headers)."""
    from zero import games as G, package
    E = G.engine()
    got, probs, heads = {}, [], []
    for f in sorted(glob.glob(os.path.join(out_dir, '*.rec.jsonl.gz'))):
        with open(f, 'rb') as fh:
            lines = gzip.decompress(fh.read()).decode('utf-8').splitlines()
        h = json.loads(lines[0])
        heads.append(h)
        if package.record_game(h) != game or h.get('aux') != aux or h.get('seed') != seed or h.get('records') not in (1, 2):
            probs.append(f'{os.path.basename(f)}: header {h}')
        parent = {}
        n_rows = 0
        for ln in lines[1:]:
            x = json.loads(ln)
            br = x.get('branch')
            if br is None:
                parent[x['gid']] = ln
                s, info = E.replay_rows(game, h['spec'], ln, aux)
            else:
                s, info = E.replay_rows(game, h['spec'], ln, aux, parent=parent[x['gid']])
            if info['seed'] != E.game_seed_of(seed, info['gid']):
                probs.append(f'game {info["gid"]}: seed {info["seed"]} is not its gid\'s')
            got[(int(info['gid']), br is not None)] = s
            n_rows += info['rows']
        if n_rows != h['rows']:
            probs.append(f'{os.path.basename(f)}: {n_rows} rows replayed, the header says {h["rows"]}')
    return got, probs, heads


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--worker', required=True, help="the build's worker_main.py")
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--gen', type=int, default=None)
    ap.add_argument('--reference')
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--gid-base', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7785777)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--precision', default='fp32')
    ap.add_argument('--conc', type=int)
    ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--allow-divergence', action='store_true')
    ap.add_argument('--no-isolate', action='store_true', help="start the worker without -I (a development Python whose "
                    "packages are in its user site; the runtime pack's Python has them in its own tree)")
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    sys.path[:0] = [os.path.abspath(a.root), os.path.join(ROOT, 'python')]
    __import__(a.package)
    from zero import farm_torch_worker as W, torch_weights as TW
    with open(a.settings, encoding='utf-8') as f:
        raw = json.load(f)
    raw.update(parse_set(a.set))
    raw['infer_precision'] = a.precision
    os.makedirs(a.out, exist_ok=True)
    gen = a.gen if a.gen is not None else int(os.path.basename(a.checkpoint).split('_')[-1].split('.')[0])
    w = TW.export(a.checkpoint, os.path.join(a.out, 'model-torch.pt'), raw, gen=gen)
    cfg = W.job_settings(raw, a.threads, a.conc, a.groups)
    for k in ('infer_precision', 'infer_token_chunks', 'infer_workers', 'torch_threads'):
        cfg[k] = raw.get(k, cfg.get(k))
    wd = os.path.join(a.out, 'worker')
    os.makedirs(wd, exist_ok=True)
    for f in glob.glob(os.path.join(wd, 'out', '*')):
        os.remove(f)
    job = {'op': 'job', 'lease_id': 'gate', 'kind': 'selfplay', 'settings': raw, 'games': a.games, 'gid_base': a.gid_base,
           'seed': a.seed, 'model': os.path.abspath(w), 'reference': os.path.abspath(a.reference) if a.reference else None,
           'gen': gen, 'threads': a.threads, 'groups': a.groups, 'out_dir': 'out', 'device': a.device}
    if a.conc:
        job['conc'] = a.conc
    t0 = time.time()
    env = dict(os.environ, ZERO_TORCH_WORKER_DEVICE=a.device)
    evs, err, rc = run_worker(a.python, os.path.abspath(a.worker), job, wd, env=env, isolate=not a.no_isolate)
    t_worker = time.time() - t0
    end = next((e for e in evs if e.get('ev') == 'end'), None)
    summary = {'worker_rc': rc, 'worker_s': round(t_worker, 1), 'end': end and {k: end[k] for k in ('reason', 'unfinished')},
               'hello': evs[0], 'selftest': [e for e in evs if e.get('ev') == 'selftest']}
    if end is None or rc != 0:
        summary['error'] = [e for e in evs if e.get('ev') == 'error'] or err[-2000:]
        print(json.dumps(summary, indent=1))
        print('torch worker gate: FAILED (the worker did not end its job)')
        return 1
    from zero import games as G
    from zero.run import Run
    aux = bool(Run(os.path.join(a.out, 'trainer'), cfg).aux)
    got, probs, heads = replay_files(os.path.join(wd, 'out'), cfg['game'], aux, a.seed)
    t0 = time.time()
    want = trainer_rows(cfg, a.checkpoint, gen, a.games, a.seed, a.gid_base, a.device, os.path.join(a.out, 'trainer'))
    summary['trainer_s'] = round(time.time() - t0, 1)
    wg = {g for g, br in got if not br}
    tg = {g for g, (r, _) in want.items() if not r.get('branch')}
    summary['games'] = {'worker': len(wg), 'trainer': len(tg), 'same_ids': wg == tg}
    ident, diffs = 0, []
    for (g, br), s in sorted(got.items()):
        if br or g not in want:
            continue
        ok, why = same(s, want[g][1])
        if ok:
            ident += 1
        elif len(diffs) < 8:
            diffs.append(f'game {g}: {why}')
    summary.update(identical=ident, compared=len([1 for g, br in got if not br and g in want]), diffs=diffs,
                   problems=probs, rows=sum(h['rows'] for h in heads), files=len(heads))
    passed = (not probs and wg == tg and len(wg) == a.games
              and (ident == summary['compared'] or a.allow_divergence))
    print(json.dumps(summary, indent=1, default=str))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1, default=str)
    print(f'torch worker gate: {"PASSED" if passed else "FAILED"} - {ident} of {summary["compared"]} games identical '
          f'to the trainer\'s ({a.device}, {a.precision}, groups {a.groups}, threads {a.threads}), {summary["rows"]} rows, '
          f'{len(probs)} replay problems')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
