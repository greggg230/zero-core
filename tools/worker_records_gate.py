"""Gate 3 (zero-farm phase 3; docs/az.md 43.4): a native worker's game records rebuild, bit for bit, the shard rows it
would have written.

  python tools/worker_records_gate.py --worker <mtgz-worker.exe> --settings <run config.json> --package mtgz.az
      (--model <graph.onnx> --runtime-dir <runtime pack> | --uniform) --games 48 [--seed 7785777] [--gid-base 0]
      [--net 80] [--threads 8] [--conc 48] [--groups 1] [--switch-model <graph.onnx> --switch-gen 81 --switch-after 4]
      [--set key=value ...] [--out DIR] [--json PATH]

The worker plays the games once with `--output both` (zero_worker: the same finished games written as shards AND as
records, files cut at the same games). Then, for every records file, each game is replayed by the trainer's path - the
game package's engine module `replay_rows` (zero_search::record::replay: new_game from the record's seed, every real
move, the samples' rows rebuilt from the game; the record's search outputs pi / q / net) - and the rows written by the
trainer's shard writer (zero.data.ShardWriter, the run's shard format); every array of that shard must equal the
worker's own shard of the same games: dtype, shape and bytes - features (sparse or dense), players, options, policy,
z / q / sc, the network ids, the auxiliary targets. A game's setup record (`SearchGame::record_setup`: MTG Zero's
per-seat decks) is compared by the replay itself. `--switch-model` sends a `checkpoint` (a phase change: games in flight
go on under the new network id) after `--switch-after` finished games, so rows of two networks are checked. The
records' game seeds must be their game ids' (`game_seed_of`). Exit 0 iff every array of every file is identical.
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


def run_worker(a, settings_path, out):
    args = [a.worker, 'selfplay', '--settings', settings_path, '--games', str(a.games), '--gid-base', str(a.gid_base),
            '--seed', str(a.seed), '--threads', str(a.threads), '--out', out, '--prefix', 'g', '--output', 'both',
            '--status-s', '30']
    if a.conc:
        args += ['--conc', str(a.conc)]
    if a.groups:
        args += ['--groups', str(a.groups)]
    if a.uniform:
        args += ['--uniform']
    else:
        dll = os.path.join(a.runtime_dir, 'onnxruntime.dll') if a.runtime_dir else a.ort_dll
        args += ['--model', a.model, '--ort-dll', dll]
    if a.net is not None:
        args += ['--net', str(a.net)]
    if a.switch_model:
        args += ['--stdin-control']
    env = dict(os.environ)
    if a.runtime_dir:
        env['PATH'] = a.runtime_dir + os.pathsep + env.get('PATH', '')
    t0 = time.time()
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding='utf-8', env=env)
    err = []
    threading.Thread(target=lambda: err.extend(p.stderr), daemon=True).start()
    events, fin, switched = [], 0, False
    for line in p.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        events.append(ev)
        if ev.get('ev') == 'fin':
            fin += ev['n']
            if a.switch_model and not switched and fin >= a.switch_after:
                p.stdin.write(json.dumps({'op': 'checkpoint', 'gen': a.switch_gen, 'model': os.path.abspath(a.switch_model)}) + '\n')
                p.stdin.flush()
                switched = True
        if ev.get('ev') == 'status':
            s = ev['stats']
            print(f'# worker: {s.get("games")} games, {s.get("leaves_per_s")} leaves/s, {s.get("wall_s")} s', flush=True)
    if a.switch_model:
        try:
            p.stdin.close()
        except OSError:
            pass
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f'the worker failed (exit {rc}):\n{"".join(err)[-3000:]}')
    end = next((e for e in events if e.get('ev') == 'end'), None)
    return events, end, time.time() - t0, switched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--package', required=True, help='the game package (e.g. mtgz.az): its engine module replays')
    ap.add_argument('--root', help='the game repo (its package importable from here; default: the cwd)')
    ap.add_argument('--model')
    ap.add_argument('--runtime-dir')
    ap.add_argument('--ort-dll')
    ap.add_argument('--uniform', action='store_true')
    ap.add_argument('--games', type=int, default=48)
    ap.add_argument('--seed', type=int, default=7785777)
    ap.add_argument('--gid-base', type=int, default=0)
    ap.add_argument('--net', type=int)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--conc', type=int)
    ap.add_argument('--groups', type=int)
    ap.add_argument('--switch-model')
    ap.add_argument('--switch-gen', type=int)
    ap.add_argument('--switch-after', type=int, default=4)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', default=None)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    if not a.uniform and not a.model:
        ap.error('--model or --uniform')
    if a.switch_model and a.switch_gen is None:
        ap.error('--switch-gen with --switch-model')
    sys.path.insert(0, a.root or os.getcwd())
    sys.path.insert(0, os.path.join(ROOT, 'python'))
    import importlib
    import numpy as np
    importlib.import_module(a.package)
    from zero import games as G
    from zero.data import ShardWriter, shard_format
    from zero.run import DEFAULTS
    E = G.engine()

    with open(a.settings, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.update(parse_set(a.set))
    out = a.out or os.path.join(os.getcwd(), f'records-gate-{int(time.time())}')
    os.makedirs(out, exist_ok=True)
    sp = os.path.join(out, 'settings.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    wout = os.path.join(out, 'worker')
    events, end, secs, switched = run_worker(a, sp, wout)
    print(f'# worker: {end["stats"]["games"]} games in {secs:.0f} s; switched: {switched}', flush=True)

    rc = dict(DEFAULTS)
    rc.update(cfg)
    game = rc['game']
    spec = G.spec_json(rc['spec'])
    fmt = shard_format(rc)
    recs = sorted(glob.glob(os.path.join(wout, 'g_*.rec.jsonl.gz')))
    shards = sorted(p for p in glob.glob(os.path.join(wout, 'g_*.npz')) if not p.endswith('.bad.npz'))
    rep = {'worker': a.worker, 'settings': a.settings, 'set': parse_set(a.set), 'game': game, 'games': 0, 'rows': 0,
           'value_rows': 0, 'aux_rows': 0, 'nets': {}, 'setups': 0, 'files': len(recs), 'shards': len(shards),
           'arrays': 0, 'different': [], 'seconds_worker': round(secs, 1), 'replay_s': 0.0, 'record_bytes': 0,
           'shard_bytes': 0, 'switched': switched}
    if len(recs) != len(shards) or not recs:
        raise SystemExit(f'{len(recs)} records files, {len(shards)} shards')
    for rp, shp in zip(recs, shards):
        with gzip.open(rp, 'rt', encoding='utf-8') as f:
            lines = f.read().splitlines()
        h = json.loads(lines[0])
        aux = bool(h['aux'])
        if json.loads(h['spec'] or '{}') != json.loads(spec):
            raise SystemExit(f'{rp}: the records\' spec is not the run\'s')
        ref = os.path.join(out, 'replayed', os.path.basename(shp)[:-4])
        w = ShardWriter(ref, 'r', rows=1 << 40, meta={'game': game, 'features': h['features'], 'actions': h['actions'], **fmt})
        for ln in lines[1:]:
            t0 = time.time()
            s, info = E.replay_rows(game, spec, ln, aux)
            rep['replay_s'] += time.time() - t0
            if info['seed'] != E.game_seed_of(a.seed, info['gid']):
                raise SystemExit(f'game {info["gid"]}: its seed is not its game id\'s')
            rep['games'] += 1
            rep['setups'] += bool(info['setup'])
            if s is None:
                continue
            rep['rows'] += len(s['player'])
            rep['value_rows'] += int((np.diff(s['opt_off']) == 0).sum())
            if 'aux' in s:
                rep['aux_rows'] += int((np.abs(s['aux']).sum(1) > 0).sum())
            for n in s['net'].tolist():
                rep['nets'][str(n)] = rep['nets'].get(str(n), 0) + 1
            w.add(s)
        w.close()
        got = sorted(glob.glob(os.path.join(ref, 'r_*.npz')))
        if len(got) != 1:
            raise SystemExit(f'{rp}: {len(got)} replayed shards')
        x, y = np.load(shp), np.load(got[0])
        rep['record_bytes'] += os.path.getsize(rp)
        rep['shard_bytes'] += os.path.getsize(shp)
        names = sorted(k for k in x.files if k != 'meta')
        if names != sorted(k for k in y.files if k != 'meta'):
            rep['different'].append({'file': os.path.basename(shp), 'members': [names, sorted(y.files)]})
            continue
        for k in names:
            rep['arrays'] += 1
            if not (x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes()):
                rep['different'].append({'file': os.path.basename(shp), 'array': k})
        mx, my = json.loads(str(x['meta'])), json.loads(str(y['meta']))
        if mx['rows'] != my['rows']:
            rep['different'].append({'file': os.path.basename(shp), 'meta rows': [mx['rows'], my['rows']]})
    rep['replay_s'] = round(rep['replay_s'], 2)
    rep['identical'] = not rep['different'] and rep['rows'] > 0
    print(json.dumps(rep))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1)
    print(('GATE 3 PASSED' if rep['identical'] else 'GATE 3 FAILED') + f': {rep["games"]} games, {rep["rows"]} rows '
          f'({rep["value_rows"]} value-only, {rep["aux_rows"]} with auxiliary targets), nets {rep["nets"]}, '
          f'{rep["setups"]} setups compared, {rep["arrays"]} arrays in {rep["files"]} files')
    return 0 if rep['identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
