"""The eval gate (zero-farm phase 2, docs/az.md 44.2): the native worker's eval job against the Python path of
an evaluation match - `Run.play_match`'s driver (zero.farm_eval.match_driver: Run.nn_seat(key, eval_sims)
seats, Run.driver(table, seed, conc, paired)) driven by `infer.drive` with zero-core's `OnnxEvaluator` on the
same ONNX graphs - at deterministic settings (groups 1): every result row must be byte-identical.

    python tools/worker_eval_gate.py --root <game checkout> --package mtgz.az --settings <job settings.json> \
        --model 1=<a.onnx> --model 2=<b.onnx> --seats '[{"net": 1}, {"net": 2}]' --games 20 --seed 31 \
        --paired --threads 4 --infer-threads 4 --worker <mtgz-worker exe> --ort-dll <onnxruntime.dll> \
        --out <dir> [--batches 50] [--side both|python|native] [--seat-cases all|merge,sims,gumbel,belief,select]

`--settings`: the eval job's settings (a run's resolved config with `eval_spec` set: zero.farm_eval.
resolve_eval_spec), given as they are to both sides (the worker reads the same file); `--threads` is both
sides' search threads (the job's threads over the settings'), `--infer-threads` ONNX Runtime's intra-op
threads on both sides (the same pieces and threads: the graph's outputs bit for bit, docs/az.md 44 gate b).
The Python side plays games 0..N in one driver (a fixed table: Run.play_match's own); the native side one
eval job of games 0..N (a rolling queue_only driver, docs/search.md 2.16). Compared: each game's row (bytes),
and the first `--batches` network batches each side sent (every array; diagnostics). Writes
<out>/python.jsonl, <out>/native/e_*.jsonl, <out>/gate.json; exit 1 on a difference.

`--seat-cases` (per-seat settings, docs/az.md 68): instead of one match, one per case, each side of `--seats` given its
own settings by zero.h2h's per-side options (zero.seat_settings.side_set over the settings: the seats' sets, as
zero.farm_h2h and farm campaigns make them) - `merge` (side A merges its trees, root-view values on both; B does not),
`sims` (A 2x, B 1/2 of eval_sims), `gumbel` (A the Gumbel root search with 4 considered options, B PUCT), `belief` (A
belief tilt 1, B 0: the networks need the auxiliary head), `select` (A 'cq' with guard 2, B 'visits'). Each case in
<out>/<case>/ (as above); gate.json holds every case; exit 1 when any case differs. `--seats '[{"net": 1}, {"net": 1}]'`
plays both sides with one network (the same checkpoint, two settings: an evaluation-only A/B).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

KEYS = ('xs_off', 'xs_idx', 'xs_val', 'n_players', 'player', 'gid', 'extra', 'opt_off', 'opt_action')
VARLEN = ('t_off', 't_seg', 't_slot', 't_val', 'ov_off', 'ov_opt', 'ov_tok')   # (variable-length rows, docs/az.md 46)


def gen_pairs(vals, what):
    out = {}
    for kv in vals or []:
        g, _, v = kv.partition('=')
        if not g.isdigit() or not v:
            raise SystemExit(f'{what} {kv!r}: GEN=VALUE')
        out[int(g)] = v
    return out


def python_side(a, cfg, seats, models):
    sys.path.insert(0, os.path.abspath(a.root))
    __import__(a.package)
    import numpy as np
    from zero import farm_eval as FE
    from zero.infer import drive
    from zero.onnx_eval import OnnxEvaluator

    rec = []

    class Recording:
        def __init__(self, ev):
            self.ev = ev

        def __call__(self, lb):
            if len(rec) < a.batches:
                rec.append({k: np.array(lb.a[k]) for k in KEYS + VARLEN if k in lb.a})
            return self.ev(lb)

        def launch(self, lb):
            return self(lb)

        def result(self, h):
            return h

    by_gen = {g: Recording(OnnxEvaluator(p, threads=a.infer_threads)) for g, p in models.items()}
    evs = {}
    for k, s in zip(('m', 'o'), seats):
        if 'net' in s:
            evs[k] = by_gen[s['net']]
    drv = FE.match_driver(cfg, seats, a.games, a.seed, paired=a.paired)
    t0 = time.time()
    drive(drv, evs)
    secs = time.time() - t0
    rows = {r['gid']: FE.row_json(FE.row(r, seats, a.paired)) for r in drv.take_results()}
    return rows, rec, secs


SEAT_CASES = {
    'merge': ['--a-infoset-merge', '1', '--b-infoset-merge', '0', '--a-leaf-value-view', 'root', '--b-leaf-value-view', 'root'],
    'sims': ['--a-sims', '{2x}', '--b-sims', '{half}'],
    'gumbel': ['--a-root-search', 'gumbel', '--a-gumbel-m', '4', '--b-root-search', 'puct'],
    'belief': ['--a-belief-tilt', '1', '--b-belief-tilt', '0'],
    'select': ['--a-select', 'cq', '--a-select-min-visits', '2', '--b-select', 'visits'],
}


def case_seats(cfg, seats, case):
    """The seats of a seat case: `seats` (two network seats) with the case's per-side options as their sets
    (zero.seat_settings: side_set over `cfg`, both sides relative to the one job's settings)."""
    import argparse
    from zero import farm_eval as FE, seat_settings as SS
    sims = int(cfg['eval_sims'])
    argv = [x.replace('{2x}', str(2 * sims)).replace('{half}', str(max(1, sims // 2))) for x in SEAT_CASES[case]]
    ap = argparse.ArgumentParser()
    SS.add_side_args(ap, local=False)
    sets = SS.sets_of(ap.parse_args(argv), cfg)
    out = [SS.net_seat(s['net'], st) for s, st in zip(seats, sets)]
    FE.seat_dicts(cfg, out)                      # (each seat's checks: Run.nn_seat)
    return out, argv


def native_side(a, settings_path, models, refs):
    out = os.path.join(a.out, 'native')
    os.makedirs(out, exist_ok=True)
    for p in glob.glob(os.path.join(out, '*.jsonl')):
        os.remove(p)
    bf = os.path.join(a.out, 'native_batches.npz')
    cmd = [a.worker, 'eval', '--settings', settings_path, '--seats', json.dumps(a.seat_list), '--games', str(a.games), '--seed',
           str(a.seed), '--threads', str(a.threads), '--infer-threads', str(a.infer_threads), '--out', out,
           '--ort-dll', a.ort_dll, '--record-batches', str(a.batches), '--record-file', bf, '--flush-games',
           str(a.flush_games)]
    if a.paired:
        cmd.append('--paired')
    for g, p in models.items():
        cmd += ['--model', f'{g}={p}']
    for g, p in refs.items():
        cmd += ['--reference', f'{g}={p}']
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    secs = time.time() - t0
    with open(os.path.join(a.out, 'native.log'), 'w', encoding='utf-8') as f:
        f.write(' '.join(cmd) + '\n' + p.stdout + '\n--- stderr\n' + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f'the worker failed ({p.returncode}): {p.stderr[-2000:]}')
    events = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, '*.jsonl'))):
        with open(f, encoding='utf-8') as fh:
            for line in fh.read().splitlines():
                g = json.loads(line)['gid']
                assert g not in rows, f'game {g} twice'
                rows[g] = line
    return rows, bf, events, secs


def compare_batches(py, bf):
    import numpy as np
    if not os.path.exists(bf):
        return {'compared': 0, 'note': 'no native batch file'}
    z = np.load(bf)
    bo = z['batch_off']
    xo, oo = z['xs_off'], z['opt_off']
    n = min(len(py), len(bo) - 1)
    for k in range(n):
        r0, r1 = int(bo[k]), int(bo[k + 1])
        nat = {'xs_off': xo[r0:r1 + 1] - xo[r0], 'xs_idx': z['xs_idx'][xo[r0]:xo[r1]], 'xs_val': z['xs_val'][xo[r0]:xo[r1]],
               'n_players': z['n_players'][r0:r1], 'player': z['player'][r0:r1], 'gid': z['gid'][r0:r1],
               'extra': z['extra'][r0:r1], 'opt_off': oo[r0:r1 + 1] - oo[r0], 'opt_action': z['opt_action'][oo[r0]:oo[r1]]}
        from worker_parity import varlen_slice
        nat.update(varlen_slice(z, r0, r1))
        keys = KEYS + tuple(x for x in VARLEN if x in nat or x in py[k])
        for key in keys:
            if key not in nat or key not in py[k]:
                return {'compared': k + 1, 'first_difference': {'batch': k, 'array': key, 'python': key in py[k],
                                                                 'native': key in nat}}
            a, b = np.asarray(py[k][key]), np.asarray(nat[key])
            if a.shape != b.shape or not np.array_equal(a.astype(np.float64), b.astype(np.float64)):
                return {'compared': k + 1, 'first_difference': {'batch': k, 'array': key, 'python': list(a.shape),
                                                                 'native': list(b.shape)}}
    return {'compared': n, 'identical': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--model', action='append', required=True)
    ap.add_argument('--reference', action='append', default=[])
    ap.add_argument('--seats', default='[{"net": 1}, {"net": 2}]')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--paired', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--infer-threads', type=int, default=4)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--ort-dll', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, default=50)
    ap.add_argument('--flush-games', type=int, default=8)
    ap.add_argument('--side', default='both', choices=['both', 'python', 'native'])
    ap.add_argument('--seat-cases', default=None, help="per-seat settings' cases (module docs): all, or a comma list of "
                    + ', '.join(SEAT_CASES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = gen_pairs(a.model, '--model')
    refs = gen_pairs(a.reference, '--reference')
    seats = json.loads(a.seats)
    with open(a.settings, encoding='utf-8') as f:
        cfg = json.load(f)
    # (the job's threads over the settings', as the worker reads them; groups as the settings say)
    cfg = dict(cfg, threads=a.threads)
    if a.seat_cases:
        sys.path.insert(0, os.path.abspath(a.root))
        __import__(a.package)
        names = list(SEAT_CASES) if a.seat_cases == 'all' else a.seat_cases.split(',')
        bad = [n for n in names if n not in SEAT_CASES]
        if bad:
            raise SystemExit(f'--seat-cases: unknown {bad} (cases: {", ".join(SEAT_CASES)})')
        out, cases = a.out, {}
        for n in names:
            a.out = os.path.join(out, n)
            os.makedirs(a.out, exist_ok=True)
            a.seat_list, argv = case_seats(cfg, seats, n)
            print(f'# case {n}: {" ".join(argv)} -> seats {json.dumps(a.seat_list)}', file=sys.stderr, flush=True)
            cases[n] = dict(match(a, cfg, models, refs), options=argv)
        res = {'settings': a.settings, 'seat_cases': cases, 'passed': all(c.get('passed') for c in cases.values())}
        with open(os.path.join(out, 'gate.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
        print(json.dumps({'seat_cases': {n: {k: c.get(k) for k in ('seats', 'identical', 'different', 'passed', 'batches',
                                                                    'tags')} for n, c in cases.items()},
                          'passed': res['passed']}))
        return 0 if res['passed'] else 1
    a.seat_list = seats
    res = match(a, cfg, models, refs)
    with open(os.path.join(a.out, 'gate.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return 0 if res.get('passed', True) else 1


def match(a, cfg, models, refs):
    """One match (`a.seat_list`) on both sides, compared -> its result dict (also written to <a.out>/gate.json)."""
    seats = a.seat_list
    res = {'settings': a.settings, 'seats': seats, 'games': a.games, 'seed': a.seed, 'paired': a.paired,
           'threads': a.threads, 'infer_threads': a.infer_threads, 'groups': cfg['groups'], 'conc': min(cfg['conc'], a.games)}
    nat = py = None
    if a.side in ('both', 'native'):
        nrows, bf, events, nsecs = native_side(a, a.settings, models, refs)
        res['native'] = {'games': len(nrows), 'seconds': round(nsecs, 1),
                         'selftests': [e for e in events if e.get('ev') == 'selftest'],
                         'end': next((e for e in events if e.get('ev') == 'end'), None)}
        nat = nrows
        print(f'# native: {len(nrows)} games in {nsecs:.0f} s', file=sys.stderr, flush=True)
    if a.side in ('both', 'python'):
        prows, pb, psecs = python_side(a, cfg, seats, models)
        with open(os.path.join(a.out, 'python.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
            for g in sorted(prows):
                f.write(prows[g] + '\n')
        res['python'] = {'games': len(prows), 'seconds': round(psecs, 1)}
        py = prows
        print(f'# python: {len(prows)} games in {psecs:.0f} s', file=sys.stderr, flush=True)
        if nat is not None:
            res['batches'] = compare_batches(pb, bf)
    if py is not None and nat is not None:
        same = sorted(g for g in py if nat.get(g) == py[g])
        diff = sorted(g for g in set(py) | set(nat) if nat.get(g) != py.get(g))
        res.update(identical=len(same), different=len(diff), different_gids=diff[:20])
        if diff:
            g = diff[0]
            res['first_difference'] = {'gid': g, 'python': py.get(g), 'native': nat.get(g)}
        from zero import farm_eval as FE
        rows = [json.loads(py[g]) for g in sorted(py)]
        if all('net' in s for s in seats):
            res['summary'] = FE.summary(rows, a.paired, seats[0])
            res['tags'] = FE.seat_tags(cfg, seats)
        res['passed'] = not diff and len(same) == a.games
    with open(os.path.join(a.out, 'gate.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == '__main__':
    raise SystemExit(main())
