"""Dump the driver settings (`SearchDriver`'s `DriverCfg`, `<engine>.driver_settings`) a game package's
runs build, for every preset and run config given - the evidence that a change to the seat parser (or to
the run settings -> seat code) changes no seat (zero-farm phase 0b: the parser moved from zero_py into
zero_search, docs/az.md 42).

    python tools/driver_settings_dump.py --root <game checkout> --package mtgz.az --configs <runs dir>... \
        --out old.json [--cfg-dir <dir for the resolved configs>]
    python tools/driver_settings_dump.py --compare old.json new.json
    python tools/driver_settings_dump.py --compare-worker new.json --worker <native worker exe>

Per case (each preset of the package, each `<runs dir>/*/config.json`): the settings resolved as a run
resolves them (DEFAULTS < preset < saved config: `zero.run.resolve_config`), then the drivers the run
builds from them, each as `driver_settings` text (or 'ERROR: ...' when the settings are refused):

* `selfplay_remote`: a remote self-play worker's (`zero.remoteplay.Worker._load`: the self-play seat,
  the table of the run's player counts - the league's table with the league on - rolling, queue_only,
  seed `--seed`, net 0);
* `selfplay_local`: this machine's self-play driver (`Run._selfplay`: rolling, or the table of a
  non-rolling iteration's games);
* `eval_yard`, `eval_random`: the evaluation seat against the yardstick / the random bot
  (`Run.play_match`: seats alternating, 6 games, paired with elo_paired);
* `elo`: the evaluation seat against two checkpoints in one driver (`Run.play_matches`, keys o0, o1);
* `bots`: the yardstick against the random bot (`Run.bots_match`).

`--compare` lists, per case and driver, equal / both refused / DIFFERENT and exits 1 on any difference.
`--compare-worker` runs `<worker> driver-cfg <resolved config> --seed S --net 0` for every case whose
`selfplay_remote` was built and compares its text with it (the native worker's settings -> driver path,
a Rust port of the run's self-play seat; exit 1 on a difference). Reads the configs only.
"""
import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback

SEED = 4242
CASES = ('selfplay_remote', 'selfplay_local', 'eval_yard', 'eval_random', 'elo', 'bots')


def _err(e):
    msg = str(e).strip().splitlines()
    return f'ERROR: {type(e).__name__}: {msg[-1] if msg else ""}'[:600]


def dump(args):
    sys.path.insert(0, os.path.abspath(args.root))
    importlib.import_module(args.package)
    from zero import games as G
    from zero import league as L
    from zero import run as R
    eng = G.engine()
    if not hasattr(eng, 'driver_settings'):
        raise SystemExit(f'{eng.__name__} has no driver_settings (built from a zero-core before it): rebuild it')
    print(f'zero: {os.path.dirname(R.__file__)}  engine: {eng.__file__}', file=sys.stderr)

    class Captured(Exception):
        pass

    def fake_driver(game, seats, spec=None, **kw):
        raise Captured(eng.driver_settings(seats, spec=G.spec_json(spec), **kw))

    real = G.Driver
    G.Driver = fake_driver
    ns = argparse.Namespace(**{k: None for k in R.DEFAULTS})
    items = [(f'preset:{p}', None) for p in sorted(R.PRESETS)]
    for d in args.configs:
        for p in sorted(glob.glob(os.path.join(d, '*', 'config.json'))):
            items.append((f'config:{p}', p))
    out = {}
    os.makedirs(args.cfg_dir, exist_ok=True)
    try:
        for name, path in items:
            row = {}
            out[name] = row
            try:
                if path is None:
                    saved = {'preset': name.split(':', 1)[1]}
                    rdir = tempfile.mkdtemp(prefix='zds-')
                else:
                    with open(path, encoding='utf-8') as f:
                        saved = json.load(f)
                    rdir = tempfile.mkdtemp(prefix="zds-")   # (never the run directory: read only)
                cfg = R.resolve_config(ns, saved)
            except BaseException as e:             # noqa: BLE001 - a refused config is a result
                row['resolve'] = _err(e)
                continue
            cp = os.path.join(args.cfg_dir, f'{len(out):03d}.json')
            with open(cp, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=1)
            row['cfg'] = cp
            try:
                run = R.Run(rdir, cfg)
            except BaseException as e:             # noqa: BLE001
                row['run'] = _err(e)
                continue
            c = cfg

            def one(key, fn):
                try:
                    fn()
                    row[key] = 'ERROR: no driver built'
                except Captured as cap:
                    row[key] = cap.args[0]
                except BaseException as e:         # noqa: BLE001
                    row[key] = _err(e)
                    if args.verbose:
                        traceback.print_exc()

            def selfplay_remote():
                seat = run.nn_seat('m', c['sims'], selfplay=True)
                table = [[seat] * p for p in c['players']]
                if L.enabled(c):
                    table, _ = L.rolling_table(c, seat)
                run.driver(table, seed=SEED, record=True, rolling=True, net=0, queue_only=True)

            def selfplay_local():
                seat = run.nn_seat('m', c['sims'], selfplay=True)
                if c['rolling']:
                    table = [[seat] * p for p in c['players']]
                    if L.enabled(c):
                        table, _ = L.rolling_table(c, seat)
                    run.driver(table, seed=c['seed'] * 7919 + 1, record=True, rolling=True, net=0)
                else:
                    import numpy as np
                    counts = np.random.default_rng(c['seed'] * 1_000_003 + 1).choice(c['players'], size=min(c['games'], 64))
                    if L.enabled(c):
                        games, _ = L.table(c, seat, counts)
                    else:
                        games = [[seat] * int(p) for p in counts]
                    run.driver(games, seed=c['seed'] * 1_000_003 + 1, record=True)

            def match(opp):
                def f():
                    a = run.nn_seat('m', c['eval_sims'])
                    o = run.opp_seat(opp, c['eval_sims'])
                    seats = [[a if s == g % 2 else o for s in range(2)] for g in range(6)]
                    run.driver(seats, seed=7919 + 1, conc=min(c['conc'], 6),
                               groups=run.match_groups(G.is_external(opp)), paired=bool(c.get('elo_paired')))
                return f

            def elo():
                a = run.nn_seat('m', c['eval_sims'])
                seats = []
                for t in range(2):
                    o = run.nn_seat(f'o{t}', c['eval_sims'])
                    seats += [[a if s == g % 2 else o for s in range(2)] for g in range(4)]
                run.driver(seats, seed=17, conc=min(c['conc'], len(seats)), paired=bool(c.get('elo_paired')))

            def bots():
                a, b = c['yardstick'], 'random'
                seats = [[G.seat_dict(a), G.seat_dict(b)] if g % 2 == 0 else [G.seat_dict(b), G.seat_dict(a)]
                         for g in range(4)]
                run.driver(seats, seed=5, conc=min(c['conc'], 4),
                           groups=run.match_groups(G.is_external(a) or G.is_external(b)))

            one('selfplay_remote', selfplay_remote)
            one('selfplay_local', selfplay_local)
            one('eval_yard', match(c['yardstick']))
            one('eval_random', match('random'))
            one('elo', elo)
            one('bots', bots)
            print(f'{name}: ' + ', '.join(f'{k} {"ERR" if v.startswith("ERROR") else "ok"}'
                                          for k, v in row.items() if k in CASES), file=sys.stderr)
    finally:
        G.Driver = real
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    built = sum(1 for r in out.values() for k in CASES if k in r and not r[k].startswith('ERROR'))
    print(f'{len(out)} cases, {built} drivers built -> {args.out}', file=sys.stderr)


def compare(a_path, b_path):
    with open(a_path, encoding='utf-8') as f:
        a = json.load(f)
    with open(b_path, encoding='utf-8') as f:
        b = json.load(f)
    bad = 0
    equal = refused = 0
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if ra is None or rb is None:
            print(f'DIFFERENT {name}: only in {"the first" if rb is None else "the second"}')
            bad += 1
            continue
        for k in ('resolve', 'run') + CASES:
            va, vb = ra.get(k), rb.get(k)
            if va is None and vb is None:
                continue
            if va == vb:
                if va.startswith('ERROR'):
                    refused += 1
                    print(f'both refused  {name} {k}: {va}')
                else:
                    equal += 1
            elif (va or '').startswith('ERROR') and (vb or '').startswith('ERROR'):
                refused += 1
                print(f'both refused  {name} {k}:\n    old: {va}\n    new: {vb}')
            else:
                bad += 1
                print(f'DIFFERENT {name} {k}:\n--- {a_path}\n{va}\n--- {b_path}\n{vb}')
    print(f'{equal} drivers identical, {refused} refused by both, {bad} different ({len(a)} / {len(b)} cases)')
    return 1 if bad else 0


def compare_worker(path, worker):
    with open(path, encoding='utf-8') as f:
        a = json.load(f)
    bad = same = skipped = unsupported = 0
    for name, r in sorted(a.items()):
        want = r.get('selfplay_remote')
        if want is None or 'cfg' not in r:
            skipped += 1
            continue
        p = subprocess.run([worker, 'driver-cfg', r['cfg'], '--seed', str(SEED), '--net', '0'], capture_output=True,
                           text=True, encoding='utf-8')
        got = p.stdout.rstrip('\n') if p.returncode == 0 else f'ERROR: {p.stderr.strip()[-600:]}'
        if want.startswith('ERROR') and got.startswith('ERROR'):
            print(f'both refused  {name}:\n    python: {want}\n    worker: {got}')
            same += 1
        elif got == want:
            same += 1
        elif got.startswith('ERROR') and 'unsupported by the native worker' in got:
            # a Python-only feature (the league, branched games, archive starts): refused by design, and
            # check-config lists it
            unsupported += 1
            print(f'worker refuses (Python-only feature)  {name}: {got}')
        else:
            bad += 1
            print(f'DIFFERENT {name} (python vs worker):\n--- python\n{want}\n--- worker\n{got}')
    print(f'{same} cases agree (built or both refused), {unsupported} refused by the worker as Python-only, {bad} '
          f'different, {skipped} without a self-play driver')
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root')
    ap.add_argument('--package')
    ap.add_argument('--configs', nargs='*', default=[])
    ap.add_argument('--out')
    ap.add_argument('--cfg-dir')
    ap.add_argument('--compare', nargs=2)
    ap.add_argument('--compare-worker')
    ap.add_argument('--worker')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if args.compare_worker:
        sys.exit(compare_worker(args.compare_worker, args.worker))
    if not (args.root and args.package and args.out):
        ap.error('--root, --package and --out (or --compare / --compare-worker)')
    args.cfg_dir = args.cfg_dir or os.path.splitext(args.out)[0] + '-cfg'
    dump(args)


if __name__ == '__main__':
    main()
