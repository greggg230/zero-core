"""Start a run from another run's checkpoint.

  python tools/fork_run.py <src.pt> <new run dir> [--game toy]
      A NEW run whose iteration 0 is the source network widened to the game's current
      feature / action counts (new inputs get zero weights: it computes what the source did).
      Settings: the new run's own flags when it is started.

  python tools/fork_run.py --source runs/a --dest runs/b [--iter K] [--set reuse=8 ...] [--copy]
      A CONTINUATION of a (possibly live) run at its checkpoint K (default: the newest): the
      checkpoints <= K that its Elo matches need, the optimizer state when it belongs to K, the
      window's sample shards (hard links on the same volume, --copy to copy), Elo results (every
      pool: elo/ and elo/<rule>/), metrics up to K and the source's config.json with the --set
      overrides. The source is only read. Then: python tools/supervise.py runs/b (the run resumes after K). A co-training run
      (setting `resolver`, docs/az.md 11) is copied with its resolver network: the same of
      <source>/resolver/ into <dest>/resolver/ (K: an iteration where both have a checkpoint).
"""
import argparse
import glob
import json
import os
import shutil
import sys


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def new_run(src, run, game=None):
    import torch  # noqa: F401
    from .. import games
    from ..net import load, save, widen
    from .. import package
    old, extra = load(src)
    info = games.info(game or old.config.get('game', 'toy'))
    grid = package.active().grids.get(game) if game and old.config.get('arch') == 'grid' else None
    new = widen(old, info['num_features'], info['num_actions'], grid=grid)
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    out = os.path.join(run, 'models', 'iter_0000.pt')
    save(new, out, {'iter': 0, 'forked_from': os.path.abspath(src), 'source_extra': extra})
    print(f'{src} ({old.config["features"]} features, {old.config["actions"]} actions) -> {out} '
          f'({new.config["features"]} features, {new.config["actions"]} actions)')


def _link(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def continuation(source, dest, it=None, overrides=None, copy=False):
    from ..run import DEFAULTS, latest_iter, model_path
    if os.path.exists(os.path.join(dest, 'config.json')):
        raise SystemExit(f'{dest} already holds a run')
    with open(os.path.join(source, 'config.json')) as f:
        cfg = json.load(f)
    for k in overrides or {}:
        if k not in DEFAULTS:
            raise SystemExit(f'--set {k}: unknown setting')
    co = bool(cfg.get('resolver'))
    rsrc = os.path.join(source, 'resolver')
    if it is None:
        it = latest_iter(source)
        if co and it is not None:
            r = latest_iter(rsrc)
            it = min(it, r) if r is not None else None
    if it is None or not os.path.exists(model_path(source, it)):
        raise SystemExit(f'{source}: no checkpoint {it}')
    if co:
        if not os.path.exists(model_path(rsrc, it)):
            raise SystemExit(f'{rsrc}: no checkpoint {it} (a co-training run forks where both networks have one)')
        rcfg = cfg
        if os.path.exists(os.path.join(rsrc, 'config.json')):
            with open(os.path.join(rsrc, 'config.json')) as f:
                rcfg = json.load(f)
        _copy_run(rsrc, os.path.join(dest, 'resolver'), it, rcfg, copy)
        if os.path.exists(os.path.join(rsrc, 'config.json')):
            shutil.copy2(os.path.join(rsrc, 'config.json'), os.path.join(dest, 'resolver', 'config.json'))
    need, dirs, rows = _copy_run(source, dest, it, cfg, copy)
    cfg.update(overrides or {})
    with open(os.path.join(dest, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    print(f'{source} @ iter {it} -> {dest}: {len(need)} checkpoints, {len(dirs)} data iterations, {len(rows)} metrics rows'
          + (' (and its resolver network)' if co else ''))
    print(f'next: python tools/supervise.py {dest}   (resumes after iteration {it})')


def _copy_run(source, dest, it, cfg, copy):
    """Checkpoint `it` and those its Elo matches need, its optimizer state, the window's shards,
    Elo results, metrics up to `it` -> (checkpoints, data dirs, metrics rows)."""
    from .. import elo
    from ..run import model_path, read_metrics, window_iters
    e = max(1, int(cfg.get('eval_every', 1)))
    need = {it} | {it - off * e for off in elo.OFFSETS if it - off * e >= 0}
    if cfg.get('elo_offsets'):
        # raw Elo offsets (docs/az.md 5.4): the opponents of K and of the fork's next evaluation
        need |= {j for k in (it, it + e) for j in elo.opponents(k, e, cfg['elo_offsets']) if 0 <= j <= it}
    for j in sorted(need):
        if os.path.exists(model_path(source, j)):
            _link(model_path(source, j), model_path(dest, j), True)
    opt = os.path.join(source, 'models', 'opt_latest.pt')
    if os.path.exists(opt):
        import torch
        o = torch.load(opt, map_location='cpu', weights_only=False)
        if o.get('iter') == it:
            shutil.copy2(opt, os.path.join(dest, 'models', 'opt_latest.pt'))
    dirs = sorted(glob.glob(os.path.join(source, 'data', 'iter_*')), reverse=True)
    dirs = [d for d in dirs if int(os.path.basename(d)[5:9]) <= it][:max(window_iters(cfg, it + 1), 1) + 1]
    for d in dirs:
        for p in glob.glob(os.path.join(d, '*.npz')):
            _link(p, os.path.join(dest, 'data', os.path.basename(d), os.path.basename(p)), copy)
    # every Elo pool: elo/ ('visits') and elo/<rule>/ (another evaluation move rule, docs/az.md 5.4)
    pools = [''] + sorted(d for d in os.listdir(os.path.join(source, 'elo'))
                          if os.path.isdir(os.path.join(source, 'elo', d))) if os.path.isdir(os.path.join(source, 'elo')) else ['']
    for sub in pools:
        for f in ('results.json', 'ratings.json'):
            p = os.path.join(source, 'elo', sub, f)
            if os.path.exists(p):
                os.makedirs(os.path.join(dest, 'elo', sub), exist_ok=True)
                shutil.copy2(p, os.path.join(dest, 'elo', sub, f))
    rows = [m for m in read_metrics(source) if m['iter'] <= it]
    with open(os.path.join(dest, 'metrics.jsonl'), 'w', encoding='utf-8') as f:
        for m in rows:
            f.write(json.dumps(m) + '\n')
    return need, dirs, rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and not argv[0].startswith('-'):
        ap = argparse.ArgumentParser()
        ap.add_argument('src')
        ap.add_argument('run')
        ap.add_argument('--game', default=None)
        a = ap.parse_args(argv)
        new_run(a.src, a.run, a.game)
        return 0
    ap = argparse.ArgumentParser(description='continue a run at one of its checkpoints (module docstring)')
    ap.add_argument('--source', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--iter', type=int, default=None)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--copy', action='store_true', help='copy shards instead of hard-linking them')
    a = ap.parse_args(argv)
    over = {}
    for kv in a.set:
        k, _, v = kv.partition('=')
        over[k.strip().replace('-', '_')] = parse_value(v)
    continuation(a.source, a.dest, a.iter, over, a.copy)
    return 0


def cli():
    """The command line (`tools/fork_run.py`): CPU only."""
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.exit(main())


if __name__ == '__main__':
    cli()
