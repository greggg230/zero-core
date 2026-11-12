"""Start a run from another run's checkpoint.

  python tools/fork_run.py <src.pt> <new run dir> [--game toy] [--ungated] [--allow-draft]
      A NEW run whose iteration 0 is the source network widened to the game's current
      feature / action counts (new inputs get zero weights: it computes what the source did); a
      token network follows the game's object layout if it changed (docs/az.md 10.3), a conv
      network the game's board grid. The source may be a network of an older encoding (the
      way to move a run across an encoding change: the encoding guard refuses to resume it,
      docs/az.md 10.4); the fork is checked against this build's encoding before it is written
      (--game: the fork's game, recorded in its config). Settings: the new run's own flags when
      it is started. Variable-length token sets (docs/az.md 43.8): a segment that gains a tail needs
      nothing, a new tail-only segment is gated (its gate exactly 0: the fork computes what the source
      did on every row the source could read; --ungated: not gated, then the game's layout must not
      gate it either). A `draft` layout (the game's) is refused unless --allow-draft (R16).

  python tools/fork_run.py --source runs/a --dest runs/b [--iter K] [--set reuse=8 ...] [--copy]
      A CONTINUATION of a (possibly live) run at its checkpoint K (default: the newest): the
      checkpoints <= K that its Elo matches need (and, with the self-play league in the fork's
      settings - e.g. --set selfplay_league=0.25 -, every one its league may play: docs/az.md 14),
      the optimizer state when it belongs to K, the
      window's sample shards (hard links on the same volume, --copy to copy), Elo results (every
      pool: elo/ and elo/<rule>/), metrics up to K and the source's config.json with the --set
      overrides. The source is only read. Then: python tools/supervise.py runs/b (the run resumes after K). A co-training run
      (setting `resolver`, docs/az.md 11) is copied with its resolver network: the same of
      <source>/resolver/ into <dest>/resolver/ (K: an iteration where both have a checkpoint).

  python tools/fork_run.py --source runs/a --dest runs/b [--iter K] --net student.pt [--set ...]
      The same continuation with ANOTHER NETWORK as its checkpoint K: e.g. a bigger one distilled from
      K (tools/distill.py, docs/az.md 13). The network must be of this build's encoding for the run's
      game (the encoding guard); the source's optimizer state is not copied (it belongs to the source's
      network: the fork's first training starts a fresh AdamW with its warm-up), K's Elo results are
      dropped from the fork's pools (they rated the source's network), and the fork's config.json
      records the network's size (hidden / blocks / heads / ff, or ch). The rest - the window's
      shards, the checkpoints before K for the Elo matches, the metrics - as above, so the fork
      trains on the run's window from iteration K + 1 (an A/B of the network against a plain
      continuation of the source). The network's auxiliary head must be what the fork's settings need
      (`check_heads`, docs/az.md 13.6): with a root belief on (selfplay_ / eval_belief_tilt > 0) the
      head for the game's targets, else net.HeadMissing - the refusal the run would give at its start
      (exit 78) -, given before anything is copied, as the encoding guard's.
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


def check_draft(game, allow=False, what='fork_run'):
    """R16 (docs/az.md 43.8): the game's token layout is a `draft` (a layout still being built, e.g. MTG Zero's
    v7 until its project ends) -> SystemExit unless `allow` (--allow-draft; a run's `allow_draft_layout`)."""
    from .. import package
    lay = package.active().tokens_of(game) or {}
    if lay.get('draft') and not allow:
        ident = ' '.join(str(lay[k]) for k in ('name', 'version') if lay.get(k) is not None)
        raise SystemExit(f'{what}: the game {game!r} has a draft token layout ({ident or "no name"}): refused unless '
                         f'--allow-draft (docs/az.md 43.8)')


def new_run(src, run, game=None, allow_draft=False, ungated=False):
    import torch  # noqa: F401
    from .. import games
    from ..net import check_encoding, layout_sans_identity, load, save, widen
    from .. import package
    # the source may be a network of an older encoding: the widen's purpose (the encoding guard's
    # opt-out; what is written is checked, docs/az.md 10.4)
    old, extra = load(src, allow_mismatch=True)
    target = game or old.config.get('game', 'toy')
    check_draft(target, allow_draft)
    info = games.info(target)
    grid = package.active().grids.get(target) if old.config.get('arch') == 'grid' else None
    # a token network follows the game's object layout as it is now (a layout change: widened
    # segment by segment, docs/az.md 10.3; the same layout: as before; new tail-only segments gated
    # unless --ungated, docs/az.md 43.8)
    layout = package.active().tokens_of(target) if old.config.get('arch') == 'tokens' else None
    new = widen(old, info['num_features'], info['num_actions'], grid=grid, layout=layout, ungated=ungated)
    if game and new.config.get('game') != game:
        new.config['game'] = game                  # (widen keeps the source's game: the fork plays --game)
    check_encoding(new.config, target, f'fork of {src}')
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    out = os.path.join(run, 'models', 'iter_0000.pt')
    save(new, out, {'iter': 0, 'forked_from': os.path.abspath(src), 'source_extra': extra})
    moved = layout is not None and layout_sans_identity(new.config['layout']) != layout_sans_identity(old.config['layout'])
    tails = ''
    if moved:
        segs = new.config['layout']['segments']
        had = {s['name'] for s in old.config['layout']['segments'] if s.get('tail')}
        got = [s['name'] for s in segs if s.get('tail') and s['name'] not in had and s['count']]
        gated = new.gates() if hasattr(new, 'gates') else {}
        fresh = [s['name'] for s in segs[len(old.config['layout']['segments']):]]
        if got:
            tails += f'; tails on {", ".join(got)}'
        if fresh:
            tails += '; new segments ' + ', '.join(f'{n} (gate {gated[n]:g})' if n in gated else f'{n} (ungated)'
                                                   for n in fresh)
    print(f'{src} ({old.config["features"]} features, {old.config["actions"]} actions) -> {out} '
          f'({new.config["features"]} features, {new.config["actions"]} actions'
          + (', across a token layout change' if moved else '') + tails + ')')


def _link(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def net_size(config):
    """The run settings that describe a network's size (`hidden`, `blocks`, ... as `run.Run.new_model`
    reads them) from its checkpoint config."""
    arch = config.get('arch', 'mlp')
    if arch == 'tokens':
        return {'arch': 'tokens', 'hidden': config['d'], 'blocks': config['layers'], 'heads': config['heads'],
                'ff': config['ff']}
    if arch == 'grid':
        return {'arch': 'grid', 'ch': config['ch'], 'blocks': config['blocks']}
    return {'arch': 'mlp', 'hidden': config['hidden'], 'blocks': config['blocks']}


def check_heads(m, settings, game, net):
    """--net: the network's auxiliary head as the fork's settings need it (docs/az.md 15, 18): with a
    root belief on (selfplay_ / eval_belief_tilt > 0) the head for the game's targets (`check_belief_head`:
    net.HeadMissing, the refusal the run itself would give at its start, exit 78); a run that trains
    auxiliary targets refuses a head for other targets at its first training, so that is refused here
    too. A network without a head in a run that only trains targets is fine (the run adds a fresh one)."""
    from .. import games as G
    from ..net import HeadMissing, aux_layout, check_belief_head
    from ..run import aux_settings
    on = [k for k in ('selfplay_belief_tilt', 'eval_belief_tilt') if float(settings.get(k, 0.0) or 0.0) > 0]
    if on:
        check_belief_head(m, game, f'{" / ".join(on)} of the fork (--net {net})')
    _, want = aux_settings(dict(settings, game=game), G.info(game))
    have = aux_layout(m.config.get('aux'))
    if want and have and have != aux_layout(want):
        raise HeadMissing(f'--net {net}: its auxiliary head predicts {have}, the fork trains {aux_layout(want)}')


def install_net(dest, it, net, game, settings=None):
    """--net: `net` (a checkpoint of this build's encoding for `game`, with the auxiliary head the fork's
    `settings` need: `check_heads`) as the fork's checkpoint `it`; no optimizer state; checkpoint `it`'s
    Elo results dropped from every pool. -> its size settings."""
    from .. import elo
    from ..net import check_encoding, load, save
    from ..run import model_path
    m, extra = load(net, game=game)
    check_encoding(m.config, game, net)
    check_heads(m, settings or {}, game, net)
    save(m, model_path(dest, it), dict(extra, iter=it, net_from=os.path.abspath(net)))
    opt = os.path.join(dest, 'models', 'opt_latest.pt')
    if os.path.exists(opt):
        os.remove(opt)
    name = elo.ckpt_name(it)
    for d, _, files in os.walk(os.path.join(dest, 'elo')):
        if 'results.json' not in files:
            continue
        p = os.path.join(d, 'results.json')
        with open(p) as f:
            res = json.load(f)
        res['pairs'] = {k: v for k, v in res.get('pairs', {}).items() if name not in k.split('|')}
        if 'logged' in res:
            res['logged'] = [x for x in res['logged'] if x != name]
        with open(p, 'w') as f:
            json.dump(res, f)                      # (ratings.json: refit at the fork's next evaluation)
    return net_size(m.config)


def continuation(source, dest, it=None, overrides=None, copy=False, net=None, allow_draft=False):
    from ..run import DEFAULTS, latest_iter, model_path
    if os.path.exists(os.path.join(dest, 'config.json')):
        raise SystemExit(f'{dest} already holds a run')
    with open(os.path.join(source, 'config.json')) as f:
        cfg = json.load(f)
    for k in overrides or {}:
        if k not in DEFAULTS:
            raise SystemExit(f'--set {k}: unknown setting')
    # (R16: a draft layout's run continues only where its settings allow it, or --allow-draft)
    check_draft(cfg.get('game', 'toy'), allow_draft or bool(dict(cfg, **(overrides or {})).get('allow_draft_layout')),
                'fork_run --source')
    co = bool(cfg.get('resolver'))
    rsrc = os.path.join(source, 'resolver')
    if it is None:
        it = latest_iter(source)
        if co and it is not None:
            r = latest_iter(rsrc)
            it = min(it, r) if r is not None else None
    if it is None or not os.path.exists(model_path(source, it)):
        raise SystemExit(f'{source}: no checkpoint {it}')
    settings = dict(cfg, **(overrides or {}))
    if net:                                        # (refused before anything is copied: the guard, the heads)
        from ..net import check_encoding, load
        m, _ = load(net, game=cfg.get('game', 'toy'))
        check_encoding(m.config, cfg.get('game', 'toy'), net)
        check_heads(m, settings, cfg.get('game', 'toy'), net)
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
    need, dirs, rows = _copy_run(source, dest, it, cfg, copy, settings=settings)
    size = install_net(dest, it, net, cfg.get('game', 'toy'), settings) if net else None
    if size:
        cfg.update(size)
    cfg.update(overrides or {})
    with open(os.path.join(dest, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    print(f'{source} @ iter {it} -> {dest}: {len(need)} checkpoints, {len(dirs)} data iterations, {len(rows)} metrics rows'
          + (' (and its resolver network)' if co else ''))
    if size:
        print(f'checkpoint {it} = {net} ({", ".join(f"{k} {v}" for k, v in size.items())}): no optimizer state, '
              f'its Elo results dropped')
    print(f'next: python tools/supervise.py {dest}   (resumes after iteration {it})')


def _copy_run(source, dest, it, cfg, copy, settings=None):
    """Checkpoint `it` and those its Elo matches need (and, with the self-play league in `settings` -
    the fork's: the source's config with the --set overrides -, every past checkpoint its league may
    play, zero/league.py), its optimizer state, the window's shards, Elo results, metrics up to `it`
    -> (checkpoints, data dirs, metrics rows)."""
    from .. import elo, league
    from ..run import model_path, read_metrics, window_iters
    e = max(1, int(cfg.get('eval_every', 1)))
    need = {it} | {it - off * e for off in elo.OFFSETS if it - off * e >= 0}
    if cfg.get('elo_offsets'):
        # raw Elo offsets (docs/az.md 5.4): the opponents of K and of the fork's next evaluation
        need |= {j for k in (it, it + e) for j in elo.opponents(k, e, cfg['elo_offsets']) if 0 <= j <= it}
    need |= league.fork_needs(settings if settings is not None else cfg, it)
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
        ap.add_argument('--ungated', action='store_true',
                        help='new tail-only segments without a gate (their tokens perturb the rows they appear in)')
        ap.add_argument('--allow-draft', action='store_true', help="fork onto the game's draft token layout (R16)")
        a = ap.parse_args(argv)
        new_run(a.src, a.run, a.game, a.allow_draft, a.ungated)
        return 0
    ap = argparse.ArgumentParser(description='continue a run at one of its checkpoints (module docstring)')
    ap.add_argument('--source', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--iter', type=int, default=None)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--copy', action='store_true', help='copy shards instead of hard-linking them')
    ap.add_argument('--net', default=None, help="another network as the fork's checkpoint K (module docstring)")
    ap.add_argument('--allow-draft', action='store_true', help="continue a run of the game's draft token layout (R16)")
    a = ap.parse_args(argv)
    over = {}
    for kv in a.set:
        k, _, v = kv.partition('=')
        over[k.strip().replace('-', '_')] = parse_value(v)
    continuation(a.source, a.dest, a.iter, over, a.copy, a.net, a.allow_draft)
    return 0


def cli():
    """The command line (`tools/fork_run.py`): CPU only."""
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.exit(main())


if __name__ == '__main__':
    cli()
