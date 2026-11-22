"""Distil a network into a new one of another size (zero.distill; docs/az.md 13).

  python tools/distill.py TEACHER.pt OUT.pt --data runs/a/data [--heldout runs/a/data/iter_0021]
      --hidden 256 --blocks 6 --heads 8 --ff 512 [--arch tokens] [--steps 20000 | --epochs 30]
      [--batch 512 --lr 5e-4 --lr-min 1e-5 --warmup 500 --wd 1e-4]
      [--teacher-w 1.0 --value-q 0.5 --margin-w 0.5] [--aux-w 1.0 --aux-teacher-w W]
      [--ram-budget-gb 20 --heldout-rows 16384]
      [--eval-every 1000 --save-every 2000 --keep-best --resume] [--device cuda --torch-threads 4 --seed 1]
      A new network of the given size for the teacher's game (this build's encoding), trained on the
      shards to compute what the teacher computes; OUT.pt is a normal checkpoint (the encoding guard
      checks it before it is written) with the record of its distillation in its `extra`
      ('distilled_from', 'distill': settings, data, steps, the held-out agreement). A run starts from it
      with `tools/fork_run.py --source <run> --dest <new run> --net OUT.pt` (the run's window and
      iteration numbers, the student as checkpoint K), or `tools/fork_run.py OUT.pt <new run>` (a new
      run whose iteration 0 it is).
      A teacher with an auxiliary head (docs/az.md 15: MTG Zero's belief) gets a student with the same
      head, distilled too (zero.distill: --aux-w its weight in the loss, --aux-teacher-w the blend of
      the teacher's predictions with the shards' targets, default --teacher-w); the teacher's head must
      be the game's targets (the student's then is: a run that trains them or reads the root's belief
      - `*_belief_tilt` - loads it). Shards with value-only rows (docs/az.md 16): the policy terms over
      the decision rows only, as the run's training; the held-out agreement per kind of row.
  python tools/distill.py --compare A.pt B.pt --data DIR ... [--rows 16384]
      The agreement of A with B as the teacher on the shards' positions (no training): policy KL(B || A),
      top-1 agreement, P(win) differences, each one's losses against the shards' targets (and, when both
      have the same auxiliary head, its agreement: 'aux').

--data / --heldout: a run's data directory (every iter_* in it), one data/iter_NNNN directory or a
shard file; repeatable. Without --heldout the last shard of --data (in path order) is held out
(--heldout none: none). Every shard must be of this build's encoding (its meta's feature count):
the student reads this build's inputs. The teacher's answers are computed once per window row
before the training. A long distillation writes OUT.pt.partial (the student, the optimizer, the
step) every --save-every steps; --resume continues from it (other batches than without the stop).
--init S.pt: the student starts from that checkpoint (its size; e.g. more steps after a gate missed)
instead of fresh (one without the teacher's auxiliary head gets a fresh one, `net.add_aux`).
--keep-best: also OUT.best.pt, the student at its lowest held-out policy KL so far (written at each
held-out evaluation that improves it; its extra's `distill.step` says when).
Progress: one JSON line per report on stdout and in OUT.pt.log.jsonl. Below-normal priority.

  python tools/distill.py --sources SOURCES.json OUT.pt [--device cuda --torch-threads 4 --keep-best --resume]
      Several teachers, each on its own rows (docs/az.md 77): SOURCES.json =
        {"student": {"init": "S.pt"},                 (or a fresh one: {"hidden": .., "blocks": .., "heads": ..,
                                                        "ff": .., "arch": .., "ch": ..}, each default the first
                                                        teacher's, as the size flags)
         "sources": [{"name": "a", "teacher": "T.pt", "data": [<as --data>, ...],
                      "select": {<the game package's row selector's keys>}, "weight": 1.0,
                      "heldout": "auto" | "none" | [<as --heldout>, ...], "config": "<run>/config.json"}, ...],
         "heldout": "auto" | "none",
         "train": {"steps": .., "epochs": .., "batch": .., "lr": .., "lr_min": .., "warmup": .., "wd": ..,
                   "teacher_w": .., "value_q": .., "margin_w": .., "aux_w": .., "aux_teacher_w": ..,
                   "ram_budget_gb": .., "heldout_rows": .., "eval_every": .., "save_every": ..,
                   "report_every": .., "seed": ..}}            (each default the flag's)
      One window of every source's selected rows (`select` {} = every row; else the game package's row
      selector, `GamePackage.row_selectors`, decides each shard's rows from its meta and its run's config -
      `<run>/config.json` beside `<run>/data`, or the source's "config"; a game without one refuses a
      non-empty select), each teacher's answers on its own source's rows only, every batch split between
      the sources by weight (stratified: a source's share does not depend on its row count). Held out:
      "auto" = each source's last shard with selected rows; the held-out agreement per source (and the
      weight-averaged policy_kl / top1 / value_mse / value_mae / margin_mse) in the progress lines and
      in OUT.pt's extra (`distill`: the sources file and its sha256, each source's teacher sha256, rows
      and agreement, the steps). The training settings come from the file only (their flags refused).
      Loud: an unknown key, a source without selected rows, a selector the game cannot decide.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def shard_list(args):
    """Shard files of --data / --heldout arguments (a run's data dir, an iter_NNNN dir, a file)."""
    from ..data import shard_paths
    out = []
    for a in args:
        if os.path.isfile(a):
            out.append(a)
        elif glob.glob(os.path.join(a, 'iter_*')):
            for d in sorted(glob.glob(os.path.join(a, 'iter_*'))):
                out.extend(shard_paths(d))
        elif os.path.isdir(a):
            out.extend(shard_paths(a))
        else:
            raise SystemExit(f'{a}: no such shard or data directory')
    return out


def _meta_game(meta, game):
    """A shard meta's game resolved (docs/az.md 73; none recorded: `game`); a bad layout record: its message."""
    from .. import package
    try:
        return package.record_game(meta, default=game)
    except package.GameAliasError as e:
        return str(e)


def check_shards(paths, features, actions, game=None):
    """Every shard of this build's encoding for `game` (its meta's game, feature and action counts),
    else SystemExit."""
    import numpy as np
    bad = []
    for p in paths:
        with np.load(p) as z:
            meta = json.loads(str(z['meta']))
        if int(meta.get('features', features)) != features or int(meta.get('actions', actions)) != actions                 or (game and _meta_game(meta, game) != game):
            bad.append(f'{p}: {meta.get("game")!r}, {meta.get("features")} features, {meta.get("actions")} actions')
    if bad:
        raise SystemExit(f'shards of another encoding than this build\'s ({game!r}, {features} features, {actions} actions): '
                         + '; '.join(bad[:3]) + (f' (+{len(bad) - 3} more)' if len(bad) > 3 else ''))


def window(paths, model, rows=None, seed=0, aux_width=0, select=None):
    """The shards' window for `model`'s inputs; `aux_width` > 0: with the shards' auxiliary targets; a token network
    with tail segments / pointer overrides: their tails and overrides too (its layout, docs/az.md 43; a shard of other
    tail segments or without them is refused, R9). `select`: per shard its rows to load (`data.Window`'s, docs/az.md
    71)."""
    from ..data import Window
    kw = {'aux_width': aux_width} if aux_width else {}
    if select is not None:
        kw['select'] = select
    lay = getattr(model, 'config', {}).get('layout')
    if lay is not None and (any(s.get('tail') for s in lay.get('segments', [])) or lay.get('option_pointers')):
        kw['layout'] = lay
    return Window(paths, model.features, model.config['actions'], budget_rows=rows, seed=seed, **kw)


def budget_rows(gb, features, aux_width=0):
    """Window rows in `gb` GB: a row's features, targets and the teacher's answers; with an auxiliary
    head of width W also the row's targets (float16 + its mask, as the run's window) and the teacher's
    predictions (float32): 6 W + 1 bytes more."""
    from ..protocol import MAX_SEATS
    bpr = features * 2 + 8 * 16 + 3 * MAX_SEATS * 4 + 2 * MAX_SEATS * 4 + 12   # + the teacher's answers
    if aux_width:
        bpr += 6 * aux_width + 1
    return int(gb * 1e9 / bpr)


def check_head(model, game, what):
    """A network with an auxiliary head: it must predict the game's targets (names and widths: a run
    trains exactly those, and the root's belief reads them, docs/az.md 15, 18), else SystemExit."""
    from ..net import EncodingMismatch, check_belief_head
    if not getattr(model, 'aux_layout', None):
        return
    try:
        check_belief_head(model, game, what)
    except EncodingMismatch as e:
        raise SystemExit(str(e)) from None


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, rec):
        rec = dict(rec, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        line = json.dumps(rec)
        print(line, flush=True)
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


def compare(a):
    from .. import distill as D
    from ..net import load
    import torch
    S, _ = load(a.compare[0], a.device, game=a.game)
    T, _ = load(a.compare[1], a.device, game=a.game or S.config.get('game'))
    paths = shard_list(a.data)
    check_shards(paths, S.features, S.config['actions'], S.config.get('game'))
    same = bool(S.aux_layout) and S.aux_layout == T.aux_layout    # (the heads' agreement: the same targets)
    W = window(paths, S, a.rows, a.seed, D.aux_width(S) if same else 0)
    strip = (lambda O: O) if same else (lambda O: {k: v for k, v in O.items() if k not in ('aux', 'aux_layout')})
    with torch.no_grad():
        out = D.agreement(strip(D.outputs(S, W, a.device)), strip(D.outputs(T, W, a.device)), W)
    print(json.dumps({'compare': a.compare[0], 'teacher': a.compare[1], 'shards': len(paths), **out}), flush=True)
    return 0


# ---------------------------------------------------------------- several teachers (docs/az.md 77)

# The flags whose settings a sources file's "train" block gives (refused beside --sources), and its keys.
SOURCE_TRAIN = ('steps', 'epochs', 'batch', 'lr', 'lr_min', 'warmup', 'wd', 'teacher_w', 'value_q', 'margin_w', 'aux_w',
                'aux_teacher_w', 'ram_budget_gb', 'heldout_rows', 'eval_every', 'save_every', 'report_every', 'seed')
SOURCE_REFUSED = SOURCE_TRAIN + ('data', 'heldout', 'init', 'arch', 'hidden', 'blocks', 'heads', 'ff', 'ch', 'rows')
STUDENT_SIZE = ('arch', 'hidden', 'blocks', 'heads', 'ff', 'ch')
SOURCE_KEYS = ('name', 'teacher', 'data', 'select', 'weight', 'heldout', 'config')


def _number(v, what, integer=False, lo=None, none=False):
    if v is None and none:
        return None
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (not integer or float(v) == int(v))
    if not ok or (lo is not None and v < lo):
        raise SystemExit(f'{what}: {v!r} is not a{"n integer" if integer else " number"}'
                         + (f' >= {lo}' if lo is not None else ''))
    return int(v) if integer else float(v)


def read_sources(path, defaults):
    """A sources file (module docstring) checked -> {'sha256', 'path', 'student', 'sources', 'heldout', 'train'}
    (`train`: every key, `defaults` filling the missing ones); anything unknown or ill-formed: SystemExit."""
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        spec = json.loads(raw.decode('utf-8'))
    except ValueError as e:
        raise SystemExit(f'{path}: not a JSON sources file ({e})') from None
    if not isinstance(spec, dict):
        raise SystemExit(f'{path}: a sources file is a JSON object')
    bad = sorted(set(spec) - {'student', 'sources', 'heldout', 'train'})
    if bad:
        raise SystemExit(f'{path}: unknown keys {bad} (student, sources, heldout, train)')
    st = spec.get('student')
    if not isinstance(st, dict) or not st:
        raise SystemExit(f'{path}: "student" must be {{"init": <checkpoint>}} or a fresh student\'s size')
    bad = sorted(set(st) - {'init'} - set(STUDENT_SIZE))
    if bad or ('init' in st and len(st) > 1):
        raise SystemExit(f'{path}: student {st}: "init" alone, or the size keys {list(STUDENT_SIZE)}')
    if 'init' in st:
        if not isinstance(st['init'], str) or not os.path.isfile(st['init']):
            raise SystemExit(f'{path}: student init {st["init"]!r}: no such checkpoint')
    else:
        for k in ('hidden', 'blocks', 'heads', 'ff', 'ch'):
            if k in st:
                _number(st[k], f'{path}: student {k}', integer=True, lo=0)
    held = spec.get('heldout', 'auto')
    if held not in ('auto', 'none'):
        raise SystemExit(f'{path}: heldout {held!r}: "auto" or "none" (a source\'s own list goes in the source)')
    srcs = spec.get('sources')
    if not isinstance(srcs, list) or not srcs:
        raise SystemExit(f'{path}: "sources" must be a non-empty list')
    out, names = [], set()
    for k, s in enumerate(srcs):
        if not isinstance(s, dict):
            raise SystemExit(f'{path}: source {k} is not an object')
        bad = sorted(set(s) - set(SOURCE_KEYS))
        if bad:
            raise SystemExit(f'{path}: source {k}: unknown keys {bad} ({list(SOURCE_KEYS)})')
        name = s.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise SystemExit(f'{path}: source {k}: a "name" of its own (a non-empty string) is required ({name!r})')
        names.add(name)
        if not isinstance(s.get('teacher'), str) or not os.path.isfile(s['teacher']):
            raise SystemExit(f'{path}: source {name!r}: teacher {s.get("teacher")!r}: no such checkpoint')
        data = s.get('data')
        data = [data] if isinstance(data, str) else data
        if not isinstance(data, list) or not data or not all(isinstance(d, str) and d for d in data):
            raise SystemExit(f'{path}: source {name!r}: "data" must be a non-empty list of shard / data paths')
        sel = s.get('select', {})
        if not isinstance(sel, dict):
            raise SystemExit(f'{path}: source {name!r}: "select" must be an object ({sel!r})')
        sh = s.get('heldout', None)
        if sh is not None and sh not in ('auto', 'none') and not (
                isinstance(sh, list) and sh and all(isinstance(d, str) and d for d in sh)):
            raise SystemExit(f'{path}: source {name!r}: heldout {sh!r}: "auto", "none" or a list of shard / data paths')
        cfg = s.get('config')
        if cfg is not None and (not isinstance(cfg, str) or not os.path.isfile(cfg)):
            raise SystemExit(f'{path}: source {name!r}: config {cfg!r}: no such file')
        out.append({'name': name, 'teacher': s['teacher'], 'data': data, 'select': sel,
                    'weight': _number(s.get('weight', 1.0), f'{path}: source {name!r}: weight'),
                    'heldout': held if sh is None else sh, 'config': cfg})
        if out[-1]['weight'] <= 0:
            raise SystemExit(f'{path}: source {name!r}: weight {out[-1]["weight"]} must be > 0')
    tr = spec.get('train', {})
    if not isinstance(tr, dict):
        raise SystemExit(f'{path}: "train" must be an object')
    bad = sorted(set(tr) - set(SOURCE_TRAIN))
    if bad:
        raise SystemExit(f'{path}: train: unknown keys {bad} ({list(SOURCE_TRAIN)})')
    train = dict(defaults)
    ints = ('steps', 'batch', 'warmup', 'heldout_rows', 'eval_every', 'save_every', 'report_every', 'seed')
    for key, v in tr.items():
        nullable = key in ('steps', 'epochs', 'aux_teacher_w')
        train[key] = _number(v, f'{path}: train {key}', integer=key in ints, lo=0, none=nullable)
    if train['batch'] < 1 or (train['steps'] is not None and train['steps'] < 1) or train['report_every'] < 1:
        raise SystemExit(f'{path}: train: batch, steps and report_every must be >= 1')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'path': os.path.abspath(path), 'student': st, 'sources': out,
            'heldout': held, 'train': train}


def _flags_given(argv, dests):
    """The `dests` (argparse names) whose flag appears in `argv` (`--x-y` or `--x-y=...`)."""
    flags = {'--' + d.replace('_', '-'): d for d in dests}
    out = []
    for t in argv:
        f = t.split('=', 1)[0]
        if f in flags and flags[f] not in out:
            out.append(flags[f])
    return out


def _source_plan(src, game, F, A):
    """One source's shards and selected rows -> its plan: 'train' / 'held' shard paths with their selected rows
    ('train_sel' / 'held_sel'), 'rows' (the training shards' rows), 'selected' (theirs selected), 'held_rows'."""
    from .. import distill as D
    from ..data import shard_rows
    name = src['name']
    paths = shard_list(src['data'])
    rule = src['heldout']
    explicit = shard_list(rule) if isinstance(rule, list) else []
    check_shards(paths + explicit, F, A, game)

    def pick(ps):
        try:
            return D.select_rows(ps, src['select'], game, name, src['config'])
        except ValueError as e:
            raise SystemExit(f'source {name!r}: {e}') from None
    if isinstance(rule, list):
        hk = {os.path.abspath(h) for h in explicit}
        train = [p for p in paths if os.path.abspath(p) not in hk]
        tsel, hsel, held = pick(train), pick(explicit), explicit
        if not sum(len(s) for s in hsel):
            raise SystemExit(f'source {name!r}: its selector takes no row of its {len(held)} held-out shards')
    else:
        allsel = pick(paths)
        have = [k for k, s in enumerate(allsel) if len(s)]
        hk = set(have[-1:]) if rule == 'auto' else set()
        train = [paths[k] for k in have if k not in hk]
        tsel = [allsel[k] for k in have if k not in hk]
        held = [paths[k] for k in sorted(hk)]
        hsel = [allsel[k] for k in sorted(hk)]
    rows = sum(shard_rows(p) for p in train)
    keep = [k for k, s in enumerate(tsel) if len(s)]
    train, tsel = [train[k] for k in keep], [tsel[k] for k in keep]
    selected = sum(len(s) for s in tsel)
    if not selected:
        raise SystemExit(f'source {name!r}: no training rows ({len(paths)} shards; select {json.dumps(src["select"])}'
                         f'{"; the held-out shard is its only one with selected rows" if rule == "auto" else ""})')
    hkeep = [k for k, s in enumerate(hsel) if len(s)]
    held, hsel = [held[k] for k in hkeep], [hsel[k] for k in hkeep]
    return dict(src, train=train, train_sel=tsel, held=held, held_sel=hsel, rows=rows, selected=selected,
                held_rows=sum(len(s) for s in hsel))


def run_sources(a, argv, ap):
    """`--sources FILE OUT.pt` (module docstring; docs/az.md 77)."""
    import types
    import torch
    from .. import distill as D
    from ..net import add_aux, aux_layout, check_encoding, load, save
    given = _flags_given(argv, SOURCE_REFUSED)
    if given:
        raise SystemExit(f'--sources: the training settings come from the sources file ("train", "student", each '
                         f'source\'s "data" / "heldout"); refused flags: {", ".join("--" + g.replace("_", "-") for g in given)}')
    if a.teacher and a.out:
        ap.error('--sources FILE OUT.pt: one positional argument (OUT.pt)')
    out = a.out or a.teacher
    if not out:
        ap.error('--sources FILE OUT.pt')
    S = read_sources(a.sources, {k: ap.get_default(k) for k in SOURCE_TRAIN})
    tr = S['train']
    log = Log(a.log or out + '.log.jsonl')
    t0 = time.time()
    # the teachers: the encoding guard, the head, the game (each loaded again for its answers below)
    info = []
    for src in S['sources']:
        t, textra = load(src['teacher'], 'cpu', game=a.game)
        g = a.game or t.config.get('game', 'toy')
        check_head(t, g, f'the teacher of source {src["name"]!r}')
        info.append({'config': t.config, 'aux': aux_layout(t.config.get('aux')), 'game': g, 'net': D.describe(t),
                     'sha256': sha256(src['teacher']),
                     'iter': textra.get('iter') if isinstance(textra, dict) else None})
        del t
    if len({i['game'] for i in info}) > 1:
        raise SystemExit(f'--sources: teachers of several games {sorted({i["game"] for i in info})}')
    if len({json.dumps(i['aux']) for i in info}) > 1:
        raise SystemExit('--sources: the teachers\' auxiliary heads differ (' + '; '.join(
            f'{s["name"]}: {i["aux"]}' for s, i in zip(S['sources'], info)) + '): the student learns one head')
    game, taux = info[0]['game'], info[0]['aux']
    st = S['student']
    init_sha = None
    if 'init' in st:
        student, _ = load(st['init'], 'cpu', game=game)
        init_sha = sha256(st['init'])
        have = aux_layout(student.config.get('aux'))
        if taux and have != taux:
            if have:
                raise SystemExit(f"student init {st['init']}: its auxiliary head {have} is not the teachers' {taux}")
            student = add_aux(student, taux, seed=tr['seed'])         # (a fresh head, the rest as it was)
        student.train()
        size = None
    else:
        tc = info[0]['config']
        size = dict(arch=st.get('arch') or tc.get('arch', 'mlp'),
                    hidden=st.get('hidden') or tc.get('d', tc.get('hidden', 128)),
                    blocks=st.get('blocks') or tc.get('layers', tc.get('blocks', 4)),
                    heads=st.get('heads') or tc.get('heads', 4), ff=st.get('ff', 0), ch=st.get('ch', 64))
        student = D.new_student(types.SimpleNamespace(config=tc), size['arch'], size['hidden'], size['blocks'],
                                size['heads'], size['ff'], size['ch'], tr['seed'], game)
    student.to(a.device)
    check_encoding(student.config, game, 'the student')
    check_head(student, game, 'the student')
    aw = D.aux_width(student)
    F, A = student.features, student.config['actions']
    plan = [_source_plan(src, game, F, A) for src in S['sources']]
    wpaths, wsel = [], []
    for p in plan:
        p['shards'] = (len(wpaths), len(wpaths) + len(p['train']))
        wpaths += p['train']
        wsel += p['train_sel']
    W = window(wpaths, student, budget_rows(tr['ram_budget_gb'], F, aw), tr['seed'], aw, select=wsel)
    for p in plan:
        p['span'] = (int(W.starts[p['shards'][0]]), int(W.starts[p['shards'][1]]))
        p['H'] = window(p['held'], student, tr['heldout_rows'], tr['seed'] + 1, aw, select=p['held_sel']) \
            if p['held'] else None
    load_s = time.time() - t0
    T, t1 = None, time.time()
    for p, i in zip(plan, info):
        teacher, _ = load(p['teacher'], a.device, game=game)
        T = D.outputs(teacher, W, a.device, span=p['span'], into=T)
        p['TH'] = D.outputs(teacher, p['H'], a.device) if p['H'] is not None else None
        del teacher
        if str(a.device).startswith('cuda'):
            torch.cuda.empty_cache()
    t_s = time.time() - t1
    n = D.shares([p['weight'] for p in plan], tr['batch'])
    strata = [(p['span'][0], p['span'][1], k) for p, k in zip(plan, n)]
    steps = tr['steps'] or max(1, int(round((tr['epochs'] or 1.0) * W.N / tr['batch'])))
    settings = dict(tr, steps=steps, game=game, size=size, device=a.device, prefetch=a.prefetch,
                    keep_best=a.keep_best, torch_threads=a.torch_threads)
    part = out + '.partial'
    opt, start = None, 0
    if a.resume and os.path.exists(part):
        ck = torch.load(part, map_location=a.device, weights_only=False)
        if ck['config'] != student.config:
            raise SystemExit(f'{part}: a student of another size or encoding than asked; delete it or drop --resume')
        if ck.get('sources_sha256') != S['sha256']:
            raise SystemExit(f'{part}: written for another sources file (sha256 {ck.get("sources_sha256")}); delete it '
                             f'or drop --resume')
        student.load_state_dict(ck['state'])
        opt, start = ck['opt'], int(ck['step'])

    def desc(p, i, k):
        return {'name': p['name'], 'teacher': os.path.abspath(p['teacher']), 'teacher_sha256': i['sha256'],
                'teacher_net': i['net'], 'teacher_iter': i['iter'], 'select': p['select'], 'weight': p['weight'],
                'batch_rows': k, 'config': os.path.abspath(p['config']) if p['config'] else None,
                'train_shards': len(p['train']), 'rows': p['rows'], 'selected': p['selected'],
                'used': p['span'][1] - p['span'][0], 'heldout_shards': len(p['held']),
                'heldout_rows': p['H'].N if p['H'] is not None else 0}
    more = {}
    nv = D.view_rows(W)
    hv = sum(D.view_rows(p['H']) for p in plan if p['H'] is not None)
    if nv or hv:
        more.update(train_view_rows=nv, heldout_view_rows=hv)
    if aw:
        more.update(aux=taux, train_aux_rows=W.aux_rows,
                    heldout_aux_rows=sum(p['H'].aux_rows for p in plan if p['H'] is not None),
                    aux_w=tr['aux_w'], aux_teacher_w=tr['teacher_w'] if tr['aux_teacher_w'] is None else tr['aux_teacher_w'])
    log({'event': 'start', 'sources_file': S['path'], 'sources_sha256': S['sha256'],
         'student_init': os.path.abspath(st['init']) if 'init' in st else None, 'student_net': D.describe(student),
         'train_rows': W.N, 'loaded_rows': W.loaded, 'steps': steps, 'start': start,
         'epochs': round(steps * tr['batch'] / W.N, 2), 'load_s': round(load_s, 1), 'teacher_s': round(t_s, 1),
         'device': a.device, 'sources': [desc(p, i, k) for p, i, k in zip(plan, info, n)], **more})
    hist = []
    best = {'kl': float('inf')}
    best_path = os.path.splitext(out)[0] + '.best.pt'
    AVG = ('policy_kl', 'top1', 'value_mse', 'value_mae', 'margin_mse')

    def record(step):
        srcs = []
        for p, i, k in zip(plan, info, n):
            d = desc(p, i, k)
            d.update(train_shards=[os.path.abspath(x) for x in p['train']],
                     heldout_shards=[os.path.abspath(x) for x in p['held']])
            srcs.append(d)
        return {'distilled_from': S['path'],
                'distill': {'sources_file': S['path'], 'sources_sha256': S['sha256'], 'sources': srcs,
                            'student_init': os.path.abspath(st['init']) if 'init' in st else None,
                            'student_init_sha256': init_sha, 'settings': settings, 'step': step,
                            'student_net': D.describe(student), 'train_rows': W.N, 'heldout': list(hist),
                            'seconds': round(time.time() - t0, 1)}}

    def held_out(step):
        per, wts = {}, {}
        for p in plan:
            if p['H'] is None:
                continue
            per[p['name']] = D.agreement(D.outputs(student, p['H'], a.device), p['TH'], p['H'])
            wts[p['name']] = p['weight']
        student.train()
        if not per:
            return None
        tot = sum(wts.values())
        r = {'step': step, **{k: round(sum(wts[s] * per[s][k] for s in per) / tot, 7) for k in AVG}, 'sources': per}
        hist.append(r)
        log({'event': 'heldout', **r})
        if a.keep_best and step > 0 and r['policy_kl'] < best['kl']:
            best['kl'] = r['policy_kl']
            save(student, best_path, record(step))
            log({'event': 'best', 'step': step, 'policy_kl': r['policy_kl'], 'out': os.path.abspath(best_path)})
        return r
    if start == 0:
        held_out(0)
    t2 = time.time()

    def on_report(step, terms, lr):
        el = time.time() - t2
        log({'event': 'train', 'step': step, 'lr': round(lr, 8), **terms, 'seconds': round(el, 1),
             's_per_step': round(el / max(1, step - start), 4)})

    def on_step(step, model, o):
        if tr['eval_every'] and step % tr['eval_every'] == 0 and step < steps:
            held_out(step)
        if tr['save_every'] and step % tr['save_every'] == 0 and step < steps:
            tmp = part + '.tmp'
            torch.save({'config': model.config, 'state': model.state_dict(), 'opt': o.state_dict(), 'step': step,
                        'settings': settings, 'sources_sha256': S['sha256']}, tmp)
            os.replace(tmp, part)
    D.distill(student, W, T, steps, tr['batch'], tr['lr'], tr['lr_min'], tr['warmup'], tr['wd'], tr['teacher_w'],
              tr['value_q'], tr['margin_w'], a.device, tr['seed'], a.prefetch, opt, start, tr['report_every'],
              on_report, on_step, aux_w=tr['aux_w'], aux_teacher_w=tr['aux_teacher_w'], strata=strata)
    final = held_out(steps)
    student.to('cpu').eval()
    check_encoding(student.config, game, 'the student')
    check_head(student, game, 'the student')
    save(student, out, record(steps))
    if os.path.exists(part):
        os.remove(part)
    log({'event': 'done', 'out': os.path.abspath(out), 'student_net': D.describe(student), 'steps': steps,
         'heldout': final, 'seconds': round(time.time() - t0, 1)})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='distil a network into a new one (module docstring)')
    ap.add_argument('teacher', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'), default=None)
    ap.add_argument('--data', action='append', default=[])
    ap.add_argument('--sources', default=None, help='a multi-teacher distillation from a sources file (docstring)')
    ap.add_argument('--heldout', action='append', default=[])
    ap.add_argument('--game', default=None)
    ap.add_argument('--arch', default=None, help="the student's arch (default: the teacher's)")
    ap.add_argument('--hidden', type=int, default=None, help="width (tokens: d; default: the teacher's)")
    ap.add_argument('--blocks', type=int, default=None, help="blocks / layers (default: the teacher's)")
    ap.add_argument('--heads', type=int, default=None)
    ap.add_argument('--ff', type=int, default=0, help='tokens: feed-forward width (0 = 2 x hidden)')
    ap.add_argument('--ch', type=int, default=64)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--epochs', type=float, default=None, help='steps = epochs x window rows / batch')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--teacher-w', type=float, default=1.0, help="weight of the teacher's answers (1 - it: the shards' targets)")
    ap.add_argument('--value-q', type=float, default=0.5, help="the shards' value target (1 - q) z + q q_root")
    ap.add_argument('--margin-w', type=float, default=0.5)
    ap.add_argument('--aux-w', type=float, default=1.0,
                    help="an auxiliary head's term in the loss (docs/az.md 13.6; the teacher's head only)")
    ap.add_argument('--aux-teacher-w', type=float, default=None,
                    help="an auxiliary head's blend: the teacher's predictions (1 - it: the shards' targets; "
                         "default --teacher-w)")
    ap.add_argument('--ram-budget-gb', type=float, default=8.0)
    ap.add_argument('--heldout-rows', type=int, default=16384)
    ap.add_argument('--rows', type=int, default=16384, help='--compare: rows of the shards compared')
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--report-every', type=int, default=100)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init', default=None, help="the student's start: a checkpoint (its size) instead of a fresh network")
    ap.add_argument('--keep-best', action='store_true', help='also OUT.best.pt: the student at its lowest held-out KL')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--torch-threads', type=int, default=4)
    ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--log', default=None, help='JSON lines log (default OUT.pt.log.jsonl)')
    ap.add_argument('--normal-priority', action='store_true')
    a = ap.parse_args(argv)
    if not a.data and not a.sources:
        ap.error('the following arguments are required: --data')
    import torch
    from ..run import below_normal, utf8_console
    utf8_console()
    torch.set_num_threads(max(1, a.torch_threads))
    if not a.normal_priority:
        below_normal()
    if a.compare:
        return compare(a)
    if a.sources:
        return run_sources(a, sys.argv[1:] if argv is None else list(argv), ap)
    if not a.teacher or not a.out:
        ap.error('TEACHER.pt OUT.pt (or --compare A B)')
    from .. import distill as D
    from ..net import add_aux, aux_layout, check_encoding, load, save
    log = Log(a.log or a.out + '.log.jsonl')
    t0 = time.time()
    teacher, textra = load(a.teacher, a.device, game=a.game)
    tsha = sha256(a.teacher)
    game = a.game or teacher.config.get('game', 'toy')
    taux = aux_layout(teacher.config.get('aux'))
    check_head(teacher, game, 'the teacher')       # (its head is the game's targets: the student's will be)
    tc = teacher.config
    size = dict(arch=a.arch or tc.get('arch', 'mlp'),
                hidden=a.hidden or tc.get('d', tc.get('hidden', 128)),
                blocks=a.blocks or tc.get('layers', tc.get('blocks', 4)),
                heads=a.heads or tc.get('heads', 4), ff=a.ff, ch=a.ch)
    if a.init:
        student, _ = load(a.init, 'cpu', game=game)
        have = aux_layout(student.config.get('aux'))
        if taux and have != taux:
            if have:
                raise SystemExit(f"--init {a.init}: its auxiliary head {have} is not the teacher's {taux}")
            student = add_aux(student, taux, seed=a.seed)          # (a fresh head, the rest as it was)
        student.train()
    else:
        student = D.new_student(teacher, size['arch'], size['hidden'], size['blocks'], size['heads'], size['ff'], size['ch'],
                                a.seed, game)
    student.to(a.device)
    check_encoding(student.config, game, 'the student')
    check_head(student, game, 'the student')
    aw = D.aux_width(student)
    paths = shard_list(a.data)
    if a.heldout == ['none']:
        held = []
    elif a.heldout:
        held = shard_list(a.heldout)
    else:
        held = paths[-1:]
    train = [p for p in paths if os.path.abspath(p) not in {os.path.abspath(h) for h in held}]
    if not train:
        raise SystemExit('no training shards left after the held-out ones')
    check_shards(train + held, student.features, student.config['actions'], game)
    W = window(train, student, budget_rows(a.ram_budget_gb, student.features, aw), a.seed, aw)
    H = window(held, student, a.heldout_rows, a.seed + 1, aw) if held else None
    load_s = time.time() - t0
    T, t_s = D.timed(D.outputs, teacher, W, a.device)
    TH = D.outputs(teacher, H, a.device) if H is not None else None
    steps = a.steps or max(1, int(round((a.epochs or 1.0) * W.N / a.batch)))
    settings = dict(vars(a), game=game, size=size, steps=steps)
    settings.pop('compare', None)
    settings.pop('sources', None)                  # (docs/az.md 77's flag: not one of this path's settings)
    part = a.out + '.partial'
    opt, start = None, 0
    if a.resume and os.path.exists(part):
        ck = torch.load(part, map_location=a.device, weights_only=False)
        if ck['config'] != student.config:
            raise SystemExit(f'{part}: a student of another size or encoding than asked; delete it or drop --resume')
        student.load_state_dict(ck['state'])
        opt, start = ck['opt'], int(ck['step'])
    more = {}
    nv, hv = D.view_rows(W), (D.view_rows(H) if H is not None else 0)
    if nv or hv:                                   # value-only rows (docs/az.md 16)
        more.update(train_view_rows=nv, heldout_view_rows=hv)
    if aw:                                         # the auxiliary head (rows whose shard has its targets)
        more.update(aux=taux, train_aux_rows=W.aux_rows, heldout_aux_rows=H.aux_rows if H is not None else 0,
                    aux_w=a.aux_w, aux_teacher_w=a.teacher_w if a.aux_teacher_w is None else a.aux_teacher_w)
    log({'event': 'start', 'teacher': os.path.abspath(a.teacher), 'teacher_net': D.describe(teacher),
         'student_net': D.describe(student), 'train_shards': len(train), 'train_rows': W.N, 'loaded_rows': W.loaded,
         'heldout_shards': len(held), 'heldout_rows': H.N if H is not None else 0, 'steps': steps, 'start': start,
         'epochs': round(steps * a.batch / W.N, 2), 'load_s': round(load_s, 1), 'teacher_s': round(t_s, 1),
         'device': a.device, **more})
    hist = []
    best = {'kl': float('inf')}
    best_path = os.path.splitext(a.out)[0] + '.best.pt'

    def record(step):
        return {'distilled_from': os.path.abspath(a.teacher), 'teacher_sha256': tsha, 'teacher_extra': textra,
                'distill': {'settings': settings, 'step': step, 'teacher_net': D.describe(teacher),
                            'student_net': D.describe(student), 'train_shards': [os.path.abspath(p) for p in train],
                            'heldout_shards': [os.path.abspath(p) for p in held], 'train_rows': W.N,
                            'heldout': list(hist), 'seconds': round(time.time() - t0, 1)}}

    def held_out(step):
        if H is None:
            return None
        r = D.agreement(D.outputs(student, H, a.device), TH, H)
        student.train()
        hist.append({'step': step, **r})
        log({'event': 'heldout', 'step': step, **r})
        if a.keep_best and step > 0 and r['policy_kl'] < best['kl']:
            best['kl'] = r['policy_kl']
            save(student, best_path, record(step))
            log({'event': 'best', 'step': step, 'policy_kl': r['policy_kl'], 'out': os.path.abspath(best_path)})
        return r
    if start == 0:
        held_out(0)
    t1 = time.time()

    def on_report(step, terms, lr):
        el = time.time() - t1
        log({'event': 'train', 'step': step, 'lr': round(lr, 8), **terms, 'seconds': round(el, 1),
             's_per_step': round(el / max(1, step - start), 4)})

    def on_step(step, model, o):
        if a.eval_every and step % a.eval_every == 0 and step < steps:
            held_out(step)
        if a.save_every and step % a.save_every == 0 and step < steps:
            tmp = part + '.tmp'
            torch.save({'config': model.config, 'state': model.state_dict(), 'opt': o.state_dict(), 'step': step,
                        'settings': settings}, tmp)
            os.replace(tmp, part)
    D.distill(student, W, T, steps, a.batch, a.lr, a.lr_min, a.warmup, a.wd, a.teacher_w, a.value_q, a.margin_w,
              a.device, a.seed, a.prefetch, opt, start, a.report_every, on_report, on_step, aux_w=a.aux_w,
              aux_teacher_w=a.aux_teacher_w)
    final = held_out(steps)
    student.to('cpu').eval()
    check_encoding(student.config, game, 'the student')
    check_head(student, game, 'the student')
    save(student, a.out, record(steps))
    if os.path.exists(part):
        os.remove(part)
    log({'event': 'done', 'out': os.path.abspath(a.out), 'student_net': D.describe(student), 'steps': steps,
         'heldout': final, 'seconds': round(time.time() - t0, 1)})
    return 0


def cli():
    sys.exit(main())


if __name__ == '__main__':
    cli()
