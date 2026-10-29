"""Distil a network into a new one of another size (zero.distill; docs/az.md 13).

  python tools/distill.py TEACHER.pt OUT.pt --data runs/a/data [--heldout runs/a/data/iter_0021]
      --hidden 256 --blocks 6 --heads 8 --ff 512 [--arch tokens] [--steps 20000 | --epochs 30]
      [--batch 512 --lr 5e-4 --lr-min 1e-5 --warmup 500 --wd 1e-4]
      [--teacher-w 1.0 --value-q 0.5 --margin-w 0.5] [--ram-budget-gb 20 --heldout-rows 16384]
      [--eval-every 1000 --save-every 2000 --resume] [--device cuda --torch-threads 4 --seed 1]
      A new network of the given size for the teacher's game (this build's encoding), trained on the
      shards to compute what the teacher computes; OUT.pt is a normal checkpoint (the encoding guard
      checks it before it is written) with the record of its distillation in its `extra`
      ('distilled_from', 'distill': settings, data, steps, the held-out agreement). A run starts from it
      with `tools/fork_run.py --source <run> --dest <new run> --net OUT.pt` (the run's window and
      iteration numbers, the student as checkpoint K), or `tools/fork_run.py OUT.pt <new run>` (a new
      run whose iteration 0 it is).
  python tools/distill.py --compare A.pt B.pt --data DIR ... [--rows 16384]
      The agreement of A with B as the teacher on the shards' positions (no training): policy KL(B || A),
      top-1 agreement, P(win) differences, each one's losses against the shards' targets.

--data / --heldout: a run's data directory (every iter_* in it), one data/iter_NNNN directory or a
shard file; repeatable. Without --heldout the last shard of --data (in path order) is held out
(--heldout none: none). Every shard must be of this build's encoding (its meta's feature count):
the student reads this build's inputs. The teacher's answers are computed once per window row
before the training. A long distillation writes OUT.pt.partial (the student, the optimizer, the
step) every --save-every steps; --resume continues from it (other batches than without the stop).
Progress: one JSON line per report on stdout and in OUT.pt.log.jsonl. Below-normal priority.
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


def check_shards(paths, features, actions, game=None):
    """Every shard of this build's encoding for `game` (its meta's game, feature and action counts),
    else SystemExit."""
    import numpy as np
    bad = []
    for p in paths:
        with np.load(p) as z:
            meta = json.loads(str(z['meta']))
        if int(meta.get('features', features)) != features or int(meta.get('actions', actions)) != actions                 or (game and meta.get('game', game) != game):
            bad.append(f'{p}: {meta.get("game")!r}, {meta.get("features")} features, {meta.get("actions")} actions')
    if bad:
        raise SystemExit(f'shards of another encoding than this build\'s ({game!r}, {features} features, {actions} actions): '
                         + '; '.join(bad[:3]) + (f' (+{len(bad) - 3} more)' if len(bad) > 3 else ''))


def window(paths, model, rows=None, seed=0):
    from ..data import Window
    return Window(paths, model.features, model.config['actions'], budget_rows=rows, seed=seed)


def budget_rows(gb, features):
    from ..protocol import MAX_SEATS
    bpr = features * 2 + 8 * 16 + 3 * MAX_SEATS * 4 + 2 * MAX_SEATS * 4 + 12   # + the teacher's answers
    return int(gb * 1e9 / bpr)


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
    W = window(paths, S, a.rows, a.seed)
    with torch.no_grad():
        out = D.agreement(D.outputs(S, W, a.device), D.outputs(T, W, a.device), W)
    print(json.dumps({'compare': a.compare[0], 'teacher': a.compare[1], 'shards': len(paths), **out}), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='distil a network into a new one (module docstring)')
    ap.add_argument('teacher', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'), default=None)
    ap.add_argument('--data', action='append', default=[], required=True)
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
    ap.add_argument('--ram-budget-gb', type=float, default=8.0)
    ap.add_argument('--heldout-rows', type=int, default=16384)
    ap.add_argument('--rows', type=int, default=16384, help='--compare: rows of the shards compared')
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--report-every', type=int, default=100)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--torch-threads', type=int, default=4)
    ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--log', default=None, help='JSON lines log (default OUT.pt.log.jsonl)')
    ap.add_argument('--normal-priority', action='store_true')
    a = ap.parse_args(argv)
    import torch
    from ..run import below_normal, utf8_console
    utf8_console()
    torch.set_num_threads(max(1, a.torch_threads))
    if not a.normal_priority:
        below_normal()
    if a.compare:
        return compare(a)
    if not a.teacher or not a.out:
        ap.error('TEACHER.pt OUT.pt (or --compare A B)')
    from .. import distill as D
    from ..net import check_encoding, load, save
    log = Log(a.log or a.out + '.log.jsonl')
    t0 = time.time()
    teacher, textra = load(a.teacher, a.device, game=a.game)
    game = a.game or teacher.config.get('game', 'toy')
    tc = teacher.config
    size = dict(arch=a.arch or tc.get('arch', 'mlp'),
                hidden=a.hidden or tc.get('d', tc.get('hidden', 128)),
                blocks=a.blocks or tc.get('layers', tc.get('blocks', 4)),
                heads=a.heads or tc.get('heads', 4), ff=a.ff, ch=a.ch)
    student = D.new_student(teacher, size['arch'], size['hidden'], size['blocks'], size['heads'], size['ff'], size['ch'],
                            a.seed, game)
    check_encoding(student.config, game, 'the student')
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
    W = window(train, student, budget_rows(a.ram_budget_gb, student.features), a.seed)
    H = window(held, student, a.heldout_rows, a.seed + 1) if held else None
    load_s = time.time() - t0
    T, t_s = D.timed(D.outputs, teacher, W, a.device)
    TH = D.outputs(teacher, H, a.device) if H is not None else None
    steps = a.steps or max(1, int(round((a.epochs or 1.0) * W.N / a.batch)))
    settings = dict(vars(a), game=game, size=size, steps=steps)
    settings.pop('compare', None)
    part = a.out + '.partial'
    opt, start = None, 0
    if a.resume and os.path.exists(part):
        ck = torch.load(part, map_location=a.device, weights_only=False)
        if ck['config'] != student.config:
            raise SystemExit(f'{part}: a student of another size or encoding than asked; delete it or drop --resume')
        student.load_state_dict(ck['state'])
        opt, start = ck['opt'], int(ck['step'])
    log({'event': 'start', 'teacher': os.path.abspath(a.teacher), 'teacher_net': D.describe(teacher),
         'student_net': D.describe(student), 'train_shards': len(train), 'train_rows': W.N, 'loaded_rows': W.loaded,
         'heldout_shards': len(held), 'heldout_rows': H.N if H is not None else 0, 'steps': steps, 'start': start,
         'epochs': round(steps * a.batch / W.N, 2), 'load_s': round(load_s, 1), 'teacher_s': round(t_s, 1),
         'device': a.device})
    hist = []

    def held_out(step):
        if H is None:
            return None
        r = D.agreement(D.outputs(student, H, a.device), TH, H)
        student.train()
        hist.append({'step': step, **r})
        log({'event': 'heldout', 'step': step, **r})
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
              a.device, a.seed, a.prefetch, opt, start, a.report_every, on_report, on_step)
    final = held_out(steps)
    student.to('cpu').eval()
    check_encoding(student.config, game, 'the student')
    extra = {'distilled_from': os.path.abspath(a.teacher), 'teacher_sha256': sha256(a.teacher), 'teacher_extra': textra,
             'distill': {'settings': settings, 'teacher_net': D.describe(teacher), 'student_net': D.describe(student),
                         'train_shards': [os.path.abspath(p) for p in train],
                         'heldout_shards': [os.path.abspath(p) for p in held], 'train_rows': W.N,
                         'heldout': hist, 'seconds': round(time.time() - t0, 1)}}
    save(student, a.out, extra)
    if os.path.exists(part):
        os.remove(part)
    log({'event': 'done', 'out': os.path.abspath(a.out), 'student_net': D.describe(student), 'steps': steps,
         'heldout': final, 'seconds': round(time.time() - t0, 1)})
    return 0


def cli():
    sys.exit(main())


if __name__ == '__main__':
    cli()
