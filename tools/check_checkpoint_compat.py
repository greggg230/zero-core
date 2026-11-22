"""Old checkout vs new checkout: the same checkpoint (or evaluator) must play identically.

  python tools/check_checkpoint_compat.py --old ~/glassroad-zero-wt/run-1fd27be \\
      --new ~/glassroad-zero-wt/run-<new sha> --package grz \\
      --model ~/glassroad-zero/runs/grz2p_cq/models/iter_0045.pt [--games 24 --sims 32 --match-games 16]
  python tools/check_checkpoint_compat.py --old A --new B --package cvz --net random --game caverna \\
      --spec '{"full": true}'                                    # golden digests without a checkpoint

Runs tools/selfplay_digest.py (next to this script) once per checkout, each in its own Python
process importing only that checkout (its `<pkg>/_engine.pyd` must be built: tools/build_py.sh),
with fixed seeds: self-play records + results for policy_target 'visits' and 'cq' (the
checkpoint as the evaluator; CPU, 1 torch thread) and a match against the built-in random bot.
Exit status 0 when every digest is identical, 1 otherwise (the differing fields are printed). The
sample arrays compared are named (`arrays hashed: ...` per side, `arrays` in the result line): a game
whose engine computes auxiliary targets has them recorded and hashed too (docs/az.md 47.12).
The checkpoint and the checkouts are only read (Python bytecode is not written). The game and
spec default to the checkpoint's game and the game's 2p spec ('{"intro": false}' Glass Road,
'{"full": true}' Caverna) - pass --spec to override. Extra arguments go to selfplay_digest.py.

**At once** (docs/az.md 75.1): a side's digests are independent of each other - each policy target's self-play and the
match is its own driver with its own fixed seed, on an evaluator built the same way in every process - so each is its
own selfplay_digest.py process (`--targets <one> --match-games 0`, and `--targets '' --match-games N`), and both
sides' processes run together, --jobs at a time (default: all of them, 6 for the two targets and the match). Each
digest is the one the one-process-per-side run computes (tests/test_checkpoint_compat.py compares the two ways);
`--jobs 1` is that run: one process per side, one side after the other.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SPECS = {'glassroad': '{"intro": false}', 'caverna': '{"full": true}'}


def digest(root, package, args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1')
    env.pop('PYTHONPATH', None)
    cmd = [sys.executable, os.path.join(HERE, 'selfplay_digest.py'), '--root', root, '--package', package, *args]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f'{root}: selfplay_digest failed ({p.returncode}):\n{p.stdout}\n{p.stderr}')
    out = json.loads(p.stdout.strip().splitlines()[-1])
    out['seconds'] = round(time.time() - t0, 1)
    return out


def phases(targets, match_games):
    """A side's independent digests as selfplay_digest.py arguments: one per policy target (no match), then the match
    (no target)."""
    out = [['--targets', t, '--match-games', '0'] for t in targets]
    if match_games > 0:
        out.append(['--targets', '', '--match-games', str(match_games)])
    return out


def merge(parts):
    """One side's phase outputs (in `phases` order) -> the output of one process that ran them all: the first's
    fields, every target, the match. The phases must agree on what they ran (game, spec, evaluator, engine)."""
    out = dict(parts[0])
    out['targets'] = {}
    out.pop('match', None)
    for r in parts:
        for k in ('game', 'spec', 'evaluator', 'engine', 'root'):
            if r.get(k) != out.get(k):
                raise SystemExit(f'{out.get("root")}: its digest processes disagree on {k}: {out.get(k)} != {r.get(k)}')
        out['targets'].update(r['targets'])
        if r.get('match'):
            out['match'] = r['match']
    out['seconds'] = round(max(r['seconds'] for r in parts), 1)
    return out


def compare(a, b):
    """-> list of differing fields (paths)."""
    bad = []
    for t in sorted(set(a['targets']) | set(b['targets'])):
        if a['targets'].get(t) != b['targets'].get(t):
            bad.append(f'targets.{t}: {a["targets"].get(t)} != {b["targets"].get(t)}')
    if a.get('match') != b.get('match'):
        bad.append(f'match: {a.get("match")} != {b.get("match")}')
    for k in ('game', 'spec', 'evaluator'):
        if a.get(k) != b.get(k):
            bad.append(f'{k}: {a.get(k)} != {b.get(k)}')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--old', required=True, help='the old checkout (e.g. the live run\'s pinned worktree)')
    ap.add_argument('--new', required=True, help='the new checkout (zero-core based, built)')
    ap.add_argument('--package', required=True, help='grz | cvz')
    ap.add_argument('--model', default=None, help='a checkpoint (.pt); default: --net')
    ap.add_argument('--net', default='random', choices=('uniform', 'random'))
    ap.add_argument('--game', default=None)
    ap.add_argument('--spec', default=None)
    ap.add_argument('--targets', default='visits,cq', help='the policy targets to play and digest (selfplay_digest.py\'s)')
    ap.add_argument('--match-games', type=int, default=16, help='games against the random bot (0: no match)')
    ap.add_argument('--jobs', type=int, default=0, help='digest processes at once (default 0: all of them - every '
                    'target and the match of both sides; 1: one process per side, one side after the other)')
    a, extra = ap.parse_known_args(argv)
    args = list(extra)
    game = a.game
    if a.model:
        import torch
        game = game or torch.load(a.model, map_location='cpu', weights_only=False)['config'].get('game', 'toy')
        args += ['--model', os.path.abspath(a.model)]
    else:
        args += ['--net', a.net]
    game = game or 'toy'
    spec = a.spec if a.spec is not None else SPECS.get(game, '')
    args += ['--game', game, '--spec', spec]
    targets = [t for t in a.targets.split(',') if t]
    sides = (('old', os.path.abspath(a.old)), ('new', os.path.abspath(a.new)))
    plan = phases(targets, a.match_games)
    res = {}
    if a.jobs == 1 or len(plan) <= 1:
        for side, root in sides:
            res[side] = digest(root, a.package, [*args, '--targets', ','.join(targets), '--match-games',
                                                 str(a.match_games)])
    else:
        # every digest of both sides at once (threads only wait on their processes); a failed one ends the run
        with ThreadPoolExecutor(max_workers=a.jobs if a.jobs > 0 else 2 * len(plan)) as pool:
            jobs = {side: [pool.submit(digest, root, a.package, [*args, *ph]) for ph in plan] for side, root in sides}
            for side, _ in sides:
                res[side] = merge([j.result() for j in jobs[side]])
    for side, _ in sides:
        r = res[side]
        print(f'{side}: {r["root"]} ({r["seconds"]} s; engine {r["engine"]})', flush=True)
        for t, d in r['targets'].items():
            print(f'  {t}: {d["samples"]} samples, records {d["records"][:16]}, results {d["results"][:16]}')
        arrays = sorted({k for d in r['targets'].values() for k in d.get('arrays', ())})
        print(f'  arrays hashed: {", ".join(arrays) or "(not named: a selfplay_digest.py from before 47.12)"}')
        if r.get('match'):
            print(f'  vs random: {r["match"]["score"]}/{r["match"]["games"]}, results {r["match"]["results"][:16]}')
    bad = compare(res['old'], res['new'])
    arrays = sorted({k for d in res['new']['targets'].values() for k in d.get('arrays', ())})
    print(json.dumps({'game': game, 'spec': spec, 'evaluator': res['new']['evaluator'] if not a.model else a.model,
                      'arrays': arrays, 'identical': not bad, 'differences': bad}))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
