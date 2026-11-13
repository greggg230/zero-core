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
"""
import argparse
import json
import os
import subprocess
import sys
import time

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
    res = {}
    for side, root in (('old', a.old), ('new', a.new)):
        res[side] = digest(os.path.abspath(root), a.package, args)
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
