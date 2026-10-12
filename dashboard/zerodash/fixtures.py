"""The fixtures snapshot (fixtures/<project>/runs/<run>/: real run files of all four projects,
taken on the desktop) as a live-looking dashboard, for development, tests and screenshots.

prepare(dest) copies the snapshot's run directories into `dest`, gives every file the
modification time it had on the desktop (fixtures/snapshot.json: git does not keep mtimes)
and returns the fixtures config (projects.fixtures.json) resolved against `dest`, with the
clock frozen at the moment the snapshot was taken. `liveness()` answers for the processes
that were alive then. The copy is what gets paused / resumed, never the repository.
"""
import json
import os
import shutil

from . import FIXTURES_CONFIG, FIXTURES_DIR, config
from .procs import FixedLiveness

PROJECTS = ('stpete', 'dominion', 'caverna', 'glassroad')


def snapshot():
    with open(os.path.join(FIXTURES_DIR, 'snapshot.json'), encoding='utf-8') as f:
        return json.load(f)


def prepare(dest, config_path=FIXTURES_CONFIG):
    """-> (config dict, snapshot dict); `dest` gets fixtures/<project>/runs/..."""
    snap = snapshot()
    for p in PROJECTS:
        shutil.copytree(os.path.join(FIXTURES_DIR, p), os.path.join(dest, 'fixtures', p), dirs_exist_ok=True)
    for rel, t in snap['mtimes'].items():
        path = os.path.join(dest, 'fixtures', *rel.split('/'))
        if os.path.exists(path):
            os.utime(path, (t, t))
    cfg = config.load(config_path, base=dest)
    cfg['now'] = snap['now']
    return cfg, snap


def liveness(snap):
    return FixedLiveness(snap['alive'])
