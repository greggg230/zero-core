import json
import os
import shutil
import sys

import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from zerodash import fixtures  # noqa: E402
from zerodash.model import Dashboard  # noqa: E402
from zerodash.procs import FixedLiveness  # noqa: E402


class Fx:
    """A temporary copy of the fixtures snapshot (mtimes restored, clock frozen at the
    snapshot) and helpers to build a Dashboard on it."""

    def __init__(self, root):
        self.root = root
        self.cfg, self.snap = fixtures.prepare(root)
        self.now = self.snap['now']

    def dir(self, project, run):
        return os.path.join(self.root, 'fixtures', project, 'runs', run)

    def path(self, project, run, name):
        return os.path.join(self.dir(project, run), name)

    def dash(self, alive=None, now=None, liveness=None, ssh=None, cfg=None):
        if liveness is None:
            liveness = FixedLiveness(self.snap['alive'] if alive is None else alive)
        t = self.now if now is None else now
        return Dashboard(cfg or self.cfg, clock=lambda: t, liveness=liveness, ssh=ssh)

    def runs(self, dash=None):
        """{'project/run': summary}"""
        snap = (dash or self.dash()).snapshot()
        return {r['key']: r for p in snap['projects'] for r in p['runs']}

    def read(self, project, run, name):
        with open(self.path(project, run, name), encoding='utf-8') as f:
            return f.read()

    def write_json(self, project, run, name, obj):
        with open(self.path(project, run, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def edit_json(self, project, run, name, **kw):
        p = self.path(project, run, name)
        with open(p, encoding='utf-8') as f:
            j = json.load(f)
        j.update(kw)
        st = os.stat(p)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(j, f)
        os.utime(p, (st.st_atime, st.st_mtime))
        return j


@pytest.fixture
def fx(tmp_path):
    return Fx(str(tmp_path))


def posix_shell():
    """A POSIX sh to stand in for the remote host's shell (Git's sh on Windows; never WSL's
    System32 bash)."""
    cands = [shutil.which('sh'), shutil.which('bash'), r'C:\Program Files\Git\usr\bin\sh.exe',
             r'C:\Program Files\Git\bin\sh.exe', '/bin/sh']
    for c in cands:
        if c and os.path.exists(c) and 'system32' not in c.lower():
            return c
    return None
