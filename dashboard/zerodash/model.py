"""The dashboard's model: the configured projects, their runs (local run directories and
remote runs), one JSON snapshot of everything for the page, and the controls."""
import os
import threading
import time

from . import config as config_mod
from . import control, runfiles, summary
from .procs import PsutilLiveness
from .remote import Prober, RemoteRun, RemoteSource, SshRunner
from .summary import LocalSource, MirrorSource


# card order within a project: live runs, paused, crashed / unknown, stopped; newest first in each
ORDER = {'paused': 1, 'crashed': 2, 'unknown': 2, 'stopped': 3}


class NotFound(KeyError):
    pass


def _real(p):
    return os.path.normcase(os.path.realpath(p))


def pause_support(proj, name, d):
    """Does this local run honour PAUSE? The per-run override, else the project's flag; with
    "auto" (St. Pete): a mac_sync mirror (PAUSE is forwarded to the Mac) or a run whose process
    writes runctl state files (code with PAUSE support) does, an older run honours only STOP."""
    v = config_mod.run_option(proj, name, 'pause', True)
    if v != 'auto':
        return bool(v)
    return summary.is_mirror(d) or any(os.path.exists(os.path.join(d, f))
                                        for f in ('run_state.json', 'supervisor_state.json'))


class Dashboard:
    def __init__(self, cfg, clock=None, liveness=None, ssh=None, machine=None):
        self.cfg = cfg
        if clock is None:
            clock = (lambda: cfg['now']) if cfg.get('now') else time.time
        self.clock = clock
        self.liveness = liveness or PsutilLiveness()
        self.ssh = ssh or SshRunner()
        self.machine = machine
        self.remote = {}
        for p in cfg['projects']:
            for r in p['remote_runs']:
                if not r.get('mirror'):
                    raise config_mod.ConfigError(f"{p['id']}/{r['name']}: remote runs need a local \"mirror\" directory")
                self.remote[(p['id'], r['id'])] = RemoteRun(r, self.ssh, self.clock)
        self.prober = Prober(list(self.remote.values()))
        self.control_lock = threading.Lock()

    def start(self):
        """Background work: the remote probes and the machine strip's sampler."""
        self.prober.start()
        if self.machine is not None:
            self.machine.start()

    # -------------------------------------------------------------- runs

    def project(self, pid):
        for p in self.cfg['projects']:
            if p['id'] == pid:
                return p
        raise NotFound(pid)

    def discover(self, proj):
        """-> (runs, errors). Local runs: every run directory under the project's roots (one
        entry per real directory: junctions are followed, and remote runs' mirrors are left to
        their remote entry, whichever project's root they are in); then the configured remote runs."""
        runs, errors, seen, ids = [], [], set(), set()
        mirrors = {_real(r['mirror']) for p in self.cfg['projects'] for r in p['remote_runs']}
        for root in proj['roots']:
            try:
                names = sorted(os.listdir(root))
            except OSError as e:
                errors.append(f'{root}: {e.strerror or e}')
                continue
            for n in names:
                d = os.path.join(root, n)
                if not runfiles.is_run_dir(d):
                    continue
                rp = _real(d)
                if rp in seen or rp in mirrors:
                    continue
                seen.add(rp)
                if n in ids:
                    errors.append(f'{d}: another run is already called {n}')
                    continue
                ids.add(n)
                runs.append({'id': n, 'name': n, 'dir': d, 'remote': None,
                             'pause': pause_support(proj, n, d),
                             'hidden': (proj['runs'].get(n) or {}).get('hidden')})
        for r in proj['remote_runs']:
            runs.append({'id': r['id'], 'name': r['name'], 'dir': r['mirror'], 'pause': r['pause'],
                         'remote': self.remote[(proj['id'], r['id'])],
                         'hidden': (proj['runs'].get(r['id']) or {}).get('hidden')})
        return runs, errors

    def source(self, run, now=None):
        if run['remote'] is not None:
            return RemoteSource(run['remote'], run['dir'])
        if summary.is_mirror(run['dir']):
            return MirrorSource(run['dir'], self.clock() if now is None else now)
        return LocalSource(run['dir'])

    def summarize(self, proj, run, now=None):
        now = self.clock() if now is None else now
        src = self.source(run, now)
        run = dict(run, remote_view=src.remote)
        try:
            return summary.summarize(proj, run, src, self.liveness, now, self.cfg)
        except Exception as e:                      # noqa: BLE001 - one broken run never breaks the page
            return {'key': f"{proj['id']}/{run['id']}", 'project': proj['id'], 'id': run['id'],
                    'name': run['name'], 'remote': src.remote, 'state': 'unknown', 'label': 'Unreadable',
                    'severity': 'muted', 'detail': f'{type(e).__name__}: {e}', 'hidden': False,
                    'pause_supported': run['pause'], 'pause': None, 'can_pause': False, 'can_resume': False,
                    'iteration': {}, 'eval': None, 'elo': None, 'throughput': None, 'modified': None}

    def find(self, pid, rid):
        proj = self.project(pid)
        for r in self.discover(proj)[0]:
            if r['id'] == rid:
                return proj, r
        raise NotFound(f'{pid}/{rid}')

    def all_runs(self):
        """-> [(project, run, summary)] for every run of every project."""
        now = self.clock()
        out = []
        for proj in self.cfg['projects']:
            for r in self.discover(proj)[0]:
                out.append((proj, r, self.summarize(proj, r, now)))
        return out

    def snapshot(self):
        now = self.clock()
        projects, pause_all = [], {}
        for proj in self.cfg['projects']:
            runs, errors = self.discover(proj)
            sums = [self.summarize(proj, r, now) for r in runs]
            sums.sort(key=lambda s: (s['hidden'], ORDER.get(s['state'], 0), -(s.get('modified') or 0), s['name']))
            for s in sums:
                if s.get('pause') and s['pause'].get('kind') == 'pause-all':
                    pid = s['pause']['by'][len(summary.PAUSE_ALL_PREFIX):].strip()
                    pause_all.setdefault(pid, []).append(s['key'])
            projects.append({
                'id': proj['id'], 'name': proj['name'], 'dashboard': proj.get('dashboard'),
                'pause_supported': proj['pause'], 'pause_note': proj['pause_note'] if proj['pause'] is False else None,
                'run_pause_note': proj['pause_note'],
                'runs': sums, 'errors': errors,
                'hidden_count': sum(1 for s in sums if s['hidden']),
            })
        return {
            'now': now, 'title': self.cfg['title'], 'refresh_s': self.cfg['refresh_s'],
            'hide_after_days': self.cfg['hide_after_days'],
            'projects': projects,
            'pause_all': {'ids': sorted(pause_all), 'runs': sorted(k for v in pause_all.values() for k in v)},
            'machine': self.machine.snapshot() if self.machine is not None else None,
        }

    # -------------------------------------------------------------- controls (control.py)

    def pause(self, pid, rid, by):
        with self.control_lock:
            return control.pause_run(self, pid, rid, by)

    def resume(self, pid, rid):
        with self.control_lock:
            return control.resume_run(self, pid, rid)

    def pause_all(self):
        with self.control_lock:
            return control.pause_all(self)

    def resume_all(self):
        with self.control_lock:
            return control.resume_all(self)
