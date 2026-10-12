"""Hard stop of a run: python tools/kill_run.py runs/toy1 [--scan]

Kills the run's supervisor (tools/supervise.py, which would otherwise restart the run) and
then the <pkg>.az.run process tree of that run (the game package's `run_module`). The
processes come from the pids the run and its supervisor record in run_state.json /
supervisor_state.json (with the process creation time, so a reused pid is never killed); --scan
also searches every process's command line (slow on Windows: ~15 s; for processes that predate
the state files). For a clean stop after the
current iteration use `touch runs/toy1/STOP`; to pause, tools/pause_run.py.
"""
import os
import sys

import psutil

from .. import package, runctl  # light: no torch


def _target(p, args, flag):
    """Absolute path of the run named in a process's arguments (after `flag`)."""
    try:
        i = args.index(flag)
        path = args[i + 1]
    except (ValueError, IndexError):
        return None
    if not os.path.isabs(path):
        try:
            path = os.path.join(p.cwd(), path)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
    return os.path.normcase(os.path.abspath(path))


def _is_run_of(p, want):
    try:
        args = p.cmdline() or []
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return package.active().run_module in args and _target(p, args, '--run') == want


def from_state(run):
    """(supervisors, runs) recorded in the run's state files and still alive."""
    want = os.path.normcase(os.path.abspath(run))
    sups, runs = [], []
    ss, rs = runctl.read_state(run, 'supervisor'), runctl.read_state(run, 'run')
    if runctl.alive(ss):
        sups.append(psutil.Process(int(ss['pid'])))
        rp = ss.get('run_pid')
        if rp:
            try:
                p = psutil.Process(int(rp))
                if _is_run_of(p, want):
                    runs.append(p)
            except psutil.NoSuchProcess:
                pass
    if runctl.alive(rs) and not any(p.pid == int(rs['pid']) for p in runs):
        runs.append(psutil.Process(int(rs['pid'])))
    return sups, runs


def scan(run):
    """Every python process whose command line names this run (slow on Windows)."""
    want = os.path.normcase(os.path.abspath(run))
    me = os.getpid()
    sups, runs = [], []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if p.pid == me or not (p.info['name'] or '').lower().startswith('python'):
                continue
            args = p.cmdline() or []
            sup = next((a for a in args if a.replace('\\', '/').endswith('tools/supervise.py')), None)
            if sup and _target(p, args, sup) == want:
                sups.append(p)
            elif _is_run_of(p, want):
                runs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return sups, runs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if a != '--scan']
    if len(args) != 1:
        print(__doc__)
        return 2
    run = args[0]
    sups, runs = from_state(run)
    if '--scan' in argv:
        s2, r2 = scan(run)
        sups += [p for p in s2 if p.pid not in {q.pid for q in sups}]
        runs += [p for p in r2 if p.pid not in {q.pid for q in runs}]
    for p in sups + runs:                       # the supervisor first: it must not restart the run
        try:
            kids = p.children(recursive=True)
            for k in kids:
                k.kill()
            p.kill()
            print(f'killed {"supervisor" if p in sups else "run"} {p.pid} and {len(kids)} children')
        except psutil.NoSuchProcess:
            pass
    if not sups and not runs:
        print(f'no processes of {run} found' + ('' if '--scan' in argv else ' in its state files (--scan searches all processes)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
