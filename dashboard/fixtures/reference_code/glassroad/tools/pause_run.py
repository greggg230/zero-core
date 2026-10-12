"""Pause or resume a run (give the machine back within seconds; continue later).

  python tools/pause_run.py runs/all1            # pause: create runs/all1/PAUSE
  python tools/pause_run.py runs/all1 --resume   # resume: remove it (the supervisor relaunches the run)
  python tools/pause_run.py runs/all1 --status   # just print the run's state
  add --wait to wait until the run is paused (or running again) and print how long it took

Pausing: the run stops at its next pause point (between self-play / evaluation batches and
training steps; games in progress are dropped) and exits with status 75; tools/supervise.py
then waits for PAUSE to be removed and relaunches the run, which resumes from its newest
checkpoint. Same as the dashboard's Pause / Resume buttons (grz/az/runctl.py; docs/az.md 2).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from grz.az import runctl  # noqa: E402


def describe(st):
    s = st['state']
    if s is None:
        return 'unknown: ' + st['detail']
    out = s
    if st['paused_since'] and s in ('pausing', 'paused'):
        out += f' (PAUSE since {time.strftime("%H:%M:%S", time.localtime(st["paused_since"]))})'
    return out + (f' - {st["detail"]}' if st['detail'] else '')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--resume', action='store_true', help='remove the PAUSE file')
    g.add_argument('--status', action='store_true', help='only print the state')
    ap.add_argument('--wait', action='store_true', help='wait until the run is paused / running again')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds --wait waits at most')
    a = ap.parse_args(argv)
    run = a.run
    if not runctl.is_run_dir(run):
        print(f'{run}: not a run directory (no config.json / metrics.jsonl)')
        return 2
    if a.status:
        print(describe(runctl.run_status(run)))
        return 0
    t0 = time.time()
    if a.resume:
        had = runctl.request_resume(run)
        st = runctl.run_status(run)
        print('PAUSE removed' if had else 'no PAUSE file (the run was not paused)')
        if had and not st['supervisor_alive']:
            print('no supervisor is running for this run: relaunch it (README, "Operating a run") to resume')
    else:
        new = runctl.request_pause(run, by='tools/pause_run.py')
        print('PAUSE created' if new else 'PAUSE file already present')
    if a.wait:
        want = ('running',) if a.resume else ('paused', 'stopped')
        while time.time() - t0 < a.timeout:
            st = runctl.run_status(run)
            if st['state'] in want or st['state'] is None:
                break
            time.sleep(0.2)
        print(f'{time.time() - t0:.1f}s: {describe(runctl.run_status(run))}')
        if not a.resume:
            rs = runctl.read_state(run, 'run') or {}
            if rs.get('state') == 'paused' and rs.get('workers_stopped') and rs.get('pause_requested'):
                print(f'the run stopped its workers {rs["workers_stopped"] - rs["pause_requested"]:.1f}s and exited '
                      f'{rs["exited"] - rs["pause_requested"]:.1f}s after the PAUSE file was created')
    else:
        print(describe(runctl.run_status(run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
