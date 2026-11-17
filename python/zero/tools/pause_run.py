"""Pause or resume a run (give the machine back within seconds, or at the next iteration boundary; continue later).

  python tools/pause_run.py runs/all1               # pause now: create runs/all1/PAUSE
  python tools/pause_run.py runs/all1 --wait        # pause at the next iteration boundary and wait for it
  python tools/pause_run.py runs/all1 --boundary    # pause at the next iteration boundary (PAUSE_NEXT), no wait
  python tools/pause_run.py runs/all1 --now --wait  # pause now and wait until the run has stopped
  python tools/pause_run.py runs/all1 --resume      # resume: remove PAUSE (and a pending PAUSE_NEXT)
  python tools/pause_run.py runs/all1 --status      # just print the run's state

Pausing now (PAUSE): the run stops at its next pause point (between self-play / evaluation batches and training
steps; games in progress are dropped, and so are a remote's shards still travelling: the interrupted self-play phase
is played again) and exits with status 75; tools/supervise.py then waits for PAUSE to be removed and relaunches the
run, which resumes from its newest checkpoint. Same as the dashboard's Pause / Resume buttons (zero/runctl.py;
docs/az.md 2).

Pausing at the boundary (PAUSE_NEXT, docs/az.md 56.3; `--wait` without `--now`, or `--boundary`): the current
iteration finishes - its self-play phase with every remote shard, its training, checkpoint, evaluation and metrics
row - nothing of the next one starts, and the run then creates PAUSE itself and exits with status 75 as above. Resume
the same way; no finished iteration's work is lost. A run of a zero-core before PAUSE_NEXT ignores the file (`--wait`
says so after its timeout); `--now` pauses any run.
"""
import argparse
import os
import sys
import time

from .. import runctl


def describe(st):
    s = st['state']
    if s is None:
        return 'unknown: ' + st['detail']
    out = s
    if st['paused_since'] and s in ('pausing', 'paused'):
        out += f' (PAUSE since {time.strftime("%H:%M:%S", time.localtime(st["paused_since"]))})'
    return out + (f' - {st["detail"]}' if st['detail'] else '')


def last_iter(run):
    """The newest iteration of the run's metrics.jsonl (None: none readable)."""
    import json
    it = None
    try:
        with open(os.path.join(run, 'metrics.jsonl'), encoding='utf-8') as f:
            for line in f:
                try:
                    it = int(json.loads(line)['iter'])
                except (ValueError, KeyError, TypeError):
                    pass
    except OSError:
        pass
    return it


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--resume', action='store_true', help='remove the PAUSE file (and a pending PAUSE_NEXT)')
    g.add_argument('--status', action='store_true', help='only print the state')
    g.add_argument('--boundary', action='store_true',
                   help='pause at the next iteration boundary (PAUSE_NEXT: the current iteration finishes first)')
    g.add_argument('--now', action='store_true', help='pause at once (PAUSE), also with --wait')
    ap.add_argument('--wait', action='store_true',
                    help='wait until the run is paused / running again; a pause with --wait waits for the iteration '
                         'boundary unless --now')
    ap.add_argument('--timeout', type=float, default=None,
                    help='seconds --wait waits at most (default 120; 3600 for a boundary pause)')
    a = ap.parse_args(argv)
    run = a.run
    if not runctl.is_run_dir(run):
        print(f'{run}: not a run directory (no config.json / metrics.jsonl)')
        return 2
    if a.status:
        print(describe(runctl.run_status(run)))
        return 0
    boundary = not a.resume and not a.now and (a.boundary or a.wait)
    timeout = a.timeout if a.timeout is not None else (3600.0 if boundary else 120.0)
    t0 = time.time()
    it0 = last_iter(run)
    if a.resume:
        had = runctl.request_resume(run)
        st = runctl.run_status(run)
        print('PAUSE removed' if had else 'no PAUSE file (the run was not paused)')
        if had and not st['supervisor_alive']:
            print('no supervisor is running for this run: relaunch it (README, "Operating a run") to resume')
    elif boundary:
        if runctl.pause_requested(run):
            print('PAUSE file already present: the run pauses (or paused) at once, not at the boundary')
        else:
            new = runctl.request_pause_next(run, by='tools/pause_run.py')
            print(('PAUSE_NEXT created' if new else 'PAUSE_NEXT file already present') +
                  f': the run pauses after its current iteration (the newest finished: {it0})')
    else:
        new = runctl.request_pause(run, by='tools/pause_run.py')
        print('PAUSE created' if new else 'PAUSE file already present')
    if a.wait:
        want = ('running',) if a.resume else ('paused', 'stopped')
        while time.time() - t0 < timeout:
            st = runctl.run_status(run)
            if st['state'] in want or st['state'] is None:
                break
            time.sleep(0.2 if not boundary else 2.0)
        st = runctl.run_status(run)
        print(f'{time.time() - t0:.1f}s: {describe(st)}')
        if boundary:
            if runctl.pause_next_requested(run) and st['state'] == 'stopped':
                print('no run process: PAUSE_NEXT stays, and the run pauses as soon as it starts (before any work)')
            elif st['state'] not in want:
                print(f'not paused after {timeout:.0f}s: PAUSE_NEXT stays (the run pauses at its boundary; remove '
                      f'{runctl.pause_next_path(run)} to cancel, or a run of an older zero-core ignores it: --now)')
            else:
                print(f'paused at the boundary after iteration {last_iter(run)} (the newest finished when asked: {it0})')
        elif not a.resume:
            rs = runctl.read_state(run, 'run') or {}
            if rs.get('state') == 'paused' and rs.get('workers_stopped') and rs.get('pause_requested'):
                print(f'the run stopped its workers {rs["workers_stopped"] - rs["pause_requested"]:.1f}s and exited '
                      f'{rs["exited"] - rs["pause_requested"]:.1f}s after the PAUSE file was created')
    else:
        print(describe(runctl.run_status(run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
