"""A stall alarm for a supervised run, from outside it (docs/az.md 56.4): alarms when the run has gone `factor` x its
own median iteration time without finishing an iteration, and takes a stack dump of the run's Python processes.

  python tools/stall_watch.py runs/r1                 # watch (every 60 s, 24 h); on a stall: print, dump, exit
  python tools/stall_watch.py runs/r1 --once          # one look: exit 3 and print STALL ... when stalled, else 0
  python tools/stall_watch.py runs/r1 --status        # print the numbers it judges by

It only reads the run: log.txt (the `iter N train:` lines - an iteration's end - and the `run ... starting after iter
N` lines - a start), progress.json (the phase), the PAUSE / PAUSE_NEXT / STOP files and the state files' pids. The
time since the newer of the last iteration's end and the last start is judged against

    threshold = max(factor x median, slack x longest, floor)

over the last `window` iteration intervals (both ends in one process: an interval across a restart is not an
iteration). `longest` is what lets an evaluation iteration through without a false alarm: with eval_every 16 and a
window of 32 the slow iterations are in it (a local Elo phase, a yardstick on the GPU). With fewer than `min_intervals`
intervals (a new run) the threshold is `fallback` (75 min: watch_run.sh's old fixed rule). No alarm while PAUSE is
there (paused), nor once progress.json says 'idle' (the run ended).

On a stall: a stack dump of every Python process of the run - the run process (run_state.json's pid), its children
(workers, the shard writer, companions) and the supervisor - by `py-spy dump` when py-spy is on the PATH or in this Python's user scripts (pip install --user
py-spy; it reads another process's stacks without stopping it), plus each process's state, threads and CPU seconds
(psutil), written under `--dump-dir` (default: the temp directory's zero-stall/<run>-<time>/); without py-spy it says
so and what to install. Then it prints the alarm (the way tools/watch_run.sh does: a line a watching session sees) and
exits.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

TAIL_BYTES = 4 << 20              # the log's tail read each look
FACTOR, SLACK, FLOOR_S, FALLBACK_S, WINDOW, MIN_INTERVALS = 3.0, 1.5, 600.0, 4500.0, 32, 3
_TS = re.compile(r'^(\d\d):(\d\d):(\d\d) (.*)$')
_TRAIN = re.compile(r'^iter (\d+) train: ')
_START = re.compile(r'^run .* starting after iter (\d+)')


def _tail(path, n=TAIL_BYTES):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        data = f.read()
    text = data.decode('utf-8', errors='replace')
    if size > n:
        text = text.split('\n', 1)[-1]           # (a cut first line)
    return text.splitlines()


def events(lines, mtime):
    """The log's timestamped lines -> [(epoch seconds, kind 'train' | 'start', iteration)], oldest first. The lines
    carry only a time of day: the last one is placed at the latest such time not after the file's modification time
    (+ a minute), each earlier one before the line after it (a time of day greater than the next line's crossed a
    midnight)."""
    raw = []
    for ln in lines:
        m = _TS.match(ln)
        if not m:
            continue
        tod = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        rest = m.group(4)
        t = _TRAIN.match(rest)
        s = None if t else _START.match(rest)
        raw.append((tod, 'train' if t else 'start' if s else None, int((t or s).group(1)) if (t or s) else None))
    if not raw:
        return []
    ref = dt.datetime.fromtimestamp(mtime + 60)
    day = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    nxt = day.timestamp() + raw[-1][0]
    if nxt > ref.timestamp():
        nxt -= 86400
    out = []
    later_tod = None
    for tod, kind, it in reversed(raw):
        if later_tod is not None:
            back = later_tod - tod
            if back < 0:
                back += 86400                    # (crossed a midnight going back)
            nxt -= back
        later_tod = tod
        if kind:
            out.append((nxt, kind, it))
    return out[::-1]


def intervals(evs):
    """Iteration intervals (seconds): consecutive iterations' `train` lines with no start between them."""
    out = []
    prev = None
    for t, kind, it in evs:
        if kind == 'start':
            prev = None
        elif prev is not None and it == prev[1] + 1:
            out.append(t - prev[0])
            prev = (t, it)
        else:
            prev = (t, it)
    return out


def judge(run, now=None, factor=FACTOR, slack=SLACK, floor=FLOOR_S, fallback=FALLBACK_S, window=WINDOW,
          min_intervals=MIN_INTERVALS):
    """One look at the run -> {stalled, reason, since, age_s, threshold_s, median_s, longest_s, intervals, last, phase}."""
    now = time.time() if now is None else now
    log = os.path.join(run, 'log.txt')
    out = {'stalled': False, 'reason': '', 'phase': None, 'last': None}
    try:
        with open(os.path.join(run, 'progress.json'), encoding='utf-8') as f:
            out['phase'] = json.load(f).get('phase')
    except (OSError, ValueError, AttributeError):
        pass
    try:
        evs = events(_tail(log), os.path.getmtime(log))
    except OSError as e:
        out['reason'] = f'no log ({e})'
        return out
    iv = intervals(evs)[-window:]
    med = sorted(iv)[len(iv) // 2] if iv else None
    longest = max(iv) if iv else None
    thr = max(factor * med, slack * longest, floor) if len(iv) >= min_intervals else fallback
    since = evs[-1] if evs else None
    age = now - since[0] if since else None
    out.update(since=since, age_s=None if age is None else round(age, 1), threshold_s=round(thr, 1),
               median_s=None if med is None else round(med, 1), longest_s=None if longest is None else round(longest, 1),
               intervals=len(iv), last=next((it for _, k, it in reversed(evs) if k == 'train'), None))
    if os.path.exists(os.path.join(run, 'PAUSE')):
        out['reason'] = 'paused (PAUSE)'
    elif out['phase'] == 'idle':
        out['reason'] = 'ended (progress.json idle)'
    elif since is None:
        out['reason'] = 'no iteration or start line yet'
    elif age > thr:
        what = 'the run started' if since[1] == 'start' else f'iteration {since[2]} ended'
        out.update(stalled=True, reason=f'{age / 60:.1f} min since {what} '
                                        f'({time.strftime("%H:%M:%S", time.localtime(since[0]))}), threshold '
                                        f'{thr / 60:.1f} min ({"%.0f x median %.0f s" % (factor, med) if med else "fallback"}'
                                        f'{", longest %.0f s" % longest if longest else ""}), phase {out["phase"]!r}')
    else:
        out['reason'] = f'ok: {age / 60:.1f} of {thr / 60:.1f} min'
    return out


def run_pids(run):
    """The run's Python processes: the run process (run_state.json), its descendants, the supervisor."""
    pids = []
    for name in ('run_state.json', 'supervisor_state.json'):
        try:
            with open(os.path.join(run, name), encoding='utf-8') as f:
                p = int(json.load(f).get('pid') or 0)
        except (OSError, ValueError, TypeError):
            p = 0
        if p:
            pids.append(p)
    try:
        import psutil
    except ImportError:
        return pids
    out = []
    for p in pids:
        try:
            pr = psutil.Process(p)
            out += [p] + [c.pid for c in pr.children(recursive=True)]
        except psutil.Error:
            continue
    keep = []
    for p in dict.fromkeys(out):
        try:
            if 'python' in psutil.Process(p).name().lower():
                keep.append(p)
        except psutil.Error:
            pass
    return keep


def find_pyspy():
    """py-spy: on the PATH, else beside this Python's scripts (a `pip install --user py-spy` puts it in the user
    scripts directory, often not on the PATH) -> path or None."""
    p = shutil.which('py-spy')
    if p:
        return p
    import sysconfig
    exe = 'py-spy.exe' if os.name == 'nt' else 'py-spy'
    dirs = [sysconfig.get_path('scripts')]
    for scheme in (f'{os.name}_user', 'nt_user', 'posix_user', 'osx_framework_user'):
        try:
            dirs.append(sysconfig.get_path('scripts', scheme))
        except KeyError:
            pass
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, exe)):
            return os.path.join(d, exe)
    return None


def dump(run, out_dir=None, pyspy=None):
    """Stack dumps of the run's Python processes into a new directory -> (directory, notes)."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    d = out_dir or os.path.join(tempfile.gettempdir(), 'zero-stall')
    d = os.path.join(d, f'{os.path.basename(os.path.normpath(run))}-{stamp}')
    os.makedirs(d, exist_ok=True)
    pyspy = pyspy if pyspy is not None else find_pyspy()
    notes = []
    pids = run_pids(run)
    if not pids:
        notes.append('no live process of the run found (run_state.json / supervisor_state.json)')
    try:
        import psutil
    except ImportError:
        psutil = None
    for p in pids:
        info = {}
        if psutil is not None:
            try:
                pr = psutil.Process(p)
                ct = pr.cpu_times()
                info = {'pid': p, 'cmdline': pr.cmdline()[:8], 'status': pr.status(), 'threads': pr.num_threads(),
                        'cpu_user_s': ct.user, 'cpu_system_s': ct.system, 'rss_mb': pr.memory_info().rss >> 20}
            except psutil.Error as e:
                info = {'pid': p, 'error': str(e)}
        with open(os.path.join(d, f'{p}.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(info) + '\n\n')
            if pyspy:
                try:
                    r = subprocess.run([pyspy, 'dump', '--pid', str(p)], capture_output=True, text=True, timeout=60)
                    f.write(r.stdout + (('\n[stderr]\n' + r.stderr) if r.stderr.strip() else ''))
                except (OSError, subprocess.SubprocessError) as e:
                    f.write(f'py-spy failed: {e}\n')
                    notes.append(f'py-spy failed on {p}: {e}')
    if not pyspy:
        notes.append('py-spy is not on the PATH: process states only (pip install py-spy for the stacks)')
    return d, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run')
    ap.add_argument('--once', action='store_true', help='one look: exit 3 when stalled (after the dump), else 0')
    ap.add_argument('--status', action='store_true', help='print the numbers and exit')
    ap.add_argument('--hours', type=float, default=24.0)
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--factor', type=float, default=FACTOR, help='x the median iteration time (3)')
    ap.add_argument('--slack', type=float, default=SLACK, help='x the longest iteration of the window (1.5)')
    ap.add_argument('--floor', type=float, default=FLOOR_S, help='seconds, the least threshold (600)')
    ap.add_argument('--fallback', type=float, default=FALLBACK_S, help='seconds, the threshold of a new run (4500)')
    ap.add_argument('--window', type=int, default=WINDOW, help='iteration intervals judged by (32)')
    ap.add_argument('--dump-dir', default=None)
    ap.add_argument('--no-dump', action='store_true')
    a = ap.parse_args(argv)
    kw = dict(factor=a.factor, slack=a.slack, floor=a.floor, fallback=a.fallback, window=a.window)
    if a.status:
        print(json.dumps(judge(a.run, **kw), default=str))
        return 0
    end = time.time() + a.hours * 3600
    if not a.once:
        j = judge(a.run, **kw)
        print(f'stall watch start {time.strftime("%F %T")} {a.run}: {j["reason"]}', flush=True)
    while True:
        j = judge(a.run, **kw)
        if j['stalled']:
            print(f'STALL {time.strftime("%F %T")}: {j["reason"]}', flush=True)
            if not a.no_dump:
                d, notes = dump(a.run, a.dump_dir)
                print(f'stack dumps: {d}' + ''.join(f'\n  {n}' for n in notes), flush=True)
            return 3 if a.once else 0
        if a.once:
            return 0
        if time.time() > end:
            print(f'stall watch expired after {a.hours:g} h without a stall {time.strftime("%F %T")}')
            return 0
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
