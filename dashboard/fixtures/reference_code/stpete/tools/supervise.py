"""Keep an AlphaZero run going unattended.

  python tools/supervise.py runs/az1 [extra stpz.run args...]

* runs `python -m stpz.run --run <run> ...` and restarts it (resuming from the newest
  checkpoint) whenever it exits with an error, after killing any orphaned workers;
* backs off if it keeps crashing (5 crashes within an hour -> wait 5 minutes);
* asks Windows not to sleep while it is running;
* stops when the run exits cleanly (finished, or a STOP file in the run dir).
Logs to <run>/supervisor.log.
"""
import ctypes
import os
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(run, msg):
    with open(os.path.join(run, 'supervisor.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def keep_awake():
    if os.name == 'nt':
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def kill_orphans(pid):
    """Pool workers of a dead run keep its pid as their parent pid."""
    n = 0
    for p in psutil.process_iter(['pid', 'ppid']):
        if p.info['ppid'] == pid:
            try:
                for k in p.children(recursive=True):
                    k.kill()
                p.kill()
                n += 1
            except psutil.NoSuchProcess:
                pass
    return n


def main():
    run = sys.argv[1]
    extra = sys.argv[2:]
    os.makedirs(run, exist_ok=True)
    keep_awake()
    crashes = []
    log(run, f'supervisor {os.getpid()} starting: {run} {" ".join(extra)}')
    while True:
        out = open(os.path.join(run, 'stdout.txt'), 'a', encoding='utf-8')
        p = subprocess.Popen([sys.executable, '-m', 'stpz.run', '--run', run] + extra,
                             cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        log(run, f'started stpz.run pid {p.pid}')
        code = p.wait()
        out.close()
        killed = kill_orphans(p.pid)
        log(run, f'stpz.run exited with code {code}; killed {killed} orphaned workers')
        if code == 0:
            log(run, 'clean exit (finished or STOP file); supervisor done')
            return
        if os.path.exists(os.path.join(run, 'STOP')):
            log(run, 'STOP file present; not restarting')
            return
        now = time.time()
        crashes = [t for t in crashes if now - t < 3600] + [now]
        wait = 300 if len(crashes) >= 5 else 60
        log(run, f'crash #{len(crashes)} in the last hour; restarting in {wait}s')
        time.sleep(wait)


if __name__ == '__main__':
    main()
