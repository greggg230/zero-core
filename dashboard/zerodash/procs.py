"""Is the process a run file names still running?

Pids are reused, so a pid alone proves nothing: a state file's pid is checked against the
process's creation time recorded next to it ("started", as runctl does); a pid read from
supervisor.log is checked against the time of that log line (the process was created at most
a few minutes before it logged its pid). Every liveness answers True / False, or None when it
cannot know (a remote host that did not answer).
"""
import os
import threading
import time


class PsutilLiveness:
    """This machine's processes (psutil). Without psutil every answer is None.

    Opening a process is slow on Windows (~10 ms), so the set of live pids is listed at most
    every `ttl` seconds and a live pid's creation time is remembered while it stays listed."""

    def __init__(self, ttl=2.0):
        try:
            import psutil
            self.psutil = psutil
        except ImportError:                        # pragma: no cover - psutil is on the desktop
            self.psutil = None
        self.ttl = ttl
        self.lock = threading.Lock()
        self.pids, self.pids_t, self.ct = None, 0.0, {}

    def _listed(self, pid):
        now = time.monotonic()
        if self.pids is None or now - self.pids_t > self.ttl:
            self.pids, self.pids_t = set(self.psutil.pids()), now
            self.ct = {p: c for p, c in self.ct.items() if p in self.pids}
        return pid in self.pids

    def create_time(self, pid):
        """-> creation time of a live process with this pid, False if there is none, None if
        unknown."""
        if self.psutil is None:
            return None
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        with self.lock:
            try:
                if not self._listed(pid):
                    return False
            except Exception:                      # noqa: BLE001
                return None
            if pid in self.ct:
                return self.ct[pid]
            try:
                p = self.psutil.Process(pid)
                if os.name != 'nt' and p.status() == self.psutil.STATUS_ZOMBIE:
                    return False
                ct = p.create_time()
            except self.psutil.NoSuchProcess:
                return False
            except Exception:                      # noqa: BLE001 - access denied etc.
                return None
            self.ct[pid] = ct
            return ct

    def alive(self, pid, started=None, near=None):
        return check(self.create_time(pid) if pid else False, started, near)


def check(ct, started=None, near=None, tol=1.5, window=(-600.0, 5.0)):
    """A process's creation time `ct` (False: no such process, None: unknown) against what a
    run file says about it: `started` = its recorded creation time, `near` = the time it was
    logged."""
    if ct is None or ct is False:
        return ct if ct is None else False
    if started is not None:
        try:
            return abs(ct - float(started)) <= tol
        except (TypeError, ValueError):
            pass
    if near is not None:
        return near + window[0] <= ct <= near + window[1]
    return True


class FixedLiveness:
    """A fixed set of live processes {pid: creation time} (the fixtures snapshot, tests).
    `strict_near=False` ignores the log-line check: log times are local times, so a snapshot
    taken in one time zone and replayed in another would otherwise never match."""

    def __init__(self, alive=None, strict_near=False, unknown=False):
        self.procs = {int(k): (float(v) if v is not None else None) for k, v in (alive or {}).items()}
        self.strict_near = strict_near
        self.unknown = unknown

    def alive(self, pid, started=None, near=None):
        if self.unknown:
            return None
        if not pid or int(pid) not in self.procs:
            return False
        ct = self.procs[int(pid)]
        if ct is None:
            return True
        return check(ct, started, near if self.strict_near else None)
