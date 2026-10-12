"""Start a command detached from the caller, in a hidden console (no window flashes,
survives the shell/session that launched it). Output goes to the given log file.

  python tools/launch_hidden.py <log file> <command> [args...]

Windows: CreateProcess with CREATE_NEW_CONSOLE + SW_HIDE (same approach as the hub launcher;
DETACHED_PROCESS / CREATE_NO_WINDOW make grandchildren flash terminals). Elsewhere: a new
session (setsid) with stdin from /dev/null, the same log file for stdout / stderr.
"""
import os
import subprocess
import sys


def _launch_windows(log_path, cmd):
    import ctypes
    import ctypes.wintypes as wt
    import msvcrt

    class SI(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                    ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                    ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                    ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE),
                    ("hStdError", wt.HANDLE)]

    class PI(ctypes.Structure):
        _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD),
                    ("dwThreadId", wt.DWORD)]

    K = ctypes.WinDLL("kernel32", use_last_error=True)
    fp = open(log_path, "a", buffering=1, encoding="utf-8")
    h = msvcrt.get_osfhandle(fp.fileno())
    K.SetHandleInformation(h, 1, 1)                       # inheritable
    si = SI()
    si.cb = ctypes.sizeof(SI)
    si.dwFlags = 0x1 | 0x100                              # STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES
    si.wShowWindow = 0                                    # SW_HIDE
    si.hStdOutput = h
    si.hStdError = h
    pi = PI()
    CREATE_NEW_CONSOLE = 0x10
    ok = K.CreateProcessW(None, subprocess.list2cmdline(cmd), None, None, True, CREATE_NEW_CONSOLE, None,
                          os.getcwd(), ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return pi.dwProcessId


def _launch_posix(log_path, cmd):
    fp = open(log_path, "a", buffering=1, encoding="utf-8")
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 2 or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv and argv[0] in ('-h', '--help') else 2
    log_path, cmd = argv[0], argv[1:]
    pid = (_launch_windows if os.name == 'nt' else _launch_posix)(log_path, cmd)
    print(f"launched pid={pid} ({'hidden console' if os.name == 'nt' else 'new session'}) log={log_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
