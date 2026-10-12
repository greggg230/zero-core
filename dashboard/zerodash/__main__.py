"""python -m zerodash [--host 0.0.0.0] [--port 5200] [--config projects.json] [--fixtures]

Binds 0.0.0.0 by default (reached over Tailscale; hub id zero_dash, port 5200).
--fixtures serves a temporary copy of fixtures/ (the desktop snapshot, clock frozen at the
snapshot) with fixtures/../projects.fixtures.json: the controls act on the copy.
"""
import argparse
import atexit
import shutil
import sys
import tempfile

from . import DEFAULT_CONFIG, config, fixtures
from .app import create_app
from .machine import Machine
from .model import Dashboard
from .remote import SshRunner


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m zerodash', description='Cross-game Zero dashboard')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5200)
    ap.add_argument('--config', default=None, help=f'projects.json (default: {DEFAULT_CONFIG})')
    ap.add_argument('--fixtures', action='store_true', help='serve a copy of the fixtures snapshot')
    ap.add_argument('--no-machine', action='store_true', help='no machine strip (nvidia-smi, ssh uptime)')
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                          # noqa: BLE001
            pass
    ssh = SshRunner()
    liveness = None
    if a.fixtures:
        tmp = tempfile.mkdtemp(prefix='zerodash-fixtures-')
        atexit.register(shutil.rmtree, tmp, True)
        cfg, snap = fixtures.prepare(tmp, a.config or fixtures.FIXTURES_CONFIG)
        liveness = fixtures.liveness(snap)
        print(f'zerodash: fixtures snapshot copied to {tmp}')
    else:
        cfg = config.load(a.config or DEFAULT_CONFIG)
    machine = None if a.no_machine else Machine(cfg, ssh=ssh)
    dash = Dashboard(cfg, liveness=liveness, ssh=ssh, machine=machine)
    dash.start()
    app = create_app(dash)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
