"""Screenshots of the dashboard on the fixtures snapshot -> dashboard/docs/*.png

    cd dashboard && python tools/screenshots.py [--out docs]

Phone (390 px) and desktop (1280 px) widths, light and dark, plus the Pause-all confirm sheet
and a variant with St. Pete's az6_cq as a run on the Mac (its mac_sync mirror). No port
is opened: Playwright's requests are answered by the Flask app's test client. The machine strip
shows fixed sample values (the desktop's GPU / disks are not part of the snapshot).
"""
import argparse
import json
import os
import sys
import tempfile
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from zerodash import fixtures  # noqa: E402
from zerodash.app import create_app  # noqa: E402
from zerodash.model import Dashboard  # noqa: E402


class SampleMachine:
    """The machine strip with fixed values (what the desktop showed around the snapshot)."""

    def __init__(self, now):
        self.now = now

    def start(self):
        pass

    def snapshot(self):
        return {'gpu': {'gpus': [{'mem_used_mb': 5727, 'mem_total_mb': 32607, 'power_w': 148.4, 'util_pct': 85}]},
                'cpu': {'percent': 41.0, 'cores': 32, 'mem_percent': 58.0, 'mem_used_gb': 37.0, 'mem_total_gb': 63.8},
                'disks': [{'label': 'C:', 'path': 'C:/', 'free_gb': 12.8, 'total_gb': 1861.9, 'warn': True,
                           'warn_below_gb': 15},
                          {'label': 'E:', 'path': 'E:/', 'free_gb': 1388.0, 'total_gb': 3726.0, 'warn': False,
                           'warn_below_gb': None}],
                'mac': {'label': 'Mac', 'host': 'greggkeithley@100.95.248.116', 'load': [3.12, 2.87, 2.65],
                        'uptime': '9:53  up 6 days', 'time': self.now - 20, 'age_s': 20},
                'sampled': self.now - 4, 'age_s': 4}


def make_mirror(run_dir, now):
    """Turn the az6_cq copy into what St. Pete's mac_sync leaves on the desktop for a run on the
    Mac (MAC_SYNC + mac_state.json; the run and its supervisor alive there)."""
    with open(os.path.join(run_dir, 'MAC_SYNC'), 'w', encoding='utf-8') as f:
        json.dump({'host': 'greggkeithley@100.95.248.116', 'remote': '/Users/greggkeithley/stpete-zero/runs/az6_cq',
                   'ctl_mirrored': {'PAUSE': None, 'STOP': None}, 'last_pass': '2026-10-12 09:49:40'}, f)
    sup = {'pid': 51234, 'started': now - 4000, 'since': now - 3999, 'state': 'running', 'run_pid': 51240}
    run = {'pid': 51240, 'started': now - 3999, 'since': now - 3998, 'state': 'running'}
    with open(os.path.join(run_dir, 'mac_state.json'), 'w', encoding='utf-8') as f:
        json.dump({'run': run, 'run_alive': True, 'supervisor': sup, 'supervisor_alive': True, 'pause': False,
                   'stop': False, 'host': 'greggkeithley@100.95.248.116',
                   'remote': '/Users/greggkeithley/stpete-zero/runs/az6_cq', 'checked': now - 220}, f)


def build(tmp, mirror=False):
    cfg, snap = fixtures.prepare(tmp)
    if mirror:
        make_mirror(os.path.join(tmp, 'fixtures', 'stpete', 'runs', 'az6_cq'), snap['now'])
    dash = Dashboard(cfg, liveness=fixtures.liveness(snap), machine=SampleMachine(snap['now']))
    return create_app(dash)


def shoot(app, out, shots):
    from playwright.sync_api import sync_playwright
    client = app.test_client()

    def handle(route):
        req = route.request
        u = urlsplit(req.url)
        path = u.path + ('?' + u.query if u.query else '')
        r = client.open(path, method=req.method, data=req.post_data, headers={'Content-Type': 'application/json'})
        route.fulfill(status=r.status_code, body=r.get_data(),
                      headers={'Content-Type': r.headers.get('Content-Type', 'text/plain')})

    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for name, width, scheme, action in shots:
            ctx = br.new_context(viewport={'width': width, 'height': 900 if width > 600 else 844},
                                 device_scale_factor=2 if width < 600 else 1, color_scheme=scheme,
                                 is_mobile=width < 600, has_touch=width < 600)
            page = ctx.new_page()
            page.route('http://zerodash.test/**', handle)
            page.goto('http://zerodash.test/')
            page.wait_for_selector('.project')
            page.wait_for_timeout(300)
            if action == 'pause-all':
                page.click('#pause-all')
                page.wait_for_selector('#confirm:not([hidden])')
                page.wait_for_timeout(200)
                page.screenshot(path=os.path.join(out, name))
            else:
                page.screenshot(path=os.path.join(out, name), full_page=True)
            print('wrote', os.path.join(out, name))
            ctx.close()
        br.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'docs'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shoot(build(os.path.join(tmp, 'a')), a.out, [
            ('phone_light.png', 390, 'light', None),
            ('phone_dark.png', 390, 'dark', None),
            ('desktop_light.png', 1280, 'light', None),
            ('desktop_dark.png', 1280, 'dark', None),
            ('phone_pause_all_confirm.png', 390, 'light', 'pause-all'),
        ])
        shoot(build(os.path.join(tmp, 'b'), mirror=True), a.out, [
            ('phone_mac_mirror_light.png', 390, 'light', None),
        ])


if __name__ == '__main__':
    main()
