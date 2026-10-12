"""projects.json: which projects and runs the dashboard shows.

    {
      "title": "Zero dashboard",
      "refresh_s": 15,                 page refresh interval
      "hide_after_days": 3,            runs untouched this long and with no live supervisor are
                                       hidden until "Show all" is on
      "stale_after_min": 45,           a run with a live process but no new output for longer than
                                       max(this, 1.5 x its recent iteration time) is "stale"
      "disks": [{"path": "C:/", "label": "C:", "warn_below_gb": 15}, ...],
      "mac": {"ssh": "user@host", "label": "Mac"},     machine strip: the Mac's `uptime` over SSH
      "projects": [{
        "id": "glassroad", "name": "Glass Road Zero",
        "repo": "C:/Users/.../glassroad-zero",
        "dashboard": "http://desktop-j05412i:5280",    the game's own dashboard
        "roots": ["C:/Users/.../glassroad-zero/runs"], every run directory (config.json or
                                                       metrics.jsonl) under these; the same
                                                       directory reached twice (a junction) is
                                                       shown once
        "pause": true,                                 the runs honour PAUSE (false: STOP only,
                                                       no Pause button, pause-all skips them;
                                                       "auto": per run - a mac_sync mirror or a
                                                       run with runctl state files honours it,
                                                       an older run only STOP)
        "pause_note": "no pause (STOP only)",          shown for runs without pause support
        "yardstick": {"key": "eval_yard", "label": "JS bot"},   metrics.jsonl evaluation shown
        "stale_after_min": 45,                         optional per-project override
        "runs": {"az6_cq": {"pause": true}},           optional per-run overrides (pause,
                                                       stale_after_min, hidden)
        "remote_runs": [{                              runs on another host (the Mac)
          "name": "az6_cq",
          "ssh": "greggkeithley@100.95.248.116",       default: mac.ssh
          "path": "~/stpete-zero/runs/az6_cq",         the run directory on that host
          "mirror": "C:/Users/.../mac_mirror/az6_cq",  a local copy of its files that a separate
                                                       sync keeps fresh (metrics, logs, progress,
                                                       elo) - read for everything but control
          "label": "Mac",
          "pause": true
        }]
      }]
    }

Relative paths are relative to the directory of the config file. "now" (an epoch) freezes the
dashboard's clock (the fixtures snapshot only).
"""
import json
import os
import re

ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
HOST_RE = re.compile(r'^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$|^[A-Za-z0-9._:-]+$')

DEFAULTS = {
    'title': 'Zero dashboard',
    'refresh_s': 15,
    'hide_after_days': 3.0,
    'stale_after_min': 45.0,
}


class ConfigError(ValueError):
    pass


def load(path, base=None):
    """Read and check a projects.json -> the normalised config dict (absolute paths)."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return normalize(raw, base or os.path.dirname(os.path.abspath(path)))


def _abs(base, p):
    if not isinstance(p, str) or not p:
        raise ConfigError(f'expected a path, got {p!r}')
    p = os.path.expanduser(p)
    return os.path.normpath(p if os.path.isabs(p) or re.match(r'^[A-Za-z]:[\\/]', p) else os.path.join(base, p))


def _host(h, where):
    if not isinstance(h, str) or not HOST_RE.match(h) or h.startswith('-'):
        raise ConfigError(f'{where}: bad ssh host {h!r} (user@host)')
    return h


def _pause_flag(v, where):
    if v == 'auto' or isinstance(v, bool):
        return v
    raise ConfigError(f'{where}: "pause" must be true, false or "auto"')


def normalize(raw, base):
    if not isinstance(raw, dict) or not isinstance(raw.get('projects'), list):
        raise ConfigError('projects.json: expected {"projects": [...]}')
    cfg = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in raw:
            cfg[k] = raw[k]
    cfg['now'] = float(raw['now']) if raw.get('now') is not None else None
    mac = raw.get('mac')
    cfg['mac'] = None
    if mac:
        cfg['mac'] = {'ssh': _host(mac.get('ssh'), 'mac'), 'label': mac.get('label') or 'Mac'}
    cfg['disks'] = []
    for d in raw.get('disks', []):
        cfg['disks'].append({'path': d['path'], 'label': d.get('label') or d['path'],
                             'warn_below_gb': d.get('warn_below_gb')})
    seen = set()
    cfg['projects'] = []
    for p in raw['projects']:
        pid = p.get('id')
        if not isinstance(pid, str) or not ID_RE.match(pid) or pid in seen:
            raise ConfigError(f'project id {pid!r}: missing, not a plain name, or used twice')
        seen.add(pid)
        proj = {
            'id': pid,
            'name': p.get('name') or pid,
            'repo': _abs(base, p['repo']) if p.get('repo') else None,
            'dashboard': p.get('dashboard'),
            'roots': [_abs(base, r) for r in p.get('roots', [])],
            'pause': _pause_flag(p.get('pause', True), pid),
            'pause_note': p.get('pause_note') or 'no pause (STOP only)',
            'yardstick': dict(p.get('yardstick') or {}),
            'stale_after_min': float(p.get('stale_after_min', cfg['stale_after_min'])),
            'runs': dict(p.get('runs') or {}),
            'remote_runs': [],
        }
        rids = set()
        for r in p.get('remote_runs', []):
            name = r.get('name')
            if not isinstance(name, str) or not ID_RE.match(name):
                raise ConfigError(f'{pid}: remote run name {name!r} is not a plain name')
            label = r.get('label') or (cfg['mac'] or {}).get('label') or 'remote'
            rid = f'{name}@{re.sub(r"[^A-Za-z0-9]", "", label).lower() or "remote"}'
            if rid in rids:
                raise ConfigError(f'{pid}: remote run {rid} listed twice')
            rids.add(rid)
            host = r.get('ssh') or (cfg['mac'] or {}).get('ssh')
            path = r.get('path')
            if not isinstance(path, str) or not path or '\n' in path or '\0' in path:
                raise ConfigError(f'{pid}/{name}: remote runs need "path" (the run directory on the host)')
            proj['remote_runs'].append({
                'id': rid, 'name': name, 'label': label,
                'ssh': _host(host, f'{pid}/{name}'),
                'path': path,
                'mirror': _abs(base, r['mirror']) if r.get('mirror') else None,
                'pause': r.get('pause', proj['pause']) is not False,
            })
        cfg['projects'].append(proj)
    return cfg


def run_option(proj, run_name, key, default=None):
    """A per-run override from the project's "runs" map, else the project's value."""
    o = proj['runs'].get(run_name) or {}
    if key in o:
        return o[key]
    return proj.get(key, default)
