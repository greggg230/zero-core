"""Pause / Resume, Pause all / Resume all - the PAUSE contract (DESIGN.md 3; the games'
runctl.py):

* <run>/PAUSE holds JSON {"time": <epoch>, "by": "<who>"}; the run notices it within seconds,
  stops at its next pause point and exits with status 75; its supervisor relaunches it once
  PAUSE is removed.
* Pause writes PAUSE with "by": "zerodash <client address>"; an existing PAUSE is left as it is
  (its time and author are the original pause's). Resume removes the run's PAUSE, whoever
  wrote it.
* Pause all writes PAUSE into every active run that honours PAUSE and has none, with "by":
  "zerodash pause-all <id>". Resume all removes only PAUSE files whose "by" starts with
  "zerodash pause-all " - a run paused by hand, by a game dashboard or by a zerodash Pause
  button stays paused.
* Projects / runs configured without pause support (St. Pete: STOP only) are never written to.
* STOP files are never written or removed: stopping a run stays manual.
"""
import json
import os
import secrets
import time

from . import runfiles, state, summary


class ControlError(RuntimeError):
    """A control the run does not allow (HTTP 409)."""


def write_pause_local(run_dir, by, now):
    """Create <run>/PAUSE unless it exists -> True if created now. The new file is linked into
    place, which fails if a PAUSE appeared meanwhile (falls back to check-and-replace where
    hard links are not supported)."""
    p = os.path.join(run_dir, 'PAUSE')
    if os.path.exists(p):
        return False
    tmp = p + f'.zerodash.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'time': now, 'by': by}, f)
    try:
        try:
            os.link(tmp, p)
            return True
        except FileExistsError:
            return False
        except OSError:
            if os.path.exists(p):
                return False
            os.replace(tmp, p)
            return True
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def remove_pause_local(run_dir, only_pause_all=False):
    """Remove <run>/PAUSE -> True if removed. only_pause_all: only a PAUSE a pause-all wrote."""
    p = os.path.join(run_dir, 'PAUSE')
    if only_pause_all:
        cur = runfiles.parse_pause(run_dir)
        if summary.pause_kind(cur) != 'pause-all':
            return False
    try:
        os.remove(p)
        return True
    except FileNotFoundError:
        return False


def pause_run(dash, pid, rid, by):
    proj, run = dash.find(pid, rid)
    if not run['pause']:
        raise ControlError(f"{pid}/{rid}: {proj['pause_note']}")
    if run['remote'] is not None:
        changed = run['remote'].write_pause(json.dumps({'time': dash.clock(), 'by': by}))
        run['remote'].refresh()
    else:
        changed = write_pause_local(run['dir'], by, dash.clock())
    return {'ok': True, 'changed': changed, 'run': dash.summarize(proj, run)}


def resume_run(dash, pid, rid):
    proj, run = dash.find(pid, rid)
    if run['remote'] is not None:
        changed = run['remote'].remove_pause()
        run['remote'].refresh()
    else:
        changed = remove_pause_local(run['dir'])
    return {'ok': True, 'changed': changed, 'run': dash.summarize(proj, run)}


def new_pause_all_id(now):
    return time.strftime('%Y%m%d-%H%M%S', time.localtime(now)) + '-' + secrets.token_hex(2)


def pause_all(dash):
    """Pause every active run that honours PAUSE and is not paused yet."""
    now = dash.clock()
    pa_id = new_pause_all_id(now)
    by = summary.PAUSE_ALL_PREFIX + pa_id
    res = {'ok': True, 'id': pa_id, 'by': by, 'paused': [], 'already': [], 'skipped': [], 'errors': []}
    for proj, run, s in dash.all_runs():
        active = s['state'] in state.ACTIVE
        if not active:
            continue
        if s.get('pause'):
            res['already'].append({'key': s['key'], 'by': s['pause'].get('by')})
            continue
        if not run['pause']:
            res['skipped'].append({'key': s['key'], 'why': proj['pause_note']})
            continue
        try:
            if run['remote'] is not None:
                ok = run['remote'].write_pause(json.dumps({'time': now, 'by': by}))
            else:
                ok = write_pause_local(run['dir'], by, now)
            if ok:
                res['paused'].append({'key': s['key']})
            else:                                   # a PAUSE appeared meanwhile: left as it is
                res['already'].append({'key': s['key'], 'by': None})
        except Exception as e:                      # noqa: BLE001 - report, carry on with the others
            res['errors'].append({'key': s['key'], 'error': str(e)})
    dash.prober.poke()
    res['ok'] = not res['errors']
    return res


def resume_all(dash):
    """Remove the PAUSE files pause-all wrote (any pause-all), in every run; leave others."""
    res = {'ok': True, 'resumed': [], 'kept': [], 'errors': []}
    for proj in dash.cfg['projects']:
        for run in dash.discover(proj)[0]:
            key = f"{proj['id']}/{run['id']}"
            try:
                if run['remote'] is not None:
                    text = run['remote'].read_pause()
                    p = runfiles.pause_from_text(text) if text is not None else None
                    kind = summary.pause_kind(p)
                    if kind == 'pause-all':
                        if run['remote'].remove_pause(expect=text):
                            res['resumed'].append({'key': key})
                    elif p is not None:
                        res['kept'].append({'key': key, 'by': p.get('by')})
                else:
                    p = runfiles.parse_pause(run['dir'])
                    if p is None:
                        continue
                    if summary.pause_kind(p) == 'pause-all':
                        if remove_pause_local(run['dir'], only_pause_all=True):
                            res['resumed'].append({'key': key})
                    else:
                        res['kept'].append({'key': key, 'by': p.get('by')})
            except Exception as e:                  # noqa: BLE001
                res['errors'].append({'key': key, 'error': str(e)})
    dash.prober.poke()
    res['ok'] = not res['errors']
    return res
