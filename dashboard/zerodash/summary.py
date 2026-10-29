"""What one run card shows: state, iteration and phase, latest yardstick evaluation, Elo and
its trend, self-play throughput, who paused it and since when.

A run is read through a `RunSource`: a local run directory, or a remote run's local mirror
plus the control files / state files / processes its SSH probe saw (remote.py).
"""
import os

from . import config, runfiles, state

SPARK_POINTS = 60
YARDSTICK_KEYS = ('eval_yard', 'eval_bmu', 'eval_shrewd')
ACTIVITY_FILES = ('log.txt', 'stdout.txt', 'metrics.jsonl', 'progress.json', 'run_state.json')
PAUSE_ALL_PREFIX = 'zerodash pause-all '


class LocalSource:
    """A run directory on this machine."""
    remote = None
    lag = 0                       # how far behind the run its files may be (s)

    def __init__(self, path):
        self.dir = path

    def pause(self):
        return runfiles.parse_pause(self.dir)

    def stop(self):
        return os.path.exists(os.path.join(self.dir, 'STOP'))

    def state_file(self, who):
        j = runfiles.read_json(os.path.join(self.dir, f'{who}_state.json'))
        return j if isinstance(j, dict) else None

    def progress(self):
        return runfiles.progress_file(self.dir)

    def supervisor_lines(self):
        return runfiles.tail_lines(os.path.join(self.dir, 'supervisor.log'), 60)

    def liveness(self, default):
        return default


MIRROR = 'MAC_SYNC'                 # stpete-zero tools/mac_sync.py: this dir mirrors a run on the Mac
MIRROR_LAG_S = 900                  # mac_sync pulls the files every ~10 min (PAUSE / STOP within ~5 s)
MIRROR_STALE_S = 1800               # no pass for this long (3 missed): the Mac's processes are unknown


def is_mirror(d):
    return os.path.isfile(os.path.join(d, MIRROR))


class MirrorLiveness:
    """A mirror's processes are on the Mac: mac_state.json says whether its run / supervisor
    (by pid + creation time) were alive at the sync's last pass. Anything else is unknown."""

    def __init__(self, ms, fresh):
        self.ms, self.fresh = ms or {}, fresh

    def alive(self, pid, started=None, near=None):
        if not self.fresh:
            return None
        for who in ('run', 'supervisor'):
            st = self.ms.get(who) or {}
            if pid and st.get('pid') == pid and (started is None or st.get('started') == started):
                return bool(self.ms.get(f'{who}_alive'))
        return None


class MirrorSource(LocalSource):
    """A desktop mirror of a run on the Mac, kept fresh (~10 s) by stpete-zero's
    tools/mac_sync.py: MAC_SYNC (host, remote dir, last pass), mac_state.json (the Mac run's
    run_state / supervisor_state and whether those processes are alive, `checked`), the run's
    files with the Mac's mtimes, and PAUSE / STOP mirrored both ways - a PAUSE written or
    removed here reaches the Mac within ~5 s, so control is plain local file control.
    The files are pulled every ~10 minutes: liveness counts while the last pass is under
    MIRROR_STALE_S old, and the run is judged 'stale' only MIRROR_LAG_S later than a local one."""
    lag = MIRROR_LAG_S

    def __init__(self, path, now):
        super().__init__(path)
        marker = runfiles.read_json(os.path.join(path, MIRROR))
        self.marker = marker if isinstance(marker, dict) else {}
        ms = runfiles.read_json(os.path.join(path, 'mac_state.json'))
        self.ms = ms if isinstance(ms, dict) else {}
        checked = self.ms.get('checked') or self.ms.get('updated')
        if not isinstance(checked, (int, float)) and self.marker.get('last_pass'):
            checked = runfiles.local_epoch(str(self.marker['last_pass'])[:19])
        self.age = now - checked if isinstance(checked, (int, float)) else None
        self.fresh = self.age is not None and self.age <= MIRROR_STALE_S

    @property
    def remote(self):
        err = None
        if not self.ms:
            err = 'mac_sync has not written mac_state.json'
        elif self.age is None:
            err = 'mac_state.json has no time of the last pass'
        elif not self.fresh:
            err = f'mac_sync last reached the Mac {self.age / 60:.0f} min ago'
        return {'kind': 'mirror', 'host': self.marker.get('host') or self.ms.get('host'),
                'path': self.marker.get('remote') or self.ms.get('remote'), 'label': 'Mac',
                'probe_age_s': round(self.age) if self.age is not None else None, 'error': err}

    def state_file(self, who):
        st = self.ms.get(who)
        return st if isinstance(st, dict) else None

    def liveness(self, default):
        return MirrorLiveness(self.ms, self.fresh)


def pause_kind(p):
    """Who wrote a PAUSE: 'pause-all' (zerodash Pause all; resumed by Resume all), 'zerodash'
    (a dashboard Pause button) or 'other' (by hand, tools/pause_run.py, a game dashboard...)."""
    if not p:
        return None
    by = p.get('by') or ''
    if by.startswith(PAUSE_ALL_PREFIX):
        return 'pause-all'
    if by.startswith('zerodash'):
        return 'zerodash'
    return 'other'


def stale_after(proj_min, seconds):
    """Seconds without output after which a live run is 'stale': the configured minimum, or
    1.5x its slowest recent iteration (a St. Pete self-play phase writes nothing for an hour)."""
    base = float(proj_min) * 60
    if seconds:
        base = max(base, 1.5 * max(seconds))
    return min(base, 12 * 3600)


def _phase(ms, prog, dprog, events):
    """-> (current iteration, phase, since, detail, fraction done) of the iteration in flight."""
    last = (ms or {}).get('last') or {}
    cur = last['iter'] + 1 if last.get('iter') is not None else None
    phase = since = detail = frac = None
    if prog and isinstance(prog.get('iter'), int) and (cur is None or prog['iter'] >= cur):
        cur, phase = prog['iter'], prog.get('phase')
        if prog.get('target'):
            frac = min(1.0, (prog.get('games') or 0) / prog['target'])
            detail = f"{prog.get('games') or 0:,} / {prog['target']:,} games"
    elif dprog and (cur is None or dprog['iter'] >= cur):
        cur = dprog['iter']
        if not dprog['done']:
            phase = 'selfplay'
        if dprog.get('games'):
            frac = min(1.0, dprog['games_done'] / dprog['games'])
            detail = f"{dprog['games_done']:,} / {dprog['games']:,} games"
    if cur is not None:
        mine = [e for e in events if e['iter'] == cur and e['what'] != 'start']
        if phase is None:
            what = mine[-1]['what'] if mine else None
            phase = {'selfplay': 'train', 'random buys': 'train', 'targets': 'train', 'board': 'train',
                     'train': 'eval', 'vs': 'eval', 'elo': 'eval', 'evaluation': 'eval'}.get(what, 'selfplay')
        if mine:
            since = mine[-1]['t']
        elif last.get('time'):
            since = last['time']
        starts = [e for e in events if e['what'] == 'start']
        if starts and starts[-1]['t'] and (since is None or starts[-1]['t'] > since):
            since = starts[-1]['t']
    return cur, phase, since, detail, frac


def _eval(proj, ms):
    evs = (ms or {}).get('evals') or {}
    key = (proj.get('yardstick') or {}).get('key')
    if not key or key not in evs:
        key = next((k for k in YARDSTICK_KEYS if k in evs), None)
    out = None
    if key:
        e = evs[key]
        label = (proj.get('yardstick') or {}).get('label') or e.get('opp') or key[5:]
        out = {'key': key, 'label': label, **e}
    pend = (ms or {}).get('pending')
    if pend:
        out = dict(out or {}, pending=pend)
    return out


def _elo(run_dir, ms):
    r = runfiles.ratings(run_dir)
    if r and r['iters']:
        pts = r['iters']
        cand = [p for p in pts if p[0] not in r.get('unevaluated', ())] or pts
        last, best = pts[-1], max(cand, key=lambda p: p[1])
        return {'iter': last[0], 'elo': round(last[1], 1), 'se': last[2], 'anchor': r.get('anchor'),
                'anchor_elo': r.get('anchor_elo'), 'best': {'iter': best[0], 'elo': round(best[1], 1)},
                'series': [[p[0], round(p[1], 1)] for p in pts[-SPARK_POINTS:]]}
    if ms and ms.get('elo'):
        e = ms['elo']
        return {'iter': e['iter'], 'elo': e['elo'], 'se': e.get('se'), 'anchor': None, 'best': None, 'series': []}
    return None


def _throughput(ms):
    sp = (ms or {}).get('selfplay')
    if not sp:
        return None
    gph = sp.get('games_per_h') or sp.get('games_per_hour')
    if gph is None and sp.get('games') and sp.get('seconds'):
        gph = sp['games'] / sp['seconds'] * 3600
    lps = sp.get('leaves_per_s') or sp.get('positions_per_s')
    return {'iter': sp['iter'], 'games_per_h': gph, 'leaves_per_s': lps}


def summarize(proj, run, src, liveness, now, opts):
    """One run -> the JSON the page shows. `run`: {'id', 'name', 'dir' (files to read),
    'remote' (None or the remote dict)}; `opts`: {'hide_after_days'}."""
    d = run['dir']
    cfg = runfiles.read_json(os.path.join(d, 'config.json')) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    ms = runfiles.metrics_summary(d)
    pause = src.pause()
    rs, ss = src.state_file('run'), src.state_file('supervisor')
    sup = runfiles.supervisor_info(runfiles.supervisor_events(src.supervisor_lines()))
    log_path = os.path.join(d, 'log.txt')
    log_mt = runfiles.mtime(log_path)
    events = runfiles.log_events(runfiles.tail_lines(log_path, 80), log_mt)
    lg = runfiles.log_flags(events)
    prog = src.progress()
    dprog = runfiles.dominion_progress(d, cfg) if not prog else None

    times = [runfiles.mtime(os.path.join(d, f)) for f in ACTIVITY_FILES]
    times += [(ms or {}).get('last', {}).get('time'), (prog or {}).get('time'), (rs or {}).get('updated'),
              (dprog or {}).get('mtime'), (dprog or {}).get('time')]
    times = [t for t in times if isinstance(t, (int, float)) and t > 0 and t <= now + 120]
    activity = max(times) if times else None
    other = [runfiles.mtime(os.path.join(d, f)) for f in ('config.json', 'supervisor.log', 'PAUSE')]
    other += [(pause or {}).get('time'), (ss or {}).get('updated')]
    modified = max([t for t in times + other if isinstance(t, (int, float)) and t <= now + 120] or [0]) or None

    stout = runfiles.tail_lines(os.path.join(d, 'stdout.txt'), 15, 16384)
    sa = stale_after(config.run_option(proj, run['name'], 'stale_after_min', 45), (ms or {}).get('seconds')) + src.lag
    facts = {'pause': pause, 'stop': src.stop(), 'rs': rs, 'ss': ss, 'sup': sup, 'log': lg,
             'traceback': any('Traceback (most recent call last)' in ln for ln in stout),
             'activity': activity, 'stale_after': sa, 'now': now}
    st = state.detect(facts, src.liveness(liveness))

    pause_ok = bool(run.get('pause'))
    cur, phase, since, pdetail, frac = _phase(ms, prog, dprog, events)
    live = st['state'] in ('running', 'stale', 'stopping', 'pausing', 'starting')
    procs_alive = bool(st['run_alive']) or bool(st['supervisor_alive'])
    hidden = run.get('hidden')
    if hidden is None:
        hidden = (not procs_alive and st['state'] not in state.ACTIVE + ('pausing',)
                  and (modified is None or now - modified > float(opts.get('hide_after_days', 3)) * 86400))
    last = (ms or {}).get('last') or {}
    out = {
        'key': f"{proj['id']}/{run['id']}",
        'project': proj['id'], 'id': run['id'], 'name': run['name'],
        'remote': run.get('remote_view'),
        **st,
        'hidden': bool(hidden),
        'modified': modified, 'activity': activity, 'stale_after_s': round(sa),
        'pause_supported': pause_ok,
        'pause': ({'time': pause.get('time'), 'by': pause.get('by'), 'kind': pause_kind(pause)} if pause else None),
        'stop': facts['stop'],
        'can_pause': pause_ok and not pause and st['state'] in state.ACTIVE,
        'can_resume': bool(pause),
        'iteration': {
            'last': last.get('iter'), 'last_time': last.get('time'), 'last_seconds': last.get('seconds'),
            'current': cur, 'phase': phase, 'phase_since': since, 'phase_detail': pdetail,
            'phase_frac': frac, 'live': live, 'paused_in': lg.get('paused'),
        },
        'eval': _eval(proj, ms),
        'elo': _elo(d, ms),
        'throughput': _throughput(ms),
        'config': {k: cfg.get(k) for k in ('preset', 'game', 'sims', 'eval_every', 'iters') if cfg.get(k) is not None},
    }
    return out

