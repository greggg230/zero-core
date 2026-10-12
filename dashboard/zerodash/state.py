"""One state per run, from its files and the liveness of the processes they name.

Glass Road, Caverna and Dominion runs (runctl.py) write run_state.json and
supervisor_state.json: {pid, started (process creation time), state, since, ...}; the
processes are checked by pid + creation time and the rules follow runctl.run_status:

  run process alive      PAUSE -> pausing, STOP -> stopping, no output for too long ->
                         stale, else running
  no run process, PAUSE  paused (whether the supervisor waits to relaunch it is in the detail)
  supervisor alive only  its state 'restarting' (or a crash as its last log event) ->
                         restarting (a crash loop when it crashed 3+ times in the hour),
                         'paused' / 'resuming' -> resuming, else starting (stale when that
                         lasts too long)
  nothing alive          clean end (run_state done, 'supervisor done', log 'done' / STOP
                         found) -> stopped; run_state aborted / still 'running', log 'run
                         aborted', a traceback in stdout, or a supervisor that died mid-run /
                         mid-restart -> crashed; else stopped

St. Pete runs have no state files (yet): the supervisor's and the run's pids come from
supervisor.log ('supervisor N starting', 'started stpz.run pid N', '... exited with code C'),
each checked against the time of its log line; then the same rules apply.

When liveness is unknown (a remote host that did not answer, a mirror whose sync stalled) the
run counts as running while its output is fresh, else 'unknown'; a local run without
supervisor.log or state files (started by hand) the same, but 'stopped' once its output is old.
"""

ACTIVE = ('running', 'stale', 'starting', 'restarting', 'resuming', 'stopping')

LABELS = {
    'running': 'Running', 'pausing': 'Pausing', 'paused': 'Paused', 'resuming': 'Resuming',
    'starting': 'Starting', 'restarting': 'Crashed - restarting', 'stopping': 'Stopping',
    'stopped': 'Stopped', 'crashed': 'Crashed', 'stale': 'Stale', 'unknown': 'Unknown',
}
SEVERITY = {
    'running': 'good', 'pausing': 'warn', 'paused': 'info', 'resuming': 'warn', 'starting': 'warn',
    'restarting': 'serious', 'stopping': 'warn', 'stopped': 'muted', 'crashed': 'critical',
    'stale': 'serious', 'unknown': 'muted',
}


def _ago(s):
    if s is None:
        return '?'
    s = max(0, s)
    return f'{s:.0f} s' if s < 90 else f'{s / 60:.0f} min' if s < 5400 else f'{s / 3600:.1f} h'


def processes(f, liveness):
    """-> (run_alive, supervisor_alive, source): True / False / None (unknown)."""
    rs, ss, sup = f.get('rs'), f.get('ss'), f.get('sup') or {}
    if rs or ss:
        r = liveness.alive(rs['pid'], started=rs.get('started')) if rs and rs.get('pid') else False
        if r and rs.get('state') in ('paused', 'done', 'aborted'):
            r = False                     # wrote its last state and is exiting (runctl does the same)
        s = liveness.alive(ss['pid'], started=ss.get('started')) if ss and ss.get('pid') else False
        if not r and s and ss.get('state') == 'running' and ss.get('run_pid') and \
                ss.get('run_pid') != (rs or {}).get('pid'):
            r = liveness.alive(ss['run_pid'])     # launched, not yet written its run_state.json
        return r, s, 'state files'
    if sup.get('sup_pid') or sup.get('run_pid'):
        s = liveness.alive(sup['sup_pid'], near=sup.get('sup_t')) if sup.get('sup_pid') else False
        r = liveness.alive(sup['run_pid'], near=sup.get('run_t')) if sup.get('run_pid') else False
        if s is None and r is None:
            return None, None, 'supervisor.log'
        return r, s, 'supervisor.log'
    return None, None, 'none'


def detect(f, liveness):
    """f: the run's facts (summary.facts) -> {'state', 'label', 'severity', 'detail',
    'run_alive', 'supervisor_alive', 'source'}."""
    r, s, source = processes(f, liveness)
    pause, stop = f.get('pause'), bool(f.get('stop'))
    rs, ss, sup, lg = f.get('rs') or {}, f.get('ss') or {}, f.get('sup') or {}, f.get('log') or {}
    now, act, stale_after = f['now'], f.get('activity'), f['stale_after']
    quiet = now - act if act else None
    fresh = quiet is not None and quiet <= stale_after
    last_sup = (sup.get('last') or {}).get('kind')
    crashes = sup.get('crashes_1h') or 0
    clean = rs.get('state') == 'done' or last_sup == 'done' or lg.get('done') or lg.get('stop_found')
    detail = ''

    if r is None and s is None:
        if pause:
            st, detail = 'paused', 'PAUSE file present; whether its processes run is not known here.'
        elif clean:
            st, detail = 'stopped', 'Finished, or ended by STOP.'
        elif lg.get('aborted'):
            st, detail = 'crashed', 'Aborted (see the log).'
        elif lg.get('paused'):
            st, detail = 'paused', 'The log ends paused but the PAUSE file is gone.'
        elif fresh:
            st, detail = 'running', f'Process state unknown ({source}); judged by its output {_ago(quiet)} ago.'
        elif source == 'none':
            st = 'stopped'
            detail = (f'Nothing written for {_ago(quiet)}; no supervisor.log or state files to check '
                      'its processes.' if quiet is not None else 'No supervisor.log or state files.')
        else:
            st = 'unknown'
            detail = (f'Process state unknown ({source}); no output for {_ago(quiet)}.' if quiet is not None
                      else f'Process state unknown ({source}).')
    elif r:
        if pause:
            st, detail = 'pausing', 'Stopping at the next pause point.'
        elif stop:
            st, detail = 'stopping', 'STOP file: the run ends after the current iteration.'
        elif quiet is not None and not fresh:
            st, detail = 'stale', f'The run process is alive but has written nothing for {_ago(quiet)}.'
        else:
            st = 'running'
    elif pause:
        st = 'paused'
        if s:
            detail = 'Its supervisor waits: Resume relaunches the run.'
        elif s is None:
            detail = 'Whether a supervisor waits to relaunch it is not known here.'
        else:
            detail = 'No supervisor is running: resuming needs the launch command.'
    elif s:
        sst = ss.get('state')
        if sst == 'restarting' or last_sup == 'crash':
            st = 'restarting'
            detail = (f'Restart loop: {crashes} crashes in the last hour; the supervisor keeps restarting it.'
                      if crashes >= 3 else 'The run crashed; the supervisor restarts it after a short wait.')
        elif sst in ('paused', 'resuming') or last_sup == 'resumed':
            st, detail = 'resuming', 'PAUSE removed: the supervisor is relaunching the run.'
        elif r is None:
            st = 'running' if fresh or quiet is None else 'stale'
            detail = '' if st == 'running' else f'The supervisor is alive but nothing was written for {_ago(quiet)}.'
        else:
            since = ss.get('since') or sup.get('sup_t')
            long = since is not None and now - since > stale_after and not fresh
            st = 'stale' if long else 'starting'
            detail = (f'The supervisor is alive but its run process is not, and nothing was written for {_ago(quiet)}.'
                      if long else 'The supervisor is launching the run.')
    else:
        rst = rs.get('state')
        if clean:
            st = 'stopped'
            detail = 'Ended by STOP.' if (stop or lg.get('stop_found')) else 'Finished.'
        elif rst == 'aborted' or lg.get('aborted'):
            st, detail = 'crashed', 'Aborted (see the log).'
        elif f.get('traceback') or rst == 'running' or last_sup in ('crash', 'run_start'):
            st = 'crashed'
            detail = ('The run process died and no supervisor is running to restart it.'
                      if last_sup != 'crash' else 'The run crashed and its supervisor is gone.')
        elif rst == 'paused':
            st, detail = 'stopped', 'Paused, and the PAUSE file is gone, but no supervisor is running: relaunch it.'
        else:
            st, detail = 'stopped', 'No run process (finished, killed or crashed).'
    if st in ('running', 'starting', 'restarting') and stop:
        detail = (detail + ' ' if detail else '') + 'STOP file present.'
    return {'state': st, 'label': ('Restart loop' if st == 'restarting' and crashes >= 3 else LABELS[st]),
            'severity': SEVERITY[st], 'detail': detail, 'run_alive': r, 'supervisor_alive': s,
            'source': source, 'crashes_1h': crashes}
