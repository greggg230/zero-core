"""The outside stall alarm (zero/tools/stall_watch.py, docs/az.md 56.4): the threshold from the run's own iteration
times (factor x median, slack x the longest, a floor; a new run's fallback), intervals across a restart left out, the
log's times of day placed across a midnight, no alarm while paused or ended, a dump that names what it lacks."""
import datetime as dt
import json
import os

from zero.tools import stall_watch as SW


def write_run(tmp_path, lines, mtime, files=()):
    run = tmp_path / 'r'
    run.mkdir(exist_ok=True)
    (run / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.utime(run / 'log.txt', (mtime, mtime))
    for f in files:
        (run / f).write_text('{}')
    return str(run)


def ts(h, m, s=0, day=16):
    return dt.datetime(2026, 11, day, h, m, s).timestamp()


def hms(t):
    return dt.datetime.fromtimestamp(t).strftime('%H:%M:%S')


def iters(t0, n, step, first=1, slow=()):
    """`n` iteration ends from t0, `step` seconds apart (an iteration in `slow` takes 2.5x), with a status line."""
    out, t = [], t0
    for k in range(n):
        t += step * (2.5 if first + k in slow else 1)
        out += [f'{hms(t - 30)} iter {first + k} selfplay: {{"games": 1350}}', f'{hms(t)} iter {first + k} train: {{}}']
    return out, t


def test_threshold_from_the_runs_own_iterations_and_the_11_16_hang(tmp_path):
    """pb3's 11-16 hang replayed: iterations of ~330 s (the 16th 2.5x: an evaluation), a restart at 16:37 that never
    finishes an iteration: no alarm before 3 x the median / 1.5 x the longest, an alarm by ~17:00 (it hung 93 min
    unnoticed); the interval across the restart is not an iteration."""
    lines, t = iters(ts(13, 0), 40, 330, first=100, slow=(112, 128))
    lines += [f'{hms(ts(16, 37, 38))} run runs/pb3 starting after iter 139; cfg {{}}',
              f'{hms(ts(16, 37, 42))} remote self-play mac: worker 0 up']
    run = write_run(tmp_path, lines, ts(16, 37, 42))
    j = SW.judge(run, now=ts(16, 50))
    assert not j['stalled'] and j['median_s'] == 330 and j['longest_s'] == 825 and j['intervals'] == 32, j
    assert j['threshold_s'] == max(3 * 330, 1.5 * 825, SW.FLOOR_S) and j['since'][1] == 'start', j
    assert not SW.judge(run, now=ts(16, 37, 38) + j['threshold_s'] - 5)['stalled']
    s = SW.judge(run, now=ts(16, 37, 38) + j['threshold_s'] + 5)
    assert s['stalled'] and 'since the run started' in s['reason'] and s['last'] == 139, s
    # paused, or ended: no alarm
    run = write_run(tmp_path, lines, ts(16, 37, 42), files=('PAUSE',))
    assert not SW.judge(run, now=ts(18, 0))['stalled']
    os.remove(os.path.join(run, 'PAUSE'))
    with open(os.path.join(run, 'progress.json'), 'w') as f:
        json.dump({'phase': 'idle'}, f)
    assert not SW.judge(run, now=ts(18, 0))['stalled']


def test_midnight_restarts_and_a_new_run(tmp_path):
    """Times of day across a midnight are placed on the right day; intervals across a start line are left out; a run
    with fewer than 3 intervals is judged by the fallback (75 min)."""
    lines, t = iters(ts(23, 50), 6, 300, first=1)              # 23:55 ... 00:20 the next day
    run = write_run(tmp_path, lines, t)
    evs = SW.events(SW._tail(os.path.join(run, 'log.txt')), t)
    assert [round(e[0]) for e in evs] == [round(ts(23, 50) + 300 * (k + 1)) for k in range(6)], evs
    assert SW.intervals(evs) == [300.0] * 5
    restart = lines[:4] + [f'{hms(ts(0, 3, day=17))} run r starting after iter 2; cfg {{}}'] + lines[4:]
    run = write_run(tmp_path, restart, t)
    assert SW.intervals(SW.events(SW._tail(os.path.join(run, 'log.txt')), t)) == [300.0] * 4   # (2 -> 3 crosses the start)
    new = write_run(tmp_path, lines[:4], ts(0, 0, day=17))
    j = SW.judge(new, now=ts(0, 0, day=17) + 3000)
    assert j['threshold_s'] == SW.FALLBACK_S and not j['stalled'], j
    assert SW.judge(new, now=ts(0, 0, day=17) + SW.FALLBACK_S + 60)['stalled']


def test_once_exit_code_and_a_dump_without_py_spy(tmp_path, capsys):
    """--once: exit 3 with STALL when stalled (the dump directory named; without py-spy it says what to install),
    0 when not."""
    lines, t = iters(ts(10, 0), 8, 300)
    run = write_run(tmp_path, lines, t)
    os.utime(os.path.join(run, 'log.txt'), (t, t))
    d, notes = SW.dump(run, str(tmp_path / 'dumps'), pyspy='')
    assert os.path.isdir(d) and any('py-spy' in n for n in notes) and any('no live process' in n for n in notes)
    # (the log's times are long past: stalled; the dump goes to tmp_path)
    assert SW.main([run, '--once', '--dump-dir', str(tmp_path / 'dumps')]) == 3
    out = capsys.readouterr().out
    assert out.startswith('STALL ') and 'stack dumps:' in out
    lines2, t2 = iters(dt.datetime.now().timestamp() - 2400, 8, 300)
    run2 = write_run(tmp_path / 'x', lines2, t2) if (tmp_path / 'x').mkdir() is None else None
    assert SW.main([run2, '--once', '--no-dump']) == 0
