"""zero.farm_report (docs/az.md 42.3): the own machines' contributions, from run directories built here
in the real formats (metrics.jsonl: zero.run.Run.finish_iter + zero.remoteplay's records; evals.jsonl:
zero.remote.RemoteEval; Elo pools: zero.elo), a fake ssh runner, a fake HTTP transport. Nothing here
reaches the network, the Mac or a real run."""
import json
import os
import subprocess
import sys
import time

import pytest

from zero import farm_report as fr

NOW = 1_800_000_000.0
H = 3600.0
TOKEN = 'tok-SECRET-1234567890'


def jsonl(path, rows, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def mrow(it, t, games, secs=150.0, sp_s=100.0, remotes=None, remote=None, **extra):
    """A metrics.jsonl row as Run.finish_iter writes it (`remotes`: {name: record} of a list of hosts;
    `remote`: one host's record)."""
    sp = {'games': games, 'samples': games * 10, 'seconds': sp_s}
    if remotes is not None:
        sp['remote'] = {'host': list(remotes), 'games': sum(r['games'] for r in remotes.values())}
        sp['remotes'] = {n: dict(r, host=n) for n, r in remotes.items()}
    if remote is not None:
        sp['remote'] = remote
    m = {'iter': it, 'selfplay': sp, 'train': {'steps': 10}, 'phases': {'selfplay_s': sp_s, 'train_s': secs - sp_s},
         'resumed_seconds': 0, 'seconds': secs, 'time': t}
    m.update(extra)
    return m


def rec(games, running_s, linger=None):
    r = {'games': games, 'games_played': games, 'accepted': games, 'seconds': running_s, 'running_s': running_s,
         'backoff_s': 0.0}
    if linger:
        r['linger'] = linger
    return r


def ey(games):
    return {'win': 0.5, 'se': 0.05, 'margin': 0.0, 'games': games, 'score': games / 2, 'rounds': 20.0, 'unfinished': 0,
            'opp': 'heuristic'}


@pytest.fixture
def farm(tmp_path):
    """Two roots of run directories, a Vast ledger, a config; -> dict of paths."""
    r1 = tmp_path / 'mtg-zero' / 'runs'
    r2 = tmp_path / 'other-zero' / 'runs'
    # alpha: self-play with a list of hosts (mac, vast:line, vast:w1), evaluations on the remote 'mac'
    a = str(r1 / 'alpha')
    jsonl(os.path.join(a, 'metrics.jsonl'), [
        mrow(1, NOW - 30 * H, 100, remotes={'mac': rec(10, 50), 'vast:line': rec(50, 100)}),        # > 24 h
        mrow(2, NOW - 5 * H, 200, secs=1000, sp_s=600,
             remotes={'mac': rec(20, 500, linger={'games': 3, 'seconds': 100}), 'vast:line': rec(300, 590)}),
        mrow(3, NOW - 40 * 60, 150, remotes={'mac': rec(25, 95), 'vast:line': rec(280, 90), 'vast:w1': rec(270, 80)}),
    ])
    jsonl(os.path.join(a, 'evals.jsonl'), [
        {'iter': 2, 'time': NOW - 3 * H, 'host': 'mac', 'seconds': 600.0, 'eval_yard': ey(100),
         'elo': {'elo': 900.0, 'se': 30.0},
         'extras': {'deckchoice': {'games': 64, 'win': 0.4, 'seconds': 100.0},
                    'real_decks': {'n': 128, 'pools': 64, 'games_per_pool': 2, 'a': 0.6}}},
        {'iter': 3, 'time': NOW - 20 * 60, 'host': 'mac', 'seconds': 300.0, 'eval_yard': ey(100)},
    ])
    jdump(os.path.join(a, 'elo', 'cq', 'results.json'), {'anchor': 'heuristic', 'pairs': {
        'heuristic|iter_0002': [50.0, 100], 'iter_0001|iter_0002': [10.0, 20], 'iter_0002|iter_0003': [10.0, 20],
        'iter_0001|iter_0003': [10.0, 20], 'anchor:a1|iter_0003': [5.0, 30], 'iter_0003|random': [9.0, 10],
        'anchor:a1|anchor:a2': [100.0, 200]}})
    # beta: one remote (an old row's host 'user@host' -> 'mac' by the aliases), evaluations here
    b = str(r1 / 'beta')
    beta_rows = [
        mrow(5, NOW - 2 * H, 400, secs=300, sp_s=200,
             remote={'host': 'greggkeithley@100.95.248.116', 'games': 40, 'running_s': 190.0, 'seconds': 195.0},
             eval_yard=ey(50), eval_random=dict(ey(10), opp='random'), extras={'x': {'games': 6, 'win': 1.0}}),
        mrow(6, NOW - 10 * 60, 500, remote={'host': 'greggkeithley@100.95.248.116', 'games': 50, 'running_s': 90.0}),
    ]
    beta_rows[0]['phases'].update(eval_s=40.0, elo_s=10.0)
    jsonl(os.path.join(b, 'metrics.jsonl'), beta_rows)
    jdump(os.path.join(b, 'elo', 'results.json'), {'pairs': {'iter_0004|iter_0005': [8.0, 16],
                                                             'heuristic|iter_0005': [25.0, 50]}})
    time.sleep(0.02)                       # (gamma's directory is created after beta's)
    # gamma: a fork of beta (its rows copied, same iter and time) and one row of its own
    jsonl(str(r1 / 'gamma' / 'metrics.jsonl'), beta_rows + [mrow(7, NOW - 5 * 60, 70)])
    jdump(str(r1 / 'gamma' / 'elo' / 'results.json'), {'pairs': {'iter_0004|iter_0005': [8.0, 16]}})
    # delta: lived on the Vast box w1 from 6 h ago until it came home an hour ago (tools/vast.py)
    d = str(r1 / 'delta')
    jsonl(os.path.join(d, 'metrics.jsonl'), [mrow(10, NOW - 3 * H, 900), mrow(11, NOW - 30 * 60, 800)])
    jdump(os.path.join(d, f'ON_VAST_{int(NOW - H)}.json'), {'box': 'w1', 'since': NOW - 6 * H, 'iter_at_push': 9})
    # notime: rows without `time` (timed from the file's mtime and the next row's seconds)
    jsonl(str(r1 / 'notime' / 'metrics.jsonl'), [{'iter': 1, 'selfplay': {'games': 7}, 'seconds': 600.0},
                                                {'iter': 2, 'selfplay': {'games': 9}, 'seconds': 600.0}],
          mtime=NOW - 15 * 60)
    # cot: a co-training run: resolver/metrics.jsonl (matches mode, one remote)
    jsonl(str(r1 / 'cot' / 'resolver' / 'metrics.jsonl'),
          [mrow(4, NOW - 50 * 60, 12, remote={'host': 'mac', 'games': 4, 'running_s': 100.0})])
    jdump(str(r1 / 'cot' / 'config.json'), {'game': 'x'})
    # oldrun: nothing changed for 47 h - skipped
    jsonl(str(r1 / 'oldrun' / 'metrics.jsonl'), [mrow(1, NOW - 48 * H, 999)], mtime=NOW - 47 * H)
    # not a run: no run file
    os.makedirs(str(r1 / 'scratch'))
    # root 2: another run named alpha (-> alpha@other-zero), remotes gpu:1 (this machine) and the box 'vast'
    jsonl(str(r2 / 'alpha' / 'metrics.jsonl'),
          [mrow(1, NOW - 2 * H, 33, remotes={'gpu:1': rec(11, 90), 'vast': rec(12, 90)})])
    for p in [a, b, str(r1 / 'gamma'), d, str(r1 / 'cot'), str(r2 / 'alpha')]:
        for f in ('metrics.jsonl', 'evals.jsonl', 'resolver/metrics.jsonl'):
            if os.path.exists(os.path.join(p, f)):
                os.utime(os.path.join(p, f), (NOW - 60, NOW - 60))
    ledger = str(tmp_path / 'ledger.json')
    jdump(ledger, {
        'line': {'name': 'line', 'cpu': 'AMD Ryzen 9 7950X3D 16-Core Processor', 'threads': 32.0, 'ram_gb': 93,
                 'geo': 'Texas, US', 'state': 'running', 'deadline': NOW + H},
        'w1': {'name': 'w1', 'cpu': 'AMD Ryzen 9 9900X 12-Core Processor', 'threads': 24.0, 'ram_gb': 60,
               'state': 'destroyed', 'deadline': NOW - 2 * H},
        'idle': {'name': 'idle', 'cpu': 'Some CPU', 'threads': 8.0, 'ram_gb': 32, 'state': 'running',
                 'deadline': NOW + 2 * H}})
    cfg = str(tmp_path / 'farm_report.json')
    jdump(cfg, {'vast_ledger': ledger, 'url': 'https://farm.example',
                'fuzz': {'host': 'mac-test', 'python': '$HOME/py', 'farm': '$HOME/farm'}})
    tok = str(tmp_path / 'admin_token')
    with open(tok, 'w', encoding='utf-8') as f:
        f.write(TOKEN + '\n')
    return {'r1': str(r1), 'r2': str(r2), 'cfg': cfg, 'token': tok, 'log': str(tmp_path / 'report.log'),
            'tmp': tmp_path}


FUZZ_OUT = {'now': NOW, 'state': 'running', 'updated': NOW - 5, 'sha': 'abcdef123456', 'truncated': False,
            'chunks': [[NOW - 30 * 60, 'games', 1000, 600.0], [NOW - 5 * H, 'matches', 50, 1200.0],
                       [NOW - 30 * H, 'games', 5, 10.0]]}


class Ssh:
    """A fake runner: the Mac answering (FUZZ_OUT) or not (offline)."""

    def __init__(self, online=True):
        self.online, self.calls = online, []

    def __call__(self, argv, input=None, timeout=None):
        self.calls.append((argv, input, timeout))
        if not self.online:
            return 255, '', 'ssh: connect to host mac-test port 22: Connection timed out\n'
        return 0, 'some noise\n' + fr.RESULT + json.dumps(FUZZ_OUT) + '\n', ''


class Transport:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, url, data, headers, timeout):
        self.calls.append((url, json.loads(data), dict(headers), timeout))
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a


def by_id(body):
    return {s['id']: s for s in body['sources']}


def report(farm, online=True, roots=None):
    cfg = fr.load_config(farm['cfg'])
    return fr.build_report(cfg, roots or [farm['r1'], farm['r2']], NOW, runner=Ssh(online))


# ------------------------------------------------------------------ the counts

def test_runs_found_once_with_ids(farm):
    runs = fr.find_runs([farm['r1'], farm['r2'], farm['r1']], NOW)        # (a root twice: each run once)
    assert [r.id for r in runs] == ['alpha', 'beta', 'cot', 'delta', 'gamma', 'notime', 'alpha@other-zero']
    assert 'oldrun' not in [r.name for r in runs] and 'scratch' not in [r.name for r in runs]


def test_selfplay_eval_fuzz_per_source(farm):
    body, _ = report(farm)
    assert body['reporter'] == 'desktop' and body['t'] == int(NOW)
    assert [s['id'] for s in body['sources']] == ['desktop', 'mac', 'vast', 'vast:idle', 'vast:line', 'vast:w1']
    s = by_id(body)
    # this machine: alpha 200 + 150, beta 400 + 500 (once, not again for its fork gamma), gamma 70, delta 800 (after
    # it came home), notime 7 + 9, alpha@other-zero 33 + gpu:1's 11, cot's resolver 12
    assert s['desktop']['selfplay_games'] == {'1h': 150 + 500 + 70 + 800 + 16 + 12,
                                              '24h': 200 + 150 + 400 + 500 + 70 + 800 + 16 + 44 + 12}
    assert s['desktop']['eval_games'] == {'1h': 0, '24h': 50 + 10 + 6 + 16}           # beta's, with its Elo match
    assert s['mac']['selfplay_games'] == {'1h': 25 + 50 + 4, '24h': 20 + 25 + 40 + 50 + 4}
    # evals rows: 100 + 64 (real_decks records no games) + Elo 20; 100 + Elo 20 + 20 + 30 (anchor)
    assert s['mac']['eval_games'] == {'1h': 170, '24h': 184 + 170}
    assert s['mac']['fuzz_seeds'] == {'1h': 1000, '24h': 1050}
    assert s['vast:line']['selfplay_games'] == {'1h': 280, '24h': 580}
    assert s['vast:w1']['selfplay_games'] == {'1h': 270, '24h': 270 + 900}                # delta on the box
    assert s['vast']['selfplay_games'] == {'1h': 0, '24h': 12}
    assert s['vast:idle']['selfplay_games'] == {'1h': 0, '24h': 0}
    for v in s.values():
        if v['id'] != 'mac':
            assert v['fuzz_seeds'] == {'1h': 0, '24h': 0}
        assert set(v) >= {'id', 'name', 'kind', 'hardware', 'online', 'working_on', 'selfplay_games', 'eval_games',
                          'fuzz_seeds', 'busy_h_24h'}
        assert 0 <= v['busy_h_24h'] <= 24


def test_working_on_kinds_hardware(farm):
    s = by_id(report(farm)[0])
    assert s['desktop']['working_on'] == ['run:alpha', 'run:alpha@other-zero', 'run:beta', 'run:cot', 'run:delta',
                                          'run:gamma', 'run:notime']
    assert s['mac']['working_on'] == ['run:alpha', 'run:beta', 'run:cot']
    assert s['vast:w1']['working_on'] == ['run:alpha', 'run:delta']
    assert s['vast:line']['working_on'] == ['run:alpha'] and s['vast:idle']['working_on'] == []
    assert [s[k]['kind'] for k in ('desktop', 'mac', 'vast', 'vast:line')] == ['local', 'ssh', 'vast', 'vast']
    assert s['desktop']['hardware'] == 'i9-13900K / RTX 5090' and s['desktop']['online'] is True
    assert s['mac']['hardware'] == 'M4 Max 16-core / 64 GB' and s['mac']['online'] is True
    assert s['mac']['fuzz_state'] == 'running' and 'error' not in s['mac']
    assert s['vast:line']['hardware'] == 'AMD Ryzen 9 7950X3D 16-Core Processor (32 threads) / RTX 5090 / 93 GB'
    assert s['vast:line']['name'] == 'Vast box line (Texas, US)' and s['vast:line']['online'] is True
    assert s['vast:w1']['online'] is False                     # destroyed in the ledger
    assert s['vast']['hardware'] == 'Vast.ai box vast' and s['vast']['online'] is None


def test_busy_hours_union(farm):
    s = by_id(report(farm)[0])
    # vast:line: running_s 590 + 90 (its iteration 1 is 30 h old)
    assert s['vast:line']['busy_h_24h'] == round(680 / 3600, 2)
    # mac: evals 600 + 300; self-play 95 + 190 + 90 + 100 (cot); fuzz chunks 600 + 1200 - and alpha's iteration 2
    # (running 500 + linger 100) lies inside the 1200 s chunk: counted once
    assert s['mac']['busy_h_24h'] == round((600 + 300 + 95 + 190 + 90 + 100 + 600 + 1200) / 3600, 2)


def test_mac_offline_is_not_a_failure(farm):
    body, _ = report(farm, online=False)
    mac = by_id(body)['mac']
    assert mac['online'] is False and 'unreachable' in mac['error'] and 'Connection timed out' in mac['error']
    assert mac['fuzz_seeds'] == {'1h': None, '24h': None}          # unknown, not 0
    assert mac['selfplay_games']['24h'] == 139                      # the runs' records still count


def test_fuzz_ssh_command(farm):
    ssh = Ssh()
    fr.build_report(fr.load_config(farm['cfg']), [farm['r1']], NOW, runner=ssh)
    (argv, inp, timeout), = ssh.calls
    assert argv == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'mac-test',
                    '"$HOME/py" - "$HOME/farm" 3600 86400']
    assert inp == fr.FUZZ_SCRIPT and timeout == 90


def test_fuzz_script_parses_the_farm_log(tmp_path):
    """FUZZ_SCRIPT itself (run here, as the Mac would): farm.log's finished chunks, status.json."""
    now = time.time()
    farm = tmp_path / 'farm'

    def ts(t):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    lines = [
        f'{ts(now - 30 * H)} games [1+1000]: rc 0, 1000 games in 10 s; old',
        f'{ts(now - 25 * H)} smoke-random [3483+2]: rc 1, 2 seeds in 866 s',
        f'{ts(now - 5 * H)} smoke-match [1761+1]: rc 0, 1 seeds in 588 s',
        f'{ts(now - 30 * 60)} games [100+50]: rc 0, 50 games in 300 s; 50 games: wins [20, 20], draws 10',
        f'{ts(now - 30 * 60)} exponential-games [10+15] (bisect): no exit after 133 s running: killing (a hang)',
        f'{ts(now - 30 * 60)} exponential-games [10+15] (bisect) hung (133 s running): bisecting',
        f'{ts(now - 30 * 60)} exponential-games [10+15] (bisect): rc -15, 0 games in 133 s',
        f'{ts(now - 10 * 60)} draft-events [21492417+17] (guaranteed): rc 0, 17 events in 1 s; 17 events, 8398 actions',
        f'{ts(now - 5 * 60)} round 12 done',
    ]
    os.makedirs(farm / 'logs')
    (farm / 'logs' / 'farm.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    jdump(str(farm / 'status.json'), {'state': 'backing off: eval lock', 'updated': now, 'sha': 'abcdef1234567890'})

    def local(argv, input=None, timeout=None):
        return fr.run_cmd([sys.executable, '-', str(farm), '3600', '86400'], input=input, timeout=timeout)
    got = fr.fuzz_probe({'host': 'mac-test', 'python': 'py', 'farm': str(farm)}, local)
    assert got['online'] and 'error' not in got and got['state'] == 'backing off: eval lock'
    assert [(c[1], c[2], c[3]) for c in got['chunks']] == [
        ('smoke-random', 2, 866.0), ('smoke-match', 1, 588.0), ('games', 50, 300.0), ('exponential-games', 0, 133.0),
        ('draft-events', 17, 1.0)]
    assert abs(got['chunks'][2][0] - (now - 30 * 60)) <= 1.0          # (the Mac's local time, read back)
    cfg = dict(fr.DEFAULTS)
    rep = fr.Report(cfg, now)
    rep.add_fuzz('mac', got)
    assert rep.acc['mac']['fuzz'] == {'1h': 50 + 17, '24h': 50 + 17 + 1}
    # no farm log: the source is online, the error says why
    os.remove(farm / 'logs' / 'farm.log')
    got = fr.fuzz_probe({'host': 'mac-test', 'python': 'py', 'farm': str(farm)}, local)
    assert got['online'] and got['chunks'] == [] and 'farm.log' in got['error']


def test_elo_games_by_iter(tmp_path):
    jdump(str(tmp_path / 'elo' / 'results.json'), {'pairs': {'iter_0002|iter_0010': [1, 20], 'heuristic|iter_0010': [1, 100],
                                                             'anchor:x|iter_0010': [1, 40], 'iter_0010|random': [1, 10]}})
    jdump(str(tmp_path / 'elo' / 'cq' / 'results.json'), {'pairs': {'iter_0008|iter_0010': [1, 16],
                                                                    'iter_0010|iter_0012': [1, 16]}})
    assert fr.elo_games_by_iter(str(tmp_path)) == {10: 20 + 40 + 16, 12: 16}


# ------------------------------------------------------------------ main: dry run, POST, token

def test_dry_run_prints_the_report(farm, capsys):
    t = Transport([])
    rc = fr.main(['--dry-run', '--config', farm['cfg'], '--roots', farm['r1'], farm['r2'], '--log', farm['log']],
                 now=NOW, runner=Ssh(), transport=t, environ={})
    out = capsys.readouterr().out
    assert rc == 0 and t.calls == [] and not os.path.exists(farm['log'])
    assert json.loads(out) == report(farm)[0]


def test_post_body_bearer_and_no_token_leak(farm, capsys):
    t = Transport([(200, '{"ok": true}')])
    rc = fr.main(['--config', farm['cfg'], '--roots', farm['r1'], farm['r2'], '--token-file', farm['token'],
                  '--log', farm['log']], now=NOW, runner=Ssh(), transport=t, environ={})
    cap = capsys.readouterr()
    assert rc == 0
    (url, body, headers, timeout), = t.calls
    assert url == 'https://farm.example/v1/admin/external'
    assert headers['Authorization'] == f'Bearer {TOKEN}' and headers['Content-Type'] == 'application/json'
    assert set(body) == {'reporter', 't', 'sources'} and body == report(farm)[0]
    for s in body['sources']:
        assert set(s['selfplay_games']) == set(s['eval_games']) == set(s['fuzz_seeds']) == {'1h', '24h'}
        assert all(w.startswith('run:') for w in s['working_on'])
    with open(farm['log'], encoding='utf-8') as f:
        log = f.read()
    assert 'posted 6 sources' in cap.out and 'posted 6 sources' in log
    assert TOKEN not in cap.out + cap.err + log


def test_env_token_first_and_url_flag(farm):
    t = Transport([(204, '')])
    rc = fr.main(['--config', farm['cfg'], '--roots', farm['r1'], '--token-file', farm['token'], '--no-fuzz',
                  '--url', 'http://other.example/', '--log', ''], now=NOW, runner=Ssh(), transport=t,
                 environ={fr.TOKEN_ENV: 'env-token'})
    assert rc == 0 and t.calls[0][0] == 'http://other.example/v1/admin/external'
    assert t.calls[0][2]['Authorization'] == 'Bearer env-token'


def test_no_token_exits_2(farm, capsys):
    rc = fr.main(['--config', farm['cfg'], '--roots', farm['r1'], '--token-file', str(farm['tmp'] / 'none'),
                  '--log', ''], now=NOW, runner=Ssh(), transport=Transport([]), environ={})
    assert rc == 2 and 'no admin token' in capsys.readouterr().err


def test_post_failure_exits_1_without_the_token(farm, capsys):
    t = Transport([(500, f'boom {TOKEN}')] * 4)
    slept = []
    rc = fr.main(['--config', farm['cfg'], '--roots', farm['r1'], '--token-file', farm['token'], '--log', farm['log']],
                 now=NOW, runner=Ssh(online=False), transport=t, sleep=slept.append, environ={})
    cap = capsys.readouterr()
    assert rc == 1 and len(t.calls) == 4 and slept == [2, 5, 10]
    assert 'FAILED' in cap.err and '(offline: mac' in cap.err
    with open(farm['log'], encoding='utf-8') as f:
        log = f.read()
    assert TOKEN not in cap.out + cap.err + log and '***' in log


def test_retries_5xx_and_network_not_4xx():
    slept = []
    t = Transport([(503, 'busy'), ConnectionResetError('reset'), (200, 'ok')])
    ok, msg = fr.post('https://x/v1/admin/external', {'a': 1}, TOKEN, transport=t, sleep=slept.append)
    assert ok and len(t.calls) == 3 and slept == [2, 5] and 'after 2 retries' in msg
    slept = []
    t = Transport([(401, 'unauthorized')])
    ok, msg = fr.post('https://x/v1/admin/external', {'a': 1}, TOKEN, transport=t, sleep=slept.append)
    assert not ok and len(t.calls) == 1 and slept == [] and msg == 'HTTP 401: unauthorized'
    t = Transport([OSError('down')] * 4)
    ok, msg = fr.post('https://x/v1/admin/external', {'a': 1}, TOKEN, transport=t, sleep=lambda s: None)
    assert not ok and len(t.calls) == 4 and 'down' in msg


def test_light_import():
    """zero.farm_report imports neither torch nor numpy (the task runs it alone)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(fr.__file__)))
    code = 'import sys, zero.farm_report; print(sorted(m for m in ("torch", "numpy") if m in sys.modules))'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=root,
                         env=dict(os.environ, PYTHONPATH=root), timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == '[]', out.stderr


# ------------------------------------------------------------------ the scheduled task (mocked)

def test_install_command_exact():
    argv = fr.task_install_argv('C:\\Py\\pythonw.exe', 'C:\\w t\\python', ['--config', 'C:\\c f\\x.json'])
    assert argv == ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command',
                    "$ErrorActionPreference = 'Stop'; "
                    "$a = New-ScheduledTaskAction -Execute 'C:\\Py\\pythonw.exe' "
                    "-Argument '-m zero.farm_report --config \"C:\\c f\\x.json\"' -WorkingDirectory 'C:\\w t\\python'; "
                    "$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5); "
                    "$p = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent()"
                    ".Name) -LogonType Interactive -RunLevel Limited; "
                    "$s = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan "
                    "-Minutes 4) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries; "
                    "Register-ScheduledTask -TaskName 'ZeroFarmReport' -Description 'zero.farm_report: the own "
                    "machines'' contributions to zero-farm (zero-core docs/az.md 42.3)' -Action $a -Trigger $t "
                    "-Principal $p -Settings $s -Force | Out-Null; Write-Output ('registered ' + 'ZeroFarmReport')"]
    assert 'SYSTEM' not in argv[-1]
    assert fr.task_uninstall_argv() == [
        'powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command',
        "$ErrorActionPreference = 'Stop'; Unregister-ScheduledTask -TaskName 'ZeroFarmReport' -Confirm:$false; "
        "Write-Output ('removed ' + 'ZeroFarmReport')"]


def test_install_and_uninstall_task_mocked(tmp_path, capsys):
    calls = []

    def runner(argv, input=None, timeout=None):
        calls.append(argv)
        return 0, 'registered ZeroFarmReport\n', ''
    cfg = str(tmp_path / 'c.json')
    rc = fr.main(['--install-task', '--config', cfg, '--no-fuzz'], runner=runner)
    assert rc == 0 and 'registered' in capsys.readouterr().out
    assert calls == [fr.task_install_argv(fr.pythonw_path(), fr.PYTHON_DIR, ['--config', cfg, '--no-fuzz'])]
    assert f"-WorkingDirectory '{fr.PYTHON_DIR}'" in calls[0][-1]
    assert os.path.basename(fr.pythonw_path()).lower() in ('pythonw.exe', os.path.basename(sys.executable).lower())
    calls.clear()
    assert fr.main(['--uninstall-task'], runner=runner) == 0 and calls == [fr.task_uninstall_argv()]
    calls.clear()
    assert fr.main(['--install-task', '--task-root', str(tmp_path)], runner=runner) == 2 and calls == []
