"""The shared work queue's model (zero/workqueue.py, docs/az.md 5.3): each consumer's in-flight cap
is its throughput's share of an iteration's units, a batch-bound consumer's its slots while its games
take at most `batch_inflight` iterations (simulated phases of synthetic consumers whose rate grows
with their in-flight: delver1's desktop); per-game sample splitting (a game beyond a quota waits
with its samples); the staleness histogram; the co-training coordinator keeping the local driver's
share in reserve. Units: no driver except `test_split_games_on_the_driver`."""
import json
import math
import time

import numpy as np
import pytest

from zero.workqueue import Shares, hist_summary, lag_hist, merge_hist, split_games


def test_caps_split_the_units_by_throughput():
    """cap = ceil(inflight x units x rate / total rate), within [lo, hi]; the caps sum to inflight x
    units when no bound binds."""
    sh = Shares(1.0)
    sh.consumer('local', 2, 192)
    sh.consumer('w0', 4, 768)
    sh.consumer('w1', 4, 768)
    sh.units = 600
    # unmeasured: proportional to the slots
    caps = sh.caps()
    assert caps == {'local': math.ceil(600 * 192 / 1728), 'w0': math.ceil(600 * 768 / 1728),
                    'w1': math.ceil(600 * 768 / 1728)}
    # measured: the desktop at 0.26 games/s, each Mac worker at 1.35
    assert sh.observe('local', 260, 1000.0, 1.0) == pytest.approx(0.26)
    sh.observe('w0', 1350, 1000.0, 1.0)
    sh.observe('w1', 1350, 1000.0, 1.0)
    caps = sh.caps()
    tot = 0.26 + 2 * 1.35
    assert caps['local'] == math.ceil(600 * 0.26 / tot) and caps['w0'] == caps['w1'] == math.ceil(600 * 1.35 / tot)
    assert abs(sum(caps.values()) - 600) <= 3
    # inflight 2: twice as much in flight (two iterations' worth)
    sh.f = 2.0
    assert sh.caps()['w0'] == math.ceil(1200 * 1.35 / tot)
    # bounds: hi = the slots, lo = a few games per group
    sh.f = 1.0
    sh.units = 100_000
    assert sh.caps() == {'local': 192, 'w0': 768, 'w1': 768}
    sh.units = 3
    assert sh.caps() == {'local': 2, 'w0': 4, 'w1': 4}
    sh.units = None                                # unknown units (co-training before any event): the slots
    assert sh.caps() == {'local': 192, 'w0': 768, 'w1': 768}


def test_a_consumer_that_stops_taking_work_gives_its_share_to_the_others():
    """A remote backing off or offline drops out of the total: its cap is 0 and the others' shares
    grow at once (the desktop takes the whole quota when it is alone)."""
    sh = Shares(1.0)
    sh.consumer('local', 2, 192)
    sh.consumer('w0', 4, 768)
    sh.units = 600
    sh.observe('local', 100, 100.0, 1.0)
    sh.observe('w0', 300, 100.0, 1.0)
    assert sh.caps() == {'local': 150, 'w0': 450}
    sh.set_active('w0', False)
    assert sh.caps() == {'local': 192, 'w0': 0}
    sh.set_active('w0', True)
    assert sh.caps() == {'local': 150, 'w0': 450}


def test_rates_ewma_utilisation_and_restore():
    """A phase's rate joins the EWMA (half weight); a consumer that had less work than its cap counts
    as if it had kept it busy at the same latency (at most 4x); too short a phase measures nothing;
    a restart's rates come back from the metrics row's state."""
    sh = Shares(1.0)
    sh.consumer('local', 2, 100)
    assert sh.observe('local', 100, 10.0, 1.0) == pytest.approx(10.0)
    sh.observe('local', 200, 10.0, 1.0)
    assert sh.c['local']['rate'] == pytest.approx(15.0)
    assert sh.observe('local', 50, 10.0, 0.5) == pytest.approx(10.0)   # half busy: 5/s measured, 10/s at its cap
    assert sh.observe('local', 50, 10.0, 0.01) == pytest.approx(20.0)  # at most x4
    assert sh.observe('local', 5, 0.1, 1.0) is None and sh.observe('local', 0, 100.0, 1.0) is None
    st = json.loads(json.dumps(sh.state()))
    sh2 = Shares(1.0)
    sh2.restore(st)
    sh2.consumer('local', 2, 100)
    sh2.consumer('w0', 4, 400)                     # a new worker: the measured consumers' rate per slot
    r = sh2.rates()
    assert r['local'] == pytest.approx(st['local']['rate'], rel=1e-5)
    assert r['w0'] == pytest.approx(st['local']['rate'] / 100 * 400, rel=1e-5)


# ---------------------------------------------------------------- batch-bound consumers

LPU = 6587.0                       # leaves per game (delver1's iteration 48)


def simulate(consumers, units=800, phases=10, inflight=1.0, batch_inflight=2.0, off=None):
    """Phases of synthetic consumers {name: (lo, hi, batch, latency(n), rate0)}: each keeps its cap
    full (single-game: it starts its own games) and finishes them at the latency its curve gives for
    that in-flight (rate = n / latency(n), Little's law); a phase lasts until the units are done;
    each consumer is measured as zero.remoteplay measures it (its leaves per running second over the
    leaves per unit). `rate0`: its rate before the first phase (a restart's restored state; None:
    unmeasured). `off`: {phase: name} backing off that phase. -> [{caps, rates, secs, lat}] per
    phase (`lat`: each consumer's game latency in phases of the joint phase)."""
    sh = Shares(inflight, batch_inflight)
    for n, (lo, hi, batch, lat, r0) in consumers.items():
        sh.consumer(n, lo, hi, cautious=n != 'local', batch=batch)
        sh.c[n]['rate'] = r0
    sh.units = units
    sh.lpu = LPU
    out = []
    for ph in range(phases):
        for n in consumers:
            sh.set_active(n, not (off and off.get(ph) == n))
        caps = sh.caps()
        rate = {n: (caps[n] / consumers[n][3](caps[n]) if caps[n] else 0.0) for n in consumers}
        secs = units / sum(rate.values())
        sh.observe_phase({n: {'units': rate[n] * secs, 'run_s': secs, 'util': 1.0, 'leaves': rate[n] * secs * LPU,
                              'lead': 0.0} for n in consumers if caps[n]})
        out.append({'caps': caps, 'rates': rate, 'secs': secs,
                    'lat': {n: consumers[n][3](caps[n]) / secs for n in consumers if caps[n]}})
    return out


# delver1 on zero-core 213a9c0 (its metrics.jsonl, iterations 19-48): the desktop's leaves per second
# grow with its games in flight (2 groups, batches of in-flight / 2: ~35 leaves/s per game at 75-192
# in flight, 6.1-6.8k at all 192), so its games take ~180-200 s whatever its in-flight - CUDA,
# batch-bound; a Mac worker's rate is its cores' (CPU: ~1.85 games/s at 340 or 768 in flight)
DESKTOP = lambda n: 180.0 + 0.1 * n                     # noqa: E731
MACW = lambda n: 5.0 + n / 1.85                         # noqa: E731


def delver1(batch, cap0):
    """The desktop (batch-bound or not) and the Mac's two workers, from the desktop at `cap0` in flight."""
    return {'local': (4, 192, batch, DESKTOP, cap0 / DESKTOP(cap0)), 'w0': (4, 768, False, MACW, 1.85),
            'w1': (4, 768, False, MACW, 1.85)}


def test_a_batch_bound_consumer_is_not_starved_by_its_own_rate():
    """The spiral of zero-core 213a9c0: the desktop's cap is its throughput's share, and a smaller cap
    measures a smaller rate - its games no shorter - so its cap only settles where its games last one
    phase: ~60% of its slots here whatever it starts from (delver1: 66-131 of 192 at 2.0-5.3k
    leaves/s, the point moving with its games' length). Batch-bound, its cap is its slots one phase
    from the spiral's state and stays there, its games within `batch_inflight` (2) phases (here 1.14),
    its throughput 1.6x; the CPU workers keep games one phase long, as before."""
    for cap0 in (75, 181):
        old = simulate(delver1(False, cap0), phases=30)
        assert all(r['caps']['local'] <= 0.65 * 192 for r in old[-5:]), [r['caps']['local'] for r in old]
        new = simulate(delver1(True, cap0), phases=30)
        assert [r['caps']['local'] for r in new[1:]] == [192] * 29, [r['caps']['local'] for r in new]
        last, prev = new[-1], old[-1]
        assert last['lat']['local'] < 1.3 and abs(prev['lat']['local'] - 1.0) < 0.05, (last, prev)
        assert last['rates']['local'] > 1.5 * prev['rates']['local'], (last['rates'], prev['rates'])
        assert last['secs'] < 0.95 * prev['secs'], 'the phase is shorter with the desktop at its slots'
        for w in ('w0', 'w1'):                 # the Mac's workers: one phase of their throughput, as before
            assert abs(last['lat'][w] - 1.0) < 0.05 and abs(prev['lat'][w] - 1.0) < 0.05, (last, prev)


def test_a_batch_bound_consumer_too_slow_for_the_iterations_falls_to_its_floor():
    """A batch-bound worker whose games take ~1,400 s at any in-flight, in phases of ~200 s (more than
    `batch_inflight` phases): its slots would only fill the data with stale rows, so its cap falls to
    its floor, as a CPU consumer's would; with batch_inflight 1 (or 'none') nothing is batch-bound."""
    slow = lambda n: 1400.0 + n / 100                  # noqa: E731
    run = simulate({'local': (4, 192, True, DESKTOP, 0.5), 'w0': (8, 768, True, slow, 0.2),
                    'w1': (4, 768, False, MACW, 1.85)}, phases=10)
    assert run[-1]['caps']['w0'] == 8 and run[-1]['caps']['local'] == 192, [r['caps'] for r in run]
    same = simulate(delver1(True, 100), phases=6, batch_inflight=1.0)
    assert [r['caps'] for r in same] == [r['caps'] for r in simulate(delver1(False, 100), phases=6)]


def test_batch_cap_formula_bounds_and_cpu_consumers_unchanged():
    """cap = max(inflight x share, min(slots, batch_inflight x share)) for a batch-bound consumer,
    within [lo, hi]; `slots` (co-training: the driver's slots below hi's queue) bounds the batch rule
    only; a consumer that is not batch-bound gets exactly the throughput share."""
    sh = Shares(1.0, 2.0)
    sh.consumer('local', 2, 40, batch=True, slots=32)   # co-training: 32 slots + a queue of 8
    sh.consumer('w0', 4, 768)
    sh.units = 60
    sh.observe('local', 100, 100.0, 1.0)                # 1 unit/s
    sh.observe('w0', 200, 100.0, 1.0)                   # 2 units/s: P = 20 s
    assert sh.caps() == {'local': 32, 'w0': 40}         # its share 20, 2 x 20 = 40: its slots
    sh.units = 30                                       # P = 10 s: a share of 10, 2 x 10 = 20 under the slots
    assert sh.caps() == {'local': 20, 'w0': 20}
    sh.set_active('w0', False)                          # alone: its share is the 30 units; its slots win
    assert sh.caps() == {'local': 32, 'w0': 0}
    sh.units = 300                                      # alone: a share of 300 is above its slots: up to hi
    assert sh.caps()['local'] == 40
    sh.set_active('w0', True)
    sh.c['local']['batch'] = False                      # not batch-bound: the share, as before the batch rule
    assert sh.caps() == {'local': 40, 'w0': 200}
    sh.units = 30
    assert sh.caps() == {'local': 10, 'w0': 20}
    # an unmeasured batch-bound remote: a quarter of what the batch rule gives it until it is measured
    sh2 = Shares(1.0, 2.0)
    sh2.consumer('local', 2, 192)
    sh2.consumer('w0', 4, 768, cautious=True, batch=True)
    sh2.units = 600
    sh2.observe('local', 100, 100.0, 1.0)               # 1/s: the rate per slot 1/192, w0's guess 4/s
    assert sh2.caps()['w0'] == math.ceil(0.25 * min(768, 2 * 4.0 * 600 / 5.0))
    # the state names the batch-bound consumers and their slots (a metrics row's caps can be recomputed)
    st = json.loads(json.dumps(sh.state()))
    assert st['w0'] == {'rate': 2.0, 'lead': 0.0, 'lo': 4, 'hi': 768}
    sh.c['local']['batch'] = True
    assert json.loads(json.dumps(sh.state()))['local']['batch'] is True
    assert json.loads(json.dumps(sh.state()))['local']['slots'] == 32


def test_a_backoff_gives_a_batch_bound_desktop_the_quota_at_once():
    """A Mac worker backing off for a phase: its cap is 0 and the desktop takes the rest at once (here
    its slots, as before); the next phase the shares come back."""
    run = simulate(delver1(True, 120), phases=5, off={3: 'w0'})
    assert run[3]['caps']['w0'] == 0 and run[3]['caps']['local'] == 192 and run[3]['caps']['w1'] > run[2]['caps']['w1']
    assert run[4]['caps']['w0'] > 0


def _samples(ks, rng):
    """A driver-like sample dict of games with ks samples each (options 2-4 per sample)."""
    n = sum(ks)
    nopt = rng.integers(2, 5, n)
    off = np.concatenate([[0], np.cumsum(nopt)]).astype(np.int64)
    game = np.repeat(np.arange(len(ks)), ks)
    return {'x': np.repeat(game[:, None], 3, 1).astype(np.float32), 'player': np.zeros(n, np.int32),
            'n_players': np.full(n, 2, np.int32), 'opt_off': off, 'opt_action': np.repeat(game, nopt).astype(np.int32),
            'pi': np.ones(off[-1], np.float32), 'z': np.zeros((n, 8), np.float32), 'q': np.zeros((n, 8), np.float32),
            'sc': np.zeros((n, 8), np.float32), 'net': game.astype(np.uint32)}


def test_split_games_cuts_a_sample_dict_into_its_games():
    """Each row's `samples` consecutive samples are its game's (opt_off rebased, options with them);
    a game without samples (cut at max_steps) gets None; a mismatch is an error."""
    from zero.protocol import concat_samples
    rng = np.random.default_rng(0)
    ks = [3, 0, 5, 1]
    s = _samples(ks, rng)
    rows = [{'gid': g, 'samples': k} for g, k in enumerate(ks)]
    parts = split_games(rows, s)
    assert [r['gid'] for r, _ in parts] == [0, 1, 2, 3] and parts[1][1] is None
    for g, (r, p) in enumerate(parts):
        if p is None:
            continue
        assert len(p['player']) == ks[g] and (p['x'][:, 0] == g).all() and (p['opt_action'] == g).all()
        assert p['opt_off'][0] == 0 and p['opt_off'][-1] == len(p['pi']) == len(p['opt_action'])
    back = concat_samples([p for _, p in parts])
    assert all(np.array_equal(back[k], s[k]) for k in s)
    with pytest.raises(RuntimeError, match='left over'):
        split_games(rows[:3], s)
    with pytest.raises(RuntimeError, match='samples'):
        split_games(rows + [{'gid': 9, 'samples': 2}], s)
    assert split_games([{'gid': 0, 'samples': 0}], None) == [({'gid': 0, 'samples': 0}, None)]


def test_split_games_on_the_driver():
    """The driver hands out rows and samples in the same order: every sliced sample's value target
    is its own game's result."""
    from zero import games as G
    from zero.infer import UniformEvaluator, drive
    from zero.run import DEFAULTS, PRESETS, Run
    cfg = dict(DEFAULTS, **PRESETS['toy-smoke'], players=[2, 3])
    R = Run('.', cfg)
    seat = R.nn_seat('m', 8, selfplay=True)
    drv = R.driver([[seat] * 2, [seat] * 3], seed=5, record=True, rolling=True, net=0, queue_only=True)
    for k in range(40):
        drv.enqueue(f't{k}', G.spec_json(cfg['spec']))
    drv.resume(1 << 62, 0)
    got = {'games': 0, 'rows': 0, 'bad': 0}

    def pump(lb=None):
        for r, sd in split_games(drv.take_results(), drv.take_samples()):
            got['games'] += 1
            if sd is None:
                continue
            got['rows'] += len(sd['player'])
            n = r['n']
            for i, p in enumerate(sd['player']):
                if not np.allclose(sd['z'][i, :n], [r['result'][(int(p) + j) % n] for j in range(n)]):
                    got['bad'] += 1
    drive(drv, {'m': UniformEvaluator()}, on_batch=pump)
    pump()
    assert got['games'] == 40 and got['rows'] > 100 and got['bad'] == 0


def test_lag_histogram():
    h = lag_hist(np.array([4, 4, 3, 1, 5]), 5)
    assert h == {'0': 3, '1': 1, '3': 1}           # net 5 (a newer one cannot happen): lag 0
    h = lag_hist(np.array([4]), 5, into=h)
    assert merge_hist(h, {'2': 4}) == {'0': 4, '1': 1, '2': 4, '3': 1}
    assert hist_summary({'0': 3, '1': 1}) == {'rows': 4, 'mean': 0.25, 'max': 1, 'le1': 1.0}
    assert hist_summary({}) is None


class FakeDrv:
    def __init__(self):
        self.q = []
        self.res = []

    @property
    def queued(self):
        return len(self.q)

    def enqueue(self, tag, spec):
        self.q.append((tag, spec))

    def take_results(self):
        r, self.res = self.res, []
        return r


def test_the_coordinator_keeps_the_local_share_in_reserve():
    """With a remote consumer the local driver's room under its cap stays in the queue for it: the
    remote takes only what is queued beyond it, however much room it has; the local driver's
    outstanding count follows what was fed and what came back."""
    from zero.cotrain import Coordinator
    co = Coordinator(lambda p: [f'{p}-{k}' for k in range(4)], lambda p, rows: '[]')
    now = time.time()
    for g in range(5):
        co.events[g] = {'payload': str(g), 'n': 4, 'rows': {}, 't': now}
        for k in range(4):
            co.to_queue.append((f'{g}/{k}', f'{g}-{k}'))
        co.outstanding += 4
    co.local_cap = 6
    assert len(co.take_for_remote(100)) == 20 - 6, 'the remote leaves the local cap queued'
    drv = FakeDrv()
    assert co.feed(drv) == 6 and co.local_out == 6
    assert co.take_for_remote(100) == [] and co.feed(drv) == 0
    row = lambda tag: {'tag': tag, 'result': [1.0, 0.0], 'scores': [1, 0]}  # noqa: E731
    drv.res = [row(t) for t, _ in drv.q[:2]]
    assert co.collect(drv) == 2 and co.local_out == 4
    co.requeue([('4/0', '4-0'), ('4/1', '4-1'), ('4/2', '4-2')])   # given back by a remote: to the front
    assert co.take_for_remote(100) == [('4/0', '4-0')], 'the local driver has room for 2 of them'
    assert co.feed(drv) == 2 and co.local_out == 6
    co.local_cap = None                            # no remote: everything to the local driver, as before
    co.to_queue.extend([('x/0', '')])
    assert co.feed(drv) == 1


def test_the_coordinator_counts_matches_per_event():
    from zero.cotrain import Coordinator

    class D:
        def take_deferred(self):
            return [(1, json.dumps({'episodes': ['a', 'b', 'c']})), (2, json.dumps({'episodes': ['a']}))]
    co = Coordinator(lambda p: json.loads(p)['episodes'], lambda p, rows: '[]')
    assert co.take_events(D()) == 2 and (co.ev_total, co.ep_total) == (2, 4)
