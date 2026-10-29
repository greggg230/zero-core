"""The shared work queue's model (zero/workqueue.py, docs/az.md 5.3): each consumer's in-flight cap
targets its game latency (bootstrapped from its throughput's share of an iteration's units), for
throughput-bound and latency-bound consumers alike - simulated phases of synthetic consumers whose
latency follows a curve of their in-flight (delver1's desktop: rate growing with the games in
flight); per-game sample splitting (a game beyond a quota waits with its samples); the staleness
histogram; the co-training coordinator keeping the local driver's share in reserve. Units: no driver
except `test_split_games_on_the_driver`."""
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
    a restart's rates (and the latency model's caps, sensitivities) come back from the metrics row's
    state."""
    sh = Shares(1.0)
    sh.consumer('local', 2, 100)
    assert sh.observe('local', 100, 10.0, 1.0) == pytest.approx(10.0)
    sh.observe('local', 200, 10.0, 1.0)
    assert sh.c['local']['rate'] == pytest.approx(15.0)
    assert sh.observe('local', 50, 10.0, 0.5) == pytest.approx(5.0)    # half busy: 5/s measured ...
    assert sh.c['local']['rate'] == pytest.approx((15.0 + 10.0) / 2)   # ... 10/s at its cap
    sh.observe('local', 50, 10.0, 0.01)
    assert sh.c['local']['rate'] == pytest.approx((12.5 + 20.0) / 2)   # at most x4
    assert sh.observe('local', 5, 0.1, 1.0) is None and sh.observe('local', 0, 100.0, 1.0) is None
    sh.c['local']['cap'], sh.c['local']['sens'] = 77.0, 0.3
    st = json.loads(json.dumps(sh.state()))
    sh2 = Shares(1.0)
    sh2.restore(st)
    sh2.consumer('local', 2, 100)
    sh2.consumer('w0', 4, 400)                     # a new worker: the measured consumers' rate per slot
    r = sh2.rates()
    assert r['local'] == pytest.approx(st['local']['rate'], rel=1e-5)
    assert r['w0'] == pytest.approx(st['local']['rate'] / 100 * 400, rel=1e-5)
    assert (sh2.c['local']['cap'], sh2.c['local']['sens'], sh2.c['w0']['cap']) == (77.0, 0.3, None)


def simulate(consumers, units=800, phases=8, inflight=1.0, off=None):
    """Phases of synthetic consumers {name: (lo, hi, gpu, latency(n), cap0 or None)}: each plays its
    cap's games at the latency its curve gives (rate = n / latency(n)), the phase lasts until the
    units are done, every consumer is measured as zero.remoteplay measures it. `off`: {phase: name}
    backing off for that phase. -> [(caps, rates, phase seconds)] per phase."""
    sh = Shares(inflight)
    for n, (lo, hi, gpu, lat, cap0) in consumers.items():
        sh.consumer(n, lo, hi, cautious=n != 'local', gpu=gpu)
        if cap0 is not None:
            sh.c[n]['cap'] = float(cap0)
            sh.c[n]['rate'] = cap0 / lat(cap0)
    sh.units = units
    sh.lpu = 1000.0
    out = []
    for ph in range(phases):
        for n in consumers:
            sh.set_active(n, not (off and off.get(ph) == n))
        caps = sh.caps()
        rate = {n: (caps[n] / consumers[n][3](caps[n]) if caps[n] else 0.0) for n in consumers}
        secs = units / sum(rate.values())
        meas = {n: {'units': rate[n] * secs, 'run_s': secs, 'util': 1.0, 'leaves': rate[n] * secs * 1000.0,
                    'cap': caps[n], 'lead': 0.0} for n in consumers if caps[n]}
        sh.observe_phase(meas, phase_s=secs)
        out.append((caps, rate, secs))
    return out


# delver1's desktop (zero-core 213a9c0, 2026-10-28, its log): 82 games in flight at ~234 s a game, 192
# at ~282 s - a latency of ~198 s + 0.44 s a game (CUDA batches: the rate grows with the games in
# flight); a Mac worker: CPU, its rate ~1.5 games/s whatever its in-flight (latency in-flight / 1.5)
DESKTOP = lambda n: 198.0 + 0.44 * n                    # noqa: E731
MACW = lambda n: 5.0 + n / 1.5                          # noqa: E731


def test_a_gpu_consumer_whose_rate_grows_with_its_in_flight_is_not_starved():
    """The spiral of zero-core 213a9c0 (its cap = rate x horizon: a smaller cap measured a lower rate
    and shrank again - delver1's desktop 181 -> 83 -> 75 games in flight, 6.5k -> 2.3k leaves/s): the
    latency target lets a consumer whose games take nearly as long at any load grow to its slots
    while they finish within 1.5 iterations, from the spiral's own state; the CPU workers converge to
    games one iteration long."""
    run = simulate({'local': (4, 192, True, DESKTOP, 75), 'w0': (8, 768, False, MACW, 300),
                    'w1': (8, 768, False, MACW, 300)}, phases=8)
    caps, rate, secs = run[-1]
    assert caps['local'] == 192, [r[0] for r in run]
    assert run[2][0]['local'] == 192, 'two phases from the spiral back to its slots'
    assert DESKTOP(192) <= 1.5 * secs, (DESKTOP(192), secs)       # its games: within 1.5 phases
    for w in ('w0', 'w1'):                                        # the CPU workers: one phase each
        assert abs(MACW(caps[w]) / secs - 1.0) < 0.1, (caps, secs)
    assert rate['local'] > 2 * 75 / DESKTOP(75), 'the desktop more than doubled its throughput'


def test_a_latency_bound_consumer_with_a_cpu_prior_is_found_by_its_measurements():
    """The same desktop curve with the CPU prior (sens 1): it shrinks first, its latency hardly falls,
    the measured sensitivity marks it latency-bound and its cap grows back to its slots."""
    run = simulate({'local': (4, 192, False, DESKTOP, 181), 'w0': (8, 768, False, MACW, 300),
                    'w1': (8, 768, False, MACW, 300)}, phases=10)
    caps = [r[0]['local'] for r in run]
    assert min(caps) < 181 and caps[-1] >= 185, caps


def test_a_throughput_bound_consumer_converges_to_one_iteration():
    """CPU workers alone (latency proportional to their in-flight): the caps settle where a game
    takes one iteration's phase (inflight 1), and half of it with inflight 0.5."""
    for f in (1.0, 0.5):
        run = simulate({'local': (4, 768, False, MACW, None), 'w0': (8, 768, False, MACW, None)}, units=600,
                       phases=10, inflight=f)
        caps, rate, secs = run[-1]
        for n in caps:
            assert abs(MACW(caps[n]) / (f * secs) - 1.0) < 0.12, (f, caps, secs)


def test_a_consumer_whose_games_outlast_the_phase_gets_its_floor():
    """A worker whose games take 1,400 s at any load in a phase of ~250 s: no in-flight makes them
    fresh, its cap falls to its floor (its stale rows stay few)."""
    run = simulate({'local': (4, 192, True, DESKTOP, 150), 'w0': (8, 768, False, lambda n: 1400.0 + n / 10, 300),
                    'w1': (8, 768, False, MACW, 300)}, phases=8)
    assert run[-1][0]['w0'] == 8 and run[-1][2] < 400, [(r[0], round(r[2])) for r in run]


def test_a_backoff_gives_the_desktop_the_whole_quota_at_once():
    """A Mac worker backing off for a phase: its cap is 0 and the desktop's grows with its horizon at
    once (here to its slots); the next phase the shares come back."""
    run = simulate({'local': (4, 192, True, DESKTOP, 110), 'w0': (8, 768, False, MACW, 300)}, phases=5,
                   off={3: 'w0'})
    assert run[3][0]['w0'] == 0 and run[3][0]['local'] == 192, run[3][0]
    assert run[4][0]['w0'] > 0


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
