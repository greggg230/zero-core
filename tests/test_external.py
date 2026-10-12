"""External seats (a game's JS bot; here the toy's Python bots): a seat answered in Python from the
exported state."""
import json
import random

import pytest

from zero import _engine as E
from zero import games as G
from zero.infer import UniformEvaluator, drive
from zero.toybots import heuristic_choice


def _results(seat_b, n=60, nn=False):
    a = G.nn_seat('m', 8) if nn else {'bot': 'random'}
    seats = [[a, seat_b] if g % 2 == 0 else [seat_b, a] for g in range(n)]
    drv = G.Driver('toy', seats, concurrency=16, threads=2, seed=5)
    drive(drv, {'m': UniformEvaluator()}, {'toyheur': G.external('toyheur')})
    return sorted((r['gid'], tuple(r['scores']), r['ext_moves']) for r in drv.take_results())


@pytest.mark.parametrize('nn', [False, True])
def test_external_heuristic_equals_builtin(nn):
    """The Python port behind the external seat plays exactly the built-in Rust bot's games."""
    builtin = _results({'bot': 'heuristic'}, nn=nn)
    external = _results({'external': 'toyheur'}, nn=nn)
    assert [r[:2] for r in builtin] == [r[:2] for r in external]
    assert all(r[2] == 0 for r in builtin) and all(r[2] > 0 for r in external)


def test_python_heuristic_is_legal_everywhere():
    rng = random.Random(0)
    n = 0
    for s in range(40):
        g = E.Game('toy', None, 2 + s % 3, s)
        while not g.over:
            a = heuristic_choice(json.loads(g.export_state()))
            assert a in g.legal()
            n += 1
            g.step(a if rng.random() < 0.5 else rng.choice(g.legal()))
    assert n > 1000


def test_external_payload_errors_stop_the_driver():
    drv = G.Driver('toy', [[{'external': 'bad'}, {'bot': 'random'}]], concurrency=1, threads=1)
    with pytest.raises(RuntimeError, match='external move failed'):
        drive(drv, {}, {'bad': lambda lb: ['[999]'] * lb.B})


def test_missing_external_player_is_loud():
    drv = G.Driver('toy', [[{'external': 'nobody'}, {'bot': 'random'}]], concurrency=1, threads=1)
    with pytest.raises(KeyError):
        drive(drv, {}, {})
    with pytest.raises(ValueError):
        G.external('nobody')


def test_async_external_answers_match_inline():
    """drive() answers external batches on threads while other groups search: same games as
    answering each batch inline, even when the external player is slow."""
    import time

    def slow_heur(lb):
        time.sleep(0.002)
        return G.external('toyheur')(lb)

    def run(inline):
        a = G.nn_seat('m', 8)
        o = {'external': 'toyheur'}
        seats = [[a, o] if g % 2 == 0 else [o, a] for g in range(48)]
        drv = G.Driver('toy', seats, concurrency=24, threads=2, groups=4, seed=9)
        ev = UniformEvaluator()
        if inline:
            while (lb := drv.next_batch()) is not None:
                if lb.external:
                    drv.submit_external(lb.id, slow_heur(lb))
                else:
                    drv.submit(lb.id, *ev(lb))
        else:
            drive(drv, {'m': ev}, {'toyheur': slow_heur})
        return sorted((r['gid'], tuple(r['scores'])) for r in drv.take_results())

    assert run(inline=False) == run(inline=True)


def test_run_match_groups():
    """run.py: matches with an external player use ext_groups groups, the others the run's groups."""
    from zero import run
    cfg = run.resolve_config(run.parse(['--run', 'x']), {})
    r = run.Run('x', cfg)
    assert cfg['ext_groups'] == 2 and r.match_groups({'jsbot': None}) == 2 and r.match_groups({}) is None
    cfg = run.resolve_config(run.parse(['--run', 'x', '--ext-groups', '6']), {})
    assert run.Run('x', cfg).match_groups({'toyheur': None}) == 6
