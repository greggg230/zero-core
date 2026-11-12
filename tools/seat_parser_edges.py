"""Edge cases of the seat / driver settings parser, old module vs new (zero-farm phase 0b, docs/az.md 42):
each case's `driver_settings` text from two builds of an engine module (e.g. the toy module of the commit
before zero_search::settings and of the commit after) - identical, or both refused (the messages may
differ), or listed as a deliberate change.

    python tools/seat_parser_edges.py <dir holding oldpkg/_engine.pyd> oldpkg  [new module: zero._engine]
"""
import importlib
import json
import sys

import numpy as np


def cases():
    nn = {'sims': 8, 'selfplay': True}
    out = {
        'empty nn': ([[{}, {}]], {}),
        'defaults null': ([[{'sims': None, 'redet': None, 'key': None, 'sims_fast': None}] * 2], {}),
        'self-play seat': ([[nn, nn]], dict(rolling=True, record=True, queue_only=True, seed=7, concurrency=9)),
        'bool ints': ([[{'sims': True, 'c_puct': False, 'temp_moves': True}] * 2], {}),
        'truthy flags': ([[{'selfplay': 'x', 'redet': 0, 'record': [], 'belief_fast': {'a': 1}, 'infoset_merge': 0.0}] * 2], {}),
        'int float': ([[{'sims': 64.0}] * 2], {}),
        'negative int': ([[{'sims': -1}] * 2], {}),
        'huge int': ([[{'sims': 2 ** 32}] * 2], {}),
        'float from int': ([[{'alpha': 1, 'eps': 0, 'margin_w': 2}] * 2], {}),
        'str for int': ([[{'sims': '8'}] * 2], {}),
        'unknown key': ([[{'sim': 8}] * 2], {}),
        'bot': ([[{'bot': 'random'}, {'bot': 'heuristic'}]], {}),
        'bot none': ([[{'bot': None}, {}]], {}),
        'external': ([[{'external': 'js'}, {'sims': 4}]], {}),
        'external and nn key clash': ([[{'external': 'js'}, {'key': 'js'}]], {}),
        'keys order': ([[{'key': 'o'}, {}], [{}, {'key': 'o'}], [{'key': 'b', 'sims': 3}, {'external': 'e'}]], {}),
        'cq target': ([[{'policy_target': 'cq', 'cq_c_visit': 30, 'cq_c_scale': 0.3, 'cq_prior_w': 0.5}] * 2], {}),
        'cq prior on visits': ([[{'cq_prior_w': 0.5}] * 2], {}),
        'select cq': ([[{'select': 'cq', 'select_min_visits': 2, 'select_c_scale': 0.1}] * 2], {}),
        'select visits with transform': ([[{'select_c_visit': 3}] * 2], {}),
        'gumbel': ([[{'root_search': 'gumbel', 'gumbel_m': 8, 'policy_target': 'cq'}] * 2], {}),
        'gumbel keys on puct': ([[{'gumbel_scale': 0.0}] * 2], {}),
        'belief': ([[{'belief_tilt': 1, 'belief_clip': 10, 'belief_mix': 0.25, 'belief_fast': True,
                      'belief_tilts': {'deck': 1, 'belief': 0.5}}] * 2], {}),
        'belief tilts list': ([[{'belief_tilts': [1, 2]}] * 2], {}),
        'root view': ([[{'leaf_value_view': 'root', 'view_samples': 0.5, 'view_sample_target': 'z',
                         'infoset_merge': True}] * 2], {}),
        'lookahead dict': ([[{'lookahead': {'plies': 2, 'worlds': 4, 'opp': 'min'}}] * 2], {}),
        'lookahead json': ([[{'lookahead': '{"plies": 3}'}] * 2], {}),
        'lookahead bad': ([[{'lookahead': {'plies': 'x'}}] * 2], {}),
        'resign': ([[{'resign_t': 0.05, 'resign_k': 3, 'resign_noresign': 0.2, 'selfplay': True}] * 2], {}),
        'value choice': ([[{'value_choice_worlds': 8, 'value_choice_temp': 0.03}] * 2], {}),
        'eval cache min': ([[{'eval_cache': 8192, 'sims_minor': 4, 'root_min_visits': 2,
                              'root_min_visits_max_actions': 5}] * 2], {}),
        'numpy values': ([[{'sims': np.int64(8), 'alpha': np.float64(0.5), 'selfplay': np.bool_(True)}] * 2], {}),
        'tuple rows': ((({'sims': 2}, {'sims': 2}),), {}),
        'branch': ([[nn, nn]], dict(record=True, rolling=True, branch={'p': 0.25, 'moves': 2, 'temp': 1.5, 'differ': 1})),
        'branch defaults': ([[nn, nn]], dict(record=True, branch={'p': 1})),
        'branch no p': ([[nn, nn]], dict(branch={'moves': 2})),
        'branch p none': ([[nn, nn]], dict(branch={'p': None})),
        'branch unknown': ([[nn, nn]], dict(branch={'p': 0.5, 'x': 1})),
        'starts tuples': ([[nn, nn]], dict(record=True, starts={'p': 0.1, 'specs': [('a', {'z': 1, 'a': [1, 2.5]}),
                                                                                    ('b', '{"q":1}'), ('c', None)]})),
        'starts no specs': ([[nn, nn]], dict(starts={'p': 0.1})),
        'starts empty': ([[nn, nn]], dict(starts={'p': 0.1, 'specs': []})),
        'starts unknown': ([[nn, nn]], dict(starts={'p': 0.1, 'specs': [('a', '{}')], 'w': 1})),
        'spec dict': ([[{}, {}]], dict(spec={'b': 1, 'a': {'y': [1, 2], 'x': 'é'}})),
        'spec str': ([[{}, {}]], dict(spec='{"x": 1}')),
        'all args': ([[{}, {}]], dict(concurrency=3, seed=99, record=True, threads=2, groups=5, max_steps=77,
                                      rolling=True, net=4, queue_only=True, move_log=6, aux=True, paired=True,
                                      sparse=True)),
    }
    return out


DELIBERATE = {
    # zero_search::settings is stricter than the old bindings for inputs no caller makes (docs/az.md 42)
    'bot with another key', 'starts list entries',
}


def main():
    sys.path.insert(0, sys.argv[1])
    old = importlib.import_module(sys.argv[2] + '._engine')
    new = importlib.import_module(sys.argv[3] if len(sys.argv) > 3 else 'zero._engine')
    cs = cases()
    # deliberate differences, shown separately
    extra = {
        'bot with another key': ([[{'bot': 'random', 'sims': 8}, {}]], {}),
        'starts list entries': ([[{'sims': 8, 'selfplay': True}] * 2], dict(record=True, starts={'p': 0.1, 'specs': [['a', {'z': 1, 'a': [1, 2.5]}]]})),
    }
    bad = 0
    for name, (seats, kw) in list(cs.items()) + list(extra.items()):
        res = []
        for m in (old, new):
            try:
                res.append(m.driver_settings(seats, **kw))
            except Exception as e:                 # noqa: BLE001
                res.append(f'ERROR {type(e).__name__}: {e}')
        a, b = res
        if a == b:
            st = 'refused by both (same message)' if a.startswith('ERROR') else 'identical'
        elif a.startswith('ERROR') and b.startswith('ERROR'):
            st = f'refused by both\n      old: {a[:200]}\n      new: {b[:200]}'
        elif name in DELIBERATE:
            st = f'DELIBERATE CHANGE\n      old: {a[:200]}\n      new: {b[:200]}'
        else:
            st = f'DIFFERENT\n--- old\n{a}\n--- new\n{b}'
            bad += 1
        print(f'{name}: {st}')
    print(json.dumps({'cases': len(cs) + len(extra), 'different': bad}))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
