"""The exhaustive lookahead per information set (docs/search.md 3.9): its settings, the position API.

A lookahead seat is a network seat with the key `lookahead` (a dict of its settings; the Rust
`lookahead::LookCfg` validates them: an unknown or bad setting is a ValueError, never ignored):

    plies           moves of a line (decisions with >= 2 options, either seat; default 4)
    horizon         'plies' (default) | 'turn' (to the end of the turn) | 'next_turn' (to the start of the
                    decider's next turn), at most `max_plies` moves (default 40; a cut line is counted)
    worlds          dealt worlds (default 8)
    topk            0 = every option (default), else the k of highest prior below the root;
                    topk_root: the root too; topk_from_turn: every option before that turn
    opp             'min' / 'best' (default: the reply worst for us), 'expect' / 'policy' (prior-weighted),
                    'top' (its top-prior reply)
    mode            'infoset' (default: our choices per information set) | 'world' (per world)
    budget          nodes per decision (0 = none, default); over it: fallback_topk (default 4) below the
                    root, then fallback_plies (a list; default max(2, plies - 2), 4 under a horizon),
                    then fallback 'mcts' (default: the seat's MCTS search) | 'prior' (the top prior)
    rows            rows per network batch of one decision (default 512)

`parse('plies=6,worlds=8,topk=0,opp=best,mode=infoset,budget=200000')` reads the command-line form
(zero.h2h --a-lookahead / --b-lookahead; `fallback_plies=4/2`), `settings(d)` the canonical dict.
The seat's MCTS settings (sims, rule, belief, leaf view) stay: they deal the lookahead's worlds (the
belief's tilt / clip / mix) and play its final fallback.

`evaluate_position(game, ev, settings, ...)`: the lookahead of one position (the Delver Lab's call):
per root option its value in each world (P(the decider wins) under the best play per information set
after it), mean +- sd, the modal principal line with labels, the counters and timing.
"""
import collections
import json
import time

KEYS = ('plies', 'horizon', 'max_plies', 'worlds', 'topk', 'topk_root', 'topk_from_turn', 'opp', 'mode', 'budget',
        'fallback_topk', 'fallback_plies', 'fallback', 'rows')
INT_KEYS = ('plies', 'max_plies', 'worlds', 'topk', 'topk_from_turn', 'budget', 'fallback_topk', 'rows')
BOOL_KEYS = ('topk_root',)
STR_KEYS = ('horizon', 'opp', 'mode', 'fallback')


def _engine():
    from .games import engine
    return engine()


def settings(d):
    """The canonical settings dict of `d` (a dict or its JSON): every setting, defaults filled in;
    checked by the engine (`lookahead_settings`): a ValueError names a bad or unknown one."""
    return json.loads(_engine().lookahead_settings(d if isinstance(d, str) else json.dumps(d)))


def parse(text):
    """'k=v,k=v,...' (or a JSON object) -> the settings dict as given, checked (`settings`). Integers,
    booleans (topk_root: 1 / 0 / true / false), names; fallback_plies '4/2' (a list; '' = none)."""
    text = (text or '').strip()
    if text.startswith('{'):
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError(f'lookahead {text!r}: a JSON object')
        settings(d)
        return d
    d = {}
    for item in [x.strip() for x in text.split(',') if x.strip()]:
        if '=' not in item:
            raise ValueError(f'lookahead setting {item!r}: key=value')
        k, v = (x.strip() for x in item.split('=', 1))
        if k not in KEYS:
            raise ValueError(f'lookahead: unknown setting {k!r} (one of {", ".join(KEYS)})')
        if k in d:
            raise ValueError(f'lookahead: {k} given twice')
        try:
            if k in INT_KEYS:
                d[k] = int(v)
            elif k in BOOL_KEYS:
                if v.lower() not in ('1', '0', 'true', 'false'):
                    raise ValueError
                d[k] = v.lower() in ('1', 'true')
            elif k == 'fallback_plies':
                d[k] = [int(x) for x in v.split('/') if x.strip()]
            else:
                d[k] = v
        except ValueError:
            raise ValueError(f'lookahead {k}={v!r}: not a valid value') from None
    settings(d)
    return d


def tag(d):
    """A short name of the settings: 'la-p6-w8-all-min-is-b200000' (horizons 'turn' / 'next_turn' as
    't<max>' / 'nt<max>')."""
    s = settings(d)
    h = {'plies': f'p{s["plies"]}', 'turn': f't{s["max_plies"]}', 'next_turn': f'nt{s["max_plies"]}'}[s['horizon']]
    k = 'all' if not s['topk'] else f'k{s["topk"]}' + ('r' if s['topk_root'] else '')
    m = {'infoset': 'is', 'world': 'w'}[s['mode']]
    b = f'-b{s["budget"]}' if s['budget'] else ''
    return f'la-{h}-w{s["worlds"]}-{k}-{s["opp"]}-{m}{b}'


def _labels(g):
    """{action: label} of g's decision (its exported state's options, when it has them)."""
    try:
        st = json.loads(g.export_state())
    except (ValueError, TypeError):
        return {}
    d = st.get('decision') if isinstance(st, dict) else None
    opts = (d or {}).get('options') or []
    out = {}
    for o in opts:
        if isinstance(o, dict) and 'a' in o and 't' in o:
            out[int(o['a'])] = o['t']
    return out


def replay_line(world, moves, reveal=None):
    """A line's moves [(player, action, minor)] replayed on a copy of `world` -> [{'p', 'a', 'label',
    'minor'[, 'seen']}] (labels from the exported state; `reveal(keys, p)` -> what the move showed,
    e.g. MTG's draws and looks) and where it is at the end ({'turn', 'step'} when the state says)."""
    from .mcts import transition
    g = world.clone()
    out = []
    for p, a, minor in moves:
        lab = _labels(g).get(int(a), f'action {a}')
        keys, _ = transition(g, int(a))
        e = {'p': int(p), 'a': int(a), 'label': lab, 'minor': bool(minor)}
        if reveal is not None:
            seen = reveal(keys, p)
            if seen:
                e['seen'] = seen
        out.append(e)
    where = None
    try:
        st = json.loads(g.export_state())
        if isinstance(st, dict) and ('turn' in st or 'step' in st):
            where = {'turn': st.get('turn'), 'step': st.get('step')}
    except (ValueError, TypeError):
        pass
    return out, where


def drive(la, ev):
    """Answer a `Lookahead`'s rounds with evaluator `ev` (NetEvaluator.evaluate's contract) until it
    decides -> (rounds, rows)."""
    rounds = rows = 0
    while True:
        b = la.next_batch()
        if b is None:
            return rounds, rows
        out = ev.evaluate(b['x'], b['n_players'], b['opt_off'], b['opt_action'], b['extra'])
        la.submit(out[0], out[1], out[3] if len(out) > 3 else None)
        rounds += 1
        rows += len(b['player'])


def evaluate_position(game, ev, settings_, worlds=None, seeds=None, seed=0, belief=None, belief_tilt=0.0,
                      belief_clip=20.0, belief_mix=0.0, lines=True, reveal=None):
    """The lookahead of `game`'s pending decision (module docstring) with evaluator `ev` and settings
    `settings_` (a dict). Worlds: `worlds` (games at the same decision), else dealt from `seeds` (one
    per world) or `seed`'s stream, by the root's belief when `belief_tilt` > 0 (`belief`: its
    predictions; None: asked of `ev` in the first round). -> {'decider', 'outcome' (an action, or
    'mcts'), 'options': [{'action', 'label', 'value', 'sd', 'n', 'values' (per world, in world order),
    'line' (the modal line: [{'p', 'a', 'label', 'minor'}]), 'line_worlds', 'end', 'ends' (per end,
    the worlds)}] sorted by value (stably: ties keep the root's order), 'stats', 'settings', 'seconds',
    'rounds', 'rows', 'worlds'}."""
    import numpy as np
    eng = _engine()
    t0 = time.time()
    kw = {}
    if worlds is not None:
        kw['worlds'] = list(worlds)
    if seeds is not None:
        kw['seeds'] = [int(s) for s in seeds]
    if belief_tilt:
        kw.update(belief_tilt=float(belief_tilt), belief_clip=float(belief_clip), belief_mix=float(belief_mix))
        if belief is not None:
            kw['belief'] = np.asarray(belief, np.float32)
    la = eng.Lookahead(game, settings_, seed=int(seed), lines=bool(lines), **kw)
    rounds, rows = drive(la, ev)
    secs = time.time() - t0
    r = la.result()
    ws = la.worlds()
    labels = _labels(game)
    opts = []
    for o in r['options']:
        a = int(o['action'])
        vals = [v for _, v in o['values']]
        row = {'action': a, 'label': labels.get(a, f'action {a}'), 'value': o['mean'], 'sd': o['sd'], 'n': len(vals),
               'values': vals}
        if lines and ws:
            got = []
            for w, _ in o['values']:
                ln = la.line(int(w), a)
                if ln is None:
                    continue
                moves, end = ln
                rl, where = replay_line(ws[int(w)], moves, reveal)
                key = tuple((e['p'], e['label']) for e in rl if not e['minor'])
                got.append((key, rl, end, where))
            if got:
                c = collections.Counter(k for k, _, _, _ in got)
                best = max(c.items(), key=lambda kv: kv[1])[0]          # (the first most common)
                k0 = next(x for x in got if x[0] == best)
                row.update(line=k0[1], line_worlds=c[best], end=k0[2], where=k0[3],
                           ends=dict(collections.Counter(x[2] for x in got)))
        opts.append(row)
    opts.sort(key=lambda x: -x['value'])
    return {'decider': game.to_move, 'outcome': r['outcome'], 'options': opts, 'stats': r['stats'],
            'settings': json.loads(r['settings']), 'seconds': round(secs, 3), 'rounds': rounds, 'rows': rows,
            'worlds': len(ws), 'belief': None if la.belief is None else list(la.belief)}
