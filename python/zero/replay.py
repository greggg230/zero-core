"""Game replays (docs/replays.md): games played by checkpoints (and bots), recorded with what the
search saw at every decision, for the game apps' replay viewers.

  python -m <pkg>.az.replays CKPT [--opp ext:jsbot | self | random | OTHER.pt] [--games 6]
      [--sims 128] [--opp-sims N] [--spec JSON] [--selfplay] [--hint ext:jsbot | none]
      [--chance-k 2] [--device cuda] [--seed 0] [--out DIR] [--title T]
      [--select visits | cq] [--select-min-visits N] [--cq-c-visit V --cq-c-scale S]
      [--upload [URL]] [--key-file F]

The checkpoint plays `--games` games against `--opp` (another checkpoint, itself, a built-in
bot or an external player such as the JS bot), seats alternating. Its search runs the Rust tree
from Python (`play.PyDriver`, impl 'rs'): evaluation settings by default (no noise, argmax),
`--selfplay` = the run's self-play settings (Dirichlet noise, temperature for the first
`temp_moves` decisions). The moves it does not sample follow the run's move rule (docs/search.md
3.3: eval_select, or selfplay_select with --selfplay, with the rule's completed-Q transform -
its *_c_visit / *_c_scale, unset: cq_c_visit / cq_c_scale; a checkpoint without a run: 'visits'),
or --select / --select-min-visits, and --cq-c-visit / --cq-c-scale for the 'cq' rule's transform;
the seat records its rule (`select`, e.g. 'cq-n4-v50-s0.3'). The spec defaults to the
checkpoint's run (`runs/<run>/config.json`), else the game package's. `--hint ext:<name>` asks that external player, at each network
decision whose state it has not seen yet, what it would do there (stored, never applied): the
viewer shows it next to the network's choice (default: the package's `replay_hint`).

Each game -> one JSON object (format 'zero-replay/1'), written gzipped to `--out` (default
`runs/<run>/replays/`) and optionally uploaded to the game app (`--upload`: POST
<url>/api/replays/upload, `Authorization: Bearer <key>`; url and key from the package's
`replay_url` / `<PREFIX>_REPLAY_KEY` or `--key-file`).

  format   'zero-replay/1'
  game, spec, seed, created (unix s), title, group, source {checkpoint, run, iter}
  seats    [{name, kind: 'nn' | 'bot' | 'ext', checkpoint?, run?, iter?, sims?, select? (the move
           rule, absent = 'visits'), aux? (the names of its checkpoint's auxiliary targets: its
           frames show the head's predictions)}]
  states   [the game's export_state(), parsed]   (a state is stored once per change)
  frames   one per step, in order:
           {s: state index before the step, p: seat to move, k: 'nn' | 'bot' | 'ext' | 'forced',
            a: action id ('nn', 'bot', 'forced'),
            'nn':  acts, n (root visits), p0 (network priors, noise-free), q (mean utility of each
                   option for the mover, null if unvisited), v (network utility of the root for
                   the mover), w (search P(win) per seat), sims, sampled (a temperature pick)
            'ext': x (the external's answer, parsed JSON)
            hint:  {by: 'ext:<name>', x: its answer at this state} (network decisions)
            the auxiliary head (a checkpoint with one, docs/az.md 15; absent otherwise): what the
                   game package's `replay_aux` makes of the head's predictions at the decision's root
                   for the decider (one forward pass of the root's features after its search; MTG
                   Zero: `belief` {for, n, cards: [[name, expected count], ...]}), by default
                   aux: {<target>: [[column, value], ...]} (its 12 largest values)}
  final    index of the final state; result (per seat), scores, rounds, over
"""
import argparse
import base64
import datetime
import gzip
import json
import os
import random
import re
import sys
import time

import numpy as np

from . import games as G, package
from .infer import NetEvaluator, drive
from .mcts import select_settings
from .net import load
from .play import NNSeat, PyDriver, Slot
from .protocol import LeafBatch

FORMAT = 'zero-replay/1'


class ExtSeat:
    """An external player's seat in the recording driver (`ext:<name>`)."""

    def __init__(self, name):
        self.name = name


class RecSlot(Slot):
    """A game of the recording driver: `play.Slot` + its states, frames and pending request."""
    __slots__ = ('states', 'frames', 'last_state', 'hinted', 'frame', 'waiting')

    def __init__(self, gid, game, spec, seats, rng):
        super().__init__(gid, game, spec, seats, rng)
        self.states, self.frames = [], []
        self.last_state = self.hinted = self.frame = None
        self.waiting = False


class RecordingDriver(PyDriver):
    """`play.PyDriver` + external seats + per-step frames. seat_fn(game index) -> seats: NNSeat
    (impl 'rs'), a built-in bot name, or ExtSeat. `hint`: an external name asked (not applied)
    at network decisions. Finished games: `take_replays()` (dicts without the header fields)."""

    def __init__(self, game, n_games, seat_fn, spec=None, concurrency=8, seed=0, max_steps=100_000, hint=None,
                 aux=None, aux_render=None):
        self.hint = hint
        # evaluator key -> its NetEvaluator, for the network seats whose checkpoint has an auxiliary
        # head (docs/az.md 15): each of their decisions' frames shows the head's predictions at the root
        self.aux = dict(aux or {})
        self.aux_render = aux_render or default_aux_render
        self.ext_out = {}
        self.replays = []
        super().__init__(game, n_games, seat_fn, spec=spec, concurrency=concurrency, seed=seed, record=False,
                         max_steps=max_steps)

    # ---- protocol additions -----------------------------------------------------------------
    def submit_external(self, bid, payloads):
        lb, items = self.ext_out.pop(bid)
        for (slot, p, kind, frame, _), payload in zip(items, payloads):
            x = json.loads(payload)
            if kind == 'hint':
                frame['hint'] = {'by': f'ext:{self.hint}', 'x': x}
                continue
            frame['x'] = x
            slot.frames.append(frame)
            slot.game.apply_external(payload)
            slot.steps += 1
            slot.trees.clear()
            slot.sims = 0
            slot.target = None
            slot.waiting = False

    @property
    def finished(self):
        return not self.active and not self.queue and not self.outstanding and not self.ext_out

    def next_batch(self):
        while not self.queue:
            if self.outstanding or self.ext_out:
                raise RuntimeError('submit the outstanding batches before asking for more')
            if not self.active:
                return None
            self._round()
        return self.queue.pop(0)

    def take_replays(self):
        r, self.replays = self.replays, []
        return r

    # ---- frames -----------------------------------------------------------------------------
    def _state(self, slot):
        s = slot.game.export_state()
        if s != slot.last_state:
            slot.states.append(json.loads(s))
            slot.last_state = s
        return len(slot.states) - 1

    def _ext_batch(self, key, items):
        a = {'states': [it[4] for it in items],
             'player': np.asarray([it[1] for it in items], np.int32),
             'n_players': np.asarray([it[0].n for it in items], np.int32),
             'gid': np.asarray([it[0].gid for it in items], np.int64)}
        lb = LeafBatch(self.next_bid, key, 'ext', a)
        self.ext_out[self.next_bid] = (lb, items)
        self.next_bid += 1
        self.queue.append(lb)

    # ---- internals (PyDriver's, with external seats and frames) ------------------------------
    def _refill(self):
        while len(self.active) < self.concurrency and self.next_gid < self.n_games:
            seats = self.seat_fn(self.next_gid)
            self.active.append(RecSlot(self.next_gid, self.game, self.spec, seats, random.Random(self.rng.randrange(1 << 30))))
            self.next_gid += 1

    def _round(self):
        need = {}
        ext = {}
        hints = []
        finished = []
        for slot in self.active:
            if slot.waiting:
                continue
            while True:
                g = slot.game
                if g.over or slot.steps >= self.max_steps:
                    finished.append(slot)
                    break
                legal = g.legal()
                p = g.to_move
                if len(legal) == 1:
                    slot.frame = {'s': self._state(slot), 'p': p, 'k': 'forced'}
                    self._apply(slot, legal[0])
                    continue
                seat = slot.seats[p]
                if isinstance(seat, ExtSeat):
                    slot.waiting = True
                    ext.setdefault(seat.name, []).append((slot, p, 'move', {'s': self._state(slot), 'p': p, 'k': 'ext'},
                                                          slot.last_state))
                    break
                if not isinstance(seat, NNSeat):
                    slot.frame = {'s': self._state(slot), 'p': p, 'k': 'bot'}
                    self._apply(slot, slot.bots[p].choose(g))
                    continue
                if slot.target is None:
                    self._start(slot, seat, p)
                    s = self._state(slot)
                    slot.frame = {'s': s, 'p': p, 'k': 'nn'}
                    if self.hint and s != slot.hinted:
                        slot.hinted = s
                        hints.append((slot, p, 'hint', slot.frame, slot.last_state))
                tree = slot.trees[p]
                if slot.sims >= slot.target:
                    self._decide(slot, seat, tree, p)
                    continue
                r = tree.simulate(g, slot.search_rng, seat)
                if r is None:
                    slot.sims += 1
                    continue
                player, x, acts, vx = r
                need.setdefault(seat.key, []).append((slot, seat, tree, player, x, acts, vx))
                break
        for slot in finished:
            self.active.remove(slot)
            self._finish(slot)
        self._refill()
        if hints:
            self._ext_batch(self.hint, hints)
        for name, items in ext.items():
            self._ext_batch(name, items)
        self._leaf_batches(need)

    def _decide(self, slot, seat, tree, p):
        t = tree.t
        acts, nv, p0 = list(t.root_acts), list(t.root_visits), list(t.root_priors0)
        wsum = list(t.root_values)
        win = t.root_win()
        f = slot.frame
        f.update(acts=acts, n=nv, p0=[round(float(x), 4) for x in p0],
                 q=[round(w / n, 4) if n else None for w, n in zip(wsum, nv)],
                 v=round(float(t.root_value0), 4), w=[round(float(x), 4) for x in win] if win else None,
                 sims=int(sum(nv)), sampled=bool(seat.selfplay and slot.decisions < seat.temp_moves and sum(nv) > 0))
        ev = self.aux.get(seat.key)
        if ev is not None:
            # the auxiliary head at the root, for the decider: one forward pass of the root's features
            # (the search's own evaluations never compute the head, and a reused tree's root was
            # evaluated during an earlier search)
            x = np.asarray(slot.game.encode(p), np.float32)[None]
            pred = {k: v[0] for k, v in ev.aux(x, np.asarray([slot.n], np.int64)).items()}
            f.update(self.aux_render(pred, slot.states[f['s']], p, f))
        super()._decide(slot, seat, tree, p)

    def _apply(self, slot, a):
        f = slot.frame
        if f is not None:
            f['a'] = int(a)
            slot.frames.append(f)
            slot.frame = None
        super()._apply(slot, a)

    def _finish(self, slot):
        super()._finish(slot)
        g = slot.game
        over = g.over
        self.replays.append({
            'gid': slot.gid, 'states': slot.states + ([] if slot.last_state == g.export_state() else [json.loads(g.export_state())]),
            'frames': slot.frames, 'result': [float(x) for x in g.result()] if over else None,
            'scores': [float(x) for x in g.scores()], 'rounds': int(g.rounds), 'over': bool(over)})
        self.replays[-1]['final'] = len(self.replays[-1]['states']) - 1


def default_aux_render(pred, state, seat, frame, top=12):
    """A frame's auxiliary predictions when the game package has no `replay_aux`: {'aux': {name:
    [[column, value], ...]}}, each target's `top` largest values (rounded to 4 places)."""
    out = {}
    for name, v in pred.items():
        v = np.asarray(v, np.float64)
        idx = np.argsort(-v, kind='stable')[:top]
        out[name] = [[int(i), round(float(v[i]), 4)] for i in idx]
    return {'aux': out}


# ------------------------------------------------------------------------------ recording

def run_of(ckpt):
    """(run dir, iteration) of a checkpoint path runs/<run>/models/iter_NNNN.pt ((None, it) when
    the directory holds no run)."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    m = re.search(r'iter_(\d+)\.pt$', ckpt)
    it = int(m.group(1)) if m else None
    return (d if os.path.exists(os.path.join(d, 'config.json')) else None), it


def run_config(run):
    if not run:
        return {}
    with open(os.path.join(run, 'config.json')) as f:
        return json.load(f)


def _nn(key, sims, cfg, selfplay, chance_k, select=None):
    """A network seat at the run's settings (`cfg`); its move rule the run's (eval_select, or
    selfplay_select with `selfplay`, with the rule's completed-Q transform: `mcts.select_settings`)
    unless `select` = (rule, min_visits) or (rule, min_visits, c_visit, c_scale) says otherwise (a
    rule None: the run's rule and guard; a transform None or left out: the run's rule's). Its leaf
    values' view is the run's (selfplay_ / eval_leaf_value_view, docs/search.md 3.6)."""
    pre = 'selfplay_select' if selfplay else 'eval_select'
    rule, mv, cv, cs = select_settings(cfg, pre)
    if select:
        s = tuple(select) + (None,) * (4 - len(select))
        if s[0] is not None:
            rule, mv = s[0], s[1]
        if rule == 'visits' and s[2:] != (None, None):
            raise ValueError("a completed-Q transform for the move rule 'visits'")
        cv, cs = (x if o is None else o for x, o in zip((cv, cs), s[2:]))
    return NNSeat(key=key, sims=sims, c_puct=cfg.get('c_puct', 1.0), selfplay=selfplay, alpha=cfg.get('alpha', 0.3),
                  eps=cfg.get('eps', 0.25), temp_moves=cfg.get('temp_moves', 30) if selfplay else 0,
                  chance_k=chance_k, redet=bool(cfg.get('redet', 1)), impl='rs', cq_c_visit=cfg.get('cq_c_visit', 50.0),
                  cq_c_scale=cfg.get('cq_c_scale', 1.0), select=rule, select_min_visits=int(mv),
                  select_c_visit=None if rule == 'visits' else cv, select_c_scale=None if rule == 'visits' else cs,
                  leaf_value_view=cfg.get(('selfplay' if selfplay else 'eval') + '_leaf_value_view', 'mover'))


def _with_rule(info, seat):
    if seat.select != 'visits':
        info['select'] = seat.select_tag
    return info


def _ckpt_seat(path, key, sims):
    run, it = run_of(path)
    name = f'{os.path.basename(run)} iter {it}' if run and it is not None else os.path.basename(path)
    return {'name': name, 'kind': 'nn', 'checkpoint': os.path.abspath(path), 'run': os.path.basename(run) if run else None,
            'iter': it, 'sims': sims, 'key': key}


EXT_TITLES = {'jsbot': 'JS bot'}


def record(ckpt, opp='self', games=6, sims=128, opp_sims=None, spec=None, selfplay=False, hint=None, chance_k=2,
           device='cpu', seed=0, title=None, conc=8, log=print, select=None):
    """Play and record `games` games of checkpoint `ckpt` vs `opp` ('self', 'ext:<name>', a bot
    name or another checkpoint), seats alternating (game k: `ckpt` is seat k % 2). `select`:
    (rule, min_visits) or (rule, min_visits, c_visit, c_scale) of the network seats (None: each
    checkpoint's run's; a transform None: the run's rule's). -> replays."""
    ma, _ = load(ckpt, device)                     # (the encoding guard: its own game's, docs/az.md 10.4)
    game = ma.config.get('game', package.active().game)
    run, it = run_of(ckpt)
    cfg = run_config(run)
    if spec is None:
        spec = cfg.get('spec', package.active().specs.get(game, ''))
    evaluators = {'a': NetEvaluator(ma, device)}
    seat_a = _nn('a', sims, cfg, selfplay, chance_k, select)
    me = _with_rule(_ckpt_seat(ckpt, 'a', sims), seat_a)
    externals = {}
    if opp == 'self':
        other, seat_b = me, seat_a
    elif opp.startswith('ext:'):
        name = opp[4:]
        other, seat_b = {'name': EXT_TITLES.get(name, name), 'kind': 'ext', 'ext': name}, ExtSeat(name)
        externals[name] = G.external(name, seed)
    elif opp.endswith('.pt'):
        mb, _ = load(opp, device)
        if mb.config.get('game', game) != game:
            raise SystemExit(f'{opp} plays {mb.config.get("game")}, {ckpt} plays {game}')
        evaluators['b'] = NetEvaluator(mb, device)
        seat_b = _nn('b', opp_sims or sims, run_config(run_of(opp)[0]) or cfg, selfplay, chance_k, select)
        other = _with_rule(_ckpt_seat(opp, 'b', opp_sims or sims), seat_b)
    else:
        other, seat_b = {'name': opp, 'kind': 'bot', 'bot': opp}, opp
    hname = hint[4:] if hint and hint.startswith('ext:') else None
    if hname:
        externals.setdefault(hname, G.external(hname, seed + 1))
    # checkpoints with an auxiliary head: their frames show its predictions (docs/az.md 15)
    aux = {k: ev for k, ev in evaluators.items() if ev.aux_layout}
    for s in (me, other):
        if s.get('key') in aux:
            s['aux'] = [n for n, _, _ in aux[s['key']].aux_layout]
    seats = lambda k: [seat_a, seat_b] if k % 2 == 0 else [seat_b, seat_a]  # noqa: E731
    names = lambda k: [me, other] if k % 2 == 0 else [other, me]  # noqa: E731
    kw = {'aux': aux, 'aux_render': package.active().replay_aux_of(game)} if aux else {}
    drv = RecordingDriver(game, games, seats, spec=spec, concurrency=min(conc, games), seed=seed, max_steps=5000,
                          hint=hname, **kw)
    t0 = time.time()
    drive(drv, evaluators, externals)
    vs = 'itself' if opp == 'self' else other['name']
    title = title or f'{me["name"]} vs {vs}'
    created = int(time.time())
    group = f'{title} · {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
    out = []
    for r in sorted(drv.take_replays(), key=lambda r: r['gid']):
        k = r.pop('gid')
        rep = {'format': FORMAT, 'game': game, 'spec': spec, 'seed': seed, 'game_index': k, 'created': created,
               'title': f'{title} #{k + 1}', 'group': group, 'selfplay': selfplay, 'hint': hint if hname else None,
               'source': {'checkpoint': os.path.abspath(ckpt), 'run': os.path.basename(run) if run else None, 'iter': it},
               'seats': [{x: v for x, v in s.items() if x != 'key'} for s in names(k)]}
        rep.update(r)
        out.append(rep)
    log(f'recorded {len(out)} games in {time.time() - t0:.0f}s ({drv.sims} simulations)')
    return out


def summary(rep):
    """The listing fields of a replay (the app's index row)."""
    sc = rep['scores']
    best = max(sc) if sc else None
    return {'format': rep['format'], 'game': rep['game'], 'title': rep['title'], 'group': rep['group'],
            'created': rep['created'], 'seats': [s['name'] for s in rep['seats']], 'kinds': [s['kind'] for s in rep['seats']],
            'scores': sc, 'winners': [i for i, x in enumerate(sc) if x == best], 'rounds': rep['rounds'], 'over': rep['over'],
            'frames': len(rep['frames']), 'spec': rep['spec'], 'selfplay': rep['selfplay'], 'source': rep['source']}


def save(rep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.fromtimestamp(rep['created']).strftime('%Y%m%d-%H%M%S')
    path = os.path.join(out_dir, f'{stamp}_{rep["game_index"]:03d}.json.gz')
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(rep, f, separators=(',', ':'))
    return path


def upload(rep, url, key, timeout=30):
    """POST one replay to the game app (`<url>/api/replays/upload`). -> the app's answer (dict)."""
    import urllib.request
    gz = gzip.compress(json.dumps(rep, separators=(',', ':')).encode('utf-8'))
    body = json.dumps({'meta': summary(rep), 'gz': base64.b64encode(gz).decode('ascii')}).encode('utf-8')
    req = urllib.request.Request(url.rstrip('/') + '/api/replays/upload', data=body, method='POST',
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}',
                                          'User-Agent': 'zero-replay/1'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def upload_key(pkg, key_file=None):
    """The app's upload key: `key_file`, else <PREFIX>_REPLAY_KEY, else <game repo>/.replay_key."""
    if key_file:
        with open(key_file) as f:
            return f.read().strip()
    k = pkg.env('REPLAY_KEY')
    if k:
        return k
    default = os.path.join(pkg.root, '.replay_key')
    if os.path.exists(default):
        with open(default) as f:
            return f.read().strip()
    raise SystemExit(f'--upload needs a key: --key-file, {pkg.env_prefix}_REPLAY_KEY or {default}')


def main(argv=None):
    pkg = package.active()
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument('ckpt')
    ap.add_argument('--opp', default=pkg.replay_opp or 'self',
                    help="'self', 'ext:<name>', a built-in bot ('random') or another checkpoint (.pt)")
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--sims', type=int, default=128)
    ap.add_argument('--opp-sims', type=int, default=None)
    ap.add_argument('--spec', default=None)
    ap.add_argument('--selfplay', action='store_true', help="the run's self-play noise and temperature")
    ap.add_argument('--hint', default=pkg.replay_hint or 'none', help="'ext:<name>' or 'none'")
    ap.add_argument('--chance-k', type=int, default=2)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=None, help='default: random')
    ap.add_argument('--conc', type=int, default=8, help='games played side by side')
    ap.add_argument('--title', default=None)
    ap.add_argument('--out', default=None, help='default: <run>/replays')
    ap.add_argument('--upload', nargs='?', const='default', default=None, help='upload to the game app (default URL: the package replay_url)')
    ap.add_argument('--key-file', default=None)
    ap.add_argument('--select', default=None, choices=['visits', 'cq'],
                    help="the network seats' move rule (default: the checkpoint's run's)")
    ap.add_argument('--select-min-visits', type=int, default=0)
    ap.add_argument('--cq-c-visit', type=float, default=None,
                    help="the 'cq' rule's completed-Q transform (default: the checkpoint's run's rule's)")
    ap.add_argument('--cq-c-scale', type=float, default=None)
    args = ap.parse_args(argv)
    tr = (args.cq_c_visit, args.cq_c_scale)
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    reps = record(args.ckpt, opp=args.opp, games=args.games, sims=args.sims, opp_sims=args.opp_sims, spec=args.spec,
                  selfplay=args.selfplay, hint=None if args.hint == 'none' else args.hint, chance_k=args.chance_k,
                  device=args.device, seed=seed, title=args.title, conc=args.conc,
                  select=(args.select, args.select_min_visits, *tr) if args.select or tr != (None, None) else None)
    run, _ = run_of(args.ckpt)
    out = args.out or os.path.join(run or '.', 'replays')
    for r in reps:
        path = save(r, out)
        s = summary(r)
        print(f'{path}: {" vs ".join(s["seats"])} {s["scores"]} ({s["frames"]} steps)')
    if args.upload:
        url = pkg.replay_url if args.upload == 'default' else args.upload
        if not url:
            raise SystemExit('--upload: no URL (give one, or set the game package replay_url)')
        key = upload_key(pkg, args.key_file)
        for r in reps:
            print(f'uploaded {r["title"]}: {upload(r, url, key)}')


if __name__ == '__main__':
    main()
