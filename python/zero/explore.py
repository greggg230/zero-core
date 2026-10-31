"""Self-play exploration (docs/search.md 3.9, docs/az.md 21): the run settings, the driver's `explore`
dict, and the metrics of a self-play phase.

Run settings (all off by default: '' = no exploration, the runs of before byte for byte):

  selfplay_explore            the habit tags to explore, comma-separated names the game declares
                              (`SearchGame::EXPLORE_TAGS`, `G.info(game)['explore_tags']`), or 'all'
  selfplay_explore_games      the share of self-play games eligible for an episode (0.25)
  selfplay_explore_q_margin   the gate: a deviation only within this much completed Q of the chosen
                              move (0.1: P(win) with margin_w 0)
  selfplay_explore_mask_z     1: records at or before the first deviation get z := q and no margin
                              target, a value-only sample without a root value by then is dropped (1)
  selfplay_explore_p          '' = each tag's default chance per opportunity; a number = every tag's;
                              'tag=p,...' = those tags' (the others their default)
  selfplay_explore_window     '' = each tag's default window (turns; 0 = the decision only); a number =
                              every tag's; 'tag=w,...' = those tags'
  selfplay_explore_pick       'first' = the game's first opportunity that triggers, of any explored tag,
                              starts its episode (the tags whose opportunities come early take most);
                              'game' = each eligible game draws one explored tag at its start, only it
                              can (the tags share the eligible games evenly)
  selfplay_explore_min_visits a deviation only to an option with at least this many root visits, and
                              every exploring search (an eligible game's recording seat at a decision
                              with an option of a tag it may still deviate from) takes this root minimum
                              visits (up to selfplay_explore_max_actions options, the budget allowing)
                              and at least the full budget `sims` - a fast draw raised, still
                              unrecorded (4; 0 = neither: an unvisited option gated on its v_mix)
  selfplay_explore_max_actions the largest root an exploring search floors (8)

A name the game does not declare, a p / window for a tag not explored, a value out of range: refused
(SystemExit), never a silent no-op. Only the recording self-play seat explores (never an evaluation,
an h2h, Elo, a replay, a league opponent); single-game runs only.

Metrics (`selfplay.explore`, a self-play phase's finished games - local and remote): games, eligible,
episodes, overridden decisions, masked / dropped records and their shares of the games' records, and
per tag: opportunities (before a game's episode), gate blocks (triggered, deviation refused), unsearched
(triggered, no alternative with min_visits visits), episodes, overridden decisions, the mean Q gap at the
first deviation (the chosen move's completed Q minus the deviation's), the episodes with a deviation to
an option under min_visits visits (0 with min_visits > 0), how the episodes ended, their turns; the
exploring searches, those floored, the simulations the raised budgets added and the forced walks (per
game too).
"""
import json

KEYS = ('selfplay_explore', 'selfplay_explore_games', 'selfplay_explore_q_margin', 'selfplay_explore_mask_z',
        'selfplay_explore_p', 'selfplay_explore_window', 'selfplay_explore_pick', 'selfplay_explore_min_visits',
        'selfplay_explore_max_actions')


def tags_of(info):
    """[(name, window, p)] the game declares (tag id k + 1 = entry k)."""
    return [(t[0], int(t[1]), float(t[2])) for t in info.get('explore_tags', [])]


def _per_tag(raw, names, what, conv):
    """'' -> {}; a number -> {every name: it}; 'tag=v,...' -> {tag: v} (the tags explored only)."""
    raw = str(raw if raw is not None else '').strip()
    if not raw:
        return {}
    if '=' not in raw:
        try:
            v = conv(raw)
        except ValueError:
            raise SystemExit(f'{what} {raw!r}: a number, or tag=value,... for the explored tags {names}') from None
        return {n: v for n in names}
    out = {}
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition('=')
        k = k.strip()
        if k not in names:
            raise SystemExit(f'{what}: {k!r} is not an explored tag (selfplay_explore: {names})')
        try:
            out[k] = conv(v.strip())
        except ValueError:
            raise SystemExit(f'{what}: {part!r} (tag=value)') from None
    return out


def driver_dict(cfg, info):
    """The driver's `explore` dict for a run's settings (None when selfplay_explore is ''), checked:
    SystemExit on a name the game does not declare or a bad value."""
    raw = str(cfg.get('selfplay_explore', '') or '').strip()
    if not raw:
        return None
    declared = tags_of(info)
    known = [t[0] for t in declared]
    if not known:
        raise SystemExit(f'selfplay_explore {raw!r}: {info.get("name")} declares no habit tags (SearchGame::EXPLORE_TAGS)')
    names = known if raw == 'all' else [n.strip() for n in raw.split(',') if n.strip()]
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f'selfplay_explore: {info.get("name")} declares no habit tag {bad} (its tags: {known})')
    if len(set(names)) != len(names):
        raise SystemExit(f'selfplay_explore {raw!r}: a tag twice')
    games = float(cfg.get('selfplay_explore_games', 0.25))
    if not 0.0 < games <= 1.0:
        raise SystemExit(f'selfplay_explore_games {games}: a share in (0, 1] (no exploration: selfplay_explore \'\')')
    margin = float(cfg.get('selfplay_explore_q_margin', 0.1))
    if not margin >= 0.0:
        raise SystemExit(f'selfplay_explore_q_margin {margin}: a number >= 0')
    mask = cfg.get('selfplay_explore_mask_z', 1)
    if mask not in (0, 1, True, False):
        raise SystemExit(f'selfplay_explore_mask_z {mask!r}: 0 or 1')
    ps = _per_tag(cfg.get('selfplay_explore_p', ''), names, 'selfplay_explore_p', float)
    ws = _per_tag(cfg.get('selfplay_explore_window', ''), names, 'selfplay_explore_window', int)
    pick = cfg.get('selfplay_explore_pick', 'first')
    if pick not in ('first', 'game'):
        raise SystemExit(f"selfplay_explore_pick {pick!r}: 'first' or 'game'")
    mv, ma = cfg.get('selfplay_explore_min_visits', 4), cfg.get('selfplay_explore_max_actions', 8)
    for k, v, lo in (('selfplay_explore_min_visits', mv, 0), ('selfplay_explore_max_actions', ma, 2)):
        if not isinstance(v, int) or isinstance(v, bool) or v < lo:
            raise SystemExit(f'{k} {v!r}: an integer >= {lo}')
    tags = []
    for n in names:
        _, w0, p0 = declared[known.index(n)]
        p, w = ps.get(n, p0), ws.get(n, w0)
        if not 0.0 < p <= 1.0:
            raise SystemExit(f'selfplay_explore_p: {n} {p}: a chance in (0, 1]')
        if w < 0:
            raise SystemExit(f'selfplay_explore_window: {n} {w}: turns >= 0 (0 = the decision only)')
        tags.append({'tag': n, 'p': float(p), 'window': int(w)})
    d = {'games': games, 'q_margin': margin, 'mask_z': bool(mask), 'min_visits': mv, 'max_actions': ma, 'tags': tags}
    if pick == 'game':
        d['game_tag'] = True                       # (absent: the driver's default, 'first')
    return d


def check(cfg, info, where=''):
    """The run's exploration settings -> the driver dict (None when off), else SystemExit."""
    try:
        d = driver_dict(cfg, info)
    except SystemExit as e:
        raise SystemExit(f'{where}{e}') from None
    if d is not None and (cfg.get('resolver') or where):
        raise SystemExit(f'{where}selfplay_explore: single-game runs only (not with co-training)')
    return d


def fingerprint(cfg, info):
    """What a remote worker's hello reports and the run checks (docs/az.md 21): the driver dict as
    canonical JSON, '' when off."""
    d = driver_dict(cfg, info)
    return '' if d is None else json.dumps(d, sort_keys=True)


# ------------------------------------------------------------------------------- metrics

def _tag_sums():
    return {'opportunities': 0, 'blocked': 0, 'unsearched': 0, 'episodes': 0, 'overridden': 0, 'q_gap_sum': 0.0,
            'alt_unvisited': 0, 'alt_under': 0, 'drawn': 0, 'ends': {}, 'turns': {}}


GAME_SUMS = ('games', 'eligible', 'samples', 'masked', 'dropped', 'searches', 'floored', 'raised', 'forced')


def new_tally(info, d):
    """An empty tally (raw sums, mergeable: `merge`) of a driver exploring as `d` (`driver_dict`) in
    the game of `info`: the rows' opportunities / blocked come in d's tag order."""
    explored = [t['tag'] for t in d['tags']]
    return dict({k: 0 for k in GAME_SUMS}, names=[t[0] for t in tags_of(info)], explored=explored,
                min_visits=int(d.get('min_visits', 0)), tags={n: _tag_sums() for n in explored})


def _tag_sums_keys():
    return ('opportunities', 'blocked', 'unsearched', 'episodes', 'overridden', 'q_gap_sum', 'alt_unvisited', 'alt_under',
            'drawn')


def tally(rows, acc):
    """Add the finished games' exploration (`row['explore']`) to the tally `acc`; rows without it (a
    driver that does not explore) are skipped."""
    for r in rows:
        x = r.get('explore')
        if x is None:
            continue
        acc['games'] += 1
        acc['eligible'] += bool(x['eligible'])
        acc['samples'] += int(r.get('samples', 0) or 0)
        acc['masked'] += int(x['masked'])
        acc['dropped'] += int(x['dropped'])
        for k in ('searches', 'floored', 'raised', 'forced'):
            acc[k] += int(x.get(k, 0))
        for k, n in enumerate(acc['explored']):
            acc['tags'][n]['opportunities'] += int(x['opportunities'][k])
            acc['tags'][n]['blocked'] += int(x['blocked'][k])
            acc['tags'][n]['unsearched'] += int(x.get('unsearched', [0] * len(acc['explored']))[k])
        if x.get('tag'):
            acc['tags'][acc['names'][int(x['tag']) - 1]]['drawn'] += 1        # (pick 'game': the game's tag)
        ep = x.get('episode')
        if ep is not None:
            t = acc['tags'][acc['names'][int(ep['tag']) - 1]]
            t['episodes'] += 1
            t['overridden'] += int(ep['overridden'])
            t['q_gap_sum'] += float(ep['q_gap'])
            t['alt_unvisited'] += int(ep['alt_visits']) == 0
            t['alt_under'] += int(ep['alt_visits']) < max(1, acc.get('min_visits', 0))
            t['ends'][ep['end']] = t['ends'].get(ep['end'], 0) + 1
            k = str(int(ep['turn']))
            t['turns'][k] = t['turns'].get(k, 0) + 1
    return acc


def merge(a, b):
    """Two tallies' sums (either may be None); tallies of different settings are refused."""
    if not a:
        return b
    if not b:
        return a
    if a['explored'] != b['explored']:
        raise ValueError(f'explore tallies of different tags: {a["explored"]} / {b["explored"]}')
    out = json.loads(json.dumps(a))
    for k in GAME_SUMS:
        out[k] += b.get(k, 0)
    for n, t in b['tags'].items():
        o = out['tags'][n]
        for k in _tag_sums_keys():
            o[k] += t.get(k, 0)
        for k in ('ends', 'turns'):
            for x, v in t[k].items():
                o[k][x] = o[k].get(x, 0) + v
    return out


def summary(acc):
    """The phase's `selfplay.explore` metrics from a tally."""
    tags = {}
    for n in acc['explored']:
        t = acc['tags'][n]
        e = t['episodes']
        tags[n] = {'drawn': t['drawn'], 'opportunities': t['opportunities'], 'blocked': t['blocked'],
                   'unsearched': t['unsearched'], 'episodes': e, 'overridden': t['overridden'],
                   'q_gap': round(t['q_gap_sum'] / e, 4) if e else None, 'alt_unvisited': t['alt_unvisited'],
                   'alt_under': t['alt_under'], 'ends': dict(sorted(t['ends'].items())),
                   'turns': dict(sorted(t['turns'].items(), key=lambda kv: int(kv[0])))}
    rec = acc['samples'] + acc['dropped']
    return {'games': acc['games'], 'eligible': acc['eligible'], 'episodes': sum(t['episodes'] for t in tags.values()),
            'overridden': sum(t['overridden'] for t in tags.values()), 'masked': acc['masked'],
            'dropped': acc['dropped'], 'masked_share': round(acc['masked'] / rec, 4) if rec else 0.0,
            'dropped_share': round(acc['dropped'] / rec, 4) if rec else 0.0, 'searches': acc['searches'],
            'floored': acc['floored'], 'raised_sims': acc['raised'], 'forced_walks': acc['forced'],
            'raised_sims_per_game': round(acc['raised'] / max(1, acc['games']), 1), 'tags': tags}
