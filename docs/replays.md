# Replays: what the AI did, in the game app

`zero.replay` plays games with a checkpoint and records, at every step, the app-renderable
state and what the search saw. A game app shows them with its own board renderer (Kingdom
Builder: the "AI replays" tab of kingdombuilder.greggg230.com, `src/ui/replay.js`).

```
python -m <pkg>.az.replays runs/<run>/models/iter_NNNN.pt [--opp ext:jsbot | self | random | other.pt]
    [--games 6] [--sims 128] [--selfplay] [--hint ext:jsbot | none] [--device cuda] [--upload]
```

* The checkpoint plays `--games` games against `--opp`, seats alternating. Evaluation
  settings by default (no noise, argmax); `--selfplay` uses the run's self-play noise and
  temperature (what training sees). The spec comes from the checkpoint's run
  (`runs/<run>/config.json`), so a run on a pinned setup records that setup.
* `--hint ext:<name>`: at every network decision on a state it has not seen yet, that external
  player is asked what it would do (never applied). Kingdom Builder: the JS bot's plan for the
  rest of the turn, shown next to the network's choice.
* Files: `runs/<run>/replays/<stamp>_<k>.json.gz`. `--upload` posts each to the game package's
  `replay_url` (`POST <url>/api/replays/upload`, `Authorization: Bearer <key>`; key from
  `--key-file`, `<PREFIX>_REPLAY_KEY` or `<game repo>/.replay_key`).
* Speed: the Rust tree driven from Python, 8 games side by side; Kingdom Builder 128 sims
  vs the JS bot ~5-10 s per game on the GPU.

## Format `zero-replay/1`

One JSON object per game (gzipped on disk and in transit):

| key | content |
|---|---|
| `format`, `game`, `spec`, `seed`, `game_index`, `created`, `title`, `group` | header; `group` = the batch (title + time) |
| `selfplay`, `hint`, `source` | settings; `source` = {checkpoint, run, iter} |
| `seats` | per seat {name, kind: 'nn' / 'ext' / 'bot', checkpoint?, run?, iter?, sims?} |
| `states` | the game's `export_state()` (the app's JSON state), parsed; a new entry only when it changed |
| `frames` | one per step, below |
| `final`, `result`, `scores`, `rounds`, `over` | the end |

A frame: `s` (state index before the step), `p` (seat to move), `k` (`'nn'`, `'ext'`,
`'forced'`, `'bot'`), `a` (the AI action id, except 'ext'), and

* `'nn'`: `acts` (legal action ids), `n` (root visits), `p0` (network priors, noise-free),
  `q` (mean utility of each option for the mover; null = unvisited), `v` (network utility of
  the root for the mover), `w` (search P(win) per seat), `sims`, `sampled` (a temperature pick),
  `hint` ({by, x}: the hint player's answer at this state);
* `'ext'`: `x` (the external player's answer: Kingdom Builder `{p, actions: [{resp}], h}`).

Compound decisions (a tile, then its target hexes) are several frames on the same state.
Sizes: Kingdom Builder ~10-12 KB per game gzipped (the states repeat a lot).

## A game app's side (Kingdom Builder as the example)

* Worker: table `replays (id, created, grp, meta, gz)`; `POST /api/replays/upload` with its own
  key (`REPLAY_KEY` secret, before the password gate, like the alert relay); signed-in
  `GET /api/replays` (list of `meta` = `zero.replay.summary`), `GET /api/replays/<id>` ({meta,
  gz: base64}), `POST /api/replays/<id>/delete`, `POST /api/replays/delete-group`.
* Viewer: decode (`DecompressionStream('gzip')`), render `states[frame.s]` with the normal board,
  map the AI's action ids to the game's terms (cells, tiles, end turn), heat map the search's
  share per option (or the network prior), show the hint and a win-chance strip from `w`.
* Deep link: `?replay=<id>`.
