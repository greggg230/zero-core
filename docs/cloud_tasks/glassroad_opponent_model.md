# Task: Glass Road opponent model (hand choice + hand beliefs) on zero-core

Start on `greggg230/zero-core` (+ `greggg230/glassroad-zero`) with this prompt:

---

You are improving how **Glass Road Zero** models the opponent's hidden hand. Repos: `greggg230/zero-core`
(shared framework: Rust crate `zero_search` in `crates/search`, PyO3 scaffolding `crates/py`, Python
package `python/zero`) and `greggg230/glassroad-zero` (the game: `engine/` incl. `engine/src/az/knowledge.rs`
determinization + `features.rs`; pins zero-core as submodule `third_party/zero-core`). Branches:
zero-core `opponent-model`, glassroad-zero `opponent-model` (its submodule pointing at your zero-core branch
commits while you work; final pin = your last zero-core commit). Never push to main; no PRs.

Read first: zero-core `DESIGN.md`, `docs/search.md`, `docs/az.md`, `docs/new_game.md`; glassroad-zero
`DESIGN.md` 3.5-3.7, `docs/azlayer.md` 5-7, and the diagnosis on glassroad-zero branch **`oracle`**:
`docs/hidden_info.md` + its code (`git log 822df27..oracle`). Key facts from it:
* Select (5-of-15 hand choice) is simultaneous but sequentialised (seat 0 first). The net could see its
  role via feature `opp_selected` and learned a weak second-chooser policy (seat 0 won ~90%). Fixed on
  main c1f56dc: `opp_selected` is always 0 (role-blind). A role-blind run `grz2p_rb` is training now.
* Oracle gap (true opponent hand in search, role-blind net, 128 sims, 1,200 games): +102 Elo overall,
  **+87 from card play**, +14 from select.
* Predictability (log-prob of the true opponent hand; self-play): uniform -5.27 nats; the net's own select
  policy from the opponent's view, restricted to consistent hands -0.72; + learned correction -0.41.
  CAVEAT: on the JS bot's real hands the net's policy is WORSE than uniform (-9.4 vs -8.0); only a heavily
  softened version beats uniform.

## Do, in order (each behind a default-off flag; defaults bit-identical: golden digests must not change)
1. **Port the oracle branch's tooling into zero-core / glassroad-zero** (it predates zero-core and touched
   `search/` + `py/`, which now live in zero-core): seat options `oracle_hands`, `oracle_p`, `class_sims`,
   `mask_features`; `DECISION_CLASSES` / `decision_class()` / `determinize_oracle()` trait hooks with
   defaults; `DriverCfg::trace`; the h2h by-seat + seat-adjusted Elo + `ext:jsbot` opponent (zero has
   `zero.h2h` from Caverna - merge features), `handprobe`. Tests as on the branch.
2. **(B) Opponent hand choice drawn from the role-blind policy inside search.** At a `select` decision of
   the opponent inside our tree (the opponent chooses after us), don't run PUCT: sample its picks from the
   network's prior (no Dirichlet noise) - a "policy-sampled chance node" for later-simultaneous choosers
   (glassroad-zero DESIGN 3.6; zero-core `docs/search.md` has a sketch). When WE choose second, the sampled
   world un-commits the opponent's pick so it is drawn after ours in the same way (no uniform picks).
   Game hook with a default (e.g. `fn policy_sampled(&self) -> bool`), tree switch, Rust/Python parity.
3. **(C) Belief prior during play.** Determinize the opponent's unplayed hand from a prior = the net's
   role-blind select policy from the opponent's perspective at the period start (hands sampled by 5
   ascending picks; cache per game per period; 64 sampled hands is ~2-3% extra inference at 128 sims),
   restricted to hands consistent with observations (played / followed / did-not-follow), mixed with a
   uniform floor (setting, e.g. `belief_mix` in [0, 1]; plus a temperature). Needs a driver request kind or
   a cheap in-driver path for the policy evaluations - design it in zero-core generically (a game can
   expose "belief samples" requests), document it.
4. **Measure** (Glass Road; the desktop orchestrator has the GPU + live checkpoints; if you lack them,
   produce scripts + exact commands and run what you can on CPU with small game counts):
   * h2h 1,200 games at 128 sims, same network (latest `grz2p_rb` checkpoint), new option vs baseline,
     for B alone, C alone (belief_mix 0.25 / 0.5 / 1.0), B+C;
   * vs the JS bot by seat (300 games) for each - the JS bot's hands are NOT the net's, so a confidently
     wrong prior must not hurt there: keep the uniform floor if it does;
   * report as in docs/hidden_info.md.
5. Docs: update glassroad-zero `docs/hidden_info.md` (move it to main), zero-core `docs/search.md`.

Guard rails: cargo test + pytest green in both repos; Glass Road parity gate unaffected; golden self-play
digests unchanged with every new flag off; `tools/smoke_all.sh` green. Final message: what was built, the
measurement tables, recommended settings for a training fork, exact commands for the orchestrator.
Commit messages end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
