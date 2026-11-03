//! The driver's worker threads: advancing games (search, bots, external requests), batch
//! assembly, finished games and training records.
use super::*;
use crate::lookahead::{Answers, Outcome, RowSink, Step as LookStep};
use crate::tree::{leaf_values_team, transition_with, Scratch as SimScratch, Sim, Walk};

struct WScratch<G: SearchGame> {
    game: Option<G>,
    sim: SimScratch,
}

/// Queue-job flag: fill job of chunk `c & !FILL` (else: search job of chunk c).
const FILL: usize = 1 << 30;

pub(super) fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "?".into())
}

pub(super) fn worker<G: SearchGame>(sh: Arc<Shared<G>>) {
    let mut scratch = WScratch { game: None, sim: SimScratch::default() };
    loop {
        let job = {
            let mut q = lock(&sh.queue);
            loop {
                if sh.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(j) = q.pop_front() {
                    break j;
                }
                q = sh.queue_cv.wait(q).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let (g, c) = (job.0 as usize, job.1 as usize);
        if c & FILL != 0 {
            let t = Instant::now();
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| fill_chunk(&sh, g, c & !FILL))) {
                sh.fail(format!("batch assembly panicked: {}", panic_msg(p)));
            }
            sh.counters.assemble_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if sh.groups[g].fill_remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                finish_assembly(&sh, g);
            }
            continue;
        }
        let t = Instant::now();
        // a game's panic is caught per game inside (`run_chunk`): this catches the rest
        match catch_unwind(AssertUnwindSafe(|| run_chunk(&sh, g, c, &mut scratch))) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => sh.fail(e),
            Err(p) => {
                sh.fail(format!("search worker panicked: {}", panic_msg(p)));
                scratch.game = None;
            }
        }
        sh.counters.busy_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let grp = &sh.groups[g];
        if grp.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
            // the group's last chunk: lay out its batches, then fill them in parallel (fill
            // jobs go to the front of the queue: the caller is waiting for this group)
            let t = Instant::now();
            let n = if sh.failed.load(Ordering::SeqCst) {
                0
            } else {
                catch_unwind(AssertUnwindSafe(|| plan_assembly(&sh, g))).unwrap_or_else(|p| {
                    sh.fail(format!("batch assembly panicked: {}", panic_msg(p)));
                    0
                })
            };
            sh.counters.assemble_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if n == 0 {
                finish_assembly(&sh, g);
            } else {
                grp.fill_remaining.store(n, Ordering::SeqCst);
                let mut q = lock(&sh.queue);
                for c in (0..n).rev() {
                    q.push_front((g as u32, (c | FILL) as u32));
                }
                drop(q);
                sh.queue_cv.notify_all();
            }
        }
    }
}

#[derive(Default)]
struct Tally {
    sims: u64,
    leaves: u64,
    ext: u64,
    lost: u64,
    restarts: u64,
    recorded: u64,
    minor: u64,
    view_rows: u64,
    view_recorded: u64,
    belief_rows: u64,
    /// Branching and archive starts (docs/search.md 2.11).
    branches: u64,
    dev_moves: u64,
    starts: u64,
}

fn run_chunk<G: SearchGame>(sh: &Shared<G>, g: usize, c: usize, scratch: &mut WScratch<G>) -> Result<(), String> {
    let cfg = &sh.cfg;
    let grp = &sh.groups[g];
    let mut ch = lock(&grp.chunks[c]);
    let ch = &mut *ch;
    for b in ch.bufs.iter_mut() {
        b.clear();
    }
    let evals = read(&grp.evals);
    let epoch = sh.epoch.load(Ordering::SeqCst);
    let net = sh.net.load(Ordering::SeqCst);
    let mut n = Tally::default();
    for slot in ch.slots.iter_mut() {
        let gid = slot.start.or(Some(slot.gid)).filter(|&x| x != NO_GAME);
        let t0 = Instant::now();
        // a panic inside the game (an engine bug) is caught here, per game, so the message
        // names the game to replay and the chunk's lock is never poisoned
        let r = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            if let Some(gid) = slot.start.take() {
                init_slot(sh, slot, gid, &mut n)?;
                slot.epoch = epoch;
                slot.net = net;
            }
            if slot.game.is_none() {
                return Ok(());
            }
            if slot.epoch != epoch {
                n.restarts += slot.target.is_some() as u64;
                n.lost += restart_search(cfg, slot)?;
                slot.epoch = epoch;
                slot.net = net;
            }
            match slot.pending.take() {
                Some(Pending::Leaf { key, local, seat, node, vrow, brow }) => {
                    let Some(Eval::Nn { opt_off, priors, value, margin, bases, aux, aux_row }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing network evaluation".into());
                    };
                    let b = (bases[c] + local) as usize;
                    let lo = opt_off[b] as usize;
                    let hi = opt_off[b + 1] as usize;
                    // the priors from the leaf's row; its value from the value-only row after it
                    // (`LeafView::Root` at another seat's leaf, docs/search.md 3.6), else its own
                    let v = if vrow { b + 1 } else { b };
                    apply_eval(cfg, slot, seat, node, vrow, &priors[lo..hi], &value[v * MAX_SEATS..(v + 1) * MAX_SEATS], &margin[v * MAX_SEATS..(v + 1) * MAX_SEATS])?;
                    n.sims += 1;
                    if brow {
                        // the fresh root's belief row after it (docs/search.md 3.8)
                        set_belief(cfg, slot, seat, belief_of(aux, aux_row, b + 1 + vrow as usize, G::aux_width())?)?;
                    }
                }
                Some(Pending::Belief { key, local, seat }) => {
                    let Some(Eval::Nn { bases, aux, aux_row, .. }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing network evaluation".into());
                    };
                    set_belief(cfg, slot, seat, belief_of(aux, aux_row, (bases[c] + local) as usize, G::aux_width())?)?;
                }
                Some(Pending::Ext { key, local }) => {
                    let Some(Eval::Ext { payloads, bases }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing external answer".into());
                    };
                    slot.ext_payload = Some(payloads[(bases[c] + local) as usize].clone());
                }
                Some(Pending::Look { key, local }) => {
                    // a lookahead's round (docs/search.md 3.9): its rows' answers
                    let Some(Eval::Nn { opt_off, priors, value, bases, aux, aux_row, .. }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing network evaluation".into());
                    };
                    let ans = Answers { base: (bases[c] + local) as usize, opt_off, priors, value, aux, aux_row, aux_width: G::aux_width() };
                    slot.look.as_mut().ok_or("internal: a lookahead answer without a lookahead")?.answer(&ans)?;
                }
                None => {}
            }
            advance_slot(sh, slot, &mut ch.bufs, &mut ch.results, &mut ch.samples, &mut ch.parked, scratch, &mut n)
        }))
        .unwrap_or_else(|p| {
            // the scratch world may be mid-transition
            scratch.game = None;
            Err(format!("search worker panicked: {}", panic_msg(p)))
        });
        if let Some(gid) = gid {
            let ns = t0.elapsed().as_nanos() as u64;
            let k = &sh.counters;
            if ns > k.step_max_ns.load(Ordering::Relaxed) && k.step_max_ns.fetch_max(ns, Ordering::Relaxed) < ns {
                k.step_max_gid.store(gid as u64, Ordering::Relaxed);
            }
            if ns > SLOW_STEP_NS {
                k.slow_steps.fetch_add(1, Ordering::Relaxed);
            }
        }
        if let Err(e) = r {
            let tag = slot.tag.as_ref().or(slot.setup.as_ref().map(|s| &s.tag)).map(|t| format!(" (queued setup {t:?})")).unwrap_or_default();
            return Err(format!("game {}{tag} (driver seed {}): {e}", gid.map(|x| x as i64).unwrap_or(-1), cfg.seed));
        }
    }
    let k = &sh.counters;
    k.sims.fetch_add(n.sims, Ordering::Relaxed);
    k.leaves.fetch_add(n.leaves, Ordering::Relaxed);
    k.ext_requests.fetch_add(n.ext, Ordering::Relaxed);
    k.restart_sims.fetch_add(n.lost, Ordering::Relaxed);
    k.restarts.fetch_add(n.restarts, Ordering::Relaxed);
    k.recorded.fetch_add(n.recorded, Ordering::Relaxed);
    k.minor.fetch_add(n.minor, Ordering::Relaxed);
    if n.view_rows > 0 || n.view_recorded > 0 {
        k.view_rows.fetch_add(n.view_rows, Ordering::Relaxed);
        k.view_recorded.fetch_add(n.view_recorded, Ordering::Relaxed);
    }
    if n.belief_rows > 0 {
        k.belief_rows.fetch_add(n.belief_rows, Ordering::Relaxed);
    }
    if n.branches > 0 || n.dev_moves > 0 || n.starts > 0 {
        k.branches.fetch_add(n.branches, Ordering::Relaxed);
        k.dev_moves.fetch_add(n.dev_moves, Ordering::Relaxed);
        k.starts.fetch_add(n.starts, Ordering::Relaxed);
    }
    Ok(())
}

/// Row `r`'s auxiliary output (a root belief row's) in an evaluation.
fn belief_of<'a>(aux: &'a [f32], aux_row: &[u32], r: usize, w: usize) -> Result<&'a [f32], String> {
    let k = aux_row.get(r).copied().unwrap_or(NO_AUX);
    if k == NO_AUX {
        return Err(format!("internal: row {r} is not a root belief row"));
    }
    Ok(&aux[k as usize * w..(k as usize + 1) * w])
}

/// The search of seat `seat` takes its root's belief `pred` (the network's auxiliary output for the
/// root's features, docs/search.md 3.8): every simulation from now on deals its world by it.
fn set_belief<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, seat: usize, pred: &[f32]) -> Result<(), String> {
    let s = nn_seat(cfg, slot.gid, seat).ok_or("internal: a belief for a non-network seat")?;
    let live = slot.game.as_ref().ok_or("internal: a belief without a game")?;
    let tree = slot.trees[seat].as_mut().ok_or("internal: a belief without a tree")?;
    let b = Belief { pred: pred.to_vec(), tilt: s.belief_tilt, clip: s.belief_clip };
    tree.set_belief(Some(b), s.belief_mix, live, &mut slot.search_rng);
    Ok(())
}

fn game_seats(cfg: &DriverCfg, gid: u32) -> &[u16] {
    &cfg.games[gid as usize % cfg.games.len()]
}

fn seat_cfg(cfg: &DriverCfg, gid: u32, p: usize) -> &SeatCfg {
    &cfg.seats[game_seats(cfg, gid)[p] as usize]
}

fn nn_seat(cfg: &DriverCfg, gid: u32, p: usize) -> Option<&NnSeat> {
    match seat_cfg(cfg, gid, p) {
        SeatCfg::Nn(s) => Some(s),
        _ => None,
    }
}

fn is_external(cfg: &DriverCfg, gid: u32, p: usize) -> bool {
    matches!(seat_cfg(cfg, gid, p), SeatCfg::External(_))
}

/// A new search epoch began (`Driver::resume` with another network): drop the slot's trees
/// and pending leaf and restart the search of its current decision from an empty tree,
/// keeping that decision's playout-cap draw. -> simulations thrown away.
fn restart_search<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>) -> Result<u64, String> {
    let leaf = matches!(slot.pending, Some(Pending::Leaf { .. }));
    let lost = slot.sims as u64 + leaf as u64;
    // (a root belief asked of the old network is asked again of the new one: the reset tree has none)
    if leaf || matches!(slot.pending, Some(Pending::Belief { .. })) {
        slot.pending = None;
    }
    // (a lookahead begins again with the new network: docs/search.md 3.9)
    if matches!(slot.pending, Some(Pending::Look { .. })) {
        slot.pending = None;
    }
    slot.look = None;
    slot.drop_trees();
    slot.sims = 0;
    if slot.target.is_some() {
        slot.ensure_legal();
        let game = slot.game.as_ref().unwrap();
        let p = game.to_move().ok_or("internal: search target set on a finished game")?;
        let seat = nn_seat(cfg, slot.gid, p).ok_or("internal: search target set for a non-network seat")?;
        let mut t = slot.spare.pop().unwrap_or_default();
        t.reset(game, p, &slot.legal, seat.redet, &mut slot.search_rng);
        if slot.dev_now {
            slot.target = Some(1);                 // (the deviation's root evaluation again)
        }
        t.set_root_min(root_floor(seat.root_min_visits, seat.root_min_visits_max_actions, slot.full, slot.legal.len(), slot.target.unwrap()));
        // (a branch's deviation search is the root's evaluation alone: no Gumbel schedule)
        if let (RootSearch::Gumbel { m, c_visit, c_scale, .. }, false) = (seat.root_search, slot.dev_now) {
            // the decision's Gumbel search again, its noise kept (as its playout-cap draw)
            t.set_gumbel(m, slot.target.unwrap(), &slot.gnoise, c_visit, c_scale);
        }
        slot.trees[p] = Some(t);
    }
    Ok(lost)
}

/// Start game `gid` in `slot` (with the spec of its queued setup, if any): every random
/// stream derives from (driver seed, gid) - with `DriverCfg::paired`, from (driver seed, gid / 2):
/// the two games of a pair share them.
fn init_slot<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, gid: u32, n_: &mut Tally) -> Result<(), String> {
    let cfg = &sh.cfg;
    let setup = slot.setup.take();
    let mut spec = setup.as_ref().map_or(&sh.spec, |s| &s.spec);
    let stream = if cfg.paired { gid / 2 } else { gid };
    // archive starts (docs/search.md 2.11): the game's own start stream, never `base`
    slot.start_ix = None;
    if let Some(sc) = &cfg.starts {
        let mut r = Rng::new(split64(cfg.seed ^ split64(stream as u64 + 1) ^ START_SALT));
        if r.unit() < sc.p {
            let j = r.below(sh.starts.len() as u32) as usize;
            spec = &sh.starts[j];
            slot.start_ix = Some(j as u32);
            n_.starts += 1;
        }
    }
    // branched games (docs/search.md 2.11): the game's own branch stream, never `base`; only a game
    // that draws a branch keeps one
    slot.fork = None;
    slot.branch = None;
    slot.dev_left = 0;
    slot.dev_now = false;
    if let Some(bc) = &cfg.branch {
        let mut r = Rng::new(split64(cfg.seed ^ split64(stream as u64 + 1) ^ BRANCH_SALT));
        if r.unit() < bc.p {
            slot.fork = Some(Box::new(Fork { rng: r, seen: 0, point: None }));
        }
    }
    let mut base = Rng::new(split64(cfg.seed ^ split64(stream as u64 + 1)));
    let seats = game_seats(cfg, gid);
    let n = seats.len();
    slot.tag = setup.as_ref().map(|s| s.tag.clone());
    let game = G::new_game(spec, n, base.next_u64())?;
    let team = game.team_size();
    #[allow(clippy::manual_is_multiple_of)] // (rustc < 1.87 on other machines)
    let divides = team != 0 && n % team == 0;
    if !divides {
        return Err(format!("{}: team_size {team} does not divide {n} players", G::NAME));
    }
    slot.gid = gid;
    slot.n = n;
    slot.rng = Rng::new(base.next_u64());
    slot.noise_rng = Rng::new(base.next_u64());
    slot.search_rng = Rng::new(base.next_u64());
    // one seed per seat whatever its kind, so a bot's stream does not depend on the other seats
    let seeds: Vec<u64> = seats.iter().map(|_| base.next_u64()).collect();
    slot.bots = seats
        .iter()
        .zip(seeds)
        .map(|(&s, seed)| match &cfg.seats[s as usize] {
            SeatCfg::Bot(name) => make_bot::<G>(spec, name, seed),
            _ => None,
        })
        .collect();
    slot.drop_trees();
    slot.game = Some(game);
    slot.legal_ok = false;
    slot.sims = 0;
    slot.target = None;
    slot.full = true;
    slot.minor = false;
    slot.belief = false;
    slot.decisions = 0;
    slot.ext_moves = 0;
    slot.steps = 0;
    slot.records.clear();
    slot.rq.clear();
    slot.view_wait.clear();
    slot.pending = None;
    slot.ext_payload = None;
    slot.moves.clear();
    slot.look = None;
    slot.look_mcts = false;
    slot.look_guard = None;
    Ok(())
}

/// Expand the slot's pending leaf with its evaluation and back it up. `vrow`: `value` / `margin`
/// are relative to the tree's viewer (a value-only row, `LeafView::Root`), else to the leaf's
/// decider.
#[allow(clippy::too_many_arguments)]
fn apply_eval<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, seat: usize, node: u32, vrow: bool, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String> {
    let s = nn_seat(cfg, slot.gid, seat).ok_or("internal: pending leaf of a non-network seat")?;
    let tree = slot.trees[seat].as_mut().ok_or("internal: pending leaf without a tree")?;
    if priors.len() != tree.acts(node).len() {
        return Err("internal: evaluation option count mismatch".into());
    }
    let vp = if vrow { tree.viewer } else { tree.player(node) };
    let (util, win) = leaf_values_team(value, margin, vp, slot.n, s.margin_w, tree.team);
    tree.expand_leaf(node, priors, &util, &win);
    tree.backup(&slot.path, &util, &win);
    // (root Dirichlet noise: PUCT's; a Gumbel search has its Gumbel noise instead, docs/search.md 3.7)
    if node == tree.root && s.selfplay && slot.full && s.root_search == RootSearch::Puct {
        let r = tree.root;
        tree.add_noise(r, s.alpha, s.eps, &mut slot.noise_rng);
    }
    slot.sims += 1;
    Ok(())
}

/// Play the slot's game until its search needs a network evaluation, an external seat is to
/// move, or the game ends (or parks at a deferred terminal).
#[allow(clippy::too_many_arguments)]
fn advance_slot<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, bufs: &mut [LeafBuf], results: &mut Vec<GameResult>, samples: &mut Vec<Record>, parked: &mut Vec<Parked<G>>, scratch: &mut WScratch<G>, n: &mut Tally) -> Result<(), String> {
    let cfg = &sh.cfg;
    loop {
        let g = slot.game.as_ref().unwrap();
        let Some(p) = g.to_move().filter(|_| slot.steps < cfg.max_steps) else {
            finish(sh, slot, results, samples, parked, n)?;
            if slot.game.is_some() {
                // the game drew a branch (docs/search.md 2.11): its slot plays it now
                continue;
            }
            return Ok(());
        };
        if let SeatCfg::External(key) = seat_cfg(cfg, slot.gid, p) {
            if let Some(pl) = slot.ext_payload.take() {
                apply_external(slot, &pl, &mut scratch.sim.key)?;
                continue;
            }
            let buf = &mut bufs[*key as usize];
            let local = buf.len() as u32;
            buf.states.push(g.export_state());
            buf.player.push(p as i32);
            buf.n_players.push(slot.n as i32);
            buf.gid.push(slot.gid as i64);
            slot.pending = Some(Pending::Ext { key: *key, local });
            n.ext += 1;
            return Ok(());
        }
        slot.ensure_legal();
        if slot.legal.is_empty() {
            return Err(format!("seat {p} to move without legal actions"));
        }
        if slot.legal.len() == 1 {
            let a = slot.legal[0];
            apply(cfg, slot, a, &mut scratch.sim.key)?;
            continue;
        }
        if let Some(bot) = slot.bots[p].as_mut() {
            let a = bot.choose(slot.game.as_ref().unwrap(), &slot.legal);
            apply(cfg, slot, a, &mut scratch.sim.key)?;
            continue;
        }
        let seat = nn_seat(cfg, slot.gid, p).ok_or("internal: a seat without a player")?;
        if let Some(lc) = seat.look.as_deref() {
            // a lookahead seat (docs/search.md 3.9): the lookahead decides, unless it fell back to the
            // seat's MCTS search for this decision
            if !slot.look_mcts {
                if look_step(sh, slot, seat, lc, p, bufs, &mut scratch.sim.key)? {
                    return Ok(());
                }
                continue;
            }
        }
        if slot.target.is_none() {
            if slot.dev_left > 0 && !slot.game.as_ref().unwrap().minor_decision() {
                // a branch's deviation move (docs/search.md 2.11): the network's priors alone
                start_dev(slot, seat, p);
            } else {
                start(slot, seat, p);
            }
        }
        if slot.sims >= slot.target.unwrap() {
            n.minor += slot.minor as u64;
            decide(cfg, slot, seat, p, &mut scratch.sim.key, n, &sh.counters.look)?;
            continue;
        }
        let walk = Walk { c_puct: seat.c_puct, chance_k: seat.chance_k, margin_w: seat.margin_w };
        let live = slot.game.as_ref().unwrap();
        let sg = scratch.game.get_or_insert_with(|| live.clone());
        let tree = slot.trees[p].as_mut().unwrap();
        // the root's belief (docs/search.md 3.8): asked once per search, from the root's features
        // (the live decision, the decider's own view: the same in every world). A fresh root's first
        // simulation is its own leaf (nothing dealt is read) and the belief row goes with it; a
        // reused root's simulations wait for the row alone.
        let brow = slot.belief && tree.belief().is_none();
        if brow && tree.expanded(tree.root) {
            let buf = &mut bufs[seat.key as usize];
            let local = buf.len() as u32;
            push_belief_row(buf, live, p, slot.n, slot.gid);
            n.belief_rows += 1;
            slot.pending = Some(Pending::Belief { key: seat.key, local, seat: p });
            return Ok(());
        }
        std::mem::swap(&mut slot.path, &mut scratch.sim.path);
        let r = tree.simulate(live, sg, &mut slot.search_rng, &walk, &mut scratch.sim);
        std::mem::swap(&mut slot.path, &mut scratch.sim.path);
        match r? {
            Sim::Terminal => {
                slot.sims += 1;
                n.sims += 1;
            }
            Sim::Leaf { node, player } => {
                let buf = &mut bufs[seat.key as usize];
                let local = buf.len() as u32;
                sg.encode(player, buf.row_x(G::NUM_FEATURES));
                buf.seal();
                let acts = tree.acts(node);
                buf.nopt.push(acts.len() as i32);
                buf.opt_action.extend(acts.iter().map(|&a| a as i32));
                buf.player.push(player as i32);
                buf.n_players.push(slot.n as i32);
                buf.gid.push(slot.gid as i64);
                buf.extra.push(ROW_LEAF);
                // root-view leaf values (docs/search.md 3.6): another seat's leaf also sends a
                // value-only row (no options) of the tree's viewer, the root decider, right after it
                let vrow = seat.leaf_value_view == LeafView::Root && player != tree.viewer;
                if vrow {
                    sg.encode(tree.viewer, buf.row_x(G::NUM_FEATURES));
                    buf.seal();
                    buf.nopt.push(0);
                    buf.player.push(tree.viewer as i32);
                    buf.n_players.push(slot.n as i32);
                    buf.gid.push(slot.gid as i64);
                    buf.extra.push(ROW_VALUE);
                    n.view_rows += 1;
                }
                if brow {
                    // (an unexpanded root: this leaf is the root itself)
                    if node != tree.root {
                        return Err("internal: a fresh root's first leaf is not the root".into());
                    }
                    push_belief_row(buf, live, p, slot.n, slot.gid);
                    n.belief_rows += 1;
                }
                slot.pending = Some(Pending::Leaf { key: seat.key, local, seat: p, node, vrow, brow });
                n.leaves += 1;
                return Ok(());
            }
        }
    }
}

/// A lookahead's rows into a key's leaf buffer (docs/search.md 3.9): one row each, `encode(viewer)`
/// with its options (none: value-only), kind as given.
struct LeafSink<'a> {
    buf: &'a mut LeafBuf,
    gid: u32,
    n: usize,
    /// The last row's features may still be written (sealed at the next row, or on drop).
    open: bool,
}

impl Drop for LeafSink<'_> {
    fn drop(&mut self) {
        if self.open {
            self.buf.seal();
        }
    }
}

impl<G: SearchGame> RowSink<G> for LeafSink<'_> {
    fn push_row(&mut self, viewer: usize, opts: &[u16], kind: u8) -> &mut [f32] {
        if self.open {
            self.buf.seal();
        }
        self.open = true;
        let b = &mut *self.buf;
        b.nopt.push(opts.len() as i32);
        b.opt_action.extend(opts.iter().map(|&a| a as i32));
        b.player.push(viewer as i32);
        b.n_players.push(self.n as i32);
        b.gid.push(self.gid as i64);
        b.extra.push(kind);
        b.row_x(G::NUM_FEATURES)
    }
    fn pop_row(&mut self) {
        // (only the row just pushed: the lookahead pops a duplicate observation's row at once)
        debug_assert!(self.open, "pop_row without an open row");
        self.open = false;
        let b = &mut *self.buf;
        let k = b.nopt.pop().unwrap_or(0) as usize;
        b.opt_action.truncate(b.opt_action.len() - k);
        b.player.pop();
        b.n_players.pop();
        b.gid.pop();
        b.extra.pop();
        b.unrow(G::NUM_FEATURES);
    }
}

/// The lookahead of seat `p`'s pending decision (docs/search.md 3.9): created at the decision's first
/// step (its belief from the seat's settings, its worlds dealt from the slot's search stream), stepped
/// until it needs rows (-> true: the slot waits for the batch) or decides: its move is played, or
/// (every stage over the budget, `Last::Mcts`) the decision goes to the seat's MCTS search (-> false
/// either way: the slot goes on). Its counters go to the driver's `LookTotals`.
#[allow(clippy::too_many_arguments)]
fn look_step<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, seat: &NnSeat, lc: &LookCfg, p: usize, bufs: &mut [LeafBuf], key: &mut Vec<u64>) -> Result<bool, String> {
    let t0 = Instant::now();
    if lc.guard > 0.0 && slot.look.is_none() && slot.game.as_ref().is_some_and(|g| g.minor_decision()) {
        // (the guarded player: a minor decision is the MCTS search's alone)
        slot.look_mcts = true;
        return Ok(false);
    }
    if slot.look.is_none() {
        let live = slot.game.as_ref().ok_or("internal: a lookahead without a game")?;
        let belief = (seat.belief_tilt > 0.0).then_some((seat.belief_tilt, seat.belief_clip, seat.belief_mix));
        slot.look = Some(Box::new(Look::new(lc.clone(), live, p, belief, Vec::new(), Vec::new(), false)?));
    }
    let buf = &mut bufs[seat.key as usize];
    let local = buf.len() as u32;
    let mut sink = LeafSink { buf, gid: slot.gid, n: slot.n, open: false };
    let look = slot.look.as_mut().unwrap();
    let r = look.step(&mut sink, &mut slot.search_rng);
    drop(sink);
    let busy = t0.elapsed().as_secs_f64();
    let r = r?;
    if let LookStep::Rows(_) = r {
        lock(&sh.counters.look).busy_s += busy;
        slot.pending = Some(Pending::Look { key: seat.key, local });
        return Ok(true);
    }
    let look = slot.look.take().unwrap();
    let outcome = look.outcome().ok_or("internal: a lookahead done without an outcome")?;
    {
        let mut t = lock(&sh.counters.look);
        t.busy_s += busy;
        t.add(look.stats(), outcome == Outcome::Mcts);
    }
    match outcome {
        Outcome::Act(_) if lc.guard > 0.0 => {
            // the guarded player: the MCTS search decides, against the lookahead's values (`decide`)
            let mut l = look;
            let means = l.options()?.into_iter().map(|o| (o.act, o.mean)).collect();
            slot.look_guard = Some((means, lc.guard));
            slot.look_mcts = true;
        }
        Outcome::Act(a) => {
            slot.decisions += 1;
            apply(&sh.cfg, slot, a, key)?;
        }
        Outcome::Mcts => slot.look_mcts = true,
    }
    Ok(false)
}

/// The guarded player's move (`LookCfg::guard`): the MCTS move `a`, unless the lookahead's best root
/// option (the first best mean) has a mean over the MCTS move's above the margin -> (the move, whether
/// the lookahead's). An MCTS move the lookahead did not expand (a capped root) stays.
fn guard_move(means: &[(u16, f64)], margin: f64, a: u16) -> (u16, bool) {
    let Some(&(_, m)) = means.iter().find(|x| x.0 == a) else { return (a, false) };
    let mut best = means[0];
    for &x in means {
        if x.1 > best.1 {
            best = x;
        }
    }
    if best.0 != a && best.1 - m > margin {
        (best.0, true)
    } else {
        (a, false)
    }
}

/// A root belief row (`ROW_BELIEF`, docs/search.md 3.8): `encode(p)` of the live game, no options.
fn push_belief_row<G: SearchGame>(buf: &mut LeafBuf, live: &G, p: usize, n: usize, gid: u32) {
    live.encode(p, buf.row_x(G::NUM_FEATURES));
    buf.seal();
    buf.nopt.push(0);
    buf.player.push(p as i32);
    buf.n_players.push(n as i32);
    buf.gid.push(gid as i64);
    buf.extra.push(ROW_BELIEF);
}

/// Start the search for the pending decision of seat `p` (legal actions in `slot.legal`).
fn start<G: SearchGame>(slot: &mut Slot<G>, seat: &NnSeat, p: usize) {
    slot.full = seat.sims_fast.is_none() || slot.rng.unit() < seat.p_full;
    slot.target = Some(if slot.full { seat.sims } else { seat.sims_fast.unwrap() });
    let game = slot.game.as_ref().unwrap();
    // a minor decision (self-play seats with sims_minor): its own budget, never recorded (the
    // playout-cap draw above is made all the same, so the streams do not depend on it)
    slot.minor = seat.selfplay && seat.sims_minor.is_some() && game.minor_decision();
    if slot.minor {
        slot.full = false;
        slot.target = seat.sims_minor;
    }
    // a lighter decision class (`SearchGame::sims_divisor`): the drawn budget divided
    let div = game.sims_divisor();
    if div > 1 {
        slot.target = slot.target.map(|t| (t / div).max(1));
    }
    // the root's belief (docs/search.md 3.8): full searches, or every one with belief_fast; a
    // reused tree's belief was its last decision's (`Tree::advance` cleared it)
    slot.belief = seat.takes_belief(slot.full);
    match slot.trees[p].as_mut() {
        None => {
            let mut t = slot.spare.pop().unwrap_or_default();
            t.reset(game, p, &slot.legal, seat.redet, &mut slot.search_rng);
            slot.trees[p] = Some(t);
        }
        Some(t) => {
            t.set_base(game, &mut slot.search_rng);
            if seat.selfplay && slot.full && t.expanded(t.root) && seat.root_search == RootSearch::Puct {
                let r = t.root;
                t.add_noise(r, seat.alpha, seat.eps, &mut slot.noise_rng);
            }
        }
    }
    // the root minimum visits of this search (docs/search.md 3.5; 0 = off: the tree's walks
    // are PUCT's alone; a reused root's visits count toward the floor)
    let k = root_floor(seat.root_min_visits, seat.root_min_visits_max_actions, slot.full, slot.legal.len(), slot.target.unwrap());
    let t = slot.trees[p].as_mut().unwrap();
    t.set_root_min(k);
    // the Gumbel root search (docs/search.md 3.7): every search of the seat (full, fast, minor). Its
    // noise replaces the root Dirichlet noise and the temperature: a self-play seat's first
    // `temp_moves` searched decisions are searched with it, later ones (and an evaluation seat's
    // with scale 0) without. The schedule is that of the search's simulations (a fresh root's own
    // evaluation takes its first one: its walks are the schedule's first target - 1); a reused root's
    // earlier visits are not in it.
    if let RootSearch::Gumbel { m, scale, c_visit, c_scale } = seat.root_search {
        let s = if seat.selfplay && slot.decisions >= seat.temp_moves { 0.0 } else { scale };
        slot.gnoise.clear();
        if s > 0.0 {
            for _ in 0..slot.legal.len() {
                slot.gnoise.push(s * crate::noise::gumbel(&mut slot.noise_rng));
            }
        }
        t.set_gumbel(m, slot.target.unwrap(), &slot.gnoise, c_visit, c_scale);
    }
}

/// Start a branch's deviation search of seat `p`'s decision (docs/search.md 2.11): the root's own
/// evaluation alone (one simulation; none when a reused root is expanded already) for the network's
/// priors - no noise, floor, Gumbel schedule or belief, never recorded.
fn start_dev<G: SearchGame>(slot: &mut Slot<G>, seat: &NnSeat, p: usize) {
    slot.full = false;
    slot.minor = false;
    slot.belief = false;
    slot.dev_now = true;
    let game = slot.game.as_ref().unwrap();
    match slot.trees[p].as_mut() {
        None => {
            let mut t = slot.spare.pop().unwrap_or_default();
            t.reset(game, p, &slot.legal, seat.redet, &mut slot.search_rng);
            slot.trees[p] = Some(t);
        }
        Some(t) => t.set_base(game, &mut slot.search_rng),
    }
    let t = slot.trees[p].as_mut().unwrap();
    t.set_root_min(0);
    t.clear_gumbel();
    slot.target = Some(if t.expanded(t.root) { 0 } else { 1 });
}

/// A branch's deviation move (docs/search.md 2.11): sampled from the root's priors P0 (the network's
/// policy, before any noise) raised to 1 / temp, from the branch's own stream; nothing recorded.
fn deviate<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, p: usize, key: &mut Vec<u64>, n: &mut Tally) -> Result<(), String> {
    let temp = cfg.branch.as_ref().ok_or("internal: a deviation without branch settings")?.temp;
    let tree = slot.trees[p].as_ref().unwrap();
    let root = tree.root;
    let p0 = tree.priors0(root);
    let acts = tree.acts(root);
    if p0.len() != acts.len() || acts.len() < 2 {
        return Err("internal: a deviation search without the root's priors".into());
    }
    let differ = cfg.branch.as_ref().is_some_and(|b| b.differ);
    // (BranchCfg::differ: the first deviation leaves the game's own move out)
    let own = slot.branch.as_ref().filter(|b| differ && b.played == 0).map(|b| b.action);
    let w: Vec<f64> = p0.iter().zip(acts).map(|(&x, &a)| if Some(a) == own { 0.0 } else { (x.max(0.0) as f64).powf(1.0 / temp) }).collect();
    let tot: f64 = w.iter().sum();
    let i = if tot > 0.0 && tot.is_finite() {
        let mut r = slot.rng.unit() * tot;
        let mut i = w.len() - 1;
        for (k, &x) in w.iter().enumerate() {
            r -= x;
            if r < 0.0 {
                i = k;
                break;
            }
        }
        i
    } else {
        slot.rng.below(acts.len() as u32) as usize
    };
    let a = acts[i];
    if let Some(b) = slot.branch.as_mut() {
        if b.played == 0 {
            b.same_first = a == b.action;
        }
        b.played += 1;
    }
    slot.dev_left -= 1;
    slot.dev_now = false;
    slot.decisions += 1;
    n.dev_moves += 1;
    apply(cfg, slot, a, key)
}

/// Choose the move after the search; the training record (`n.recorded`) and value-only records
/// (`n.view_recorded`) it makes.
#[allow(clippy::too_many_arguments)]
fn decide<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, seat: &NnSeat, p: usize, key: &mut Vec<u64>, n: &mut Tally, look: &Mutex<LookTotals>) -> Result<(), String> {
    if slot.dev_now {
        return deviate(cfg, slot, p, key, n);
    }
    let tree = slot.trees[p].as_ref().unwrap();
    let root = tree.root;
    // value-only records waiting for this seat's next search (`ViewTarget::NextQ`, docs/search.md
    // 3.6) take its root value (without forced walks, as a record's q); a minor search is too small
    let gumbel = tree.gumbel_on();
    if !slot.view_wait.is_empty() && !slot.minor && slot.view_wait.iter().any(|w| w.1 as usize == p) {
        let w = if gumbel {
            tree.gumbel_q()
        } else if tree.root_min() > 0 {
            tree.root_win_pruned()
        } else {
            tree.win(root)
        };
        let rq = &mut slot.rq;
        slot.view_wait.retain(|&(i, o)| {
            if o as usize != p {
                return true;
            }
            rq[i as usize] = w;
            false
        });
    }
    let nv_all = tree.visits(root);
    // with the root minimum visits (docs/search.md 3.5) the visit-based outputs - temperature
    // sampling, the 'visits' rule, the 'visits' target, the root value q - leave the forced walks
    // out; the 'cq' rule and target read Q from every visit
    let pruned = if tree.root_min() > 0 { Some(tree.root_visits_pruned()) } else { None };
    let nv: &[u32] = pruned.as_deref().unwrap_or(&nv_all);
    let p0 = tree.priors0(root);
    let acts = tree.acts(root);
    let tot: u64 = nv.iter().map(|&x| x as u64).sum();
    let i = if nv.is_empty() {
        0
    } else if let Some(i) = tree.gumbel_choose() {
        // the Gumbel search's move (docs/search.md 3.7): its noise did the sampling
        i
    } else if seat.selfplay && slot.decisions < seat.temp_moves && tot > 0 {
        let mut r = slot.rng.unit() * tot as f64;
        let mut i = nv.len() - 1;
        for (k, &c) in nv.iter().enumerate() {
            r -= c as f64;
            if r < 0.0 {
                i = k;
                break;
            }
        }
        i
    } else {
        // the seat's rule (docs/search.md 3.3): argmax (N, P0) by default, or the completed-Q
        // logits' argmax (Select::choose, the code SearchTree.root_select runs)
        match seat.select {
            Select::Visits => argmax_visits(nv, &p0),
            sel => sel.choose(&p0, &nv_all, &tree.values(root), tree.value0(root)),
        }
    };
    let mut a = acts[i];
    if let Some((means, margin)) = slot.look_guard.take() {
        // the guarded lookahead player (docs/search.md 3.9): this search's move unless the lookahead's
        // margin over it exceeds the guard
        let (b, over) = guard_move(&means, margin, a);
        a = b;
        let mut t = lock(look);
        t.guard_decisions += 1;
        t.guard_overrides += over as u64;
    }
    // (a seat with `record` false - a league opponent - records nothing; the move is the same)
    // (a branch records nothing before its deviation moves are played: docs/search.md 2.11)
    let rec = cfg.record && seat.selfplay && seat.record && slot.full && slot.dev_left == 0;
    if rec {
        let pi: Vec<f32> = match seat.policy_target {
            PolicyTarget::Visits if tot > 0 => nv.iter().map(|&c| (c as f64 / tot as f64) as f32).collect(),
            PolicyTarget::Visits => p0.clone(),
            PolicyTarget::CompletedQ { c_visit, c_scale } => tree.cq_target(root, c_visit, c_scale),
        };
        let g = slot.game.as_ref().unwrap();
        let mut x = vec![0.0f32; G::NUM_FEATURES];
        g.encode(p, &mut x);
        // the decider's auxiliary targets from the true state (a label only: never in `x`)
        let mut aux = Vec::new();
        if cfg.aux {
            aux.resize(G::aux_width(), 0.0);
            g.aux_target(p, &mut aux);
        }
        slot.records.push(Record { x, player: p as u8, n_players: slot.n as u8, opts: acts.to_vec(), pi, z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: slot.net, aux });
        slot.rq.push(if gumbel {
            tree.gumbel_q()
        } else if pruned.is_some() {
            tree.root_win_pruned()
        } else {
            tree.win(root)
        });
        n.recorded += 1;
        // value-only samples (docs/search.md 3.6): the real state from every other recording
        // seat's view, in its perspective, no options; q per the seat's view_target
        if seat.view_samples > 0.0 && view_pick(cfg.seed, slot.gid, slot.decisions, seat.view_samples) {
            let g = slot.game.as_ref().unwrap();
            for o in 0..slot.n {
                if o == p || !nn_seat(cfg, slot.gid, o).is_some_and(|s| s.selfplay && s.record) {
                    continue;
                }
                let mut x = vec![0.0f32; G::NUM_FEATURES];
                g.encode(o, &mut x);
                // (with auxiliary targets: the observer's, as a decider's are the decider's)
                let mut aux = Vec::new();
                if cfg.aux {
                    aux.resize(G::aux_width(), 0.0);
                    g.aux_target(o, &mut aux);
                }
                if seat.view_target == ViewTarget::NextQ {
                    slot.view_wait.push((slot.records.len() as u32, o as u8));
                }
                slot.records.push(Record { x, player: o as u8, n_players: slot.n as u8, opts: Vec::new(), pi: Vec::new(), z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: slot.net, aux });
                slot.rq.push(None);
                n.view_recorded += 1;
            }
        }
    }
    // branching (docs/search.md 2.11): the reservoir over the game's eligible decisions - a
    // recording self-play seat's search of a decision (>= 2 options) that is not minor - keeps a copy
    // of the state at a uniformly drawn one, from the game's branch stream
    if let Some(f) = slot.fork.as_mut() {
        let g = slot.game.as_ref().unwrap();
        if cfg.record && seat.selfplay && seat.record && !g.minor_decision() {
            f.seen += 1;
            if f.rng.unit() * (f.seen as f64) < 1.0 {
                f.point = Some(ForkPoint { game: g.clone(), decisions: slot.decisions, steps: slot.steps, moves: slot.moves.clone(), at: f.seen - 1, action: a });
            }
        }
    }
    slot.decisions += 1;
    apply(cfg, slot, a, key)?;
    Ok(())
}

/// Apply a real move: the transition (single-option decisions of non-external seats are
/// auto-stepped, as in the search), then tree reuse.
fn apply<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, a: u16, key: &mut Vec<u64>) -> Result<(), String> {
    let gid = slot.gid;
    let g = slot.game.as_mut().unwrap();
    let p = g.to_move().ok_or("internal: move in a finished game")?;
    if (slot.moves.len() as u32) < cfg.move_log {
        // (the move log, a diagnostic: every caller has the decision's legal actions in slot.legal)
        slot.moves.push((p as u8, a, slot.legal.clone()));
    }
    transition_with(g, a, None, key, &mut slot.legal, |q| !is_external(cfg, gid, q)).map_err(|e| format!("move {a} failed: {e}"))?;
    slot.legal_ok = true;
    slot.steps += 1;
    slot.look_mcts = false;
    slot.look_guard = None;
    let nxt = g.to_move();
    for q in 0..slot.n {
        let keep = match (slot.trees[q].as_mut(), nxt) {
            (Some(t), Some(np)) => q == p && np == p && t.redet && t.advance(a, key, np, &slot.legal),
            _ => false,
        };
        if !keep {
            slot.drop_tree(q);
        }
    }
    slot.sims = 0;
    slot.target = None;
    Ok(())
}

/// Apply an external seat's answer: the game may have changed in any way, so every tree goes.
fn apply_external<G: SearchGame>(slot: &mut Slot<G>, payload: &str, key: &mut Vec<u64>) -> Result<(), String> {
    key.clear();
    let g = slot.game.as_mut().unwrap();
    g.apply_external(payload, key).map_err(|e| format!("external move failed: {e} (payload {payload:?})"))?;
    slot.ext_moves += 1;
    slot.steps += 1;
    slot.legal_ok = false;
    slot.look = None;
    slot.look_mcts = false;
    slot.look_guard = None;
    slot.drop_trees();
    slot.sims = 0;
    slot.target = None;
    Ok(())
}

/// The game ended (or hit max_steps): result row and training samples; a game over at a
/// deferred terminal is parked instead (its records, root values and payload; `Driver::resolve`
/// emits them later).
fn finish<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, results: &mut Vec<GameResult>, samples: &mut Vec<Record>, parked: &mut Vec<Parked<G>>, n_: &mut Tally) -> Result<(), String> {
    let cfg = &sh.cfg;
    let g = slot.game.take().unwrap();
    let over = g.to_move().is_none();
    let gid = slot.gid;
    let start = slot.start_ix.take();
    let fork = slot.fork.take();
    let done = Done { gid, tag: slot.tag.take(), n: slot.n, steps: slot.steps, decisions: slot.decisions, ext_moves: slot.ext_moves, moves: std::mem::take(&mut slot.moves), branch: slot.branch.take(), start };
    slot.dev_left = 0;
    slot.dev_now = false;
    let records = std::mem::take(&mut slot.records);
    let rq = std::mem::take(&mut slot.rq);
    slot.view_wait.clear();
    slot.drop_trees();
    slot.look = None;
    slot.look_mcts = false;
    slot.look_guard = None;
    slot.gid = NO_GAME;
    slot.pending = None;
    slot.ext_payload = None;
    slot.legal_ok = false;
    if over && g.deferred() {
        if fork.as_ref().is_some_and(|f| f.point.is_some()) {
            return Err("branch: a game at a deferred terminal cannot branch (docs/search.md 2.11)".into());
        }
        let payload = g.deferred_payload().map_err(|e| format!("deferred_payload: {e}"))?;
        let records: Vec<ParkedRecord> = if cfg.record { records.into_iter().map(ParkedRecord::park).collect() } else { Vec::new() };
        let bytes = records.iter().map(|r| r.bytes()).sum::<usize>() as u64;
        let Done { gid, tag, n, steps, decisions, ext_moves, .. } = done;
        parked.push(Parked { gid, tag, game: g, n, records, rq, steps, decisions, ext_moves, payload, bytes, start });
        return Ok(());
    }
    results.push(emit(cfg.record, &g, over, done, records, rq, samples));
    if let Some(mut f) = fork {
        if let Some(fp) = f.point.take() {
            spawn_branch(sh, slot, gid, f, fp, start)?;
            n_.branches += 1;
        }
    }
    Ok(())
}

/// The finished game `gid`'s branch into its slot (docs/search.md 2.11): the state at its branch
/// point, the game's searched decisions, real moves and move log before it; streams of its own (from
/// the game's branch stream), fresh trees and records; its deviation moves drawn (1..=moves).
fn spawn_branch<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, gid: u32, mut f: Box<Fork<G>>, fp: ForkPoint<G>, start: Option<u32>) -> Result<(), String> {
    let cfg = &sh.cfg;
    let bc = cfg.branch.as_ref().ok_or("internal: a branch without branch settings")?;
    let moves = 1 + f.rng.below(bc.moves);
    let mut base = Rng::new(f.rng.next_u64());
    let seats = game_seats(cfg, gid);
    // (bots of a branch: from the driver's spec, or the archive entry's the game started from)
    let spec = start.map_or(&sh.spec, |j| &sh.starts[j as usize]);
    slot.gid = gid;
    slot.n = seats.len();
    slot.rng = Rng::new(base.next_u64());
    slot.noise_rng = Rng::new(base.next_u64());
    slot.search_rng = Rng::new(base.next_u64());
    let seeds: Vec<u64> = seats.iter().map(|_| base.next_u64()).collect();
    slot.bots = seats
        .iter()
        .zip(seeds)
        .map(|(&s, seed)| match &cfg.seats[s as usize] {
            SeatCfg::Bot(name) => make_bot::<G>(spec, name, seed),
            _ => None,
        })
        .collect();
    slot.drop_trees();
    slot.game = Some(fp.game);
    slot.tag = None;
    slot.legal_ok = false;
    slot.sims = 0;
    slot.target = None;
    slot.full = true;
    slot.minor = false;
    slot.belief = false;
    slot.decisions = fp.decisions;
    slot.ext_moves = 0;
    slot.steps = fp.steps;
    slot.records.clear();
    slot.rq.clear();
    slot.view_wait.clear();
    slot.pending = None;
    slot.ext_payload = None;
    slot.moves = fp.moves;
    slot.look = None;
    slot.look_mcts = false;
    slot.look_guard = None;
    slot.fork = None;
    slot.start_ix = None;
    slot.branch = Some(BranchInfo { at: fp.at, of: f.seen, decision: fp.decisions, step: fp.steps, action: fp.action, moves, played: 0, same_first: false, start });
    slot.dev_left = moves;
    slot.dev_now = false;
    Ok(())
}

/// What a finished game's result row needs besides the game.
pub(super) struct Done {
    pub gid: u32,
    pub tag: Option<String>,
    pub n: usize,
    pub steps: u64,
    pub decisions: u32,
    pub ext_moves: u32,
    pub moves: Vec<(u8, u16, Vec<u16>)>,
    pub branch: Option<BranchInfo>,
    pub start: Option<u32>,
}

/// A finished (or resolved) game's result row; its records become samples (`record`, the game
/// over and records made): `z` = `result()`, `q` = the root search's win estimate (`z` without
/// one), `sc` = `margin()`, each relative to the record's decider. A game cut at max_steps
/// (`over` false) is a draw and emits nothing.
pub(super) fn emit<G: SearchGame>(record: bool, g: &G, over: bool, d: Done, records: Vec<Record>, rq: Vec<Option<[f64; MAX_SEATS]>>, samples: &mut Vec<Record>) -> GameResult {
    let n = d.n;
    let res: [f32; MAX_SEATS] = if over {
        g.result()
    } else {
        let mut r = [0.0; MAX_SEATS];
        r[..n].fill(1.0 / n as f32);
        r
    };
    let emit = record && over && !records.is_empty();
    let row = GameResult {
        gid: d.gid,
        n,
        over,
        result: res[..n].to_vec(),
        scores: (0..n).map(|i| g.score(i)).collect(),
        steps: d.steps,
        decisions: d.decisions,
        ext_moves: d.ext_moves,
        rounds: g.rounds(),
        samples: if emit { records.len() as u32 } else { 0 },
        tag: d.tag,
        moves: d.moves,
        branch: d.branch,
        start: d.start,
    };
    if emit {
        let margins: Vec<f32> = (0..n).map(|i| g.margin(i)).collect();
        for (mut r, rq) in records.into_iter().zip(rq) {
            for j in 0..n {
                let s = (r.player as usize + j) % n;
                r.z[j] = res[s];
                r.q[j] = match rq {
                    Some(w) => w[s] as f32,
                    None => res[s],
                };
                r.sc[j] = margins[s];
            }
            samples.push(r);
        }
    }
    row
}

/// Lay out the group's batches (one per key with rows): external batches are complete here,
/// network batches get their final sizes and every chunk's row / option offsets. Returns the
/// number of fill jobs (0: nothing to fill).
fn plan_assembly<G: SearchGame>(sh: &Shared<G>, g: usize) -> usize {
    let grp = &sh.groups[g];
    let f = G::NUM_FEATURES;
    let mut chunks: Vec<_> = grp.chunks.iter().map(lock).collect();
    let mut asm = Asm::default();
    let mut bases_all = lock(&grp.bases);
    for (k, (_, kind)) in sh.cfg.keys.iter().enumerate() {
        let (mut b, mut nopt, mut nz) = (0usize, 0usize, 0usize);
        let mut row_base = Vec::with_capacity(chunks.len());
        let mut opt_base = Vec::with_capacity(chunks.len());
        let mut nz_base = Vec::with_capacity(chunks.len());
        for ch in chunks.iter() {
            row_base.push(b);
            opt_base.push(nopt);
            nz_base.push(nz);
            b += ch.bufs[k].len();
            nopt += ch.bufs[k].opt_action.len();
            nz += ch.bufs[k].xs_idx.len();
        }
        bases_all[k] = row_base.iter().map(|&r| r as u32).collect();
        if b == 0 {
            continue;
        }
        let mut bt = Batch { key: k as u16, external: *kind == KeyKind::External, group: g, b, ..Default::default() };
        if bt.external {
            for ch in chunks.iter_mut() {
                let x = &mut ch.bufs[k];
                bt.states.append(&mut x.states);
                bt.player.extend_from_slice(&x.player);
                bt.n_players.extend_from_slice(&x.n_players);
                bt.gid.extend_from_slice(&x.gid);
            }
            asm.batches.push(bt);
            continue;
        }
        if sh.cfg.sparse {
            bt.xs_off = vec![0; b + 1];
            bt.xs_idx = vec![0; nz];
            bt.xs_val = vec![0.0; nz];
        } else {
            bt.x = vec![0.0; b * f];
        }
        bt.player = vec![0; b];
        bt.n_players = vec![0; b];
        bt.gid = vec![0; b];
        bt.opt_off = vec![0; b + 1];
        bt.opt_action = vec![0; nopt];
        bt.extra = vec![ROW_LEAF; b];
        let dest = Dest {
            x: bt.x.as_mut_ptr(),
            player: bt.player.as_mut_ptr(),
            n_players: bt.n_players.as_mut_ptr(),
            gid: bt.gid.as_mut_ptr(),
            opt_off: bt.opt_off.as_mut_ptr(),
            opt_action: bt.opt_action.as_mut_ptr(),
            extra: bt.extra.as_mut_ptr(),
            xs_off: bt.xs_off.as_mut_ptr(),
            xs_idx: bt.xs_idx.as_mut_ptr(),
            xs_val: bt.xs_val.as_mut_ptr(),
        };
        asm.keys.push(AsmKey { key: k, dest, row_base, opt_base, nz_base });
        asm.batches.push(bt);
    }
    drop(bases_all);
    let n = if asm.keys.is_empty() { 0 } else { chunks.len() };
    drop(chunks);
    *lock(&grp.asm) = Some(asm);
    n
}

/// Fill job: copy chunk `c`'s rows of every network key into the group's batches.
fn fill_chunk<G: SearchGame>(sh: &Shared<G>, g: usize, c: usize) {
    let grp = &sh.groups[g];
    let f = G::NUM_FEATURES;
    let sparse = sh.cfg.sparse;
    let parts: Vec<(usize, Dest, usize, usize, usize)> = {
        let a = lock(&grp.asm);
        a.as_ref().map(|a| a.keys.iter().map(|k| (k.key, k.dest, k.row_base[c], k.opt_base[c], k.nz_base[c])).collect()).unwrap_or_default()
    };
    let ch = lock(&grp.chunks[c]);
    for (k, d, rb, ob, zb) in parts {
        let x = &ch.bufs[k];
        let n = x.len();
        if n == 0 {
            continue;
        }
        let no = x.opt_action.len();
        // SAFETY: rows rb..rb + n and options ob..ob + no belong to this chunk alone (the plan's
        // prefix sums) and lie inside the batch's buffers (sized for all rows / options); see Dest.
        unsafe {
            if sparse {
                // (rows rb..rb + n's non-zeros zb..zb + nz: this chunk's alone, the plan's prefix sums)
                let nz = x.xs_idx.len();
                std::slice::from_raw_parts_mut(d.xs_idx.add(zb), nz).copy_from_slice(&x.xs_idx);
                std::slice::from_raw_parts_mut(d.xs_val.add(zb), nz).copy_from_slice(&x.xs_val);
                let xo = std::slice::from_raw_parts_mut(d.xs_off.add(rb + 1), n);
                let mut z = zb;
                for l in 0..n {
                    z += x.xs_nnz[l] as usize;
                    xo[l] = z as i64;
                }
            } else {
                std::slice::from_raw_parts_mut(d.x.add(rb * f), n * f).copy_from_slice(&x.x);
            }
            std::slice::from_raw_parts_mut(d.player.add(rb), n).copy_from_slice(&x.player);
            std::slice::from_raw_parts_mut(d.n_players.add(rb), n).copy_from_slice(&x.n_players);
            std::slice::from_raw_parts_mut(d.gid.add(rb), n).copy_from_slice(&x.gid);
            std::slice::from_raw_parts_mut(d.opt_action.add(ob), no).copy_from_slice(&x.opt_action);
            std::slice::from_raw_parts_mut(d.extra.add(rb), n).copy_from_slice(&x.extra);
            let oo = std::slice::from_raw_parts_mut(d.opt_off.add(rb + 1), n);
            let mut o = ob;
            for l in 0..n {
                o += x.nopt[l] as usize;
                oo[l] = o as i32;
            }
        }
    }
}

/// After the last fill job (or with nothing to fill): publish the group's batches, results and
/// samples and wake `next_batch`. Never panics on a poisoned lock (the group is always marked
/// done: a caller waiting for it must wake up, to see the driver's error if there is one).
fn finish_assembly<G: SearchGame>(sh: &Shared<G>, g: usize) {
    let grp = &sh.groups[g];
    let asm = lock(&grp.asm).take().unwrap_or_default();
    let mut out = GroupOut { batches: asm.batches, ..Default::default() };
    for c in &grp.chunks {
        let mut c = lock(c);
        out.results.append(&mut c.results);
        out.samples.append(&mut c.samples);
        out.parked.append(&mut c.parked);
    }
    *lock(&grp.out) = Some(out);
    lock(&sh.done)[g] = true;
    sh.done_cv.notify_all();
}
