//! The driver's worker threads: advancing games (search, bots, external requests), batch
//! assembly, finished games and training records.
use super::*;
use crate::tree::{leaf_values, transition_with, Scratch as SimScratch, Sim, Walk};

struct WScratch<G: SearchGame> {
    game: Option<G>,
    sim: SimScratch,
}

/// Queue-job flag: fill job of chunk `c & !FILL` (else: search job of chunk c).
const FILL: usize = 1 << 30;

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "?".into())
}

pub(super) fn worker<G: SearchGame>(sh: Arc<Shared<G>>) {
    let mut scratch = WScratch { game: None, sim: SimScratch::default() };
    loop {
        let job = {
            let mut q = sh.queue.lock().unwrap();
            loop {
                if sh.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(j) = q.pop_front() {
                    break j;
                }
                q = sh.queue_cv.wait(q).unwrap();
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
            let n = if sh.error.lock().unwrap().is_none() { plan_assembly(&sh, g) } else { 0 };
            sh.counters.assemble_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if n == 0 {
                finish_assembly(&sh, g);
            } else {
                grp.fill_remaining.store(n, Ordering::SeqCst);
                let mut q = sh.queue.lock().unwrap();
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
}

fn run_chunk<G: SearchGame>(sh: &Shared<G>, g: usize, c: usize, scratch: &mut WScratch<G>) -> Result<(), String> {
    let cfg = &sh.cfg;
    let grp = &sh.groups[g];
    let mut ch = grp.chunks[c].lock().unwrap();
    let ch = &mut *ch;
    for b in ch.bufs.iter_mut() {
        b.clear();
    }
    let evals = grp.evals.read().unwrap();
    let epoch = sh.epoch.load(Ordering::SeqCst);
    let net = sh.net.load(Ordering::SeqCst);
    let mut n = Tally::default();
    for slot in ch.slots.iter_mut() {
        let gid = slot.start.or(Some(slot.gid)).filter(|&x| x != NO_GAME);
        let r = (|| -> Result<(), String> {
            if let Some(gid) = slot.start.take() {
                init_slot(sh, slot, gid)?;
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
                Some(Pending::Leaf { key, local, seat, node }) => {
                    let Some(Eval::Nn { opt_off, priors, value, margin, bases }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing network evaluation".into());
                    };
                    let b = (bases[c] + local) as usize;
                    let lo = opt_off[b] as usize;
                    let hi = opt_off[b + 1] as usize;
                    apply_eval(cfg, slot, seat, node, &priors[lo..hi], &value[b * MAX_SEATS..(b + 1) * MAX_SEATS], &margin[b * MAX_SEATS..(b + 1) * MAX_SEATS])?;
                    n.sims += 1;
                }
                Some(Pending::Ext { key, local }) => {
                    let Some(Eval::Ext { payloads, bases }) = evals[key as usize].as_ref() else {
                        return Err("internal: missing external answer".into());
                    };
                    slot.ext_payload = Some(payloads[(bases[c] + local) as usize].clone());
                }
                None => {}
            }
            advance_slot(sh, slot, &mut ch.bufs, &mut ch.results, &mut ch.samples, scratch, &mut n)
        })();
        if let Err(e) = r {
            return Err(format!("game {} (driver seed {}): {e}", gid.map(|x| x as i64).unwrap_or(-1), cfg.seed));
        }
    }
    let k = &sh.counters;
    k.sims.fetch_add(n.sims, Ordering::Relaxed);
    k.leaves.fetch_add(n.leaves, Ordering::Relaxed);
    k.ext_requests.fetch_add(n.ext, Ordering::Relaxed);
    k.restart_sims.fetch_add(n.lost, Ordering::Relaxed);
    k.restarts.fetch_add(n.restarts, Ordering::Relaxed);
    k.recorded.fetch_add(n.recorded, Ordering::Relaxed);
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
    if leaf {
        slot.pending = None;
    }
    slot.drop_trees();
    slot.sims = 0;
    if slot.target.is_some() {
        slot.ensure_legal();
        let game = slot.game.as_ref().unwrap();
        let p = game.to_move().ok_or("internal: search target set on a finished game")?;
        let seat = nn_seat(cfg, slot.gid, p).ok_or("internal: search target set for a non-network seat")?;
        let mut t = slot.spare.pop().unwrap_or_default();
        t.reset(game, p, &slot.legal, seat.redet, &mut slot.search_rng);
        slot.trees[p] = Some(t);
    }
    Ok(lost)
}

/// Start game `gid` in `slot`: every random stream derives from (driver seed, gid).
fn init_slot<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, gid: u32) -> Result<(), String> {
    let cfg = &sh.cfg;
    let mut base = Rng::new(split64(cfg.seed ^ split64(gid as u64 + 1)));
    let seats = game_seats(cfg, gid);
    let n = seats.len();
    let game = G::new_game(&sh.spec, n, base.next_u64())?;
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
            SeatCfg::Bot(name) => make_bot::<G>(&sh.spec, name, seed),
            _ => None,
        })
        .collect();
    slot.drop_trees();
    slot.game = Some(game);
    slot.legal_ok = false;
    slot.sims = 0;
    slot.target = None;
    slot.full = true;
    slot.decisions = 0;
    slot.ext_moves = 0;
    slot.steps = 0;
    slot.records.clear();
    slot.rq.clear();
    slot.pending = None;
    slot.ext_payload = None;
    Ok(())
}

/// Expand the slot's pending leaf with its evaluation and back it up.
fn apply_eval<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, seat: usize, node: u32, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String> {
    let s = nn_seat(cfg, slot.gid, seat).ok_or("internal: pending leaf of a non-network seat")?;
    let tree = slot.trees[seat].as_mut().ok_or("internal: pending leaf without a tree")?;
    if priors.len() != tree.acts(node).len() {
        return Err("internal: evaluation option count mismatch".into());
    }
    let (util, win) = leaf_values(value, margin, tree.player(node), slot.n, s.margin_w);
    tree.expand(node, priors, &util);
    tree.backup(&slot.path, &util, &win);
    if node == tree.root && s.selfplay && slot.full {
        let r = tree.root;
        tree.add_noise(r, s.alpha, s.eps, &mut slot.noise_rng);
    }
    slot.sims += 1;
    Ok(())
}

/// Play the slot's game until its search needs a network evaluation, an external seat is to
/// move, or the game ends.
fn advance_slot<G: SearchGame>(sh: &Shared<G>, slot: &mut Slot<G>, bufs: &mut [LeafBuf], results: &mut Vec<GameResult>, samples: &mut Vec<Record>, scratch: &mut WScratch<G>, n: &mut Tally) -> Result<(), String> {
    let cfg = &sh.cfg;
    loop {
        let g = slot.game.as_ref().unwrap();
        let Some(p) = g.to_move().filter(|_| slot.steps < cfg.max_steps) else {
            finish(cfg, slot, results, samples);
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
        if slot.target.is_none() {
            start(slot, seat, p);
        }
        if slot.sims >= slot.target.unwrap() {
            n.recorded += decide(cfg, slot, seat, p, &mut scratch.sim.key)? as u64;
            continue;
        }
        let walk = Walk { c_puct: seat.c_puct, chance_k: seat.chance_k, margin_w: seat.margin_w };
        let live = slot.game.as_ref().unwrap();
        let sg = scratch.game.get_or_insert_with(|| live.clone());
        let tree = slot.trees[p].as_mut().unwrap();
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
                let off = buf.x.len();
                buf.x.resize(off + G::NUM_FEATURES, 0.0);
                sg.encode(player, &mut buf.x[off..]);
                let acts = tree.acts(node);
                buf.nopt.push(acts.len() as i32);
                buf.opt_action.extend(acts.iter().map(|&a| a as i32));
                buf.player.push(player as i32);
                buf.n_players.push(slot.n as i32);
                buf.gid.push(slot.gid as i64);
                slot.pending = Some(Pending::Leaf { key: seat.key, local, seat: p, node });
                n.leaves += 1;
                return Ok(());
            }
        }
    }
}

/// Start the search for the pending decision of seat `p` (legal actions in `slot.legal`).
fn start<G: SearchGame>(slot: &mut Slot<G>, seat: &NnSeat, p: usize) {
    slot.full = seat.sims_fast.is_none() || slot.rng.unit() < seat.p_full;
    slot.target = Some(if slot.full { seat.sims } else { seat.sims_fast.unwrap() });
    let game = slot.game.as_ref().unwrap();
    match slot.trees[p].as_mut() {
        None => {
            let mut t = slot.spare.pop().unwrap_or_default();
            t.reset(game, p, &slot.legal, seat.redet, &mut slot.search_rng);
            slot.trees[p] = Some(t);
        }
        Some(t) => {
            t.set_base(game, &mut slot.search_rng);
            if seat.selfplay && slot.full && t.expanded(t.root) {
                let r = t.root;
                t.add_noise(r, seat.alpha, seat.eps, &mut slot.noise_rng);
            }
        }
    }
}

/// Choose the move after the search. -> whether a training record was made.
fn decide<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, seat: &NnSeat, p: usize, key: &mut Vec<u64>) -> Result<bool, String> {
    let tree = slot.trees[p].as_ref().unwrap();
    let root = tree.root;
    let nv = tree.visits(root);
    let p0 = tree.priors0(root);
    let acts = tree.acts(root);
    let tot: u64 = nv.iter().map(|&x| x as u64).sum();
    let i = if nv.is_empty() {
        0
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
        // max by (N, P0), first maximum on exact ties
        let mut best = 0;
        for k in 1..nv.len() {
            if (nv[k], p0[k]) > (nv[best], p0[best]) {
                best = k;
            }
        }
        best
    };
    let a = acts[i];
    let rec = cfg.record && seat.selfplay && slot.full;
    if rec {
        let pi: Vec<f32> = match seat.policy_target {
            PolicyTarget::Visits if tot > 0 => nv.iter().map(|&c| (c as f64 / tot as f64) as f32).collect(),
            PolicyTarget::Visits => p0.clone(),
            PolicyTarget::CompletedQ { c_visit, c_scale } => tree.cq_target(root, c_visit, c_scale),
        };
        let mut x = vec![0.0f32; G::NUM_FEATURES];
        slot.game.as_ref().unwrap().encode(p, &mut x);
        slot.records.push(Record { x, player: p as u8, n_players: slot.n as u8, opts: acts.to_vec(), pi, z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: slot.net });
        slot.rq.push(tree.win(root));
    }
    slot.decisions += 1;
    apply(cfg, slot, a, key)?;
    Ok(rec)
}

/// Apply a real move: the transition (single-option decisions of non-external seats are
/// auto-stepped, as in the search), then tree reuse.
fn apply<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, a: u16, key: &mut Vec<u64>) -> Result<(), String> {
    let gid = slot.gid;
    let g = slot.game.as_mut().unwrap();
    let p = g.to_move().ok_or("internal: move in a finished game")?;
    transition_with(g, a, None, key, &mut slot.legal, |q| !is_external(cfg, gid, q)).map_err(|e| format!("move {a} failed: {e}"))?;
    slot.legal_ok = true;
    slot.steps += 1;
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
    slot.drop_trees();
    slot.sims = 0;
    slot.target = None;
    Ok(())
}

/// The game ended (or hit max_steps): result row and training samples.
fn finish<G: SearchGame>(cfg: &DriverCfg, slot: &mut Slot<G>, results: &mut Vec<GameResult>, samples: &mut Vec<Record>) {
    let g = slot.game.take().unwrap();
    let n = slot.n;
    let over = g.to_move().is_none();
    let res: [f32; MAX_SEATS] = if over {
        g.result()
    } else {
        let mut r = [0.0; MAX_SEATS];
        r[..n].fill(1.0 / n as f32);
        r
    };
    let emit = cfg.record && over && !slot.records.is_empty();
    results.push(GameResult {
        gid: slot.gid,
        n,
        over,
        result: res[..n].to_vec(),
        scores: (0..n).map(|i| g.score(i)).collect(),
        steps: slot.steps,
        decisions: slot.decisions,
        ext_moves: slot.ext_moves,
        rounds: g.rounds(),
        samples: if emit { slot.records.len() as u32 } else { 0 },
    });
    if emit {
        let margins: Vec<f32> = (0..n).map(|i| g.margin(i)).collect();
        for (mut r, rq) in slot.records.drain(..).zip(slot.rq.drain(..)) {
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
    slot.records.clear();
    slot.rq.clear();
    slot.drop_trees();
    slot.gid = NO_GAME;
    slot.pending = None;
    slot.ext_payload = None;
    slot.legal_ok = false;
}

/// Lay out the group's batches (one per key with rows): external batches are complete here,
/// network batches get their final sizes and every chunk's row / option offsets. Returns the
/// number of fill jobs (0: nothing to fill).
fn plan_assembly<G: SearchGame>(sh: &Shared<G>, g: usize) -> usize {
    let grp = &sh.groups[g];
    let f = G::NUM_FEATURES;
    let mut chunks: Vec<_> = grp.chunks.iter().map(|c| c.lock().unwrap()).collect();
    let mut asm = Asm::default();
    let mut bases_all = grp.bases.lock().unwrap();
    for (k, (_, kind)) in sh.cfg.keys.iter().enumerate() {
        let (mut b, mut nopt) = (0usize, 0usize);
        let mut row_base = Vec::with_capacity(chunks.len());
        let mut opt_base = Vec::with_capacity(chunks.len());
        for ch in chunks.iter() {
            row_base.push(b);
            opt_base.push(nopt);
            b += ch.bufs[k].len();
            nopt += ch.bufs[k].opt_action.len();
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
        bt.x = vec![0.0; b * f];
        bt.player = vec![0; b];
        bt.n_players = vec![0; b];
        bt.gid = vec![0; b];
        bt.opt_off = vec![0; b + 1];
        bt.opt_action = vec![0; nopt];
        let dest = Dest {
            x: bt.x.as_mut_ptr(),
            player: bt.player.as_mut_ptr(),
            n_players: bt.n_players.as_mut_ptr(),
            gid: bt.gid.as_mut_ptr(),
            opt_off: bt.opt_off.as_mut_ptr(),
            opt_action: bt.opt_action.as_mut_ptr(),
        };
        asm.keys.push(AsmKey { key: k, dest, row_base, opt_base });
        asm.batches.push(bt);
    }
    drop(bases_all);
    let n = if asm.keys.is_empty() { 0 } else { chunks.len() };
    drop(chunks);
    *grp.asm.lock().unwrap() = Some(asm);
    n
}

/// Fill job: copy chunk `c`'s rows of every network key into the group's batches.
fn fill_chunk<G: SearchGame>(sh: &Shared<G>, g: usize, c: usize) {
    let grp = &sh.groups[g];
    let f = G::NUM_FEATURES;
    let parts: Vec<(usize, Dest, usize, usize)> = {
        let a = grp.asm.lock().unwrap();
        a.as_ref().map(|a| a.keys.iter().map(|k| (k.key, k.dest, k.row_base[c], k.opt_base[c])).collect()).unwrap_or_default()
    };
    let ch = grp.chunks[c].lock().unwrap();
    for (k, d, rb, ob) in parts {
        let x = &ch.bufs[k];
        let n = x.len();
        if n == 0 {
            continue;
        }
        let no = x.opt_action.len();
        // SAFETY: rows rb..rb + n and options ob..ob + no belong to this chunk alone (the plan's
        // prefix sums) and lie inside the batch's buffers (sized for all rows / options); see Dest.
        unsafe {
            std::slice::from_raw_parts_mut(d.x.add(rb * f), n * f).copy_from_slice(&x.x);
            std::slice::from_raw_parts_mut(d.player.add(rb), n).copy_from_slice(&x.player);
            std::slice::from_raw_parts_mut(d.n_players.add(rb), n).copy_from_slice(&x.n_players);
            std::slice::from_raw_parts_mut(d.gid.add(rb), n).copy_from_slice(&x.gid);
            std::slice::from_raw_parts_mut(d.opt_action.add(ob), no).copy_from_slice(&x.opt_action);
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
/// samples and wake `next_batch`.
fn finish_assembly<G: SearchGame>(sh: &Shared<G>, g: usize) {
    let grp = &sh.groups[g];
    let asm = grp.asm.lock().unwrap().take().unwrap_or_default();
    let mut out = GroupOut { batches: asm.batches, ..Default::default() };
    for c in &grp.chunks {
        let mut c = c.lock().unwrap();
        out.results.append(&mut c.results);
        out.samples.append(&mut c.samples);
    }
    *grp.out.lock().unwrap() = Some(out);
    sh.done.lock().unwrap()[g] = true;
    sh.done_cv.notify_all();
}
