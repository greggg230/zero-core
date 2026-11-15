//! Game records of variable-length rows (docs/search.md 2.15, 2.17): `toy_tail`'s traced games replay into the rows
//! the driver emitted - the head, every tail token and every pointer override, bit for bit - and a game the driver
//! aborted at its row limit replays as aborted (no samples, the game still in progress where the driver stopped it).
use super::*;
use crate::record::{game_json, replay};
use crate::toy_tail::TailGame;

fn seat(sims: u32) -> NnSeat {
    let mut s = NnSeat::eval(0, sims);
    s.selfplay = true;
    s.temp_moves = 6;
    s.record = true;
    s
}

fn rcfg(s: NnSeat, spec: &str, sparse: bool) -> DriverCfg {
    DriverCfg {
        spec: spec.into(),
        seats: vec![SeatCfg::Nn(s)],
        games: vec![vec![0, 0]],
        keys: vec![("m".into(), KeyKind::Nn)],
        concurrency: 6,
        seed: 31,
        record: true,
        threads: 2,
        groups: 2,
        max_steps: 10_000,
        rolling: true,
        net: 0,
        queue_only: false,
        move_log: 0,
        aux: false,
        paired: false,
        sparse,
        branch: None,
        starts: None,
        max_row_tokens: None,
    }
}

/// Answers from each row's head and tail (so the games depend on both); uniform priors.
fn answer(d: &mut Driver<TailGame>, b: &Batch) {
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let m = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
        let h = if b.xs_off.is_empty() {
            b.x[r * TailGame::NUM_FEATURES..(r + 1) * TailGame::NUM_FEATURES].iter().enumerate().map(|(i, x)| x * ((i % 5) as f32 - 2.0)).sum::<f32>()
        } else {
            (b.xs_off[r] as usize..b.xs_off[r + 1] as usize).map(|k| b.xs_val[k] * ((b.xs_idx[k] % 5) as f32 - 2.0)).sum::<f32>()
        } + (b.t_off[r + 1] - b.t_off[r]) as f32 * 0.3;
        let w = 1.0 / (1.0 + (-h * 0.2).exp());
        v[r * MAX_SEATS] = w;
        v[r * MAX_SEATS + 1] = 1.0 - w;
    }
    d.submit(b.id, &pri, &v, &m).unwrap();
}

/// A traced rolling driver's finished games, each with its samples.
fn traced(c: DriverCfg, quota: u64) -> Vec<(GameResult, Vec<Record>)> {
    let mut d = Driver::<TailGame>::new(c).unwrap();
    d.set_trace(true).unwrap();
    d.resume(quota, 0).unwrap();
    let mut out = Vec::new();
    let take = |d: &mut Driver<TailGame>, out: &mut Vec<(GameResult, Vec<Record>)>| {
        let rows = d.take_results();
        let mut s = d.take_samples().into_iter();
        for r in rows {
            let k = r.samples as usize;
            let mine: Vec<Record> = s.by_ref().take(k).collect();
            assert_eq!(mine.len(), k);
            out.push((r, mine));
        }
        assert!(s.next().is_none());
    };
    while let Some(b) = d.next_batch(true).unwrap() {
        answer(&mut d, &b);
        take(&mut d, &mut out);
    }
    take(&mut d, &mut out);
    out
}

/// A replayed row against the driver's, every value a shard stores bit for bit (pi and q as float16): the head, the
/// tail tokens (segment, slot, floats) and the overrides included.
fn same_row(a: &Record, b: &Record) -> Result<(), String> {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let f16s = |v: &[f32]| v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<_>>();
    let chk = |what: &str, ok: bool| if ok { Ok(()) } else { Err(format!("{what} differs")) };
    chk("x", bits(&a.x) == bits(&b.x))?;
    chk("player", a.player == b.player)?;
    chk("n_players", a.n_players == b.n_players)?;
    chk("opts", a.opts == b.opts)?;
    chk("pi", f16s(&a.pi) == f16s(&b.pi))?;
    chk("z", bits(&a.z) == bits(&b.z))?;
    chk("q", f16s(&a.q) == f16s(&b.q))?;
    chk("sc", bits(&a.sc) == bits(&b.sc))?;
    chk("net", a.net == b.net)?;
    chk("step", a.step == b.step)?;
    chk("tail seg", a.tail.seg == b.tail.seg)?;
    chk("tail slot", a.tail.slot == b.tail.slot)?;
    chk("tail val", bits(&a.tail.val) == bits(&b.tail.val))?;
    chk("overrides", a.ov == b.ov)
}

/// Every game replayed through its record's text; -> (rows, rows with a tail token, overrides, aborted games).
fn replay_all(games: &[(GameResult, Vec<Record>)], spec: &str) -> (usize, usize, usize, usize) {
    let sp = TailGame::parse_spec(spec).unwrap();
    let (mut rows, mut tails, mut ovs, mut aborted) = (0, 0, 0, 0);
    for (r, s) in games {
        let rec: serde_json::Value = serde_json::from_str(&game_json(r, s).unwrap().to_string()).unwrap();
        assert_eq!(rec.get("abort_reason").and_then(|v| v.as_str()), r.abort_reason, "game {}", r.gid);
        let rp = replay::<TailGame>(&sp, &rec, false).unwrap_or_else(|e| panic!("game {}: {e}", r.gid));
        assert_eq!((rp.gid, rp.over, rp.aborted, rp.abort_reason), (r.gid, r.over, r.aborted, r.abort_reason));
        assert_eq!(rp.records.len(), s.len());
        for (k, (a, b)) in rp.records.iter().zip(s).enumerate() {
            same_row(a, b).unwrap_or_else(|e| panic!("game {} row {k}: {e}", r.gid));
        }
        rows += s.len();
        tails += s.iter().filter(|x| !x.tail.seg.is_empty()).count();
        ovs += s.iter().map(|x| x.ov.len()).sum::<usize>();
        aborted += r.aborted as usize;
    }
    (rows, tails, ovs, aborted)
}

#[test]
fn traced_tail_games_replay_to_their_rows_bit_for_bit() {
    for sparse in [false, true] {
        let spec = r#"{"items0": 9}"#;
        let games = traced(rcfg(seat(16), spec, sparse), 24);
        assert!(games.len() >= 24);
        let (rows, tails, ovs, aborted) = replay_all(&games, spec);
        assert!(rows > 100 && tails > 10 && ovs > 10, "rows {rows}, with a tail {tails}, overrides {ovs}");
        assert_eq!(aborted, 0);
    }
}

#[test]
fn a_game_aborted_at_the_row_limit_replays_as_aborted() {
    // the game's own limit at the first row (70 items: every row past 64 tokens), and a driver's lower limit that
    // lets the first rows through (games that grow past it abort mid-game, their moves so far in the record)
    for (spec, limit) in [(r#"{"items0": 70}"#, None), (r#"{"items0": 8}"#, Some(12))] {
        let mut c = rcfg(seat(8), spec, true);
        c.max_row_tokens = limit;
        let games = traced(c, 12);
        let (_, _, _, aborted) = replay_all(&games, spec);
        assert!(aborted > 0, "{spec}: no game aborted");
        assert!(games.iter().filter(|(r, _)| r.aborted).all(|(r, s)| s.is_empty() && r.abort_reason == Some("max_row_tokens")));
        if limit.is_some() {
            assert!(games.iter().any(|(r, _)| r.aborted && r.trace.as_ref().is_some_and(|t| !t.steps.is_empty())), "a mid-game abort");
        }
    }
    // a contest whose second game passes the limit: aborted with game 1's samples, and so is its replay
    let spec = r#"{"items0": 6, "max_moves": 10, "episodes": 3, "items_step": 70}"#;
    let games = traced(rcfg(seat(16), spec, true), 4);
    let (rows, _, _, aborted) = replay_all(&games, spec);
    assert!(aborted == games.len() && rows > 0, "{aborted} of {} contests aborted, {rows} rows", games.len());
}

#[test]
fn an_abort_reason_is_checked() {
    let spec = r#"{"items0": 70}"#;
    let games = traced(rcfg(seat(8), spec, true), 2);
    let sp = TailGame::parse_spec(spec).unwrap();
    let (r, s) = &games[0];
    let good = game_json(r, s).unwrap();
    assert!(replay::<TailGame>(&sp, &good, false).is_ok());
    let mut v = good.clone();
    v["abort_reason"] = serde_json::json!("tired");
    assert!(replay::<TailGame>(&sp, &v, false).unwrap_err().contains("abort_reason"));
    let mut v = good.clone();
    v.as_object_mut().unwrap().remove("abort_reason");
    assert!(replay::<TailGame>(&sp, &v, false).unwrap_err().contains("aborted"), "an abort without its reason");
    // a game without tail tokens has no row limit
    let toy = crate::toy::ToyGame::parse_spec("{}").unwrap();
    let mut t = traced_toy();
    t["abort_reason"] = serde_json::json!("max_row_tokens");
    assert!(replay::<crate::toy::ToyGame>(&toy, &t, false).unwrap_err().contains("no tail tokens"));
}

/// One finished toy game's record (no tail segments).
fn traced_toy() -> serde_json::Value {
    let mut c = rcfg(seat(4), "{}", true);
    c.spec = "{}".into();
    let mut d = Driver::<crate::toy::ToyGame>::new(c).unwrap();
    d.set_trace(true).unwrap();
    d.resume(1, 0).unwrap();
    let mut rows = Vec::new();
    let mut smp = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        let mut pri = Vec::new();
        for r in 0..b.b {
            let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
            pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
        }
        d.submit(b.id, &pri, &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS]).unwrap();
        rows.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    rows.extend(d.take_results());
    smp.extend(d.take_samples());
    let r = &rows[0];
    game_json(r, &smp[..r.samples as usize]).unwrap()
}
