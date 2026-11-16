//! Rolling self-play jobs (zero-farm PROTOCOL.md 10.5, zero-core docs/az.md 54) on the toy game with the uniform
//! evaluator (no ONNX Runtime): the job keeps `conc` games in flight while ids are left, sends `low` once per games
//! value, idles when every released id is played (no `end`), takes `extend` (never lower), writes a records file
//! `flush_s` after its first game finished even while idle, switches networks in place at a `checkpoint` (each row
//! records the network that searched it) and ends `finish` / `quit` with the released ids not finished as
//! `unfinished`. Every game of a rolling job is the game of the same id in a classic job, byte for byte.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use zero_worker::job::{run_job, Checkpoint, Control, EvalKind, Job, Outcome, Progress, Rolling, Sink};
use zero_worker::npz::read_npz;
use zero_worker::records::Output;
use zero_worker::settings::Settings;
use zero_worker::GameEntry;

static TOY: GameEntry = GameEntry::of::<zero_search::toy::ToyGame>("toy");

#[derive(Default)]
struct Events(Mutex<Vec<Value>>);

impl Sink for Events {
    fn event(&self, v: Value) {
        self.0.lock().unwrap().push(v);
    }
    fn log(&self, _msg: &str) {}
}

impl Events {
    fn of(&self, ev: &str) -> Vec<Value> {
        self.0.lock().unwrap().iter().filter(|e| e["ev"] == ev).cloned().collect()
    }
    fn wait(&self, what: &str, f: impl Fn(&[Value]) -> bool) {
        let t0 = Instant::now();
        while !f(&self.0.lock().unwrap()) {
            assert!(t0.elapsed() < Duration::from_secs(120), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

fn settings(shard_rows: i64) -> Settings {
    Settings::from_value(&json!({
        "game": "toy", "spec": "", "players": [2], "conc": 6, "threads": 2, "groups": 1, "max_steps": 10000,
        "sims": 16, "sims_fast": 4, "p_full": 0.5, "c_puct": 1.0, "alpha": 0.3, "eps": 0.25, "temp_moves": 10,
        "chance_k": 0, "margin_w": 0.0, "redet": 1, "shard_rows": shard_rows, "shard_sparse": 1, "sparse": 0,
        "policy_target": "cq", "cq_c_scale": 0.3
    }))
    .unwrap()
}

fn job(dir: &Path, games: u64, conc: usize, rolling: Option<Rolling>, output: Output, shard_rows: i64) -> Job {
    Job {
        lease_id: "r".into(),
        settings: settings(shard_rows),
        eval: EvalKind::Uniform,
        games,
        gid_base: 500,
        seed: 99,
        net: 7,
        search_threads: Some(2),
        conc: Some(conc),
        groups: Some(1),
        out_dir: dir.to_path_buf(),
        prefix: "f".into(),
        meta: Map::new(),
        record_batches: None,
        status_s: f64::INFINITY,
        output,
        rolling,
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zero-worker-rolling-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Run `j` while `during` drives its control; `during` must end the job (finish / quit) for a rolling one.
fn run(j: &Job, during: impl Fn(&Control, &Mutex<Progress>, &Events) + Send + Sync) -> (Outcome, Vec<Value>) {
    let ctl = Control::default();
    let ev = Events::default();
    let progress = Arc::new(Mutex::new(Progress::default()));
    let out = std::thread::scope(|s| {
        let h = s.spawn(|| run_job(&TOY, j, &ctl, &ev, progress.clone(), &|_| {}));
        during(&ctl, &progress, &ev);
        h.join().unwrap()
    });
    (out.unwrap(), ev.0.into_inner().unwrap())
}

fn fins(e: &Events) -> u64 {
    fins_of(&e.0.lock().unwrap())
}

/// The games finished (`fin` events) - inside `Events::wait`'s predicate, which holds the events' lock.
fn fins_of(v: &[Value]) -> u64 {
    v.iter().filter(|x| x["ev"] == "fin").map(|x| x["n"].as_u64().unwrap()).sum()
}

/// One shard per game (shard_rows 1): gid -> its arrays.
fn by_gid(o: &Outcome) -> BTreeMap<u32, BTreeMap<String, Vec<u8>>> {
    o.shards
        .iter()
        .map(|w| {
            assert_eq!(w.gids.len(), 1);
            (w.gids[0], read_npz(&w.path).unwrap().into_iter().filter(|(k, _)| k != "meta").map(|(k, a)| (k, a.data)).collect())
        })
        .collect()
}

#[test]
fn a_rolling_job_idles_takes_extend_and_ends_on_finish() {
    let d = tmp("extend");
    let j = job(&d, 4, 2, Some(Rolling::default()), Output::Shards, 1);
    let (o, ev) = run(&j, |ctl, p, e| {
        // 4 ids, 2 in flight: `low` once the 3rd game started (4 - 3 < 2)
        e.wait("low 4", |v| v.iter().any(|x| x["ev"] == "low" && x["games"] == 4));
        e.wait("4 games", |v| fins_of(v) == 4);
        // every released id played: the job idles (no end), still one `low` for games 4
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(p.lock().unwrap().games, 4);
        assert_eq!(e.of("low").len(), 1, "one low per games value");
        // never lower: an extend to fewer or as many ids changes nothing
        ctl.extend(3);
        ctl.extend(4);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(fins(e), 4, "no game started on a lower or equal extend");
        assert!(ctl.extend(10));
        assert!(!ctl.extend(9), "a lower total than one asked is ignored");
        e.wait("low 10", |v| v.iter().any(|x| x["ev"] == "low" && x["games"] == 10));
        e.wait("10 games", |v| fins_of(v) == 10);
        std::thread::sleep(Duration::from_millis(100));
        ctl.extend(13);
        e.wait("a game of the 3 more", |v| fins_of(v) >= 11);
        ctl.finish();
    });
    assert_eq!(o.reason, "finish");
    let lows: Vec<(u64, u64)> = ev.iter().filter(|x| x["ev"] == "low").map(|x| (x["started"].as_u64().unwrap(), x["games"].as_u64().unwrap())).collect();
    assert_eq!(lows.iter().map(|l| l.1).collect::<Vec<_>>(), vec![4, 10, 13], "one low per games value: {lows:?}");
    for (s, g) in &lows {
        assert!(g - s < 2, "low when fewer than conc ids are left: {lows:?}");
    }
    // finish: the games in flight finished, none started after; the ids never started are unfinished
    let done: Vec<u32> = by_gid(&o).keys().copied().collect();
    assert!(done.len() >= 11 && done.len() <= 13, "{done:?}");
    assert_eq!(done, (500..500 + done.len() as u32).collect::<Vec<_>>());
    let n = done.len() as u32;
    let want: Vec<(u32, u32)> = if n < 13 { vec![(500 + n, 513)] } else { vec![] };
    assert_eq!(o.unfinished, want);
    assert!(ev.iter().all(|x| x["ev"] != "end"), "run_job sends no end (serve does)");
}

#[test]
fn rolling_games_are_the_classic_jobs_games() {
    let (a, b) = (tmp("classic"), tmp("rolled"));
    let (oa, _) = run(&job(&a, 9, 3, None, Output::Shards, 1), |_, _, _| {});
    assert_eq!(oa.reason, "done");
    let (ob, _) = run(&job(&b, 2, 3, Some(Rolling::default()), Output::Shards, 1), |ctl, _, e| {
        // ids released a few at a time, as the agent extends them
        for (prev, g) in [(2, 5), (5, 7), (7, 9)] {
            e.wait("low", |v| v.iter().any(|x| x["ev"] == "low" && x["games"] == prev));
            ctl.extend(g);
        }
        e.wait("9 games", |v| fins_of(v) == 9);
        ctl.finish();
    });
    assert_eq!(ob.reason, "finish");
    assert!(ob.unfinished.is_empty(), "{:?}", ob.unfinished);
    assert_eq!(by_gid(&oa), by_gid(&ob), "every game of a rolling job == that id's game in a classic job");
}

#[test]
fn quit_lists_the_released_ids_not_finished() {
    let d = tmp("quit");
    let (o, _) = run(&job(&d, 6, 3, Some(Rolling::default()), Output::Shards, 1), |ctl, _, e| {
        e.wait("a game", |v| fins_of(v) >= 1);
        ctl.extend(30);
        e.wait("8 games", |v| fins_of(v) >= 8);
        ctl.quit();
    });
    assert_eq!(o.reason, "quit");
    let done: Vec<u32> = by_gid(&o).keys().copied().collect();
    let missing: u32 = o.unfinished.iter().map(|r| r.1 - r.0).sum();
    assert_eq!(done.len() as u32 + missing, 30, "released ids = finished + unfinished: {:?}", o.unfinished);
    assert!(o.unfinished.iter().all(|r| r.0 >= 500 && r.1 <= 530));
}

#[test]
fn flush_s_writes_the_held_games_while_idle() {
    let d = tmp("flush");
    // a records file only at 100k rows, or flush_s after its first game finished
    let j = job(&d, 3, 3, Some(Rolling { flush_s: 0.3 }), Output::Records, 100_000);
    let (o, ev) = run(&j, |ctl, _, e| {
        e.wait("3 games", |v| fins_of(v) == 3);
        let t = Instant::now();
        e.wait("a records file", |v| v.iter().any(|x| x["ev"] == "records"));
        assert!(t.elapsed() < Duration::from_secs(5));
        let r = e.of("records");
        assert_eq!((r.len(), r[0]["games"].as_u64()), (1, Some(3)), "{r:?}");
        // more games: another file flush_s after the first of them finished
        ctl.extend(5);
        e.wait("5 games", |v| fins_of(v) == 5);
        e.wait("a 2nd records file", |v| v.iter().filter(|x| x["ev"] == "records").count() == 2);
        ctl.finish();
    });
    assert_eq!(o.reason, "finish");
    assert_eq!(o.records.iter().map(|w| w.games).collect::<Vec<_>>(), vec![3, 2]);
    assert_eq!(ev.iter().filter(|x| x["ev"] == "records").count(), 2, "nothing left at the end");
    let mut gids: Vec<u32> = o.records.iter().flat_map(|w| w.gids.clone()).collect();
    gids.sort();
    assert_eq!(gids, (500..505).collect::<Vec<_>>());
    assert!(o.unfinished.is_empty());
}

#[test]
fn a_classic_job_never_flushes_by_time_and_ends_done() {
    let d = tmp("classic-flush");
    let (o, ev) = run(&job(&d, 3, 3, None, Output::Records, 100_000), |ctl, _, e| {
        e.wait("3 games", |v| fins_of(v) == 3);
        // (an extend reaches a classic job's control but changes nothing)
        ctl.extend(10);
    });
    assert_eq!(o.reason, "done");
    assert_eq!(o.records.len(), 1);
    assert!(ev.iter().all(|x| x["ev"] != "low"));
    assert!(o.unfinished.is_empty());
}

#[test]
fn a_checkpoint_switches_in_place_while_the_games_go_on() {
    let d = tmp("checkpoint");
    let j = job(&d, 200, 4, Some(Rolling::default()), Output::Shards, 1);
    let (o, ev) = run(&j, |ctl, _, e| {
        e.wait("3 games", |v| fins_of(v) >= 3);
        ctl.checkpoint(Checkpoint { gen: 9, model: PathBuf::from("uniform"), sha256: None, reference: None });
        // games finish with rows of both networks, then gen 9's only
        let at = fins(e);
        e.wait("12 more games", |v| fins_of(v) >= at + 12);
        // a second switch, while the games go on
        ctl.checkpoint(Checkpoint { gen: 11, model: PathBuf::from("uniform"), sha256: None, reference: None });
        let at = fins(e);
        e.wait("12 more games", |v| fins_of(v) >= at + 12);
        ctl.finish();
    });
    assert_eq!(o.reason, "finish");
    let mut nets_by_gid: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for w in &o.shards {
        let z = read_npz(&w.path).unwrap();
        nets_by_gid.insert(w.gids[0], z["net"].i64s().unwrap());
    }
    let all: Vec<i64> = nets_by_gid.values().flatten().copied().collect();
    for n in [7, 9, 11] {
        assert!(all.contains(&n), "rows of net {n}: {nets_by_gid:?}");
    }
    // within a game the rows' networks never go back; a game that straddled a switch has rows of both
    for (g, ns) in &nets_by_gid {
        assert!(ns.windows(2).all(|w| w[0] <= w[1]), "game {g}: {ns:?}");
    }
    assert!(nets_by_gid.values().any(|ns| ns.first() != ns.last()), "a game in flight went on across a switch");
    // the games kept their ids: one contiguous run from gid_base
    let done: Vec<u32> = nets_by_gid.keys().copied().collect();
    assert_eq!(done, (500..500 + done.len() as u32).collect::<Vec<_>>());
    assert!(ev.iter().all(|x| x["ev"] != "error"));
}

#[test]
fn a_classic_uniform_job_still_refuses_a_checkpoint() {
    let d = tmp("classic-ck");
    let ctl = Control::default();
    let ev = Events::default();
    let j = job(&d, 50, 2, None, Output::Shards, 1);
    let r = std::thread::scope(|s| {
        let h = s.spawn(|| run_job(&TOY, &j, &ctl, &ev, Arc::new(Mutex::new(Progress::default())), &|_| {}));
        ev.wait("a game", |v| v.iter().any(|x| x["ev"] == "fin"));
        ctl.checkpoint(Checkpoint { gen: 9, model: PathBuf::from("uniform"), sha256: None, reference: None });
        h.join().unwrap()
    });
    let e = r.err().expect("a uniform classic job has no network to switch");
    assert!(e.contains("no network to switch"), "{e}");
}
