//! The self-play job loop on the toy game with the uniform evaluator (no ONNX Runtime needed): a
//! yield / resume in the middle of a job finishes with the rows of a job never yielded; a job of game
//! ids `gid_base..` plays exactly those games of a job from 0; the shards hold every finished game's
//! samples in zero-core's format; quit and finish end it as the protocol says.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};
use zero_worker::job::{do_resume, do_yield, run_job, Control, EvalKind, EvalLink, Job, Progress, Sink};
use zero_worker::npz::read_npz;
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

fn settings(groups: usize, sparse: bool) -> Settings {
    Settings::from_value(&json!({
        "game": "toy", "spec": "", "players": [2], "conc": 6, "threads": 2, "groups": groups, "max_steps": 10000,
        "sims": 16, "sims_fast": 4, "p_full": 0.5, "c_puct": 1.0, "alpha": 0.3, "eps": 0.25, "temp_moves": 10,
        "chance_k": 0, "margin_w": 0.0, "redet": 1, "shard_rows": 150, "shard_sparse": sparse as i32, "sparse": 0,
        "policy_target": "cq", "cq_c_scale": 0.3
    }))
    .unwrap()
}

fn job(dir: &Path, games: u64, gid_base: u32, groups: usize, sparse: bool) -> Job {
    Job {
        lease_id: "t".into(),
        settings: settings(groups, sparse),
        eval: EvalKind::Uniform,
        games,
        gid_base,
        seed: 99,
        net: 7,
        search_threads: Some(2),
        conc: None,
        groups: None,
        out_dir: dir.to_path_buf(),
        prefix: "f".into(),
        meta: Map::new(),
        record_batches: None,
        status_s: 0.0,
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zero-worker-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Every shard's arrays (name -> bytes), in shard order.
fn shards(dir: &Path) -> Vec<BTreeMap<String, Vec<u8>>> {
    let mut ps: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "npz")).collect();
    ps.sort();
    ps.iter().map(|p| read_npz(p).unwrap().into_iter().filter(|(k, _)| k != "meta").map(|(k, a)| (k, a.data)).collect()).collect()
}

fn run(j: &Job, during: impl Fn(&Control, &Mutex<Option<EvalLink>>, &Mutex<Progress>, &Events) + Send + Sync) -> (zero_worker::job::Outcome, Vec<Value>) {
    let ctl = Control::default();
    let ev = Events::default();
    let link = Mutex::new(None);
    let progress = Arc::new(Mutex::new(Progress::default()));
    let out = std::thread::scope(|s| {
        let h = s.spawn(|| run_job(&TOY, j, &ctl, &ev, progress.clone(), &|l| *link.lock().unwrap() = Some(l)));
        during(&ctl, &link, &progress, &ev);
        h.join().unwrap()
    });
    (out.unwrap(), ev.0.into_inner().unwrap())
}

#[test]
fn yield_resume_finishes_with_the_same_rows() {
    let (a_dir, b_dir) = (tmp("plain"), tmp("yield"));
    let small = |d: &Path| {
        let mut j = job(d, 48, 0, 1, true);
        j.settings.set("shard_rows", zero_worker::pyjson::PyJson::Int(25));
        j
    };
    let (oa, ea) = run(&small(&a_dir), |_, _, _, _| {});
    assert_eq!(oa.reason, "done");
    assert!(oa.unfinished.is_empty());
    let (ob, eb) = run(&small(&b_dir), |ctl, link, p, ev| {
        // yield once a game finished, hold it a moment, resume
        while !ev.0.lock().unwrap().iter().any(|e| e["ev"] == "fin") {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let l = link.lock().unwrap().clone();
        assert!(do_yield(ctl, l.as_ref(), p).is_some());
        let games = p.lock().unwrap().games;
        assert!(games < 40, "the yield came mid-job ({games} of 48 games done)");
        std::thread::sleep(std::time::Duration::from_millis(300));
        // (yielded: no batch was answered meanwhile, so no game finished)
        let g2 = p.lock().unwrap().games;
        assert!(g2 <= games + 2, "games went on while yielded: {games} -> {g2}");
        assert!(do_resume(ctl, l.as_ref()).unwrap());
    });
    assert_eq!(ob.reason, "done");
    assert_eq!(ob.stats["yields"], 1);
    let (sa, sb) = (shards(&a_dir), shards(&b_dir));
    assert!(sa.len() >= 2, "several shards: {}", sa.len());
    assert_eq!(sa, sb, "the shards of a yielded job == an uninterrupted job's");
    let fins = |e: &[Value]| e.iter().filter(|x| x["ev"] == "fin").map(|x| x["n"].as_u64().unwrap()).sum::<u64>();
    assert_eq!((fins(&ea), fins(&eb)), (48, 48));
    let shard_games = |e: &[Value]| e.iter().filter(|x| x["ev"] == "shard").map(|x| x["games"].as_u64().unwrap()).sum::<u64>();
    assert_eq!(shard_games(&ea), 48);
}

#[test]
fn a_gid_block_plays_those_games() {
    let (a_dir, b_dir) = (tmp("all"), tmp("tail"));
    // shards of one game each (shard_rows 1) so each shard is one gid's rows
    let mut ja = job(&a_dir, 7, 0, 1, true);
    ja.settings.set("shard_rows", zero_worker::pyjson::PyJson::Int(1));
    let mut jb = job(&b_dir, 3, 4, 1, true);
    jb.settings.set("shard_rows", zero_worker::pyjson::PyJson::Int(1));
    let (oa, _) = run(&ja, |_, _, _, _| {});
    let (ob, _) = run(&jb, |_, _, _, _| {});
    let by_gid = |o: &zero_worker::job::Outcome| -> BTreeMap<u32, BTreeMap<String, Vec<u8>>> {
        o.shards.iter().map(|w| {
            assert_eq!(w.gids.len(), 1);
            (w.gids[0], read_npz(&w.path).unwrap().into_iter().filter(|(k, _)| k != "meta").map(|(k, a)| (k, a.data)).collect())
        }).collect()
    };
    let (ga, gb) = (by_gid(&oa), by_gid(&ob));
    assert_eq!(ga.keys().copied().collect::<Vec<_>>(), (0..7).collect::<Vec<u32>>());
    assert_eq!(gb.keys().copied().collect::<Vec<_>>(), vec![4, 5, 6]);
    for g in 4..7 {
        assert_eq!(ga[&g], gb[&g], "game {g}");
    }
}

#[test]
fn shards_hold_every_game_and_meta() {
    let d = tmp("meta");
    let (o, ev) = run(&job(&d, 9, 100, 2, false), |_, _, _, _| {});
    assert_eq!(o.reason, "done");
    let mut gids: Vec<u32> = o.shards.iter().flat_map(|w| w.gids.clone()).collect();
    gids.sort();
    assert_eq!(gids, (100..109).collect::<Vec<u32>>());
    let rows: usize = o.shards.iter().map(|w| w.rows).sum();
    let z = read_npz(&o.shards[0].path).unwrap();
    let meta: Value = serde_json::from_str(&z["meta"].text().unwrap()).unwrap();
    assert_eq!((meta["game"].as_str(), meta["rows"].as_u64()), (Some("toy"), Some(o.shards[0].rows as u64)));
    assert!(meta.get("x_format").is_none() && z.contains_key("x"), "dense shards with shard_sparse 0");
    assert_eq!(z["net"].i64s().unwrap()[0], 7);
    let shard_rows: u64 = ev.iter().filter(|e| e["ev"] == "shard").map(|e| e["rows"].as_u64().unwrap()).sum();
    assert_eq!(shard_rows as usize, rows);
    assert!(ev.iter().any(|e| e["ev"] == "status"));
}

#[test]
fn quit_and_finish() {
    let d = tmp("quit");
    let (o, _) = run(&job(&d, 40, 0, 2, true), |ctl, _, p, _| {
        while p.lock().unwrap().games < 2 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        ctl.quit();
    });
    assert_eq!(o.reason, "quit");
    let done: u64 = o.shards.iter().map(|w| w.games as u64).sum();
    let missing: u32 = o.unfinished.iter().map(|r| r.1 - r.0).sum();
    assert_eq!(done + missing as u64, 40, "{:?}", o.unfinished);
    let d = tmp("finish");
    let (o, _) = run(&job(&d, 40, 0, 2, true), |ctl, _, p, _| {
        while p.lock().unwrap().games < 2 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        ctl.finish();
    });
    assert_eq!(o.reason, "finish");
    let done: u64 = o.shards.iter().map(|w| w.games as u64).sum();
    assert!((2..40).contains(&done), "{done}");
    // finish: every game started was finished; the unfinished are the ones never started (one range)
    assert_eq!(o.unfinished.len(), 1);
    assert_eq!(o.unfinished[0], (done as u32, 40));
}
