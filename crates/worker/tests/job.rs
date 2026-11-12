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
        output: Output::Shards,
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

/// A records file (`zero_worker::records`): its header and its games' records.
fn read_records(p: &Path) -> (Value, Vec<String>) {
    use std::io::Read;
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(p).unwrap()).read_to_string(&mut text).unwrap();
    let mut lines = text.lines();
    let h: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    (h, lines.map(String::from).collect())
}

/// Toy settings with every kind of row a record must rebuild: auxiliary targets and value-only rows.
fn rich(j: &mut Job) {
    use zero_worker::pyjson::PyJson;
    j.settings.set("aux_next_card_w", PyJson::Float(1.0));
    j.settings.set("selfplay_view_samples", PyJson::Float(0.5));
    j.settings.set("shard_rows", PyJson::Int(60));
}

#[test]
fn records_replay_to_the_shards_rows_bit_for_bit() {
    for sparse in [true, false] {
        let d = tmp(&format!("records-{sparse}"));
        let mut j = job(&d, 30, 40, 2, sparse);
        rich(&mut j);
        j.output = Output::Both;
        let (o, ev) = run(&j, |_, _, _, _| {});
        assert_eq!(o.reason, "done");
        assert!(o.records.len() >= 2, "several records files: {}", o.records.len());
        assert_eq!(o.records.len(), o.shards.len(), "records files and shards are cut at the same games");
        let (mut rows, mut views, mut auxed) = (0, 0, 0);
        for (rw, sw) in o.records.iter().zip(&o.shards) {
            let (h, lines) = read_records(&rw.path);
            assert_eq!((h["records"].as_u64(), h["game"].as_str(), h["aux"].as_bool(), h["seed"].as_u64()), (Some(1), Some("toy"), Some(true), Some(99)));
            assert_eq!(h["games"].as_u64(), Some(lines.len() as u64));
            let mut gids: Vec<u32> = rw.gids.clone();
            gids.sort();
            let mut sg = sw.gids.clone();
            sg.sort();
            assert_eq!(gids, sg, "the file's games are the shard's");
            // the rows rebuilt from the records, through the same shard writer, against the shard written directly
            let games: Vec<zero_worker::shard::GameSamples> = lines
                .iter()
                .map(|l| {
                    let rp = zero_search::record::replay_json::<zero_search::toy::ToyGame>(h["spec"].as_str().unwrap(), l, true).unwrap();
                    assert_eq!(rp.seed, zero_search::record::game_seed(99, rp.gid, false), "the game's seed is its id's");
                    zero_worker::shard::GameSamples { gid: rp.gid, records: rp.records }
                })
                .filter(|g| !g.records.is_empty())
                .collect();
            let f = <zero_search::toy::ToyGame as zero_search::SearchGame>::NUM_FEATURES;
            let (arrays, _, probs) = zero_worker::shard::shard_arrays(&games, f, sparse);
            assert!(probs.is_empty());
            let z = read_npz(&sw.path).unwrap();
            assert_eq!(z.len() - 1, arrays.len(), "the same members");
            for (name, a) in &arrays {
                assert_eq!(z[*name].data, a.data, "{name}: rebuilt rows == the shard's");
                assert_eq!(z[*name].shape, a.shape, "{name}");
            }
            rows += games.iter().map(|g| g.records.len()).sum::<usize>();
            views += games.iter().flat_map(|g| g.records.iter()).filter(|r| r.opts.is_empty()).count();
            auxed += games.iter().flat_map(|g| g.records.iter()).filter(|r| r.aux.iter().any(|&x| x != 0.0)).count();
        }
        assert_eq!(rows, o.shards.iter().map(|w| w.rows).sum::<usize>());
        assert!(views > 0 && auxed > 0, "value-only rows {views}, rows with auxiliary targets {auxed}");
        let n_rec = ev.iter().filter(|e| e["ev"] == "records").count();
        assert_eq!(n_rec, o.records.len());
        assert_eq!(o.stats["rows"].as_u64(), Some(rows as u64), "rows counted once");
    }
}

#[test]
fn a_served_job_writes_records_only() {
    let d = tmp("records-only");
    let mut j = job(&d, 9, 0, 1, true);
    j.output = Output::Records;
    let (o, ev) = run(&j, |_, _, _, _| {});
    assert_eq!(o.reason, "done");
    assert!(o.shards.is_empty() && !o.records.is_empty());
    assert!(!std::fs::read_dir(&d).unwrap().any(|e| e.unwrap().path().extension().is_some_and(|x| x == "npz")));
    assert!(ev.iter().all(|e| e["ev"] != "shard"));
    let games: u64 = ev.iter().filter(|e| e["ev"] == "records").map(|e| e["games"].as_u64().unwrap()).sum();
    assert_eq!(games, 9);
    for e in ev.iter().filter(|e| e["ev"] == "records") {
        assert!(e["path"].as_str().unwrap().ends_with(".rec.jsonl.gz"));
        assert_eq!(e["gen"].as_u64(), Some(7), "the rows' network");
    }
}
