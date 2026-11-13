//! The self-play job on `toy_tail` (variable-length rows: tail tokens, a tail-only segment, pointer overrides; zero-core
//! docs/search.md 2.15, docs/az.md 46) with the uniform evaluator (no ONNX Runtime): the shards are format v2 (x_version 2,
//! the meta's t_segs / option_pointers / max_row_tokens, the members t_ptr .. ov_tok after aux) and hold every row's
//! tail tokens and overrides; the records replay into exactly those rows; games aborted at the row limit are recorded
//! (abort_reason) and replay as aborted, with no samples; dense shards are refused for such a game.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};
use zero_search::toy_tail::TailGame;
use zero_search::SearchGame;
use zero_worker::job::{run_job, Control, EvalKind, Job, Progress, Sink};
use zero_worker::npz::read_npz;
use zero_worker::records::Output;
use zero_worker::settings::Settings;
use zero_worker::GameEntry;

static TAIL: GameEntry = GameEntry::of::<TailGame>("toy_tail");

#[derive(Default)]
struct Events(Mutex<Vec<Value>>);

impl Sink for Events {
    fn event(&self, v: Value) {
        self.0.lock().unwrap().push(v);
    }
    fn log(&self, _msg: &str) {}
}

fn job(dir: &Path, spec: &str, games: u64, sparse: bool) -> Job {
    let settings = Settings::from_value(&json!({
        "game": "toy_tail", "spec": spec, "players": [2], "conc": 6, "threads": 2, "groups": 2, "max_steps": 10000,
        "sims": 12, "sims_fast": 4, "p_full": 0.5, "c_puct": 1.0, "alpha": 0.3, "eps": 0.25, "temp_moves": 10,
        "chance_k": 0, "margin_w": 0.0, "redet": 1, "shard_rows": 120, "shard_sparse": sparse as i32, "sparse": 0,
        "selfplay_view_samples": 0.5
    }))
    .unwrap();
    Job {
        lease_id: "t".into(),
        settings,
        eval: EvalKind::Uniform,
        games,
        gid_base: 0,
        seed: 41,
        net: 3,
        search_threads: Some(2),
        conc: None,
        groups: None,
        out_dir: dir.to_path_buf(),
        prefix: "f".into(),
        meta: Map::new(),
        record_batches: None,
        status_s: 0.0,
        output: Output::Both,
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zero-worker-tail-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn run(j: &Job) -> Result<zero_worker::job::Outcome, String> {
    let ctl = Control::default();
    let ev = Events::default();
    run_job(&TAIL, j, &ctl, &ev, Arc::new(Mutex::new(Progress::default())), &|_| {})
}

fn read_records(p: &Path) -> (Value, Vec<String>) {
    use std::io::Read;
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(p).unwrap()).read_to_string(&mut text).unwrap();
    let mut lines = text.lines();
    let h: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    (h, lines.map(String::from).collect())
}

#[test]
fn tail_shards_are_format_v2_and_records_replay_to_them() {
    let d = tmp("v2");
    let j = job(&d, r#"{"items0": 9}"#, 24, true);
    let o = run(&j).unwrap();
    assert_eq!(o.reason, "done");
    assert!(!o.shards.is_empty() && o.records.len() == o.shards.len());
    let fmt = zero_worker::shard::RowsFmt::of(&(TAIL.info)()).unwrap();
    let (mut tails, mut ovs) = (0usize, 0usize);
    for (rw, sw) in o.records.iter().zip(&o.shards) {
        let z = read_npz(&sw.path).unwrap();
        let meta: Value = serde_json::from_str(&z["meta"].text().unwrap()).unwrap();
        assert_eq!(meta["x_version"], json!(2));
        assert_eq!(meta["x_format"], json!("sparse"));
        assert_eq!(meta["t_segs"], json!([[0, "items", 16, 4], [1, "marks", 4, 0]]));
        assert_eq!((meta["option_pointers"].clone(), meta["max_row_tokens"].clone()), (json!(true), json!(TailGame::MAX_ROW_TOKENS)));
        let names: Vec<&str> = ["t_ptr", "t_seg", "t_slot", "t_val", "ov_ptr", "ov_opt", "ov_tok"].to_vec();
        for n in &names {
            assert!(z.contains_key(*n), "{n}");
        }
        assert_eq!((z["t_seg"].descr.as_str(), z["t_slot"].descr.as_str(), z["t_val"].descr.as_str()), ("<u2", "<u4", "<f2"));
        tails += z["t_seg"].shape[0];
        ovs += z["ov_opt"].shape[0];
        // the records' rows, rebuilt and written by the same function, are the shard's arrays byte for byte
        let (h, lines) = read_records(&rw.path);
        let games: Vec<zero_worker::shard::GameSamples> = lines
            .iter()
            .map(|l| {
                let rp = zero_search::record::replay_json::<TailGame>(h["spec"].as_str().unwrap(), l, false).unwrap();
                zero_worker::shard::GameSamples { gid: rp.gid, records: rp.records }
            })
            .filter(|g| !g.records.is_empty())
            .collect();
        let (arrays, _, probs) = zero_worker::shard::shard_arrays(&games, TailGame::NUM_FEATURES, true, Some(&fmt)).unwrap();
        assert!(probs.is_empty());
        assert_eq!(z.len() - 1, arrays.len());
        for (name, a) in &arrays {
            assert_eq!(z[*name].data, a.data, "{name}: rebuilt rows == the shard's");
        }
    }
    assert!(tails > 20 && ovs > 20, "tail tokens {tails}, overrides {ovs}");
}

#[test]
fn games_aborted_at_the_row_limit_are_recorded_and_replay() {
    // 70 items: every row is past the game's 64 tokens - every game aborted at its first row, no samples
    let d = tmp("abort");
    let j = job(&d, r#"{"items0": 70}"#, 6, true);
    let o = run(&j).unwrap();
    assert_eq!(o.reason, "done");
    assert!(o.shards.is_empty(), "no samples, no shard");
    let mut n = 0;
    for rw in &o.records {
        let (h, lines) = read_records(&rw.path);
        for l in &lines {
            let rec: Value = serde_json::from_str(l).unwrap();
            assert_eq!(rec["abort_reason"], json!("max_row_tokens"));
            let rp = zero_search::record::replay_json::<TailGame>(h["spec"].as_str().unwrap(), l, false).unwrap();
            assert!(rp.aborted && !rp.over && rp.records.is_empty() && rp.abort_reason == Some("max_row_tokens"));
            n += 1;
        }
    }
    assert_eq!(n, 6, "every game recorded");
}

#[test]
fn dense_shards_are_refused_for_a_game_with_tails() {
    let d = tmp("dense");
    let mut j = job(&d, r#"{"items0": 9}"#, 2, false);
    j.output = Output::Shards;
    let e = run(&j).err().expect("a dense shard of tail rows");
    assert!(e.contains("sparse"), "{e}");
}
