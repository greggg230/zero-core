//! The eval job (zero-farm PROTOCOL.md 5.2) on the toy game with the uniform evaluator (no ONNX Runtime):
//! one result row per game in the protocol's format, the seats swapped between the games of a pair; a lease
//! of game ids `gid_base..` plays exactly those games of the whole match; a yield / resume changes nothing;
//! files hold whole pairs; quit and finish end it as the protocol says; network against bot.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use zero_worker::evaljob::{run_eval_job, EvalJob, ModelSpec};
use zero_worker::job::{do_resume, do_yield, Control, EvalLink, Outcome, Progress, Sink};
use zero_worker::settings::{EvalSeat, Settings};
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

fn settings() -> Settings {
    Settings::from_value(&json!({
        "game": "toy", "spec": "", "eval_spec": "{}", "players": [2], "conc": 6, "threads": 2, "groups": 1,
        "max_steps": 10000, "sims": 16, "eval_sims": 12, "c_puct": 1.0, "chance_k": 0, "margin_w": 0.0, "redet": 1,
        "eval_select": "cq", "cq_c_scale": 0.3, "sparse": 0
    }))
    .unwrap()
}

fn job(dir: &Path, games: u64, gid_base: u32, seats: Vec<EvalSeat>) -> EvalJob {
    let mut models = BTreeMap::new();
    for g in [1u32, 2] {
        models.insert(g, ModelSpec { model: PathBuf::from("uniform"), sha256: None, reference: None });
    }
    EvalJob {
        lease_id: "t".into(),
        settings: settings(),
        seats,
        models,
        ort: None,
        games,
        gid_base,
        seed: 31,
        paired: true,
        search_threads: Some(2),
        conc: None,
        groups: None,
        out_dir: dir.to_path_buf(),
        prefix: "e".into(),
        flush_games: 4,
        flush_s: 1e9,
        record_batches: None,
        status_s: 0.0,
    }
}

fn nets() -> Vec<EvalSeat> {
    vec![EvalSeat::Net(1), EvalSeat::Net(2)]
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zero-worker-eval-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn run(j: &EvalJob, during: impl Fn(&Control, &Mutex<Option<EvalLink>>, &Mutex<Progress>, &Events) + Send + Sync) -> (Outcome, Vec<Value>) {
    let ctl = Control::default();
    let ev = Events::default();
    let link = Mutex::new(None);
    let progress = Arc::new(Mutex::new(Progress::default()));
    let out = std::thread::scope(|s| {
        let h = s.spawn(|| run_eval_job(&TOY, j, &ctl, &ev, progress.clone(), &|l| *link.lock().unwrap() = Some(l)));
        during(&ctl, &link, &progress, &ev);
        h.join().unwrap()
    });
    (out.unwrap(), ev.0.into_inner().unwrap())
}

/// Every row of the job's files (gid -> the line), and the files' gids.
fn rows(o: &Outcome) -> (BTreeMap<u32, String>, Vec<Vec<u32>>) {
    let mut by = BTreeMap::new();
    let mut files = Vec::new();
    for w in &o.shards {
        let text = std::fs::read_to_string(&w.path).unwrap();
        let mut gs = Vec::new();
        for line in text.lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            let g = v["gid"].as_u64().unwrap() as u32;
            assert!(by.insert(g, line.to_string()).is_none(), "game {g} twice");
            gs.push(g);
        }
        assert_eq!(gs, w.gids, "a file's rows are its gids, in order");
        files.push(gs);
    }
    (by, files)
}

#[test]
fn rows_pairs_and_files() {
    let d = tmp("rows");
    let (o, ev) = run(&job(&d, 10, 0, nets()), |_, _, _, _| {});
    assert_eq!(o.reason, "done");
    assert!(o.unfinished.is_empty());
    let (by, files) = rows(&o);
    assert_eq!(by.keys().copied().collect::<Vec<_>>(), (0..10).collect::<Vec<u32>>());
    for f in &files {
        assert!(f.len() % 2 == 0 && f.chunks(2).all(|p| p[0] % 2 == 0 && p[1] == p[0] + 1), "whole pairs in a file: {f:?}");
    }
    for (g, line) in &by {
        let v: Value = serde_json::from_str(line).unwrap();
        let want = if g % 2 == 0 { json!([{"net": 1}, {"net": 2}]) } else { json!([{"net": 2}, {"net": 1}]) };
        assert_eq!(v["seats"], want, "game {g}");
        assert_eq!(v["pair"], json!(g / 2));
        let r: Vec<f64> = v["result"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(r.len(), 2);
        assert!((r[0] + r[1] - 1.0).abs() < 1e-9, "a two-player result: {r:?}");
        assert!(v["margin"].is_f64() && v["rounds"].is_u64() && v["over"].is_boolean());
        // the line is json.dumps(row, separators=(',', ':')): keys in the protocol's order
        assert!(line.starts_with(&format!("{{\"gid\":{g},\"pair\":{},\"seats\":[", g / 2)), "{line}");
    }
    let evals: Vec<&Value> = ev.iter().filter(|e| e["ev"] == "evals").collect();
    assert_eq!(evals.len(), files.len());
    assert_eq!(evals.iter().map(|e| e["games"].as_u64().unwrap()).sum::<u64>(), 10);
    assert_eq!(evals[0]["path"], "e_000.jsonl");
    assert_eq!(evals[0]["gids"], json!([[files[0][0], files[0].last().unwrap() + 1]]));
}

#[test]
fn a_lease_plays_those_games_of_the_match() {
    let (a, b, c) = (tmp("all"), tmp("tail"), tmp("head"));
    let (oa, _) = run(&job(&a, 12, 0, nets()), |_, _, _, _| {});
    let (ob, _) = run(&job(&b, 6, 6, nets()), |_, _, _, _| {});
    let (oc, _) = run(&job(&c, 6, 0, nets()), |_, _, _, _| {});
    let (ra, rb, rc) = (rows(&oa).0, rows(&ob).0, rows(&oc).0);
    assert_eq!(rb.keys().copied().collect::<Vec<_>>(), (6..12).collect::<Vec<u32>>());
    for (g, line) in rb.iter().chain(rc.iter()) {
        assert_eq!(&ra[g], line, "game {g}: a lease's row == the whole match's");
    }
}

#[test]
fn yield_resume_changes_nothing() {
    let (a, b) = (tmp("plain"), tmp("yield"));
    let (oa, _) = run(&job(&a, 16, 0, nets()), |_, _, _, _| {});
    let (ob, _) = run(&job(&b, 16, 0, nets()), |ctl, link, p, ev| {
        while !ev.0.lock().unwrap().iter().any(|e| e["ev"] == "fin") {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let l = link.lock().unwrap().clone();
        assert!(do_yield(ctl, l.as_ref(), p).is_some());
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(do_resume(ctl, l.as_ref()).unwrap());
    });
    assert_eq!((oa.reason, ob.reason), ("done", "done"));
    assert_eq!(ob.stats["yields"], 1);
    assert_eq!(rows(&oa).0, rows(&ob).0);
}

#[test]
fn quit_and_finish() {
    let d = tmp("quit");
    let (o, _) = run(&job(&d, 40, 0, nets()), |ctl, _, p, _| {
        while p.lock().unwrap().games < 3 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        ctl.quit();
    });
    assert_eq!(o.reason, "quit");
    let done = rows(&o).0.len() as u32;
    let missing: u32 = o.unfinished.iter().map(|r| r.1 - r.0).sum();
    assert!(done >= 3);
    assert_eq!(done + missing, 40, "{:?}", o.unfinished);
    let d = tmp("finish");
    let (o, _) = run(&job(&d, 40, 0, nets()), |ctl, _, p, _| {
        while p.lock().unwrap().games < 3 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        ctl.finish();
    });
    assert_eq!(o.reason, "finish");
    let done = rows(&o).0.len() as u32;
    assert!((3..40).contains(&done), "{done}");
    // every game started finished (conc 6); the rest never started
    assert!(done <= 3 + 6 + 6, "{done}");
    assert_eq!(o.unfinished.last().unwrap().1, 40);
}

#[test]
fn network_against_a_bot_and_refusals() {
    let d = tmp("bot");
    let (o, _) = run(&job(&d, 6, 2, vec![EvalSeat::Bot("heuristic".into()), EvalSeat::Net(2)]), |_, _, _, _| {});
    let by = rows(&o).0;
    assert_eq!(by.len(), 6);
    let v: Value = serde_json::from_str(&by[&3]).unwrap();
    assert_eq!(v["seats"], json!([{"net": 2}, {"bot": "heuristic"}]));
    // refused: odd paired leases, a seat without its model, an external bot, an unknown bot, no eval_spec
    let ctl = Control::default();
    let ev = Events::default();
    let p = Arc::new(Mutex::new(Progress::default()));
    let try_ = |j: &EvalJob| run_eval_job(&TOY, j, &ctl, &ev, p.clone(), &|_| {}).err().unwrap_or_default();
    assert!(try_(&job(&d, 5, 0, nets())).contains("must be even"));
    assert!(try_(&job(&d, 4, 0, vec![EvalSeat::Net(1), EvalSeat::Net(3)])).contains("models has"));
    assert!(try_(&job(&d, 4, 0, vec![EvalSeat::Net(1), EvalSeat::Bot("nobody".into())])).contains("built-in bots"));
    let mut j = job(&d, 4, 0, nets());
    j.settings.set("eval_spec", zero_worker::pyjson::PyJson::Str(String::new()));
    assert!(try_(&j).contains("eval_spec is unset"));
    assert!(EvalSeat::parse(&json!({"bot": "ext:jsbot"})).unwrap_err().contains("external"));
    assert!(EvalSeat::parse(&json!({"net": 1, "bot": "x"})).is_err());
    assert_eq!(EvalSeat::parse(&json!({"net": 7})).unwrap(), EvalSeat::Net(7));
}
