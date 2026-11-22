//! The eval job (zero-farm PROTOCOL.md 5.2) on the toy game with the uniform evaluator (no ONNX Runtime):
//! one result row per game in the protocol's format, the seats swapped between the games of a pair; a lease
//! of game ids `gid_base..` plays exactly those games of the whole match; a yield / resume changes nothing;
//! files hold whole pairs; quit and finish end it as the protocol says; network against bot.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use zero_worker::evaljob::{run_eval_job, EvalJob, LaneMode, ModelSpec};
use zero_worker::job::{do_resume, do_yield, Control, EvalLink, Outcome, Progress, Sink};
use zero_worker::settings::{EvalSeat, Settings};
use zero_worker::GameEntry;

static TOY: GameEntry = GameEntry::of::<zero_search::toy::ToyGame>("toy");

#[derive(Default)]
struct Events(Mutex<Vec<Value>>, Mutex<Vec<String>>);

impl Sink for Events {
    fn event(&self, v: Value) {
        self.0.lock().unwrap().push(v);
    }
    fn log(&self, msg: &str) {
        self.1.lock().unwrap().push(msg.to_string());
    }
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
        own_infer_threads: false,
        lanes: LaneMode::Auto,
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
    let (o, ev, _) = run_logged(j, during);
    (o, ev)
}

/// `run`, with the job's log lines.
fn run_logged(j: &EvalJob, during: impl Fn(&Control, &Mutex<Option<EvalLink>>, &Mutex<Progress>, &Events) + Send + Sync) -> (Outcome, Vec<Value>, Vec<String>) {
    let ctl = Control::default();
    let ev = Events::default();
    let link = Mutex::new(None);
    let progress = Arc::new(Mutex::new(Progress::default()));
    let out = std::thread::scope(|s| {
        let h = s.spawn(|| run_eval_job(&TOY, j, &ctl, &ev, progress.clone(), &|l| *link.lock().unwrap() = Some(l)));
        during(&ctl, &link, &progress, &ev);
        h.join().unwrap()
    });
    (out.unwrap(), ev.0.into_inner().unwrap(), ev.1.into_inner().unwrap())
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

/// `eval_trace` (docs/az.md 50): every row is the untraced row plus `digest` and `rec`; tracing changes no game; each
/// record replays to its row's result from its game's seed (`record::game_seed` of the lease's seed and its gid,
/// paired); the digest is the SHA-256 of the record's sorted-key compact JSON; a lease's traced rows are the whole
/// match's; a bad value is refused.
#[test]
fn traced_rows_are_the_rows_plus_their_records() {
    use zero_search::record::{game_seed, replay_json};
    use zero_search::toy::ToyGame;
    let (a, b, c) = (tmp("untraced"), tmp("traced"), tmp("traced-tail"));
    let mut jt = job(&b, 10, 0, nets());
    let mut s: Value = serde_json::from_str(&settings().raw.dumps()).unwrap();
    s["eval_trace"] = json!(1);
    jt.settings = Settings::from_value(&s).unwrap();
    let (oa, _) = run(&job(&a, 10, 0, nets()), |_, _, _, _| {});
    let (ob, _) = run(&jt, |_, _, _, _| {});
    let mut jc = jt.clone();
    jc.out_dir = c.clone();
    jc.gid_base = 4;
    jc.games = 6;
    let (oc, _) = run(&jc, |_, _, _, _| {});
    let (ra, rb, rc) = (rows(&oa).0, rows(&ob).0, rows(&oc).0);
    assert_eq!(ra.keys().collect::<Vec<_>>(), rb.keys().collect::<Vec<_>>());
    for (g, line) in &rb {
        let plain = ra[g].strip_suffix('}').unwrap();
        assert!(line.starts_with(&format!("{plain},\"digest\":\"")), "game {g}: the untraced row, then digest and rec\n{line}\n{}", ra[g]);
        let v: Value = serde_json::from_str(line).unwrap();
        let rec = serde_json::to_string(&v["rec"]).unwrap();
        assert!(line.ends_with(&format!(",\"rec\":{rec}}}")), "game {g}: rec is the last key, as written");
        assert_eq!(v["digest"].as_str().unwrap(), &zero_worker::proto::sha256_hex(rec.as_bytes())[..16]);
        assert_eq!(v["rec"]["gid"], json!(g));
        assert_eq!(v["rec"]["seed"].as_u64().unwrap(), game_seed(31, *g, true), "game {g}: the game's seed");
        let rp = replay_json::<ToyGame>("{}", &rec, false).unwrap_or_else(|e| panic!("game {g}: {e}"));
        let res: Vec<f64> = v["result"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(rp.result.iter().map(|&x| x as f64).collect::<Vec<_>>(), res, "game {g}: the replay's result is the row's");
        assert_eq!(rp.rounds as u64, v["rounds"].as_u64().unwrap());
        assert!(rp.records.is_empty(), "an eval game records no samples");
        // a tampered move is refused
        let mut bad = v["rec"].clone();
        let steps = bad["steps"].as_array_mut().unwrap();
        if let Some(Value::Number(n)) = steps.first().cloned() {
            steps[0] = json!(n.as_u64().unwrap() + 1000);
            assert!(replay_json::<ToyGame>("{}", &serde_json::to_string(&bad).unwrap(), false).is_err(), "game {g}: a bad move replays");
        }
    }
    for (g, line) in &rc {
        assert_eq!(&rb[g], line, "game {g}: a lease's traced row == the whole match's");
    }
    let mut s2 = s.clone();
    s2["eval_trace"] = json!("yes");
    let mut jbad = jt.clone();
    jbad.settings = Settings::from_value(&s2).unwrap();
    jbad.out_dir = tmp("traced-bad");
    let ctl = Control::default();
    let ev = Events::default();
    let e = run_eval_job(&TOY, &jbad, &ctl, &ev, Arc::new(Mutex::new(Progress::default())), &|_| {}).err().unwrap();
    assert!(e.contains("eval_trace"), "{e}");
}

/// Per-seat settings (docs/az.md 68): a network seat's `set` plays the job's settings with the set over it. A set that
/// restates the job's settings plays the same games as no set (the rows then differ only in their seats, which echo the
/// seat as given); a set that changes a seat's search changes its games; two seats of one network (gen 1 twice) each play
/// their own settings; the seats keep their sides between a pair's games; a lease of such a match plays those games.
#[test]
fn seat_sets_play_their_own_settings() {
    let seat = |v: Value| EvalSeat::parse(&v).unwrap();
    let (a, b, c, d, e) = (tmp("set-none"), tmp("set-same"), tmp("set-sims"), tmp("set-tail"), tmp("set-gen1"));
    let (oa, _) = run(&job(&a, 8, 0, nets()), |_, _, _, _| {});
    // eval_sims 12 and eval_select 'cq' are the job's own: the same games
    let same = vec![seat(json!({"net": 1, "set": {"eval_sims": 12, "eval_select": "cq"}})), EvalSeat::Net(2)];
    let (ob, _) = run(&job(&b, 8, 0, same), |_, _, _, _| {});
    let (ra, rb) = (rows(&oa).0, rows(&ob).0);
    let set_text = r#"{"net":1,"set":{"eval_select":"cq","eval_sims":12}}"#;
    for (g, line) in &rb {
        assert!(line.contains(set_text), "game {g}: the seat as given (keys sorted): {line}");
        assert_eq!(&line.replace(set_text, r#"{"net":1}"#), &ra[g], "game {g}: a set restating the job's settings, the same game");
    }
    // other simulations for side A (and the 'visits' rule for B, from the 'cq' rule of the job): other games
    let other = vec![seat(json!({"net": 1, "set": {"eval_sims": 40}})), seat(json!({"net": 2, "set": {"eval_select": "visits", "cq_c_scale": 1.0}}))];
    let (oc, _) = run(&job(&c, 8, 0, other.clone()), |_, _, _, _| {});
    let rc = rows(&oc).0;
    let strip = |s: &str| {
        let v: Value = serde_json::from_str(s).unwrap();
        (v["result"].clone(), v["rounds"].clone(), v["margin"].clone())
    };
    assert!(rc.iter().any(|(g, l)| strip(l) != strip(&ra[g])), "a set that changes a seat's search plays other games");
    for (g, line) in &rc {
        let v: Value = serde_json::from_str(line).unwrap();
        let (s0, s1) = (json!({"net": 1, "set": {"eval_sims": 40}}), json!({"net": 2, "set": {"cq_c_scale": 1.0, "eval_select": "visits"}}));
        assert_eq!(v["seats"], if g % 2 == 0 { json!([s0, s1]) } else { json!([s1, s0]) }, "game {g}");
    }
    // a lease of games 4..8 of that match: those games, byte for byte
    let (od, _) = run(&job(&d, 4, 4, other), |_, _, _, _| {});
    for (g, line) in &rows(&od).0 {
        assert_eq!(&rc[g], line, "game {g}: a lease's row == the whole match's");
    }
    // one network on both seats, each with its own settings
    let mut j = job(&e, 6, 0, vec![seat(json!({"net": 1, "set": {"eval_sims": 30}})), EvalSeat::Net(1)]);
    j.models.remove(&2);
    let (oe, _) = run(&j, |_, _, _, _| {});
    let re = rows(&oe).0;
    assert_eq!(re.len(), 6);
    let v: Value = serde_json::from_str(&re[&1]).unwrap();
    assert_eq!(v["seats"], json!([{"net": 1}, {"net": 1, "set": {"eval_sims": 30}}]));
    // a set the seat's checks refuse is the job's refusal, before any game
    let mut bad = job(&tmp("set-bad"), 4, 0, vec![seat(json!({"net": 1, "set": {"eval_select": "visits", "eval_select_min_visits": 2}})), EvalSeat::Net(2)]);
    bad.out_dir = tmp("set-bad");
    let ctl = Control::default();
    let ev = Events::default();
    let err = run_eval_job(&TOY, &bad, &ctl, &ev, Arc::new(Mutex::new(Progress::default())), &|_| {}).err().unwrap();
    assert!(err.contains("seat 0") && err.contains("select_min_visits"), "{err}");
}

/// The layout of docs/az.md 71: a job without `groups` plays in one group whatever the settings' `groups`, its seats'
/// batches on an evaluator lane each; the rows are those of the layout before (the settings' groups, one evaluator
/// thread for both networks) byte for byte - a lease of the match too, traced rows too, a yield / resume in the middle
/// too -, with fewer, larger batches.
#[test]
fn one_group_and_lanes_play_the_rows_of_before() {
    let with_groups = |g: u64| {
        let mut s: Value = serde_json::from_str(&settings().raw.dumps()).unwrap();
        s["groups"] = json!(g);
        s["conc"] = json!(12);
        s["eval_trace"] = json!(1);
        Settings::from_value(&s).unwrap()
    };
    let (a, b, c, d) = (tmp("lanes-before"), tmp("lanes-new"), tmp("lanes-tail"), tmp("lanes-yield"));
    // before: the settings' groups 3 (what a job without groups played), one evaluator thread
    let mut ja = job(&a, 24, 0, nets());
    ja.settings = with_groups(3);
    ja.groups = Some(3);
    ja.lanes = LaneMode::One;
    let (oa, _, la) = run_logged(&ja, |_, _, _, _| {});
    // now: the same settings, no job groups -> one group, a lane per seat
    let mut jb = ja.clone();
    jb.out_dir = b.clone();
    jb.groups = None;
    jb.lanes = LaneMode::Auto;
    let (ob, _, lb) = run_logged(&jb, |_, _, _, _| {});
    let line = |l: &[String]| l.iter().find(|x| x.starts_with("eval job")).cloned().unwrap();
    assert!(line(&la).contains("groups 3; 1 evaluator lane(s) (One"), "{}", line(&la));
    assert!(line(&lb).contains("groups 1; 2 evaluator lane(s) (Auto"), "{}", line(&lb));
    assert!(line(&lb).contains("lane 0 (keys m)") && line(&lb).contains("lane 1 (keys o)"), "{}", line(&lb));
    let (ra, rb) = (rows(&oa).0, rows(&ob).0);
    assert_eq!(ra.len(), 24);
    assert_eq!(ra, rb, "the rows (and their records) of before, byte for byte");
    let (na, nb) = (oa.stats["infer_batches"].as_u64().unwrap(), ob.stats["infer_batches"].as_u64().unwrap());
    assert!(nb * 2 < na, "one group: fewer, larger batches ({nb} against {na})");
    // 16 threads and 64 games in flight, the job's groups 2: two lanes per seat (batches to them in turn); no job groups:
    // one group, a lane per seat - the rows of before either way
    let wide = |dir: &str, lanes: LaneMode, groups: Option<usize>| {
        let mut s: Value = serde_json::from_str(&with_groups(3).raw.dumps()).unwrap();
        s["conc"] = json!(64);
        let mut j = job(&tmp(dir), 64, 0, nets());
        j.settings = Settings::from_value(&s).unwrap();
        (j.lanes, j.groups, j.search_threads) = (lanes, groups, Some(16));
        j
    };
    let (o64a, _, l64a) = run_logged(&wide("lanes-64-before", LaneMode::One, Some(3)), |_, _, _, _| {});
    let (o64, _, l64) = run_logged(&wide("lanes-64", LaneMode::Auto, Some(2)), |_, _, _, _| {});
    let (o64b, _, l64b) = run_logged(&wide("lanes-64-g1", LaneMode::Auto, None), |_, _, _, _| {});
    assert!(line(&l64a).contains("conc 64, threads 16, groups 3; 1 evaluator lane(s)"), "{}", line(&l64a));
    assert!(line(&l64).contains("groups 2; 4 evaluator lane(s)"), "{}", line(&l64));
    assert!(line(&l64).contains("lane 1 (keys m)") && line(&l64).contains("lane 3 (keys o)"), "{}", line(&l64));
    assert!(line(&l64b).contains("groups 1; 2 evaluator lane(s) (Auto, 4 intra-op threads each)"), "{}", line(&l64b));
    assert_eq!(rows(&o64).0.len(), 64);
    assert_eq!(rows(&o64).0, rows(&o64a).0);
    assert_eq!(rows(&o64b).0, rows(&o64a).0);
    // ... and 16 threads with 12 games in flight: one group, two lanes
    let mut j16 = jb.clone();
    j16.out_dir = tmp("lanes-16");
    j16.search_threads = Some(16);
    let (o16, _, l16) = run_logged(&j16, |_, _, _, _| {});
    assert!(line(&l16).contains("groups 1; 2 evaluator lane(s)"), "{}", line(&l16));
    assert_eq!(rows(&o16).0, ra);
    // a lease of games 8..20 in the new layout: those games of the match
    let mut jc = jb.clone();
    jc.out_dir = c;
    jc.gid_base = 8;
    jc.games = 12;
    let (oc, _) = run(&jc, |_, _, _, _| {});
    for (g, l) in &rows(&oc).0 {
        assert_eq!(&ra[g], l, "game {g}: a lease's row == the whole match's");
    }
    // a job's own groups is still the job's (the agent's tuning field)
    let mut jg = jb.clone();
    jg.out_dir = tmp("lanes-groups");
    jg.groups = Some(2);
    let (og, _, lg) = run_logged(&jg, |_, _, _, _| {});
    assert!(line(&lg).contains("groups 2;"), "{}", line(&lg));
    assert_eq!(rows(&og).0, ra);
    // a yield and resume with two lanes changes nothing
    let mut jd = jb.clone();
    jd.out_dir = d;
    let (od, _) = run(&jd, |ctl, link, p, ev| {
        while !ev.0.lock().unwrap().iter().any(|e| e["ev"] == "fin") {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let l = link.lock().unwrap().clone();
        assert_eq!(l.as_ref().map(|l| l.workers()), Some(2), "an evaluator thread per lane");
        assert!(do_yield(ctl, l.as_ref(), p).is_some());
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(do_resume(ctl, l.as_ref()).unwrap());
    });
    assert_eq!(od.stats["yields"], 1);
    assert_eq!(rows(&od).0, ra);
    // one network on both seats: two lanes (two sessions of it) on the CPU, the rows of one lane
    let mut je = job(&tmp("lanes-gen1"), 8, 0, vec![EvalSeat::parse(&json!({"net": 1, "set": {"eval_sims": 20}})).unwrap(), EvalSeat::Net(1)]);
    je.models.remove(&2);
    let mut jf = je.clone();
    jf.out_dir = tmp("lanes-gen1-one");
    jf.lanes = LaneMode::One;
    jf.groups = Some(3);
    let (oe, _, le) = run_logged(&je, |_, _, _, _| {});
    let (of, _) = run(&jf, |_, _, _, _| {});
    assert!(line(&le).contains("2 evaluator lane(s)"), "{}", line(&le));
    assert_eq!(rows(&oe).0, rows(&of).0);
    // a bot seat: one lane, for the network's key
    let mut jh = job(&tmp("lanes-bot"), 6, 0, vec![EvalSeat::Bot("heuristic".into()), EvalSeat::Net(2)]);
    jh.settings = with_groups(3);
    let mut ji = jh.clone();
    ji.out_dir = tmp("lanes-bot-one");
    ji.lanes = LaneMode::One;
    ji.groups = Some(3);
    let (oh, _, lh) = run_logged(&jh, |_, _, _, _| {});
    let (oi, _) = run(&ji, |_, _, _, _| {});
    assert!(line(&lh).contains("1 evaluator lane(s) (Auto") && line(&lh).contains("lane 0 (keys o)"), "{}", line(&lh));
    assert_eq!(rows(&oh).0, rows(&oi).0);
}
