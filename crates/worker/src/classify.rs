//! `check-config`: every run setting classified for a native self-play worker, from the explicit
//! table `run_settings.json` (one entry per zero-core `zero.run.DEFAULTS` key; zero-core's
//! tests/test_worker_settings.py fails when the two differ, so a new setting can never be ignored in
//! silence - docs/az.md 44).
//!
//! Classes: `play` (read by the worker and supported), `trainer` (deliberately unused by a self-play
//! worker), `python_only` (play-affecting, implemented in Python only: unsupported unless off),
//! `param` (a setting of a python_only switch: unused while it is off), and the game's own
//! `aux_<target>_w` (play) / `reader_<target>` (trainer) for its auxiliary targets, and the game
//! package's own run settings (zero-core `GamePackage.settings`, docs/az.md 40; the worker binary lists
//! them with their classes: `Worker::package_settings`). A key in no class is unknown. Unknown keys and
//! python_only switches that are on make the check fail.
//!
//! An eval job (`classify_eval`, docs/az.md 44.2) reads only the settings of an evaluation match
//! (`settings::EVAL_READS`: `Run.nn_seat(key, eval_sims)` and `Run.driver` of a match): those are `play`,
//! every other known setting `trainer` (unused by the job, whatever its value: a self-play feature that is
//! Python-only does not change an evaluation's games), unknown keys unknown.
use serde_json::Value;
use zero_search::dynapi::GameInfo;

use crate::pyjson::PyJson;
use crate::settings::{truthy, Settings};

pub const TABLE: &str = include_str!("run_settings.json");

/// A game package's own run settings (zero-core `GamePackage.settings`): (name, class, why), class `play`,
/// `trainer`, `python_only` or `param:<its switch>` (the table's classes).
pub type PackageSettings = &'static [(&'static str, &'static str, &'static str)];

#[derive(Clone, Debug, PartialEq)]
pub enum Class {
    Play,
    Trainer,
    /// Off: unused; on: unsupported.
    PythonOnly,
    /// A setting of a python_only switch.
    Param(String),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub class: Class,
    pub why: String,
}

/// The table (keys in name order) and its patterns.
pub fn table() -> Result<(Vec<(String, Entry)>, Vec<(String, String, Class, String)>), String> {
    let v: Value = serde_json::from_str(TABLE).map_err(|e| format!("run_settings.json: {e}"))?;
    let class_of = |o: &Value| -> Result<Class, String> {
        Ok(match o["class"].as_str() {
            Some("play") => Class::Play,
            Some("trainer") => Class::Trainer,
            Some("python_only") => Class::PythonOnly,
            Some("param") => Class::Param(o["of"].as_str().ok_or("run_settings.json: a param without 'of'")?.to_string()),
            c => return Err(format!("run_settings.json: class {c:?}")),
        })
    };
    let mut keys = Vec::new();
    for (k, o) in v["keys"].as_object().ok_or("run_settings.json: no keys")? {
        keys.push((k.clone(), Entry { class: class_of(o)?, why: o["why"].as_str().unwrap_or("").to_string() }));
    }
    let mut pats = Vec::new();
    for p in v["patterns"].as_array().ok_or("run_settings.json: no patterns")? {
        pats.push((p["prefix"].as_str().unwrap_or("").to_string(), p["suffix"].as_str().unwrap_or("").to_string(), class_of(p)?, p["why"].as_str().unwrap_or("").to_string()));
    }
    Ok((keys, pats))
}

/// One setting's verdict.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub key: String,
    pub value: String,
    /// 'play' | 'trainer' | 'off' (a python_only switch that is off, or its param) | 'unsupported' |
    /// 'unknown'
    pub status: &'static str,
    pub why: String,
}

/// Every setting of `c` for a worker of `info`'s game; `others`: the other games the worker build
/// registers (a game package's settings `aux_<t>_w` / `reader_<t>` exist for every game's targets:
/// another game's are harmless while off - Python ignores them - and refused when on).
pub fn classify(c: &Settings, info: &GameInfo, others: &[GameInfo]) -> Result<Vec<Verdict>, String> {
    classify_with(c, info, others, &[])
}

/// `classify` with the game package's own settings.
pub fn classify_with(c: &Settings, info: &GameInfo, others: &[GameInfo], package: PackageSettings) -> Result<Vec<Verdict>, String> {
    let (mut keys, pats) = table()?;
    for (k, cl, why) in package {
        if keys.iter().any(|e| e.0 == *k) {
            return Err(format!("the game package's setting {k} is a zero-core run setting too"));
        }
        let class = match *cl {
            "play" => Class::Play,
            "trainer" => Class::Trainer,
            "python_only" => Class::PythonOnly,
            p if p.starts_with("param:") => Class::Param(p[6..].to_string()),
            other => return Err(format!("the game package's setting {k}: class {other:?}")),
        };
        keys.push((k.to_string(), Entry { class, why: format!("the game package's: {why}") }));
    }
    let mut out = Vec::new();
    let null = PyJson::Null;
    for k in c.keys() {
        let v = c.get(k).unwrap_or(&null);
        let value = v.dumps();
        let found = keys.iter().find(|e| e.0 == k).map(|e| (e.1.class.clone(), e.1.why.clone())).or_else(|| {
            pats.iter().find_map(|(pre, suf, cl, why)| {
                let t = k.strip_prefix(pre.as_str())?.strip_suffix(suf.as_str())?;
                info.aux.iter().any(|a| a.0 == t).then(|| (cl.clone(), format!("{why} (target {t:?})")))
            })
        });
        let other = || {
            pats.iter().find_map(|(pre, suf, _, _)| {
                let t = k.strip_prefix(pre.as_str())?.strip_suffix(suf.as_str())?;
                others.iter().find(|g| g.aux.iter().any(|a| a.0 == t)).map(|g| format!("a setting of {:?}'s auxiliary target {t:?}, which {} has not (Python ignores it)", g.name, info.name))
            })
        };
        let (status, why) = match found {
            None if other().is_some() => (if truthy(v) { "unsupported" } else { "off" }, other().unwrap()),
            None => ("unknown", format!("not a zero-core run setting of this build, nor an auxiliary target setting of {} ({:?})", info.name, info.aux.iter().map(|a| a.0).collect::<Vec<_>>())),
            Some((Class::Play, w)) => ("play", w),
            Some((Class::Trainer, w)) => ("trainer", w),
            Some((Class::PythonOnly, w)) => {
                if truthy(v) {
                    ("unsupported", w)
                } else {
                    ("off", w)
                }
            }
            Some((Class::Param(s), w)) => ("off", w + if c.get(&s).is_some_and(truthy) { " (its switch is on: see that one)" } else { "" }),
        };
        out.push(Verdict { key: k.to_string(), value, status, why });
    }
    if c.get("game").and_then(|g| match g {
        PyJson::Str(s) => Some(s.as_str()),
        _ => None,
    }) != Some(info.name)
    {
        out.push(Verdict { key: "game".into(), value: c.get("game").map_or("missing".into(), |g| g.dumps()), status: "unsupported", why: format!("this worker plays {:?} here", info.name) });
    }
    Ok(out)
}

/// Every setting of `c` for an eval job (module docs): the settings an evaluation match reads are `play`,
/// every other known one `trainer` (unused), unknown keys and another game's settings as `classify` says.
pub fn classify_eval(c: &Settings, info: &GameInfo, others: &[GameInfo], package: PackageSettings) -> Result<Vec<Verdict>, String> {
    let mut v = classify_with(c, info, others, package)?;
    for x in &mut v {
        if crate::settings::EVAL_READS.contains(&x.key.as_str()) {
            if x.status != "unsupported" {
                x.status = "play";
                x.why = "read by an evaluation match (Run.nn_seat(key, eval_sims), Run.driver of a match)".into();
            }
        } else if matches!(x.status, "play" | "trainer" | "off" | "unsupported") && !(x.key == "game" && x.status == "unsupported") {
            x.why = format!("not read by an evaluation match ({})", x.why);
            x.status = "trainer";
        }
    }
    Ok(v)
}

/// The failing verdicts (unknown keys, python_only switches on).
pub fn failures(v: &[Verdict]) -> Vec<&Verdict> {
    v.iter().filter(|x| matches!(x.status, "unknown" | "unsupported")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_parses_and_is_consistent() {
        let (keys, pats) = table().unwrap();
        assert!(keys.len() >= 150);
        assert_eq!(pats.len(), 2);
        for (k, e) in &keys {
            if let Class::Param(s) = &e.class {
                let sw = keys.iter().find(|x| &x.0 == s).unwrap_or_else(|| panic!("{k}: its switch {s} is not in the table"));
                assert_eq!(sw.1.class, Class::PythonOnly, "{k}: its switch {s} is not python_only");
            }
            assert!(!e.why.is_empty(), "{k}: no reason");
        }
    }

    #[test]
    fn verdicts() {
        let info = GameInfo { name: "t", num_actions: 2, num_features: 2, min_players: 2, max_players: 2, bots: vec![], aux: vec![("deck", 3)] };
        let c = Settings::parse(r#"{"game": "t", "sims": 4, "lr": 0.1, "selfplay_league": 0.0, "selfplay_league_pool": "2,4", "resolver": "x", "aux_deck_w": 0.5, "reader_deck": 1, "aux_hand_w": 1, "bogus": 1}"#).unwrap();
        let other = GameInfo { name: "o", num_actions: 2, num_features: 2, min_players: 2, max_players: 2, bots: vec![], aux: vec![("hand", 3)] };
        let v = classify(&c, &info, &[other.clone()]).unwrap();
        let st: Vec<(&str, &str)> = v.iter().map(|x| (x.key.as_str(), x.status)).collect();
        assert_eq!(st, vec![("game", "play"), ("sims", "play"), ("lr", "trainer"), ("selfplay_league", "off"), ("selfplay_league_pool", "off"), ("resolver", "unsupported"), ("aux_deck_w", "play"), ("reader_deck", "trainer"), ("aux_hand_w", "unsupported"), ("bogus", "unknown")]);
        assert_eq!(failures(&v).len(), 3);
        let c = Settings::parse(r#"{"game": "t", "aux_hand_w": 0.0, "aux_cards_w": 0.0}"#).unwrap();
        let st: Vec<&str> = classify(&c, &info, &[other.clone()]).unwrap().iter().map(|x| x.status).collect();
        assert_eq!(st, vec!["play", "off", "unknown"]);
        // a game package's own settings; an eval job's reading of the same settings
        const PKG: PackageSettings = &[("deck_every", "python_only", "a self-play spec hook"), ("deck_window", "param:deck_every", "its window"), ("real_games", "trainer", "an extra evaluation")];
        let c = Settings::parse(r#"{"game": "t", "sims": 4, "eval_sims": 8, "eval_select": "cq", "selfplay_league": 0.5, "resolver": "x", "deck_every": 2, "deck_window": 4, "real_games": 1, "aux_deck_w": 0.5, "bogus": 1}"#).unwrap();
        let vv = classify_with(&c, &info, &[other.clone()], PKG).unwrap();
        let st: Vec<(&str, &str)> = vv.iter().map(|x| (x.key.as_str(), x.status)).collect();
        assert_eq!(st, vec![("game", "play"), ("sims", "play"), ("eval_sims", "trainer"), ("eval_select", "trainer"), ("selfplay_league", "unsupported"), ("resolver", "unsupported"), ("deck_every", "unsupported"), ("deck_window", "off"), ("real_games", "trainer"), ("aux_deck_w", "play"), ("bogus", "unknown")]);
        let vv = classify_eval(&c, &info, &[other], PKG).unwrap();
        let st: Vec<(&str, &str)> = vv.iter().map(|x| (x.key.as_str(), x.status)).collect();
        assert_eq!(st, vec![("game", "play"), ("sims", "trainer"), ("eval_sims", "play"), ("eval_select", "play"), ("selfplay_league", "trainer"), ("resolver", "trainer"), ("deck_every", "trainer"), ("deck_window", "trainer"), ("real_games", "trainer"), ("aux_deck_w", "trainer"), ("bogus", "unknown")]);
        assert!(classify_with(&c, &info, &[], &[("sims", "play", "x")]).is_err());
    }
}
