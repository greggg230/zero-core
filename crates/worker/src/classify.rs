//! `check-config`: every run setting classified for a native self-play worker, from the explicit
//! table `run_settings.json` (one entry per zero-core `zero.run.DEFAULTS` key; zero-core's
//! tests/test_worker_settings.py fails when the two differ, so a new setting can never be ignored in
//! silence - docs/az.md 42).
//!
//! Classes: `play` (read by the worker and supported), `trainer` (deliberately unused by a self-play
//! worker), `python_only` (play-affecting, implemented in Python only: unsupported unless off),
//! `param` (a setting of a python_only switch: unused while it is off), and the game's own
//! `aux_<target>_w` (play) / `reader_<target>` (trainer) for its auxiliary targets. A key in no
//! class is unknown. Unknown keys and python_only switches that are on make the check fail.
use serde_json::Value;
use zero_search::dynapi::GameInfo;

use crate::pyjson::PyJson;
use crate::settings::{truthy, Settings};

pub const TABLE: &str = include_str!("run_settings.json");

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

/// Every setting of `c` for a worker of `info`'s game.
pub fn classify(c: &Settings, info: &GameInfo) -> Result<Vec<Verdict>, String> {
    let (keys, pats) = table()?;
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
        let (status, why) = match found {
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
        let v = classify(&c, &info).unwrap();
        let st: Vec<(&str, &str)> = v.iter().map(|x| (x.key.as_str(), x.status)).collect();
        assert_eq!(st, vec![("game", "play"), ("sims", "play"), ("lr", "trainer"), ("selfplay_league", "off"), ("selfplay_league_pool", "off"), ("resolver", "unsupported"), ("aux_deck_w", "play"), ("reader_deck", "trainer"), ("aux_hand_w", "unknown"), ("bogus", "unknown")]);
        assert_eq!(failures(&v).len(), 3);
    }
}
