//! Self-play records (zero-farm DESIGN.md 6.1, PROTOCOL.md 5 `records`; zero-core docs/search.md 2.17): a job's
//! finished games as what replays them - each game's seed, every real move and only the search's outputs -
//! instead of their sample rows. The trainer rebuilds the rows (`zero_search::record::replay`, the engine module's
//! `replay_rows`) bit for bit as the shards would hold them, and replay is its validation.
//!
//! A records file is gzip'd JSON lines: first a header
//!
//! ```text
//! {"records": 1, "game", "spec" (the spec the games were played with, as the driver got it), "aux" (rows carry
//!  the auxiliary targets), "features", "actions", "aux_width", "seed" (the driver's), "lease", "build_id",
//!  "games", "rows", "gids" ([[a, b), ...]), "gen" (the oldest network of its rows, or null)}
//! ```
//!
//! and, only in a file written after a game of the job was dropped after a panic (zero-core docs/search.md 2.18),
//! `"failures"`: those games' failure records (`GameFailure::to_json`) since the last file - a key older readers
//! ignore (the desktop's pool collector appends them to the run's game_failures.jsonl). A dropped game itself is in
//! no file (no samples, no trace).
//!
//! then one game record per line (`zero_search::record::game_json`), in the order the games finished. Like the
//! shard writer it cuts a file once it holds `rows` training rows (whole games) and writes the rest at the end;
//! games without rows (cut short or aborted) are recorded too: they are finished games of the job's ids.
//!
//! Branched games (`selfplay_branch`, zero-core docs/search.md 2.11 and 2.17; `RecordWriter::set_branches`): the
//! header says `"records": 2` and a branch game is one more line under its game's gid - **always in its game's file**:
//! a game whose row says `branched` is held (no file is cut, and a timed flush leaves it out) until its branch's line
//! follows, so a branch's record always finds its game's record to replay with, and the coordinator credits the two as
//! one game id (zero-farm PROTOCOL.md 5). Only the job's end writes a game whose branch never came (the phase ended).
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::{Compression, GzBuilder};
use serde_json::{json, Map, Value};
use zero_search::driver::{GameResult, Record};
use zero_search::record::{game_json, RECORDS_VERSION, RECORDS_VERSION_BRANCH};

use crate::job::ranges;
use crate::shard::Written;

/// What a job's records files are written as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// Sample shards (zero-core's npz; before phase 3, and the desktop's `selfplay` default).
    Shards,
    /// Game records (`records` events; a `serve` job's default from phase 3).
    Records,
    /// Both (the equivalence gate: rows rebuilt from the records == the shards' rows).
    Both,
}

impl Output {
    pub fn parse(s: &str) -> Result<Output, String> {
        match s {
            "shards" => Ok(Output::Shards),
            "records" => Ok(Output::Records),
            "both" => Ok(Output::Both),
            _ => Err(format!("output {s:?}: 'records', 'shards' or 'both'")),
        }
    }
    pub fn shards(self) -> bool {
        self != Output::Records
    }
    pub fn records(self) -> bool {
        self != Output::Shards
    }
}

pub struct RecordWriter {
    dir: PathBuf,
    prefix: String,
    rows: usize,
    header: Map<String, Value>,
    level: u32,
    lines: Vec<String>,
    gids: Vec<u32>,
    nbuf: usize,
    gen: Option<u32>,
    /// Failure records of dropped games (module docs) for the next file.
    failures: Vec<Value>,
    /// The header's `records` version: `RECORDS_VERSION`, or `RECORDS_VERSION_BRANCH` for a job with branched games.
    version: u32,
    /// Games (by gid) whose row says `branched` and whose branch's line has not come yet (module docs).
    waiting: std::collections::HashSet<u32>,
    pub seq: usize,
    pub written: Vec<Written>,
}

impl RecordWriter {
    /// `header`: the file header's fixed fields (module docs: game, spec, aux, features, actions, aux_width, seed,
    /// lease, build_id); `rows`: a file once this many rows are held (whole games).
    pub fn new(dir: &Path, prefix: &str, rows: usize, header: Map<String, Value>, level: u32) -> Result<RecordWriter, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(RecordWriter { dir: dir.to_path_buf(), prefix: prefix.to_string(), rows: rows.max(1), header, level, lines: Vec::new(), gids: Vec::new(), nbuf: 0, gen: None, failures: Vec::new(), version: RECORDS_VERSION, waiting: Default::default(), seq: 0, written: Vec::new() })
    }

    /// The job plays branched games: its files are `records/2` (module docs).
    pub fn set_branches(&mut self) {
        self.version = RECORDS_VERSION_BRANCH;
    }

    /// A finished game (its trace on) and its samples -> the file written, if any.
    pub fn add(&mut self, row: &GameResult, samples: &[Record]) -> Result<Option<Written>, String> {
        let rec = game_json(row, samples)?;
        self.lines.push(serde_json::to_string(&rec).map_err(|e| e.to_string())?);
        self.gids.push(row.gid);
        self.nbuf += samples.len();
        if let Some(m) = samples.iter().map(|r| r.net).min() {
            self.gen = Some(self.gen.map_or(m, |g| g.min(m)));
        }
        // (a game and its branch, one file: module docs)
        if row.branch.is_some() {
            self.waiting.remove(&row.gid);
        } else if row.branched {
            if self.version != RECORDS_VERSION_BRANCH {
                return Err(format!("game {}: a branched game in a records/1 writer (RecordWriter::set_branches)", row.gid));
            }
            self.waiting.insert(row.gid);
        }
        if self.nbuf >= self.rows && self.waiting.is_empty() {
            return self.flush();
        }
        Ok(None)
    }

    /// A game dropped after a panic: its failure record rides in the next file's header (module docs).
    pub fn add_failure(&mut self, f: Value) {
        self.failures.push(f);
    }

    /// Failure records no file holds yet (a file is written only with a game in it).
    pub fn pending_failures(&self) -> usize {
        self.failures.len()
    }

    /// Write what is held, but a game still waiting for its branch's line (module docs): it stays for the next file.
    pub fn flush(&mut self) -> Result<Option<Written>, String> {
        if self.waiting.is_empty() {
            return self.write_all();
        }
        let (mut keep_l, mut keep_g, mut keep_n) = (Vec::new(), Vec::new(), 0usize);
        let (mut out_l, mut out_g, mut keep_gen) = (Vec::new(), Vec::new(), None);
        for (l, g) in std::mem::take(&mut self.lines).into_iter().zip(std::mem::take(&mut self.gids)) {
            if self.waiting.contains(&g) {
                // (its rows and networks: re-read from the line - they count toward the next file's rows and gen)
                let v = serde_json::from_str::<Value>(&l).unwrap_or_default();
                keep_n += v["samples"]["step"].as_array().map_or(0, |a| a.len());
                if let Some(m) = v["samples"]["net"].as_array().and_then(|a| a.iter().filter_map(|x| x.as_u64()).min()) {
                    keep_gen = Some(keep_gen.map_or(m as u32, |k: u32| k.min(m as u32)));
                }
                keep_l.push(l);
                keep_g.push(g);
            } else {
                out_l.push(l);
                out_g.push(g);
            }
        }
        self.lines = out_l;
        self.gids = out_g;
        self.nbuf -= keep_n;
        let w = if self.lines.is_empty() { None } else { self.write_all()? };
        self.lines = keep_l;
        self.gids = keep_g;
        self.nbuf = keep_n;
        self.gen = keep_gen;
        Ok(w)
    }

    /// The job's end: everything held, a game whose branch never came included.
    pub fn finish(&mut self) -> Result<Option<Written>, String> {
        self.waiting.clear();
        self.write_all()
    }

    fn write_all(&mut self) -> Result<Option<Written>, String> {
        if self.lines.is_empty() {
            return Ok(None);
        }
        let lines = std::mem::take(&mut self.lines);
        let gids = std::mem::take(&mut self.gids);
        let rows = std::mem::take(&mut self.nbuf);
        let gen = self.gen.take();
        let mut h = Value::Object(self.header.clone());
        h["records"] = json!(self.version);
        h["games"] = json!(gids.len());
        h["rows"] = json!(rows);
        h["gids"] = json!(ranges(&gids).into_iter().map(|r| [r.0, r.1]).collect::<Vec<_>>());
        h["gen"] = json!(gen);
        if !self.failures.is_empty() {
            h["failures"] = Value::Array(std::mem::take(&mut self.failures));
        }
        let name = format!("{}_{:03}.rec.jsonl.gz", self.prefix, self.seq);
        let path = self.dir.join(&name);
        let tmp = self.dir.join(format!("{name}.{}.tmp", std::process::id()));
        let ioe = |e: std::io::Error| format!("{}: {e}", tmp.display());
        {
            let f = File::create(&tmp).map_err(ioe)?;
            let mut z = GzBuilder::new().mtime(0).write(BufWriter::new(f), Compression::new(self.level));
            z.write_all(serde_json::to_string(&h).map_err(|e| e.to_string())?.as_bytes()).map_err(ioe)?;
            z.write_all(b"\n").map_err(ioe)?;
            for l in &lines {
                z.write_all(l.as_bytes()).map_err(ioe)?;
                z.write_all(b"\n").map_err(ioe)?;
            }
            let mut w = z.finish().map_err(ioe)?;
            w.flush().map_err(ioe)?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let bytes = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?.len();
        let w = Written { seq: self.seq, path, rows, bytes, games: gids.len(), gids, gen };
        self.seq += 1;
        self.written.push(w.clone());
        Ok(Some(w))
    }

    /// Games added that are in no file yet.
    pub fn pending_games(&self) -> usize {
        self.lines.len()
    }
}
