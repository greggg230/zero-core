//! Self-play records (zero-farm DESIGN.md 6.1, PROTOCOL.md 5 `records`; zero-core docs/search.md 2.15): a job's
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
//! then one game record per line (`zero_search::record::game_json`), in the order the games finished. Like the
//! shard writer it cuts a file once it holds `rows` training rows (whole games) and writes the rest at the end;
//! games without rows (cut short or aborted) are recorded too: they are finished games of the job's ids.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::{Compression, GzBuilder};
use serde_json::{json, Map, Value};
use zero_search::driver::{GameResult, Record};
use zero_search::record::{game_json, RECORDS_VERSION};

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
    pub seq: usize,
    pub written: Vec<Written>,
}

impl RecordWriter {
    /// `header`: the file header's fixed fields (module docs: game, spec, aux, features, actions, aux_width, seed,
    /// lease, build_id); `rows`: a file once this many rows are held (whole games).
    pub fn new(dir: &Path, prefix: &str, rows: usize, header: Map<String, Value>, level: u32) -> Result<RecordWriter, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(RecordWriter { dir: dir.to_path_buf(), prefix: prefix.to_string(), rows: rows.max(1), header, level, lines: Vec::new(), gids: Vec::new(), nbuf: 0, gen: None, seq: 0, written: Vec::new() })
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
        if self.nbuf >= self.rows {
            return self.flush();
        }
        Ok(None)
    }

    pub fn flush(&mut self) -> Result<Option<Written>, String> {
        if self.lines.is_empty() {
            return Ok(None);
        }
        let lines = std::mem::take(&mut self.lines);
        let gids = std::mem::take(&mut self.gids);
        let rows = std::mem::take(&mut self.nbuf);
        let gen = self.gen.take();
        let mut h = Value::Object(self.header.clone());
        h["records"] = json!(RECORDS_VERSION);
        h["games"] = json!(gids.len());
        h["rows"] = json!(rows);
        h["gids"] = json!(ranges(&gids).into_iter().map(|r| [r.0, r.1]).collect::<Vec<_>>());
        h["gen"] = json!(gen);
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
