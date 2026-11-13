//! The core's toy games as a worker binary (`toy`, `toy_tail`): zero-farm's gate tools on zero-core alone - e.g.
//! `tools/worker_parity.py --package zero` on `toy_tail` (variable-length rows: format v2 shards, the batches' tail
//! tokens and overrides; docs/az.md 46). `cargo build -p zero_worker --example zw_toy --release`.
static GAMES: &[zero_worker::GameEntry] = &[
    zero_worker::GameEntry::of::<zero_search::toy::ToyGame>("toy"),
    zero_worker::GameEntry::of::<zero_search::toy_tail::TailGame>("toy_tail"),
];

fn main() {
    std::process::exit(zero_worker::main(GAMES, "zero-core"))
}
