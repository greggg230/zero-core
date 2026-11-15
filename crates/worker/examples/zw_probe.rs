//! A game-less worker binary for the provider tools: `probe-providers`, `eval-batches`, `info` (no game is
//! registered: self-play, eval and fuzz jobs need a game's binary). `cargo build -p zero_worker --example
//! zw_probe --release`; zero-farm PROTOCOL.md 8's gate measurements use it with a real network
//! (`probe-providers --model M --reference R`).
static WORKER: zero_worker::Worker = zero_worker::Worker { games: &[], repo: "none", selfplay: true, fuzz: None, package_settings: &[] };

fn main() {
    std::process::exit(zero_worker::main_with(&WORKER))
}
