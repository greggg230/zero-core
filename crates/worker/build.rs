//! A diagnostic worker build may embed the probe's graph variants (zero-farm PROTOCOL 8 `probe-providers`,
//! `src/probe_variants.rs`): `ZERO_PROBE_VARIANTS=<variants.npz>` at build time (made by
//! `tools/onnx_probe_variants.py`; it may hold real networks, so it is never committed). Without it nothing is
//! embedded and `probe-providers` runs as before.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(probe_variants)");
    println!("cargo:rerun-if-env-changed=ZERO_PROBE_VARIANTS");
    if let Ok(p) = std::env::var("ZERO_PROBE_VARIANTS") {
        let p = std::path::PathBuf::from(p);
        let p = p.canonicalize().unwrap_or_else(|e| panic!("ZERO_PROBE_VARIANTS={}: {e}", p.display()));
        println!("cargo:rerun-if-changed={}", p.display());
        println!("cargo:rustc-env=ZERO_PROBE_VARIANTS_FILE={}", p.display());
        println!("cargo:rustc-cfg=probe_variants");
    }
}
