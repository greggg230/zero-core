#!/usr/bin/env bash
# Build the core's own engine module (crate crates/toy_py -> python/zero/_engine.pyd, the toy
# game only) for THIS checkout: the core's tests and toy runs use it. Games build theirs with
# their own tools/build_py.sh (which calls tools/build_engine.sh of their pinned core).
#
#   bash tools/build_py.sh            # release build
#   PYTHON=py bash tools/build_py.sh  # other interpreter
#
# Target dir: $CARGO_TARGET_DIR or E:/cargo-target/zero-core/<branch> (C: is full) on Windows,
# <repo>/target elsewhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) TDEF="E:/cargo-target/zero-core" ;;
  *) TDEF="" ;;
esac
exec bash "$ROOT/tools/build_engine.sh" "$ROOT" python/zero zero_toy_py "$TDEF"
