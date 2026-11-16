#!/usr/bin/env bash
# Build the core's own engine module (crate crates/toy_py -> python/zero/_engine.pyd, the toy
# game only) for THIS checkout: the core's tests and toy runs use it. Games build theirs with
# their own tools/build_py.sh (which calls tools/build_engine.sh of their pinned core).
#
#   bash tools/build_py.sh            # release build
#   PYTHON=py bash tools/build_py.sh  # other interpreter
#
# Target dir: $CARGO_TARGET_DIR, else on Windows $ZERO_BIG_DISK/cargo-target/zero-core/<branch> when the machine
# names a big disk (~/.config/zero/local.env: tools/local.env.example), else <repo>/target.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/tools/local_env.sh"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) TDEF="${ZERO_BIG_DISK:+$ZERO_BIG_DISK/cargo-target/zero-core}" ;;
  *) TDEF="" ;;
esac
exec bash "$ROOT/tools/build_engine.sh" "$ROOT" python/zero zero_toy_py "$TDEF"
