#!/usr/bin/env bash
# Build a PyO3 engine module (a game's cdylib crate: zero_py::engine_module! + its games) for ONE
# checkout and install it next to its Python package, so `import <pkg>._engine` from that
# checkout uses that checkout's build (a plain cargo build of the cdylib: no maturin needed).
# Called by each repo's tools/build_py.sh (docs/pinning.md):
#
#   bash build_engine.sh <repo root> <package dir, relative to the root> <crate> [default target dir]
#   e.g. bash third_party/zero-core/tools/build_engine.sh "$ROOT" grz grz_py E:/cargo-target/glassroad-zero
#
# Environment: PYTHON (interpreter to build for and check with; default `python`),
# CARGO_TARGET_DIR (default: <default target dir>/<branch> - a detached checkout: detached-<its directory's
# name>, never one `HEAD` dir shared by every detached worktree -, or <root>/target without one; docs/az.md 46.13).
# Windows locks a loaded .pyd: if the copy fails, the old file is renamed out of the way
# (running processes keep using it; new ones get the new build).
set -euo pipefail
ROOT="$1"
PKGDIR="$2"
CRATE="$3"
TDEF="${4:-}"
BRANCH="$(git -C "$ROOT" symbolic-ref -q --short HEAD 2>/dev/null | tr '/' '_' || true)"
if [ -z "$BRANCH" ]; then
  if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then BRANCH="detached-$(basename "$ROOT")"; else BRANCH=nogit; fi
fi
if [ -z "${CARGO_TARGET_DIR:-}" ]; then
  if [ -n "$TDEF" ]; then CARGO_TARGET_DIR="$TDEF/$BRANCH"; else CARGO_TARGET_DIR="$ROOT/target"; fi
fi
export CARGO_TARGET_DIR
PY="${PYTHON:-python}"
export PYO3_PYTHON="$PY"
cargo build --manifest-path "$ROOT/Cargo.toml" --release -p "$CRATE" --lib

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) SRC="$CARGO_TARGET_DIR/release/$CRATE.dll"; DST="$ROOT/$PKGDIR/_engine.pyd" ;;
  Darwin) SRC="$CARGO_TARGET_DIR/release/lib$CRATE.dylib"; DST="$ROOT/$PKGDIR/_engine.so" ;;
  *) SRC="$CARGO_TARGET_DIR/release/lib$CRATE.so"; DST="$ROOT/$PKGDIR/_engine.so" ;;
esac
rm -f "$ROOT/$PKGDIR"/_engine.*.old 2>/dev/null || true
[ "$(uname -s)" = Darwin ] && rm -f "$DST"
if ! cp -f "$SRC" "$DST" 2>/dev/null; then
  OLD="$DST.$(date +%s).old"
  mv "$DST" "$OLD"
  cp -f "$SRC" "$DST"
  echo "note: $DST was in use; the old build was moved to $OLD"
fi
MOD="$(echo "$PKGDIR" | sed 's#^python/##; s#/#.#g')._engine"
PARENT="$ROOT"
case "$PKGDIR" in python/*) PARENT="$ROOT/python" ;; esac
command -v cygpath >/dev/null 2>&1 && PARENT="$(cygpath -w "$PARENT")"   # Git Bash: a Windows path for Python
"$PY" -c "import sys; sys.path.insert(0, r'$PARENT'); import importlib; e = importlib.import_module('$MOD'); print('$MOD OK:', e.__file__, '| games:', e.game_names())"
