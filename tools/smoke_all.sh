#!/usr/bin/env bash
# Cross-game smoke (DESIGN.md 1 guard rails): zero-core's own tests, then each migrated game's
# build, tests and CPU smoke training run.
#
#   bash tools/smoke_all.sh                         # the core + the sibling game repos (../glassroad-zero, ../caverna-zero, ../kingdombuilder-zero)
#   bash tools/smoke_all.sh ~/glassroad-zero-wt/zero-core ~/caverna-zero-wt/zero-core
#   bash tools/smoke_all.sh --with-this-core [games...]   # the games against THIS core checkout's HEAD
#   bash tools/smoke_all.sh --with-this-core --build-only [games...]   # build only, no tests/smoke (tiered gates: an
#                                                                        # opt-in core feature with byte-identical
#                                                                        # defaults - "do the other games still BUILD?")
#
# Per game (a repo with <pkg>/az/__init__.py registering a GamePackage and third_party/zero-core):
# tools/build_py.sh, cargo test, python -m pytest tests, then `python -m <pkg>.az.run --preset
# <game>-smoke` (3 iterations, the JS bot through Node as the yardstick) checked for 4 metrics
# rows. By default each game runs at its own pinned core. --with-this-core runs each game in a
# temporary detached worktree of its HEAD whose submodule is moved to this core checkout's HEAD
# (commit your core changes first), built with ZERO_CORE_UNPINNED=1: "do the games still work
# with this core?" before a pin moves. --build-only runs each game's build step alone (no cargo
# test, pytest or smoke run) - cheap confirmation that a core change compiles against every game
# without paying for their full suites, e.g. before a core commit a run is about to pin moves.
#
# Env: SMOKE_DIR (scratch runs / worktrees; default E:/zero-smoke on Windows, /tmp/zero-smoke),
# SKIP_CORE=1 (games only), SKIP_CARGO=1, SKIP_SMOKE=1, CARGO_BUILD_JOBS (default 8),
# PYTHON (default python), SMOKE_TARGET (the cargo target dirs' last component on Windows,
# E:/cargo-target/<repo>/<SMOKE_TARGET>; default smoke: give concurrent or scratch smokes their
# own, a target dir shared by two checkouts can silently reuse the other's build).
# Exit status: the number of failed steps (0 = green).
set -uo pipefail
CORE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export PYTHONIOENCODING=utf-8
WITH_CORE=0
BUILD_ONLY=0
GAMES=()
for a in "$@"; do
  case "$a" in
    --with-this-core) WITH_CORE=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) GAMES+=("$(cd "$a" && pwd)") ;;
  esac
done
if [ ${#GAMES[@]} -eq 0 ]; then
  for g in glassroad-zero caverna-zero kingdombuilder-zero; do
    [ -e "$CORE/../$g/third_party/zero-core" ] && GAMES+=("$(cd "$CORE/../$g" && pwd)")
  done
fi
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) WIN=1; SMOKE_DIR="${SMOKE_DIR:-E:/zero-smoke}"; TBASE="E:/cargo-target" ;;
  *) WIN=0; SMOKE_DIR="${SMOKE_DIR:-/tmp/zero-smoke}"; TBASE="" ;;
esac
mkdir -p "$SMOKE_DIR"
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
STAMP="$(date +%Y%m%d-%H%M%S)"
FAILED=0
SUMMARY=()

step() {   # step <label> <command...>
  local label="$1"; shift
  local t0=$SECONDS
  echo "=== $label"
  if "$@"; then
    SUMMARY+=("ok    $label ($((SECONDS - t0)) s)")
  else
    SUMMARY+=("FAIL  $label ($((SECONDS - t0)) s)")
    FAILED=$((FAILED + 1))
  fi
}


if [ "${SKIP_CORE:-0}" != 1 ]; then
  CTGT="${CARGO_TARGET_DIR:-${TBASE:+$TBASE/zero-core/${SMOKE_TARGET:-smoke}}}"
  [ "${SKIP_CARGO:-0}" != 1 ] && step "core: cargo test" \
    env ${CTGT:+CARGO_TARGET_DIR=$CTGT} cargo test --release --manifest-path "$CORE/Cargo.toml" -q
  step "core: build zero._engine" bash "$CORE/tools/build_py.sh"
  step "core: pytest" bash -c "cd '$CORE' && '$PY' -m pytest tests -q -p no:cacheprovider -W ignore::pytest.PytestWarning --basetemp='$(winpath "$SMOKE_DIR/$STAMP-pytest-core")'"
fi

smoke_run() {   # smoke_run <root> <pkg> <preset>
  local root="$1" pkg="$2" preset="$3"
  local run="$SMOKE_DIR/$STAMP-$pkg-smoke"
  (cd "$root" && "$PY" -m "$pkg.az.run" --run "$(winpath "$run")" --preset "$preset" > "$run.out" 2>&1) || {
    echo "smoke run failed: $run.out"; tail -20 "$run.out"; return 1; }
  "$PY" - "$(winpath "$run")" <<'EOF'
import json, sys
m = [json.loads(l) for l in open(sys.argv[1] + '/metrics.jsonl', encoding='utf-8')]
its = [r['iter'] for r in m]
last = m[-1]
print('iterations', its, '| last: samples', last['selfplay']['samples'], 'eval', last.get('eval_yard'), 'elo', last.get('elo'))
assert its == list(range(len(its))) and len(its) >= 2 and last['selfplay']['unfinished'] == 0
EOF
}

for G in "${GAMES[@]}"; do
  ROOT="$G"
  PKG="$(cd "$G" && ls -d */az/__init__.py 2>/dev/null | grep -v '^third_party' | head -1 | cut -d/ -f1)"
  if [ -z "$PKG" ]; then echo "$G: no <pkg>/az/__init__.py - skipped"; SUMMARY+=("skip  $G"); continue; fi
  # the repo's name (a worktree's too: the main repo's directory)
  NAME="$(basename "$(dirname "$(git -C "$G" rev-parse --path-format=absolute --git-common-dir)")")"
  WT=""
  if [ "$WITH_CORE" = 1 ]; then
    WT="$SMOKE_DIR/$STAMP-$NAME-wt"
    git -C "$G" worktree add --detach "$WT" HEAD >/dev/null 2>&1 || { echo "worktree failed"; FAILED=$((FAILED + 1)); continue; }
    git -C "$WT" submodule update --init third_party/zero-core >/dev/null 2>&1
    git -C "$WT/third_party/zero-core" fetch -q "$(winpath "$CORE")" HEAD && git -C "$WT/third_party/zero-core" checkout -q FETCH_HEAD
    ROOT="$WT"
    echo "=== $NAME: worktree $WT with zero-core $(git -C "$CORE" rev-parse --short HEAD)"
  fi
  TGT="${TBASE:+$TBASE/$NAME/${SMOKE_TARGET:-smoke}}"
  GAME="$(cd "$ROOT" && "$PY" -c "import $PKG.az as a; print(a.PACKAGE.game)" 2>/dev/null)"
  step "$NAME: build $PKG._engine" env ZERO_CORE_UNPINNED="$WITH_CORE" ${TGT:+CARGO_TARGET_DIR=$TGT} bash "$ROOT/tools/build_py.sh"
  if [ "$BUILD_ONLY" = 1 ]; then
    SUMMARY+=("skip  $NAME: cargo test, pytest, smoke run (--build-only)")
  else
    [ "${SKIP_CARGO:-0}" != 1 ] && step "$NAME: cargo test" \
      env ${TGT:+CARGO_TARGET_DIR=$TGT} cargo test --release --manifest-path "$ROOT/Cargo.toml" -q
    step "$NAME: pytest" bash -c "cd '$ROOT' && '$PY' -m pytest tests -q -p no:cacheprovider -W ignore::pytest.PytestWarning --basetemp='$(winpath "$SMOKE_DIR/$STAMP-pytest-$NAME")'"
    [ "${SKIP_SMOKE:-0}" != 1 ] && step "$NAME: smoke run ($GAME-smoke)" smoke_run "$ROOT" "$PKG" "$GAME-smoke"
  fi
  if [ -n "$WT" ]; then
    git -C "$G" worktree remove --force "$WT" >/dev/null 2>&1 || echo "note: remove $WT by hand (git worktree remove --force)"
  fi
done

echo
echo "=== smoke_all summary (core $(git -C "$CORE" rev-parse --short HEAD))"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
[ $FAILED -eq 0 ] && echo "GREEN" || echo "$FAILED step(s) FAILED"
exit $FAILED
