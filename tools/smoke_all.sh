#!/usr/bin/env bash
# Cross-game smoke (DESIGN.md 1 guard rails): zero-core's own tests, then each migrated game's
# build, tests and CPU smoke training run.
#
#   bash tools/smoke_all.sh                         # the core + the game repos it finds (glassroad-zero, caverna-zero, kingdombuilder-zero; "Finding the games")
#   bash tools/smoke_all.sh ~/glassroad-zero-wt/zero-core ~/caverna-zero-wt/zero-core
#   bash tools/smoke_all.sh --with-this-core [games...]   # the games against THIS core checkout's HEAD
#   bash tools/smoke_all.sh --with-this-core --build-only [games...]   # build only, no tests/smoke (tiered gates: an
#                                                                        # opt-in core feature with byte-identical
#                                                                        # defaults - "do the other games still BUILD?")
#   bash tools/smoke_all.sh --core-only             # the core's own steps, deliberately no game (or SKIP_GAMES=1)
#   bash tools/smoke_all.sh --dry-run [...]         # print the plan (core steps, each game and where it was found), run nothing
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
# Finding the games (no game arguments): glassroad-zero, caverna-zero and kingdombuilder-zero (a
# directory with third_party/zero-core) beside this checkout (../<game>); beside its main checkout
# when this is a git worktree (the directory holding `git rev-parse --git-common-dir`); and when
# this is a submodule (a game's or MTG Zero's third_party/zero-core, in a worktree or not), beside
# the superproject's checkout and the superproject's main checkout. The first hit per game wins.
# SMOKE_GAMES_ROOT=<dir> looks in <dir> only. Finding NO game fails before anything runs (exit 1,
# never GREEN; before 2026-11-04 a run from a worktree found none and printed GREEN - docs/az.md
# 29); a game not found is listed as missing in the summary. An explicit game path that does not
# exist, or is not a game repo, fails as well.
#
# Env: SMOKE_DIR (scratch runs / worktrees; default E:/zero-smoke on Windows, /tmp/zero-smoke),
# SKIP_CORE=1 (games only), SKIP_GAMES=1 (= --core-only), SMOKE_GAMES_ROOT (above), SKIP_CARGO=1,
# SKIP_SMOKE=1, CARGO_BUILD_JOBS (default 8), PYTHON (default python), SMOKE_TARGET (the cargo
# target dirs' last component on Windows, E:/cargo-target/<repo>/<SMOKE_TARGET>; default smoke:
# give concurrent or scratch smokes their own, a target dir shared by two checkouts can silently
# reuse the other's build).
# Exit status: the number of failed steps (0 = green; finding no game counts as one); 2 = bad
# arguments.
set -uo pipefail
CORE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export PYTHONIOENCODING=utf-8
KNOWN_GAMES=(glassroad-zero caverna-zero kingdombuilder-zero)
WITH_CORE=0
BUILD_ONLY=0
DRY_RUN=0
CORE_ONLY=0
[ "${SKIP_GAMES:-0}" = 1 ] && CORE_ONLY=1
GAMES=()
for a in "$@"; do
  case "$a" in
    --with-this-core) WITH_CORE=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    --core-only) CORE_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "smoke_all: unknown option $a (see --help)" >&2; exit 2 ;;
    *) [ -d "$a" ] || { echo "smoke_all: no such game checkout: $a" >&2; exit 2; }
       GAMES+=("$(cd "$a" && pwd)") ;;
  esac
done
EXPLICIT=${#GAMES[@]}
if [ "$CORE_ONLY" = 1 ] && [ "$EXPLICIT" -gt 0 ]; then
  echo "smoke_all: --core-only (SKIP_GAMES=1) and game paths contradict each other" >&2; exit 2
fi
if [ "$CORE_ONLY" = 1 ] && [ "${SKIP_CORE:-0}" = 1 ]; then
  echo "smoke_all: --core-only (SKIP_GAMES=1) with SKIP_CORE=1 checks nothing" >&2; exit 2
fi

# ---- finding the games (no explicit paths)
main_checkout() {   # main_checkout <dir>: the main working tree of <dir>'s repo (a worktree's too); nothing for a submodule
  local common
  common="$(git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || return 0
  case "$common" in */.git) dirname "$common" ;; esac
}
ROOTS=()
add_root() {        # add_root <dir>: a directory to look for the games in (once each)
  local d r
  d="$(cd "$1" 2>/dev/null && pwd)" || return 0
  for r in ${ROOTS[@]+"${ROOTS[@]}"}; do [ "$r" = "$d" ] && return 0; done
  ROOTS+=("$d")
}
MISSING=()
if [ "$EXPLICIT" -eq 0 ] && [ "$CORE_ONLY" = 0 ]; then
  if [ -n "${SMOKE_GAMES_ROOT:-}" ]; then
    add_root "$SMOKE_GAMES_ROOT"
    [ ${#ROOTS[@]} -eq 0 ] && echo "smoke_all: SMOKE_GAMES_ROOT=$SMOKE_GAMES_ROOT is not a directory"
  else
    add_root "$CORE/.."
    m="$(main_checkout "$CORE")"; [ -n "$m" ] && add_root "$m/.."
    sp="$(git -C "$CORE" rev-parse --show-superproject-working-tree 2>/dev/null)"
    if [ -n "$sp" ]; then
      add_root "$sp/.."
      m="$(main_checkout "$sp")"; [ -n "$m" ] && add_root "$m/.."
    fi
  fi
  for g in "${KNOWN_GAMES[@]}"; do
    found=""
    for r in ${ROOTS[@]+"${ROOTS[@]}"}; do
      if [ -e "$r/$g/third_party/zero-core" ]; then found="$r/$g"; break; fi
    done
    if [ -n "$found" ]; then GAMES+=("$found"); else MISSING+=("$g"); fi
  done
fi
LOOKED=""
for r in ${ROOTS[@]+"${ROOTS[@]}"}; do LOOKED="${LOOKED:+$LOOKED; }$r"; done

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) WIN=1; SMOKE_DIR="${SMOKE_DIR:-E:/zero-smoke}"; TBASE="E:/cargo-target" ;;
  *) WIN=0; SMOKE_DIR="${SMOKE_DIR:-/tmp/zero-smoke}"; TBASE="" ;;
esac
CORE_SHA="$(git -C "$CORE" rev-parse --short HEAD 2>/dev/null)"
NO_GAMES_MSG="no game repo found and none given (looked for ${KNOWN_GAMES[*]} in: ${LOOKED:-nowhere}) - pass the game checkouts as arguments, set SMOKE_GAMES_ROOT, or --core-only for a deliberate core-only run"

if [ "$DRY_RUN" = 1 ]; then
  echo "smoke_all --dry-run: core $CORE ($CORE_SHA), with-this-core $WITH_CORE, build-only $BUILD_ONLY"
  [ "${SKIP_CORE:-0}" = 1 ] && echo "  core steps: skipped (SKIP_CORE=1)" || echo "  core steps: cargo test, build zero._engine, pytest"
  for G in ${GAMES[@]+"${GAMES[@]}"}; do echo "  game: $G"; done
  for g in ${MISSING[@]+"${MISSING[@]}"}; do echo "  missing: $g (looked in: ${LOOKED:-nowhere})"; done
  if [ "$CORE_ONLY" = 1 ]; then echo "  games: none (--core-only)"; exit 0; fi
  echo "  games: ${#GAMES[@]}"
  [ ${#GAMES[@]} -gt 0 ] && exit 0
  echo "FAIL  $NO_GAMES_MSG"
  exit 1
fi
if [ "$CORE_ONLY" = 0 ] && [ ${#GAMES[@]} -eq 0 ]; then
  echo
  echo "=== smoke_all summary (core $CORE_SHA)"
  echo "  FAIL  games: $NO_GAMES_MSG"
  echo "1 step(s) FAILED"
  exit 1
fi

mkdir -p "$SMOKE_DIR"
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
STAMP="$(date +%Y%m%d-%H%M%S)"
FAILED=0
SUMMARY=()
CHECKED=()

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

if [ "$WITH_CORE" = 1 ] && [ -n "$(git -C "$CORE" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "note: $CORE has uncommitted changes - --with-this-core builds the games against its HEAD $CORE_SHA, not them"
  SUMMARY+=("note  uncommitted core changes are not in the games' builds (--with-this-core uses HEAD $CORE_SHA)")
fi

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

for G in ${GAMES[@]+"${GAMES[@]}"}; do
  ROOT="$G"
  echo "=== game: $G"
  PKG="$(cd "$G" && ls -d */az/__init__.py 2>/dev/null | grep -v '^third_party' | head -1 | cut -d/ -f1)"
  if [ -z "$PKG" ]; then
    echo "$G: no <pkg>/az/__init__.py - not a game repo"
    SUMMARY+=("FAIL  $G: no <pkg>/az/__init__.py (not a game repo)"); FAILED=$((FAILED + 1)); continue
  fi
  # the repo's name (a worktree's too: the main repo's directory)
  NAME="$(basename "$(dirname "$(git -C "$G" rev-parse --path-format=absolute --git-common-dir)")")"
  WT=""
  if [ "$WITH_CORE" = 1 ]; then
    WT="$SMOKE_DIR/$STAMP-$NAME-wt"
    if ! git -C "$G" worktree add --detach "$WT" HEAD >/dev/null 2>&1; then
      echo "$NAME: git worktree add $WT failed"
      SUMMARY+=("FAIL  $NAME: scratch worktree"); FAILED=$((FAILED + 1)); continue
    fi
    git -C "$WT" submodule update --init third_party/zero-core >/dev/null 2>&1
    want="$(git -C "$CORE" rev-parse HEAD)"
    git -C "$WT/third_party/zero-core" fetch -q "$(winpath "$CORE")" HEAD && git -C "$WT/third_party/zero-core" checkout -q FETCH_HEAD
    have="$(git -C "$WT/third_party/zero-core" rev-parse HEAD 2>/dev/null)"
    if [ "$have" != "$want" ]; then
      # never build the game's own pin and call it "this core"
      echo "$NAME: could not move the scratch worktree's zero-core to $want (it is at ${have:-nothing})"
      SUMMARY+=("FAIL  $NAME: zero-core at this core's HEAD"); FAILED=$((FAILED + 1))
      git -C "$G" worktree remove --force "$WT" >/dev/null 2>&1 || echo "note: remove $WT by hand (git worktree remove --force)"
      continue
    fi
    ROOT="$WT"
    echo "=== $NAME: worktree $WT with zero-core $CORE_SHA"
  fi
  TGT="${TBASE:+$TBASE/$NAME/${SMOKE_TARGET:-smoke}}"
  GAME="$(cd "$ROOT" && "$PY" -c "import $PKG.az as a; print(a.PACKAGE.game)" 2>/dev/null)"
  CHECKED+=("$NAME")
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
echo "=== smoke_all summary (core $CORE_SHA)"
for s in ${SUMMARY[@]+"${SUMMARY[@]}"}; do echo "  $s"; done
if [ "$CORE_ONLY" = 1 ]; then
  echo "  games: none (--core-only)"
else
  echo "  games: ${#CHECKED[@]} of ${#GAMES[@]} checked${CHECKED[*]+ (${CHECKED[*]})}"
  for g in ${MISSING[@]+"${MISSING[@]}"}; do echo "  missing: $g (not found in: ${LOOKED:-nowhere})"; done
fi
[ $FAILED -eq 0 ] && echo "GREEN" || echo "$FAILED step(s) FAILED"
exit $FAILED
