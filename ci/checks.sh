#!/usr/bin/env bash
# Integration-repository checks.
#
# Written as a script rather than only as a CI workflow because this gitea
# has Actions disabled and no runner, so a workflow file alone would never
# execute. This runs today -- by hand, from a git hook, or from CI once a
# runner exists. .gitea/workflows/ci.yml is a thin wrapper around it.
#
# Lives in ci/ rather than scripts/ because scripts/ is gitignored at the
# repo root as "non-project workspace junk" -- this is project
# infrastructure and has to be tracked.
#
# Usage:
#   ci/checks.sh            # everything, including the CPU-only build
#   ci/checks.sh --quick    # skip the build (~seconds instead of minutes)
#
# Exits non-zero if any check fails. Each check prints PASS or FAIL with the
# reason, and a failure never stops later checks -- one broken thing should
# not hide the others.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

FAILURES=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
section() { printf '\n\033[1m%s\033[0m\n' "$1"; }

VENV="$REPO/modelctl/.venv/bin/python"
[ -x "$VENV" ] || VENV=python3

# --- 1. submodule pin -------------------------------------------------
# The specific failure this exists for: the repo sat five commits ahead of
# its own pin for a day, so a clone built a runtime the docs did not
# describe, and the drift check that should have caught it was reading the
# checked-out commit under the name of the pinned one.
section "submodule pin"
PINNED=$(git -C "$REPO" rev-parse HEAD:llama.cpp 2>/dev/null)
CHECKED=$(git -C "$REPO/llama.cpp" rev-parse HEAD 2>/dev/null)
if [ -z "$PINNED" ]; then
    fail "cannot read the pinned llama.cpp commit"
elif [ "$PINNED" != "$CHECKED" ]; then
    fail "working tree is at ${CHECKED:0:12}, pinned at ${PINNED:0:12}"
else
    pass "pin and working tree agree (${PINNED:0:12})"
fi

URL=$(git -C "$REPO" config -f .gitmodules submodule.llama.cpp.url 2>/dev/null)
if [ -n "$URL" ]; then pass "submodule URL declared: $URL"
else fail "no submodule URL in .gitmodules"; fi

MANIFEST_RT=$("$VENV" - <<'EOF' 2>/dev/null
import json, pathlib, sys
try:
    print(json.loads(pathlib.Path("integration-manifest.json").read_text())
          .get("llama_cpp_commit", ""))
except Exception:
    sys.exit(1)
EOF
)
if [ -z "$MANIFEST_RT" ]; then
    fail "integration-manifest.json unreadable or missing llama_cpp_commit"
elif [ "${MANIFEST_RT:0:12}" != "${PINNED:0:12}" ]; then
    fail "manifest names ${MANIFEST_RT:0:12}, pin is ${PINNED:0:12}"
else
    pass "manifest agrees with the pin"
fi

# --- 2. static import / compile --------------------------------------
section "static checks"
if "$VENV" -m compileall -q "$REPO/modelctl" > /tmp/ci-compileall.log 2>&1; then
    pass "compileall"
else
    fail "compileall -- see /tmp/ci-compileall.log"
fi

if (cd "$REPO/modelctl" && "$VENV" -c "
import importlib, pathlib, sys
bad = []
for p in sorted(pathlib.Path('.').glob('modelctl*.py')):
    if p.name.startswith('test_'):
        continue
    try:
        importlib.import_module(p.stem)
    except Exception as e:
        bad.append(f'{p.stem}: {type(e).__name__}: {e}')
if bad:
    print('\n'.join(bad)); sys.exit(1)
" > /tmp/ci-imports.log 2>&1); then
    pass "every modelctl module imports"
else
    fail "import failure -- $(command head -1 /tmp/ci-imports.log)"
fi

# --- 3. test suite ----------------------------------------------------
# Covers the golden command/provenance and transaction rollback
# jobs, which already live in the suite rather than as separate CI steps.
# Parallel via pytest-xdist (unittest-parallel cannot pickle the TUI's
# IsolatedAsyncioTestCase classes); serial fallback ~218s, parallel ~43s.
section "test suite"
TESTS_T0=$SECONDS
if (cd "$REPO/modelctl" && "$VENV" -m pytest -n auto -q . \
        > /tmp/ci-tests.log 2>&1); then
    pass "$(grep -oE '[0-9]+ passed(, [0-9]+ skipped)?' /tmp/ci-tests.log | tail -1) in $((SECONDS - TESTS_T0))s wall"
else
    fail "tests -- $(grep -oE '[0-9]+ (failed|error)[a-z]*' /tmp/ci-tests.log | tail -1 || echo 'failure'), see /tmp/ci-tests.log"
fi

# --- 3b. console offline build ----------------------------------------
# The /v2 console's toolchain is frozen and self-contained (node_modules
# vendored in-tree, lockfile committed): a clean clone must build with
# zero network. Building to a temp dir catches toolchain rot the week it
# happens without touching the committed dist/ (no byte-comparison by
# design -- content hashes move with every source edit).
section "console offline build"
CONSOLE="$REPO/modelctl/console"
CONSOLE_OUT="${MODELCTL_CI_CONSOLE_DIR:-/tmp/ci-console-dist}"
if ! command -v node >/dev/null; then
    fail "node not found (the console toolchain needs only node itself)"
elif (cd "$CONSOLE" && npm run --offline build -- --outDir "$CONSOLE_OUT" \
        --emptyOutDir > /tmp/ci-console.log 2>&1); then
    pass "console builds offline from the vendored tree"
else
    fail "console build -- see /tmp/ci-console.log"
fi

# --- 4. CPU-only build and capability truthfulness --------------------
# One non-negotiable: "CPU-only capability truthfulness must
# be enforced on every push". A CPU-only build claiming a SYCL feature
# would let modelctl generate cache flags for a binary that cannot honour
# them.
section "CPU-only build and capability truthfulness"
if [ "$QUICK" = "1" ]; then
    printf '  \033[33mSKIP\033[0m  build (--quick)\n'
else
    # Overridable so worktrees and parallel runs don't fight over one
    # cmake cache configured against a different source path.
    BUILD="${MODELCTL_CI_BUILD_DIR:-/tmp/ci-build-cpu}"
    if cmake -B "$BUILD" -S "$REPO/llama.cpp" -DGGML_SYCL=OFF \
            -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
            > /tmp/ci-cmake.log 2>&1 \
       && cmake --build "$BUILD" --target llama-server test-moe-cache \
            test-moe-hybrid -j"$(nproc)" > /tmp/ci-build.log 2>&1; then
        pass "CPU-only build"

        if "$BUILD/bin/test-moe-cache" > /dev/null 2>&1; then
            pass "MoE cache unit tests (host-only, no GPU)"
        else fail "MoE cache unit tests"; fi
        if "$BUILD/bin/test-moe-hybrid" > /dev/null 2>&1; then
            pass "MoE hybrid partition tests (host-only, no GPU)"
        else fail "MoE hybrid partition tests"; fi

        CAPS=$("$BUILD/bin/llama-server" --modelctl-capabilities 2>/dev/null)
        LIARS=$(printf '%s' "$CAPS" | "$VENV" -c "
import json, sys
try:
    f = json.load(sys.stdin).get('features', {})
except Exception:
    print('UNPARSEABLE'); sys.exit(0)
print(','.join(k for k, v in f.items() if v) or 'none')
")
        if [ "$LIARS" = "none" ]; then
            pass "CPU-only build reports every SYCL/cache feature false"
        elif [ "$LIARS" = "UNPARSEABLE" ]; then
            fail "capability response is not valid JSON"
        else
            fail "CPU-only build claims: $LIARS"
        fi
    else
        fail "CPU-only build -- see /tmp/ci-build.log"
    fi
fi

# --- 5. sanitizer pass over the host-only cache/hybrid tests ----------
# The staging/hybrid path's worst failure mode is a lifetime bug that
# corrupts silently (the 2026-07 use-after-free shipped green through
# every functional test). ASan/UBSan turns that whole class into a
# deterministic red test, with no GPU required.
section "sanitizer pass (ASan/UBSan, host-only tests)"
if [ "$QUICK" = "1" ]; then
    printf '  \033[33mSKIP\033[0m  sanitizers (--quick)\n'
else
    SBUILD="${MODELCTL_CI_SAN_BUILD_DIR:-/tmp/ci-build-san}"
    # -fno-sanitize=function: upstream ggml's type-traits table stores
    # dequantizers cast to a generic signature; every indirect call
    # through it trips clang's function-type check. That UB is upstream's
    # to fix -- the checks we are here for (lifetimes, bounds, overflow)
    # all stay on.
    SANFLAGS="-fsanitize=address,undefined -fno-sanitize=function -fno-sanitize-recover=all"
    # Pick a toolchain whose sanitizer runtime actually links: gcc needs
    # the libasan/libubsan RPMs, clang carries compiler-rt statically.
    SAN_CC=""; SAN_CXX=""
    for pair in "gcc:g++" "clang:clang++"; do
        c="${pair%%:*}"; cxx="${pair##*:}"
        if command -v "$c" >/dev/null && command -v "$cxx" >/dev/null \
           && echo 'int main(){return 0;}' | "$c" $SANFLAGS -x c - \
                -o /tmp/ci-santest 2>/dev/null; then
            SAN_CC="$c"; SAN_CXX="$cxx"; break
        fi
    done
    rm -f /tmp/ci-santest
    if [ -z "$SAN_CC" ]; then
        printf '  \033[33mSKIP\033[0m  no toolchain with a working sanitizer runtime (dnf install libasan libubsan, or install clang)\n'
    elif cmake -B "$SBUILD" -S "$REPO/llama.cpp" -DGGML_SYCL=OFF \
            -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_C_COMPILER="$SAN_CC" -DCMAKE_CXX_COMPILER="$SAN_CXX" \
            -DCMAKE_C_FLAGS="$SANFLAGS" -DCMAKE_CXX_FLAGS="$SANFLAGS" \
            > /tmp/ci-cmake-san.log 2>&1 \
       && cmake --build "$SBUILD" --target test-moe-cache test-moe-hybrid \
            -j"$(nproc)" > /tmp/ci-build-san.log 2>&1; then
        if "$SBUILD/bin/test-moe-cache" > /dev/null 2>&1; then
            pass "test-moe-cache under ASan/UBSan"
        else fail "test-moe-cache under ASan/UBSan"; fi
        if "$SBUILD/bin/test-moe-hybrid" > /dev/null 2>&1; then
            pass "test-moe-hybrid under ASan/UBSan"
        else fail "test-moe-hybrid under ASan/UBSan"; fi
    else
        fail "sanitizer build -- see /tmp/ci-build-san.log"
    fi
fi


# --- 6. layering: leaf modules must not import the CLI hub ------------
# modelctl.py is a 4.4k-line CLI hub. When low-level modules import it
# back, CLI process semantics (sys.exit, print, argparse) leak into
# long-lived callers -- that inversion is what let a missing profile
# kill a web job-lane worker thread. These modules stay leaves so the
# class cannot come back.
section "layering (leaf modules do not import modelctl)"
LEAF_VIOLATIONS=$(cd "$REPO/modelctl" && "$VENV" -c "
import ast, pathlib, sys
LEAVES = ['modelctl_paths.py', 'modelctl_fsutil.py', 'modelctl_errors.py',
          'modelctl_profiles.py', 'modelctl_vram.py', 'modelctl_tiers.py',
          'modelctl_capabilities.py', 'modelctl_hardware.py',
          'modelctl_storage.py', 'modelctl_backends.py']
bad = []
for name in LEAVES:
    path = pathlib.Path(name)
    if not path.exists():
        continue
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        # Module level only: a lazy import inside a function is the
        # documented escape hatch for genuinely optional callbacks.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            col = getattr(node, 'col_offset', 0)
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''])
            if col == 0 and any(n == 'modelctl' for n in names):
                bad.append(f'{name}:{node.lineno}')
print(' '.join(bad))
")
if [ -z "$LEAF_VIOLATIONS" ]; then
    pass "no leaf module imports modelctl at module level"
else
    fail "leaf modules import the CLI hub: $LEAF_VIOLATIONS"
fi

section "result"
if [ "$FAILURES" -eq 0 ]; then
    printf '  \033[32mall checks passed\033[0m\n\n'
    exit 0
fi
printf '  \033[31m%d check(s) failed\033[0m\n\n' "$FAILURES"
exit 1
