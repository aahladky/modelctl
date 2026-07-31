#!/usr/bin/env bash
# Weekly upstream drift report for the llama.cpp runtime fork.
# Purely informational: fetches upstream, reports drift, never touches the
# pinned submodule commit, feature/sycl-moe-expert-cache, or any branch.
set -euo pipefail

TOP=/home/aaron/workspace/moe-serving
RUNTIME="$TOP/llama.cpp"
REPORT_DIR="$TOP/docs/upstream-sync/drift-reports"
DATE="$(date +%Y-%m-%d)"
REPORT="$REPORT_DIR/$DATE.md"

mkdir -p "$REPORT_DIR"

cd "$RUNTIME"
git fetch origin master --quiet

# Read the *pinned* commit from the top-level repo's index, not whatever
# happens to be checked out locally.
# submodule status prints an optional 1-char prefix (+/-/U) directly
# butted up against the 40-char hash with no separator -- always take the
# last 40 chars of the first field so this works regardless of prefix.
PINNED="$(git -C "$TOP" submodule status llama.cpp | awk '{h=$1; print substr(h, length(h)-39)}')"
UPSTREAM_HEAD="$(git rev-parse origin/master)"

if [ "$PINNED" = "$UPSTREAM_HEAD" ]; then
  DRIFT_COUNT=0
else
  DRIFT_COUNT="$(git rev-list --count "$PINNED".."$UPSTREAM_HEAD" 2>/dev/null || echo "unknown (pinned commit not an ancestor of origin/master? check manually)")"
fi

WATCH_PATHS=(
  "ggml/src/ggml-sycl/"
  "ggml/src/ggml-backend.cpp"
  "tools/server/"
  "common/"
  "src/llama-model"
  "src/llama-context"
  "tests/"
)

COMMITS="$(git log --oneline --no-merges "$PINNED".."$UPSTREAM_HEAD" -- "${WATCH_PATHS[@]}" 2>/dev/null || true)"

# Crude keyword heuristic for "needs immediate review" -- this is a signal
# for a human to look closer, not a judgment call in itself.
FLAGGED="$(echo "$COMMITS" | grep -iE \
  'fix|cve|security|vulnerab|sched|correctness|crash|overflow|regression|mtp|nextn|moe|expert|placement|deadlock|race|use-after-free|uaf| leak' \
  || true)"

{
  echo "# Upstream drift report -- $DATE"
  echo
  echo "Pinned runtime commit: \`$PINNED\`"
  echo "Upstream (\`origin/master\`) head: \`$UPSTREAM_HEAD\`"
  echo "Drift: **$DRIFT_COUNT commits behind**"
  echo
  echo "This report is informational only. It does not advance the pinned"
  echo "submodule commit, merge any branch, or touch"
  echo "\`feature/sycl-moe-expert-cache\`. Promotion still requires the full"
  echo "build/correctness/performance/integration gate (Phase 0, Tasks"
  echo "0.5-0.9)."
  echo
  echo "## Commits touching watched paths since the pinned commit"
  echo
  if [ -n "$COMMITS" ]; then
    echo '```'
    echo "$COMMITS"
    echo '```'
  else
    echo "None."
  fi
  echo
  echo "## Flagged for immediate review (keyword heuristic, not a judgment call)"
  echo
  if [ -n "$FLAGGED" ]; then
    echo "The following commits matched a keyword suggesting a SYCL"
    echo "correctness fix, security fix, scheduler change, model-format"
    echo "change, or MTP/MoE placement fix -- read these before waiting for"
    echo "the next scheduled integration cycle:"
    echo
    echo '```'
    echo "$FLAGGED"
    echo '```'
  else
    echo "None matched. Still read the full commit list above if drift is"
    echo "large -- this heuristic is a filter, not a substitute for review."
  fi
} > "$REPORT"

cd "$TOP"
git add "$REPORT"
if ! git diff --cached --quiet; then
  git commit -q -m "Upstream drift report $DATE ($DRIFT_COUNT commits behind)"
  git push -q origin master
  echo "Committed and pushed $REPORT"
else
  echo "No changes to commit (report identical to last run?)"
fi

echo "Drift: $DRIFT_COUNT commits behind. Flagged commits: $(echo "$FLAGGED" | grep -c . || true)"
