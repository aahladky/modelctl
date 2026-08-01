#!/usr/bin/env bash
# Live, human-readable tail of the newest Claude Code session for this repo.
# Usage: scripts/agent-tail.sh   (ctrl-c to stop)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
P="$HOME/.claude/projects/$(echo "$ROOT" | tr '/' '-')"
F=$(ls -t "$P"/*.jsonl 2>/dev/null | head -1)
[ -n "${F:-}" ] || { echo "no sessions found under $P"; exit 1; }
echo "== tailing $F"
tail -n 30 -f "$F" | python3 -u -c '
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
    except Exception:
        continue
    c = (e.get("message") or {}).get("content")
    if not isinstance(c, list):
        continue
    for b in c:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text" and (b.get("text") or "").strip():
            print("\n[claude]", b["text"].strip()[:500])
        elif t == "tool_use":
            print("[tool]", b.get("name", "?"),
                  json.dumps(b.get("input", {}))[:200])
        elif t == "tool_result":
            s = b.get("content")
            s = s if isinstance(s, str) else json.dumps(s)
            print("   ->", (s or "")[:200].replace("\n", " "))
'
