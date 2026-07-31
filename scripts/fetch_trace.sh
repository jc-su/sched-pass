#!/usr/bin/env bash
# fetch_trace.sh -- pull the real Qwen-Bailian usage traces WITHOUT git-lfs.
# The .jsonl files in the repo are 133-byte LFS pointers; GitHub serves the real
# content from the media CDN (media.githubusercontent.com/media/...), which is not
# API-rate-limited. Used by sched_trace_loadgen.py for live trace replay.
#
#   ./scripts/fetch_trace.sh                 # all four scenarios -> data/traces/
#   ./scripts/fetch_trace.sh traceA          # just the To-C chat trace
#   BYTES=2000000 ./scripts/fetch_trace.sh traceA   # first ~2MB only (a sample)
set -u
REPO=alibaba-edu/qwen-bailian-usagetraces-anon
BRANCH=main
BASE="https://media.githubusercontent.com/media/$REPO/$BRANCH"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/data/traces"
mkdir -p "$OUT"

declare -A FILES=(
  [traceA]=qwen_traceA_blksz_16.jsonl      # To-C   chat
  [traceB]=qwen_traceB_blksz_16.jsonl      # To-B   API automation
  [thinking]=qwen_thinking_blksz_16.jsonl  # reasoning (long output)
  [coder]=qwen_coder_blksz_16.jsonl        # code generation
)

want=("$@"); [ ${#want[@]} -eq 0 ] && want=(traceA traceB thinking coder)
for k in "${want[@]}"; do
  f=${FILES[$k]:-}
  [ -z "$f" ] && { echo "unknown scenario: $k (traceA|traceB|thinking|coder)"; continue; }
  if [ -n "${BYTES:-}" ]; then
    dst="$OUT/${f%.jsonl}.sample.jsonl"   # committed fixture name (see .gitignore)
    echo "fetching $k sample (~${BYTES}B) -> $dst"
    curl -fsSL -r "0-${BYTES}" "$BASE/$f" -o "$dst" \
      && sed -i '$ { /}$/!d }' "$dst"     # drop a truncated final line
  else
    dst="$OUT/$f"                         # full pull (gitignored)
    echo "fetching $k (full) -> $dst"
    curl -fsSL "$BASE/$f" -o "$dst"
  fi
  [ -f "$dst" ] && echo "  $(wc -l <"$dst") records, $(wc -c <"$dst") bytes"
done
