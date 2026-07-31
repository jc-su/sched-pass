#!/usr/bin/env bash
# start_server_detached.sh -- launch the woven SGLang server detached from
# the calling session (survives SSH drops / Claude session restarts).
#
#   MODEL=JackFram/llama-160m ./scripts/start_server_detached.sh [args...]
#
# Env: LOG (/tmp/sglang_server.log), PORT (30070). Truncates LOG first so
# readiness watchers can never match a stale line. Prints the detached PID
# and exits immediately; poll ${PORT}/health for readiness.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=${LOG:-/tmp/sglang_server.log}
PORT=${PORT:-30070}
: > "$LOG"
setsid nohup bash "$ROOT/scripts/serve_sglang_armed.sh" \
  --port "$PORT" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "launch FAILED; log tail:" >&2
  tail -5 "$LOG" >&2
  exit 1
fi
echo "detached: pid=$PID log=$LOG port=$PORT"
