#!/usr/bin/env bash
# run_trace_ab.sh -- live trace-driven A/B against the woven SGLang server, using
# the real Qwen-Bailian trace re-crafted by sched_trace_loadgen.py. This is the
# turnkey "does the whole system work end to end under a real workload" run.
#
# Two regimes it can exercise:
#   ATTENTION levers (ORDER / shed / split-skew) -- runs against ANY cached model
#     (no MoE needed). This is runnable NOW.
#   MoE cap (SCHED_MOE_CAP=1)                     -- needs a MoE model (none cached;
#     set MODEL=Qwen/Qwen1.5-MoE-A2.7B or similar; multi-GB download on first use).
#
# Usage:
#   ./scripts/run_trace_ab.sh                     # woven vs stock, cached dense model
#   MODEL=Qwen/Qwen3-8B ./scripts/run_trace_ab.sh
#   MOE=1 MODEL=Qwen/Qwen1.5-MoE-A2.7B ./scripts/run_trace_ab.sh   # + expert cap
#
# It boots ONE server per arm (woven, then stock), replays the trace, prints the
# latency table. Boots are done in the background + health-polled (this session's
# servers die under setsid-nohup; harness-backgrounded is the reliable path).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-30000}
TRACE=${TRACE:-$ROOT/data/traces/qwen_traceA_blksz_16.sample.jsonl}
LIMIT=${LIMIT:-200}
SPEED=${SPEED:-0}                 # 0 = as-fast-as-possible (throughput A/B)
PY=${PY:-python3}
BASE="http://127.0.0.1:$PORT"

if [ ! -f "$TRACE" ]; then
  echo "trace $TRACE missing -- fetching a sample"; BYTES=350000 bash "$ROOT/scripts/fetch_trace.sh" traceA
fi

boot() { # boot <arm: woven|stock>
  local arm=$1
  echo ">>> booting $arm server ($MODEL) on :$PORT"
  if [ "$arm" = woven ]; then
    [ "${MOE:-0}" = 1 ] && export SCHED_MOE_CAP=1
    MODEL="$MODEL" bash "$ROOT/scripts/serve_sglang_armed.sh" --port "$PORT" &
  else
    "$PY" -m sglang.launch_server --model-path "$MODEL" --port "$PORT" \
      --attention-backend flashinfer &
  fi
  echo $!
}

wait_health() { # wait_health <pid>
  for _ in $(seq 1 120); do
    curl -fsS "$BASE/health" >/dev/null 2>&1 && { echo "  healthy"; return 0; }
    kill -0 "$1" 2>/dev/null || { echo "  server died during boot"; return 1; }
    sleep 2
  done
  echo "  health timeout"; return 1
}

run_arm() { # run_arm <arm>
  local arm=$1 pid
  pid=$(boot "$arm" | tail -1)
  if wait_health "$pid"; then
    "$PY" "$ROOT/python/sched_trace_loadgen.py" --base-url "$BASE" \
      --trace "$TRACE" --limit "$LIMIT" --speed "$SPEED" --tag "$arm"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
}

echo "== trace A/B: $MODEL, $LIMIT reqs from $(basename "$TRACE"), speed=$SPEED, MoE=${MOE:-0} =="
run_arm woven
run_arm stock
echo "== done -- compare the two 'replay summary' tables above (woven vs stock) =="
