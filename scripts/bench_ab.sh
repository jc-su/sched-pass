#!/usr/bin/env bash
# bench_ab.sh -- the B3 serving A/B/C, self-contained and drop-proof:
# stock (plugin off) vs woven observe-only vs woven ENFORCE (pi live),
# one bench_serving round each, results appended to $OUT as they land.
# Run DETACHED (setsid nohup) on a flaky host; poll $OUT.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/bench_ab_results.txt}
PORT=${PORT:-30071}
MODEL=${MODEL:-JackFram/llama-160m}
NPROMPT=${NPROMPT:-300}
CONC=${CONC:-64}
INLEN=${INLEN:-128}
OUTLEN=${OUTLEN:-128}
RATIO=${RATIO:-0.5}
SRV_ARGS=${SRV_ARGS:---disable-cuda-graph}
MEMFRAC=${MEMFRAC:-0.5}

note() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT"; }

bench() { # bench <label>
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 \
    --port "$PORT" --dataset-name random --num-prompts "$NPROMPT" \
    --random-input-len "$INLEN" --random-output-len "$OUTLEN" \
    --random-range-ratio "$RATIO" --max-concurrency "$CONC" 2>&1 |
    grep -E "Request throughput|Output token throughput|Median TTFT|Median TPOT|P99 TPOT|Median ITL" |
    sed "s/^/[$1] /" >> "$OUT"
}

wait_health() {
  for _ in $(seq 1 120); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 4
  done
  return 1
}

run_config() { # run_config <label> <extra-env...>
  local label=$1; shift
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 3
  note "starting $label"
  env "$@" MODEL="$MODEL" PY=python3 LOG="/tmp/srv_$label.log" PORT="$PORT" \
    bash "$ROOT/scripts/start_server_detached.sh" \
    $SRV_ARGS --mem-fraction-static "$MEMFRAC" >> "$OUT" 2>&1
  if wait_health; then
    note "$label healthy; benching"
    bench "$label"
  else
    note "$label FAILED to become healthy; log tail:"
    tail -5 "/tmp/srv_$label.log" >> "$OUT" 2>/dev/null
  fi
}

: > "$OUT"
note "== B3 serving A/B/C (model=$MODEL, n=$NPROMPT, conc=$CONC) =="
run_config stock SCHED_SITE_OFF=1
run_config observe SCHED_DEBUG=1
run_config enforce SCHED_DEBUG=1 SCHED_SGLANG_ENFORCE=1
pkill -9 -f "sglang.launch_server" 2>/dev/null
note "== DONE =="
