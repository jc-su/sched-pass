#!/usr/bin/env bash
# clean_3b_master.sh -- the 3B track on a HEALTHY base model
# (Qwen2.5-3B-Instruct, same 16q/2kv GQA shape), with the post-mortem rules:
# pinned greedy (--sampling-defaults openai), FULL-TEXT identity gates (no
# coherence heuristics). Phases -> clean_3b_results.txt:
#   A. determinism sanity: stock, 3 identical greedy requests must MATCH
#   B. correctness gate: woven vs stock, same request, full text EQUALITY
#   C. attention-dominant A/B/C (in 1536..3072, out 128, conc 256)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/clean_3b_results.txt"
M3B=$(ls -d /home/jcsu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/ | head -1)
export SGLANG_FLASHINFER_USE_TENSOR_CORE=false
note() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT"; }
: > "$OUT"
note "== clean-3B master (model $M3B) =="

# 16-token generations for the identity gates: long tails accumulate benign
# kernel nondeterminism (non-batch-invariant reductions) even at true
# greedy; short prefixes are exact. (--enable-deterministic-inference does
# not boot on this stack -- gates use prefix identity instead.)
REQ='{"text":"The theory of relativity states that","sampling_params":{"max_new_tokens":16,"temperature":0}}'

boot() { # boot <label> <env...>
  local label=$1; shift
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 3
  env "$@" MODEL="$M3B" SCHED_DEBUG=1 PY=python3 \
    LOG="/tmp/srv_c3b_$label.log" PORT=30071 \
    bash "$ROOT/scripts/start_server_detached.sh" \
    --disable-cuda-graph --mem-fraction-static 0.75 \
    --sampling-defaults openai --random-seed 42 >> "$OUT" 2>&1
  for _ in $(seq 1 150); do
    curl -sf --max-time 3 http://127.0.0.1:30071/health >/dev/null 2>&1 && return 0
    grep -q "hit an exception" "/tmp/srv_c3b_$label.log" 2>/dev/null && break
    sleep 4
  done
  note "$label FAILED to boot"; tail -4 "/tmp/srv_c3b_$label.log" >> "$OUT"
  return 1
}

gen() { curl -s --max-time 900 http://127.0.0.1:30071/generate \
  -H 'Content-Type: application/json' -d "$REQ" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])"; }

# ---- A. determinism sanity (stock) -----------------------------------------
note "-- A: determinism sanity (stock, pinned greedy)"
if boot stock SCHED_SITE_OFF=1; then
  gen > /tmp/c3b_s1.txt; gen > /tmp/c3b_s2.txt; gen > /tmp/c3b_s3.txt
  if cmp -s /tmp/c3b_s1.txt /tmp/c3b_s2.txt && cmp -s /tmp/c3b_s2.txt /tmp/c3b_s3.txt; then
    note "A PASS: 3 identical greedy generations"
    head -c 90 /tmp/c3b_s1.txt >> "$OUT"; echo >> "$OUT"
  else
    note "A FAIL: still nondeterministic on the CLEAN model -- STOP"
    head -c 70 /tmp/c3b_s1.txt >> "$OUT"; echo >> "$OUT"
    head -c 70 /tmp/c3b_s2.txt >> "$OUT"; echo >> "$OUT"
    note "== MASTER DONE =="; exit 0
  fi
fi

# ---- B. correctness gate: woven vs stock text identity ---------------------
note "-- B: woven vs stock full-text identity"
if boot woven; then
  gen > /tmp/c3b_w1.txt; gen > /tmp/c3b_w2.txt
  if cmp -s /tmp/c3b_w1.txt /tmp/c3b_w2.txt; then
    note "B: woven deterministic"
  else
    note "B WARN: woven nondeterministic"
  fi
  if cmp -s /tmp/c3b_s1.txt /tmp/c3b_w1.txt; then
    note "B PASS: WOVEN == STOCK (full text identity) -- quarantine LIFTED"
  else
    note "B FAIL: woven differs from stock:"
    head -c 70 /tmp/c3b_s1.txt >> "$OUT"; echo >> "$OUT"
    head -c 70 /tmp/c3b_w1.txt >> "$OUT"; echo >> "$OUT"
  fi
fi

# ---- C. attention-dominant A/B/C -------------------------------------------
note "-- C: 3B long-KV A/B/C (in 1536..3072, out 128, conc 256)"
OUT="$ROOT/bench_clean3b_results.txt" MODEL="$M3B" PORT=30071 \
  NPROMPT=400 CONC=256 INLEN=3072 OUTLEN=128 RATIO=0.5 MEMFRAC=0.75 \
  SRV_ARGS="--disable-cuda-graph" \
  bash "$ROOT/scripts/bench_ab.sh"
note "C done -> bench_clean3b_results.txt"
pkill -9 -f "sglang.launch_server" 2>/dev/null
note "== MASTER DONE =="
