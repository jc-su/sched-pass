#!/usr/bin/env bash
# run_all.sh -- the whole validation suite in one command (the CI gate).
#
#   ./test/run_all.sh                  # build plugin + run everything
#   SKIP_FLASHINFER=1 ./test/run_all.sh   # fixture-only (no FlashInfer JIT)
#
# Env (defaults match the Blackwell node):
#   LLVM_DIR   cmake dir of the LLVM to build against
#   CLANG      CUDA-capable clang++            CUDA  toolkit root
#   ARCH       --cuda-gpu-arch                 PY    python with torch
set -u
LLVM_DIR=${LLVM_DIR:-/usr/lib/llvm-22/lib/cmake/llvm}
CLANG=${CLANG:-clang++-22}
CUDA=${CUDA:-/usr/local/cuda-12.9}
ARCH=${ARCH:-sm_120a}
PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PL=$ROOT/build/libSchedPass.so
# opt MUST be the SAME LLVM the plugin is built against (a foreign opt on PATH
# fails the plugin ABI -> loads nothing -> the manifest dump is empty). Derive
# it from LLVM_DIR (.../lib/cmake/llvm -> .../bin), not from `command -v opt`.
LLVM_BIN=${LLVM_BIN:-$(cd "$LLVM_DIR/../../.." 2>/dev/null && pwd)/bin}
CC="$CLANG -x cuda --cuda-gpu-arch=$ARCH --cuda-path=$CUDA -O2 -std=c++17 \
    -fpass-plugin=$PL -Wno-unknown-cuda-version -L$CUDA/lib64 -lcudart"
export LD_LIBRARY_PATH=$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SCHED_PLUGIN=$PL

pass=0; fail=0
gate() { # gate <name> <command...>
  local name=$1; shift
  if "$@" >/tmp/sched_gate.log 2>&1; then
    echo "  [PASS] $name"; pass=$((pass+1))
  else
    echo "  [FAIL] $name"; fail=$((fail+1)); tail -5 /tmp/sched_gate.log | sed 's/^/     | /'
  fi
}
expect_all_pass() { # expect_all_pass <name> <command...>
  local name=$1; shift
  if "$@" 2>&1 | tee /tmp/sched_gate.log | grep -qE "== (ALL )?(PASS|ALL EXACT) =="; then
    echo "  [PASS] $name"; pass=$((pass+1))
  else
    echo "  [FAIL] $name"; fail=$((fail+1)); tail -8 /tmp/sched_gate.log | sed 's/^/     | /'
  fi
}

echo "== sched-pass suite ($CLANG, $CUDA, $ARCH) =="

echo "-- build --"
gate "cmake+ninja plugin" bash -c \
  "cmake -S '$ROOT' -B '$ROOT/build' -DLLVM_DIR='$LLVM_DIR' -GNinja && ninja -C '$ROOT/build'"

echo "-- golden IR (CPU-only FileCheck) --"
expect_all_pass "weave IR shapes (remap/timer/gates)" \
  bash "$ROOT/test/ir/run_ir_tests.sh"
expect_all_pass "capability manifest (C++<->Python consistency)" \
  env PLUGIN="$PL" LLVM_BIN="$LLVM_BIN" \
  "$PY" "$ROOT/test/py/test_manifest.py"

echo "-- host-app ABI fixtures --"
gate "compile paged_decode" bash -c "$CC '$ROOT/test/paged_decode.cu' -o '$ROOT/build/pd'"
expect_all_pass "paged_decode (A-H, shed incl.)" "$ROOT/build/pd"
gate "compile WQ ticket" bash -c \
  "SCHED_WORKQUEUE=1 $CC -DSCHED_FIXTURE_WQ=1 '$ROOT/test/paged_decode.cu' -o '$ROOT/build/pdwq'"
expect_all_pass "work-queue ticket claim" "$ROOT/build/pdwq"
gate "compile CLC hetero" bash -c \
  "SCHED_WORKQUEUE=1 SCHED_CLC=1 $CC -DSCHED_FIXTURE_CLC=1 '$ROOT/test/hetero_batch.cu' -o '$ROOT/build/hclc'"
expect_all_pass "CLC try_cancel (sm_100+)" "$ROOT/build/hclc"
gate "compile PDL overlap" bash -c \
  "SCHED_PDL=1 $CC '$ROOT/test/pdl_overlap.cu' -o '$ROOT/build/pdl'"
expect_all_pass "PDL griddepcontrol (sm_90+)" "$ROOT/build/pdl"

echo "-- baked ABI (JIT, SchedArena fixed VA) --"
expect_all_pass "dynamic loop (all levers)" \
  env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
  "$PY" "$ROOT/test/py/test_dynamic_loop.py"
expect_all_pass "shed quality curve (E2 contract)" \
  env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
  "$PY" "$ROOT/test/py/eval_quality.py"
expect_all_pass "baked timer gate (ctrl.flags)" \
  env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
  "$PY" "$ROOT/test/py/test_timer_gate.py"
expect_all_pass "device timer channel (indirect)" \
  env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
  "$PY" "$ROOT/test/py/test_timer_indirect.py"
expect_all_pass "controller (estimator/LPT/hysteresis)" \
  "$PY" "$ROOT/python/sched_controller.py"
expect_all_pass "failure injection (VA miss/oversize/layout/shim)" \
  "$PY" "$ROOT/test/py/test_failsafe.py"
gate "MoE expert-cap hook wireup (real sglang types, CPU)" \
  "$PY" "$ROOT/test/py/test_moe_hook.py"
expect_all_pass "trace loadgen re-craft (shared-prefix reconstruction)" \
  "$PY" "$ROOT/python/sched_trace_loadgen.py" --selftest

if [ -z "${SKIP_FLASHINFER:-}" ]; then
  echo "-- FlashInfer (real serving kernels) --"
  export FLASHINFER_NVCC="$PY $ROOT/python/nvcc_clang_shim.py"
  expect_all_pass "armed pi on FlashInfer decode" "$PY" "$ROOT/test/py/test_flashinfer_arm.py"
  expect_all_pass "kernel-resident device order (SCHED_ORDER_INDIRECT, bit-exact)" \
    env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
    SCHED_WEAVE_ONLY=batch_decode "$PY" "$ROOT/test/py/test_device_order.py"
  expect_all_pass "woven prefill: bit-exact pi + exact per-request fold" \
    env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
    SCHED_WEAVE_ONLY=batch_prefill "$PY" "$ROOT/test/py/test_flashinfer_prefill.py"
  expect_all_pass "cross-process JIT-cache contract" "$PY" "$ROOT/test/py/test_fixed_va.py"
  expect_all_pass "FlashInfer CLC work-queue A/B (grid guard, disarm, SASS)" \
    env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
    "$PY" "$ROOT/test/py/test_flashinfer_clc.py"
  expect_all_pass "serving gate (8-token identity, woven vs stock)" \
    env SCHED_CLANG="$CLANG" SCHED_CUDA_PATH="$CUDA" SCHED_ARCH="$ARCH" \
    SCHED_GATE_GQA=1 "$PY" "$ROOT/test/py/test_serving_gate.py"
  gate "MoE expert-cap in a live GPU FusedMoE forward (no model download)" \
    "$PY" "$ROOT/test/py/test_moe_forward_gpu.py"
fi

echo "== $pass passed, $fail failed =="
exit $fail
