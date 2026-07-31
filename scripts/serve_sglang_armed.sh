#!/usr/bin/env bash
# serve_sglang_armed.sh -- launch a real SGLang server with the woven,
# ARMED FlashInfer decode path.
#
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct ./scripts/serve_sglang_armed.sh [args...]
#
# What this wires (each piece individually validated by test/run_all.sh):
#   * FLASHINFER_NVCC -> nvcc_clang_shim.py: FlashInfer's JIT compiles with
#     clang + -fpass-plugin=libSchedPass.so (weaves pi/timer/policy/shed).
#   * PYTHONPATH + SCHED_SGLANG=1 -> sitecustomize.py registers the plugin in
#     the spawned scheduler process; its first batch creates the SchedPlane
#     (fixed-VA SchedArena) and arms the process env, so the JIT bakes stable
#     addresses and the va-keyed cache is reused across restarts.
#   * decode attention = flashinfer (the woven path); prefill = triton by
#     default (confines the woven surface). SCHED_WEAVE_PREFILL=1 swaps prefill
#     to flashinfer AND weaves it -- live-validated bit-exact (24/24 tokens) vs
#     stock on a real server, 411-token multi-tile prefill (ROADMAP 2026-07-09).
#
# Control at runtime:
#   SCHED_SGLANG_ENFORCE=1   apply LPT pi + polite hints (default: observe only)
#   SCHED_MAX_TASKS=4096     arena capacity (decode tiles per step ceiling)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:?set MODEL (e.g. Qwen/Qwen2.5-0.5B-Instruct)}
PY=${PY:-python3}

export SCHED_PLUGIN=${SCHED_PLUGIN:-$ROOT/build/libSchedPass.so}
# SCHED_SITE_OFF=1 keeps sitecustomize (the plugin bootstrap) OUT of the
# compiler processes -- importing torch inside the shim deadlocked the JIT's
# `nvcc --version` probe and would tax every compile.
export FLASHINFER_NVCC="env SCHED_SITE_OFF=1 $PY $ROOT/python/nvcc_clang_shim.py"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SCHED_SGLANG=1
export PYTHONUNBUFFERED=1
export SCHED_MAX_TASKS=${SCHED_MAX_TASKS:-4096}
# MINIMAL WEAVE SURFACE for serving: compile in only the levers the control
# plane uses (pi + timer). shed carries a known open codegen fault on
# multi-lever JIT softmax kernels (host-fixture-quarantined) and the plugin
# never sets tau; the policy/prefetch lever is unused unless sigma hints are
# configured. Unset these to re-enable for experiments (cache keys carry the
# lever mask, so variants never collide).
export SCHED_NO_SHED=${SCHED_NO_SHED:-1}
if [ -z "${SCHED_KV_BYTES_PER_TOKEN:-}" ]; then
  export SCHED_NO_POLICY=${SCHED_NO_POLICY:-1}
fi
# Weave the ATTENTION modules; everything else (norm, rope, sampling) compiles
# UNWOVEN via SCHED_REAL_NVCC. Weave-only confines the woven surface AND keeps
# rope/norm CTAs from writing the decode control tables (timer pollution).
#
#   decode  -- always woven (the proven, shipping path).
#   prefill -- OPT-IN via SCHED_WEAVE_PREFILL=1. The prefill "clang wall" was
#              closed 2026-07-09: it was NOT a codegen gap but FlashInfer gating
#              its cp.async/ldmatrix fast path behind __CUDACC_VER_MAJOR__ (an
#              nvcc-only macro); the shim now defines it, and prefill is
#              bit-exact under pi with exact per-request folds
#              (test/py/test_flashinfer_prefill.py). Gated OFF by default until
#              validated in the live serving loop; decode-only can never be
#              destabilized by enabling it.
# UNWOVEN prefill (default) routes to SCHED_REAL_NVCC; that nvcc needs
# scripts/fix_cuda_glibc241.sh applied ONCE (CUDA 12.9 + glibc 2.41 sinpi/cospi
# cudafe clash) or long-input serving fails to BOOT even stock. WOVEN prefill
# builds via clang (shim) instead, bypassing that fallback for the module.
# export so the spawned SCHEDULER process (where the plugin runs) sees it --
# the plugin's _attn_mode() gates prefill weaving on it.
export SCHED_WEAVE_PREFILL="${SCHED_WEAVE_PREFILL:-0}"
if [ "$SCHED_WEAVE_PREFILL" = "1" ]; then
  export SCHED_WEAVE_ONLY=${SCHED_WEAVE_ONLY:-batch_decode,batch_prefill}
else
  export SCHED_WEAVE_ONLY=${SCHED_WEAVE_ONLY:-batch_decode}
fi
export SCHED_REAL_NVCC=${SCHED_REAL_NVCC:-/usr/local/cuda-12.9/bin/nvcc}

# SGLang's OWN jit kernels (sgl_kernel_jit via tvm-ffi) find nvcc through
# CUDA_HOME -- point them at a shim-backed fake toolkit so they compile with
# clang too (tvm-ffi uses $CUDA_HOME/bin/nvcc for -c compiles only; links
# with plain c++, so no link-mode handling is needed).
# Route SGLang's tvm-ffi JIT to our nvcc wrapper via PATH, NEVER CUDA_HOME.
# Root-caused 2026-07-07 (see ROADMAP.md "3B corruption saga"): repointing CUDA_HOME corrupts
# Triton's runtime-compiled GQA prefill kernels (wrong-topic generations)
# EVEN when the fake toolkit is a complete symlink overlay of the real one;
# a skeletal fake additionally poisoned ~/.triton across boots. tvm-ffi's
# nvcc discovery falls back to `which nvcc` (extension.py), so a PATH-front
# wrapper reaches it while Triton keeps its own untouched toolchain
# discovery (pure-upstream behavior, verified deterministic-and-correct).
FAKECUDA="$ROOT/build/fakecuda"
CUDA_REAL=${SCHED_CUDA_PATH:-/usr/local/cuda-12.9}
rm -rf "$FAKECUDA"
mkdir -p "$FAKECUDA/bin"
for x in "$CUDA_REAL"/bin/*; do
  b=$(basename "$x")
  [ "$b" = nvcc ] && continue
  ln -sfn "$x" "$FAKECUDA/bin/$b"
done
ln -sfn "$CUDA_REAL/include" "$FAKECUDA/include"
ln -sfn "$CUDA_REAL/lib64" "$FAKECUDA/lib64"
ln -sfn "$CUDA_REAL/targets" "$FAKECUDA/targets"
ln -sfn "$CUDA_REAL/nvvm" "$FAKECUDA/nvvm"
cat > "$FAKECUDA/bin/nvcc" <<EOF
#!/usr/bin/env bash
export SCHED_SITE_OFF=1  # keep the plugin bootstrap out of compiler procs
exec $PY $ROOT/python/nvcc_clang_shim.py "\$@"
EOF
chmod +x "$FAKECUDA/bin/nvcc"
unset CUDA_HOME CUDA_PATH 2>/dev/null || true
export PATH="$FAKECUDA/bin:$PATH"

# Empty SCHED_PLUGIN is a legitimate bisect/debug mode: the shim compiles
# clang-unwoven, the plugin plane never arms compiles.
[ -z "$SCHED_PLUGIN" ] || [ -f "$SCHED_PLUGIN" ] || { echo "plugin missing: $SCHED_PLUGIN (run test/run_all.sh)"; exit 1; }

# Prefill backend: TRITON by default (confines the woven surface to decode --
# FlashInfer's prefill kernel is not even compiled). When weaving prefill it
# MUST be FLASHINFER: that is the kernel the pass weaves and the plugin controls
# (Triton prefill is off the woven path, so SCHED_WEAVE_PREFILL=1 alone would be
# a silent no-op without this swap).
PREFILL_BACKEND=triton
if [ "${SCHED_WEAVE_PREFILL:-0}" = "1" ]; then
  PREFILL_BACKEND=flashinfer
  echo "[serve] SCHED_WEAVE_PREFILL=1 -> prefill backend = flashinfer (woven)"
fi

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --attention-backend flashinfer \
  --prefill-attention-backend "$PREFILL_BACKEND" \
  --decode-attention-backend flashinfer \
  "$@"
