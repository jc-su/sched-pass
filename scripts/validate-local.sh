#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${NTA_BUILD_DIR:-"${root}/build"}
iterations=${NTA_ITERATIONS:-50}
requests=${NTA_REQUESTS:-32}
results=${NTA_RESULTS_DIR:-"${TMPDIR:-/tmp}/nta-local-results"}
mkdir -p "${results}"

python_bin=${NTA_PYTHON:-python3}
cuda_request=${NTA_CUDA_ROOT:-${CUDAToolkit_ROOT:-}}
if [[ -n ${cuda_request} ]]; then
  cuda_root=$("${python_bin}" "${root}/tools/jit/cuda_toolkit.py" \
    --cuda-path "${cuda_request}" --print-root)
else
  cuda_root=$("${python_bin}" "${root}/tools/jit/cuda_toolkit.py" --print-root)
fi
ptxas=${NTA_PTXAS:-${cuda_root}/bin/ptxas}

cmake -S "${root}" -B "${build}" -GNinja \
  -DCMAKE_BUILD_TYPE="${NTA_BUILD_TYPE:-Release}" \
  -DLLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-22/lib/cmake/llvm}" \
  -DNTA_CLANG_CUDA="${NTA_CLANG_CUDA:-/usr/bin/clang++-22}" \
  -DCUDAToolkit_ROOT="${cuda_root}" \
  -DNTA_CUDA_ROOT="${cuda_root}" \
  -DNTA_CUDA_ARCH="${NTA_CUDA_ARCH:-sm_120}"
cmake --build "${build}" -j"${NTA_BUILD_JOBS:-2}"
ctest --test-dir "${build}" --output-on-failure

"${build}/nta-kv-bench" \
  --mode=mixed --requests=96 --coalesce=3 --dependencies=4 \
  --tile-bytes=65536 --iterations="${iterations}" \
  --cancel-stride=17 --stale-stride=19 | \
  tee "${results}/dependency-set.log"

"${build}/nta-kv-bench" \
  --mode=host-direct --requests=96 --tile-bytes=65536 \
  --iterations="${iterations}" --external-registration=1 | \
  tee "${results}/registered-host-direct.log"

"${build}/nta-kv-bench" \
  --mode=host-staged --requests=96 --tile-bytes=65536 \
  --iterations="${iterations}" --external-registration=1 \
  --external-offset=1 | tee "${results}/registered-host-staged.log"

"${build}/nta-moe-bench" \
  --mode=mixed --tokens=64 --experts=16 --top-k=2 --hidden=128 \
  --iterations="${iterations}" | tee "${results}/moe.log"

matrix_log="${results}/attention-matrix.log"
: >"${matrix_log}"
for copy in global tma; do
  for mode in resident host-direct host-staged mixed; do
    "${build}/nta-paged-attention" \
      --mode="${mode}" \
      --copy="${copy}" \
      --requests="${requests}" \
      --min-pages=4 \
      --max-pages=16 \
      --iterations="${iterations}" \
      --progress-rounds=1 | tee -a "${matrix_log}"
  done
done

"${build}/nta-paged-attention" \
  --mode=host-staged \
  --requests=16 \
  --min-pages=8 \
  --max-pages=16 \
  --sparse-top-k=2 \
  --iterations="${iterations}" \
  --progress-rounds=1 \
  --request-credit-pages=2 | tee "${results}/sparse-late-bound.log"

"${build}/nta-paged-attention" \
  --mode=host-staged \
  --requests=16 \
  --min-pages=8 \
  --max-pages=16 \
  --sparse-top-k=2 \
  --sparse-policy=overfetch \
  --iterations="${iterations}" \
  --progress-rounds=1 \
  --request-credit-pages=2 | tee "${results}/sparse-overfetch.log"

"${ptxas}" -v -arch="${NTA_CUDA_ARCH:-sm_120}" -O3 \
  "${build}/kernel/PagedAttention.ptx" \
  -o "${results}/PagedAttention.cubin" \
  2>"${results}/paged-attention-ptxas.log"
"${ptxas}" -v -arch="${NTA_CUDA_ARCH:-sm_120}" -O3 \
  "${build}/kernel/KvAcquire.ptx" \
  -o "${results}/KvAcquire.cubin" \
  2>"${results}/kv-ptxas.log"

if [[ ${NTA_SANITIZE:-0} == 1 ]]; then
  sanitizer=${NTA_COMPUTE_SANITIZER:-${cuda_root}/bin/compute-sanitizer}
  have_flashinfer=0
  if "${python_bin}" -c 'import flashinfer' >/dev/null 2>&1; then
    have_flashinfer=1
  fi
  for tool in memcheck racecheck synccheck; do
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-paged-attention" \
      --mode=mixed --copy=tma --requests=3 --min-pages=2 --max-pages=3 \
      --iterations=1 --progress-rounds=3 \
      2>&1 | tee "${results}/${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-paged-attention" \
      --mode=host-staged --requests=3 --min-pages=4 --max-pages=6 \
      --sparse-top-k=2 --iterations=1 --progress-rounds=1 \
      --request-credit-pages=2 \
      2>&1 | tee "${results}/sparse-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-paged-attention" \
      --mode=host-staged --requests=3 --min-pages=4 --max-pages=6 \
      --sparse-top-k=2 --sparse-policy=overfetch --iterations=1 \
      --progress-rounds=1 --request-credit-pages=2 \
      2>&1 | tee "${results}/sparse-overfetch-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-kv-bench" \
      --mode=mixed --requests=6 --coalesce=2 --dependencies=3 \
      --tile-bytes=8192 --iterations=1 \
      2>&1 | tee "${results}/dependency-set-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-kv-bench" \
      --mode=host-staged --requests=6 --tile-bytes=8192 --iterations=1 \
      --external-registration=1 --external-offset=1 \
      2>&1 | tee "${results}/registered-host-staged-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-moe-bench" \
      --mode=mixed --tokens=6 --experts=6 --top-k=2 --hidden=64 \
      --iterations=1 \
      2>&1 | tee "${results}/moe-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-cta-nvme-try-issue-test" \
      2>&1 | tee "${results}/cta-nvme-${tool}.log"
    if [[ ${have_flashinfer} == 1 ]]; then
      "${root}/tools/jit/activate.py" \
        --build-dir "${build}" \
        --cache-root "${build}/flashinfer-jit-cache" \
        --clang "${NTA_CLANG_CUDA:-/usr/bin/clang++-22}" \
        --cuda-path "${cuda_root}" \
        --flashinfer-hook -- \
        "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
        "${python_bin}" "${root}/tests/flashinfer/hooked_decode.py" --sanitizer \
        2>&1 | tee "${results}/flashinfer-${tool}.log"
    fi
  done
fi
