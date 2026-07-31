#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${NTA_BUILD_DIR:-"${root}/build"}
iterations=${NTA_ITERATIONS:-50}
requests=${NTA_REQUESTS:-32}
results=${NTA_RESULTS_DIR:-"${root}/results"}
mkdir -p "${results}"

cmake -S "${root}" -B "${build}" -GNinja \
  -DCMAKE_BUILD_TYPE="${NTA_BUILD_TYPE:-Release}" \
  -DLLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-22/lib/cmake/llvm}" \
  -DNTA_CLANG_CUDA="${NTA_CLANG_CUDA:-/usr/bin/clang++-22}" \
  -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda-12.9}" \
  -DNTA_CUDA_ROOT="${NTA_CUDA_ROOT:-/usr/local/cuda-12.9}" \
  -DNTA_CUDA_ARCH="${NTA_CUDA_ARCH:-sm_120}"
cmake --build "${build}" -j"${NTA_BUILD_JOBS:-2}"
ctest --test-dir "${build}" --output-on-failure

"${build}/nta-kv-bench" \
  --mode=mixed --requests=96 --coalesce=3 --dependencies=4 \
  --tile-bytes=65536 --iterations="${iterations}" \
  --cancel-stride=17 --stale-stride=19 | \
  tee "${results}/dependency-set.log"

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
      --progress-passes=1 | tee -a "${matrix_log}"
  done
done

ptxas=${NTA_PTXAS:-/usr/local/cuda-12.9/bin/ptxas}
"${ptxas}" -v -arch="${NTA_CUDA_ARCH:-sm_120}" -O3 \
  "${build}/kernel/PagedAttention.ptx" \
  -o "${results}/PagedAttention.cubin" \
  2>"${results}/paged-attention-ptxas.log"
"${ptxas}" -v -arch="${NTA_CUDA_ARCH:-sm_120}" -O3 \
  "${build}/kernel/KvAcquire.ptx" \
  -o "${results}/KvAcquire.cubin" \
  2>"${results}/kv-ptxas.log"

if [[ ${NTA_SANITIZE:-0} == 1 ]]; then
  sanitizer=${NTA_COMPUTE_SANITIZER:-/usr/local/cuda-13.0/bin/compute-sanitizer}
  for tool in memcheck racecheck synccheck; do
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-paged-attention" \
      --mode=mixed --copy=tma --requests=3 --min-pages=2 --max-pages=3 \
      --iterations=1 --progress-passes=3 \
      2>&1 | tee "${results}/${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-kv-bench" \
      --mode=mixed --requests=6 --coalesce=2 --dependencies=3 \
      --tile-bytes=8192 --iterations=1 \
      2>&1 | tee "${results}/dependency-set-${tool}.log"
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 \
      "${build}/nta-moe-bench" \
      --mode=mixed --tokens=6 --experts=6 --top-k=2 --hidden=64 \
      --iterations=1 \
      2>&1 | tee "${results}/moe-${tool}.log"
  done
fi
