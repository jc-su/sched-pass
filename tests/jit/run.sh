#!/usr/bin/env bash
set -euo pipefail

shim=$1
plugin=$2
clang=$3
cuda_path=$4
project_root=$5
architecture=$6
llvm_dis=$7
output_dir=$8

mkdir -p "${output_dir}"
NTA_CLANG="${clang}" \
NTA_PLUGIN="${plugin}" \
NTA_CUDA_PATH="${cuda_path}" \
NTA_PROJECT_ROOT="${project_root}" \
"${shim}" \
  -O3 -std=c++20 --use_fast_math --generate-line-info \
  --cuda-device-only -emit-llvm -c \
  -arch "${architecture}" \
  "${project_root}/tests/jit/ForeignKernel.cu" \
  -o "${output_dir}/ForeignKernel.bc"

"${llvm_dis}" "${output_dir}/ForeignKernel.bc" \
  -o "${output_dir}/ForeignKernel.ll"
rg -q 'define.*ptx_kernel.*@nta_foreign_kernel' \
  "${output_dir}/ForeignKernel.ll"
rg -q 'call i1 @nta_acquire_set_slow' "${output_dir}/ForeignKernel.ll"
rg -q '!nta.acquire' "${output_dir}/ForeignKernel.ll"
rg -q '"no-nans-fp-math"="true"' "${output_dir}/ForeignKernel.ll"
rg -q 'call i1 @nta_acquire_set_slow.*!dbg' \
  "${output_dir}/ForeignKernel.ll"
if rg -q '__nta_(bind_request|acquire_set_marker|defer_marker)' \
  "${output_dir}/ForeignKernel.ll"; then
  echo "JIT-compiled foreign kernel still contains an NTA marker" >&2
  exit 1
fi

echo "NTA clang JIT test passed"
