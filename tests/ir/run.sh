#!/usr/bin/env bash
set -euo pipefail

plugin=$1
opt=$2
output_dir=$3
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${output_dir}"

"${opt}" \
  -load-pass-plugin="${plugin}" \
  -passes=nta-acquire \
  -S "${source_dir}/batched.ll" \
  -o "${output_dir}/batched.lowered.ll"

rg -q 'call i1 @nta_request_live' "${output_dir}/batched.lowered.ll"
rg -q 'br i1 %nta.direct' "${output_dir}/batched.lowered.ll"
rg -q 'call ptr @nta_acquire_slow' "${output_dir}/batched.lowered.ll"
rg -q 'call void @nta_defer' "${output_dir}/batched.lowered.ll"
rg -q '!nta.acquire' "${output_dir}/batched.lowered.ll"
if rg -q '__nta_(bind_request|acquire_marker|defer_marker)' \
  "${output_dir}/batched.lowered.ll"; then
  echo "lowered module still contains an NTA marker" >&2
  exit 1
fi

"${opt}" \
  -load-pass-plugin="${plugin}" \
  -passes=nta-acquire \
  -S "${source_dir}/tensor-map.ll" \
  -o "${output_dir}/tensor-map.lowered.ll"
rg -q 'call ptr @nta_acquire_tensor_map_slow' \
  "${output_dir}/tensor-map.lowered.ll"
rg -Fq '!{!"request-bound", i32 7, !"tensor-map"}' \
  "${output_dir}/tensor-map.lowered.ll"
rg -q 'phi ptr \[ null, %entry \], \[ %direct.map, %nta.acquire.direct \]' \
  "${output_dir}/tensor-map.lowered.ll"
if rg -q '__nta_(bind_request|acquire_tensor_map_marker|defer_marker)' \
  "${output_dir}/tensor-map.lowered.ll"; then
  echo "lowered tensor-map module still contains an NTA marker" >&2
  exit 1
fi

for fixture in reject-no-binding reject-live-state reject-wrong-token \
               reject-pending-use; do
  "${opt}" \
    -load-pass-plugin="${plugin}" \
    -passes=nta-acquire \
    -S "${source_dir}/${fixture}.ll" \
    -o "${output_dir}/${fixture}.lowered.ll" \
    2>"${output_dir}/${fixture}.stderr"
  rg -q '__nta_acquire_marker' "${output_dir}/${fixture}.lowered.ll"
  rg -q 'nta: warning:' "${output_dir}/${fixture}.stderr"
done

echo "NTA IR tests passed"
