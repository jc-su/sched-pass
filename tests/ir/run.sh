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
  -S "${source_dir}/dependency-set.ll" \
  -o "${output_dir}/dependency-set.lowered.ll"
rg -q 'call i1 @nta_acquire_set_slow' \
  "${output_dir}/dependency-set.lowered.ll"
rg -Fq '!{!"request-bound", i32 20, !"dependency-set", !"split-phase-cta"}' \
  "${output_dir}/dependency-set.lowered.ll"
if rg -q '__nta_(bind_request|acquire_set_marker|defer_marker)' \
  "${output_dir}/dependency-set.lowered.ll"; then
  echo "lowered dependency-set module still contains an NTA marker" >&2
  exit 1
fi

# `clang -fpass-plugin` uses the optimizer extension point rather than an
# explicit `-passes=nta-acquire` pipeline. Exercise that JIT entry path too.
"${opt}" \
  -load-pass-plugin="${plugin}" \
  -passes='default<O3>' \
  -S "${source_dir}/dependency-set.ll" \
  -o "${output_dir}/dependency-set.jit-lowered.ll"
rg -q 'call i1 @nta_acquire_set_slow' \
  "${output_dir}/dependency-set.jit-lowered.ll"
if rg -q '__nta_(bind_request|acquire_set_marker|defer_marker)' \
  "${output_dir}/dependency-set.jit-lowered.ll"; then
  echo "JIT optimizer pipeline still contains an NTA marker" >&2
  exit 1
fi

"${opt}" \
  -load-pass-plugin="${plugin}" \
  -passes=nta-acquire \
  -S "${source_dir}/tensor-map.ll" \
  -o "${output_dir}/tensor-map.lowered.ll"
rg -q 'call ptr @nta_acquire_tensor_map_slow' \
  "${output_dir}/tensor-map.lowered.ll"
rg -Fq '!{!"request-bound", i32 20, !"tensor-map", !"split-phase-cta"}' \
  "${output_dir}/tensor-map.lowered.ll"
rg -q 'phi ptr \[ null, %entry \], \[ %direct.map, %nta.acquire.direct \]' \
  "${output_dir}/tensor-map.lowered.ll"
if rg -q '__nta_(bind_request|acquire_tensor_map_marker|defer_marker)' \
  "${output_dir}/tensor-map.lowered.ll"; then
  echo "lowered tensor-map module still contains an NTA marker" >&2
  exit 1
fi

"${opt}" \
  -load-pass-plugin="${plugin}" \
  -passes=nta-acquire \
  -S "${source_dir}/late-bound.ll" \
  -o "${output_dir}/late-bound.lowered.ll"
rg -q 'call ptr @nta_acquire_slow' "${output_dir}/late-bound.lowered.ll"
rg -q 'and i32 %cta, %catalog.mask' "${output_dir}/late-bound.lowered.ll"
rg -Fq '!{!"request-bound", i32 20, !"byte-address", !"split-phase-cta"}' \
  "${output_dir}/late-bound.lowered.ll"
if rg -q '__nta_(bind_request|acquire_marker|defer_marker)' \
  "${output_dir}/late-bound.lowered.ll"; then
  echo "lowered late-bound module still contains an NTA marker" >&2
  exit 1
fi

fixtures=(
  "reject-no-binding:no valid request binding dominates acquisition"
  "reject-live-state:pending edge contains state that cannot cross CTA deferral"
  "reject-wrong-token:pending edge defers a different acquisition token"
  "reject-pending-use:acquired value is used outside the ready edge"
  "reject-set-live-state:pending edge contains state that cannot cross CTA deferral"
  "reject-divergent-control:acquisition marker is control-dependent on a non-CTA-uniform branch"
  "reject-nondominating-divergent:acquisition marker is control-dependent on a non-CTA-uniform branch"
  "reject-divergent-operand:acquisition marker has a non-CTA-uniform operand"
  "reject-divergent-value-phi:request binding has a non-CTA-uniform operand"
  "reject-device-helper:acquisition markers must be inlined into a GPU kernel entry"
)
for entry in "${fixtures[@]}"; do
  fixture=${entry%%:*}
  reason=${entry#*:}
  if "${opt}" \
      -load-pass-plugin="${plugin}" \
      -passes=nta-acquire \
      -S "${source_dir}/${fixture}.ll" \
      -o "${output_dir}/${fixture}.lowered.ll" \
      2>"${output_dir}/${fixture}.stderr"; then
    echo "unsafe fixture ${fixture} was accepted" >&2
    exit 1
  fi
  rg -Fq "nta: error:" "${output_dir}/${fixture}.stderr"
  rg -Fq "${reason}" "${output_dir}/${fixture}.stderr"
  rg -Fq "NTA acquisition verification failed" \
    "${output_dir}/${fixture}.stderr"
done

echo "NTA IR tests passed"
