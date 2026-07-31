#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CLANG=${CLANG:-/usr/bin/clang++-20}
CUDA=${CUDA:-/usr/local/cuda-12.8}
ARCH=${ARCH:-sm_120}
CUOBJDUMP=${CUOBJDUMP:-$CUDA/bin/cuobjdump}
OUTDIR=${OUTDIR:-experiments/clc/results}

mkdir -p "$OUTDIR" build
stamp=$(date +%Y%m%d_%H%M%S)

if [ "$#" -eq 0 ]; then
  set -- clc_probe clc_decode_probe clc_2d_probe clc_trace_probe \
    clc_cluster_probe clc_tuple_probe clc_participation_probe \
    clc_runtime_probe clc_mapping_probe clc_pressure_probe clc_graph_probe
fi

for target in "$@"; do
  src="experiments/clc/${target}.cu"
  bin="build/${target}"
  ptx="$OUTDIR/isa_${target}_${stamp}.ptx"
  sass="$OUTDIR/isa_${target}_${stamp}.sass"
  resource="$OUTDIR/isa_${target}_${stamp}.resources.txt"
  clc_summary="$OUTDIR/isa_${target}_${stamp}.clc_summary.txt"

  echo "== build $target =="
  "$CLANG" -x cuda --cuda-gpu-arch="$ARCH" --cuda-path="$CUDA" \
    -O2 -std=c++17 -Wno-unknown-cuda-version \
    -L"$CUDA/lib64" -lcudart "$src" -o "$bin"

  echo "== PTX $target =="
  "$CLANG" -x cuda --cuda-gpu-arch="$ARCH" --cuda-path="$CUDA" \
    -O2 -std=c++17 -Wno-unknown-cuda-version \
    --cuda-device-only -S "$src" -o "$ptx"

  echo "== SASS/resources $target =="
  "$CUOBJDUMP" --dump-sass "$bin" > "$sass"
  "$CUOBJDUMP" --dump-resource-usage "$bin" > "$resource"

  {
    echo "# $target"
    echo
    echo "## PTX CLC sequence"
    grep -niE "clusterlaunchcontrol|mbarrier|query_cancel|get_first_ctaid|try_cancel" \
      "$ptx" || true
    echo
    echo "## SASS likely lowered CLC/mbarrier sequence"
    grep -niE "UGETNEXTWORKID|SYNCS|PHASECHK|TRYWAIT|FENCE|MEMBAR|S2R|CgaCtaId" \
      "$sass" || true
    echo
    echo "## Resources"
    cat "$resource"
  } > "$clc_summary"

  echo "ptx=$ptx"
  echo "sass=$sass"
  echo "resource=$resource"
  echo "summary=$clc_summary"
done
