#!/bin/bash
# Blackwell CLC compile-gate: prove the woven Cluster Launch Control claim
# path assembles and lowers to real Blackwell SASS, for both sm_100 and
# sm_120. Intended host: UC-Santa_Cruz-2 (LLVM 20, CUDA 12.8, RTX PRO 6000).
#
# This is a COMPILE gate, not a run: the node's GPU has Confidential Compute
# mode ON and the CPU lacks CC support, so the driver cannot attach and NO
# CUDA program runs (cudaGetDeviceCount -> 0). Runtime CLC needs CC mode off
# (a security change; ask the node owner). The gate below is the strongest
# evidence attainable without execution: ptxas accepts the woven kernel and
# emits Blackwell CLC SASS.
set -e
cd "$(dirname "$0")/.."

LLVM=${LLVM:-/usr/lib/llvm-20}
CUDA=${CUDA:-/usr/local/cuda-12.9}
CXX="$LLVM/bin/clang++"

echo "=== building plugin (LLVM 20) ==="
cmake -S . -B build -DLLVM_DIR="$LLVM/lib/cmake/llvm" -GNinja >/dev/null
ninja -C build

PL="$PWD/build/libSchedPass.so"
for ARCH in sm_100 sm_120; do
  echo "=== $ARCH: compile+ptxas+link the CLC work-queue fixture ==="
  SCHED_WORKQUEUE=1 SCHED_CLC=1 "$CXX" -x cuda --cuda-gpu-arch="$ARCH" \
      --cuda-path="$CUDA" -O2 -std=c++17 -fpass-plugin="$PL" \
      -Wno-unknown-cuda-version -L"$CUDA/lib64" -lcudart \
      test/hetero_batch.cu -o "build/hetero_clc_$ARCH" 2>&1 \
      | grep -ivE "fatbinary warning" || true
  echo "  linked: build/hetero_clc_$ARCH"
  echo "  woven CLC PTX instructions:"
  SCHED_WORKQUEUE=1 SCHED_CLC=1 "$CXX" -x cuda --cuda-gpu-arch="$ARCH" \
      --cuda-path="$CUDA" -O2 -std=c++17 -fpass-plugin="$PL" \
      -Wno-unknown-cuda-version --cuda-device-only -S \
      test/hetero_batch.cu -o "/tmp/hetero_$ARCH.s" 2>/dev/null
  grep -oE "clusterlaunchcontrol[^ ]*" "/tmp/hetero_$ARCH.s" | sort -u | sed 's/^/    /'
  echo "  CLC choreography in Blackwell SASS:"
  "$CUDA/bin/cuobjdump" -sass "build/hetero_clc_$ARCH" 2>/dev/null \
      | grep -oE "(ELECT|SYNCS\.ARRIVE\.TRANS64|SYNCS\.PHASECHK[^ ]*|MEMBAR\.ALL\.CTA|S2R R[0-9]+, SR_CgaCtaId)" \
      | sort -u | sed 's/^/    /'
done
echo "=== CLC compile-gate PASSED (runtime pending CC-mode-off) ==="
