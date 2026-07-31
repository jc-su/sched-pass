#!/bin/bash
# Build the plugin + both fixture binaries and run them on the GPU.
# Intended host: cloudsys01 (LLVM 19, CUDA 12.9, RTX A6000 sm_86).
set -e
cd "$(dirname "$0")/.."

LLVM=${LLVM:-/usr/lib/llvm-19}
CUDA=${CUDA:-/usr/local/cuda-12.9}
ARCH=${ARCH:-sm_86}

echo "=== building plugin (LLVM at $LLVM) ==="
cmake -S . -B build -DLLVM_DIR="$LLVM/lib/cmake/llvm" -GNinja >/dev/null
ninja -C build

CXX="$LLVM/bin/clang++"
FLAGS=(-x cuda --cuda-gpu-arch="$ARCH" --cuda-path="$CUDA" -O2 -std=c++17
       -fpass-plugin="$PWD/build/libSchedPass.so"
       -L"$CUDA/lib64" -lcudart -Wno-unknown-cuda-version)

echo "=== compiling fixture (basic) ==="
SCHED_DEBUG=1 "$CXX" "${FLAGS[@]}" test/paged_decode.cu -o build/paged_decode

echo "=== compiling fixture (work-queue) ==="
SCHED_DEBUG=1 SCHED_WORKQUEUE=1 "$CXX" "${FLAGS[@]}" -DSCHED_FIXTURE_WQ=1 \
    test/paged_decode.cu -o build/paged_decode_wq

echo "=== compiling hetero-batch demo (work-queue) ==="
SCHED_DEBUG=1 SCHED_WORKQUEUE=1 "$CXX" "${FLAGS[@]}" \
    test/hetero_batch.cu -o build/hetero_batch

echo "=== running (basic) ==="
./build/paged_decode

echo "=== running (work-queue) ==="
./build/paged_decode_wq

echo "=== running (hetero-batch scheduler demo) ==="
./build/hetero_batch

# Blackwell CLC: run test/blackwell_clc_gate.sh on an sm_100+ node (LLVM 20).
# On sm_86 SCHED_CLC falls back to the ticket claim, so the fixtures above
# already cover the acquisition-layer logic; CLC hardware claim is validated
# separately on the Blackwell node.
