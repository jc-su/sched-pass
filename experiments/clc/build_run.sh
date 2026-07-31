#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CLANG=${CLANG:-/usr/bin/clang++-20}
CUDA=${CUDA:-/usr/local/cuda-12.8}
ARCH=${ARCH:-sm_120}

mkdir -p build

TARGET=${TARGET:-clc_probe}
SRC="experiments/clc/${TARGET}.cu"

"$CLANG" -x cuda --cuda-gpu-arch="$ARCH" --cuda-path="$CUDA" \
  -O2 -std=c++17 -Wno-unknown-cuda-version \
  -L"$CUDA/lib64" -lcudart \
  "$SRC" -o "build/$TARGET"

./build/"$TARGET" "$@"
