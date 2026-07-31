#!/usr/bin/env bash
# One-shot host fix for the CUDA 12.9 + glibc 2.41 cudafe clash that blocks
# compiling FlashInfer long-input prefill kernels (sinpi/cospi noexcept
# mismatch). Adds `noexcept(true)` to CUDA's 4 decls to match glibc. Needs
# sudo (CUDA header is root-owned). Reversible: keeps a .bak. See ROADMAP.md.
set -eu
H=/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h
[ -w "$H" ] || { echo "run with sudo: sudo bash $0"; exit 1; }
cp -n "$H" "$H.bak"
sed -i -E 's/(double[[:space:]]+(sinpi|cospi)\(double x\));/\1 noexcept(true);/' "$H"
sed -i -E 's/(float[[:space:]]+(sinpif|cospif)\(float x\));/\1 noexcept(true);/' "$H"
echo "patched $H (backup at $H.bak). Now: clear the failed cache and re-run --"
echo "  rm -rf ~/.cache/flashinfer/0.6.12/120f/cached_ops/*batch_prefill*"
