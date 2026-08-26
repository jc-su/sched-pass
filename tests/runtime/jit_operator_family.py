#!/usr/bin/env python3
"""Regression coverage for semantic JIT operator-family identification."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile


ROOT = pathlib.Path(__file__).parents[2]
SHIM = ROOT / "tools" / "jit" / "nvcc_clang.py"
SPEC = importlib.util.spec_from_file_location("nta_nvcc_clang", SHIM)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SHIM}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    decode_module_with_prefill_source = [
        "-c",
        "/tmp/generated/nta_sglang_decode_request_bound_v10_h128/"
        "batch_prefill_paged_kernel_mask_0.cu",
    ]
    assert MODULE.operator_family(decode_module_with_prefill_source) == 1
    assert (
        MODULE.operator_family(
            [
                "-c",
                "/tmp/generated/nta_sglang_prefill_demand_acquire_v10_h128/"
                "batch_prefill_paged_kernel_mask_0.cu",
            ]
        )
        == 2
    )
    assert MODULE.operator_family(["-c", "batch_decode_kernel.cu"]) == 1
    assert (
        MODULE.operator_family(
            ["-c", "nta_batch_decode_default_v2_baseline/batch_decode_kernel.cu"]
        )
        == 0
    )
    assert MODULE.operator_family(["-c", "unrelated.cu"]) == 0
    assert MODULE.has_typed_operator_kernel_source(
        ["-c", "batch_prefill_paged_kernel_mask_0.cu"]
    )
    assert MODULE.has_typed_operator_kernel_source(["-c", "batch_decode_kernel.cu"])
    assert not MODULE.has_typed_operator_kernel_source(
        ["-c", "batch_prefill_jit_binding.cu"]
    )
    filtered = MODULE.filter_cuda_include_args(
        [
            "-I",
            "/usr/local/cuda-13.0/targets/x86_64-linux/include",
            "-isystem",
            "/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl",
            "-isystem/usr/local/cuda-12.9/include",
            "-I",
            "/tmp/project/include",
        ]
    )
    assert filtered == ["-I", "/tmp/project/include"]
    with tempfile.TemporaryDirectory() as temporary:
        cuda_root = pathlib.Path(temporary)
        (cuda_root / "include").mkdir()
        target_include = cuda_root / "targets" / "aarch64-linux" / "include"
        (target_include / "cccl").mkdir(parents=True)
        assert MODULE.cuda_include_dirs(cuda_root) == (
            cuda_root / "include",
            target_include,
            target_include / "cccl",
        )

    try:
        MODULE.operator_family(
            ["/tmp/nta_sglang_decode_x/nta_sglang_prefill_x/kernel.cu"]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("ambiguous operator family was accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
