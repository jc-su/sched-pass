#!/usr/bin/env python3
"""Regression coverage for semantic JIT operator-family identification."""

from __future__ import annotations

import importlib.util
import pathlib


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
    assert MODULE.operator_family(
        [
            "-c",
            "/tmp/generated/nta_sglang_prefill_demand_acquire_v10_h128/"
            "batch_prefill_paged_kernel_mask_0.cu",
        ]
    ) == 2
    assert MODULE.operator_family(["-c", "batch_decode_kernel.cu"]) == 1
    assert MODULE.operator_family(["-c", "unrelated.cu"]) == 0

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
