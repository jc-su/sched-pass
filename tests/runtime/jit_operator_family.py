#!/usr/bin/env python3
"""Regression coverage for semantic JIT operator-family identification."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import tempfile


ROOT = pathlib.Path(__file__).parents[2]
SHIM = ROOT / "tools" / "jit" / "nvcc_clang.py"
SPEC = importlib.util.spec_from_file_location("nta_nvcc_clang", SHIM)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SHIM}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ACTIVATE = ROOT / "tools" / "jit" / "activate.py"
ACTIVATE_SPEC = importlib.util.spec_from_file_location("nta_activate", ACTIVATE)
if ACTIVATE_SPEC is None or ACTIVATE_SPEC.loader is None:
    raise RuntimeError(f"could not load {ACTIVATE}")
ACTIVATE_MODULE = importlib.util.module_from_spec(ACTIVATE_SPEC)
ACTIVATE_SPEC.loader.exec_module(ACTIVATE_MODULE)


def main() -> int:
    activation = ACTIVATE.read_text(encoding="utf-8")
    for required in (
        "include/nta/TicketProtocol.cuh",
        "runtime/device/TypedInstrumentation.cuh",
        "runtime/device/OperatorMetadata.cuh",
        "tool_identity(clang)",
        "tool_identity(real_nvcc)",
        "NTA_TRANSPORT_PROGRAM_SHA256",
    ):
        assert required in activation
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
    with tempfile.TemporaryDirectory() as temporary:
        generated = pathlib.Path(temporary)
        binding = generated / "batch_prefill.cu"
        binding.touch()
        instantiation = generated / "batch_prefill_ragged_kernel_mask_0.cu"
        instantiation.write_text("/*CTA_TILE_Q=*/16\n", encoding="utf-8")
        baseline_arguments = [
            "-c",
            str(binding),
            str(generated / "nta_batch_prefill_default_v2_baseline"),
        ]
        assert MODULE.operator_family(baseline_arguments) == 0
        assert MODULE.needs_prefill_cta_tile_32_suppression(baseline_arguments)
        instantiation.write_text("/*CTA_TILE_Q=*/32\n", encoding="utf-8")
        assert not MODULE.needs_prefill_cta_tile_32_suppression(baseline_arguments)

    numerical_inputs = ACTIVATE_MODULE.numerical_fingerprint_inputs(
        ROOT,
        ROOT / "build/libNtaPass.so",
        SHIM,
        ROOT / "include/nta/RuntimeABI.h",
    )
    relative_inputs = {
        path.relative_to(ROOT).as_posix()
        for path in numerical_inputs
        if path.is_relative_to(ROOT)
    }
    assert "runtime/device/Acquire.cuh" in relative_inputs
    assert "runtime/device/OperatorMetadata.cuh" in relative_inputs
    assert "runtime/device/JitRuntime.cuh" not in relative_inputs
    assert "runtime/device/TransportProgram.cu" not in relative_inputs
    assert "build/libnta-runtime.so" not in relative_inputs
    assert "build/libnta-transport-program.so" not in relative_inputs

    previous = {
        name: os.environ.get(name)
        for name in (
            "NTA_FLASHINFER_HOOK",
            "NTA_JIT_METADATA_SOURCE",
            "NTA_JIT_REQUEST_BOUND_SOURCE",
            "NTA_JIT_STREAM_ORDERED_SOURCE",
        )
    }
    old_plugin = MODULE.PLUGIN
    try:
        MODULE.PLUGIN = str(SHIM)
        os.environ["NTA_FLASHINFER_HOOK"] = "1"
        os.environ["NTA_JIT_METADATA_SOURCE"] = "batch_decode_kernel.cu"
        os.environ["NTA_JIT_REQUEST_BOUND_SOURCE"] = "request_bound"
        os.environ["NTA_JIT_STREAM_ORDERED_SOURCE"] = "stream_ordered"
        typed_source = [
            "-c",
            "/tmp/generated/nta_batch_decode_default_v2_hooked/"
            "batch_decode_kernel.cu",
        ]
        translated = MODULE.translate(typed_source, True)
        rendered = " ".join(translated)
        assert "-DNTA_DEVICE_PHASE_KERNELS=0" in translated
        assert "runtime/device/Acquire.cuh" in rendered
        assert "runtime/device/OperatorMetadata.cuh" in rendered
        assert "runtime/device/JitRuntime.cuh" not in rendered

        helper = MODULE.translate(
            [
                "-c",
                "/tmp/generated/nta_batch_decode_default_v2_hooked/"
                "batch_decode_jit_binding.cu",
            ],
            True,
        )
        helper_rendered = " ".join(helper)
        assert "runtime/device/Acquire.cuh" in helper_rendered
        assert "runtime/device/OperatorMetadata.cuh" not in helper_rendered
        assert "runtime/device/JitRuntime.cuh" not in helper_rendered
    finally:
        MODULE.PLUGIN = old_plugin
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
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
