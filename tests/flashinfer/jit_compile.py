#!/usr/bin/env python3
"""Compile hooked FlashInfer decode and paged-prefill JIT modules through NTA."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import pathlib
import subprocess

import torch
from flashinfer.jit.attention.modules import (
    gen_customize_batch_decode_module,
    gen_customize_batch_prefill_module,
)

TENSOR_NAMES = ["nta_runtime", "nta_work_items", "nta_dependencies"]
TENSOR_DTYPES = ["uint8_t", "uint8_t", "uint8_t"]
SCALAR_NAMES = ["sm_scale", "nta_work_count", "nta_skip_merge"]
SCALAR_DTYPES = ["double", "int64_t", "int64_t"]
VARIANT_NAME = "DefaultAttention<false, false, false, false>"
VARIANT_DECL = "#include <flashinfer/attention/variants.cuh>"
ABI_VERSION = int(os.environ["NTA_ABI_VERSION"])


def check_module(
    name: str, specification: object, run_functions: tuple[str, ...]
) -> pathlib.Path:
    module = specification.build_and_load()
    module.get_function("plan")
    for function in run_functions:
        module.get_function(function)
    workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    matches = list(workspace.rglob(f"{name}.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one FlashInfer JIT module, found {len(matches)} in {workspace}"
        )
    module_path = matches[0].resolve()
    library = ctypes.CDLL(str(module_path))
    typed = "_baseline" not in name
    if not typed:
        try:
            library.nta_jit_operator_contract
        except AttributeError:
            return module_path
        raise RuntimeError("baseline numerical module unexpectedly exports NTA metadata")
    library.nta_jit_abi_version.restype = ctypes.c_uint32
    version = library.nta_jit_abi_version()
    if version != ABI_VERSION:
        raise RuntimeError(f"unexpected NTA runtime ABI {version} in {module_path}")
    tvm_ffi_spec = importlib.util.find_spec("tvm_ffi")
    if tvm_ffi_spec is None or tvm_ffi_spec.origin is None:
        raise RuntimeError("tvm_ffi is required to load a FlashInfer JIT module")
    tvm_ffi_library = (
        pathlib.Path(tvm_ffi_spec.origin).resolve().parent / "lib/libtvm_ffi.so"
    )
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = os.pathsep.join(
        value
        for value in (str(tvm_ffi_library), environment.get("LD_PRELOAD", ""))
        if value
    )
    subprocess.run(
        [
            os.path.join(os.environ["NTA_BUILD_DIR"], "nta-jit-operator-load"),
            module_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        env=environment,
    )
    return module_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "prefill-baseline", "stream-typed"),
        default="all",
        help="materialize only the requested differential artifact",
    )
    options = parser.parse_args()
    baseline_name = "nta_batch_decode_default_v2_baseline"
    baseline = gen_customize_batch_decode_module(
        baseline_name,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        [],
        [],
        ["sm_scale"],
        ["double"],
        VARIANT_NAME,
        VARIANT_DECL,
    )
    decode_name = "nta_batch_decode_default_v2_hooked"
    decode = gen_customize_batch_decode_module(
        decode_name,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        TENSOR_NAMES,
        TENSOR_DTYPES,
        SCALAR_NAMES,
        SCALAR_DTYPES,
        VARIANT_NAME,
        VARIANT_DECL,
    )
    decode_bf16_name = "nta_batch_decode_default_v2_hooked_bf16"
    decode_bf16 = gen_customize_batch_decode_module(
        decode_bf16_name,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        TENSOR_NAMES,
        TENSOR_DTYPES,
        SCALAR_NAMES,
        SCALAR_DTYPES,
        VARIANT_NAME,
        VARIANT_DECL,
    )

    def prefill_specification(
        name: str, dtype: torch.dtype, *, typed: bool = True
    ) -> object:
        return gen_customize_batch_prefill_module(
            "fa2",
            name,
            dtype,
            dtype,
            dtype,
            torch.int32,
            128,
            128,
            TENSOR_NAMES if typed else [],
            TENSOR_DTYPES if typed else [],
            SCALAR_NAMES if typed else ["sm_scale"],
            SCALAR_DTYPES if typed else ["double"],
            VARIANT_NAME,
            VARIANT_DECL,
        )

    prefill_name = "nta_batch_prefill_default_v2_hooked"
    prefill = prefill_specification(prefill_name, torch.float16)
    prefill_baseline_name = "nta_batch_prefill_default_v2_baseline"
    prefill_baseline = prefill_specification(
        prefill_baseline_name, torch.float16, typed=False
    )
    prefill_bf16_name = "nta_batch_prefill_default_v2_hooked_bf16"
    prefill_bf16 = prefill_specification(prefill_bf16_name, torch.bfloat16)
    if options.only == "prefill-baseline":
        print(
            "flashinfer_prefill_baseline_module="
            f"{check_module(prefill_baseline_name, prefill_baseline, ('ragged_run', 'paged_run'))}"
        )
        return
    if options.only == "stream-typed":
        stream_typed_name = (
            "nta_sglang_decode_stream_ordered_v1_demand_acquire_tc_h128_float16_float16"
        )
        stream_typed = prefill_specification(stream_typed_name, torch.float16)
        print(
            "flashinfer_stream_typed_module="
            f"{check_module(stream_typed_name, stream_typed, ('ragged_run', 'paged_run'))}"
        )
        return
    print(
        f"flashinfer_baseline_module={check_module(baseline_name, baseline, ('run',))}"
    )
    print(f"flashinfer_decode_module={check_module(decode_name, decode, ('run',))}")
    print(
        "flashinfer_decode_bf16_module="
        f"{check_module(decode_bf16_name, decode_bf16, ('run',))}"
    )
    print(
        "flashinfer_prefill_module="
        f"{check_module(prefill_name, prefill, ('ragged_run', 'paged_run'))}"
    )
    print(
        "flashinfer_prefill_baseline_module="
        f"{check_module(prefill_baseline_name, prefill_baseline, ('ragged_run', 'paged_run'))}"
    )
    print(
        "flashinfer_prefill_bf16_module="
        f"{check_module(prefill_bf16_name, prefill_bf16, ('ragged_run', 'paged_run'))}"
    )


if __name__ == "__main__":
    main()
