#!/usr/bin/env python3
"""Compile hooked FlashInfer decode and paged-prefill JIT modules through NTA."""

from __future__ import annotations

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
from flashinfer.jit.attention.variants import attention_sink_fa2_decl


TENSOR_NAMES = ["sink", "nta_runtime", "nta_work_items", "nta_dependencies"]
TENSOR_DTYPES = ["float", "uint8_t", "uint8_t", "uint8_t"]
SCALAR_NAMES = ["sm_scale", "nta_work_count"]
SCALAR_DTYPES = ["double", "int64_t"]
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
        [os.path.join(os.environ["NTA_BUILD_DIR"], "nta-jit-phase-load"), module_path],
        check=True,
        stdout=subprocess.DEVNULL,
        env=environment,
    )
    return module_path


def main() -> None:
    baseline_name = "nta_batch_decode_baseline"
    baseline = gen_customize_batch_decode_module(
        baseline_name,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        ["sink"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "AttentionSink",
        attention_sink_fa2_decl,
    )
    decode_name = "nta_batch_decode_hooked"
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
        "AttentionSink",
        attention_sink_fa2_decl,
    )
    prefill_name = "nta_batch_prefill_hooked"
    prefill = gen_customize_batch_prefill_module(
        "fa2",
        prefill_name,
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
        "AttentionSink",
        attention_sink_fa2_decl,
    )
    print(
        "flashinfer_baseline_module="
        f"{check_module(baseline_name, baseline, ('run',))}"
    )
    print(
        f"flashinfer_decode_module={check_module(decode_name, decode, ('run',))}"
    )
    print(
        "flashinfer_prefill_module="
        f"{check_module(prefill_name, prefill, ('ragged_run', 'paged_run'))}"
    )


if __name__ == "__main__":
    main()
