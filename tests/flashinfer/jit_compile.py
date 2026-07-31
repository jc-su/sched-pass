#!/usr/bin/env python3
"""Compile FlashInfer's real multi-source batch-decode JIT through NTA."""

from __future__ import annotations

import os
import pathlib

import torch
from flashinfer.jit.attention.modules import gen_customize_batch_decode_module
from flashinfer.jit.attention.variants import attention_sink_fa2_decl


def main() -> None:
    name = "nta_batch_decode_jit_smoke"
    specification = gen_customize_batch_decode_module(
        name,
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
    module = specification.build_and_load()
    module.get_function("plan")
    module.get_function("run")
    workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    matches = list(workspace.rglob(f"{name}.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one FlashInfer JIT module, found {len(matches)} in {workspace}"
        )
    module_path = matches[0].resolve()
    print(f"flashinfer_jit_module={module_path}")


if __name__ == "__main__":
    main()
