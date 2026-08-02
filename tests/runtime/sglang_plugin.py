#!/usr/bin/env python3
"""Validate SGLang plugin registration in frontend and spawned workers."""

from __future__ import annotations

import multiprocessing as mp


def load_in_spawn(result) -> None:
    from importlib.metadata import PackageNotFoundError, distribution, entry_points
    import os
    import sys

    from sglang.srt.plugins import load_plugins

    discovered = [
        (entry.name, entry.value) for entry in entry_points(group="sglang.srt.plugins")
    ]
    try:
        dist_path = str(distribution("nta-runtime")._path)
    except PackageNotFoundError:
        dist_path = "missing"
    load_plugins()
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

    result.put(
        (
            "nta_flashinfer" in ATTENTION_BACKENDS,
            ATTENTION_BACKEND_CHOICES.count("nta_flashinfer"),
            discovered,
            sys.executable,
            sys.path,
            os.getuid(),
            dist_path,
        )
    )


def main() -> None:
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

    from nta_runtime.plugins.sglang import BACKEND_NAME, register

    register()
    register()
    assert BACKEND_NAME in ATTENTION_BACKENDS
    assert ATTENTION_BACKEND_CHOICES.count(BACKEND_NAME) == 1
    assert callable(ATTENTION_BACKENDS[BACKEND_NAME])

    from sglang.srt.model_executor.runner import decode_cuda_graph_runner

    assert getattr(
        decode_cuda_graph_runner.build_replay_fb_view,
        "_nta_preserves_request_metadata",
        False,
    )

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from nta_runtime.plugins.sglang import (
        _ABORT_TARGET,
        _HICACHE_LOAD_TARGET,
        _RELEASE_TARGET,
        _FORWARD_BATCH_TARGET,
    )

    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_HICACHE_LOAD_TARGET]
    )
    assert any(
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_ABORT_TARGET]
    )
    assert any(
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_RELEASE_TARGET]
    )
    assert any(
        kind == HookType.AFTER
        for kind, _, _ in HookRegistry._hooks[_FORWARD_BATCH_TARGET]
    )

    from nta_runtime.engines.sglang import (
        NtaFlashInferAttnBackend,
        _plan_cache_signature,
    )

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"
    signature = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), (7,), 100, 200, 4096, 4096, None
    )
    remapped = _plan_cache_signature(
        (0, 0), (3, 4), (((11,), (21,)),), (7,), 100, 200, 4096, 4096, None
    )
    rebound = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), (7,), 100, 200, 4096, 4096, None
    )
    assert signature != remapped, "plan cache aliased different HiCache page rows"
    assert signature == rebound, "request rebinding invalidated a structural plan"

    context = mp.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=load_in_spawn, args=(result,))
    process.start()
    process.join(30)
    assert process.exitcode == 0
    observed = result.get(timeout=1)
    assert observed[:2] == (True, 1), observed


if __name__ == "__main__":
    main()
