#!/usr/bin/env python3
"""Run a real SGLang/FlashInfer baseline smoke workload and emit JSON."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shutil
import statistics
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument(
        "--cuda-graph-decode",
        choices=("disabled", "full"),
        default="full",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        default="flashinfer",
    )
    parser.add_argument(
        "--flashinfer-workspace-base",
        type=pathlib.Path,
        default=ROOT / "results" / "qualification" / "sglang-flashinfer",
    )
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if (
        min(
            args.requests,
            args.max_new_tokens,
            args.iterations,
            args.warmup_iterations,
            args.context_length,
        )
        <= 0
    ):
        parser.error("request, token, iteration, and context values must be positive")
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("--mem-fraction-static must be between zero and one")
    return args


def configure_jit_environment(args: argparse.Namespace) -> pathlib.Path:
    host_cxx = args.cuda_host_cxx
    if host_cxx is None:
        discovered = next(
            (shutil.which(name) for name in ("g++-14", "g++-13", "g++-12")),
            None,
        )
        if discovered is None:
            raise RuntimeError(
                "CUDA-compatible host compiler not found; pass --cuda-host-cxx"
            )
        host_cxx = pathlib.Path(discovered)
    host_cxx = host_cxx.resolve()
    if not host_cxx.is_file():
        raise RuntimeError(f"CUDA host compiler does not exist: {host_cxx}")

    configured_launcher = os.environ.get("FLASHINFER_NVCC")
    launcher = (
        pathlib.Path(configured_launcher).resolve()
        if configured_launcher
        else (ROOT / "tools" / "flashinfer" / "nvcc_compat.py").resolve()
    )
    if not launcher.is_file():
        raise RuntimeError(f"FlashInfer NVCC launcher does not exist: {launcher}")
    configured_workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    workspace = (
        pathlib.Path(configured_workspace).resolve()
        if configured_workspace
        else args.flashinfer_workspace_base.resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    for stale in workspace.glob("nta-engine.*.json"):
        stale.unlink()
    os.environ["NTA_ENGINE_STATS_FILE"] = str(workspace / "nta-engine.json")

    # CC serves two masters: Triton compiles C11 launcher stubs through $CC
    # (a C++ compiler breaks them), while FlashInfer's ninja passes $CC to
    # nvcc as -ccbin (a CUDA-incompatible GCC breaks that). The CUDA-matched
    # C driver paired with the chosen host C++ compiler satisfies both.
    host_cc = os.environ.get("CC") or shutil.which(host_cxx.name.replace("g++", "gcc"))
    if host_cc is None:
        raise RuntimeError(
            f"no CUDA-compatible C compiler matches {host_cxx.name}; set CC explicitly"
        )
    os.environ["CC"] = str(host_cc)
    os.environ["CXX"] = str(host_cxx)
    os.environ["CUDAHOSTCXX"] = str(host_cxx)
    os.environ["NTA_NVCC_HOST_COMPILER"] = str(host_cxx)
    os.environ["FLASHINFER_NVCC"] = str(launcher)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
    return host_cxx


def output_tokens(result: dict[str, Any]) -> int:
    metadata = result.get("meta_info", {})
    for field in ("completion_tokens", "completion_tokens_without_jump_forward"):
        value = metadata.get(field)
        if isinstance(value, int):
            return value
    output_ids = metadata.get("output_ids")
    return len(output_ids) if isinstance(output_ids, list) else 0


def main() -> int:
    args = parse_args()
    host_cxx = configure_jit_environment(args)
    import sglang as sgl
    import torch

    prompts = [
        f"Request {index}: explain one property of finite GPU kernels."
        for index in range(args.requests)
    ]
    sampling = {"temperature": 0, "max_new_tokens": args.max_new_tokens}
    load_started = time.perf_counter()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        cuda_graph_backend_decode=args.cuda_graph_decode,
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=512,
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        for _ in range(args.warmup_iterations):
            engine.generate(prompts, sampling)
        samples: list[float] = []
        results = None
        for _ in range(args.iterations):
            started = time.perf_counter()
            results = engine.generate(prompts, sampling)
            samples.append(time.perf_counter() - started)

    assert results is not None
    if not isinstance(results, list) or len(results) != args.requests:
        raise RuntimeError("SGLang returned an incomplete result batch")
    generated = sum(output_tokens(result) for result in results)
    if generated == 0 or any(
        not isinstance(result.get("text"), str) for result in results
    ):
        raise RuntimeError("SGLang returned no generated token data")
    elapsed = statistics.median(samples)
    output_digest = hashlib.sha256()
    for result in results:
        output_digest.update(result["text"].encode("utf-8"))
        output_digest.update(b"\0")
    stats_workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    stats_files = sorted(stats_workspace.glob("nta-engine.*.json"))
    engine_stats = []
    for path in stats_files:
        try:
            engine_stats.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    integrated = args.attention_backend == "nta_flashinfer"
    report = {
        "schema": 1,
        "classification": "serving-integration-smoke",
        "nta_integrated": integrated,
        "engine": "sglang",
        "attention_backend": args.attention_backend,
        "engine_version": importlib.metadata.version("sglang"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_host_cxx": str(host_cxx),
        "flashinfer_workspace_base": os.environ["FLASHINFER_WORKSPACE_BASE"],
        "model": str(args.model.resolve()),
        "requests": args.requests,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "cuda_graph_decode": args.cuda_graph_decode,
        "generated_tokens": generated,
        "generated_text_sha256": output_digest.hexdigest(),
        "load_seconds": load_seconds,
        "batch_seconds_samples": samples,
        "median_batch_seconds": elapsed,
        "requests_per_second": args.requests / elapsed,
        "generated_tokens_per_second": generated / elapsed,
        "engine_stats": engine_stats,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
