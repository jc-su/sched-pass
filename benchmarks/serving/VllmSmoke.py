#!/usr/bin/env python3
"""Run a reproducible vLLM reference/native tier integration smoke.

This is an integration gate, not a headline performance benchmark.  It uses
the same model, request batch, CUDA/JIT environment, and output digest for the
stock FlashInfer and NTA CUSTOM arms.  Native evidence is collected from the
worker process through an explicit per-PID file; parent-process generation
success alone is never treated as proof of NTA execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import statistics
import subprocess
import time
import uuid
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--backend", choices=("stock", "stock_offload", "nta"), required=True
    )
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument(
        "--prompt-profile",
        choices=("auto", "short", "long_prefix"),
        default="auto",
        help=(
            "prompt shape; auto selects long_prefix for host reload and short otherwise"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--serving-tier", choices=("hbm", "host_staged"), default="hbm")
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        help="explicit vLLM HBM KV capacity for deterministic tier tests",
    )
    parser.add_argument("--host-cache-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-home", type=pathlib.Path)
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="optional JSON report path; stdout remains the machine-readable report",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if (
        min(
            args.requests,
            args.max_new_tokens,
            args.iterations,
            args.warmup_iterations,
            args.max_model_len,
            args.max_num_seqs,
        )
        <= 0
    ):
        parser.error("request, token, iteration, and model limits must be positive")
    if args.max_num_seqs < args.requests:
        parser.error("--max-num-seqs must cover the request batch")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be between zero and one")
    if args.backend == "stock" and args.serving_tier != "hbm":
        parser.error("stock backend supports only the resident HBM smoke")
    if args.backend == "stock_offload" and args.serving_tier != "host_staged":
        parser.error("stock_offload requires --serving-tier host_staged")
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        parser.error("--kv-cache-memory-bytes must be positive")
    if args.host_cache_bytes <= 0:
        parser.error("--host-cache-bytes must be positive")
    return args


def configure_environment(args: argparse.Namespace) -> pathlib.Path:
    from cuda_environment import configure_jit_environment

    _, _, workspace = configure_jit_environment(
        root=ROOT,
        workspace=args.flashinfer_workspace_base,
        host_cxx=args.cuda_host_cxx,
        cuda_home=args.cuda_home,
        instrumented=args.backend == "nta",
    )
    return workspace


def output_text(result: Any) -> str:
    outputs = getattr(result, "outputs", ())
    return str(getattr(outputs[0], "text", "")) if outputs else ""


def output_tokens(result: Any) -> int:
    outputs = getattr(result, "outputs", ())
    return len(getattr(outputs[0], "token_ids", ())) if outputs else 0


def output_token_ids(result: Any) -> tuple[int, ...]:
    outputs = getattr(result, "outputs", ())
    return (
        tuple(int(token) for token in getattr(outputs[0], "token_ids", ()))
        if outputs
        else ()
    )


def repository_metadata() -> tuple[str, bool]:
    """Return the source revision used by this smoke, without importing tools."""

    revision = os.environ.get("NTA_REVISION", "").strip()
    if not revision:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            revision = completed.stdout.strip()
    if not revision:
        revision = "unrecorded"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        dirty = False
    else:
        dirty = status.returncode != 0 or bool(status.stdout.strip())
    return revision, dirty


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    run_token = uuid.uuid4().hex
    os.environ["NTA_VLLM_EVIDENCE_TOKEN"] = run_token
    if args.backend == "nta":
        os.environ.update(
            {
                "NTA_VLLM_NATIVE": "1",
                "NTA_VLLM_ALLOW_STOCK_FALLBACK": "0",
                "NTA_SERVING_TIER": args.serving_tier,
            }
        )
        if args.serving_tier == "host_staged":
            os.environ["NTA_VLLM_VERIFY_TRANSFER"] = "1"
        else:
            os.environ.pop("NTA_VLLM_VERIFY_TRANSFER", None)
    else:
        os.environ.pop("NTA_VLLM_NATIVE", None)
        os.environ.pop("NTA_VLLM_ALLOW_STOCK_FALLBACK", None)
        os.environ.pop("NTA_VLLM_VERIFY_TRANSFER", None)
        os.environ.pop("NTA_SERVING_TIER", None)

    import torch
    from vllm import LLM, SamplingParams

    prompt_profile = args.prompt_profile
    if prompt_profile == "auto":
        prompt_profile = (
            "long_prefix" if args.serving_tier == "host_staged" else "short"
        )
    if prompt_profile == "long_prefix":
        prefix = " ".join(
            [
                "A finite GPU kernel has explicit work identity, bounded progress, "
                "and exact data dependencies."
            ]
            * 12
        )
        prompts = [
            f"{prefix} Request {index}: summarize the invariant."
            for index in range(args.requests)
        ]
    else:
        prompts = [
            f"Request {index}: explain one property of finite GPU kernels."
            for index in range(args.requests)
        ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=0,
    )
    attention_backend = "CUSTOM" if args.backend == "nta" else "FLASHINFER"
    connector_config = (
        {
            "kv_connector": (
                "NtaVllmConnector"
                if args.backend == "nta"
                else "SimpleCPUOffloadConnector"
            ),
            **(
                {"kv_connector_module_path": "nta_runtime.connectors.vllm"}
                if args.backend == "nta"
                else {}
            ),
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": args.host_cache_bytes,
                "lazy_offload": False,
            },
        }
        if args.backend in {"nta", "stock_offload"}
        else None
    )
    load_started = time.perf_counter()
    engine = LLM(
        model=str(args.model.resolve()),
        attention_backend=attention_backend,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
        seed=0,
        enable_prefix_caching=True,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        kv_transfer_config=connector_config,
    )
    try:
        load_seconds = time.perf_counter() - load_started
        samples: list[float] = []
        results: list[Any] | None = None
        if args.serving_tier == "host_staged":
            baseline = engine.generate(prompts, sampling)
            baseline_tokens = tuple(output_token_ids(result) for result in baseline)

            def reset_resident_cache(attempt: int) -> None:
                # Store completion is reported on a later engine step. A
                # unique drain request advances that lifecycle without sharing
                # the target prefix. Preserve the connector's CPU directory
                # while resetting only vLLM's resident prefix cache.
                engine.generate(
                    [f"Drain {attempt}: give one word about stream ordering."],
                    sampling,
                )
                if not engine.reset_prefix_cache(reset_connector=False):
                    raise RuntimeError(
                        "vLLM could not quiesce resident KV before host reload"
                    )

            for iteration in range(args.warmup_iterations + args.iterations):
                reset_resident_cache(iteration)
                started = time.perf_counter()
                current = engine.generate(prompts, sampling)
                elapsed = time.perf_counter() - started
                if (
                    tuple(output_token_ids(result) for result in current)
                    != baseline_tokens
                ):
                    raise RuntimeError("vLLM host reload changed generated token IDs")
                if iteration >= args.warmup_iterations:
                    samples.append(elapsed)
                    results = current
        else:
            for _ in range(args.warmup_iterations):
                engine.generate(prompts, sampling)
            for _ in range(args.iterations):
                started = time.perf_counter()
                results = engine.generate(prompts, sampling)
                samples.append(time.perf_counter() - started)
    finally:
        # vLLM 0.26's offline LLM wrapper is not a context manager.  Explicit
        # EngineCore shutdown is required so worker atexit evidence is flushed
        # before the parent validates the run.
        engine.llm_engine.engine_core.shutdown()

    if results is None or len(results) != args.requests:
        raise RuntimeError("vLLM returned an incomplete result batch")
    texts = [output_text(result) for result in results]
    generated = sum(output_tokens(result) for result in results)
    if generated <= 0 or any(not text for text in texts):
        raise RuntimeError("vLLM returned no generated token data")
    digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    for result in results:
        for token in output_token_ids(result):
            token_digest.update(token.to_bytes(8, "little", signed=False))
        token_digest.update(b"\0")

    evidence = []
    for path in sorted(workspace.glob(f"nta-vllm-engine.{run_token}.*.json")):
        try:
            evidence.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid vLLM evidence file: {path}") from error
    contracts = [
        entry.get("consumer_contract") for entry in evidence if isinstance(entry, dict)
    ]
    native_verified = any(
        isinstance(contract, dict) and contract.get("kind") == "native_work_unit"
        for contract in contracts
    )
    if args.backend == "nta" and not native_verified:
        raise RuntimeError(
            "NTA vLLM generated output but no worker native_work_unit evidence"
        )
    if args.backend == "nta":
        consumer_contract = next(
            contract
            for contract in contracts
            if isinstance(contract, dict) and contract.get("kind") == "native_work_unit"
        )
    else:
        consumer_contract = {
            "schema": 1,
            "engine": "vllm",
            "backend": (
                "flashinfer+simple_cpu_offload"
                if args.backend == "stock_offload"
                else "flashinfer"
            ),
            "kind": "framework_reference",
            "exact_demand": True,
            "typed_work_plan": False,
            "native_submission": False,
            "numerical_consumer": True,
            "engine_version": importlib.metadata.version("vllm"),
        }
    native_launches = sum(
        int(entry.get("stats", {}).get("native_decode_launches", 0))
        + int(entry.get("stats", {}).get("native_prefill_launches", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    reference_fallback_launches = sum(
        int(entry.get("stats", {}).get("reference_fallback_launches", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    host_transfer_blocks = sum(
        int(entry.get("stats", {}).get("host_transfer_blocks", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    host_transfer_bytes = sum(
        int(entry.get("stats", {}).get("host_transfer_bytes", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    host_launches = sum(
        int(entry.get("stats", {}).get("host_decode_launches", 0))
        + int(entry.get("stats", {}).get("host_prefill_launches", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    host_preload_batches = sum(
        int(entry.get("stats", {}).get("host_preload_batches", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    host_preload_waits = sum(
        int(entry.get("stats", {}).get("host_preload_waits", 0))
        for entry in evidence
        if isinstance(entry, dict)
    )
    native_worker_stats = [
        entry.get("stats", {})
        for entry in evidence
        if isinstance(entry, dict)
        and isinstance(entry.get("consumer_contract"), dict)
        and entry["consumer_contract"].get("kind") == "native_work_unit"
    ]
    worker_incremental_wrapper_builds = sum(
        int(stats.get("worker_incremental_wrapper_builds", 0))
        for stats in native_worker_stats
    )
    worker_incremental_plan_builds = sum(
        int(stats.get("worker_incremental_plan_builds", 0))
        for stats in native_worker_stats
    )
    worker_incremental_plan_reuses = sum(
        int(stats.get("worker_incremental_plan_reuses", 0))
        for stats in native_worker_stats
    )
    worker_incremental_workspace_allocated_bytes = sum(
        int(stats.get("worker_incremental_workspace_allocated_bytes", 0))
        for stats in native_worker_stats
    )
    worker_request_bound_wrapper_builds = sum(
        int(stats.get("worker_request_bound_wrapper_builds", 0))
        for stats in native_worker_stats
    )
    worker_request_bound_plan_builds = sum(
        int(stats.get("worker_request_bound_plan_builds", 0))
        for stats in native_worker_stats
    )
    worker_request_bound_plan_reuses = sum(
        int(stats.get("worker_request_bound_plan_reuses", 0))
        for stats in native_worker_stats
    )
    worker_request_bound_workspace_allocated_bytes = sum(
        int(stats.get("worker_request_bound_workspace_allocated_bytes", 0))
        for stats in native_worker_stats
    )
    worker_request_bound_workspace_borrowed_bindings = sum(
        int(stats.get("worker_request_bound_workspace_borrowed_bindings", 0))
        for stats in native_worker_stats
    )
    worker_attention_workspace_peak_bytes = max(
        (
            int(stats.get("worker_attention_workspace_peak_bytes", 0))
            for stats in native_worker_stats
        ),
        default=0,
    )
    worker_attention_borrowed_workspace_peak_bytes = max(
        (
            int(stats.get("worker_attention_borrowed_workspace_peak_bytes", 0))
            for stats in native_worker_stats
        ),
        default=0,
    )
    if args.backend == "nta" and args.serving_tier == "host_staged":
        if (
            host_transfer_blocks <= 0
            or host_transfer_bytes <= 0
            or host_launches <= 0
            or host_preload_batches <= 0
            or host_preload_waits <= 0
        ):
            raise RuntimeError(
                "vLLM host smoke completed without evidenced host materialization"
            )
        expected_phases_per_worker = 2 if args.max_new_tokens > 1 else 1
        expected_wrapper_builds = expected_phases_per_worker * len(native_worker_stats)
        if worker_request_bound_wrapper_builds != expected_wrapper_builds:
            raise RuntimeError(
                "vLLM host smoke did not preserve worker-scoped FlashInfer "
                "workspace lifetime: "
                f"expected {expected_wrapper_builds} wrappers, observed "
                f"{worker_request_bound_wrapper_builds}"
            )
        if (
            worker_incremental_wrapper_builds != 0
            or worker_request_bound_plan_builds <= 0
            or worker_request_bound_plan_reuses <= 0
            or worker_request_bound_workspace_allocated_bytes != 0
            or worker_request_bound_workspace_borrowed_bindings <= 0
            or worker_attention_workspace_peak_bytes != 0
            or worker_attention_borrowed_workspace_peak_bytes <= 0
        ):
            raise RuntimeError(
                "vLLM optimized host smoke did not use only the worker-shared "
                "request-bound planner with framework-owned workspace"
            )
    revision, dirty = repository_metadata()
    median_seconds = statistics.median(samples)
    report = {
        "schema": 1,
        "classification": "vllm-serving-integration-smoke",
        "backend": args.backend,
        "serving_tier": args.serving_tier,
        "backend_selected": True,
        "native_execution_verified": native_verified,
        "native_launches": native_launches,
        "reference_fallback_launches": reference_fallback_launches,
        "host_launches": host_launches,
        "host_preload_batches": host_preload_batches,
        "host_preload_waits": host_preload_waits,
        "host_transfer_blocks": host_transfer_blocks,
        "host_transfer_bytes": host_transfer_bytes,
        "worker_incremental_wrapper_builds": worker_incremental_wrapper_builds,
        "worker_incremental_plan_builds": worker_incremental_plan_builds,
        "worker_incremental_plan_reuses": worker_incremental_plan_reuses,
        "worker_incremental_workspace_allocated_bytes": (
            worker_incremental_workspace_allocated_bytes
        ),
        "worker_request_bound_wrapper_builds": worker_request_bound_wrapper_builds,
        "worker_request_bound_plan_builds": worker_request_bound_plan_builds,
        "worker_request_bound_plan_reuses": worker_request_bound_plan_reuses,
        "worker_request_bound_workspace_allocated_bytes": (
            worker_request_bound_workspace_allocated_bytes
        ),
        "worker_request_bound_workspace_borrowed_bindings": (
            worker_request_bound_workspace_borrowed_bindings
        ),
        "worker_attention_workspace_peak_bytes": (
            worker_attention_workspace_peak_bytes
        ),
        "worker_attention_borrowed_workspace_peak_bytes": (
            worker_attention_borrowed_workspace_peak_bytes
        ),
        "stock_fallback_enabled": (
            args.backend == "nta"
            and os.environ.get("NTA_VLLM_ALLOW_STOCK_FALLBACK") == "1"
        ),
        "engine": "vllm",
        "engine_version": importlib.metadata.version("vllm"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": str(args.model.resolve()),
        "revision": revision,
        "dirty": dirty,
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "requests": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "prompt_profile": prompt_profile,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "host_cache_bytes": (
            args.host_cache_bytes if args.serving_tier == "host_staged" else None
        ),
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "generated_tokens": generated,
        "generated_text_sha256": digest.hexdigest(),
        "generated_token_ids_sha256": token_digest.hexdigest(),
        "load_seconds": load_seconds,
        "batch_seconds_samples": samples,
        "median_batch_seconds": median_seconds,
        "requests_per_second": args.requests / median_seconds,
        "generated_tokens_per_second": generated / median_seconds,
        "evidence": evidence,
        "consumer_contract": consumer_contract,
        "flashinfer_workspace_base": str(workspace),
    }
    serialized = json.dumps(report, sort_keys=True)
    if args.output is not None:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
