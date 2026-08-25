#!/usr/bin/env python3
"""Run a reproducible vLLM resident-reference/native integration smoke.

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
import statistics
import time
import uuid
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--backend", choices=("stock", "nta"), required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-home", type=pathlib.Path)
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if min(
        args.requests,
        args.max_new_tokens,
        args.iterations,
        args.warmup_iterations,
        args.max_model_len,
        args.max_num_seqs,
    ) <= 0:
        parser.error("request, token, iteration, and model limits must be positive")
    if args.max_num_seqs < args.requests:
        parser.error("--max-num-seqs must cover the request batch")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be between zero and one")
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
                "NTA_SERVING_TIER": "host_staged",
            }
        )
    else:
        os.environ.pop("NTA_VLLM_NATIVE", None)
        os.environ.pop("NTA_VLLM_ALLOW_STOCK_FALLBACK", None)

    import torch
    from vllm import LLM, SamplingParams

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
    )
    try:
        load_seconds = time.perf_counter() - load_started
        for _ in range(args.warmup_iterations):
            engine.generate(prompts, sampling)
        samples: list[float] = []
        results: list[Any] | None = None
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
        entry.get("consumer_contract")
        for entry in evidence
        if isinstance(entry, dict)
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
            if isinstance(contract, dict)
            and contract.get("kind") == "native_work_unit"
        )
    else:
        consumer_contract = {
            "schema": 1,
            "engine": "vllm",
            "backend": "flashinfer",
            "kind": "framework_reference",
            "exact_demand": True,
            "typed_work_plan": False,
            "native_submission": False,
            "numerical_consumer": True,
            "engine_version": importlib.metadata.version("vllm"),
        }
    median_seconds = statistics.median(samples)
    report = {
        "schema": 1,
        "classification": "vllm-serving-integration-smoke",
        "backend": args.backend,
        "backend_selected": True,
        "native_execution_verified": native_verified,
        "engine": "vllm",
        "engine_version": importlib.metadata.version("vllm"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": str(args.model.resolve()),
        "requests": args.requests,
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
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
