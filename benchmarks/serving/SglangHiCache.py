#!/usr/bin/env python3
"""Measure repeated CPU-DRAM KV promotion through SGLang HiCache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import statistics
import subprocess
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
MIN_HICACHE_PREFIX_TOKENS = 64


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="fail unless this many attempts yield the requested promotions",
    )
    parser.add_argument("--hot-tokens", type=int, default=160)
    parser.add_argument("--hot-requests", type=int, default=1)
    parser.add_argument("--churn-tokens", type=int, default=240)
    parser.add_argument("--resident-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=320)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=4.0)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-home", type=pathlib.Path)
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    parser.add_argument(
        "--cuda-graph-decode",
        choices=("disabled", "full"),
        default="disabled",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if (
        min(
            args.iterations,
            args.hot_tokens,
            args.hot_requests,
            args.churn_tokens,
            args.max_total_tokens,
            args.context_length,
        )
        <= 0
    ):
        parser.error("token counts and iterations must be positive")
    if args.resident_tokens < 0:
        parser.error("resident token count cannot be negative")
    if args.hot_tokens < MIN_HICACHE_PREFIX_TOKENS:
        parser.error(
            "hot token count must be at least "
            f"{MIN_HICACHE_PREFIX_TOKENS} to form a promotable HiCache prefix"
        )
    if args.hicache_ratio <= 1.0:
        parser.error("HiCache ratio must exceed the device-cache size")
    if args.max_attempts is None:
        args.max_attempts = 4 * args.iterations
    if args.max_attempts < args.iterations:
        parser.error("max attempts must be at least the requested iterations")
    if args.hot_requests * args.hot_tokens + args.churn_tokens <= args.max_total_tokens:
        parser.error(
            "hot request set and churn prompt must exceed the device token pool together"
        )
    return args


def configure_environment(args: argparse.Namespace) -> pathlib.Path:
    from cuda_environment import configure_jit_environment

    _, _, workspace = configure_jit_environment(
        root=ROOT,
        workspace=args.flashinfer_workspace_base,
        host_cxx=getattr(args, "cuda_host_cxx", None),
        cuda_home=getattr(args, "cuda_home", None),
        revision=os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
    )
    return workspace


def make_prompt(tokenizer: Any, label: str, token_count: int) -> str:
    seed = f"{label}: finite GPU kernels acquire cache pages by request identity. "
    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < token_count:
        text += seed
    ids = tokenizer.encode(text, add_special_tokens=False)[:token_count]
    return tokenizer.decode(ids, skip_special_tokens=True)


def generated_text(result: Any) -> str:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError("SGLang returned an invalid generation result")
    return result["text"]


def generation_results(result: Any) -> list[dict[str, Any]]:
    values = result if isinstance(result, list) else [result]
    for value in values:
        generated_text(value)
    return values


def host_cached_tokens(result: dict[str, Any]) -> int:
    details = result.get("meta_info", {}).get("cached_tokens_details", {})
    if not isinstance(details, dict):
        return 0
    return int(details.get("host", 0))


def device_cached_tokens(result: dict[str, Any]) -> int:
    details = result.get("meta_info", {}).get("cached_tokens_details", {})
    if not isinstance(details, dict):
        return 0
    return int(details.get("device", 0))


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    hot = [
        make_prompt(tokenizer, f"hot-prefix-{index}", args.hot_tokens)
        for index in range(args.hot_requests)
    ]
    shape_warmup = [
        make_prompt(tokenizer, f"shape-warmup-hot-{index}", args.hot_tokens)
        for index in range(args.hot_requests)
    ]
    if args.resident_tokens:
        shape_warmup.append(
            make_prompt(tokenizer, "shape-warmup-resident", args.resident_tokens)
        )
    eviction_rounds = args.max_total_tokens // args.churn_tokens + 1
    churn = [
        make_prompt(tokenizer, f"eviction-{attempt}", args.churn_tokens)
        for attempt in range(eviction_rounds * (args.max_attempts + 1))
    ]
    resident = (
        [
            make_prompt(tokenizer, f"resident-{attempt}", args.resident_tokens)
            for attempt in range(args.max_attempts)
        ]
        if args.resident_tokens
        else []
    )
    sampling = {"temperature": 0, "max_new_tokens": 1}
    load_started = time.perf_counter()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=8,
        cuda_graph_backend_decode=args.cuda_graph_decode,
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=args.context_length,
        enable_hierarchical_cache=True,
        hicache_ratio=args.hicache_ratio,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        generation_results(engine.generate(shape_warmup, sampling))
        # Warm the exact two-cache-hit attention shape outside the timed region.
        # Placement does not affect FlashInfer specialization; the measured arm
        # below separately proves host versus device residency from SGLang's
        # response metadata.
        generation_results(engine.generate(shape_warmup, sampling))
        generation_results(engine.generate(hot, sampling))
        churn_cursor = 0
        for _ in range(eviction_rounds):
            generated_text(engine.generate(churn[churn_cursor], sampling))
            churn_cursor += 1
        samples: list[float] = []
        metadata: list[Any] = []
        attempt_seconds: list[float] = []
        attempt_metadata: list[Any] = []
        external_attempt_indices: list[int] = []
        generated_samples: list[list[str]] = []
        digest = hashlib.sha256()
        for attempt in range(args.max_attempts):
            if resident:
                # Make the peer a device-cache hit, rather than an uncached
                # prefill that SGLang can place in a different forward batch.
                # This setup is deliberately excluded from the measured call.
                generated_text(engine.generate(resident[attempt], sampling))
            started = time.perf_counter()
            prompts = hot + ([resident[attempt]] if resident else [])
            result = engine.generate(prompts, sampling)
            elapsed = time.perf_counter() - started
            values = generation_results(result)
            texts = [generated_text(value) for value in values]
            result_metadata = [value.get("meta_info", {}) for value in values]
            attempt_seconds.append(elapsed)
            attempt_metadata.append(result_metadata)
            hot_is_external = all(
                host_cached_tokens(value) > 0 for value in values[: args.hot_requests]
            )
            peer_is_resident = not resident or (
                device_cached_tokens(values[args.hot_requests]) > 0
                and host_cached_tokens(values[args.hot_requests]) == 0
            )
            if hot_is_external and peer_is_resident:
                samples.append(elapsed)
                metadata.append(result_metadata)
                external_attempt_indices.append(attempt)
                generated_samples.append(texts)
                for text in texts:
                    digest.update(text.encode("utf-8"))
                    digest.update(b"\0")
            for _ in range(eviction_rounds):
                generated_text(engine.generate(churn[churn_cursor], sampling))
                churn_cursor += 1
            if len(samples) == args.iterations:
                break
        if len(samples) != args.iterations:
            raise RuntimeError(
                f"only {len(samples)} of {args.max_attempts} attempts loaded the "
                f"hot prefix from host memory; requested {args.iterations}"
            )

    stats = []
    for path in sorted(workspace.glob("nta-engine.*.json")):
        stats.append(json.loads(path.read_text(encoding="utf-8")))
    median = statistics.median(samples)
    hot_request_seconds = [
        max(float(value["e2e_latency"]) for value in values[: args.hot_requests])
        for values in metadata
        if len(values) >= args.hot_requests
        and all(
            isinstance(value.get("e2e_latency"), (int, float))
            for value in values[: args.hot_requests]
        )
    ]
    peer_request_seconds = [
        float(values[args.hot_requests]["e2e_latency"])
        for values in metadata
        if len(values) > args.hot_requests
        and isinstance(values[args.hot_requests].get("e2e_latency"), (int, float))
    ]
    peer_delay_seconds = [
        max(0.0, peer - hot)
        for hot, peer in zip(hot_request_seconds, peer_request_seconds)
    ]
    report = {
        "schema": 1,
        "classification": "sglang-hicache-promotion",
        "revision": os.environ["NTA_REVISION"],
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "engine_version": importlib.metadata.version("sglang"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_cuda_version": torch.version.cuda,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
        "attention_backend": args.attention_backend,
        "cuda_graph_decode": args.cuda_graph_decode,
        "model": str(args.model.resolve()),
        "hot_tokens": args.hot_tokens,
        "hot_requests": args.hot_requests,
        "churn_tokens": args.churn_tokens,
        "eviction_rounds": eviction_rounds,
        "eviction_tokens_per_attempt": eviction_rounds * args.churn_tokens,
        "hicache_ratio": args.hicache_ratio,
        "resident_tokens": args.resident_tokens,
        "batch_width": args.hot_requests + (1 if resident else 0),
        "max_total_tokens": args.max_total_tokens,
        "iterations": args.iterations,
        "attempts": len(attempt_seconds),
        "max_attempts": args.max_attempts,
        "external_attempt_indices": external_attempt_indices,
        "load_seconds": load_seconds,
        "shape_warmup_excluded": True,
        "resident_setup_excluded": bool(resident),
        "placement_proof_required": bool(resident),
        "promotion_seconds_samples": samples,
        "attempt_seconds_samples": attempt_seconds,
        "median_promotion_seconds": median,
        "promotions_per_second": 1.0 / median,
        "completed_requests_per_second": (args.hot_requests + (1 if resident else 0))
        / median,
        "generated_text_sha256": digest.hexdigest(),
        "generated_text_samples": generated_samples,
        "result_metadata": metadata,
        "attempt_metadata": attempt_metadata,
        "engine_stats": stats,
    }
    if len(hot_request_seconds) == len(samples):
        report["hot_request_seconds_samples"] = hot_request_seconds
        report["median_hot_request_seconds"] = statistics.median(hot_request_seconds)
    if len(peer_request_seconds) == len(samples):
        report["peer_request_seconds_samples"] = peer_request_seconds
        report["peer_delay_seconds_samples"] = peer_delay_seconds
        report["median_peer_request_seconds"] = statistics.median(peer_request_seconds)
        report["median_peer_delay_seconds"] = statistics.median(peer_delay_seconds)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
