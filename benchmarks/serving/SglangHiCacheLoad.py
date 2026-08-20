#!/usr/bin/env python3
"""Run a placement-proven mixed HiCache load through an in-process SGLang engine."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import pathlib
import random
import time
from typing import Any

from SglangHiCache import (
    configure_environment,
    device_cached_tokens,
    generated_text,
    git_value,
    host_cached_tokens,
    make_prompt,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument("--external-requests", type=int, default=3)
    parser.add_argument("--external-tokens", type=int, default=8192)
    parser.add_argument(
        "--external-suffix-tokens",
        type=int,
        default=0,
        help=(
            "uncached tokens appended to each host-resident prefix so the timed "
            "request executes chunked prefill instead of an exact-prefix decode"
        ),
    )
    parser.add_argument("--resident-requests", type=int, default=1)
    parser.add_argument("--resident-tokens", type=int, default=8192)
    parser.add_argument("--resident-output-tokens", type=int, default=128)
    parser.add_argument("--external-output-tokens", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--churn-tokens", type=int, default=12000)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--context-length", type=int, default=32768)
    # 0 keeps the historical setting (chunk == context length, i.e. a
    # 16K prefill runs as one unchunked forward). Smaller values are
    # the standard decode-protection configuration and apply to both
    # arms identically.
    parser.add_argument("--chunked-prefill-size", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument(
        "--batch-mode",
        choices=("coalesced", "separate"),
        default="coalesced",
        help=(
            "coalesced enables SGLang mixed-chunk batching so resident decode and "
            "external-prefix work share the paged FlashInfer launch; separate is "
            "the scheduler-level ablation"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--allow-oversubscribed-pool",
        action="store_true",
        help=(
            "admit timed contexts whose dense KV exceeds the device pool; "
            "this is the capacity experiment's operating condition — the "
            "dense arm honestly queues and retracts under pressure while "
            "the sidecar arm holds only bounded staging"
        ),
    )
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument(
        "--cuda-graph-decode", choices=("disabled", "full"), default="disabled"
    )
    parser.add_argument(
        "--cuda-graph-prefill",
        choices=("disabled", "breakable"),
        default="disabled",
        help=(
            "prefill-phase CUDA graph backend for BOTH arms; breakable "
            "captures the dense per-layer compute piecewise and leaves "
            "attention (and the tiered staging chain) eager between "
            "pieces, shrinking the extend forward's launch-overhead span"
        ),
    )
    parser.add_argument(
        "--load-warmup-iterations",
        type=int,
        default=2,
        help="performance-excluded mixed arrivals before measurement",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    integer_fields = (
        args.external_requests,
        args.external_tokens,
        args.resident_requests,
        args.resident_tokens,
        args.resident_output_tokens,
        args.external_output_tokens,
        args.churn_tokens,
        args.max_total_tokens,
        args.context_length,
        args.max_running_requests,
    )
    if min(integer_fields) <= 0:
        parser.error("request and token counts must be positive")
    if args.external_suffix_tokens < 0:
        parser.error("external suffix token count cannot be negative")
    if args.load_warmup_iterations < 0:
        parser.error("load warmup iterations cannot be negative")
    if args.request_rate <= 0:
        parser.error("request rate must be positive")
    if args.hicache_ratio <= 1:
        parser.error("HiCache ratio must exceed device cache capacity")
    active_tokens = (
        args.resident_requests * args.resident_tokens
        + args.external_requests
        * (args.external_tokens + args.external_suffix_tokens)
    )
    if (
        active_tokens >= args.max_total_tokens
        and not args.allow_oversubscribed_pool
    ):
        parser.error(
            "all timed resident and external contexts must fit together "
            "(pass --allow-oversubscribed-pool for capacity-pressure runs)"
        )
    if (
        args.external_requests * args.external_tokens + args.churn_tokens
        <= args.max_total_tokens
    ):
        parser.error("external contexts and churn must exceed the device token pool")
    return args


def _meta(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("SGLang omitted request metadata")
    return meta


async def _stream_request(
    engine: Any,
    prompt: str,
    sampling: dict[str, Any],
    *,
    kind: str,
    index: int,
    gate: asyncio.Event | None,
    first_token_event: asyncio.Event | None,
    offset_seconds: float,
) -> dict[str, Any]:
    if gate is not None:
        await gate.wait()
    if offset_seconds:
        await asyncio.sleep(offset_seconds)
    submitted = time.perf_counter()
    stream = await engine.async_generate(
        prompt,
        sampling,
        stream=True,
        rid=f"nta-load-{kind}-{index}",
    )
    first = 0.0
    token_times: list[float] = []
    final: dict[str, Any] | None = None
    async for result in stream:
        now = time.perf_counter()
        if first == 0.0:
            first = now
            if first_token_event is not None:
                first_token_event.set()
        token_times.append(now)
        final = result
    finished = time.perf_counter()
    if final is None or first == 0.0:
        raise RuntimeError(f"SGLang returned no streamed output for {kind}-{index}")
    meta = _meta(final)
    completion_tokens = int(meta.get("completion_tokens", len(token_times)))
    intervals = [
        current - previous for previous, current in zip(token_times, token_times[1:])
    ]
    return {
        "kind": kind,
        "index": index,
        "arrival_offset_seconds": offset_seconds,
        "submitted_seconds": submitted,
        "first_token_seconds": first,
        "finished_seconds": finished,
        "ttft_seconds": first - submitted,
        "e2e_seconds": finished - submitted,
        "tpot_seconds": (
            (finished - first) / (completion_tokens - 1)
            if completion_tokens > 1
            else 0.0
        ),
        "inter_token_seconds": intervals,
        "p99_itl_seconds": _percentile(intervals, 0.99),
        "completion_tokens": completion_tokens,
        "device_cached_tokens": device_cached_tokens(final),
        "host_cached_tokens": host_cached_tokens(final),
        "text": generated_text(final),
    }


async def _run_load(
    engine: Any,
    resident_prompts: list[str],
    external_prompts: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    resident_started = asyncio.Event()
    resident_sampling = {
        "temperature": 0,
        "max_new_tokens": args.resident_output_tokens,
        "ignore_eos": True,
    }
    external_sampling = {
        "temperature": 0,
        "max_new_tokens": args.external_output_tokens,
        "ignore_eos": True,
    }

    async def resident(index: int, prompt: str) -> dict[str, Any]:
        record = await _stream_request(
            engine,
            prompt,
            resident_sampling,
            kind="resident",
            index=index,
            gate=None,
            first_token_event=resident_started,
            offset_seconds=0.0,
        )
        return record

    rng = random.Random(args.seed)
    offsets: list[float] = [0.0] if external_prompts else []
    arrival = 0.0
    for _ in external_prompts[1:]:
        arrival += rng.expovariate(args.request_rate)
        offsets.append(arrival)

    started = time.perf_counter()
    resident_tasks = [
        asyncio.create_task(resident(index, prompt))
        for index, prompt in enumerate(resident_prompts)
    ]
    external_tasks = [
        asyncio.create_task(
            _stream_request(
                engine,
                prompt,
                external_sampling,
                kind="external",
                index=index,
                gate=resident_started,
                first_token_event=None,
                offset_seconds=offsets[index],
            )
        )
        for index, prompt in enumerate(external_prompts)
    ]
    records = await asyncio.gather(*(resident_tasks + external_tasks))
    return records, time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    external_prefixes = [
        make_prompt(tokenizer, f"load-external-{index}", args.external_tokens)
        for index in range(args.external_requests)
    ]
    external_prompts = [
        (
            prefix
            if args.external_suffix_tokens == 0
            else prefix
            + "\n"
            + make_prompt(
                tokenizer,
                f"load-external-suffix-{index}",
                args.external_suffix_tokens,
            )
        )
        for index, prefix in enumerate(external_prefixes)
    ]
    resident_prompts = [
        make_prompt(tokenizer, f"load-resident-{index}", args.resident_tokens)
        for index in range(args.resident_requests)
    ]
    shape_prompt = make_prompt(tokenizer, "load-shape", args.external_tokens)
    eviction_rounds = args.max_total_tokens // args.churn_tokens + 1
    churn_prompts = [
        make_prompt(tokenizer, f"load-churn-{index}", args.churn_tokens)
        for index in range((2 + args.load_warmup_iterations) * eviction_rounds)
    ]
    setup_sampling = {"temperature": 0, "max_new_tokens": 1}

    load_started = time.perf_counter()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        cuda_graph_backend_decode=args.cuda_graph_decode,
        cuda_graph_backend_prefill=args.cuda_graph_prefill,
        chunked_prefill_size=(
            args.chunked_prefill_size
            if args.chunked_prefill_size > 0
            else args.context_length
        ),
        enable_mixed_chunk=args.batch_mode == "coalesced",
        enable_hierarchical_cache=True,
        hicache_ratio=args.hicache_ratio,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        generated_text(engine.generate(shape_prompt, setup_sampling))
        generated_text(engine.generate(shape_prompt, setup_sampling))
        for prompt in external_prefixes:
            generated_text(engine.generate(prompt, setup_sampling))
        for prompt in churn_prompts[:eviction_rounds]:
            generated_text(engine.generate(prompt, setup_sampling))
        external_probe = engine.generate(external_prefixes[0], setup_sampling)
        if host_cached_tokens(external_probe) <= 0:
            raise RuntimeError("external JIT warmup did not load from host cache")
        for prompt in churn_prompts[eviction_rounds:]:
            generated_text(engine.generate(prompt, setup_sampling))
        for prompt in resident_prompts:
            generated_text(engine.generate(prompt, setup_sampling))
            resident_probe = engine.generate(prompt, setup_sampling)
            if device_cached_tokens(resident_probe) <= 0:
                raise RuntimeError("resident warmup did not remain in device cache")

        for warmup in range(args.load_warmup_iterations):
            # Demand graphs warm on the first occurrence and capture on the
            # second. Both are excluded so the measured occurrence is replay.
            engine.loop.run_until_complete(
                _run_load(engine, resident_prompts, external_prompts, args)
            )
            begin = (2 + warmup) * eviction_rounds
            end = begin + eviction_rounds
            for prompt in churn_prompts[begin:end]:
                generated_text(engine.generate(prompt, setup_sampling))
            for prompt in resident_prompts:
                generated_text(engine.generate(prompt, setup_sampling))
                resident_probe = engine.generate(prompt, setup_sampling)
                if device_cached_tokens(resident_probe) <= 0:
                    raise RuntimeError(
                        "resident request was not restored after load warmup"
                    )

        records, elapsed = engine.loop.run_until_complete(
            _run_load(engine, resident_prompts, external_prompts, args)
        )

    external = [record for record in records if record["kind"] == "external"]
    resident = [record for record in records if record["kind"] == "resident"]
    if not all(record["host_cached_tokens"] > 0 for record in external):
        raise RuntimeError("a timed external request was not served from host cache")
    minimum_host_prefix = min(
        record["host_cached_tokens"] for record in external
    )
    if not all(
        record["device_cached_tokens"] > 0 and record["host_cached_tokens"] == 0
        for record in resident
    ):
        raise RuntimeError("a timed resident request was not device-resident")

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: (value["kind"], value["index"])):
        text = record.pop("text").encode("utf-8")
        record["text_sha256"] = hashlib.sha256(text).hexdigest()
        digest.update(text)
        digest.update(b"\0")
    stats = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(workspace.glob("nta-engine.*.json"))
    ]
    total_tokens = sum(record["completion_tokens"] for record in records)
    report = {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": os.environ["NTA_REVISION"],
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "engine_version": importlib.metadata.version("sglang"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "attention_backend": args.attention_backend,
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "request_rate": args.request_rate,
        "external_requests": args.external_requests,
        "external_tokens": args.external_tokens,
        "external_suffix_tokens": args.external_suffix_tokens,
        "minimum_external_host_cached_tokens": minimum_host_prefix,
        "resident_requests": args.resident_requests,
        "resident_tokens": args.resident_tokens,
        "resident_output_tokens": args.resident_output_tokens,
        "external_output_tokens": args.external_output_tokens,
        "eviction_rounds": eviction_rounds,
        "churn_tokens": args.churn_tokens,
        "max_total_tokens": args.max_total_tokens,
        "batch_mode": args.batch_mode,
        "mixed_chunk_enabled": args.batch_mode == "coalesced",
        "chunked_prefill_size": (
            args.chunked_prefill_size
            if args.chunked_prefill_size > 0
            else args.context_length
        ),
        "hicache_ratio": args.hicache_ratio,
        "cuda_graph_decode": args.cuda_graph_decode,
        "cuda_graph_prefill": args.cuda_graph_prefill,
        "load_warmup_iterations": args.load_warmup_iterations,
        "load_warmup_excluded": args.load_warmup_iterations >= 2,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "request_throughput": len(records) / elapsed,
        "output_token_throughput": total_tokens / elapsed,
        "generated_text_sha256": digest.hexdigest(),
        "placement_proven": True,
        "records": records,
        "resident_p95_ttft_seconds": _percentile(
            [record["ttft_seconds"] for record in resident], 0.95
        ),
        "resident_p95_tpot_seconds": _percentile(
            [record["tpot_seconds"] for record in resident], 0.95
        ),
        "resident_p99_itl_seconds": _percentile(
            [
                interval
                for record in resident
                for interval in record["inter_token_seconds"]
            ],
            0.99,
        ),
        "external_p95_ttft_seconds": _percentile(
            [record["ttft_seconds"] for record in external], 0.95
        ),
        "engine_stats": stats,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
