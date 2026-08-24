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
import platform
import random
import subprocess
import sys
import time
from typing import Any

try:
    from experiments.bailian import demand_trace_digest, read_jsonl
    from experiments.queueing import finite_window_littles_law
    from experiments.validate_workload import validate as validate_workload
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from experiments.bailian import demand_trace_digest, read_jsonl
    from experiments.queueing import finite_window_littles_law
    from experiments.validate_workload import validate as validate_workload

from SglangHiCache import (
    configure_environment,
    device_cached_tokens,
    generated_text,
    git_value,
    host_cached_tokens,
    make_prompt,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _machine_metadata() -> dict[str, Any]:
    def command(argv: list[str]) -> str | None:
        try:
            result = subprocess.run(
                argv,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
    }


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
        "--workload-manifest",
        type=pathlib.Path,
        help=(
            "normalized Bailian workload manifest; the same manifest is used "
            "for both stock and NTA arms"
        ),
    )
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
            "attention and exact staging remain eager between "
            "pieces, shrinking the extend forward's launch-overhead span"
        ),
    )
    parser.add_argument(
        "--load-warmup-iterations",
        type=int,
        default=2,
        help="performance-excluded mixed arrivals before measurement",
    )
    parser.add_argument("--slo-ttft-seconds", type=float, default=8.0)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
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
    if args.slo_ttft_seconds <= 0 or args.slo_p99_itl_seconds <= 0:
        parser.error("SLO thresholds must be positive")
    if args.request_rate <= 0:
        parser.error("request rate must be positive")
    if args.hicache_ratio <= 1:
        parser.error("HiCache ratio must exceed device cache capacity")
    if args.workload_manifest is None:
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


def _tokenized_structure_prompt(tokenizer: Any, row: dict[str, Any]) -> tuple[str, int]:
    block_size = int(row.get("block_size", 16))
    token_count = int(row["input_length"])
    block_count = (token_count + block_size - 1) // block_size
    block_ids = list(row.get("hash_ids", ()))
    block_ids.extend(
        f"{row['request_id']}:unique:{index}"
        for index in range(len(block_ids), block_count)
    )
    token_ids: list[int] = []
    for block_id in block_ids:
        seed = f"nta-bailian-block-{block_id} "
        block_text = seed
        while len(tokenizer.encode(block_text, add_special_tokens=False)) < block_size:
            block_text += seed
        block_tokens = tokenizer.encode(block_text, add_special_tokens=False)[:block_size]
        token_ids.extend(block_tokens)
    token_ids = token_ids[:token_count]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, measured


def _load_workload(
    path: pathlib.Path, tokenizer: Any
) -> tuple[
    list[str], list[str], list[str], list[str], list[float], list[int], list[int], dict[str, Any]
]:
    manifest = validate_workload(path.resolve())
    records_path = path.resolve().parent / str(manifest["records_file"])
    rows = read_jsonl(records_path)
    if not rows:
        raise RuntimeError("normalized workload contains no requests")
    computed_demand_digest = demand_trace_digest(rows)
    if computed_demand_digest != manifest["demand_trace_digest"]:
        raise RuntimeError("normalized workload demand digest does not match its records")
    explicit_states = [row.get("request_state") for row in rows]
    if any(state is not None for state in explicit_states):
        resident_rows = [row for row in rows if row.get("request_state") == "resident"]
        external_rows = [row for row in rows if row.get("request_state") != "resident"]
    else:
        # A structure-only manifest without an application state annotation is
        # still usable, but this deterministic split is recorded as harness
        # policy rather than mistaken for a production label.
        resident_rows = rows[:1]
        external_rows = rows[1:]
    if not resident_rows or not external_rows:
        raise RuntimeError(
            "serving replay needs at least one resident and one external request; "
            "add request_state to the normalized workload"
        )

    tokenization_errors = 0
    structure_only = not bool(manifest["prompt"].get("semantic_representativeness_claim"))

    def prompt(row: dict[str, Any]) -> str:
        nonlocal tokenization_errors
        value = row.get("prompt_text")
        if structure_only or value is None:
            value, measured = _tokenized_structure_prompt(tokenizer, row)
            if measured != int(row["input_length"]):
                tokenization_errors += 1
            return value
        return str(value)

    external_offsets = [float(row["arrival_seconds"]) for row in external_rows]
    origin = min(external_offsets)
    external_offsets = [offset - origin for offset in external_offsets]
    request_arrival_offsets = {
        str(row["request_id"]): 0.0 for row in resident_rows
    }
    request_arrival_offsets.update(
        {
            str(row["request_id"]): offset
            for row, offset in zip(external_rows, external_offsets)
        }
    )
    metadata = {
        "manifest": str(path.resolve()),
        "manifest_digest": hashlib.sha256(path.resolve().read_bytes()).hexdigest(),
        "records_digest": str(manifest["records_digest"]),
        "demand_trace_digest": str(manifest["demand_trace_digest"]),
        "arrival": manifest["arrival"],
        "prompt": manifest["prompt"],
        "state_mapping": "explicit_request_state" if any(state is not None for state in explicit_states) else "first_row_resident_fallback",
        "request_count": len(rows),
        "request_id_order": [str(row["request_id"]) for row in rows],
        "request_arrival_offsets": request_arrival_offsets,
        "tokenization_errors": tokenization_errors,
        "resident_input_tokens": [int(row["input_length"]) for row in resident_rows],
        "external_input_tokens": [int(row["input_length"]) for row in external_rows],
        "resident_output_tokens": [int(row["output_length"]) for row in resident_rows],
        "external_output_tokens": [int(row["output_length"]) for row in external_rows],
    }
    return (
        [str(row["request_id"]) for row in resident_rows],
        [str(row["request_id"]) for row in external_rows],
        [prompt(row) for row in resident_rows],
        [prompt(row) for row in external_rows],
        external_offsets,
        metadata["resident_output_tokens"],
        metadata["external_output_tokens"],
        metadata,
    )


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
    request_id: str | None,
    gate: asyncio.Event | None,
    first_token_event: asyncio.Event | None,
    offset_seconds: float,
    load_start_seconds: float,
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
        rid=request_id or f"nta-load-{kind}-{index}",
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
        "request_id": request_id or f"nta-load-{kind}-{index}",
        "arrival_offset_seconds": offset_seconds,
        "arrival_seconds": offset_seconds,
        "submitted_offset_seconds": submitted - load_start_seconds,
        "first_token_offset_seconds": first - load_start_seconds,
        "finished_offset_seconds": finished - load_start_seconds,
        "submitted_seconds": submitted,
        "first_token_seconds": first,
        "finished_seconds": finished,
        "ttft_seconds": first - submitted,
        "e2e_seconds": finished - submitted,
        "admission_delay_seconds": max(
            0.0, submitted - (load_start_seconds + offset_seconds)
        ),
        "system_time_seconds": max(0.0, finished - (load_start_seconds + offset_seconds)),
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
    external_offsets: list[float] | None = None,
    resident_output_tokens: list[int] | None = None,
    external_output_tokens: list[int] | None = None,
    resident_request_ids: list[str] | None = None,
    external_request_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
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
        sampling = dict(resident_sampling)
        if resident_output_tokens is not None:
            sampling["max_new_tokens"] = max(1, resident_output_tokens[index])
        record = await _stream_request(
            engine,
            prompt,
            sampling,
            kind="resident",
            index=index,
            request_id=(resident_request_ids[index] if resident_request_ids is not None else None),
            gate=None,
            first_token_event=resident_started,
            offset_seconds=0.0,
            load_start_seconds=started,
        )
        return record

    if external_offsets is not None:
        if len(external_offsets) != len(external_prompts):
            raise RuntimeError("workload arrival count does not match external prompts")
        offsets = external_offsets
    else:
        rng = random.Random(args.seed)
        offsets = [0.0] if external_prompts else []
        arrival = 0.0
        for _ in external_prompts[1:]:
            arrival += rng.expovariate(args.request_rate)
            offsets.append(arrival)

    resident_tasks = [
        asyncio.create_task(resident(index, prompt))
        for index, prompt in enumerate(resident_prompts)
    ]
    external_tasks = [
        asyncio.create_task(
            _stream_request(
                engine,
                prompt,
                {
                    **external_sampling,
                    "max_new_tokens": max(
                        1,
                        external_output_tokens[index]
                        if external_output_tokens is not None
                        else args.external_output_tokens,
                    ),
                },
                kind="external",
                index=index,
                request_id=(external_request_ids[index] if external_request_ids is not None else None),
                gate=None if external_offsets is not None else resident_started,
                first_token_event=None,
                offset_seconds=offsets[index],
                load_start_seconds=started,
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


def _latency_percentiles(
    records: list[dict[str, Any]], field: str
) -> dict[str, float]:
    values = [float(record[field]) for record in records]
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _itl_values(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(interval)
        for record in records
        for interval in record["inter_token_seconds"]
    ] or [0.0]


def _slo_goodput(
    records: list[dict[str, Any]],
    elapsed: float,
    *,
    ttft_seconds: float,
    p99_itl_seconds: float,
) -> dict[str, Any]:
    qualified = sum(
        float(record["ttft_seconds"]) <= ttft_seconds
        and float(record["p99_itl_seconds"]) <= p99_itl_seconds
        for record in records
    )
    return {
        "qualified_requests": qualified,
        "total_requests": len(records),
        "attainment": qualified / len(records),
        "goodput_requests_per_second": qualified / elapsed,
        "thresholds_seconds": {
            "ttft": ttft_seconds,
            "p99_itl": p99_itl_seconds,
        },
    }


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    import sglang as sgl
    from transformers import AutoTokenizer

    workload_metadata: dict[str, Any] | None = None
    external_offsets: list[float] | None = None
    resident_output_tokens: list[int] | None = None
    external_output_tokens: list[int] | None = None
    resident_request_ids: list[str] | None = None
    external_request_ids: list[str] | None = None
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    if args.workload_manifest is not None:
        (
            resident_request_ids,
            external_request_ids,
            resident_prompts,
            external_prompts,
            external_offsets,
            resident_output_tokens,
            external_output_tokens,
            workload_metadata,
        ) = _load_workload(args.workload_manifest, tokenizer)
        if workload_metadata["tokenization_errors"]:
            raise RuntimeError(
                "Bailian structure prompt could not preserve exact tokenizer lengths; "
                "use a tokenizer-compatible prompt adapter before claiming serving evidence"
            )
        args.resident_requests = len(resident_prompts)
        args.external_requests = len(external_prompts)
        args.resident_tokens = max(workload_metadata["resident_input_tokens"])
        args.external_tokens = max(workload_metadata["external_input_tokens"])
        shape_prompt = external_prompts[0]
        external_prefixes = external_prompts
    else:
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
                _run_load(
                    engine,
                    resident_prompts,
                    external_prompts,
                    args,
                    external_offsets,
                    resident_output_tokens,
                    external_output_tokens,
                    resident_request_ids,
                    external_request_ids,
                )
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
            _run_load(
                engine,
                resident_prompts,
                external_prompts,
                args,
                external_offsets,
                resident_output_tokens,
                external_output_tokens,
                resident_request_ids,
                external_request_ids,
            )
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
    selected_tokens = sum(
        record["host_cached_tokens"] + record["device_cached_tokens"]
        for record in records
    )
    physical_tokens = sum(
        record["host_cached_tokens"] + record["device_cached_tokens"]
        for record in records
    )
    admission_delays = [record["admission_delay_seconds"] for record in records]
    littles_law = finite_window_littles_law(records, elapsed)
    ttft = _latency_percentiles(records, "ttft_seconds")
    tpot = _latency_percentiles(records, "tpot_seconds")
    itl_values = _itl_values(records)
    itl = {
        "p50": _percentile(itl_values, 0.50),
        "p95": _percentile(itl_values, 0.95),
        "p99": _percentile(itl_values, 0.99),
    }
    slo_goodput = _slo_goodput(
        records,
        elapsed,
        ttft_seconds=args.slo_ttft_seconds,
        p99_itl_seconds=args.slo_p99_itl_seconds,
    )
    correctness = {
        "verification_failures": 0,
        "placement_proven": True,
        "generated_text_sha256": digest.hexdigest(),
        "demand_trace_digest": (
            workload_metadata["demand_trace_digest"]
            if workload_metadata is not None
            else None
        ),
    }
    report = {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": os.environ.get("NTA_REVISION") or git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "machine": _machine_metadata(),
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
        "workload": workload_metadata,
        "demand_semantics": "exact",
        "demand_trace_digest": (
            workload_metadata["demand_trace_digest"]
            if workload_metadata is not None
            else None
        ),
        "selected_bytes": None,
        "physical_bytes": None,
        "byte_accounting_status": "not exposed by SGLang engine metadata",
        "selected_kv_tokens": selected_tokens,
        "physical_kv_tokens": physical_tokens,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "request_throughput": len(records) / elapsed,
        "output_token_throughput": total_tokens / elapsed,
        "p50_ttft_seconds": ttft["p50"],
        "p95_ttft_seconds": ttft["p95"],
        "p99_ttft_seconds": ttft["p99"],
        "p50_tpot_seconds": tpot["p50"],
        "p95_tpot_seconds": tpot["p95"],
        "p99_tpot_seconds": tpot["p99"],
        "p99_itl_seconds": itl["p99"],
        "latency_percentiles": {
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "inter_token_seconds": itl,
        },
        "slo_goodput": slo_goodput,
        "generated_text_sha256": digest.hexdigest(),
        "placement_proven": True,
        "verification_failures": 0,
        "correctness": correctness,
        "littles_law": littles_law,
        "admission_delay_seconds": {
            "mean": sum(admission_delays) / len(admission_delays),
            "p95": _percentile(admission_delays, 0.95),
            "scope": "client_admission_delay",
        },
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
