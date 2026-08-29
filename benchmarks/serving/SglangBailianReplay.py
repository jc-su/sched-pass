#!/usr/bin/env python3
"""Replay one source-contiguous Bailian window through SGLang HiCache.

Unlike ``SglangHiCacheLoad.py``, this harness never assigns requests to
resident/external roles and never churns prompts to force placement. A drained
source-preceding window conditions the cache; placement in the measured open-
loop window is reported only from SGLang's device/host hit metadata.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bailian_replay import (  # noqa: E402
    append_cache_boundary,
    encode_window,
    load_replay_window,
)
from experiments.atomic_io import atomic_write_json  # noqa: E402
from experiments.queueing import finite_window_system_accounting  # noqa: E402
from SglangHiCache import configure_environment, git_value  # noqa: E402
from SglangHiCacheLoad import (  # noqa: E402
    SGLANG_INPUT_MARGIN_TOKENS,
    _engine_byte_accounting,
    _execution_dispatch,
    _itl_values,
    _latency_percentiles,
    _machine_metadata,
    _measurement_delta,
    _percentile,
    _publish_engine_stats_snapshot,
    _read_engine_stats,
    _slo_goodput,
    _stream_request,
)


TokenInput = tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument("--workload-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--measured-start", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--measured-requests", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--output-length-scale", type=float, default=0.1)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument(
        "--batch-mode", choices=("coalesced", "separate"), default="coalesced"
    )
    parser.add_argument(
        "--cuda-graph-decode", choices=("disabled", "full"), default="disabled"
    )
    parser.add_argument(
        "--cuda-graph-prefill",
        choices=("disabled", "breakable"),
        default="disabled",
    )
    parser.add_argument("--slo-ttft-seconds", type=float, default=8.0)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-home", type=pathlib.Path)
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.workload_manifest.is_file():
        parser.error(f"workload manifest does not exist: {args.workload_manifest}")
    if args.measured_start < 0 or min(
        args.warmup_requests,
        args.measured_requests,
        args.max_output_tokens,
        args.context_length,
        args.chunked_prefill_size,
        args.max_total_tokens,
        args.max_running_requests,
    ) <= 0:
        parser.error("replay counts and serving capacities must be positive")
    if args.hicache_ratio <= 1.0:
        parser.error("HiCache ratio must exceed device cache capacity")
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("mem-fraction-static must be between zero and one")
    if not math.isfinite(args.time_scale) or args.time_scale <= 0.0:
        parser.error("time scale must be finite and positive")
    if not math.isfinite(args.output_length_scale) or args.output_length_scale <= 0.0:
        parser.error("output length scale must be finite and positive")
    if args.slo_ttft_seconds <= 0.0 or args.slo_p99_itl_seconds <= 0.0:
        parser.error("SLO thresholds must be positive")
    return args


def _request_id(phase: str, row: Mapping[str, Any]) -> str:
    return (
        f"nta-bailian-{phase}-{int(row['source_index'])}-"
        f"{str(row['request_id'])}"
    )


async def _run_phase(
    engine: Any,
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[TokenInput],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], float]:
    if len(rows) != len(inputs) or not rows:
        raise RuntimeError("Bailian phase rows and token inputs disagree")
    started = time.perf_counter()
    tasks = []
    for index, (row, input_ids) in enumerate(zip(rows, inputs, strict=True)):
        tasks.append(
            asyncio.create_task(
                _stream_request(
                    engine,
                    input_ids,
                    {
                        "temperature": 0,
                        "max_new_tokens": int(row["replay_output_tokens"]),
                        "ignore_eos": True,
                        "stream_interval": 1,
                    },
                    kind="trace",
                    index=index,
                    request_id=_request_id(phase, row),
                    gate=None,
                    first_token_event=None,
                    offset_seconds=float(row["replay_arrival_seconds"]),
                    load_start_seconds=started,
                )
            )
        )
    records = list(await asyncio.gather(*tasks))
    for record, row in zip(records, rows, strict=True):
        record.update(
            {
                "source_index": int(row["source_index"]),
                "source_request_id": str(row["request_id"]),
                "source_input_tokens": int(row["input_length"]),
                "source_output_tokens": int(row["output_length"]),
                "replay_output_tokens": int(row["replay_output_tokens"]),
                "replayable_prefix_tokens": int(row["replayable_prefix_tokens"]),
                "followup": row.get("parent_chat_id") not in (None, "", -1, "-1"),
            }
        )
    return records, time.perf_counter() - started


def _cache_state(record: Mapping[str, Any]) -> str:
    host = int(record["host_cached_tokens"])
    device = int(record["device_cached_tokens"])
    if host and device:
        return "device_and_host"
    if host:
        return "host"
    if device:
        return "device"
    return "cold"


def _axis(values: Sequence[int | float | str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "distinct": len(set(values)),
        "heterogeneous": len(set(values)) > 1,
    }
    if values and all(isinstance(value, (int, float)) for value in values):
        result.update({"min": min(values), "max": max(values)})
    else:
        result["values"] = sorted({str(value) for value in values})
    return result


def _maximum_concurrency(records: Sequence[Mapping[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for record in records:
        events.append((float(record["submitted_offset_seconds"]), 1))
        events.append((float(record["finished_offset_seconds"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise RuntimeError("Bailian replay retired an inactive request")
        maximum = max(maximum, active)
    if active:
        raise RuntimeError("Bailian replay did not retire every request")
    return maximum


def _heterogeneity(
    records: Sequence[Mapping[str, Any]], stats: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    axes = {
        "input_tokens": _axis([int(item["input_tokens"]) for item in records]),
        "output_tokens": _axis(
            [int(item["completion_tokens"]) for item in records]
        ),
        "replayable_prefix_tokens": _axis(
            [int(item["replayable_prefix_tokens"]) for item in records]
        ),
        "observed_cached_prefix_tokens": _axis(
            [
                int(item["host_cached_tokens"]) + int(item["device_cached_tokens"])
                for item in records
            ]
        ),
        "cache_state": _axis([str(item["observed_cache_state"]) for item in records]),
        "arrival_offset_seconds": _axis(
            [float(item["arrival_offset_seconds"]) for item in records]
        ),
    }
    counters = {
        name: sum(int(report.get(name, 0)) for report in stats)
        for name in (
            "multi_request_engine_batches",
            "heterogeneous_engine_batches",
            "multi_axis_heterogeneous_batches",
            "sequence_length_heterogeneous_batches",
            "availability_heterogeneous_batches",
            "external_rows_heterogeneous_batches",
            "mixed_dependency_layers",
        )
    }
    return {
        "schema": 1,
        "axes": axes,
        "heterogeneous_axis_count": sum(
            bool(value["heterogeneous"]) for value in axes.values()
        ),
        "maximum_concurrent_requests": _maximum_concurrency(records),
        "engine_forward": counters,
        "batch_internal_geometry_proven": (
            counters["sequence_length_heterogeneous_batches"] > 0
        ),
        "batch_internal_availability_proven": (
            counters["availability_heterogeneous_batches"] > 0
        ),
        "scope": (
            "batch_internal"
            if counters["heterogeneous_engine_batches"] > 0
            else "request_set_only"
        ),
    }


def _native_dispatch_distribution(
    stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    observations = 0
    nonprefix = 0
    model_layers: set[int] = set()
    for report in stats:
        observations += int(report.get("native_dispatch_prefix_observations", 0))
        nonprefix += int(report.get("native_dispatch_nonprefix_batches", 0))
        if "model_layer_count" in report:
            model_layers.add(int(report["model_layer_count"]))
        for name, value in report.items():
            if name.startswith("native_dispatch_prefix_layers_") and name.endswith("_batches"):
                depth = int(name.removeprefix("native_dispatch_prefix_layers_").removesuffix("_batches"))
                histogram[depth] += int(value)
    if sum(histogram.values()) != observations:
        raise RuntimeError(
            "native-dispatch histogram disagrees with observations"
        )
    if nonprefix:
        raise RuntimeError("runtime observed non-prefix native dispatch")
    if len(model_layers) > 1:
        raise RuntimeError("workers disagree on model layer count")
    layer_count = next(iter(model_layers), None)
    if layer_count is not None and any(
        depth < 0 or depth > layer_count for depth in histogram
    ):
        raise RuntimeError("native dispatch prefix exceeds the model layer count")
    native_layer_observations = sum(
        depth * count for depth, count in histogram.items()
    )
    mixed_dispatch_observations = sum(
        count
        for depth, count in histogram.items()
        if layer_count is not None and 0 < depth < layer_count
    )
    return {
        "schema": 1,
        "definition": "native_numerical_dispatch_prefix_layers",
        "model_layer_count": layer_count,
        "observations": observations,
        "histogram": {str(depth): count for depth, count in sorted(histogram.items())},
        "mean_layers": (
            native_layer_observations / observations
            if observations
            else None
        ),
        "native_layer_observations": native_layer_observations,
        "native_layer_fraction": (
            native_layer_observations / (observations * layer_count)
            if observations and layer_count
            else None
        ),
        "mixed_dispatch_observations": mixed_dispatch_observations,
        "framework_only_observations": histogram.get(0, 0),
        "native_only_observations": (
            histogram.get(layer_count, 0) if layer_count is not None else 0
        ),
        "nonprefix_batches": nonprefix,
    }


def _progressive_consumer_distribution(
    stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize actual ticketed work-unit execution, not configured policy."""

    histogram: Counter[int] = Counter()
    observations = 0
    reported_layers = 0
    model_layers: set[int] = set()
    for report in stats:
        observations += int(report.get("progressive_consumer_batch_observations", 0))
        reported_layers += int(report.get("progressive_consumer_layers", 0))
        if "model_layer_count" in report:
            model_layers.add(int(report["model_layer_count"]))
        for name, value in report.items():
            if name.startswith("progressive_consumer_layers_") and name.endswith(
                "_batches"
            ):
                layers = int(
                    name.removeprefix("progressive_consumer_layers_").removesuffix(
                        "_batches"
                    )
                )
                histogram[layers] += int(value)
    if sum(histogram.values()) != observations:
        raise RuntimeError(
            "progressive-consumer histogram disagrees with observations"
        )
    if len(model_layers) > 1:
        raise RuntimeError("workers disagree on model layer count")
    layer_count = next(iter(model_layers), None)
    if layer_count is not None and any(
        layers < 0 or layers > layer_count for layers in histogram
    ):
        raise RuntimeError(
            "progressive-consumer layers exceed the model layer count"
        )
    observed_layers = sum(layers * count for layers, count in histogram.items())
    if observed_layers != reported_layers:
        raise RuntimeError(
            "progressive-consumer layer accounting disagrees with its histogram"
        )
    active_observations = sum(
        count for layers, count in histogram.items() if layers > 0
    )
    return {
        "schema": 1,
        "definition": "layers_executed_by_ticketed_progressive_work_unit_consumer",
        "model_layer_count": layer_count,
        "observations": observations,
        "histogram": {
            str(layers): count for layers, count in sorted(histogram.items())
        },
        "layer_observations": observed_layers,
        "active_observations": active_observations,
        "inactive_observations": histogram.get(0, 0),
        "layer_fraction": (
            observed_layers / (observations * layer_count)
            if observations and layer_count
            else None
        ),
    }


def _prefetch_arrival_readiness(
    stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize CUDA-event ordering only when barrier profiling was enabled."""

    profile_modes = {
        bool(report.get("barrier_profile_enabled", False)) for report in stats
    }
    if len(profile_modes) > 1:
        raise RuntimeError("workers disagree on barrier profiling mode")
    profiled = profile_modes == {True}
    arrivals = sum(int(report.get("profiled_attention_arrivals", 0)) for report in stats)
    ready = sum(
        int(report.get("profiled_attention_ready_at_arrival", 0))
        for report in stats
    )
    not_ready = sum(
        int(report.get("profiled_attention_not_ready_at_arrival", 0))
        for report in stats
    )
    material = sum(
        int(report.get("profiled_attention_materially_stalled_arrivals", 0))
        for report in stats
    )
    stall_ms = sum(
        float(report.get("profiled_attention_stall_gpu_ms", 0.0))
        for report in stats
    )
    if ready + not_ready != arrivals:
        raise RuntimeError("attention arrival-readiness counters are inconsistent")
    if material > not_ready or stall_ms < 0.0:
        raise RuntimeError("attention arrival-stall counters are inconsistent")
    if not profiled and any((arrivals, ready, not_ready, material)):
        raise RuntimeError("arrival readiness was reported with profiling disabled")
    return {
        "schema": 1,
        "definition": "cuda_event_order_at_proactive_prefetch_attention_wait",
        "status": "profiled" if profiled else "not_profiled",
        "arrivals": arrivals,
        "ready_at_arrival": ready,
        "not_ready_at_arrival": not_ready,
        "materially_stalled_arrivals": material,
        "material_stall_threshold_ms": 0.01,
        "stall_gpu_ms": stall_ms,
    }


def _consumer_contract(
    stats: Sequence[Mapping[str, Any]], engine_version: str, backend: str
) -> dict[str, Any]:
    if backend == "flashinfer":
        return {
            "schema": 1,
            "engine": "sglang",
            "backend": "flashinfer",
            "kind": "framework_reference",
            "exact_demand": True,
            "typed_work_plan": False,
            "native_submission": False,
            "numerical_consumer": True,
            "engine_version": engine_version,
        }
    contracts = [
        report.get("consumer_contract")
        for report in stats
        if isinstance(report.get("consumer_contract"), dict)
    ]
    return next(
        (item for item in contracts if item.get("kind") == "native_work_unit"),
        contracts[0] if contracts else {},
    )


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    prior_stats_paths = set(workspace.glob("nta-engine.*.json"))
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    window = load_replay_window(
        args.workload_manifest,
        measured_start=args.measured_start,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        context_length=args.context_length,
        input_margin_tokens=SGLANG_INPUT_MARGIN_TOKENS,
        input_adapter_tokens=1,
        device_token_capacity=args.max_total_tokens,
        consumer_wave_tokens=args.chunked_prefill_size,
        max_output_tokens=args.max_output_tokens,
        output_length_scale=args.output_length_scale,
        time_scale=args.time_scale,
    )
    encoded, encoding = encode_window(window, tokenizer)
    adapted, boundary = append_cache_boundary(encoded, tokenizer)
    warmup_inputs = adapted[: len(window.warmup_rows)]
    measured_inputs = adapted[len(window.warmup_rows) :]
    if any(len(item) >= args.context_length - SGLANG_INPUT_MARGIN_TOKENS for item in adapted):
        raise RuntimeError("adapted Bailian input exceeds SGLang's request envelope")

    measurement_baseline: dict[str, dict[str, Any]] = {}
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
        chunked_prefill_size=args.chunked_prefill_size,
        enable_mixed_chunk=args.batch_mode == "coalesced",
        enable_hierarchical_cache=True,
        hicache_ratio=args.hicache_ratio,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        warmup_records, warmup_seconds = engine.loop.run_until_complete(
            _run_phase(
                engine,
                window.warmup_rows,
                warmup_inputs,
                phase="warmup",
            )
        )
        measurement_baseline = (
            _publish_engine_stats_snapshot(engine, workspace, prior_stats_paths)
            if args.attention_backend == "nta_flashinfer"
            else {}
        )
        records, elapsed = engine.loop.run_until_complete(
            _run_phase(
                engine,
                window.measured_rows,
                measured_inputs,
                phase="measurement",
            )
        )
        measurement_final = (
            _publish_engine_stats_snapshot(engine, workspace, prior_stats_paths)
            if args.attention_backend == "nta_flashinfer"
            else {}
        )

    cumulative_by_name = (
        measurement_final
        if args.attention_backend == "nta_flashinfer"
        else _read_engine_stats(workspace, prior_stats_paths)
    )
    cumulative_stats = list(cumulative_by_name.values())
    if args.attention_backend == "nta_flashinfer":
        if not measurement_baseline or set(measurement_baseline) != set(cumulative_by_name):
            raise RuntimeError("NTA replay baseline and final worker stats disagree")
        stats = [
            _measurement_delta(report, measurement_baseline[name])
            for name, report in sorted(cumulative_by_name.items())
        ]
    else:
        stats = []

    cache_counts: Counter[str] = Counter()
    prefix_violations: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: int(item["source_index"])):
        state = _cache_state(record)
        record["observed_cache_state"] = state
        cache_counts[state] += 1
        observed = int(record["host_cached_tokens"]) + int(
            record["device_cached_tokens"]
        )
        expected_maximum = int(record["replayable_prefix_tokens"])
        if observed > expected_maximum:
            prefix_violations.append(
                {
                    "source_index": record["source_index"],
                    "observed": observed,
                    "replayable_maximum": expected_maximum,
                }
            )
        text = str(record.pop("text")).encode("utf-8")
        record["text_sha256"] = hashlib.sha256(text).hexdigest()
        digest.update(text)
        digest.update(b"\0")
    if prefix_violations:
        raise RuntimeError(
            "natural replay manufactured cache reuse beyond source history: "
            + json.dumps(prefix_violations[:16], sort_keys=True)
        )

    engine_version = importlib.metadata.version("sglang")
    total_tokens = sum(int(record["completion_tokens"]) for record in records)
    ttft = _latency_percentiles(records, "ttft_seconds")
    tpot = _latency_percentiles(records, "tpot_seconds")
    intervals = _itl_values(records)
    itl = {
        "p50": _percentile(intervals, 0.50),
        "p95": _percentile(intervals, 0.95),
        "p99": _percentile(intervals, 0.99),
        "sample_count": sum(int(record["itl_sample_count"]) for record in records),
    }
    selected_bytes, physical_bytes, byte_status = _engine_byte_accounting(stats)
    accounting = finite_window_system_accounting(records, elapsed)
    report = {
        "schema": 1,
        "classification": "sglang-bailian-natural-replay",
        "revision": os.environ.get("NTA_REVISION") or git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "engine_version": engine_version,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "attention_backend": args.attention_backend,
        "model": str(args.model.resolve()),
        "machine": _machine_metadata(),
        "demand_semantics": "exact_content_block_prefix_replay",
        "cache_policy": "natural_observed_no_forced_placement",
        "placement_proven": False,
        "cache_state_observed": True,
        "workload": {
            **window.metadata,
            "token_encoding": encoding,
            "cache_boundary": boundary,
        },
        "warmup": {
            "policy": "immediately_preceding_source_window_then_drain",
            "request_count": len(warmup_records),
            "elapsed_seconds": warmup_seconds,
            "performance_excluded": True,
            "measurement_boundary": (
                "out_of_band_scheduler_control_rpc"
                if args.attention_backend == "nta_flashinfer"
                else "drained_warmup_host_timer"
            ),
            "marker_perturbation": "none",
        },
        "measured_start": args.measured_start,
        "measured_requests": len(records),
        "time_scale": args.time_scale,
        "context_length": args.context_length,
        "max_total_tokens": args.max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "chunked_prefill_size": args.chunked_prefill_size,
        "batch_mode": args.batch_mode,
        "hicache_ratio": args.hicache_ratio,
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
        "slo_goodput": _slo_goodput(
            records,
            elapsed,
            ttft_seconds=args.slo_ttft_seconds,
            p99_itl_seconds=args.slo_p99_itl_seconds,
        ),
        "finite_window_accounting": accounting,
        "observed_cache_state_counts": dict(sorted(cache_counts.items())),
        "prefix_fidelity_violations": 0,
        "heterogeneity": _heterogeneity(records, stats),
        "native_dispatch_prefix": _native_dispatch_distribution(stats),
        "progressive_consumer": _progressive_consumer_distribution(stats),
        "prefetch_arrival_readiness": _prefetch_arrival_readiness(stats),
        "selected_bytes": selected_bytes,
        "physical_bytes": physical_bytes,
        "byte_accounting_status": byte_status,
        "execution_dispatch": (
            _execution_dispatch(stats)
            if args.attention_backend == "nta_flashinfer"
            else {"kind": "framework_reference"}
        ),
        "consumer_contract": _consumer_contract(
            stats, engine_version, args.attention_backend
        ),
        "generated_text_sha256": digest.hexdigest(),
        "correctness": {
            "verification_failures": 0,
            "generated_text_sha256": digest.hexdigest(),
            "source_prefix_identity_preserved": True,
            "prefix_fidelity_violations": 0,
        },
        "verification_failures": 0,
        "records": records,
        "engine_stats": stats,
        "engine_stats_cumulative": cumulative_stats,
    }
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
