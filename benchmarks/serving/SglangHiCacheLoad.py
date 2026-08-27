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
from typing import Any, Sequence

try:
    from experiments.bailian import (
        demand_trace_digest,
        input_page_ids,
        read_jsonl,
        unique_input_page_ids,
    )
    from experiments.queueing import finite_window_littles_law
    from experiments.validate_workload import validate as validate_workload
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from experiments.bailian import (
        demand_trace_digest,
        input_page_ids,
        read_jsonl,
        unique_input_page_ids,
    )
    from experiments.queueing import finite_window_littles_law
    from experiments.validate_workload import validate as validate_workload

from SglangHiCache import (
    configure_environment,
    device_cached_tokens,
    generated_text,
    generation_results,
    git_value,
    host_cached_tokens,
    make_prompt,
)


_MEASUREMENT_COUNTERS = frozenset(
    {
        "batches",
        "canonical_initial_cta_bound",
        "canonical_resume_cta_bound",
        "compact_initial_cta_bound",
        "compact_initial_launches",
        "compact_resume_cta_bound",
        "compact_resume_launches",
        "copy_engine_bytes",
        "copy_engine_issue_cpu_ns",
        "copy_engine_layout_cpu_ns",
        "copy_engine_operations",
        "copy_engine_selected_rows",
        "copy_engine_selected_runs",
        "copy_engine_submissions",
        "copy_engine_waves",
        "cost_model_transfer_samples",
        "cross_layer_frontier_batches",
        "cross_layer_frontier_layers",
        "cta_work_items",
        "decode_launches",
        "demand_graph_captures",
        "demand_graph_evictions",
        "demand_graph_paged_prefill_warmups",
        "demand_graph_replays",
        "demand_graph_warmups",
        "demand_host_layers",
        "direct_host_layers",
        "direct_staging_bytes",
        "direct_staging_launches",
        "exact_resume_window_layers",
        "external_launches",
        "fragment_lookahead_layers",
        "fragment_lookahead_objects",
        "graph_captures",
        "graph_external_batches",
        "graph_replays",
        "hicache_external_batches",
        "hicache_fallback_batches",
        "host_direct_batches",
        "host_bound_after_full_publication_batches",
        "host_incremental_batches",
        "host_mixed_direct_batches",
        "host_typed_mixed_batches",
        "host_progress_rounds",
        "hybrid_parallel_waves",
        "indexed_host_bytes",
        "indexed_host_objects",
        "indexed_layout_candidate_bytes",
        "indexed_layout_eligible_rows",
        "indexed_layout_profile_cpu_ns",
        "indexed_layout_profiles",
        "indexed_layout_rows",
        "indexed_layout_runs",
        "indexed_object_lifetime_guard_fallbacks",
        "indexed_object_quiesced_registrations",
        "initial_acquisition_batches",
        "initial_acquisition_layers",
        "initial_typed_gap_layers",
        "incremental_host_layers",
        "incremental_initialization_samples",
        "incremental_initialization_setup_ns",
        "incremental_setup_observed_ns_total",
        "incremental_setup_samples",
        "lookahead_acquisition_layers",
        "lookahead_acquisition_objects",
        "lookahead_bound_launches",
        "lookahead_copy_waves",
        "metadata_cpu_ns",
        "mixed_dependency_layers",
        "mixed_direct_work_items",
        "mixed_external_work_items",
        "mixed_forward_batches",
        "mixed_forward_requests",
        "mixed_scheduled_requests",
        "native_external_attention_launches",
        "nvme_bytes",
        "nvme_destination_rebinds",
        "nvme_destination_slice_bytes",
        "nvme_destination_slice_count",
        "nvme_epochs",
        "nvme_fresh_slot_installs",
        "nvme_object_quiesced_replacements",
        "nvme_progress_rounds",
        "nvme_region_bytes",
        "nvme_region_count",
        "nvme_region_prepare_ns",
        "nvme_same_destination_installs",
        "nvme_shared_region_slices",
        "nvme_view_publications",
        "paired_lookahead_layers",
        "parallel_indexed_progress_layers",
        "pipeline_cpu_ns",
        "plan_cpu_ns",
        "plan_uploads",
        "prefetched_host_bytes",
        "prefetched_layers",
        "prefill_graph_capture_batches",
        "prefill_graph_served_batches",
        "prefill_launches",
        "prevalidated_indexed_progress_layers",
        "progress_snapshots",
        "ready_stock_wrapper_pairs",
        "request_acquisition_groups",
        "request_cancellations",
        "request_retirements",
        "request_compute_completed_ns",
        "request_metadata_updates",
        "request_overlap_layers",
        "request_rebindings",
        "request_work_completed",
        "request_work_failed",
        "layer_service_plan_key_missing_batches",
        "layer_service_plan_curve_missing_batches",
        "layer_service_plan_curve_uncalibrated_batches",
        "layer_service_plan_curve_calibrated_batches",
        "layer_service_retirement_commits",
        "layer_service_profiled_intervals",
        "resident_reference_batches",
        "work_topology_builds",
        "work_topology_cache_hits",
        "work_topology_cpu_ns",
        "work_topology_items",
        "semantic_dense_tiles",
        "semantic_plan_builds",
        "semantic_plan_cpu_ns",
        "semantic_verifier_sessions",
        "sm_mover_bytes",
        "sm_mover_rows",
        "stock_attention_launches",
        "stock_prefetch_metadata_fastpath_batches",
        "stock_prefetched_external_attention_launches",
        "stock_prefetched_external_batches",
        "stock_ready_external_attention_launches",
        "stock_resident_attention_launches",
        "stock_resident_batches",
        "ticketed_incremental_launches",
        "tier_external_layers",
        "tier_host_proxy_bytes",
        "tier_candidate_bytes",
        "tier_selected_bytes",
        "tier_selected_leases",
        "tier_selected_rows",
        "tile_acquisition_groups",
        "transformed_direct_launches",
        "typed_acquisition_batches",
        "typed_acquisition_rows",
        "typed_acquisition_work_items",
        "typed_bulk_attention_launches",
        "typed_demand_gap_layers",
        "typed_exact_dependency_groups",
        "typed_granularity_constrained_batches",
        "typed_transfer_groups",
    }
)


def _read_engine_stats(
    workspace: pathlib.Path, prior_paths: set[pathlib.Path]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(set(workspace.glob("nta-engine.*.json")) - prior_paths):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            reports[path.name] = report
    return reports


def _wait_for_engine_stats(
    workspace: pathlib.Path,
    prior_paths: set[pathlib.Path],
    *,
    after_unix_ns: int,
    timeout_seconds: float = 10.0,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        reports = _read_engine_stats(workspace, prior_paths)
        if reports and all(
            int(report.get("snapshot_unix_ns", 0)) >= after_unix_ns
            for report in reports.values()
        ):
            return reports
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "NTA engine statistics did not publish the measurement marker"
            )
        time.sleep(0.01)


def _measured_consumer_contract(report: dict[str, Any]) -> dict[str, Any]:
    native = int(report.get("transformed_direct_launches", 0)) + int(
        report.get("ticketed_incremental_launches", 0)
    )
    stock = int(report.get("stock_prefetched_external_attention_launches", 0))
    kind = (
        "native_work_unit"
        if native
        else "framework_reference"
        if stock
        else "projection_only"
    )
    return {
        "schema": 1,
        "engine": "sglang",
        "backend": "nta_flashinfer",
        "kind": kind,
        "exact_demand": True,
        "typed_work_plan": kind == "native_work_unit",
        "native_submission": kind == "native_work_unit",
        "numerical_consumer": kind != "projection_only",
        "engine_version": os.environ.get("NTA_SGLANG_VERSION", "0.5.16"),
    }


def _execution_dispatch(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and classify the timed selection-to-consumer boundary.

    Typed metadata may prove that a mixed batch has no profitable overlap.
    That decision must dispatch stock FlashInfer; entering the native
    work-unit consumer in a direct-only window is pure mechanism overhead and
    invalidates a no-regression result.  Mixed direct/incremental windows are
    represented explicitly instead of guessing which batch owned a launch.
    """

    counters = {
        name: sum(int(report.get(name, 0)) for report in reports)
        for name in (
            "host_direct_batches",
            "host_incremental_batches",
            "host_mixed_direct_batches",
            "host_typed_mixed_batches",
            "stock_prefetched_external_attention_launches",
            "transformed_direct_launches",
            "ticketed_incremental_launches",
            "plan_uploads",
            "work_topology_builds",
        )
    }
    direct = counters["host_direct_batches"]
    incremental = counters["host_incremental_batches"]
    stock_launches = counters["stock_prefetched_external_attention_launches"]
    native_launches = (
        counters["transformed_direct_launches"]
        + counters["ticketed_incremental_launches"]
    )
    if direct and not incremental:
        residual = {
            name: counters[name]
            for name in (
                "transformed_direct_launches",
                "ticketed_incremental_launches",
                "plan_uploads",
                "work_topology_builds",
            )
            if counters[name]
        }
        if residual:
            raise RuntimeError(
                "direct-only host selection entered the native work-unit path: "
                + json.dumps(residual, sort_keys=True)
            )
        if stock_launches == 0:
            raise RuntimeError(
                "direct-only host selection did not execute stock FlashInfer"
            )
        kind = "stock_direct"
    elif incremental and not direct:
        if native_launches == 0:
            raise RuntimeError(
                "incremental host selection did not execute a native work unit"
            )
        kind = "native_incremental"
    elif direct and incremental:
        kind = "mixed_dispatch"
    else:
        kind = "unclassified"
    return {"kind": kind, **counters}


def _measurement_delta(
    final: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Project cumulative engine counters onto the timed request window."""
    measured = dict(final)
    counter_names = set(_MEASUREMENT_COUNTERS)
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("admission_") and name != "admission_lead_layers"
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("host_selection_") and name != "host_selection"
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("host_mover_")
        and (
            name.endswith("_batches")
            or name
            in {
                "host_mover_overlap_compute_ns",
                "host_mover_predicted_sm_ns",
                "host_mover_predicted_selected_ns",
            }
        )
    )
    # Profiling and CPU accounting fields are created lazily, so a static
    # allow-list cannot know every operator form in advance.  Only cumulative
    # quantities are projected; rates and maxima are derived/gauge values and
    # must never be subtracted as counters.
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.endswith("_cpu_ns")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.endswith("_enqueue_layers")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("profiled_")
        and "_max_" not in name
        and name.endswith(("_batches", "_bytes", "_gpu_ms", "_layers", "_waits"))
    )
    for name in counter_names:
        end = final.get(name, 0)
        begin = baseline.get(name, 0)
        if isinstance(end, bool) or isinstance(begin, bool):
            raise RuntimeError(f"measurement counter {name} is boolean")
        if not isinstance(end, (int, float)) or not isinstance(begin, (int, float)):
            continue
        value = end - begin
        if value < 0:
            raise RuntimeError(f"measurement counter {name} decreased")
        measured[name] = value
    for prefix in (
        "profiled_transfer",
        "profiled_pipeline_transfer",
        "profiled_demand_transfer",
    ):
        elapsed_ms = float(measured.get(f"{prefix}_gpu_ms", 0.0))
        transfer_bytes = int(measured.get(f"{prefix}_bytes", 0))
        rate_name = f"{prefix}_gib_per_second"
        if elapsed_ms > 0.0 and transfer_bytes > 0:
            measured[rate_name] = (
                transfer_bytes / (1 << 30) / (elapsed_ms / 1_000.0)
            )
        else:
            measured.pop(rate_name, None)
    measured["measurement_scope"] = "timed_load_delta"
    measured["measurement_baseline_unix_ns"] = int(
        baseline.get("snapshot_unix_ns", 0)
    )
    measured["measurement_counter_fields"] = sorted(counter_names)
    measured["consumer_contract"] = _measured_consumer_contract(measured)
    measured["execution_protocol_status"] = measured["consumer_contract"]["kind"]
    return measured


ROOT = pathlib.Path(__file__).resolve().parents[2]

# SGLang 0.5.16 publishes ``max_req_input_len`` as ``context_len - 6``
# (``max_req_len`` is ``context_len - 1`` and the scheduler reserves another
# five tokens), then rejects inputs at the published bound.  Keep a small
# adapter margin so generated pressure requests remain valid across that
# tokenizer/scheduler boundary.  This is a request-envelope constraint, not a
# change to the normalized workload's exact token counts.
SGLANG_INPUT_MARGIN_TOKENS = 8


def _max_request_input_tokens(context_length: int) -> int:
    return context_length - SGLANG_INPUT_MARGIN_TOKENS


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


def _engine_byte_accounting(
    stats: list[dict[str, Any]],
) -> tuple[int | None, int | None, str]:
    """Project NTA's physical transfer counters into the serving report.

    ``tier_selected_bytes`` counts each ownership lease's exact logical payload
    once. Per-work-unit demand is deliberately separate because many CTAs may
    share one acquired object. The physical bar is the bytes actually staged
    into device-side destinations: the sum of mutually exclusive host-pipeline,
    indexed-host, and NVMe counters. Keeping the projection here avoids both
    CTA-fanout overcounting and inference from token/cache-size estimates.
    """
    nta_stats = [
        entry
        for entry in stats
        if isinstance(entry, dict) and entry.get("backend") == "nta_flashinfer"
    ]
    if not nta_stats:
        return None, None, "not exposed by SGLang engine metadata"
    if any("tier_selected_bytes" not in entry for entry in nta_stats):
        raise RuntimeError("NTA engine omitted typed tier-selection accounting")
    selected = sum(int(entry["tier_selected_bytes"]) for entry in nta_stats)
    physical = sum(
        int(entry.get("prefetched_host_bytes", 0))
        + int(entry.get("indexed_host_bytes", 0))
        + int(entry.get("nvme_bytes", 0))
        for entry in nta_stats
    )
    if selected < 0 or physical < 0:
        raise RuntimeError("NTA engine published negative byte accounting")
    if physical == 0:
        return None, None, "not exposed by SGLang engine metadata"
    if physical < selected:
        raise RuntimeError(
            "NTA engine physical transfer bytes are below selected demand bytes"
        )
    return selected, physical, "exact_engine_transfer_counters"


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
        "--eviction-rounds",
        type=int,
        help=(
            "explicit cache-churn rounds; zero disables churn for a capacity-fit "
            "workload, while the default derives rounds from max_total_tokens"
        ),
    )
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
        "--output",
        type=pathlib.Path,
        help="optional persistent JSON report path",
    )
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
    if args.context_length <= SGLANG_INPUT_MARGIN_TOKENS:
        parser.error("context length is too small for the SGLang request envelope")
    max_request_input_tokens = _max_request_input_tokens(args.context_length)
    if args.churn_tokens >= max_request_input_tokens:
        parser.error("churn token count exceeds the SGLang request input budget")
    if args.load_warmup_iterations < 0:
        parser.error("load warmup iterations cannot be negative")
    if args.eviction_rounds is not None and args.eviction_rounds < 0:
        parser.error("eviction rounds cannot be negative")
    if args.slo_ttft_seconds <= 0 or args.slo_p99_itl_seconds <= 0:
        parser.error("SLO thresholds must be positive")
    if args.request_rate <= 0:
        parser.error("request rate must be positive")
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("--mem-fraction-static must be between zero and one")
    if args.external_tokens + args.external_suffix_tokens >= max_request_input_tokens:
        parser.error("external prompt exceeds the SGLang request input budget")
    if args.resident_tokens >= max_request_input_tokens:
        parser.error("resident prompt exceeds the SGLang request input budget")
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
            parser.error(
                "external contexts and churn must exceed the device token pool"
            )
    return args


def _tokenized_structure_prompt(tokenizer: Any, row: dict[str, Any]) -> tuple[str, int]:
    block_size = int(row.get("block_size", 16))
    token_count = int(row["input_length"])
    block_ids = input_page_ids(row, block_size=block_size)
    token_ids: list[int] = []
    for block_id in block_ids:
        seed = f"nta-bailian-block-{block_id} "
        block_text = seed
        while len(tokenizer.encode(block_text, add_special_tokens=False)) < block_size:
            block_text += seed
        block_tokens = tokenizer.encode(block_text, add_special_tokens=False)[
            :block_size
        ]
        token_ids.extend(block_tokens)
    token_ids = token_ids[:token_count]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, measured


def _load_workload(
    path: pathlib.Path, tokenizer: Any
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[float],
    list[int],
    list[int],
    dict[str, Any],
]:
    manifest = validate_workload(path.resolve())
    records_path = path.resolve().parent / str(manifest["records_file"])
    rows = read_jsonl(records_path)
    if not rows:
        raise RuntimeError("normalized workload contains no requests")
    computed_demand_digest = demand_trace_digest(rows)
    if computed_demand_digest != manifest["demand_trace_digest"]:
        raise RuntimeError(
            "normalized workload demand digest does not match its records"
        )
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
    structure_only = not bool(
        manifest["prompt"].get("semantic_representativeness_claim")
    )

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
    request_arrival_offsets = {str(row["request_id"]): 0.0 for row in resident_rows}
    request_arrival_offsets.update(
        {
            str(row["request_id"]): offset
            for row, offset in zip(external_rows, external_offsets)
        }
    )
    block_size = int(manifest["block_size"])
    resident_page_ids = unique_input_page_ids(resident_rows, block_size=block_size)
    external_cached_prefix_tokens = [
        min(
            int(row["input_length"]) - 1,
            int(row.get("shared_prefix_blocks", 0)) * block_size,
        )
        for row in external_rows
    ]
    external_cached_page_ids = frozenset(
        page_id
        for row, prefix_tokens in zip(
            external_rows, external_cached_prefix_tokens, strict=True
        )
        for page_id in input_page_ids(row, block_size=block_size)[
            : (prefix_tokens + block_size - 1) // block_size
        ]
    )
    resident_prompt_text = [prompt(row) for row in resident_rows]
    external_prompt_text = [prompt(row) for row in external_rows]
    metadata = {
        "manifest": str(path.resolve()),
        "manifest_digest": hashlib.sha256(path.resolve().read_bytes()).hexdigest(),
        "records_digest": str(manifest["records_digest"]),
        "demand_trace_digest": str(manifest["demand_trace_digest"]),
        "block_size": block_size,
        "arrival": manifest["arrival"],
        "prompt": manifest["prompt"],
        "state_mapping": "explicit_request_state"
        if any(state is not None for state in explicit_states)
        else "first_row_resident_fallback",
        "request_count": len(rows),
        "request_id_order": [str(row["request_id"]) for row in rows],
        "request_arrival_offsets": request_arrival_offsets,
        "tokenization_errors": tokenization_errors,
        "resident_input_tokens": [int(row["input_length"]) for row in resident_rows],
        "external_input_tokens": [int(row["input_length"]) for row in external_rows],
        # The trace exposes exact shared-prefix block identities.  Only that
        # prefix is materialized during placement; request-local continuation
        # rows remain uncached and are the timed prefill query.
        "external_cached_prefix_tokens": external_cached_prefix_tokens,
        "resident_output_tokens": [int(row["output_length"]) for row in resident_rows],
        "external_output_tokens": [int(row["output_length"]) for row in external_rows],
        # These are logical page counts, not raw request-length sums.  Shared
        # Bailian prefix pages occur once in the combined pressure budget.
        "resident_input_cache_pages": len(resident_page_ids),
        "external_input_cache_pages": len(external_cached_page_ids),
        "combined_input_cache_pages": len(
            resident_page_ids | external_cached_page_ids
        ),
        "shared_input_cache_pages": len(
            resident_page_ids & external_cached_page_ids
        ),
    }
    return (
        [str(row["request_id"]) for row in resident_rows],
        [str(row["request_id"]) for row in external_rows],
        resident_prompt_text,
        external_prompt_text,
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


TokenInput = tuple[int, ...]


def _token_input(tokenizer: Any, prompt: str) -> TokenInput:
    token_ids = tuple(
        int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)
    )
    if not token_ids:
        raise RuntimeError("serving request tokenized to an empty input")
    return token_ids


def _generate_one(
    engine: Any,
    input_ids: TokenInput,
    sampling: dict[str, Any],
    *,
    rid: str | None = None,
) -> Any:
    return engine.generate(
        input_ids=list(input_ids),
        sampling_params=sampling,
        rid=rid,
    )


def _generate_many(
    engine: Any,
    inputs: Sequence[TokenInput],
    samplings: Sequence[dict[str, Any]],
) -> Any:
    if len(inputs) != len(samplings):
        raise RuntimeError("batched serving inputs and sampling parameters disagree")
    return engine.generate(
        input_ids=[list(input_ids) for input_ids in inputs],
        sampling_params=list(samplings),
    )


def _exact_calibration_input(
    tokenizer: Any,
    prefix: Sequence[int],
    measured: Sequence[int],
    *,
    label: str,
    forbidden_first_tokens: set[int],
) -> TokenInput:
    """Build an exact-prefix, exact-query-row calibration input.

    Replaying the measured input during warmup turns its nominally uncached
    suffix into a radix-cache hit.  That silently changes a mixed prefill with
    hundreds of query rows into an almost-exact-prefix request.  Text-level
    length matching is insufficient because tokenizer merges can alter the
    boundary.  This harness submits token IDs directly: the cached prefix is
    byte-for-byte identical and the continuation has exactly the same row
    count, while its first token is unique across measured and warmup inputs.
    """
    prefix_ids = tuple(int(value) for value in prefix)
    measured_ids = tuple(int(value) for value in measured)
    if not prefix_ids or measured_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("calibration input does not retain the exact cached prefix")
    query_rows = len(measured_ids) - len(prefix_ids)
    if query_rows <= 0:
        raise RuntimeError("calibration prompt has no uncached continuation")
    suffix = list(_token_input(tokenizer, make_prompt(tokenizer, label, query_rows)))
    if len(suffix) != query_rows:
        raise RuntimeError("calibration continuation changed the exact query-row count")
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    vocabulary_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    if vocabulary_size <= len(special) + len(forbidden_first_tokens):
        raise RuntimeError("tokenizer has no distinct calibration token")
    if suffix[0] in forbidden_first_tokens or suffix[0] in special:
        seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
        for offset in range(vocabulary_size):
            candidate = (seed + offset) % vocabulary_size
            if candidate not in special and candidate not in forbidden_first_tokens:
                suffix[0] = candidate
                break
        else:  # pragma: no cover - guarded by the vocabulary-size check above
            raise RuntimeError("could not select a distinct calibration token")
    forbidden_first_tokens.add(suffix[0])
    candidate = prefix_ids + tuple(suffix)
    if len(candidate) != len(measured_ids) or candidate == measured_ids:
        raise RuntimeError("calibration input did not preserve the exact request shape")
    return candidate


async def _stream_request(
    engine: Any,
    input_ids: TokenInput,
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
        input_ids=list(input_ids),
        sampling_params=sampling,
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
        "system_time_seconds": max(
            0.0, finished - (load_start_seconds + offset_seconds)
        ),
        "tpot_seconds": (
            (finished - first) / (completion_tokens - 1)
            if completion_tokens > 1
            else 0.0
        ),
        "inter_token_seconds": intervals,
        "p99_itl_seconds": _percentile(intervals, 0.99),
        "completion_tokens": completion_tokens,
        "input_tokens": len(input_ids),
        "device_cached_tokens": device_cached_tokens(final),
        "host_cached_tokens": host_cached_tokens(final),
        "text": generated_text(final),
    }


async def _run_load(
    engine: Any,
    resident_prompts: list[TokenInput],
    external_prompts: list[TokenInput],
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
        # A performance-excluded warmup must preserve the timed concurrency
        # shape. Truncating resident decode to one token lets it retire before
        # the external request joins, so graphs and service curves calibrate a
        # different single-request forward and the measured mixed shape stays
        # cold.
        "max_new_tokens": args.resident_output_tokens,
        "ignore_eos": True,
    }
    external_sampling = {
        "temperature": 0,
        "max_new_tokens": args.external_output_tokens,
        "ignore_eos": True,
    }

    async def resident(index: int, prompt: TokenInput) -> dict[str, Any]:
        sampling = dict(resident_sampling)
        if resident_output_tokens is not None:
            sampling["max_new_tokens"] = max(1, resident_output_tokens[index])
        record = await _stream_request(
            engine,
            prompt,
            sampling,
            kind="resident",
            index=index,
            request_id=(
                resident_request_ids[index]
                if resident_request_ids is not None
                else None
            ),
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
                request_id=(
                    external_request_ids[index]
                    if external_request_ids is not None
                    else None
                ),
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


def _latency_percentiles(records: list[dict[str, Any]], field: str) -> dict[str, float]:
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
    prior_stats_paths = set(workspace.glob("nta-engine.*.json"))
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
            resident_texts,
            external_texts,
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
        prompt_lengths = [
            *workload_metadata["resident_input_tokens"],
            *workload_metadata["external_input_tokens"],
        ]
        output_lengths = [
            *workload_metadata["resident_output_tokens"],
            *workload_metadata["external_output_tokens"],
        ]
        max_request_input_tokens = _max_request_input_tokens(args.context_length)
        if any(length >= max_request_input_tokens for length in prompt_lengths):
            raise RuntimeError(
                "Bailian prompt exceeds the SGLang request input budget "
                f"({max_request_input_tokens} tokens for context length "
                f"{args.context_length})"
            )
        if any(
            input_length + output_length > args.context_length
            for input_length, output_length in zip(prompt_lengths, output_lengths)
        ):
            raise RuntimeError(
                "Bailian input and output lengths exceed the configured context "
                f"length ({args.context_length})"
            )
        args.resident_requests = len(resident_texts)
        args.external_requests = len(external_texts)
        args.resident_tokens = max(workload_metadata["resident_input_tokens"])
        args.external_tokens = max(workload_metadata["external_input_tokens"])
        resident_prompts = [_token_input(tokenizer, value) for value in resident_texts]
        external_prompts = [_token_input(tokenizer, value) for value in external_texts]
        cached_prefix_lengths = [
            int(value)
            for value in workload_metadata["external_cached_prefix_tokens"]
        ]
        if len(cached_prefix_lengths) != len(external_prompts) or any(
            length <= 0 or length >= len(prompt)
            for length, prompt in zip(
                cached_prefix_lengths, external_prompts, strict=True
            )
        ):
            raise RuntimeError(
                "Bailian serving replay requires a nonempty exact shared prefix "
                "and at least one uncached query token for every external request"
            )
        external_prefixes = [
            prompt[:length]
            for prompt, length in zip(
                external_prompts, cached_prefix_lengths, strict=True
            )
        ]
        # Shape setup must never insert a timed request's continuation into the
        # radix cache. Only the token count is shared with the measured input.
        shape_prompt = _token_input(
            tokenizer,
            make_prompt(tokenizer, "load-shape-workload", len(external_prompts[0])),
        )
    else:
        external_prefixes = [
            _token_input(
                tokenizer,
                make_prompt(tokenizer, f"load-external-{index}", args.external_tokens),
            )
            for index in range(args.external_requests)
        ]
        external_prompts = [
            (
                prefix
                if args.external_suffix_tokens == 0
                else prefix
                + _token_input(
                    tokenizer,
                    make_prompt(
                        tokenizer,
                        f"load-external-suffix-{index}",
                        args.external_suffix_tokens,
                    ),
                )
            )
            for index, prefix in enumerate(external_prefixes)
        ]
        resident_prompts = [
            _token_input(
                tokenizer,
                make_prompt(tokenizer, f"load-resident-{index}", args.resident_tokens),
            )
            for index in range(args.resident_requests)
        ]
        shape_prompt = _token_input(
            tokenizer,
            make_prompt(
                tokenizer,
                "load-shape",
                max(len(prompt) for prompt in external_prompts),
            ),
        )
    external_query_rows = [
        len(prompt) - len(prefix)
        for prefix, prompt in zip(external_prefixes, external_prompts, strict=True)
    ]
    if any(
        len(prompt) >= _max_request_input_tokens(args.context_length)
        for prompt in [*resident_prompts, *external_prompts]
    ):
        raise RuntimeError(
            "a generated prompt exceeds the SGLang request input budget "
            f"({_max_request_input_tokens(args.context_length)} tokens for "
            f"context length {args.context_length})"
        )
    eviction_rounds = (
        args.eviction_rounds
        if args.eviction_rounds is not None
        else args.max_total_tokens // args.churn_tokens + 1
    )
    churn_prompts = [
        _token_input(
            tokenizer,
            make_prompt(tokenizer, f"load-churn-{index}", args.churn_tokens),
        )
        for index in range((2 + args.load_warmup_iterations) * eviction_rounds)
    ]
    setup_sampling = {"temperature": 0, "max_new_tokens": 1}
    resident_cache_tokens = 0
    external_cache_tokens = 0
    combined_cache_tokens = 0
    shared_cache_tokens = 0
    required_placement_pressure = 0
    placement_eviction_prompts: list[TokenInput] = []
    if workload_metadata is not None:
        block_size = int(workload_metadata["block_size"])
        resident_cache_tokens = (
            int(workload_metadata["resident_input_cache_pages"]) * block_size
        )
        external_cache_tokens = (
            int(workload_metadata["external_input_cache_pages"]) * block_size
        )
        combined_cache_tokens = (
            int(workload_metadata["combined_input_cache_pages"]) * block_size
        )
        shared_cache_tokens = (
            int(workload_metadata["shared_input_cache_pages"]) * block_size
        )
        if resident_cache_tokens > args.max_total_tokens:
            raise RuntimeError(
                "resident input working set cannot fit in the configured SGLang "
                f"KV pool ({resident_cache_tokens} > {args.max_total_tokens} tokens)"
            )
        # SGLang's reusable radix/HiCache prefix is the request input. The
        # generated completion is not itself a prefix of the next request, so
        # it does not provide reliable pressure for this placement phase.
        # Force at least one page beyond the combined logical working set when
        # it fits.  If it already exceeds the pool, one churn request is still
        # retained as an explicit placement perturbation; the subsequent
        # resident warmup establishes the final device/host split.
        required_placement_pressure = max(
            block_size,
            args.max_total_tokens - combined_cache_tokens + block_size,
        )
        remaining = max(args.churn_tokens, required_placement_pressure)
        maximum_prompt_tokens = _max_request_input_tokens(args.context_length) - 1
        while remaining > 0:
            token_count = min(maximum_prompt_tokens, remaining)
            placement_eviction_prompts.append(
                _token_input(
                    tokenizer,
                    make_prompt(
                        tokenizer,
                        f"load-placement-eviction-{len(placement_eviction_prompts)}",
                        token_count,
                    ),
                )
            )
            remaining -= token_count

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

        def warm_residents() -> None:
            """Materialize and verify the resident working set in bounded batches."""

            for begin in range(0, len(resident_prompts), args.max_running_requests):
                end = begin + args.max_running_requests
                results = generation_results(
                    _generate_many(
                        engine,
                        resident_prompts[begin:end],
                        [dict(setup_sampling)] * len(resident_prompts[begin:end]),
                    )
                )
                for result in results:
                    if device_cached_tokens(result) <= 0:
                        raise RuntimeError(
                            "resident warmup did not remain in device cache"
                        )

        load_seconds = time.perf_counter() - load_started
        generated_text(_generate_one(engine, shape_prompt, setup_sampling))
        generated_text(_generate_one(engine, shape_prompt, setup_sampling))

        def warm_external_prefixes() -> None:
            """Create a reusable, write-through-backed external prefix.

            SGLang may mark a long prefill as ``chunked`` and deliberately
            defer its write-through hit-count update.  A second setup hit is
            therefore part of the placement protocol: it makes the external
            prefix eligible for D→H backup before the pressure phase.  This
            is setup-only and never contributes to the timed records.
            """

            generation_results(
                _generate_many(
                    engine,
                    external_prefixes,
                    [dict(setup_sampling)] * len(external_prefixes),
                )
            )
            generation_results(
                _generate_many(
                    engine,
                    external_prefixes,
                    [dict(setup_sampling)] * len(external_prefixes),
                )
            )

        warm_external_prefixes()
        for prompt in churn_prompts[:eviction_rounds]:
            generated_text(_generate_one(engine, prompt, setup_sampling))
        if workload_metadata is None:
            external_probe = _generate_one(engine, external_prefixes[0], setup_sampling)
            if host_cached_tokens(external_probe) <= 0:
                raise RuntimeError("external JIT warmup did not load from host cache")
        for prompt in churn_prompts[eviction_rounds:]:
            generated_text(_generate_one(engine, prompt, setup_sampling))
        warm_residents()

        def establish_final_placement() -> None:
            """Restore the requested device/host split after setup traffic.

            Warmup requests intentionally exercise the same mixed scheduler
            path as the timed load, but they also perturb the device LRU.  A
            timed external request must not accidentally become a device hit
            merely because a graph warmup touched it.  Repeat the explicit
            pressure-and-resident-warm protocol after every excluded warmup so
            placement is a property of the timed phase, not of incidental
            setup order.
            """

            for prompt in placement_eviction_prompts:
                generated_text(_generate_one(engine, prompt, setup_sampling))
            if placement_eviction_prompts:
                warm_residents()

        establish_final_placement()

        calibration_first_tokens = [
            ({prompt[len(prefix)]} if len(prompt) > len(prefix) else set())
            for prefix, prompt in zip(
                external_prefixes, external_prompts, strict=True
            )
        ]
        calibration_shape_records: list[list[dict[str, int]]] = []
        for warmup in range(args.load_warmup_iterations):
            # Demand graphs warm on the first occurrence and capture on the
            # second. Both are excluded so the measured occurrence is replay.
            # Every continuation-bearing request uses the exact measured prefix
            # and row count, but diverges on its first uncached token. This is a
            # token-level contract for both synthetic and Bailian replays.
            calibration_external_prompts = [
                (
                    prompt
                    if len(prompt) == len(prefix)
                    else _exact_calibration_input(
                        tokenizer,
                        prefix,
                        prompt,
                        label=f"load-calibration-suffix-{warmup}-{index}",
                        forbidden_first_tokens=calibration_first_tokens[index],
                    )
                )
                for index, (prefix, prompt) in enumerate(
                    zip(external_prefixes, external_prompts, strict=True)
                )
            ]
            warmup_records, _ = engine.loop.run_until_complete(
                _run_load(
                    engine,
                    resident_prompts,
                    calibration_external_prompts,
                    args,
                    external_offsets,
                    resident_output_tokens,
                    external_output_tokens,
                    resident_request_ids,
                    external_request_ids,
                )
            )
            calibration_shape_records.append(
                [
                    {
                        "index": int(record["index"]),
                        "input_tokens": int(record["input_tokens"]),
                        "host_cached_tokens": int(record["host_cached_tokens"]),
                        "device_cached_tokens": int(record["device_cached_tokens"]),
                    }
                    for record in warmup_records
                    if record["kind"] == "external"
                ]
            )
            begin = (2 + warmup) * eviction_rounds
            end = begin + eviction_rounds
            for prompt in churn_prompts[begin:end]:
                generated_text(_generate_one(engine, prompt, setup_sampling))
            if eviction_rounds:
                warm_residents()
            establish_final_placement()

        # Delimit the timed counter window with a resident-only marker. The
        # same marker runs in both paired arms; in NTA it also forces the
        # asynchronous stats publisher past every warmup/calibration event.
        marker_started_ns = time.time_ns()
        from nta_runtime.engines.sglang import OBSERVATION_MARKER_REQUEST_PREFIX

        generated_text(
            _generate_one(
                engine,
                resident_prompts[0],
                setup_sampling,
                rid=f"{OBSERVATION_MARKER_REQUEST_PREFIX}baseline",
            )
        )
        if args.attention_backend == "nta_flashinfer":
            measurement_baseline = _wait_for_engine_stats(
                workspace,
                prior_stats_paths,
                after_unix_ns=marker_started_ns,
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
    missing_external = [
        {
            "index": record["index"],
            "device": record["device_cached_tokens"],
            "host": record["host_cached_tokens"],
        }
        for record in external
        if record["host_cached_tokens"] <= 0
    ]
    if missing_external:
        raise RuntimeError(
            "a timed external request was not served from host cache: "
            + json.dumps(missing_external, sort_keys=True)
        )
    missing_resident = [
        {
            "index": record["index"],
            "device": record["device_cached_tokens"],
            "host": record["host_cached_tokens"],
        }
        for record in resident
        if record["device_cached_tokens"] <= 0 or record["host_cached_tokens"] != 0
    ]
    minimum_host_prefix = min(record["host_cached_tokens"] for record in external)
    if missing_resident:
        raise RuntimeError(
            "a timed resident request was not device-resident: "
            + json.dumps(missing_resident, sort_keys=True)
        )

    def external_shapes(values: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
        return sorted(
            (
                {
                    "index": int(record["index"]),
                    "input_tokens": int(record["input_tokens"]),
                    "host_cached_tokens": int(record["host_cached_tokens"]),
                    "device_cached_tokens": int(record["device_cached_tokens"]),
                }
                for record in values
                if record["kind"] == "external"
            ),
            key=lambda value: value["index"],
        )

    timed_external_shapes = external_shapes(records)
    mismatched_calibrations = [
        {
            "warmup": warmup,
            "calibration": sorted(shape, key=lambda value: value["index"]),
            "timed": timed_external_shapes,
        }
        for warmup, shape in enumerate(calibration_shape_records)
        if sorted(shape, key=lambda value: value["index"]) != timed_external_shapes
    ]
    if mismatched_calibrations:
        raise RuntimeError(
            "performance warmup did not preserve exact cached-prefix placement "
            "and uncached query rows: "
            + json.dumps(mismatched_calibrations, sort_keys=True)
        )
    calibration_contract = {
        "kind": "exact_token_prefix_and_query_rows",
        "verified": (
            args.load_warmup_iterations > 0
            and len(calibration_shape_records) == args.load_warmup_iterations
        ),
        "warmup_iterations": len(calibration_shape_records),
        "cached_prefix_tokens": [len(value) for value in external_prefixes],
        "uncached_query_rows": external_query_rows,
        "timed_shapes": timed_external_shapes,
    }

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: (value["kind"], value["index"])):
        text = record.pop("text").encode("utf-8")
        record["text_sha256"] = hashlib.sha256(text).hexdigest()
        digest.update(text)
        digest.update(b"\0")
    cumulative_stats_by_name = _read_engine_stats(workspace, prior_stats_paths)
    cumulative_stats = list(cumulative_stats_by_name.values())
    if args.attention_backend == "nta_flashinfer":
        if not measurement_baseline or (
            set(measurement_baseline) != set(cumulative_stats_by_name)
        ):
            raise RuntimeError(
                "NTA measurement baseline and final worker statistics disagree"
            )
        stats = [
            _measurement_delta(report, measurement_baseline[name])
            for name, report in sorted(cumulative_stats_by_name.items())
        ]
    else:
        stats = []
    execution_dispatch = (
        _execution_dispatch(stats)
        if args.attention_backend == "nta_flashinfer"
        else {"kind": "framework_reference"}
    )
    engine_version = importlib.metadata.version("sglang")
    consumer_contract: dict[str, Any] | None = None
    if args.attention_backend == "nta_flashinfer":
        contracts = [
            entry.get("consumer_contract")
            for entry in stats
            if isinstance(entry, dict)
            and entry.get("backend") == "nta_flashinfer"
            and isinstance(entry.get("consumer_contract"), dict)
        ]
        # Prefer proof that the native work-unit consumer launched.  If the
        # report only contains a projection/reference contract, preserve it so
        # the formal evaluator can reject the trial with an actionable reason
        # instead of silently manufacturing evidence.
        consumer_contract = next(
            (
                contract
                for contract in contracts
                if contract.get("kind") == "native_work_unit"
            ),
            contracts[0] if contracts else None,
        )
    else:
        # Stock FlashInfer is the numerical framework reference.  It is not a
        # typed NTA work-unit consumer, but it still consumes the same exact
        # demand and must be represented explicitly in a formal arm.
        consumer_contract = {
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
    tier_entries = {
        str(entry["serving_tier"])
        for entry in stats
        if isinstance(entry, dict) and "serving_tier" in entry
    }
    if len(tier_entries) > 1:
        raise RuntimeError("NTA worker processes disagree on the serving tier")
    serving_tier = next(iter(tier_entries), "host_staged")
    tier_catalog_digests = {
        str(entry["tier_catalog_digest"])
        for entry in stats
        if isinstance(entry, dict) and entry.get("tier_catalog_digest")
    }
    if len(tier_catalog_digests) > 1:
        raise RuntimeError("NTA worker processes disagree on the tier catalog")
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
    selected_bytes, physical_bytes, byte_accounting_status = _engine_byte_accounting(
        stats
    )
    report = {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": os.environ.get("NTA_REVISION") or git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "machine": _machine_metadata(),
        "engine_version": engine_version,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "attention_backend": args.attention_backend,
        "serving_tier": serving_tier,
        "tier_catalog_digest": next(iter(tier_catalog_digests), None),
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
        "placement_eviction_rounds": len(placement_eviction_prompts),
        "placement_eviction_tokens": sum(map(len, placement_eviction_prompts)),
        "resident_input_cache_tokens": resident_cache_tokens,
        "external_input_cache_tokens": external_cache_tokens,
        "combined_input_cache_tokens": combined_cache_tokens,
        "shared_input_cache_tokens": shared_cache_tokens,
        "required_placement_pressure_tokens": required_placement_pressure,
        "churn_tokens": args.churn_tokens,
        "max_total_tokens": args.max_total_tokens,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
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
        "load_warmup_excluded": (
            args.load_warmup_iterations >= 2
            and calibration_contract["verified"]
        ),
        "calibration_input_contract": calibration_contract,
        "workload": workload_metadata,
        "demand_semantics": "exact",
        "execution_dispatch": execution_dispatch,
        "demand_trace_digest": (
            workload_metadata["demand_trace_digest"]
            if workload_metadata is not None
            else None
        ),
        "selected_bytes": selected_bytes,
        "physical_bytes": physical_bytes,
        "byte_accounting_status": byte_accounting_status,
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
        "engine_stats_cumulative": cumulative_stats,
    }
    if consumer_contract is not None:
        report["consumer_contract"] = consumer_contract
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
