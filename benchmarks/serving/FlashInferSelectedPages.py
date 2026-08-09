#!/usr/bin/env python3
"""Compare GPU-selected host-page acquisition with candidate overfetch."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
from collections.abc import Callable
from typing import Any

import flashinfer
import torch
from nta_runtime import (
    FlashInferLayerEpoch,
    JitPhaseProgram,
    Runtime,
    RuntimeConfig,
    attention_jit_args,
    build_selected_page_work_plan,
    register_selected_host_pages,
    request_bound_attention_jit_args,
    require_operator_pair,
)
from nta_runtime.flashinfer_schedule import decode_schedule
from nta_runtime.execution_policy import DeviceDemandCostModel, plan_device_demand


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_NAME = "nta_batch_decode_selected_pages_v1"
DIRECT_MODULE_NAME = "nta_batch_decode_selected_pages_request_bound_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-pages", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--selector",
        choices=("random", "quest"),
        default="random",
        help=(
            "random preserves the controlled-score sweep; quest derives the "
            "scores from the actual candidate keys and query via the "
            "upper-bound envelope formula"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    if (
        min(
            arguments.batch_size,
            arguments.candidate_pages,
            arguments.top_k,
            arguments.iterations,
            arguments.trials,
        )
        <= 0
    ):
        parser.error("all dimensions, trials, and iterations must be positive")
    if arguments.top_k > arguments.candidate_pages:
        parser.error("top-k cannot exceed the candidate-page count")
    return arguments


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def make_hooked_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
        jit_args=attention_jit_args(
            MODULE_NAME,
            dtype_q=torch.float16,
            dtype_kv=torch.float16,
            dtype_o=torch.float16,
            idtype=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
        ),
    )


def make_request_bound_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
        jit_args=request_bound_attention_jit_args(
            DIRECT_MODULE_NAME,
            dtype_q=torch.float16,
            dtype_kv=torch.float16,
            dtype_o=torch.float16,
            idtype=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
        ),
    )


def make_stock_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )


def plan_decode(
    wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    indices: torch.Tensor,
    batch_size: int,
    pages_per_request: int,
) -> None:
    wrapper.plan(
        torch.arange(
            0,
            (batch_size + 1) * pages_per_request,
            pages_per_request,
            dtype=torch.int32,
            device="cuda",
        ),
        indices,
        torch.full((batch_size,), 16, dtype=torch.int32, device="cuda"),
        4,
        2,
        128,
        16,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        disable_split_kv=True,
    )


def phase_program(module_name: str) -> JitPhaseProgram:
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not workspace:
        raise RuntimeError(
            "FLASHINFER_WORKSPACE_BASE is required; use tools/jit/activate.py "
            "with --flashinfer-hook"
        )
    modules = list(pathlib.Path(workspace).rglob(f"{module_name}.so"))
    if len(modules) != 1:
        raise RuntimeError(f"expected one compiled {module_name}.so, found {modules}")
    return JitPhaseProgram(modules[0])


def event_sample(call: Callable[[], None], iterations: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1_000.0 / iterations


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_geomean_interval(
    ratios: list[float], *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    if not ratios or any(value <= 0 or not math.isfinite(value) for value in ratios):
        raise ValueError("bootstrap ratios must be finite and positive")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.geometric_mean(
            ratios[rng.randrange(len(ratios))] for _ in ratios
        )
        for _ in range(samples)
    )
    return estimates[round(0.025 * (samples - 1))], estimates[
        round(0.975 * (samples - 1))
    ]


def main() -> int:
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    batch_size = arguments.batch_size
    candidate_pages = arguments.candidate_pages
    top_k = arguments.top_k
    total_candidates = batch_size * candidate_pages
    selected_count = batch_size * top_k
    page_shape = (2, 16, 2, 128)

    host_pages = torch.randn(
        (total_candidates, *page_shape), dtype=torch.float16, pin_memory=True
    )
    full_device_pages = torch.empty_like(host_pages, device="cuda")
    selected_device_pages = torch.zeros(
        (selected_count, *page_shape), dtype=torch.float16, device="cuda"
    )
    query = torch.randn((batch_size, 4, 128), dtype=torch.float16, device="cuda")
    if arguments.selector == "quest":
        from nta_runtime.quest_selector import quest_candidate_scores

        candidate_keys = (
            host_pages[:, 0]
            .view(batch_size, candidate_pages, *page_shape[1:])
            .to("cuda")
        )
        scores = quest_candidate_scores(
            query, candidate_keys, group_size=query.shape[1] // page_shape[2]
        ).to(torch.float16)
        del candidate_keys
    else:
        scores = torch.randn(
            (batch_size, candidate_pages), dtype=torch.float16, device="cuda"
        )
    source_page_table = torch.arange(
        total_candidates, dtype=torch.int32, device="cuda"
    ).view(batch_size, candidate_pages)
    lengths = torch.full(
        (batch_size,), candidate_pages, dtype=torch.int32, device="cuda"
    )
    selected_source_indices = torch.empty(
        (batch_size, top_k), dtype=torch.int32, device="cuda"
    )

    def select_pages() -> torch.Tensor:
        selected_source_indices.copy_(
            flashinfer.top_k_page_table_transform(
                scores,
                source_page_table,
                lengths,
                top_k,
                deterministic=True,
            )
        )
        return selected_source_indices

    # NTA registers this stable device tensor directly. The CPU copy below is
    # used only to construct the precomputed offline-oracle arm before timing.
    select_pages()
    selected_host_indices = selected_source_indices.cpu()
    if bool((selected_host_indices < 0).any()) or bool(
        (selected_host_indices >= total_candidates).any()
    ):
        raise RuntimeError("FlashInfer selector produced an out-of-range page")
    oracle_host_pages = host_pages.index_select(
        0, selected_host_indices.reshape(-1).to(torch.int64)
    ).pin_memory()
    oracle_device_pages = torch.empty_like(oracle_host_pages, device="cuda")
    request_bound_filter = os.environ.get("NTA_JIT_REQUEST_BOUND_SOURCE", "")
    os.environ["NTA_JIT_REQUEST_BOUND_SOURCE"] = ",".join(
        token
        for token in (request_bound_filter, DIRECT_MODULE_NAME)
        if token
    )
    hooked = make_hooked_wrapper()
    request_bound = make_request_bound_wrapper()
    compact_indices = torch.arange(selected_count, dtype=torch.int32, device="cuda")
    plan_decode(hooked, compact_indices, batch_size, top_k)
    plan_decode(
        request_bound, selected_source_indices.reshape(-1), batch_size, top_k
    )
    schedule = decode_schedule(hooked)
    if set(schedule.request_indices) != set(range(batch_size)):
        raise RuntimeError(f"FlashInfer omitted requests from its schedule: {schedule}")

    runtime = Runtime(
        RuntimeConfig(
            request_capacity=batch_size,
            object_capacity=batch_size,
            intent_capacity=max(batch_size, schedule.work_count),
            work_ticket_capacity=schedule.work_count,
            max_dependencies_per_work_ticket=1,
        )
    )
    page_bytes = host_pages[0].numel() * host_pages.element_size()
    for request in range(batch_size):
        runtime.set_request(
            request,
            0x5350415253450000 + request,
            1,
            priority=request % 8,
            max_outstanding_bytes=top_k * page_bytes,
        )
    acquisition = register_selected_host_pages(
        runtime,
        host_pages,
        selected_device_pages,
        selected_source_indices,
        stream=torch.cuda.current_stream(),
    )
    plan = build_selected_page_work_plan(
        runtime,
        acquisition,
        schedule.request_indices,
        estimated_compute_ns=2_500,
        stream=torch.cuda.current_stream(),
    )
    plan.wait_on(torch.cuda.current_stream())
    phases = phase_program(MODULE_NAME)
    direct_phases = phase_program(DIRECT_MODULE_NAME)
    require_operator_pair(direct_phases, phases)
    epoch = FlashInferLayerEpoch(
        runtime,
        plan,
        phases,
        object_count=batch_size,
        max_progress_passes=1,
        wait_for_plan=False,
    )
    nta_output = torch.empty_like(query)

    stock = make_stock_wrapper()
    plan_decode(stock, selected_source_indices.reshape(-1), batch_size, top_k)
    stock_output = torch.empty_like(query)
    request_bound_output = torch.empty_like(query)
    oracle_stock = make_stock_wrapper()
    plan_decode(oracle_stock, compact_indices, batch_size, top_k)
    oracle_output = torch.empty_like(query)

    def nta_retained_call() -> None:
        epoch.enqueue_host(
            hooked,
            query,
            selected_device_pages,
            nta_output,
            progress_blocks=batch_size,
            sm_scale=1.0 / math.sqrt(128),
            stream=torch.cuda.current_stream(),
        )

    def nta_cold_call() -> None:
        phases.invalidate_cached_objects(
            runtime, 0, batch_size, stream=torch.cuda.current_stream()
        )
        nta_retained_call()

    def overfetch_call() -> None:
        full_device_pages.copy_(host_pages, non_blocking=True)
        request_bound.run(
            query,
            full_device_pages,
            runtime.device_view_tensor,
            1.0 / math.sqrt(128),
            0,
            out=request_bound_output,
        )

    def offline_oracle_call() -> None:
        oracle_device_pages.copy_(oracle_host_pages, non_blocking=True)
        oracle_stock.run(query, oracle_device_pages, out=oracle_output)

    def nta_cold_pipeline_call() -> None:
        select_pages()
        nta_cold_call()

    def nta_retained_pipeline_call() -> None:
        select_pages()
        nta_retained_call()

    def overfetch_pipeline_call() -> None:
        select_pages()
        overfetch_call()

    def candidate_retained_pipeline_call() -> None:
        select_pages()
        request_bound.run(
            query,
            full_device_pages,
            runtime.device_view_tensor,
            1.0 / math.sqrt(128),
            0,
            out=request_bound_output,
        )

    selected_bytes = selected_count * page_bytes
    candidate_bytes = total_candidates * page_bytes
    demand_plan = plan_device_demand(
        candidate_bytes=candidate_bytes,
        selected_bytes=selected_bytes,
        selected_pages=selected_count,
        model=DeviceDemandCostModel(),
    )
    full_device_pages.copy_(host_pages, non_blocking=True)

    for _ in range(3):
        nta_cold_pipeline_call()
        overfetch_pipeline_call()
        candidate_retained_pipeline_call()
        offline_oracle_call()
    torch.cuda.synchronize()

    nta_cold_samples = []
    nta_retained_samples = []
    overfetch_samples = []
    oracle_samples = []
    candidate_retained_samples = []
    topk_samples = []
    for trial in range(arguments.trials):
        arms = (
            (nta_cold_samples, nta_cold_pipeline_call),
            (overfetch_samples, overfetch_pipeline_call),
            (oracle_samples, offline_oracle_call),
            (nta_retained_samples, nta_retained_pipeline_call),
            (candidate_retained_samples, candidate_retained_pipeline_call),
        )
        for offset in range(len(arms)):
            samples, call = arms[(trial + offset) % len(arms)]
            samples.append(event_sample(call, arguments.iterations))
        topk_samples.append(event_sample(select_pages, arguments.iterations))

    result = epoch.check(1, torch.cuda.current_stream())
    if not result.status.succeeded:
        raise RuntimeError(f"selected-page epoch did not complete: {result}")
    full_device_pages.copy_(host_pages, non_blocking=True)
    stock.run(
        query,
        full_device_pages,
        out=stock_output,
    )
    request_bound.run(
        query,
        full_device_pages,
        runtime.device_view_tensor,
        1.0 / math.sqrt(128),
        0,
        out=request_bound_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(nta_output, stock_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        request_bound_output, stock_output, rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(oracle_output, stock_output, rtol=2e-3, atol=2e-3)

    nta_cold_median = statistics.median(nta_cold_samples)
    nta_retained_median = statistics.median(nta_retained_samples)
    overfetch_median = statistics.median(overfetch_samples)
    oracle_median = statistics.median(oracle_samples)
    candidate_retained_median = statistics.median(candidate_retained_samples)
    topk_median = statistics.median(topk_samples)
    adaptive_source = "nta_cold_us" if demand_plan.mode == "indexed" else "overfetch_us"
    adaptive_samples = (
        nta_cold_samples if demand_plan.mode == "indexed" else overfetch_samples
    )
    adaptive_median = statistics.median(adaptive_samples)
    indexed_speedup_samples = [
        overfetch / indexed
        for overfetch, indexed in zip(overfetch_samples, nta_cold_samples)
    ]
    policy_speedup_samples = [
        overfetch / adaptive
        for overfetch, adaptive in zip(overfetch_samples, adaptive_samples)
    ]
    policy_regret_samples = [
        adaptive / min(indexed, overfetch)
        for adaptive, indexed, overfetch in zip(
            adaptive_samples, nta_cold_samples, overfetch_samples
        )
    ]
    indexed_speedup_ci = bootstrap_geomean_interval(
        indexed_speedup_samples, seed=7
    )
    policy_speedup_ci = bootstrap_geomean_interval(policy_speedup_samples, seed=11)
    report: dict[str, Any] = {
        "schema": 1,
        "classification": "flashinfer-gpu-selected-host-pages",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "flashinfer_version": flashinfer.__version__,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": batch_size,
        "candidate_pages_per_request": candidate_pages,
        "selected_pages_per_request": top_k,
        "candidate_bytes": candidate_bytes,
        "selected_bytes": selected_bytes,
        "bytes_avoided_fraction": 1.0 - selected_bytes / candidate_bytes,
        "gpu_selected_pages": True,
        "selector_mode": arguments.selector,
        "content_derived_scores": arguments.selector == "quest",
        # Scores are computed once at setup; per-step recomputation is the
        # serving integration's job and is not claimed by this benchmark.
        "scores_recomputed_per_step": False,
        "nta_hot_path_host_identity_round_trips": 0,
        "offline_oracle_host_identity_round_trips": 1,
        "offline_oracle_precomputed": True,
        "online_transfer_mode": demand_plan.mode,
        "predicted_bulk_ns": demand_plan.predicted_bulk_ns,
        "predicted_indexed_ns": demand_plan.predicted_indexed_ns,
        "real_flashinfer_selector": True,
        "real_flashinfer_attention": True,
        "all_policy_attention_transformed": True,
        "paired_operator_contract_verified": True,
        "stock_output_parity": True,
        "topk_us": {
            "median": topk_median,
            "p95": percentile(topk_samples, 0.95),
            "samples": topk_samples,
        },
        "nta_cold_us": {
            "median_with_topk": nta_cold_median,
            "estimated_median_without_topk": max(0.0, nta_cold_median - topk_median),
            "p95_with_topk": percentile(nta_cold_samples, 0.95),
            "samples": nta_cold_samples,
        },
        "nta_retained_us": {
            "median_with_topk": nta_retained_median,
            "estimated_median_without_topk": max(
                0.0, nta_retained_median - topk_median
            ),
            "p95_with_topk": percentile(nta_retained_samples, 0.95),
            "samples": nta_retained_samples,
        },
        "candidate_retained_us": {
            "median_with_topk": candidate_retained_median,
            "estimated_median_without_topk": max(
                0.0, candidate_retained_median - topk_median
            ),
            "p95_with_topk": percentile(candidate_retained_samples, 0.95),
            "samples": candidate_retained_samples,
        },
        "overfetch_us": {
            "median_with_topk": overfetch_median,
            "estimated_median_without_topk": max(0.0, overfetch_median - topk_median),
            "p95_with_topk": percentile(overfetch_samples, 0.95),
            "samples": overfetch_samples,
        },
        "offline_oracle_us": {
            "median_precomputed": oracle_median,
            "median_with_online_selector": oracle_median + topk_median,
            "p95_precomputed": percentile(oracle_samples, 0.95),
            "samples": oracle_samples,
        },
        "adaptive_us": {
            "median_with_topk": adaptive_median,
            "p95_with_topk": percentile(adaptive_samples, 0.95),
            "samples": adaptive_samples,
            "derived_from": adaptive_source,
        },
        "speedup_over_overfetch": (overfetch_median / nta_cold_median),
        "speedup_over_overfetch_geometric_mean": statistics.geometric_mean(
            indexed_speedup_samples
        ),
        "speedup_over_overfetch_bootstrap_95_percent_ci": indexed_speedup_ci,
        "online_policy_speedup_over_forced_overfetch": (
            overfetch_median / adaptive_median
        ),
        "online_policy_speedup_geometric_mean": statistics.geometric_mean(
            policy_speedup_samples
        ),
        "online_policy_speedup_bootstrap_95_percent_ci": policy_speedup_ci,
        "online_policy_regret_to_best_measured": statistics.geometric_mean(
            policy_regret_samples
        ),
        "online_policy_regret_samples": policy_regret_samples,
        "online_policy_regret_bootstrap_95_percent_ci": (
            bootstrap_geomean_interval(policy_regret_samples, seed=13)
        ),
        "regret_to_offline_oracle": nta_cold_median / oracle_median,
        "indexed_fixed_overhead_over_oracle_us": nta_cold_median - oracle_median,
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
