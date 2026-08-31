#!/usr/bin/env python3
"""Run paired stock/NTA SGLang arms on one natural Bailian replay window."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
from typing import Any

from gpu_trial import CotenantSampler, TRIAL_OWNER_ENV, wait_for_free_gpu
from CompareSglangHiCache import require_clean_mechanism


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402
from experiments.validate_bailian_replay import (  # noqa: E402
    validate as validate_bailian_replay,
)


RESULTS_ROOT = pathlib.Path(os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
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
    parser.add_argument("--slo-tpot-seconds", type=float, default=0.050)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument(
        "--workspace-root",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving-natural",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--execution-order",
        choices=("seeded", "stock_first", "nta_first"),
        default="seeded",
    )
    parser.add_argument(
        "--allow-output-divergence",
        action="store_true",
        help="record rather than reject deterministic output divergence",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving-natural" / "comparison.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir() or not args.workload_manifest.is_file():
        parser.error("model and workload manifest must exist")
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
    if (
        not math.isfinite(args.time_scale)
        or args.time_scale <= 0.0
        or not math.isfinite(args.output_length_scale)
        or args.output_length_scale <= 0.0
        or not 0.0 < args.mem_fraction_static < 1.0
        or min(
            args.slo_ttft_seconds,
            args.slo_tpot_seconds,
            args.slo_p99_itl_seconds,
        )
        <= 0.0
    ):
        parser.error("time scale and memory fraction are invalid")
    return args


def _worker_report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("classification") == "sglang-bailian-natural-replay":
            return value
    raise RuntimeError("natural replay worker emitted no JSON report")


def _worker_command(
    args: argparse.Namespace, backend: str, workspace: pathlib.Path
) -> list[str]:
    return [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangBailianReplay.py"),
        "--model",
        str(args.model.resolve()),
        "--attention-backend",
        backend,
        "--workload-manifest",
        str(args.workload_manifest.resolve()),
        "--measured-start",
        str(args.measured_start),
        "--warmup-requests",
        str(args.warmup_requests),
        "--measured-requests",
        str(args.measured_requests),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--output-length-scale",
        str(args.output_length_scale),
        "--time-scale",
        str(args.time_scale),
        "--context-length",
        str(args.context_length),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        str(args.max_running_requests),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--batch-mode",
        args.batch_mode,
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--cuda-graph-prefill",
        args.cuda_graph_prefill,
        "--slo-ttft-seconds",
        str(args.slo_ttft_seconds),
        "--slo-tpot-seconds",
        str(args.slo_tpot_seconds),
        "--slo-p99-itl-seconds",
        str(args.slo_p99_itl_seconds),
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]


def run(
    args: argparse.Namespace, backend: str, *, run_id: str
) -> dict[str, Any]:
    cache_name = os.environ.get(
        "NTA_NATURAL_COMPARE_CACHE_NAME", "sglang-bailian-natural-cache"
    )
    workspace = args.workspace_root.resolve() / cache_name / backend
    environment = os.environ.copy()
    environment["NTA_EXECUTION_ADMISSION"] = "1"
    owner_token = f"{os.getpid()}:{time.monotonic_ns()}:{backend}"
    environment[TRIAL_OWNER_ENV] = owner_token
    wait_for_free_gpu()
    with CotenantSampler(owner_token) as sampler:
        completed = subprocess.run(
            _worker_command(args, backend, workspace),
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log = workspace.parent / "logs" / f"{run_id}.{backend}.stdout.log"
    atomic_write_text(log, completed.stdout)
    failures: list[str] = []
    if completed.returncode:
        failures.append(f"worker exited with status {completed.returncode}")
    if not sampler.complete:
        failures.append("co-tenant sampler did not terminate")
    if sampler.sampling_errors:
        failures.append(f"GPU sampler had {sampler.sampling_errors} errors")
    if failures:
        raise RuntimeError(
            f"{backend} natural replay failed ({'; '.join(failures)}):\n"
            + "\n".join(completed.stdout.splitlines()[-120:])
        )
    report = _worker_report(completed.stdout)
    report.update(
        {
            "cotenant_gpu_samples": sampler.foreign_samples,
            "gpu_samples": sampler.samples,
            "gpu_sampling_errors": sampler.sampling_errors,
            "gpu_sampling_complete": sampler.complete,
            "cotenant_pids_seen": sorted(sampler.foreign_pids),
            "stdout_log": str(log.resolve()),
        }
    )
    return report


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _request_identity(report: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            item["source_index"],
            item["source_request_id"],
            item["source_input_tokens"],
            item["replay_output_tokens"],
            item["arrival_offset_seconds"],
            item["replayable_prefix_tokens"],
        )
        for item in report["records"]
    ]


def _placement_outcomes(report: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            item["source_index"],
            item["observed_cache_state"],
            item["device_cached_tokens"],
            item["host_cached_tokens"],
        )
        for item in report["records"]
    ]


def main() -> int:
    args = parse_args()
    run_id = f"{time.time_ns()}-{os.getpid()}"
    order = ["flashinfer", "nta_flashinfer"]
    if args.execution_order == "seeded":
        random.Random(args.seed).shuffle(order)
    elif args.execution_order == "nta_first":
        order.reverse()
    reports = {
        backend: run(args, backend, run_id=run_id) for backend in order
    }
    contaminated = {
        backend: {
            "foreign_samples": report["cotenant_gpu_samples"],
            "foreign_pids": report["cotenant_pids_seen"],
        }
        for backend, report in reports.items()
        if int(report["cotenant_gpu_samples"]) > 0
    }
    if contaminated:
        raise RuntimeError(
            "natural replay was contaminated by foreign GPU activity: "
            + json.dumps(contaminated, sort_keys=True)
        )
    stock = reports["flashinfer"]
    nta = reports["nta_flashinfer"]
    if _request_identity(stock) != _request_identity(nta):
        raise RuntimeError("paired natural replay arms used different request demand")
    workload_fields = (
        "manifest_digest",
        "records_digest",
        "selected_demand_trace_digest",
    )
    if any(
        stock["workload"].get(field) != nta["workload"].get(field)
        for field in workload_fields
    ) or stock["workload"]["token_encoding"]["identity_digest"] != nta[
        "workload"
    ]["token_encoding"]["identity_digest"]:
        raise RuntimeError("paired natural replay provenance diverged")
    outputs_diverge = (
        stock["generated_text_sha256"] != nta["generated_text_sha256"]
    )
    if outputs_diverge and not args.allow_output_divergence:
        raise RuntimeError("stock and NTA natural replay outputs differ")
    placement_matched = _placement_outcomes(stock) == _placement_outcomes(nta)
    stock_goodput = float(stock["slo_goodput"]["goodput_requests_per_second"])
    nta_goodput = float(nta["slo_goodput"]["goodput_requests_per_second"])
    stock_host = int(stock["observed_cache_state_counts"].get("host", 0)) + int(
        stock["observed_cache_state_counts"].get("device_and_host", 0)
    )
    nta_host = int(nta["observed_cache_state_counts"].get("host", 0)) + int(
        nta["observed_cache_state_counts"].get("device_and_host", 0)
    )
    tier_opportunity = stock_host > 0 or nta_host > 0
    mechanism_activation = (
        require_clean_mechanism(nta) if nta_host > 0 else None
    )
    claim_blockers: list[str] = []
    if not tier_opportunity:
        claim_blockers.append("no_observed_host_tier_demand")
    if not placement_matched:
        claim_blockers.append("paired_placement_outcomes_differ")
    if mechanism_activation is None:
        claim_blockers.append("nta_mechanism_not_exercised")
    if nta["heterogeneity"].get("scope") != "batch_internal":
        claim_blockers.append("batch_internal_heterogeneity_not_observed")
    if int(nta["progressive_consumer"].get("active_observations", 0)) == 0:
        claim_blockers.append("no_progressive_consumer_execution")
    comparison = {
        "schema": 1,
        "classification": "sglang-bailian-natural-replay-comparison",
        "revision": nta.get("revision") or stock.get("revision"),
        "run_id": run_id,
        "execution_order": order,
        "outputs_diverge": outputs_diverge,
        "request_demand_matched": True,
        "placement_outcome_matched": placement_matched,
        "evidence_scope": (
            "matched_natural_tier_opportunity"
            if tier_opportunity and placement_matched
            else "end_to_end_natural_tier_opportunity"
            if tier_opportunity
            else "natural_no_tier_opportunity"
        ),
        "tier_opportunity_observed": tier_opportunity,
        "stock_host_requests": stock_host,
        "nta_host_requests": nta_host,
        "mechanism_activation": mechanism_activation,
        "claim_eligibility": {
            "matched_causal_serving": not claim_blockers,
            "blockers": claim_blockers,
            "interpretation": (
                "Eligibility is structural only; performance significance and "
                "effect size are evaluated across repeated paired trials."
            ),
        },
        "output_throughput_ratio": _ratio(
            float(nta["output_token_throughput"]),
            float(stock["output_token_throughput"]),
        ),
        "request_throughput_ratio": _ratio(
            float(nta["request_throughput"]), float(stock["request_throughput"])
        ),
        "goodput_ratio": _ratio(nta_goodput, stock_goodput),
        "p95_ttft_ratio": _ratio(
            float(nta["p95_ttft_seconds"]), float(stock["p95_ttft_seconds"])
        ),
        "p95_tpot_ratio": _ratio(
            float(nta["p95_tpot_seconds"]), float(stock["p95_tpot_seconds"])
        ),
        "p99_itl_ratio": _ratio(
            float(nta["p99_itl_seconds"]), float(stock["p99_itl_seconds"])
        ),
        "native_dispatch_prefix": nta["native_dispatch_prefix"],
        "progressive_consumer": nta["progressive_consumer"],
        "prefetch_arrival_readiness": nta["prefetch_arrival_readiness"],
        "stock": stock,
        "nta": nta,
    }
    validate_bailian_replay(comparison)
    atomic_write_json(args.output, comparison)
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
