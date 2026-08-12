#!/usr/bin/env python3
"""Compare stock and transformed NTA under a mixed HiCache arrival trace."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Any

from CompareSglangHiCache import require_clean_mechanism


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--external-requests", type=int, default=3)
    parser.add_argument("--external-tokens", type=int, default=8192)
    parser.add_argument(
        "--external-suffix-tokens",
        type=int,
        default=0,
        help="uncached chunked-prefill tokens appended to each external prefix",
    )
    parser.add_argument("--resident-requests", type=int, default=1)
    parser.add_argument("--resident-tokens", type=int, default=8192)
    parser.add_argument("--resident-output-tokens", type=int, default=128)
    parser.add_argument("--external-output-tokens", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--churn-tokens", type=int, default=12000)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument(
        "--batch-mode",
        choices=("coalesced", "separate"),
        default="coalesced",
    )
    parser.add_argument("--slo-scale", type=float, default=1.5)
    parser.add_argument("--admission-lead-layers", type=int, default=36)
    parser.add_argument("--admission-max-delay-us", type=int, default=10000)
    parser.add_argument(
        "--incremental-setup-ns",
        type=int,
        default=0,
        help=(
            "policy-model setup cost for the mechanism stress arm; zero forces the "
            "trial to expose the request-overlap path but does not remove real cost"
        ),
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--allow-oversubscribed-pool",
        action="store_true",
        help="forwarded to the load harness for capacity-pressure shapes",
    )
    parser.add_argument(
        "--cuda-graph-decode", choices=("disabled", "full"), default="disabled"
    )
    parser.add_argument(
        "--require-demand-graph",
        action="store_true",
        help="require finite NTA demand-operator graph capture and replay",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-hicache-load.json",
    )
    args = parser.parse_args()
    if args.slo_scale <= 0:
        parser.error("SLO scale must be positive")
    if args.admission_lead_layers <= 0 or args.admission_max_delay_us < 0:
        parser.error("admission bounds are invalid")
    if args.incremental_setup_ns < 0:
        parser.error("incremental setup cost must be nonnegative")
    if args.external_suffix_tokens < 0:
        parser.error("external suffix token count cannot be negative")
    return args


def _report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("classification") == "sglang-hicache-load":
            return value
    raise RuntimeError("load trial emitted no JSON report")


def run(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    workspace = ROOT / "results" / "serving" / "sglang-hicache-load-cache" / backend
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangHiCacheLoad.py"),
        "--model",
        str(args.model.resolve()),
        "--attention-backend",
        backend,
        "--external-requests",
        str(args.external_requests),
        "--external-tokens",
        str(args.external_tokens),
        "--external-suffix-tokens",
        str(args.external_suffix_tokens),
        "--resident-requests",
        str(args.resident_requests),
        "--resident-tokens",
        str(args.resident_tokens),
        "--resident-output-tokens",
        str(args.resident_output_tokens),
        "--external-output-tokens",
        str(args.external_output_tokens),
        "--request-rate",
        str(args.request_rate),
        "--churn-tokens",
        str(args.churn_tokens),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--max-running-requests",
        str(args.max_running_requests),
        "--batch-mode",
        args.batch_mode,
        "--seed",
        str(args.seed),
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    if args.allow_oversubscribed_pool:
        command.append("--allow-oversubscribed-pool")
    environment = os.environ.copy()
    environment["NTA_SGLANG_ACQUISITION_ADMISSION"] = "1"
    environment["NTA_SGLANG_ADMISSION_LEAD_LAYERS"] = str(args.admission_lead_layers)
    environment["NTA_SGLANG_ADMISSION_MAX_DELAY_US"] = str(args.admission_max_delay_us)
    environment["NTA_SGLANG_REQUIRE_MIXED_ATTENTION"] = (
        "1" if backend == "nta_flashinfer" and args.batch_mode == "coalesced" else "0"
    )
    if backend == "nta_flashinfer" and args.batch_mode == "coalesced":
        # Exercise the actual request-aware finite-kernel path. One transfer
        # wave isolates overlap from deeper transfer pipelining: resident CTAs
        # run immediately, then only the externally dependent CTAs resume.
        environment.update(
            {
                "NTA_SGLANG_PIPELINE_HOST": "0",
                "NTA_SGLANG_REQUEST_OVERLAP": "1",
                "NTA_SGLANG_MAX_HOST_ROUNDS": "1",
                "NTA_SGLANG_MIN_PREDICTED_GAIN": "1.0",
                "NTA_SGLANG_INCREMENTAL_SETUP_NS": str(args.incremental_setup_ns),
            }
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{backend} load trial failed:\n"
            + "\n".join(completed.stdout.splitlines()[-120:])
        )
    return _report(completed.stdout)


def _thresholds(stock: dict[str, Any], scale: float) -> dict[str, float]:
    return {
        "resident_ttft": scale * float(stock["resident_p95_ttft_seconds"]),
        "resident_tpot": scale * float(stock["resident_p95_tpot_seconds"]),
        "resident_itl": scale * float(stock["resident_p99_itl_seconds"]),
        "external_ttft": scale * float(stock["external_p95_ttft_seconds"]),
    }


def _goodput(report: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    resident_ok = []
    external_ok = []
    for record in report["records"]:
        if record["kind"] == "resident":
            resident_ok.append(
                float(record["ttft_seconds"]) <= thresholds["resident_ttft"]
                and float(record["tpot_seconds"]) <= thresholds["resident_tpot"]
                and float(record["p99_itl_seconds"]) <= thresholds["resident_itl"]
            )
        else:
            external_ok.append(
                float(record["ttft_seconds"]) <= thresholds["external_ttft"]
            )
    elapsed = float(report["elapsed_seconds"])
    passed = sum(resident_ok) + sum(external_ok)
    return {
        "passed_requests": passed,
        "total_requests": len(resident_ok) + len(external_ok),
        "slo_attainment": passed / (len(resident_ok) + len(external_ok)),
        "goodput_requests_per_second": passed / elapsed,
        "resident_slo_attainment": sum(resident_ok) / len(resident_ok),
        "external_slo_attainment": sum(external_ok) / len(external_ok),
    }


def _write_failed_comparison(
    output: pathlib.Path,
    reports: dict[str, dict[str, Any]],
    order: list[str],
    reason: str,
) -> None:
    failure = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison-failure",
        "execution_order": order,
        "reason": reason,
        "stock": reports["flashinfer"],
        "nta": reports["nta_flashinfer"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    order = ["flashinfer", "nta_flashinfer"]
    random.Random(args.seed).shuffle(order)
    reports = {backend: run(args, backend) for backend in order}
    stock = reports["flashinfer"]
    nta = reports["nta_flashinfer"]
    activation = require_clean_mechanism(
        nta,
        require_graph_replay=args.cuda_graph_decode == "full",
        require_demand_graph=args.require_demand_graph,
        require_physical_compaction=args.batch_mode == "coalesced",
    )
    if not stock.get("load_warmup_excluded") or not nta.get(
        "load_warmup_excluded"
    ):
        _write_failed_comparison(
            args.output, reports, order, "mixed-arrival warmup was not excluded"
        )
        raise RuntimeError("load trial did not exclude mixed-arrival graph warmup")
    if not stock["placement_proven"] or not nta["placement_proven"]:
        _write_failed_comparison(
            args.output, reports, order, "cache placement was not proven"
        )
        raise RuntimeError("load trial did not prove cache placement")
    if stock["batch_mode"] != args.batch_mode or nta["batch_mode"] != args.batch_mode:
        _write_failed_comparison(
            args.output, reports, order, "requested batch mode was not preserved"
        )
        raise RuntimeError("load trial did not preserve the requested batch mode")
    if stock["generated_text_sha256"] != nta["generated_text_sha256"]:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "stock and NTA load outputs differ",
        )
        raise RuntimeError("stock and NTA load outputs differ")
    stats = [
        entry
        for entry in nta["engine_stats"]
        if entry.get("backend") == "nta_flashinfer"
    ]
    considered = sum(
        int(entry.get("admission_considered_batches", 0)) for entry in stats
    )
    admission_bytes = sum(
        int(entry.get("admission_external_bytes", 0)) for entry in stats
    )
    delayed = sum(int(entry.get("admission_delayed_batches", 0)) for entry in stats)
    credit_rows = sum(
        int(entry.get("external_admission_credit_rows", 0)) for entry in stats
    )
    if considered == 0 or (admission_bytes == 0 and credit_rows == 0):
        # Tiered serving admits external prefixes through pre-allocation
        # credits rather than transfer-byte accounting.
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "acquisition-aware admission was not exercised",
        )
        raise RuntimeError(
            "NTA load trial did not exercise acquisition-aware admission"
        )
    tiered_layers = sum(
        int(entry.get("tiered_decode_layers", 0))
        + int(entry.get("tiered_prefill_layers", 0))
        for entry in stats
    )
    tiered_compaction = sum(
        int(entry.get("tiered_device_compaction_launches", 0))
        for entry in stats
    )
    if args.batch_mode == "coalesced" and tiered_layers > 0:
        # Tiered serving coalesces every claimed and resident request into
        # one compact plan per layer; its witnesses are the served tiered
        # layers and device selection compaction, not the demand-acquire
        # overlap counters.
        if tiered_compaction == 0:
            _write_failed_comparison(
                args.output,
                reports,
                order,
                "tiered selection compaction was not exercised",
            )
            raise RuntimeError(
                "tiered coalesced trial never ran device selection compaction"
            )
    elif args.batch_mode == "coalesced":
        mixed_layers = sum(
            int(entry.get("mixed_dependency_layers", 0)) for entry in stats
        )
        overlap_layers = sum(
            int(entry.get("request_overlap_layers", 0)) for entry in stats
        )
        ticketed_layers = sum(
            int(entry.get("ticketed_incremental_launches", 0)) for entry in stats
        )
        compact_initial = sum(
            int(entry.get("compact_initial_cta_bound", 0)) for entry in stats
        )
        parallel_progress = sum(
            int(entry.get("parallel_indexed_progress_layers", 0)) for entry in stats
        )
        if (
            min(
                mixed_layers,
                overlap_layers,
                ticketed_layers,
                compact_initial,
                parallel_progress,
            )
            == 0
        ):
            _write_failed_comparison(
                args.output,
                reports,
                order,
                "request-overlapped transformed attention was not exercised",
            )
            raise RuntimeError(
                "coalesced trial did not execute request-overlapped FlashInfer "
                "attention with compact grids and parallel indexed progress"
            )

    thresholds = _thresholds(stock, args.slo_scale)
    stock_goodput = _goodput(stock, thresholds)
    nta_goodput = _goodput(nta, thresholds)
    stock_rate = float(stock["output_token_throughput"])
    nta_rate = float(nta["output_token_throughput"])
    stock_gp = float(stock_goodput["goodput_requests_per_second"])
    nta_gp = float(nta_goodput["goodput_requests_per_second"])
    resident_ttft_ratio = float(nta["resident_p95_ttft_seconds"]) / float(
        stock["resident_p95_ttft_seconds"]
    )
    resident_tpot_ratio = float(nta["resident_p95_tpot_seconds"]) / float(
        stock["resident_p95_tpot_seconds"]
    )
    resident_itl_ratio = float(nta["resident_p99_itl_seconds"]) / float(
        stock["resident_p99_itl_seconds"]
    )
    external_ttft_ratio = float(nta["external_p95_ttft_seconds"]) / float(
        stock["external_p95_ttft_seconds"]
    )
    comparison = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison",
        "execution_order": order,
        "batch_mode": args.batch_mode,
        "slo_scale": args.slo_scale,
        "incremental_setup_ns": args.incremental_setup_ns,
        "external_suffix_tokens": args.external_suffix_tokens,
        "slo_thresholds_seconds": thresholds,
        "stock": stock,
        "nta": nta,
        "stock_goodput": stock_goodput,
        "nta_goodput": nta_goodput,
        "output_throughput_ratio": nta_rate / stock_rate,
        "goodput_ratio": nta_gp / stock_gp if stock_gp else None,
        "resident_p95_ttft_ratio": resident_ttft_ratio,
        "resident_p95_tpot_ratio": resident_tpot_ratio,
        "resident_p99_itl_ratio": resident_itl_ratio,
        "external_p95_ttft_ratio": external_ttft_ratio,
        "mechanism_activation": activation,
        "admission_considered_batches": considered,
        "admission_external_bytes": admission_bytes,
        "admission_delayed_batches": delayed,
        "mixed_dependency_layers": sum(
            int(entry.get("mixed_dependency_layers", 0)) for entry in stats
        ),
        "request_overlap_layers": sum(
            int(entry.get("request_overlap_layers", 0)) for entry in stats
        ),
        "parallel_indexed_progress_layers": sum(
            int(entry.get("parallel_indexed_progress_layers", 0)) for entry in stats
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
