#!/usr/bin/env python3
"""Run matched stock and NTA HiCache promotion trials."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.atomic_io import atomic_write_json  # noqa: E402

RESULTS_ROOT = pathlib.Path(os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--hot-tokens", type=int, default=160)
    parser.add_argument("--hot-requests", type=int, default=1)
    parser.add_argument("--churn-tokens", type=int, default=240)
    parser.add_argument("--resident-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=320)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--hicache-ratio", type=float, default=4.0)
    parser.add_argument(
        "--cuda-graph-decode",
        choices=("disabled", "full"),
        default="disabled",
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--max-latency-regression-percent",
        type=float,
        help="fail when the NTA median exceeds stock by more than this percent",
    )
    parser.add_argument(
        "--verify-transfer",
        action="store_true",
        help=(
            "run a separate performance-excluded NTA arm that compares every "
            "promoted KV layer with its pinned-host source"
        ),
    )
    parser.add_argument(
        "--require-demand-graph",
        action="store_true",
        help=(
            "require captures and launches of the finite incremental NTA "
            "operator graph, not only SGLang's model decode graph"
        ),
    )
    parser.add_argument(
        "--require-physical-compaction",
        action="store_true",
        help=(
            "require the incremental form to launch fewer resume CTAs than "
            "the canonical full grid"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "sglang-hicache.json",
    )
    args = parser.parse_args()
    if (
        args.max_latency_regression_percent is not None
        and args.max_latency_regression_percent < 0
    ):
        parser.error("latency regression limit cannot be negative")
    return args


def parse_report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(report, dict)
            and report.get("classification") == "sglang-hicache-promotion"
        ):
            return report
    raise RuntimeError("HiCache trial did not emit a report")


def require_clean_mechanism(
    report: dict[str, Any],
    *,
    require_graph_replay: bool = False,
    require_demand_graph: bool = False,
    require_physical_compaction: bool = False,
) -> dict[str, Any]:
    """Validate the exact execution contract exercised by one serving arm."""
    stats = [
        entry
        for entry in report.get("engine_stats", [])
        if entry.get("backend") == "nta_flashinfer"
    ]
    if not stats:
        raise RuntimeError("NTA HiCache trial did not publish engine statistics")

    def total(key: str) -> int:
        return sum(int(entry.get(key, 0)) for entry in stats)

    protocols = {
        str(entry.get("execution_protocol"))
        for entry in stats
        if entry.get("execution_protocol")
    }
    if len(protocols) != 1:
        raise RuntimeError(
            "NTA HiCache trial did not publish one execution protocol "
            f"({sorted(protocols)})"
        )
    protocol = next(iter(protocols))
    physical_tier = any(
        entry.get("serving_tier") in {"nvme", "cxl_dax"} for entry in stats
    )
    auto_host_entries = [
        entry
        for entry in stats
        if entry.get("serving_tier") == "host_staged"
        and entry.get("host_execution_mode") == "auto"
    ]
    auto_calibration_closed = all(
        entry.get("incremental_setup_calibrated") is True
        and int(entry.get("incremental_calibration_probes_remaining", -1)) == 0
        for entry in auto_host_entries
    )
    if auto_host_entries and not auto_calibration_closed:
        raise RuntimeError(
            "host AUTO trial reached measurement before incremental setup "
            "calibration closed"
        )

    fallbacks = total("hicache_fallback_batches")
    external_batches = total("hicache_external_batches")
    external_launches = total("external_launches")
    prefetched_layers = total("prefetched_layers")
    demand_layers = total("demand_host_layers")
    transformed = total("transformed_direct_launches")
    incremental = total("ticketed_incremental_launches")
    event_ordered_incremental = total("event_ordered_incremental_launches")
    native_incremental = incremental + event_ordered_incremental
    attention = total("decode_launches") + total("prefill_launches")
    stock_launches = total("stock_attention_launches")
    stock_resident_launches = total("stock_resident_attention_launches")
    stock_external_launches = total("stock_prefetched_external_attention_launches")
    native_external_launches = total("native_external_attention_launches")
    accounted_external_launches = external_launches
    tier_external_layers = total("tier_external_layers")
    acquisition_layers = prefetched_layers + demand_layers + tier_external_layers
    if fallbacks:
        raise RuntimeError(f"NTA HiCache trial used {fallbacks} fallback batches")
    if external_batches == 0 or accounted_external_launches == 0:
        raise RuntimeError("NTA HiCache trial did not execute an external batch")
    if accounted_external_launches != acquisition_layers:
        raise RuntimeError(
            "external attention layers do not match acquisition layers "
            f"({accounted_external_launches} != {prefetched_layers} + "
            f"{demand_layers} + {tier_external_layers})"
        )
    if external_launches != native_external_launches + stock_external_launches:
        raise RuntimeError(
            "external numerical-consumer accounting is not disjoint "
            f"({external_launches} != native {native_external_launches} + "
            f"stock {stock_external_launches})"
        )
    if (
        stock_launches != stock_resident_launches + stock_external_launches
        or transformed + native_incremental + stock_launches != attention
    ):
        raise RuntimeError(
            "attention accounting is not exact "
            f"(stock={stock_launches}, resident_stock={stock_resident_launches}, "
            f"external_stock={stock_external_launches}, transformed={transformed}, "
            f"ticketed={incremental}, event_ordered={event_ordered_incremental}, "
            f"total={attention})"
        )
    contracts = [
        contract for entry in stats for contract in entry.get("operator_contracts", [])
    ]
    verified_modules = total("verified_operator_modules")
    compiler_verification_required = (
        transformed > 0 or native_incremental > 0 or native_external_launches > 0
    )
    if compiler_verification_required and (verified_modules == 0 or not contracts):
        raise RuntimeError(
            "NTA native numerical path did not verify compiler contracts"
        )

    # Transport is now a runtime-owned artifact, independent of whichever
    # numerical operator the selector executes.  Every external path must
    # prove that this artifact was content-checked and that it carries the
    # generic incremental/tier-ownership contract.  Loading an unused typed
    # attention module is not a substitute for this proof.
    transport_entries = [
        entry
        for entry in stats
        if entry.get("transport_program_loaded") is True
        and isinstance(entry.get("transport_contract"), dict)
    ]
    if not transport_entries:
        raise RuntimeError("NTA external path did not verify its transport program")
    for entry in transport_entries:
        transport_contract = entry["transport_contract"]
        digest = str(entry.get("transport_program_sha256", ""))
        if (
            transport_contract.get("family") != "generic"
            or transport_contract.get("form") != "incremental"
            or int(transport_contract.get("tier_mask", 0)) == 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise RuntimeError("NTA transport program contract is incomplete")

    framework_preacquired_verified = False
    if stock_external_launches > 0:
        framework_preacquired_verified = any(
            isinstance(entry.get("consumer_contract"), dict)
            and entry["consumer_contract"].get("kind") == "framework_reference"
            and entry["consumer_contract"].get("exact_demand") is True
            and entry["consumer_contract"].get("numerical_consumer") is True
            for entry in stats
        )
        if not framework_preacquired_verified:
            raise RuntimeError(
                "preacquired stock consumer lacks an exact framework contract"
            )

    mixed_layers = total("mixed_dependency_layers")
    if (
        protocol == "late_bound"
        and mixed_layers == 0
        and accounted_external_launches > 0
        and stock_external_launches == 0
    ):
        raise RuntimeError(
            "late-bound trial formed no heterogeneous layer with direct and "
            "external work"
        )

    compact_launches = total("compact_resume_launches")
    compact_ctas = total("compact_resume_cta_bound")
    canonical_ctas = total("canonical_resume_cta_bound")
    native_work_unit_active = native_incremental > 0 and native_external_launches > 0
    heterogeneous_work_unit_active = native_work_unit_active and mixed_layers > 0
    # A complete exact prefetch legitimately uses the stock consumer for the
    # external pages, so there is no resume grid to compact.  Compaction is a
    # gate for the incremental form, not for this stock-consumer control arm.
    if (
        require_physical_compaction
        and not physical_tier
        and native_incremental > 0
        and (
            compact_launches == 0
            or compact_ctas == 0
            or canonical_ctas == 0
            or compact_ctas >= canonical_ctas
        )
    ):
        raise RuntimeError(
            "incremental trial did not physically compact resume work "
            f"({compact_launches} launches, {compact_ctas}/{canonical_ctas} CTAs)"
        )

    if require_graph_replay:
        captures = total("graph_captures")
        replays = total("graph_replays")
        if captures == 0 or replays == 0:
            raise RuntimeError("NTA graph trial did not capture and replay")
    demand_graph = {
        key: total(key)
        for key in (
            "demand_graph_warmups",
            "demand_graph_captures",
            "demand_graph_replays",
        )
    }
    if require_demand_graph and min(demand_graph.values()) == 0:
        raise RuntimeError(
            "NTA trial did not warm, capture, and replay the demand graph "
            f"({demand_graph})"
        )
    return {
        "all_attention_transformed": stock_launches == 0,
        "external_attention_transformed": (
            fallbacks == 0 and native_external_launches == acquisition_layers
        ),
        "external_attention_stock_consumer": stock_external_launches > 0,
        "external_attention_accounted": (
            fallbacks == 0 and accounted_external_launches == acquisition_layers
        ),
        # A stock consumer after complete prefetch is valid deployment
        # behavior, but it is transport evidence rather than evidence that
        # heterogeneous work ran as individual dependencies became ready.
        "native_work_unit_active": native_work_unit_active,
        "heterogeneous_work_unit_active": heterogeneous_work_unit_active,
        "transport_only": (
            stock_external_launches > 0
            and transformed == 0
            and native_incremental == 0
        ),
        "resident_reference_attention_launches": stock_resident_launches,
        "active_forms": [
            name
            for name, count in (
                ("direct", transformed),
                ("ticketed_incremental", incremental),
                ("event_ordered_incremental", event_ordered_incremental),
            )
            if count
        ],
        "external_batches": external_batches,
        "external_launches": accounted_external_launches,
        "transformed_external_launches": native_external_launches,
        "stock_prefetched_external_launches": stock_external_launches,
        "transformed_direct_launches": transformed,
        "ticketed_incremental_launches": incremental,
        "event_ordered_incremental_launches": event_ordered_incremental,
        "total_attention_launches": attention,
        "fallback_batches": fallbacks,
        "verified_operator_modules": verified_modules,
        "operator_contract_count": len(contracts),
        "compiler_verification_required": compiler_verification_required,
        "compiler_verification_active": (
            compiler_verification_required
            and verified_modules > 0
            and len(contracts) > 0
        ),
        "framework_preacquired_verified": framework_preacquired_verified,
        "transport_program_verified": True,
        "verification_domain": (
            "compiler_typed_work_unit"
            if compiler_verification_required
            else "framework_exact_preacquired"
        ),
        "mixed_dependency_layers": mixed_layers,
        "compact_resume_launches": compact_launches,
        "compact_resume_cta_bound": compact_ctas,
        "canonical_resume_cta_bound": canonical_ctas,
        "compact_resume_cta_ratio": (
            compact_ctas / canonical_ctas if canonical_ctas else None
        ),
        "physical_compaction_applicable": native_incremental > 0 and not physical_tier,
        "physical_compaction_proven": (
            physical_tier
            or native_incremental == 0
            or (
                compact_launches > 0
                and compact_ctas > 0
                and canonical_ctas > compact_ctas
            )
        ),
        "demand_graph": demand_graph,
        "execution_protocol": protocol,
        "auto_calibration_applicable": bool(auto_host_entries),
        "auto_calibration_closed": auto_calibration_closed,
    }


def run(
    args: argparse.Namespace, backend: str, *, verify_transfer: bool = False
) -> dict[str, Any]:
    workspace = RESULTS_ROOT / "serving" / "sglang-hicache-cache" / backend
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangHiCache.py"),
        "--model",
        str(args.model.resolve()),
        "--attention-backend",
        backend,
        "--iterations",
        str(args.iterations),
        "--hot-tokens",
        str(args.hot_tokens),
        "--hot-requests",
        str(args.hot_requests),
        "--churn-tokens",
        str(args.churn_tokens),
        "--resident-tokens",
        str(args.resident_tokens),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    if args.max_attempts is not None:
        command.extend(("--max-attempts", str(args.max_attempts)))
    environment = os.environ.copy()
    environment.pop("NTA_VERIFY_TRANSFER", None)
    if verify_transfer:
        if backend != "nta_flashinfer":
            raise ValueError("transfer verification is defined only for NTA")
        environment["NTA_VERIFY_TRANSFER"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-100:])
        raise RuntimeError(
            f"HiCache {backend} trial failed with exit code "
            f"{completed.returncode}:\n{tail}"
        )
    return parse_report(completed.stdout)


def main() -> int:
    args = parse_args()
    execution_order = ["flashinfer", "nta_flashinfer"]
    random.Random(args.seed).shuffle(execution_order)
    reports = {backend: run(args, backend) for backend in execution_order}
    baseline = reports["flashinfer"]
    mechanism = reports["nta_flashinfer"]
    activation = require_clean_mechanism(
        mechanism,
        require_graph_replay=args.cuda_graph_decode == "full",
        require_demand_graph=args.require_demand_graph,
        require_physical_compaction=args.require_physical_compaction,
    )
    if not baseline.get("shape_warmup_excluded") or not mechanism.get(
        "shape_warmup_excluded"
    ):
        raise RuntimeError("matched HiCache trials did not exclude shape JIT warmup")
    if args.resident_tokens and (
        not baseline.get("resident_setup_excluded")
        or not mechanism.get("resident_setup_excluded")
        or not baseline.get("placement_proof_required")
        or not mechanism.get("placement_proof_required")
    ):
        raise RuntimeError(
            "matched heterogeneous trials did not prove host/device placement"
        )
    if baseline.get("revision") != mechanism.get("revision"):
        raise RuntimeError("stock and NTA trials used different revisions")
    if baseline["generated_text_sha256"] != mechanism["generated_text_sha256"]:
        raise RuntimeError(
            "stock and NTA HiCache generations differ: "
            f"stock={baseline.get('generated_text_samples')} "
            f"NTA={mechanism.get('generated_text_samples')}"
        )
    if baseline["external_attempt_indices"] != mechanism["external_attempt_indices"]:
        raise RuntimeError(
            "stock and NTA observed different host-residency sequences: "
            f"stock={baseline['external_attempt_indices']} "
            f"NTA={mechanism['external_attempt_indices']}"
        )
    transfer_verification = None
    if args.verify_transfer:
        transfer_verification = run(args, "nta_flashinfer", verify_transfer=True)
        require_clean_mechanism(
            transfer_verification,
            require_graph_replay=args.cuda_graph_decode == "full",
            require_demand_graph=args.require_demand_graph,
            require_physical_compaction=args.require_physical_compaction,
        )
        if (
            transfer_verification["generated_text_sha256"]
            != baseline["generated_text_sha256"]
        ):
            raise RuntimeError("transfer-verification generation differs from stock")
        if (
            transfer_verification["external_attempt_indices"]
            != baseline["external_attempt_indices"]
        ):
            raise RuntimeError(
                "transfer-verification host-residency sequence differs from stock"
            )
    baseline_time = float(baseline["median_promotion_seconds"])
    mechanism_time = float(mechanism["median_promotion_seconds"])
    latency_change = mechanism_time / baseline_time - 1.0
    if (
        args.max_latency_regression_percent is not None
        and latency_change > args.max_latency_regression_percent / 100.0
    ):
        raise RuntimeError(
            f"NTA median latency changed by {100.0 * latency_change:.2f}%; "
            f"limit is {args.max_latency_regression_percent:.2f}%"
        )
    report = {
        "schema": 1,
        "classification": "matched-sglang-hicache-comparison",
        "revision": baseline["revision"],
        "dirty": bool(baseline.get("dirty") or mechanism.get("dirty")),
        "correctness": True,
        "execution_order": execution_order,
        "randomization_seed": args.seed,
        "baseline": baseline,
        "mechanism": mechanism,
        "mechanism_activation": activation,
        "promotion_throughput_ratio": baseline_time / mechanism_time,
        "promotion_latency_change_fraction": latency_change,
        "max_latency_regression_percent": args.max_latency_regression_percent,
        "transfer_verification": transfer_verification,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
