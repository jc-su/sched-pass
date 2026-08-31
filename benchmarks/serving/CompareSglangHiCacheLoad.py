#!/usr/bin/env python3
"""Compare stock and transformed NTA under a mixed HiCache arrival trace."""

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

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CompareSglangHiCache import require_clean_mechanism  # noqa: E402
from experiments.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402
from experiments.serving_metrics import (  # noqa: E402
    preregistered_goodput,
    preregistered_joint_goodput,
    relative_goodput,
    relative_thresholds,
    safe_ratio,
)
from experiments.serving_path_evidence import (  # noqa: E402
    EXERCISED_PATHS,
    require_exercised_paths,
    require_frontier_shape,
)
from experiments.validate_serving_report import (  # noqa: E402
    validate as validate_serving_report,
)
from gpu_trial import (  # noqa: E402
    CotenantSampler,
    TRIAL_OWNER_ENV,
    query_gpu_power_limits,
    trial_environment_evidence,
    wait_for_free_gpu,
)


RESULTS_ROOT = pathlib.Path(os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=pathlib.Path,
        help="normalized Bailian manifest replayed identically by both arms",
    )
    parser.add_argument(
        "--scale-workload-arrivals-to-request-rate",
        action="store_true",
        help=(
            "uniformly replay a rate-bearing manifest at --request-rate in "
            "both arms"
        ),
    )
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
    parser.add_argument("--external-output-tokens", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--churn-tokens", type=int, default=12000)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--chunked-prefill-size", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument(
        "--numa-node",
        type=int,
        help="SGLang scheduler/HiCache NUMA node, applied identically to both arms",
    )
    parser.add_argument(
        "--cpu-affinity",
        help="fail-closed CPU-list contract applied identically to both arms",
    )
    parser.add_argument(
        "--eviction-rounds",
        type=int,
        help=(
            "explicit cache-churn rounds forwarded to both arms; zero disables "
            "churn for a capacity-fit workload"
        ),
    )
    parser.add_argument("--load-warmup-iterations", type=int, default=8)
    parser.add_argument("--setup-idle-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--gpu-start-max-temperature-c",
        type=int,
        default=60,
        help="maximum GPU temperature for two samples before either paired arm",
    )
    parser.add_argument(
        "--batch-mode",
        choices=("coalesced", "separate"),
        default="coalesced",
    )
    parser.add_argument("--slo-scale", type=float, default=1.5)
    parser.add_argument("--slo-ttft-seconds", type=float, default=8.0)
    parser.add_argument("--slo-tpot-seconds", type=float, default=0.050)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
    parser.add_argument("--admission-max-delay-us", type=int, default=10000)
    parser.add_argument(
        "--incremental-setup-ns",
        type=int,
        default=None,
        help=(
            "optional deployment calibration; when omitted, the runtime starts "
            "uncalibrated and may use its explicitly counted calibration probe"
        ),
    )
    parser.add_argument(
        "--nta-calibration-profile",
        type=pathlib.Path,
        help=(
            "compatibility-bound AUTO profile for the NTA arm; paired timing "
            "always opens it read-only and rejects online calibration"
        ),
    )
    parser.add_argument(
        "--prepare-nta-calibration-profile",
        action="store_true",
        help=(
            "run one excluded writable NTA calibration arm before the paired "
            "trial, then reopen --nta-calibration-profile read-only for timing"
        ),
    )
    parser.add_argument(
        "--nta-calibration-profile-tag",
        default="default",
        help="deployment tag used when validating --nta-calibration-profile",
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument(
        "--workspace-root",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("NTA_SERVING_WORKSPACE_ROOT", RESULTS_ROOT / "serving")
        ),
        help="external JIT/cache workspace root; defaults to results/serving for direct runs",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--execution-order",
        choices=("seeded", "stock_first", "nta_first"),
        default="seeded",
        help=(
            "arm order; explicit values keep pre-registered seeds intact "
            "instead of searching for a seed that shuffles to the order"
        ),
    )
    parser.add_argument(
        "--allow-output-divergence",
        action="store_true",
        help=(
            "record instead of reject arm output divergence; graph replay's "
            "floating-point reordering can flip near-tie tokens, and the "
            "scored quality battery — not text equality — is the registered "
            "quality metric"
        ),
    )
    parser.add_argument(
        "--allow-oversubscribed-pool",
        action="store_true",
        help="forwarded to the load harness for capacity-pressure shapes",
    )
    parser.add_argument(
        "--cuda-graph-decode", choices=("disabled", "full"), default="disabled"
    )
    parser.add_argument(
        "--cuda-graph-prefill",
        choices=("disabled", "breakable"),
        default="disabled",
    )
    parser.add_argument(
        "--require-demand-graph",
        action="store_true",
        help="require finite NTA demand-operator graph capture and replay",
    )
    parser.add_argument(
        "--require-exercised-path",
        action="append",
        choices=EXERCISED_PATHS,
        default=[],
        help=(
            "fail unless timed counters prove the requested physical path; "
            "repeat for compound arms (for example native_demand_sm plus "
            "prefetch_copy_engine plus partial_consumer)"
        ),
    )
    parser.add_argument("--require-native-frontier-layers", type=int)
    parser.add_argument("--require-ready-stock-layers", type=int)
    parser.add_argument("--require-progressive-layers", type=int)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "sglang-hicache-load.json",
    )
    args = parser.parse_args()
    if (
        args.slo_scale <= 0
        or args.slo_ttft_seconds <= 0
        or args.slo_tpot_seconds <= 0
        or args.slo_p99_itl_seconds <= 0
    ):
        parser.error("SLO scale and thresholds must be positive")
    if args.admission_max_delay_us < 0:
        parser.error("admission bounds are invalid")
    if args.eviction_rounds is not None and args.eviction_rounds < 0:
        parser.error("eviction rounds cannot be negative")
    if args.numa_node is not None and args.numa_node < 0:
        parser.error("NUMA node cannot be negative")
    if args.load_warmup_iterations < 0:
        parser.error("load warmup iterations cannot be negative")
    if args.setup_idle_timeout_seconds <= 0.0:
        parser.error("setup idle timeout must be positive")
    if args.gpu_start_max_temperature_c <= 0:
        parser.error("GPU start temperature must be positive")
    if args.incremental_setup_ns is not None and args.incremental_setup_ns < 0:
        parser.error("incremental setup cost must be nonnegative")
    if not args.nta_calibration_profile_tag.strip():
        parser.error("calibration profile tag cannot be empty")
    if (
        args.prepare_nta_calibration_profile
        and args.nta_calibration_profile is None
    ):
        parser.error(
            "--prepare-nta-calibration-profile requires --nta-calibration-profile"
        )
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("--mem-fraction-static must be between zero and one")
    if args.external_suffix_tokens < 0:
        parser.error("external suffix token count cannot be negative")
    if args.scale_workload_arrivals_to_request_rate and args.workload_manifest is None:
        parser.error("arrival scaling requires --workload-manifest")
    if args.churn_tokens > args.context_length:
        parser.error(
            "--churn-tokens cannot exceed --context-length; choose a smaller "
            "churn request or a larger model context"
        )
    if (
        args.workload_manifest is None
        and args.external_tokens + args.external_suffix_tokens > args.context_length
    ):
        parser.error(
            "--external-tokens plus --external-suffix-tokens cannot exceed "
            "--context-length"
        )
    if args.resident_tokens > args.context_length:
        parser.error("--resident-tokens cannot exceed --context-length")
    if any(
        value is not None and value < 0
        for value in (
            args.require_native_frontier_layers,
            args.require_ready_stock_layers,
            args.require_progressive_layers,
        )
    ):
        parser.error("required frontier layer counts cannot be negative")
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


def run(
    args: argparse.Namespace,
    backend: str,
    *,
    calibration_profile_access: str = "read_only",
    trial_label: str | None = None,
) -> dict[str, Any]:
    if calibration_profile_access not in {"read_only", "writable"}:
        raise ValueError("unknown calibration-profile access mode")
    if calibration_profile_access == "writable" and (
        backend != "nta_flashinfer" or args.nta_calibration_profile is None
    ):
        raise ValueError("writable calibration requires an NTA profile path")
    # Kernel-byte-forking toggles (e.g. NTA_STAGING_STREAMING) require a
    # variant-tagged cache so the shim's fail-closed guard can prove a
    # toggled env never reuses the other variant's compiled kernels.
    cache_name = os.environ.get("NTA_COMPARE_CACHE_NAME", "sglang-hicache-load-cache")
    workspace = args.workspace_root.resolve() / cache_name / backend
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
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--max-running-requests",
        str(args.max_running_requests),
        "--batch-mode",
        args.batch_mode,
        "--slo-ttft-seconds",
        str(args.slo_ttft_seconds),
        "--slo-tpot-seconds",
        str(args.slo_tpot_seconds),
        "--slo-p99-itl-seconds",
        str(args.slo_p99_itl_seconds),
        "--seed",
        str(args.seed),
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--cuda-graph-prefill",
        args.cuda_graph_prefill,
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    if args.eviction_rounds is not None:
        command.extend(("--eviction-rounds", str(args.eviction_rounds)))
    if args.numa_node is not None:
        command.extend(("--numa-node", str(args.numa_node)))
    if args.cpu_affinity is not None:
        command.extend(("--cpu-affinity", args.cpu_affinity))
    command.extend(("--load-warmup-iterations", str(args.load_warmup_iterations)))
    command.extend(
        ("--setup-idle-timeout-seconds", str(args.setup_idle_timeout_seconds))
    )
    if args.workload_manifest is not None:
        command.extend(("--workload-manifest", str(args.workload_manifest.resolve())))
    if args.scale_workload_arrivals_to_request_rate:
        command.append("--scale-workload-arrivals-to-request-rate")
    if args.allow_oversubscribed_pool:
        command.append("--allow-oversubscribed-pool")
    if calibration_profile_access == "writable":
        command.append("--auto-calibration-training-run")
    environment = os.environ.copy()
    for name in (
        "NTA_EXECUTION_CALIBRATION_PROFILE",
        "NTA_EXECUTION_CALIBRATION_PROFILE_READ_ONLY",
        "NTA_EXECUTION_CALIBRATION_PROFILE_TAG",
    ):
        environment.pop(name, None)
    environment["NTA_EXECUTION_ADMISSION"] = "1"
    environment["NTA_EXECUTION_ADMISSION_MAX_DELAY_US"] = str(
        args.admission_max_delay_us
    )
    role = backend if trial_label is None else trial_label
    owner_token = f"{os.getpid()}:{time.monotonic_ns()}:{role}"
    environment[TRIAL_OWNER_ENV] = owner_token
    if backend == "nta_flashinfer":
        # Execution form is orthogonal to the semantic protocol. ``auto`` is
        # the production policy; direct/dependency_aware are explicit causal
        # arms and are carried into the artifact below.
        environment["NTA_EXECUTION_HOST_FORM"] = os.environ.get(
            "NTA_COMPARE_EXECUTION_HOST_FORM", "auto"
        )
        if args.nta_calibration_profile is not None:
            environment["NTA_EXECUTION_CALIBRATION_PROFILE"] = str(
                args.nta_calibration_profile.expanduser().resolve()
            )
            if calibration_profile_access == "read_only":
                environment["NTA_EXECUTION_CALIBRATION_PROFILE_READ_ONLY"] = "1"
            environment["NTA_EXECUTION_CALIBRATION_PROFILE_TAG"] = (
                args.nta_calibration_profile_tag.strip()
            )
    if backend == "nta_flashinfer" and args.batch_mode == "coalesced":
        # Exercise the production selector.  The benchmark must not silently
        # force a one-wave/direct arm: doing so turns a mechanism comparison
        # into a hand-picked scheduling-policy comparison.  Dedicated
        # ablations may still override planner parameters explicitly.
        protocol = os.environ.get("NTA_COMPARE_EXECUTION_PROTOCOL", "late_bound")
        environment["NTA_EXECUTION_PROTOCOL"] = protocol
        if args.incremental_setup_ns is not None:
            environment["NTA_EXECUTION_INCREMENTAL_SETUP_NS"] = str(
                args.incremental_setup_ns
            )
        for compare_name, runtime_name in (
            ("NTA_COMPARE_EXECUTION_MAX_ROUNDS", "NTA_EXECUTION_MAX_ROUNDS"),
            (
                "NTA_COMPARE_EXECUTION_MIN_PREDICTED_GAIN",
                "NTA_EXECUTION_MIN_PREDICTED_GAIN",
            ),
        ):
            override = os.environ.get(compare_name)
            if override is not None:
                environment[runtime_name] = override
    wait_for_free_gpu(
        max_temperature_c=args.gpu_start_max_temperature_c,
        stable_samples=2,
    )
    power_limits = query_gpu_power_limits()
    if len(power_limits) != 1:
        raise RuntimeError(
            "SGLang serving artifact currently requires exactly one visible GPU"
        )
    environment["NTA_EXECUTION_CALIBRATION_GPU_POWER_LIMIT_WATTS"] = (
        f"{power_limits[0]:.2f}"
    )
    with CotenantSampler(owner_token) as sampler:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    # Preserve every arm log beside its trial artifact. A
    # failed activation is part of the artifact's diagnosis; reporting only
    # "no engine stats" makes an otherwise reproducible failure impossible to
    # audit after the parent process exits. Including the output stem and seed
    # prevents a multi-trial campaign from overwriting an earlier arm.
    log_path = (
        args.output.resolve().parent
        / "logs"
        / f"{args.output.stem}.seed-{args.seed}.{role}.stdout.log"
    )
    atomic_write_text(log_path, completed.stdout)
    environment_evidence, failures = trial_environment_evidence(
        sampler,
        expected_power_limit_watts=power_limits[0],
        start_max_temperature_c=args.gpu_start_max_temperature_c,
    )
    if completed.returncode:
        failures.append(f"worker exited with status {completed.returncode}")
    environment_path = log_path.with_suffix(".environment.json")
    atomic_write_json(
        environment_path,
        {
            "schema": 1,
            "classification": "sglang-hicache-arm-environment",
            "backend": backend,
            "returncode": completed.returncode,
            "failures": failures,
            **environment_evidence,
        },
    )
    if failures:
        raise RuntimeError(
            f"{backend} load trial failed ({'; '.join(failures)}):\n"
            + "\n".join(completed.stdout.splitlines()[-120:])
        )
    report = _report(completed.stdout)
    report.update(environment_evidence)
    report["arm_environment"] = str(environment_path)
    report["arm_log"] = str(log_path)
    return report


def _write_failed_comparison(
    output: pathlib.Path,
    reports: dict[str, dict[str, Any]],
    order: list[str],
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    failure = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison-failure",
        "execution_order": order,
        "reason": reason,
        "stock": reports["flashinfer"],
        "nta": reports["nta_flashinfer"],
    }
    if diagnostics:
        failure["diagnostics"] = diagnostics
    atomic_write_json(output, failure)


def main() -> int:
    args = parse_args()
    if args.prepare_nta_calibration_profile:
        run(
            args,
            "nta_flashinfer",
            calibration_profile_access="writable",
            trial_label="nta_calibration",
        )
    order = ["flashinfer", "nta_flashinfer"]
    if args.execution_order == "seeded":
        random.Random(args.seed).shuffle(order)
    elif args.execution_order == "nta_first":
        order.reverse()
    reports = {backend: run(args, backend) for backend in order}
    stock = reports["flashinfer"]
    nta = reports["nta_flashinfer"]
    contaminated = {
        backend: {
            "foreign_samples": int(report.get("cotenant_gpu_samples", 0)),
            "foreign_pids": report.get("cotenant_pids_seen", []),
        }
        for backend, report in reports.items()
        if int(report.get("cotenant_gpu_samples", 0)) > 0
    }
    if contaminated:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "co-tenant GPU activity contaminated a serving arm",
            {"contaminated_arms": contaminated},
        )
        raise RuntimeError(
            "serving comparison was contaminated by a foreign GPU process: "
            + json.dumps(contaminated, sort_keys=True)
        )
    try:
        activation = require_clean_mechanism(
            nta,
            require_graph_replay=args.cuda_graph_decode == "full",
            require_demand_graph=args.require_demand_graph,
            require_physical_compaction=args.batch_mode == "coalesced",
            require_read_only_calibration_profile=(
                args.nta_calibration_profile is not None
            ),
        )
        batch_heterogeneity = nta.get("batch_heterogeneity")
        activation["batch_heterogeneity_proven"] = bool(
            isinstance(batch_heterogeneity, dict)
            and batch_heterogeneity.get("proven") is True
        )
    except RuntimeError as error:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "NTA mechanism activation failed",
            {"activation_error": str(error)},
        )
        raise
    if not stock.get("load_warmup_excluded") or not nta.get("load_warmup_excluded"):
        _write_failed_comparison(
            args.output, reports, order, "mixed-arrival warmup was not excluded"
        )
        raise RuntimeError("load trial did not exclude mixed-arrival graph warmup")
    calibration_contracts = {
        backend: report.get("calibration_input_contract")
        for backend, report in reports.items()
    }
    invalid_calibration = {
        backend: contract
        for backend, contract in calibration_contracts.items()
        if not isinstance(contract, dict)
        or contract.get("kind") != "exact_token_prefix_and_query_rows"
        or contract.get("verified") is not True
    }
    if invalid_calibration:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "paired warmups did not prove an exact token-prefix/query-row contract",
            {"invalid_calibration_contracts": invalid_calibration},
        )
        raise RuntimeError("load warmup did not preserve the exact request shape")
    stock_calibration = calibration_contracts["flashinfer"]
    nta_calibration = calibration_contracts["nta_flashinfer"]
    assert isinstance(stock_calibration, dict) and isinstance(nta_calibration, dict)
    for field in (
        "materialized_prefix_tokens",
        "cached_prefix_tokens",
        "uncached_query_rows",
        "timed_shapes",
    ):
        if stock_calibration.get(field) != nta_calibration.get(field):
            _write_failed_comparison(
                args.output,
                reports,
                order,
                f"paired warmup calibration disagreed on {field}",
            )
            raise RuntimeError(
                f"stock and NTA calibration contracts disagree on {field}"
            )
    if not stock["placement_proven"] or not nta["placement_proven"]:
        _write_failed_comparison(
            args.output, reports, order, "cache placement was not proven"
        )
        raise RuntimeError("load trial did not prove cache placement")
    for backend, report in reports.items():
        if int(report.get("verification_failures", -1)) != 0:
            _write_failed_comparison(
                args.output,
                reports,
                order,
                f"{backend} reported verification failures",
            )
            raise RuntimeError(f"{backend} reported verification failures")
        accounting = report.get("finite_window_accounting")
        required_accounting = (
            "arrival_rate_per_second",
            "completion_rate_per_second",
            "mean_in_system",
            "mean_system_time_seconds",
            "occupancy_area_request_seconds",
            "sum_residence_seconds",
        )
        if (
            not isinstance(accounting, dict)
            or accounting.get("method")
            != "finite_window_arrival_departure_accounting"
            or accounting.get("interpretation")
            != "descriptive_client_timestamp_accounting"
            or not all(
                math.isfinite(float(accounting.get(field, float("nan"))))
                for field in required_accounting
            )
        ):
            _write_failed_comparison(
                args.output,
                reports,
                order,
                f"{backend} omitted finite-window client timestamp accounting",
            )
            raise RuntimeError(
                f"{backend} omitted finite-window client timestamp accounting"
            )
    if stock["batch_mode"] != args.batch_mode or nta["batch_mode"] != args.batch_mode:
        _write_failed_comparison(
            args.output, reports, order, "requested batch mode was not preserved"
        )
        raise RuntimeError("load trial did not preserve the requested batch mode")
    if args.workload_manifest is not None:
        stock_workload = stock.get("workload")
        nta_workload = nta.get("workload")
        if not stock_workload or not nta_workload:
            _write_failed_comparison(
                args.output, reports, order, "normalized workload was not replayed"
            )
            raise RuntimeError("normalized workload was not replayed")
        paired_identity_fields = (
            "manifest_digest",
            "demand_trace_digest",
            "token_input_identity_digest",
            "external_suffix_tokens",
        )
        if any(
            stock_workload.get(field) != nta_workload.get(field)
            for field in paired_identity_fields
        ):
            _write_failed_comparison(
                args.output,
                reports,
                order,
                "paired arms used different workload manifests",
            )
            raise RuntimeError("paired arms used different workload manifests")
    outputs_diverge = stock["generated_text_sha256"] != nta["generated_text_sha256"]
    if outputs_diverge and not args.allow_output_divergence:
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
    try:
        transport_execution = require_exercised_paths(
            stats, args.require_exercised_path
        )
        require_frontier_shape(
            transport_execution,
            native_layers=args.require_native_frontier_layers,
            ready_stock_layers=args.require_ready_stock_layers,
            progressive_layers=args.require_progressive_layers,
        )
    except ValueError as error:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "required physical execution path was not exercised",
            {"execution_error": str(error)},
        )
        raise
    considered = sum(
        int(entry.get("admission_considered_batches", 0)) for entry in stats
    )
    admission_bytes = sum(
        int(entry.get("admission_external_bytes", 0)) for entry in stats
    )
    delayed = sum(int(entry.get("admission_delayed_batches", 0)) for entry in stats)
    if considered == 0 or admission_bytes == 0:
        _write_failed_comparison(
            args.output,
            reports,
            order,
            "acquisition-aware admission was not exercised",
        )
        raise RuntimeError(
            "NTA load trial did not exercise acquisition-aware admission"
        )
    thresholds = relative_thresholds(stock, args.slo_scale)
    stock_goodput = relative_goodput(stock, thresholds)
    nta_goodput = relative_goodput(nta, thresholds)
    stock_prereg = preregistered_goodput(stock)
    nta_prereg = preregistered_goodput(nta)
    stock_joint_prereg = preregistered_joint_goodput(stock)
    nta_joint_prereg = preregistered_joint_goodput(nta)
    stock_rate = float(stock["output_token_throughput"])
    nta_rate = float(nta["output_token_throughput"])
    stock_resident_rate = float(stock["resident_output_token_throughput"])
    nta_resident_rate = float(nta["resident_output_token_throughput"])
    stock_external_rate = float(stock["external_output_token_throughput"])
    nta_external_rate = float(nta["external_output_token_throughput"])
    stock_gp = float(stock_goodput["goodput_requests_per_second"])
    nta_gp = float(nta_goodput["goodput_requests_per_second"])
    resident_ttft_ratio = safe_ratio(
        float(nta["resident_p95_ttft_seconds"]),
        float(stock["resident_p95_ttft_seconds"]),
    )
    resident_tpot_ratio = safe_ratio(
        float(nta["resident_p95_tpot_seconds"]),
        float(stock["resident_p95_tpot_seconds"]),
    )
    resident_itl_ratio = safe_ratio(
        float(nta["resident_p99_itl_seconds"]),
        float(stock["resident_p99_itl_seconds"]),
    )
    external_ttft_ratio = safe_ratio(
        float(nta["external_p95_ttft_seconds"]),
        float(stock["external_p95_ttft_seconds"]),
    )
    harness_args = {
        key: (str(value) if isinstance(value, pathlib.Path) else value)
        for key, value in sorted(vars(args).items())
        if key not in ("output", "seed", "execution_order")
    }
    # Record explicit experiment-level overrides.  ``auto`` means the runtime
    # production default selected the execution form from its cost model.
    harness_args.update(
        {
            "nta_execution_max_rounds": os.environ.get(
                "NTA_COMPARE_EXECUTION_MAX_ROUNDS", "auto"
            ),
            "nta_execution_min_predicted_gain": os.environ.get(
                "NTA_COMPARE_EXECUTION_MIN_PREDICTED_GAIN", "auto"
            ),
            "nta_execution_protocol": os.environ.get(
                "NTA_COMPARE_EXECUTION_PROTOCOL", "late_bound"
            ),
            "nta_execution_host_form": os.environ.get(
                "NTA_COMPARE_EXECUTION_HOST_FORM", "auto"
            ),
            "nta_execution_host_mover": os.environ.get(
                "NTA_EXECUTION_HOST_MOVER", "auto"
            ),
        }
    )
    evidence_scope = (
        "heterogeneous_work_unit"
        if activation["heterogeneous_work_unit_active"]
        and activation["batch_heterogeneity_proven"]
        else "native_work_unit"
        if activation["native_work_unit_active"]
        else "transport_only"
        if activation["transport_only"]
        else "exact_execution_only"
    )
    comparison = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison",
        "evidence_scope": evidence_scope,
        "execution_order": order,
        "outputs_diverge": outputs_diverge,
        # Trial identity for strict resume validation: the revision both
        # arms ran and the full workload-shaping argument set (seed and
        # order are validated separately; output is location-only).
        "revision": (
            os.environ.get("NTA_REVISION")
            or str(nta.get("revision") or stock.get("revision") or "")
        ),
        "harness_args": harness_args,
        "nta_selected_bytes": sum(
            int(entry.get("tier_selected_bytes", 0)) for entry in stats
        ),
        "nta_candidate_bytes": sum(
            int(entry.get("tier_candidate_bytes", 0)) for entry in stats
        ),
        "nta_work_selected_bytes": sum(
            int(entry.get("work_selected_bytes", 0)) for entry in stats
        ),
        "nta_work_candidate_bytes": sum(
            int(entry.get("work_candidate_bytes", 0)) for entry in stats
        ),
        "nta_staged_bytes": (
            int(nta["physical_bytes"])
            if isinstance(nta.get("physical_bytes"), int)
            else None
        ),
        "batch_mode": args.batch_mode,
        "slo_scale": args.slo_scale,
        "slo_ttft_seconds": args.slo_ttft_seconds,
        "slo_tpot_seconds": args.slo_tpot_seconds,
        "slo_p99_itl_seconds": args.slo_p99_itl_seconds,
        "incremental_setup_ns": args.incremental_setup_ns,
        "external_suffix_tokens": args.external_suffix_tokens,
        "slo_thresholds_seconds": thresholds,
        "stock": stock,
        "nta": nta,
        "stock_goodput": stock_goodput,
        "nta_goodput": nta_goodput,
        "stock_slo_goodput": float(stock["slo_goodput"]["goodput_requests_per_second"]),
        "nta_slo_goodput": float(nta["slo_goodput"]["goodput_requests_per_second"]),
        "stock_p50_ttft_seconds": float(stock["p50_ttft_seconds"]),
        "stock_p95_ttft_seconds": float(stock["p95_ttft_seconds"]),
        "stock_p99_ttft_seconds": float(stock["p99_ttft_seconds"]),
        "stock_p99_itl_seconds": float(stock["p99_itl_seconds"]),
        "nta_p50_ttft_seconds": float(nta["p50_ttft_seconds"]),
        "nta_p95_ttft_seconds": float(nta["p95_ttft_seconds"]),
        "nta_p99_ttft_seconds": float(nta["p99_ttft_seconds"]),
        "nta_p99_itl_seconds": float(nta["p99_itl_seconds"]),
        "stock_preregistered_goodput": stock_prereg,
        "nta_preregistered_goodput": nta_prereg,
        "stock_preregistered_joint_goodput": stock_joint_prereg,
        "nta_preregistered_joint_goodput": nta_joint_prereg,
        "output_throughput_ratio": safe_ratio(nta_rate, stock_rate),
        "resident_output_throughput_ratio": safe_ratio(
            nta_resident_rate, stock_resident_rate
        ),
        "external_output_throughput_ratio": safe_ratio(
            nta_external_rate, stock_external_rate
        ),
        "goodput_ratio": safe_ratio(nta_gp, stock_gp),
        "preregistered_goodput_ratio": safe_ratio(
            float(nta_prereg["goodput_requests_per_second"]),
            float(stock_prereg["goodput_requests_per_second"]),
        ),
        "preregistered_joint_goodput_ratio": safe_ratio(
            float(nta_joint_prereg["goodput_requests_per_second"]),
            float(stock_joint_prereg["goodput_requests_per_second"]),
        ),
        "resident_p95_ttft_ratio": resident_ttft_ratio,
        "resident_p95_tpot_ratio": resident_tpot_ratio,
        "resident_p99_itl_ratio": resident_itl_ratio,
        "external_p95_ttft_ratio": external_ttft_ratio,
        "mechanism_activation": activation,
        "transport_execution": transport_execution,
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
    # A standalone paired run is itself an artifact-producing command.  Apply
    # the same closed evidence contract used by the repeated-trial wrapper so
    # an internally inconsistent result can never be printed as a success.
    validate_serving_report(comparison)
    atomic_write_json(args.output, comparison)
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
