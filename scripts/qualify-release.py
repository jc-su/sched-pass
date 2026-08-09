#!/usr/bin/env python3
"""Execute NTA release gates and emit a machine-readable claim verdict."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]
    environment: dict[str, str] | None = None


def run(command: Command) -> dict[str, Any]:
    environment = os.environ.copy()
    if command.environment:
        environment.update(command.environment)
    started = time.monotonic()
    result = subprocess.run(
        command.argv,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = result.stdout
    return {
        "name": command.name,
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "command": command.argv,
        "output_tail": output[-12000:],
    }


def read_json(
    path: pathlib.Path, *, expected_schema: int = 1
) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"missing evidence file: {path}"
    except (OSError, json.JSONDecodeError) as error:
        return None, f"invalid evidence file {path}: {error}"
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        return None, (
            f"evidence must use schema {expected_schema}: {path}"
        )
    return value, ""


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def sequence_contains(value: Any, required: set[str]) -> bool:
    return isinstance(value, list) and required.issubset(
        {item for item in value if isinstance(item, str)}
    )


def production_checks(
    evidence_dir: pathlib.Path, revision: str
) -> list[dict[str, Any]]:
    evidence, error = read_json(
        evidence_dir / "production-evidence.json", expected_schema=3
    )
    if evidence is None:
        return [check("production evidence", False, error)]

    serving = evidence.get("serving", {})
    reliability = evidence.get("reliability", {})
    portability = evidence.get("portability", {})
    required_metrics = {
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "slo_attainment",
        "goodput",
        "cpu_utilization",
        "sm_utilization",
    }
    required_faults = {
        "nvme_status",
        "malformed_cqe",
        "timeout",
        "controller_reset",
        "iommu_fault",
        "process_crash",
    }
    provenance_valid = (
        evidence.get("revision") == revision and evidence.get("dirty") is False
    )
    return [
        check(
            "production evidence provenance",
            provenance_valid,
            "evidence is from the exact clean revision"
            if provenance_valid
            else "evidence must report the exact revision and dirty=false",
        ),
        check(
            "serving integration",
            serving.get("engine") in {"sglang", "vllm"}
            and serving.get("mechanism_integrated") is True
            and serving.get("mechanism_mode") == "request_aware_dual_form"
            and serving.get("correctness") is True
            and serving.get("transfer_verification") is True
            and serving.get("all_attention_layers_executed") is True
            and serving.get("baseline_and_mechanism") is True
            and serving.get("zero_fallback") is True
            and serving.get("all_attention_transformed") is True
            and serving.get("bounded_hbm_tier_streaming") is True
            and serving.get("generation_safe_request_completion") is True
            and serving.get("jit_cache_primed") is True
            and serving.get("compiler_contract_verified") is True
            and serving.get("compiler_plan_verified") is True
            and serving.get("verified_operator_modules", 0) > 0
            and serving.get("verified_operator_pairs", 0) > 0
            and serving.get("verified_operator_plan_pairs", 0) > 0
            and serving.get("transformed_direct_launches", 0) > 0
            and serving.get("ticketed_incremental_launches", 0) > 0
            and serving.get("stock_attention_launches") == 0
            and serving.get("matched_cache_and_admission") is True,
            "requires both compiler-generated forms in a real integration with "
            "complete execution, verified transfers, matched state, no stock "
            "dispatch, generation-safe bounded-HBM streaming, verified compiler "
            "contracts, warm JIT caches, and zero fallback",
        ),
        check(
            "serving graph path",
            serving.get("decode_cuda_graph_replay") is True
            and serving.get("paged_prefill_integrated") is True
            and serving.get("demand_operator_graph_replay") is True
            and serving.get("demand_graph_captures", 0) > 0
            and serving.get("demand_graph_replays", 0) > 0,
            "requires whole-model decode replay, paged-prefill integration, and "
            "positive finite demand-operator graph capture/replay counters",
        ),
        check(
            "serving performance bounds",
            isinstance(serving.get("resident_p50_overhead_fraction"), (int, float))
            and serving["resident_p50_overhead_fraction"] <= 0.05
            and isinstance(serving.get("dense_bulk_throughput_ratio"), (int, float))
            and serving["dense_bulk_throughput_ratio"] >= 0.90,
            "requires <=5% resident overhead and >=90% of matched dense bulk throughput",
        ),
        check(
            "serving tier coverage",
            sequence_contains(serving.get("tiers"), {"host_staged", "nvme"})
            and serving.get("simultaneous_host_nvme") is True
            and serving.get("cancellation_and_slot_reuse") is True,
            "requires simultaneous CPU-DRAM/NVMe serving plus cancellation "
            "and request-slot reuse",
        ),
        check(
            "serving coverage",
            serving.get("model_count", 0) >= 2
            and serving.get("trace_count", 0) >= 3
            and serving.get("soak_hours", 0) >= 24,
            "requires at least two models, three traces, and a 24-hour soak",
        ),
        check(
            "serving metrics",
            sequence_contains(serving.get("metrics"), required_metrics),
            "requires TTFT, TPOT, SLO, goodput, CPU, and SM metrics",
        ),
        check(
            "fault recovery",
            sequence_contains(reliability.get("faults"), required_faults)
            and reliability.get("zero_leaks") is True
            and reliability.get("bounded_recovery") is True,
            "requires injected transport/process faults with bounded recovery",
        ),
        check(
            "hardware portability",
            portability.get("machine_count", 0) >= 2
            and portability.get("gpu_model_count", 0) >= 2
            and portability.get("nvme_model_count", 0) >= 2
            and portability.get("multi_gpu") is True,
            "requires two machines, two GPU and SSD models, and multi-GPU",
        ),
    ]


def osdi_checks(evidence_dir: pathlib.Path, revision: str) -> list[dict[str, Any]]:
    evidence, error = read_json(
        evidence_dir / "osdi-evidence.json", expected_schema=3
    )
    if evidence is None:
        return [check("OSDI evidence", False, error)]

    required_baselines = {
        "untouched_flashinfer",
        "compiler_direct",
        "layer_complete_prefetch",
        "coalesced_bulk",
        "request_skip_rebatch",
        "forced_fine_incremental",
        "best_fixed_hindsight",
        "cpu_completion",
        "persistent_gpu_progress",
    }
    required_ablations = {
        "request_semantics",
        "runnable_tile_set",
        "incremental_kernel_form",
        "complete_contributor_merge",
        "elastic_grouping",
        "replica_selection",
        "engine_progress_feedback",
        "cta_try_issue",
    }
    provenance_valid = (
        evidence.get("revision") == revision and evidence.get("dirty") is False
    )
    opportunity = evidence.get("opportunity", {})
    compiler = evidence.get("compiler", {})
    incremental = evidence.get("incremental_execution", {})
    scheduler = evidence.get("scheduler", {})
    sparse = evidence.get("sparse_flashinfer", {})
    performance = evidence.get("performance", {})
    workload = evidence.get("workload", {})
    return [
        check(
            "OSDI evidence provenance",
            provenance_valid,
            "evidence is from the exact clean revision"
            if provenance_valid
            else "evidence must report the exact revision and dirty=false",
        ),
        check(
            "comparative baselines",
            sequence_contains(evidence.get("baselines"), required_baselines),
            "requires matched prefetch, direct, overlap, CPU, and persistent baselines",
        ),
        check(
            "mechanism ablations",
            sequence_contains(evidence.get("ablations"), required_ablations),
            "requires all compiler, semantic, transport, and scheduling ablations",
        ),
        check(
            "statistical methodology",
            evidence.get("independent_trials", 0) >= 10
            and evidence.get("confidence_intervals") is True
            and evidence.get("controlled_clocks") is True,
            "requires controlled clocks, at least ten trials, and intervals",
        ),
        check(
            "measured dense opportunity",
            opportunity.get("model_count", 0) >= 2
            and sequence_contains(opportunity.get("tiers"), {"host_staged", "nvme"})
            and opportunity.get("gpu_timestamped_arrivals") is True
            and opportunity.get("material_barrier_cost") is True,
            "requires GPU-timestamped dense traces from two models and both external tiers",
        ),
        check(
            "compiler-generated forms",
            compiler.get("same_source_direct_and_incremental") is True
            and compiler.get("generated_kernel_family_count", 0) >= 2
            and compiler.get("convergence_verified") is True
            and compiler.get("acquired_edge_identity_verified") is True
            and compiler.get("exactly_once_publication_verified") is True
            and compiler.get("differential_correctness") is True
            and compiler.get("versioned_operator_contracts") is True
            and compiler.get("typed_operator_plans") is True
            and compiler.get("runtime_abi_verified") is True
            and compiler.get("direct_incremental_source_fingerprint_match") is True
            and compiler.get("direct_incremental_plan_fingerprint_match") is True
            and compiler.get("online_softmax_plan_verified") is True,
            "requires sound same-source forms, convergence, identity continuity, "
            "and exactly-once publication for at least two generated-kernel families",
        ),
        check(
            "real FlashInfer incremental execution",
            incremental.get("decode") is True
            and incremental.get("paged_prefill") is True
            and incremental.get("canonical_flashinfer_attention") is True
            and incremental.get("custom_attention_kernel") is False
            and incremental.get("partial_before_last_arrival") is True
            and incremental.get("generation_safe_request_completion") is True
            and incremental.get("stock_output_parity") is True
            and incremental.get("demand_cuda_graph_replay") is True
            and sequence_contains(
                incremental.get("demand_graph_families"),
                {"decode", "paged_prefill"},
            )
            and incremental.get("demand_graph_captures", 0) > 0
            and incremental.get("demand_graph_replays", 0) > 0
            and incremental.get("all_attention_transformed") is True
            and incremental.get("transformed_direct_launches", 0) > 0
            and incremental.get("ticketed_incremental_launches", 0) > 0
            and incremental.get("stock_attention_launches") == 0
            and isinstance(
                incremental.get("bounded_hbm_staging_reduction"), (int, float)
            )
            and incremental["bounded_hbm_staging_reduction"] >= 4.0
            and isinstance(
                incremental.get("speedup_over_atomic_promotion"), (int, float)
            )
            and incremental["speedup_over_atomic_promotion"] >= 1.15
            and isinstance(incremental.get("speedup_95ci_lower"), (int, float))
            and incremental["speedup_95ci_lower"] > 1.0,
            "requires canonical FlashInfer decode/prefill partial execution, "
            "graph replay, generation-safe request completion, a bounded-HBM "
            "crossover, and proof that no NTA arm dispatched stock attention",
        ),
        check(
            "heterogeneous serving workload",
            workload.get("mixed_resident_external_requests") is True
            and workload.get("heterogeneous_context_and_prefix") is True
            and workload.get("fragmented_kv_placement") is True
            and workload.get("simultaneous_host_nvme") is True
            and workload.get("admission_churn") is True
            and workload.get("cancellation_and_slot_reuse") is True,
            "requires the complete long-context/agent batch-barrier scenario, "
            "not separate homogeneous tier microbenchmarks",
        ),
        check(
            "unified scheduler and engine feedback",
            scheduler.get("single_elastic_policy") is True
            and scheduler.get("engine_admission_feedback") is True
            and isinstance(scheduler.get("median_decision_regret"), (int, float))
            and scheduler["median_decision_regret"] <= 1.05
            and isinstance(scheduler.get("p95_decision_regret"), (int, float))
            and scheduler["p95_decision_regret"] <= 1.10,
            "requires one online policy, admission feedback, and bounded identical-snapshot regret",
        ),
        check(
            "real GPU-selected FlashInfer acquisition",
            sparse.get("gpu_selected_pages") is True
            and sparse.get("nta_hot_path_host_identity_round_trips") == 0
            and sparse.get("real_flashinfer_selector") is True
            and sparse.get("real_flashinfer_attention") is True
            and sparse.get("all_policy_attention_transformed") is True
            and sparse.get("paired_operator_contract_verified") is True
            and sparse.get("stock_output_parity") is True
            and sparse.get("candidate_sweep_points", 0) >= 5
            and sparse.get("selectivity_crossover_measured") is True
            and isinstance(sparse.get("peak_speedup_over_overfetch"), (int, float))
            and sparse["peak_speedup_over_overfetch"] >= 2.0
            and isinstance(
                sparse.get("peak_speedup_bootstrap_95_percent_ci"), list
            )
            and len(sparse["peak_speedup_bootstrap_95_percent_ci"]) == 2
            and sparse["peak_speedup_bootstrap_95_percent_ci"][0] > 1.0
            and isinstance(sparse.get("maximum_online_policy_regret"), (int, float))
            and sparse["maximum_online_policy_regret"] <= 1.05
            and sparse.get("policy_regret_definition")
            == "same_trial_chosen_over_best"
            and sparse.get("candidate_retained_baseline") is True
            and isinstance(
                sparse.get(
                    "minimum_cold_indexed_latency_ratio_to_candidate_retained"
                ),
                (int, float),
            )
            and sparse[
                "minimum_cold_indexed_latency_ratio_to_candidate_retained"
            ] >= 1.0
            and sparse.get("no_selectivity_policy_mode") == "bulk"
            and isinstance(sparse.get("no_selectivity_speedup"), (int, float))
            and sparse["no_selectivity_speedup"] >= 0.99
            and isinstance(
                sparse.get("no_selectivity_forced_indexed_throughput_ratio"),
                (int, float),
            )
            and isinstance(sparse.get("maximum_regret_to_offline_oracle"), (int, float))
            and sparse["maximum_regret_to_offline_oracle"] <= 2.0,
            "requires paired transformed FlashInfer forms, a five-point crossover, "
            "a confidence-bounded overfetch win, same-trial policy regret, "
            "explicit forced-indexed and resident-candidate costs, and bounded oracle regret",
        ),
        check(
            "mechanism performance bounds",
            isinstance(performance.get("resident_p50_overhead_fraction"), (int, float))
            and performance["resident_p50_overhead_fraction"] <= 0.05
            and isinstance(performance.get("dense_bulk_throughput_ratio"), (int, float))
            and performance["dense_bulk_throughput_ratio"] >= 0.90
            and performance.get("end_to_end_gain_over_skip_rebatch") is True,
            "requires direct-path, dense-transfer, and equal-state serving gates",
        ),
        check(
            "artifact reproduction",
            evidence.get("artifact_reproduced_on_clean_host") is True
            and evidence.get("raw_results_published") is True,
            "requires clean-host reproduction and published raw results",
        ),
    ]


def local_commands(
    build: pathlib.Path, cpu_build: pathlib.Path, evidence_dir: pathlib.Path
) -> list[Command]:
    jobs = os.environ.get("NTA_BUILD_JOBS", str(min(8, max(1, os.cpu_count() or 1))))
    static_build = evidence_dir / "scan-build"
    llvm_dir = os.environ.get("LLVM_DIR", "/usr/lib/llvm-22/lib/cmake/llvm")
    clang_cuda = os.environ.get("NTA_CLANG_CUDA", "/usr/bin/clang++-22")
    cuda_root = os.environ.get("NTA_CUDA_ROOT", "/usr/local/cuda-12.9")
    sanitizer_env = {
        "NTA_BUILD_DIR": str(build),
        "NTA_ITERATIONS": "1",
        "NTA_REQUESTS": "3",
        "NTA_SANITIZE": "1",
    }
    shell_scripts = sorted(str(path) for path in (ROOT / "scripts").glob("*.sh"))
    shell_scripts.append(str(ROOT / "tests/ir/run.sh"))
    return [
        Command(
            "configure",
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(build),
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DLLVM_DIR={llvm_dir}",
                f"-DNTA_CLANG_CUDA={clang_cuda}",
                f"-DCUDAToolkit_ROOT={cuda_root}",
                f"-DNTA_CUDA_ROOT={cuda_root}",
                f"-DNTA_CUDA_ARCH={os.environ.get('NTA_CUDA_ARCH', 'sm_120')}",
            ],
        ),
        Command("build", ["cmake", "--build", str(build), f"-j{jobs}"]),
        Command(
            "functional and sanitizer matrix",
            [str(ROOT / "scripts/validate-local.sh")],
            sanitizer_env,
        ),
        Command(
            "CPU-only configure",
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(cpu_build),
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DLLVM_DIR={llvm_dir}",
                "-DNTA_ENABLE_CUDA=OFF",
            ],
        ),
        Command("CPU-only build", ["cmake", "--build", str(cpu_build), f"-j{jobs}"]),
        Command(
            "CPU-only tests",
            ["ctest", "--test-dir", str(cpu_build), "--output-on-failure"],
        ),
        Command(
            "10k lifecycle stress",
            [
                str(build / "nta-kv-bench"),
                "--mode=mixed",
                "--requests=48",
                "--coalesce=3",
                "--dependencies=4",
                "--tile-bytes=8192",
                "--iterations=1",
                "--lifecycle-epochs=10000",
                "--cancel-stride=11",
                "--stale-stride=13",
            ],
        ),
        Command(
            "static-analysis configure",
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(static_build),
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DLLVM_DIR={llvm_dir}",
                f"-DNTA_CLANG_CUDA={clang_cuda}",
                f"-DCUDAToolkit_ROOT={cuda_root}",
                f"-DNTA_CUDA_ROOT={cuda_root}",
                f"-DNTA_CUDA_ARCH={os.environ.get('NTA_CUDA_ARCH', 'sm_120')}",
            ],
        ),
        Command(
            "Clang static analysis",
            [
                "scan-build-22",
                "--status-bugs",
                "-o",
                str(evidence_dir / "scan-reports"),
                "cmake",
                "--build",
                str(static_build),
                "--clean-first",
                "--target",
                "nta-runtime",
                "nta-runtime-host",
                "nta-kv-bench",
                "nta-moe-bench",
                f"-j{jobs}",
            ],
        ),
        Command(
            "Python analysis",
            [
                "ruff",
                "check",
                str(ROOT / "python"),
                str(ROOT / "benchmarks" / "serving"),
                str(ROOT / "scripts"),
                str(ROOT / "tests" / "runtime"),
                str(ROOT / "tools"),
            ],
        ),
        Command(
            "Python package",
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(evidence_dir / "wheel"),
                str(ROOT),
            ],
        ),
        Command("shell analysis", ["shellcheck", *shell_scripts]),
        Command("patch hygiene", ["git", "diff", "--check"]),
    ]


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("local", "production", "osdi"), default="local"
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--cpu-build-dir", type=pathlib.Path, default=ROOT / "build-cpu"
    )
    parser.add_argument(
        "--evidence-dir", type=pathlib.Path, default=ROOT / "results/qualification"
    )
    parser.add_argument("--skip-local", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    command_results = []
    prior_local_ready = False
    prior_local_detail = "local gates were not executed"
    revision = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    if not args.skip_local:
        for command in local_commands(
            args.build_dir.resolve(), args.cpu_build_dir.resolve(), evidence_dir
        ):
            result = run(command)
            command_results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {result['name']} ({result['duration_seconds']:.3f}s)")
            if not result["passed"]:
                break
    else:
        prior, _ = read_json(evidence_dir / "local-qualification.json")
        prior_local_ready = (
            prior is not None
            and prior.get("ready") is True
            and prior.get("revision") == revision
            and prior.get("dirty") is False
            and not dirty
        )
        prior_local_detail = (
            "cached local evidence matches this clean revision"
            if prior_local_ready
            else "cached local evidence is missing, failed, or stale"
        )

    checks = [
        check(
            "local executable gates",
            prior_local_ready
            or (
                bool(command_results)
                and all(item["passed"] for item in command_results)
            ),
            prior_local_detail
            if args.skip_local
            else "all build, CTest, sanitizer, stress, and hygiene commands must pass",
        )
    ]
    if args.profile in {"production", "osdi"}:
        checks.extend(production_checks(evidence_dir, revision))
    if args.profile == "osdi":
        checks.extend(osdi_checks(evidence_dir, revision))
    if args.profile != "local":
        checks.append(
            check(
                "immutable revision",
                not dirty,
                "release evidence requires a clean commit",
            )
        )

    report = {
        "schema": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": args.profile,
        "revision": revision,
        "branch": git_value("branch", "--show-current"),
        "dirty": dirty,
        "commands": command_results,
        "checks": checks,
        "ready": all(item["passed"] for item in checks),
    }
    report_path = evidence_dir / f"{args.profile}-qualification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verdict={'READY' if report['ready'] else 'NOT_READY'}")
    print(f"report={report_path}")
    if not report["ready"]:
        for item in checks:
            if not item["passed"]:
                print(f"missing={item['name']}: {item['detail']}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
