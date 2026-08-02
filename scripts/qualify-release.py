#!/usr/bin/env python3
"""Execute NTA release gates and emit a machine-readable claim verdict."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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


def verify_artifacts(
    evidence: dict[str, Any], required_classes: set[str], revision: str
) -> tuple[bool, str]:
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        return False, "evidence must contain an artifact manifest"
    observed_classes: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False, "artifact manifest entries must be objects"
        classification = artifact.get("class")
        relative_path = artifact.get("path")
        expected_digest = artifact.get("sha256")
        if not all(
            isinstance(value, str)
            for value in (classification, relative_path, expected_digest)
        ):
            return False, "artifact entries require class, path, and sha256 strings"
        path = (ROOT / relative_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            return False, f"artifact escapes the repository: {relative_path}"
        if not path.is_file():
            return False, f"artifact is missing: {relative_path}"
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            return False, f"artifact digest mismatch: {relative_path}"
        try:
            if path.suffix == ".jsonl":
                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                records = value if isinstance(value, list) else [value]
        except (OSError, json.JSONDecodeError) as error:
            return False, f"artifact is not valid JSON data: {relative_path}: {error}"
        if not records or not all(isinstance(record, dict) for record in records):
            return False, f"artifact has no structured records: {relative_path}"
        if any(record.get("revision") != revision for record in records):
            return False, f"artifact record revision mismatch: {relative_path}"
        observed_classes.add(classification)
    missing = sorted(required_classes - observed_classes)
    if missing:
        return False, f"artifact classes are missing: {', '.join(missing)}"
    return True, "all required raw artifacts exist and match their SHA-256 digests"


def production_checks(
    evidence_dir: pathlib.Path, revision: str
) -> list[dict[str, Any]]:
    evidence, error = read_json(
        evidence_dir / "production-evidence.json", expected_schema=2
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
    artifacts_valid, artifact_detail = verify_artifacts(
        evidence,
        {"serving", "correctness", "reliability", "portability"},
        revision,
    )
    return [
        check(
            "production evidence provenance",
            evidence.get("revision") == revision and artifacts_valid,
            artifact_detail
            if evidence.get("revision") == revision
            else "evidence revision does not match the qualified revision",
        ),
        check(
            "serving integration",
            serving.get("engine") in {"sglang", "vllm"}
            and serving.get("mechanism_integrated") is True
            and serving.get("mechanism_mode") == "incremental_demand"
            and serving.get("correctness") is True
            and serving.get("transfer_verification") is True
            and serving.get("all_attention_layers_executed") is True
            and serving.get("baseline_and_mechanism") is True
            and serving.get("zero_fallback") is True
            and serving.get("post_acquisition_instrumented_launches") == 0
            and serving.get("matched_cache_and_admission") is True,
            "requires a real demand-mode integration with complete execution, "
            "verified transfers, matched state, and zero fallback",
        ),
        check(
            "serving graph path",
            serving.get("decode_cuda_graph_replay") is True
            and serving.get("paged_prefill_integrated") is True,
            "requires demand-mode decode graph replay and paged-prefill integration",
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
            sequence_contains(serving.get("tiers"), {"host_staged", "nvme"}),
            "requires real CPU-DRAM and NVMe serving paths",
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
        evidence_dir / "osdi-evidence.json", expected_schema=2
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
    artifacts_valid, artifact_detail = verify_artifacts(
        evidence,
        {
            "opportunity",
            "dense_flashinfer",
            "sparse_flashinfer",
            "baselines",
            "ablations",
            "statistics",
            "reproduction",
        },
        revision,
    )
    opportunity = evidence.get("opportunity", {})
    compiler = evidence.get("compiler", {})
    incremental = evidence.get("incremental_execution", {})
    scheduler = evidence.get("scheduler", {})
    sparse = evidence.get("sparse_flashinfer", {})
    performance = evidence.get("performance", {})
    return [
        check(
            "OSDI evidence provenance",
            evidence.get("revision") == revision and artifacts_valid,
            artifact_detail
            if evidence.get("revision") == revision
            else "evidence revision does not match the qualified revision",
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
            and compiler.get("differential_correctness") is True,
            "requires sound same-source forms for at least two generated-kernel families",
        ),
        check(
            "real FlashInfer incremental execution",
            incremental.get("decode") is True
            and incremental.get("paged_prefill") is True
            and incremental.get("partial_before_last_arrival") is True
            and incremental.get("stock_output_parity") is True
            and incremental.get("demand_cuda_graph_replay") is True,
            "requires real decode/prefill partial execution and demand graph replay",
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
            "real GPU-selected sparse stress",
            sparse.get("gpu_selected_pages") is True
            and sparse.get("nta_hot_path_host_identity_round_trips") == 0
            and sparse.get("real_flashinfer_selector") is True
            and sparse.get("real_flashinfer_attention") is True
            and sparse.get("stock_output_parity") is True
            and sparse.get("candidate_sweep_points", 0) >= 5
            and sparse.get("selectivity_crossover_measured") is True
            and isinstance(sparse.get("peak_speedup_over_overfetch"), (int, float))
            and sparse["peak_speedup_over_overfetch"] >= 2.0
            and isinstance(sparse.get("maximum_regret_to_offline_oracle"), (int, float))
            and sparse["maximum_regret_to_offline_oracle"] <= 2.0,
            "requires real FlashInfer selection/attention, a five-point crossover, "
            "a matched overfetch win, and bounded regret to an offline oracle",
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


def workspace_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(git_value("rev-parse", "HEAD").encode("ascii"))
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout
    digest.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout.split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        path = ROOT / os.fsdecode(encoded_path)
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    fingerprint = workspace_fingerprint()
    revision = git_value("rev-parse", "HEAD")
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
            and prior.get("workspace_fingerprint") == fingerprint
        )
        prior_local_detail = (
            "cached local evidence matches this workspace"
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
    dirty = bool(git_value("status", "--porcelain"))
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
        "workspace_fingerprint": fingerprint,
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
