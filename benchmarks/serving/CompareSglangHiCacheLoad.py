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
import threading
import time
from typing import Any

from CompareSglangHiCache import require_clean_mechanism


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = pathlib.Path(os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=pathlib.Path,
        help="normalized Bailian manifest replayed identically by both arms",
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
    parser.add_argument("--external-output-tokens", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--churn-tokens", type=int, default=12000)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--chunked-prefill-size", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument(
        "--eviction-rounds",
        type=int,
        help=(
            "explicit cache-churn rounds forwarded to both arms; zero disables "
            "churn for a capacity-fit workload"
        ),
    )
    parser.add_argument("--load-warmup-iterations", type=int, default=2)
    parser.add_argument(
        "--batch-mode",
        choices=("coalesced", "separate"),
        default="coalesced",
    )
    parser.add_argument("--slo-scale", type=float, default=1.5)
    parser.add_argument("--slo-ttft-seconds", type=float, default=8.0)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
    parser.add_argument("--admission-lead-layers", type=int, default=36)
    parser.add_argument("--admission-max-delay-us", type=int, default=10000)
    parser.add_argument(
        "--incremental-setup-ns",
        type=int,
        default=0,
        help=(
            "modeled setup cost for the mechanism stress arm; zero forces the "
            "trial to expose the request-overlap path but does not remove real cost"
        ),
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
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "sglang-hicache-load.json",
    )
    args = parser.parse_args()
    if (
        args.slo_scale <= 0
        or args.slo_ttft_seconds <= 0
        or args.slo_p99_itl_seconds <= 0
    ):
        parser.error("SLO scale and thresholds must be positive")
    if args.admission_lead_layers <= 0 or args.admission_max_delay_us < 0:
        parser.error("admission bounds are invalid")
    if args.eviction_rounds is not None and args.eviction_rounds < 0:
        parser.error("eviction rounds cannot be negative")
    if args.load_warmup_iterations < 0:
        parser.error("load warmup iterations cannot be negative")
    if args.incremental_setup_ns < 0:
        parser.error("incremental setup cost must be nonnegative")
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("--mem-fraction-static must be between zero and one")
    if args.external_suffix_tokens < 0:
        parser.error("external suffix token count cannot be negative")
    if args.churn_tokens > args.context_length:
        parser.error(
            "--churn-tokens cannot exceed --context-length; choose a smaller "
            "churn request or a larger model context"
        )
    if args.external_tokens + args.external_suffix_tokens > args.context_length:
        parser.error(
            "--external-tokens plus --external-suffix-tokens cannot exceed "
            "--context-length"
        )
    if args.resident_tokens > args.context_length:
        parser.error("--resident-tokens cannot exceed --context-length")
    return args


def _wait_for_free_gpu(limit_mib: int = 8000, timeout_s: float = 600.0) -> None:
    """Block until GPU memory drops below limit_mib.

    The previous arm's scheduler subprocesses release device memory a few
    seconds after their parent exits, and co-tenant jobs on the shared box come
    and go; launching a server into that window fails its memory profile with a
    misleading mem-fraction error. Startup ordering only — no timed phase has
    begun when this runs.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        try:
            used_mib = max(int(line) for line in probe.stdout.split() if line.strip())
        except ValueError:
            used_mib = None
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        compute_apps = [line for line in apps.stdout.split() if line.strip()]
        if probe.returncode == 0 and used_mib is not None:
            # Memory alone misses an idle-but-resident co-tenant process;
            # a live compute app also disqualifies the device.
            if used_mib < limit_mib and not compute_apps:
                return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory still at {used_mib} MiB after {timeout_s:.0f}s; "
                "refusing to launch a serving arm into an occupied device"
            )
        time.sleep(5.0)


def _report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("classification") == "sglang-hicache-load":
            return value
    raise RuntimeError("load trial emitted no JSON report")


class _CotenantSampler:
    """Sample foreign GPU compute apps during one arm's run.

    The pre-run wait-gate only proves the GPU was free at launch; on this
    shared box co-tenant jobs can land mid-trial and arbitrarily inflate
    whichever arm they overlap. Every report therefore carries the number
    of one-second samples that saw a compute app outside our process tree,
    so contaminated trials are identifiable by an objective environmental
    criterion instead of by their metric values.
    """

    def __init__(self, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("co-tenant sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.samples = 0
        self.sampling_errors = 0
        self.foreign_samples = 0
        self.foreign_pids: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _descendants(self) -> set[int]:
        pids = {os.getpid()}
        try:
            children = pathlib.Path("/proc")
            grew = True
            while grew:
                grew = False
                for stat in children.glob("[0-9]*/stat"):
                    try:
                        parts = stat.read_text().rsplit(") ", 1)[1].split()
                        pid = int(stat.parent.name)
                        ppid = int(parts[1])
                    except (OSError, IndexError, ValueError):
                        continue
                    if ppid in pids and pid not in pids:
                        pids.add(pid)
                        grew = True
        except OSError:
            pass
        return pids

    def _sample_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                self.sampling_errors += 1
                return
            apps = {int(line) for line in out.stdout.split() if line.strip().isdigit()}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            self.sampling_errors += 1
            return
        self.samples += 1
        foreign = apps - self._descendants()
        if foreign:
            self.foreign_samples += 1
            self.foreign_pids |= foreign

    def _loop(self) -> None:
        # Sample immediately so a co-tenant that arrived during the launch
        # gate cannot hide inside the first sampling interval.
        while not self._stop.is_set():
            self._sample_once()
            if self._stop.wait(self.interval_seconds):
                return

    def __enter__(self) -> "_CotenantSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=7)

    @property
    def complete(self) -> bool:
        return not self._thread.is_alive()


def run(args: argparse.Namespace, backend: str) -> dict[str, Any]:
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
    command.extend(("--load-warmup-iterations", str(args.load_warmup_iterations)))
    if args.workload_manifest is not None:
        command.extend(("--workload-manifest", str(args.workload_manifest.resolve())))
    if args.allow_oversubscribed_pool:
        command.append("--allow-oversubscribed-pool")
    environment = os.environ.copy()
    environment["NTA_EXECUTION_ADMISSION"] = "1"
    environment["NTA_EXECUTION_ADMISSION_LEAD_LAYERS"] = str(args.admission_lead_layers)
    environment["NTA_EXECUTION_ADMISSION_MAX_DELAY_US"] = str(
        args.admission_max_delay_us
    )
    if backend == "nta_flashinfer" and args.batch_mode == "coalesced":
        # Exercise the actual request-aware finite-kernel path. One transfer
        # wave isolates overlap from deeper transfer pipelining: resident CTAs
        # run immediately, then only the externally dependent CTAs resume.
        protocol = os.environ.get("NTA_COMPARE_EXECUTION_PROTOCOL", "late_bound")
        prefetch = os.environ.get("NTA_COMPARE_EXECUTION_PREFETCH", "0")
        max_rounds = os.environ.get("NTA_COMPARE_EXECUTION_MAX_ROUNDS", "1")
        environment.update(
            {
                "NTA_EXECUTION_PREFETCH": prefetch,
                "NTA_EXECUTION_PROTOCOL": protocol,
                "NTA_EXECUTION_MAX_ROUNDS": max_rounds,
                "NTA_EXECUTION_MIN_PREDICTED_GAIN": "1.0",
                "NTA_EXECUTION_INCREMENTAL_SETUP_NS": str(args.incremental_setup_ns),
            }
        )
    _wait_for_free_gpu()
    with _CotenantSampler() as sampler:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    # Preserve the complete arm log beside its isolated JIT workspace.  A
    # failed activation is part of the artifact's diagnosis; reporting only
    # "no engine stats" makes an otherwise reproducible failure impossible to
    # audit after the parent process exits.
    log_path = workspace.parent / f"{workspace.name}.{backend}.stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    failures: list[str] = []
    if not sampler.complete:
        failures.append("co-tenant sampler did not terminate cleanly")
    if sampler.sampling_errors:
        failures.append(
            "co-tenant sampler lost environmental samples: "
            f"{sampler.sampling_errors} errors"
        )
    if completed.returncode:
        failures.append(f"worker exited with status {completed.returncode}")
    if failures:
        raise RuntimeError(
            f"{backend} load trial failed ({'; '.join(failures)}):\n"
            + "\n".join(completed.stdout.splitlines()[-120:])
        )
    report = _report(completed.stdout)
    report["cotenant_gpu_samples"] = sampler.foreign_samples
    report["gpu_samples"] = sampler.samples
    report["gpu_sampling_errors"] = sampler.sampling_errors
    report["gpu_sampling_complete"] = sampler.complete
    report["cotenant_pids_seen"] = sorted(sampler.foreign_pids)
    return report


def _thresholds(stock: dict[str, Any], scale: float) -> dict[str, float]:
    return {
        "resident_ttft": scale * float(stock["resident_p95_ttft_seconds"]),
        "resident_tpot": scale * float(stock["resident_p95_tpot_seconds"]),
        "resident_itl": scale * float(stock["resident_p99_itl_seconds"]),
        "external_ttft": scale * float(stock["external_p95_ttft_seconds"]),
    }


def _preregistered_goodput(report: dict[str, Any]) -> dict[str, Any]:
    """The registered primary metric: completed requests per second whose
    TTFT <= 8.0s and P99 ITL <= 100ms; both request kinds count and a
    request violating either threshold contributes zero."""
    qualified = 0
    total = 0
    for record in report["records"]:
        total += 1
        if (
            float(record["ttft_seconds"]) <= 8.0
            and float(record["p99_itl_seconds"]) <= 0.100
        ):
            qualified += 1
    elapsed = float(report["elapsed_seconds"])
    return {
        "qualified_requests": qualified,
        "total_requests": total,
        "goodput_requests_per_second": qualified / elapsed,
    }


def _ratio(numerator: float, denominator: float) -> float:
    """Return a finite neutral ratio for metrics with no interval samples."""
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
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
        little = report.get("littles_law")
        if not isinstance(little, dict) or not math.isfinite(
            float(little.get("residual", float("nan")))
        ):
            _write_failed_comparison(
                args.output,
                reports,
                order,
                f"{backend} omitted finite-window Little's Law evidence",
            )
            raise RuntimeError(f"{backend} omitted finite-window Little's Law evidence")
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
        if stock_workload.get("manifest_digest") != nta_workload.get(
            "manifest_digest"
        ) or stock_workload.get("demand_trace_digest") != nta_workload.get(
            "demand_trace_digest"
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
    thresholds = _thresholds(stock, args.slo_scale)
    stock_goodput = _goodput(stock, thresholds)
    nta_goodput = _goodput(nta, thresholds)
    stock_prereg = _preregistered_goodput(stock)
    nta_prereg = _preregistered_goodput(nta)
    stock_rate = float(stock["output_token_throughput"])
    nta_rate = float(nta["output_token_throughput"])
    stock_gp = float(stock_goodput["goodput_requests_per_second"])
    nta_gp = float(nta_goodput["goodput_requests_per_second"])
    resident_ttft_ratio = _ratio(
        float(nta["resident_p95_ttft_seconds"]),
        float(stock["resident_p95_ttft_seconds"]),
    )
    resident_tpot_ratio = _ratio(
        float(nta["resident_p95_tpot_seconds"]),
        float(stock["resident_p95_tpot_seconds"]),
    )
    resident_itl_ratio = _ratio(
        float(nta["resident_p99_itl_seconds"]),
        float(stock["resident_p99_itl_seconds"]),
    )
    external_ttft_ratio = _ratio(
        float(nta["external_p95_ttft_seconds"]),
        float(stock["external_p95_ttft_seconds"]),
    )
    harness_args = {
        key: (str(value) if isinstance(value, pathlib.Path) else value)
        for key, value in sorted(vars(args).items())
        if key not in ("output", "seed", "execution_order")
    }
    # These are experiment-level controls rather than runtime defaults. Record
    # them in the report so a banked trial cannot be mistaken for the default
    # late-bound variant when the prefetch/control ablation changes.
    harness_args.update(
        {
            "nta_execution_max_rounds": os.environ.get(
                "NTA_COMPARE_EXECUTION_MAX_ROUNDS", "1"
            ),
            "nta_execution_prefetch": os.environ.get(
                "NTA_COMPARE_EXECUTION_PREFETCH", "0"
            ),
            "nta_execution_protocol": os.environ.get(
                "NTA_COMPARE_EXECUTION_PROTOCOL", "late_bound"
            ),
        }
    )
    comparison = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison",
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
            int(entry.get("work_selected_bytes", 0)) for entry in stats
        ),
        "nta_candidate_bytes": sum(
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
        "output_throughput_ratio": _ratio(nta_rate, stock_rate),
        "goodput_ratio": nta_gp / stock_gp if stock_gp else None,
        "preregistered_goodput_ratio": (
            nta_prereg["goodput_requests_per_second"]
            / stock_prereg["goodput_requests_per_second"]
            if stock_prereg["goodput_requests_per_second"]
            else None
        ),
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
