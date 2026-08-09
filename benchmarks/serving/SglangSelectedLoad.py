#!/usr/bin/env python3
"""First end-to-end comparison for selected-demand serving (1D stage 4).

Runs the identical HiCache load workload through three arms and reports the
comparative metrics the selected-demand thesis is judged by:

  stock       SGLang's FlashInfer backend, bulk promotion, dense attention.
  nta_dense   NTA backend on its default path: bulk promotion, direct form.
              Must match stock output exactly; this is the never-lose arm.
  nta_tiered  Claim-time envelopes, no bulk promotion, per-layer device
              selection, bounded indexed staging, selected attention.
              Output is approximate by design; its quality is judged by the
              separate quality harness, not by text equality here.

The round-trip-ablation and prediction-prefetch arms are deliberately absent
until the device-resident selection optimization lands: the current tiered
path still pays per-layer host orchestration, so those ablations would not
yet measure what they claim to measure.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOAD = ROOT / "benchmarks" / "serving" / "SglangHiCacheLoad.py"

ARMS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("stock", "flashinfer", {}),
    ("nta_dense", "nta_flashinfer", {}),
    (
        "nta_tiered",
        "nta_flashinfer",
        {
            "NTA_SGLANG_SELECTED_SERVE": "1",
            "NTA_SGLANG_SELECTED_TIERED": "1",
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--external-tokens", type=int, default=16384)
    parser.add_argument("--resident-tokens", type=int, default=2048)
    parser.add_argument("--resident-output-tokens", type=int, default=64)
    parser.add_argument("--external-output-tokens", type=int, default=16)
    parser.add_argument("--churn-tokens", type=int, default=17000)
    parser.add_argument("--max-total-tokens", type=int, default=19000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--selected-budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.selected_budget <= 0:
        parser.error("selected budget must be positive")
    return args


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def run_arm(
    args: argparse.Namespace, name: str, backend: str, extra: dict[str, str]
) -> dict[str, Any]:
    workspace = ROOT / "results" / "serving" / "sglang-hicache-load-cache" / backend
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir", args.build_dir,
        "--cache-root", str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable, str(LOAD),
        "--model", str(args.model.resolve()),
        "--attention-backend", backend,
        "--external-requests", "1",
        "--external-tokens", str(args.external_tokens),
        "--resident-requests", "1",
        "--resident-tokens", str(args.resident_tokens),
        "--resident-output-tokens", str(args.resident_output_tokens),
        "--external-output-tokens", str(args.external_output_tokens),
        "--request-rate", str(args.request_rate),
        "--churn-tokens", str(args.churn_tokens),
        "--max-total-tokens", str(args.max_total_tokens),
        "--context-length", str(args.context_length),
        "--batch-mode", "separate",
        "--seed", str(args.seed),
        "--flashinfer-workspace-base", str(workspace / "flashinfer"),
    ]
    environment = os.environ.copy()
    environment.update(extra)
    if "NTA_SGLANG_SELECTED_TIERED" in extra:
        environment["NTA_SGLANG_SELECTED_BUDGET"] = str(args.selected_budget)
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{name} arm failed:\n"
            + "\n".join(completed.stdout.splitlines()[-80:])
        )
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if report.get("classification") == "sglang-hicache-load":
            return report
    raise RuntimeError(f"{name} arm emitted no load report")


def tiered_counters(report: dict[str, Any]) -> dict[str, Any]:
    for entry in report.get("engine_stats", ()):
        if "tiered_claims" in entry:
            return {
                key: value
                for key, value in entry.items()
                if key.startswith("tiered")
            }
    return {}


def metric(report: dict[str, Any], key: str) -> float:
    return float(report[key])


def main() -> int:
    args = parse_args()
    arm_reports: dict[str, dict[str, Any]] = {}
    for name, backend, extra in ARMS:
        arm_reports[name] = run_arm(args, name, backend, extra)

    stock = arm_reports["stock"]
    dense = arm_reports["nta_dense"]
    tiered = arm_reports["nta_tiered"]
    if (
        stock["generated_text_sha256"] != dense["generated_text_sha256"]
    ):
        raise RuntimeError(
            "the dense NTA arm must reproduce stock output exactly"
        )
    counters = tiered_counters(tiered)
    kept = int(counters.get("tiered_tokens_kept", 0))
    total = int(counters.get("tiered_tokens_total", 0))
    rows_copied = int(counters.get("tiered_rows_copied", 0))

    def ratios(base: dict[str, Any], subject: dict[str, Any]) -> dict[str, float]:
        return {
            "external_p95_ttft_ratio": metric(subject, "external_p95_ttft_seconds")
            / metric(base, "external_p95_ttft_seconds"),
            "resident_p99_itl_ratio": metric(subject, "resident_p99_itl_seconds")
            / metric(base, "resident_p99_itl_seconds"),
            "resident_p95_tpot_ratio": metric(subject, "resident_p95_tpot_seconds")
            / metric(base, "resident_p95_tpot_seconds"),
        }

    report = {
        "schema": 1,
        "classification": "sglang-selected-load-comparison",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "model": str(args.model),
        "external_tokens": args.external_tokens,
        "selected_budget_pages": args.selected_budget,
        "seed": args.seed,
        "dense_output_exact": True,
        "tiered_output_sha256": tiered["generated_text_sha256"],
        "tiered_tokens_avoided_fraction": (
            (1.0 - kept / total) if total else None
        ),
        "tiered_rows_copied": rows_copied,
        "tiered_counters": counters,
        "tiered_vs_stock": ratios(stock, tiered),
        "tiered_vs_dense": ratios(dense, tiered),
        "dense_vs_stock": ratios(stock, dense),
        "arms": {
            name: {
                "external_p95_ttft_seconds": metric(r, "external_p95_ttft_seconds"),
                "resident_p99_itl_seconds": metric(r, "resident_p99_itl_seconds"),
                "resident_p95_tpot_seconds": metric(r, "resident_p95_tpot_seconds"),
                "generated_text_sha256": r["generated_text_sha256"],
            }
            for name, r in arm_reports.items()
        },
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
