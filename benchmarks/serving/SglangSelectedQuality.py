#!/usr/bin/env python3
"""Compare selected-demand budgets on scored SGLang external-prefix tasks."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any

from SglangSelectedLoad import tiered_counters, validate_tiered_activation


ROOT = pathlib.Path(__file__).resolve().parents[2]
QUALITY = ROOT / "benchmarks" / "serving" / "SglangHiCacheQuality.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--external-tokens", type=int, default=16384)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--resident-tokens", type=int, default=2048)
    parser.add_argument("--resident-output-tokens", type=int, default=64)
    parser.add_argument("--churn-tokens", type=int, default=17000)
    parser.add_argument("--max-total-tokens", type=int, default=19000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--selected-budgets", default="32,64,128")
    parser.add_argument("--selected-page-tokens", type=int, default=16)
    parser.add_argument("--selection-refresh-interval", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--arm-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--min-stock-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-quality-delta", type=float, default=0.01)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    budgets = tuple(
        int(value) for value in args.selected_budgets.split(",") if value
    )
    if not budgets or min(budgets) <= 0:
        parser.error("selected budgets must be positive")
    args.selected_budgets = budgets
    if min(
        args.external_tokens,
        args.task_count,
        args.max_new_tokens,
        args.resident_tokens,
        args.resident_output_tokens,
        args.churn_tokens,
        args.max_total_tokens,
        args.context_length,
        args.selected_page_tokens,
    ) <= 0:
        parser.error("token counts must be positive")
    if not 0.0 <= args.min_stock_pass_rate <= 1.0:
        parser.error("minimum stock pass rate must be in [0, 1]")
    if not 0.0 <= args.max_quality_delta <= 1.0:
        parser.error("maximum quality delta must be in [0, 1]")
    return args


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def run_arm(
    args: argparse.Namespace,
    *,
    name: str,
    backend: str,
    extra: dict[str, str],
) -> dict[str, Any]:
    workspace = (
        ROOT
        / "results"
        / "serving"
        / "sglang-quality-cache"
        / name
    )
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(QUALITY),
        "--model",
        str(args.model.resolve()),
        "--attention-backend",
        backend,
        "--external-tokens",
        str(args.external_tokens),
        "--task-count",
        str(args.task_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--resident-tokens",
        str(args.resident_tokens),
        "--resident-output-tokens",
        str(args.resident_output_tokens),
        "--churn-tokens",
        str(args.churn_tokens),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--seed",
        str(args.seed),
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    environment = os.environ.copy()
    environment.update(extra)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.arm_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        raise RuntimeError(
            f"{name} arm exceeded {args.arm_timeout_seconds:.1f}s:\n"
            + "\n".join(output.splitlines()[-80:])
        ) from error
    if process.returncode:
        raise RuntimeError(
            f"{name} arm failed:\n" + "\n".join(output.splitlines()[-80:])
        )
    for line in reversed(output.splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if report.get("classification") == "sglang-hicache-quality":
            return report
    raise RuntimeError(f"{name} arm emitted no quality report")


def normalize_finished_tiered_counters(report: dict[str, Any]) -> dict[str, Any]:
    counters = tiered_counters(report)
    if counters.get("tiered_rows_copied", 0) == 0:
        counters["tiered_rows_copied"] = counters.get(
            "tiered_rows_copied_released", 0
        )
    if counters.get("tiered_rows_rehit", 0) == 0:
        counters["tiered_rows_rehit"] = counters.get(
            "tiered_rows_rehit_released", 0
        )
    return {"engine_stats": [counters]}


def main() -> int:
    args = parse_args()
    stock = run_arm(args, name="stock", backend="flashinfer", extra={})
    stock_pass_rate = float(stock["pass_rate"])
    if not bool(stock["all_tasks_host_served"]):
        raise RuntimeError("stock quality arm did not exercise host-cache load")
    if stock_pass_rate < args.min_stock_pass_rate:
        raise RuntimeError(
            f"stock quality pass rate {stock_pass_rate:.3f} below "
            f"{args.min_stock_pass_rate:.3f}; task set is not valid"
        )
    budget_reports = {}
    for budget in args.selected_budgets:
        report = run_arm(
            args,
            name=f"budget{budget}",
            backend="nta_flashinfer",
            extra={
                "NTA_SGLANG_SELECTED_SERVE": "1",
                "NTA_SGLANG_SELECTED_TIERED": "1",
                "NTA_SGLANG_SELECTED_BUDGET": str(budget),
                "NTA_SGLANG_SELECTED_PAGE_TOKENS": str(args.selected_page_tokens),
                "NTA_SGLANG_SELECTED_REFRESH_INTERVAL": str(
                    args.selection_refresh_interval
                ),
            },
        )
        _, activation = validate_tiered_activation(
            normalize_finished_tiered_counters(report),
            require_overlap=False,
        )
        pass_rate = float(report["pass_rate"])
        quality_delta = stock_pass_rate - pass_rate
        budget_reports[str(budget)] = {
            "pass_rate": pass_rate,
            "quality_delta": quality_delta,
            "quality_parity": quality_delta <= args.max_quality_delta,
            "all_tasks_host_served": bool(report["all_tasks_host_served"]),
            "generated_text_sha256": report["generated_text_sha256"],
            "records": report["records"],
            "mechanism_activation": activation,
        }
    result = {
        "schema": 1,
        "classification": "sglang-selected-quality-comparison",
        "revision": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "model": str(args.model.resolve()),
        "external_tokens": args.external_tokens,
        "resident_tokens": args.resident_tokens,
        "task_count": args.task_count,
        "selected_page_tokens": args.selected_page_tokens,
        "selection_refresh_interval": args.selection_refresh_interval,
        "stock": {
            "pass_rate": stock_pass_rate,
            "all_tasks_host_served": bool(stock["all_tasks_host_served"]),
            "generated_text_sha256": stock["generated_text_sha256"],
            "records": stock["records"],
        },
        "budgets": budget_reports,
    }
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
