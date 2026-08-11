#!/usr/bin/env python3
"""End-to-end diagnostic for selected-demand SGLang serving.

Runs the identical HiCache load workload through stock and tiered NTA arms,
with an optional dense NTA diagnostic, and reports the comparative metrics the
selected-demand thesis is judged by:

  stock       SGLang's FlashInfer backend, bulk promotion, dense attention.
  nta_dense   NTA backend on its default path: bulk promotion, direct form.
              Optional diagnostic; when enabled it must match stock output
              exactly.
  nta_tiered  Claim-time envelopes, no bulk promotion, per-layer device
              selection, validated indexed staging, selected attention.
              Output is approximate by design; its quality is judged by the
              separate quality harness, not by text equality here.

This harness fails if external-prefix allocation, selected acquisition, GPU
miss compaction, or the compiler-generated request-bound consumer did not
execute. Live/high-water counters must prove that bounded staging used fewer
physical rows than the represented dense prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOAD = ROOT / "benchmarks" / "serving" / "SglangHiCacheLoad.py"

BASE_ARMS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("stock", "flashinfer", {}),
    (
        "nta_tiered",
        "nta_flashinfer",
        {
            "NTA_SGLANG_SELECTED_SERVE": "1",
            "NTA_SGLANG_SELECTED_TIERED": "1",
        },
    ),
)

DENSE_DIAGNOSTIC_ARM: tuple[str, str, dict[str, str]] = (
    "nta_dense",
    "nta_flashinfer",
    {},
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
    parser.add_argument("--selected-page-tokens", type=int, default=16)
    parser.add_argument("--selection-refresh-interval", type=int, default=1)
    parser.add_argument("--selector-quality-report", type=pathlib.Path)
    parser.add_argument("--min-selector-mean-recall", type=float)
    parser.add_argument("--min-selector-min-layer-recall", type=float)
    parser.add_argument("--max-selector-oracle-gap", type=float)
    parser.add_argument(
        "--batch-mode", choices=("coalesced", "separate"), default="coalesced"
    )
    parser.add_argument(
        "--require-request-overlap",
        action="store_true",
        help="require the split peer/external request-overlap form to execute",
    )
    parser.add_argument("--load-warmup-iterations", type=int, default=2)
    parser.add_argument("--arm-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-hicache-load-cache",
        help="base directory for per-backend JIT and engine reports",
    )
    parser.add_argument(
        "--include-dense-diagnostic",
        action="store_true",
        help=(
            "also run the legacy dense NTA exact-output diagnostic before the "
            "tiered selected arm"
        ),
    )
    parser.add_argument(
        "--require-tiered-output-match",
        action="store_true",
        help=(
            "require the selected tiered arm to reproduce stock text on this "
            "deterministic workload; this is a smoke quality gate, not a "
            "general proof for approximate attention"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.selected_budget <= 0 or args.selected_page_tokens <= 0:
        parser.error("selected budget and page tokens must be positive")
    if args.selection_refresh_interval <= 0:
        parser.error("selection refresh interval must be positive")
    for name in (
        "min_selector_mean_recall",
        "min_selector_min_layer_recall",
        "max_selector_oracle_gap",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"{name.replace('_', '-')} must be in [0, 1]")
    if args.load_warmup_iterations < 0 or args.arm_timeout_seconds <= 0:
        parser.error("warmup count must be nonnegative and timeout positive")
    return args


def selected_arms(
    args: argparse.Namespace,
) -> tuple[tuple[str, str, dict[str, str]], ...]:
    if not args.include_dense_diagnostic:
        return BASE_ARMS
    return (BASE_ARMS[0], DENSE_DIAGNOSTIC_ARM, BASE_ARMS[1])


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def run_arm(
    args: argparse.Namespace, name: str, backend: str, extra: dict[str, str]
) -> dict[str, Any]:
    workspace = args.cache_root / backend
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
        "--batch-mode", args.batch_mode,
        "--seed", str(args.seed),
        "--flashinfer-workspace-base", str(workspace / "flashinfer"),
        "--load-warmup-iterations", str(args.load_warmup_iterations),
    ]
    environment = os.environ.copy()
    environment.update(extra)
    if "NTA_SGLANG_SELECTED_TIERED" in extra:
        environment["NTA_SGLANG_SELECTED_BUDGET"] = str(args.selected_budget)
        environment["NTA_SGLANG_SELECTED_PAGE_TOKENS"] = str(
            args.selected_page_tokens
        )
        environment["NTA_SGLANG_SELECTED_REFRESH_INTERVAL"] = str(
            args.selection_refresh_interval
        )
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
            f"{name} arm failed:\n"
            + "\n".join(output.splitlines()[-80:])
        )
    for line in reversed(output.splitlines()):
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
                or key.startswith("external_")
                or key
                in {
                    "selected_compiler_launches",
                    "hicache_fallback_batches",
                    "stock_attention_launches",
                }
            }
    return {}


def validate_tiered_activation(
    report: dict[str, Any],
    *,
    require_overlap: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    counters = tiered_counters(report)
    kept = int(counters.get("tiered_tokens_kept", 0))
    total = int(counters.get("tiered_tokens_total", 0))
    activation = {
        "claims": int(counters.get("tiered_claims", 0)),
        "device_compaction_launches": int(
            counters.get("tiered_device_compaction_launches", 0)
        ),
        "bounded_cache_launches": int(
            counters.get("tiered_bounded_cache_launches", 0)
        ),
        "request_overlap_layers": int(
            counters.get("tiered_request_overlap_layers", 0)
        ),
        "overlapped_peer_requests": int(
            counters.get("tiered_request_overlap_peer_requests", 0)
        ),
        "compiler_attention_launches": int(
            counters.get("selected_compiler_launches", 0)
        ),
        "rows_copied": int(counters.get("tiered_rows_copied", 0)),
        "rows_reused": int(counters.get("tiered_rows_rehit", 0)),
        "selection_reuse_layers": int(
            counters.get("tiered_selection_reuse_layers", 0)
        ),
        "external_prefix_claims": int(
            counters.get("external_prefix_claims", 0)
        ),
        "dense_slots_avoided": int(
            counters.get("external_dense_slots_avoided", 0)
        ),
        "staging_slots": int(counters.get("external_staging_slots", 0)),
        "admission_credit_rows": int(
            counters.get("external_admission_credit_rows", 0)
        ),
        "dense_high_water_rows": int(
            counters.get("external_dense_high_water_rows", 0)
        ),
        "staging_high_water_rows": int(
            counters.get("external_staging_high_water_rows", 0)
        ),
        "fallback_batches": int(counters.get("hicache_fallback_batches", 0)),
        "stock_attention_launches": int(
            counters.get("stock_attention_launches", 0)
        ),
    }
    failed = [
        name
        for name in (
            "claims",
            "device_compaction_launches",
            "bounded_cache_launches",
            "compiler_attention_launches",
            "rows_copied",
            "external_prefix_claims",
            "dense_slots_avoided",
            "staging_slots",
            "admission_credit_rows",
            "dense_high_water_rows",
            "staging_high_water_rows",
        )
        if activation[name] <= 0
    ]
    if activation["rows_reused"] <= 0 and activation["selection_reuse_layers"] <= 0:
        failed.append("rows_reused_or_selection_reuse_layers")
    if require_overlap and activation["request_overlap_layers"] <= 0:
        failed.append("request_overlap_layers")
    if require_overlap and activation["overlapped_peer_requests"] <= 0:
        failed.append("overlapped_peer_requests")
    if failed:
        raise RuntimeError(
            "tiered arm bypassed required mechanisms: " + ", ".join(failed)
        )
    if activation["fallback_batches"] != 0:
        raise RuntimeError("tiered arm reported a HiCache fallback")
    if activation["stock_attention_launches"] != 0:
        raise RuntimeError(
            "tiered arm bypassed the compiler-generated attention path"
        )
    if activation["staging_high_water_rows"] >= activation["dense_high_water_rows"]:
        raise RuntimeError("tiered arm did not reduce live KV allocation")
    if total <= 0 or not 0 < kept < total:
        raise RuntimeError("tiered arm did not exercise selective attention")
    return counters, activation


def metric(report: dict[str, Any], key: str) -> float:
    return float(report[key])


def selector_quality_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.selector_quality_report is None:
        if any(
            value is not None
            for value in (
                args.min_selector_mean_recall,
                args.min_selector_min_layer_recall,
                args.max_selector_oracle_gap,
            )
        ):
            raise RuntimeError(
                "selector quality thresholds require --selector-quality-report"
            )
        return None
    try:
        quality = json.loads(
            args.selector_quality_report.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read selector quality report: {args.selector_quality_report}"
        ) from error
    if quality.get("classification") != "quest-attention-mass-recall":
        raise RuntimeError("selector quality report has the wrong classification")
    model = pathlib.Path(str(quality.get("model", ""))).resolve()
    if model != args.model.resolve():
        raise RuntimeError(
            f"selector quality model mismatch: {model} != {args.model.resolve()}"
        )
    if int(quality.get("prompt_tokens", 0)) != args.external_tokens:
        raise RuntimeError(
            "selector quality context length does not match external tokens"
        )
    if int(quality.get("page_tokens", 0)) != args.selected_page_tokens:
        raise RuntimeError("selector quality page size does not match serving run")
    aggregate = quality.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError("selector quality report omitted aggregate recall")
    quest_key = f"quest_recall_at_{args.selected_budget}"
    oracle_key = f"oracle_recall_at_{args.selected_budget}"
    quest = aggregate.get(quest_key)
    oracle = aggregate.get(oracle_key)
    if not isinstance(quest, dict) or not isinstance(oracle, dict):
        raise RuntimeError(
            f"selector quality report omitted budget {args.selected_budget}"
        )
    quest_mean = float(quest.get("mean"))
    quest_min_layer = float(quest.get("min_layer"))
    oracle_mean = float(oracle.get("mean"))
    oracle_min_layer = float(oracle.get("min_layer"))
    oracle_gap = max(0.0, oracle_mean - quest_mean)
    result = {
        "report": str(args.selector_quality_report),
        "budget_pages": args.selected_budget,
        "page_tokens": args.selected_page_tokens,
        "quest_mean_recall": quest_mean,
        "quest_min_layer_recall": quest_min_layer,
        "oracle_mean_recall": oracle_mean,
        "oracle_min_layer_recall": oracle_min_layer,
        "oracle_mean_gap": oracle_gap,
    }
    if (
        args.min_selector_mean_recall is not None
        and quest_mean < args.min_selector_mean_recall
    ):
        raise RuntimeError(
            "selector mean recall below gate: "
            f"{quest_mean:.4f} < {args.min_selector_mean_recall:.4f}"
        )
    if (
        args.min_selector_min_layer_recall is not None
        and quest_min_layer < args.min_selector_min_layer_recall
    ):
        raise RuntimeError(
            "selector min-layer recall below gate: "
            f"{quest_min_layer:.4f} < {args.min_selector_min_layer_recall:.4f}"
        )
    if (
        args.max_selector_oracle_gap is not None
        and oracle_gap > args.max_selector_oracle_gap
    ):
        raise RuntimeError(
            "selector oracle gap above gate: "
            f"{oracle_gap:.4f} > {args.max_selector_oracle_gap:.4f}"
        )
    return result


def main() -> int:
    args = parse_args()
    selector_quality = selector_quality_gate(args)
    arm_reports: dict[str, dict[str, Any]] = {}
    for name, backend, extra in selected_arms(args):
        arm_reports[name] = run_arm(args, name, backend, extra)

    stock = arm_reports["stock"]
    tiered = arm_reports["nta_tiered"]
    dense = arm_reports.get("nta_dense")
    if dense is not None:
        if stock["generated_text_sha256"] != dense["generated_text_sha256"]:
            raise RuntimeError(
                "the dense NTA arm must reproduce stock output exactly"
            )
    tiered_matches_stock = (
        stock["generated_text_sha256"] == tiered["generated_text_sha256"]
    )
    if args.require_tiered_output_match and not tiered_matches_stock:
        raise RuntimeError(
            "the selected tiered arm changed generated text for the "
            "deterministic quality-smoke workload"
        )
    counters, activation = validate_tiered_activation(
        tiered, require_overlap=args.require_request_overlap
    )
    kept = int(counters.get("tiered_tokens_kept", 0))
    total = int(counters.get("tiered_tokens_total", 0))
    rows_copied = int(counters.get("tiered_rows_copied", 0))
    rows_reused = int(counters.get("tiered_rows_rehit", 0))
    rows_requested = rows_copied + rows_reused

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
        "schema": 2,
        "classification": "sglang-selected-load-comparison",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "model": str(args.model),
        "external_tokens": args.external_tokens,
        "selected_budget_pages": args.selected_budget,
        "selected_page_tokens": args.selected_page_tokens,
        "selection_refresh_interval": args.selection_refresh_interval,
        "selector_quality": selector_quality,
        "batch_mode": args.batch_mode,
        "request_overlap_required": args.require_request_overlap,
        "seed": args.seed,
        "dense_output_exact": dense is not None,
        "tiered_output_matches_stock": tiered_matches_stock,
        "tiered_output_sha256": tiered["generated_text_sha256"],
        "tiered_tokens_avoided_fraction": (
            (1.0 - kept / total) if total else None
        ),
        "tiered_rows_copied": rows_copied,
        "tiered_rows_reused": rows_reused,
        "tiered_physical_copy_fraction": (
            rows_copied / rows_requested if rows_requested else None
        ),
        "mechanism_activation": activation,
        "tiered_counters": counters,
        "tiered_vs_stock": ratios(stock, tiered),
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
    if dense is not None:
        report["tiered_vs_dense"] = ratios(dense, tiered)
        report["dense_vs_stock"] = ratios(stock, dense)
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
