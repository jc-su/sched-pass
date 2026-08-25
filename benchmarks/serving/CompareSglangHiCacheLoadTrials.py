#!/usr/bin/env python3
"""Run arm-balanced, paired SGLang HiCache qualification trials."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = pathlib.Path(
    os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results")
)
RATIO_FIELDS = (
    # The registered primary: absolute-SLO goodput (TTFT <= 8.0s AND P99
    # ITL <= 100ms, all requests). Its omission until 2026-08-15 made the
    # aggregate report only the legacy relative-threshold goodput_ratio;
    # campaign records before that date were corrected from the banked
    # per-trial artifacts, which always carried both fields.
    "preregistered_goodput_ratio",
    "output_throughput_ratio",
    "goodput_ratio",
    "resident_p95_ttft_ratio",
    "resident_p95_tpot_ratio",
    "resident_p99_itl_ratio",
    "external_p95_ttft_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260801)
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "sglang-hicache-load-trials",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "sglang-hicache-load-qualification.json",
    )
    parser.add_argument(
        "--allow-mixed-revisions",
        action="store_true",
        help=(
            "accept banked artifacts from more than one recorded revision "
            "(the mix is recorded in the aggregate); without this flag a "
            "revision mismatch across trials is fatal"
        ),
    )
    parser.add_argument(
        "comparison_args",
        nargs=argparse.REMAINDER,
        help="arguments for CompareSglangHiCacheLoad.py after --",
    )
    args = parser.parse_args()
    if args.trials < 3:
        parser.error("qualification requires at least three paired trials")
    if args.comparison_args[:1] == ["--"]:
        args.comparison_args = args.comparison_args[1:]
    if not args.comparison_args:
        parser.error("comparison arguments are required after --")
    forbidden = {"--seed", "--output"}.intersection(args.comparison_args)
    if forbidden:
        parser.error(
            "the trial runner owns these comparison arguments: "
            + ", ".join(sorted(forbidden))
        )
    return args


def _expected_harness_args(comparison_args: list[str]) -> dict:
    """Parse the child harness's arguments exactly as it would."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))
    try:
        import CompareSglangHiCacheLoad as child
    finally:
        sys.path.pop(0)
    argv = sys.argv
    sys.argv = ["CompareSglangHiCacheLoad.py", *comparison_args]
    try:
        parsed = child.parse_args()
    except SystemExit as error:
        raise RuntimeError(
            "comparison arguments do not parse; cannot validate banked trials"
        ) from error
    finally:
        sys.argv = argv
    result = {
        key: (str(value) if isinstance(value, pathlib.Path) else value)
        for key, value in sorted(vars(parsed).items())
        if key not in ("output", "seed", "execution_order")
    }
    result.update(
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
    return result


def _seed_for_order(seed: int, first: str) -> int:
    while True:
        order = ["flashinfer", "nta_flashinfer"]
        random.Random(seed).shuffle(order)
        if order[0] == first:
            return seed
        seed += 1


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("ratios must be finite and positive")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _bootstrap_interval(
    values: list[float], *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(
        _geometric_mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return estimates[round(0.025 * (samples - 1))], estimates[
        round(0.975 * (samples - 1))
    ]


def _aggregate(reports: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, field in enumerate(RATIO_FIELDS):
        values = [float(report[field]) for report in reports]
        zero_values = sum(1 for value in values if value == 0.0)
        if zero_values and field in (
            "goodput_ratio",
            "preregistered_goodput_ratio",
        ):
            # A zero goodput arm makes the geometric mean undefined. Report
            # the zero-trial count and aggregate the positive trials instead
            # of discarding the entire series; consumers must read both.
            positive = [value for value in values if value > 0.0]
            result[field] = {
                "zero_goodput_trials": zero_values,
                "positive_trial_geometric_mean": (
                    _geometric_mean(positive) if positive else None
                ),
                "paired_values": values,
            }
            continue
        low, high = _bootstrap_interval(values, seed=seed + index)
        result[field] = {
            "paired_values": values,
            "median": statistics.median(values),
            "geometric_mean": _geometric_mean(values),
            "bootstrap_95_percent_ci": [low, high],
        }
    return result


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    expected_args = _expected_harness_args(args.comparison_args)
    reports: list[dict[str, Any]] = []
    artifacts: list[str] = []
    for trial in range(args.trials):
        first = "flashinfer" if trial % 2 == 0 else "nta_flashinfer"
        # Pre-registered seeds are used verbatim: arm balancing is an
        # explicit argument, never a seed search — searching mutated the
        # registered seed list (found 2026-08-13, external review).
        seed = args.seed_base + trial
        artifact = (args.artifact_dir / f"trial-{trial:02d}.json").resolve()
        if artifact.is_file():
            # Resume after an interrupted campaign: the seed chain above is
            # deterministic, so a banked trial re-derives the same seed and
            # arm order; accept it only when both match.
            report = json.loads(artifact.read_text(encoding="utf-8"))
            arm_seed = report.get("nta", {}).get("seed", -1)
            banked_args = report.get("harness_args")
            args_match = True
            if banked_args is not None:
                current_args = expected_args
                mismatched = {
                    key: (banked_args.get(key), current_args.get(key))
                    for key in set(banked_args) | set(current_args)
                    if banked_args.get(key) != current_args.get(key)
                }
                if mismatched:
                    args_match = False
                    raise RuntimeError(
                        f"banked trial {trial} was produced with different "
                        f"harness arguments: {mismatched}"
                    )
            if (
                args_match
                and report.get("classification") == "sglang-hicache-load-comparison"
                and int(arm_seed) == seed
                and report.get("execution_order", [None])[0] == first
            ):
                reports.append(report)
                artifacts.append(str(artifact))
                continue
        command = [
            sys.executable,
            str(ROOT / "benchmarks" / "serving" / "CompareSglangHiCacheLoad.py"),
            *args.comparison_args,
            "--seed",
            str(seed),
            "--execution-order",
            "stock_first" if first == "flashinfer" else "nta_first",
            "--output",
            str(artifact),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"paired trial {trial} failed:\n"
                + "\n".join(completed.stdout.splitlines()[-120:])
            )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        if report.get("classification") != "sglang-hicache-load-comparison":
            raise RuntimeError(f"paired trial {trial} emitted an invalid artifact")
        if report.get("execution_order", [None])[0] != first:
            raise RuntimeError(f"paired trial {trial} did not preserve arm balancing")
        reports.append(report)
        artifacts.append(str(artifact))

    diverged_trials = [
        trial
        for trial, report in enumerate(reports)
        if report["stock"]["generated_text_sha256"]
        != report["nta"]["generated_text_sha256"]
    ]
    revisions = sorted(
        {str(report.get("revision") or "unrecorded") for report in reports}
    )
    if len(revisions) > 1 and not args.allow_mixed_revisions:
        raise RuntimeError(
            "banked trials span more than one revision "
            f"({revisions}); rerun on one revision or pass "
            "--allow-mixed-revisions to aggregate anyway (recorded)"
        )
    aggregate = {
        "schema": 1,
        "classification": "sglang-hicache-load-qualification",
        "trial_count": len(reports),
        # Ten process-level trials are the documented evidence standard for a
        # serving claim; smaller runs are diagnostics and must say so.
        "evidence_grade": "qualified" if len(reports) >= 10 else "diagnostic",
        "arm_order": [report["execution_order"] for report in reports],
        "artifacts": artifacts,
        "all_outputs_exact": not diverged_trials,
        "diverged_trials": diverged_trials,
        "all_attention_transformed": all(
            bool(report["mechanism_activation"]["all_attention_transformed"])
            for report in reports
        ),
        "all_external_attention_transformed": all(
            bool(
                report["mechanism_activation"].get(
                    "external_attention_transformed", False
                )
            )
            for report in reports
        ),
        "all_fallback_free": all(
            int(report["mechanism_activation"]["fallback_batches"]) == 0
            for report in reports
        ),
        "revisions": revisions,
        "harness_args": expected_args,
        "selected_bytes_per_trial": [
            report.get("nta_selected_bytes") for report in reports
        ],
        "candidate_bytes_per_trial": [
            report.get("nta_candidate_bytes") for report in reports
        ],
        "ratios": _aggregate(reports, args.seed_base),
    }
    # Registered-bar status: "qualified" alone only certifies trial count
    # and mechanism purity; this block states each pre-registered bar's
    # verdict so no consumer mistakes one passing bar for a passing run.
    registered = aggregate["ratios"].get("preregistered_goodput_ratio", {})
    resident = aggregate["ratios"].get("resident_p99_itl_ratio", {})
    goodput_geomean = registered.get("geometric_mean")
    goodput_floor = (registered.get("bootstrap_95_percent_ci") or [None])[0]
    resident_geomean = resident.get("geometric_mean")
    aggregate["bars"] = {
        "registered_goodput": {
            "bar": 1.5,
            "geometric_mean": goodput_geomean,
            "ci_floor": goodput_floor,
            "passes": bool(
                goodput_geomean is not None
                and goodput_floor is not None
                and goodput_geomean >= 1.5
                and goodput_floor > 1.0
            ),
        },
        "resident_p99_itl": {
            "bar": 1.05,
            "geometric_mean": resident_geomean,
            "passes": bool(resident_geomean is not None and resident_geomean <= 1.05),
        },
        "outputs": {
            # With divergence reporting armed, a recorded divergence is
            # not a bar failure — the scored quality battery is the
            # registered arbiter; without the flag exactness is mandatory.
            "exact": aggregate["all_outputs_exact"],
            "divergence_reporting_armed": (
                "--allow-output-divergence" in args.comparison_args
            ),
            "diverged_trials": diverged_trials,
            "passes": aggregate["all_outputs_exact"]
            or "--allow-output-divergence" in args.comparison_args,
        },
        "mechanism": {
            "passes": aggregate["all_external_attention_transformed"]
            and aggregate["all_fallback_free"],
        },
        "physical_bytes": {
            # The registered evidence standard records physically staged
            # bytes per trial; artifacts predating the ledger fail this
            # bar and must be regenerated rather than waived.
            "recorded_trials": sum(
                1
                for value in (report.get("nta_staged_bytes") for report in reports)
                if isinstance(value, int) and value > 0
            ),
            "passes": all(
                isinstance(report.get("nta_staged_bytes"), int)
                and report.get("nta_staged_bytes") > 0
                for report in reports
            ),
        },
    }
    aggregate["all_bars_pass"] = all(
        bar["passes"] for bar in aggregate["bars"].values()
    )
    # Output exactness is mandatory unless the trials themselves ran with
    # divergence reporting armed; then the aggregate records which trials
    # diverged instead of refusing, and the scored quality battery remains
    # the arbiter — the posture recorded with campaign three.
    mandatory = ["all_external_attention_transformed", "all_fallback_free"]
    if "--allow-output-divergence" not in args.comparison_args:
        mandatory.append("all_outputs_exact")
    if not all(aggregate[key] for key in mandatory):
        raise RuntimeError("qualification violated a mandatory mechanism invariant")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
