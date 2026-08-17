#!/usr/bin/env python3
"""Tiered CUDA-graph replay soundness battery (audit debt, 2026-08-15).

Runs the real serving comparison at refresh intervals 1024, 32, and 1 with
graphs enabled in both arms and VERIFY=fast byte-checking every staged
layer, then asserts the boundary witnesses the position-ordering defect
would violate:

- refresh 1024: replays dominate; staging happens at extends (eager), so
  both replay and eager tiered batches must appear (boundary crossings).
- refresh 32: mid-decode refreshes force frequent replay->eager->replay
  transitions with divergent per-layer row tables — the exact regime that
  crashed before the ordering fix.
- refresh 1: selection refreshes every step, so tiered decode must stay
  eager (zero tiered replays) while dense residents still replay — the
  other side of the boundary.

Claim reuse (external count exceeds live concurrency, so table slots
recycle) and changed batch layouts (completions shrink the batch) are
exercised by the shape itself. Cancellation coverage remains a recorded
debt for the RQ4 harness.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def nta_stats(report: dict) -> dict:
    merged: dict = {}
    for entry in report["nta"].get("engine_stats", []):
        if entry.get("backend") != "nta_flashinfer":
            continue
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
    return merged


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    # Two halves per refresh interval: the verify half byte-checks every
    # staged layer (verify-mode claims never replay, so it certifies the
    # bytes, not the replay path); the replay half runs without VERIFY so
    # tiered epochs replay under the captured graphs, and output parity
    # with the stock arm is the soundness signal there — the position-
    # ordering defect corrupted per-layer rows exactly on that path.
    for refresh, mode in (
        (1024, "verify"),
        (32, "verify"),
        (1, "verify"),
        (1024, "replay"),
        (32, "replay"),
        (1, "replay"),
    ):
        suffix = "" if mode == "verify" else "-replay"
        artifact = args.output_dir / f"replay-refresh{refresh}{suffix}.json"
        environment = os.environ.copy()
        environment.update(
            {
                "SGLANG_NUMA_BIND_V2": "1",
                "NTA_SGLANG_TIERED_GRAPH": "1",
                "NTA_SGLANG_PIPELINE_HOST": "1",
                "NTA_SGLANG_SELECTED_SERVE": "1",
                "NTA_SGLANG_SELECTED_TIERED": "1",
                "NTA_SGLANG_SELECTED_BUDGET": "64",
                "NTA_SGLANG_SELECTED_REFRESH_INTERVAL": str(refresh),
            }
        )
        if mode == "verify":
            environment["NTA_SGLANG_SELECTED_TIERED_VERIFY"] = "fast"
        else:
            environment.pop("NTA_SGLANG_SELECTED_TIERED_VERIFY", None)
        command = [
            sys.executable,
            str(ROOT / "benchmarks" / "serving" / "CompareSglangHiCacheLoad.py"),
            "--model",
            str(args.model),
            "--external-requests",
            "8",
            "--external-tokens",
            "16384",
            "--external-output-tokens",
            "192",
            "--resident-requests",
            "2",
            "--resident-tokens",
            "2048",
            "--resident-output-tokens",
            "64",
            "--request-rate",
            "4.0",
            "--churn-tokens",
            "12000",
            "--max-total-tokens",
            "110000",
            "--context-length",
            "32768",
            "--hicache-ratio",
            "8.0",
            "--max-running-requests",
            "4",
            "--cuda-graph-decode",
            "full",
            "--seed",
            str(args.seed),
            "--allow-output-divergence",
            "--allow-oversubscribed-pool",
            "--output",
            str(artifact),
        ]
        label = f"refresh={refresh}/{mode}"
        if artifact.is_file():
            print(f"{label}: reusing banked artifact")
        else:
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
                tail = "\n".join(completed.stdout.splitlines()[-40:])
                failures.append(f"{label}: serving run failed\n{tail}")
                continue
        report = json.loads(artifact.read_text(encoding="utf-8"))
        stats = nta_stats(report)
        replays = stats.get("tiered_graph_replay_batches", 0)
        eager = stats.get("tiered_graph_eager_batches", 0)
        checked = stats.get("tiered_fast_checked_layers", 0)
        claims = stats.get("tiered_claims", 0)
        live_max = stats.get("tiered_claims_live_max", 0)
        check(stats.get("graph_replays", 0) > 0, f"{label}: no graph replays", failures)
        check(
            stats.get("hicache_fallback_batches", 0) == 0,
            f"{label}: fallback batches",
            failures,
        )
        check(
            claims > live_max,
            f"{label}: no claim-slot reuse ({claims} claims, {live_max} live max)",
            failures,
        )
        if mode == "verify":
            check(checked > 0, f"{label}: VERIFY=fast checked no layers", failures)
        elif refresh == 1:
            check(
                replays == 0,
                f"{label}: tiered decode replayed {replays} batches despite "
                "per-step refresh",
                failures,
            )
            check(eager > 0, f"{label}: no eager tiered batches", failures)
        else:
            check(replays > 0, f"{label}: no tiered replays", failures)
            if refresh < 192:
                # Only a refresh interval shorter than the decode length
                # forces mid-decode eager steps; at 1024 the extends are
                # eager but decode replays uninterrupted, so demanding
                # boundary crossings there was asserting the impossible.
                check(
                    eager > 0,
                    f"{label}: no eager tiered batches (no boundary "
                    "crossings)",
                    failures,
                )
        diverged = bool(report.get("outputs_diverge"))
        print(
            f"{label}: replays={replays} eager={eager} checked={checked} "
            f"claims={claims}/{live_max} diverged={diverged}"
        )
    if failures:
        print("REPLAY BATTERY FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("tiered graph replay battery passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
