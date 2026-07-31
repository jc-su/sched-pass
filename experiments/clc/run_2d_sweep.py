#!/usr/bin/env python3
"""Run 2D-grid CLC characterization sweeps."""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_2d_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

NUMERIC = {
    "grid_x": int,
    "grid_y": int,
    "tasks": int,
    "threads": int,
    "work_cycles": int,
    "smem_bytes": int,
    "static_us": float,
    "clc_us": float,
    "delta_pct": float,
    "processed": int,
    "active_workers": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "success_rate": float,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "claimed_min_x": int,
    "claimed_min_y": int,
    "claimed_max_x": int,
    "claimed_max_y": int,
    "claim_cycles_avg": float,
    "claim_cycles_max": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "r_x": int,
    "r_y": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "max_processed": int,
    "multi_claim_workers": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}

CFG_FIELDS = ("grid_x", "grid_y", "threads", "work_cycles", "smem_bytes")


def cfg(suite, grid_x, grid_y, threads=128, work_cycles=4096, smem_bytes=0):
    return {
        "suite": suite,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "threads": threads,
        "work_cycles": work_cycles,
        "smem_bytes": smem_bytes,
    }


def parse_probe_csv(stdout):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("grid_x,"))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    return {k: NUMERIC[k](v) for k, v in row.items()}


def run_one(config):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cp = subprocess.run([str(BIN), *[str(config[k]) for k in CFG_FIELDS]],
                        env=env, text=True, capture_output=True, check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    return row


def validate_row(row):
    errors = []
    tasks = row["tasks"]
    predicted_r = row["predicted_r"]
    expected_active = min(tasks, predicted_r)
    expected_claimed = max(0, tasks - predicted_r)

    if row["missed"]:
        errors.append(f"missed={row['missed']}")
    if row["duplicates"]:
        errors.append(f"duplicates={row['duplicates']}")
    if row["duplicate_claims"]:
        errors.append(f"duplicate_claims={row['duplicate_claims']}")
    if row["claim_range_holes"]:
        errors.append(f"claim_range_holes={row['claim_range_holes']}")
    if row["active_workers"] != expected_active:
        errors.append(
            f"active_workers={row['active_workers']} expected={expected_active}")

    if tasks <= predicted_r:
        if row["successes"] or row["unique_claimed"]:
            errors.append("unexpected claim below/equal R")
    else:
        if row["unique_claimed"] != expected_claimed:
            errors.append(
                f"unique_claimed={row['unique_claimed']} expected={expected_claimed}")
        if row["claimed_min"] != predicted_r:
            errors.append(
                f"claimed_min={row['claimed_min']} expected={predicted_r}")
        if row["claimed_max"] != tasks - 1:
            errors.append(
                f"claimed_max={row['claimed_max']} expected={tasks - 1}")
        if row["claimed_min_x"] != row["r_x"] or row["claimed_min_y"] != row["r_y"]:
            errors.append("claimed min coord does not match R coord")

    if row["structural_ok"] != 1:
        errors.append("probe structural_ok=0")
    return errors


def configs_for_suite(suite):
    shapes = (
        (8192, 1),
        (4096, 2),
        (2048, 4),
        (1024, 8),
        (512, 16),
        (256, 32),
        (128, 64),
        (64, 128),
        (32, 256),
        (17, 512),
    )
    if suite in ("shape", "all"):
        for grid_x, grid_y in shapes:
            yield cfg("2d-shape", grid_x, grid_y, 128, 4096, 0)

    if suite in ("threshold", "all"):
        for grid_x, grid_y in ((47, 48), (48, 48), (64, 35), (64, 36),
                               (64, 128), (32, 256)):
            yield cfg("2d-threshold", grid_x, grid_y, 128, 4096, 0)

    if suite in ("smem", "all"):
        for smem in (0, 8192, 16384, 24576):
            for grid_x, grid_y in ((64, 128), (32, 256)):
                yield cfg("2d-smem", grid_x, grid_y, 128, 4096, smem)

    if suite in ("latency", "all"):
        for work_cycles in (0, 128, 1024, 4096):
            for grid_x, grid_y in ((64, 128), (8192, 1)):
                yield cfg("2d-latency", grid_x, grid_y, 128, work_cycles, 0)


def group_key(row):
    return (row["suite"], row["grid_x"], row["grid_y"], row["threads"],
            row["work_cycles"], row["smem_bytes"])


def mean(rows, field):
    return statistics.fmean(r[field] for r in rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out = []
    for key, rs in sorted(grouped.items()):
        out.append({
            "suite": key[0],
            "grid_x": key[1],
            "grid_y": key[2],
            "threads": key[3],
            "work_cycles": key[4],
            "smem_bytes": key[5],
            "tasks": rs[0]["tasks"],
            "repeats": len(rs),
            "static_us_mean": mean(rs, "static_us"),
            "clc_us_mean": mean(rs, "clc_us"),
            "delta_pct_mean": mean(rs, "delta_pct"),
            "delta_pct_stdev": statistics.stdev(r["delta_pct"] for r in rs)
            if len(rs) > 1 else 0.0,
            "success_rate_mean": mean(rs, "success_rate"),
            "active_workers_mean": mean(rs, "active_workers"),
            "occ_blocks_per_sm_mean": mean(rs, "occ_blocks_per_sm"),
            "predicted_r_mean": mean(rs, "predicted_r"),
            "r_x": rs[0]["r_x"],
            "r_y": rs[0]["r_y"],
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "claimed_min_x": rs[0]["claimed_min_x"],
            "claimed_min_y": rs[0]["claimed_min_y"],
            "claimed_max_x": rs[0]["claimed_max_x"],
            "claimed_max_y": rs[0]["claimed_max_y"],
            "first_claim_min_min": min(r["first_claim_min"] for r in rs),
            "first_claim_max_max": max(r["first_claim_max"] for r in rs),
            "last_claim_min_min": min(r["last_claim_min"] for r in rs),
            "last_claim_max_max": max(r["last_claim_max"] for r in rs),
            "claim_cycles_avg_mean": mean(rs, "claim_cycles_avg"),
            "claim_cycles_max_max": max(r["claim_cycles_max"] for r in rs),
            "max_processed_max": max(r["max_processed"] for r in rs),
            "multi_claim_workers_mean": mean(rs, "multi_claim_workers"),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "valid_runs": sum(1 for r in rs if r["validation"] == "ok"),
        })
    return out


def fieldnames(rows):
    out = []
    for row in rows:
        for key in row.keys():
            if key not in out:
                out.append(key)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("shape", "threshold", "smem",
                                        "latency", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; run: TARGET=clc_2d_probe build_run.sh")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    failures = []
    configs = list(configs_for_suite(args.suite))
    total = len(configs) * args.repeats
    n = 0
    for config in configs:
        for rep in range(args.repeats):
            n += 1
            row = run_one(config)
            row["repeat"] = rep
            errors = validate_row(row)
            row["validation"] = "ok" if not errors else "; ".join(errors)
            rows.append(row)
            if errors:
                failures.append((row, errors))
            print(f"[{n:03d}/{total}] suite={config['suite']} "
                  f"grid={config['grid_x']}x{config['grid_y']} "
                  f"R={row['predicted_r']} Rcoord=({row['r_x']},{row['r_y']}) "
                  f"claim={row['claimed_min']}..{row['claimed_max']} "
                  f"coord=({row['claimed_min_x']},{row['claimed_min_y']}).."
                  f"({row['claimed_max_x']},{row['claimed_max_y']}) "
                  f"valid={row['validation'] == 'ok'}")

    raw = OUTDIR / f"clc_2d_{args.suite}_raw_{stamp}.csv"
    summary = OUTDIR / f"clc_2d_{args.suite}_summary_{stamp}.csv"
    write_csv(raw, rows)
    write_csv(summary, aggregate(rows))
    print(f"raw={raw}")
    print(f"summary={summary}")

    if failures:
        print("validation failures:")
        for row, errors in failures[:20]:
            print(f"  grid={row['grid_x']}x{row['grid_y']}: "
                  f"{'; '.join(errors)}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
