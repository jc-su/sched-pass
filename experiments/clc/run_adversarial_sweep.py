#!/usr/bin/env python3
"""Run focused adversarial CLC probes.

This covers behavior that is not a timing sweep:

* 3D get_first_ctaid.v4 tuple semantics.
* Partial CLC participation, where only some initial CTAs keep claiming.
"""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).resolve().parent / "results"
TUPLE_BIN = ROOT / "build" / "clc_tuple_probe"
PART_BIN = ROOT / "build" / "clc_participation_probe"

TUPLE_NUMERIC = {
    "grid_x": int,
    "grid_y": int,
    "grid_z": int,
    "tasks": int,
    "threads": int,
    "work_cycles": int,
    "processed": int,
    "active_workers": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "w_min": int,
    "w_max": int,
    "w_nonzero": int,
    "w_bins": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "max_processed": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}

PART_NUMERIC = {
    "tasks": int,
    "threads": int,
    "claim_stride": int,
    "work_cycles": int,
    "processed": int,
    "active_workers": int,
    "participants_launched": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "max_processed": int,
    "multi_claim_workers": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "exactly_once": int,
}


def parse_probe_csv(stdout, prefix, numeric):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith(prefix))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    return {k: numeric[k](v) for k, v in row.items()}


def run_cmd(bin_path, args, prefix, numeric):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cp = subprocess.run([str(bin_path), *map(str, args)], env=env, text=True,
                        capture_output=True, check=True)
    return parse_probe_csv(cp.stdout, prefix, numeric)


def tuple_configs():
    for grid_x, grid_y, grid_z in ((64, 16, 8), (16, 16, 32),
                                   (17, 19, 29)):
        yield {
            "suite": "tuple",
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "threads": 128,
            "work_cycles": 4096,
        }


def part_configs():
    for stride in (1, 2, 4, 8, 16):
        yield {
            "suite": "participation",
            "tasks": 8192,
            "threads": 128,
            "claim_stride": stride,
            "work_cycles": 4096,
        }


def run_tuple(config):
    args = (config["grid_x"], config["grid_y"], config["grid_z"],
            config["threads"], config["work_cycles"])
    row = run_cmd(TUPLE_BIN, args, "grid_x,", TUPLE_NUMERIC)
    row["suite"] = config["suite"]
    return row


def run_part(config):
    args = (config["tasks"], config["threads"], config["claim_stride"],
            config["work_cycles"])
    row = run_cmd(PART_BIN, args, "tasks,", PART_NUMERIC)
    row["suite"] = config["suite"]
    return row


def validate_tuple(row):
    errors = []
    tasks = row["tasks"]
    predicted_r = row["predicted_r"]
    expected_active = min(tasks, predicted_r)
    expected_claimed = max(0, tasks - predicted_r)

    checks = (
        ("missed", 0),
        ("duplicates", 0),
        ("duplicate_claims", 0),
        ("claim_range_holes", 0),
        ("active_workers", expected_active),
        ("expected_active_workers", expected_active),
        ("expected_claimed", expected_claimed),
        ("w_min", 0),
        ("w_max", 0),
        ("w_nonzero", 0),
        ("w_bins", 1),
        ("structural_ok", 1),
    )
    for field, expected in checks:
        if row[field] != expected:
            errors.append(f"{field}={row[field]} expected={expected}")

    if tasks <= predicted_r:
        if row["successes"] or row["unique_claimed"]:
            errors.append("unexpected claim below/equal R")
    else:
        if row["unique_claimed"] != expected_claimed:
            errors.append("unique_claimed mismatch")
        if row["claimed_min"] != predicted_r:
            errors.append("claimed_min mismatch")
        if row["claimed_max"] != tasks - 1:
            errors.append("claimed_max mismatch")

    return errors


def validate_part(row):
    errors = []
    if row["processed"] != row["tasks"]:
        errors.append("processed != tasks")
    for field in ("missed", "duplicates", "duplicate_claims"):
        if row[field]:
            errors.append(f"{field}={row[field]}")
    if row["exactly_once"] != 1:
        errors.append("exactly_once=0")

    if row["claim_stride"] == 1:
        if row["active_workers"] != row["predicted_r"]:
            errors.append("stride=1 active_workers mismatch")
        if row["claimed_min"] != row["predicted_r"]:
            errors.append("stride=1 claimed_min mismatch")
        if row["claimed_max"] != row["tasks"] - 1:
            errors.append("stride=1 claimed_max mismatch")
        if row["claim_range_holes"] != 0:
            errors.append("stride=1 holes")
    else:
        if row["claim_range_holes"] == 0:
            errors.append("partial participation did not create sparse claims")
        if row["active_workers"] <= row["predicted_r"]:
            errors.append("partial participation did not launch extra workers")

    return errors


def tuple_key(row):
    return (row["suite"], row["grid_x"], row["grid_y"], row["grid_z"],
            row["threads"], row["work_cycles"])


def part_key(row):
    return (row["suite"], row["tasks"], row["threads"], row["claim_stride"],
            row["work_cycles"])


def mean(rows, field):
    return statistics.fmean(r[field] for r in rows)


def aggregate_tuple(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple_key(row)].append(row)

    out = []
    for key, rs in sorted(grouped.items()):
        out.append({
            "suite": key[0],
            "grid_x": key[1],
            "grid_y": key[2],
            "grid_z": key[3],
            "threads": key[4],
            "work_cycles": key[5],
            "tasks": rs[0]["tasks"],
            "repeats": len(rs),
            "active_workers_mean": mean(rs, "active_workers"),
            "predicted_r_mean": mean(rs, "predicted_r"),
            "unique_claimed_mean": mean(rs, "unique_claimed"),
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "first_claim_min_min": min(r["first_claim_min"] for r in rs),
            "first_claim_max_max": max(r["first_claim_max"] for r in rs),
            "last_claim_min_min": min(r["last_claim_min"] for r in rs),
            "last_claim_max_max": max(r["last_claim_max"] for r in rs),
            "w_min_min": min(r["w_min"] for r in rs),
            "w_max_max": max(r["w_max"] for r in rs),
            "w_nonzero_max": max(r["w_nonzero"] for r in rs),
            "w_bins_max": max(r["w_bins"] for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "valid_runs": sum(1 for r in rs if r["validation"] == "ok"),
        })
    return out


def aggregate_part(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[part_key(row)].append(row)

    out = []
    for key, rs in sorted(grouped.items()):
        out.append({
            "suite": key[0],
            "tasks": key[1],
            "threads": key[2],
            "claim_stride": key[3],
            "work_cycles": key[4],
            "repeats": len(rs),
            "active_workers_mean": mean(rs, "active_workers"),
            "active_workers_min": min(r["active_workers"] for r in rs),
            "active_workers_max": max(r["active_workers"] for r in rs),
            "participants_mean": mean(rs, "participants_launched"),
            "successes_mean": mean(rs, "successes"),
            "unique_claimed_mean": mean(rs, "unique_claimed"),
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_min_max": max(r["claimed_min"] for r in rs),
            "claimed_max_min": min(r["claimed_max"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "claim_range_holes_mean": mean(rs, "claim_range_holes"),
            "claim_range_holes_min": min(r["claim_range_holes"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "multi_claim_workers_mean": mean(rs, "multi_claim_workers"),
            "max_processed_max": max(r["max_processed"] for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "exactly_once_min": min(r["exactly_once"] for r in rs),
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(name, repeats):
    if name == "tuple":
        if not TUPLE_BIN.exists():
            raise SystemExit(f"missing {TUPLE_BIN}; build clc_tuple_probe first")
        runner = run_tuple
        validator = validate_tuple
        configs = list(tuple_configs())
        aggregator = aggregate_tuple
    elif name == "participation":
        if not PART_BIN.exists():
            raise SystemExit(
                f"missing {PART_BIN}; build clc_participation_probe first")
        runner = run_part
        validator = validate_part
        configs = list(part_configs())
        aggregator = aggregate_part
    else:
        raise ValueError(name)

    rows = []
    failures = []
    for config in configs:
        for repeat in range(repeats):
            row = runner(config)
            row["repeat"] = repeat
            errors = validator(row)
            row["validation"] = "ok" if not errors else ";".join(errors)
            rows.append(row)
            status = "ok" if not errors else row["validation"]
            print(f"[{name}] config={config} repeat={repeat} {status}")
            if errors:
                failures.append(row)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = OUTDIR / f"clc_{name}_raw_{stamp}.csv"
    summary_path = OUTDIR / f"clc_{name}_summary_{stamp}.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, aggregator(rows))
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")

    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("tuple", "participation", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    failures = []
    suites = ("tuple", "participation") if args.suite == "all" else (args.suite,)
    for suite in suites:
        failures.extend(run_suite(suite, args.repeats))

    if failures:
        raise SystemExit(f"{len(failures)} invalid rows")


if __name__ == "__main__":
    main()
