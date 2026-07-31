#!/usr/bin/env python3
"""Run clustered-launch CLC characterization sweeps."""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_cluster_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

NUMERIC = {
    "tasks": int,
    "threads": int,
    "cluster_x": int,
    "work_cycles": int,
    "active_clusters": int,
    "predicted_r": int,
    "sm_count": int,
    "processed": int,
    "active_ctas": int,
    "active_leaders": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "base_unique": int,
    "base_min": int,
    "base_max": int,
    "base_duplicates": int,
    "base_alignment_errors": int,
    "claim_range_holes": int,
    "first_claim_base_min": int,
    "first_claim_base_max": int,
    "last_claim_base_min": int,
    "last_claim_base_max": int,
    "max_processed": int,
    "expected_active_ctas": int,
    "expected_claimed": int,
    "expected_claimed_clusters": int,
    "structural_ok": int,
}


def cfg(suite, tasks, threads, cluster_x, work_cycles=4096):
    return {
        "suite": suite,
        "tasks": tasks,
        "threads": threads,
        "cluster_x": cluster_x,
        "work_cycles": work_cycles,
    }


def parse_probe_csv(stdout):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("tasks,"))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    return {k: NUMERIC[k](v) for k, v in row.items()}


def run_one(config):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cp = subprocess.run(
        [str(BIN), str(config["tasks"]), str(config["threads"]),
         str(config["cluster_x"]), str(config["work_cycles"])],
        env=env, text=True, capture_output=True, check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    return row


def predict_r(threads, cluster_x):
    row = run_one(cfg("predict", 8192, threads, cluster_x, 1024))
    return row["predicted_r"]


def configs_for_suite(suite):
    if suite in ("cluster-size", "all"):
        for cluster_x in (1, 2, 4, 8):
            yield cfg("cluster-size", 8192, 128, cluster_x, 4096)
            yield cfg("cluster-size", 16384, 128, cluster_x, 4096)

    if suite in ("threshold", "all"):
        for cluster_x in (1, 2, 4, 8):
            r = predict_r(128, cluster_x)
            for tasks in (r, r + cluster_x, r + 16 * cluster_x):
                yield cfg("cluster-threshold", tasks, 128, cluster_x, 4096)


def validate(row):
    errors = []
    if row["missed"]:
        errors.append(f"missed={row['missed']}")
    if row["duplicates"]:
        errors.append(f"duplicates={row['duplicates']}")
    if row["base_duplicates"]:
        errors.append(f"base_duplicates={row['base_duplicates']}")
    if row["base_alignment_errors"]:
        errors.append(f"base_alignment_errors={row['base_alignment_errors']}")
    if row["claim_range_holes"]:
        errors.append(f"claim_range_holes={row['claim_range_holes']}")
    if row["active_ctas"] != row["expected_active_ctas"]:
        errors.append("active_ctas mismatch")

    if row["tasks"] <= row["predicted_r"]:
        if row["successes"] or row["unique_claimed"] or row["base_unique"]:
            errors.append("unexpected claim below/equal R")
    else:
        if row["claimed_min"] != row["predicted_r"]:
            errors.append("claimed_min mismatch")
        if row["claimed_max"] != row["tasks"] - 1:
            errors.append("claimed_max mismatch")
        if row["base_min"] != row["predicted_r"]:
            errors.append("base_min mismatch")
        if row["base_max"] != row["tasks"] - row["cluster_x"]:
            errors.append("base_max mismatch")
        if row["base_unique"] != row["expected_claimed_clusters"]:
            errors.append("claimed cluster count mismatch")

    if row["structural_ok"] != 1:
        errors.append("probe structural_ok=0")
    return errors


def group_key(row):
    return (row["suite"], row["tasks"], row["threads"], row["cluster_x"],
            row["work_cycles"])


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
            "tasks": key[1],
            "threads": key[2],
            "cluster_x": key[3],
            "work_cycles": key[4],
            "repeats": len(rs),
            "active_clusters_mean": mean(rs, "active_clusters"),
            "predicted_r_mean": mean(rs, "predicted_r"),
            "active_ctas_mean": mean(rs, "active_ctas"),
            "active_leaders_mean": mean(rs, "active_leaders"),
            "successes_mean": mean(rs, "successes"),
            "unique_claimed_mean": mean(rs, "unique_claimed"),
            "base_unique_mean": mean(rs, "base_unique"),
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "base_min_min": min(r["base_min"] for r in rs),
            "base_max_max": max(r["base_max"] for r in rs),
            "first_claim_base_min_min": min(r["first_claim_base_min"]
                                            for r in rs),
            "first_claim_base_max_max": max(r["first_claim_base_max"]
                                            for r in rs),
            "last_claim_base_min_min": min(r["last_claim_base_min"]
                                           for r in rs),
            "last_claim_base_max_max": max(r["last_claim_base_max"]
                                           for r in rs),
            "max_processed_max": max(r["max_processed"] for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "base_duplicates_max": max(r["base_duplicates"] for r in rs),
            "base_alignment_errors_max": max(r["base_alignment_errors"]
                                             for r in rs),
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("cluster-size", "threshold", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; run TARGET=clc_cluster_probe build_run.sh")

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
            errors = validate(row)
            row["validation"] = "ok" if not errors else "; ".join(errors)
            rows.append(row)
            if errors:
                failures.append((row, errors))
            print(f"[{n:03d}/{total}] suite={config['suite']} "
                  f"tasks={config['tasks']} cluster={config['cluster_x']} "
                  f"active_clusters={row['active_clusters']} R={row['predicted_r']} "
                  f"base={row['base_min']}..{row['base_max']} "
                  f"valid={row['validation'] == 'ok'}")

    raw = OUTDIR / f"clc_cluster_{args.suite}_raw_{stamp}.csv"
    summary = OUTDIR / f"clc_cluster_{args.suite}_summary_{stamp}.csv"
    write_csv(raw, rows)
    write_csv(summary, aggregate(rows))
    print(f"raw={raw}")
    print(f"summary={summary}")

    if failures:
        print("validation failures:")
        for row, errors in failures[:20]:
            print(f"  tasks={row['tasks']} cluster={row['cluster_x']}: "
                  f"{'; '.join(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
