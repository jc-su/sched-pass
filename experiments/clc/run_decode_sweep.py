#!/usr/bin/env python3
"""Run decode-shaped Blackwell CLC characterization sweeps."""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_decode_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

LAYOUT_NAMES = {
    0: "interleaved",
    1: "long-prefix",
    2: "long-suffix",
}

NUMERIC = {
    "tasks": int,
    "long_every": int,
    "short_blocks": int,
    "long_blocks": int,
    "page_tokens": int,
    "layout": int,
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
    "claim_cycles_avg": float,
    "claim_cycles_max": int,
    "bytes_static": float,
    "threads": int,
    "smem_bytes": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "max_processed": int,
    "multi_claim_workers": int,
    "max_successes_worker": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}

CFG_FIELDS = (
    "tasks",
    "long_every",
    "short_blocks",
    "long_blocks",
    "page_tokens",
    "layout",
)


def cfg(suite, tasks, long_every, short_blocks, long_blocks, page_tokens,
        layout=0):
    return {
        "suite": suite,
        "tasks": tasks,
        "long_every": long_every,
        "short_blocks": short_blocks,
        "long_blocks": long_blocks,
        "page_tokens": page_tokens,
        "layout": layout,
    }


def parse_probe_csv(stdout):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("tasks,"))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    return {k: NUMERIC[k](v) for k, v in row.items()}


def run_one(config):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cp = subprocess.run([str(BIN), *[str(config[k]) for k in CFG_FIELDS]],
                        env=env, text=True, capture_output=True, check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    row["layout_name"] = LAYOUT_NAMES.get(row["layout"], str(row["layout"]))
    return row


def validate_row(row):
    errors = []
    tasks = row["tasks"]
    predicted_r = row["predicted_r"]
    expected_active = min(tasks, predicted_r)
    expected_claimed = max(0, tasks - predicted_r)

    if row["missed"] != 0:
        errors.append(f"missed={row['missed']}")
    if row["duplicates"] != 0:
        errors.append(f"duplicates={row['duplicates']}")
    if row["duplicate_claims"] != 0:
        errors.append(f"duplicate_claims={row['duplicate_claims']}")
    if row["claim_range_holes"] != 0:
        errors.append(f"claim_range_holes={row['claim_range_holes']}")
    if row["active_workers"] != expected_active:
        errors.append(
            f"active_workers={row['active_workers']} expected={expected_active}")

    if tasks <= predicted_r:
        if row["successes"] != 0 or row["unique_claimed"] != 0:
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

    if row["structural_ok"] != 1:
        errors.append("probe structural_ok=0")
    return errors


def threshold_configs():
    for tasks in (1024, 2048, 2256, 2304, 3072, 4096, 8192):
        yield cfg("decode-threshold", tasks, 0, 2, 16, 16, 0)


def workload_configs():
    for tasks in (4096, 8192, 16384):
        for long_every in (8, 4):
            for layout in (0, 1, 2):
                yield cfg("decode-workload", tasks, long_every, 2, 16, 16,
                          layout)

    for tasks in (4096, 8192):
        for long_every in (8, 4):
            for layout in (0, 1, 2):
                yield cfg("decode-workload", tasks, long_every, 4, 32, 16,
                          layout)


def configs_for_suite(suite):
    if suite == "threshold":
        return list(threshold_configs())
    if suite == "workload":
        return list(workload_configs())
    if suite == "all":
        return list(threshold_configs()) + list(workload_configs())
    raise ValueError(suite)


def group_key(row):
    return (row["suite"], row["tasks"], row["long_every"],
            row["short_blocks"], row["long_blocks"], row["page_tokens"],
            row["layout"])


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
            "long_every": key[2],
            "short_blocks": key[3],
            "long_blocks": key[4],
            "page_tokens": key[5],
            "layout": key[6],
            "layout_name": LAYOUT_NAMES.get(key[6], str(key[6])),
            "threads": rs[0]["threads"],
            "smem_bytes": rs[0]["smem_bytes"],
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
            "expected_claimed_mean": mean(rs, "expected_claimed"),
            "unique_claimed_mean": mean(rs, "unique_claimed"),
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "first_claim_min_min": min(r["first_claim_min"] for r in rs),
            "first_claim_max_max": max(r["first_claim_max"] for r in rs),
            "last_claim_min_min": min(r["last_claim_min"] for r in rs),
            "last_claim_max_max": max(r["last_claim_max"] for r in rs),
            "claim_cycles_avg_mean": mean(rs, "claim_cycles_avg"),
            "claim_cycles_max_max": max(r["claim_cycles_max"] for r in rs),
            "max_processed_max": max(r["max_processed"] for r in rs),
            "multi_claim_workers_mean": mean(rs, "multi_claim_workers"),
            "max_successes_worker_max": max(r["max_successes_worker"]
                                            for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "valid_runs": sum(1 for r in rs if r["validation"] == "ok"),
            "bytes_static_mean": mean(rs, "bytes_static"),
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
    ap.add_argument("--suite", choices=("threshold", "workload", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; run: TARGET=clc_decode_probe build_run.sh")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    rows = []
    failures = []
    cfgs = configs_for_suite(args.suite)
    total = len(cfgs) * args.repeats
    n = 0
    for config in cfgs:
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
                  f"tasks={config['tasks']} blocks={config['short_blocks']}/"
                  f"{config['long_blocks']} layout={row['layout_name']} "
                  f"R={row['predicted_r']} claim={row['claimed_min']}.."
                  f"{row['claimed_max']} delta={row['delta_pct']:+.1f}% "
                  f"valid={row['validation'] == 'ok'}")

    raw = OUTDIR / f"clc_decode_{args.suite}_raw_{stamp}.csv"
    summary = OUTDIR / f"clc_decode_{args.suite}_summary_{stamp}.csv"
    write_csv(raw, rows)
    write_csv(summary, aggregate(rows))
    print(f"raw={raw}")
    print(f"summary={summary}")

    if failures:
        print("validation failures:")
        for row, errors in failures[:20]:
            print(f"  suite={row['suite']} tasks={row['tasks']} "
                  f"blocks={row['short_blocks']}/{row['long_blocks']}: "
                  f"{'; '.join(errors)}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
