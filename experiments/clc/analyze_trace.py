#!/usr/bin/env python3
"""Summarize CLC trace-event CSVs.

The trace probe records successful claim completions in the order visible to
kernel code. This analyzer turns a full trace into stable policy facts: whether
the claimed set is contiguous, how non-FIFO the observed order is, how claims
split into waves, and how many claims each worker performs.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


NUMERIC_EVENTS = {
    "seq": int,
    "worker_linear": int,
    "worker_x": int,
    "worker_y": int,
    "processed_before": int,
    "claimed_linear": int,
    "claimed_x": int,
    "claimed_y": int,
    "claimed_z": int,
    "claim_cycles": int,
}

NUMERIC_SUMMARY = {
    "grid_x": int,
    "grid_y": int,
    "tasks": int,
    "threads": int,
    "work_cycles": int,
    "trace_cap": int,
    "trace_count": int,
    "trace_recorded": int,
    "monotonic_trace": int,
    "processed": int,
    "active_workers": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "max_processed": int,
    "multi_claim_workers": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}


def read_rows(path, numeric):
    with Path(path).open(newline="") as f:
        return [{k: numeric[k](v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def quantile(sorted_values, p):
    if not sorted_values:
        return 0
    idx = round((len(sorted_values) - 1) * p)
    return sorted_values[idx]


def inversion_stats(claimed):
    adjacent = 0
    total = 0
    max_backstep = 0
    max_forward_step = 0
    for i in range(1, len(claimed)):
        step = claimed[i] - claimed[i - 1]
        if step < 0:
            adjacent += 1
            max_backstep = max(max_backstep, -step)
        else:
            max_forward_step = max(max_forward_step, step)

    # Full inversion count is useful for traces of this size and still cheap.
    seen = []
    for value in claimed:
        total += sum(1 for prior in seen if prior > value)
        seen.append(value)
    return adjacent, total, max_backstep, max_forward_step


def summarize(events, summary_row):
    claimed = [row["claimed_linear"] for row in events]
    cycles = sorted(row["claim_cycles"] for row in events)
    workers = Counter(row["worker_linear"] for row in events)
    waves = defaultdict(list)
    for row in events:
        waves[row["processed_before"]].append(row["claimed_linear"])

    adjacent_inv, total_inv, max_backstep, max_forward_step = inversion_stats(
        claimed)
    claimed_min = min(claimed) if claimed else 0
    claimed_max = max(claimed) if claimed else 0
    unique_claims = len(set(claimed))
    holes = (claimed_max - claimed_min + 1 - unique_claims) if claimed else 0
    attempts_minus_successes = 0
    if summary_row:
        attempts_minus_successes = (
            summary_row["attempts"] - summary_row["successes"])

    row = {
        "events": len(events),
        "claimed_min": claimed_min,
        "claimed_max": claimed_max,
        "unique_claims": unique_claims,
        "claim_range_holes": holes,
        "first_claim": claimed[0] if claimed else 0,
        "last_claim": claimed[-1] if claimed else 0,
        "adjacent_inversions": adjacent_inv,
        "total_inversions": total_inv,
        "max_backstep": max_backstep,
        "max_forward_step": max_forward_step,
        "cycle_min": cycles[0] if cycles else 0,
        "cycle_p50": quantile(cycles, 0.50),
        "cycle_p90": quantile(cycles, 0.90),
        "cycle_p99": quantile(cycles, 0.99),
        "cycle_max": cycles[-1] if cycles else 0,
        "cycle_mean": statistics.fmean(cycles) if cycles else 0.0,
        "workers_with_claims": len(workers),
        "worker_claims_min": min(workers.values()) if workers else 0,
        "worker_claims_max": max(workers.values()) if workers else 0,
        "worker_claims_mean": statistics.fmean(workers.values())
        if workers else 0.0,
        "waves": len(waves),
        "attempts_minus_successes": attempts_minus_successes,
        "final_failed_attempts_equal_active_workers": 0,
    }

    if summary_row:
        row.update({
            "tasks": summary_row["tasks"],
            "predicted_r": summary_row["predicted_r"],
            "active_workers": summary_row["active_workers"],
            "attempts": summary_row["attempts"],
            "successes": summary_row["successes"],
            "structural_ok": summary_row["structural_ok"],
        })
        row["final_failed_attempts_equal_active_workers"] = int(
            attempts_minus_successes == summary_row["active_workers"])

    for wave, values in sorted(waves.items()):
        values = sorted(values)
        prefix = f"wave{wave}"
        row[f"{prefix}_count"] = len(values)
        row[f"{prefix}_min"] = values[0]
        row[f"{prefix}_max"] = values[-1]
        row[f"{prefix}_holes"] = values[-1] - values[0] + 1 - len(set(values))

    return row


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = read_rows(args.events, NUMERIC_EVENTS)
    summary_row = None
    if args.summary:
        summaries = read_rows(args.summary, NUMERIC_SUMMARY)
        if summaries:
            summary_row = summaries[0]

    row = summarize(events, summary_row)
    write_csv(args.out, [row])
    print(f"analysis={args.out}")


if __name__ == "__main__":
    main()
