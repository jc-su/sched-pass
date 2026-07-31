#!/usr/bin/env python3
"""Analyze CLC task-to-worker/SM mapping CSVs."""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EVENT_NUMERIC = {
    "raw": int,
    "worker": int,
    "initial_smid": int,
    "exec_smid": int,
    "ordinal": int,
    "claimed": int,
}

SUMMARY_NUMERIC = {
    "tasks": int,
    "threads": int,
    "work_cycles": int,
    "smem_bytes": int,
    "processed": int,
    "active_workers": int,
    "attempts": int,
    "successes": int,
    "failures": int,
    "claimed_records": int,
    "claimed_min": int,
    "claimed_max": int,
    "missed": int,
    "duplicates": int,
    "bad_records": int,
    "claim_range_holes": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "workers_per_sm_min": int,
    "workers_per_sm_max": int,
    "workers_per_sm_mean": float,
    "smid_changes": int,
    "structural_ok": int,
}


def read_rows(path, numeric):
    with Path(path).open(newline="") as f:
        return [{k: numeric[k](v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def read_summary(path):
    rows = read_rows(path, SUMMARY_NUMERIC)
    return rows[0] if rows else {}


def mean(values):
    return statistics.fmean(values) if values else 0.0


def analyze(events, summary):
    tasks = summary["tasks"]
    sm_count = summary["sm_count"]
    r = summary["predicted_r"]
    active = summary["active_workers"]

    by_raw = {row["raw"]: row for row in events}
    prefix = [by_raw[i] for i in range(min(active, tasks))]
    suffix = [by_raw[i] for i in range(r, tasks)] if tasks > r else []

    prefix_seq = [row["initial_smid"] for row in prefix]
    full_waves = len(prefix_seq) // sm_count if sm_count else 0
    first_order = prefix_seq[:sm_count]
    repeating_order = 1
    unique_per_full_wave = 1
    for wave in range(full_waves):
      chunk = prefix_seq[wave * sm_count:(wave + 1) * sm_count]
      if len(set(chunk)) != sm_count:
        unique_per_full_wave = 0
      if chunk != first_order:
        repeating_order = 0

    by_smid_prefix = defaultdict(list)
    for row in prefix:
        by_smid_prefix[row["initial_smid"]].append(row["raw"])

    stride_values = []
    residue_counts = []
    contiguous_smid_ranges = 1
    for raws in by_smid_prefix.values():
        raws = sorted(raws)
        residue_counts.append(len(set(raw % sm_count for raw in raws)))
        if len(raws) > 1:
            diffs = [b - a for a, b in zip(raws, raws[1:])]
            stride_values.extend(diffs)
            if any(diff != 1 for diff in diffs):
                contiguous_smid_ranges = 0

    prefix_per_sm = [len(v) for v in by_smid_prefix.values()]

    by_smid_suffix = defaultdict(list)
    suffix_worker_prefix_ok = 1
    suffix_exec_same_smid = 1
    for row in suffix:
        by_smid_suffix[row["initial_smid"]].append(row["raw"])
        if row["worker"] >= active:
            suffix_worker_prefix_ok = 0
        if row["exec_smid"] != row["initial_smid"]:
            suffix_exec_same_smid = 0

    suffix_per_sm = [len(v) for v in by_smid_suffix.values()]
    ordinal_counts = Counter(row["ordinal"] for row in suffix)

    row = {
        "tasks": tasks,
        "threads": summary["threads"],
        "work_cycles": summary["work_cycles"],
        "smem_bytes": summary["smem_bytes"],
        "predicted_r": r,
        "active_workers": active,
        "sm_count": sm_count,
        "occ_blocks_per_sm": summary["occ_blocks_per_sm"],
        "prefix_full_sm_waves": full_waves,
        "prefix_repeating_sm_order": repeating_order,
        "prefix_unique_sms_per_full_wave": unique_per_full_wave,
        "prefix_contiguous_fill_by_smid": contiguous_smid_ranges,
        "prefix_sm_order_first_32": " ".join(map(str, first_order[:32])),
        "prefix_per_sm_min": min(prefix_per_sm) if prefix_per_sm else 0,
        "prefix_per_sm_max": max(prefix_per_sm) if prefix_per_sm else 0,
        "prefix_per_sm_mean": mean(prefix_per_sm),
        "prefix_stride_min": min(stride_values) if stride_values else 0,
        "prefix_stride_max": max(stride_values) if stride_values else 0,
        "prefix_single_residue_per_smid": int(
            all(count == 1 for count in residue_counts)),
        "suffix_records": len(suffix),
        "suffix_worker_is_prefix": suffix_worker_prefix_ok,
        "suffix_exec_smid_equals_worker_smid": suffix_exec_same_smid,
        "suffix_active_sms": len(by_smid_suffix),
        "suffix_per_sm_min": min(suffix_per_sm) if suffix_per_sm else 0,
        "suffix_per_sm_max": max(suffix_per_sm) if suffix_per_sm else 0,
        "suffix_per_sm_mean": mean(suffix_per_sm),
        "summary_structural_ok": summary["structural_ok"],
    }

    for ordinal in sorted(ordinal_counts):
        row[f"suffix_ordinal_{ordinal}_count"] = ordinal_counts[ordinal]

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
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = read_rows(args.events, EVENT_NUMERIC)
    summary = read_summary(args.summary)
    row = analyze(events, summary)
    write_csv(args.out, [row])
    print(f"analysis={args.out}")


if __name__ == "__main__":
    main()
