#!/usr/bin/env python3
"""Extract high-signal Nsight Compute metrics from CSV output."""

import argparse
import csv
from pathlib import Path


WANTED = {
    "Duration",
    "SM Frequency",
    "Elapsed Cycles",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "DRAM Throughput",
    "Block Size",
    "Grid Size",
    "Cluster Scheduling Policy",
    "Registers Per Thread",
    "Driver Shared Memory Per Block",
    "Dynamic Shared Memory Per Block",
    "Static Shared Memory Per Block",
    "# SMs",
    "# TPCs",
    "Waves Per SM",
    "Block Limit Barriers",
    "Block Limit SM",
    "Block Limit Registers",
    "Block Limit Shared Mem",
    "Block Limit Warps",
    "Theoretical Active Warps per SM",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Achieved Active Warps Per SM",
    "One or More Eligible",
    "Issued Warp Per Scheduler",
    "No Eligible",
    "Active Warps Per Scheduler",
    "Eligible Warps Per Scheduler",
    "Warp Cycles Per Issued Instruction",
    "Warp Cycles Per Executed Instruction",
    "Avg. Active Threads Per Warp",
    "Avg. Not Predicated Off Threads Per Warp",
    "Local Memory Spilling Requests",
    "Shared Memory Spilling Requests",
    "Avg. Executed Instructions Per Scheduler",
    "Executed Instructions",
    "Avg. Issued Instructions Per Scheduler",
    "Issued Instructions",
}


def read_ncu_csv(path):
    with Path(path).open(newline="") as f:
        header = None
        for line in f:
            if line.startswith('"ID"'):
                header = next(csv.reader([line]))
                break
        if header is None:
            return []
        return list(csv.DictReader(f, fieldnames=header))


def summarize(paths):
    out = []
    for path in paths:
        for row in read_ncu_csv(path):
            if row["Metric Name"] not in WANTED:
                continue
            out.append({
                "source": str(path),
                "kernel": row["Kernel Name"],
                "section": row["Section Name"],
                "metric": row["Metric Name"],
                "unit": row["Metric Unit"],
                "value": row["Metric Value"],
            })
    return out


def write_csv(path, rows):
    fieldnames = ["source", "kernel", "section", "metric", "unit", "value"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args()
    rows = summarize(args.csv)
    write_csv(args.out, rows)
    print(f"summary={args.out}")


if __name__ == "__main__":
    main()
