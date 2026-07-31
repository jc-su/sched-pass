#!/usr/bin/env python3
"""Run CLC worker/SM mapping captures."""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import analyze_mapping


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_mapping_probe"
OUTDIR = Path(__file__).resolve().parent / "results"


def configs_for_suite(suite):
    if suite in ("threads", "all"):
        for threads in (64, 128, 256):
            yield {
                "suite": "mapping-threads",
                "tasks": 8192,
                "threads": threads,
                "work_cycles": 4096,
                "smem_bytes": 0,
            }
    if suite in ("smem", "all"):
        for smem_bytes in (0, 8192, 16384, 24576):
            yield {
                "suite": "mapping-smem",
                "tasks": 8192,
                "threads": 128,
                "work_cycles": 4096,
                "smem_bytes": smem_bytes,
            }
    if suite in ("threshold", "all"):
        for tasks in (1024, 2256, 2304):
            yield {
                "suite": "mapping-threshold",
                "tasks": tasks,
                "threads": 128,
                "work_cycles": 4096,
                "smem_bytes": 0,
            }


def run_probe(config, env_key, path):
    env = dict(os.environ)
    env[env_key] = "1"
    cmd = [
        str(BIN),
        str(config["tasks"]),
        str(config["threads"]),
        str(config["work_cycles"]),
        str(config["smem_bytes"]),
    ]
    with path.open("w", newline="") as f:
        subprocess.run(cmd, env=env, text=True, stdout=f, check=True)


def validate(row):
    errors = []
    if int(row["summary_structural_ok"]) != 1:
        errors.append("summary_structural_ok=0")
    if int(row["prefix_unique_sms_per_full_wave"]) != 1:
        errors.append("prefix wave did not cover all SMs")
    if int(row["prefix_repeating_sm_order"]) != 1:
        errors.append("prefix SM order did not repeat")
    if int(row["prefix_contiguous_fill_by_smid"]) != 0:
        errors.append("prefix unexpectedly looked contiguous-filled")
    if int(row["tasks"]) >= int(row["predicted_r"]) and \
            int(row["prefix_single_residue_per_smid"]) != 1:
        errors.append("SM raw ids were not one residue class")
    if int(row["suffix_records"]) > 0:
        if int(row["suffix_worker_is_prefix"]) != 1:
            errors.append("suffix worker outside prefix")
        if int(row["suffix_exec_smid_equals_worker_smid"]) != 1:
            errors.append("suffix execution SM changed")
    return errors


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("threads", "smem", "threshold", "all"),
                    default="all")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; build clc_mapping_probe first")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    failures = []
    for idx, config in enumerate(configs_for_suite(args.suite)):
        base = (
            f"clc_mapping_{args.suite}_{idx}_"
            f"t{config['tasks']}_b{config['threads']}_"
            f"s{config['smem_bytes']}_{stamp}"
        )
        events_path = OUTDIR / f"{base}_events.csv"
        summary_path = OUTDIR / f"{base}_summary.csv"
        run_probe(config, "CLC_MAP_EVENTS_CSV", events_path)
        run_probe(config, "CLC_PROBE_CSV", summary_path)

        events = analyze_mapping.read_rows(events_path,
                                           analyze_mapping.EVENT_NUMERIC)
        summary = analyze_mapping.read_summary(summary_path)
        row = analyze_mapping.analyze(events, summary)
        row.update(config)
        errors = validate(row)
        row["validation"] = "ok" if not errors else ";".join(errors)
        rows.append(row)
        status = "ok" if not errors else row["validation"]
        print(f"[mapping] config={config} {status}")
        if errors:
            failures.append(row)

    analysis_path = OUTDIR / f"clc_mapping_{args.suite}_analysis_{stamp}.csv"
    write_csv(analysis_path, rows)
    print(f"analysis={analysis_path}")
    if failures:
        raise SystemExit(f"{len(failures)} invalid mapping rows")


if __name__ == "__main__":
    main()
