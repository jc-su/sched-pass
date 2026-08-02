#!/usr/bin/env python3
"""Analyze request/tile arrival traces without assuming an online oracle."""

from __future__ import annotations

import argparse
import json
import pathlib

from nta_runtime.opportunity import (
    load_json_lines,
    summarize,
    summarize_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--launch-overhead-ns", type=int, default=0)
    parser.add_argument("--grouping-window-ns", type=int, default=0)
    arguments = parser.parse_args()
    with arguments.trace.open(encoding="utf-8") as source:
        records = load_json_lines(source)
    result = summarize(
        records,
        launch_overhead_ns=arguments.launch_overhead_ns,
        grouping_window_ns=arguments.grouping_window_ns,
    )
    provenance = summarize_provenance(records)
    if len(provenance.revisions) != 1 or not provenance.revisions[0]:
        raise ValueError("trace must contain one non-empty qualified revision")
    report = {
        "schema": 2,
        "revision": provenance.revisions[0],
        "classification": "incremental-execution-opportunity",
        "provenance": provenance.as_json(),
        "opportunity": result.as_json(),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
