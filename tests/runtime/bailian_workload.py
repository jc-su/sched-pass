#!/usr/bin/env python3
"""Deterministic contract tests for Bailian workload preparation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))

from experiments.bailian import (  # noqa: E402
    input_page_ids,
    normalize,
    read_jsonl_selection,
    unique_input_page_ids,
    write_workload,
)
from experiments.check_regression import compare  # noqa: E402
from experiments.prepare_serving_cohort import build_cohort  # noqa: E402
from experiments.run_evaluation import validate_spec  # noqa: E402
from experiments.validate_workload import validate  # noqa: E402
from SglangHiCacheLoad import _load_workload  # noqa: E402


ONLINE = [
    {
        "chat_id": "a",
        "timestamp": 10.0,
        "input_length": 32,
        "output_length": 4,
        "hash_ids": ["x", "y"],
        "turn": 0,
    },
    {
        "chat_id": "b",
        "timestamp": 10.5,
        "input_length": 48,
        "output_length": 5,
        "hash_ids": ["x", "y", "z"],
        "turn": 0,
    },
]


def main() -> None:
    manifest, rows = normalize(ONLINE, arrival_mode="trace", synthesize_prompts=True)
    assert manifest["arrival"]["production_arrival_claim"] is True
    assert rows[0]["arrival_seconds"] == 0.0
    assert rows[1]["arrival_seconds"] == 0.5
    assert rows[1]["shared_prefix_blocks"] == 2
    assert len(rows[1]["prompt_token_ids"]) == 48
    assert input_page_ids(rows[1]) == ("x", "y", "z")
    assert len(unique_input_page_ids(rows)) == 3

    # The prefix statistic must remain exact while using a linear trie rather
    # than rebuilding every tuple prefix for every request.
    trie_manifest, trie_rows = normalize(
        [
            {
                "chat_id": "p0",
                "input_length": 64,
                "output_length": 1,
                "hash_ids": ["a", "b", "c", "d"],
            },
            {
                "chat_id": "p1",
                "input_length": 48,
                "output_length": 1,
                "hash_ids": ["a", "b", "x"],
            },
            {
                "chat_id": "p2",
                "input_length": 32,
                "output_length": 1,
                "hash_ids": ["a", "z"],
            },
        ]
    )
    assert trie_manifest["statistics"]["shared_prefix_blocks"] == 3
    assert [row["shared_prefix_blocks"] for row in trie_rows] == [0, 2, 1]

    scaled_manifest, _ = normalize(ONLINE, arrival_mode="trace", time_scale=0.5)
    assert scaled_manifest["arrival"]["production_arrival_claim"] is False
    assert scaled_manifest["claims"]["arrival_is_production_trace"] is False

    session_rows = [
        {
            "chat_id": "session",
            "parent_chat_id": -1,
            "turn": 0,
            "timestamp": 1.0,
            "input_length": 16,
            "output_length": 2,
            "hash_ids": ["root"],
        },
        {
            "chat_id": "session",
            "parent_chat_id": 0,
            "turn": 1,
            "timestamp": 1.2,
            "input_length": 32,
            "output_length": 3,
            "hash_ids": ["root", "followup"],
        },
    ]
    session_manifest, session_rows = normalize(
        session_rows, arrival_mode="trace", state_policy="root_resident"
    )
    assert session_manifest["serving_state"]["synthetic"] is True
    assert [row["request_state"] for row in session_rows] == [
        "resident",
        "external",
    ]

    offline = [
        {"request_id": "one", "input_length": 16, "output_length": 1, "hash_ids": ["a"]}
    ]
    offline_manifest, offline_rows = normalize(offline, arrival_mode="batch_release")
    assert offline_manifest["claims"]["offline_row_order_is_arrival"] is False
    assert (
        offline_manifest["claims"]["serving_state_is_production_cache_state"] is False
    )
    assert offline_rows[0]["arrival_source"] == "batch_release_no_arrival_claim"
    try:
        normalize(offline, arrival_mode="trace")
    except ValueError as error:
        assert "timestamps" in str(error)
    else:
        raise AssertionError(
            "offline trace was incorrectly accepted as production arrival"
        )

    with tempfile.TemporaryDirectory(prefix="nta-bailian-") as directory:
        root = Path(directory)
        manifest, rows = normalize(
            [
                {**ONLINE[0], "parent_chat_id": -1, "turn": 0},
                {**ONLINE[1], "parent_chat_id": "a", "turn": 1},
            ],
            arrival_mode="trace",
            synthesize_prompts=True,
            state_policy="root_resident",
        )
        write_workload(root / "manifest.json", root / "records.jsonl", manifest, rows)
        validate(root / "manifest.json")

        absent_manifest, absent_rows = normalize(offline, arrival_mode="batch_release")
        write_workload(
            root / "absent-manifest.json",
            root / "absent-records.jsonl",
            absent_manifest,
            absent_rows,
        )
        absent_document = json.loads(
            (root / "absent-manifest.json").read_text(encoding="utf-8")
        )
        absent_document["claims"]["serving_state_is_production_cache_state"] = True
        (root / "absent-manifest.json").write_text(
            json.dumps(absent_document), encoding="utf-8"
        )
        try:
            validate(root / "absent-manifest.json")
        except ValueError as error:
            assert "absent serving state" in str(error)
        else:
            raise AssertionError("absent serving state was accepted as production")

        class LossyTokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                del add_special_tokens
                return list(range(len(text.split())))

            def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
                del skip_special_tokens
                return " ".join(f"token-{value}" for value in token_ids[:-1])

        loaded = _load_workload(root / "manifest.json", LossyTokenizer())
        assert loaded.metadata["tokenization_errors"] == len(rows)
        assert loaded.resident_arrival_offsets == (0.0,)
        assert loaded.external_arrival_offsets == (0.5,)

        cohort_manifest, cohort_rows = build_cohort(
            root / "manifest.json",
            resident_requests=1,
            external_requests=1,
            context_length=64,
            max_input_tokens=56,
            max_output_tokens=8,
            active_token_budget=96,
            arrival_mode="batch_release",
        )
        cohort_path = root / "cohort-manifest.json"
        write_workload(
            cohort_path,
            root / "cohort-records.jsonl",
            cohort_manifest,
            cohort_rows,
        )
        validated_cohort = validate(cohort_path)
        assert validated_cohort["cohort_heterogeneity"][
            "joint_shape_heterogeneity"
        ]
        assert validated_cohort["selection"]["active_tokens"] == sum(
            row["input_length"] + max(1, row["output_length"])
            for row in cohort_rows
        )
        assert {row["request_state"] for row in cohort_rows} == {
            "resident",
            "external",
        }
        assert all(row["arrival_seconds"] == 0.0 for row in cohort_rows)
        loaded_cohort = _load_workload(cohort_path, LossyTokenizer())
        assert loaded_cohort.resident_arrival_offsets == (0.0,)
        assert loaded_cohort.external_arrival_offsets == (0.0,)
        document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert document["request_count"] == 2
        fixture = (
            Path(__file__).resolve().parents[1] / "fixtures" / "bailian-online.jsonl"
        )
        cli_manifest = root / "cli-manifest.json"
        cli_records = root / "cli-records.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "experiments"
                    / "prepare_bailian.py"
                ),
                "--input",
                str(fixture),
                "--arrival-mode",
                "trace",
                "--synthesize-prompts",
                "--max-requests",
                "2",
                "--manifest",
                str(cli_manifest),
                "--records",
                str(cli_records),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        validate(cli_manifest)
        cli_document = json.loads(cli_manifest.read_text(encoding="utf-8"))
        assert cli_document["source_file"] == fixture.name
        assert not Path(cli_document["source_file"]).is_absolute()
        assert cli_document["selection"] == {
            "mode": "source_prefix",
            "max_requests": 2,
            "source_request_count": 3,
        }
        selected, source_count, source_digest = read_jsonl_selection(fixture, 2)
        assert len(selected) == 2
        assert source_count == 3
        assert source_digest == cli_document["source_digest"]
        malformed = root / "malformed.jsonl"
        malformed.write_bytes(b'{"input_length": 1}\nnot-json\n')
        try:
            read_jsonl_selection(malformed, 1)
        except ValueError as error:
            assert "line 2" in str(error)
        else:
            raise AssertionError("malformed suffix was silently ignored")
        opportunity = root / "rq0.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "experiments"
                    / "analyze_workload.py"
                ),
                str(cli_manifest),
                "--output",
                str(opportunity),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        opportunity_report = json.loads(opportunity.read_text(encoding="utf-8"))
        assert opportunity_report["classification"] == "bailian-rq0-opportunity-report"
        assert (
            opportunity_report["provenance"]["demand_trace_digest"]
            == cli_document["demand_trace_digest"]
        )
        assert (
            opportunity_report["compute_transfer_regime"]["status"]
            == "trace_only_not_identifiable"
        )
        assert (
            opportunity_report["exact_demand_shape"]["candidate_kv_blocks"]["total"] > 0
        )
        validate_spec(
            {
                "schema": 1,
                "classification": "nta-paired-evaluation",
                "workload_manifest": str(cli_manifest),
                "repetitions": 5,
                "experiments": [
                    {
                        "name": "fixture",
                        "variant": "nta",
                        "arm": "B5",
                        "tier": "host_mem",
                        "demand_semantics": "exact",
                        "stratum": {
                            "request_state": "mixed",
                            "granularity": "page_group",
                            "load_ratio": "balanced",
                            "availability_skew": "medium",
                            "staging_pressure": "near_capacity",
                            "arrival": "trace_timestamp",
                        },
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'classification':'nta-evaluation-fixture','verification_failures':0,'slo_goodput':1.0}))",
                        ],
                        "metrics": ["slo_goodput"],
                    },
                    {
                        "name": "fixture",
                        "variant": "stock",
                        "arm": "B1",
                        "tier": "host_mem",
                        "demand_semantics": "exact",
                        "stratum": {
                            "request_state": "mixed",
                            "granularity": "page_group",
                            "load_ratio": "balanced",
                            "availability_skew": "medium",
                            "staging_pressure": "near_capacity",
                            "arrival": "trace_timestamp",
                        },
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'classification':'nta-evaluation-fixture','verification_failures':0,'slo_goodput':0.9}))",
                        ],
                        "metrics": ["slo_goodput"],
                    },
                ],
                "comparisons": [
                    {
                        "name": "fixture-goodput",
                        "experiment": "fixture",
                        "numerator_variant": "nta",
                        "denominator_variant": "stock",
                        "metric": "slo_goodput",
                    }
                ],
            },
            cli_manifest,
        )
        spec_path = root / "evaluation.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "classification": "nta-paired-evaluation",
                    "workload_manifest": str(cli_manifest),
                    "repetitions": 5,
                    "seed": 7,
                    "experiments": [
                        {
                            "name": "fixture",
                            "variant": "nta",
                            "arm": "B5",
                            "tier": "host_mem",
                            "demand_semantics": "exact",
                            "stratum": {
                                "request_state": "mixed",
                                "granularity": "page_group",
                                "load_ratio": "balanced",
                                "availability_skew": "medium",
                                "staging_pressure": "near_capacity",
                                "arrival": "trace_timestamp",
                            },
                            "command": [
                                sys.executable,
                                "-c",
                                "import json; print(json.dumps({'classification':'nta-evaluation-fixture','verification_failures':0,'slo_goodput': 1.0}))",
                            ],
                            "metrics": ["slo_goodput"],
                        },
                        {
                            "name": "fixture",
                            "variant": "stock",
                            "arm": "B1",
                            "tier": "host_mem",
                            "demand_semantics": "exact",
                            "stratum": {
                                "request_state": "mixed",
                                "granularity": "page_group",
                                "load_ratio": "balanced",
                                "availability_skew": "medium",
                                "staging_pressure": "near_capacity",
                                "arrival": "trace_timestamp",
                            },
                            "command": [
                                sys.executable,
                                "-c",
                                "import json; print(json.dumps({'classification':'nta-evaluation-fixture','verification_failures':0,'slo_goodput': 0.9}))",
                            ],
                            "metrics": ["slo_goodput"],
                        },
                    ],
                    "comparisons": [
                        {
                            "name": "fixture-goodput",
                            "experiment": "fixture",
                            "numerator_variant": "nta",
                            "denominator_variant": "stock",
                            "metric": "slo_goodput",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        evaluation_output = root / "evaluation-output"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "experiments"
                    / "run_evaluation.py"
                ),
                "--spec",
                str(spec_path),
                "--output-dir",
                str(evaluation_output),
                "--allow-dirty",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "experiments"
                    / "validate_evaluation_artifact.py"
                ),
                str(evaluation_output),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        artifact_output = root / "evaluation-artifact"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2] / "experiments" / "reproduce.py"
                ),
                "--profile",
                "evaluation",
                "--spec",
                str(spec_path),
                "--output",
                str(artifact_output),
                "--allow-dirty",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2]
                    / "experiments"
                    / "validate_bundle.py"
                ),
                str(artifact_output),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        record = json.loads(
            (evaluation_output / "trials.jsonl").read_text().splitlines()[0]
        )
        assert record["tier"] == "host_mem"
        assert record["demand_semantics"] == "exact"
        baseline = {
            "schema": 1,
            "classification": "nta-performance-baseline",
            "report": {"verification_failures": 0, "graph_ms": 1.0},
            "metrics": [
                {
                    "name": "graph",
                    "path": "graph_ms",
                    "direction": "lower_is_better",
                    "relative_tolerance": 0.05,
                }
            ],
        }
        assert compare(baseline, {"verification_failures": 0, "graph_ms": 1.01})["pass"]
        assert not compare(baseline, {"verification_failures": 0, "graph_ms": 1.2})[
            "pass"
        ]
    print("bailian_workload=pass")


if __name__ == "__main__":
    main()
