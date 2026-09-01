#!/usr/bin/env python3
"""Pure tests for the stock-only serving saturation-rate freezer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.freeze_serving_overload_rate import (  # noqa: E402
    FreezeRule,
    PilotInput,
    build_freeze,
    main as freeze_main,
)


REVISION = "0123456789abcdef0123456789abcdef01234567"
MACHINE = {"hostname": "pilot-host", "gpu": "pilot-gpu", "kernel": "pilot"}
REQUESTS = 100


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def report(
    *, rate: float, throughput: float, system_latency: float, attainment: float
) -> dict[str, object]:
    qualified = round(REQUESTS * attainment)
    elapsed = REQUESTS / throughput
    records: list[dict[str, object]] = []
    outputs: list[dict[str, str]] = []
    for index in range(REQUESTS):
        request_id = f"request-{index:03d}"
        text_sha256 = hashlib.sha256(f"output-{index}".encode()).hexdigest()
        outputs.append({"request_id": request_id, "text_sha256": text_sha256})
        records.append(
            {
                "request_id": request_id,
                "kind": "resident" if index % 2 == 0 else "external",
                "input_tokens": 4096 + index,
                "completion_tokens": 2,
                "host_cached_tokens": 3072 if index % 2 else 0,
                "device_cached_tokens": 1024,
                "system_time_seconds": system_latency,
                "ttft_seconds": 0.05,
                "tpot_seconds": 0.01 if index < qualified else 0.20,
                "p99_itl_seconds": 0.01,
                "itl_sample_count": 1,
                "token_timestamps_exact": True,
                "text_sha256": text_sha256,
            }
        )
    aggregate_output = digest(sorted(outputs, key=lambda item: item["request_id"]))
    return {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "attention_backend": "flashinfer",
        "engine_stats": [],
        "dirty": False,
        "revision": REVISION,
        "machine": MACHINE,
        "model": "Qwen/test-model",
        "engine": "sglang",
        "engine_version": "test",
        "batch_mode": "open-loop",
        "serving_tier": "host",
        "seed": 17,
        "context_length": 32768,
        "resident_requests": 50,
        "external_requests": 50,
        "workload_family": "bailian-coder-window-17",
        "demand_semantics": "exact",
        "placement_proven": True,
        "load_warmup_excluded": True,
        "verification_failures": 0,
        "correctness": {
            "verification_failures": 0,
            "generated_text_sha256": aggregate_output,
        },
        "generated_text_sha256": aggregate_output,
        "request_rate": rate,
        "arrival_schedule": {
            "method": "seeded_exponential",
            "source_mode": "synthetic_open_loop",
            "source_target_rate_per_second": rate,
            "target_rate_per_second": rate,
            "uniform_time_scale": 1.0,
            "request_order_preserved": True,
        },
        "records": records,
        "elapsed_seconds": elapsed,
        "request_throughput": throughput,
        "p95_system_time_seconds": system_latency,
        "p99_system_time_seconds": system_latency,
        "slo_goodput": {
            "qualified_requests": qualified,
            "total_requests": REQUESTS,
            "requests_with_token_level_itl": REQUESTS,
            "attainment": qualified / REQUESTS,
            "goodput_requests_per_second": qualified / elapsed,
            "thresholds_seconds": {
                "ttft": 1.0,
                "tpot": 0.10,
                "p99_itl": 0.10,
            },
        },
    }


def write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(callable_value, text: str) -> None:
    try:
        callable_value()
    except ValueError as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"invalid saturation pilot was accepted: {text}")


def inputs(directory: Path) -> list[PilotInput]:
    values = (
        (8.0, 7.8, 1.0, 1.00),
        (12.0, 11.4, 1.1, 0.98),
        (18.0, 15.0, 1.9, 0.80),
        (26.0, 15.5, 2.1, 0.70),
    )
    result: list[PilotInput] = []
    for rate, throughput, latency, attainment in values:
        path = directory / f"stock-{rate:g}.json"
        write(
            path,
            report(
                rate=rate,
                throughput=throughput,
                system_latency=latency,
                attainment=attainment,
            ),
        )
        result.append(PilotInput(rate, path))
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        pilots = inputs(directory)
        output = directory / "frozen.json"
        arguments: list[str] = []
        for pilot in pilots:
            arguments.extend(("--input", f"{pilot.offered_rate:g}={pilot.path}"))
        arguments.extend(("--output", str(output)))
        assert freeze_main(arguments) == 0
        first_bytes = output.read_bytes()
        frozen = json.loads(first_bytes)
        assert frozen["arm"] == "stock"
        assert frozen["knee"]["offered_rate"] == 12.0
        assert frozen["knee"]["saturation_onset_rate"] == 18.0
        assert frozen["frozen_overload_rate"] == 18.0
        assert frozen["formal_overload"]["offered_rate"] == 18.0
        assert frozen["rule"]["name"] == "first_sustained_stock_multisignal_knee_v2"
        assert frozen["knee"]["transition_gray_zone"] == []
        assert len(frozen["inputs"]) == 4
        assert all(len(item["sha256"]) == 64 for item in frozen["inputs"])
        # No timestamp or ambient state enters the freeze: identical evidence
        # and thresholds produce byte-identical output.
        assert freeze_main(arguments) == 0
        assert output.read_bytes() == first_bytes
        reversed_arguments: list[str] = []
        for pilot in reversed(pilots):
            reversed_arguments.extend(
                ("--input", f"{pilot.offered_rate:g}={pilot.path}")
            )
        reversed_arguments.extend(("--output", str(output)))
        assert freeze_main(reversed_arguments) == 0
        assert output.read_bytes() == first_bytes

        # A rate can cross the throughput/latency thresholds before the SLO
        # threshold.  It is a measured transition gray-zone point, not a new
        # sustainable knee and not a reason to hide a later formal collapse.
        pilots = inputs(directory)
        gray_path = directory / "stock-16.json"
        write(
            gray_path,
            report(
                rate=16.0,
                throughput=14.3,
                system_latency=1.7,
                attainment=0.98,
            ),
        )
        gray_pilots = [*pilots[:2], PilotInput(16.0, gray_path), *pilots[2:]]
        gray_frozen = build_freeze(gray_pilots, FreezeRule())
        assert gray_frozen["knee"]["offered_rate"] == 12.0
        assert gray_frozen["knee"]["saturation_onset_rate"] == 18.0
        assert [
            item["offered_rate"]
            for item in gray_frozen["knee"]["transition_gray_zone"]
        ] == [16.0]
        assert gray_frozen["frozen_overload_rate"] == 18.0

        no_knee = copy.deepcopy(pilots)
        for pilot in no_knee:
            value = json.loads(pilot.path.read_text(encoding="utf-8"))
            value["request_throughput"] = pilot.offered_rate * 0.97
            value["elapsed_seconds"] = REQUESTS / value["request_throughput"]
            value["slo_goodput"]["goodput_requests_per_second"] = (
                value["slo_goodput"]["qualified_requests"] / value["elapsed_seconds"]
            )
            write(pilot.path, value)
        expect_failure(
            lambda: build_freeze(no_knee, FreezeRule()),
            "no stable multi-signal",
        )

        pilots = inputs(directory)
        unstable_path = pilots[-1].path
        unstable = json.loads(unstable_path.read_text(encoding="utf-8"))
        unstable["p95_system_time_seconds"] = 0.5
        unstable["p99_system_time_seconds"] = 0.5
        for record in unstable["records"]:
            record["system_time_seconds"] = 0.5
        write(unstable_path, unstable)
        expect_failure(
            lambda: build_freeze(pilots, FreezeRule()),
            "system-latency reversal",
        )

        pilots = inputs(directory)
        nta_path = pilots[2].path
        nta = json.loads(nta_path.read_text(encoding="utf-8"))
        nta["attention_backend"] = "nta_flashinfer"
        write(nta_path, nta)
        expect_failure(
            lambda: build_freeze(pilots, FreezeRule()),
            "not the stock FlashInfer arm",
        )

        pilots = inputs(directory)
        mislabeled_path = pilots[2].path
        mislabeled = json.loads(mislabeled_path.read_text(encoding="utf-8"))
        mislabeled["arrival_schedule"]["target_rate_per_second"] = 12.0
        write(mislabeled_path, mislabeled)
        expect_failure(
            lambda: build_freeze(pilots, FreezeRule()),
            "arrival schedule disagrees",
        )

        pilots = inputs(directory)
        divergent_path = pilots[1].path
        divergent = json.loads(divergent_path.read_text(encoding="utf-8"))
        divergent["records"][0]["text_sha256"] = "f" * 64
        outputs = [
            {
                "request_id": record["request_id"],
                "text_sha256": record["text_sha256"],
            }
            for record in divergent["records"]
        ]
        aggregate = digest(sorted(outputs, key=lambda item: item["request_id"]))
        divergent["generated_text_sha256"] = aggregate
        divergent["correctness"]["generated_text_sha256"] = aggregate
        write(divergent_path, divergent)
        expect_failure(
            lambda: build_freeze(pilots, FreezeRule()),
            "output identity digest",
        )
    print("serving_overload_rate=pass")


if __name__ == "__main__":
    main()
