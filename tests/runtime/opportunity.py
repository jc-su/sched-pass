#!/usr/bin/env python3

import pathlib
import subprocess
import sys
import tempfile

from nta_runtime.opportunity import (
    OperatorArrival,
    TileArrival,
    append_json_line,
    load_json_lines,
    summarize,
    summarize_provenance,
)


def main() -> None:
    records = load_json_lines(
        [
            '{"batch_id":"b0","layer":0,"tiles":['
            '{"request_id":"a","tile_id":0,"available_ns":0,"compute_ns":30},'
            '{"request_id":"a","tile_id":1,"available_ns":100,"compute_ns":30},'
            '{"request_id":"b","tile_id":0,"available_ns":20,"compute_ns":30}]}'
        ]
    )
    fine = summarize(
        records,
        launch_overhead_ns=5,
        grouping_window_ns=0,
        parallel_slots=1,
        material_delay_ns=50,
    )
    grouped = summarize(
        records,
        launch_overhead_ns=5,
        grouping_window_ns=100,
        parallel_slots=1,
        material_delay_ns=50,
    )
    assert fine.operators == 1 and fine.tiles == 3 and fine.requests == 2
    assert fine.available_before_atomic_launch == 2 / 3
    assert fine.material_available_before_atomic_launch == 2 / 3
    assert fine.operators_with_material_opportunity == 1.0
    assert fine.blocked_compute_ns == 60
    assert fine.blocked_compute_area_ns2 == 5400
    assert fine.atomic_makespan_ns == 195
    assert fine.incremental_makespan_ns == 135
    assert grouped.incremental_makespan_ns == 195
    assert fine.incremental_speedup > grouped.incremental_speedup
    parallel = summarize(
        records,
        launch_overhead_ns=5,
        grouping_window_ns=0,
        parallel_slots=2,
        material_delay_ns=50,
    )
    assert parallel.atomic_makespan_ns == 165
    assert parallel.incremental_makespan_ns == 135
    assert parallel.ideal_incremental_speedup >= parallel.incremental_speedup
    try:
        summarize(records, parallel_slots=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero CTA slots were accepted")
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary) / "trace.jsonl"
        append_json_line(path, records[0])
        restored = load_json_lines(path.read_text(encoding="utf-8").splitlines())
        assert restored == records
        qualified_path = pathlib.Path(temporary) / "qualified.jsonl"
        append_json_line(
            qualified_path,
            OperatorArrival(
                "qualified",
                0,
                (TileArrival("request", 0, 10, 20),),
                revision="abc",
                engine="sglang",
                model="model",
                tier="host_staged",
            ),
        )
        output = pathlib.Path(temporary) / "analysis.json"
        analyzer = (
            pathlib.Path(__file__).resolve().parents[2]
            / "scripts"
            / ("analyze-opportunity.py")
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                str(qualified_path),
                "--output",
                str(output),
            ],
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert completed.returncode == 0 and output.is_file()
    observed = OperatorArrival(
        "b1",
        1,
        (
            TileArrival("a", 0, 0, 30, availability_source="resident_at_launch"),
            TileArrival("b", 1, 100, 30),
        ),
        revision="abc",
        engine="sglang",
        model="model",
        tier="host_staged",
    )
    provenance = summarize_provenance((observed,))
    assert provenance.revisions == ("abc",)
    assert provenance.gpu_timestamped_tiles == 1
    assert provenance.resident_at_launch_tiles == 1
    assert provenance.calibrated_compute_tiles == 2
    assert provenance.measured_compute_tiles == 0


if __name__ == "__main__":
    main()
