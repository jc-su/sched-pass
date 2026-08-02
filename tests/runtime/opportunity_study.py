#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize-opportunity-study.py"


def report(model: str, tier: str, proceed: bool) -> dict[str, object]:
    return {
        "schema": 2,
        "classification": "incremental-execution-opportunity",
        "revision": "revision",
        "provenance": {
            "models": [model],
            "tiers": [tier],
            "measured_compute_tiles": 4,
            "gpu_timestamped_tiles": 2,
            "resident_at_launch_tiles": 2,
        },
        "opportunity": {"tiles": 4},
        "proceed": proceed,
    }


def run(
    paths: list[pathlib.Path], output: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *map(str, paths),
            "--output",
            str(output),
            "--require-proceed",
        ],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        paths = [directory / "host.json", directory / "nvme.json"]
        paths[0].write_text(
            json.dumps(report("model-a", "host_staged", True)), encoding="utf-8"
        )
        paths[1].write_text(
            json.dumps(report("model-b", "nvme", True)), encoding="utf-8"
        )
        output = directory / "study.json"
        assert run(paths, output).returncode == 0
        assert json.loads(output.read_text(encoding="utf-8"))["proceed"] is True

        paths[1].write_text(
            json.dumps(report("model-b", "host_staged", False)), encoding="utf-8"
        )
        assert run(paths, output).returncode == 2
        assert json.loads(output.read_text(encoding="utf-8"))["proceed"] is False


if __name__ == "__main__":
    main()
