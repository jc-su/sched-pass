#!/usr/bin/env python3
"""Run the small dependency-free matrix used by the CTest contract gate.

This is an experiment smoke test, not implementation logic.  It intentionally
executes the canonical runner and validator in a temporary directory so CTest
proves the artifact workflow without placing generated data in the source
tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_work_unit_matrix.py"
VALIDATOR = ROOT / "experiments" / "validate_matrix_artifact.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nta-matrix-") as temporary:
        artifact = Path(temporary) / "matrix.json"
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--max-cases",
                "2",
                "--repetitions",
                "1",
                "--ablation",
                "all",
                "--output",
                str(artifact),
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                str(artifact),
                "--require-all-ablations",
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        data = json.loads(artifact.read_text(encoding="utf-8"))
        if len(data["records"]) != 2 * 7 * 8:
            raise RuntimeError("matrix smoke artifact has incomplete record coverage")
    print("experiment_matrix=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
