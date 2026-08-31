#!/usr/bin/env python3
"""Pure tests for bounded serving-GPU readiness probes."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.serving import gpu_trial  # noqa: E402


def completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


def main() -> None:
    hung = Mock()
    hung.communicate.side_effect = subprocess.TimeoutExpired(["nvidia-smi"], 5)
    hung.wait.return_value = -9
    with patch.object(gpu_trial.subprocess, "Popen", return_value=hung):
        assert gpu_trial._run_nvidia_smi(["--query-gpu=memory.used"]) is None
    hung.kill.assert_called_once_with()

    calls: list[list[str]] = []

    def observe(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[0].startswith("--query-gpu"):
            return completed("512, 34\n")
        return completed("")

    with patch.object(gpu_trial, "_run_nvidia_smi", side_effect=observe):
        gpu_trial.wait_for_free_gpu(
            limit_mib=1024,
            timeout_seconds=1.0,
            max_temperature_c=40,
            stable_samples=1,
        )
    assert len(calls) == 2

    with (
        patch.object(
            gpu_trial,
            "_run_nvidia_smi",
            side_effect=(completed("512, 34\n"), completed("", returncode=1)),
        ),
        patch.object(gpu_trial.time, "monotonic", side_effect=(0.0, 2.0)),
    ):
        try:
            gpu_trial.wait_for_free_gpu(timeout_seconds=1.0)
        except RuntimeError as error:
            assert "refusing to launch" in str(error)
        else:
            raise AssertionError("failed compute-process query was accepted")

    print("gpu_trial=pass")


if __name__ == "__main__":
    main()
