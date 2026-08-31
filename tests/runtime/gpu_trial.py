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

    with patch.object(
        gpu_trial, "_run_nvidia_smi", return_value=completed("500.00\n")
    ):
        assert gpu_trial.query_gpu_power_limits() == (500.0,)

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

    sampler = gpu_trial.CotenantSampler("test-owner")
    with patch.object(
        gpu_trial,
        "_run_nvidia_smi",
        side_effect=(completed(""), completed("34, 1200, 88.5, 0x0, 500.0\n")),
    ):
        sampler._sample_once()
    assert sampler.samples == 1
    assert sampler.sampling_errors == 0
    assert sampler.telemetry_samples == 1
    assert sampler.temperature_min_c == sampler.temperature_max_c == 34
    assert sampler.power_limit_min_watts == 500.0
    assert sampler.power_limit_max_watts == 500.0
    evidence, failures = gpu_trial.trial_environment_evidence(
        sampler,
        expected_power_limit_watts=500.0,
        start_max_temperature_c=40,
    )
    assert not failures
    assert evidence["gpu_samples"] == 1
    assert evidence["gpu_graphics_clock_limit_mhz"] is None
    telemetry = evidence["gpu_environment"]
    assert isinstance(telemetry, dict)
    assert telemetry["thermal_slowdown_samples"] == 0

    sampler.thermal_slowdown_samples = 1
    _, failures = gpu_trial.trial_environment_evidence(
        sampler,
        expected_power_limit_watts=500.0,
        start_max_temperature_c=40,
    )
    assert any("thermal slowdown" in failure for failure in failures)

    sampler.thermal_slowdown_samples = 0
    _, failures = gpu_trial.trial_environment_evidence(
        sampler,
        expected_power_limit_watts=500.0,
        start_max_temperature_c=40,
        expected_graphics_clock_limit_mhz=1100,
    )
    assert any("graphics clock did not match" in failure for failure in failures)
    _, failures = gpu_trial.trial_environment_evidence(
        sampler,
        expected_power_limit_watts=500.0,
        start_max_temperature_c=40,
        expected_graphics_clock_limit_mhz=1200,
    )
    assert not failures

    failed_sampler = gpu_trial.CotenantSampler("test-owner")
    with patch.object(gpu_trial, "_run_nvidia_smi", return_value=None):
        failed_sampler._sample_once()
    assert failed_sampler.samples == 0
    assert failed_sampler.sampling_errors == 1

    print("gpu_trial=pass")


if __name__ == "__main__":
    main()
