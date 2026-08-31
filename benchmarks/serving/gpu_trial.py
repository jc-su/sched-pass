"""Fail-closed GPU exclusivity checks shared by serving experiments."""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time


TRIAL_OWNER_ENV = "NTA_BENCHMARK_TRIAL_OWNER"
NVIDIA_SMI_TIMEOUT_SECONDS = 5.0


def _run_nvidia_smi(arguments: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run one bounded telemetry query.

    ``nvidia-smi`` can block in driver teardown even when the GPU is otherwise
    healthy.  A readiness probe must therefore be a bounded observation, not
    an unbounded prerequisite that can strand an entire trial campaign.
    """

    command = ["nvidia-smi", *arguments]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    try:
        stdout, stderr = process.communicate(timeout=NVIDIA_SMI_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Popen.communicate's usual timeout recipe kills and then waits.  That
        # second wait is itself unbounded when the NVIDIA process is stuck in
        # an uninterruptible driver call.  Signal it without waiting and let a
        # daemon reaper collect it whenever the kernel releases the task.
        process.kill()
        threading.Thread(target=process.wait, daemon=True).start()
        return None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


def query_gpu_power_limits() -> tuple[float, ...]:
    """Return one bounded snapshot of every visible GPU power-limit contract."""

    result = _run_nvidia_smi(
        ["--query-gpu=power.limit", "--format=csv,noheader,nounits"]
    )
    try:
        limits = tuple(
            float(line.strip())
            for line in (result.stdout if result is not None else "").splitlines()
            if line.strip()
        )
    except ValueError as error:
        raise RuntimeError("GPU power-limit query returned invalid data") from error
    if (
        result is None
        or result.returncode != 0
        or not limits
        or any(limit <= 0.0 for limit in limits)
    ):
        raise RuntimeError("GPU power-limit contract is unavailable")
    return limits


def wait_for_free_gpu(
    limit_mib: int = 8000,
    timeout_seconds: float = 600.0,
    *,
    max_temperature_c: int | None = None,
    stable_samples: int = 1,
) -> None:
    """Wait for an unoccupied GPU at a reproducible thermal starting point."""

    if stable_samples <= 0:
        raise ValueError("GPU readiness stable-sample count must be positive")
    if max_temperature_c is not None and max_temperature_c <= 0:
        raise ValueError("GPU readiness temperature must be positive")

    deadline = time.monotonic() + timeout_seconds
    used_mib: int | None = None
    temperature_c: int | None = None
    ready_samples = 0
    while True:
        gpu_state = _run_nvidia_smi(
            [
                "--query-gpu=memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        try:
            rows = [
                tuple(part.strip() for part in line.split(","))
                for line in (
                    gpu_state.stdout if gpu_state is not None else ""
                ).splitlines()
                if line.strip()
            ]
            used_mib = max(int(row[0]) for row in rows)
            temperature_c = max(int(row[1]) for row in rows)
        except (IndexError, ValueError):
            used_mib = None
            temperature_c = None
        applications = _run_nvidia_smi(
            [
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        compute_pids = [
            line
            for line in (
                applications.stdout if applications is not None else ""
            ).split()
            if line.strip()
        ]
        if (
            gpu_state is not None
            and applications is not None
            and gpu_state.returncode == 0
            and applications.returncode == 0
            and used_mib is not None
        ):
            temperature_ready = max_temperature_c is None or (
                temperature_c is not None and temperature_c <= max_temperature_c
            )
            if used_mib < limit_mib and not compute_pids and temperature_ready:
                ready_samples += 1
                if ready_samples >= stable_samples:
                    return
            else:
                ready_samples = 0
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory still at {used_mib} MiB after "
                f"{timeout_seconds:.0f}s (temperature={temperature_c}C, "
                f"required<={max_temperature_c}C); refusing to launch a "
                "serving arm from an occupied or thermally biased device"
            )
        time.sleep(5.0)


class CotenantSampler:
    """Sample GPU processes not owned by one benchmark process tree."""

    def __init__(self, owner_token: str, interval_seconds: float = 1.0) -> None:
        if not owner_token or "\0" in owner_token:
            raise ValueError("co-tenant sampler owner token is invalid")
        if interval_seconds <= 0:
            raise ValueError("co-tenant sampling interval must be positive")
        self.owner_token = owner_token
        self.interval_seconds = interval_seconds
        self.samples = 0
        self.sampling_errors = 0
        self.foreign_samples = 0
        self.foreign_pids: set[int] = set()
        self.telemetry_samples = 0
        self.telemetry_errors = 0
        self.temperature_min_c: int | None = None
        self.temperature_max_c: int | None = None
        self.graphics_clock_min_mhz: int | None = None
        self.graphics_clock_max_mhz: int | None = None
        self.graphics_clock_sum_mhz = 0
        self.power_max_watts = 0.0
        self.power_limit_min_watts: float | None = None
        self.power_limit_max_watts: float | None = None
        self.thermal_slowdown_samples = 0
        self.clock_reason_masks: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _descendants() -> set[int]:
        pids = {os.getpid()}
        try:
            process_root = pathlib.Path("/proc")
            grew = True
            while grew:
                grew = False
                for stat in process_root.glob("[0-9]*/stat"):
                    try:
                        parts = stat.read_text().rsplit(") ", 1)[1].split()
                        pid = int(stat.parent.name)
                        parent = int(parts[1])
                    except (OSError, IndexError, ValueError):
                        continue
                    if parent in pids and pid not in pids:
                        pids.add(pid)
                        grew = True
        except OSError:
            pass
        return pids

    def _owned_by_trial(self, pid: int, descendants: set[int]) -> bool:
        if pid in descendants:
            return True
        try:
            environment = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            return False
        marker = f"{TRIAL_OWNER_ENV}={self.owner_token}".encode()
        return marker in environment

    def _sample_once(self) -> None:
        result = _run_nvidia_smi(["--query-compute-apps=pid", "--format=csv,noheader"])
        if result is None or result.returncode != 0:
            self.sampling_errors += 1
            return
        try:
            applications = {
                int(line) for line in result.stdout.split() if line.strip().isdigit()
            }
        except ValueError:
            self.sampling_errors += 1
            return
        self.samples += 1
        descendants = self._descendants()
        foreign = {
            pid for pid in applications if not self._owned_by_trial(pid, descendants)
        }
        if foreign:
            self.foreign_samples += 1
            self.foreign_pids |= foreign
        self._sample_telemetry()

    def _sample_telemetry(self) -> None:
        try:
            result = _run_nvidia_smi(
                [
                    "--query-gpu=temperature.gpu,clocks.current.graphics,"
                    "power.draw,clocks_throttle_reasons.active,power.limit",
                    "--format=csv,noheader,nounits",
                ]
            )
            rows = [
                tuple(part.strip() for part in line.split(","))
                for line in (result.stdout if result is not None else "").splitlines()
                if line.strip()
            ]
            if result is None or result.returncode != 0 or not rows:
                raise ValueError("nvidia-smi returned no GPU telemetry")
            temperatures = [int(row[0]) for row in rows]
            clocks = [int(row[1]) for row in rows]
            powers = [float(row[2]) for row in rows]
            masks = [int(row[3], 0) for row in rows]
            power_limits = [float(row[4]) for row in rows]
        except (OSError, IndexError, subprocess.TimeoutExpired, ValueError):
            self.telemetry_errors += 1
            return

        temperature_min = min(temperatures)
        temperature_max = max(temperatures)
        clock_min = min(clocks)
        clock_max = max(clocks)
        self.temperature_min_c = (
            temperature_min
            if self.temperature_min_c is None
            else min(self.temperature_min_c, temperature_min)
        )
        self.temperature_max_c = (
            temperature_max
            if self.temperature_max_c is None
            else max(self.temperature_max_c, temperature_max)
        )
        self.graphics_clock_min_mhz = (
            clock_min
            if self.graphics_clock_min_mhz is None
            else min(self.graphics_clock_min_mhz, clock_min)
        )
        self.graphics_clock_max_mhz = (
            clock_max
            if self.graphics_clock_max_mhz is None
            else max(self.graphics_clock_max_mhz, clock_max)
        )
        self.graphics_clock_sum_mhz += sum(clocks) // len(clocks)
        self.power_max_watts = max(self.power_max_watts, max(powers))
        minimum_limit = min(power_limits)
        maximum_limit = max(power_limits)
        self.power_limit_min_watts = (
            minimum_limit
            if self.power_limit_min_watts is None
            else min(self.power_limit_min_watts, minimum_limit)
        )
        self.power_limit_max_watts = (
            maximum_limit
            if self.power_limit_max_watts is None
            else max(self.power_limit_max_watts, maximum_limit)
        )
        self.clock_reason_masks.update(masks)
        # NVML clock-event bits 5 and 6 are software and hardware thermal
        # slowdown. Idle and power-cap reasons are recorded but are not thermal
        # contamination by themselves.
        if any(mask & 0x60 for mask in masks):
            self.thermal_slowdown_samples += 1
        self.telemetry_samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            if self._stop.wait(self.interval_seconds):
                return

    def __enter__(self) -> "CotenantSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2 * NVIDIA_SMI_TIMEOUT_SECONDS + 2.0)

    @property
    def complete(self) -> bool:
        return not self._thread.is_alive()

    def telemetry(self) -> dict[str, object]:
        return {
            "samples": self.telemetry_samples,
            "errors": self.telemetry_errors,
            "temperature_min_c": self.temperature_min_c,
            "temperature_max_c": self.temperature_max_c,
            "graphics_clock_min_mhz": self.graphics_clock_min_mhz,
            "graphics_clock_max_mhz": self.graphics_clock_max_mhz,
            "graphics_clock_mean_mhz": (
                self.graphics_clock_sum_mhz / self.telemetry_samples
                if self.telemetry_samples
                else None
            ),
            "power_max_watts": self.power_max_watts,
            "power_limit_min_watts": self.power_limit_min_watts,
            "power_limit_max_watts": self.power_limit_max_watts,
            "thermal_slowdown_samples": self.thermal_slowdown_samples,
            "thermal_slowdown_sample_fraction": (
                self.thermal_slowdown_samples / self.telemetry_samples
                if self.telemetry_samples
                else None
            ),
            "clock_reason_masks": [
                f"0x{mask:016x}" for mask in sorted(self.clock_reason_masks)
            ],
        }


def trial_environment_evidence(
    sampler: CotenantSampler,
    *,
    expected_power_limit_watts: float,
    start_max_temperature_c: int,
    expected_graphics_clock_limit_mhz: int | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Seal one serving arm's shared, fail-closed GPU environment evidence.

    Power, temperature, and clock observations are nuisance/outcome telemetry,
    never mechanism inputs. Missing telemetry, a changed administrator policy,
    or a foreign process invalidates a trial. Natural temperature and thermal-
    slowdown variation remains in the report and is handled by randomized
    paired repetitions. An explicit graphics-clock limit identifies a
    diagnostic sensitivity run; omission identifies the production-default
    DVFS policy used by headline serving trials.
    """

    if expected_power_limit_watts <= 0.0:
        raise ValueError("expected GPU power limit must be positive")
    if start_max_temperature_c <= 0:
        raise ValueError("GPU start temperature must be positive")
    if (
        expected_graphics_clock_limit_mhz is not None
        and expected_graphics_clock_limit_mhz <= 0
    ):
        raise ValueError("expected GPU graphics-clock limit must be positive")

    failures: list[str] = []
    if not sampler.complete:
        failures.append("co-tenant sampler did not terminate cleanly")
    if not sampler.samples:
        failures.append("GPU environment sampler recorded no successful sample")
    if sampler.sampling_errors:
        failures.append(
            "co-tenant sampler lost environmental samples: "
            f"{sampler.sampling_errors} errors"
        )
    if sampler.foreign_samples:
        failures.append(
            "GPU co-tenant contamination was observed in "
            f"{sampler.foreign_samples} samples "
            f"(pids={sorted(sampler.foreign_pids)})"
        )

    telemetry = sampler.telemetry()
    if not int(telemetry["samples"]):
        failures.append("GPU telemetry sampler recorded no successful sample")
    if int(telemetry["errors"]):
        failures.append(
            "GPU telemetry sampler lost environmental samples: "
            f"{telemetry['errors']} errors"
        )
    observed_graphics_clock_max = telemetry["graphics_clock_max_mhz"]
    if expected_graphics_clock_limit_mhz is not None and (
        observed_graphics_clock_max is None
        or abs(int(observed_graphics_clock_max) - expected_graphics_clock_limit_mhz)
        > 15
    ):
        failures.append(
            "GPU graphics clock did not match the declared sustainable limit "
            f"(expected={expected_graphics_clock_limit_mhz} MHz, "
            f"observed={observed_graphics_clock_max})"
        )
    observed_power_limits = (
        telemetry["power_limit_min_watts"],
        telemetry["power_limit_max_watts"],
    )
    if any(value is None for value in observed_power_limits) or any(
        abs(float(value) - expected_power_limit_watts) > 0.01
        for value in observed_power_limits
        if value is not None
    ):
        failures.append(
            "GPU power limit changed during the serving arm "
            f"(expected={expected_power_limit_watts:.2f}, "
            f"observed={observed_power_limits})"
        )

    evidence: dict[str, object] = {
        "gpu_samples": sampler.samples,
        "gpu_sampling_errors": sampler.sampling_errors,
        "gpu_sampling_complete": sampler.complete,
        "cotenant_gpu_samples": sampler.foreign_samples,
        "cotenant_pids_seen": sorted(sampler.foreign_pids),
        "gpu_environment": telemetry,
        "gpu_start_max_temperature_c": start_max_temperature_c,
        "gpu_graphics_clock_limit_mhz": expected_graphics_clock_limit_mhz,
        "gpu_clock_policy": (
            "production_default_dvfs"
            if expected_graphics_clock_limit_mhz is None
            else "fixed_diagnostic"
        ),
    }
    return evidence, failures
