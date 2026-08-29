"""Fail-closed GPU exclusivity checks shared by serving experiments."""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time


TRIAL_OWNER_ENV = "NTA_BENCHMARK_TRIAL_OWNER"


def wait_for_free_gpu(limit_mib: int = 8000, timeout_seconds: float = 600.0) -> None:
    """Wait until no compute process owns the benchmark GPU."""

    deadline = time.monotonic() + timeout_seconds
    used_mib: int | None = None
    while True:
        memory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        try:
            used_mib = max(
                int(line) for line in memory.stdout.split() if line.strip()
            )
        except ValueError:
            used_mib = None
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        compute_pids = [
            line for line in applications.stdout.split() if line.strip()
        ]
        if memory.returncode == 0 and used_mib is not None:
            if used_mib < limit_mib and not compute_pids:
                return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory still at {used_mib} MiB after "
                f"{timeout_seconds:.0f}s; refusing to launch a serving arm into "
                "an occupied device"
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
            environment = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(
                b"\0"
            )
        except OSError:
            return False
        marker = f"{TRIAL_OWNER_ENV}={self.owner_token}".encode()
        return marker in environment

    def _sample_once(self) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                self.sampling_errors += 1
                return
            applications = {
                int(line)
                for line in result.stdout.split()
                if line.strip().isdigit()
            }
        except (OSError, subprocess.TimeoutExpired, ValueError):
            self.sampling_errors += 1
            return
        self.samples += 1
        descendants = self._descendants()
        foreign = {
            pid
            for pid in applications
            if not self._owned_by_trial(pid, descendants)
        }
        if foreign:
            self.foreign_samples += 1
            self.foreign_pids |= foreign

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
        self._thread.join(timeout=7)

    @property
    def complete(self) -> bool:
        return not self._thread.is_alive()
