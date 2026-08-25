"""Small, dependency-free helpers for reproducible experiment artifacts.

This module belongs to the experiment layer.  It records the environment and
executes commands, but it does not import or implement runtime mechanisms.
Keeping it standard-library-only lets the artifact contract run before the
CUDA/PyTorch/SGLang environment is installed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _command(argv: Sequence[str], *, timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _source_digest() -> str:
    """Hash tracked source names and bytes without depending on Git objects."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256()
    for encoded_path in listing.split(b"\0"):
        if not encoded_path:
            continue
        path = ROOT / encoded_path.decode("utf-8")
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    revision = _command(("git", "rev-parse", "HEAD")) or "unrecorded"
    status = _command(("git", "status", "--porcelain=v1", "--untracked-files=all"))
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        staged = subprocess.run(
            ["git", "diff", "--binary", "--cached", "--"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        diff_digest = _digest(diff + staged)
    except (OSError, subprocess.SubprocessError):
        diff_digest = None
    try:
        source_digest = _source_digest()
    except (OSError, subprocess.SubprocessError):
        source_digest = None
    return {
        "revision": revision,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "source_digest": source_digest,
        "worktree_diff_digest": diff_digest,
    }


def machine_metadata() -> dict[str, Any]:
    distributions = {}
    for name in (
        "torch",
        "flashinfer-python",
        "sglang",
        "sglang-kernel",
        "vllm",
        "apache-tvm-ffi",
        "openai",
        "setuptools",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "commands": {
            name: _command((name, "--version"))
            for name in ("cmake", "ninja", "clang++", "nvcc")
            if shutil.which(name)
        },
        "nvidia_smi": _command(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            )
        ),
        "cuda_devices": _command(("nvidia-smi", "-L")),
        "python_distributions": distributions,
    }


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    environment: dict[str, str]
    returncode: int
    duration_seconds: float
    log: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ArtifactRun:
    """Own one artifact directory and its command/log manifest."""

    def __init__(self, output: Path, *, profile: str, arguments: Sequence[str]):
        self.output = output.resolve()
        try:
            self.output.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise ValueError(
                "artifact output must be outside the source tree; use /tmp or "
                "another external artifact directory"
            )
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(
                f"artifact output is not empty; refusing to overwrite an older run: {self.output}"
            )
        self.logs = self.output / "logs"
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.arguments = tuple(arguments)
        self.commands: list[CommandResult] = []
        self.metadata: dict[str, Any] = {
            "schema": 1,
            "classification": "nta-reproducible-artifact",
            "profile": profile,
            "arguments": list(arguments),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repository": git_metadata(),
            "machine": machine_metadata(),
            "status": "running",
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.metadata["commands"] = [item.as_dict() for item in self.commands]
        self.metadata["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if self.metadata.get("status") != "running":
            self.metadata["finished_at"] = self.metadata["updated_at"]
        (self.output / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output / "commands.json").write_text(
            json.dumps(self.metadata["commands"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def update(self, **fields: Any) -> None:
        self.metadata.update(fields)
        self._write_metadata()

    def command(
        self,
        argv: Sequence[str],
        *,
        name: str,
        cwd: Path = ROOT,
        environment: dict[str, str] | None = None,
    ) -> CommandResult:
        log_path = self.logs / f"{len(self.commands):03d}-{name}.log"
        env = os.environ.copy()
        if environment:
            env.update(environment)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        result = CommandResult(
            tuple(str(value) for value in argv),
            str(cwd),
            dict(environment or {}),
            completed.returncode,
            time.monotonic() - started,
            str(log_path.relative_to(self.output)),
        )
        self.commands.append(result)
        self._write_metadata()
        if completed.returncode != 0:
            raise RuntimeError(
                f"artifact command failed ({completed.returncode}); see {log_path}"
            )
        return result

    def finish(self, *, status: str = "complete", **fields: Any) -> None:
        self.metadata.update(fields)
        self.metadata["status"] = status
        self._write_metadata()
