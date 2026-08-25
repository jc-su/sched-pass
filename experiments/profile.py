#!/usr/bin/env python3
"""Run an explicitly selected profiler with external artifact provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _choose(tool: str) -> str | None:
    if tool != "auto":
        return shutil.which(tool)
    for candidate in ("nsys", "ncu", "perf"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _profile_command(tool: str, output: Path, command: Sequence[str]) -> list[str]:
    prefix = output / "profile"
    if tool == "nsys":
        return [
            tool,
            "profile",
            "--force-overwrite=true",
            "--output",
            str(prefix),
            "--",
            *command,
        ]
    if tool == "ncu":
        return [
            tool,
            "--force-overwrite",
            "--set",
            "full",
            "--export",
            str(prefix),
            "--",
            *command,
        ]
    if tool == "perf":
        return [
            tool,
            "stat",
            "-x,",
            "-o",
            str(output / "perf-stat.csv"),
            "--",
            *command,
        ]
    raise ValueError(f"unsupported profiler {tool}")


def _outputs(output: Path, tool: str) -> list[str]:
    """Return non-empty profiler outputs relative to the artifact directory."""
    patterns = {
        "nsys": ("profile*.nsys-rep", "profile*.qdrep"),
        "ncu": ("profile*.ncu-rep",),
        "perf": ("perf-stat.csv",),
    }
    found: set[Path] = set()
    for pattern in patterns[tool]:
        found.update(path for path in output.glob(pattern) if path.is_file())
    return sorted(
        str(path.relative_to(output))
        for path in found
        if path.stat().st_size > 0
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tool", choices=("auto", "nsys", "ncu", "perf"), default="auto"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--require-tool",
        action="store_true",
        help="fail instead of recording unavailable profiler evidence",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        parser.error("a command after -- is required")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("profiling output must be outside the source tree")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"profiling output is not empty: {output}")
    status = _git("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit(
            "profiling requires a clean checkout; use --allow-dirty for local debugging"
        )
    tool_path = _choose(args.tool)
    if tool_path is None:
        document = {
            "schema": 1,
            "classification": "nta-profile",
            "status": "unavailable",
            "requested_tool": args.tool,
            "command": command,
            "outputs": [],
            "revision": _git("rev-parse", "HEAD"),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "profile.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("profiler=unavailable")
        return 2 if args.require_tool else 0
    tool = Path(tool_path).name
    profiled = _profile_command(tool, output, command)
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": 1,
        "classification": "nta-profile",
        "status": "planned",
        "tool": tool,
        "tool_path": tool_path,
        "command": command,
        "profile_command": profiled,
        "outputs": [],
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output / "profile.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({"tool": tool, "command": profiled}, sort_keys=True))
        return 0
    started = time.monotonic()
    profiler_tmp: Path | None = None
    profiler_environment = os.environ.copy()
    if tool == "nsys":
        # Nsight Systems creates a session directory independently of the
        # report output. A stale root-owned /tmp/nvidia directory must not
        # force an otherwise unprivileged artifact run through sudo.
        profiler_tmp = Path(
            tempfile.mkdtemp(prefix="nta-nsys-", dir=str(output.parent))
        )
        profiler_environment["TMPDIR"] = str(profiler_tmp)
    try:
        completed = subprocess.run(
            profiled,
            cwd=ROOT,
            env=profiler_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finally:
        if profiler_tmp is not None:
            shutil.rmtree(profiler_tmp, ignore_errors=True)
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    document.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "duration_seconds": time.monotonic() - started,
            "outputs": _outputs(output, tool),
        }
    )
    (output / "profile.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode:
        print(f"profiler failed; see {output / 'stdout.log'}", file=sys.stderr)
        return completed.returncode
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
