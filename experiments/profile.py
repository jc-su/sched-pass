#!/usr/bin/env python3
"""Run an explicitly selected profiler with external artifact provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
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
    completed = subprocess.run(
        profiled,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    document.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "duration_seconds": time.monotonic() - started,
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
