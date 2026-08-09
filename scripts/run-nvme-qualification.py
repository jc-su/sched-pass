#!/usr/bin/env python3
"""Run matched CPU and GPU-controlled read qualification on one VFIO NVMe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdf", default=os.environ.get("NTA_NVME_BDF", "0000:d8:00.0"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--namespace", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--progress-passes", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--fio-runtime", type=int, default=10)
    parser.add_argument("--fio-size", default="2G")
    parser.add_argument(
        "--media-policy",
        choices=("hardware-write-protect", "trusted-read-only-code"),
        default="hardware-write-protect",
    )
    parser.add_argument("--minimum-bandwidth-ratio", type=float, default=0.5)
    parser.add_argument("--cta-try-issue", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "results" / "qualification" / "nvme-qualification.json",
    )
    args = parser.parse_args()
    if min(
        args.queue_depth,
        args.bytes,
        args.requests,
        args.progress_passes,
        args.iterations,
        args.fio_runtime,
    ) <= 0:
        parser.error("NVMe qualification dimensions must be positive")
    if not 0 < args.minimum_bandwidth_ratio <= 1:
        parser.error("minimum bandwidth ratio must be in (0, 1]")
    return args


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    output: pathlib.Path | None = None,
) -> str:
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if output is not None:
        output.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + "\n".join(completed.stdout.splitlines()[-80:])
        )
    return completed.stdout


def revision() -> tuple[str, bool]:
    value = run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(run(["git", "status", "--porcelain"]).strip())
    return value, dirty


def namespace_block_device(bdf: str, namespace: int) -> pathlib.Path:
    directory = pathlib.Path("/sys/bus/pci/devices") / bdf / "nvme"
    candidates = []
    for controller in directory.glob("nvme[0-9]*"):
        for entry in controller.glob("nvme*n*"):
            nsid = entry / "nsid"
            if nsid.is_file() and int(nsid.read_text(encoding="utf-8")) == namespace:
                candidates.append(pathlib.Path("/dev") / entry.name)
    run(["sudo", "udevadm", "settle"])
    candidates = [path for path in candidates if path.is_block_device()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one kernel block device for {bdf} namespace {namespace}; "
            f"found {candidates}"
        )
    return candidates[0]


def fio_baseline(args: argparse.Namespace, block: pathlib.Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="nta-fio-", suffix=".json") as raw:
        run(
            [
                "sudo",
                "fio",
                "--name=nta-matched-read",
                f"--filename={block}",
                "--readonly",
                "--direct=1",
                "--ioengine=io_uring",
                "--rw=read",
                f"--bs={args.bytes}",
                f"--iodepth={args.requests}",
                "--numjobs=1",
                f"--size={args.fio_size}",
                f"--runtime={args.fio_runtime}",
                "--time_based=1",
                "--group_reporting=1",
                "--output-format=json",
            ],
            output=pathlib.Path(raw.name),
        )
        document = json.loads(pathlib.Path(raw.name).read_text(encoding="utf-8"))
    read = document["jobs"][0]["read"]
    bandwidth = float(read["bw_bytes"])
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise RuntimeError("fio returned an invalid read bandwidth")
    return {
        "engine": "fio-io_uring",
        "block_device": str(block),
        "block_bytes": args.bytes,
        "queue_depth": args.requests,
        "runtime_seconds": args.fio_runtime,
        "bandwidth_bytes_per_second": bandwidth,
        "bandwidth_mib_per_second": bandwidth / (1024 * 1024),
        "iops": float(read["iops"]),
        "mean_latency_ns": float(read["lat_ns"]["mean"]),
    }


def gpu_read(args: argparse.Namespace, git_revision: str) -> dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f"{args.output.stem}-gpu.json")
    environment = {
        "NTA_NVME_BDF": args.bdf,
        "NTA_NVME_NSID": str(args.namespace),
        "NTA_NVME_QUEUE_DEPTH": str(args.queue_depth),
        "NTA_GPU": str(args.gpu),
        "NTA_NVME_MEDIA_POLICY": args.media_policy,
        "NTA_NVME_REFERENCE_BYTES": str(args.bytes * args.requests),
        "NTA_REVISION": git_revision,
    }
    run(
        [
            str(ROOT / "scripts" / "nta-vfio-device.sh"),
            "qualify",
            f"--bytes={args.bytes}",
            f"--requests={args.requests}",
            f"--progress-passes={args.progress_passes}",
            f"--iterations={args.iterations}",
            f"--cta-try-issue={int(args.cta_try_issue)}",
            f"--output={raw_output}",
        ],
        environment=environment,
    )
    run(
        [
            "sudo",
            "chown",
            f"{os.getuid()}:{os.getgid()}",
            str(raw_output),
        ]
    )
    return json.loads(raw_output.read_text(encoding="utf-8"))


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    git_revision, dirty = revision()
    block = namespace_block_device(args.bdf, args.namespace)
    baseline = fio_baseline(args, block)
    gpu = gpu_read(args, git_revision)
    ratio = float(gpu["physical_mib_per_second"]) / float(
        baseline["bandwidth_mib_per_second"]
    )
    ready = (
        gpu.get("revision") == git_revision
        and gpu.get("verified") is True
        and gpu.get("translated_iommu") is True
        and gpu.get("gpu_doorbell_mapping_validated") is True
        and int(gpu.get("verification_failures", 1)) == 0
        and int(gpu.get("failed", 1)) == 0
        and int(gpu.get("outstanding", 1)) == 0
        and ratio >= args.minimum_bandwidth_ratio
    )
    raw_output = args.output.with_name(f"{args.output.stem}-gpu.json")
    report = {
        "schema": 1,
        "classification": "nta-vfio-nvme-qualification",
        "revision": git_revision,
        "dirty": dirty,
        "ready": ready,
        "minimum_bandwidth_ratio": args.minimum_bandwidth_ratio,
        "matched_bandwidth_ratio": ratio,
        "baseline": baseline,
        "gpu_controlled": gpu,
        "raw_gpu_artifact": {
            "path": str(raw_output.resolve()),
            "sha256": digest(raw_output),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if args.require_ready and not ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
