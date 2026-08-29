#!/usr/bin/env python3
"""Run matched CPU and GPU-controlled read qualification on one VFIO NVMe."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hardware import platform_identity  # noqa: E402

RESULTS_ROOT = pathlib.Path(
    os.environ.get(
        "NTA_RESULTS_DIR", pathlib.Path(tempfile.gettempdir()) / "nta-results"
    )
)


def _privileged(command: list[str]) -> list[str]:
    """Run a control-plane command with a non-interactive privilege boundary."""
    if os.geteuid() == 0:
        return command
    return ["sudo", "-n", *command]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdf", default=os.environ.get("NTA_NVME_BDF", "0000:d8:00.0"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--namespace", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument(
        "--progress-rounds",
        type=int,
        default=1,
        help="exact dependency/consumer rounds per replay; NVMe polling is completion-driven",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="measured GPU graph replays; 100 reduces clock/queue noise",
    )
    parser.add_argument("--fio-runtime", type=int, default=10)
    parser.add_argument("--fio-size", default="2G")
    parser.add_argument(
        "--media-policy",
        choices=("hardware-write-protect", "trusted-read-only-code"),
        default="hardware-write-protect",
    )
    parser.add_argument(
        "--dma-target",
        choices=("hbm-peer", "host-mapped"),
        default="hbm-peer",
        help="NVMe data destination; host-mapped is an explicit baseline",
    )
    parser.add_argument(
        "--minimum-bandwidth-ratio",
        type=float,
        default=0.9,
        help="minimum direct-HBM/fio bandwidth ratio for qualification",
    )
    parser.add_argument(
        "--require-hbm-backend",
        choices=("auto", "cuda-dmabuf-ioas", "nvidia-peer-pages"),
        default="auto",
        help=(
            "set the native setup-time HBM mapping policy; auto may select "
            "either qualified direct-HBM backend"
        ),
    )
    parser.add_argument("--cta-try-issue", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--allow-device-rebind",
        action="store_true",
        help="confirm that the selected NVMe controller may be rebound to VFIO",
    )
    parser.add_argument(
        "--keep-vfio",
        action="store_true",
        help=(
            "leave a successful qualification on VFIO for subsequent tests; "
            "failures still restore the previous driver"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_ROOT / "qualification" / "nvme-qualification.json",
    )
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        default=RESULTS_ROOT / "qualification" / "nvme-reference.bin",
        help=(
            "build/artifact-scoped read-only namespace reference; it is "
            "captured before VFIO binding and reused by the exact-data checks"
        ),
    )
    args = parser.parse_args()
    if (
        min(
            args.queue_depth,
            args.bytes,
            args.requests,
            args.progress_rounds,
            args.iterations,
            args.fio_runtime,
        )
        <= 0
    ):
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


def runtime_abi_version() -> int:
    header = (ROOT / "include" / "nta" / "RuntimeABI.h").read_text(encoding="utf-8")
    match = re.search(r"\bVersion\s*=\s*([0-9]+)\s*;", header)
    if match is None:
        raise RuntimeError("cannot read the native runtime ABI version")
    return int(match.group(1))


def namespace_block_device(bdf: str, namespace: int) -> pathlib.Path:
    run(["udevadm", "settle"])
    directory = pathlib.Path("/sys/bus/pci/devices") / bdf / "nvme"
    candidates = []
    for controller in directory.glob("nvme[0-9]*"):
        for entry in controller.glob("nvme*n*"):
            nsid = entry / "nsid"
            if nsid.is_file() and int(nsid.read_text(encoding="utf-8")) == namespace:
                candidates.append(pathlib.Path("/dev") / entry.name)
    candidates = [path for path in candidates if path.is_block_device()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one kernel block device for {bdf} namespace {namespace}; "
            f"found {candidates}"
        )
    return candidates[0]


def validate_bdf(bdf: str) -> None:
    if (
        re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf)
        is None
    ):
        raise RuntimeError(f"invalid PCI BDF: {bdf!r}")
    device = pathlib.Path("/sys/bus/pci/devices") / bdf
    if not device.is_dir():
        raise RuntimeError(f"PCI device does not exist: {bdf}")


def read_only_preflight(args: argparse.Namespace) -> None:
    validate_bdf(args.bdf)
    run(
        [
            str(ROOT / "scripts" / "nta-vfio-device.sh"),
            "preflight",
        ],
        environment={
            "NTA_NVME_BDF": args.bdf,
            "NTA_NVME_NSID": str(args.namespace),
            "NTA_NVME_QUEUE_DEPTH": str(args.queue_depth),
            "NTA_GPU": str(args.gpu),
            "NTA_NVME_MEDIA_POLICY": args.media_policy,
            "NTA_NVME_DMA_TARGET": args.dma_target,
            "NTA_NVME_REFERENCE": str(args.reference),
            "NTA_NVME_REFERENCE_BYTES": str(args.bytes * args.requests),
        },
    )


def fio_baseline(args: argparse.Namespace, block: pathlib.Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="nta-fio-", suffix=".json") as raw:
        run(
            _privileged(
                [
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
                ]
            ),
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


def iommu_fault_count(bdf: str) -> int:
    requester = bdf.split(":", 1)[1]
    output = run(_privileged(["dmesg", "--color=never"]))
    marker = f"Request device [{requester}] fault"
    return sum(marker in line for line in output.splitlines())


def hbm_mapping_contract_ready(
    *, dma_target: str, required_backend: str, gpu: dict[str, Any]
) -> bool:
    """Verify that native policy enforcement and the selected mapper agree.

    The policy field is emitted by the native benchmark from the options used
    to construct the transport.  Requiring it here prevents a post-hoc backend
    label from being mistaken for proof that an explicit fail-closed policy was
    active before the first HBM mapping attempt.
    """

    if gpu.get("hbm_mapping_policy") != required_backend:
        return False
    if dma_target != "hbm-peer":
        return required_backend == "auto"
    selected = gpu.get("hbm_mapping_backend")
    return (
        gpu.get("hbm_peer_dma_supported") is True
        and selected in {"cuda-dmabuf-ioas", "nvidia-peer-pages"}
        and (required_backend == "auto" or selected == required_backend)
    )


def gpu_read(args: argparse.Namespace, git_revision: str) -> dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f"{args.output.stem}-gpu.json")
    environment = {
        "NTA_NVME_BDF": args.bdf,
        "NTA_NVME_NSID": str(args.namespace),
        "NTA_NVME_QUEUE_DEPTH": str(args.queue_depth),
        "NTA_GPU": str(args.gpu),
        "NTA_NVME_MEDIA_POLICY": args.media_policy,
        "NTA_NVME_DMA_TARGET": args.dma_target,
        "NTA_NVME_HBM_BACKEND": args.require_hbm_backend,
        "NTA_NVME_REFERENCE": str(args.reference),
        "NTA_NVME_REFERENCE_BYTES": str(args.bytes * args.requests),
        "NTA_ALLOW_DEVICE_REBIND": "1",
        "NTA_NVME_KEEP_VFIO": "1" if args.keep_vfio else "0",
        "NTA_REVISION": git_revision,
    }
    run(
        [
            str(ROOT / "scripts" / "nta-vfio-device.sh"),
            "qualify",
            f"--bytes={args.bytes}",
            f"--requests={args.requests}",
            f"--progress-rounds={args.progress_rounds}",
            f"--iterations={args.iterations}",
            f"--cta-try-issue={int(args.cta_try_issue)}",
            f"--dma-target={args.dma_target}",
            f"--output={raw_output}",
        ],
        environment=environment,
    )
    run(_privileged(["chown", f"{os.getuid()}:{os.getgid()}", str(raw_output)]))
    return json.loads(raw_output.read_text(encoding="utf-8"))


def write_report(
    args: argparse.Namespace,
    *,
    git_revision: str,
    dirty: bool,
    fields: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema": 1,
        "classification": "nta-vfio-nvme-qualification",
        "tier": "nvme",
        "revision": git_revision,
        "dirty": dirty,
        "runtime_abi": runtime_abi_version(),
        "platform_identity": platform_identity(),
        "media_policy": args.media_policy,
        "read_only_contract": args.media_policy == "trusted-read-only-code",
        **fields,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    if not args.allow_device_rebind:
        raise SystemExit(
            "refusing NVMe qualification without --allow-device-rebind; "
            "review the read-only preflight and explicitly authorize VFIO rebinding"
        )
    if args.dma_target != "hbm-peer" and args.require_hbm_backend != "auto":
        raise SystemExit(
            "an explicit --require-hbm-backend requires --dma-target hbm-peer"
        )
    git_revision, dirty = revision()
    expected_abi = runtime_abi_version()
    args.reference = args.reference.expanduser().resolve()
    if args.reference.exists() and not args.reference.is_file():
        raise SystemExit(f"NVMe reference is not a regular file: {args.reference}")
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    phase = "preflight"
    try:
        read_only_preflight(args)
        phase = "kernel-baseline"
        block = namespace_block_device(args.bdf, args.namespace)
        baseline = fio_baseline(args, block)
        phase = "gpu-qualification"
        faults_before = iommu_fault_count(args.bdf)
        gpu = gpu_read(args, git_revision)
        faults_after = iommu_fault_count(args.bdf)
        iommu_fault_free = faults_after == faults_before
        ratio = float(gpu["end_to_end_mib_per_second"]) / float(
            baseline["bandwidth_mib_per_second"]
        )
        selected_hbm_backend = gpu.get("hbm_mapping_backend")
        reported_hbm_policy = gpu.get("hbm_mapping_policy")
        hbm_backend_ready = hbm_mapping_contract_ready(
            dma_target=args.dma_target,
            required_backend=args.require_hbm_backend,
            gpu=gpu,
        )
        transport_ready = (
            gpu.get("revision") == git_revision
            and int(gpu.get("runtime_abi", -1)) == expected_abi
            and gpu.get("verified") is True
            and gpu.get("selected_data_path_verified") is True
            and gpu.get("destination") == args.dma_target
            and hbm_backend_ready
            and gpu.get("translated_iommu") is True
            and gpu.get("gpu_doorbell_mapping_validated") is True
            and iommu_fault_free
            and int(gpu.get("verification_failures", 1)) == 0
            and int(gpu.get("failed", 1)) == 0
            and int(gpu.get("outstanding", 1)) == 0
        )
        provenance_ready = not dirty
        performance_qualified = ratio >= args.minimum_bandwidth_ratio
        qualified = transport_ready and provenance_ready and performance_qualified
        raw_output = args.output.with_name(f"{args.output.stem}-gpu.json")
        write_report(
            args,
            git_revision=git_revision,
            dirty=dirty,
            fields={
                "ready": qualified,
                "transport_ready": transport_ready,
                "provenance_ready": provenance_ready,
                "qualified": qualified,
                "status": "qualified" if qualified else "not_qualified",
                "demand_semantics": "exact",
                "minimum_bandwidth_ratio": args.minimum_bandwidth_ratio,
                "matched_bandwidth_ratio": ratio,
                "performance_qualified": performance_qualified,
                "required_hbm_backend": args.require_hbm_backend,
                "reported_hbm_mapping_policy": reported_hbm_policy,
                "selected_hbm_backend": selected_hbm_backend,
                "iommu_fault_free": iommu_fault_free,
                "iommu_fault_count_before": faults_before,
                "iommu_fault_count_after": faults_after,
                "baseline": baseline,
                "gpu_controlled": gpu,
                "raw_gpu_result": str(raw_output.resolve()),
            },
        )
        if args.require_ready and not qualified:
            return 2
        return 0
    except (RuntimeError, OSError, ValueError) as error:
        write_report(
            args,
            git_revision=git_revision,
            dirty=dirty,
            fields={
                "ready": False,
                "qualified": False,
                "status": "blocked" if phase == "preflight" else "failed",
                "failure_phase": phase,
                "failure": str(error),
                "demand_semantics": "exact",
            },
        )
        if args.require_ready:
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
