"""Read-only hardware capability inventory for tiered evaluation.

This module deliberately does not bind PCI devices, open block namespaces, or
run a qualification workload.  It records enough host state to explain why a
physical-tier trial is qualified, unavailable, or still needs an explicit
qualification step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


SCHEMA = 1
CLASSIFICATION = "nta-hardware-inventory"


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _command(argv: list[str]) -> str | None:
    if not shutil.which(argv[0]):
        return None
    try:
        result = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _driver(device: Path) -> str | None:
    try:
        return device.joinpath("driver").resolve().name
    except (OSError, RuntimeError):
        return None


def _iommu_group(device: Path) -> str | None:
    try:
        return device.joinpath("iommu_group").resolve().name
    except (OSError, RuntimeError):
        return None


def _nvme_namespaces(device: Path, dev_root: Path) -> list[str]:
    names: list[str] = []
    for controller in sorted(device.joinpath("nvme").glob("nvme[0-9]*")):
        for namespace in sorted(controller.glob("nvme*n*")):
            name = namespace.name
            if (dev_root / name).exists():
                names.append(name)
    return names


def _nvme_controllers(sysfs_root: Path, dev_root: Path) -> list[dict[str, Any]]:
    controllers: list[dict[str, Any]] = []
    for device in sorted(sysfs_root.glob("*:*:*.*")):
        if not device.is_dir():
            continue
        class_code = _text(device / "class")
        if class_code is None or not class_code.lower().startswith("0x0108"):
            continue
        driver = _driver(device)
        controllers.append(
            {
                "bdf": device.name,
                "class": class_code,
                "driver": driver,
                "vfio_owned": driver == "vfio-pci",
                "iommu_group": _iommu_group(device),
                "namespaces": _nvme_namespaces(device, dev_root),
            }
        )
    return controllers


def _dax_devices(dev_root: Path) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for path in sorted(dev_root.glob("dax*")):
        try:
            mode = path.stat().st_mode
            character_device = path.is_char_device()
        except OSError:
            continue
        devices.append(
            {
                "path": str(path),
                "character_device": character_device,
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "mode": oct(mode & 0o777),
            }
        )
    return devices


def collect(
    *,
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
    dev_root: Path = Path("/dev"),
) -> dict[str, Any]:
    nvme = _nvme_controllers(sysfs_root, dev_root)
    dax = _dax_devices(dev_root)
    vfio_nvme = [entry for entry in nvme if entry["vfio_owned"]]
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
        },
        "gpu": {
            "nvidia_smi": _command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                    "--format=csv,noheader",
                ]
            )
        },
        "nvme": {
            "controllers": nvme,
            "vfio_controller_count": len(vfio_nvme),
            "status": (
                "qualification_candidate_present" if vfio_nvme else "no_vfio_controller"
            ),
        },
        "dax": {
            "devices": dax,
            "status": (
                "endpoint_present"
                if any(entry["character_device"] for entry in dax)
                else "no_devdax_endpoint"
            ),
        },
        "safety": {
            "read_only_inventory": True,
            "pci_binding_performed": False,
            "block_namespace_opened": False,
            "qualification_performed": False,
        },
    }


def validate(document: dict[str, Any]) -> dict[str, Any]:
    if (
        document.get("schema") != SCHEMA
        or document.get("classification") != CLASSIFICATION
    ):
        raise ValueError("unsupported hardware inventory")
    for section in ("machine", "gpu", "nvme", "dax", "safety"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"hardware inventory lacks {section}")
    safety = document["safety"]
    if safety.get("read_only_inventory") is not True:
        raise ValueError("hardware inventory is not marked read-only")
    if any(
        safety.get(field) is not False
        for field in (
            "pci_binding_performed",
            "block_namespace_opened",
            "qualification_performed",
        )
    ):
        raise ValueError("hardware inventory is not read-only")
    controllers = document["nvme"].get("controllers")
    if not isinstance(controllers, list):
        raise ValueError("hardware inventory has no NVMe controller list")
    for controller in controllers:
        if not isinstance(controller, dict) or not isinstance(
            controller.get("bdf"), str
        ):
            raise ValueError("invalid NVMe controller entry")
    dax = document["dax"].get("devices")
    if not isinstance(dax, list):
        raise ValueError("hardware inventory has no DAX device list")
    return document


def write_inventory(output: Path) -> dict[str, Any]:
    document = collect()
    validate(document)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document
