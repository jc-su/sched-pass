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
DEFAULT_CXL_SYSFS_ROOT = Path("/sys/bus/cxl/devices")


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


def _cxl_attr(path: Path, name: str) -> str | None:
    return _text(path / name)


def _cxl_decoders(cxl_sysfs_root: Path) -> list[dict[str, Any]]:
    decoders: list[dict[str, Any]] = []
    for path in sorted(cxl_sysfs_root.glob("decoder*")):
        if not path.is_dir():
            continue
        decoders.append(
            {
                "device": path.name,
                "size": _cxl_attr(path, "size"),
                "target_list": _cxl_attr(path, "target_list"),
                "interleave_ways": _cxl_attr(path, "interleave_ways"),
                "interleave_granularity": _cxl_attr(path, "interleave_granularity"),
                "locked": _cxl_attr(path, "locked"),
                "cap_ram": _cxl_attr(path, "cap_ram"),
                "cap_pmem": _cxl_attr(path, "cap_pmem"),
                "create_ram_region": (path / "create_ram_region").is_file(),
                "create_pmem_region": (path / "create_pmem_region").is_file(),
            }
        )
    return decoders


def _cxl_inventory(dev_root: Path, cxl_sysfs_root: Path) -> dict[str, Any]:
    names = (
        sorted(path.name for path in cxl_sysfs_root.iterdir() if path.is_dir())
        if cxl_sysfs_root.is_dir()
        else []
    )
    memdevs = [name for name in names if name.startswith("mem")]
    endpoints = [name for name in names if name.startswith("endpoint")]
    regions = [name for name in names if name.startswith("region")]
    ports = [name for name in names if name.startswith("port")]
    root_ports = [name for name in names if name.startswith("root")]
    decoders = _cxl_decoders(cxl_sysfs_root)
    dax_devices = _dax_devices(dev_root)
    dax_present = any(entry["character_device"] for entry in dax_devices)
    if dax_present:
        status = "devdax_endpoint_present"
        next_step = "run the explicit DAX qualification probe"
    elif regions:
        status = "region_present_without_devdax"
        next_step = "reconfigure an idle region namespace to devdax"
    elif memdevs or endpoints:
        status = "memdev_present_without_region"
        next_step = "provision a region only after platform ownership review"
    elif decoders:
        status = "root_decoder_only"
        next_step = "platform has no enumerated CXL memory endpoint to provision"
    else:
        status = "no_cxl_topology"
        next_step = "enable and enumerate CXL memory in platform firmware"
    inventory: dict[str, Any] = {
        "status": status,
        "next_step": next_step,
        "sysfs_root": str(cxl_sysfs_root),
        "root_ports": root_ports,
        "ports": ports,
        "endpoints": endpoints,
        "memdevs": memdevs,
        "regions": regions,
        "decoders": decoders,
        "cxl_list": {},
        "ndctl_list": None,
        "daxctl_list": None,
    }
    if cxl_sysfs_root == DEFAULT_CXL_SYSFS_ROOT:
        inventory["cxl_list"] = {
            "buses": _command(["cxl", "list", "-BMu"]),
            "memdevs": _command(["cxl", "list", "-iM"]),
            "decoders": _command(["cxl", "list", "-D"]),
        }
        inventory["ndctl_list"] = _command(["ndctl", "list", "-Ru"])
        inventory["daxctl_list"] = _command(["daxctl", "list", "-u"])
    return inventory


def collect(
    *,
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
    dev_root: Path = Path("/dev"),
    cxl_sysfs_root: Path = DEFAULT_CXL_SYSFS_ROOT,
) -> dict[str, Any]:
    nvme = _nvme_controllers(sysfs_root, dev_root)
    dax = _dax_devices(dev_root)
    cxl = _cxl_inventory(dev_root, cxl_sysfs_root)
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
        "cxl": cxl,
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
    for section in ("machine", "gpu", "nvme", "dax", "cxl", "safety"):
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
    cxl = document["cxl"]
    for field in ("memdevs", "regions", "decoders"):
        if not isinstance(cxl.get(field), list):
            raise ValueError(f"hardware inventory has no CXL {field} list")
    if not isinstance(cxl.get("status"), str):
        raise ValueError("hardware inventory has no CXL status")
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
