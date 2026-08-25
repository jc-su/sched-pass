#!/usr/bin/env python3
"""Check that physical-tier bring-up is read-only and fail-closed."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cxl_script = (ROOT / "scripts" / "nta-cxl-dax-module.sh").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "create-region",
        "destroy-region",
        "create_dax",
        "destroy_dax",
        "mkfs",
        "nvme format",
    ):
        assert forbidden not in cxl_script
    for required in (
        "cxl_core",
        "cxl_port",
        "cxl_pci",
        "cxl_mem",
        "cxl_acpi",
        "cxl_pmem",
        "dax_cxl",
        "device_dax",
        "cxl list -M -i",
        "daxctl list -u",
        "topology=",
        "root_decoders_only",
    ):
        assert required in cxl_script

    cxl_runtime = (ROOT / "runtime" / "host" / "CxlRuntime.cpp").read_text(
        encoding="utf-8"
    )
    assert "regular files are not qualification targets" in cxl_runtime
    assert "is not backed by an enumerated CXL region" in cxl_runtime

    qualification = (ROOT / "scripts" / "run-nvme-qualification.py").read_text(
        encoding="utf-8"
    )
    assert "--allow-device-rebind" in qualification
    nvme_script = (ROOT / "scripts" / "nta-vfio-device.sh").read_text(
        encoding="utf-8"
    )
    assert "require_media_policy" in nvme_script
    assert "restore" in nvme_script
    print("physical_tier_safety=pass")


if __name__ == "__main__":
    main()
