#!/usr/bin/env python3
"""Run the read-only physical-tier capability frontends on every host.

This test is intentionally not a qualification test.  It proves that the
kernel/module and discovery frontends are executable and records unavailable
hardware as a capability state.  The separate VFIO and devdax probes remain
the only tests allowed to qualify actual device DMA.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "cxl_core",
    "cxl_port",
    "cxl_pci",
    "cxl_mem",
    "cxl_acpi",
    "cxl_pmem",
    "dax_cxl",
    "device_dax",
)


def main() -> int:
    script = ROOT / "scripts" / "nta-cxl-dax-module.sh"
    result = subprocess.run(
        [str(script), "status"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"CXL read-only status command failed: {result.stderr.strip()}"
        )
    output = result.stdout
    for module in MODULES:
        if not re.search(rf"\b{re.escape(module)}=(?:loaded|unloaded)\b", output):
            raise AssertionError(f"CXL status omitted module state for {module}")

    topology = re.search(r"\btopology=([a-z_]+)\b", output)
    if topology is None:
        raise AssertionError("CXL status omitted the topology state")
    if topology.group(1) not in {
        "devdax_ready",
        "region_without_devdax",
        "type3_memdev_without_region",
        "root_decoders_only",
        "no_cxl_topology",
    }:
        raise AssertionError(f"unknown CXL topology state: {topology.group(1)}")

    daxctl_available = "unavailable (daxctl utility is not installed)" not in output
    cxl_available = "unavailable (cxl utility is not installed)" not in output
    live_endpoint = topology.group(1) == "devdax_ready"
    print(
        "physical_tier_capability=pass"
        f" cxl_cli={int(cxl_available)}"
        f" daxctl_cli={int(daxctl_available)}"
        f" devdax_inventory_candidate={int(live_endpoint)}"
        f" topology={topology.group(1)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
