#!/usr/bin/env python3
"""Ensure physical NVMe qualification remains explicitly fail-closed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "scripts" / "run-nvme-qualification.py"
    spec = importlib.util.spec_from_file_location("nta_nvme_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load NVMe qualification runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    vfio_script = (ROOT / "scripts" / "nta-vfio-device.sh").read_text(encoding="utf-8")
    assert '"$(basename "$block")"p*' in vfio_script
    assert "hidden multipath namespace" in vfio_script
    assert "require_media_policy" in vfio_script
    assert "nvme get-feature" in vfio_script
    assert "reference capture has an unexpected size" in vfio_script
    control_plane = (ROOT / "runtime" / "host" / "NvmeVfioControlPlane.cpp").read_text(encoding="utf-8")
    assert "mediaPolicy_(options.mediaPolicy)" in control_plane
    assert "Do not issue Set Features" in control_plane
    devices = sorted((Path("/sys/bus/pci/devices")).glob("*:*:*.*"))
    if devices:
        runner.validate_bdf(devices[0].name)
    for invalid in ("invalid", "0000:d8:00", "../../nvme"):
        try:
            runner.validate_bdf(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid BDF was accepted: {invalid}")

    environment = dict(os.environ)
    environment.pop("NTA_ALLOW_DEVICE_REBIND", None)
    direct = subprocess.run(
        [str(ROOT / "scripts" / "nta-vfio-device.sh"), "bind"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert direct.returncode != 0
    assert "device rebind is destructive" in direct.stdout

    runner_process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run-nvme-qualification.py"),
            "--bdf",
            "0000:00:00.0",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert runner_process.returncode != 0
    assert "--allow-device-rebind" in runner_process.stdout
    print("nvme_safety=pass")


if __name__ == "__main__":
    main()
