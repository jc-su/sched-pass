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
    assert "NTA_NVME_BDF must use DDDD:BB:SS.F syntax" in vfio_script
    assert "nvme get-feature" in vfio_script
    assert "reference capture has an unexpected size" in vfio_script
    assert "wait_for_nvme_namespace" in vfio_script
    assert "returned to the nvme driver but no live namespace appeared" in vfio_script
    assert "driver_override is only a transactional bind aid" in vfio_script
    runtime = (ROOT / "runtime" / "host" / "NvmeRuntime.cpp").read_text(
        encoding="utf-8"
    )
    assert "controlPlane->mappingBackend().mapHbm" in runtime
    assert "retainPagePrefix" in runtime
    assert "mappingBackend().mapHost" in runtime
    assert "NvmeHbmMappingBackend::NvidiaPeerPages" in runtime
    assert "cuMemAlloc(&allocation.base" in runtime
    assert "cuMemGetHandleForAddressRange" not in runtime
    assert "CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER" in runtime
    assert "ioctl" not in (
        ROOT / "runtime" / "device" / "Acquire.cuh"
    ).read_text(encoding="utf-8")
    assert "NvmeDmaMapping" not in (
        ROOT / "runtime" / "host" / "NvmeControlPlane.h"
    ).read_text(encoding="utf-8")
    python_runtime = (ROOT / "python" / "nta_runtime" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert '("dma_target", ctypes.c_uint32)' in python_runtime
    assert "class NvmeHbmMappingBackend(enum.IntEnum)" in python_runtime
    qualification = (ROOT / "scripts" / "run-nvme-qualification.py").read_text(
        encoding="utf-8"
    )
    assert '"NTA_NVME_DMA_TARGET": args.dma_target' in qualification
    assert 'gpu.get("selected_data_path_verified") is True' in qualification
    assert 'gpu.get("destination") == args.dma_target' in qualification
    assert "and iommu_fault_free" in qualification
    control_plane = (ROOT / "runtime" / "host" / "NvmeVfioControlPlane.cpp").read_text(
        encoding="utf-8"
    )
    assert "class VfioNvmeMappingBackend final" in control_plane
    assert "class VfioNvmeControlPlane final : public NvmeControlPlane" in control_plane
    assert "class VfioNvmeControlPlane final : public NvmeMappingBackend" not in control_plane
    assert "mappingBackend_->shutdown()" in control_plane
    assert "mediaPolicy_(options.mediaPolicy)" in control_plane
    assert "Do not issue Set Features" in control_plane
    assert "NTA_NVME_P2P_IOCTL_MAP" in control_plane
    assert "progressNvmeUntilIdle" in runtime or "progressNvmeUntilIdle" in (
        ROOT / "include" / "nta" / "FinitePhase.h"
    ).read_text(encoding="utf-8")
    bridge = (ROOT / "kernel" / "nta_nvme_p2p" / "nta_nvme_p2p.c").read_text(
        encoding="utf-8"
    )
    for required in (
        "nvidia_p2p_get_pages_persistent",
        "nvidia_p2p_dma_map_pages",
        "IOMMU_COOKIE_IOMMUFD",
        "__IOMMU_DOMAIN_PAGING",
        "iommu_map(domain, iova, iova",
        "iommu_unmap(domain, iova, page_size)",
        'strcmp(driver->name, "vfio-pci")',
        "PCI_CLASS_STORAGE_EXPRESS",
    ):
        assert required in bridge
    release = bridge[
        bridge.index("static void nta_release_mapping") : bridge.index(
            "static bool nta_peer_is_vfio_nvme"
        )
    ]
    assert release.index("nta_unmap_peer_iovas(mapping)") < release.index(
        "nvidia_p2p_dma_unmap_pages"
    )
    runtime_c = (ROOT / "include" / "nta" / "RuntimeC.h").read_text(encoding="utf-8")
    assert "nta_jit_phase_progress_nvme_until_idle" in runtime_c
    runtime_c_impl = (ROOT / "runtime" / "host" / "RuntimeC.cpp").read_text(
        encoding="utf-8"
    )
    install_begin = runtime_c_impl.index("nta_runtime_install_nvme_object")
    install_end = runtime_c_impl.index(
        "nta_runtime_read_pending_count", install_begin
    )
    install_body = runtime_c_impl[install_begin:install_end]
    assert "runtime->nvme->allocate" not in install_body
    assert "sourceByteOffset, nativeBytes);" in install_body
    epoch = (ROOT / "python" / "nta_runtime" / "epoch.py").read_text(encoding="utf-8")
    assert "self.phases.progress_nvme_until_idle(" in epoch
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
