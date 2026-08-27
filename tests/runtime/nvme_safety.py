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
    vfio_path = ROOT / "scripts" / "nta-vfio-device.sh"
    vfio_script = vfio_path.read_text(encoding="utf-8")
    syntax = subprocess.run(
        ["bash", "-n", str(vfio_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stdout
    assert '"$(basename "$block")"p*' in vfio_script
    assert "hidden multipath namespace" in vfio_script
    assert "require_media_policy" in vfio_script
    assert "NTA_NVME_BDF must use DDDD:BB:SS.F syntax" in vfio_script
    assert "nvme get-feature" in vfio_script
    assert "reference capture has an unexpected size" in vfio_script
    assert (
        "no namespace block device exists and reference file is absent" in vfio_script
    )
    assert '.nta-nvme-reference.XXXXXX' in vfio_script
    assert 'chmod 0444 "$temporary"' in vfio_script
    assert 'mv -f "$temporary" "$reference"' in vfio_script
    assert "wait_for_nvme_namespace" in vfio_script
    assert "returned to the nvme driver but no live namespace appeared" in vfio_script
    assert "driver_override is only a transactional bind aid" in vfio_script
    runtime = (ROOT / "runtime" / "host" / "NvmeRuntime.cpp").read_text(
        encoding="utf-8"
    )
    assert "controlPlane->mappingBackend().mapHbm" in runtime
    assert "retainPagePrefix" in runtime
    assert "registerExternalHbm" in runtime
    assert "NvmeHbmRegion::view" in runtime
    assert "ownsDestinationMemory" in runtime
    assert "mapping.cacheable = cacheable" in runtime
    assert "impl_ == nullptr ? nullptr : impl_->deviceAddress" in runtime
    assert "cannot allocate from a moved-from NVMe transport" in runtime
    assert "cannot register HBM through a moved-from NVMe transport" in runtime
    description_begin = runtime.index("NvmeTransport::describeExternalHbm")
    registration_begin = runtime.index(
        "NvmeTransport::registerExternalHbm", description_begin
    )
    view_begin = runtime.index("NvmeHbmRegion::view", registration_begin)
    description = runtime[description_begin:registration_begin]
    registration = runtime[registration_begin:view_begin]
    view = runtime[view_begin:]
    assert "cudaMemoryTypeDevice" in description
    assert "cuMemGetAddressRange" in description
    assert "mappingBackend().mapHbm" not in description
    assert "cudaMalloc" not in description
    assert "describeExternalHbm(deviceAddress, bytes)" in registration
    assert "mappingBackend().mapHbm" in registration
    assert "cudaMalloc registered NVMe HBM page table" in registration
    assert "mappingBackend().mapHbm" not in view
    assert "cudaMalloc" not in view
    assert "externalRegion = impl_" in view
    assert "mappingBackend().mapHost" in runtime
    assert "NvmeHbmMappingBackend::NvidiaPeerPages" in runtime
    assert "cuMemAlloc(&allocation.base" in runtime
    assert "cuMemGetHandleForAddressRange" not in runtime
    assert "CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER" in runtime
    release_mapping = runtime[
        runtime.index("void releaseMapping(RetiredMapping mapping)") : runtime.index(
            "int deviceOrdinal = 0"
        )
    ]
    assert "cudaDeviceSynchronize" not in release_mapping
    assert "retiredMappings" in release_mapping
    host_runtime = (ROOT / "runtime" / "host" / "Runtime.cpp").read_text(
        encoding="utf-8"
    )
    assert "installNvmeObjectAsync" in host_runtime
    assert "reapRetiredObjects" in host_runtime
    assert "cudaStreamWaitEvent(stream, priorConsumerEvent, 0)" in host_runtime
    retire_begin = host_runtime.index("void retireObject(")
    retire_end = host_runtime.index("void reserveStaging(", retire_begin)
    retirement = host_runtime[retire_begin:retire_end]
    assert "const cudaError_t eventStatus" in retirement
    assert "cudaStreamSynchronize(stream)" in retirement
    assert "cudaDeviceSynchronize()" in retirement
    assert "releaseObject(retired.object)" in retirement
    assert "ioctl" not in (ROOT / "runtime" / "device" / "Acquire.cuh").read_text(
        encoding="utf-8"
    )
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
    assert qualification.count('f"--requests={args.requests}"') == 1
    assert '"NTA_NVME_DMA_TARGET": args.dma_target' in qualification
    assert qualification.count('"NTA_NVME_REFERENCE": str(args.reference)') == 2
    assert 'RESULTS_ROOT / "qualification" / "nvme-reference.bin"' in qualification
    assert 'gpu.get("selected_data_path_verified") is True' in qualification
    assert 'gpu.get("destination") == args.dma_target' in qualification
    assert "and iommu_fault_free" in qualification
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    preflight = cmake[cmake.index("add_custom_target(\n  nta-vfio-preflight") :]
    assert "${CMAKE_COMMAND} -E env" in preflight
    assert "${NTA_TEST_NVME_ENVIRONMENT}" in preflight
    control_plane = (ROOT / "runtime" / "host" / "NvmeVfioControlPlane.cpp").read_text(
        encoding="utf-8"
    )
    assert "class VfioNvmeMappingBackend final" in control_plane
    assert "class VfioNvmeControlPlane final : public NvmeControlPlane" in control_plane
    assert (
        "class VfioNvmeControlPlane final : public NvmeMappingBackend"
        not in control_plane
    )
    assert "mappingBackend_->shutdown()" in control_plane
    assert "std::shared_ptr<NvmeTransport::Impl> owner;" in runtime
    assert "detail::NvmeMapping dmaMapping;" in runtime
    buffer_impl = runtime[runtime.index("struct NvmeBuffer::Impl") :]
    assert buffer_impl.index("detail::NvmeMapping dmaMapping;") > buffer_impl.index(
        "std::shared_ptr<NvmeTransport::Impl> owner;"
    )
    assert runtime.index("controlPlane.reset()") > runtime.index(
        "releaseMappingResources(mapping)"
    )
    assert "mediaPolicy_(options.mediaPolicy)" in control_plane
    assert "Do not issue Set Features" in control_plane
    assert "NTA_NVME_P2P_IOCTL_MAP" in control_plane
    attention = (ROOT / "benchmarks" / "attention" / "PagedAttention.cpp").read_text(
        encoding="utf-8"
    )
    assert "An NVMe object owns an HBM destination" in attention
    assert "options.mode == Mode::Nvme" in attention
    assert "nvmeStats.completed == nvmeStats.submitted" in attention
    assert '\\",\\"queue_depth\\":' in attention
    assert "progressNvmeUntilIdle" in runtime or "progressNvmeUntilIdle" in (
        ROOT / "include" / "nta" / "FinitePhase.h"
    ).read_text(encoding="utf-8")
    bridge = (ROOT / "kernel" / "nta_nvme_p2p" / "nta_nvme_p2p.c").read_text(
        encoding="utf-8"
    )
    for required in (
        "nvidia_p2p_get_pages_persistent",
        "nvidia_p2p_dma_map_pages",
        "__IOMMU_DOMAIN_PAGING",
        "iommu_map(domain, iova, iova",
        "iommu_unmap(domain, iova, page_size)",
        'strcmp(driver->name, "vfio-pci")',
        "PCI_CLASS_STORAGE_EXPRESS",
    ):
        assert required in bridge
    # iommu_domain::cookie_type and IOMMU_COOKIE_IOMMUFD are private/removed
    # kernel internals.  The module must remain buildable on the supported
    # kernel while retaining the VFIO driver, paging-domain, and identity-PTE
    # checks above as its public safety boundary.
    assert "domain->cookie_type" not in bridge
    assert "IOMMU_COOKIE_IOMMUFD" not in bridge
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
    assert "nta_runtime_install_nvme_object_async" in runtime_c
    assert "nta_nvme_transport_register_hbm_region" in runtime_c
    assert "nta_nvme_transport_describe_hbm_region" in runtime_c
    assert "nta_runtime_install_registered_nvme_object" in runtime_c
    assert "nta_runtime_install_registered_nvme_object_async" in runtime_c
    runtime_c_impl = (ROOT / "runtime" / "host" / "RuntimeC.cpp").read_text(
        encoding="utf-8"
    )
    install_begin = runtime_c_impl.index("nta_runtime_install_nvme_object")
    install_end = runtime_c_impl.index("nta_runtime_read_pending_count", install_begin)
    install_body = runtime_c_impl[install_begin:install_end]
    assert "runtime->nvme->allocate" not in install_body
    assert "region->value->view" in install_body
    assert "mapExternalHbm" not in install_body
    planner = (
        ROOT / "python" / "nta_runtime" / "hbm_registration.py"
    ).read_text(encoding="utf-8")
    assert "begin < mutable[-1]" in planner
    assert "HBM registration plan contains duplicate destination keys" in planner
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
