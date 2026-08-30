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
    assert ".nta-nvme-reference.XXXXXX" in vfio_script
    assert 'chmod 0444 "$temporary"' in vfio_script
    assert 'mv -f "$temporary" "$reference"' in vfio_script
    assert "wait_for_nvme_namespace" in vfio_script
    assert "wait_for_namespace_partitions" in vfio_script
    assert 'partx --show --noheadings --output NR "$block_device"' in vfio_script
    safe_device_begin = vfio_script.index("require_safe_device()")
    containment_begin = vfio_script.index("require_containment()", safe_device_begin)
    safe_device = vfio_script[safe_device_begin:containment_begin]
    assert safe_device.index("wait_for_namespace_partitions") < safe_device.index(
        'check_block_device "$block"'
    )
    assert "returned to the nvme driver but no live namespace appeared" in vfio_script
    assert "not a valid byte oracle for the raw namespace" in vfio_script
    assert "wait_for_driver nvme" in vfio_script
    assert "/sys/module/vmem_sw/parameters/target_bdf" in vfio_script
    assert "active references to the selected controller" in vfio_script
    assert vfio_script.index("wait_for_driver nvme") < vfio_script.index(
        "capture_reference", vfio_script.index("bind_vfio()")
    )
    bind_begin = vfio_script.index("bind_vfio()")
    case_begin = vfio_script.index("case ${1:-status}", bind_begin)
    bind_body = vfio_script[bind_begin:case_begin]
    capture_index = bind_body.index("capture_reference")
    assert bind_body.index("require_safe_device", capture_index) > capture_index
    assert "run)" in vfio_script
    run_begin = vfio_script.index("run)")
    restore_begin = vfio_script.index("restore)", run_begin)
    run_body = vfio_script[run_begin:restore_begin]
    assert "require_rebind_confirmation" in run_body
    assert "validate_session" in run_body
    assert '[[ -r $state ]]' in run_body
    assert "bind_vfio" not in run_body
    assert 'exec "$@"' in run_body
    assert "smart_write_counters" in vfio_script
    assert '"data_units_written"' in vfio_script
    assert '"host_write_commands"' in vfio_script
    assert "zero-write gate failed" in vfio_script
    assert "session-start)" in vfio_script
    assert "session-stop)" in vfio_script
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
    assert "mappingKeyBytes" in runtime
    assert "context.mappingKey >= mappingKey" in runtime
    assert "externalRegion == nullptr" in runtime
    assert "mappingBackend().mapHost" in runtime
    assert "mappingBackend().hbmMappingBackend()" in runtime
    preflight_begin = runtime.index("HbmAllocation preflight;")
    capabilities_begin = runtime.index(
        "capabilities.supportsHbmPeerDma = true;", preflight_begin
    )
    constructor_preflight = runtime[preflight_begin:capabilities_begin]
    assert "mappingBackend().mapHbm" in constructor_preflight
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
    device_acquire = (ROOT / "runtime" / "device" / "Acquire.cuh").read_text(
        encoding="utf-8"
    )
    assert "controllerPageSize) /" in device_acquire
    assert "sizeof(std::uint64_t) +" in device_acquire
    host_runtime = (ROOT / "runtime" / "host" / "Runtime.cpp").read_text(
        encoding="utf-8"
    )
    assert "installNvmeObjectAsync" in host_runtime
    assert "installNvmeObjectsAsync" in host_runtime
    assert "publish NVMe object batch asynchronously" in host_runtime
    assert "nvmeTransferPageCount(*buffer, bytes, capabilities)" in host_runtime
    indexed_async = host_runtime[
        host_runtime.index(
            "registerIndexedHostObjectsAsyncQuiesced"
        ) : host_runtime.index(
            "ObjectHandle HostRuntime::installNvmeObject(",
            host_runtime.index("registerIndexedHostObjectsAsyncQuiesced"),
        )
    ]
    assert "cudaStreamSynchronize(stream)" in indexed_async
    nvme_async = host_runtime[
        host_runtime.index(
            "ObjectHandle HostRuntime::installNvmeObjectAsync("
        ) : host_runtime.index("void HostRuntime::bindTensorMaps")
    ]
    assert "cudaStreamSynchronize(stream)" in nvme_async
    assert "reapRetiredObjects" in host_runtime
    assert "cudaStreamWaitEvent(stream, priorConsumerEvent, 0)" in host_runtime
    retire_begin = host_runtime.index("void retireObject(")
    retire_end = host_runtime.index("void reserveStaging(", retire_begin)
    retirement = host_runtime[retire_begin:retire_end]
    assert "const cudaError_t eventStatus" in retirement
    assert "cudaStreamSynchronize(stream)" in retirement
    assert "cudaDeviceSynchronize()" in retirement
    assert "releaseObject(owned)" in retirement
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
    assert '"gpu_queue_depth_calibration": gpu_calibration' in qualification
    assert '"target_identity": target_identity' in qualification
    assert '"recommended_serving_config"' in qualification
    assert '"NTA_NVME_HBM_BACKEND": args.require_hbm_backend' in qualification
    assert 'gpu.get("hbm_mapping_policy") != required_backend' in qualification
    assert "provenance_ready = not dirty" in qualification
    assert (
        "qualified = transport_ready and provenance_ready and performance_qualified"
        in qualification
    )
    assert "and iommu_fault_free" in qualification
    assert "retain_vfio = args.keep_vfio and qualified" in qualification
    assert 'if not retain_vfio and pci_driver(args.bdf) == "vfio-pci"' in qualification
    winner, winner_path, summaries = runner.select_calibrated_result(
        [
            (8, {"end_to_end_mib_per_second": 7000.0}, Path("q8-t0")),
            (5, {"end_to_end_mib_per_second": 7300.0}, Path("q5-t0")),
            (5, {"end_to_end_mib_per_second": 7350.0}, Path("q5-t1")),
            (5, {"end_to_end_mib_per_second": 7325.0}, Path("q5-t2")),
            (8, {"end_to_end_mib_per_second": 7100.0}, Path("q8-t1")),
            (8, {"end_to_end_mib_per_second": 7050.0}, Path("q8-t2")),
        ]
    )
    assert winner["end_to_end_mib_per_second"] == 7325.0
    assert winner_path == Path("q5-t2")
    assert [entry["queue_depth"] for entry in summaries] == [5, 8]
    assert runner._depth_candidates("5,8,16") == (5, 8, 16)
    for invalid in ("", "0", "5,5", "five"):
        try:
            runner._depth_candidates(invalid)
        except Exception:
            pass
        else:
            raise AssertionError(f"invalid queue-depth sweep accepted: {invalid!r}")
    direct_gpu = {
        "hbm_mapping_policy": "cuda-dmabuf-ioas",
        "hbm_mapping_backend": "cuda-dmabuf-ioas",
        "hbm_peer_dma_supported": True,
    }
    assert runner.hbm_mapping_contract_ready(
        dma_target="hbm-peer",
        required_backend="cuda-dmabuf-ioas",
        gpu=direct_gpu,
    )
    assert not runner.hbm_mapping_contract_ready(
        dma_target="hbm-peer",
        required_backend="nvidia-peer-pages",
        gpu=direct_gpu,
    )
    assert not runner.hbm_mapping_contract_ready(
        dma_target="hbm-peer",
        required_backend="cuda-dmabuf-ioas",
        gpu={**direct_gpu, "hbm_mapping_policy": "auto"},
    )
    assert runner.hbm_mapping_contract_ready(
        dma_target="host-mapped",
        required_backend="auto",
        gpu={"hbm_mapping_policy": "auto"},
    )
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    preflight = cmake[cmake.index("add_custom_target(\n  nta-vfio-preflight") :]
    assert "${CMAKE_COMMAND} -E env" in preflight
    assert "${NTA_TEST_NVME_ENVIRONMENT}" in preflight
    assert "NTA_TEST_NVME_HBM_BACKEND" in cmake
    assert "--hbm-backend=${NTA_TEST_NVME_HBM_BACKEND}" in cmake
    assert 'set(NTA_TEST_NVME_MANAGE_SESSION "OFF"' in cmake
    assert "NTA_TEST_NVME_MANAGE_SESSION=ON requires" in cmake
    assert '"NTA_ALLOW_DEVICE_REBIND=1"' in cmake
    assert '"run" "--"' in cmake
    assert "FIXTURES_SETUP nta_nvme_session" in cmake
    assert "FIXTURES_CLEANUP nta_nvme_session" in cmake
    assert "FIXTURES_REQUIRED nta_nvme_session" in cmake
    assert 'PASS_REGULAR_EXPRESSION "zero_write=pass"' in cmake
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
    assert "NvmeHbmMappingBackend::CudaDmaBufIoas" in control_plane
    assert "cuMemGetHandleForAddressRange" in control_plane
    assert "IOMMU_IOAS_MAP_FILE" in control_plane
    assert "NvmeMappingToken::Kind::CudaDmaBufIoas" in control_plane
    assert "policy_ == NvmeHbmMappingPolicy::CudaDmaBufIoas" in control_plane
    assert "policy_ == NvmeHbmMappingPolicy::NvidiaPeerPages" in control_plane
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
    assert "nta_runtime_install_registered_nvme_objects_async" in runtime_c
    runtime_c_impl = (ROOT / "runtime" / "host" / "RuntimeC.cpp").read_text(
        encoding="utf-8"
    )
    install_begin = runtime_c_impl.index("nta_runtime_install_nvme_object")
    install_end = runtime_c_impl.index("nta_runtime_read_pending_count", install_begin)
    install_body = runtime_c_impl[install_begin:install_end]
    assert "runtime->nvme->allocate" not in install_body
    assert "region->value->view" in install_body
    assert "mapExternalHbm" not in install_body
    planner = (ROOT / "python" / "nta_runtime" / "hbm_registration.py").read_text(
        encoding="utf-8"
    )
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

    uncalibrated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run-nvme-qualification.py"),
            "--queue-depth-candidates",
            "5",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert uncalibrated.returncode != 0
    assert "at least 2 GPU queue depths" in uncalibrated.stdout
    print("nvme_safety=pass")


if __name__ == "__main__":
    main()
