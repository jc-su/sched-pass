#!/usr/bin/env python3
"""Correctness tests for the physical-tier qualification boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qualify_tiers import assemble  # noqa: E402
from experiments.validate_tier_qualification import validate  # noqa: E402


def native(tier: str) -> dict[str, object]:
    return {
        "schema": 1,
        "classification": "nta-paged-attention",
        "tier": tier,
        "demand_semantics": "exact",
        "graph_ms": 1.0,
        "verification_failures": 0,
        "qualification": {"backend": "cuda", "qualified": True},
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-tier-qualification-") as directory:
        root = Path(directory)
        reports = {
            "hbm": native("hbm"),
            "host_mem": native("host_mem"),
            "nvme": {
                "schema": 1,
                "classification": "nta-vfio-nvme-qualification",
                "tier": "nvme",
                "ready": True,
                "transport_ready": True,
                "provenance_ready": True,
                "qualified": True,
                "demand_semantics": "exact",
                "revision": "a" * 40,
                "dirty": False,
                "runtime_abi": 32,
                "platform_identity": {
                    "boot_id": "11111111-2222-3333-4444-555555555555",
                    "kernel": "7.0.0-test",
                    "nvidia_driver_versions": ["595.84"],
                },
                "iommu_fault_free": True,
                "minimum_bandwidth_ratio": 0.5,
                "matched_bandwidth_ratio": 0.95,
                "performance_qualified": True,
                "required_hbm_backend": "cuda-dmabuf-ioas",
                "reported_hbm_mapping_policy": "cuda-dmabuf-ioas",
                "selected_hbm_backend": "cuda-dmabuf-ioas",
                "target_identity": {
                    "bdf": "0000:d8:00.0",
                    "namespace_id": 1,
                    "block_device": "/dev/nvme-test",
                    "kernel_driver": "nvme",
                    "model": "fixture-nvme",
                    "serial": "fixture-serial",
                },
                "baseline": {
                    "block_device": "/dev/nvme-test",
                    "queue_depth": 4,
                    "bandwidth_mib_per_second": 1000.0,
                },
                "baseline_candidates": [
                    {
                        "block_device": "/dev/nvme-test",
                        "block_bytes": 2097152,
                        "queue_depth": 4,
                        "bandwidth_mib_per_second": 1000.0,
                    },
                    {
                        "block_device": "/dev/nvme-test",
                        "block_bytes": 2097152,
                        "queue_depth": 8,
                        "bandwidth_mib_per_second": 990.0,
                    },
                ],
                "gpu_queue_depth_calibration": [
                    {
                        "queue_depth": 5,
                        "trials": 3,
                        "bandwidth_mib_per_second": [940.0, 950.0, 960.0],
                        "median_bandwidth_mib_per_second": 950.0,
                    },
                    {
                        "queue_depth": 8,
                        "trials": 3,
                        "bandwidth_mib_per_second": [900.0, 920.0, 910.0],
                        "median_bandwidth_mib_per_second": 910.0,
                    },
                ],
                "recommended_serving_config": {
                    "NTA_NVME_QUEUE_DEPTH": 5,
                    "transfer_bytes": 2097152,
                    "selection_metric": "median_exact_end_to_end_mib_per_second",
                },
                "gpu_controlled": {
                    "revision": "a" * 40,
                    "runtime_abi": 32,
                    "verified": True,
                    "selected_data_path_verified": True,
                    "destination": "hbm-peer",
                    "hbm_peer_dma_supported": True,
                    "hbm_mapping_policy": "cuda-dmabuf-ioas",
                    "hbm_mapping_backend": "cuda-dmabuf-ioas",
                    "translated_iommu": True,
                    "gpu_doorbell_mapping_validated": True,
                    "verification_failures": 0,
                    "failed": 0,
                    "outstanding": 0,
                    "device": "vfio:0000:d8:00.0",
                    "namespace_id": 1,
                    "queue_depth": 5,
                    "bytes_per_request": 2097152,
                    "end_to_end_mib_per_second": 950.0,
                },
            },
            "dax": {
                "schema": 1,
                "classification": "nta-dax-qualification",
                "tier": "dax",
                "status": "qualified",
                "qualified": True,
                "verification_failures": 0,
                "direct_device_visible": True,
            },
        }
        paths = {}
        for tier, report in reports.items():
            path = root / f"{tier}.json"
            path.write_text(json.dumps(report) + "\n", encoding="utf-8")
            paths[tier] = path
        output = root / "qualification.json"
        document = assemble(paths, output)
        validate(document)
        assert document["required_tiers"] == ["hbm", "host_mem", "nvme", "dax"]
        assert output.is_file()

        partial = assemble(
            {"hbm": paths["hbm"], "host_mem": paths["host_mem"]},
            root / "partial.json",
            required_tiers=("hbm", "host_mem"),
        )
        validate(partial, required_tiers=("hbm", "host_mem"))
        assert partial["required_tiers"] == ["hbm", "host_mem"]
        mismatched_scope = json.loads(json.dumps(partial))
        try:
            validate(mismatched_scope)
        except ValueError as error:
            assert "required_tiers" in str(error)
        else:
            raise AssertionError("partial qualification passed an all-tier scope")

        skipped = json.loads(json.dumps(document))
        skipped["entries"][3]["status"] = "skipped"
        skipped["entries"][3]["qualified"] = False
        try:
            validate(skipped)
        except ValueError as error:
            assert "not qualified" in str(error) or "status" in str(error)
        else:
            raise AssertionError("skipped DAX qualification was accepted")

        host_proxy = json.loads(json.dumps(document))
        host_proxy["entries"][2]["report"]["gpu_controlled"]["destination"] = (
            "host-mapped"
        )
        try:
            validate(host_proxy)
        except ValueError as error:
            assert "direct-HBM" in str(error)
        else:
            raise AssertionError(
                "host-mapped NVMe baseline passed direct-HBM admission"
            )

        iommu_fault = json.loads(json.dumps(document))
        iommu_fault["entries"][2]["report"]["iommu_fault_free"] = False
        try:
            validate(iommu_fault)
        except ValueError as error:
            assert "IOMMU fault" in str(error)
        else:
            raise AssertionError("faulting NVMe run passed tier admission")

        mismatched_policy = json.loads(json.dumps(document))
        mismatched_policy["entries"][2]["report"]["gpu_controlled"][
            "hbm_mapping_policy"
        ] = "auto"
        try:
            validate(mismatched_policy)
        except ValueError as error:
            assert "enforced natively" in str(error)
        else:
            raise AssertionError("post-hoc NVMe backend label passed admission")

        slow = json.loads(json.dumps(document))
        slow_report = slow["entries"][2]["report"]
        slow_report["matched_bandwidth_ratio"] = 0.35
        slow_report["performance_qualified"] = False
        slow_report["gpu_controlled"]["end_to_end_mib_per_second"] = 350.0
        slow_report["gpu_queue_depth_calibration"][0]["bandwidth_mib_per_second"] = [
            340.0,
            350.0,
            360.0,
        ]
        slow_report["gpu_queue_depth_calibration"][0][
            "median_bandwidth_mib_per_second"
        ] = 350.0
        slow_report["gpu_queue_depth_calibration"][1]["bandwidth_mib_per_second"] = [
            300.0,
            310.0,
            320.0,
        ]
        slow_report["gpu_queue_depth_calibration"][1][
            "median_bandwidth_mib_per_second"
        ] = 310.0
        try:
            validate(slow)
        except ValueError as error:
            assert "performance threshold" in str(error)
        else:
            raise AssertionError("slow NVMe transport passed tier admission")

        untuned = json.loads(json.dumps(document))
        untuned["entries"][2]["report"]["gpu_controlled"]["queue_depth"] = 8
        try:
            validate(untuned)
        except ValueError as error:
            assert "calibrated GPU queue depth" in str(error)
        else:
            raise AssertionError("an untuned NVMe queue passed tier admission")

        dirty = json.loads(json.dumps(document))
        dirty["entries"][2]["report"]["dirty"] = True
        try:
            validate(dirty)
        except ValueError as error:
            assert "dirty" in str(error)
        else:
            raise AssertionError("dirty NVMe qualification was accepted")
    print("tier_qualification=pass")


if __name__ == "__main__":
    main()
