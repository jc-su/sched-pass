#!/usr/bin/env python3
"""Validate crash-safe, compatibility-bound SGLang calibration profiles."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines.sglang_calibration_profile import (  # noqa: E402
    PROFILE_CPU_AFFINITY_ENV,
    PROFILE_PATH_ENV,
    PROFILE_READ_ONLY_ENV,
    SglangCalibrationProfileConfig,
    SglangCalibrationProfileStore,
    _declared_cpu_affinity,
)
from nta_runtime.execution_planner import HostCostModel  # noqa: E402


class Owner:
    def __init__(self, state: dict[str, object] | None = None) -> None:
        self.state = {} if state is None else state
        self.imports = 0

    def export_state(self) -> dict[str, object]:
        return self.state

    def import_state(self, state: dict[str, object]) -> int:
        self.state = state
        self.imports += 1
        return int(state.get("samples", 0))


def main() -> None:
    with patch.dict(os.environ, {PROFILE_CPU_AFFINITY_ENV: "16-18,20,18"}):
        assert _declared_cpu_affinity() == [16, 17, 18, 20]
    with patch.dict(os.environ, {PROFILE_CPU_AFFINITY_ENV: "18-16"}):
        try:
            _declared_cpu_affinity()
        except ValueError as error:
            assert PROFILE_CPU_AFFINITY_ENV in str(error)
        else:
            raise AssertionError("calibration profile accepted invalid CPU affinity")

    with tempfile.TemporaryDirectory() as directory:
        profile_path = Path(directory) / "auto.json"
        compatibility = {
            "schema": 1,
            "runtime": {"revision": "test-revision"},
            "model": {"layers": 32},
        }
        initial = HostCostModel()
        stats: dict[str, object] = {}
        store = SglangCalibrationProfileStore(
            config=SglangCalibrationProfileConfig(profile_path, False),
            compatibility=compatibility,
            stats=stats,
        )
        owners = (Owner({"samples": 2}), Owner({"samples": 4}), Owner({"samples": 6}))
        absent = store.restore(
            layer_calibration=owners[0],
            consumer_calibration=owners[1],
            host_movers=owners[2],
            host_cost_model=initial,
            incremental_calibration_probes=2,
            incremental_initialization_probes=1,
            incremental_setup_samples=0,
            incremental_service_samples=0,
            cost_model_transfer_samples=0,
        )
        assert absent.host_cost_model == initial
        assert stats["calibration_profile_status"] == "absent"
        assert all(owner.imports == 0 for owner in owners)

        learned = replace(
            initial,
            bandwidth_bytes_per_second=24_000_000_000,
            incremental_setup_ns=42_000,
            incremental_service_scale=1.25,
        )
        store.save(
            layer_calibration=owners[0],
            consumer_calibration=owners[1],
            host_movers=owners[2],
            host_cost_model=learned,
            incremental_calibration_probes_remaining=0,
            incremental_initialization_probes_remaining=0,
            incremental_setup_samples=3,
            incremental_service_samples=4,
            cost_model_transfer_samples=5,
        )
        assert profile_path.is_file()
        document = json.loads(profile_path.read_text(encoding="utf-8"))
        assert document["classification"] == "nta-sglang-auto-calibration"
        assert "prompt" not in profile_path.read_text(encoding="utf-8").lower()
        assert stats["calibration_profile_status"] == "saved"
        assert stats["calibration_profile_sha256"] == hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()

        restored_owners = (Owner(), Owner(), Owner())
        restored_stats: dict[str, object] = {}
        read_only = SglangCalibrationProfileStore(
            config=SglangCalibrationProfileConfig(profile_path, True),
            compatibility=compatibility,
            stats=restored_stats,
        )
        restored = read_only.restore(
            layer_calibration=restored_owners[0],
            consumer_calibration=restored_owners[1],
            host_movers=restored_owners[2],
            host_cost_model=initial,
            incremental_calibration_probes=2,
            incremental_initialization_probes=1,
            incremental_setup_samples=0,
            incremental_service_samples=0,
            cost_model_transfer_samples=0,
        )
        assert restored.host_cost_model == learned
        assert restored.incremental_calibration_probes_remaining == 0
        assert restored.incremental_initialization_probes_remaining == 0
        assert restored.incremental_setup_samples == 3
        assert restored.incremental_service_samples == 4
        assert restored.cost_model_transfer_samples == 5
        assert tuple(owner.state["samples"] for owner in restored_owners) == (2, 4, 6)
        assert restored_stats["calibration_profile_loaded_samples"] == 12
        assert restored_stats["calibration_profile_sha256"] == hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()

        before = profile_path.read_bytes()
        read_only.save(
            layer_calibration=restored_owners[0],
            consumer_calibration=restored_owners[1],
            host_movers=restored_owners[2],
            host_cost_model=restored.host_cost_model,
            incremental_calibration_probes_remaining=0,
            incremental_initialization_probes_remaining=0,
            incremental_setup_samples=3,
            incremental_service_samples=4,
            cost_model_transfer_samples=5,
        )
        assert profile_path.read_bytes() == before
        assert restored_stats["calibration_profile_status"] == "loaded_read_only"

        incompatible_owners = (Owner(), Owner(), Owner())
        incompatible = SglangCalibrationProfileStore(
            config=SglangCalibrationProfileConfig(profile_path, True),
            compatibility={**compatibility, "model": {"layers": 48}},
            stats={},
        )
        try:
            incompatible.restore(
                layer_calibration=incompatible_owners[0],
                consumer_calibration=incompatible_owners[1],
                host_movers=incompatible_owners[2],
                host_cost_model=initial,
                incremental_calibration_probes=2,
                incremental_initialization_probes=1,
                incremental_setup_samples=0,
                incremental_service_samples=0,
                cost_model_transfer_samples=0,
            )
        except ValueError as error:
            assert "compatibility.model.layers" in str(error)
        else:
            raise AssertionError("profile accepted a different model geometry")
        assert all(owner.imports == 0 for owner in incompatible_owners)

        runner = SimpleNamespace(
            ps=SimpleNamespace(
                tp_rank=1,
                tp_size=2,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
            )
        )
        with patch.dict(
            os.environ,
            {PROFILE_PATH_ENV: str(profile_path), PROFILE_READ_ONLY_ENV: "1"},
            clear=False,
        ):
            partitioned = SglangCalibrationProfileConfig.from_environment(
                model_runner=runner, applicable=True
            )
        assert partitioned.path == profile_path.with_name("auto.tp1.json")
        assert partitioned.read_only

        with patch.dict(
            os.environ,
            {PROFILE_PATH_ENV: str(profile_path), PROFILE_READ_ONLY_ENV: "0"},
            clear=False,
        ):
            try:
                SglangCalibrationProfileConfig.from_environment(
                    model_runner=runner, applicable=False
                )
            except ValueError as error:
                assert "requires host-staged" in str(error)
            else:
                raise AssertionError("profile was enabled for an inapplicable backend")


if __name__ == "__main__":
    main()
