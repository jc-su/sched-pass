"""Crash-safe deployment profiles for SGLang AUTO calibration.

Profiles contain only bounded timing models. They never contain prompt tokens,
request generations, CUDA events, pointers, leases, or tier-resource identity.
An explicit compatibility document binds every profile to one software,
hardware, model-partition, and execution-policy configuration before any state
is restored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch

from nta_runtime.execution_planner import HostCostModel


PROFILE_SCHEMA = 1
PROFILE_CLASSIFICATION = "nta-sglang-auto-calibration"
PROFILE_PATH_ENV = "NTA_EXECUTION_CALIBRATION_PROFILE"
PROFILE_READ_ONLY_ENV = "NTA_EXECUTION_CALIBRATION_PROFILE_READ_ONLY"
PROFILE_TAG_ENV = "NTA_EXECUTION_CALIBRATION_PROFILE_TAG"
_MAX_PROFILE_BYTES = 64 * 1024 * 1024


def _boolean_environment(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _parallel_value(model_runner: Any, name: str, default: int) -> int:
    value = getattr(getattr(model_runner, "ps", None), name, default)
    if value is None:
        value = default
    result = int(value)
    if result < 0:
        raise ValueError(f"SGLang parallel field {name} cannot be negative")
    return result


@dataclass(frozen=True, slots=True)
class SglangCalibrationProfileConfig:
    """Explicit profile path and write policy for one worker partition."""

    path: Path | None
    read_only: bool

    @classmethod
    def from_environment(
        cls, *, model_runner: Any, applicable: bool
    ) -> "SglangCalibrationProfileConfig":
        raw_path = os.environ.get(PROFILE_PATH_ENV, "").strip()
        read_only = _boolean_environment(PROFILE_READ_ONLY_ENV)
        if not raw_path:
            if read_only:
                raise ValueError(f"{PROFILE_READ_ONLY_ENV}=1 requires {PROFILE_PATH_ENV}")
            return cls(None, False)
        if not applicable:
            raise ValueError(
                f"{PROFILE_PATH_ENV} requires host-staged late-bound AUTO execution"
            )
        path = Path(raw_path).expanduser().resolve()
        ranks = {
            "tp": (
                _parallel_value(model_runner, "tp_rank", 0),
                _parallel_value(model_runner, "tp_size", 1),
            ),
            "pp": (
                _parallel_value(model_runner, "pp_rank", 0),
                _parallel_value(model_runner, "pp_size", 1),
            ),
            "dp": (
                _parallel_value(model_runner, "dp_rank", 0),
                _parallel_value(model_runner, "dp_size", 1),
            ),
        }
        active = [(name, rank) for name, (rank, size) in ranks.items() if size > 1]
        if active:
            suffix = "".join(f".{name}{rank}" for name, rank in active)
            path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        return cls(path, read_only)

    @property
    def enabled(self) -> bool:
        return self.path is not None


@dataclass(frozen=True, slots=True)
class SglangCalibrationRuntimeState:
    host_cost_model: HostCostModel
    incremental_calibration_probes_remaining: int
    incremental_initialization_probes_remaining: int
    incremental_setup_samples: int
    incremental_service_samples: int
    cost_model_transfer_samples: int


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _model_value(model_config: Any, *names: str, default: Any = None) -> Any:
    hf_config = getattr(model_config, "hf_config", None)
    for owner in (model_config, hf_config):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return value
    return default


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_sglang_calibration_compatibility(
    *,
    model_runner: Any,
    token_pool: Any,
    model_partition: Any,
    execution_config: Any,
    tuning: Any,
    engine_version: str,
    revision: str,
    runtime_api_version: int,
) -> dict[str, Any]:
    """Build content-independent identity for one calibrated deployment."""

    if not revision or revision == "unknown":
        raise ValueError(
            f"{PROFILE_PATH_ENV} requires a concrete NTA_REVISION so Python "
            "policy changes cannot reuse stale timing state"
        )
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    model_config = model_runner.model_config
    server_args = getattr(model_runner, "server_args", None)
    ps = getattr(model_runner, "ps", None)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    host_model = tuning.host_cost_model
    mover_model = tuning.host_mover_default_service_model
    return {
        "schema": PROFILE_SCHEMA,
        "runtime": {
            "nta_revision": revision,
            "nta_runtime_api": int(runtime_api_version),
            "nta_package": _package_version("nta-runtime"),
            "sglang_engine_contract": engine_version,
            "sglang_package": _package_version("sglang"),
            "flashinfer_package": _package_version("flashinfer-python"),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "python": platform.python_version(),
            "platform": platform.machine(),
            "profile_tag": os.environ.get(PROFILE_TAG_ENV, "default").strip()
            or "default",
        },
        "gpu": {
            "ordinal": int(device),
            "uuid": str(getattr(properties, "uuid", "unknown")),
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "multiprocessors": int(properties.multi_processor_count),
            "total_memory": int(properties.total_memory),
        },
        "cpu": {"affinity": affinity},
        "model": {
            "path": str(getattr(model_config, "model_path", "unknown")),
            "architectures": list(
                _model_value(model_config, "architectures", default=()) or ()
            ),
            "global_layers": int(model_partition.global_layer_count),
            "first_local_layer": int(model_partition.first_layer),
            "end_local_layer": int(model_partition.end_layer),
            "head_dim": int(_model_value(model_config, "head_dim", default=0) or 0),
            "attention_heads": int(
                _model_value(model_config, "num_attention_heads", default=0) or 0
            ),
            "kv_heads": int(
                _model_value(model_config, "num_key_value_heads", default=0) or 0
            ),
            "hidden_size": int(
                _model_value(model_config, "hidden_size", default=0) or 0
            ),
            "dtype": str(getattr(model_runner, "dtype", "unknown")),
            "kv_cache_dtype": str(
                getattr(model_runner, "kv_cache_dtype", "unknown")
            ),
            "page_size": int(getattr(token_pool, "page_size")),
            "quantization": _json_scalar(
                _model_value(model_config, "quantization", default=None)
            ),
            "attention_backend": _json_scalar(
                getattr(server_args, "attention_backend", None)
            ),
        },
        "parallel": {
            name: _json_scalar(getattr(ps, name, None))
            for name in (
                "tp_rank",
                "tp_size",
                "pp_rank",
                "pp_size",
                "dp_rank",
                "dp_size",
                "attn_tp_rank",
                "attn_tp_size",
                "attn_cp_rank",
                "attn_cp_size",
                "attn_dp_rank",
                "attn_dp_size",
                "moe_ep_rank",
                "moe_ep_size",
                "dcp_size",
                "gpu_id",
            )
        },
        "execution": {
            "protocol": execution_config.protocol.kind.value,
            "granularity": execution_config.protocol.granularity.value,
            "max_inflight_units": execution_config.protocol.max_inflight_units,
            "host_execution_mode": execution_config.host_execution_mode.value,
            "grouping": tuning.grouping,
            "frontier_layers_per_wave": tuning.frontier_layers_per_wave,
            "sm_acquisition_waves": tuning.sm_acquisition_waves,
            "sm_mover_max_worker_ctas": tuning.sm_mover_max_worker_ctas,
            "copy_engine_max_operations": tuning.copy_engine_max_operations,
            "indexed_copy_target_bytes": tuning.indexed_copy_target_bytes,
            "indexed_copy_max_blocks": tuning.indexed_copy_max_blocks,
            "host_mover_policy": tuning.host_mover_policy,
            "host_mover_calibration_samples": (
                tuning.host_mover_calibration_samples
            ),
            "layer_service_minimum_samples": (
                tuning.layer_service_minimum_samples
            ),
            "layer_service_maximum_samples": (
                tuning.layer_service_maximum_samples
            ),
            "incremental_calibration_probes": (
                tuning.incremental_calibration_probes
            ),
            "cost_model_prior": {
                field.name: getattr(host_model, field.name)
                for field in fields(HostCostModel)
            },
            "mover_prior": {
                "sm_bandwidth_bytes_per_second": (
                    mover_model.sm_bandwidth_bytes_per_second
                ),
                "copy_bandwidth_bytes_per_second": (
                    mover_model.copy_bandwidth_bytes_per_second
                ),
                "copy_operation_ns": mover_model.copy_operation_ns,
                "hybrid_join_ns": mover_model.hybrid_join_ns,
                "minimum_gain": mover_model.minimum_gain,
                "minimum_calibration_samples": (
                    mover_model.minimum_calibration_samples
                ),
                "copy_compute_overlap_efficiency": (
                    mover_model.copy_compute_overlap_efficiency
                ),
            },
        },
    }


def _mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{owner} must be a string-keyed object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{owner} fields disagree "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )


def _integer(value: Any, owner: str, *, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{owner} is not a bounded nonnegative integer")
    return value


def _first_difference(expected: Any, actual: Any, path: str = "compatibility") -> str:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            if key not in expected:
                return f"{path}.{key} is unexpected"
            if key not in actual:
                return f"{path}.{key} is missing"
            if expected[key] != actual[key]:
                return _first_difference(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length differs ({len(expected)} != {len(actual)})"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            if left != right:
                return _first_difference(left, right, f"{path}[{index}]")
    return f"{path} differs ({expected!r} != {actual!r})"


def _host_cost_state(
    model: HostCostModel,
    *,
    incremental_calibration_probes_remaining: int,
    incremental_initialization_probes_remaining: int,
    incremental_setup_samples: int,
    incremental_service_samples: int,
    cost_model_transfer_samples: int,
) -> dict[str, Any]:
    return {
        "schema": PROFILE_SCHEMA,
        "model": {field.name: getattr(model, field.name) for field in fields(model)},
        "incremental_calibration_probes_remaining": (
            incremental_calibration_probes_remaining
        ),
        "incremental_initialization_probes_remaining": (
            incremental_initialization_probes_remaining
        ),
        "incremental_setup_samples": incremental_setup_samples,
        "incremental_service_samples": incremental_service_samples,
        "cost_model_transfer_samples": cost_model_transfer_samples,
    }


def _restore_host_cost_state(
    value: Any,
    *,
    configured: HostCostModel,
    configured_probe_count: int,
) -> SglangCalibrationRuntimeState:
    state = _mapping(value, "host-cost calibration")
    fields_expected = frozenset(
        {
            "schema",
            "model",
            "incremental_calibration_probes_remaining",
            "incremental_initialization_probes_remaining",
            "incremental_setup_samples",
            "incremental_service_samples",
            "cost_model_transfer_samples",
        }
    )
    _exact_fields(state, fields_expected, "host-cost calibration")
    if _integer(state["schema"], "host-cost schema") != PROFILE_SCHEMA:
        raise ValueError("host-cost calibration schema is incompatible")
    model_state = _mapping(state["model"], "host-cost model")
    model_fields = frozenset(field.name for field in fields(HostCostModel))
    _exact_fields(model_state, model_fields, "host-cost model")
    kwargs: dict[str, Any] = {}
    for field in fields(HostCostModel):
        raw = model_state[field.name]
        if field.name == "incremental_service_scale":
            if raw is not None and (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError("host-cost service scale is invalid")
            kwargs[field.name] = None if raw is None else float(raw)
        elif field.name == "minimum_predicted_gain":
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError("host-cost minimum gain is invalid")
            kwargs[field.name] = float(raw)
        elif field.name == "incremental_setup_ns":
            kwargs[field.name] = None if raw is None else _integer(raw, field.name)
        else:
            kwargs[field.name] = _integer(raw, field.name)
    restored = HostCostModel(**kwargs)
    restored.validate()
    for name in (
        "round_overhead_ns",
        "tile_compute_ns",
        "max_rounds",
        "minimum_predicted_gain",
        "dependency_width",
    ):
        if getattr(restored, name) != getattr(configured, name):
            raise ValueError(f"host-cost static field {name} is incompatible")
    remaining = _integer(
        state["incremental_calibration_probes_remaining"],
        "remaining incremental probes",
        maximum=configured_probe_count,
    )
    initialization_remaining = _integer(
        state["incremental_initialization_probes_remaining"],
        "remaining initialization probes",
        maximum=configured_probe_count,
    )
    if initialization_remaining > remaining:
        raise ValueError("initialization probe state exceeds the remaining budget")
    return SglangCalibrationRuntimeState(
        host_cost_model=restored,
        incremental_calibration_probes_remaining=remaining,
        incremental_initialization_probes_remaining=initialization_remaining,
        incremental_setup_samples=_integer(
            state["incremental_setup_samples"], "incremental setup samples"
        ),
        incremental_service_samples=_integer(
            state["incremental_service_samples"], "incremental service samples"
        ),
        cost_model_transfer_samples=_integer(
            state["cost_model_transfer_samples"], "cost-model transfer samples"
        ),
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary_path = Path(output.name)
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class SglangCalibrationProfileStore:
    """Load/save one explicit, compatibility-bound worker profile."""

    def __init__(
        self,
        *,
        config: SglangCalibrationProfileConfig,
        compatibility: Mapping[str, Any],
        stats: dict[str, Any],
    ) -> None:
        if config.path is None:
            raise ValueError("enabled calibration profile has no path")
        self._path = config.path
        self._read_only = config.read_only
        self._compatibility = dict(compatibility)
        self._stats = stats
        self._last_state_encoding: bytes | None = None
        self._stats.update(
            {
                "calibration_profile_enabled": True,
                "calibration_profile_path": str(self._path),
                "calibration_profile_read_only": self._read_only,
                "calibration_profile_status": "unloaded",
                "calibration_profile_loaded_samples": 0,
                "calibration_profile_save_count": 0,
                "calibration_profile_sha256": None,
            }
        )

    @property
    def read_only(self) -> bool:
        return self._read_only

    def restore(
        self,
        *,
        layer_calibration: Any,
        consumer_calibration: Any,
        host_movers: Any,
        host_cost_model: HostCostModel,
        incremental_calibration_probes: int,
        incremental_initialization_probes: int,
        incremental_setup_samples: int,
        incremental_service_samples: int,
        cost_model_transfer_samples: int,
    ) -> SglangCalibrationRuntimeState:
        default = SglangCalibrationRuntimeState(
            host_cost_model,
            incremental_calibration_probes,
            incremental_initialization_probes,
            incremental_setup_samples,
            incremental_service_samples,
            cost_model_transfer_samples,
        )
        if not self._path.exists():
            if self._read_only:
                raise FileNotFoundError(
                    f"read-only calibration profile does not exist: {self._path}"
                )
            self._stats["calibration_profile_status"] = "absent"
            return default
        size = self._path.stat().st_size
        if size <= 0 or size > _MAX_PROFILE_BYTES:
            raise ValueError("calibration profile size is invalid")
        encoded_document = self._path.read_bytes()
        document = json.loads(encoded_document.decode("utf-8"))
        profile = _mapping(document, "calibration profile")
        _exact_fields(
            profile,
            frozenset(
                {
                    "schema",
                    "classification",
                    "written_unix_ns",
                    "compatibility",
                    "state",
                }
            ),
            "calibration profile",
        )
        if (
            _integer(profile["schema"], "calibration profile schema")
            != PROFILE_SCHEMA
            or profile["classification"] != PROFILE_CLASSIFICATION
        ):
            raise ValueError("calibration profile type is incompatible")
        _integer(profile["written_unix_ns"], "calibration profile timestamp")
        saved_compatibility = _mapping(
            profile["compatibility"], "calibration compatibility"
        )
        if saved_compatibility != self._compatibility:
            raise ValueError(
                "calibration profile compatibility mismatch: "
                + _first_difference(self._compatibility, saved_compatibility)
            )
        state = _mapping(profile["state"], "calibration state")
        _exact_fields(
            state,
            frozenset(
                {"layer_service", "consumer_policy", "host_mover", "host_cost"}
            ),
            "calibration state",
        )
        runtime_state = _restore_host_cost_state(
            state["host_cost"],
            configured=host_cost_model,
            configured_probe_count=incremental_calibration_probes,
        )
        loaded = layer_calibration.import_state(state["layer_service"])
        loaded += consumer_calibration.import_state(state["consumer_policy"])
        loaded += host_movers.import_state(state["host_mover"])
        self._last_state_encoding = json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._stats["calibration_profile_status"] = (
            "loaded_read_only" if self._read_only else "loaded"
        )
        self._stats["calibration_profile_loaded_samples"] = loaded
        self._stats["calibration_profile_loaded_unix_ns"] = time.time_ns()
        self._stats["calibration_profile_sha256"] = hashlib.sha256(
            encoded_document
        ).hexdigest()
        return runtime_state

    def save(
        self,
        *,
        layer_calibration: Any,
        consumer_calibration: Any,
        host_movers: Any,
        host_cost_model: HostCostModel,
        incremental_calibration_probes_remaining: int,
        incremental_initialization_probes_remaining: int,
        incremental_setup_samples: int,
        incremental_service_samples: int,
        cost_model_transfer_samples: int,
    ) -> None:
        if self._read_only:
            self._stats["calibration_profile_status"] = "loaded_read_only"
            return
        state = {
            "layer_service": layer_calibration.export_state(),
            "consumer_policy": consumer_calibration.export_state(),
            "host_mover": host_movers.export_state(),
            "host_cost": _host_cost_state(
                host_cost_model,
                incremental_calibration_probes_remaining=(
                    incremental_calibration_probes_remaining
                ),
                incremental_initialization_probes_remaining=(
                    incremental_initialization_probes_remaining
                ),
                incremental_setup_samples=incremental_setup_samples,
                incremental_service_samples=incremental_service_samples,
                cost_model_transfer_samples=cost_model_transfer_samples,
            ),
        }
        state_encoding = json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if state_encoding == self._last_state_encoding:
            self._stats["calibration_profile_status"] = "current"
            return
        document = {
            "schema": PROFILE_SCHEMA,
            "classification": PROFILE_CLASSIFICATION,
            "written_unix_ns": time.time_ns(),
            "compatibility": self._compatibility,
            "state": state,
        }
        _atomic_write_json(self._path, document)
        self._last_state_encoding = state_encoding
        self._stats["calibration_profile_status"] = "saved"
        self._stats["calibration_profile_save_count"] += 1
        self._stats["calibration_profile_saved_unix_ns"] = time.time_ns()
        self._stats["calibration_profile_sha256"] = hashlib.sha256(
            self._path.read_bytes()
        ).hexdigest()


__all__ = [
    "PROFILE_PATH_ENV",
    "PROFILE_READ_ONLY_ENV",
    "PROFILE_TAG_ENV",
    "SglangCalibrationProfileConfig",
    "SglangCalibrationProfileStore",
    "SglangCalibrationRuntimeState",
    "build_sglang_calibration_compatibility",
]
