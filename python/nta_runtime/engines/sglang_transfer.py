"""Typed, lease-scoped transfer planning for SGLang HiCache loads."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedMoverServiceModel,
    IndexedPairLayout,
)


def host_mover_service_model_from_environment(
    environ: dict[str, str] | None = None,
) -> IndexedMoverServiceModel:
    """Load an explicit deployment calibration, failing closed to SM.

    A copy-engine bandwidth without its per-operation issue cost (or vice
    versa) is not a usable calibration.  Keeping both optional makes the
    uncalibrated state representable instead of silently substituting a byte
    threshold.
    """

    values = os.environ if environ is None else environ
    host_bandwidth = int(
        values.get("NTA_TIER_HOST_STAGED_BANDWIDTH_BPS", 30_000_000_000)
    )
    sm_bandwidth_text = values.get("NTA_EXECUTION_HOST_SM_BANDWIDTH_BPS")
    sm_bandwidth = int(
        host_bandwidth if sm_bandwidth_text is None else sm_bandwidth_text
    )
    copy_bandwidth_text = values.get(
        "NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS"
    )
    copy_operation_text = values.get("NTA_EXECUTION_HOST_COPY_OPERATION_NS")
    if (copy_bandwidth_text is None) != (copy_operation_text is None):
        raise ValueError(
            "NTA host-copy calibration requires both COPY_BANDWIDTH_BPS "
            "and COPY_OPERATION_NS"
        )
    return IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=sm_bandwidth,
        copy_bandwidth_bytes_per_second=(
            None if copy_bandwidth_text is None else int(copy_bandwidth_text)
        ),
        copy_operation_ns=(
            None if copy_operation_text is None else int(copy_operation_text)
        ),
        hybrid_join_ns=int(values.get("NTA_EXECUTION_HOST_HYBRID_JOIN_NS", 0)),
        minimum_gain=float(
            values.get("NTA_EXECUTION_HOST_MOVER_MIN_GAIN", 1.03)
        ),
        # Explicit deployment calibrations are trusted observations. Defaults
        # remain priors and trigger bounded in-process probes before ``auto``
        # compares the two issuers.
        sm_samples=0 if sm_bandwidth_text is None else 1,
        copy_samples=0 if copy_bandwidth_text is None else 1,
    )


@dataclass(frozen=True)
class HostMoverLeasePlan:
    """One exact index partition reused by every layer of a host-load lease.

    SGLang publishes one source/destination page map for the whole model.  Run
    decomposition is therefore a lease property, not a layer-wave property;
    retaining it here prevents O(pages) analysis and descriptor D2H syncs from
    recurring when a later frontier is enqueued.
    """

    row_count: int
    kind: str
    copy_runs: tuple[ContiguousPairRun, ...]
    sm_source_indices: torch.Tensor
    sm_destination_indices: torch.Tensor
    layout: IndexedPairLayout | None
    layout_cpu_ns: int
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ValueError("host mover lease plan has no rows")
        if self.kind not in {"sm", "copy_engine", "hybrid"}:
            raise ValueError("host mover lease plan has an invalid kind")
        if self.layout_cpu_ns < 0:
            raise ValueError("host mover lease planning time cannot be negative")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None
            and self.predicted_selected_ns <= 0
        ):
            raise ValueError("host mover lease predictions must be positive")
        if self.sm_source_indices.device != self.sm_destination_indices.device:
            raise ValueError("host mover SM maps must share one device")
        if self.sm_source_indices.dtype is not torch.int32 or (
            self.sm_destination_indices.dtype is not torch.int32
        ):
            raise ValueError("host mover SM maps must use ABI int32 storage")
        if self.sm_source_indices.ndim != 1 or (
            self.sm_destination_indices.ndim != 1
        ):
            raise ValueError("host mover SM maps must be vectors")
        if self.sm_source_indices.numel() != self.sm_destination_indices.numel():
            raise ValueError("host mover SM maps disagree")
        if self.copy_row_count + self.sm_row_count != self.row_count:
            raise ValueError("host mover lease plan does not cover every row")
        expected_kind = (
            "sm"
            if not self.copy_runs
            else "copy_engine"
            if self.sm_row_count == 0
            else "hybrid"
        )
        if self.kind != expected_kind:
            raise ValueError("host mover lease kind disagrees with its partition")

    @property
    def copy_row_count(self) -> int:
        return sum(run.row_count for run in self.copy_runs)

    @property
    def sm_row_count(self) -> int:
        return int(self.sm_source_indices.numel())

    @property
    def retained_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.sm_row_count == 0:
            return ()
        return (self.sm_source_indices, self.sm_destination_indices)
