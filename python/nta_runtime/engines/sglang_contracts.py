"""Typed ownership and geometry contracts for the SGLang integration.

The records in this module carry no scheduler policy and own no framework
lifecycle.  They are the shared boundary between the SGLang hook, the HiCache
lease owner, topology construction, and the numerical attention adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PagePair = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class LeaseOperationTransfer:
    """Unmerged SGLang load operation captured into one owned lease."""

    operation_id: int
    node_id: int
    row_count: int

    def __post_init__(self) -> None:
        if self.operation_id < 0 or self.node_id < 0 or self.row_count <= 0:
            raise ValueError("SGLang lease operation geometry is invalid")


@dataclass(frozen=True, slots=True)
class LeaseOperationRequest:
    """Stable request owner captured at SGLang's unmerged operation edge.

    Generation is intentionally absent here: SGLang allocates the request slot
    before it constructs ``ForwardBatch``, while the runtime registry is the
    sole authority allowed to assign a generation.  The acquisition owner binds
    this record through that registry before any scheduled transfer is issued.
    """

    operation_id: int
    request_id: str
    request_slot: int
    logical_begin: int
    row_count: int
    tenant_id: int

    def __post_init__(self) -> None:
        if (
            self.operation_id < 0
            or not isinstance(self.request_id, str)
            or not self.request_id
            or not 0 <= self.request_slot < 1 << 32
            or self.logical_begin < 0
            or self.row_count <= 0
            or not 0 <= self.tenant_id < 1 << 32
        ):
            raise ValueError("SGLang lease request ownership is invalid")

    @property
    def logical_end(self) -> int:
        return self.logical_begin + self.row_count


@dataclass(frozen=True, slots=True)
class LeaseOperationRange:
    """One unmerged load operation's contiguous range in the merged lease."""

    operation_id: int
    row_begin: int
    row_count: int

    def __post_init__(self) -> None:
        if self.operation_id < 0 or self.row_begin < 0 or self.row_count <= 0:
            raise ValueError("SGLang lease operation range is invalid")

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class LeaseAcquisitionSlice:
    """Exact operation-local rows consumed by one numerical work unit."""

    operation_id: int
    row_begin: int
    row_count: int

    def __post_init__(self) -> None:
        if self.operation_id < 0 or self.row_begin < 0 or self.row_count <= 0:
            raise ValueError("SGLang work dependency geometry is invalid")

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class LeaseAcquisitionGroup:
    """One shared transfer/readiness interval within an owned lease operation."""

    operation_id: int
    row_begin: int
    row_count: int

    def __post_init__(self) -> None:
        if self.operation_id < 0 or self.row_begin < 0 or self.row_count <= 0:
            raise ValueError("SGLang acquisition-group geometry is invalid")

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class LeaseDeviceIndexMap:
    """Lease-owned device ABI map retained for every indexed acquisition."""

    source_indices: torch.Tensor
    destination_indices: torch.Tensor
    operations: tuple[LeaseOperationRange, ...]

    def __post_init__(self) -> None:
        source = self.source_indices
        destination = self.destination_indices
        if (
            source.device != destination.device
            or source.device.type != "cuda"
            or source.dtype is not torch.int32
            or destination.dtype is not torch.int32
            or source.ndim != 1
            or destination.ndim != 1
            or source.numel() <= 0
            or source.numel() != destination.numel()
        ):
            raise ValueError("SGLang lease device index map has an invalid ABI")
        cursor = 0
        operation_ids: set[int] = set()
        for operation in self.operations:
            if operation.operation_id in operation_ids or operation.row_begin != cursor:
                raise ValueError("SGLang lease operation ranges are not a partition")
            operation_ids.add(operation.operation_id)
            cursor = operation.row_end
        if cursor != int(source.numel()):
            raise ValueError("SGLang lease operation ranges do not cover the map")

    def operation(self, operation_id: int) -> LeaseOperationRange:
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        raise KeyError(operation_id)

    @property
    def retained_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_indices, self.destination_indices
