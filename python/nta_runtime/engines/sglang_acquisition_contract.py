"""Typed producer/consumer contract shared by SGLang tier backends.

Framework metadata proves which KV rows are needed; a tier backend publishes
when those rows may be consumed.  This module is the single boundary between
those concerns.  Attention dispatch sees one layer record regardless of
whether Host DMA, an SM mover, or NVMe produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class AcquisitionTier(str, Enum):
    HOST_STAGED = "host_staged"
    NVME = "nvme"


class AcquisitionConsumerPlan(str, Enum):
    """How numerical work is bound after the producer publishes a fence."""

    HOST_MATERIALIZED = "host_materialized"
    PREACQUIRED = "preacquired"


@dataclass(frozen=True, slots=True)
class HostArrivalProfileKey:
    """Stable deployment-local class for producer/attention timing policy.

    Exact request lengths are deliberately not part of this key.  Production
    traces rarely repeat them, which would turn an online policy into a
    per-request cold start.  Power-of-two geometry classes preserve the
    dominant transfer/compute scale while each class still makes a
    conservative decision from its observed minimum arrival margin and
    maximum consumer cost.
    """

    phase: str
    query_rows_bucket: int
    batch_size_bucket: int
    transfer_rows_bucket: int
    transfer_bytes_bucket: int
    mover_kind: str
    layers_per_submission: int
    sm_waves_per_layer: int

    def __post_init__(self) -> None:
        if self.phase not in {"decode", "extend"}:
            raise ValueError("Host arrival profile has an invalid phase")
        if min(
            self.query_rows_bucket,
            self.batch_size_bucket,
            self.transfer_rows_bucket,
            self.transfer_bytes_bucket,
        ) < 0 or min(self.layers_per_submission, self.sm_waves_per_layer) <= 0:
            raise ValueError("Host arrival profile geometry must be positive")
        if self.mover_kind not in {"sm", "copy_engine", "hybrid"}:
            raise ValueError("Host arrival profile has an invalid mover kind")


@dataclass(frozen=True)
class HostLayerPublication:
    """One Host-produced layer and any exact intra-layer completion waves."""

    key_bytes: int
    value_bytes: int
    ready_event: torch.cuda.Event
    # SM movers use runtime object slots. Copy-engine movers are ordered only
    # by the CUDA event and therefore own no acquisition-directory entry.
    transfer_first_slot: int | None
    transfer_object_id_base: int | None
    transfer_object_version: int | None
    registration_event: torch.cuda.Event | None
    wave_events: tuple[torch.cuda.Event, ...]
    wave_object_slots: tuple[int, ...]
    wave_row_ends: tuple[int, ...]
    # A lease-unique timing marker is present only while the bounded consumer
    # policy is collecting producer-vs-attention arrival samples. Numerical
    # correctness continues to use ``ready_event``.
    profile_ready_event: torch.cuda.Event | None = None

    def __post_init__(self) -> None:
        if min(self.key_bytes, self.value_bytes) <= 0:
            raise ValueError("published Host layer byte geometry must be positive")
        if self.transfer_first_slot is None:
            if (
                self.transfer_object_id_base is not None
                or self.transfer_object_version is not None
                or self.registration_event is not None
                or self.wave_events
                or self.wave_object_slots
                or self.wave_row_ends
            ):
                raise ValueError("copy-engine layer retained SM wave state")
            return
        if (
            self.transfer_first_slot < 0
            or self.transfer_object_id_base is None
            or self.transfer_object_id_base <= 0
            or self.transfer_object_version is None
            or self.transfer_object_version <= 0
            or not self.wave_row_ends
            or any(end <= 0 for end in self.wave_row_ends)
            or tuple(sorted(set(self.wave_row_ends))) != self.wave_row_ends
        ):
            raise ValueError("SM-published layer wave geometry is invalid")
        event_owned = (
            len(self.wave_events) == len(self.wave_row_ends)
            and not self.wave_object_slots
            and self.registration_event is None
            and self.ready_event is self.wave_events[-1]
        )
        object_owned = (
            not self.wave_events
            and len(self.wave_object_slots) == len(self.wave_row_ends)
            and self.registration_event is not None
            and self.wave_object_slots
            == tuple(
                self.transfer_first_slot + 2 * wave
                for wave in range(len(self.wave_row_ends))
            )
        )
        if event_owned == object_owned:
            raise ValueError("SM-published layer readiness owner is ambiguous")

    @property
    def wave_count(self) -> int:
        return len(self.wave_row_ends)


@dataclass(frozen=True, slots=True)
class SglangLayerAcquisition:
    """Tier-neutral readiness record for one transformer layer."""

    owner: "SglangForwardAcquisition"
    local_layer: int
    layer_id: int
    ready_event: torch.cuda.Event
    tier: AcquisitionTier
    consumer_plan: AcquisitionConsumerPlan
    partial_publication: HostLayerPublication | None = None
    backend_record: Any = None

    def __post_init__(self) -> None:
        if self.local_layer < 0 or self.layer_id < 0:
            raise ValueError("layer acquisition identity cannot be negative")
        if self.tier is AcquisitionTier.HOST_STAGED:
            if (
                self.consumer_plan is not AcquisitionConsumerPlan.HOST_MATERIALIZED
                or self.partial_publication is None
            ):
                raise ValueError("Host acquisition omitted its materialization owner")
        elif (
            self.consumer_plan is not AcquisitionConsumerPlan.PREACQUIRED
            or self.partial_publication is not None
            or self.backend_record is None
        ):
            raise ValueError("NVMe acquisition omitted its preacquired owner")

    @property
    def progressive(self) -> bool:
        publication = self.partial_publication
        return publication is not None and publication.transfer_first_slot is not None


class SglangForwardAcquisition(ABC):
    """One forward-scoped producer owner consumed by numerical attention."""

    @property
    @abstractmethod
    def tier(self) -> AcquisitionTier:
        """Return the physical tier owned by this forward."""

    @abstractmethod
    def layer(self, local_layer: int) -> SglangLayerAcquisition | None:
        """Return the published layer, or ``None`` for typed on-demand work."""

    @abstractmethod
    def consume_layer(
        self,
        layer: SglangLayerAcquisition,
        stream: torch.cuda.Stream,
        *,
        wait_for_ready: bool,
    ) -> None:
        """Order or bind exactly one numerical consumer to a producer fence."""

    @abstractmethod
    def finish(self, stream: torch.cuda.Stream) -> None:
        """Fence the final consumer before producer resources may be reused."""

    @abstractmethod
    def abort_after_quiescence(self) -> None:
        """Release ownership after the caller has synchronized every stream."""


class HostForwardAcquisition(SglangForwardAcquisition):
    """Dynamic view over one HiCache Host lease's published layers."""

    def __init__(self, pending: Any) -> None:
        self._pending = pending
        self._start_layer = int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        self._layer_count = int(pending.controller.layer_num)
        if self._layer_count <= 0:
            raise ValueError("Host acquisition has no model layers")
        self._consumed: set[int] = set()

    @property
    def tier(self) -> AcquisitionTier:
        return AcquisitionTier.HOST_STAGED

    def layer(self, local_layer: int) -> SglangLayerAcquisition | None:
        if local_layer < 0 or local_layer >= self._layer_count:
            raise RuntimeError("Host acquisition layer is outside the model")
        publication = self._pending.prefetched_layers.get(local_layer)
        if publication is None:
            return None
        if not isinstance(publication, HostLayerPublication):
            raise RuntimeError("Host acquisition published an untyped layer")
        return SglangLayerAcquisition(
            self,
            local_layer,
            self._start_layer + local_layer,
            publication.ready_event,
            AcquisitionTier.HOST_STAGED,
            AcquisitionConsumerPlan.HOST_MATERIALIZED,
            publication,
        )

    def consume_layer(
        self,
        layer: SglangLayerAcquisition,
        stream: torch.cuda.Stream,
        *,
        wait_for_ready: bool,
    ) -> None:
        if layer.owner is not self or self.layer(layer.local_layer) != layer:
            raise RuntimeError("Host consumer uses a foreign layer publication")
        if layer.local_layer in self._consumed:
            raise RuntimeError("Host acquisition layer was consumed more than once")
        if wait_for_ready:
            stream.wait_event(layer.ready_event)
        elif not layer.progressive:
            raise RuntimeError("event-only Host acquisition cannot be consumed early")
        self._consumed.add(layer.local_layer)

    def finish(self, stream: torch.cuda.Stream) -> None:
        del stream

    def abort_after_quiescence(self) -> None:
        self._consumed.clear()
