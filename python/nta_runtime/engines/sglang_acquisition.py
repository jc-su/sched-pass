"""Lease-scoped Host acquisition submission for the SGLang adapter.

The tier-neutral scheduler owns ordering and job state.  This narrow adapter
coalesces adjacent layer jobs into transport ranges and verifies that the Host
backend published one readiness fence for every claimed job.  It owns no CUDA
stream, request identity, attention policy, or HiCache acknowledgement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nta_runtime.acquisition_scheduler import (
    AcquisitionJobState,
    AcquisitionQueue,
    AcquisitionWork,
    LayerAcquisitionModel,
    schedule_acquisition_jobs,
)


@dataclass(frozen=True, slots=True)
class AcquisitionSubmission:
    """One work-conserving submission result."""

    job_count: int
    ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.job_count < 0 or any(
            begin < 0 or end <= begin for begin, end in self.ranges
        ):
            raise ValueError("Host acquisition submission geometry is invalid")
        if self.job_count != sum(end - begin for begin, end in self.ranges):
            raise ValueError("Host acquisition ranges do not cover the claimed jobs")


class HostLayerAcquisition:
    """Own the finite layer-job lifecycle for one exact HiCache lease."""

    def __init__(self, layer_bytes: tuple[int, ...]) -> None:
        if not layer_bytes or any(value <= 0 for value in layer_bytes):
            raise ValueError("Host acquisition requires positive layer bytes")
        self._layer_bytes = tuple(layer_bytes)
        self._model: LayerAcquisitionModel | None = None
        self.queue = AcquisitionQueue(
            tuple(
                AcquisitionWork(layer, payload_bytes)
                for layer, payload_bytes in enumerate(self._layer_bytes)
            ),
            # Transformer attention consumes local layers in this structural
            # order.  Once timing calibration is available, bind_model proves
            # that simultaneous-release EDF has the same order.
            ordered_job_ids=range(len(self._layer_bytes)),
            # A Host mover stream is already the serialized physical link.
            # Publish the complete finite queue once so the link never idles
            # between Python layer callbacks; CUDA events retain per-layer
            # readiness for the numerical consumer.
            max_inflight_jobs=len(self._layer_bytes),
        )

    @property
    def model(self) -> LayerAcquisitionModel | None:
        return self._model

    def bind_model(self, model: LayerAcquisitionModel) -> bool:
        """Attach calibrated feasibility without changing physical ownership.

        Returns true only for the first binding. Repeated calls must be exactly
        equivalent so admission polling cannot silently change a live proof.
        """

        if model.layer_bytes != self._layer_bytes:
            raise RuntimeError("Host EDF model changed acquisition byte ownership")
        if schedule_acquisition_jobs(model.admission_jobs()).ordered_job_ids != (
            self.queue.job_ids
        ):
            raise RuntimeError("Host EDF order disagrees with numerical layer order")
        if self._model is None:
            self._model = model
            return True
        if self._model != model:
            raise RuntimeError("Host acquisition changed its calibrated EDF model")
        return False

    @property
    def started(self) -> bool:
        return self.queue.count_states(AcquisitionJobState.PLANNED) != len(
            self.queue.job_ids
        )

    @property
    def fully_published(self) -> bool:
        return self.queue.count_states(
            AcquisitionJobState.FENCE_PUBLISHED,
            AcquisitionJobState.CONSUMED,
        ) == len(self.queue.job_ids)

    def submit_available(
        self,
        *,
        publish_range: Callable[[int, int], None],
        published_layers: Mapping[int, Any],
    ) -> AcquisitionSubmission:
        """Fill available link slots and publish each claimed readiness fence."""

        claimed = self.queue.claim()
        if not claimed:
            return AcquisitionSubmission(0, ())
        claimed_ids = tuple(job.job_id for job in claimed)
        ranges = _contiguous_ranges(claimed_ids)
        try:
            for begin, end in ranges:
                publish_range(begin, end)
                for layer in range(begin, end):
                    if layer not in published_layers:
                        raise RuntimeError(
                            "Host transport returned without publishing layer "
                            f"{layer}'s readiness fence"
                        )
                    self.queue.publish_fence(layer)
        except BaseException:
            for job_id in claimed_ids:
                if self.queue.state(job_id) is AcquisitionJobState.SUBMITTED:
                    self.queue.fail(job_id)
            raise
        return AcquisitionSubmission(len(claimed), ranges)

    def retire(self, layer: int) -> None:
        """Retire one layer after its numerical consumer has been ordered."""

        if layer not in self.queue.job_ids:
            raise RuntimeError(f"Host acquisition does not own layer {layer}")
        state = self.queue.state(layer)
        if state is not AcquisitionJobState.FENCE_PUBLISHED:
            raise RuntimeError(
                f"Host acquisition layer {layer} reached its consumer in "
                f"state {state.value}"
            )
        self.queue.retire(layer)

    def retire_published(self) -> None:
        """Retire a fully published graph batch at its stream handoff."""

        for job_id in self.queue.job_ids:
            state = self.queue.state(job_id)
            if state is AcquisitionJobState.FENCE_PUBLISHED:
                self.queue.retire(job_id)
            elif state is not AcquisitionJobState.CONSUMED:
                raise RuntimeError(
                    "Host graph handoff contains an unpublished acquisition job"
                )

    def cancel_unfinished(self) -> None:
        self.queue.cancel_unfinished()


def _contiguous_ranges(job_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Coalesce only adjacent EDF jobs, preserving the scheduler's order."""

    if not job_ids:
        return ()
    ranges: list[tuple[int, int]] = []
    begin = previous = job_ids[0]
    for job_id in job_ids[1:]:
        if job_id == previous + 1:
            previous = job_id
            continue
        ranges.append((begin, previous + 1))
        begin = previous = job_id
    ranges.append((begin, previous + 1))
    return tuple(ranges)
