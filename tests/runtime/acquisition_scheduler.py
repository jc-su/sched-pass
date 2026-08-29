#!/usr/bin/env python3

from dataclasses import replace

from nta_runtime.acquisition_scheduler import (
    AcquisitionJob,
    AcquisitionJobState,
    AcquisitionQueue,
    AcquisitionServiceCurve,
    AcquisitionWork,
    LayerAcquisitionModel,
    schedule_acquisition_jobs,
)
from nta_runtime.engines.sglang_acquisition import HostLayerAcquisition


def main() -> None:
    curve = AcquisitionServiceCurve(minimum_samples=3, maximum_samples=4)
    for sample in (2_000, 1_500):
        curve = curve.with_observation(sample)
    assert not curve.calibrated
    assert curve.overlap_budget_ns(35) == 0
    curve = curve.with_observation(1_750)
    assert curve.calibrated
    assert curve.conservative_interval_ns == 1_500
    assert curve.overlap_budget_ns(35) == 52_500
    curve = curve.with_observation(1_600)
    curve = curve.with_observation(1_700)
    assert curve.samples_ns == (1_500, 1_750, 1_600, 1_700)

    # Descriptor order is not policy. EDF establishes one deterministic order
    # and identifies the first causal miss and exact missing slack.
    jobs = (
        AcquisitionJob(2, 4096, 100, 300),
        AcquisitionJob(0, 4096, 120, 120),
        AcquisitionJob(1, 4096, 100, 220),
    )
    generic_edf = schedule_acquisition_jobs(
        (
            jobs[0],
            jobs[1],
            jobs[2],
        )
    )
    assert generic_edf.ordered_job_ids == (0, 1, 2)
    assert generic_edf.completion_ns == (120, 220, 320)
    assert not generic_edf.feasible
    assert generic_edf.first_missed_job_id == 2
    assert generic_edf.required_initial_slack_ns == 20
    assert schedule_acquisition_jobs(()).feasible

    try:
        schedule_acquisition_jobs(
            (AcquisitionJob(0, 1, 1, 1), AcquisitionJob(0, 1, 1, 2))
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("EDF accepted duplicate execution-local identities")

    # Queue capacity is backpressure, not a workload-size limit. Submission is
    # work-conserving and follows the same EDF order used by feasibility.
    queue = AcquisitionQueue.from_edf(jobs, max_inflight_jobs=2)
    assert tuple(job.job_id for job in queue.claim()) == (0, 1)
    assert not queue.claim()
    queue.publish_fence(0)
    queue.publish_fence(1)
    queue.retire(0)
    assert tuple(job.job_id for job in queue.claim()) == (2,)
    queue.cancel(1)
    queue.publish_fence(2)
    queue.retire(2)
    assert queue.terminal
    assert queue.state(0) is AcquisitionJobState.CONSUMED
    assert queue.state(1) is AcquisitionJobState.CANCELLED
    assert queue.state(2) is AcquisitionJobState.CONSUMED

    cancelled = AcquisitionQueue.from_edf(jobs, max_inflight_jobs=1)
    assert cancelled.claim()[0].job_id == 0
    cancelled.cancel_unfinished()
    assert cancelled.terminal
    assert all(
        cancelled.state(job_id) is AcquisitionJobState.CANCELLED
        for job_id in cancelled.job_ids
    )

    failed = AcquisitionQueue.from_edf(jobs[:1], max_inflight_jobs=1)
    assert failed.claim()[0].job_id == 2
    failed.fail(2)
    assert failed.terminal
    try:
        failed.publish_fence(2)
    except ValueError as error:
        assert "failed" in str(error)
    else:
        raise AssertionError("terminal acquisition job was republished")

    model = LayerAcquisitionModel(
        layer_bytes=(4096, 4096, 4096),
        transfer_service_ns=(100, 100, 100),
        initial_compute_ns=120,
        inter_layer_compute_ns=100,
    )
    feasible = model.analyze_admission(ready_prefix_layers=0)
    assert feasible.feasible
    assert feasible.first_missed_layer is None
    assert feasible.required_initial_slack_ns == 0
    assert feasible.cumulative_completion_ns == (100, 200, 300)
    assert feasible.deadlines_ns == (120, 220, 320)

    # The SGLang seam submits the complete finite layer queue in one coalesced
    # range. Fence publication and numerical retirement remain per layer.
    owner = HostLayerAcquisition(model.layer_bytes)
    assert owner.model is None
    assert owner.bind_model(model)
    assert owner.model == model
    assert not owner.bind_model(model)
    published: dict[int, object] = {}
    submitted_ranges: list[tuple[int, int]] = []

    def publish_range(begin: int, end: int) -> None:
        submitted_ranges.append((begin, end))
        published.update((layer, object()) for layer in range(begin, end))

    submission = owner.submit_available(
        publish_range=publish_range,
        published_layers=published,
    )
    assert submission.job_count == 3
    assert submission.ranges == ((0, 3),)
    assert submitted_ranges == [(0, 3)]
    assert owner.started and owner.fully_published
    assert owner.submit_available(
        publish_range=publish_range,
        published_layers=published,
    ).job_count == 0
    for layer in range(3):
        owner.retire(layer)
    assert owner.queue.terminal

    missing_fence = HostLayerAcquisition(model.layer_bytes)
    try:
        missing_fence.submit_available(
            publish_range=lambda _begin, _end: None,
            published_layers={},
        )
    except RuntimeError as error:
        assert "readiness fence" in str(error)
    else:
        raise AssertionError("Host acquisition accepted an unpublished fence")
    assert missing_fence.queue.terminal

    # Physical work may start before a calibrated feasibility model exists.
    # Explicit consumer order is lifecycle state, not a fake EDF estimate.
    unmodeled = AcquisitionQueue(
        (AcquisitionWork(0, 1024), AcquisitionWork(1, 2048)),
        ordered_job_ids=(0, 1),
        max_inflight_jobs=2,
    )
    assert tuple(job.job_id for job in unmodeled.claim()) == (0, 1)
    try:
        HostLayerAcquisition(model.layer_bytes).bind_model(
            replace(model, layer_bytes=(4096, 4096, 8192))
        )
    except RuntimeError as error:
        assert "byte ownership" in str(error)
    else:
        raise AssertionError("Host acquisition accepted a changed physical model")

    cold_model = replace(model, initial_compute_ns=0)
    cold = cold_model.analyze_admission(ready_prefix_layers=0)
    assert not cold.feasible
    assert cold.first_missed_layer == 0
    assert cold.required_initial_slack_ns == 100
    assert cold_model.minimum_admission_ready_prefix() == 1
    warmed = cold_model.analyze_admission(ready_prefix_layers=1)
    assert warmed.feasible
    assert warmed.cumulative_completion_ns == (100, 200)
    assert warmed.deadlines_ns == (100, 200)

    # Once layer-one attention has arrived, its historical compute interval is
    # gone. The suffix proof receives only future service.
    suffix = replace(
        cold_model, transfer_service_ns=(100, 100, 150)
    ).analyze_after_attention(completed_layer=1)
    assert not suffix.feasible
    assert suffix.first_missed_layer == 2
    assert suffix.cumulative_completion_ns == (150,)
    assert suffix.deadlines_ns == (100,)
    ready_suffix = replace(
        cold_model, transfer_service_ns=(100, 100, 150)
    ).analyze_after_attention(completed_layer=1, ready_prefix_layers=3)
    assert ready_suffix.feasible and not ready_suffix.cumulative_completion_ns
    complete = cold_model.analyze_admission(ready_prefix_layers=3)
    assert complete.feasible and not complete.cumulative_completion_ns

    # The forward-scoped table must be exactly equivalent to rebuilding the
    # simultaneous-release EDF suffix at every layer, including nonuniform
    # service and multiple possible first misses.
    frontier_model = LayerAcquisitionModel(
        layer_bytes=(1, 2, 3, 4, 5),
        transfer_service_ns=(50, 120, 40, 180, 20),
        initial_compute_ns=0,
        inter_layer_compute_ns=100,
    )
    frontier = frontier_model.compile_after_attention_frontier()
    assert frontier.layer_count == 5
    for completed_layer in range(frontier.layer_count):
        rebuilt = frontier_model.analyze_after_attention(
            completed_layer=completed_layer
        )
        expected_end = (
            frontier.layer_count
            if rebuilt.first_missed_layer is None
            else rebuilt.first_missed_layer
        )
        assert frontier.feasible_end_after_attention(completed_layer) == expected_end

    nonuniform = LayerAcquisitionModel(
        layer_bytes=(1024, 8192, 1024),
        transfer_service_ns=(80, 160, 40),
        initial_compute_ns=100,
        inter_layer_compute_ns=100,
    ).analyze_admission(ready_prefix_layers=0)
    assert not nonuniform.feasible
    assert nonuniform.first_missed_layer == 1
    assert nonuniform.required_initial_slack_ns == 40

    for invalid in (
        lambda: model.analyze_admission(ready_prefix_layers=4),
        lambda: model.analyze_after_attention(completed_layer=3),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError(
                "layer acquisition scheduler accepted invalid geometry"
            )


if __name__ == "__main__":
    main()
