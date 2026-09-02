#!/usr/bin/env python3

from dataclasses import replace

from nta_runtime.acquisition_scheduler import (
    AcquisitionGroupIdentity,
    AcquisitionJob,
    AcquisitionJobState,
    AcquisitionQueue,
    AcquisitionServiceCurve,
    AcquisitionWork,
    LayerAcquisition,
    LayerAcquisitionModel,
    SharedAcquisitionJob,
    SharedAcquisitionPacket,
    SharedAcquisitionQueue,
    SharedAcquisitionState,
    TenantCreditCharge,
    TenantCreditLedger,
    schedule_acquisition_jobs,
    schedule_shared_acquisition_jobs,
    schedule_shared_acquisition_packets,
)


def main() -> None:
    credits = TenantCreditLedger(((1, 4096), (2, (1 << 64) - 1)))
    assert credits.finite and credits.active_lease_count == 0
    lease = credits.try_reserve(
        (
            TenantCreditCharge(1, 1024),
            TenantCreditCharge(1, 2048),
            TenantCreditCharge(2, 8192),
        )
    )
    assert lease is not None
    assert lease.charges == (
        TenantCreditCharge(1, 3072),
        TenantCreditCharge(2, 8192),
    )
    assert credits.outstanding_bytes(1) == 3072
    assert credits.outstanding_bytes(2) == 8192
    assert credits.try_reserve((TenantCreditCharge(1, 2048),)) is None

    # Numeric lease IDs are only diagnostic. A capability minted by another
    # ledger must neither release nor corrupt the live reservation, even when
    # both ledgers have the same first ID and exact charge tuple.
    foreign_credits = TenantCreditLedger(((1, 4096), (2, (1 << 64) - 1)))
    foreign_lease = foreign_credits.try_reserve(lease.charges)
    assert foreign_lease is not None and foreign_lease == lease
    try:
        credits.release(foreign_lease)
    except RuntimeError as error:
        assert "stale or foreign" in str(error)
    else:
        raise AssertionError("tenant ledger accepted a foreign capability")
    assert credits.active_lease_count == 1
    assert credits.outstanding_bytes(1) == 3072
    assert credits.outstanding_bytes(2) == 8192
    foreign_credits.release(foreign_lease)

    credits.release(lease)
    assert credits.active_lease_count == 0
    assert credits.outstanding_bytes(1) == credits.outstanding_bytes(2) == 0
    try:
        credits.release(lease)
    except RuntimeError as error:
        assert "stale or foreign" in str(error)
    else:
        raise AssertionError("tenant ledger accepted a duplicate release")

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

    # Shared-link policy uses complete semantic identities and absolute release
    # times. This example is deliberately not a simultaneous-release theorem:
    # work-conserving non-preemptive EDF starts the only available group, so a
    # later urgent group can miss even though an idling policy could wait for it.
    identity_a = AcquisitionGroupIdentity(3, 7, 0, 0, 8, 41)
    identity_b = AcquisitionGroupIdentity(9, 2, 0, 16, 4, 52)
    shared_jobs = (
        SharedAcquisitionJob(identity_a, 1, 8192, 8192, 0, 60, 100, 0),
        SharedAcquisitionJob(identity_b, 2, 4096, 4096, 10, 20, 50, 7),
    )
    shared_policy = schedule_shared_acquisition_jobs(shared_jobs)
    assert shared_policy.ordered_identities == (identity_a, identity_b)
    assert shared_policy.start_ns == (0, 60)
    assert shared_policy.completion_ns == (60, 80)
    assert not shared_policy.feasible
    assert shared_policy.first_missed_identity == identity_b
    assert shared_policy.maximum_lateness_ns == 30
    delayed_policy = schedule_shared_acquisition_jobs(
        (replace(shared_jobs[0], release_ns=20),), available_ns=10
    )
    assert delayed_policy.start_ns == (20,)

    # A physical packet is the non-preemptive boundary, while exact groups stay
    # independently named. A later urgent packet cannot be inserted into a
    # coalesced wave; splitting that wave into two packets makes the insertion
    # legal and is precisely the scheduler/transport co-design boundary.
    identity_c = AcquisitionGroupIdentity(10, 2, 0, 24, 4, 41)
    packet_a = SharedAcquisitionPacket.from_jobs(
        (
            replace(shared_jobs[0], service_ns=45, deadline_ns=200),
            SharedAcquisitionJob(identity_c, 2, 4096, 4096, 0, 45, 200),
        )
    )
    packet_b = SharedAcquisitionPacket.from_jobs(
        (replace(shared_jobs[1], service_ns=20, deadline_ns=100),)
    )
    packet_policy = schedule_shared_acquisition_packets((packet_a, packet_b))
    assert tuple(packet.identities for packet in packet_policy.ordered_packets) == (
        packet_a.identities,
        packet_b.identities,
    )
    assert packet_policy.completion_ns == (90, 110)
    assert packet_policy.first_missed_packet is packet_b
    split_policy = schedule_shared_acquisition_packets(
        (
            SharedAcquisitionPacket.from_jobs(
                (replace(shared_jobs[0], service_ns=45, deadline_ns=200),)
            ),
            packet_b,
            SharedAcquisitionPacket.from_jobs(
                (SharedAcquisitionJob(identity_c, 2, 4096, 4096, 0, 45, 200),)
            ),
        )
    )
    assert split_policy.feasible
    assert tuple(packet.identities[0] for packet in split_policy.ordered_packets) == (
        identity_a,
        identity_b,
        identity_c,
    )
    packet_queue = SharedAcquisitionQueue(
        staging_capacity_bytes=12288,
        tenant_credits=TenantCreditLedger(((1, 8192), (2, 8192))),
        max_inflight_groups=2,
    )
    packet_queue.add(
        packet_a_jobs := (
            replace(shared_jobs[0], service_ns=45, deadline_ns=200),
            SharedAcquisitionJob(identity_c, 2, 4096, 4096, 0, 45, 200),
        )
    )
    packet_queue.add(
        packet_b_jobs := (replace(shared_jobs[1], service_ns=20, deadline_ns=100),)
    )
    analyzed_packets = packet_queue.analyze_packets(
        (
            tuple(job.identity for job in packet_a_jobs),
            tuple(job.identity for job in packet_b_jobs),
        ),
        now_ns=0,
    )
    assert analyzed_packets.completion_ns == packet_policy.completion_ns
    assert packet_queue.claim_cohort(packet_a.identities, now_ns=0)
    # Both exact groups share one event, so neither byte reservation is modeled
    # as available at the first group's internal 45 ns service prefix.
    assert packet_queue.cohort_resource_delay_ns(packet_b.identities, now_ns=10) == 80

    # Dynamic dispatch spans requests and batches but exposes only one finite
    # non-preemptive group at a time. Staging and tenant credits are acquired in
    # the same claim transaction and released by physical readiness, while a
    # consumer may already be ordered behind the published fence.
    shared_credits = TenantCreditLedger(((1, 8192), (2, 4096)))
    shared_queue = SharedAcquisitionQueue(
        staging_capacity_bytes=8192,
        tenant_credits=shared_credits,
        max_inflight_groups=1,
    )
    shared_queue.add(shared_jobs)
    assert shared_queue.group_count == 2
    assert tuple(
        job.identity for job in shared_queue.claim_cohort((identity_a,), now_ns=0)
    ) == (identity_a,)
    assert shared_queue.staging_outstanding_bytes == 8192
    assert shared_credits.outstanding_bytes(1) == 8192
    assert not shared_queue.claim_cohort((identity_b,), now_ns=10)
    assert shared_queue.cohort_resource_delay_ns((identity_b,), now_ns=10) == 50
    assert (
        shared_queue.cohort_resource_delay_ns((identity_a, identity_b), now_ns=10)
        is None
    )
    shared_queue.publish_fence(identity_a)
    shared_queue.consume(identity_a)
    assert shared_queue.state(identity_a) is SharedAcquisitionState.FENCE_PUBLISHED
    shared_queue.mark_ready(identity_a)
    assert shared_queue.state(identity_a) is SharedAcquisitionState.CONSUMED
    assert shared_queue.staging_outstanding_bytes == 0
    assert shared_credits.outstanding_bytes(1) == 0
    assert shared_queue.cohort_resource_delay_ns((identity_b,), now_ns=60) == 0
    assert tuple(
        job.identity for job in shared_queue.claim_cohort((identity_b,), now_ns=60)
    ) == (identity_b,)
    shared_queue.publish_fence(identity_b)
    shared_queue.mark_ready(identity_b)
    shared_queue.consume(identity_b)
    assert shared_queue.state(identity_b) is SharedAcquisitionState.CONSUMED

    cohort_credits = TenantCreditLedger(((1, 8192), (2, 4096)))
    cohort_queue = SharedAcquisitionQueue(
        staging_capacity_bytes=12288,
        tenant_credits=cohort_credits,
        max_inflight_groups=2,
    )
    cohort_identity = AcquisitionGroupIdentity(9, 2, 0, 16, 4, 41)
    cohort_jobs = (
        replace(shared_jobs[0], deadline_ns=100),
        SharedAcquisitionJob(cohort_identity, 2, 4096, 4096, 0, 20, 100, 7),
    )
    cohort_queue.add(cohort_jobs)
    assert cohort_queue.next_released_identity(now_ns=0) == cohort_identity
    assert tuple(
        job.identity
        for job in cohort_queue.claim_cohort((identity_a, cohort_identity), now_ns=0)
    ) == (identity_a, cohort_identity)
    assert cohort_queue.staging_outstanding_bytes == 12288
    for identity in (identity_a, cohort_identity):
        cohort_queue.publish_fence(identity)
        cohort_queue.mark_ready(identity)
        cohort_queue.consume(identity)
    cohort_queue.forget_terminal((identity_a, cohort_identity))
    assert cohort_queue.group_count == 0

    cancelled_identity = AcquisitionGroupIdentity(3, 8, 1, 0, 1, 53)
    shared_queue.add((SharedAcquisitionJob(cancelled_identity, 1, 1, 1, 0, 1, 100),))
    assert shared_queue.cancel_request(3, 8) == 1
    assert shared_queue.state(cancelled_identity) is SharedAcquisitionState.CANCELLED

    inflight_identity = AcquisitionGroupIdentity(4, 1, 2, 0, 1, 54)
    inflight_queue = SharedAcquisitionQueue(
        staging_capacity_bytes=1,
        tenant_credits=TenantCreditLedger(()),
    )
    inflight_queue.add((SharedAcquisitionJob(inflight_identity, 0, 1, 1, 0, 1, 10),))
    assert inflight_queue.claim_cohort((inflight_identity,), now_ns=0)
    inflight_queue.publish_fence(inflight_identity)
    inflight_queue.cancel(inflight_identity)
    assert (
        inflight_queue.state(inflight_identity)
        is SharedAcquisitionState.FENCE_PUBLISHED
    )
    inflight_queue.mark_ready(inflight_identity)
    assert inflight_queue.state(inflight_identity) is SharedAcquisitionState.CANCELLED
    inflight_queue.forget_terminal((inflight_identity,))

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
    owner = LayerAcquisition(model.layer_bytes)
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
    assert (
        owner.submit_available(
            publish_range=publish_range,
            published_layers=published,
        ).job_count
        == 0
    )
    for layer in range(3):
        owner.retire(layer)
    assert owner.queue.terminal

    missing_fence = LayerAcquisition(model.layer_bytes)
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
        LayerAcquisition(model.layer_bytes).bind_model(
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
