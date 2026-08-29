#!/usr/bin/env python3
"""Validate SGLang forward ownership, retirement, and commit ordering."""

from __future__ import annotations

from collections import defaultdict
import operator
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines.sglang import NtaFlashInferAttnBackend  # noqa: E402
from nta_runtime.engines.sglang_lifecycle import (  # noqa: E402
    SglangForwardLifecycle,
)
from nta_runtime.engines.sglang_state import (  # noqa: E402
    SglangForwardEpoch,
    SglangForwardPlan,
)


class FakeAdapter:
    last_publish_count = 0
    last_metadata_publish_count = 0


class FakeHiCache:
    def __init__(self, pending=None, order=None) -> None:
        self._live = {} if pending is None else {pending.consumer_index: pending}
        self.order = [] if order is None else order

    def get(self, consumer_index: int):
        return self._live.get(consumer_index)

    def complete_layer(self, pending, local_layer: int) -> None:
        self.order.append("hicache")
        assert local_layer == 0
        assert self._live.pop(pending.consumer_index) is pending

    def retire(self, pending, *, stream) -> bool:
        self.order.append("hicache-retire")
        assert stream is not None
        return self._live.pop(pending.consumer_index, None) is pending


def new_stats():
    return defaultdict(int)


def new_lifecycle(*, layer_count: int, hicache, counters=None):
    return SglangForwardLifecycle(
        request_adapter=FakeAdapter(),
        hicache=hicache,
        granularity=object(),
        model_layer_count=layer_count,
        stats=new_stats() if counters is None else counters,
    )


def new_epoch(*, semantic_plans=None, pending=None, **state):
    return SglangForwardEpoch(
        plan=SglangForwardPlan(
            bindings=(),
            semantic_plans={} if semantic_plans is None else semantic_plans,
            pending_host_load=pending,
        ),
        **state,
    )


def assert_raises(fragment: str, operation) -> None:
    try:
        operation()
    except (RuntimeError, TypeError) as error:
        assert fragment in str(error)
    else:
        raise AssertionError(f"operation did not reject {fragment!r}")


def test_immutable_plan_and_atomic_alias_adoption() -> None:
    source_plan = SglangForwardPlan(
        bindings=(),
        semantic_plans={11: object()},
        pending_host_load=None,
    )
    assert_raises(
        "does not support item assignment",
        lambda: operator.setitem(source_plan.semantic_plans, 11, object()),
    )

    active = SglangForwardEpoch(plan=source_plan)
    owner = new_lifecycle(layer_count=1, hicache=FakeHiCache())
    owner.activate(active)
    assert_raises(
        "aliases disagree",
        lambda: owner.adopt_wrapper_aliases(active, {11: 23}, {}),
    )
    assert active.plan is source_plan
    assert owner.wrapper_alias_count == 0

    source_wrapper = object()
    owner.adopt_wrapper_aliases(active, {11: 23}, {23: source_wrapper})
    assert tuple(active.semantic_plans) == (23,)
    assert active.plan is not source_plan
    assert tuple(source_plan.semantic_plans) == (11,)
    assert owner.stock_wrapper(23) is source_wrapper


def test_unstarted_forward_form_transition() -> None:
    pending = SimpleNamespace(consumer_index=2)
    planned = new_epoch(semantic_plans={11: object()}, pending=pending)
    stock = new_epoch(pending=pending)
    owner = new_lifecycle(layer_count=1, hicache=FakeHiCache())
    owner.activate(planned)
    owner.replace_unstarted_epoch(planned, stock)
    assert owner.active is stock
    assert stock.semantic_plans == {}

    started = new_epoch(pending=pending)
    started.external_last_local_layer = 0
    started_owner = new_lifecycle(layer_count=1, hicache=FakeHiCache())
    started_owner.activate(started)
    assert_raises(
        "after execution began",
        lambda: started_owner.replace_unstarted_epoch(
            started,
            new_epoch(pending=pending),
        ),
    )
    assert started_owner.active is started

    different_pending = SimpleNamespace(consumer_index=2)
    assert_raises(
        "changed request or resource identity",
        lambda: owner.replace_unstarted_epoch(
            stock,
            new_epoch(pending=different_pending),
        ),
    )
    assert owner.active is stock


def test_profile_classification_survives_forward_release() -> None:
    owner = new_lifecycle(layer_count=1, hicache=FakeHiCache())

    external_cursor = owner.profile_cursor()
    external = new_epoch(pending=SimpleNamespace(consumer_index=41))
    owner.activate(external)
    owner.finish(external, retain_for_graph=False)
    assert owner.active is None
    assert owner.external_since(external_cursor)

    resident_cursor = owner.profile_cursor()
    resident = new_epoch()
    owner.activate(resident)
    owner.finish(resident, retain_for_graph=False)
    assert not owner.external_since(resident_cursor)
    assert_raises(
        "exactly one lifecycle epoch",
        lambda: owner.external_since(external_cursor),
    )


def test_lifecycle_boundaries() -> None:
    counters = new_stats()
    unfinished = new_epoch(stream_ordered_epoch=object())
    owner = new_lifecycle(layer_count=1, hicache=FakeHiCache(), counters=counters)
    owner.activate(unfinished)
    owner._engine_batch = object()
    owner._wrapper_aliases[7] = object()
    engine_batch = owner.engine_batch
    with patch("torch.cuda.synchronize") as synchronize:
        assert_raises("stream-ordered work window", owner.begin)
    synchronize.assert_not_called()
    assert owner.active is unfinished
    assert owner.engine_batch is engine_batch
    assert owner.wrapper_alias_count == 1

    pending = SimpleNamespace(consumer_index=3)
    live = new_epoch(pending=pending)
    live_cache = FakeHiCache(pending)
    live_owner = new_lifecycle(layer_count=1, hicache=live_cache)
    live_owner.activate(live)
    with patch("torch.cuda.synchronize") as synchronize:
        assert_raises("HiCache acquisition lease", live_owner.begin)
    synchronize.assert_not_called()
    assert live_owner.active is live

    complete = new_epoch()
    complete_owner = new_lifecycle(
        layer_count=1,
        hicache=FakeHiCache(),
        counters=counters,
    )
    complete_owner.activate(complete)
    complete_owner._engine_batch = object()
    complete_owner._wrapper_aliases[9] = object()
    with patch("torch.cuda.synchronize") as synchronize:
        complete_owner.finish(complete, retain_for_graph=False)
    synchronize.assert_not_called()
    assert complete_owner.active is None
    assert complete_owner.engine_batch is None
    assert complete_owner.wrapper_alias_count == 0
    assert counters["forward_lifecycle_completions"] == 1
    assert_raises(
        "lost its active epoch",
        lambda: complete_owner.finish(complete, retain_for_graph=False),
    )
    assert counters["forward_lifecycle_completions"] == 1

    retained = new_epoch()
    retained_owner = new_lifecycle(layer_count=1, hicache=FakeHiCache())
    retained_owner.activate(retained)
    retained_owner.finish(retained, retain_for_graph=True)
    assert retained_owner.active is retained
    assert_raises(
        "lost its active epoch",
        lambda: retained_owner.finish(new_epoch(), retain_for_graph=True),
    )


def assert_same(left, right) -> None:
    assert left is right


def test_abort_order_and_idempotence() -> None:
    order = []
    pending = SimpleNamespace(consumer_index=5)
    active = new_epoch(pending=pending, nvme_acquisition=object())
    counters = new_stats()
    owner = new_lifecycle(
        layer_count=1,
        hicache=FakeHiCache(pending, order),
        counters=counters,
    )
    owner.activate(active)
    stream = object()

    def abort_nvme(acquisition) -> None:
        assert_same(acquisition, active.nvme_acquisition)
        order.append("nvme-abort")

    with (
        patch(
            "torch.cuda.synchronize",
            side_effect=lambda: order.append("sync"),
        ) as synchronize,
        patch("torch.cuda.current_stream", return_value=stream),
    ):
        assert owner.abort(abort_nvme=abort_nvme)
        assert not owner.abort(abort_nvme=lambda _acquisition: None)
    synchronize.assert_called_once_with()
    assert order == ["sync", "nvme-abort", "hicache-retire"]
    assert owner.active is None
    assert counters["forward_lifecycle_aborts"] == 1


def test_abort_preserves_owners_until_quiescent() -> None:
    for failed_stage in ("sync", "nvme-abort", "hicache-retire"):
        order = []
        pending = SimpleNamespace(consumer_index=7)
        active = new_epoch(pending=pending, nvme_acquisition=object())
        counters = new_stats()

        class FaultHiCache(FakeHiCache):
            failed_stage = None

            def retire(self, item, *, stream) -> bool:
                order.append("hicache-retire")
                if self.failed_stage == "hicache-retire":
                    raise RuntimeError("fault:hicache-retire")
                return self._live.pop(item.consumer_index, None) is item

        hicache = FaultHiCache(pending, order)
        hicache.failed_stage = failed_stage
        owner = new_lifecycle(
            layer_count=1,
            hicache=hicache,
            counters=counters,
        )
        owner.activate(active)

        def synchronize() -> None:
            order.append("sync")
            if failed_stage == "sync":
                raise RuntimeError("fault:sync")

        def abort_nvme(_acquisition) -> None:
            order.append("nvme-abort")
            if failed_stage == "nvme-abort":
                raise RuntimeError("fault:nvme-abort")

        with (
            patch("torch.cuda.synchronize", side_effect=synchronize),
            patch("torch.cuda.current_stream", return_value=object()),
        ):
            assert_raises(
                f"fault:{failed_stage}",
                lambda: owner.abort(abort_nvme=abort_nvme),
            )
        expected_first = {
            "sync": ["sync"],
            "nvme-abort": ["sync", "nvme-abort", "hicache-retire"],
            "hicache-retire": ["sync", "nvme-abort", "hicache-retire"],
        }
        assert order == expected_first[failed_stage]
        assert owner.active is active
        assert counters["forward_lifecycle_aborts"] == 0
        assert (active.nvme_acquisition is None) == (
            failed_stage == "hicache-retire"
        )
        assert (hicache.get(pending.consumer_index) is pending) == (
            failed_stage in {"sync", "hicache-retire"}
        )

        # A retry completes only the owners that remain live and publishes one
        # lifecycle abort, never one count per attempted cleanup.
        order.clear()
        hicache.failed_stage = None

        def retry_nvme(_acquisition) -> None:
            order.append("nvme-abort")

        with (
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda: order.append("sync"),
            ),
            patch("torch.cuda.current_stream", return_value=object()),
        ):
            assert owner.abort(abort_nvme=retry_nvme)
            assert not owner.abort(
                pending,
                abort_nvme=lambda _acquisition: None,
            )
        expected_retry = {
            "sync": ["sync", "nvme-abort", "hicache-retire"],
            "nvme-abort": ["sync", "nvme-abort"],
            "hicache-retire": ["sync", "hicache-retire"],
        }
        assert order == expected_retry[failed_stage]
        assert owner.active is None
        assert counters["forward_lifecycle_aborts"] == 1

    order = []
    pending = SimpleNamespace(consumer_index=8)
    active = new_epoch(pending=pending, nvme_acquisition=object())
    counters = new_stats()
    owner = new_lifecycle(
        layer_count=1,
        hicache=FakeHiCache(pending, order),
        counters=counters,
    )
    owner.activate(active)
    with (
        patch("torch.cuda.synchronize", side_effect=lambda: order.append("sync")),
        patch("torch.cuda.current_stream", return_value=object()),
    ):
        assert_raises("no pipeline owner", owner.abort)
    assert order == ["sync", "hicache-retire"]
    assert owner.active is active
    assert counters["forward_lifecycle_aborts"] == 0
    with patch("torch.cuda.synchronize"):
        assert owner.abort(abort_nvme=lambda _acquisition: None)
    assert owner.active is None
    assert counters["forward_lifecycle_aborts"] == 1


def test_dispatch_validation_is_transactional() -> None:
    counters = new_stats()
    active = new_epoch()
    owner = new_lifecycle(layer_count=2, hicache=FakeHiCache(), counters=counters)
    owner.activate(active)

    assert_raises(
        "cannot claim progressive work",
        lambda: owner.record_external_dispatch(
            active,
            0,
            native_dispatch=False,
            progressive_consumer=True,
            final_layer=False,
        ),
    )
    assert active.external_last_local_layer == -1
    assert active.framework_dispatch_external_layers == 0

    assert_raises(
        "final-layer identity",
        lambda: owner.record_external_dispatch(
            active,
            0,
            native_dispatch=True,
            progressive_consumer=False,
            final_layer=True,
        ),
    )
    assert active.external_last_local_layer == -1

    owner.record_external_dispatch(
        active,
        0,
        native_dispatch=True,
        progressive_consumer=True,
        final_layer=False,
    )
    owner.record_external_dispatch(
        active,
        1,
        native_dispatch=False,
        progressive_consumer=False,
        final_layer=True,
    )
    assert active.external_dispatch_recorded
    assert counters["native_dispatch_prefix_layers_1_batches"] == 1
    assert counters["progressive_consumer_layers_1_batches"] == 1
    assert_raises(
        "after completion",
        lambda: owner.record_external_dispatch(
            active,
            2,
            native_dispatch=True,
            progressive_consumer=False,
            final_layer=False,
        ),
    )


def commit_backend(*, fail_at=None):
    order = []
    pending = SimpleNamespace(consumer_index=13)
    counters = new_stats()
    hicache = FakeHiCache(pending, order)
    active = new_epoch(pending=pending)
    owner = new_lifecycle(layer_count=1, hicache=hicache, counters=counters)
    owner.activate(active)
    backend = object.__new__(NtaFlashInferAttnBackend)
    backend._forward_lifecycle = owner
    backend._hicache = hicache
    backend._model_layer_count = 1
    backend._tier_service = SimpleNamespace(is_host_staged=True)
    backend._profile_cpu = False
    backend._stats = counters
    backend._nvme_pipeline = None

    def stage(name, action=lambda: None):
        def run(*_args, **_kwargs):
            order.append(name)
            if fail_at == name:
                raise RuntimeError(f"fault:{name}")
            return action()

        return run

    backend._materializer = SimpleNamespace(
        record_host_consumer=stage("quiescence")
    )
    backend._advance_deadline_frontier = stage("frontier")
    commit_dispatch = owner.commit_external_dispatch

    def record_dispatch(dispatch) -> None:
        order.append("dispatch")
        if fail_at == "dispatch":
            raise RuntimeError("fault:dispatch")
        commit_dispatch(dispatch)

    owner.commit_external_dispatch = record_dispatch
    hicache.complete_layer = stage(
        "hicache",
        lambda: assert_same(
            hicache._live.pop(pending.consumer_index),
            pending,
        ),
    )
    backend._finalize_stream_ordered_batch = stage("stream-retire")
    backend._commit_incremental_setup_observation = stage("observation")
    backend._publish_stats = stage("publish")
    backend._finish_forward = stage(
        "finish",
        lambda: owner.finish(active, retain_for_graph=False),
    )
    return backend, active, pending, owner, order


def test_external_commit_order_and_fault_containment() -> None:
    expected = [
        "quiescence",
        "stream-retire",
        "dispatch",
        "frontier",
        "hicache",
        "observation",
        "publish",
        "finish",
    ]
    backend, active, pending, owner, order = commit_backend()
    with (
        patch("torch.cuda.current_stream", return_value=object()),
        patch("torch.cuda.synchronize") as synchronize,
    ):
        backend._commit_external_layer(
            batch=active,
            pending=pending,
            layer=object(),
            local_layer=0,
            native_dispatch=True,
            progressive_consumer=True,
            publish_stats=True,
        )
    synchronize.assert_not_called()
    assert order == expected
    assert owner.active is None

    for failed_index, failed_stage in enumerate(expected):
        backend, active, pending, owner, order = commit_backend(fail_at=failed_stage)
        with (
            patch("torch.cuda.current_stream", return_value=object()),
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda: order.append("abort-sync"),
            ),
        ):
            assert_raises(
                f"fault:{failed_stage}",
                lambda: backend._commit_external_layer(
                    batch=active,
                    pending=pending,
                    layer=object(),
                    local_layer=0,
                    native_dispatch=True,
                    progressive_consumer=True,
                    publish_stats=True,
                ),
            )
        cleanup = ["abort-sync"]
        if failed_stage not in {"observation", "publish", "finish"}:
            cleanup.append("hicache-retire")
        assert order == expected[: failed_index + 1] + cleanup
        assert owner.active is None
        assert owner._hicache.get(pending.consumer_index) is None
        assert backend._stats["forward_lifecycle_aborts"] == 1

        # Automatic cleanup leaves the owner reusable by the next framework
        # forward instead of stranding a half-committed dispatch/lease pair.
        successor = new_epoch()
        owner.begin()
        owner.activate(successor)
        owner.finish(successor, retain_for_graph=False)
        assert owner.active is None

    # Duplicate/out-of-order identity is rejected before quiescence, frontier,
    # launch counters, or HiCache ownership can change.
    backend, active, pending, owner, order = commit_backend()
    active.external_last_local_layer = 0
    with (
        patch("torch.cuda.current_stream", return_value=object()),
        patch(
            "torch.cuda.synchronize",
            side_effect=lambda: order.append("abort-sync"),
        ),
    ):
        assert_raises(
            "not contiguous",
            lambda: backend._commit_external_layer(
                batch=active,
                pending=pending,
                layer=object(),
                local_layer=0,
                native_dispatch=True,
                progressive_consumer=True,
                publish_stats=True,
            ),
        )
    assert order == ["abort-sync", "hicache-retire"]
    assert backend._stats["external_launches"] == 0
    assert owner._hicache.get(pending.consumer_index) is None
    assert owner.active is None


def main() -> None:
    test_immutable_plan_and_atomic_alias_adoption()
    test_unstarted_forward_form_transition()
    test_profile_classification_survives_forward_release()
    test_lifecycle_boundaries()
    test_abort_order_and_idempotence()
    test_abort_preserves_owners_until_quiescent()
    test_dispatch_validation_is_transactional()
    test_external_commit_order_and_fault_containment()


if __name__ == "__main__":
    main()
