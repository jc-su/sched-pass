"""Single-owner lifecycle for SGLang forward execution epochs."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.sglang import SglangAdapter
from nta_runtime.engines.sglang_hicache import PendingHostLoad, SglangHiCacheBridge
from nta_runtime.engines.sglang_state import SglangForwardEpoch
from nta_runtime.requests import RequestBinding


@dataclass(frozen=True, slots=True)
class _ValidatedExternalDispatch:
    epoch: SglangForwardEpoch
    local_layer: int
    native_dispatch: bool
    progressive_consumer: bool
    final_layer: bool
    previous_local_layer: int


class SglangForwardLifecycle:
    """Own the one live framework epoch and its request/wrapper identity.

    Long-lived runtime, transport, and numerical services borrow an epoch; they
    never replace it. Normal completion is synchronization-free. Only abort is
    a CUDA quiescence boundary because auxiliary transport may still reference
    framework-owned source rows.
    """

    def __init__(
        self,
        *,
        request_adapter: SglangAdapter,
        hicache: SglangHiCacheBridge,
        granularity: Any,
        model_layer_count: int,
        stats: MutableMapping[str, Any],
    ) -> None:
        if model_layer_count <= 0:
            raise ValueError("SGLang forward lifecycle needs model layers")
        self._request_adapter = request_adapter
        self._hicache = hicache
        self._granularity = granularity
        self._model_layer_count = model_layer_count
        self._stats = stats
        self._next_engine_epoch = 0
        self._engine_batch: EngineBatch | None = None
        self._active: SglangForwardEpoch | None = None
        self._wrapper_aliases: dict[int, Any] = {}
        self._activation_serial = 0
        self._last_activation_external = False

    @property
    def active(self) -> SglangForwardEpoch | None:
        return self._active

    @property
    def engine_batch(self) -> EngineBatch | None:
        return self._engine_batch

    @property
    def wrapper_alias_count(self) -> int:
        return len(self._wrapper_aliases)

    def has_wrapper_alias(self, wrapper_id: int) -> bool:
        return wrapper_id in self._wrapper_aliases

    def stock_wrapper(self, wrapper_id: int) -> Any | None:
        return self._wrapper_aliases.get(wrapper_id)

    def begin(self) -> None:
        """Open a new framework boundary or reject a live predecessor."""

        epoch = self._active
        if epoch is None:
            self._engine_batch = None
            return
        if epoch.stream_ordered_epoch is not None:
            raise RuntimeError(
                "the preceding typed forward did not retire its stream-ordered "
                "work window"
            )
        pending = epoch.pending_host_load
        if pending is not None and self._hicache.get(pending.consumer_index) is pending:
            raise RuntimeError(
                "the preceding forward did not retire its HiCache acquisition lease"
            )
        self._release()

    def bind_requests(
        self, forward_batch: Any, *, allow_capture_ids: bool
    ) -> tuple[RequestBinding, ...]:
        """Bind request generations once and retain their engine batch."""

        batch = self._request_adapter.bind_forward(
            forward_batch,
            allow_capture_ids=allow_capture_ids,
            stream=torch.cuda.current_stream(),
            epoch=self._next_engine_epoch,
            granularity=self._granularity,
        )
        self._next_engine_epoch += 1
        self._engine_batch = batch
        self._stats["engine_batch_epoch"] = batch.epoch
        self._stats["engine_batch_size"] = len(batch.bindings)
        self._stats["request_rebindings"] += self._request_adapter.last_publish_count
        self._stats["request_metadata_updates"] = self._stats.get(
            "request_metadata_updates", 0
        ) + self._request_adapter.last_metadata_publish_count
        return batch.bindings

    def activate(self, epoch: SglangForwardEpoch) -> None:
        if self._active is not None:
            raise RuntimeError("SGLang activated two forward epochs")
        self._active = epoch
        self._activation_serial += 1
        self._last_activation_external = epoch.pending_host_load is not None

    def profile_cursor(self) -> int:
        """Return a token that identifies the next measured forward activation."""

        return self._activation_serial

    def external_since(self, cursor: int) -> bool:
        """Classify exactly one activation even after normal completion releases it."""

        if cursor < 0 or self._activation_serial != cursor + 1:
            raise RuntimeError(
                "profiled SGLang forward did not produce exactly one lifecycle epoch"
            )
        return self._last_activation_external

    def record_reference_forward(self) -> None:
        """Publish one resource-free framework-reference forward.

        Resident work has no request binding, acquisition lease, wrapper alias,
        or stream-ordered owner.  It therefore must not allocate an execution
        epoch merely so optional forward profiling can classify it.  The
        serial is the complete observation contract for this path; external
        forwards continue to use ``activate`` and ``finish``.
        """

        if self._active is not None or self._engine_batch is not None:
            raise RuntimeError(
                "resident reference forward observed a live execution owner"
            )
        if self._wrapper_aliases:
            raise RuntimeError(
                "resident reference forward observed live wrapper aliases"
            )
        self._activation_serial += 1
        self._last_activation_external = False

    def replace_unstarted_epoch(
        self,
        expected: SglangForwardEpoch,
        replacement: SglangForwardEpoch,
    ) -> None:
        """Change only the numerical form before any consumer can observe it."""

        if self._active is not expected:
            raise RuntimeError("SGLang forward-form transition lost its active epoch")
        if self._wrapper_aliases:
            raise RuntimeError("SGLang forward-form transition retained wrapper aliases")
        expected.require_unstarted("replace the forward numerical form")
        replacement.require_unstarted("install a replacement forward form")
        if (
            expected.bindings != replacement.bindings
            or expected.pending_host_load is not replacement.pending_host_load
        ):
            raise RuntimeError(
                "SGLang forward-form transition changed request or resource identity"
            )
        self._active = replacement

    def adopt_wrapper_aliases(
        self,
        epoch: SglangForwardEpoch,
        source_to_target: Mapping[int, int],
        target_to_source: Mapping[int, Any],
    ) -> None:
        """Atomically re-key the plan and install its numerical aliases."""

        if self._active is not epoch:
            raise RuntimeError("wrapper adoption lost its active forward epoch")
        target_ids = set(source_to_target.values())
        aliases = dict(target_to_source)
        if set(aliases) != target_ids:
            raise RuntimeError("wrapper aliases disagree with adopted identities")
        # Validate every fallible alias condition before replacing the epoch's
        # immutable plan.  A rejected adoption must leave both identity owners
        # unchanged so the caller can abort or retry without split identity.
        epoch.adopt_wrapper_identity(source_to_target)
        self._wrapper_aliases.clear()
        self._wrapper_aliases.update(aliases)

    def record_external_dispatch(
        self,
        epoch: SglangForwardEpoch,
        local_layer: int,
        *,
        native_dispatch: bool,
        progressive_consumer: bool,
        final_layer: bool,
    ) -> None:
        """Commit one exactly-once numerical dispatch to epoch state."""

        dispatch = self.validate_external_dispatch(
            epoch,
            local_layer,
            native_dispatch=native_dispatch,
            progressive_consumer=progressive_consumer,
            final_layer=final_layer,
        )
        self.commit_external_dispatch(dispatch)

    def commit_external_dispatch(
        self, dispatch: _ValidatedExternalDispatch
    ) -> None:
        """Commit a validation token if no intervening dispatch changed it."""

        epoch = dispatch.epoch
        if (
            self._active is not epoch
            or epoch.external_dispatch_recorded
            or epoch.external_last_local_layer != dispatch.previous_local_layer
        ):
            raise RuntimeError("validated external dispatch became stale")

        epoch.external_last_local_layer = dispatch.local_layer
        if dispatch.native_dispatch:
            if epoch.framework_dispatch_seen:
                epoch.native_dispatch_nonprefix_seen = True
            epoch.native_dispatch_external_layers += 1
        else:
            epoch.framework_dispatch_seen = True
            epoch.framework_dispatch_external_layers += 1
        if dispatch.progressive_consumer:
            epoch.progressive_consumer_external_layers += 1
        if not dispatch.final_layer:
            return

        observed_layers = (
            epoch.native_dispatch_external_layers
            + epoch.framework_dispatch_external_layers
        )
        if observed_layers != self._model_layer_count:
            raise RuntimeError(
                "external dispatch did not account for every model layer"
            )
        native_layers = epoch.native_dispatch_external_layers
        if epoch.native_dispatch_nonprefix_seen:
            self._stats["native_dispatch_nonprefix_batches"] += 1
            key = f"native_dispatch_nonprefix_layers_{native_layers}_batches"
        else:
            self._stats["native_dispatch_prefix_observations"] += 1
            key = f"native_dispatch_prefix_layers_{native_layers}_batches"
        self._stats[key] = self._stats.get(key, 0) + 1
        progressive_layers = epoch.progressive_consumer_external_layers
        self._stats["progressive_consumer_batch_observations"] += 1
        self._stats["progressive_consumer_layers"] += progressive_layers
        self._stats["progressive_consumer_batches"] = self._stats.get(
            "progressive_consumer_batches", 0
        ) + int(progressive_layers > 0)
        progressive_key = f"progressive_consumer_layers_{progressive_layers}_batches"
        self._stats[progressive_key] = self._stats.get(progressive_key, 0) + 1
        epoch.external_dispatch_recorded = True

    def validate_external_dispatch(
        self,
        epoch: SglangForwardEpoch,
        local_layer: int,
        *,
        native_dispatch: bool,
        progressive_consumer: bool,
        final_layer: bool,
    ) -> _ValidatedExternalDispatch:
        """Validate one dispatch without mutating lifecycle or resource state."""

        if self._active is not epoch:
            raise RuntimeError("external dispatch lost its active forward epoch")
        if epoch.external_dispatch_recorded:
            raise RuntimeError("external dispatch received a layer after completion")
        if local_layer != epoch.external_last_local_layer + 1:
            raise RuntimeError("external dispatch layers are not contiguous")
        if local_layer < 0 or local_layer >= self._model_layer_count:
            raise RuntimeError("external dispatch layer is outside the model")
        expected_final = local_layer + 1 == self._model_layer_count
        if final_layer != expected_final:
            raise RuntimeError("external dispatch final-layer identity is inconsistent")
        if progressive_consumer and not native_dispatch:
            raise RuntimeError(
                "framework external dispatch cannot claim progressive work"
            )
        return _ValidatedExternalDispatch(
            epoch=epoch,
            local_layer=local_layer,
            native_dispatch=native_dispatch,
            progressive_consumer=progressive_consumer,
            final_layer=final_layer,
            previous_local_layer=epoch.external_last_local_layer,
        )

    def finish(self, epoch: SglangForwardEpoch, *, retain_for_graph: bool) -> None:
        """Release normal forward ownership after its final consumer."""

        if self._active is not epoch:
            raise RuntimeError("SGLang forward completion lost its active epoch")
        if epoch.stream_ordered_epoch is not None:
            raise RuntimeError("SGLang forward completed with an unretired work epoch")
        pending = epoch.pending_host_load
        if pending is not None and self._hicache.get(pending.consumer_index) is pending:
            raise RuntimeError("SGLang forward completed with a live acquisition lease")
        if retain_for_graph:
            return
        self._release()
        self._stats["forward_lifecycle_completions"] += 1

    def abort(
        self,
        pending: PendingHostLoad | None = None,
    ) -> bool:
        """Quiesce and retire one abnormal forward exactly once."""

        epoch = self._active
        target = epoch.pending_host_load if epoch is not None else pending
        target_live = (
            target is not None
            and self._hicache.get(target.consumer_index) is target
        )
        if epoch is None and not target_live:
            self._engine_batch = None
            return False

        errors: list[BaseException] = []
        try:
            torch.cuda.synchronize()
        except BaseException:
            # No resource owner may be retired without a fence that joins every
            # numerical and auxiliary transport stream.  A current-stream event
            # is insufficient here: Host and NVMe producers use other streams.
            # Preserve the entire epoch so a later abort can retry quiescence.
            raise

        acquisition_complete = True
        if epoch is not None and epoch.acquisition is not None:
            try:
                epoch.acquisition.abort_after_quiescence()
                epoch.acquisition = None
            except BaseException as error:
                errors.append(error)
                acquisition_complete = False

        retired = False
        retire_complete = not target_live
        if target_live:
            try:
                # Global quiescence above makes source-row reuse safe.  The
                # stream argument retains HiCache's normal event contract; it
                # is no longer being used as a substitute for quiescence.
                retired = self._hicache.retire(
                    target,
                    stream=torch.cuda.current_stream(),
                )
                retire_complete = retired
                if not retired:
                    errors.append(
                        RuntimeError("live HiCache acquisition lease was not retired")
                    )
            except BaseException as error:
                errors.append(error)
                retire_complete = False

        cleanup_complete = acquisition_complete and retire_complete
        if cleanup_complete:
            self._release()
            self._stats["forward_lifecycle_aborts"] += 1
            return epoch is not None or retired

        if not errors:
            errors.append(RuntimeError("SGLang forward abort did not quiesce owners"))
        primary = errors[0]
        for additional in errors[1:]:
            primary.add_note(
                "additional abort failure: "
                f"{type(additional).__name__}: {additional}"
            )
        raise primary

    def reset_after_quiescence(self) -> None:
        """Drop Python references after the caller has synchronized CUDA."""

        self._release()

    def _release(self) -> None:
        self._active = None
        self._engine_batch = None
        self._wrapper_aliases.clear()
