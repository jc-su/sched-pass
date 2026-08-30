"""Framework-neutral NVMe-to-HBM materialization ownership.

The engine owns the numerical HBM tensors, while the tier service owns storage
identity and the NVMe mapping.  This module is the narrow boundary between
those owners: it decomposes exact source/destination index pairs into bounded
NVMe runs, publishes every unique run exactly once, and retains the
single-use consumer proof required before a runtime directory slot is reused.

There is deliberately no request scheduling or framework lifecycle here.
Callers provide an immutable layer identity, typed tensor lanes, and an extent
resolver from their selected tier service.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any

from nta_runtime.indexed_transfer import ContiguousPairRun, analyze_index_pairs
from nta_runtime.runtime import RegisteredNvmeObjectInstall


IndexPair = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class NvmeRunPlan:
    """Exact, capacity-bounded transfer runs shared by all numerical work."""

    pair_runs: tuple[tuple[IndexPair, tuple[ContiguousPairRun, ...]], ...]
    unique_runs: tuple[ContiguousPairRun, ...]
    lane_element_bytes: tuple[int, ...]
    object_capacity: int
    rows_per_lba: int
    maximum_rows_per_command: int
    _lookup: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if (
            not self.lane_element_bytes
            or min(self.lane_element_bytes) <= 0
            or self.object_capacity <= 0
            or self.rows_per_lba <= 0
            or self.maximum_rows_per_command <= 0
        ):
            raise ValueError("NVMe run plan has invalid command geometry")
        lookup = dict(self.pair_runs)
        if len(lookup) != len(self.pair_runs):
            raise ValueError("NVMe run plan repeats an index pair")
        referenced = tuple(
            dict.fromkeys(run for _pair, runs in self.pair_runs for run in runs)
        )
        if referenced != self.unique_runs or not self.unique_runs:
            raise ValueError("NVMe run plan does not exactly own its unique runs")
        if self.object_count > self.object_capacity:
            raise ValueError("NVMe run plan exceeds its recorded object capacity")
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    @property
    def object_count(self) -> int:
        return len(self.lane_element_bytes) * len(self.unique_runs)

    def runs_for(self, pair: IndexPair) -> tuple[ContiguousPairRun, ...]:
        try:
            return self._lookup[pair]
        except KeyError as error:
            raise RuntimeError("NVMe index pair was not run-validated") from error


def plan_nvme_runs(
    index_pairs: Iterable[IndexPair],
    *,
    lane_element_bytes: tuple[int, ...],
    lba_size: int,
    max_transfer_bytes: int,
    object_capacity: int,
) -> NvmeRunPlan:
    """Build one exact run plan without opening or mutating a transport.

    Runs are maximal in both source and destination index spaces and then split
    only at the controller transfer limit.  The same run referenced by more
    than one work item remains one acquisition owner and consumes one set of
    directory slots.
    """

    if (
        not lane_element_bytes
        or min(lane_element_bytes) <= 0
        or lba_size <= 0
        or max_transfer_bytes <= 0
        or object_capacity <= 0
    ):
        raise ValueError("NVMe run planning requires positive resource geometry")
    rows_per_lba = math.lcm(
        *(
            lba_size // math.gcd(lba_size, element_bytes)
            for element_bytes in lane_element_bytes
        )
    )
    maximum_rows = min(
        max_transfer_bytes // element_bytes for element_bytes in lane_element_bytes
    )
    maximum_rows -= maximum_rows % rows_per_lba
    if maximum_rows <= 0:
        raise RuntimeError("one LBA-aligned row group exceeds the NVMe transfer limit")

    unique_pairs = tuple(dict.fromkeys(pair for pair in index_pairs if pair[0]))
    if not unique_pairs:
        raise RuntimeError("NVMe run planning has no external index pair")
    pair_runs: list[tuple[IndexPair, tuple[ContiguousPairRun, ...]]] = []
    for pair in unique_pairs:
        source_ordinals, destination_rows = pair
        if len(source_ordinals) != len(destination_rows):
            raise RuntimeError("NVMe source and destination index counts disagree")
        layout = analyze_index_pairs(source_ordinals, destination_rows)
        runs: list[ContiguousPairRun] = []
        for contiguous in layout.runs:
            if contiguous.row_count % rows_per_lba:
                raise RuntimeError("NVMe run is not exactly LBA materializable")
            consumed = 0
            while consumed < contiguous.row_count:
                row_count = min(maximum_rows, contiguous.row_count - consumed)
                runs.append(
                    ContiguousPairRun(
                        contiguous.source_first + consumed,
                        contiguous.destination_first + consumed,
                        row_count,
                    )
                )
                consumed += row_count
        if not runs:
            raise RuntimeError("NVMe index pair produced no transfer run")
        pair_runs.append((pair, tuple(runs)))

    unique_runs = tuple(dict.fromkeys(run for _pair, runs in pair_runs for run in runs))
    required_objects = len(unique_runs) * len(lane_element_bytes)
    if required_objects > object_capacity:
        raise RuntimeError(
            "NVMe layer needs more HBM object slots than the runtime capacity"
        )
    return NvmeRunPlan(
        tuple(pair_runs),
        unique_runs,
        tuple(lane_element_bytes),
        object_capacity,
        rows_per_lba,
        maximum_rows,
    )


@dataclass(frozen=True, slots=True)
class NvmeTensorLane:
    """One engine-owned numerical tensor lane materialized from NVMe."""

    component: str
    destination_address: int
    destination_rows: int
    row_bytes: int
    row_stride_bytes: int
    region: Any

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("NVMe tensor lane component cannot be empty")
        if (
            min(
                self.destination_address,
                self.destination_rows,
                self.row_bytes,
                self.row_stride_bytes,
            )
            <= 0
        ):
            raise ValueError("NVMe tensor lane geometry must be positive")
        if self.row_bytes != self.row_stride_bytes:
            raise ValueError("NVMe numerical destination rows must be contiguous")
        if self.region is None:
            raise ValueError("NVMe tensor lane has no registered HBM region")

    def address(self, run: ContiguousPairRun) -> int:
        if run.destination_first > self.destination_rows - run.row_count:
            raise RuntimeError("NVMe run exceeds its numerical destination")
        return self.destination_address + run.destination_first * self.row_stride_bytes


@dataclass(frozen=True, slots=True)
class NvmeObjectReference:
    slot: int
    object_id: int
    bytes: int


def publish_registered_nvme_objects(
    bindings: tuple[RegisteredNvmeObjectInstall, ...],
    *,
    runtime: Any,
    stream: Any,
) -> tuple[int, ...]:
    """Publish one fully prevalidated registered-HBM directory image.

    Validation completes before the first native mutation.  A reused slot's
    consumer event is passed for that slot, not merely once per batch: native
    object retirement is field-scoped even when stream ordering is shared.
    """

    if not bindings:
        raise ValueError("registered NVMe publication cannot be empty")
    if runtime is None or stream is None:
        raise ValueError("registered NVMe publication requires runtime and stream")
    if len({binding.slot for binding in bindings}) != len(bindings):
        raise ValueError("registered NVMe publication repeats a directory slot")
    if len({binding.object_id for binding in bindings}) != len(bindings):
        raise ValueError("registered NVMe publication repeats an object identity")

    expected_slots = tuple(range(bindings[0].slot, bindings[0].slot + len(bindings)))
    if tuple(binding.slot for binding in bindings) != expected_slots:
        raise ValueError(
            "registered NVMe publication slots must be contiguous and increasing"
        )
    installed = runtime.install_registered_nvme_objects_async(bindings, stream)
    expected = tuple(binding.destination_device_address for binding in bindings)
    if installed != expected:
        raise RuntimeError("NVMe objects do not alias their numerical destinations")
    return installed


@dataclass(frozen=True, slots=True)
class NvmePublicationCounters:
    fresh_slots: int = 0
    same_destination_slots: int = 0
    destination_rebinds: int = 0
    quiesced_replacements: int = 0


@dataclass(frozen=True, slots=True)
class NvmeRunPublication:
    """Immutable directory image produced by one successful publication."""

    plan: NvmeRunPlan
    lane_count: int
    objects_by_run: tuple[
        tuple[ContiguousPairRun, tuple[NvmeObjectReference, ...]], ...
    ]
    slot_destinations: tuple[tuple[int, int], ...]
    counters: NvmePublicationCounters
    _lookup: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        lookup = dict(self.objects_by_run)
        if (
            self.lane_count <= 0
            or tuple(lookup) != self.plan.unique_runs
            or any(len(objects) != self.lane_count for objects in lookup.values())
        ):
            raise ValueError("NVMe publication does not cover its run plan")
        if len(self.slot_destinations) != self.lane_count * len(lookup):
            raise ValueError("NVMe publication slot image is incomplete")
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    @property
    def object_count(self) -> int:
        return len(self.slot_destinations)

    @property
    def transfer_bytes(self) -> int:
        return sum(
            object_.bytes
            for _run, objects in self.objects_by_run
            for object_ in objects
        )

    def objects_for(
        self, pair: IndexPair
    ) -> tuple[tuple[NvmeObjectReference, ...], ...]:
        return tuple(self._lookup[run] for run in self.plan.runs_for(pair))


@dataclass(frozen=True, slots=True)
class PreparedNvmeRunPublication:
    """Fully validated host descriptor image, not yet visible to the GPU."""

    plan: NvmeRunPlan
    lane_count: int
    bindings: tuple[RegisteredNvmeObjectInstall, ...]
    runs_by_object: tuple[ContiguousPairRun, ...]
    previous_destinations: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if (
            self.lane_count <= 0
            or len(self.bindings) != self.plan.object_count
            or len(self.runs_by_object) != len(self.bindings)
            or len(self.previous_destinations) != len(self.bindings)
        ):
            raise ValueError("prepared NVMe publication is incomplete")

    @property
    def object_count(self) -> int:
        return len(self.bindings)


class NvmeSlotLifetime:
    """Single-use stream proof for runtime directory slot replacement.

    Directory entries belong to the transport acquisition, not to the later
    numerical kernel.  A slot may therefore be recycled once the transport
    epoch that referenced it has retired.  Callers that keep directory-backed
    numerical dependencies alive must record that later consumer instead; the
    same event contract covers both cases without conflating their owners.
    """

    def __init__(self, consumer_event: Any) -> None:
        self._consumer_event = consumer_event
        self._destinations: dict[int, int] = {}
        self._retired_slots: set[int] = set()

    def previous(self, slot: int) -> int | None:
        return self._destinations.get(slot)

    def prior_consumer_event(self, slot: int) -> Any | None:
        if slot not in self._destinations:
            return None
        if slot not in self._retired_slots:
            raise RuntimeError(
                f"NVMe slot {slot} replacement has no prior-consumer event"
            )
        return self._consumer_event

    def commit(self, publications: tuple[tuple[int, int], ...]) -> None:
        if not publications:
            raise ValueError("NVMe slot publication batch is empty")
        slots = tuple(slot for slot, _address in publications)
        if len(set(slots)) != len(slots) or any(
            address <= 0 for _slot, address in publications
        ):
            raise ValueError("NVMe slot publications must be unique and non-null")
        replaced = {slot for slot in slots if slot in self._destinations}
        missing = replaced - self._retired_slots
        if missing:
            raise RuntimeError(
                "NVMe replacement consumed no prior-consumer proof for slots "
                f"{sorted(missing)}"
            )
        self._destinations.update(publications)
        self._retired_slots.difference_update(replaced)

    def record_retirement(self, stream: Any) -> None:
        if not self._destinations:
            raise RuntimeError("cannot retire NVMe slots before publication")
        self._consumer_event.record(stream)
        self._retired_slots.update(self._destinations)


def prepare_nvme_runs(
    plan: NvmeRunPlan,
    lanes: tuple[NvmeTensorLane, ...],
    *,
    extent_resolver: Callable[[int, tuple[int, ...], str, int], Any],
    layer_id: int,
    object_version: int,
    object_id_base: int,
    first_object_slot: int = 0,
    lifetime: NvmeSlotLifetime,
) -> PreparedNvmeRunPublication:
    """Validate and materialize one run image without native mutation."""

    if (
        not lanes
        or object_version <= 0
        or object_version >= 1 << 32
        or object_id_base < 0
        or object_id_base >= 1 << 64
        or first_object_slot < 0
        or first_object_slot >= 1 << 32
        or layer_id < 0
    ):
        raise ValueError("NVMe publication identity is invalid")
    if tuple(lane.row_bytes for lane in lanes) != plan.lane_element_bytes:
        raise ValueError("NVMe publication lanes disagree with the run plan")
    if (
        plan.object_count >= 1 << 32
        or first_object_slot > (1 << 32) - plan.object_count
        or object_id_base > (1 << 64) - plan.object_count
    ):
        raise ValueError("NVMe publication exceeds the runtime slot ABI")
    object_ids = tuple(
        object_id_base + relative for relative in range(plan.object_count)
    )
    if len(set(object_ids)) != plan.object_count or any(
        object_id >= 1 << 64 for object_id in object_ids
    ):
        raise ValueError("NVMe object ID base aliases runtime directory slots")

    # Resolve and validate the entire catalog before mutating the runtime
    # directory.  A bad key or destination bound therefore cannot leave a
    # partially installed layer behind.
    resolved: list[tuple[ContiguousPairRun, NvmeTensorLane, Any]] = []
    for run in plan.unique_runs:
        ordinals = tuple(range(run.source_first, run.source_first + run.row_count))
        for lane in lanes:
            destination = lane.address(run)
            extent = extent_resolver(layer_id, ordinals, lane.component, lane.row_bytes)
            expected_bytes = run.row_count * lane.row_bytes
            if int(getattr(extent, "bytes", -1)) != expected_bytes:
                raise RuntimeError("NVMe catalog extent changed transfer byte geometry")
            if int(getattr(extent, "offset", -1)) < 0 or destination <= 0:
                raise RuntimeError("NVMe catalog produced an invalid physical extent")
            resolved.append((run, lane, extent))

    bindings: list[RegisteredNvmeObjectInstall] = []
    previous_destinations: list[int | None] = []
    runs_by_object: list[ContiguousPairRun] = []
    for relative, (run, lane, extent) in enumerate(resolved):
        slot = first_object_slot + relative
        object_id = object_ids[relative]
        previous = lifetime.previous(slot)
        quiescence = lifetime.prior_consumer_event(slot)
        destination = lane.address(run)
        bindings.append(
            RegisteredNvmeObjectInstall(
                slot,
                object_id,
                object_version,
                int(extent.offset),
                int(extent.bytes),
                lane.region,
                destination,
                quiescence,
            )
        )
        previous_destinations.append(previous)
        runs_by_object.append(run)

    return PreparedNvmeRunPublication(
        plan,
        len(lanes),
        tuple(bindings),
        tuple(runs_by_object),
        tuple(previous_destinations),
    )


def publish_prepared_nvme_runs(
    preparations: tuple[PreparedNvmeRunPublication, ...],
    *,
    runtime: Any,
    stream: Any,
    lifetime: NvmeSlotLifetime,
) -> tuple[NvmeRunPublication, ...]:
    """Commit contiguous prepared images as one native directory transaction."""

    if not preparations:
        raise ValueError("prepared NVMe publication batch cannot be empty")
    bindings = tuple(
        binding for preparation in preparations for binding in preparation.bindings
    )
    installed_addresses = publish_registered_nvme_objects(
        bindings, runtime=runtime, stream=stream
    )

    publications: list[NvmeRunPublication] = []
    cursor = 0
    for preparation in preparations:
        installed = installed_addresses[cursor : cursor + preparation.object_count]
        cursor += preparation.object_count
        expected = tuple(
            binding.destination_device_address for binding in preparation.bindings
        )
        if installed != expected:
            raise RuntimeError("NVMe objects do not alias their numerical destinations")

        run_objects: dict[ContiguousPairRun, list[NvmeObjectReference]] = {
            run: [] for run in preparation.plan.unique_runs
        }
        slot_destinations: list[tuple[int, int]] = []
        fresh = same = rebound = quiesced = 0
        for run, binding, previous, address in zip(
            preparation.runs_by_object,
            preparation.bindings,
            preparation.previous_destinations,
            installed,
            strict=True,
        ):
            if previous is None:
                fresh += 1
            elif previous == address:
                same += 1
            else:
                rebound += 1
            if previous is not None:
                quiesced += 1
            run_objects[run].append(
                NvmeObjectReference(binding.slot, binding.object_id, binding.bytes)
            )
            slot_destinations.append((binding.slot, address))

        committed = tuple(slot_destinations)
        lifetime.commit(committed)
        publications.append(
            NvmeRunPublication(
                preparation.plan,
                preparation.lane_count,
                tuple(
                    (run, tuple(run_objects[run]))
                    for run in preparation.plan.unique_runs
                ),
                committed,
                NvmePublicationCounters(fresh, same, rebound, quiesced),
            )
        )
    if cursor != len(installed_addresses):
        raise RuntimeError("NVMe publication result exceeds its prepared image")
    return tuple(publications)


def publish_nvme_runs(
    plan: NvmeRunPlan,
    lanes: tuple[NvmeTensorLane, ...],
    *,
    runtime: Any,
    extent_resolver: Callable[[int, tuple[int, ...], str, int], Any],
    layer_id: int,
    object_version: int,
    object_id_base: int,
    first_object_slot: int = 0,
    stream: Any,
    lifetime: NvmeSlotLifetime,
) -> NvmeRunPublication:
    """Prepare and publish one immutable run image."""

    preparation = prepare_nvme_runs(
        plan,
        lanes,
        extent_resolver=extent_resolver,
        layer_id=layer_id,
        object_version=object_version,
        object_id_base=object_id_base,
        first_object_slot=first_object_slot,
        lifetime=lifetime,
    )
    return publish_prepared_nvme_runs(
        (preparation,), runtime=runtime, stream=stream, lifetime=lifetime
    )[0]
