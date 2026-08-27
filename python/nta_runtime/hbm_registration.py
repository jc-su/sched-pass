"""Framework-neutral setup planner for caller-owned NVMe HBM destinations.

CUDA caching allocators commonly pack several tensors into one allocation.
NVIDIA peer-pages pins whole peer-page envelopes, so independently registering
two logical tensors can attempt to own the same IOMMU PTE.  This module turns
validated native range descriptions into a disjoint registration plan.  It is
pure Python and performs no CUDA or transport work, which keeps the ownership
rule directly unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Protocol


_UINT64_LIMIT = 1 << 64


class HbmRangeDescription(Protocol):
    allocation_address: int
    allocation_bytes: int
    registration_address: int
    registration_bytes: int


@dataclass(frozen=True)
class HbmDestinationSlice:
    """One stable framework tensor slice that may receive NVMe payload bytes."""

    key: Hashable
    address: int
    bytes: int

    def __post_init__(self) -> None:
        if self.address <= 0 or self.bytes <= 0:
            raise ValueError("HBM destination address and bytes must be positive")
        if self.address >= _UINT64_LIMIT or self.bytes >= _UINT64_LIMIT:
            raise ValueError("HBM destination geometry exceeds uint64")
        if self.address + self.bytes >= _UINT64_LIMIT:
            raise ValueError("HBM destination range overflows uint64")
        try:
            hash(self.key)
        except TypeError as error:
            raise TypeError("HBM destination key must be hashable") from error


@dataclass(frozen=True)
class DescribedHbmDestination:
    destination: HbmDestinationSlice
    allocation_address: int
    allocation_bytes: int
    registration_address: int
    registration_bytes: int

    @classmethod
    def from_native(
        cls,
        destination: HbmDestinationSlice,
        description: HbmRangeDescription,
    ) -> "DescribedHbmDestination":
        return cls(
            destination,
            int(description.allocation_address),
            int(description.allocation_bytes),
            int(description.registration_address),
            int(description.registration_bytes),
        )

    def __post_init__(self) -> None:
        values = (
            self.allocation_address,
            self.allocation_bytes,
            self.registration_address,
            self.registration_bytes,
        )
        if any(value <= 0 or value >= _UINT64_LIMIT for value in values):
            raise ValueError("HBM range description must use positive uint64 values")
        allocation_end = self.allocation_address + self.allocation_bytes
        registration_end = self.registration_address + self.registration_bytes
        destination_end = self.destination.address + self.destination.bytes
        if allocation_end >= _UINT64_LIMIT or registration_end >= _UINT64_LIMIT:
            raise ValueError("HBM range description overflows uint64")
        if not (
            self.allocation_address
            <= self.registration_address
            <= self.destination.address
            and destination_end <= registration_end <= allocation_end
        ):
            raise ValueError(
                "HBM registration envelope does not contain its destination slice"
            )


@dataclass(frozen=True)
class HbmRegistrationGroup:
    """One disjoint peer mapping shared by all overlapping member slices."""

    allocation_address: int
    allocation_bytes: int
    registration_address: int
    registration_bytes: int
    destinations: tuple[HbmDestinationSlice, ...]

    @property
    def registration_end(self) -> int:
        return self.registration_address + self.registration_bytes


def coalesce_hbm_destinations(
    descriptions: Iterable[DescribedHbmDestination],
) -> tuple[HbmRegistrationGroup, ...]:
    """Return sorted, non-overlapping mappings for all described slices.

    Envelopes are merged only when they overlap within the same native CUDA
    allocation.  Adjacent envelopes remain independent because they share no
    PTE.  Conflicting allocation descriptions and duplicate logical keys fail
    closed before any caller performs a physical registration.
    """

    ordered = sorted(
        tuple(descriptions),
        key=lambda item: (
            item.allocation_address,
            item.registration_address,
            item.registration_bytes,
            repr(item.destination.key),
        ),
    )
    if not ordered:
        raise ValueError("HBM registration plan cannot be empty")
    keys = tuple(item.destination.key for item in ordered)
    if len(set(keys)) != len(keys):
        raise ValueError("HBM registration plan contains duplicate destination keys")

    mutable: list[dict[str, Any]] = []
    allocation_sizes: dict[int, int] = {}
    for item in ordered:
        previous_size = allocation_sizes.setdefault(
            item.allocation_address, item.allocation_bytes
        )
        if previous_size != item.allocation_bytes:
            raise ValueError("one CUDA allocation has conflicting native geometry")
        begin = item.registration_address
        end = begin + item.registration_bytes
        if (
            mutable
            and mutable[-1]["allocation_address"] == item.allocation_address
            and begin < mutable[-1]["registration_end"]
        ):
            mutable[-1]["registration_end"] = max(mutable[-1]["registration_end"], end)
            mutable[-1]["destinations"].append(item.destination)
            continue
        mutable.append(
            {
                "allocation_address": item.allocation_address,
                "allocation_bytes": item.allocation_bytes,
                "registration_address": begin,
                "registration_end": end,
                "destinations": [item.destination],
            }
        )

    groups = tuple(
        HbmRegistrationGroup(
            allocation_address=group["allocation_address"],
            allocation_bytes=group["allocation_bytes"],
            registration_address=group["registration_address"],
            registration_bytes=(
                group["registration_end"] - group["registration_address"]
            ),
            destinations=tuple(group["destinations"]),
        )
        for group in mutable
    )
    for previous, current in zip(groups, groups[1:]):
        if previous.registration_end > current.registration_address:
            raise RuntimeError("HBM registration planner produced overlapping mappings")
    return groups
