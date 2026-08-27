#!/usr/bin/env python3
"""Validate disjoint setup planning for allocator-packed HBM tensors."""

from __future__ import annotations

from dataclasses import dataclass

from nta_runtime.hbm_registration import (
    DescribedHbmDestination,
    HbmDestinationSlice,
    coalesce_hbm_destinations,
)


@dataclass(frozen=True)
class NativeRange:
    allocation_address: int
    allocation_bytes: int
    registration_address: int
    registration_bytes: int


def described(
    key: str,
    address: int,
    bytes: int,
    *,
    allocation: int,
    allocation_bytes: int,
    registration: int,
    registration_bytes: int,
) -> DescribedHbmDestination:
    return DescribedHbmDestination.from_native(
        HbmDestinationSlice(key, address, bytes),
        NativeRange(
            allocation,
            allocation_bytes,
            registration,
            registration_bytes,
        ),
    )


def main() -> None:
    packed_key = described(
        "layer-0-key",
        0x11000,
        0x7000,
        allocation=0x10000,
        allocation_bytes=0x80000,
        registration=0x10000,
        registration_bytes=0x10000,
    )
    packed_value = described(
        "layer-0-value",
        0x19000,
        0x15000,
        allocation=0x10000,
        allocation_bytes=0x80000,
        registration=0x10000,
        registration_bytes=0x20000,
    )
    adjacent = described(
        "layer-1-key",
        0x31000,
        0x1000,
        allocation=0x10000,
        allocation_bytes=0x80000,
        registration=0x30000,
        registration_bytes=0x10000,
    )
    independent = described(
        "layer-1-value",
        0x101000,
        0x1000,
        allocation=0x100000,
        allocation_bytes=0x40000,
        registration=0x100000,
        registration_bytes=0x10000,
    )
    groups = coalesce_hbm_destinations(
        (independent, packed_value, adjacent, packed_key)
    )
    assert len(groups) == 3
    assert groups[0].registration_address == 0x10000
    assert groups[0].registration_bytes == 0x20000
    assert {item.key for item in groups[0].destinations} == {
        "layer-0-key",
        "layer-0-value",
    }
    assert groups[1].registration_address == 0x30000
    assert groups[1].registration_bytes == 0x10000
    assert groups[2].registration_address == 0x100000

    duplicate = described(
        "layer-0-key",
        0x41000,
        0x1000,
        allocation=0x10000,
        allocation_bytes=0x80000,
        registration=0x40000,
        registration_bytes=0x10000,
    )
    try:
        coalesce_hbm_destinations((packed_key, duplicate))
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("HBM planner accepted a duplicate destination key")

    conflicting = described(
        "conflicting-allocation",
        0x41000,
        0x1000,
        allocation=0x10000,
        allocation_bytes=0x90000,
        registration=0x40000,
        registration_bytes=0x10000,
    )
    try:
        coalesce_hbm_destinations((packed_key, conflicting))
    except ValueError as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("HBM planner accepted conflicting allocation geometry")

    try:
        described(
            "outside-envelope",
            0x25000,
            0x1000,
            allocation=0x10000,
            allocation_bytes=0x80000,
            registration=0x10000,
            registration_bytes=0x10000,
        )
    except ValueError as error:
        assert "does not contain" in str(error)
    else:
        raise AssertionError("HBM planner accepted a slice outside its envelope")

    print("hbm_registration=pass")


if __name__ == "__main__":
    main()
