"""Authoritative request ownership projection for one SGLang forward.

The plugin captures a transient request-to-load edge before HiCache leases the
physical queue.  After that lease exists, only its immutable operation demand
records may define numerical ownership.  Keeping this join outside the hook
module prevents framework view construction from becoming a second lifetime
owner.
"""

from __future__ import annotations

from typing import Any

from nta_runtime.adapters.sglang import SglangAcquisitionSpan
from nta_runtime.engines.sglang_hicache import find_bridge
from nta_runtime.engines.sglang_topology import project_forward_operation_owners


_LEASE_ID_ATTRIBUTE = "_nta_hicache_lease_id"


def lease_forward_acquisitions(
    batch: Any,
    request_ids: tuple[str, ...],
    request_slots: tuple[int, ...],
) -> tuple[SglangAcquisitionSpan, ...] | None:
    """Rebuild a sidecar from the live lease's immutable ownership.

    ``None`` means no bridge-backed lease has been assigned yet, so the caller
    may use the one-shot request attribute captured at ``add_one_req``.  Once a
    bridge exists, a missing indexed owner is an invariant failure; silently
    returning to the transient attribute could bind stale or incomplete demand
    after chunking or admission staging.
    """

    consumer_index = int(getattr(batch, "hicache_consumer_index", -1))
    if consumer_index < 0:
        return None

    tree_cache = getattr(batch, "tree_cache", None)
    bridge = None
    for _ in range(3):
        controller = getattr(tree_cache, "cache_controller", None)
        device_pool = getattr(controller, "mem_pool_device", None)
        if device_pool is not None:
            bridge = find_bridge(device_pool)
            break
        tree_cache = getattr(tree_cache, "inner", None)
        if tree_cache is None:
            break
    if bridge is None:
        return None
    pending = bridge.get(consumer_index)
    bound_lease_id = getattr(batch, _LEASE_ID_ATTRIBUTE, None)
    if bound_lease_id is not None:
        bound_lease_id = int(bound_lease_id)
        if bound_lease_id <= 0:
            raise RuntimeError("SGLang batch carries an invalid HiCache lease identity")
        # ScheduleBatch survives the external prefill and becomes the running
        # decode batch.  Its framework consumer index is not cleared when the
        # final prefill layer retires NTA's lease, and the finite producer ring
        # may subsequently reuse that index.  The monotone lease ID is the ABA
        # guard: a historical batch is resident/direct after its own lease
        # retires, regardless of what currently occupies the framework slot.
        if pending is None or pending.lease_id != bound_lease_id:
            return tuple(SglangAcquisitionSpan.direct() for _ in request_ids)
    elif pending is None:
        raise RuntimeError("SGLang forward names an unknown HiCache lease")
    else:
        setattr(batch, _LEASE_ID_ATTRIBUTE, int(pending.lease_id))

    transfers = pending.transfers_by_operation()
    lease_operation_ids = (
        pending.demand_operation_ids
        if pending.demand_operation_ids is not None
        else frozenset(transfers)
    )
    expected = project_forward_operation_owners(
        request_ids,
        request_slots,
        pending.operation_demands,
        pending.operation_requests,
        lease_operation_ids=lease_operation_ids,
    )
    demands = {demand.operation_id: demand for demand in pending.operation_demands}
    acquisitions: list[SglangAcquisitionSpan] = []
    for operation_id in expected:
        if operation_id is None:
            acquisitions.append(SglangAcquisitionSpan.direct())
            continue
        transfer = transfers.get(operation_id)
        demand = demands.get(operation_id)
        if (
            transfer is None
            or demand is None
            or transfer.row_count != demand.row_count
        ):
            raise RuntimeError(
                "SGLang live lease has inconsistent transfer/demand ownership"
            )
        acquisitions.append(
            SglangAcquisitionSpan(
                operation_id,
                transfer.node_id,
                demand.logical_begin,
                demand.row_count,
            )
        )
    return tuple(acquisitions)
