#!/usr/bin/env python3
"""Writeback summary store invariants: alignment, union bound, recycling."""

import torch

from nta_runtime.engines.summary_store import WritebackSummaryStore


class FakePool:
    def __init__(self, rows, heads, dim, layers):
        self.buffers = [
            torch.randn(rows, heads, dim, dtype=torch.float16)
            for _ in range(layers)
        ]

    def _get_key_buffer(self, layer_id):
        return self.buffers[layer_id]


class FakeHostPool:
    def __init__(self, pool, layers):
        self.k_data_refs = [pool.buffers[layer] for layer in range(layers)]


PAGE = 16
LAYERS = 3
store = WritebackSummaryStore(PAGE, 64 << 20, device="cpu")
pool = FakePool(4096, 2, 8, LAYERS)
host_pool = FakeHostPool(pool, LAYERS)
layer_ids = tuple(range(LAYERS))

# One aligned node of 4 pages; rows are arbitrary pool rows.
rows_a = torch.arange(100, 100 + 4 * PAGE, dtype=torch.int64)
store.record(1, rows_a, rows_a, pool, layer_ids)
assert store.recorded_nodes == 1

# Gather with the exact grid: envelopes must equal the direct reduction.
gathered = store.gather((), rows_a, LAYERS, 4, 4 * PAGE)
assert gathered is not None
kmin, kmax = gathered
for layer in range(LAYERS):
    direct = pool.buffers[layer][rows_a].view(4, PAGE, 2, 8)
    assert torch.equal(kmin[layer], direct.amin(dim=1).to(torch.float32))
    assert torch.equal(kmax[layer], direct.amax(dim=1).to(torch.float32))

# Phase-shifted claim (starts 5 rows in): union bound must contain the
# true envelope of every full page.
shifted = rows_a[5 : 5 + 3 * PAGE]
gathered = store.gather((), shifted, LAYERS, 3, 3 * PAGE)
assert gathered is not None
smin, smax = gathered
for layer in range(LAYERS):
    true = pool.buffers[layer][shifted].view(3, PAGE, 2, 8).to(torch.float32)
    assert bool((smin[layer] <= true.amin(dim=1) + 1e-6).all())
    assert bool((smax[layer] >= true.amax(dim=1) - 1e-6).all())

# Chunk-boundary stitching: second chunk at offset 24 uses ancestor rows.
rows_b = torch.arange(600, 600 + 40, dtype=torch.int64)
store.record(2, rows_b[:24], rows_b[:24], pool, layer_ids)
store.record(
    3,
    rows_b[24:],
    rows_b[24:],
    pool,
    layer_ids,
    ancestor_host_rows=[rows_b[:24]],
    host_pool=host_pool,
)
gathered = store.gather((), rows_b[:32], LAYERS, 2, 32)
assert gathered is not None, store.miss_reasons
bmin, _ = gathered
direct = (
    pool.buffers[0][rows_b[:32]].view(2, PAGE, 2, 8).amin(dim=1).to(torch.float32)
)
assert torch.equal(bmin[0], direct)

# A row never recorded fails closed.
missing = torch.arange(3000, 3000 + PAGE, dtype=torch.int64)
assert store.gather((), missing, LAYERS, 1, PAGE) is None

# Re-recording a node under new rows retires the old mapping.
rows_c = torch.arange(2000, 2000 + PAGE, dtype=torch.int64)
store.record(1, rows_c, rows_c, pool, layer_ids)
assert store.gather((), rows_a, LAYERS, 4, 4 * PAGE) is None

# Eviction under a tiny budget frees slots and keeps gathers consistent.
small = WritebackSummaryStore(PAGE, 2_000, device="cpu")
for index in range(8):
    rows = torch.arange(index * PAGE, (index + 1) * PAGE, dtype=torch.int64)
    small.record(10 + index, rows, rows, pool, layer_ids)
assert small.evicted_nodes > 0
newest = torch.arange(7 * PAGE, 8 * PAGE, dtype=torch.int64)
assert small.gather((), newest, LAYERS, 1, PAGE) is not None

print("writeback summary store invariants hold")
