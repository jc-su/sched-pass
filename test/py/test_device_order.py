"""test_device_order.py -- prove the KERNEL-RESIDENT order table
(SCHED_ORDER_INDIRECT) is BIT-EXACT: the arena ORDER word is a retargetable
pointer to a DEVICE order tensor, the woven kernel loads it and derefs, and a
device-tensor permutation reorders CTA->tile with NO host sync and IDENTICAL
output (E1). Also checks the fail-safe (0 pointer -> identity) via the direct
disarm path. Compiles a NEW kernel variant (cache-tagged -oi).

Run:
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  SCHED_WEAVE_ONLY=batch_decode python test/py/test_device_order.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python"))
from sched_rt import SchedPlane


def main():
    torch.manual_seed(0)
    dev = "cuda"
    torch.cuda.init()

    plane = SchedPlane(max_tasks=1024, device=dev)
    plane.use_device_order()             # <-- kernel-resident order table
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    plane.set_num_tasks(plane.N)
    plane.push()

    import flashinfer
    bs, max_pages, page_size = 8, 8, 16
    nqo, nkv, hd = 8, 8, 128
    kv_indptr = torch.arange(0, bs * max_pages + 1, max_pages,
                             dtype=torch.int32, device=dev)
    kv_indices = torch.arange(bs * max_pages, dtype=torch.int32, device=dev)
    last_len = torch.full((bs,), page_size, dtype=torch.int32, device=dev)
    q = torch.randn(bs, nqo, hd, dtype=torch.float16, device=dev)
    kv = torch.randn(bs * max_pages, 2, page_size, nkv, hd,
                     dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(kv_indptr, kv_indices, last_len, nqo, nkv, hd, page_size,
           data_type=torch.float16, q_data_type=torch.float16)

    fails = [0]

    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails[0] += 0 if cond else 1

    print("== device-resident order (SCHED_ORDER_INDIRECT) bit-exact test ==")
    ok(plane._order_dev is not None, "order table is device-resident")

    plane.reset_order(); torch.cuda.synchronize()
    golden = w.run(q, kv).clone()

    # install a permutation that NEVER leaves the GPU (device tensor in ->
    # device->device copy into _order_dev, no host sync)
    order = torch.arange(plane.N, dtype=torch.int32, device=dev)  # ON DEVICE
    npair = bs - (bs % 2)
    order[:npair] = order[:npair].view(-1, 2).flip(1).reshape(-1)
    arena_before = bytes(plane.arena.read(0x00000, 64))  # ORDER region head
    plane.install_order(order, plane.N)
    arena_after = bytes(plane.arena.read(0x00000, 64))
    ok(order.is_cuda and bool((plane._order_dev[:bs] == order[:bs]).all()),
       "device-tensor order installed device->device (never left GPU)")
    ok(arena_before[8:] == arena_after[8:],
       "arena ORDER array untouched by install (only the device tensor changed)")
    torch.cuda.synchronize()
    permuted = w.run(q, kv).clone()
    ok(torch.equal(permuted, golden),
       "device-order swap-pairs permutation is BIT-EXACT vs identity (E1)")

    # reversed order over the real tiles -> still bit-exact
    rev = torch.arange(plane.N, dtype=torch.int32, device=dev)
    rev[:bs] = torch.arange(bs - 1, -1, -1, dtype=torch.int32, device=dev)
    plane.install_order(rev, plane.N)
    torch.cuda.synchronize()
    rev_out = w.run(q, kv).clone()
    ok(torch.equal(rev_out, golden),
       "device-order reversed permutation is BIT-EXACT vs identity (E1)")

    # fail-safe: back to identity
    plane.reset_order(); torch.cuda.synchronize()
    ident = w.run(q, kv).clone()
    ok(torch.equal(ident, golden), "reset to identity -> golden (fail-safe)")

    print("== PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    sys.exit(main())
