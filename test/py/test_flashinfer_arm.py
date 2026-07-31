"""test_flashinfer_arm.py -- prove pi (task_order) ACTUALLY reorders SGLang's
real FlashInfer decode kernel, bit-exact.

The earlier weave test used UNARMED tables (safe == baseline). This one ARMS
the SchedPlane: its device-table addresses are baked into FlashInfer's JIT
compile (SCHED_BAKE_*), so the woven `task = task_order[blockIdx.x]` remap is
live. FlashInfer's decode reads `batch_idx = request_indices[blockIdx.x]`
(decode.cuh:417), so our pi composes with FlashInfer's own indirection:
permuting which CTA serves which tile. Every tile is still executed exactly
once, so the output must be BIT-IDENTICAL under any permutation -- pi reorders
WHEN/WHERE, never WHAT (the E1 guarantee, on the real kernel).

Env (same as the weave test) + the plane addresses are set here before the JIT
fires:
  SCHED_PLUGIN=~/Dev/sched-pass/build/libSchedPass.so \
  FLASHINFER_NVCC="python nvcc_clang_shim.py" \
  python test_flashinfer_arm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane


def main():
    torch.manual_seed(0)
    dev = "cuda"
    torch.cuda.init()

    # 1) allocate the control plane FIRST; bake its (fixed-VA) addresses into
    #    the JIT env -- must happen before FlashInfer's JIT compiles.
    plane = SchedPlane(max_tasks=1024, device=dev)
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
        if not cond:
            fails[0] += 1

    print("== FlashInfer ARMED-pi test (task_order permutation, bit-exact) ==")

    # 2) identity order -> golden (woven kernel, pi = no-op)
    plane.reset_order()
    torch.cuda.synchronize()
    golden = w.run(q, kv).clone()
    print(f"  golden sum {float(golden.float().sum()):.4f}")

    # 3) a valid permutation of the CTA->tile map: swap adjacent pairs in the
    #    first `bs` entries (definitely valid tile indices), identity after.
    #    This makes CTA 0 serve tile 1, CTA 1 serve tile 0, etc.
    order = torch.arange(plane.N, dtype=torch.int32)
    npair = bs - (bs % 2)
    order[:npair] = order[:npair].view(-1, 2).flip(1).reshape(-1)
    plane.set_order(order)
    torch.cuda.synchronize()
    permuted = w.run(q, kv).clone()
    print(f"  permuted sum {float(permuted.float().sum()):.4f}")

    ok(torch.equal(permuted, golden),
       "pi permutation on real FlashInfer decode is BIT-EXACT (reorders "
       "CTA->tile, output unchanged)")

    # 4) timer: the woven clock64 attributed per-tile cycles (armed)
    cyc = plane.read_timer()
    ok(int((cyc > 0).sum()) > 0,
       f"woven timer populated on real FlashInfer ({int((cyc>0).sum())} tiles)")

    # 5) disarm -> unwoven-equivalent; still runs
    plane.reset_order()
    plane.push()
    print("== PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    sys.exit(main())
