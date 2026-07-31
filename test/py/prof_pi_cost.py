"""prof_pi_cost.py -- isolate the pi-ORDERING cost on the real woven decode.

Runs the SAME dispersed decode batch under identity vs reversed task_order and
launches each several times so ncu can compare L2 hit rate, DRAM bytes,
coalescing (sectors/request), achieved occupancy, and duration. If reversed
(scattered tile order) shows lower L2 hit / worse coalescing than identity, the
permutation itself costs cache locality -- the missing piece between the
kernel-level -35% (high dispersion) and the E2E enforce<observe gap.

Launch sequence (ncu profiles in order): 5 warmup(identity), 3 identity, 3
reversed -> in the CSV the LAST 3 decode rows are reversed, earlier are identity.

Run under ncu:  see the sudo ncu command in the shell harness.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

# DEFAULT to the real GQA serving shape (Qwen-3B: 16 q-heads, 2 kv-heads,
# head_dim 128) -> the 2D decode grid the earlier nkv=1 microbench MISSED.
# Override via env to compare shapes.
PAGE, HD = 16, 128
NQO = int(os.environ.get("PROF_NQO", "16"))
NKV = int(os.environ.get("PROF_NKV", "2"))
BS = int(os.environ.get("PROF_BS", "2048"))


def main():
    torch.cuda.init()
    plane = SchedPlane(max_tasks=16384, device="cuda")
    plane.use_device_timer()
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    import flashinfer

    gen = torch.Generator().manual_seed(42)
    pages = torch.exp(torch.normal(2.5, 1.0, (BS,), generator=gen)) \
        .clamp(1, 128).to(torch.int32)
    npages = int(pages.sum())
    indptr = torch.zeros(BS + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    indptr = indptr.cuda()
    indices = torch.arange(npages, dtype=torch.int32, device="cuda")
    last = torch.full((BS,), PAGE, dtype=torch.int32, device="cuda")
    q = torch.randn(BS, NQO, HD, dtype=torch.float16, device="cuda")
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device="cuda")

    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    print(f"[prof] shape nqo={NQO} nkv={NKV} (GQA group {NQO // NKV})",
          flush=True)
    v = getattr(w, "_plan_info", None)
    v = v.tolist() if hasattr(v, "tolist") else list(v)
    ntiles = int(v[0]) if (len(v) == 10 and int(v[0]) >= BS) else BS

    plane.set_num_tasks(0)
    plane.set_timer_enabled(False)  # isolate ORDERING; no timer atomics
    ident = torch.arange(plane.N, dtype=torch.int32)
    rev = torch.arange(plane.N, dtype=torch.int32)
    rev[:ntiles] = torch.arange(ntiles - 1, -1, -1, dtype=torch.int32)

    def run(order, label, reps):
        full = ident.clone()
        full[:ntiles] = order[:ntiles]
        plane.install_order(full, ntiles)
        plane.push()
        torch.cuda.synchronize()
        for _ in range(reps):
            out = w.run(q, kv)
        torch.cuda.synchronize()
        print(f"[prof] {label} x{reps} (ntiles={ntiles})", flush=True)
        return out

    o_id = run(ident, "warmup-identity", 5)
    run(ident, "PROF-identity", 3)
    o_rev = run(rev, "PROF-reversed", 3)
    print(f"[prof] bit-exact(identity==reversed)={torch.equal(o_id, o_rev)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
