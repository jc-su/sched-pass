"""eval_timer_overhead.py -- the ISOLATED timer overhead on the REAL GQA decode
kernel (device-L2 channel), and the amortized cost at each sampling cadence.
Clean microbench (fixed batch, no serving/radix-cache noise): measures the
woven decode step time with the timer gated OFF vs armed EVERY step, then the
1-in-K amortized overhead is arithmetic (each armed launch pays the same delta;
K-1 launches pay nothing). Answers "how close to eKV-free can we get, and at
what cadence".

Run (armed):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  python eval_timer_overhead.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD = 16, 128
NQO, NKV = 16, 2                       # Qwen-3B GQA serving shape (group 8)
BS = int(os.environ.get("BS", "2048"))
ITERS = 60


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer()           # device-L2 channel (the serving default)
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    import flashinfer

    gen = torch.Generator().manual_seed(42)
    pages = torch.exp(torch.normal(2.5, 1.0, (BS,), generator=gen)) \
        .clamp(1, 128).to(torch.int32)
    npages = int(pages.sum())
    indptr = torch.zeros(BS + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    indptr = indptr.cuda()
    indices = torch.arange(npages, dtype=torch.int32, device=dev)
    last = torch.full((BS,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(BS, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    plane.set_num_tasks(0); plane.reset_order(); plane.push()

    for _ in range(5):                 # warm / JIT
        w.run(q, kv)
    torch.cuda.synchronize()

    # timer OFF (gate) -- baseline woven cost
    plane.set_timer_enabled(False); plane.push()
    t_off = min(step_us(w, q, kv) for _ in range(3))
    # timer ARMED every step -- device-L2 atomic per CTA per step
    plane.set_timer_enabled(True); plane.push()
    t_on = min(step_us(w, q, kv) for _ in range(3))
    plane.set_timer_enabled(False); plane.push()

    delta = t_on - t_off
    print(f"== isolated timer overhead, REAL GQA decode "
          f"(nqo={NQO}/nkv={NKV}, {BS} tiles, device-L2 channel) ==")
    print(f"  timer OFF (gated)        {t_off:8.2f} us/step   baseline")
    print(f"  timer ON  (every step)   {t_on:8.2f} us/step   "
          f"{100*delta/t_off:+.1f}%   <- the every-step tax (the #1 bug's cost)")
    print(f"  --- amortized overhead at 1-in-K sampling (device-L2) ---")
    for k in (1, 8, 16, 32, 64):
        amort = delta / k
        print(f"    1-in-{k:<3d}  +{amort:6.2f} us/step   "
              f"{100*amort/t_off:+.2f}%")
    print(f"  (probe readback: one device->host copy per probe, host-side, "
          f"overlapped -- not on the kernel critical path)")
    print("== DONE ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
