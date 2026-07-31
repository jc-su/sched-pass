"""test_dynamic_loop.py -- the end-to-end "GPU is not static" demonstration.

Proves the full runtime path the SGLang integration rides on, on a REAL
online-softmax paged-attention kernel, with NO Triton and NO engine:

  1. the Python control plane (SchedPlane) allocates the tables as CUDA tensors;
  2. their fixed device addresses are baked into the kernel by the clang JIT
     compile + the LLVM pass plugin (SCHED_BAKE_*), so the woven kernel reads
     Python-owned state -- this is the JIT+baked ABI that FlashInfer would use;
  3. between launches the plane reprograms behavior by WRITING DATA (pi order,
     tau budget) -- no recompile, no different code: the GPU stops being static;
  4. we verify: woven-neutral == stock (bit-exact), reorder is bit-exact, the
     woven timer attributes per-request cycles, and shed (tau) changes output to
     match a truncated-attention reference (the -inf softmax mask).

Run on the Blackwell node:
  SCHED_PLUGIN=~/Dev/sched-pass/build/libSchedPass.so \
  ~/miniconda3/bin/python test_dynamic_loop.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane, HINT_URGENT, HINT_POLITE

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.environ["SCHED_PLUGIN"]
SHIM = os.path.join(HERE, "..", "..", "python", "nvcc_clang_shim.py")
CLANG = os.environ.get("SCHED_CLANG", "clang++-22")
CUDA = os.environ.get("SCHED_CUDA_PATH", "/usr/local/cuda-12.9")
ARCH = os.environ.get("SCHED_ARCH", "sm_120")

NSEQ, SD, PT = 64, 32, 32
NBLK_LONG, NBLK_SHORT = 24, 3
NPAGES = NSEQ * NBLK_LONG
PAD = 4096


def is_long(t):
    return t % 8 == 0


def compile_kernel(so_path, env):
    """Compile paged_softmax.cu -> .so with clang (+ plugin if env bakes it)."""
    src = os.path.join(HERE, "..", "..", "python", "kernels", "paged_softmax.cu")
    cmd = [CLANG, "-x", "cuda", f"--cuda-gpu-arch={ARCH}", f"--cuda-path={CUDA}",
           "-O2", "-std=c++17", "-Wno-unknown-cuda-version", "-shared", "-fPIC",
           f"-L{CUDA}/lib64", "-lcudart", src, "-o", so_path]
    if env.get("SCHED_PLUGIN"):
        cmd.insert(1, f"-fpass-plugin={env['SCHED_PLUGIN']}")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"compile failed: {so_path}")
    lib = ctypes.CDLL(so_path)
    lib.launch_paged_softmax.restype = None
    lib.launch_paged_softmax.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return lib


def run(lib, kv, bt, nbl, q, out, bt_stride):
    out.zero_()
    lib.launch_paged_softmax(kv.data_ptr(), bt.data_ptr(), nbl.data_ptr(),
                             q.data_ptr(), out.data_ptr(), NSEQ, bt_stride, PT,
                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return out.clone()


def cpu_ref(kv, bt, nbl, q, tau=0):
    kvh, bth, nblh, qh = (x.cpu() for x in (kv, bt, nbl, q))
    out = torch.zeros(NSEQ, SD)
    for s in range(NSEQ):
        for lane in range(SD):
            m, l, acc = -1e30, 0.0, 0.0
            for b in range(int(nblh[s])):
                page = int(bth[s * NBLK_LONG + b])
                base = page * PT * 2 * SD
                for t in range(PT):
                    if tau and t >= tau:
                        continue
                    sc = 0.0
                    for j in range(SD):
                        sc += float(kvh[base + t * 2 * SD + j]) * float(qh[s * SD + j])
                    sc *= 0.125
                    mn = max(m, sc)
                    c = pow(2.718281828, m - mn)
                    w = pow(2.718281828, sc - mn)
                    l = l * c + w
                    acc = acc * c + w * float(kvh[base + t * 2 * SD + SD + lane])
                    m = mn
            out[s, lane] = acc / l
    return out.reshape(-1)


def make_inputs(dev="cuda"):
    """The fixture workload: hetero paged batch (kv, bt, nbl, q, out)."""
    torch.cuda.init()
    g = torch.Generator().manual_seed(1)
    kv = (torch.rand(NPAGES * PT * 2 * SD + PAD, generator=g) - 0.5).to(dev)
    q = (torch.rand(NSEQ * SD, generator=g) - 0.5).to(dev)
    bt = torch.empty(NSEQ * NBLK_LONG, dtype=torch.int32)
    nbl = torch.empty(NSEQ, dtype=torch.int32)
    for t in range(NSEQ):
        nbl[t] = NBLK_LONG if is_long(t) else NBLK_SHORT
        for b in range(NBLK_LONG):
            bt[t * NBLK_LONG + b] = (t * NBLK_LONG + b * 7 + 3) % NPAGES
    return kv, bt.to(dev), nbl.to(dev), q, torch.zeros(NSEQ * SD, device=dev)


def main():
    dev = "cuda"
    kv, bt, nbl, q, out = make_inputs(dev)

    fails = [0]
    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails[0] += 1

    print("== dynamic-loop demo: real online-softmax paged attention ==")

    # -- STOCK: compile WITHOUT the plugin; the untouched kernel ------------
    stock_so = os.path.join(tempfile.gettempdir(), "paged_stock.so")
    stock_env = {k: v for k, v in os.environ.items() if not k.startswith("SCHED_")}
    stock_env["PATH"] = os.environ["PATH"]
    stock = compile_kernel(stock_so, stock_env)
    golden = run(stock, kv, bt, nbl, q, out, NBLK_LONG)
    ref = cpu_ref(kv, bt, nbl, q).to(dev)
    ok(torch.allclose(golden, ref, atol=2e-2, rtol=2e-2),
       "stock kernel matches CPU softmax reference")

    # -- WOVEN: plane owns the tables; addresses baked into the compile ----
    # ALL levers woven together (pi indirection + policy + timer + shed).
    # Shed's score-mask counts by the loop's canonical IV with a dominance-
    # checked replacement (see SchedWeave), so co-weaving with the other
    # levers is safe; tau=0 (the default row) keeps it bit-exact below.
    plane = SchedPlane(max_tasks=NSEQ, device=dev)
    env = plane.bake_env(PLUGIN)
    env["PATH"] = os.environ["PATH"]
    woven_so = os.path.join(tempfile.gettempdir(), "paged_woven.so")
    woven = compile_kernel(woven_so, env)

    # neutral tables -> woven kernel must equal stock (bit-exact) -----------
    plane.set_num_tasks(NSEQ)
    plane.push()
    plane.clear_timer()
    o_neutral = run(woven, kv, bt, nbl, q, out, NBLK_LONG)
    ok(torch.equal(o_neutral, golden), "woven+neutral tables == stock (bit-exact)")
    cyc = plane.read_timer()
    ok(int((cyc > 0).sum()) == NSEQ, "woven timer: one row per request")

    # the timer sees the straggler: long requests cost more ----------------
    long_c = cyc[[t for t in range(NSEQ) if is_long(t)]].float().mean()
    short_c = cyc[[t for t in range(NSEQ) if not is_long(t)]].float().mean()
    ok(long_c > 2 * short_c,
       f"timer attributes cost: long {long_c:.0f} >> short {short_c:.0f} cyc")

    # reprogram pi WITHOUT recompiling -> bit-exact (order changes, not math)
    plane.set_order(torch.arange(NSEQ - 1, -1, -1))  # reversed
    o_rev = run(woven, kv, bt, nbl, q, out, NBLK_LONG)
    ok(torch.equal(o_rev, golden), "reprogrammed pi (reversed) -> bit-exact")

    # LPT order from measured cycles (the closed loop) ---------------------
    order = torch.argsort(cyc, descending=True).to(torch.int32)
    plane.set_order(order)
    o_lpt = run(woven, kv, bt, nbl, q, out, NBLK_LONG)
    ok(torch.equal(o_lpt, golden), "LPT pi from measured cycles -> bit-exact")
    plane.reset_order()

    # SHED step, same multi-lever build: tau>0 must match the truncated-
    # attention reference (-inf softmax score mask) on the JIT kernel too.
    TAU = 8
    plane.reset_order()
    plane.set_rows(tau=[TAU] * NSEQ, n=NSEQ)
    plane.push()
    o_shed = run(woven, kv, bt, nbl, q, out, NBLK_LONG)
    ref_shed = cpu_ref(kv, bt, nbl, q, tau=TAU).to(dev)
    ok(torch.allclose(o_shed, ref_shed, atol=2e-2, rtol=2e-2),
       "shed tau>0 on the multi-lever JIT kernel matches TRUNCATED-attention "
       "CPU ref (-inf mask)")
    plane.set_rows(tau=[0] * NSEQ, n=NSEQ)
    plane.push()

    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
