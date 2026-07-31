#!/usr/bin/env python3
"""nvcc -> clang++ CUDA shim so an LLVM pass plugin can weave JIT-compiled
CUDA (FlashInfer's JIT reads $FLASHINFER_NVCC for its compiler).

nvcc and clang-CUDA take different flags; this translates the nvcc-style argv
that FlashInfer's ninja rule emits into an equivalent clang++ invocation and
injects `-fpass-plugin=<SchedPass.so>`. The pass reads SCHED_BAKE_* from the
environment (inherited here) to bake the control-plane buffer addresses.

Config via env:
  SCHED_CLANG        clang++ to use            (default: clang++-22)
  SCHED_PLUGIN       path to libSchedPass.so   (required to weave)
  SCHED_CUDA_PATH    --cuda-path               (default: /usr/local/cuda-12.9)
  SCHED_STRIP_ARCH   strip arch suffix for clang <= 21 (sm_120a -> sm_120)
  SCHED_SHIM_LOG     if set, append translated commands here
  SCHED_WEAVE_ONLY   comma list of substrings; compiles whose argv does NOT
                     match any are NOT WOVEN: routed to SCHED_REAL_NVCC if
                     set (and existing), else compiled by clang WITHOUT the
                     pass plugin. Two birds: confines the dialect surface,
                     and keeps non-attention kernels (rope, norm, sampling)
                     from being woven against the same control tables --
                     their CTAs would pollute the timer's per-tile rows.
                     Unset = weave everything (the census mode).
  SCHED_REAL_NVCC    real nvcc for routed compiles. No default: only routes
                     when explicitly set (a broken system nvcc must not be
                     picked up implicitly -- glibc 2.41 vs CUDA<=12.9
                     headers, and partial CUDA-13 crt/ installs, both exist
                     in the wild).
Everything SCHED_BAKE_* / SCHED_* the pass needs is passed through untouched.
"""
import os
import re
import subprocess
import sys


def clang_arch(nvcc_code):
    """nvcc code target -> clang --cuda-gpu-arch. clang >= 22 accepts the
    arch-conditional 'a' suffix (sm_120a) but NOT the family 'f' suffix
    (sm_120f), which CUDA 13 + FlashInfer emit for SM 12.x. Map 'f' -> 'a'
    (same Blackwell SM, arch-conditional PTX clang supports).
    SCHED_STRIP_ARCH=1 strips the suffix entirely for clang <= 21 (plain sm_NN)."""
    if os.environ.get("SCHED_STRIP_ARCH"):
        m = re.match(r"(sm_\d+)", nvcc_code)
        return m.group(1) if m else nvcc_code
    m = re.match(r"(sm_\d+)f$", nvcc_code)
    return (m.group(1) + "a") if m else nvcc_code

CLANG = os.environ.get("SCHED_CLANG", "clang++-22")
PLUGIN = os.environ.get("SCHED_PLUGIN", "")
CUDA_PATH = os.environ.get("SCHED_CUDA_PATH", "/usr/local/cuda-12.9")
LOG = os.environ.get("SCHED_SHIM_LOG", "")


PRELUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "clang_cuda_prelude.h")


def translate(argv, weave=True):
    out = [CLANG, "-x", "cuda", f"--cuda-path={CUDA_PATH}",
           "-Wno-unknown-cuda-version"]
    # Force the CUDA toolkit's own headers ahead of anything on the default
    # system path. (Historically load-bearing: a stale distro nvidia-cuda-dev
    # left CUDA-11.x headers in /usr/include that clang picked up first. Those
    # packages are removed from the node; this stays as cheap insurance so a
    # reinstalled distro package can never silently shadow the toolkit again.)
    inc = os.path.join(CUDA_PATH, "include")
    if os.path.isdir(inc):
        out += ["-isystem", inc]
    # Force-include the nvcc->clang dialect prelude (global min/max) before any
    # FlashInfer header is parsed.
    if os.path.exists(PRELUDE):
        out += ["-include", PRELUDE]
    # CRITICAL (bisected 2026-07-09): FlashInfer + CUTLASS gate their inline-PTX
    # FAST PATH -- cp.async, ldmatrix/stmatrix, mma.sync -- behind
    # `#if __CUDACC_VER_MAJOR__ >= 11`, an nvcc-ONLY macro clang does not define.
    # Without it clang silently compiles the SCALAR fallback (mismatched
    # barriers -> race -> illegal access; the whole prefill "codegen wall").
    # The fast path is compiler-agnostic INLINE ASM, so defining the macro (to
    # match the toolkit) makes clang emit the SAME cp.async/ldmatrix nvcc does.
    m = re.search(r"cuda-(\d+)\.(\d+)", CUDA_PATH)
    maj, mnr = (m.group(1), m.group(2)) if m else ("12", "0")
    out += [f"-D__CUDACC_VER_MAJOR__={maj}", f"-D__CUDACC_VER_MINOR__={mnr}",
            "-D__CUDACC_VER_BUILD__=0"]
    if PLUGIN and weave:
        out.append(f"-fpass-plugin={PLUGIN}")
    have_std = False
    i = 0
    while i < len(argv):
        a = argv[i]
        # arch: -gencode=arch=compute_120a,code=sm_120a -> --cuda-gpu-arch=sm_120
        if a.startswith("-gencode="):
            m = re.search(r",code=([A-Za-z0-9_]+)", a)
            if m:
                out.append(f"--cuda-gpu-arch={clang_arch(m.group(1))}")
            i += 1
            continue
        if a == "-arch" and i + 1 < len(argv):
            out.append(f"--cuda-gpu-arch={clang_arch(argv[i+1])}")
            i += 2
            continue
        # host-compiler passthrough: --compiler-options=X / -Xcompiler X -> X
        if a.startswith("--compiler-options="):
            out.append(a.split("=", 1)[1])
            i += 1
            continue
        if a in ("-Xcompiler", "--compiler-options") and i + 1 < len(argv):
            out.append(argv[i + 1])
            i += 2
            continue
        # -Xfatbin / --fatbin-options: nvcc fatbin flags clang lacks -> drop.
        if a.startswith("-Xfatbin") or a.startswith("--fatbin-options"):
            if a in ("-Xfatbin", "--fatbin-options") and i + 1 < len(argv):
                i += 2
            else:
                i += 1
            continue
        # -Xptxas X / -Xptxas=X -> clang's -Xcuda-ptxas X.
        if a == "-Xptxas" and i + 1 < len(argv):
            out += ["-Xcuda-ptxas", argv[i + 1]]
            i += 2
            continue
        if a.startswith("-Xptxas="):
            out += ["-Xcuda-ptxas", a.split("=", 1)[1]]
            i += 1
            continue
        # deps: nvcc --generate-dependencies-with-compile [--dependency-output F]
        if a == "--generate-dependencies-with-compile":
            i += 1
            continue
        if a == "--dependency-output" and i + 1 < len(argv):
            out += ["-MMD", "-MF", argv[i + 1]]
            i += 2
            continue
        # nvcc-only flags clang doesn't understand -> drop
        if a in ("--expt-relaxed-constexpr", "--expt-extended-lambda",
                 "-static-global-template-stub=false",
                 "-static-global-template-stub=true",
                 "--use_fast_math", "-use_fast_math",
                 "-forward-unknown-to-host-compiler",
                 "--generate-line-info", "-lineinfo"):
            i += 1
            continue
        if a.startswith("--threads") or a.startswith("-static-global-template-stub"):
            # --threads=N (fused) drop; --threads N handled below
            if a == "--threads" and i + 1 < len(argv):
                i += 2
            else:
                i += 1
            continue
        if a == "-ccbin" and i + 1 < len(argv):
            i += 2  # clang is its own host driver
            continue
        if a.startswith("-ccbin="):
            i += 1
            continue
        if a.startswith("-std="):
            have_std = True
        out.append(a)
        i += 1
    if not have_std:
        out.append("-std=c++17")
    out.append("-fPIC")
    return out


def main():
    # Version probe: JIT frameworks (sglang jit_kernel, tvm-ffi) run
    # `nvcc --version` and PARSE the banner for the CUDA release; clang's
    # banner does not match and stalls their toolkit detection. Impersonate
    # nvcc, deriving the release from the toolkit we compile against.
    if "--version" in sys.argv[1:]:
        m = re.search(r"cuda-(\d+\.\d+)", CUDA_PATH)
        rel = m.group(1) if m else "12.9"
        print("nvcc: NVIDIA (R) Cuda compiler driver (sched-pass clang shim)")
        print(f"Cuda compilation tools, release {rel}, V{rel}.0")
        print(f"Build cuda_{rel}.r{rel}/shim.0")
        return 0

    # INVARIANT: baked addresses only ever compile into a cache KEYED BY THAT
    # BASE ADDRESS. If SCHED_BAKE_* is set but the FlashInfer workspace path
    # does not carry the baked base's va-hex (import-order bug, stray env),
    # STRIP the bake vars for this compile -- a baked kernel in an unkeyed
    # cache is a landmine: any later process without the arena dereferences
    # the unmapped VA (measured: cudaErrorIllegalAddress at the first decode
    # plan in stock mode). The check is the ADDRESS itself, not a path name:
    # custom SCHED_CACHE_ROOTs are legitimate (the CLC A/B test uses one).
    order_addr = os.environ.get("SCHED_BAKE_TASK_ORDER", "")
    if order_addr.isdigit():
        va_tag = f"va{int(order_addr):x}"
        if va_tag not in os.environ.get("FLASHINFER_WORKSPACE_BASE", ""):
            for k in [k for k in os.environ if k.startswith("SCHED_BAKE_")]:
                del os.environ[k]
            sys.stderr.write(f"[sched-shim] bake vars set but workspace is "
                             f"not keyed by {va_tag}: stripped (compiling "
                             f"unwoven; an unkeyed cache must never hold "
                             f"baked kernels)\n")

    # Routing: modules outside SCHED_WEAVE_ONLY are NOT woven -- real nvcc if
    # one is explicitly configured, else clang WITHOUT the pass plugin (keeps
    # rope/norm/sampling CTAs from writing into the decode control tables).
    weave = True
    only = os.environ.get("SCHED_WEAVE_ONLY", "")
    if only:
        hay = " ".join(sys.argv[1:])
        if not any(s.strip() and s.strip() in hay for s in only.split(",")):
            weave = False
            real = os.environ.get("SCHED_REAL_NVCC", "")
            if real and os.path.exists(real):
                # FlashInfer's ninja embeds `-isystem /usr/local/cuda/include`
                # (the unversioned symlink). If that resolves to a DIFFERENT
                # toolkit than `real`, nvcc's cudafe output clashes with the
                # foreign crt headers (__cudaLaunch macro arity, sinpi/cospi
                # exception specs under glibc 2.41). nvcc injects its own
                # toolkit includes anyway -- DROP the redundant pair.
                args, i = [], 0
                argv = sys.argv[1:]
                while i < len(argv):
                    if (argv[i] == "-isystem" and i + 1 < len(argv) and
                            argv[i + 1] == "/usr/local/cuda/include"):
                        i += 2
                        continue
                    args.append(argv[i])
                    i += 1
                if LOG:
                    with open(LOG, "a") as f:
                        f.write("ROUTE->" + real + " " + hay + "\n")
                os.execv(real, [real] + args)
    cmd = translate(sys.argv[1:], weave=weave)
    if LOG:
        with open(LOG, "a") as f:
            f.write(("" if weave else "NOWEAVE ") + " ".join(cmd) + "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
