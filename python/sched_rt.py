"""sched_rt.py -- the Python control plane for the sched-pass baked ABI.

The runtime half of "the GPU is not static anymore": owns the control tables,
hands their device addresses to the compiler via SCHED_BAKE_* env (so a
JIT-compiled woven kernel reads them), and reprograms them between steps by
writing data -- no recompile, no relaunch of different code. Mirrors
runtime/sched_rt.h's ABI exactly (pinned by the constants below).

Address stability across processes (the JIT-cache contract)
-----------------------------------------------------------
Baked kernels embed table addresses at compile time, and FlashInfer caches
compiled kernels on disk. For a cached kernel to be valid in a LATER process,
the tables must live at the SAME virtual addresses. CUDA cannot promise that
for device memory (cuMemAddressReserve's fixed-address request is a hint the
driver is free to ignore -- and does, measured on driver 590). The OS can:
SchedArena mmap()s the tables at a canonical fixed VA (MAP_FIXED_NOREPLACE at
SCHED_VA_BASE) and registers the mapping with CUDA (cuMemHostRegister
DEVICEMAP); under unified addressing the device pointer EQUALS the host
pointer, so table addresses are stable across processes by OS guarantee.

Consequences, all deliberate:
  * pushes are plain memcpy into the mapping -- no CUDA calls on the step
    path; timer readback is a plain host read (the eKV zero-touch readout);
  * the kernel reads tables over PCIe once per CTA (tens of bytes) and the
    timer's atomicAdd is a system-scope PCIe atomic -- the measured, known
    cost of the observation lever;
  * correctness never depends on winning the canonical VA: bake_env() keys
    the FlashInfer workspace (its JIT cache root) by the ACTUAL base, so a
    canonical hit reuses the cache forever and a miss recompiles once into a
    different dir -- still correct, loudly noted. Unarmed kernels stay stock.

Usage:
    plane = SchedPlane(max_tasks=256)
    env = plane.bake_env(plugin=".../libSchedPass.so")  # for the JIT compile
    ... compile / import flashinfer under `env` ...
    plane.set_order([...])                              # per step: write pi
    plane.set_rows(q=..., tau=..., hint=...)            # per step: policy
    plane.push()                                        # publish (generation++)
    ... launch ...
    cyc = plane.read_timer()                            # per-task cycles
"""
from __future__ import annotations

import atexit
import ctypes
import os
import struct

import torch

# --- ABI (must match runtime/sched_rt.h / lib/SchedUtil.h) -----------------
CTRL_NUM_TASKS_OFF = 4
CTRL_LAMBDA_OFF = 8      # f32[4]
CTRL_FLAGS_OFF = 24      # u32 flags (was reserved sentinel_key; offset pinned)
CTRL_ORDER_SIZE_OFF = 28  # u32: tile count the installed pi is FOR (0=uncheck)
CTRL_ROWS_OFF = 32
FLAG_TIMER_OFF = 1       # ctrl.flags bit0: suppress the woven timer this step
ROW_SIZE = 8             # {f32 q, u16 tau, u8 hint, u8 pad}
ROW_Q_OFF = 0
ROW_TAU_OFF = 4
ROW_HINT_OFF = 6

HINT_AUTO, HINT_URGENT, HINT_POLITE = 0, 1, 2

# --- capability manifest MIRROR of include/sched/SchedManifest.h -------------
# The single cross-language source of truth for each woven instrument; the
# C++ manifest is authoritative and test/py/test_manifest.py asserts this
# mirror matches its dump ROW-FOR-ROW (name, effect, min_sm, knob, disable
# sense, cache tag, AND list order == the C++ `order` field). List order IS
# the pass emit order. Fields:
#   name, effect, min_sm, compile knob, disable_knob (SCHED_NO_* if True), tag
MANIFEST = [
    ("workqueue", "acquire",    0,  "SCHED_WORKQUEUE",   False, "-wq"),
    ("pi",        "E1/permute", 0,  "SCHED_NO_INDIRECT", True,  ""),
    ("shed",      "E2/budget",  0,  "SCHED_NO_SHED",     True,  "-nos"),
    ("policy",    "E0/hint",    80, "SCHED_NO_POLICY",   True,  "-nop"),
    ("timer",     "O/observe",  0,  "SCHED_NO_TIMER",    True,  "-ti"),
    ("pdl",       "E0/hint",    90, "SCHED_PDL",         False, ""),
]

# --- arena layout (stable offsets; addresses = base + offset) ---------------
# 256 KiB per table window keeps every table's start 64-bit aligned and leaves
# generous headroom; one 2 MiB granule holds the whole plane.
ORDER_OFF = 0x00000      # i32[max_tasks]
CTRL_OFF = 0x40000       # SchedCtrl struct (32 B header + 8 B rows)
TIMER_OFF = 0x80000      # u64[max_tasks]
QUEUE_OFF = 0xC0000      # i32 ticket counter
ARENA_BYTES = 0x100000   # 1 MiB used; reservation rounds up to granularity
MAX_TASKS_LIMIT = 16384  # keeps every table inside its 256 KiB window

DEFAULT_VA_BASE = 0x5C00_0000_0000  # quiet region between heap and mmap space

_CUDA_SUCCESS = 0
_MAP_FIXED_NOREPLACE = 0x100000  # Linux
_MAP_ANON_PRIVATE = 0x22         # MAP_PRIVATE | MAP_ANONYMOUS
_PROT_RW = 0x3


class SchedArenaError(RuntimeError):
    """An OS or CUDA call underlying the arena failed."""


def compute_bake_env(base: int, max_tasks: int, plugin: str, base_env,
                     timer_indirect: bool = False,
                     order_indirect: bool = False) -> dict:
    """The woven-JIT compile environment for a plane at `base`. Pure function
    of (base, N, mode): callable BEFORE the arena exists with the PREDICTED
    canonical base -- required because FlashInfer freezes its workspace dir
    at import time (flashinfer/jit/env.py module constant), long before the
    first batch can create the plane. The cache key includes every
    compile-time input that changes the woven code: base VA + table capacity
    + work-queue mode (structurally different driver) + timer channel layout."""
    env = dict(base_env)
    env["SCHED_PLUGIN"] = plugin
    env["SCHED_BAKE_TASK_ORDER"] = str(base + ORDER_OFF)
    env["SCHED_BAKE_CTRL"] = str(base + CTRL_OFF)
    env["SCHED_BAKE_TIMER"] = str(base + TIMER_OFF)
    env["SCHED_BAKE_QUEUE"] = str(base + QUEUE_OFF)
    env["SCHED_MAX_TASKS"] = str(max_tasks)
    if timer_indirect:
        env["SCHED_TIMER_INDIRECT"] = "1"
    if order_indirect:
        env["SCHED_ORDER_INDIRECT"] = "1"
    root = env.get("SCHED_CACHE_ROOT",
                   os.path.expanduser("~/.cache/flashinfer-sched"))
    mode = ""
    # work-queue and timer channel are VALUE-variants (not on/off), kept
    # explicit; the acquisition driver and timer layout are structurally
    # different codegen.
    if env.get("SCHED_WORKQUEUE"):
        mode = "-wqclc" if env.get("SCHED_CLC") else "-wqtk"
    if timer_indirect or env.get("SCHED_TIMER_INDIRECT"):
        mode += "-ti"
    if order_indirect or env.get("SCHED_ORDER_INDIRECT"):
        mode += "-oi"
    # Lever-disable knobs change the woven code too: without them in the key a
    # SCHED_NO_* bisect build would poison the full-weave cache. DERIVED from
    # the manifest (the disable-knob rows) -- so a new lever added to MANIFEST
    # keys the cache automatically, with no second list to keep in sync. Mask
    # letter = the knob's distinguishing word (SCHED_NO_<WORD>), manifest order.
    mask = "".join(knob[len("SCHED_NO_"):][0].lower()
                   for _n, _e, _sm, knob, disable, _tag in MANIFEST
                   if disable and env.get(knob))
    if mask:
        mode += f"-no{mask}"
    # WEAVE-VERSION tag: the pass BINARY is a compile-time input -- without
    # it, kernels woven by an older plugin are reused forever and shipped
    # compiler fixes silently never take effect (bit us: a crash fix was
    # 'deployed' but the cached kernel predated it).
    mode += f"-w{_weave_tag(plugin)}"
    env["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
        root, f"va{base:x}-n{max_tasks}{mode}")
    return env


def _weave_tag(plugin: str) -> str:
    """Short content hash of the pass plugin (cached per path+mtime)."""
    try:
        st = os.stat(plugin)
        key = (plugin, st.st_mtime_ns, st.st_size)
        if _weave_tag._memo and _weave_tag._memo[0] == key:
            return _weave_tag._memo[1]
        import hashlib
        h = hashlib.md5()
        with open(plugin, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        tag = h.hexdigest()[:8]
        _weave_tag._memo = (key, tag)
        return tag
    except OSError:
        return "none"


_weave_tag._memo = None


def predicted_base() -> int:
    """The canonical VA the arena will request (SCHED_VA_BASE override or the
    default). MAP_FIXED_NOREPLACE virtually always gets it in a fresh
    process; the plane VERIFIES on creation and re-arms with the actual base
    if not (correct-or-recompile, never wrong)."""
    return int(os.environ.get("SCHED_VA_BASE", "0"), 0) or DEFAULT_VA_BASE


class SchedArena:
    """The control-plane tables in one host mapping at a fixed canonical VA,
    device-visible via cuMemHostRegister(DEVICEMAP) + unified addressing.

    One arena per process (the canonical VA maps once; the control plane is
    one-per-process). A second construction raises.
    """

    _instance: "SchedArena | None" = None

    def __init__(self, device_ordinal: int = 0, va_base: int | None = None):
        if SchedArena._instance is not None:
            raise SchedArenaError(
                "SchedArena already exists in this process; the control plane "
                "is one-per-process (reuse the existing SchedPlane)")
        requested = int(os.environ.get("SCHED_VA_BASE", "0"), 0) or \
            (va_base if va_base is not None else DEFAULT_VA_BASE)
        self.size = ARENA_BYTES

        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self._libc = libc

        base = libc.mmap(ctypes.c_void_p(requested), self.size, _PROT_RW,
                         _MAP_ANON_PRIVATE | _MAP_FIXED_NOREPLACE, -1, 0)
        if base in (None, ctypes.c_void_p(-1).value) or base != requested:
            if base not in (None, ctypes.c_void_p(-1).value):
                libc.munmap(ctypes.c_void_p(base), self.size)
            base = libc.mmap(ctypes.c_void_p(0), self.size, _PROT_RW,
                             _MAP_ANON_PRIVATE, -1, 0)
            if base in (None, ctypes.c_void_p(-1).value):
                raise SchedArenaError("mmap failed for the arena")
            print(f"[sched] WARNING: canonical VA 0x{requested:x} unavailable "
                  f"(arena at 0x{base:x}); the JIT cache for this base is "
                  f"separate -- kernels recompile once, correctness unaffected")
        self.base: int = base
        self.canonical: bool = (self.base == requested)

        # Register with CUDA: device-visible, and UVA makes devptr == hostptr.
        self.cu = ctypes.CDLL("libcuda.so.1")
        self._check("cuInit", self.cu.cuInit(0))
        self._ensure_context(device_ordinal)
        self._check("cuMemHostRegister", self.cu.cuMemHostRegister_v2(
            ctypes.c_void_p(self.base), ctypes.c_size_t(self.size),
            ctypes.c_uint(0x02)))  # CU_MEMHOSTREGISTER_DEVICEMAP
        dptr = ctypes.c_void_p()
        self._check("cuMemHostGetDevicePointer",
                    self.cu.cuMemHostGetDevicePointer_v2(
                        ctypes.byref(dptr), ctypes.c_void_p(self.base), 0))
        if dptr.value != self.base:
            raise SchedArenaError(
                f"unified addressing violated: device 0x{dptr.value:x} != "
                f"host 0x{self.base:x} -- baked addresses would be wrong")
        ctypes.memset(ctypes.c_void_p(self.base), 0, self.size)
        SchedArena._instance = self
        atexit.register(self.close)

    def _check(self, name: str, rc: int) -> None:
        if rc != _CUDA_SUCCESS:
            msg = ctypes.c_char_p()
            self.cu.cuGetErrorName(rc, ctypes.byref(msg))
            err = msg.value.decode() if msg.value else f"code {rc}"
            raise SchedArenaError(f"{name} failed: {err}")

    def _ensure_context(self, device_ordinal: int) -> None:
        """Use the current context (torch's, usually); create the device's
        primary context for standalone use."""
        ctx = ctypes.c_void_p()
        self._check("cuCtxGetCurrent",
                    self.cu.cuCtxGetCurrent(ctypes.byref(ctx)))
        if ctx.value:
            return
        dev = ctypes.c_int()
        self._check("cuDeviceGet",
                    self.cu.cuDeviceGet(ctypes.byref(dev), device_ordinal))
        pctx = ctypes.c_void_p()
        self._check("cuDevicePrimaryCtxRetain",
                    self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), dev))
        self._check("cuCtxSetCurrent", self.cu.cuCtxSetCurrent(pctx))

    # -- raw ops: the mapping is host memory, so these are plain memcpy ------
    def write(self, offset: int, data: bytes) -> None:
        ctypes.memmove(ctypes.c_void_p(self.base + offset), data, len(data))

    def read(self, offset: int, nbytes: int) -> bytes:
        return ctypes.string_at(ctypes.c_void_p(self.base + offset), nbytes)

    def memset(self, offset: int, value: int, nbytes: int) -> None:
        ctypes.memset(ctypes.c_void_p(self.base + offset), value, nbytes)

    def close(self) -> None:
        if self.base and SchedArena._instance is self:
            self.cu.cuMemHostUnregister(ctypes.c_void_p(self.base))
            self._libc.munmap(ctypes.c_void_p(self.base),
                              ctypes.c_size_t(self.size))
            self.base = 0
            SchedArena._instance = None


class SchedPlane:
    """The per-step scheduling control plane over a SchedArena."""

    def __init__(self, max_tasks: int = 4096, device: str = "cuda",
                 va_base: int | None = None):
        if max_tasks > MAX_TASKS_LIMIT:
            raise ValueError(f"max_tasks {max_tasks} > {MAX_TASKS_LIMIT} "
                             f"(the arena's per-table window)")
        self.N = max_tasks
        self.dev = device
        torch.cuda.init()  # torch owns the primary context when present
        ordinal = torch.cuda.current_device() if device.startswith("cuda") else 0
        self.arena = SchedArena(device_ordinal=ordinal, va_base=va_base)
        self.ctrl_bytes = CTRL_ROWS_OFF + ROW_SIZE * self.N
        self._ctrl_host = bytearray(self.ctrl_bytes)
        self._generation = 0
        self._timer_dev = None      # device row buffer (indirect channel)
        self._timer_stream = None   # side stream for backlog-free readback
        self._order_dev = None      # device order table (kernel-resident pi)
        self.reset_order()
        self.push()

    # -- addresses (baked into the kernel at compile time) ------------------
    def bake_env(self, plugin: str, base_env: dict | None = None) -> dict:
        """Environment for the woven JIT compile. Also pins the FlashInfer
        cache dir to this arena's base+layout, so cached kernels and baked
        addresses can never disagree (SCHED_CACHE_ROOT overrides the root)."""
        return compute_bake_env(self.arena.base, self.N, plugin,
                                base_env or os.environ,
                                timer_indirect=self._timer_dev is not None,
                                order_indirect=self._order_dev is not None)

    def use_device_timer(self) -> None:
        """Switch the observation channel to DEVICE memory (the E1 verdict:
        device atomics are ~free, -0.5% vs +26.5% for host-mapped PCIe; only
        the READOUT then costs a small D2H per probe). Call BEFORE bake_env/
        arm_process_env: it changes the woven layout (SCHED_TIMER_INDIRECT --
        the arena's timer word becomes a retargetable pointer, cache-tagged
        -ti). Fail-safe inherited: word==0 -> timer off."""
        if self._timer_dev is None:
            self._timer_dev = torch.zeros(self.N, dtype=torch.int64,
                                          device=self.dev)
            self._timer_stream = torch.cuda.Stream()
        self.arena.write(TIMER_OFF,
                         struct.pack("<Q", self._timer_dev.data_ptr()))

    def use_device_order(self) -> None:
        """Kernel-resident order table (SCHED_ORDER_INDIRECT): the arena ORDER
        word becomes a retargetable POINTER to a DEVICE order tensor, so
        install_order runs DEVICE->DEVICE with NO host sync -- this is what
        lets the control loop drop the per-step .tolist()/host-sort drain that
        plan_every exists to mask. Call BEFORE bake_env/arm_process_env (it
        changes the woven layout, cache-tagged -oi). Fail-safe inherited:
        pointer 0 -> identity remap. The tensor starts as identity so a partial
        install leaves a valid tail."""
        if self._order_dev is None:
            self._order_dev = torch.arange(self.N, dtype=torch.int32,
                                           device=self.dev)
        self.arena.write(ORDER_OFF,
                         struct.pack("<Q", self._order_dev.data_ptr()))

    BAKE_KEYS = ("SCHED_PLUGIN", "SCHED_BAKE_TASK_ORDER", "SCHED_BAKE_CTRL",
                 "SCHED_BAKE_TIMER", "SCHED_BAKE_QUEUE", "SCHED_MAX_TASKS",
                 "FLASHINFER_WORKSPACE_BASE", "SCHED_TIMER_INDIRECT",
                 "SCHED_ORDER_INDIRECT")

    def arm_process_env(self, plugin: str) -> None:
        """Apply bake_env to THIS process's os.environ. Call before importing
        flashinfer (its JIT reads the env at compile time). Keys absent from
        the bake (e.g. SCHED_TIMER_INDIRECT when the channel is direct) are
        REMOVED -- a stale value would weave a mismatched layout."""
        env = self.bake_env(plugin)
        for k in self.BAKE_KEYS:
            if k in env:
                os.environ[k] = env[k]
            else:
                os.environ.pop(k, None)

    # -- pi ------------------------------------------------------------------
    def set_order(self, tasks) -> None:
        """tasks[i] = the task the i-th claim/CTA serves (a permutation
        prefix; the tail stays identity). With the kernel-resident channel on
        (use_device_order), installs DEVICE->DEVICE with no host sync -- a
        device-tensor `tasks` never leaves the GPU."""
        t = torch.as_tensor(tasks, dtype=torch.int32)
        n = min(t.numel(), self.N)
        if self._order_dev is not None:
            self._order_dev[:n] = t[:n].to(self._order_dev.device,
                                           non_blocking=True)
            return
        self.arena.write(ORDER_OFF, t[:n].cpu().numpy().tobytes())

    def reset_order(self) -> None:
        self.set_order(torch.arange(self.N, dtype=torch.int32))

    def set_order_size(self, n: int) -> None:
        """Stamp the tile count the installed permutation is FOR (mirror;
        lands at the next push -- write the table via set_order FIRST, so the
        size acts as the commit flag). The weave honors the order table only
        when this equals the launch's grid; 0 = unchecked (legacy)."""
        struct.pack_into("<I", self._ctrl_host, CTRL_ORDER_SIZE_OFF, int(n))

    def install_order(self, tasks, n: int) -> None:
        """The concurrent-safe pi install: table first, validity stamp second
        (published by the caller's push). Use this from serving control
        loops; bare set_order is for single-threaded harnesses."""
        self.set_order(tasks)
        self.set_order_size(n)

    # -- ctrl (packed into a host mirror, published by push()) --------------
    def set_lambda(self, bw=0.0, l2=0.0, smem=0.0, comp=0.0) -> None:
        # The sigma-policy shadow prices (the score's lambda*dR term). UNWIRED
        # in serving because the policy/prefetch lever declines on FlashInfer's
        # cp.async KV stream (host-fixture only today); this is its control
        # surface, kept for the host-app path and the future cp.async-detect
        # work. Host fixtures (paged_decode modes D/E) exercise it.
        struct.pack_into("<4f", self._ctrl_host, CTRL_LAMBDA_OFF,
                         float(bw), float(l2), float(smem), float(comp))

    def set_num_tasks(self, n: int) -> None:
        """Also the work-queue/CLC per-step arming switch: the woven driver
        takes the stock (static) path when num_tasks == 0, the claim loop when
        num_tasks > 0. Toggling is a data write -- no recompile, and in CLC
        mode no launch change (grid == tasks either way)."""
        struct.pack_into("<I", self._ctrl_host, CTRL_NUM_TASKS_OFF, int(n))

    def set_timer_enabled(self, on: bool) -> None:
        """Per-step observation gate (takes effect on the next push): the
        baked ABI cannot null its slots, so the woven timer is suppressed by
        data -- ctrl.flags bit0. Default (0) is timer ON. Sample the timer on
        probe steps only; its PCIe atomic costs ~+5.6% at serving scale."""
        (flags,) = struct.unpack_from("<I", self._ctrl_host, CTRL_FLAGS_OFF)
        flags = (flags & ~FLAG_TIMER_OFF) | (0 if on else FLAG_TIMER_OFF)
        struct.pack_into("<I", self._ctrl_host, CTRL_FLAGS_OFF, flags)

    def set_row(self, i: int, q: float = 0.0, tau: int = 0,
                hint: int = HINT_AUTO) -> None:
        off = CTRL_ROWS_OFF + ROW_SIZE * i
        struct.pack_into("<fHB", self._ctrl_host, off, float(q),
                         int(tau) & 0xFFFF, int(hint) & 0xFF)

    def set_rows(self, q=None, tau=None, hint=None, n: int | None = None) -> None:
        n = self.N if n is None else n
        for i in range(n):
            self.set_row(i,
                         q=(q[i] if q is not None else 0.0),
                         tau=(tau[i] if tau is not None else 0),
                         hint=(hint[i] if hint is not None else HINT_AUTO))

    def push(self) -> None:
        """Publish the ctrl mirror, bumping the generation. Body first, the
        generation word last: x86 stores and PCIe posted writes preserve
        order, so a kernel that sees the new generation sees complete rows."""
        self._generation += 1
        struct.pack_into("<I", self._ctrl_host, 0, self._generation)
        self.arena.write(CTRL_OFF + 4, bytes(self._ctrl_host[4:]))
        self.arena.write(CTRL_OFF, bytes(self._ctrl_host[:4]))

    # -- timer (the adjoint: per-task cycles out) ----------------------------
    def clear_timer(self) -> None:
        if self._timer_dev is not None:
            self._timer_dev.zero_()  # stream-ordered before the next launch
            return
        self.arena.memset(TIMER_OFF, 0, 8 * self.N)

    def read_timer(self) -> torch.Tensor:
        if self._timer_dev is not None:
            # Side-stream D2H so the read never waits on the main stream's
            # queued work. Caller guarantees the writes are complete (post-
            # sync, or the plugin's post-kernel event has fired).
            with torch.cuda.stream(self._timer_stream):
                t = self._timer_dev.to("cpu", non_blocking=True)
            self._timer_stream.synchronize()
            return t
        raw = self.arena.read(TIMER_OFF, 8 * self.N)
        return torch.frombuffer(bytearray(raw), dtype=torch.int64)

    # -- work-queue ticket (persistent/CLC transform only) -------------------
    # UNWIRED in serving: the ticket claim is the pre-Blackwell (workers<<tasks)
    # acquisition path; on this Blackwell node CLC mode is used (no ticket
    # counter) and serving launches full grids. This primes the counter for
    # the baked-ABI ticket path -- kept as the Python peer of the C
    # sched_rt_queue_reset (host fixtures) for sm<100 / worker-pool serving.
    def reset_queue(self, workers: int) -> None:
        self.arena.write(QUEUE_OFF, struct.pack("<i", int(workers)))

    # -- R: the CLC resident-prefix size (occupancy model) --------------------
    # Measured contract (experiments/clc/FINDINGS.md): the hardware launches
    # raw [0, R) immediately; raw [R, N) is the CLC-claimable suffix, with
    #   R = maxActiveBlocksPerMultiprocessor(kernel, block, dyn_smem) * SMs.
    # Region-aware pi placement and the "expect no CLC benefit when N <= R"
    # gate both need R, computed for the ACTUAL kernel (regs/smem change it).
    def sm_count(self) -> int:
        dev = torch.cuda.current_device() if self.dev.startswith("cuda") else 0
        return torch.cuda.get_device_properties(dev).multi_processor_count

    def r_from_occupancy(self, blocks_per_sm: int) -> int:
        """R when the kernel owner already knows blocks/SM (runtime-API
        cudaOccupancy* on its own kernel -- the preferred path; see the
        exported occupancy query in python/kernels/paged_softmax.cu)."""
        return int(blocks_per_sm) * self.sm_count()

    def r_for_cached_so(self, so_path: str, symbol_substr: str,
                        threads: int, dyn_smem: int = 0) -> int:
        """EXACT R for a kernel inside a cached JIT .so (e.g. FlashInfer's),
        via the DRIVER API on the embedded cubin: extract with cuobjdump,
        cuModuleLoadData, cuOccupancyMaxActiveBlocksPerMultiprocessor on the
        CUfunction. Deliberately avoids the runtime-API stub handle: a torch
        process can hold two cudart instances (pip-wheel + system toolkit)
        and the fatbin registers with only one -- querying the other returns
        cudaErrorInvalidResourceHandle. libcuda is one-per-process.
        Measured on the real woven BatchDecode kernel: blocks/SM in the 3..5
        range -> R 564..940 on the 188-SM sm_120, far below the light-probe
        2256 (register/smem pressure) -- so typical decode batches sit at
        N <= R and the arming gate correctly keeps CLC off. Returns 0 on any
        failure -- unknown, never a guess."""
        import re
        import subprocess
        import tempfile as tf
        cu = self.arena.cu
        try:
            cuobjdump = os.path.join(
                os.environ.get("SCHED_CUDA_PATH", "/usr/local/cuda-12.9"),
                "bin", "cuobjdump")
            if not os.path.exists(cuobjdump):
                cuobjdump = "cuobjdump"
            # Device symbol = stub symbol minus the __device_stub__ infix
            # (Itanium mangling: the length prefix shrinks by 15).
            nm = subprocess.run(["nm", "-D", "--defined-only", so_path],
                                capture_output=True, text=True, timeout=30)
            stubs = [l.split()[-1] for l in nm.stdout.splitlines()
                     if "__device_stub__" in l and symbol_substr in l]
            if not stubs:
                return 0
            def devname(stub):
                # "48__device_stub__Foo..." -> "33Foo..." (length prefix
                # shrinks by len("__device_stub__") == 15).
                m = re.search(r"(\d+)__device_stub__", stub)
                if not m:
                    return stub
                return (stub[:m.start()] + str(int(m.group(1)) - 15) +
                        stub[m.end():])
            kname = devname(stubs[0])
            with tf.TemporaryDirectory() as td:
                subprocess.run([cuobjdump, "-xelf", "all", so_path],
                               capture_output=True, text=True, timeout=120,
                               cwd=td)
                blocks = ctypes.c_int(0)
                for cb in sorted(os.listdir(td)):
                    data = open(os.path.join(td, cb), "rb").read()
                    mod = ctypes.c_void_p()
                    if cu.cuModuleLoadData(ctypes.byref(mod), data) != 0:
                        continue  # wrong arch for this device
                    fn = ctypes.c_void_p()
                    if cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                              kname.encode()) != 0:
                        cu.cuModuleUnload(mod)
                        continue
                    rc = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        ctypes.byref(blocks), fn, ctypes.c_int(int(threads)),
                        ctypes.c_size_t(int(dyn_smem)))
                    cu.cuModuleUnload(mod)
                    if rc == 0 and blocks.value > 0:
                        return blocks.value * self.sm_count()
            return 0
        except Exception as e:
            if os.environ.get("SCHED_DEBUG"):
                print(f"[sched] r_for_cached_so failed: {e!r}")
            return 0

    @staticmethod
    def _grouped_lpt(c, block: int) -> "torch.Tensor":
        """Locality-preserving LPT: chunk the tiles into blocks of `block`
        ADJACENT indices, order the BLOCKS by descending aggregate cost, keep
        original (ascending) index order WITHIN a block. block=1 is plain
        per-tile LPT. Coarser blocks trade makespan balance for L2 locality --
        adjacent tiles (adjacent requests -> adjacent KV) stay co-scheduled.
        Measured (eval_pi_grouped.py, sm_120): block~16 Pareto-beats per-tile
        LPT -- it ERASES the mid-batch L2-scramble penalty (2048 tiles: +0% ->
        -2.5%) while keeping the large-batch makespan win (8192: -10.4% ->
        -11.0%). Still an E1 permutation (bit-exact); pure control-plane order."""
        n = c.numel()
        if block <= 1 or n <= block:
            return torch.argsort(c, descending=True).to(torch.int32)
        nb = (n + block - 1) // block
        pad = nb * block - n
        cc = torch.cat([c, c.new_zeros(pad)]) if pad else c
        bcost = cc.reshape(nb, block).sum(1)
        border = torch.argsort(bcost, descending=True).to(torch.int64)
        base = border[:, None] * block + torch.arange(block, dtype=torch.int64)[None, :]
        order = base.flatten()
        return order[order < n].to(torch.int32)     # drop padding slots

    @staticmethod
    def region_order(cost, R: int, block: int = None) -> "torch.Tensor":
        """Region-aware pi for a CLC-armed step (FINDINGS 'Scheduler
        Implications'). The hardware launches raw [0, R) immediately (striped
        across SMs: consecutive prefix ids land on different SMs) and exposes
        [R, N) to late-binding steals, so:
          prefix [0, R): alternate heaviest / lightest -- heavy work starts
            early AND light work frees workers to steal (all-heavy LPT would
            leave nobody free until the tail);
          suffix [R, N): the remaining middle, heaviest first, so whichever
            worker frees first pulls the most work forward.
        Static (non-CLC) path uses locality-preserving grouped-LPT
        (SCHED_LPT_BLOCK, default 16); the CLC region-aware path below needs
        TRUE per-tile heaviest/lightest, so it keeps plain per-tile LPT."""
        c = torch.as_tensor(cost).float()
        n = c.numel()
        if R <= 0 or n <= int(R):
            if block is None:
                block = int(os.environ.get("SCHED_LPT_BLOCK", "16"))
            return SchedPlane._grouped_lpt(c, block)
        R = int(R)
        lpt = torch.argsort(c, descending=True).to(torch.int32)
        nh = (R + 1) // 2                     # heavy slots in the prefix
        nl = R - nh                           # light slots in the prefix
        pre = torch.empty(R, dtype=torch.int32)
        pre[0::2] = lpt[:nh]                  # heaviest, one per SM stripe
        if nl:
            pre[1::2] = lpt[n - nl:].flip(0)  # lightest, interleaved
        suf = lpt[nh:n - nl] if nl else lpt[nh:]
        return torch.cat([pre, suf])
