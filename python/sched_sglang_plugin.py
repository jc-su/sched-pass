"""sched_sglang_plugin.py -- wire the sched-pass control plane into SGLang.

Zero engine edits: registers hooks on SGLang's real scheduler via its plugin
system (sglang.srt.plugins). The control loop, per decode batch:

  BEFORE run_batch:
    * IDENTITY BINDING: read batch.req_pool_indices -> the request each slot
      serves this step (rid <-> slot, bumped by generation). This is the anchor
      every measurement/policy is keyed through; it must be rebuilt each step
      because the batch mutates as requests join/leave (the eKV anchor lesson).
    * DECIDE + ENFORCE: from last step's per-request cycles (the woven timer)
      and per-request KV lengths, build pi (LPT/SRPT/EDF) and per-task
      {q, tau, hint}; write the SchedPlane tables; push (generation++).
      Ordered before the batch's kernels on the stream (CUDA-graph-safe data
      writes; CapKV's wait-for-metadata rule).

  AFTER run_batch:
    * OBSERVE: read the woven timer -> per-request residency cycles; fold into
      the estimator for next step's pi. Zero GPU API calls (host-mapped read).

Install (no SGLang fork):
  SGLANG_PLUGINS=sched python -m sglang.launch_server ...     # via entry point
or force-register by importing this module before the scheduler starts (e.g. a
sitecustomize.py that calls sched_sglang_plugin.register()).

STATUS (2026-07): both halves are LIVE. FlashInfer decode compiles under
clang+plugin (the header nvcc-isms are closed by clang_cuda_prelude.h +
patch_flashinfer.py + the nvcc_clang_shim), so the woven kernel reads the
control tables on a real SGLang server. ENFORCE mode (pi ordering + arming,
SCHED_SGLANG_ENFORCE=1) is measured end to end; OBSERVE mode (default) binds
identities and reads timers without changing the schedule. Correctness is
gated by test_serving_gate.py (woven==stock token identity on boot-stable
models); serving overhead is ~zero (ROADMAP.md B3). Remaining boundary:
the pi latency GAIN needs a stable batch in the queued, attention-dominant
regime (ROADMAP.md) -- the machinery is production-posture, the headline
serving-gain measurement is the open item.
"""
import os
import torch

from sched_controller import SchedControlPlane
from sched_rt import (SchedPlane, HINT_URGENT, HINT_POLITE, HINT_AUTO,
                      compute_bake_env, predicted_base)

_PLANE = None
_STATE = {
    "last_cycles": None, "step": 0,
    "observe_only": os.environ.get("SCHED_SGLANG_ENFORCE") is None,
    # the estimator/policy (estimation -> ordering -> damping; SGLANG.md #4)
    "ctl": SchedControlPlane(policy=os.environ.get("SCHED_SGLANG_POLICY",
                                                   "lpt")),
    # OBSERVATION CADENCE: the timer's PCIe atomic costs ~+5.6% at serving
    # scale, so sample it: arm the timer 1 step in SCHED_TIMER_EVERY (the
    # ctrl.flags gate; 1 = every step, the old behavior).
    "timer_every": max(1, int(os.environ.get("SCHED_TIMER_EVERY", "8"))),
    "timer_armed": False,
    # PLANNING CADENCE (the overlap-preservation knob): binding + cost + order
    # each require a GPU->CPU read (slot indices, seq_lens, plan_info) that
    # DRAINS the stream. Doing that every step collapses SGLang's overlap
    # scheduler (a ~20us sync stalls a whole step's worth of hidden work ->
    # measured -63% throughput at conc 1024). The control plane is
    # staleness-tolerant (fail-safe defaults + the kernel-side order_size guard
    # retires a stale-size order to identity), so re-plan only 1 step in N;
    # between re-plans the installed tables persist and the kernel keeps using
    # them. 1 = every step (old behavior). Probe steps always re-plan (they
    # need a fresh binding to attribute the timer).
    "plan_every": max(1, int(os.environ.get("SCHED_PLAN_EVERY", "8"))),
    # CLC ARMING (needs kernels JIT-woven with SCHED_WORKQUEUE=1 SCHED_CLC=1;
    # harmless data writes otherwise): off | on | auto. In auto, arm the claim
    # loop only when BOTH (a) prediction uncertainty exceeds SCHED_CLC_RESID
    # and (b) ntiles > R (the resident prefix: no suffix -> nothing to steal).
    # Disarm = num_tasks=0 -> the woven driver's stock path (static + pi).
    #
    # Calibration (experiments/clc/clc_noise_probe.cu, sm_120): LPT ranking of
    # a 32x-bimodal decode mix survives +-50% multiplicative cost noise with
    # recall 100% (CLC ties, keep it off), and CLC CANNOT rescue a single
    # late-issued straggler (eps=1.0: both +50% vs oracle -- late binding
    # balances WHO runs a task, not WHEN it is issued). CLC pays only under
    # SEVERE order breakdown (recall <~75%: -2..-7.5%). uncertainty ~= eps/2
    # for U(-1,1) noise, so arm around eps>=1.5-2 => threshold ~0.75. Cold
    # batches (nothing observed yet) score ~1.0 and arm -- the right call.
    "clc": os.environ.get("SCHED_SGLANG_CLC", "auto"),
    "clc_resid": float(os.environ.get("SCHED_CLC_RESID", "0.75")),
    # CLC's tail-imbalance veto: arm only when max/mean of predicted per-tile
    # cost exceeds this. Uniform serving batches sit ~1.5-2.0 (no tail to
    # steal); dispersed sharegpt-like mixes reach 4-8. 3.0 keeps CLC off in the
    # common uniform case where it only adds cost (measured -2.2x at conc 1024).
    "clc_imbalance": float(os.environ.get("SCHED_CLC_IMBALANCE", "3.0")),
    "clc_r": int(os.environ.get("SCHED_CLC_R", "0")),  # 0 = unknown -> gate off
    # sigma hints only pay when the batch KV working set exceeds L2 (MODEL.md);
    # needs bytes/token to compute the footprint. 0 = hints disabled.
    "kv_bytes_per_token": int(os.environ.get("SCHED_KV_BYTES_PER_TOKEN", "0")),
    "binding": None,  # (rids, kv_lens, tile_to_req or None) of the last step
    "pending": None,  # (cuda Event, binding) of an unread probe step
    "probes_folded": 0,
    "stats_every": int(os.environ.get("SCHED_STATS_EVERY", "512")),
}


def _plane(max_tasks):
    """The process's one SchedPlane (SchedArena is one-per-process). Created
    at first use, sized from SCHED_MAX_TASKS (default 4096), and it ARMS the
    process env (SCHED_BAKE_* + the va-keyed FlashInfer workspace) so every
    later JIT compile in this scheduler process bakes the fixed-VA tables."""
    global _PLANE
    if _PLANE is None:
        cap = int(os.environ.get("SCHED_MAX_TASKS", "4096"))
        _PLANE = SchedPlane(max_tasks=cap, device="cuda")
        # DEVICE observation channel by default (E1: device atomics ~free vs
        # +26.5% host-mapped; test_timer_indirect.py gates it). Collection is
        # then always cheap; only the probe READOUT costs a small D2H.
        # SCHED_TIMER_DEVICE=0 reverts to the host-mapped zero-touch channel.
        if os.environ.get("SCHED_TIMER_DEVICE", "1") != "0":
            _PLANE.use_device_timer()
        # KERNEL-RESIDENT ORDER: install pi device->device (no host sync). The
        # primary order is grouped-LPT over KV LENGTH -- known on-device and
        # monotone in true decode cost, so the host ESTIMATOR is not needed to
        # ORDER (it only refines the residual, on the probe cadence). This is
        # what lets the hot path drop the per-step seq_lens .cpu() drain.
        if os.environ.get("SCHED_ORDER_DEVICE", "1") != "0":
            _PLANE.use_device_order()
        plugin = os.environ.get("SCHED_PLUGIN")
        if plugin:
            # Normally a no-op re-assertion of the env pre-armed at import
            # (pre_arm_env below): FlashInfer froze its workspace path back
            # then. If the arena missed the canonical VA, this re-arms with
            # the ACTUAL base so every FUTURE compile is correct (fresh cache
            # dir; the decode JIT is lazy and only fires after this point) --
            # correct-or-recompile, never wrong.
            _PLANE.arm_process_env(plugin)
            if not _PLANE.arena.canonical:
                print("[sched-sglang] WARNING: canonical VA missed; re-armed "
                      f"at 0x{_PLANE.arena.base:x} (kernels recompile once)")
    if max_tasks > _PLANE.N:
        raise RuntimeError(
            f"batch has {max_tasks} tiles > SCHED_MAX_TASKS={_PLANE.N}; "
            f"raise SCHED_MAX_TASKS before server start (the arena is "
            f"one-per-process and its size is baked into the kernels)")
    return _PLANE


def plane():
    """The live SchedPlane (for the launch site to bake its addresses)."""
    return _PLANE


# --- the per-step control loop ---------------------------------------------
def on_batch_begin(scheduler, batch):
    """BEFORE run_batch: bind identities (rid<->slot<->TILE), decide
    pi/policy/arming, write + push. Heavy (GPU->CPU-syncing) work runs on a
    CADENCE (plan_every) to preserve SGLang's overlap pipeline; off-cadence
    steps return immediately and the last-installed tables persist."""
    step = _STATE["step"]
    _STATE["step"] = step + 1
    # CADENCE GATE: re-plan only every plan_every steps, plus on probe steps
    # (which need a fresh binding to attribute the timer). Off-cadence: do
    # NOTHING -- no slot/seqlen/plan GPU->CPU reads, no push -- so the stream
    # never drains and the overlap scheduler keeps hiding the CPU prep. Safe:
    # a stale order on a changed batch is retired to identity kernel-side by
    # the order_size guard (bit-exact); a stale order on a stable batch is
    # still the order we want.
    probe = ((step % _STATE["timer_every"]) == 0 and
             _STATE.get("pending") is None)
    if (step % _STATE["plan_every"]) != 0 and not probe:
        # off-cadence: clear timer_armed so on_batch_end does not mistake this
        # step's completion event for the probe's (attribution stays correct).
        _STATE["timer_armed"] = False
        return

    slots = _slot_indices(batch)
    if slots is None:
        return
    n = len(slots)
    kvlen = _kv_lengths(batch, n)
    kv_list = kvlen.tolist() if kvlen is not None else [1.0] * n

    # rid<->TILE binding (SGLANG.md 4c): tile k serves request tile_to_req[k].
    # Decode: one tile/request unless the plan split KV across tiles. Prefill:
    # a request is tiled over qo (and kv) chunks -> always several tiles. pi
    # permutes TILES; the timer folds per-REQUEST exactly (both cases).
    if _attn_mode(batch) == "prefill":
        tile_to_req, ntiles = _prefill_tile_binding(scheduler, n)
    else:
        tile_to_req, ntiles = _tile_binding(scheduler, n)
    p = _plane(ntiles)
    _consume_probe(p)  # fold the LAST probe step's cycles, if the GPU is done
    _resolve_r(p)      # lazily wire R from the cached woven .so (occupancy)

    ctl = _STATE["ctl"]
    # per-REQUEST predicted cost, VECTORIZED (t_hat = alpha*kv + beta +
    # resid, clamped): the scheduler thread runs this EVERY step, and a
    # python predict/sort loop over thousands of tiles measurably stalls the
    # serving loop in the queued regime (the enforce bench died on it twice).
    # Estimation stays in the controller; the hot path is tensor ops.
    alpha, beta = ctl._fit.coeffs()
    kv_t = torch.as_tensor(kv_list, dtype=torch.float32)
    resid_t = torch.tensor([ctl._resid.get(r, 0.0) for r in slots],
                           dtype=torch.float32)
    cost_req_t = (alpha * kv_t + beta + resid_t).clamp_min_(0.0)
    if tile_to_req is not None:
        idx = torch.as_tensor(tile_to_req, dtype=torch.long).clamp_(0, n - 1)
        cost_tile_t = cost_req_t[idx]
    else:
        cost_tile_t = cost_req_t
    cost_req = cost_req_t.tolist()  # scalar consumers (hints threshold)

    # observation cadence: arm the timer only on probe steps -- and never
    # start a new probe while the previous one is still unconsumed (its rows
    # would be cleared unread; the pending event says the GPU may still be
    # writing them under the OVERLAPPED scheduler).
    # (probe was decided at the top, pre-increment; reuse it)
    p.set_timer_enabled(probe)
    _STATE["timer_armed"] = probe

    if not _STATE["observe_only"]:
        cost_t = cost_tile_t

        # CLC arming (auto): late-binding acquisition helps ONLY when ALL of:
        #   (a) there is an unlaunched suffix to steal      (ntiles > R)
        #   (b) a real stealable TAIL exists -- high measured imbalance; on a
        #       uniform batch LPT already ~= identity and there is nothing for
        #       late binding to rescue (FINDINGS.md #3). max/mean of predicted
        #       per-tile cost is the cheap tail proxy (CPU tensor, no sync).
        #   (c) the estimate might be WRONG (high uncertainty) -- if it is
        #       right, static+pi is already optimal (clc_pipeline_probe.cu).
        # Measured why (b) matters: at conc 1024 UNIFORM (imbalance ~1.6) the
        # old gate armed on cold-estimator uncertainty alone and cost 2.2x
        # (enforce+CLC 13.06 vs enforce 28.71 req/s). Imbalance now vetoes it.
        clc, R = _STATE["clc"], _STATE["clc_r"]
        imbalance = (float(cost_tile_t.max() / cost_tile_t.mean().clamp_min(1e-6))
                     if ntiles else 1.0)
        arm = (clc == "on" or
               (clc == "auto" and R > 0 and ntiles > R and
                imbalance > _STATE["clc_imbalance"] and
                ctl.uncertainty(slots, kv_list) > _STATE["clc_resid"]))
        if arm:
            p.install_order(SchedPlane.region_order(cost_t, R), ntiles)
            _STATE["order_n"] = ntiles  # remember: an order is live (for retire)
            p.set_num_tasks(ntiles)  # claim loop on
        elif R > 0 and ntiles <= R:
            # ONE WAVE (tiles <= resident capacity): every tile is issued
            # immediately, so ordering cannot change the makespan (THEORY
            # S9's regime gate) -- and the estimator+sort per step is pure
            # scheduler-thread cost. Measured: un-gated enforcement regressed
            # 3B serving ~14% at conc 256 < R while buying nothing. Skip --
            # and RETIRE any order a prior queued step installed (else the
            # kernel honors a stale permutation whenever order_size still
            # matches this tile count; bit-exact, but not the intended
            # identity, and it wastes the skip).
            if _STATE.get("order_n") is not None:
                p.reset_order(); p.set_order_size(0)
                _STATE["order_n"] = None
            p.set_num_tasks(0)
        else:
            # QUEUED regime (or R unknown): LPT over tiles, vectorized.
            #
            # WRITE-SAFETY (root cause of cudaErrorIllegalAddress under the
            # overlap scheduler): the woven kernel reads the order table at
            # EXECUTION time, and the previous step's kernel may still be in
            # flight when this hook runs -- a permutation sized for THIS
            # batch can send an in-flight smaller/larger launch out of its
            # arrays. Install a new order ONLY when (a) the last launched
            # step's event has fired and (b) the tile count is unchanged
            # since the last install; otherwise leave the committed table
            # (identity, or a same-size previous order -- both always
            # in-bounds). pi thus engages in stable same-size stretches,
            # which is exactly where ordering pays. The structural fix
            # (kernel-side per-launch bound on the mapped task) is D1 work.
            # Bijectivity is now guaranteed KERNEL-side (order_size header:
            # a stale different-size permutation makes the whole launch take
            # identity; the per-entry clamp guards faults). The idle-gating
            # below is therefore an OPTIMIZATION -- avoid churning the table
            # while a step is in flight -- not a correctness requirement.
            ev = _STATE.get("inflight")
            if ev is None or ev.query():
                # LOCALITY-PRESERVING grouped-LPT (measured Pareto win over
                # per-tile argsort: erases the mid-batch L2-scramble penalty,
                # keeps the large-batch makespan win -- eval_pi_grouped.py).
                # region_order(cost, 0) == grouped-LPT (block SCHED_LPT_BLOCK).
                order = SchedPlane.region_order(cost_tile_t, 0)
                p.install_order(order, ntiles)
                _STATE["order_n"] = ntiles  # an order is live (for retire)
            p.set_num_tasks(0)  # claim loop off -> stock static + pi

        # sigma hints only when the batch KV working set exceeds L2 (the
        # capacity regime where residency control pays -- MODEL.md).
        bpt = _STATE["kv_bytes_per_token"]
        if bpt and kvlen is not None:
            l2 = torch.cuda.get_device_properties(
                torch.cuda.current_device()).L2_cache_size
            if float(kvlen.sum()) * bpt > l2:
                thr = float(torch.tensor(cost_req).median()) if n else 0.0
                for k in range(ntiles):
                    i = int(tile_to_req[k]) if tile_to_req is not None else k
                    long_i = cost_req[min(i, n - 1)] >= thr
                    p.set_row(k, q=float(cost_tile_t[k]),
                              hint=HINT_POLITE if long_i else HINT_URGENT)
    else:
        p.set_num_tasks(0)  # observe-only: never arm the claim loop

    p.push()
    if probe:
        p.clear_timer()
    _STATE["binding"] = (list(slots), kv_list, tile_to_req, ntiles)
    # (step already incremented at the top, before the cadence gate)
    # ops visibility: one line per stats window -- the controller's decisions
    # must be observable in production (arming, probing, estimator health).
    if _STATE["step"] % _STATE["stats_every"] == 0:
        u = _STATE["ctl"].uncertainty(slots, kv_list) if n else 0.0
        print(f"[sched-sglang] step={_STATE['step']} tiles={ntiles} "
              f"split={'y' if tile_to_req is not None else 'n'} "
              f"R={_STATE['clc_r']} probes={_STATE['probes_folded']} "
              f"uncert={u:.2f} hook_errors={_STATE.get('hook_errors', 0)} "
              f"mode={'observe' if _STATE['observe_only'] else 'enforce'}")


def on_batch_end(result, batch):
    """AFTER run_batch (probe steps only): record a CUDA event AFTER the
    batch's kernels and defer the timer read to a later on_batch_begin once
    the event has fired. NO synchronize() here: under SGLang's overlapped
    event loop a sync would serialize the CPU/GPU pipeline every probe step
    -- the observation must not distort the thing it observes."""
    if _PLANE is None:
        return result
    # ALWAYS mark the step's completion point: table writes at the next
    # begin-hook are gated on this event (the in-flight kernel reads the
    # tables at execution time -- see the write-safety note in begin).
    ev = torch.cuda.Event()
    ev.record()  # stream-ordered after this step's kernels
    _STATE["inflight"] = ev
    if _STATE["timer_armed"]:
        _STATE["pending"] = (ev, _STATE["binding"])
        # #1 timer-race fix (host half): the probe kernel latched timer-ON at
        # its ENTRY, so pushing OFF now cannot suppress its own write -- and it
        # makes every later OFF-CADENCE launch latch OFF, so the probe samples
        # EXACTLY one step (no cross-step accumulation that distorts per-request
        # cost by presence-duration under batch churn). Cheap: host memcpy into
        # the UVA arena, no CUDA call/sync.
        if _PLANE is not None:
            _PLANE.set_timer_enabled(False)
            _PLANE.push()
    return result


def _consume_probe(p):
    """Non-blocking: if the pending probe step's GPU work has completed
    (event fired), read its timer rows and fold them into the estimator;
    otherwise leave it pending and try again next step."""
    pend = _STATE.get("pending")
    if pend is None:
        return
    ev, b = pend
    if not ev.query():
        return  # GPU still busy with (or before) the probe step
    _STATE["pending"] = None
    cyc = p.read_timer()
    _STATE["last_cycles"] = cyc
    if b is None:
        return
    slots, kv_list, tile_to_req, ntiles = b
    n = len(slots)
    # EXACT per-request fold, VECTORIZED (no python per-tile loop): sum every
    # tile's cycles into its request. scatter_add is exact no matter how many
    # qo/kv tiles a long request was split into -- the split-safe attribution
    # the whole design turns on. cyc is int64 (read_timer).
    c = (cyc if torch.is_tensor(cyc) else torch.as_tensor(cyc)).to(torch.int64)
    if tile_to_req is None:
        m = min(n, c.numel())
        per_req = c[:m].tolist() + [0] * (n - m)
    else:
        m = min(ntiles, c.numel(), len(tile_to_req))
        idx = torch.as_tensor(tile_to_req[:m], dtype=torch.long).clamp_(0, n - 1)
        per_req = torch.zeros(n, dtype=torch.int64) \
            .scatter_add_(0, idx, c[:m]).tolist()
    _STATE["ctl"].observe(slots, kv_list, per_req)
    _STATE["probes_folded"] = _STATE.get("probes_folded", 0) + 1


def _resolve_r(p):
    """Wire R (the CLC resident-prefix size) from the cached woven decode .so
    via the driver-API occupancy query -- lazily, because the .so only exists
    after the first decode JIT. Retries sparsely; env SCHED_CLC_R overrides."""
    if _STATE["clc_r"] != 0 or _STATE["clc"] == "off":
        return
    step = _STATE["step"]
    if step == 0 or step % 64 != 1:
        return
    try:
        import glob
        base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
        sos = sorted(glob.glob(os.path.join(
            base, ".cache", "flashinfer", "**", "*batch_decode*.so"),
            recursive=True)) if base else []
        if sos:
            r = p.r_for_cached_so(sos[0], "BatchDecode", 128)
            if r > 0:
                _STATE["clc_r"] = r
                print(f"[sched-sglang] R resolved from woven kernel: {r}")
    except Exception as e:
        if os.environ.get("SCHED_DEBUG"):
            print(f"[sched-sglang] R resolution failed (will retry): {e}")


# --- batch introspection (SGLang ScheduleBatch shapes) ---------------------
def _slot_indices(batch):
    for attr in ("req_pool_indices_cpu", "req_pool_indices"):
        v = getattr(batch, attr, None)
        if v is not None:
            return v.to("cpu").tolist() if hasattr(v, "tolist") else list(v)
    return None


def _kv_lengths(batch, n, device=False):
    """Per-request KV lengths. device=True keeps them ON THE GPU (batch.seq_lens
    is a device tensor) -- the hot-path order sorts by these with NO .cpu()
    sync; device=False materializes them on the host for the estimator (probe
    cadence only)."""
    attrs = (("seq_lens", "seq_lens_cpu") if device
             else ("seq_lens_cpu", "seq_lens"))
    for attr in attrs:
        v = getattr(batch, attr, None)
        if v is not None:
            t = torch.as_tensor(v) if not hasattr(v, "to") else v
            t = t if device else t.to("cpu")
            return t[:n].float()
    return None


def _attn_mode(batch):
    """'decode' | 'prefill' | None -- which woven attention kernel this batch
    runs. Prefill weaving is OPT-IN (SCHED_WEAVE_PREFILL=1) until live-validated
    end-to-end; decode-only stays the proven default so enabling prefill can
    never destabilize the shipping decode path. (The prefill MECHANISM --
    compile, weave, bit-exact pi, exact per-request fold -- is proven by
    test/py/test_flashinfer_prefill.py; this gate governs the SERVING loop.)"""
    fm = getattr(batch, "forward_mode", None)
    if fm is None:
        return None
    if getattr(fm, "is_decode", lambda: False)():
        return "decode"
    if os.environ.get("SCHED_WEAVE_PREFILL") == "1":
        for name in ("is_extend", "is_prefill"):
            if getattr(fm, name, lambda: False)():
                return "prefill"
    return None


# --- rid<->tile binding through FlashInfer's plan (SGLANG.md 4c) ------------
def _plan_request_indices(w, n, want_len, req_field, require_flag=None):
    """Shared plan reader -> (tile_to_req, ntiles). Recover request_indices[tile]
    (which request each grid TILE serves) from a FlashInfer PlanInfo vector and
    its int workspace buffer, so pi permutes tiles and the timer folds
    per-REQUEST EXACTLY even when one request's KV/QO is split across tiles.
      want_len     pins the PlanInfo layout (version guard): decode=10, prefill=15
      req_field    vector slot holding request_indices_offset (BYTES): decode 3,
                   prefill 4
      require_flag vector slot that must be truthy to bind, or None to always
                   bind (decode gates on split_kv@9 -- unsplit decode is one tile
                   per request; prefill tiles = qo x kv so a long request spans
                   several tiles regardless -> no gate)
    Any layout mismatch -> (None, n): one tile per request, fail-soft, never
    wrong. request_indices are `padded` int32s; read via a BYTE view (correct
    for any buf dtype) then clone() so .view(int32) is aligned for any offset."""
    vec = getattr(w, "_plan_info", None)
    buf = getattr(w, "_int_workspace_buffer", None)
    if vec is None or buf is None:
        return None, n
    v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    if len(v) != want_len:
        return None, n
    if require_flag is not None and not bool(v[require_flag]):
        return None, n
    padded, req_off = int(v[0]), int(v[req_field])
    nbytes = buf.numel() * buf.element_size()
    if padded < n or req_off < 0 or req_off + 4 * padded > nbytes:
        return None, n
    bb = buf.reshape(-1).contiguous().view(torch.uint8)  # byte view
    ti = bb[req_off:req_off + 4 * padded].clone().view(torch.int32).cpu()
    return ti.tolist(), padded


def _tile_binding(scheduler, n):
    """DECODE tile->request. DecodePlanInfo.ToVector() is 10 int64s (flashinfer
    0.6.x): [0]padded_batch_size [3]request_indices_offset(bytes) [9]split_kv.
    Bind only when the plan split KV across tiles (else one tile per request)."""
    try:
        w = _decode_wrapper(scheduler)
        if w is None:
            return None, n
        r = _plan_request_indices(w, n, 10, 3, require_flag=9)
        if (r[0] is not None and os.environ.get("SCHED_DEBUG")
                and not _STATE.get("_tb_dtype_logged")):
            _STATE["_tb_dtype_logged"] = True
            print(f"[sched-sglang] decode tile-binding armed: {r[1]} tiles / "
                  f"{n} reqs")
        return r
    except Exception as e:
        if os.environ.get("SCHED_DEBUG"):
            print(f"[sched-sglang] decode tile binding fallback (1/req): {e}")
        return None, n


def _prefill_tile_binding(scheduler, n):
    """PREFILL/extend tile->request. PrefillPlanInfo.ToVector() is 15 int64s:
    [0]padded_batch_size [4]request_indices_offset(bytes) [14]split_kv. A
    prefill request is tiled over qo (and kv) chunks -> it spans several grid
    tiles even without kv-split, so ALWAYS read request_indices (no gate).
    Verified empirically (test/py/probe_prefill_plan.py): request_indices is
    non-decreasing, in [0,B), covers every request -- the exact
    one-request->several-tiles map."""
    try:
        w = _prefill_wrapper(scheduler)
        if w is None:
            return None, n
        return _plan_request_indices(w, n, 15, 4, require_flag=None)
    except Exception as e:
        if os.environ.get("SCHED_DEBUG"):
            print(f"[sched-sglang] prefill tile binding fallback (1/req): {e}")
        return None, n


def _model_runner(scheduler):
    """SGLang model runner via the scheduler (paths differ by version)."""
    for path in (("tp_worker", "model_runner"),
                 ("tp_worker", "worker", "model_runner")):
        o = scheduler
        for name in path:
            o = getattr(o, name, None)
            if o is None:
                break
        if o is not None:
            return o
    return None


def _backend_wrapper(scheduler, backend_attrs, wrapper_names):
    """FlashInfer wrapper via the model runner's attn backend (defensive: names
    differ by SGLang version). First present backend attr, first present wrapper
    name (unwrapping a per-layer list to element 0)."""
    mr = _model_runner(scheduler)
    if mr is None:
        return None
    backend = None
    for a in backend_attrs:
        backend = getattr(mr, a, None)
        if backend is not None:
            break
    if backend is None:
        return None
    for name in wrapper_names:
        w = getattr(backend, name, None)
        if isinstance(w, (list, tuple)):
            w = w[0] if w else None
        if w is not None:
            return w
    return None


def _decode_wrapper(scheduler):
    return _backend_wrapper(
        scheduler, ("attn_backend", "decode_attn_backend"),
        ("decode_wrapper", "_decode_wrapper", "decode_wrappers"))


def _prefill_wrapper(scheduler):
    return _backend_wrapper(
        scheduler, ("attn_backend", "prefill_attn_backend", "extend_attn_backend"),
        ("prefill_wrapper_paged", "prefill_wrapper", "_prefill_wrapper",
         "prefill_wrappers_paged", "prefill_wrappers"))


# --- registration (SGLang hook system) -------------------------------------
def pre_arm_env():
    """Set the woven-JIT env NOW, at import time, from the PREDICTED canonical
    arena base: FlashInfer freezes FLASHINFER_WORKSPACE_BASE as a module
    constant at import (jit/env.py), long before the first decode batch can
    create the plane -- arming at batch time silently reuses stock-cache
    kernels. The plane verifies the prediction on creation and re-arms with
    the actual base if the canonical VA was unavailable."""
    plugin = os.environ.get("SCHED_PLUGIN")
    if not plugin:
        return
    cap = int(os.environ.get("SCHED_MAX_TASKS", "4096"))
    ti = os.environ.get("SCHED_TIMER_DEVICE", "1") != "0"
    env = compute_bake_env(predicted_base(), cap, plugin, os.environ,
                           timer_indirect=ti)
    for k in SchedPlane.BAKE_KEYS:
        if k in env:
            os.environ[k] = env[k]
        else:
            os.environ.pop(k, None)


def _hook_error(where, exc):
    """Hooks must NEVER crash the serving process, but a silent failure that
    disables scheduling/observation is worse than a loud one. Log the FIRST
    error per site always (not only under SCHED_DEBUG), count the rest, and
    surface the count in the periodic stats line so operators can see a
    degraded control plane."""
    _STATE["hook_errors"] = _STATE.get("hook_errors", 0) + 1
    seen = _STATE.setdefault("hook_err_seen", set())
    if where not in seen:
        seen.add(where)
        print(f"[sched-sglang] {where} hook error (control plane degraded; "
              f"serving continues): {exc!r}")
    elif os.environ.get("SCHED_DEBUG"):
        print(f"[sched-sglang] {where} hook error: {exc!r}")


def register():
    """Register the control-loop hooks on SGLang's scheduler. Idempotent
    (guarded: registering through both sitecustomize and an entry point must
    not double-install). BEFORE run_batch binds identities + writes tables;
    AFTER reads the timer. BEFORE __init__ creates the plane: woven kernels
    carry baked arena addresses, and CUDA-graph capture (and init-time warmup)
    launch
    decode OUTSIDE run_batch -- the arena must exist before the first woven
    launch is even possible, or capture dereferences an unmapped VA."""
    if _STATE.get("registered"):
        return  # idempotent: never double-install (sitecustomize + entrypoint)
    _STATE["registered"] = True
    pre_arm_env()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    TARGET_INIT = "sglang.srt.managers.scheduler.Scheduler.__init__"

    def before_init(self, *a, **kw):
        try:
            import torch as _t
            _t.cuda.init()
            _plane(1)  # capacity from SCHED_MAX_TASKS; arena mapped NOW
            if os.environ.get("SCHED_DEBUG"):
                print("[sched-sglang] plane created at scheduler init "
                      f"(base 0x{_PLANE.arena.base:x})")
        except Exception as e:
            print(f"[sched-sglang] init-time plane creation failed: {e!r}; "
                  "woven kernels would fault -- disarming compiles")
            for k in SchedPlane.BAKE_KEYS:
                os.environ.pop(k, None)
        return None

    if os.environ.get("SCHED_NO_INIT_HOOK") != "1":
        HookRegistry.register(TARGET_INIT, before_init, HookType.BEFORE)

    TARGET = "sglang.srt.managers.scheduler.Scheduler.run_batch"

    # BEFORE: fn(*args, **kwargs) -> (args, kwargs) or None. args = (self, batch, ...)
    def before(self, batch, *a, **kw):
        if os.environ.get("SCHED_HOOKS_NOOP") == "1":
            return None
        # Arm the MoE expert cap on the FIRST run_batch: FusedMoE is imported
        # during model load (after plugin register()), so patch the class method
        # now -- before the model forward inside this very run_batch, hence before
        # the first MoE forward. Idempotent + gated by SCHED_MOE_CAP inside.
        if not _STATE.get("moe_armed"):
            _STATE["moe_armed"] = True
            try:
                from sched_moe_hook import register_moe_cap
                if register_moe_cap() and os.environ.get("SCHED_DEBUG"):
                    print("[sched-sglang] MoE expert cap armed on FusedMoE.forward "
                          f"(capacity={os.environ.get('SCHED_MOE_CAPACITY','1.25')})")
            except Exception as e:
                _hook_error("moe-arm", e)
        try:
            if _attn_mode(batch):
                on_batch_begin(self, batch)
        except Exception as e:
            _hook_error("begin", e)
        return None  # do not modify args

    # AFTER: fn(result, *args, **kwargs) -> new_result or None. args=(self,batch,...)
    def after(result, self, batch, *a, **kw):
        if os.environ.get("SCHED_HOOKS_NOOP") == "1":
            return result
        try:
            if _attn_mode(batch):
                on_batch_end(result, batch)
        except Exception as e:
            _hook_error("end", e)
        return result

    HookRegistry.register(TARGET, before, HookType.BEFORE)
    HookRegistry.register(TARGET, after, HookType.AFTER)
    _dump_effective_config()
    # Ensure the target class exists before patching. When SGLang itself calls
    # load_plugins() the scheduler is already imported; a direct register()
    # (e.g. from sitecustomize) may run earlier, so import it defensively.
    try:
        import sglang.srt.managers.scheduler  # noqa: F401
        HookRegistry.apply_hooks()
    except Exception as e:
        if os.environ.get("SCHED_DEBUG"):
            print(f"[sched-sglang] apply deferred (SGLang will apply): {e}")
    if os.environ.get("SCHED_DEBUG"):
        print("[sched-sglang] hooks registered on Scheduler.run_batch")


def _dump_effective_config():
    """Print the RESOLVED mode once at registration -- the effective schedule
    is the product of many env vars, and an operator must be able to read it
    at a glance (not reverse-engineer it from the shell). Always on: config
    provenance is not debug-only."""
    e = os.environ.get
    compile_levers = [n for n, k in (("pi", "SCHED_NO_INDIRECT"),
                                     ("timer", "SCHED_NO_TIMER"),
                                     ("policy", "SCHED_NO_POLICY"),
                                     ("shed", "SCHED_NO_SHED")) if e(k) != "1"]
    if e("SCHED_WORKQUEUE"):
        compile_levers.append("clc" if e("SCHED_CLC") else "ticket")
    print("[sched-sglang] effective config: "
          f"mode={'observe' if _STATE['observe_only'] else 'enforce'} "
          f"weave={'+'.join(compile_levers)} "
          f"weave_only={e('SCHED_WEAVE_ONLY', 'all')} "
          f"timer={'device' if e('SCHED_TIMER_DEVICE', '1') != '0' else 'host'}"
          f"/every-{_STATE['timer_every']} "
          f"plan_every={_STATE['plan_every']} "
          f"clc_arm={_STATE['clc']}(imbalance>{_STATE['clc_imbalance']},"
          f"resid>{_STATE['clc_resid']}) "
          f"max_tasks={e('SCHED_MAX_TASKS', '4096')}")


# entry-point target: `sched = sched_sglang_plugin:plugin` in a package's
# [project.entry-points."sglang.general_plugins"], or call register() directly.
def plugin():
    register()
