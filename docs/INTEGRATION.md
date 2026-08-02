# Kernel And Engine Integration

Public design documents call work whose inputs are available a runnable tile and
call its finite execution a runnable-work launch. `Ready` remains only the
literal ABI-v19 ticket state for a tile whose dependencies are available.

NTA is integrated at two independent boundaries: an engine publishes request
and object metadata, and a source-available finite GPU kernel calls one policy
function at a reconstructible work boundary. The benchmark executables are
self-contained validation programs, not the production API.

## Host Boundary

An engine keeps ownership of its allocations. Register existing HBM, mapped
host, or staged-host replicas with `HostRuntime::registerObject`; NTA neither
allocates nor frees those buffers. Owned `install*Object` methods remain useful
for standalone programs.

```cpp
nta::RegisteredReplicaSpec replica{
    engine_page_pointer,
    nta::Placement::Hbm,
};
runtime.registerObject(object_slot, object_id, version, page_bytes,
                       /*stagingDeviceAddress=*/nullptr,
                       std::span(&replica, 1));

nta::DeviceWorkPlan device_plan(work_capacity, dependency_capacity);
device_plan.uploadAsync(work_plan, stream);
```

`WorkPlan` is the engine-neutral batch contract. It maps canonical CTA work to
request slot/generation and to one or more versioned object ranges. A vLLM,
SGLang, FlashInfer, MoE, or ANNS adapter may form that plan; the runtime and
compiler do not branch on an engine or kernel name.

`FinitePhaseProgram` loads the common reset and bounded progress kernels from an
instrumented module. Backend progress directly appends newly runnable tickets;
the explicit publication entry point remains available for compatibility and
diagnostics. It enqueues into an ordinary CUDA stream or an existing graph
capture:

```cpp
nta::FinitePhaseProgram phases(module);
phases.enqueueHost(stream, runtime.deviceView(), config,
  [&] { launch_initial_kernel(device_plan.workItems(),
                              device_plan.dependencies(),
                              device_plan.workItemCount()); },
  [&] { launch_ready_kernel(device_plan.workItems(),
                            device_plan.dependencies(),
                            device_plan.workItemCount()); });
```

`enqueueHost` and `enqueueNvme` retire completed initial/ready launches in
stream order. For a shared-object JIT, `JitPhaseProgram` loads equivalent
exported launchers and verifies `nta::abi::Version` before use:

```cpp
nta::JitPhaseProgram phases(flashinfer_module_path);
phases.enqueueHost(stream, runtime.deviceView(), config,
  [&] { flashinfer_initial_run(); },
  [&] { flashinfer_ready_run(); });
```

The engine still owns graph lifetime, batch formation, output synchronization,
request cancellation, and generation retirement. NTA does not insert a CPU
submission or completion loop.

### C and Python boundary

`libnta-runtime.so` exports the versioned C API in `nta/RuntimeC.h`. It owns the
same `HostRuntime`, `DeviceWorkPlan`, `JitPhaseProgram`, and `NvmeTransport`
objects used by the C++ workloads; it is not a second runtime implementation.
Every operation returns a status and thread-local diagnostic, validates struct
size/API version, and leaves output handles null on failure. The API exposes
non-owning object and tensor-map registration, asynchronous plan upload,
work ticket state, phase launches, VFIO transport construction, NVMe object
installation, capabilities, and queue statistics.

The dependency-free `python/nta_runtime` module binds that API with `ctypes` and
accepts integer addresses from Torch or another CUDA framework:

```python
runtime = nta_runtime.Runtime(config)
runtime.set_request(slot, request_id, generation)
direct = runtime.register_object(
    object_slot, object_id, version, page_bytes,
    [nta_runtime.Replica(kv_page.data_ptr(), nta_runtime.Placement.HBM)],
)
plan.upload(work_items, dependencies, request_ranges, torch.cuda.current_stream())
```

The Python owner keeps an attached NVMe transport alive for the runtime, reuses
stable device-plan allocations, and accepts Torch stream objects without
importing Torch itself. `nta-python-runtime` tests this path with real CUDA
memory and cross-stream event ordering.

`Runtime.device_view_tensor`, `DeviceWorkPlan.work_items_tensor`, and
`DeviceWorkPlan.dependencies_tensor` are non-owning DLPack byte views. Their
Torch `data_ptr()` values are the native allocations, so FlashInfer receives
real tensors without copying the ABI or hand-packing benchmark structures.
The native owner must outlive every exported view.

For paged host pools, `registerIndexedHostObjectsAsync` and the equivalent
Python `register_indexed_host_objects(..., stream=stream)` publish source rows,
destination rows, element size, and both strides from a reusable pinned staging
ring. Publication is ordered on the engine's current CUDA stream; it does not
force a device synchronization or repack KV rows on the CPU.

### SGLang lifecycle adapter

The installed `sglang.srt.plugins` entry point registers `nta_flashinfer` for
SGLang 0.5.14. It translates SGLang request slots, HiCache page maps, and
FlashInfer schedule coordinates into this host boundary. Resident batches
retain stock FlashInfer; supported host-backed batches use the instrumented
wrappers; planning failures restore SGLang's original transfer path before
attention starts. Scheduler aborts are mirrored into the current request
generation. See `SGLANG.md` for the exact support matrix and command line.

vLLM has no registered adapter. vLLM 0.13's experimental KVConnector carries
request IDs and block-transfer metadata, but the stock offloading connector
finishes loading before attention and the Blackwell FlashInfer backend normally
uses TRTLLM entry points outside this overlay. A correct adapter must jointly
own that connector lifecycle, request-generation policy, kernel selection,
cancellation, and stream/graph lifetime; inferring missing state from batch
position or pointers is not accepted by this contract.

### Layer epochs

A work ticket describes one finite application-kernel invocation. It reaches
`Done` after that invocation's ready launch and must not be reused by another
transformer layer. An attention adapter therefore runs one stream-ordered epoch
per layer invocation, using a layer-specific plan slice or resetting the shared
work ticket allocation before the layer's initial launch:

```text
layer N: reset -> initial -> complete -> progress+publish -> ready -> complete
layer N+1: reset -> initial -> complete -> progress+publish -> ready -> complete
```

This rule prevents later layers from skipping work because an earlier layer
retired the same CTA index. A serving adapter must also keep the request
generation and page/object version stable through the whole epoch. Fixed graph
captures use a configured number of bounded progress rounds; non-graph control
paths must inspect work ticket state and fail the request if the bound expires.
The C++ `FinitePhaseProgram` and Python `BoundedEpoch` provide fixed graph
enqueue paths; Python `BoundedEpoch` also provides synchronized early-check
paths. The fixed path enqueues every GPU round and performs one bulk
work ticket transfer at the layer boundary. Its `enqueue_*_fixed` form does no
synchronization and can be captured; the engine calls `check` after graph replay
before consuming the layer result.

## Kernel Boundary

A source-available kernel needs one early hook after it knows canonical work
identity and before it initializes shared memory, barriers, TMA state, or live
numerical state:

```cpp
nta::kernel::WorkContext work{};
if (!nta::kernel::acquireWork(runtime, work_items, dependencies,
                              canonical_work_index, work)) {
  nta::kernel::defer(runtime, work);
  return;
}

// Existing kernel mainloop. Resolve each dependency where its native pointer
// or tensor-map descriptor is consumed.
auto *page = static_cast<const half *>(
    nta::kernel::address(runtime, work, dependency_index));
```

The call is CTA-uniform. The LLVM pass proves that the pending edge returns
before state that would have to survive a relaunch. It lowers the ready path to
direct pointer/descriptor selection and the miss path to bounded intent and
work ticket state. The kernel never waits for I/O.

For an NVMe transport, `RuntimeConfig::enableCtaNvmeTryIssue` enables a miss
latency fast path. The CTA leader attempts one queue lease only when that NVMe
backend has no older acquisition. Backend demand is counted before the lease
attempt, so concurrent misses cannot pass a request that is still publishing its
intent. Success constructs the SQE/PRPs, publishes the command context, rings
the SQ doorbell, and returns with the rest of the CTA. Lock contention, queue
credit, request credit, or existing NVMe work causes immediate intent
publication instead. The application CTA never reads a CQ or holds the lease
across I/O; `nta_progress_nvme` remains the only completion consumer. The
current direct path is capped at 32 destination PRP pages so a single CTA lane
performs at most bounded small-transfer setup; larger reads use the
warp-cooperative scheduled path. Setting the option to false provides the
scheduled-only ablation.

`NvmeTransportOptions::endpoint` must explicitly name
`vfio:DDDD:BB:SS.F`. The constructor owns the complete controller through a
VFIO cdev and private IOMMUFD IOAS; it does not coexist with a kernel NVMe or
SPDK qpair. Non-VFIO endpoints fail before queue setup. Construction
fails before queue publication unless the selected media policy, IOAS mapping,
CPU queue/DMA self-test, and an end-to-end GPU SQ-doorbell-CQ qualification all
pass. `RequireHardwareWriteProtection` is the default;
`TrustReadOnlyDeviceCode` explicitly supports dedicated controllers without
feature `0x84` and does not insert a CPU workload-I/O broker.

Object identity may be selected by CTA-uniform GPU computation. The pass accepts
catalog-derived request/object SSA values and rejects lane- or thread-derived
collective operands. This supports device-selected experts, sparse KV tiles, and
graph pages without claiming that opaque pointer arithmetic reveals object
semantics automatically.

## JIT Delivery

For source-generated CUDA, run the generator under the activation tool:

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 your_flashinfer_or_kernel_generator.py
```

The activator points `FLASHINFER_NVCC` at the Clang CUDA shim, loads
`libNtaPass.so`, and uses a cache directory fingerprinted by the pass, shim,
device integration headers, and ABI. FlashInfer or another nvcc-style JIT still
owns source generation, host binding generation, linking, and cache lookup.
`NTA_JIT_ONLY` can restrict instrumentation to comma-separated source-name
fragments; `NTA_REAL_NVCC` can compile all other sources with nvcc.

The pass is registered at Clang's optimizer-last extension point, so explicit
`opt -passes=nta-acquire` is unnecessary in this path. The shim preserves the
target architecture, optimization level, fast-math choice, line tables,
dependency output, and host compiler options used by the generator.

The instrumented kernel translation unit must also compile
`runtime/device/Acquire.cuh`. FlashInfer activation force-includes the runtime
only in paged-attention kernel instantiations and emits phase wrappers from
exactly one source per shared object. Planning and TVM-FFI binding sources stay
stock. `tools/flashinfer/prepare_overlay.py` copies the installed include tree
into the fingerprinted cache, checks the complete 0.6.12 tree plus patched-file
hashes and insertion anchors, then adds the pre-state wrappers. Creation is
process-locked and atomically published; reuse verifies the immutable overlay.
The installed package is never edited.

Python integrations can use `nta_runtime.attention_jit_args` to generate the
custom parameter ABI and `FlashInferLayerEpoch` to bind a native runtime and
uploaded plan to decode or paged-prefill `wrapper.run`. The adapter derives
external acquisition from each work item's direct-dependency count. For
split-K, initial and intermediate ready launches suppress reduction; the final
bounded ready launch performs the only merge.

`tools/flashinfer/schedule.py` is the only code that reads 0.6.12 `PlanInfo`
offsets. It extracts active request/KV-tile order and chunk size.
`planScheduledDecode` checks that schedule against the engine-neutral
CSR/object plan before upload.

## Transparency Contract

The build step is transparent for a supported source JIT: no custom offline
LLVM command sequence is required. Object registration is non-owning, and
finite phases fit inside an engine's existing stream or graph.

The semantic hook is intentionally not invisible. An arbitrary cubin has no
reliable request/object identity and no compiler-proven point where a CTA may
return without abandoning barrier or numerical state. Precompiled cubins are
therefore unsupported. Source-available kernels need the small pre-state policy
call above, either directly or through an upstream template hook.

FlashInfer 0.6.12's attention-variant interface has no callback that can reject
a whole CTA. This does not block NTA: the checked JIT overlay adds the hook to
the global decode and FA2 paged-prefill wrappers before shared memory or
numerical state. An upstream optional begin-work hook would remove the overlay
and private schedule adapter. This is source-JIT integration, not
zero-source-change binary instrumentation.

## Why Benchmarks Contain C++

The benchmark host files deliberately include facilities a serving engine
already has: argument parsing, deterministic tensor generation, tier
allocation, CPU references, error injection, timing, and result reporting.
Their kernel launch and phase-orchestration code now uses the same public APIs
above. Production integration should reuse engine allocations and kernels and
should not copy the benchmark harness.
