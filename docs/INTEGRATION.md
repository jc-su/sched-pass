# Kernel And Engine Integration

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

nta::DeviceWorkPlan device_plan(capacity);
device_plan.updateAsync(work_plan, stream);
```

`WorkPlan` is the engine-neutral batch contract. It maps canonical CTA work to
request slot/generation and to one or more versioned object ranges. A vLLM,
SGLang, FlashInfer, MoE, or ANNS adapter may form that plan; the runtime and
compiler do not branch on an engine or kernel name.

`FinitePhaseProgram` loads the common reset, bounded progress, and publication
kernels from an instrumented module. It enqueues into an ordinary CUDA stream
or an existing graph capture:

```cpp
nta::FinitePhaseProgram phases(module);
phases.enqueueHost(stream, runtime.deviceView(), config,
  [&] { launch_initial_kernel(device_plan.view()); },
  [&] { launch_ready_kernel(device_plan.view()); });
```

The engine still owns graph lifetime, batch formation, output synchronization,
request cancellation, and generation retirement. NTA does not insert a CPU
submission or completion loop.

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
continuation state. The kernel never waits for I/O.

## JIT Delivery

For source-generated CUDA, run the generator under the activation tool:

```bash
tools/jit/activate.py --build-dir build -- \
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
`runtime/device/Acquire.cuh`, which supplies the lowered slow path and finite
phase kernels. In a multi-source JIT it must be included in exactly that device
kernel source, not every host-binding source. The foreign-kernel test exercises
this complete single-translation-unit path. A future FlashInfer begin-chunk
integration must add the policy and device runtime to its kernel-instantiation
source while leaving planning and TVM-FFI binding sources unchanged.

## Transparency Contract

The build step is transparent for a supported source JIT: no custom offline
LLVM command sequence is required. Object registration is non-owning, and
finite phases fit inside an engine's existing stream or graph.

The semantic hook is intentionally not invisible. An arbitrary cubin has no
reliable request/object identity and no compiler-proven point where a CTA may
return without abandoning barrier or numerical state. Precompiled cubins are
therefore unsupported. Source-available kernels need the small pre-state policy
call above, either directly or through an upstream template hook.

FlashInfer 0.6.12 exposes request and KV-tile indices at the needed point, but
its current custom attention-variant interface has no begin-chunk callback that
can reject the whole CTA. JIT compilation is implemented and verified; placing
NTA deferral inside the optimized decode and paged-prefill CTAs still requires
an upstreamable begin-chunk hook. Claiming zero-source-change optimized
FlashInfer integration before that hook exists would be incorrect.

## Why Benchmarks Contain C++

The benchmark host files deliberately include facilities a serving engine
already has: argument parsing, deterministic tensor generation, tier
allocation, CPU references, error injection, timing, and result reporting.
Their kernel launch and phase-orchestration code now uses the same public APIs
above. Production integration should reuse engine allocations and kernels and
should not copy the benchmark harness.
