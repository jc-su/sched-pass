# GPU Behavior Unboxing Map

Date: 2026-07-07.

This project can instrument more than CLC. The useful goal is not to observe
everything all the time; it is to expose enough GPU behavior to decide when to
enable each scheduling lever without letting instrumentation overhead dominate
the result.

## Instrumentation Surfaces

| Behavior to unbox | Existing / near-existing signal | Cost class | Design decision |
|---|---|---:|---|
| Per-request/task cost | Woven `clock64` timer, `__sched_timer[task]`; `SCHED_TIMER_INDIRECT`; `ctrl.flags` timer gate | medium to high if host-mapped every step; low with device buffer or sampling | Build `task_order[]`; decide estimator confidence |
| Task-order correctness | `task_order[]` permutation tests, FlashInfer armed-pi test, fixed-VA test | low | Prove cross-request remap changes when/where, not what |
| CTA residency and mapping | `%smid`, `%globaltimer`, CTA tuple probes, CLC mapping/pressure probes | research-only | Estimate `R`; understand prefix stripes and suffix execution locality |
| Claim behavior | CLC runtime/trace/noise/pipeline probes | research-only | Decide when CLC adds value beyond `task_order[]` |
| Occupancy and `R` | CUDA occupancy APIs; cubin resource extraction; Nsight LaunchStats/Occupancy | low to research | Gate CLC on `N > R`; choose CTA granularity |
| Timer channel overhead | `python/eval_timer_channel.py`, `python/test_timer_gate.py`, `python/test_timer_indirect.py` | research, then production-gated | Choose host-mapped vs device-buffer observation |
| Cache / memory policy | Woven prefetch and L2 hints; `test/probe_instr.sh`; Nsight Compute L2/DRAM counters | medium/research | Decide when hints/policy pay |
| Memory-bound regimes | `experiments/clc/clc_membound_probe.cu`; decode evals; Nsight DRAM/L2 counters | research | Avoid enabling CLC/policy when bandwidth-saturated |
| Quality shedding | `tau`, shed masks, `python/eval_quality.py`, softmax fixture | low to medium | Trade quality for SLO under overload |
| Programmatic dependent launch | `SCHED_PDL`, `test/pdl_overlap.cu` | low if launch path opts in | Hide control-table read latency behind producer tail |
| Inter-kernel behavior | pressure probe, Nsight Systems timeline | research | Avoid assuming CLC or priority preempts other kernels |
| CUDA Graph behavior | CLC graph probe; baked ABI fixed-VA tests | low/research | Check graph replay safety |
| Serving behavior | `python/test_serving_gate.py`; live SGLang TPOT/throughput/load generator | production eval | Ship/no-ship gate |
| Generated ISA | `dump_generated_isa.sh`, `cuobjdump`, FileCheck IR gates | compile-time | Confirm intended PTX/SASS exists |

## Production Signals

Use these in the real serving loop:

```text
batch shape:
  ntiles / requests / heads / split-kv count

R estimate:
  occupancy-derived active CTA prefix for the actual cached kernel

task cost estimate:
  sampled timer, preferably device-buffer or gated host-mapped timer

policy decision:
  task_order[] every step
  CLC armed only if estimator uncertainty is high and ntiles > R
  timer enabled only on probe/sample steps
```

This gives the scheduler enough feedback while keeping steady-state overhead
small.

## Research Probes

Use these offline or in short calibration windows:

```text
1. Nsight Compute:
   LaunchStats, Occupancy, SchedulerStats, WarpStateStats,
   MemoryWorkloadAnalysis, L2/TEX/DRAM counters.

2. Nsight Systems:
   kernel timeline, queue wait, PDL/graph/stream interactions.

3. Device-side probes:
   smid/globaltimer mapping, CLC claim traces, timer channel ablation,
   memory-bound and noise-threshold sweeps.

4. Generated code:
   PTX and SASS checks for clusterlaunchcontrol, griddepcontrol,
   prefetch.L2 eviction hints, and shed masks.
```

Do not run these continuously in production. They are calibration tools.

## What We Can Unbox Next

The CLC primitive is already characterized. The most valuable next unboxing is
serving behavior:

```text
SGLang live serving run:
  collect TPOT p50/p99, throughput, batch size, ntiles, R, timer sample rows,
  chosen policy, CLC armed/off, and output/error checks.

Counter pass:
  for identity, LPT, LPT+policy, CLC-armed/off:
    collect L2 hit rate, DRAM throughput, achieved occupancy,
    eligible/no-eligible warp stalls, and instruction count.

Timer channel decision:
  compare host-mapped sampled timer vs device-buffer timer on serving-shape
  kernels; use device-buffer if readback cost is acceptable.

Cache policy decision:
  verify whether urgent/polite hints improve L2 behavior on real FlashInfer
  kernels; keep disabled if counters show no benefit.

PDL decision:
  test whether consumer control-table reads can overlap a real producer tail,
  not only the synthetic fixture.

sm_100 validation:
  repeat R/noise/tie-boundary measurements on GB200/B200-class silicon.
```

## Acceptance Rules

For any new instrumentation:

```text
semantic safety:
  bit-exact output unless the experiment intentionally enables tau shedding

cost visibility:
  always report overhead vs a disabled/control variant

policy usefulness:
  the signal must change a scheduling decision, not just produce telemetry

production gate:
  steady-state path must be cheap; expensive signals must be sampled or offline
```

## Recommended Design Direction

```text
Default:
  task_order[] every decode step

Observation:
  timer sampled or device-buffered, not host-mapped every step

CLC:
  off when N <= R
  off when estimator confidence is high
  on only when N > R and residual/noise says late binding may beat static pi

Policy / cache hints:
  disabled by default until counters prove benefit on real serving traces

Shed:
  SLO/overload lever only, guarded by quality curves
```

