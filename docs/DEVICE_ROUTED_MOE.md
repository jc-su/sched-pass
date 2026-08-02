# Device-Routed MoE Acquisition Fixture

Status: mechanism and crossover evidence, not the primary system workload

## Problem

Userspace can publish request lifetimes, tensor identities, placement, and
transport capabilities, but it cannot know an expert selected from GPU-produced
hidden state without synchronizing the route back to the CPU. The consuming
CTA knows the selected expert, but an ordinary kernel has no request-safe way to
turn a nonresident selection into asynchronous acquisition and later execute
the affected runnable tiles.

This semantic gap leaves two conventional choices:

1. copy the selected IDs to the CPU, form or validate an I/O plan, and resume
   GPU execution after a host-visible synchronization; or
2. preserve GPU execution by conservatively staging every expert that might be
   selected.

The first adds a control-path bubble. The second amplifies physical I/O as the
expert catalog grows. Early userspace prefetch cannot remove this tradeoff when
the exact dependency is produced after launch.

## Implemented Mechanism

The MoE path is one finite CUDA graph epoch:

```text
GPU hidden-state producer
  -> GPU top-k router
  -> device-built WorkItem + AcquireRequirement records
  -> compiler-lowered expert consumer CTAs
       direct expert: compute
       missing expert: publish bounded intent and exit
  -> finite host-DRAM progress CTAs
  -> data-availability publication
  -> runnable expert consumer CTAs
```

Userspace installs an expert capability catalog and request generations. It
does not compute or upload the selected expert IDs. One warp per token computes
real gate dot products and writes the canonical runtime ABI records consumed by
`nta_moe_tile_kernel`. The LLVM pass lowers that consumer's acquisition boundary
to direct-address classification, request-liveness validation, bounded intent
publication, and deferral before expert state is consumed.

There is no persistent kernel, CTA-wide polling, suspended CTA state, or CPU
completion thread. A missing CTA exits. Progress and runnable-work execution are
ordinary bounded kernels in the captured graph.

The intent pool is sized by the maximum active acquisition frontier,
`min(expert_count, token_count * top_k)`, rather than by the catalog size.
Exhaustion remains explicit and fails work safely.

## Matched Policies

`nta-moe-bench` runs the same GPU hidden-state producer, top-k route, expert
matrices, and numerical output under three controls. `late-bound` is retained as
a legacy command-line value for reproducibility:

- `late-bound`: device-built requirements flow directly into compiler-lowered
  finite acquisition;
- `cpu-sync`: selected IDs are copied to pinned host memory and the stream is
  synchronized before the same acquisition path. The device has already built
  the plan, so this is a generous lower bound on CPU-mediated routing;
- `overfetch`: all staged experts are copied with no-allocate vector operations,
  concurrently with hidden-state production and routing, after which the
  identical routed expert compute runs without acquisition bookkeeping.

All policies regenerate hidden states on the GPU each epoch. Correctness is
checked against a CPU matrix-vector reference using the final GPU-selected
route and hidden state.

## Development Result

On the current RTX PRO 6000 Blackwell host, a randomized 10-process trial used
512 staged experts, top-2 routing for eight tokens, hidden size 256, and 50
measured epochs per process. Median epoch latency was:

| Policy | Median latency | Final measured epoch transfer |
| --- | ---: | ---: |
| late-bound | 0.540 ms | 4 MiB |
| CPU sync | 0.553 ms | 4 MiB |
| overfetch | 2.669 ms | 128 MiB |

The paired median ratios were 1.023x for CPU-sync/late-bound and 4.941x for
overfetch/late-bound. The 95% confidence-interval half-widths were 0.0014 and
0.0060 respectively.

The same randomized specification measured the all-resident direct-path
control: late-bound median latency was 0.418 ms versus 0.407 ms for the direct
consumer, a 1.027x ratio or 2.7% overhead (95% interval half-width 0.0022).
All 50 process runs had zero numerical failures.

These are developmental results from a dirty worktree with uncontrolled GPU
clocks. They establish that the implemented mechanism can close the two-sided
route/I/O visibility gap and can avoid large sparse-demand transfer
amplification. They are not production-serving or OSDI qualification evidence.

Reproduce the randomized experiment after building:

```bash
./scripts/run-qualified-trials.py \
  --spec experiments/moe-late-bound.json \
  --output-dir results/moe-late-bound
```

The runner requires a clean revision unless `--allow-dirty` is explicitly used
for development.

## Novelty Boundary

This fixture validates device-generated request/object binding,
generation-safe non-polling acquisition, and runnable-work execution with a
native direct path. Those mechanisms support the incremental-operator co-design
in `SYSTEM_PLAN.md`; this standalone MoE path is not the primary contribution.

The mechanism benchmark supplies genuine device-generated demand. A systems
paper still requires production incremental-operator integration, end-to-end
serving SLO results, matched prior-system baselines, dense-demand and resident
crossover experiments, controlled clocks, multiple machines, and clean-host
artifact reproduction.
