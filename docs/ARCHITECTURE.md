# Architecture

Status: compatibility entry point for `refactor/late-bound-work-unit`

The canonical architecture for this branch is
[REFRACTOR_DESIGN.md](REFRACTOR_DESIGN.md).  This file is intentionally short:
it prevents older implementation notes from being mistaken for the current
contract while preserving the familiar architecture link used by scripts and
people.

## Mechanism

NTA binds each heterogeneous work unit to three facts as late as correctness
permits:

1. request identity, including slot and generation;
2. exact demand and its epoch; and
3. availability and dependencies.

It then executes bounded runnable groups at a measured-cost granularity.  A
direct launch is the all-ready special case.  Late-bound execution exposes
ready groups while other units are blocked.  Partial execution adds an exact
consumer state that can be continued later.  These are protocol forms of one
mechanism, not separate schedulers.

## Layer ownership

```text
engine adapter       request IDs, slots, generations, cancellation, graphs
demand provider      exact selected IDs; approximate selectors are explicit
protocol planner     granularity, grouping, overlap, partial-form choice
NTA runtime          claims, dependencies, staging, transport, retirement
compiler plugin      marker validation and work-coordinate mapping
consumer operator    numerical partial state and exact merge
```

The runtime is engine-neutral.  SGLang and vLLM translate their metadata at
the adapter boundary; they do not create duplicate identity or staging
policies.  The native ABI in `include/nta/RuntimeABI.h` remains the storage
contract, while the Python contract in `python/nta_runtime/` carries semantic
policy.

## Invariants

- A stale slot/generation cannot transition or publish work.
- A stale demand epoch cannot publish selection or completion.
- Approximate demand never satisfies an exact protocol implicitly.
- All protocol forms use the same work IDs and demand trace in a comparison.
- Granularity is chosen from transfer, compute, control, and availability
  exposure costs; it is not a quality oracle.
- Engine-specific code stays in adapters; compiler and runtime code do not
  branch on SGLang or vLLM.

## Migration status

The branch currently has the semantic work-unit contract, checked ledger,
cost model, SGLang identity adapter, and vLLM adapter seam.  Full SGLang
staging/selection migration, conventional-E6 integration, compiler/consumer
contract wiring, and a pinned vLLM implementation remain explicit work items.
See [DOCUMENT_STATUS.md](DOCUMENT_STATUS.md) and
[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the current boundaries.
