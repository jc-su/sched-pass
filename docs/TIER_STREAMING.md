# Exact partial execution

Status: optional operator-level form of the current work-unit mechanism.

Tier streaming is not a second serving policy. It is the `partial` protocol
form applied when an exact consumer can publish a mergeable intermediate state
and continue with later contributors. The work remains bound to request slot,
generation, epoch, and reduction group throughout the continuation.

The implementation is split by responsibility:

- `python/nta_runtime/tier_streaming.py` plans request-owned segments and
  bounded waves;
- `python/nta_runtime/flashinfer_tier_streaming.py` executes exact
  copy/compute/merge waves and lifecycle fences;
- `python/nta_runtime/execution_core.py` and
  `python/nta_runtime/execution_protocol.py` define the shared identity and
  availability contract;
- `benchmarks/serving/FlashInferTierStreaming.py` is an operator-level
  differential test, not a serving claim.

The partial form is valid only when the workload exposes a real continuation
opportunity. Otherwise the exact late-bound form or conventional baseline is
the appropriate arm. No arm may drop contributors, change the numerical
contract, or use a quality selector to manufacture a speedup. The D6
diagnostic profile therefore measures a protocol capability and is not
silently treated as the source of the SGLang headline result. Formal serving
attributes progressive release through A3/A2.

The current SGLang serving path creates one `ExecutionSession` per real
FlashInfer attention launch. The tier-streaming executor is a reusable
operator-level implementation using the same exact work-unit contract; its
historical performance snapshots are not current serving evidence.

Evaluate this form with the shared exact demand trace in
`experiments/heterogeneous-work-unit.json`, isolating partial continuation
from request heterogeneity, transport bytes, and granularity.
