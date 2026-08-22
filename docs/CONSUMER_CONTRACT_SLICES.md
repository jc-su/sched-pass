# Consumer-Contract Build State (branch consumer-contract)

Slices 1-3a are committed and compile-verified. This file pins the
remaining integration so it is executed, not re-derived.

## Done
- ABI v28 `ClaimContext` in `RuntimeView` (fits in former padding; view
  stays 320B; golden-offset test updated) — `11e81bd`.
- `HostRuntime::publishClaim` + C API v31 + Python `publish_claim`;
  engine publishes at activate (valid=1) and all three retire paths
  (valid=0, same generation) — `27fa2e5`.
- `bindValidatedClaimConsumer` in the policy header + overlay injection
  at all four request-bound guard sites, gated by `HasClaimBindingV`;
  reworked to a per-request bindings table (four int64 words per
  request: slot|-1, generation, row bound, stamp) because one launch
  serves residents plus up to sixteen claims — `6ca0b18`, `19f660d`.
- Claim-bound module form in `flashinfer.py`
  (`claim_bound_attention_jit_args`: request-bound tensors +
  `nta_claim_bindings` int64 tensor).

## Remaining (3b): arm the check on the tiered serve path
1. **Wrapper family.** In `engines/sglang.py` `decode_wrappers` /
   `prefill_wrappers`, add policy `"claim_bound"` using
   `claim_bound_attention_jit_args`; build
   `_nta_claim_bound_decode_wrappers` (+ prefill analog). Dense
   request-bound wrappers stay untouched.
2. **Routing.** Tiered batches (claims present) select the claim-bound
   wrapper where the serve wrapper is chosen today; `verifier = wrapper`
   in `_build_multi_claim_ctx` then carries it.
3. **Bindings buffer.** Per-wrapper stable int64 buffer
   `[max_batch, 4]` (graph-stable storage, like `_tg_cats`):
   - eager fill in `_build_multi_claim_ctx` after `claim_entries`:
     position with claim -> {table_slot.index, table_slot.generation,
     kept_rows bound, stamp}, else {-1,0,0,0};
   - tiered-graph epoch fill refreshes it in place beside the existing
     `_tg_cats` per-layer refresh (`init_forward_metadata_out_graph`);
     tail-default rows are {-1,0,0,0}.
4. **Run sites.** `_run_paged` (and the graph-capture run path) pass the
   bindings buffer as the second tensor arg for claim-bound wrappers.
   FlashInfer passes tensors positionally before scalars.
5. **Stamp scaffolding.** Until prep stamps device-side (slice 3c),
   engine and bindings both use stamp 0; `publish_claim` already writes
   tableStamp=0, so the check is exact and trivially satisfied on the
   stamp word while remaining armed for slot/generation/valid/bound.

## Remaining (3c): stamping
Engine increments a per-claim stamp at every selection epoch
(refresh), republishes the row, and writes the same stamp into the
bindings buffer; later depth moves the stamp write into the prep
kernel itself.

## Remaining (4): reject fixtures
Standalone probe (pattern: ExtendStagingCaptureProbe): build one claim,
serve once (accept), then five mutations each asserting REFUSAL:
stale generation (bump table generation, keep bindings),
retired slot (publish valid=0), foreign slot (bindings point at another
live claim's slot with mismatched generation), out-of-extent row bound
(bound > stagedRows), stale stamp (mismatch stamp word). Register in
ctest as nta-claim-consumer-fixtures (GPU).

## Gate before merge
Replay battery + quality battery green with the claim-bound wrapper
armed; PREREGISTRATION mechanism-change entry recorded before any
qualifying campaign runs on it.
