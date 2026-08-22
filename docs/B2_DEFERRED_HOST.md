# B2: the deferred one-step-stale host baseline (design of record)

Registered future work in RELATED_WORK ("the strongest conceivable host
system") and Tier 1 item five in the evaluation plan. Recorded before
implementation, 2026-08-22.

## What B1 pays that B2 must not

`stage_layer_host_orchestrated` (B1) pays, per (claim, layer) refresh,
ON the serve critical path: one device-to-host synchronization of the
selected page identities, host directory/index arithmetic, a pinned
upload, and the indexed transfer — then attention waits. The same-
revision campaign measured dense parity (0.982x) and 14.4x resident
interference. The standing objection: a smarter host system would move
that work OFF the critical path by serving one refresh stale.

## B2 design

- **Serve stale, build ahead.** At a layer's refresh step, the serve
  path adopts the PREPARED table for that (claim, layer) if its
  ready-event has fired, else keeps serving the current (stale) table
  without stalling; either way it enqueues a background build from the
  layer's CURRENT queries. Only the first-ever refresh (no table yet)
  blocks — the honest cold-start cost.
- **Background builder.** One host worker thread per engine: drains a
  queue of (claim, layer, free-page snapshot event); for each item,
  waits the snapshot event, runs B1's exact host arithmetic (same
  directory/index code), uploads indices non-blocking on a side stream,
  fires the SAME indexed-transfer primitive on that stream, records a
  ready event. Selection quality, transfer primitive, and bounded-cache
  discipline are identical to B1 by construction — only the placement
  of the control edge differs.
- **Double-generation staging.** The stale table must remain servable
  while the prepared generation stages, so B2 claims allocate 2x
  staging capacity, partitioned even/odd by refresh generation; a
  generation's slots recycle only after the next adoption. The extra
  memory is disclosed as B2's pipelining cost and flows into the
  physical-bytes comparison rather than being hidden.
- **Attestation.** Stats record adopted-fresh vs served-stale steps,
  staleness depth (refresh generations behind), cold-start blocks, and
  builder queue depth, so the arm's behavior is auditable per trial.
- **Quality.** Stale-by-one-refresh selection is a policy change gated
  by the same scored battery as every selection policy; divergence
  reporting stays armed.

## Routing

`NTA_SGLANG_HOST_ORCHESTRATED=deferred` selects B2 (`=1` remains B1).
The serve-path branch in `selected_form` routes to
`stage_layer_host_deferred`, which adopts-or-reuses and never
synchronizes on the critical path after warm-up.

## Fairness notes for the paper

B2 concedes nothing we would not concede: same selector, same budget,
same transfer primitive, extra memory disclosed, staleness measured.
If B2 closes most of B1's gap, the delegation claim narrows honestly to
what remains (identity round trips at claim creation, graph
compatibility, admission integration); if it does not, the per-layer
control edge was never the whole story and the ablation strengthens.
