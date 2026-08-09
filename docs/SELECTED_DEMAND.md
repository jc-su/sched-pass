# Selected-Demand Serving (1D) Implementation Plan

Status: declared plan; implementation not started. This is the campaign
centerpiece defined in `ONE_GPU_EVALUATION.md` (RQ1/1D). Measured inputs it
builds on: near-oracle envelope selection quality with sink/recent retention
(`results/serving/quest-recall-*.json`), the 8.17x/2.14x selected-page
acquisition crossovers with exact parity, and the finding that selection
economics require 16K+ contexts (short-context attention is inherently
diffuse). Every stage below lands behind a health check before any timed run.

## Goal

One end-to-end serving experiment: SGLang decode where each step's attention
reads only device-selected KV pages, the unselected pool stays in host tiers,
selected pages move through the existing indexed acquisition path, and output
quality is evaluated against dense attention. Gate: `>=1.5x` goodput at
quality parity against the strongest baseline, n=10, clean revision.

## Design

1. **Page-summary maintenance.** Per (layer, kv-head, page): key min/max
   envelopes in HBM, updated incrementally when pages are appended or
   promoted; invalidated with page eviction through the existing generation
   machinery. Envelope updates are pure elementwise min/max over the new
   tokens: no rescan of resident pages.
2. **Per-step selection.** At each decode step and layer, score envelopes
   against the live query (`quest_selector.quest_page_scores`, aggregated
   across the head group because the acquisition unit is a whole page), always
   retain sink and recent pages, take top-k under the per-request page budget.
   Selection output is the device-resident selected-index tensor the existing
   `register_selected_host_pages`/`build_selected_page_work_plan` machinery
   already consumes.
3. **Acquisition and caching.** Selected pages resolve through the tier
   directory: HBM-resident pages consume directly; host/CXL/NVMe pages
   acquire through the indexed gather path into a staging cache keyed by
   (page, version) with reuse across steps (temporal locality of selection is
   the expected hit source; report the measured hit rate).
4. **Quality evaluation.** Same checkpoint, dense versus selected, on
   LongBench-style long-context tasks at 16K+; report task metrics and the
   per-layer attention-mass recall of the deployed selector at the chosen
   budget. Quality parity is a gate, not an assumption.

## File-by-file

- `python/nta_runtime/quest_selector.py`: add incremental envelope update
  (`update_page_envelopes(envelopes, appended_tokens)`) and a budgeted
  selection helper with sink/recent retention; property tests extend
  `tests/runtime/quest_selector.py` (update equivalence against full
  recompute; retention invariants).
- `python/nta_runtime/engines/sglang.py`: a `selected` attention form beside
  `preloaded`/`incremental`: per-step selection on the compute stream, staging
  cache lookup, indexed acquisition for misses, compact-table FlashInfer run.
  Envelope maintenance hooks where HiCache writes device pages (the same
  mutation points the bridge already observes).
- `python/nta_runtime/engines/sglang_hicache.py`: expose page-write events to
  the envelope maintainer; no transfer-path changes.
- `benchmarks/serving/SglangSelectedLoad.py`: the 1D load harness (long-prompt
  requests over a host-tiered pool, measured promotions, budgets swept), plus
  the baseline arms: dense full-promotion, overfetch-candidates, host-side
  selection with the identity round trip, prediction-based prefetch.
- `benchmarks/serving/QuestRecall.py`: long-context mode using the verified
  logit-reconstruction path (no materialized attention) for 16K-32K recall.

## Health-check sequence (each stage gates the next)

1. Envelope update equivalence tests (CPU+CUDA, no engine).
2. Selected form on a resident-only batch: selection runs, output exactly
   matches dense attention over the same selected set, zero acquisition.
3. Selected form with host-tiered pages at one budget: exact output versus a
   reference computed on the same selection; staging-cache hit accounting
   conserved.
4. Short timed smoke with mechanism counters asserted (selected launches,
   bytes avoided, zero fallback) before any qualified series.

## Boundaries

Per-KV-head selection (finer acquisition unit), CXL as a distinct credited
tier, and NVMe-backed candidate pools compose later through the same
directory; none blocks the first 1D result. Speculative or learned selectors
are out of scope: the claim is about serving device-generated demand, not
about inventing selectors.
