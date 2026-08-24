# Branch and Worktree Map

Maintained by hand; update when a branch is created, merged, or
retired. Cleaned 2026-08-24: eleven fully-merged branches deleted
locally (their commits live on in `nonresident-acquisition` history;
same-named branches may still exist on `origin` until pushed deletions).

## Live branches

| branch | checkout | purpose | merge condition |
|---|---|---|---|
| `nonresident-acquisition` | main repo | THE working branch; every sealed result and doc | — |
| `consumer-contract` | `~/Dev/sched-pass-wt/contract` | in-kernel claim-consumer contract (ABI v28), fully built, fixtures passed on GPU | replay + quality batteries green with the check armed, then a registered mechanism-change entry |
| `e6-conventional` | `~/Dev/sched-pass-wt/e6` | E6 execution protocol (GPU selection + conventional gather), the decomposition arm of docs/CAUSAL_CHAIN.md; campaign queued from this worktree | after the E6 campaign seals, merge (mode is env-gated off) |
| `isolation-admission` | none | archival WIP: naive external-admission deferral, self-recorded as not validated and probably self-defeating | not planned; keep as recorded negative design |
| `main` | `~/Dev/sched-pass-main` | the original prototype (SchedWeave lineage), kept read-only as the recognition/integration design source (docs/RECOGNITION_LINEAGE.md) | never |

## Deleted 2026-08-24 (all fully merged into nonresident-acquisition)

`forward-profile` (co-tenant sampler + per-forward profiler — the
powered-campaign revision f0cecf1), `obs-hardening` (fail-open
observability), `chunk-ladder` (chunk knob), `extend-capture`
(capture negatives + composition counters, retained in history per
PREREGISTRATION), `b2-deferred-host` (B2 design + scaffold),
`audit-p2-debts`, `fast-gather`, `p31-host-orchestrated`,
`p33-virtual-recycling`, `writeback-summaries` (all pre-window feature
branches already absorbed).

## Rules

- One worktree per live unmerged branch, under
  `~/Dev/sched-pass-wt/<name>`, with `build` symlinked to the main
  tree's build — and never committed (see the build-symlink hazard in
  project memory: check `git ls-tree <branch> --name-only | grep -x
  build` before any merge).
- A branch merges only through its stated merge condition; mechanism
  branches additionally require their PREREGISTRATION entry before any
  qualifying campaign uses them.
- Delete a branch when merged; record it here.
