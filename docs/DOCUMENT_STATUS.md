# Document Status

The refactor branch has a smaller source-of-truth set:

1. Code and tests.
2. [REFRACTOR_DESIGN.md](REFRACTOR_DESIGN.md).
3. [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).
4. [ARCHITECTURE.md](ARCHITECTURE.md) as a short compatibility pointer.
5. [VALIDATION.md](VALIDATION.md) and result artifacts as historical evidence.

The following documents contain earlier hypotheses, implementation ledgers, or
pre-refactor claims.  They are retained for provenance but must not be used to
choose new code structure or paper claims until reconciled with the two
documents above:

- `SYSTEM_PLAN.md`
- `OSDI_EVALUATION_PLAN.md`
- `ONE_GPU_EVALUATION.md`
- `PREREGISTRATION.md`
- `CAUSAL_CHAIN.md`
- `SELECTED_DEMAND.md`
- `ARCHITECTURE.md` from the parent branch (the current file is only a short
  compatibility pointer)

When a historical document disagrees with code or the refactor documents, the
historical statement is stale.  New experiments must be recorded in
`EXPERIMENT_DESIGN.md` and a fresh artifact directory, not appended to an old
campaign ledger.
