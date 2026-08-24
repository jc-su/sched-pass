# Document status

The canonical sources for the current branch are:

1. source code and tests;
2. [REFRACTOR_DESIGN.md](REFRACTOR_DESIGN.md);
3. [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md);
4. [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md);
5. build/CTest output and newly generated result artifacts.

The old prototype ledgers were removed from the working tree during this
refactor. They remain recoverable from Git history, but are intentionally not
part of the current design surface. This includes retired selected/tiered
serving notes, campaign ledgers, validation snapshots, and the duplicate
framework-integration note superseded by `ENGINE_INTEGRATION.md`.

The current tree therefore has no active document that defines a retired API,
activation flag, selector, or completion-resume claim. Earlier hypotheses and
measurements can be inspected with `git log` when provenance is needed.

New experiments must use the exact work-unit contract, record the Git
revision, protocol, granularity, demand trace, and activation counters, and
write a fresh artifact instead of appending to an old campaign ledger.
