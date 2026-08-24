# Branch and Worktree Map

Maintained by hand; update when a branch is created, merged, or retired.
This file describes development topology only; it is not part of the runtime
contract.

## Live branches

| branch | checkout | purpose | merge condition |
|---|---|---|---|
| `nonresident-acquisition` | parent history | pre-refactor implementation and sealed measurements | — |
| `refactor/late-bound-work-unit` | main repo (current checkout) | unified exact work-unit contract, engine adapters, and redesigned evaluation | current refactor branch |
| `consumer-contract` | `~/Dev/sched-pass-wt/contract` | independent consumer-contract worktree | do not modify from this checkout |
| `e6-conventional` | `~/Dev/sched-pass-wt/e6` | independent conventional-gather campaign worktree | do not modify from this checkout |
| `main` | `~/Dev/sched-pass-main` | original prototype history and recognition lineage | read-only |

## Rules

- One worktree per live unmerged branch, under
  `~/Dev/sched-pass-wt/<name>`; never modify another worktree while working
  on this branch.
- A branch merges only through an explicit review and a passing validation
  run.
- Delete a branch when merged; record it here.
