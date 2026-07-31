# Nonresident Acquisition

This branch is a clean-slate implementation of request-semantic external data
acquisition for finite GPU kernels.

The primary application is SLO-critical external KV access in continuously
batched attention. The mechanism also targets GPU-routed MoE experts and, as a
secondary generality case, graph or ANNS objects.

The architecture contract and implementation sequence are defined in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Design baseline only. No code from the previous scheduling prototype is carried
in this branch. The validated prototype remains available on `main` at commit
`4789f16`.

Implementation starts with the IR contract and a deterministic mock transport.
Real host-memory, NVMe, TMA, and RDMA paths are admitted only after the
corresponding correctness and overhead gates in the architecture document pass.
