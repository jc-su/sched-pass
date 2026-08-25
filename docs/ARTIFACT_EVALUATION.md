# Artifact evaluation and reproduction

The repository has an explicit source boundary:

| Layer | Paths | Responsibility |
| --- | --- | --- |
| implementation | `include/`, `lib/`, `runtime/`, `kernel/`, `tools/`, `python/nta_runtime/` | compiler pass, native ABI, device/runtime transports, JIT/build integration, semantic execution core |
| experiment drivers | `experiments/`, `benchmarks/`, `scripts/` | workloads, trial orchestration, measurements, artifact validation |
| correctness tests | `tests/` | deterministic pass/runtime correctness gates; no experiment output is stored there |

Experiment drivers may import the implementation under test. They must not
reimplement runtime state transitions or become a second runtime API. The
canonical dependency-free matrix is under `experiments/`; CTest invokes only
its smoke entrypoint, `experiments/smoke_matrix.py`.

## Reproduction bundle

`experiments/reproduce.py` creates an output directory containing:

```text
metadata.json       # revision, dirty state, source/diff digests, machine, packages
commands.json       # structured command records
logs/               # exact stdout/stderr for every command
matrix.json         # core/matrix profiles only
evaluation-manifest.json  # exact tier/RQ/fairness/statistical contract
```

The command records in `metadata.json` include the exact argv, working
directory, explicit environment overrides, return code, duration, and log
path. Generated data is written to the requested artifact directory, never to
the source tree or CMake build tree.

The output directory must be external to the checkout and empty (or absent);
the wrapper refuses to overwrite an older run. Native profiles likewise use a
fresh external build directory, either supplied with `--build-dir` or created
next to the artifact directory.

The machine-readable profile contract is
`experiments/artifact-manifest.json`.
Every completed bundle can be checked independently with
`python experiments/validate_bundle.py /path/to/bundle`.

An `osdi-complete` evaluation additionally requires a separately captured
performance evidence directory. It contains the successful `profile.json`,
the same-machine `baseline.json`, the measured correctness report
`measured.json`, and the machine-generated `regression.json`. Validate it with
`experiments/validate_performance_artifact.py`, then pass it to
`reproduce.py --performance-evidence`; a missing profiler is not a passing
performance gate.

Before selecting a physical tier, capture a read-only capability inventory:

```bash
python experiments/reproduce.py \
  --profile hardware \
  --output /tmp/nta-artifacts/hardware
python experiments/validate_bundle.py /tmp/nta-artifacts/hardware
```

This records GPU visibility, NVMe driver/IOMMU ownership, `/dev/dax*`
visibility, and the CXL topology state. In particular, it distinguishes a
root-decoder-only platform from an unconfigured memory endpoint, an existing
region, and an exposed devdax endpoint. It does not bind PCI devices, open
block namespaces, create/delete CXL regions, or qualify a tier. Qualification
remains an explicit, separately validated operation.

If the kernel modules are not already loaded, the non-destructive software
bring-up is:

```bash
sudo modprobe cxl_mem
sudo modprobe dax_cxl
cxl list -M -i
cxl list -R
daxctl list -u
```

These commands can expose an endpoint that firmware has already enumerated;
they cannot create a CXL Type-3 device or safely invent a DAX region. An empty
memdev/region list therefore remains a platform-capability result, not a
permission problem that the test should bypass.

Serving physical tiers also require an immutable exact page catalog. Validate
it without opening the endpoint before launching the worker:

```bash
python3 experiments/validate_tier_catalog.py /path/catalog.json --tier nvme
# or: --tier cxl_dax
```

Set `NTA_SERVING_TIER`, `NTA_TIER_CATALOG`, and the corresponding explicit
endpoint in the serving environment. The worker records the catalog digest in
its engine statistics; a physical-tier artifact is invalid if that digest or
the transport capability evidence is missing. The catalog maps SGLang device
page IDs to exact contiguous K/V byte ranges. It is not a sampling policy and
does not approximate attention.

The evaluation profile is self-contained with respect to normalized workload
inputs: it copies `manifest.json` and `records.jsonl` into `workload/` and
rewrites the copied trial specification to use them.  Physical NVMe/DAX
experiments additionally require the validated `tier-qualification.json`; the
bundle validator checks that it is qualified for every requested physical tier.

The Bailian preparation contract is independent of the native build and is
validated with `experiments/validate_workload.py`. Keep the normalized
`manifest.json` and `records.jsonl` beside the serving result in the external
artifact directory; do not copy anonymous source data into the repository.

### Core contract smoke

This needs only Python and the standard library in addition to the checked-out
runtime package:

```bash
python experiments/reproduce.py \
  --profile core \
  --output /tmp/nta-artifacts/core
```

The runner executes a small B0--B6/all-ablation matrix and the validator checks
shared exact demand traces, activation counters, tier/granularity strata, and
Little's Law accounting.

### Full matrix

Use a clean checkout and choose the size explicitly:

```bash
python experiments/reproduce.py \
  --profile matrix \
  --max-cases 128 \
  --repetitions 10 \
  --output /tmp/nta-artifacts/matrix
```

The matrix timing is labeled modeled regime data and cannot be cited as
serving performance.

### Native and GPU validation

```bash
python experiments/reproduce.py \
  --profile test \
  --cuda auto \
  --build-dir /tmp/nta-artifact-build \
  --cmake-arg=-DLLVM_DIR=/path/to/matching/llvm/lib/cmake/llvm \
  --output /tmp/nta-artifacts/test
```

`LLVM_DIR` must name the same LLVM installation that provides the `opt`,
`llc`, and (for CUDA builds) `clang++` tools. This is important on hosts with
multiple LLVM installations; the pass plugin is an in-process C++ extension
and cannot be mixed across LLVM builds. The exact CMake argument and tool
versions are retained in the artifact logs and metadata. With the production
LLVM package, the CUDA frontend and `llc` use `-O3`; the CMake configuration
automatically selects the conservative `-O0` fallback only for the known
clangir development toolchain whose optimizer currently asserts on these
device ABI types.

When no `--cmake-arg=-DLLVM_DIR=...` or `LLVM_DIR` environment override is
provided, `reproduce.py` prefers the validated packaged LLVM 22 CMake package
(`/usr/lib/llvm-22/`) and then falls back to another packaged
`/usr/lib/llvm-*/` package. This avoids accidentally pairing the pass with an
unqualified development `opt` discovered under `/usr/local`; the selected
package and every tool version remain recorded in the bundle.

Capability-gated tests report explicit skips. A skipped physical CXL DAX test
means no qualified devdax endpoint was supplied; it is not a passing CXL
qualification. The always-on `nta-cxl-dax-allocator` test still verifies the
bounded reservation ledger, reuse, coalescing, and input rejection without
claiming that ordinary host memory is CXL. Likewise, `nta-nvme-discovery` and
`nta-nvme-safety` verify the read-only candidate and ownership contracts on
every machine, while only `nta-vfio-nvme-probe` and the qualification runner
can qualify the GPU-controlled NVMe data path. The same rule applies to
multi-GPU and framework-specific serving tests.

`nta-cxl-dax-discovery` and `nta-nvme-discovery` always execute the read-only
capability-discovery path, including a temporary sysfs/devdax fixture, and
report `candidate=none` for the live platform when it has no usable endpoint.
`nta-physical-tier-capability` also executes the live, read-only CXL module and
CLI inventory on every host; it records whether the plumbing is loaded and
whether a devdax inventory is visible without treating that state as physical
qualification. The module bring-up loads `cxl_pci` in addition to the CXL
memory and DAX modules, which is required for PCIe CXL.mem enumeration.
The physical mapping probes remain capability-gated because turning an absent
endpoint into a passing result would invalidate the artifact. On a new host,
`./scripts/nta-cxl-dax-module.sh load` loads the read-only kernel plumbing and
prints the CXL memdev/region and devdax inventory; it does not create regions or
reconfigure namespaces.

For an already provisioned devdax endpoint, install
`config/udev/99-nta-dax.rules`, create the system group `dax`, and add the
artifact user to that group. Reload udev rules (or reboot) before qualification.
For example:

```bash
sudo groupadd --system --force dax
sudo usermod --append --groups dax "$USER"
sudo install -m 0644 config/udev/99-nta-dax.rules \
  /etc/udev/rules.d/99-nta-dax.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=dax --action=change
```

Start a new login session after changing group membership.
This changes access control only; it does not create a CXL region, reconfigure
a namespace, or initialize device media. A missing `/dev/daxX.Y` must therefore
be fixed in the platform CXL enumeration path rather than hidden by permissions.

The last recorded hardware inventory had no `/dev/dax*` endpoint, so its DAX
tests were correctly skipped. Always use a fresh read-only hardware inventory
for the machine and kernel used by an artifact; the group/rule establishes
permission readiness only and must not be reported as tier qualification.

### Physical NVMe-to-HBM qualification

The direct NVMe path needs the kernel-specific NVIDIA peer-page bridge in
addition to the native build. Build and load it against the exact running
kernel and NVIDIA driver:

```bash
./scripts/nta-nvme-p2p-module.sh load
./scripts/nta-nvme-p2p-module.sh status
```

Then qualify one dedicated, unmounted controller. This operation temporarily
rebinds the selected PCI function to `vfio-pci`, issues READ commands only, and
restores the original driver even when the benchmark fails:

```bash
python3 scripts/run-nvme-qualification.py \
  --bdf 0000:d8:00.0 \
  --dma-target hbm-peer \
  --media-policy trusted-read-only-code \
  --bytes 2097152 --requests 32 --progress-rounds 1 --iterations 100 \
  --fio-runtime 10 --minimum-bandwidth-ratio 0.9 \
  --allow-device-rebind --require-ready \
  --output /tmp/nta-artifacts/nvme/nvme-qualification.json

python3 experiments/qualify_tiers.py \
  --required-tier nvme \
  --nvme-report /tmp/nta-artifacts/nvme/nvme-qualification.json \
  --output /tmp/nta-artifacts/nvme/tier-qualification.json

python3 experiments/validate_tier_qualification.py \
  --required-tier nvme \
  /tmp/nta-artifacts/nvme/tier-qualification.json
```

When several physical tests will run and the dedicated controller is not being
used by another process, add `--keep-vfio` to the qualification command. A
successful run then retains the explicit VFIO ownership so later tests do not
repeat the bind/restore cycle; a failed run still restores the previous driver.
Configure CTest with the same endpoint and immutable reference file, then run
only the root-required physical tests through `sudo` if the peer bridge device
is root-only:

```bash
cmake -S . -B build \
  -DNTA_TEST_NVME_ENDPOINT=vfio:0000:d8:00.0 \
  -DNTA_TEST_NVME_REFERENCE=/tmp/nta-nvme-reference.bin \
  -DNTA_TEST_NVME_MEDIA_POLICY=trusted-read-only-code \
  -DNTA_TEST_NVME_DMA_TARGET=hbm-peer
sudo -E env NTA_NVME_ENDPOINT=vfio:0000:d8:00.0 \
  NTA_NVME_REFERENCE=/tmp/nta-nvme-reference.bin \
  NTA_NVME_MEDIA_POLICY=trusted-read-only-code \
  NTA_NVME_DMA_TARGET=hbm-peer \
  ctest --test-dir build -R 'nta-(vfio-nvme-probe|paged-attention-nvme-gpu)'
```

The JIT launcher namespaces FlashInfer caches by Unix UID, so privileged
physical tests cannot leave root-owned compilation outputs in a later user
framework run.

Use `hardware-write-protect` unless the controller reports no Namespace Write
Protection support. `trusted-read-only-code` is an explicit weaker boundary for
a dedicated experiment controller; it relies on the generated device program
emitting only NVMe READ commands. Neither policy authorizes formatting,
filesystem changes, writes, discard, sanitize, or deletion.

The native attention consumer can be checked after the transport qualification
without changing the device policy.  Keep the controller bound to VFIO and
run the benchmark as a user with access to `/dev/nta_nvme_p2p` (or through
`sudo`):

```bash
sudo env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  NTA_NVME_MEDIA_POLICY=trusted-read-only-code \
  build/nta-paged-attention \
  --mode=nvme --nvme-endpoint=vfio:0000:d8:00.0 \
  --nvme-reference=/path/to/reference.bin \
  --nvme-media-policy=trusted-read-only-code --json=1
```

The report is only `qualified:true` when the exact attention run submitted and
completed NVMe reads, had no transport failures, and had no output or
verification failures.  An HBM destination owned by the NVMe transport is not
published as a direct consumer source; doing so would make the planner bypass
the typed acquisition path.

The machine-readable report separates transport qualification from matched
performance qualification. It records exact checksums, queue counters, HBM
mapping backend, translated-IOMMU and GPU-doorbell evidence, target DMAR fault
counts, a kernel-owned read-only `fio` baseline, the bandwidth ratio, revision,
and dirty-worktree state. See `docs/NVME_SECURITY.md` for the complete threat
model and teardown contract.

### Serving experiments

Serving is intentionally a supplied workload command because model weights,
SGLang/FlashInfer versions, GPU allocation, and dataset paths are external
inputs. `--result` is required so a completed serving bundle has a structured,
machine-checkable report in addition to the raw command log:

```bash
python experiments/reproduce.py \
  --profile serving \
  --output /tmp/nta-artifacts/serving \
  --result /tmp/serving-report.json \
  --workload-manifest /tmp/bailian/manifest.json \
  -- python benchmarks/serving/CompareSglangHiCacheLoad.py \
    --model /path/to/pinned/model \
    --workload-manifest /tmp/bailian/manifest.json \
    --output /tmp/serving-report.json
```

The reproduction profile runs
`experiments/validate_serving_report.py` after the command and the bundle
validator runs it again. A serving result must prove exact demand, zero
verification failures, output digests, finite latency percentiles, SLO
goodput, Little's Law accounting, engine statistics, and machine metadata.
Comparison reports must also prove identical stock/NTA outputs and explicit
mechanism activation.

For paper artifacts, pass every non-default variable with `--env` and keep the
model/configuration manifest beside the copied serving report; never overwrite
an older trial. A
serving claim is valid only when its report contains the revision, workload
configuration, exact demand/activation counters, correctness digest, and
machine metadata.

The wrapper also sets `NTA_SERVING_WORKSPACE_ROOT` to a directory inside the
external artifact bundle. The canonical comparison driver accepts the same
value via `--workspace-root`; this keeps JIT caches and engine sidecars out of
the checkout.

### Paired evaluation artifact

The qualified evaluation profile records the spec, raw randomized trials, and
the strata-first/causal reports as one bundle:

```bash
python experiments/reproduce.py \
  --profile evaluation \
  --spec /path/to/paired-evaluation.json \
  --output /tmp/nta-artifacts/evaluation
```

It runs `run_evaluation.py` and then independently validates
`evaluation-report.json`, `strata-report.json`, and `causal-report.json`.

By default reproduction rejects a dirty worktree. `--allow-dirty` is reserved
for local debugging and records the dirty state plus a worktree diff digest.
