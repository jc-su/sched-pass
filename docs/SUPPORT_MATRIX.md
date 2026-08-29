# Support and qualification matrix

This is the engineering scope of the current branch. “Implemented” means the
path is wired and fail-closed in source and tests. “Qualified” additionally
requires machine-specific hardware and the artifact gate; it is never inferred
from a skipped test or a previous boot.

| Boundary | HBM / resident | Host-staged | NVMe -> HBM | CXL-DAX | Multi-tenant |
| --- | --- | --- | --- | --- | --- |
| Native runtime/resource contract | implemented | implemented | implemented; physical qualification required | implemented; endpoint qualification required | typed budget/admission implemented |
| SGLang + FlashInfer | exact resident and mixed external work | exact indexed path | exact catalog path, GPU-owned progress | not implemented: FlashInfer page tables name engine HBM, not direct DAX | sidecar tenant IDs plus startup budgets |
| vLLM 0.26 v1 API / V2 model runner | reference by default; opt-in exact eager single-token decode/FA2 NTA consumer | not a serving claim | explicit catalog-replay single-token decode; hardware qualification required | not implemented: native consumer still names engine HBM | request-prefix adapter plus startup budgets |
| LLVM/compiler | typed marker lowering and contract checks | same | same | same | identity comes from typed binding, not guessing |

The compiler's marker-free paged-signature discovery is diagnostic only. It
does not authorize an unmarked raw pointer as a transport request. Likewise,
the vLLM plugin's scheduler projection is not evidence of a numerical NTA
consumer; artifact validation requires the `native_work_unit` contract.

The physical NVMe implementation separates control-plane mapping/allocation
from the steady-state GPU queue. It performs no per-request mapping ioctl and
does not use host memory as a data proxy. The native CXL-DAX consumer reads a
validated device-visible mapping directly, but neither framework adapter yet
binds its paged-attention numerical pointers to that address. Serving artifact
gates reject CXL-DAX instead of reporting the native path as framework closure.
On a host without a qualified VFIO/IOMMU or devdax endpoint, the corresponding
native row remains unavailable and the correct result is an explicit
qualification skip.

This does not make the tier code untested: the default CTest suite always runs
the CXL reservation-ledger test and the read-only NVMe discovery/safety tests.
Only the physical probes and physical attention commands are capability-gated;
they are the tests that may issue device traffic or require exclusive device
ownership.

The matrix is deliberately narrower than the internal API surface. Prefill,
mixed batches, CUDA graph replay, multi-GPU physical routes, storage fault
recovery, and a vLLM scheduler-side persistent KVConnector still require their
own correctness gates before they can be presented as supported evaluation
claims. The vLLM NVMe row is the native NTA attention consumer for the current
exact block-table projection; it is not a claim that vLLM's upstream
prefix-cache/eviction lifecycle has been replaced.
