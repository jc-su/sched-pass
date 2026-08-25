#!/usr/bin/env python3
"""Prepare a source-checked FlashInfer include overlay with NTA CTA hooks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import shutil


SOURCE_PROFILES = {
    "0.6.12": {
        "tree": "71b994241bb85a3e3d5d0e40ac02d3c5652a07bbc4afaa8627adcb1914bb8b1c",
        "hashes": {
            "attention/decode.cuh": (
                "019d673aa848a938798a2c58b34b9b5813a3f137962cbbd90ef7bd71f636f373"
            ),
            "attention/prefill.cuh": (
                "0620738f4c1f3e64fd1713cec863f3ae67ab2bdbf2f4002d9e1a1bf6d69ea737"
            ),
            "attention/cascade.cuh": (
                "3ce6de505884bbc4efc78e63b0719eef4fe31fe152defa9fa4f050898fa51210"
            ),
            "vec_dtypes.cuh": (
                "8c7654d8acb6b872a32fb6f292614c208597e3d27b3316befdfe3461dae7c2e8"
            ),
            "utils.cuh": (
                "3d91b642002f143d74d840861b908b985e142392ebf331a67159461162c99507"
            ),
        },
    },
    "0.6.14": {
        "tree": "9760fbad07bcc6fd6b98c5137e5c07b4d082dd181954a567297b0091df00d203",
        "hashes": {
            "attention/decode.cuh": (
                "bd699c64cff0af922305e532de587ff4db8bc377c198bd653db1d48aa74c2942"
            ),
            "attention/prefill.cuh": (
                "9bb66eeae98340e72a7cef68691d7198a8538d58f2fc64f08f77a314a9f4d98c"
            ),
            "attention/cascade.cuh": (
                "3ce6de505884bbc4efc78e63b0719eef4fe31fe152defa9fa4f050898fa51210"
            ),
            "vec_dtypes.cuh": (
                "5b6c312a3736be4bdd2991c80038a3e3e8040ea8a302d3fb52332a50e3bac7a0"
            ),
            "utils.cuh": (
                "da42b5f100ccb02a4c05c6f82157ce604dcc535b5d7c1b38a68e42865a7e650e"
            ),
        },
    },
}
SUPPORTED_VERSIONS = frozenset(SOURCE_PROFILES)

POLICY_INCLUDE_ANCHOR = "#include <cooperative_groups.h>"
POLICY_INCLUDE = '#include "nta/FlashInferKernelPolicy.cuh"\n'
VEC_CAST_ANCHOR = "vec_cast<tgt_float_t, src_float_t>::cast<vec_size>("
VEC_CAST_REPLACEMENT = "vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>("
CLANG_PREFILL_TEMPLATE_REPLACEMENTS = (
    ("smem.load_64b_async<", "smem.template load_64b_async<"),
    ("smem.load_128b_async<", "smem.template load_128b_async<"),
    ("qo_smem.get_permuted_offset<", "qo_smem.template get_permuted_offset<"),
    (
        "vec_cast<typename KTraits::DTypeQ, float>::cast<",
        "vec_cast<typename KTraits::DTypeQ, float>::template cast<",
    ),
    (
        "vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<",
        "vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::template cast<",
    ),
    (
        "vec_cast<DTypeO, float>::cast<",
        "vec_cast<DTypeO, float>::template cast<",
    ),
)
CLANG_PREFILL_LDCA_ANCHOR = (
    "__ldca(token_pos_in_items + idx_in_original_seq - prefix_len)"
)
CLANG_PREFILL_LDCA_REPLACEMENT = (
    "token_pos_in_items[idx_in_original_seq - prefix_len]"
)
CLANG_PREFILL_BITWISE_AND_ANCHOR = (
    "if (idx_in_original_seq >= prefix_len & idx_in_original_seq < kv_len)"
)
CLANG_PREFILL_BITWISE_AND_REPLACEMENT = (
    "if (idx_in_original_seq >= prefix_len && idx_in_original_seq < kv_len)"
)
CTA_TILE_Q_DISPATCH_ANCHOR = "#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \\\n"
CTA_TILE_Q_CASE_ANCHOR = (
    "    case 32: {                                             \\\n"
    "      constexpr uint32_t CTA_TILE_Q = 32;                  \\\n"
    "      __VA_ARGS__                                          \\\n"
    "      break;                                               \\\n"
    "    }                                                      \\\n"
)
CTA_TILE_Q_HELPER = (
    "#if defined(NTA_FLASHINFER_SKIP_CTA_TILE_32)\n"
    "#define NTA_FLASHINFER_CTA_TILE_Q_32_CASE(...)\n"
    "#else\n"
    "#define NTA_FLASHINFER_CTA_TILE_Q_32_CASE(...) \\\n"
    "    case 32: {                                             \\\n"
    "      constexpr uint32_t CTA_TILE_Q = 32;                  \\\n"
    "      __VA_ARGS__                                          \\\n"
    "      break;                                               \\\n"
    "    }\n"
    "#endif\n"
)
CTA_TILE_Q_CASE_REPLACEMENT = (
    "    NTA_FLASHINFER_CTA_TILE_Q_32_CASE(__VA_ARGS__)       \\\n"
)
CASCADE_INCLUDE_ANCHOR = '#include "../cp_async.cuh"'
CASCADE_INCLUDE = '#include "nta/RuntimeABI.h"\n'

DECODE_ANCHOR = """__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
"""
DECODE_REPLACEMENT = """__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  nta::abi::RuntimeView* nta_runtime = nullptr;
#if !NTA_FLASHINFER_REQUEST_BOUND
  nta::kernel::WorkContext nta_work{};
#endif
  uint32_t nta_work_index = blockIdx.x;
#if NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasRequestBindingV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    if (!nta::flashinfer::validRequestBoundLaunch(params, nta_runtime)) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (!nta::flashinfer::bindValidatedRequestOnly(
            params, nta_runtime, nta_request_index)) return;
  }
#else
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    nta_work_index = nta::flashinfer::launchWorkIndex(
        params, nta_runtime, blockIdx.x);
    if (nta_work_index == nta::abi::InvalidIndex) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
#if NTA_FLASHINFER_PREACQUIRED_ONLY
    if (nta::flashinfer::tracksCompletion(params)) return;
#endif
    if (nta::flashinfer::usesPlanlessPreacquired(params)) {
      if (nta_runtime == nullptr || nta_runtime->abiVersion != nta::abi::Version ||
          nta_request_index >= nta_runtime->requestCapacity) return;
      if (!nta::kernel::acquireCurrentRequest(
              nta_runtime, nta_request_index, nta_work)) {
        return;
      }
    } else if (!nta::flashinfer::validWork(
                   params, nta_work_index, nta_request_index)) {
      return;
    } else if (!nta::flashinfer::tracksCompletion(params)) {
      if (!nta::kernel::acquirePreacquiredWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta_work_index, nta_work)) return;
    } else if (nta::flashinfer::bindsCurrentGeneration(params)) {
      if (!nta::kernel::acquireCurrentWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta::flashinfer::dependencies(params), nta_work_index, nta_work)) {
#if !NTA_FLASHINFER_STREAM_ORDERED_DIRECT
        nta::kernel::defer(nta_runtime, nta_work);
#endif
        return;
      }
    } else if (!nta::kernel::acquireWork(
                   nta_runtime, nta::flashinfer::workItems(params),
                   nta::flashinfer::dependencies(params), nta_work_index,
                   nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(params, nta_runtime, nta_work)) return;
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::kernel::beginPartial(nta_runtime, nta_work);
#endif
  }
#endif
  extern __shared__ uint8_t smem[];
"""

DECODE_EXIT_ANCHOR = """  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}
"""
DECODE_EXIT_REPLACEMENT = """  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem, nta_work_index);
#if !NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::flashinfer::finish(params, nta_runtime, nta_work);
#endif
  }
#endif
}
"""

MLA_DECODE_ANCHOR = """__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
"""
MLA_DECODE_REPLACEMENT = """__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  nta::abi::RuntimeView* nta_runtime = nullptr;
#if !NTA_FLASHINFER_REQUEST_BOUND
  nta::kernel::WorkContext nta_work{};
#endif
  uint32_t nta_work_index = blockIdx.x;
#if NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasRequestBindingV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    if (!nta::flashinfer::validRequestBoundLaunch(params, nta_runtime)) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
#if NTA_FLASHINFER_PREACQUIRED_ONLY
    if (nta::flashinfer::tracksCompletion(params)) return;
#endif
    if (!nta::flashinfer::bindValidatedRequestOnly(
            params, nta_runtime, nta_request_index)) return;
  }
#else
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    nta_work_index = nta::flashinfer::launchWorkIndex(
        params, nta_runtime, blockIdx.x);
    if (nta_work_index == nta::abi::InvalidIndex) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (nta::flashinfer::usesPlanlessPreacquired(params)) {
      if (nta_runtime == nullptr || nta_runtime->abiVersion != nta::abi::Version ||
          nta_request_index >= nta_runtime->requestCapacity) return;
      if (!nta::kernel::acquireCurrentRequest(
              nta_runtime, nta_request_index, nta_work)) {
        return;
      }
    } else if (!nta::flashinfer::validWork(
                   params, nta_work_index, nta_request_index)) {
      return;
    } else if (!nta::flashinfer::tracksCompletion(params)) {
      if (!nta::kernel::acquirePreacquiredWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta_work_index, nta_work)) return;
    } else if (nta::flashinfer::bindsCurrentGeneration(params)) {
      if (!nta::kernel::acquireCurrentWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta::flashinfer::dependencies(params), nta_work_index, nta_work)) {
#if !NTA_FLASHINFER_STREAM_ORDERED_DIRECT
        nta::kernel::defer(nta_runtime, nta_work);
#endif
        return;
      }
    } else if (!nta::kernel::acquireWork(
                   nta_runtime, nta::flashinfer::workItems(params),
                   nta::flashinfer::dependencies(params), nta_work_index,
                   nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(params, nta_runtime, nta_work)) return;
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::kernel::beginPartial(nta_runtime, nta_work);
#endif
  }
#endif
  auto block = cg::this_thread_block();
"""

MLA_DECODE_EXIT_ANCHOR = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA"""
MLA_DECODE_EXIT_REPLACEMENT = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
#if !NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::flashinfer::finish(params, nta_runtime, nta_work);
#endif
  }
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA"""

MLA_WORK_INDEX_ANCHOR = "  const uint32_t batch_idx = blockIdx.x;"
MLA_WORK_INDEX_REPLACEMENT = "  const uint32_t batch_idx = nta_work_index;"

DECODE_GRID_ANCHOR = "      dim3 nblks(padded_batch_size, num_kv_heads);"
DECODE_GRID_REPLACEMENT = """      dim3 nblks(
          nta::flashinfer::launchWorkCount(params, padded_batch_size), num_kv_heads);"""
MLA_GRID_ANCHOR = "    dim3 nblks(padded_batch_size, gdy);"
MLA_GRID_REPLACEMENT = """    dim3 nblks(
        nta::flashinfer::launchWorkCount(params, padded_batch_size), gdy);"""

RAGGED_PREFILL_ANCHOR = """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
"""
RAGGED_PREFILL_REPLACEMENT = """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(
    const __grid_constant__ Params params) {
  nta::abi::RuntimeView* nta_runtime = nullptr;
#if !NTA_FLASHINFER_REQUEST_BOUND
  nta::kernel::WorkContext nta_work{};
#endif
  uint32_t nta_work_index = blockIdx.x;
#if NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasRequestBindingV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    if (!nta::flashinfer::validRequestBoundLaunch(params, nta_runtime)) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (!nta::flashinfer::bindValidatedRequestOnly(
            params, nta_runtime, nta_request_index)) return;
  }
#else
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    nta_work_index = nta::flashinfer::launchWorkIndex(
        params, nta_runtime, blockIdx.x);
    if (nta_work_index == nta::abi::InvalidIndex) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (nta::flashinfer::usesPlanlessPreacquired(params)) {
      if (nta_runtime == nullptr || nta_runtime->abiVersion != nta::abi::Version ||
          nta_request_index >= nta_runtime->requestCapacity) return;
      if (!nta::kernel::acquireCurrentRequest(
              nta_runtime, nta_request_index, nta_work)) return;
    } else if (!nta::flashinfer::validWork(
                   params, nta_work_index, nta_request_index)) {
      return;
    } else if (!nta::flashinfer::tracksCompletion(params)) {
      if (!nta::kernel::acquirePreacquiredWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta_work_index, nta_work)) return;
    } else if (nta::flashinfer::bindsCurrentGeneration(params)) {
      if (!nta::kernel::acquireCurrentWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta::flashinfer::dependencies(params), nta_work_index, nta_work)) {
#if !NTA_FLASHINFER_STREAM_ORDERED_DIRECT
        nta::kernel::defer(nta_runtime, nta_work);
#endif
        return;
      }
    } else if (!nta::kernel::acquireWork(
                   nta_runtime, nta::flashinfer::workItems(params),
                   nta::flashinfer::dependencies(params), nta_work_index,
                   nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(params, nta_runtime, nta_work)) return;
    nta::kernel::beginPartial(nta_runtime, nta_work);
  }
#endif
  using DTypeQ = typename Params::DTypeQ;
"""

RAGGED_WORK_INDEX_ANCHOR = """    const uint32_t bx = blockIdx.x, lane_idx = tid.x,
"""
RAGGED_WORK_INDEX_REPLACEMENT = """    const uint32_t bx = nta_work_index, lane_idx = tid.x,
"""

RAGGED_VALID_MASK_ANCHOR = """    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
"""
RAGGED_VALID_MASK_REPLACEMENT = """    if constexpr (!nta::flashinfer::HasWorkPlanV<Params> &&
                  !nta::flashinfer::HasRequestBindingV<Params>) {
      if (block_valid_mask && !block_valid_mask[bx]) {
        return;
      }
    }
"""

RAGGED_PREFILL_EXIT_ANCHOR = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice(
"""
RAGGED_PREFILL_EXIT_REPLACEMENT = """#if !NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    nta::flashinfer::finish(params, nta_runtime, nta_work);
  }
#endif
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice(
"""

RAGGED_PREFILL_EXIT_ANCHOR_V614 = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

// VO-split helpers (HEAD_DIM_VO >= 512)
template <typename KTraits>
__device__ __forceinline__ void vosplit_softmax_store_p(
"""

PAGED_PREFILL_ANCHOR = """__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
"""
PAGED_PREFILL_REPLACEMENT = """__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  nta::abi::RuntimeView* nta_runtime = nullptr;
#if !NTA_FLASHINFER_REQUEST_BOUND
  nta::kernel::WorkContext nta_work{};
#endif
  uint32_t nta_work_index = blockIdx.x;
#if NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasRequestBindingV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    if (!nta::flashinfer::validRequestBoundLaunch(params, nta_runtime)) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
#if NTA_FLASHINFER_PREACQUIRED_ONLY
    if (nta::flashinfer::tracksCompletion(params)) return;
#endif
    if (!nta::flashinfer::bindValidatedRequestOnly(
            params, nta_runtime, nta_request_index)) return;
  }
#else
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    nta_runtime = nta::flashinfer::runtime(params);
    nta_work_index = nta::flashinfer::launchWorkIndex(
        params, nta_runtime, blockIdx.x);
    if (nta_work_index == nta::abi::InvalidIndex) return;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (nta::flashinfer::usesPlanlessPreacquired(params)) {
      if (nta_runtime == nullptr || nta_runtime->abiVersion != nta::abi::Version ||
          nta_request_index >= nta_runtime->requestCapacity) return;
      if (!nta::kernel::acquireCurrentRequest(
              nta_runtime, nta_request_index, nta_work)) {
        return;
      }
    } else if (!nta::flashinfer::validWork(
                   params, nta_work_index, nta_request_index)) {
      return;
    } else if (!nta::flashinfer::tracksCompletion(params)) {
      if (!nta::kernel::acquirePreacquiredWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta_work_index, nta_work)) return;
    } else if (nta::flashinfer::bindsCurrentGeneration(params)) {
      if (!nta::kernel::acquireCurrentWork(
              nta_runtime, nta::flashinfer::workItems(params),
              nta::flashinfer::dependencies(params), nta_work_index, nta_work)) {
#if !NTA_FLASHINFER_STREAM_ORDERED_DIRECT
        nta::kernel::defer(nta_runtime, nta_work);
#endif
        return;
      }
    } else if (!nta::kernel::acquireWork(
                   nta_runtime, nta::flashinfer::workItems(params),
                   nta::flashinfer::dependencies(params), nta_work_index,
                   nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(params, nta_runtime, nta_work)) return;
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::kernel::beginPartial(nta_runtime, nta_work);
#endif
  }
#endif
  extern __shared__ uint8_t smem[];
"""

PAGED_PREFILL_EXIT_ANCHOR = """  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}
"""
PAGED_PREFILL_EXIT_REPLACEMENT = """  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage, threadIdx, nta_work_index);
#if !NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::flashinfer::finish(params, nta_runtime, nta_work);
#endif
  }
#endif
}
"""

PAGED_PREFILL_EXIT_ANCHOR_V614 = """  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStoragePaged&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}
"""
PAGED_PREFILL_EXIT_REPLACEMENT_V614 = """  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStoragePaged&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage, threadIdx, nta_work_index);
#if !NTA_FLASHINFER_REQUEST_BOUND
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
#if !NTA_FLASHINFER_PREACQUIRED_ONLY
    nta::flashinfer::finish(params, nta_runtime, nta_work);
#endif
  }
#endif
}
"""

PAGED_PREFILL_DISPATCH_ANCHOR = "cudaError_t BatchPrefillWithPagedKVCacheDispatched("
PREFILL_GRID_ANCHOR = "  dim3 nblks(padded_batch_size, 1, num_kv_heads);"
PREFILL_GRID_REPLACEMENT = """  dim3 nblks(
      nta::flashinfer::launchWorkCount(params, padded_batch_size), 1, num_kv_heads);"""

DECODE_MERGE_ANCHOR = """        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
"""
DECODE_MERGE_REPLACEMENT = """        if (nta::flashinfer::shouldMerge(params)) {
          const auto nta_merge_gate = nta::flashinfer::mergeGate(params);
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
                num_qo_heads, HEAD_DIM, enable_pdl, stream, nta_merge_gate.runtime,
                static_cast<decltype(params.request_indices)>(nullptr), 0,
                nta_merge_gate.reductionGroupOffset));
          } else {
            FLASHINFER_CUDA_CALL(
                VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                           nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream,
                                           nta_merge_gate.runtime,
                                           static_cast<decltype(params.request_indices)>(nullptr),
                                           0, nta_merge_gate.reductionGroupOffset));
          }
        }
"""

PREFILL_MERGE_ANCHOR = """        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
              num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        }
"""
PREFILL_MERGE_REPLACEMENT = """        if (nta::flashinfer::shouldMerge(params)) {
          const auto nta_merge_gate = nta::flashinfer::mergeGate(params);
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
                params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream,
                nta_merge_gate.runtime, params.q_indptr,
                nta::flashinfer::requestGroupCount(params),
                nta_merge_gate.reductionGroupOffset));
          } else {
            FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
                tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
                num_qo_heads, HEAD_DIM_VO, enable_pdl, stream, nta_merge_gate.runtime,
              params.q_indptr, nta::flashinfer::requestGroupCount(params),
              nta_merge_gate.reductionGroupOffset));
          }
        }
"""

MERGE_KERNEL_ANCHOR = """__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads) {"""
MERGE_KERNEL_REPLACEMENT = """__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads, const nta::abi::RuntimeView* __restrict__ nta_runtime,
    const IdType* __restrict__ nta_group_indptr, uint32_t nta_group_count,
    uint32_t nta_reduction_group_offset) {"""

SUM_KERNEL_ANCHOR = """__global__ void PersistentVariableLengthAttentionSumKernel(DTypeIn* __restrict__ V, IdType* indptr,
                                                           DTypeO* __restrict__ v_sum,
                                                           uint32_t max_seq_len,
                                                           uint32_t* __restrict__ seq_len_ptr,
                                                           uint32_t num_heads) {"""
SUM_KERNEL_REPLACEMENT = """__global__ void PersistentVariableLengthAttentionSumKernel(DTypeIn* __restrict__ V, IdType* indptr,
                                                           DTypeO* __restrict__ v_sum,
                                                           uint32_t max_seq_len,
                                                           uint32_t* __restrict__ seq_len_ptr,
                                                           uint32_t num_heads,
                                                           const nta::abi::RuntimeView* __restrict__ nta_runtime,
                                                           const IdType* __restrict__ nta_group_indptr,
                                                           uint32_t nta_group_count,
                                                           uint32_t nta_reduction_group_offset) {"""

MERGE_WAIT_ANCHOR = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1"""
MERGE_WAIT_REPLACEMENT = """#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  if (nta_runtime != nullptr &&
      (nta_runtime->abiVersion != nta::abi::Version ||
       nta_runtime->reductionExpected == nullptr ||
       nta_runtime->reductionCompleted == nullptr ||
       nta_runtime->reductionFailed == nullptr || nta_runtime->workTicketCapacity == 0)) {
    return;
  }

#pragma unroll 1"""

MERGE_GROUP_ANCHOR = """    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;"""
MERGE_GROUP_REPLACEMENT = """    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    if (nta_runtime != nullptr) {
      uint32_t reduction_group = nta_reduction_group_offset + pos;
      if (nta_group_indptr != nullptr) {
        if (nta_group_count == 0 || pos >= uint32_t(nta_group_indptr[nta_group_count])) {
          continue;
        }
        uint32_t lower = 0, upper = nta_group_count;
        while (lower + 1 < upper) {
          const uint32_t middle = lower + (upper - lower) / 2;
          if (uint32_t(nta_group_indptr[middle]) <= pos) {
            lower = middle;
          } else {
            upper = middle;
          }
        }
        reduction_group = nta_reduction_group_offset + lower;
      }
      if (reduction_group >= nta_runtime->workTicketCapacity ||
          nta_runtime->reductionExpected[reduction_group] == 0 ||
          nta_runtime->reductionFailed[reduction_group] != 0 ||
          nta_runtime->reductionCompleted[reduction_group] !=
              nta_runtime->reductionExpected[reduction_group]) {
        continue;
      }
    }"""

MERGE_FUNCTION_ANCHOR = """cudaError_t VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeO* v_merged,
                                      float* s_merged, uint32_t max_seq_len, uint32_t* seq_len,
                                      uint32_t num_heads, uint32_t head_dim, bool enable_pdl,
                                      cudaStream_t stream = nullptr) {"""
MERGE_FUNCTION_REPLACEMENT = """cudaError_t VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeO* v_merged,
                                      float* s_merged, uint32_t max_seq_len, uint32_t* seq_len,
                                      uint32_t num_heads, uint32_t head_dim, bool enable_pdl,
                                      cudaStream_t stream = nullptr,
                                      const nta::abi::RuntimeView* nta_runtime = nullptr,
                                      const IdType* nta_group_indptr = nullptr,
                                      uint32_t nta_group_count = 0,
                                      uint32_t nta_reduction_group_offset = 0) {"""

SUM_FUNCTION_ANCHOR = """cudaError_t VariableLengthAttentionSum(DTypeIn* v, IdType* indptr, DTypeO* v_sum,
                                       uint32_t max_seq_len, uint32_t* seq_len, uint32_t num_heads,
                                       uint32_t head_dim, bool enable_pdl,
                                       cudaStream_t stream = nullptr) {"""
SUM_FUNCTION_REPLACEMENT = """cudaError_t VariableLengthAttentionSum(DTypeIn* v, IdType* indptr, DTypeO* v_sum,
                                       uint32_t max_seq_len, uint32_t* seq_len, uint32_t num_heads,
                                       uint32_t head_dim, bool enable_pdl,
                                       cudaStream_t stream = nullptr,
                                       const nta::abi::RuntimeView* nta_runtime = nullptr,
                                       const IdType* nta_group_indptr = nullptr,
                                       uint32_t nta_group_count = 0,
                                       uint32_t nta_reduction_group_offset = 0) {"""

MERGE_ARGS_ANCHOR = """    void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &max_seq_len, &seq_len, &num_heads};"""
MERGE_ARGS_REPLACEMENT = """    void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &max_seq_len, &seq_len,
                    &num_heads, &nta_runtime, &nta_group_indptr, &nta_group_count,
                    &nta_reduction_group_offset};"""
MERGE_EX_ANCHOR = """      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v, s, indptr, v_merged, s_merged,
                                              max_seq_len, seq_len, num_heads));"""
MERGE_EX_REPLACEMENT = """      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v, s, indptr, v_merged, s_merged,
                                              max_seq_len, seq_len, num_heads,
                                              nta_runtime, nta_group_indptr,
                                              nta_group_count,
                                              nta_reduction_group_offset));"""

SUM_ARGS_ANCHOR = (
    """    void* args[] = {&v, &indptr, &v_sum, &max_seq_len, &seq_len, &num_heads};"""
)
SUM_ARGS_REPLACEMENT = """    void* args[] = {&v, &indptr, &v_sum, &max_seq_len, &seq_len, &num_heads,
                    &nta_runtime, &nta_group_indptr, &nta_group_count,
                    &nta_reduction_group_offset};"""
SUM_EX_ANCHOR = """          cudaLaunchKernelEx(&config, kernel, v, indptr, v_sum, max_seq_len, seq_len, num_heads));"""
SUM_EX_REPLACEMENT = """          cudaLaunchKernelEx(&config, kernel, v, indptr, v_sum, max_seq_len, seq_len, num_heads,
                             nta_runtime, nta_group_indptr, nta_group_count,
                             nta_reduction_group_offset));"""


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and not candidate.name.endswith((".bak", ".sched_bak", "~"))
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def flashinfer_include() -> tuple[str, pathlib.Path]:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise RuntimeError("flashinfer-python is not installed")
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "flashinfer-python distribution metadata is missing"
        ) from error
    include = pathlib.Path(spec.origin).resolve().parent / "data" / "include"
    if not (include / "flashinfer").is_dir():
        raise RuntimeError(f"FlashInfer include tree is missing: {include}")
    return version, include


def checked_replace(
    source: str, anchor: str, replacement: str, description: str
) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected exactly one {description} anchor, found {count}")
    return source.replace(anchor, replacement)


def checked_replace_count(
    source: str,
    anchor: str,
    replacement: str,
    description: str,
    expected_count: int,
) -> str:
    count = source.count(anchor)
    if count != expected_count:
        raise RuntimeError(
            f"expected exactly {expected_count} {description} anchors, found {count}"
        )
    return source.replace(anchor, replacement)


def patch_header(path: pathlib.Path, replacements: list[tuple[str, str, str]]) -> None:
    source = path.read_text(encoding="utf-8")
    source = checked_replace(
        source,
        POLICY_INCLUDE_ANCHOR,
        POLICY_INCLUDE + POLICY_INCLUDE_ANCHOR,
        "policy include",
    )
    for anchor, replacement, description in replacements:
        source = checked_replace(source, anchor, replacement, description)
    path.write_text(source, encoding="utf-8")


def patch_vec_dtypes(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    if VEC_CAST_ANCHOR in source:
        source = checked_replace(
            source,
            VEC_CAST_ANCHOR,
            VEC_CAST_REPLACEMENT,
            "dependent vec cast template disambiguator",
        )
    elif VEC_CAST_REPLACEMENT not in source:
        raise RuntimeError("FlashInfer vec_dtypes.cuh has no supported vec cast form")
    path.write_text(source, encoding="utf-8")


def patch_utils(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    if CTA_TILE_Q_CASE_ANCHOR not in source:
        # FlashInfer 0.6.12 already omits the unreachable CTA_TILE_Q=32 case
        # from its dispatch macro.  The conditional helper is only needed for
        # 0.6.14, whose host dispatcher references that specialization even
        # when the generated module does not instantiate it.
        return
    source = checked_replace(
        source,
        CTA_TILE_Q_DISPATCH_ANCHOR,
        CTA_TILE_Q_HELPER + CTA_TILE_Q_DISPATCH_ANCHOR,
        "CTA tile dispatch compatibility helper",
    )
    source = checked_replace(
        source,
        CTA_TILE_Q_CASE_ANCHOR,
        CTA_TILE_Q_CASE_REPLACEMENT,
        "CTA tile dispatch compatibility case",
    )
    path.write_text(source, encoding="utf-8")


def patch_prefill_header(path: pathlib.Path, version: str) -> None:
    source = path.read_text(encoding="utf-8")
    source = checked_replace(
        source,
        POLICY_INCLUDE_ANCHOR,
        POLICY_INCLUDE + POLICY_INCLUDE_ANCHOR,
        "policy include",
    )
    source = checked_replace(
        source,
        RAGGED_PREFILL_ANCHOR,
        RAGGED_PREFILL_REPLACEMENT,
        "ragged prefill",
    )
    source = checked_replace(
        source,
        RAGGED_WORK_INDEX_ANCHOR,
        RAGGED_WORK_INDEX_REPLACEMENT,
        "ragged prefill work-index remap",
    )
    source = checked_replace_count(
        source,
        RAGGED_VALID_MASK_ANCHOR,
        RAGGED_VALID_MASK_REPLACEMENT,
        "prefill valid mask",
        2,
    )
    ragged_exit_anchor = RAGGED_PREFILL_EXIT_ANCHOR
    if source.count(ragged_exit_anchor) != 1:
        ragged_exit_anchor = RAGGED_PREFILL_EXIT_ANCHOR_V614
    source = checked_replace(
        source,
        ragged_exit_anchor,
        RAGGED_PREFILL_EXIT_REPLACEMENT,
        "ragged prefill completion",
    )
    source = checked_replace(
        source,
        PAGED_PREFILL_ANCHOR,
        PAGED_PREFILL_REPLACEMENT,
        "paged prefill",
    )
    paged_exit_anchor = PAGED_PREFILL_EXIT_ANCHOR
    paged_exit_replacement = PAGED_PREFILL_EXIT_REPLACEMENT
    if source.count(paged_exit_anchor) != 1:
        paged_exit_anchor = PAGED_PREFILL_EXIT_ANCHOR_V614
        paged_exit_replacement = PAGED_PREFILL_EXIT_REPLACEMENT_V614
    source = checked_replace(
        source,
        paged_exit_anchor,
        paged_exit_replacement,
        "paged prefill completion",
    )
    source = checked_replace_count(
        source,
        PREFILL_GRID_ANCHOR,
        PREFILL_GRID_REPLACEMENT,
        "prefill compact grid",
        2,
    )
    source = checked_replace_count(
        source,
        PREFILL_MERGE_ANCHOR,
        PREFILL_MERGE_REPLACEMENT,
        "prefill split-K merge",
        2,
    )
    if version == "0.6.14":
        for anchor, replacement in CLANG_PREFILL_TEMPLATE_REPLACEMENTS:
            source = source.replace(anchor, replacement)
        source = checked_replace(
            source,
            CLANG_PREFILL_LDCA_ANCHOR,
            CLANG_PREFILL_LDCA_REPLACEMENT,
            "Clang-compatible uint16 prefill cache load",
        )
        source = checked_replace(
            source,
            CLANG_PREFILL_BITWISE_AND_ANCHOR,
            CLANG_PREFILL_BITWISE_AND_REPLACEMENT,
            "prefill prefix-boundary condition",
        )
    path.write_text(source, encoding="utf-8")


def patch_cascade_header(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = checked_replace(
        source,
        CASCADE_INCLUDE_ANCHOR,
        CASCADE_INCLUDE + CASCADE_INCLUDE_ANCHOR,
        "runtime ABI include",
    )
    replacements = [
        (MERGE_KERNEL_ANCHOR, MERGE_KERNEL_REPLACEMENT, "merge kernel gate"),
        (SUM_KERNEL_ANCHOR, SUM_KERNEL_REPLACEMENT, "sum kernel gate"),
        (MERGE_FUNCTION_ANCHOR, MERGE_FUNCTION_REPLACEMENT, "merge gate API"),
        (SUM_FUNCTION_ANCHOR, SUM_FUNCTION_REPLACEMENT, "sum gate API"),
        (MERGE_ARGS_ANCHOR, MERGE_ARGS_REPLACEMENT, "merge gate arguments"),
        (MERGE_EX_ANCHOR, MERGE_EX_REPLACEMENT, "merge PDL gate arguments"),
        (SUM_ARGS_ANCHOR, SUM_ARGS_REPLACEMENT, "sum gate arguments"),
        (SUM_EX_ANCHOR, SUM_EX_REPLACEMENT, "sum PDL gate arguments"),
    ]
    for anchor, replacement, description in replacements:
        source = checked_replace(source, anchor, replacement, description)
    source = checked_replace_count(
        source,
        MERGE_WAIT_ANCHOR,
        MERGE_WAIT_REPLACEMENT,
        "merge dependency wait",
        2,
    )
    source = checked_replace_count(
        source,
        MERGE_GROUP_ANCHOR,
        MERGE_GROUP_REPLACEMENT,
        "per-request merge gate",
        2,
    )
    path.write_text(source, encoding="utf-8")


def validate_existing(
    output: pathlib.Path,
    expected: dict[str, object],
    expected_hashes: dict[str, str],
) -> dict[str, object] | None:
    manifest_path = output / "manifest.json"
    if not output.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid existing FlashInfer overlay: {output}") from error
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"stale existing FlashInfer overlay: {output}")
    overlay_hashes = manifest.get("overlay_hashes")
    overlay_tree_hash = manifest.get("overlay_tree_hash")
    if (
        not isinstance(overlay_hashes, dict)
        or set(overlay_hashes) != set(expected_hashes)
        or not isinstance(overlay_tree_hash, str)
        or tree_hash(output / "flashinfer") != overlay_tree_hash
        or any(
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not (output / "flashinfer" / relative).is_file()
            or sha256(output / "flashinfer" / relative) != expected_hash
            for relative, expected_hash in overlay_hashes.items()
            )
    ):
        raise RuntimeError(f"corrupt existing FlashInfer overlay: {output}")
    return manifest


def prepare_locked(output: pathlib.Path) -> dict[str, object]:
    version, include = flashinfer_include()
    if version not in SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"unsupported FlashInfer {version}; expected one of "
            f"{sorted(SUPPORTED_VERSIONS)}"
        )
    profile = SOURCE_PROFILES[version]
    expected_hashes = profile["hashes"]
    expected_tree_hash = profile["tree"]
    observed = {
        relative: sha256(include / "flashinfer" / relative)
        for relative in expected_hashes
    }
    observed_tree_hash = tree_hash(include / "flashinfer")
    if observed != expected_hashes or observed_tree_hash != expected_tree_hash:
        mismatches = [
            relative
            for relative, expected in expected_hashes.items()
            if observed[relative] != expected
        ]
        if observed_tree_hash != expected_tree_hash:
            mismatches.append("complete include tree")
        raise RuntimeError(
            f"FlashInfer {version} headers differ from the validated sources: "
            + ", ".join(mismatches)
        )

    manifest: dict[str, object] = {
        "flashinfer_version": version,
        "source_include": str(include),
        "source_hashes": observed,
        "source_tree_hash": observed_tree_hash,
        "hooks": [
            "batch-decode",
            "mla-decode",
            "paged-prefill-fa2",
            "ragged-prefill-fa2",
            "phase-aware-split-k-merge",
            "multi-cta-in-kernel-completion",
            "request-bound-runnable-work-remap",
            "bounded-physical-runnable-grid",
            "clang-dependent-template-fix",
        ],
    }
    existing = validate_existing(output, manifest, expected_hashes)
    if existing is not None:
        return existing

    destination = output / "flashinfer"
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    shutil.copytree(include / "flashinfer", temporary / "flashinfer")
    patch_header(
        temporary / "flashinfer/attention/decode.cuh",
        [
            (DECODE_ANCHOR, DECODE_REPLACEMENT, "batch decode"),
            (DECODE_EXIT_ANCHOR, DECODE_EXIT_REPLACEMENT, "batch decode completion"),
            (MLA_DECODE_ANCHOR, MLA_DECODE_REPLACEMENT, "MLA decode"),
            (DECODE_GRID_ANCHOR, DECODE_GRID_REPLACEMENT, "decode compact grid"),
            (MLA_GRID_ANCHOR, MLA_GRID_REPLACEMENT, "MLA compact grid"),
            (
                MLA_WORK_INDEX_ANCHOR,
                MLA_WORK_INDEX_REPLACEMENT,
                "MLA work-index remap",
            ),
            (
                MLA_DECODE_EXIT_ANCHOR,
                MLA_DECODE_EXIT_REPLACEMENT,
                "MLA decode completion",
            ),
            (DECODE_MERGE_ANCHOR, DECODE_MERGE_REPLACEMENT, "decode split-K merge"),
        ],
    )
    patch_prefill_header(temporary / "flashinfer/attention/prefill.cuh", version)
    patch_cascade_header(temporary / "flashinfer/attention/cascade.cuh")
    patch_vec_dtypes(temporary / "flashinfer/vec_dtypes.cuh")
    patch_utils(temporary / "flashinfer/utils.cuh")
    manifest["overlay_hashes"] = {
        relative: sha256(temporary / "flashinfer" / relative)
        for relative in expected_hashes
    }
    manifest["overlay_tree_hash"] = tree_hash(temporary / "flashinfer")
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.rename(output)
    if not destination.is_dir():
        raise RuntimeError(f"failed to prepare FlashInfer overlay: {destination}")
    return manifest


def prepare(output: pathlib.Path) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_locked(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    options = parser.parse_args()
    print(json.dumps(prepare(options.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
