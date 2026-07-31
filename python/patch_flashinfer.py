"""patch_flashinfer.py -- make FlashInfer's headers clang-compilable.

Idempotent, reversible-by-reinstall patches to the *installed* FlashInfer
headers so they compile under clang (our LLVM pass plugin only runs under
clang). Run once after installing/upgrading flashinfer:

    python patch_flashinfer.py            # apply
    python patch_flashinfer.py --check    # report only

Most of the nvcc->clang gap is handled WITHOUT touching FlashInfer:
  * global min/max         -> clang_cuda_prelude.h (force-included by the shim)
  * cooperative_groups.h   -> shim adds -isystem <cuda>/include (a stale
                              /usr/include symlink to an old CUDA shadowed it)
  * nvcc-only flags        -> nvcc_clang_shim.py drops/translates them
  * compute_120a arch      -> obsolete on CUDA >= 12.9 (FlashInfer detects
    12.0f natively); the old FLASHINFER_CUDA_ARCH_LIST=12.0f bypassed the
                              CUDA>=12.9 gate; shim maps sm_120f->sm_120)

The ONE thing that needs a source edit is a standards-conformance bug clang
enforces but nvcc/MSVC tolerate: a dependent member-template call missing the
`template` disambiguator. `vec_cast<...>::cast<vec_size>(...)` must be
`vec_cast<...>::template cast<vec_size>(...)`.
"""
import argparse
import os
import sys

# (file-relative-to-data-root, old, new). Three families of clang-vs-nvcc
# standards gaps, each patched at its unique full signature (idempotent):
#   * missing `template` disambiguator (clang enforces, nvcc tolerates);
#   * host-only deduction guides in bundled CCCL (clang-CUDA requires
#     host+device or none; annotation only affects viability, so promoting
#     to _CCCL_HOST_DEVICE is semantics-preserving);
#   * int64 % IntFastDiv ambiguity (fastdiv's operator% takes uint32; nvcc
#     picks it silently, clang correctly reports the ambiguity with the
#     IntFastDiv->int conversion path; the cast keeps the fastdiv fast path).
PATCHES = [
    ("include/flashinfer/vec_dtypes.cuh",
     "vec_cast<tgt_float_t, src_float_t>::cast<vec_size>",
     "vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>"),
    ("cccl/libcudacxx/include/cuda/std/string_view",
     "_CCCL_HOST basic_string_view(::std::basic_string<_CharT, "
     "::std::char_traits<_CharT>, _Alloc>)",
     "_CCCL_HOST_DEVICE basic_string_view(::std::basic_string<_CharT, "
     "::std::char_traits<_CharT>, _Alloc>)"),
    ("cccl/libcudacxx/include/cuda/std/string_view",
     "_CCCL_HOST basic_string_view(::std::basic_string<_CharT, _Traits, "
     "_Alloc>)",
     "_CCCL_HOST_DEVICE basic_string_view(::std::basic_string<_CharT, "
     "_Traits, _Alloc>)"),
    ("cccl/libcudacxx/include/cuda/std/string_view",
     "_CCCL_HOST basic_string_view(::std::basic_string_view<_CharT>)",
     "_CCCL_HOST_DEVICE basic_string_view(::std::basic_string_view<_CharT>)"),
    ("cccl/libcudacxx/include/cuda/std/string_view",
     "_CCCL_HOST basic_string_view(::std::basic_string_view<_CharT, _Traits>)",
     "_CCCL_HOST_DEVICE basic_string_view(::std::basic_string_view<_CharT, "
     "_Traits>)"),
    ("include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh",
     "int const token_idx_in_seq = tokenIdx % seq_len;",
     # int (not uint32) LHS: makes the fastdiv operator%(int, IntFastDiv)
     # an EXACT match, strictly better than builtin-% via IntFastDiv->int.
     "int const token_idx_in_seq = static_cast<int>(tokenIdx) % seq_len;"),
    # --- prefill.cuh: unblock clang weaving of BatchPrefillWithPagedKVCache ---
    # FlashInfer already `.template`'s MOST dependent member-template calls but
    # missed a few (clang enforces two-phase lookup; nvcc/cudafe tolerate).
    # Broad patterns -- safe: `new` never contains `old`, and the old-first
    # check above patches only the remaining unpatched occurrences.
    ("include/flashinfer/attention/prefill.cuh",
     ".load_128b_async<", ".template load_128b_async<"),
    ("include/flashinfer/attention/prefill.cuh",
     ">::cast<", ">::template cast<"),
    ("include/flashinfer/attention/prefill.cuh",
     ".get_permuted_offset<", ".template get_permuted_offset<"),
    # __ldca on a pointer type CUDA's overloads miss: it is a pure L1-cache
    # HINT, so drop to a plain load -- bit-exact (same bytes from DRAM), only
    # the caching advisory is lost.
    ("include/flashinfer/attention/prefill.cuh",
     "__ldca(token_pos_in_items + idx_in_original_seq - prefix_len)",
     "(*(token_pos_in_items + idx_in_original_seq - prefix_len))"),
]


def include_root():
    try:
        import flashinfer
        base = os.path.dirname(flashinfer.__file__)
    except Exception:
        print("flashinfer not importable", file=sys.stderr)
        sys.exit(2)
    root = os.path.join(base, "data")
    if not os.path.isdir(root):
        print(f"data root not found: {root}", file=sys.stderr)
        sys.exit(2)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = include_root()
    applied = already = missing = drifted = 0
    for rel, old, new in PATCHES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"[missing] {rel}")
            missing += 1
            continue
        s = open(path).read()
        # Check `old` FIRST (before `new`): a file may MIX already-patched and
        # unpatched occurrences of the same pattern (FlashInfer itself
        # .template's most sites but misses a few). If any `old` remains, apply
        # -- replace(old,new) patches exactly those; the already-`new`
        # occurrences do not contain `old`, so they are untouched. Only when NO
        # `old` remains is it "already patched". (Requires: `new` must not
        # contain `old` as a substring, true for all patterns here.)
        if old in s:
            if args.check:
                print(f"[needs-patch] {rel}")
            else:
                open(path, "w").write(s.replace(old, new))
                print(f"[patched] {rel}")
            applied += 1
        elif new in s:
            print(f"[ok] {rel}: already patched")
            already += 1
        else:
            # version DRIFT: neither the pre- nor post-patch pattern is present.
            # This patcher exists to close clang-vs-nvcc dialect gaps, so a
            # silent no-match means the gap is UNCLOSED -- a hard CI failure,
            # not a warning (the review's point: [skip] must not exit 0).
            print(f"[DRIFT] {rel}: neither old nor new pattern found "
                  "(FlashInfer version drift -- inspect + re-derive the patch)")
            drifted += 1
    print(f"== applied={applied} already={already} missing={missing} "
          f"drifted={drifted} ==")
    return 1 if (missing or drifted) else 0


if __name__ == "__main__":
    sys.exit(main())
