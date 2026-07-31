"""Discover FlashInfer's PREFILL PlanInfo layout empirically: which int in
_plan_info is request_indices_offset, and verify request_indices[tile] recovers
the qo-tile -> request map (cross-checked against qo_indptr / cta_tile_q).
Pure introspection -- no weaving, no plugin. Real nvcc is fine here."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

PAGE, HD, NQO, NKV = 16, 128, 16, 2


def main():
    dev = "cuda"
    import flashinfer

    gen = torch.Generator().manual_seed(7)
    B = 48
    qo_len = torch.exp(torch.normal(4.5, 0.8, (B,), generator=gen)) \
        .clamp(16, 1024).to(torch.int32)
    pages = ((qo_len + PAGE - 1) // PAGE).to(torch.int32)
    npages = int(pages.sum())
    total_qo = int(qo_len.sum())

    qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(qo_len, 0)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(pages, 0)
    last = ((qo_len - 1) % PAGE + 1).to(torch.int32)
    qo_indptr_d, kv_indptr_d, last_d = qo_indptr.cuda(), kv_indptr.cuda(), last.cuda()
    kv_indices = torch.arange(npages, dtype=torch.int32, device=dev)

    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(qo_indptr_d, kv_indptr_d, kv_indices, last_d, NQO, NKV, HD, PAGE,
           causal=True, q_data_type=torch.float16, kv_data_type=torch.float16)

    vec = getattr(w, "_plan_info", None)
    v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    print(f"== prefill plan: qo_len(min/med/max)="
          f"{int(qo_len.min())}/{int(qo_len.median())}/{int(qo_len.max())} "
          f"B={B} total_qo={total_qo} ==")
    print(f"_plan_info (len={len(v)}): {v}")

    buf = getattr(w, "_int_workspace_buffer", None)
    print(f"_int_workspace_buffer: dtype={buf.dtype} numel={buf.numel()} "
          f"elt={buf.element_size()}B")
    bb = buf.reshape(-1).view(torch.uint8)
    nbytes = bb.numel()

    # v[0] is padded_batch_size (ntiles) for prefill too (the test uses v[0]).
    padded = int(v[0])
    print(f"padded_batch_size (v[0]) = {padded}")

    # For each int in the vector that plausibly is a BYTE offset into buf,
    # read `padded` int32s there and see if they look like request ids in
    # [0, B). request_indices is the one whose values are all < B and whose
    # per-request tile COUNT matches ceil(qo_len / cta_tile_q).
    print("-- candidate offsets (read padded int32s, show head + range) --")
    seen = set()
    for i, off in enumerate(v):
        off = int(off)
        if off < 0 or off + 4 * padded > nbytes or off in seen:
            continue
        seen.add(off)
        # clone() -> contiguous copy at storage_offset 0 so .view(int32) is
        # aligned regardless of `off`'s alignment.
        vals = bb[off:off + 4 * padded].clone().view(torch.int32).cpu()
        vmin, vmax = int(vals.min()), int(vals.max())
        tag = ""
        # request_indices: every tile's request id, grouped (non-decreasing),
        # all in [0,B), covering all B requests. (prefill tiles = qo x kv, so
        # per-request counts != ceil(qo_len/ctq) -- don't match on that.)
        if 0 <= vmin and vmax < B:
            nondec = bool((vals[1:] >= vals[:-1]).all())
            covers = int(vals.unique().numel())
            counts = torch.bincount(vals.clamp(0, B - 1), minlength=B)
            tag = (f"  <== request_indices? nondec={nondec} covers={covers}/{B}"
                   f" per-req[:8]={counts[:8].tolist()}")
        elif vmax < padded:
            tag = f"  (tile-local index? qo/kv tile, range<padded={padded})"
        print(f"  v[{i:2d}]={off:>8}B  int32 head={vals[:12].tolist()} "
              f"range=[{vmin},{vmax}]{tag}")
    # cross-check: sum of per-request tile counts must equal padded (366)
    print(f"-- ntiles={padded}; sum(ceil(qo_len/{int(v[3])}))="
          f"{int(((qo_len + int(v[3]) - 1) // int(v[3])).sum())} "
          f"(qo-only tiles; < ntiles means kv-split too) --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
