"""sched_trace_loadgen.py -- trace-driven load generator that REPLAYS the real
Qwen-Bailian usage trace (alibaba-edu/qwen-bailian-usagetraces-anon) against a
woven or stock SGLang server. The raw prompts are absent by design, but the trace
carries what the scheduler actually reacts to: per-request input_length /
output_length, arrival timestamps, and hash_ids -- the 16-token KV-block hashes
that ENCODE the cross-request shared-prefix structure.

FAITHFUL RE-CRAFT (the point):
  * Each KV block hash_id -> a deterministic block of 16 token-ids.
  * A request's prompt = concat of its hash_ids' blocks, trimmed to input_length.
  * => two requests that share a hash_id PREFIX share an input_ids PREFIX, so
    SGLang's radix tree sees the SAME reuse the production trace had (39% block
    reuse in the To-C sample). We send input_ids directly (/generate) so lengths
    and shared prefixes are EXACT -- no lossy tokenizer round-trip.

This is the load side of the moat's live validation: heavy-tailed input_length
drives the dispersion/straggler regime (ORDER, split-skew); the reconstructed
shared prefixes drive the radix/SHAPE regime; output_length sets decode work.

Offline (no server): `python sched_trace_loadgen.py --selftest` verifies the
re-craft (shared-prefix reconstruction + exact lengths). Live: `--base-url
http://127.0.0.1:30000 --trace <file.jsonl>` replays by timestamp.
"""
import argparse
import json
import os
import random
import sys
import time

BLOCK = 16          # tokens per KV block (matches blksz_16 == our PAGE=16)
VOCAB = 151936      # Qwen2 vocab; token-ids only need to be valid, not meaningful
_BLOCK_CACHE = {}


def block_tokens(hash_id, vocab=VOCAB, block=BLOCK):
    """Deterministic 16 token-ids for a KV-block hash. Same hash_id -> same
    block, ALWAYS (across requests and processes) -> shared hash_id => shared
    tokens => radix reuse. Cached; seeded only by hash_id (reproducible)."""
    key = (hash_id, vocab, block)
    b = _BLOCK_CACHE.get(key)
    if b is None:
        r = random.Random(hash_id * 2654435761 + 0x9E3779B9)
        b = [r.randrange(vocab) for _ in range(block)]
        _BLOCK_CACHE[key] = b
    return b


def craft_input_ids(rec, vocab=VOCAB, block=BLOCK):
    """Reconstruct a request's input_ids from its hash_ids, trimmed/padded to the
    recorded input_length. Shared hash_id prefixes -> shared token prefixes."""
    want = int(rec.get("input_length", 0)) or block
    ids = []
    for h in rec.get("hash_ids", []):
        ids.extend(block_tokens(h, vocab, block))
        if len(ids) >= want:
            break
    if len(ids) < want:                      # pad tail deterministically
        pad = block_tokens(0x7FFFFFFF ^ int(rec.get("chat_id", 0)), vocab, block)
        while len(ids) < want:
            ids.extend(pad)
    return ids[:want]


def load_trace(path, limit=None):
    """Parse a trace jsonl. Tolerant of a truncated final line (range-fetched
    samples). Returns records sorted by arrival timestamp."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break                        # truncated tail (partial fetch)
            if limit and len(recs) >= limit:
                break
    recs.sort(key=lambda r: r.get("timestamp", 0.0))
    return recs


async def _fire(client, base_url, ids, max_new, sem, results, idx, rec):
    import httpx  # noqa: F401 (imported lazily; only needed for live replay)
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
        "stream": False,
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{base_url}/generate", json=payload, timeout=600)
            ok = r.status_code == 200
        except Exception:
            ok = False
        dt = time.perf_counter() - t0
    results[idx] = {"lat_s": dt, "ok": ok, "in": len(ids),
                    "out": max_new, "type": rec.get("type"),
                    "chat_id": rec.get("chat_id")}


async def replay(recs, base_url, speed=1.0, concurrency=32, vocab=VOCAB):
    """Replay records by arrival timestamp (wall-clock / speed). Returns per-
    request results. speed>1 compresses time; speed=0 fires as fast as possible."""
    import asyncio

    import httpx
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(recs)
    t_start = time.perf_counter()
    t0_trace = recs[0].get("timestamp", 0.0) if recs else 0.0
    async with httpx.AsyncClient() as client:
        tasks = []
        for idx, rec in enumerate(recs):
            if speed > 0:
                due = (rec.get("timestamp", 0.0) - t0_trace) / speed
                now = time.perf_counter() - t_start
                if due > now:
                    await asyncio.sleep(due - now)
            ids = craft_input_ids(rec, vocab)
            max_new = max(1, int(rec.get("output_length", 1)))
            tasks.append(asyncio.create_task(
                _fire(client, base_url, ids, max_new, sem, results, idx, rec)))
        await asyncio.gather(*tasks)
    return [r for r in results if r]


def summarize(results, wall_s, tag=""):
    lat = sorted(r["lat_s"] for r in results if r["ok"])
    n_ok = len(lat)
    n = len(results)
    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float("nan")
    print(f"== replay summary {tag} ==")
    print(f"  requests   : {n_ok}/{n} ok")
    print(f"  wall       : {wall_s:.1f}s ({n_ok/wall_s:.1f} req/s completed)")
    if lat:
        print(f"  latency s  : p50={pct(.5):.2f} p90={pct(.9):.2f} "
              f"p99={pct(.99):.2f} max={lat[-1]:.2f}")
    return {"n_ok": n_ok, "n": n, "p50": pct(.5), "p90": pct(.9), "p99": pct(.99)}


def selftest():
    """Verify the re-craft offline: shared hash_id prefixes -> shared input_ids
    prefixes, and crafted length == input_length. No server needed."""
    fails = 0
    def ok(c, name):
        nonlocal fails
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
        fails += 0 if c else 1
    print("== trace loadgen re-craft self-test ==")
    # determinism: same hash_id -> same block, always
    ok(block_tokens(42) == block_tokens(42), "block_tokens deterministic")
    ok(block_tokens(42) != block_tokens(43), "distinct hash -> distinct block")
    # shared prefix: two records sharing the first K hash_ids share K*16 tokens
    rA = {"chat_id": 1, "input_length": 96,
          "hash_ids": [10, 11, 12, 13, 14, 15]}          # 6 blocks = 96 tok
    rB = {"chat_id": 2, "input_length": 96,
          "hash_ids": [10, 11, 12, 99, 98, 97]}          # shares first 3 blocks
    a = craft_input_ids(rA)
    b = craft_input_ids(rB)
    ok(len(a) == 96 and len(b) == 96, "crafted length == input_length")
    ok(a[:48] == b[:48], "shared 3-block prefix -> identical 48-token prefix")
    ok(a[48:] != b[48:], "divergent tail -> different tokens")
    # trimming: input_length not a multiple of block
    rC = {"chat_id": 3, "input_length": 40, "hash_ids": [1, 2, 3, 4]}
    ok(len(craft_input_ids(rC)) == 40, "trims to non-block-multiple length")
    # padding: input_length exceeds available blocks
    rD = {"chat_id": 4, "input_length": 200, "hash_ids": [1, 2]}
    ok(len(craft_input_ids(rD)) == 200, "pads when hash_ids under-cover")
    # real fixture (if present): exact lengths + FAITHFUL reuse reconstruction.
    # block_tokens is a deterministic injection hash_id->block, so the crafted
    # 16-token-block reuse must equal the trace's hash_id reuse.
    fix = os.path.join(os.path.dirname(__file__), "..", "data", "traces",
                       "qwen_traceA_blksz_16.sample.jsonl")
    if os.path.exists(fix):
        recs = load_trace(fix)
        ok(len(recs) > 0, f"real fixture parsed ({len(recs)} records)")
        ok(all(len(craft_input_ids(r)) == r["input_length"] for r in recs),
           "every crafted input_ids == recorded input_length")
        seen_h, th, rh = set(), 0, 0
        for r in recs:
            for h in r["hash_ids"]:
                th += 1; rh += (h in seen_h); seen_h.add(h)
        seen_b, tb, rb = set(), 0, 0
        for r in recs:
            ids = craft_input_ids(r)
            for i in range(0, (len(ids) // BLOCK) * BLOCK, BLOCK):
                blk = tuple(ids[i:i + BLOCK]); tb += 1
                rb += (blk in seen_b); seen_b.add(blk)
        hr, br = (rh / th if th else 0), (rb / tb if tb else 0)
        ok(abs(hr - br) < 0.01,
           f"crafted block reuse {br:.1%} == trace hash reuse {hr:.1%} (faithful)")
    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="path to a qwen_*_blksz_16.jsonl trace")
    ap.add_argument("--base-url", help="SGLang server, e.g. http://127.0.0.1:30000")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.0,
                    help="time compression; 0 = as-fast-as-possible")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.base_url:
        return selftest()
    if not a.trace:
        print("need --trace for a live replay", file=sys.stderr)
        return 2
    recs = load_trace(a.trace, a.limit)
    print(f"loaded {len(recs)} records from {os.path.basename(a.trace)}")
    import asyncio
    t0 = time.perf_counter()
    results = asyncio.run(replay(recs, a.base_url.rstrip("/"), a.speed,
                                 a.concurrency, a.vocab))
    summarize(results, time.perf_counter() - t0, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
