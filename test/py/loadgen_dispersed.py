"""loadgen_dispersed.py -- a controlled high-dispersion load generator, the
clean E2E test for pi. Uses LOGNORMAL input lengths (the motivation's
straggler dispersion, not bench_serving's uniform ~1.5x) and a FIXED SEED so
stock/observe/enforce process the IDENTICAL request set -- the two things the
noisy uniform-random A/B lacked.

Each request: input_ids of a lognormal length (exact KV control via /generate
input_ids), fixed max_new_tokens with ignore_eos (comparable decode work),
temperature 0. Concurrency-bounded (queued regime when conc > R). Reports
throughput, median/p99 latency, and the achieved input p99/p50 dispersion.

  python loadgen_dispersed.py --port 30071 --n 2000 --conc 1024 \
    --out 128 --mu 5.8 --sigma 1.1 --lo 32 --hi 1536 --seed 42
"""
import argparse
import asyncio
import random
import time

import aiohttp


def make_requests(n, mu, sigma, lo, hi, out, seed):
    rng = random.Random(seed)
    reqs, lens = [], []
    for _ in range(n):
        L = int(round(2.718281828 ** rng.normalvariate(mu, sigma)))
        L = max(lo, min(hi, L))
        lens.append(L)
        ids = [rng.randint(1, 1000) for _ in range(L)]
        reqs.append({"input_ids": ids,
                     "sampling_params": {"max_new_tokens": out,
                                         "temperature": 0.0,
                                         "ignore_eos": True}})
    lens.sort()
    p50, p99 = lens[n // 2], lens[int(0.99 * n)]
    return reqs, (p50, p99, lens[0], lens[-1])


async def worker(sess, url, req, out, results):
    t0 = time.perf_counter()
    try:
        async with sess.post(url, json=req) as r:
            await r.json()
        dt = time.perf_counter() - t0
        results.append((dt, out))
    except Exception as e:
        results.append((None, str(e)[:40]))


async def run(args):
    reqs, disp = make_requests(args.n, args.mu, args.sigma, args.lo, args.hi,
                               args.out, args.seed)
    url = f"http://127.0.0.1:{args.port}/generate"
    sem = asyncio.Semaphore(args.conc)
    results = []
    conn = aiohttp.TCPConnector(limit=args.conc + 16)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async def bounded(sess, req):
        async with sem:
            await worker(sess, url, req, args.out, results)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
        await asyncio.gather(*(bounded(sess, r) for r in reqs))
    wall = time.perf_counter() - t0

    ok = [dt for dt, _ in results if dt is not None]
    ok.sort()
    errs = [m for dt, m in results if dt is None]
    tot_out = sum(o for dt, o in results if dt is not None)
    print(f"[loadgen] input dispersion p50={disp[0]} p99={disp[1]} "
          f"(p99/p50={disp[1]/max(disp[0],1):.1f}x) range[{disp[2]},{disp[3]}]")
    print(f"[loadgen] completed={len(ok)}/{args.n} errors={len(errs)} "
          f"wall={wall:.1f}s")
    if ok:
        print(f"[loadgen] Request throughput (req/s): {len(ok)/wall:8.2f}")
        print(f"[loadgen] Output token throughput (tok/s): {tot_out/wall:8.1f}")
        print(f"[loadgen] Median latency (ms): {1000*ok[len(ok)//2]:8.1f}")
        print(f"[loadgen] P99 latency (ms): {1000*ok[int(0.99*len(ok))]:8.1f}")
    if errs:
        print(f"[loadgen] sample errors: {errs[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30071)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--conc", type=int, default=1024)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--mu", type=float, default=5.8)      # exp(5.8)~330 median
    ap.add_argument("--sigma", type=float, default=1.1)   # wide -> high dispersion
    ap.add_argument("--lo", type=int, default=32)
    ap.add_argument("--hi", type=int, default=1536)       # KV-memory bound
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
