"""Closed-loop load generator for an FGAO service.

    python loadgen.py --url http://127.0.0.1:8000 --questions data/questions.jsonl --concurrency 64 --requests 2000

Prints throughput, latency percentiles, HTTP codes, answer correctness against
``expected`` (null = the right output is an abstention) and mean cost.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--api-key")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    lat, codes, right, cost, verified = [], {}, 0, [], 0
    lock = asyncio.Lock()
    nxt = iter(range(a.requests))
    limits = httpx.Limits(max_connections=a.concurrency, max_keepalive_connections=a.concurrency)
    async with httpx.AsyncClient(base_url=a.url, timeout=a.timeout, headers=headers, limits=limits) as c:
        async def worker():
            nonlocal right, verified
            for i in nxt:
                q = qs[i % len(qs)]
                t0 = time.perf_counter()
                try:
                    r = await c.post("/v1/answer", json={"question": q["question"], "trace": False})
                    code = r.status_code
                except httpx.HTTPError as e:
                    code = type(e).__name__
                dt = time.perf_counter() - t0
                async with lock:
                    codes[code] = codes.get(code, 0) + 1
                    if code == 200:
                        d = r.json()
                        lat.append(dt)
                        cost.append(d["cost"])
                        verified += d["answer"] is not None
                        right += d["answer"] == q["expected"]

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(a.concurrency)))
        wall = time.perf_counter() - t0
        metrics = (await c.get("/metrics")).text
    deps = [l for l in metrics.splitlines() if l.startswith("fgao_dependency_") and 'name="llm"' in l]
    ok = codes.get(200, 0)
    print(json.dumps({
        "concurrency": a.concurrency, "requests": a.requests, "seconds": round(wall, 2),
        "throughput_qps": round(a.requests / wall, 1), "http_codes": {str(k): v for k, v in codes.items()},
        "latency_s": {"p50": round(pct(lat, 50), 3), "p95": round(pct(lat, 95), 3), "p99": round(pct(lat, 99), 3),
                      "max": round(max(lat), 3) if lat else None},
        "correct": f"{right}/{ok}", "verified_answers": verified,
        "mean_cost": round(statistics.mean(cost), 2) if cost else None,
        "llm_dependency": {l.split("{")[0].replace("fgao_dependency_", "").replace("_total", ""): float(l.split()[-1])
                           for l in deps},
    }, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
