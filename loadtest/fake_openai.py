"""A fake OpenAI-compatible chat endpoint for load and fault testing.

Answers FGAO's four prompts with the rule-based ToyLLM from ``examples/toy.py``
after a random delay, and injects faults: HTTP 500s and hung requests.

    python fake_openai.py --port 9100 --latency 0.15 0.45 --error-rate 0.05 --hang-rate 0.01
"""
import argparse
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from toy import ToyLLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9100)
ap.add_argument("--latency", type=float, nargs=2, default=(0.15, 0.45))
ap.add_argument("--error-rate", type=float, default=0.05)
ap.add_argument("--hang-rate", type=float, default=0.01)
ap.add_argument("--hang-seconds", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
A, _ = ap.parse_known_args()

rng = random.Random(A.seed)
llm = ToyLLM()
app = FastAPI()
counts = {"requests": 0, "errors": 0, "hangs": 0}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    counts["requests"] += 1
    prompt = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
    u = rng.random()
    if u < A.error_rate:
        counts["errors"] += 1
        await asyncio.sleep(rng.uniform(0.01, 0.05))
        return JSONResponse({"error": {"message": "injected failure", "type": "server_error"}}, status_code=500)
    if u < A.error_rate + A.hang_rate:
        counts["hangs"] += 1
        await asyncio.sleep(A.hang_seconds)
    await asyncio.sleep(rng.uniform(*A.latency))
    return {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body.get("model", "toy"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": llm.complete(prompt)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


@app.get("/stats")
async def stats():
    return counts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=A.port, log_level="warning")
