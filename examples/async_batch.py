"""Serving many questions: thread-pooled batch and asyncio.

    python async_batch.py
"""
import asyncio

from fgao import CachedLLM, FGAORAG
from toy import KB, WEB, ToyLLM

llm = CachedLLM(ToyLLM())  # thread-safe cache: identical prompts are paid once
rag = FGAORAG.from_retrievers(kb=KB, web=WEB, llm=llm)
questions = [f"What is the capital of {c}?" for c in ("France", "Japan", "Kenya", "Spain", "Peru", "Italy")] * 20

results = rag.batch(questions, max_concurrency=16)
print(sum(r.answer is not None for r in results), "verified answers of", len(results))


async def main():
    rs = await rag.abatch(questions[:6], max_concurrency=6)
    print([r.answer for r in rs])
    async for event in rag.astream("What is the capital of Peru?"):
        print(type(event).__name__, getattr(event, "action", getattr(event, "answer", None)))


asyncio.run(main())
print("cache hits:", llm.hits, "misses:", llm.misses)
