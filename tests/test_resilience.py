import asyncio
import time

import pytest

from fgao import FGAORAG
from fgao.resilience import RateLimiter, ResilientLLM, ResilientRetriever, resilient

from test_rag import KB, WEB, Scripted


class Flaky:
    def __init__(self, fails, inner=None):
        self.fails, self.n, self.inner = fails, 0, inner or Scripted()

    def complete(self, prompt):
        self.n += 1
        if self.n <= self.fails:
            raise ConnectionError("boom")
        return self.inner(prompt)


def test_sync_retries_then_succeeds_and_counts():
    llm = resilient(Flaky(2), attempts=3, backoff=0.0, seed=0)
    assert isinstance(llm, ResilientLLM)
    assert llm.complete("Does the passage fully support x") == "yes"
    assert llm.stats.as_dict() == {"calls": 1, "attempts": 3, "retries": 2, "timeouts": 0, "failures": 0}
    llm = resilient(Flaky(5), attempts=2, backoff=0.0)
    with pytest.raises(ConnectionError):
        llm.complete("x")
    assert llm.stats.failures == 1


def test_async_timeout_retry():
    class Slow:
        def __init__(self):
            self.n = 0

        async def acomplete(self, prompt):
            self.n += 1
            await asyncio.sleep(1.0 if self.n == 1 else 0)
            return "ok"

        def complete(self, prompt):  # pragma: no cover
            return "ok"

    llm = resilient(Slow(), attempts=2, timeout=0.05, backoff=0.0)
    assert asyncio.run(llm.acomplete("x")) == "ok"
    assert llm.stats.timeouts == 1 and llm.stats.retries == 1


def test_retry_on_filter():
    llm = resilient(Flaky(1), attempts=3, backoff=0.0, retry_on=(TimeoutError,))
    with pytest.raises(ConnectionError):
        llm.complete("x")


def test_retriever_wrapping_and_rate_limit():
    r = resilient(KB, attempts=2)
    assert isinstance(r, ResilientRetriever) and r.retrieve("France", 1)[0].text == KB[0]
    assert asyncio.run(r.aretrieve("France", 1))[0].text == KB[0]
    rl = RateLimiter(rate=50, burst=1)
    t0 = time.monotonic()
    for _ in range(6):
        rl.acquire()
    assert time.monotonic() - t0 >= 0.09


def test_max_concurrency_async():
    state = {"cur": 0, "max": 0}

    class LLM:
        def complete(self, p):  # pragma: no cover
            return ""

        async def acomplete(self, p):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1
            return ""

    llm = resilient(LLM(), max_concurrency=3)

    async def go():
        await asyncio.gather(*(llm.acomplete("x") for _ in range(20)))

    asyncio.run(go())
    assert state["max"] == 3


def test_on_error_record_keeps_answering():
    class Broken:
        def retrieve(self, q, k):
            raise ConnectionError("kb down")

    rag = FGAORAG.from_retrievers(Broken(), Scripted(), web=WEB, on_error="record")
    r = rag.invoke("What is the capital of Canada?")
    assert r.answer == "Ottawa" and any(s.error and "kb down" in s.error for s in r.steps)
    with pytest.raises(ConnectionError):
        FGAORAG.from_retrievers(Broken(), Scripted(), web=WEB).invoke("What is the capital of Canada?")
    r = asyncio.run(rag.ainvoke("What is the capital of Canada?"))
    assert r.answer == "Ottawa"
