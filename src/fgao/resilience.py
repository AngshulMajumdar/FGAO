"""Retries, timeouts, rate limiting and concurrency caps for LLMs and retrievers.

    from fgao.resilience import resilient

    llm = resilient(OpenAILLM("gpt-4o-mini"), attempts=4, timeout=20, rate=50, max_concurrency=64)
    kb  = resilient(my_retriever, attempts=3, timeout=5)

``resilient(x)`` returns an object with the same interface as ``x`` (``complete`` /
``acomplete`` for an LLM, ``retrieve`` / ``aretrieve`` for a retriever):

* retries with exponential backoff and full jitter on any exception in ``retry_on``
* per-attempt ``timeout`` (seconds) on the async path; the sync path relies on the
  client's own timeout, which you should also set
* ``rate``: token-bucket limit on calls per second (shared by all threads and tasks)
* ``max_concurrency``: cap on simultaneous calls
* ``stats``: calls, attempts, retries, timeouts, failures (thread-safe counters)

A call that still fails after the last attempt raises. Combine with
``FGAORAG(..., on_error="record")`` to let FGAO continue without that operation.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type


@dataclass
class RetryPolicy:
    attempts: int = 3
    backoff: float = 0.5  # base delay (s); attempt i waits U(0, min(max_backoff, backoff * 2**i))
    max_backoff: float = 8.0
    timeout: Optional[float] = None  # per attempt, async path
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)

    def delay(self, i: int, rng: random.Random) -> float:
        return rng.uniform(0.0, min(self.max_backoff, self.backoff * (2 ** i)))


class RateLimiter:
    """Token bucket: ``rate`` calls per second, burst ``burst`` (default: ``rate``)."""

    def __init__(self, rate: float, burst: Optional[float] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1.0, rate))
        self.tokens = self.capacity
        self.t = time.monotonic()
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        """Take a token; return how long the caller must wait before using it."""
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            self.tokens -= 1.0
            return 0.0 if self.tokens >= 0 else -self.tokens / self.rate

    def acquire(self) -> None:
        w = self._reserve()
        if w > 0:
            time.sleep(w)

    async def aacquire(self) -> None:
        w = self._reserve()
        if w > 0:
            await asyncio.sleep(w)


@dataclass
class Stats:
    calls: int = 0
    attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **kw: int) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> Dict[str, int]:
        return {k: getattr(self, k) for k in ("calls", "attempts", "retries", "timeouts", "failures")}


class _Resilient:
    def __init__(self, inner: Any, policy: RetryPolicy, rate: Optional[float], max_concurrency: Optional[int],
                 name: str, seed: Optional[int]) -> None:
        self.inner = inner
        self.policy = policy
        self.limiter = RateLimiter(rate) if rate else None
        self.max_concurrency = max_concurrency
        self._tsem = threading.BoundedSemaphore(max_concurrency) if max_concurrency else None
        self._asems: Dict[int, asyncio.Semaphore] = {}
        self.name = name
        self.stats = Stats()
        self._rng = random.Random(seed)

    def _asem(self) -> Optional[asyncio.Semaphore]:
        if not self.max_concurrency:
            return None
        loop = id(asyncio.get_running_loop())
        sem = self._asems.get(loop)
        if sem is None:
            sem = self._asems[loop] = asyncio.Semaphore(self.max_concurrency)
        return sem

    def _sync(self, fn, *args):
        p = self.policy
        self.stats.add(calls=1)
        for i in range(max(1, p.attempts)):
            if self.limiter:
                self.limiter.acquire()
            self.stats.add(attempts=1)
            try:
                if self._tsem:
                    with self._tsem:
                        return fn(*args)
                return fn(*args)
            except p.retry_on:
                if i + 1 >= p.attempts:
                    self.stats.add(failures=1)
                    raise
                self.stats.add(retries=1)
                time.sleep(p.delay(i, self._rng))
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _async(self, make, *args):
        p = self.policy
        self.stats.add(calls=1)
        sem = self._asem()
        for i in range(max(1, p.attempts)):
            if self.limiter:
                await self.limiter.aacquire()
            self.stats.add(attempts=1)
            try:
                coro = make(*args)
                if sem is not None:
                    async with sem:
                        return await (asyncio.wait_for(coro, p.timeout) if p.timeout else coro)
                return await (asyncio.wait_for(coro, p.timeout) if p.timeout else coro)
            except asyncio.TimeoutError:
                self.stats.add(timeouts=1)
                if i + 1 >= p.attempts:
                    self.stats.add(failures=1)
                    raise
                self.stats.add(retries=1)
            except asyncio.CancelledError:
                raise
            except p.retry_on:
                if i + 1 >= p.attempts:
                    self.stats.add(failures=1)
                    raise
                self.stats.add(retries=1)
            await asyncio.sleep(p.delay(i, self._rng))
        raise RuntimeError("unreachable")  # pragma: no cover


class ResilientLLM(_Resilient):
    def complete(self, prompt: str) -> str:
        return self._sync(self.inner.complete, prompt)

    async def acomplete(self, prompt: str) -> str:
        from .rag.llms import acomplete

        return await self._async(lambda p: acomplete(self.inner, p), prompt)


class ResilientRetriever(_Resilient):
    def retrieve(self, query: str, k: int):
        return self._sync(self.inner.retrieve, query, k)

    async def aretrieve(self, query: str, k: int):
        from .rag.retrievers import aretrieve

        return await self._async(lambda q, kk: aretrieve(self.inner, q, kk), query, k)


def resilient(x: Any, *, attempts: int = 3, backoff: float = 0.5, max_backoff: float = 8.0,
              timeout: Optional[float] = None, rate: Optional[float] = None, max_concurrency: Optional[int] = None,
              retry_on: Tuple[Type[BaseException], ...] = (Exception,), name: str = "", seed: Optional[int] = None,
              kind: Optional[str] = None):
    """Wrap an LLM or retriever (anything ``as_llm`` / ``as_retriever`` accepts).

    ``kind="llm"`` or ``"retriever"`` resolves objects that look like both
    (e.g. LangChain runnables).
    """
    from .rag.llms import as_llm
    from .rag.retrievers import as_retriever

    policy = RetryPolicy(attempts, backoff, max_backoff, timeout, retry_on)
    looks_retriever = (hasattr(x, "retrieve") or isinstance(x, (list, tuple)) or hasattr(x, "similarity_search")
                       or hasattr(x, "_get_relevant_documents"))
    if kind == "retriever" or (kind is None and looks_retriever):
        return ResilientRetriever(as_retriever(x), policy, rate, max_concurrency, name or "retriever", seed)
    return ResilientLLM(as_llm(x), policy, rate, max_concurrency, name or "llm", seed)


__all__ = ["resilient", "RetryPolicy", "RateLimiter", "ResilientLLM", "ResilientRetriever", "Stats"]
