"""Generic LLM wrappers and coercion."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import threading
from typing import Any, Callable, Dict, Optional


async def acomplete(llm: Any, prompt: str) -> str:
    """Native ``acomplete`` when the LLM has one, otherwise ``complete`` in a worker thread."""
    fn = getattr(llm, "acomplete", None)
    if fn is not None:
        return await fn(prompt)
    return await asyncio.to_thread(llm.complete, prompt)


class FunctionLLM:
    """Wrap ``fn(prompt) -> str`` (or an ``async def`` function)."""

    def __init__(self, fn: Callable[[str], Any]) -> None:
        self.fn = fn
        if inspect.iscoroutinefunction(fn):
            self.acomplete = self._acomplete  # type: ignore[assignment]

    def complete(self, prompt: str) -> str:
        if inspect.iscoroutinefunction(self.fn):
            raise TypeError("this LLM is async-only; use the async API (ainvoke / abatch / astream)")
        return str(self.fn(prompt))

    async def _acomplete(self, prompt: str) -> str:
        return str(await self.fn(prompt))


class CachedLLM:
    """Thread-safe prompt -> response cache in front of any LLM.

    With ``path`` the cache persists as JSON (written after every new
    response). Useful for reproducible runs and to avoid paying twice for the
    same deterministic call.
    """

    def __init__(self, llm: Any, path: Optional[str] = None) -> None:
        self.llm = as_llm(llm)
        self.path = path
        self._lock = threading.Lock()
        self.cache: Dict[str, Dict[str, str]] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.cache = json.load(fh)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def complete(self, prompt: str) -> str:
        k = self.key(prompt)
        with self._lock:
            hit = self.cache.get(k)
            if hit is not None:
                self.hits += 1
                return hit["response"]
        response = self.llm.complete(prompt)
        self._store(k, prompt, response)
        return response

    async def acomplete(self, prompt: str) -> str:
        k = self.key(prompt)
        with self._lock:
            hit = self.cache.get(k)
            if hit is not None:
                self.hits += 1
                return hit["response"]
        response = await acomplete(self.llm, prompt)
        self._store(k, prompt, response)
        return response

    def _store(self, k: str, prompt: str, response: str) -> None:
        with self._lock:
            self.misses += 1
            self.cache[k] = {"prompt": prompt, "response": response}
            if self.path:
                tmp = self.path + ".tmp"
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self.cache, fh, ensure_ascii=False)
                os.replace(tmp, self.path)


def as_llm(x: Any):
    """Accept an object with ``complete``, a LangChain model, or ``fn(prompt) -> str``."""
    if x is None:
        raise ValueError("an LLM is required")
    if hasattr(x, "complete") and callable(getattr(x, "complete")):
        return x
    if hasattr(x, "invoke") and callable(getattr(x, "invoke")):
        from ..integrations.langchain import LangChainLLM

        return LangChainLLM(x)
    if callable(x):
        return FunctionLLM(x)
    raise TypeError(f"cannot use {type(x).__name__} as an LLM: give .complete(prompt), .invoke(prompt) or a function")
