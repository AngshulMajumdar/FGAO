"""Default LLM-backed RAG operations (the evaluated v2 prompts and parsers)."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, List, Optional, Sequence

from . import prompts as P
from .documents import Document
from .llms import acomplete, as_llm


class LLMGenerator:
    """Extractive answer generation; returns '' to abstain."""

    def __init__(self, llm: Any, prompt: Callable[[str, Sequence[Document]], str] = P.generate_prompt,
                 parse: Callable[[str], str] = P.parse_answer) -> None:
        self.llm, self.prompt, self.parse = as_llm(llm), prompt, parse

    def __call__(self, question: str, docs: Sequence[Document]) -> str:
        return self.parse(self.llm.complete(self.prompt(question, docs)))

    async def acall(self, question: str, docs: Sequence[Document]) -> str:
        return self.parse(await acomplete(self.llm, self.prompt(question, docs)))


class LLMGrader:
    """Relevance grading: one score in {0, 1, 2} per passage; passages scoring >= 1 are kept."""

    def __init__(self, llm: Any, prompt: Callable[[str, Sequence[Document]], str] = P.grade_prompt,
                 parse: Callable[[str, int], List[int]] = P.parse_grades) -> None:
        self.llm, self.prompt, self.parse = as_llm(llm), prompt, parse

    def __call__(self, question: str, docs: Sequence[Document]) -> List[int]:
        return self.parse(self.llm.complete(self.prompt(question, docs)), len(docs))

    async def acall(self, question: str, docs: Sequence[Document]) -> List[int]:
        return self.parse(await acomplete(self.llm, self.prompt(question, docs)), len(docs))


class LLMRewriter:
    """Query rewriting for retrieval; '' keeps the original question."""

    def __init__(self, llm: Any, prompt: Callable[[str], str] = P.rewrite_prompt,
                 parse: Callable[[str], str] = P.parse_query) -> None:
        self.llm, self.prompt, self.parse = as_llm(llm), prompt, parse

    def __call__(self, question: str) -> str:
        return self.parse(self.llm.complete(self.prompt(question)))

    async def acall(self, question: str) -> str:
        return self.parse(await acomplete(self.llm, self.prompt(question)))


class LLMSupportChecker:
    """Support check: does the passage that contains the answer support it?"""

    def __init__(self, llm: Any, prompt: Callable[[str, Document, str], str] = P.verify_prompt,
                 parse: Callable[[str], bool] = P.parse_yes) -> None:
        self.llm, self.prompt, self.parse = as_llm(llm), prompt, parse

    def __call__(self, question: str, passage: Document, answer: str) -> bool:
        return bool(self.parse(self.llm.complete(self.prompt(question, passage, answer))))

    async def acall(self, question: str, passage: Document, answer: str) -> bool:
        return bool(self.parse(await acomplete(self.llm, self.prompt(question, passage, answer))))


async def acall(fn: Any, *args: Any) -> Any:
    """Call an operation natively async when it supports it, otherwise in a worker thread."""
    a = getattr(fn, "acall", None)
    if a is not None:
        return await a(*args)
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
        return await fn(*args)
    return await asyncio.to_thread(fn, *args)
