"""Pluggable interfaces. Any object with the right method (or a plain function) works.

| role            | interface                                                   |
|-----------------|-------------------------------------------------------------|
| LLM             | ``complete(prompt: str) -> str``                            |
| Retriever       | ``retrieve(query: str, k: int) -> list[Document]``          |
| Generator       | ``(question, docs) -> str``  ('' means abstain)             |
| Grader          | ``(question, docs) -> list[int]`` (0, 1, 2 per passage)     |
| Rewriter        | ``(question) -> str`` (a search query; '' keeps the question)|
| SupportChecker  | ``(question, passage, answer) -> bool``                     |

LangChain chat models / retrievers / vector stores, Hugging Face clients, and
plain callables are converted automatically by ``as_llm`` / ``as_retriever``.
"""
from __future__ import annotations

from typing import Callable, List, Protocol, Sequence, runtime_checkable

from .documents import Document


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> List[Document]: ...


Generator = Callable[[str, Sequence[Document]], str]
Grader = Callable[[str, Sequence[Document]], List[int]]
Rewriter = Callable[[str], str]
SupportChecker = Callable[[str, Document, str], bool]
