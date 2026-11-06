"""Built-in retrievers and retriever coercion."""
from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections import Counter
from typing import Any, Callable, List, Optional, Sequence

from .documents import Document, as_document, as_documents

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for and or is are was were be been by with as from that this which who whom what "
            "when where why how did does do it its into than then there their they he she his her".split())


def tokens(text: str) -> List[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    """In-memory Okapi BM25 over title + text. No dependencies."""

    def __init__(self, documents: Sequence[Any], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = as_documents(documents)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d.title + " " + d.text)) for d in self.docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self._post: dict = {}
        for i, t in enumerate(self.tf):
            for w in t:
                self._post.setdefault(w, []).append(i)

    def retrieve(self, query: str, k: int) -> List[Document]:
        q = tokens(query)
        scores: dict = {}
        for w in q:
            for i in self._post.get(w, ()):
                f = self.tf[i][w]
                scores[i] = scores.get(i, 0.0) + self.idf[w] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        ranked = sorted(((s, i) for i, s in scores.items() if s > 0), key=lambda x: (-x[0], x[1]))
        return [self.docs[i] for _, i in ranked[:k]]

    async def aretrieve(self, query: str, k: int) -> List[Document]:
        # in-memory and sub-millisecond for typical corpora: cheaper inline than a thread hop
        return self.retrieve(query, k)


async def aretrieve(retriever: Any, query: str, k: int) -> List[Document]:
    """Native ``aretrieve`` when available, otherwise ``retrieve`` in a worker thread."""
    fn = getattr(retriever, "aretrieve", None)
    if fn is not None:
        return list(await fn(query, k))
    return list(await asyncio.to_thread(retriever.retrieve, query, k))


class FunctionRetriever:
    """Wrap ``fn(query, k) -> iterable of Document | str | dict`` (or an ``async def`` function)."""

    def __init__(self, fn: Callable[[str, int], Any]) -> None:
        self.fn = fn
        if inspect.iscoroutinefunction(fn):
            self.aretrieve = self._aretrieve  # type: ignore[assignment]

    def retrieve(self, query: str, k: int) -> List[Document]:
        if inspect.iscoroutinefunction(self.fn):
            raise TypeError("this retriever is async-only; use the async API")
        return as_documents(self.fn(query, k))[:k]

    async def _aretrieve(self, query: str, k: int) -> List[Document]:
        return as_documents(await self.fn(query, k))[:k]


class DenseRetriever:
    """Cosine-similarity retriever over precomputed embeddings.

    ``embed(texts: list[str]) -> list[list[float]]`` is any embedding function,
    e.g. ``fgao.integrations.huggingface.hf_embedder("BAAI/bge-small-en-v1.5")``.
    Uses numpy when installed, plain Python otherwise.
    """

    def __init__(self, documents: Sequence[Any], embed: Callable[[List[str]], Any], batch_size: int = 64) -> None:
        self.docs = as_documents(documents)
        self.embed = embed
        vecs: List[List[float]] = []
        texts = [(d.title + "\n" + d.text).strip() for d in self.docs]
        for i in range(0, len(texts), batch_size):
            vecs.extend(_rows(embed(texts[i:i + batch_size])))
        self._np = _numpy()
        if self._np is not None:
            m = self._np.asarray(vecs, dtype=float)
            self._m = m / (self._np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        else:
            self._m = [_unit(v) for v in vecs]

    def retrieve(self, query: str, k: int) -> List[Document]:
        (q,) = _rows(self.embed([query]))
        if self._np is not None:
            qv = self._np.asarray(q, dtype=float)
            sims = self._m @ (qv / (self._np.linalg.norm(qv) + 1e-12))
            order = self._np.argsort(-sims, kind="stable")[:k]
            return [self.docs[int(i)] for i in order]
        qv = _unit(q)
        sims = [sum(a * b for a, b in zip(row, qv)) for row in self._m]
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[:k]
        return [self.docs[i] for i in order]


def _numpy():
    try:
        import numpy

        return numpy
    except ImportError:  # pragma: no cover
        return None


def _rows(x: Any) -> List[List[float]]:
    rows = x.tolist() if hasattr(x, "tolist") else list(x)
    return [list(map(float, r)) for r in rows]


def _unit(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(a * a for a in v)) or 1.0
    return [a / n for a in v]


def as_retriever(x: Any):
    """Accept a retriever, a LangChain retriever / vector store, a list of texts or documents
    (indexed with BM25), or ``fn(query, k)``."""
    if x is None:
        return None
    if hasattr(x, "retrieve") and callable(getattr(x, "retrieve")):
        return x
    if hasattr(x, "similarity_search") or (hasattr(x, "invoke") and callable(getattr(x, "invoke"))):
        from ..integrations.langchain import LangChainRetriever

        return LangChainRetriever(x)
    if isinstance(x, (list, tuple)):
        return BM25Retriever(x)
    if callable(x):
        return FunctionRetriever(x)
    raise TypeError(f"cannot use {type(x).__name__} as a retriever")


__all__ = ["BM25Retriever", "FunctionRetriever", "DenseRetriever", "as_retriever", "aretrieve", "as_document", "tokens"]
