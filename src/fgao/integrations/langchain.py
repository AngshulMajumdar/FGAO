"""LangChain interop (requires ``langchain-core``; install ``fgao[langchain]``).

* ``LangChainLLM(model)``           any chat model / LLM with ``.invoke`` as an FGAO LLM
* ``LangChainRetriever(r)``         any retriever (``.invoke``) or vector store (``.similarity_search``)
* ``rag.as_runnable()``             FGAORAG as an LCEL Runnable (``|``, ``.batch``, ``.ainvoke``, ...)
* ``rag.as_tool()``                 FGAORAG as a tool for LangChain / LangGraph agents
* ``workflow_runnable(app)``        any compiled FGAO Workflow as a Runnable

``LangChainLLM`` and ``LangChainRetriever`` only duck-type, so they work
without importing LangChain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..rag.documents import Document, as_document


def _text(out: Any) -> str:
    content = getattr(out, "content", out)
    if isinstance(content, list):  # content blocks
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif hasattr(b, "text"):
                parts.append(getattr(b, "text"))
        return "".join(parts)
    return str(content)


class LangChainLLM:
    """Adapter: ``complete(prompt)`` -> ``model.invoke(prompt)`` text."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        return _text(self.model.invoke(prompt))

    async def acomplete(self, prompt: str) -> str:
        return _text(await self.model.ainvoke(prompt))


class LangChainRetriever:
    """Adapter for LangChain retrievers and vector stores.

    Vector stores are queried with ``similarity_search(query, k=k)``. A
    retriever returns its own number of documents; the first ``k`` are kept,
    so configure it to return at least the largest ``k`` you use (10 for the
    default source layout).
    """

    def __init__(self, obj: Any, search_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.obj = obj
        self.search_kwargs = dict(search_kwargs or {})

    def retrieve(self, query: str, k: int) -> List[Document]:
        if hasattr(self.obj, "similarity_search"):
            docs = self.obj.similarity_search(query, k=k, **self.search_kwargs)
        else:
            docs = self.obj.invoke(query, **self.search_kwargs) if self.search_kwargs else self.obj.invoke(query)
        return [as_document(d) for d in list(docs)[:k]]

    async def aretrieve(self, query: str, k: int) -> List[Document]:
        if hasattr(self.obj, "asimilarity_search"):
            docs = await self.obj.asimilarity_search(query, k=k, **self.search_kwargs)
        elif hasattr(self.obj, "ainvoke"):
            docs = await (self.obj.ainvoke(query, **self.search_kwargs) if self.search_kwargs else self.obj.ainvoke(query))
        else:
            import asyncio

            return await asyncio.to_thread(self.retrieve, query, k)
        return [as_document(d) for d in list(docs)[:k]]


def _require():
    try:
        import langchain_core  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("this needs langchain-core: pip install 'fgao[langchain]'") from e


def rag_runnable(rag):
    """FGAORAG as a ``RunnableLambda`` (sync + async)."""
    _require()
    from langchain_core.runnables import RunnableLambda

    def run(x):
        return rag.invoke(x).to_dict()

    async def arun(x):
        return (await rag.ainvoke(x)).to_dict()

    return RunnableLambda(run, afunc=arun, name="FGAORAG")


def rag_tool(rag, name: str = "fgao_rag", description: Optional[str] = None):
    """FGAORAG as a ``StructuredTool`` taking ``question: str``."""
    _require()
    from langchain_core.tools import StructuredTool

    desc = description or ("Answer a factual question from the configured document sources. Returns the "
                           "answer only when it is grounded in a passage and passes a support check; "
                           "otherwise reports that no verified answer was found.")

    def answer(question: str) -> Dict[str, Any]:
        r = rag.invoke(question)
        return {
            "answer": r.answer,
            "abstained": r.abstained,
            "evidence": r.support_passage.text if r.support_passage else None,
            "status": r.status,
        }

    async def aanswer(question: str) -> Dict[str, Any]:
        r = await rag.ainvoke(question)
        return {"answer": r.answer, "abstained": r.abstained,
                "evidence": r.support_passage.text if r.support_passage else None, "status": r.status}

    return StructuredTool.from_function(func=answer, coroutine=aanswer, name=name, description=desc)


def add_langchain_tool(wf, tool, *, inputs, output, name: Optional[str] = None, **declared):
    """Register a LangChain tool (``BaseTool`` / ``@tool``) as an FGAO workflow tool.

    inputs(state) -> the tool input (dict or str); output(state, result) -> state update.
    Sync and async calls use ``tool.invoke`` / ``tool.ainvoke``.
    """
    from ..workflow import ToolResult

    def fn(state, **_):
        return ToolResult(update=output(state, tool.invoke(inputs(state))))

    async def afn(state, **_):
        return ToolResult(update=output(state, await tool.ainvoke(inputs(state))))

    return wf.add_tool(name or tool.name, fn, afn=afn, description=getattr(tool, "description", ""), **declared)


def workflow_runnable(app, *, seed: int = 0):
    """A compiled ``Workflow`` as a Runnable: input state, output ``WorkflowResult``."""
    _require()
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(lambda s: app.invoke(s, seed=seed),
                          afunc=lambda s: app.ainvoke(s, seed=seed), name=f"FGAO:{app.name}")


__all__ = ["LangChainLLM", "LangChainRetriever", "rag_runnable", "rag_tool", "workflow_runnable", "add_langchain_tool"]
