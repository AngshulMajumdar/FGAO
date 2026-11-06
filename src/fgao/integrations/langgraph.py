"""LangGraph interop. Nodes are plain functions ``state -> partial update``, so
nothing here imports LangGraph.

    from langgraph.graph import StateGraph, START, END
    g = StateGraph(MyState)
    g.add_node("rag", rag.as_node())
    g.add_edge(START, "rag"); g.add_edge("rag", END)

``workflow_node`` embeds any compiled FGAO ``Workflow`` as a LangGraph node:
FGAO orchestrates the tools inside the node, LangGraph routes around it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional


def rag_node(rag, *, question_key: str = "question", answer_key: str = "answer",
             result_key: Optional[str] = "fgao") -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    def node(state: Mapping[str, Any]) -> Dict[str, Any]:
        r = rag.invoke(state[question_key])
        out: Dict[str, Any] = {answer_key: r.answer}
        if result_key:
            out[result_key] = r.to_dict()
        return out

    node.__name__ = "fgao_rag"
    return node


def workflow_node(app, *, input_key: Optional[str] = None, output_key: str = "fgao",
                  seed: int = 0) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    """Run a compiled Workflow on ``dict(state)`` (or ``state[input_key]``).

    Returns the final workflow state's keys as the update (or, with
    ``input_key``, ``{input_key: final_state}``), plus a summary under ``output_key``.
    """

    def node(state: Mapping[str, Any]) -> Dict[str, Any]:
        inner = dict(state[input_key]) if input_key else dict(state)
        r = app.invoke(inner, seed=seed)
        upd: Dict[str, Any] = {input_key: r.state} if input_key else dict(r.state)
        upd[output_key] = {"status": r.status, "success": r.success, "values": r.values, "cost": r.cost,
                           "trace": r.trace()}
        return upd

    node.__name__ = f"fgao_{app.name}"
    return node


__all__ = ["rag_node", "workflow_node"]
