"""MCP (Model Context Protocol) tools as FGAO workflow tools.

Works with any client exposing ``await call_tool(name, arguments)`` and
``await list_tools()``: ``mcp.Client`` (SDK 2.x) or ``mcp.ClientSession``
(SDK 1.x). FGAO decides *when* to call a tool from its declared effects; the
``inputs`` function builds the call's arguments from the workflow state and
``output`` turns the result into a state update.

    from mcp import Client
    from fgao import Workflow
    from fgao.integrations.mcp import add_mcp_tool

    async with Client("http://localhost:8000/mcp") as client:
        wf = Workflow("ops")
        wf.add_goal("ticket", lambda s: "ticket_id" in s)
        add_mcp_tool(wf, client, "create_ticket", inputs=lambda s: {"title": s["title"]},
                     output=lambda s, r: {"ticket_id": r.data["id"]}, gain={"ticket": 0.9}, cost=1)
        result = await wf.compile(budget=10).ainvoke({"title": "disk full"})

MCP calls are async, so these tools run through the async API
(``ainvoke`` / ``abatch`` / ``astream`` / ``Run.arun``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..types import Action
from ..workflow import ToolResult, Workflow


@dataclass
class MCPResult:
    """Normalised tool result: concatenated text, structured content (if any), error flag."""

    text: str
    data: Any = None
    is_error: bool = False
    raw: Any = field(default=None, repr=False)

    @classmethod
    def of(cls, res: Any) -> "MCPResult":
        content = getattr(res, "content", None) or []
        text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", "text") == "text")
        data = getattr(res, "structured_content", None)
        if data is None:
            data = getattr(res, "structuredContent", None)
        err = getattr(res, "is_error", None)
        if err is None:
            err = getattr(res, "isError", False)
        return cls(text=text, data=data, is_error=bool(err), raw=res)


async def list_mcp_tools(client: Any) -> List[Dict[str, Any]]:
    """[{"name", "description", "input_schema"}] for every tool the server offers."""
    res = await client.list_tools()
    out = []
    for t in getattr(res, "tools", res):
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        out.append({"name": t.name, "description": getattr(t, "description", "") or "", "input_schema": schema})
    return out


def add_mcp_tool(
    wf: Workflow,
    client: Any,
    tool: str,
    *,
    inputs: Callable[..., Mapping[str, Any]],
    output: Callable[[Any, MCPResult], Optional[Mapping[str, Any]]],
    name: Optional[str] = None,
    args: Optional[Mapping[str, Any]] = None,
    timeout: Optional[float] = None,
    **declared: Any,
) -> Action:
    """Register MCP tool ``tool`` as an FGAO tool.

    inputs(state, **args) -> the tool arguments; output(state, MCPResult) -> state update.
    An MCP error result is recorded as a failed call (no update), so FGAO's feedback learns from it.
    ``declared`` are the usual ``add_tool`` declarations (gain, harm, produces, requires, cost, group).
    """

    async def afn(state: Any, **kw: Any):
        call_kw = {"read_timeout_seconds": timeout} if timeout is not None else {}
        res = MCPResult.of(await client.call_tool(tool, dict(inputs(state, **kw)), **call_kw))
        if res.is_error:
            return ToolResult(error=f"MCP tool {tool} failed: {res.text[:300]}")
        return ToolResult(update=output(state, res))

    return wf.add_tool(name or tool, afn, args=args, **declared)


__all__ = ["add_mcp_tool", "list_mcp_tools", "MCPResult"]
