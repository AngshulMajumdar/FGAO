"""MCP tools orchestrated by FGAO (in-process MCP server; swap in any MCP server URL).

    pip install mcp
    python mcp_tools.py
"""
import asyncio

from mcp import Client
from mcp.server.mcpserver import MCPServer

from fgao import Workflow
from fgao.integrations.mcp import add_mcp_tool, list_mcp_tools

server = MCPServer("inventory")
STOCK = {"widget": 0, "gadget": 12}


@server.tool()
def check_stock(item: str) -> int:
    """Units in stock."""
    return STOCK[item]


@server.tool()
def reorder(item: str, units: int) -> str:
    """Place a supplier order."""
    STOCK[item] += units
    return f"ordered {units} {item}"


async def main():
    async with Client(server) as client:  # or Client("http://host:8000/mcp")
        print("tools:", [t["name"] for t in await list_mcp_tools(client)])
        wf = Workflow("restock")
        wf.add_goal("checked", lambda s: "stock" in s)
        wf.add_goal("stocked", lambda s: s.get("stock", 0) > 0 or s.get("ordered", False), weight=2)
        add_mcp_tool(wf, client, "check_stock", inputs=lambda s: {"item": s["item"]},
                     output=lambda s, r: {"stock": int(r.text)}, gain={"checked": 0.95}, cost=1)
        add_mcp_tool(wf, client, "reorder", inputs=lambda s: {"item": s["item"], "units": 50},
                     output=lambda s, r: {"ordered": r.text.startswith("ordered")}, gain={"stocked": 0.9},
                     requires_goals=["checked"], cost=3)
        result = await wf.compile(budget=10).ainvoke({"item": "widget"})
        print(result.status, [str(s.action) for s in result.steps], result.state)


asyncio.run(main())
