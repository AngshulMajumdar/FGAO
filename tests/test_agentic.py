"""Agent-scale features: checkpoint/resume, human approval, sub-agents, MCP and LangChain tools, many tools."""
import asyncio
import copy
import json
import os
import time

import pytest

import random_tasks as RT
from fgao import (FGAOConfig, FGAOController, FGAORAG, Interrupt, Observation, Status, ToolResult, Workflow,
                  WorkflowResult)
from test_golden import GOLD_CTRL, run_controller, specs_for
from test_rag import KB, WEB, Scripted

from fgao import Condition


# ------------------------------------------------------------ checkpointing
@pytest.mark.parametrize("config", ["default", "eq16", "eq17", "raw", "beta", "type"])
def test_snapshot_restore_at_every_step_reproduces_decisions(config):
    cfg = FGAOConfig(**RT.CONFIGS[config])
    for g in [g for g in GOLD_CTRL if g["config"] == config][:40]:
        p = RT.make_problem(g["seed"])
        conds = [Condition(x["name"], x["weight"], x["mandatory"]) for x in p["goals"]]
        n = len(g["actions"])
        for k in range(n + 1):  # checkpoint after k steps, continue in a fresh controller
            env = RT.Env(p, g["run_seed"])
            c = FGAOController(conds, specs_for(p), cfg, budget=p["budget"], seed=g["run_seed"])
            c.reset(Observation(env.values(), env.predicates()))
            acts = []
            for _ in range(k):
                c.should_stop()
                d = c.select_action()
                cost = env.execute(d.action.tool, d.action.args)
                acts.append(str(d.action))
                c.update(d.action, Observation(env.values(), env.predicates()), cost)
            c.should_stop()  # a cached pending decision must not leak into the snapshot
            snap = json.loads(json.dumps(c.snapshot()))
            env2 = copy.deepcopy(env)
            c2 = FGAOController(conds, specs_for(p), cfg, budget=p["budget"], seed=g["run_seed"]).restore(snap)
            while c2.should_stop() is None:
                d = c2.select_action()
                cost = env2.execute(d.action.tool, d.action.args)
                acts.append(str(d.action))
                c2.update(d.action, Observation(env2.values(), env2.predicates()), cost)
            assert acts == g["actions"] and c2.status == g["status"], (config, g["seed"], k)


def test_snapshot_rejects_other_problem():
    p = RT.make_problem(3)
    conds = [Condition(x["name"], x["weight"], x["mandatory"]) for x in p["goals"]]
    c = FGAOController(conds, specs_for(p), budget=10).reset(Observation({x.name: 0.0 for x in conds}))
    snap = c.snapshot()
    other = RT.make_problem(4)
    with pytest.raises(ValueError):
        FGAOController([Condition(x["name"]) for x in other["goals"]], specs_for(other)).restore(snap)


# --------------------------------------------------------- run / interrupts
def mail_workflow(log):
    wf = Workflow("mail")
    wf.add_goal("drafted", lambda s: "draft" in s)
    wf.add_goal("sent", lambda s: s.get("sent", False))

    @wf.tool(gain={"drafted": 0.9}, cost=1)
    def draft(s):
        log.append("draft")
        return {"draft": "hello"}

    @wf.tool(gain={"sent": 0.95}, requires_goals=["drafted"], cost=1)
    def send_email(s):
        log.append("send_email")
        return {"sent": True}

    @wf.tool(gain={"sent": 0.6}, requires_goals=["drafted"], cost=2)
    def send_slack(s):
        log.append("send_slack")
        return {"sent": True}

    return wf.compile(budget=10)


def test_run_equals_invoke_and_interrupt_approve():
    log = []
    ref = mail_workflow(log).invoke({})
    assert log == ["draft", "send_email"] and ref.success
    log2 = []
    app = mail_workflow(log2)
    run = app.start({}, interrupt_before=["send_email"])
    out = run.run()
    assert isinstance(out, Interrupt) and out.action.tool == "send_email" and log2 == ["draft"]
    assert run.status == "INTERRUPTED"
    snap = json.loads(json.dumps(run.snapshot()))  # persist while waiting for a human
    run2 = app.resume(snap, run.state, interrupt_before=["send_email"])
    assert isinstance(run2.step(), Interrupt)  # still waiting after resume
    run2.approve()
    res = run2.run()
    assert isinstance(res, WorkflowResult) and res.success and log2 == ["draft", "send_email"]
    assert [h["action"] for h in run2.ctrl.history] == [str(s.action) for s in ref.steps]


def test_interrupt_reject_picks_alternative():
    log = []
    run = mail_workflow(log).start({}, interrupt_before=["send_email"])
    assert isinstance(run.run(), Interrupt)
    run.reject()
    res = run.run()
    assert res.success and log == ["draft", "send_slack"]
    with pytest.raises(RuntimeError):
        run.approve()


def test_async_run():
    log = []
    run = mail_workflow(log).start({})
    res = asyncio.run(run.arun())
    assert res.success and log == ["draft", "send_email"]


# ------------------------------------------------------------- sub-agents
def test_subworkflow_and_rag_as_agents_charge_measured_cost():
    child = Workflow("research")
    child.add_goal("notes", lambda s: "notes" in s)
    child.add_tool("read", lambda s: ToolResult(update={"notes": s["topic"] + " notes"}, cost=3.0), gain={"notes": 0.9}, cost=1)
    child_app = child.compile(budget=10)
    rag = FGAORAG.from_retrievers(KB, Scripted(), web=WEB)

    parent = Workflow("report")
    parent.add_goal("researched", lambda s: "notes" in s)
    parent.add_goal("answered", lambda s: s.get("capital") is not None)
    parent.add_agent("research", child_app, inputs=lambda s: {"topic": s["topic"]},
                     output=lambda s, r: {"notes": r.state["notes"]}, gain={"researched": 0.9}, cost=2)
    parent.add_agent("ask", rag, inputs=lambda s: f"What is the capital of {s['country']}?",
                     output=lambda s, r: {"capital": r.answer}, gain={"answered": 0.8}, cost=6,
                     budget=lambda s: 16)
    app = parent.compile(budget=40)
    r = app.invoke({"topic": "France", "country": "France"})
    assert r.success and r.state["capital"] == "Paris" and r.state["notes"] == "France notes"
    costs = {s.tool: s.cost for s in r.steps}
    assert costs["research"] == 3.0 and costs["ask"] == 5.0  # measured, not declared
    ra = asyncio.run(app.ainvoke({"topic": "Italy", "country": "Italy"}))
    assert ra.state["capital"] == "Rome"


def test_langgraph_graph_as_agent():
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        x: int
        y: int

    g = StateGraph(S)
    g.add_node("double", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "double")
    g.add_edge("double", END)
    graph = g.compile()
    wf = Workflow("outer").add_goal("done", lambda s: "y" in s)
    wf.add_agent("lg", graph, inputs=lambda s: {"x": s["x"]}, output=lambda s, r: {"y": r["y"]}, gain={"done": 0.9},
                 cost=1)
    assert wf.compile(budget=5).invoke({"x": 21}).state["y"] == 42
    assert asyncio.run(wf.compile(budget=5).ainvoke({"x": 4})).state["y"] == 8


# -------------------------------------------------------------------- MCP
def test_mcp_tools_in_process():
    mcp = pytest.importorskip("mcp")
    from mcp.server.mcpserver import MCPServer

    from fgao.integrations.mcp import add_mcp_tool, list_mcp_tools

    server = MCPServer("ops")
    calls = []

    @server.tool()
    def lookup(city: str) -> str:
        """Weather lookup."""
        calls.append(("lookup", city))
        if city == "Atlantis":
            raise ValueError("unknown city")
        return f"{city}: 21C"

    @server.tool()
    def notify(text: str) -> str:
        calls.append(("notify", text))
        return "sent"

    async def go():
        async with mcp.Client(server) as client:
            names = {t["name"] for t in await list_mcp_tools(client)}
            wf = Workflow("ops")
            wf.add_goal("weather", lambda s: "weather" in s)
            wf.add_goal("notified", lambda s: s.get("notified", False))
            for city in ("Atlantis", "Paris"):
                add_mcp_tool(wf, client, "lookup", args={"city": city}, inputs=lambda s, city: {"city": city},
                             output=lambda s, r: {"weather": r.text}, gain={"weather": 0.9 if city == "Atlantis" else 0.7},
                             cost=1)
            add_mcp_tool(wf, client, "notify", inputs=lambda s: {"text": s["weather"]},
                         output=lambda s, r: {"notified": r.text == "sent"}, gain={"notified": 0.9},
                         requires_goals=["weather"], cost=1)
            return names, await wf.compile(budget=10).ainvoke({})

    names, r = asyncio.run(go())
    assert names == {"lookup", "notify"}
    assert r.success and r.state["weather"] == "Paris: 21C"
    # Atlantis declared 0.9: after one failure 0.9*4/5 = 0.72 > Paris 0.7, after two 0.6 < 0.7 -> switch
    assert [str(s.action) for s in r.steps] == ["lookup(city=Atlantis)", "lookup(city=Atlantis)", "lookup(city=Paris)",
                                                 "notify()"]
    assert r.steps[0].error and "failed" in r.steps[0].error  # MCP error result -> recorded failure
    assert calls == [("lookup", "Atlantis"), ("lookup", "Atlantis"), ("lookup", "Paris"), ("notify", "Paris: 21C")]


# --------------------------------------------------------------- LangChain
def test_langchain_tool_adapter():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    from fgao.integrations.langchain import add_langchain_tool

    @tool
    def word_count(text: str) -> int:
        """Count words."""
        return len(text.split())

    wf = Workflow("lc").add_goal("counted", lambda s: "n" in s)
    add_langchain_tool(wf, word_count, inputs=lambda s: {"text": s["doc"]}, output=lambda s, r: {"n": r},
                       gain={"counted": 0.9}, cost=1)
    app = wf.compile(budget=5)
    assert app.invoke({"doc": "a b c"}).state["n"] == 3
    assert asyncio.run(app.ainvoke({"doc": "a b"})).state["n"] == 2


# ------------------------------------------------------------ many tools
def test_workflow_with_1000_tools_is_fast():
    import random

    rng = random.Random(0)
    wf = Workflow("big")
    for i in range(20):
        wf.add_goal(f"g{i}", (lambda k: lambda s: k in s["done"])(f"g{i}"))
    ok = {}
    for j in range(1000):
        targets = rng.sample(range(20), 2)
        ok[j] = rng.random() < 0.3

        def fn(s, j, _t=targets):
            if ok[j]:
                s["done"].update(f"g{t}" for t in _t)

        wf.add_tool("tool", fn, args={"j": j}, gain={f"g{t}": round(rng.uniform(.2, .9), 3) for t in targets},
                    cost=round(rng.uniform(.5, 2), 3), group=f"family{j % 25}")
    app = wf.compile(budget=200)
    t0 = time.perf_counter()
    r = app.invoke({"done": set()})
    per = (time.perf_counter() - t0) / max(1, len(r.steps))
    assert r.steps and per < 0.5, per  # generous bound for slow CI machines
