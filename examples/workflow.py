"""A non-RAG workflow: declare goals and tools, let FGAO choose the order.

    python workflow.py
"""
import csv
import io
import random

from fgao import ToolResult, Workflow

RAW = "id,amount\n1,10\n2,\n3,7\n"

wf = Workflow("report")
wf.add_goal("loaded", lambda s: "rows" in s, weight=1.0)
wf.add_goal("clean", lambda s: s.get("clean", False), weight=1.0)
wf.add_goal("summary", lambda s: "total" in s, weight=2.0)
wf.add_predicate("schema_known", lambda s: "columns" in s)

rng = random.Random(1)


@wf.tool(gain={"loaded": 0.95}, cost=1.0)
def load(state):
    rows = list(csv.DictReader(io.StringIO(state["raw"])))
    return {"rows": rows, "columns": list(rows[0])}


@wf.tool(gain={"clean": 0.9}, requires=["schema_known"], requires_goals=["loaded"], cost=1.0)
def drop_missing(state):
    return {"rows": [r for r in state["rows"] if r["amount"]], "clean": True}


# two ways to summarise: a cheap one that claims to work but is flaky, and a reliable one
@wf.tool(name="summarise", args={"method": "fast"}, gain={"summary": 0.9}, requires_goals=["clean"], cost=0.5,
         group="summarise")
def summarise_fast(state, method):
    if rng.random() < 0.8:  # fails silently most of the time
        return ToolResult(cost=0.5)
    return {"total": sum(int(r["amount"]) for r in state["rows"])}


@wf.tool(name="summarise", args={"method": "exact"}, gain={"summary": 0.7}, requires_goals=["clean"], cost=1.5,
         group="summarise_exact")
def summarise_exact(state, method):
    return {"total": sum(int(r["amount"]) for r in state["rows"])}


app = wf.compile(budget=10)
result = app.invoke({"raw": RAW})
for step in result.steps:
    print(step.step, step.action, f"Q={step.decision.scores.Q:.3f}", step.values_after)
print(result.status, result.state["total"], f"cost={result.cost:g}")
