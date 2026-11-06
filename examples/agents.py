"""Agent-scale orchestration without any LLM in the controller:
sub-agents as tools, human approval before a side effect, and checkpoint / resume.

    python agents.py
"""
import json

from fgao import FGAORAG, Interrupt, ToolResult, Workflow
from toy import KB, WEB, ToyLLM

# a sub-agent: FGAO RAG over the toy corpus (its LLM calls live inside its operations)
rag = FGAORAG.from_retrievers(kb=KB, web=WEB, llm=ToyLLM())

# another sub-agent: a small FGAO workflow
writer = Workflow("writer")
writer.add_goal("text", lambda s: "text" in s)
writer.add_tool("compose", lambda s: ToolResult(update={"text": f"{s['city']} is the capital of {s['country']}."},
                                                cost=1.0), gain={"text": 0.95}, cost=1)
writer_app = writer.compile(budget=5)

outbox = []
wf = Workflow("brief")
wf.add_goal("fact", lambda s: s.get("capital") is not None, weight=2)
wf.add_goal("draft", lambda s: "draft" in s)
wf.add_goal("published", lambda s: s.get("published", False))
wf.add_agent("research", rag, inputs=lambda s: f"What is the capital of {s['country']}?",
             output=lambda s, r: {"capital": r.answer}, gain={"fact": 0.8}, cost=8)
wf.add_agent("write", writer_app, inputs=lambda s: {"city": s["capital"], "country": s["country"]},
             output=lambda s, r: {"draft": r.state["text"]}, gain={"draft": 0.9}, requires_goals=["fact"], cost=2)


@wf.tool(gain={"published": 0.95}, requires_goals=["draft"], cost=1)
def publish(state):
    outbox.append(state["draft"])
    return {"published": True}


app = wf.compile(budget=40)
state = {"country": "Japan"}
run = app.start(state, interrupt_before=["publish"])
out = run.run()
assert isinstance(out, Interrupt)
print("paused before", out.action, "| draft:", state["draft"])

checkpoint = json.dumps(run.snapshot())          # persist; a person reviews the draft
resumed = app.resume(json.loads(checkpoint), state, interrupt_before=["publish"])
resumed.step()                                   # still paused
resumed.approve()
result = resumed.run()
print(result.status, "| published:", outbox, "| cost:", result.cost)
for h in resumed.ctrl.history:
    print("  ", h["step"], h["action"], "cost", h["cost"])
