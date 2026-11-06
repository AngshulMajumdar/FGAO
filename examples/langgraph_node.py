"""FGAO RAG as one node of a LangGraph graph; LangGraph routes on its outcome.

    pip install "fgao[langgraph]"
    python langgraph_node.py
"""
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from fgao import FGAORAG
from toy import KB, WEB, ToyLLM


class State(TypedDict, total=False):
    question: str
    answer: Optional[str]
    fgao: dict
    reply: str


rag = FGAORAG.from_retrievers(kb=KB, web=WEB, llm=ToyLLM())


def respond(state: State) -> State:
    return {"reply": f"{state['answer']} (verified, cost {state['fgao']['cost']:g})"}


def escalate(state: State) -> State:
    return {"reply": "No verified answer; routed to a human."}


g = StateGraph(State)
g.add_node("rag", rag.as_node())
g.add_node("respond", respond)
g.add_node("escalate", escalate)
g.add_edge(START, "rag")
g.add_conditional_edges("rag", lambda s: "respond" if s["answer"] else "escalate")
g.add_edge("respond", END)
g.add_edge("escalate", END)
graph = g.compile()

for q in ["What is the capital of Italy?", "What is the capital of Spain?"]:
    print(q, "->", graph.invoke({"question": q})["reply"])
