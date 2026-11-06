"""Hugging Face Space: the FGAO library, its API, and a live playground.

Runs the real ``fgao`` package: paste passages, ask a question, pick a model
served by Hugging Face Inference Providers, and inspect every decision the
controller makes.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "src"), _HERE):  # repo checkout, or the package copied next to app.py
    if os.path.isdir(os.path.join(_p, "fgao")):
        sys.path.insert(0, _p)
        break

import gradio as gr  # noqa: E402

from fgao import FGAORAG, Document  # noqa: E402

DEFAULT_MODEL = os.environ.get("FGAO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SAMPLE_KB = """Paris is the capital of France. The Seine flows through the city.

Berlin is the capital of Germany. It has around 3.7 million inhabitants.

Japan is an island country in East Asia. Mount Fuji is its highest peak."""

SAMPLE_WEB = """Tokyo is the capital of Japan.

Nairobi is the capital of Kenya."""

OVERVIEW = """
# FGAO — Feedback-Greedy Agentic Orchestration

A **training-free numerical controller** for RAG and tool-using pipelines. No controller LLM, no
learned weights: at every step FGAO scores each admissible operation by predicted progress on
*verifiable* goals, minus its cost, plus a one-step lookahead. Feedback from the outcomes corrects
those predictions online. It stops when every goal is verified, or when nothing left is worth its cost.

For RAG it decides **which source to query, whether to grade passages, when to escalate to web
search, when to run a support check, and when to abstain**. It returns an answer only when that
answer is grounded in a retrieved passage and has passed the support check.

```bash
pip install fgao                    # core: zero dependencies
pip install "fgao[langchain]"       # + LangChain / LangGraph adapters
pip install "fgao[huggingface]"     # + Hugging Face Inference / embeddings
```

```python
from fgao import FGAORAG
from fgao.integrations.huggingface import HFInferenceLLM

rag = FGAORAG.from_retrievers(
    kb=my_retriever,            # any retriever, LangChain retriever/vector store, or a list of texts
    web=my_web_search,          # optional second source; FGAO escalates to it when the KB fails
    llm=HFInferenceLLM("Qwen/Qwen2.5-7B-Instruct", temperature=0),
)
result = rag.invoke("What is the capital of Japan?")
result.answer          # None = abstained (no grounded, supported answer)
result.explain()       # every decision with its numerical justification
```

See the **API** tab for the three layers (RAG, Workflow, Controller) and the integrations. The
**Playground** tab runs the library itself.
"""

API = """
## Layer 1 — `FGAORAG`: a drop-in RAG orchestrator

```python
from fgao import FGAORAG, Source, Costs

rag = FGAORAG.from_retrievers(kb=retriever, web=web_search, llm=llm, budget=16)
# or choose the sources yourself:
rag = FGAORAG([Source("docs", retriever, k=5, cost=1.0, family="kb"),
               Source("web", web_search, k=5, cost=4.0, family="web")], llm=llm)

r = rag.invoke("question")                 # RAGResult
for event in rag.stream("question"): ...   # Step ... Step, RAGResult
rag.batch(questions, max_concurrency=16)   # thread pool, input order kept
await rag.ainvoke("question"); await rag.abatch(questions)
```

`RAGResult`: `answer` (None = abstained), `status`, `values` (answered / grounded / supported),
`grounded_answer`, `evidence`, `support_passage`, `support`, `cost`, `calls`, `steps`, `retrieved`,
`explain()`, `to_dict()`.

Any operation can be replaced: `generator(question, docs) -> str`, `grader(question, docs) -> [0|1|2]`,
`rewriter(question) -> query`, `support_checker(question, passage, answer) -> bool`.

## Layer 2 — `Workflow`: declare goals and tools, FGAO picks the order

```python
from fgao import Workflow, ToolResult

wf = Workflow("report")
wf.add_goal("loaded", check=lambda s: "rows" in s, weight=1)
wf.add_goal("summary", check=lambda s: "total" in s, weight=2)

@wf.tool(gain={"loaded": 0.95}, cost=1)
def load(state): return {"rows": read(state["path"])}

@wf.tool(gain={"summary": 0.9}, requires_goals=["loaded"], cost=1)
def summarise(state): return {"total": sum(state["rows"])}

app = wf.compile(budget=10)
app.invoke({"path": "data.csv"})        # .stream / .batch / .ainvoke / .abatch / .astream
```

## Layer 3 — `FGAOController`: the step API for your own loop

```python
from fgao import FGAOController, Condition, ActionSpec, Action, Semantics, Observation

ctrl = FGAOController([Condition("solved")],
                      [ActionSpec(Action("solve"), Semantics(gain={"solved": 0.8}, cost=2))], budget=10)
ctrl.reset(Observation({"solved": 0.0}))
while ctrl.should_stop() is None:
    decision = ctrl.select_action()          # action + scores + every alternative
    obs, cost = run(decision.action)         # your execution
    ctrl.update(decision.action, obs, cost)  # online feedback
```

## Agent-scale orchestration

```python
from fgao.integrations.mcp import add_mcp_tool          # MCP tools
from fgao.integrations.langchain import add_langchain_tool

wf.add_agent("research", rag, inputs=lambda s: s["q"], output=lambda s, r: {"fact": r.answer},
             gain={"fact": 0.8}, cost=8)                 # sub-agents: FGAO, LangGraph, LangChain

run = app.start(state, interrupt_before=["send_email"])  # human approval
out = run.run()                                          # Interrupt before the gated call
checkpoint = run.snapshot()                              # JSON; app.resume(checkpoint, state) continues exactly
run.approve()                                            # or run.reject(): FGAO replans without it
```

The controller itself uses no LLM. LLMs appear only inside the tools and sub-agents you give it.

## Run it as an HTTP service

```bash
pip install "fgao[server,openai]"
fgao serve --config fgao.yaml --host 0.0.0.0 --port 8000     # or: --factory myapp.rag:build
```

| endpoint | |
|---|---|
| `POST /v1/answer` | `{"question", "budget"?, "trace"?}` returns the result as JSON |
| `POST /v1/answer/stream` | Server-Sent Events: one `step` per decision, then `result` |
| `POST /v1/batch` | `{"questions": [...]}` |
| `GET /v1/info`, `/health`, `/ready`, `/metrics`, `/docs` | layout, probes, Prometheus metrics, OpenAPI |

The service provides retries and timeouts (`fgao.resilience`), admission control (429), per-query deadlines
(504), API keys (`FGAO_API_KEYS`) and a Docker image.

## Integrations

| | |
|---|---|
| LangChain | pass chat models, retrievers, vector stores directly; `rag.as_runnable()` (LCEL), `rag.as_tool()` |
| LangGraph | `graph.add_node("rag", rag.as_node())`; `workflow_node(app)` for any Workflow |
| Hugging Face | `HFInferenceLLM`, `TransformersLLM`, `hf_embedder` + `DenseRetriever`, `load_hf_corpus` |
| OpenAI-compatible / Anthropic | `OpenAILLM(model, base_url=...)`, `AnthropicLLM(model)` |
"""


def _passages(text: str, title: str):
    parts = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    return [Document(text=p, title=f"{title} {i + 1}") for i, p in enumerate(parts)]


def answer(question, kb_text, web_text, model, token, budget, llm=None):
    """Playground handler. ``llm`` lets tests inject a model."""
    question = (question or "").strip()
    if not question:
        return "Enter a question.", [], {}
    kb = _passages(kb_text, "KB")
    if not kb:
        return "Add at least one knowledge-base passage.", [], {}
    web = _passages(web_text, "Web")
    if llm is None:
        token = (token or "").strip() or os.environ.get("HF_TOKEN")
        if not token:
            return ("This Space has no `HF_TOKEN` secret. Paste a Hugging Face token (read access, "
                    "Inference Providers enabled) to run the playground."), [], {}
        from fgao.integrations.huggingface import HFInferenceLLM

        llm = HFInferenceLLM((model or DEFAULT_MODEL).strip(), token=token, temperature=0)
    rag = FGAORAG.from_retrievers(kb=kb, web=web or None, llm=llm, budget=float(budget))
    try:
        r = rag.invoke(question)
    except Exception as e:  # noqa: BLE001 - surface provider errors in the UI
        return f"Model call failed: `{type(e).__name__}: {e}`", [], {}
    rows = [[s["step"], s["action"], round(s["Q"], 3), round(s["predicted_gain"], 3), round(s["continuation"], 3),
             s["cost"], s["budget_left"], " ".join(f"{k}={v:g}" for k, v in s["values"].items()), s["error"] or ""]
            for s in r.trace()]
    if r.answer is not None:
        head = f"### {r.answer}\n\nVerified: grounded in *{r.support_passage.title}* and accepted by the support check."
    else:
        head = "### Abstained\n\nNo answer was both grounded in a passage and accepted by the support check."
        if r.grounded_answer:
            head += f" Best unverified candidate: *{r.grounded_answer}*."
    head += f"\n\nStatus `{r.status}` · cost {r.cost:g} of {float(budget):g} · calls {r.calls}"
    return head, rows, r.to_dict()


HEADERS = ["step", "action", "Q", "predicted gain", "lookahead", "cost", "budget left", "goals after", "error"]


def build() -> gr.Blocks:
    with gr.Blocks(title="FGAO") as demo:
        with gr.Tab("Overview"):
            gr.Markdown(OVERVIEW)
        with gr.Tab("Playground"):
            with gr.Row():
                with gr.Column():
                    q = gr.Textbox(label="Question", value="What is the capital of Japan?")
                    kb = gr.Textbox(label="Knowledge base passages (blank line between passages)", lines=8,
                                    value=SAMPLE_KB)
                    web = gr.Textbox(label="Second source, e.g. web results (optional)", lines=4, value=SAMPLE_WEB)
                    with gr.Row():
                        model = gr.Textbox(label="Model (Hugging Face Inference Providers)", value=DEFAULT_MODEL)
                        budget = gr.Slider(4, 40, value=16, step=1, label="Budget")
                    token = gr.Textbox(label="HF token (optional if the Space has an HF_TOKEN secret)", type="password")
                    go = gr.Button("Run FGAO", variant="primary")
                with gr.Column():
                    out = gr.Markdown()
                    trace = gr.Dataframe(headers=HEADERS, label="Decisions", wrap=True)
                    raw = gr.JSON(label="RAGResult.to_dict()")
            go.click(lambda *a: answer(*a), [q, kb, web, model, token, budget], [out, trace, raw])
        with gr.Tab("API"):
            gr.Markdown(API)
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
