# fgao — Feedback-Greedy Agentic Orchestration

`fgao` is a **training-free, model-free controller** for retrieval-augmented generation and other tool-using pipelines.

It does not ask an LLM what to do next, and there are no weights to learn. The controller is pure arithmetic over declared effects and observed outcomes. An LLM appears only if *your tools* use one: inside RAG operations, or as a sub-agent. Workflows without any LLM are first-class. At every step it does the following:
- **Score.** Each admissible operation is scored by its predicted progress on *verifiable* goals, minus its cost, plus a one-step lookahead.
- **Run.** The best-scoring operation runs.
- **Learn.** The observed outcome corrects the predictions online.
- **Stop.** It stops when every goal is verified, when nothing left is worth its cost, or when the budget runs out.

For RAG, this lets it decide:
- which source to query;
- whether to grade passages first;
- when to escalate from the knowledge base to web search;
- when to run a support check;
- when to **abstain**.

It returns an answer only if the answer is grounded in a retrieved passage and accepted by a support check.

The library can be used three ways:

| Layer | Use it when | Feels like |
|---|---|---|
| **`FGAORAG`** | you want a RAG pipeline that decides its own retrieval strategy | a LangChain chain |
| **`Workflow`** | you have tools and checkable goals and want FGAO to order the tools | a LangGraph graph without hand-written edges |
| **`FGAOController`** | you own the loop and just want the decisions | an optimiser's `ask` / `tell` |
| **`fgao serve`** | you want it as an HTTP service | a model server |
| **`Run`** | long-running agents: pause for human approval, checkpoint, resume | LangGraph checkpointers and interrupts |

It also plugs into existing code:
- **Inputs:** LangChain models, retrievers and vector stores are accepted as they are.
- **LangChain:** `rag.as_runnable()` composes with LCEL, and `rag.as_tool()` hands the pipeline to an agent.
- **LangGraph:** `rag.as_node()` drops into a `StateGraph`.
- **Other models:** Hugging Face, OpenAI-compatible endpoints and Anthropic are covered by small adapters.

The core has **no dependencies**.

```bash
pip install fgao                    # core
pip install "fgao[langchain]"       # LangChain adapters (also: langgraph, huggingface, openai, anthropic, all)
pip install "fgao[server,openai]"   # HTTP service
pip install "fgao[mcp]"             # MCP tool adapter
```

**Documentation:**
- [Algorithm](https://github.com/AngshulMajumdar/FGAO/blob/main/docs/algorithm.md): exactly what the controller computes.
- [API reference](https://github.com/AngshulMajumdar/FGAO/blob/main/docs/api.md): every public class and function.
- [Deployment](https://github.com/AngshulMajumdar/FGAO/blob/main/docs/deployment.md): the HTTP service, config file, Docker and monitoring.
- [Hugging Face Spaces](https://github.com/AngshulMajumdar/FGAO/blob/main/docs/huggingface-space.md): deploying the playground and the HTTP API.
- [Examples](https://github.com/AngshulMajumdar/FGAO/tree/main/examples).

---

## Quickstart

```python
from fgao import FGAORAG
from fgao.integrations.huggingface import HFInferenceLLM

rag = FGAORAG.from_retrievers(
    kb=["Paris is the capital of France.", "Berlin is the capital of Germany."],  # or any retriever
    web=my_web_search,                                                          # optional second source
    llm=HFInferenceLLM("Qwen/Qwen2.5-7B-Instruct", temperature=0),
)

r = rag.invoke("What is the capital of France?")
r.answer            # 'Paris', or None when FGAO abstains
r.support_passage   # the passage that contains and supports the answer
print(r.explain())  # every decision and its numerical justification
```

```
Q: What is the capital of France?
  0. answer(src=kb5, graded=False)  Q=0.507 gain=0.370 cont=0.174 cost=3 left=13  -> answered=1, grounded=1, supported=0
  1. verify()  Q=0.388 gain=0.400 cont=0.000 cost=2 left=11  -> answered=1, grounded=1, supported=1
  stop: SUCCESS; answer: Paris
```

There are runnable scripts in [`examples/`](https://github.com/AngshulMajumdar/FGAO/tree/main/examples). All of them except `quickstart.py` run offline, using a rule-based stand-in LLM.

---

## How FGAO decides

The task is described by verifiable **goals** *gᵢ*, each with weight *wᵢ* and a value *vᵢ* ∈ [0, 1] produced by a mechanical check, never by the tool's own claim.

Each **operation** *a* declares four things:
- the fraction of each goal's deficiency it closes (`gain`);
- the fraction of achieved goals it may undo (`harm`);
- the predicates it produces;
- its preconditions and cost.

At each step FGAO picks

```
a* = argmax_a   Δ̂(a) − λ ĉ(a)/B₀ + γ H(a)
```

where:
- **Δ̂(a) = Σᵢ wᵢ δ̂ᵢ(a)** is the predicted weighted progress. Each δ̂ᵢ blends the declared prior with the outcomes observed for comparable operations, using ρ = ν/(ν+n).
- **ĉ(a)** is the estimated cost. B₀ is the budget.
- **H(a)** is the value of what *a* unlocks one step ahead. This is what makes a zero-progress preparatory step (retrieve, fetch, set up) worth taking.

FGAO stops in three cases:
- **Success:** every mandatory goal is verified.
- **No positive value:** no admissible, affordable operation has positive value.
- **Budget:** the budget is spent.

Each decision costs O(n · m) arithmetic for n admissible operations and m goals. There are no model calls.

Operations of the same **group** share feedback. In RAG, when the knowledge-base variants keep failing, all their estimates drop together and FGAO escalates to the web source.

The defaults (`FGAOConfig()` and the RAG priors, costs and weights) are the FGAO v2 configuration, unchanged. [`docs/algorithm.md`](https://github.com/AngshulMajumdar/FGAO/blob/main/docs/algorithm.md) has the details.

---

## Layer 1 — `FGAORAG`

### What the controller sees

**Goals (all mandatory):**

| goal | weight | how it is checked |
|---|---|---|
| `answered` | 0.2 | a non-abstaining answer exists |
| `grounded` | 0.3 | the normalised answer occurs in a passage it was generated from |
| `supported` | 0.5 | the support checker accepted the answer, reading the passage that contains it |

**Operations:**

| operation | what it does |
|---|---|
| `answer(src, graded=False)` | retrieve from `src` (once per query, cached), then generate |
| `answer(src, graded=True)` | retrieve, grade relevance, then generate from the passages kept |
| `verify()` | support check of the current answer |

Each `answer` variant runs at most once per query, and `verify` runs once per distinct answer.

### Sources

`from_retrievers(kb, web=None, llm=...)` builds this layout:

| source | what it retrieves | cost |
|---|---|---|
| `kb5` | top 5 from the KB | 1 |
| `kb10` | top 10 from the KB | 1.5 |
| `rw` | top 5 for an LLM-rewritten query | 2 + 1 |
| `web` | top 5 from the second source | 4 |

A generation or grading call costs 2, and the default budget is 16. To use your own layout, pass the sources explicitly:

```python
from fgao import FGAORAG, Source

rag = FGAORAG(
    [Source("docs", vectorstore, k=5, cost=1.0, family="kb"),
     Source("docs-deep", vectorstore, k=20, cost=2.0, family="kb"),
     Source("web", tavily_retriever, k=5, cost=4.0, family="web")],
    llm=llm, budget=16,
)
```

### Running

```python
r = rag.invoke("question")                      # -> RAGResult
for event in rag.stream("question"):            # Step, Step, ..., RAGResult
    ...
results = rag.batch(questions, max_concurrency=16)
r = await rag.ainvoke("question")
results = await rag.abatch(questions, max_concurrency=32)
async for event in rag.astream("question"): ...
rag.invoke("question", budget=8)                # per-call budget
rag.invoke("question", callbacks=[log_step])    # called after every step
```

### `RAGResult`

| field | meaning |
|---|---|
| `answer` | the verified answer, or `None` (abstained) |
| `status` | `SUCCESS`, `STALLED` (nothing worth its cost), `BUDGET_EXHAUSTED`, `BLOCKED` |
| `values` | final `answered` / `grounded` / `supported` |
| `grounded_answer` | the grounded answer even if unverified, for callers that prefer coverage |
| `evidence`, `support_passage` | the passages the answer came from, and the one that contains it |
| `support` | the support-check verdict on the final answer (`None` if never checked) |
| `cost`, `calls` | spent budget, and per-operation call counts |
| `steps`, `trace()` | every decision with Q, predicted gain, lookahead value, cost, budget left and goal values |
| `retrieved` | everything retrieved, per source |
| `explain()`, `to_dict()` | human-readable trace, and a JSON-serialisable dict |

### Replacing operations

Each operation is a plain callable. The defaults are LLM-backed and use the same prompts as the evaluated v2 pipeline.

```python
rag = FGAORAG.from_retrievers(
    kb=retriever, llm=llm,
    generator=lambda q, docs: my_reader(q, docs),              # -> answer span, '' to abstain
    grader=lambda q, docs: [2 if relevant(d) else 0 for d in docs],
    rewriter=lambda q: my_query_expansion(q),                  # -> search query
    support_checker=lambda q, passage, ans: nli_entails(passage.text, ans),
)
```

Declared costs, priors and goal weights can also be overridden, through `Costs(...)`, `Priors(...)` and `Weights(...)`. The defaults are the evaluated ones, so if you change them, validate on your own data.

---

## Layer 2 — `Workflow`

Declare *what must become true* and *what each tool claims to do*. FGAO decides the order at run time.

LangGraph needs edges and routing functions. Here, preconditions and declared effects replace them, and the stopping rule replaces `END`.

```python
from fgao import Workflow, ToolResult

wf = Workflow("report")
wf.add_goal("loaded",  check=lambda s: "rows" in s, weight=1)
wf.add_goal("clean",   check=lambda s: s.get("clean", False), weight=1)
wf.add_goal("summary", check=lambda s: "total" in s, weight=2)
wf.add_predicate("schema_known", lambda s: "columns" in s)

@wf.tool(gain={"loaded": 0.95}, cost=1)
def load(state):
    rows = read_csv(state["path"])
    return {"rows": rows, "columns": list(rows[0])}           # dict -> merged into the state

@wf.tool(gain={"clean": 0.9}, requires=["schema_known"], requires_goals=["loaded"], cost=1)
def drop_missing(state):
    return {"rows": [r for r in state["rows"] if r["amount"]], "clean": True}

@wf.tool(name="summarise", args={"method": "sql"}, gain={"summary": 0.9}, requires_goals=["clean"], cost=2,
         group="summarise")
def summarise(state, method):
    total, seconds = run_query(state)
    return ToolResult(update={"total": total}, cost=seconds)   # measured cost

app = wf.compile(budget=10)                  # FGAOConfig optional; on_error="record" to learn from exceptions
result = app.invoke({"path": "data.csv"})    # .stream / .batch / .ainvoke / .abatch / .astream
result.status, result.success, result.state, result.trace()
```

### Declarations

| argument | meaning |
|---|---|
| `check(state) -> float` | goal value in [0, 1]. It must use evidence, not the tool's report. |
| `gain={goal: p}` | fraction of the remaining deficiency one call closes (for a binary goal: its success probability) |
| `harm={goal: p}` | probability that one call undoes an achieved goal |
| `produces={predicate: p}` | probability that one call makes a predicate true |
| `requires=[...]`, `requires_goals=[...]` | preconditions, on predicates and on verified goals |
| `cost` | declared cost. A `ToolResult(cost=...)` reports the measured cost instead. |
| `args` | fixed arguments. Register the same function several times to offer several variants. |
| `group` | variants in one group share feedback |

Tools receive the state and their `args`, and may mutate the state. They return one of:
- `None`, when they changed the state in place;
- a `dict`, which is merged into the state;
- a `ToolResult`.

The state can be any object. Every run builds its own controller, so runs are independent and safe to execute concurrently.

---

## Layer 3 — `FGAOController`

The controller does no I/O. Use it inside an existing agent loop:

```python
from fgao import FGAOController, Condition, ActionSpec, Action, Semantics, Observation

actions = [
    ActionSpec(Action("fetch_context"), Semantics(produces={"ctx": 0.9}, cost=1)),
    ActionSpec(Action("solve"), Semantics(gain={"solved": 0.8}, requires_predicates=frozenset({"ctx"}), cost=2)),
]
ctrl = FGAOController([Condition("solved", weight=1)], actions, budget=10)
ctrl.reset(Observation({"solved": 0.0}, {"ctx": False}))

while ctrl.should_stop() is None:           # None while running, else the stop reason
    decision = ctrl.select_action()         # .action, .scores (Q, gain, cost, continuation), .ranked()
    obs, cost = execute(decision.action)    # your code
    ctrl.update(decision.action, obs, cost) # feedback, budget, history
```

The step API makes exactly the same decisions as running the whole episode at once. The test suite checks this against the original implementation.

---

## Integrations

### LangChain

LangChain objects are accepted directly. A chat model or LLM is called with `.invoke`, a retriever is called with `.invoke`, and a vector store is called with `.similarity_search`.

```python
from langchain_openai import ChatOpenAI
rag = FGAORAG.from_retrievers(kb=vectorstore, web=tavily_retriever, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))

chain = rag.as_runnable() | (lambda r: r["answer"] or "No verified answer.")   # LCEL Runnable (sync + async + batch)
agent_tool = rag.as_tool()                                                      # StructuredTool(question: str)
```

To wrap a LangChain object explicitly, use `fgao.integrations.langchain.LangChainLLM(model)` or `LangChainRetriever(retriever_or_vectorstore)`. A plain retriever returns its own number of documents. Configure it to return at least the largest `k` you use (10 for the default layout).

### LangGraph

```python
from langgraph.graph import StateGraph, START, END

g = StateGraph(State)
g.add_node("rag", rag.as_node())             # reads state["question"], writes "answer" and "fgao"
g.add_conditional_edges("rag", lambda s: "respond" if s["answer"] else "escalate")
```

`fgao.integrations.langgraph.workflow_node(app)` embeds any compiled `Workflow` as a node. FGAO orchestrates inside the node, and LangGraph routes around it.

### Hugging Face

```python
from fgao.integrations.huggingface import HFInferenceLLM, TransformersLLM, hf_embedder, st_embedder, load_hf_corpus
from fgao import DenseRetriever

llm = HFInferenceLLM("Qwen/Qwen2.5-7B-Instruct", temperature=0)          # Inference Providers, token from HF_TOKEN
local = TransformersLLM("Qwen/Qwen2.5-1.5B-Instruct")                    # local, greedy decoding
docs = load_hf_corpus("my-org/my-corpus", text_field="text", title_field="title")
kb = DenseRetriever(docs, hf_embedder("BAAI/bge-small-en-v1.5"))        # or st_embedder(...) locally
```

### Other providers

```python
from fgao.integrations.providers import OpenAILLM, AnthropicLLM
OpenAILLM("gpt-4o-mini")                                       # also any OpenAI-compatible server:
OpenAILLM("meta-llama/Llama-3.1-8B-Instruct", base_url="http://localhost:8000/v1", api_key="x")
AnthropicLLM("claude-sonnet-4-5")
```

Anything with `.complete(prompt) -> str`, or a plain function `prompt -> str`, also works as the LLM. A retriever is anything with `.retrieve(query, k)`, a function `(query, k) -> docs`, or a list of texts or dicts (indexed with the built-in BM25).

---

## Agent-scale orchestration

### Tools from anywhere

FGAO decides *when* to call a tool from what the tool declares (gain, harm, preconditions, cost). Each adapter says *how* to call it:

```python
from fgao.integrations.mcp import add_mcp_tool, list_mcp_tools          # MCP servers (SDK 1.x or 2.x)
from fgao.integrations.langchain import add_langchain_tool              # LangChain BaseTool / @tool

async with mcp.Client("http://tools:8000/mcp") as client:
    add_mcp_tool(wf, client, "create_ticket", inputs=lambda s: {"title": s["title"]},
                 output=lambda s, r: {"ticket": r.data}, gain={"ticketed": 0.9}, cost=1)
    result = await wf.compile(budget=10).ainvoke(state)

add_langchain_tool(wf, tavily_search, inputs=lambda s: {"query": s["q"]}, output=lambda s, r: {"hits": r},
                   gain={"found": 0.7}, cost=2)
```

An MCP error result is recorded as a failed call: there is no update, and the cost is charged. FGAO's feedback lowers its estimate for that tool and moves on.

### Sub-agents

Anything with `invoke` (and optionally `ainvoke`) can be one tool of a larger workflow: another FGAO `Workflow`, an `FGAORAG`, a LangGraph graph or a LangChain agent. Its measured cost (`result.cost`) is charged to the parent budget, so budgets compose across levels.

```python
wf.add_agent("research", rag, inputs=lambda s: s["question"], output=lambda s, r: {"fact": r.answer},
             gain={"fact": 0.8}, cost=8, budget=lambda s: 16)
wf.add_agent("plan", langgraph_app, inputs=lambda s: {"goal": s["goal"]}, output=lambda s, r: {"plan": r["plan"]},
             gain={"planned": 0.7}, cost=5)
```

### Human approval, checkpoints, resume

```python
run = app.start(state, interrupt_before=["send_email", "pay_invoice"])
out = run.run()                                   # WorkflowResult, or Interrupt before a gated call
if isinstance(out, Interrupt):
    db.save(run.snapshot(), state)                # JSON; wait for a person
    run = app.resume(db.load_snapshot(), db.load_state(), interrupt_before=[...])
    run.approve()                                 # or run.reject(): FGAO excludes that call and replans
    out = run.run()
```

`snapshot()` captures the whole orchestration state: goal values, predicates, budget, history, online statistics, tie-breaking RNG and rejected calls. After `resume`, the run continues with exactly the decisions the uninterrupted run would make; the tests check this at every step of thousands of runs. Your application state is yours to persist alongside it. The controller exposes the same pair directly: `FGAOController.snapshot()` / `restore()`.

### Parallel tool calls

```python
app = wf.compile(budget=20, max_parallel=4)      # up to 4 concurrent calls per round (threads or asyncio)
wf.add_tool("write_db", fn, gain={...}, resources=("db",))   # tools sharing a resource never run together
```

- **First slot:** the sequential v2 choice.
- **Extra slots:** filled with calls that are admissible and affordable now and still worth their cost in the *expected* state after the calls already in the round.
- **Independence:** by default (`independent=True`), an extra call may not gain, harm or produce anything another call in the round touches. Calls on the same goal therefore never race (no hedged duplicates, no conflicting writers), and every observed change has one declared owner, so feedback stays attributable.
- **Observation:** the round is observed once, after all its calls finish.

**Trade-off.** Concurrent calls cannot learn from each other's outcomes. Where tools overclaim their reliability or have undisclosed side effects, parallel rounds spend budget that sequential v2 would have redirected after the first failure. Use `max_parallel` where latency matters more than the last points of success, and measure on your own tasks.

`max_parallel=1` (the default) is sequential v2. A `Run` (interrupts, checkpoints) executes one call per step. On the controller, the API is `select_batch(k, conflicts=None, independent=True)` and `update_batch(actions, observation, costs)`.

### Many tools

Rates are cached between feedback updates, and each tool's gain is summed only over the goals it can affect, so decision cost scales to large tool sets. The regression tests run 100–1,500-tool problems against the reference implementation and require identical decisions. Like every v2 step, a decision executes one tool.

## Run it as a service

```bash
pip install "fgao[server,openai]"
fgao check --config deploy/fgao.yaml                 # build the pipeline and print its layout
fgao ask   --config deploy/fgao.yaml "What is ...?"  # one query from the shell
fgao serve --config deploy/fgao.yaml --host 0.0.0.0 --port 8000
```

`deploy/fgao.yaml` sets four things:
- **Model:** any OpenAI-compatible server (vLLM, TGI, Ollama, ...), Anthropic, Hugging Face, local transformers, or your own factory.
- **Retrievers:** BM25 over a JSONL file, dense retrieval, an HTTP search service, or a `module:function` factory.
- **Failure handling:** retries, timeouts, a rate limit and a concurrency cap.
- **Limits:** the budget and the server limits.

To build the pipeline in Python instead, use `fgao serve --factory myapp.rag:build`, where `build()` returns an `FGAORAG`.

| endpoint | |
|---|---|
| `POST /v1/answer` | `{"question", "budget"?, "seed"?, "trace"?}` returns a `RAGResult` as JSON |
| `POST /v1/answer/stream` | Server-Sent Events: one `step` event per decision, then `result` |
| `POST /v1/batch` | `{"questions": [...]}` returns `{"results": [...]}` |
| `GET /v1/info` | sources, costs, priors, controller configuration, limits |
| `GET /health`, `GET /ready` | probes |
| `GET /metrics` | Prometheus metrics: outcomes, stop status, latency and cost histograms, operation calls, in-flight and queued queries, and per-dependency retries, timeouts and failures |
| `GET /docs` | interactive OpenAPI documentation |

How the service behaves:
- **Admission control.** `max_inflight` queries run at once and `max_queue` more wait. Beyond that the server returns 429 with `Retry-After`.
- **Deadline.** Each query has a deadline, `request_timeout`. On expiry the server returns 504 and cancels the model call in flight.
- **Failures.** When a dependency still fails after retries, the server returns 502. With `on_error: record`, FGAO instead skips the failed operation and keeps answering.
- **Authentication.** Set `FGAO_API_KEYS=key1,key2` to require `Authorization: Bearer <key>` or `X-API-Key: <key>`. Probes stay open.
- **Logging.** Every response carries `X-Request-ID`, and each query writes one JSON log line.

### Docker

```bash
docker build -t fgao .                                  # --build-arg BASE=<mirror>/python:3.12-slim behind a registry mirror
docker run -p 8000:8000 -e OPENAI_API_KEY -e FGAO_API_KEYS=change-me \
  -v $PWD/deploy/fgao.yaml:/etc/fgao/fgao.yaml:ro -v $PWD/data:/data:ro fgao
```

The image runs as a non-root user and has a health check. Run one worker per container and scale with replicas, because admission control and metrics are per process. `deploy/docker-compose.yml` is a starting point.

### Scaling

- **The controller is not the bottleneck.** A decision is a few hundred arithmetic operations over the admissible operations and makes no model calls. Throughput is set by the model and retriever latency (handled by concurrency) and by per-request CPU in the HTTP and client layers (handled by more processes). Measure your own deployment with `loadtest/`.
- **No shared state.** Queries are independent, so replicas scale horizontally behind any load balancer.
- **Async throughout.** The async path (`ainvoke`, `abatch`, `astream`, and the server) awaits native async clients: OpenAI, Anthropic, Hugging Face, LangChain and the HTTP retriever. Sync-only components run in worker threads. The decisions are identical to the sync path, and the test suite checks this.
- **Production retrievers.** The built-in BM25 is in-memory pure Python. At millions of passages, use a vector store or search service: an HTTP retriever, a LangChain vector store, or `DenseRetriever`.
- **Caching.** `CachedLLM` is a thread-safe prompt cache, and `resilient()` adds retries to any LLM or retriever:

```python
from fgao.resilience import resilient
llm = resilient(OpenAILLM("gpt-4o-mini"), attempts=4, timeout=20, max_concurrency=128, rate=50)
```

### Load testing your deployment

`loadtest/` holds three tools for testing your own setup:
- **`make_corpus.py`:** a synthetic corpus and question set.
- **`fake_openai.py`:** an OpenAI-compatible endpoint that adds latency and injects HTTP 500s and hung calls.
- **`loadgen.py`:** a closed-loop client that reports throughput, latency percentiles, HTTP codes, answer correctness and retry counts.

### Notes

- **Determinism.** Decisions are a deterministic function of the operation outcomes. Exact ties are broken by a seeded RNG (`seed=`). Run the LLM at temperature 0 if you need reproducible runs.
- **Budget units.** Costs and budget share one unit. The default units are an LLM call = 2 and a KB retrieval = 1. Rescale `Costs` to latency or money if you prefer.
- **Abstention.** `answer is None` means no answer passed both grounding and support checking within budget. Use `grounded_answer` to trade verification for coverage.

## Configuration

`FGAOConfig` fields and their defaults:

| field | default | meaning |
|---|---|---|
| `nu` | 4.0 | prior strength; ρ = ν/(ν+n) |
| `lam` | 0.1 | cost weight, in goal units per whole budget |
| `gamma` | 0.9 | lookahead weight |
| `lookahead` | `"unlock"` | `"unlock"`, `"eq16"` (deterministic), `"eq17"` (outcome expectation) or `"none"` |
| `lookahead_depth` | 1 | depth of the unlock lookahead |
| `corrective` | True | credit repair operations after predicted harm |
| `similarity` | `"task"` | feedback pooling: `"task"` (groups), `"exact"` or `"type"` |
| `estimator` | `"rate"` | `"rate"` (state-normalised) or `"raw"` (raw deltas) |
| `feedback` | True | online correction on or off |
| `objective` | `"additive"` | `"additive"` or `"ratio"` (ranking only) |
| `beta` | 0.0 | exploration bonus |
| `optional_mass` | None | weight share of optional goals |
| `cost_scale` | budget | B₀ |
| `max_steps` | 10000 | step limit |

## Project layout

```
src/fgao/
  controller.py        FGAOController, FGAOConfig: the v2 algorithm with the step API
  estimator.py         online effect estimator (prior/feedback blend, state-normalised rates)
  types.py             Action, Condition, Semantics, ActionSpec, Observation, Decision, Scores, Status
  workflow.py          Workflow, CompiledWorkflow, ToolResult, Step, WorkflowResult
  rag/                 FGAORAG, sources, documents, prompts, LLM-backed operations, retrievers, LLM wrappers
  integrations/        langchain.py, langgraph.py, huggingface.py, providers.py (OpenAI-compatible, Anthropic)
  resilience.py        retries, timeouts, rate limiting, concurrency caps
  integrations/mcp.py  MCP tools as workflow tools
  server/              FastAPI service, YAML/JSON config, Prometheus metrics, `fgao` CLI
examples/              offline demos, workflow, step API, LangChain, LangGraph, async/batch, quickstart
space/                 Hugging Face Space (Gradio): overview, API reference, playground
space-api/             Hugging Face Space (Docker): the HTTP API on port 7860
deploy/                example service configuration, docker-compose
loadtest/              corpus generator, fault-injecting fake LLM endpoint, load generator
Dockerfile             service image
docs/                  algorithm.md, api.md, deployment.md, huggingface-space.md
tests/                 unit, integration and regression tests
```

## Tests

```bash
pip install -e ".[dev,server,langchain,langgraph]"
pytest
```

The regression tests replay recorded decisions of the original FGAO v2 implementation:
- **Controller:** 3,120 controller episodes over 13 configurations, plus 19 large problems with 100–1,500 tools.
- **RAG:** 400 RAG episodes.

For every episode they require the same action sequence, the same Q values (to 1e-10), the same cost and the same stop reason. The episodes are also run through other entry points:
- the `Workflow` API (default-configuration controller episodes);
- the native async path (all RAG episodes);
- an async-only LLM (100 RAG episodes);
- snapshot and restore after every step (6 configurations);
- the first slot of every parallel round (it must be the v2 choice).
