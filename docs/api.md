# API reference

Everything is importable from `fgao` unless another module is named. Signatures match the code; arguments after `*` are keyword-only.

1. [Controller](#1-controller)
2. [Workflow](#2-workflow)
3. [RAG](#3-rag)
4. [Integrations](#4-integrations)
5. [Resilience](#5-resilience)
6. [HTTP server](#6-http-server)

---

## 1. Controller

### `FGAOConfig`

These are the hyperparameters. Defaults are the evaluated FGAO v2 configuration.

| field | default | meaning |
|---|---|---|
| `nu` | `4.0` | prior strength: ρ = ν / (ν + n) |
| `lam` | `0.1` | cost weight, in goal units per whole budget |
| `gamma` | `0.9` | lookahead weight |
| `lookahead` | `"unlock"` | `"unlock"`, `"eq16"` (deterministic), `"eq17"` (outcome expectation), `"none"` |
| `lookahead_depth` | `1` | depth of the unlock lookahead |
| `corrective` | `True` | credit repair actions after predicted harm |
| `similarity` | `"task"` | feedback pooling: `"task"` (declared groups), `"exact"`, `"type"` |
| `estimator` | `"rate"` | `"rate"` (state-normalised) or `"raw"` (raw deltas) |
| `feedback` | `True` | online correction on or off |
| `objective` | `"additive"` | `"additive"` or `"ratio"` (ranking only) |
| `beta` | `0.0` | exploration bonus weight |
| `optional_mass` | `None` | weight share of optional goals (`None`: declared weights) |
| `cost_scale` | `None` | B₀; `None` means the initial budget |
| `max_steps` | `10000` | step limit |

### `FGAOController(conditions, actions, config=None, *, budget=inf, seed=0, comparability_key=None, prior=None)`

This is the step-wise controller for one episode. It does no I/O.

- `conditions`: a list of `Condition`.
- `actions`: a list of `ActionSpec`, or `{Action: Semantics}`, in a fixed order. The order only matters for exact ties, which a seeded RNG breaks.
- `comparability_key`: `Action -> Hashable`. It overrides `ActionSpec.group`.
- `prior`: `Action -> Semantics`, used for the effect prior only.

| member | what it does |
|---|---|
| `reset(observation)` | starts a fresh episode: statistics are empty and the RNG is re-seeded |
| `should_stop()` | `None` while running, otherwise the `Status`. It computes and caches the next decision. |
| `select_action()` | the next `Decision`. Raises `ControllerStopped` after a stop. |
| `update(action, observation, cost=None, error=None)` | records an outcome (the cost defaults to the declared cost). Returns the history record. |
| `select_batch(max_parallel, conflicts=None, independent=True)` | up to `max_parallel` decisions for one concurrent round; the first is `select_action()` (see the [algorithm](algorithm.md#parallel-rounds)) |
| `update_batch(actions, observation, costs=None, errors=None)` | records a round observed once; a single-action round is `update` |
| `snapshot()` / `restore(snapshot)` | JSON checkpoint of the episode. After `restore`, the decisions are exactly those the original would have made. |
| `block(action)` | excludes an action for the rest of the episode |
| `status`, `success`, `budget`, `spent`, `history`, `state`, `deficiency` | run state |
| `admissible_actions()`, `predicted_gain(a, state)`, `estimated_cost(a)`, `score(a, state)` | inspection |

### Types

| type | fields |
|---|---|
| `Condition(name, weight=1.0, mandatory=True, description="")` | a verifiable goal |
| `Action(tool, args=())`, `Action.of(tool, {"k": v})` | a concrete call; `.arg(k)`, `.kwargs`, `.signature` |
| `Semantics(gain={}, harm={}, produces={}, requires_predicates=frozenset(), requires_conditions=frozenset(), cost=1.0)` | the declared effects |
| `ActionSpec(action, semantics, group=None)` | an action plus its feedback group |
| `Observation(values, predicates={})` | goal values in [0, 1] and boolean predicates |
| `Decision` | `step`, `action`, `scores`, `candidates`, `n_admissible`, `select_time`, `ranked()` |
| `Scores` | `Q`, `rank`, `exploit`, `gain`, `cost`, `bonus`, `continuation` |
| `Status` | `RUNNING`, `SUCCESS`, `STALLED`, `BLOCKED`, `BUDGET_EXHAUSTED`, `STEP_LIMIT` |

---

## 2. Workflow

### `Workflow(name="workflow")`

| method | what it does |
|---|---|
| `add_goal(name, check, *, weight=1.0, mandatory=True, description="")` | `check(state) -> float` in [0, 1] (or `bool`) |
| `add_predicate(name, fn)` | `fn(state) -> bool`, for preconditions and `produces` |
| `add_tool(name, fn, *, args=None, cost=1.0, gain=None, harm=None, produces=None, requires=(), requires_goals=(), group=None, description="", afn=None, resources=())` | declares one call `fn(state, **args)` and returns its `Action`. `fn` may be `async def`; `afn` is an optional async twin. Tools sharing a `resource` never run in the same parallel round. |
| `tool(name=None, **declared)` | decorator form of `add_tool` |
| `add_agent(name, agent, *, inputs, output, measured_cost=True, budget=None, **declared)` | a sub-agent as one tool: anything with `invoke` / `ainvoke` (a compiled `Workflow`, `FGAORAG`, a LangGraph graph, a LangChain agent). `inputs(state)` builds its input; `output(state, result)` returns the update. `result.cost` is charged when present. |
| `validate()` | checks references and rates |
| `compile(config=None, *, budget=inf, on_error="raise", max_parallel=1, independent=True)` | returns a `CompiledWorkflow`. With `on_error="record"`, a tool exception is charged and recorded instead of raised. With `max_parallel > 1`, calls run in concurrent rounds. |

Tool return values:

| return value | effect |
|---|---|
| `None` | the tool changed the state in place |
| a `dict` | merged into a dict state |
| `ToolResult(update=None, cost=None, error=None)` | an update with a measured cost and/or an error |

### `CompiledWorkflow`

Each run builds its own controller, so runs are independent.

| method | what it does |
|---|---|
| `invoke(state, *, seed=0, budget=None, callbacks=())` | runs to the end and returns a `WorkflowResult` |
| `stream(state, *, seed=0, budget=None, callbacks=())` | yields each `Step`, then the `WorkflowResult` |
| `batch(states, *, max_concurrency=8, seed=0, budget=None)` | many states in a thread pool, in input order |
| `ainvoke`, `astream`, `abatch(states, *, max_concurrency=64, ...)` | native async |
| `start(state, *, seed=0, budget=None, interrupt_before=(), callbacks=())` | a resumable `Run` |
| `resume(snapshot, state, *, interrupt_before=(), callbacks=())` | continues a `Run` from its snapshot |
| `controller(*, budget=None, seed=0)`, `observe(state)`, `actions`, `conditions` | inspection |

### `Run`

A `Run` executes one call per step (sequential v2).

| member | what it does |
|---|---|
| `step()` / `astep()` | one call; returns a `Step`, an `Interrupt`, or the final `WorkflowResult` |
| `run()` / `arun()` | steps until the run ends or pauses |
| `approve()` / `reject()` | runs the interrupted call next, or excludes it and lets FGAO replan |
| `snapshot()` | JSON checkpoint of the orchestration; your application state is persisted separately |
| `done`, `status`, `result`, `steps`, `interrupted`, `ctrl` | run state |

### Results

| type | fields |
|---|---|
| `Step` | `step`, `action`, `tool`, `decision`, `values_before`, `values_after`, `cost`, `budget_left`, `error`, `to_dict()` |
| `WorkflowResult` | `state`, `status`, `success`, `values`, `cost`, `steps`, `trace()` |
| `Interrupt` | `step`, `action`, `decision` |

---

## 3. RAG

### `FGAORAG(sources, llm=None, *, generator=None, grader=None, rewriter=None, support_checker=None, costs=Costs(), priors=Priors(), weights=Weights(), budget=16.0, config=None, on_error="raise")`

| member | what it does |
|---|---|
| `FGAORAG.from_retrievers(kb, llm=None, *, web=None, costs=Costs(), **kw)` | the evaluated layout: `kb5`, `kb10`, `rw` (rewritten query), and `web` if given |
| `invoke(question, *, seed=0, budget=None, callbacks=())` | returns a `RAGResult`; `question` is a string or a dict with `question`, `query` or `input` |
| `stream`, `batch(questions, *, max_concurrency=8)`, `ainvoke`, `astream`, `abatch(questions, *, max_concurrency=64)` | as for `CompiledWorkflow` |
| `as_runnable()`, `as_tool(name="fgao_rag", description=None)`, `as_node(*, question_key="question", answer_key="answer", result_key="fgao")` | LangChain Runnable, LangChain tool, LangGraph node |
| `app` | the underlying `CompiledWorkflow` |

Goals (all mandatory): `answered` (weight 0.2), `grounded` (0.3) and `supported` (0.5). Operations: `answer(src, graded)` for each source, and `verify()`.

### Configuration

| type | fields and defaults |
|---|---|
| `Source(name, retriever, k=5, cost=1.0, family="kb", rewrite=False)` | one retrieval source; a `family` shares feedback |
| `Costs(llm=2.0, kb=1.0, kb_deep=1.5, web=4.0)` | declared costs |
| `Priors(answer_answered=0.8, answer_grounded=0.7, graded_answered=0.75, graded_grounded=0.75, answer_harms_support=0.9, answer_produces_unverified=0.8, verify_supported=0.8)` | declared effect priors |
| `Weights(answered=0.2, grounded=0.3, supported=0.5)` | goal weights |
| `default_sources(kb, web=None, costs=Costs())` | the evaluated source list |

### `RAGResult`

| field | meaning |
|---|---|
| `answer` | the verified answer, or `None` (abstained); `abstained` and `verified` are booleans |
| `status` | `SUCCESS`, `STALLED`, `BUDGET_EXHAUSTED`, `BLOCKED` |
| `values` | final `answered` / `grounded` / `supported` |
| `grounded_answer` | the grounded answer even when unverified |
| `evidence`, `support_passage` | the passages the answer came from, and the one that contains it |
| `support` | the support-check verdict on the final answer; `None` if never checked |
| `cost`, `calls` | budget spent, and calls per operation |
| `steps`, `trace()`, `explain()` | every decision with its scores |
| `retrieved` | documents per source |
| `to_dict()` | JSON-serialisable form |

### Components (`fgao.rag`)

| component | what it is |
|---|---|
| `LLMGenerator(llm)`, `LLMGrader(llm)`, `LLMRewriter(llm)`, `LLMSupportChecker(llm)` | the default operations; each takes `prompt=` and `parse=` overrides |
| `fgao.rag.prompts` | `generate_prompt`, `grade_prompt`, `rewrite_prompt`, `verify_prompt`, the parsers, `SYSTEM_PROMPT` |
| `Document(text, title="", id="", metadata={})` | a passage; `as_document` / `as_documents` convert other forms |
| `contains_answer(passage, answer)`, `normalize_answer(s)` | the grounding check |
| `BM25Retriever(documents, k1=1.5, b=0.75)` | in-memory BM25, no dependencies |
| `DenseRetriever(documents, embed, batch_size=64)` | cosine similarity over `embed(texts)` |
| `FunctionRetriever(fn)` | `fn(query, k)`, sync or async |
| `FunctionLLM(fn)`, `CachedLLM(llm, path=None)` | function LLM; thread-safe prompt cache |
| `as_llm(x)`, `as_retriever(x)` | accept objects with `complete` / `retrieve`, LangChain objects, functions and lists of texts |

Pluggable interfaces:

| role | interface |
|---|---|
| LLM | `complete(prompt) -> str`, optional `async acomplete(prompt)` |
| retriever | `retrieve(query, k) -> list[Document]`, optional `async aretrieve(query, k)` |
| generator | `(question, docs) -> str`; `""` means abstain |
| grader | `(question, docs) -> list[int]`, with 0, 1 or 2 per passage |
| rewriter | `(question) -> str` |
| support checker | `(question, passage, answer) -> bool` |

---

## 4. Integrations

| module | members |
|---|---|
| `fgao.integrations.langchain` | `LangChainLLM(model)`, `LangChainRetriever(obj, search_kwargs=None)`, `add_langchain_tool(wf, tool, *, inputs, output, name=None, **declared)`, `rag_runnable(rag)`, `rag_tool(rag, ...)`, `workflow_runnable(app, *, seed=0)` |
| `fgao.integrations.langgraph` | `rag_node(rag, *, question_key="question", answer_key="answer", result_key="fgao")`, `workflow_node(app, *, input_key=None, output_key="fgao", seed=0)` |
| `fgao.integrations.mcp` | `add_mcp_tool(wf, client, tool, *, inputs, output, name=None, args=None, timeout=None, **declared)`, `list_mcp_tools(client)`, `MCPResult` (`text`, `data`, `is_error`, `raw`) |
| `fgao.integrations.huggingface` | `HFInferenceLLM(model, *, token=None, provider=None, base_url=None, max_tokens=128, temperature=None, system=..., timeout=None)`, `TransformersLLM(model, *, max_new_tokens=64)`, `hf_embedder(model, *, token=None)`, `st_embedder(model)`, `load_hf_corpus(path, *, split="train", text_field="text", title_field=None, id_field=None, limit=None)` |
| `fgao.integrations.providers` | `OpenAILLM(model, *, base_url=None, api_key=None, max_tokens=128, temperature=0.0, system=..., timeout=None, max_retries=None)` (any OpenAI-compatible server), `AnthropicLLM(model, *, api_key=None, max_tokens=128, temperature=0.0, system=..., timeout=None, max_retries=None)` |

The MCP adapter works with `mcp.Client` (SDK 2.x) and `mcp.ClientSession` (SDK 1.x). MCP tools are async, so they run through the async API.

---

## 5. Resilience

### `fgao.resilience.resilient(x, *, attempts=3, backoff=0.5, max_backoff=8.0, timeout=None, rate=None, max_concurrency=None, retry_on=(Exception,), name="", seed=None, kind=None)`

This wraps an LLM or a retriever and keeps its interface.

| parameter | effect |
|---|---|
| `attempts`, `backoff`, `max_backoff` | retries, with exponential backoff and full jitter |
| `timeout` | per attempt, on the async path; on the sync path, set the client's own timeout |
| `rate` | calls per second, using a token bucket shared across threads and tasks |
| `max_concurrency` | cap on simultaneous calls |
| `kind` | `"llm"` or `"retriever"`, for objects that look like both |

The wrappers (`ResilientLLM`, `ResilientRetriever`) expose `.stats`: `calls`, `attempts`, `retries`, `timeouts`, `failures`. The building blocks are `RetryPolicy` and `RateLimiter(rate, burst=None)`.

---

## 6. HTTP server

Install with `pip install "fgao[server]"`.

### `fgao.server.create_app(rag, *, api_keys=None, max_inflight=256, max_queue=1024, request_timeout=60.0, max_batch=256, max_budget=None, max_documents=500, title="FGAO RAG")`

Returns the FastAPI application.

| endpoint | body / result |
|---|---|
| `POST /v1/answer` | `question`, plus optional `budget`, `seed`, `trace`, `documents`, `web_documents`; returns `RAGResult.to_dict()` |
| `POST /v1/answer/stream` | the same body; Server-Sent Events: `step` per decision, then `result` |
| `POST /v1/batch` | `questions`, plus the same optional fields; returns `{"results": [...]}` |
| `GET /v1/info` | sources, costs, priors, controller configuration, limits |
| `GET /health`, `GET /ready`, `GET /metrics`, `GET /docs` | probes, Prometheus metrics, OpenAPI |

With `documents` (strings or `{"text", "title", "id"}` objects), the request is answered from those passages. The server's LLM, costs, priors and controller are used, with up to `max_documents` passages per source.

Status codes:

| code | when |
|---|---|
| 401 | missing or invalid API key |
| 422 | invalid request, or the budget is above the limit |
| 429 | overloaded, with `Retry-After` |
| 502 | a dependency failed after retries |
| 504 | the deadline expired |

### Configuration and CLI

| member | what it does |
|---|---|
| `fgao.server.load_config(path)`, `build_from_config(cfg, base_dir=".")` | return `(FGAORAG, server_settings)` from YAML or JSON; see [deployment](deployment.md) |
| `fgao.server.config.server_kwargs(settings)` | settings → `create_app` keyword arguments; `FGAO_API_KEYS` adds keys |
| `fgao.server.config.HTTPRetriever(url, *, headers=None, timeout=10.0)` | retriever backed by a search service: POST `{"query", "k"}` |
| `fgao serve --config FILE` or `--factory module:function` | run the server; also `--host`, `--port`, `--workers`, `--log-level`, `--keep-alive` |
| `fgao ask --config FILE "question"` | one query; `--budget`, `--explain` |
| `fgao check --config FILE` | build the pipeline and print its layout |
