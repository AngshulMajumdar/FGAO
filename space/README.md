---
title: FGAO
emoji: 🧭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Training-free numerical orchestration for RAG pipelines
---

# FGAO: Feedback-Greedy Agentic Orchestration

This Space hosts the `fgao` Python library: its API reference and a playground that runs it.

FGAO is a **training-free, model-free controller** for retrieval-augmented generation and other
tool-using pipelines.
- **No LLM in the loop.** It never asks an LLM what to do next.
- **How it decides.** Each step goes to the operation with the highest predicted progress on
  verifiable goals, net of cost, plus a one-step lookahead. Observed outcomes correct those
  predictions while it runs.
- **What it returns.** An answer only when the answer is grounded in a retrieved passage and passed
  a support check. Otherwise it abstains.

## Install

```bash
pip install fgao                    # core, no dependencies
pip install "fgao[langchain]"       # LangChain / LangGraph adapters
pip install "fgao[huggingface]"     # Hugging Face Inference and embeddings
```

## Use

```python
from fgao import FGAORAG
from fgao.integrations.huggingface import HFInferenceLLM

rag = FGAORAG.from_retrievers(
    kb=retriever,          # your retriever, a LangChain retriever/vector store, or a list of texts
    web=web_search,        # optional second source
    llm=HFInferenceLLM("Qwen/Qwen2.5-7B-Instruct", temperature=0),
)
r = rag.invoke("What is the capital of Japan?")
print(r.answer)            # None means FGAO abstained
print(r.explain())         # the decision trace
```

The library has three layers:
- **`FGAORAG`** — the ready-made RAG orchestrator.
- **`Workflow`** — declare goals and tools, and FGAO orders them.
- **`FGAOController`** — `select_action`, `update` and `should_stop` for your own loop.

It also has adapters for LangChain (`as_runnable`, `as_tool`), LangGraph (`as_node`), Hugging Face,
OpenAI-compatible endpoints and Anthropic. The **API** tab documents all of it.

## Documentation

Source: [https://github.com/AngshulMajumdar/FGAO](https://github.com/AngshulMajumdar/FGAO)

- [Algorithm](https://huggingface.co/spaces/Angshul/FGAO/blob/main/docs/algorithm.md): exactly what the controller computes.
- [API reference](https://huggingface.co/spaces/Angshul/FGAO/blob/main/docs/api.md): every public class and function.
- [Deployment](https://huggingface.co/spaces/Angshul/FGAO/blob/main/docs/deployment.md): the HTTP service, config file, Docker and monitoring.

## Playground setup

The playground calls a model through Hugging Face Inference Providers:
- **Shared setup:** add an `HF_TOKEN` secret in the Space settings.
- **Per visitor:** each visitor pastes their own token instead.
- **Model choice:** set `FGAO_MODEL` to change the default model.
