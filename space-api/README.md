---
title: FGAO API
emoji: 🧭
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: HTTP API for training-free FGAO orchestration of RAG
---

# FGAO API

FGAO (Feedback-Greedy Agentic Orchestration) is a **retrieval-augmented question-answering API**.

## What it does

For each question, FGAO decides step by step which retrieval and LLM operations to run:
- **Sources:** which source to query, and how deep to search it.
- **Grading:** whether to grade the passages first.
- **Fallback:** when to fall back to a second source.
- **Checking:** when to check the answer against the passage it came from.
- **Abstaining:** when to stop and give no answer.

It returns an answer only if that answer is **grounded** in a retrieved passage and **passed a support check**; otherwise it returns `answer: null`. Every response includes the decision trace.

## Why it is supposed to be better

- **No LLM decides the control flow.** Each step is chosen by an explicit score: predicted progress on checkable goals, minus cost, plus a one-step lookahead. Decisions are cheap, deterministic and inspectable. A prompt cannot talk the pipeline out of verifying.
- **Feedback within the question.** When an operation fails, FGAO's estimate for it and for similar operations drops, so it escalates instead of repeating itself. Fixed pipelines such as CRAG and Self-RAG hard-code those rules.
- **Budget-aware.** Each operation has a cost. FGAO stops when nothing left is worth its cost, rather than running a fixed sequence.
- **Abstains when unsure.** An answer that is not grounded and supported is never returned.
- **Training-free.** There are no learned weights, so any LLM and any retriever can be swapped in.

## How to use

1. Add a Space secret `HF_TOKEN` (Settings → Variables and secrets). The LLM is called through Hugging Face Inference Providers; change `model` in `fgao.yaml` to use another model.
2. Optionally add `FGAO_API_KEYS=key1,key2` to require `Authorization: Bearer <key>`.

Ask with your own passages:

```bash
curl -X POST https://<space-host>/v1/answer -H "Content-Type: application/json" -d '{
  "question": "What is the capital of Peru?",
  "documents": ["Lima is the capital of Peru.", {"title": "Chile", "text": "Santiago is the capital of Chile."}],
  "trace": false
}'
```

To ask over the built-in corpus instead (`data/kb.jsonl`, one `{"text", "title", "id"}` per line; replace it with your own), omit `documents`.

From Python:

```python
import requests
r = requests.post("https://<space-host>/v1/answer", json={"question": "...", "documents": [...]}).json()
r["answer"], r["support_passage"], r["trace"]
```

| endpoint | |
|---|---|
| `POST /v1/answer` | `question`, plus optional `documents`, `web_documents`, `budget`, `trace` |
| `POST /v1/answer/stream` | the same, as Server-Sent Events: one `step` per decision, then `result` |
| `POST /v1/batch` | `questions: [...]`, plus optional `documents` |
| `GET /v1/info`, `/health`, `/metrics`, `/docs` | configuration, health, Prometheus metrics, interactive API docs |

The `<space-host>` is shown under "Embed this Space" → "Direct URL", for example `angshul-fgao-api.hf.space`.

Library and source: https://github.com/AngshulMajumdar/FGAO
