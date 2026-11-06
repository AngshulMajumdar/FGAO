# Deployment

## Configuration file

`deploy/fgao.yaml` is a complete example. It has these sections:

| section | keys |
|---|---|
| `llm` | `provider`: one of `openai`, `anthropic`, `huggingface`, `transformers`, `factory`; plus `model`, `base_url`, `api_key_env`, `temperature`, `max_tokens`, `timeout`, `system`; `factory` + `kwargs` for your own LLM |
| `resilience` | `attempts`, `backoff`, `max_backoff`, `timeout`, `rate`, `max_concurrency`. Applied to the LLM and to every retriever. SDK-level retries are switched off when this section is present. |
| `cache` | `{path}`: a prompt cache in front of the LLM |
| `kb`, `web` | retriever specs for the default layout (`kb5`, `kb10`, `rw`, and `web` if given) |
| `sources` | an explicit layout instead: `[{name, retriever, k, cost, family, rewrite}]` |
| `budget`, `costs`, `priors`, `weights`, `controller` | pipeline settings. Defaults are the evaluated v2 configuration. |
| `on_error` | `raise` or `record` |
| `server` | `max_inflight`, `max_queue`, `request_timeout`, `max_batch`, `max_budget`, `max_documents`, `api_keys`, `api_keys_env` |
| `factory` | `module:function` returning a ready `FGAORAG`. It replaces the whole pipeline section. |

Retriever specs:

| type | fields |
|---|---|
| `bm25` | `path` (`.jsonl` / `.json` / `.txt`) |
| `dense` | `path`, plus `embedder: {provider, model}` with provider `huggingface` or `sentence-transformers` |
| `http` | `url`, `headers`, `timeout` |
| `factory` | `factory: module:function` + `kwargs` |

**Request documents.** Clients may send their own passages with each request (`documents`, optionally `web_documents`). These are answered with the configured LLM, costs, priors and controller, so a deployment does not need a corpus of its own; `kb` can then be a small placeholder file.

Relative paths are resolved against the config file's directory. Secrets come only from environment variables (`api_key_env`, `FGAO_API_KEYS`).

## Processes, replicas and probes

- **One worker per container.** Run one worker per container and scale with replicas. Admission control (`max_inflight`, `max_queue`) and `/metrics` are per process. `--workers N` is available for single-host deployments.
- **Probes.** Use `GET /health` for liveness and `GET /ready` for readiness.
- **Keep-alive.** Keep `--keep-alive` (default 75 s) above your load balancer's idle timeout. Otherwise the server may close idle connections the balancer is about to reuse.
- **Capacity.** Size `max_inflight` from your model provider's concurrency limit, divided by the number of replicas. Each query makes a few model calls one after another (typically 2–4 with the default layout).
- **Deadlines.** Set `request_timeout` above the `resilience.timeout` × `attempts` you allow for one model call, multiplied by the number of calls a query makes.

## Monitoring

Scrape `GET /metrics`. The useful series:

| series | what to watch |
|---|---|
| `fgao_queries_total{outcome}` | `verified`, `abstained`, `timeout`, `error`: answer rate and failure rate |
| `fgao_stop_status_total{status}` | why FGAO stopped |
| `fgao_request_duration_seconds`, `fgao_query_cost` | histograms |
| `fgao_operation_calls_total{op}` | where calls go: generate, grade, rewrite, verify, `retrieve:<source>` |
| `fgao_dependency_{retries,timeouts,failures}_total{name}` | provider health |
| `fgao_inflight`, `fgao_queue` | saturation |
| `fgao_http_requests_total{route,code}` | 429s mean the service is shedding load |

## Load testing

```bash
python loadtest/make_corpus.py --out /tmp/lt/data
python loadtest/fake_openai.py --port 9100 --error-rate 0.05 --hang-rate 0.01 &
fgao serve --config my-test-config.yaml --port 8000 &       # llm.base_url: http://127.0.0.1:9100/v1
python loadtest/loadgen.py --questions /tmp/lt/data/questions.jsonl --concurrency 64 --requests 2000
```
