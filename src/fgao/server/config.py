"""Build an ``FGAORAG`` (and server settings) from a YAML or JSON file.

Example (see ``deploy/fgao.yaml``)::

    llm:
      provider: openai              # openai | anthropic | huggingface | transformers | factory
      model: gpt-4o-mini
      base_url: null                # any OpenAI-compatible server (vLLM, TGI, Ollama, ...)
      api_key_env: OPENAI_API_KEY
      temperature: 0
      max_tokens: 128
      timeout: 30
    resilience:                     # applied to the LLM and to every retriever
      attempts: 3
      backoff: 0.5
      timeout: 30
      rate: null                    # calls per second, per process
      max_concurrency: 64
    cache: {path: null}             # prompt cache (JSON file) in front of the LLM
    kb:  {type: bm25, path: /data/kb.jsonl}
    web: {type: http, url: http://search:9000/search}
    budget: 16
    on_error: record
    server: {max_inflight: 256, max_queue: 1024, request_timeout: 60, max_batch: 256}  # keys: FGAO_API_KEYS env

Retriever types

  bm25     path to .jsonl (one {"text", "title"?, "id"?} per line), .json (list), or .txt (blank-line paragraphs)
  dense    the same files + embedder {provider: huggingface | sentence-transformers, model}
  http     POST {"query", "k"} to ``url``; expects a list of documents or {"documents": [...]}
  factory  "module:function" (+ ``kwargs``) returning any retriever, LangChain retriever or vector store

Instead of ``kb`` / ``web`` an explicit ``sources:`` list may be given:
``[{name, retriever: <spec>, k, cost, family, rewrite}]``.
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..controller import FGAOConfig
from ..rag.documents import Document, as_documents
from ..rag.llms import CachedLLM
from ..rag.pipeline import FGAORAG, Costs, Priors, Source, Weights, default_sources
from ..rag.retrievers import BM25Retriever, DenseRetriever
from ..resilience import resilient


def load_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("YAML configs need pyyaml: pip install 'fgao[server]'") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def import_object(ref: str) -> Any:
    mod, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {ref!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_documents(path: str) -> List[Document]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return as_documents(json.loads(line) for line in fh if line.strip())
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return as_documents(data["documents"] if isinstance(data, dict) else data)
    with open(path, encoding="utf-8") as fh:
        return as_documents(p.strip() for p in fh.read().split("\n\n") if p.strip())


class HTTPRetriever:
    """Retriever backed by a search service: POST {"query", "k"} -> documents."""

    def __init__(self, url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 10.0) -> None:
        import httpx

        self.url, self.headers, self.timeout = url, dict(headers or {}), timeout
        self._client = httpx.Client(timeout=timeout, headers=self.headers)
        self._aclient: Any = None

    @staticmethod
    def _docs(data: Any) -> List[Document]:
        return as_documents(data.get("documents", data.get("results", [])) if isinstance(data, dict) else data)

    def retrieve(self, query: str, k: int) -> List[Document]:
        r = self._client.post(self.url, json={"query": query, "k": k})
        r.raise_for_status()
        return self._docs(r.json())[:k]

    async def aretrieve(self, query: str, k: int) -> List[Document]:
        import httpx

        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        r = await self._aclient.post(self.url, json={"query": query, "k": k})
        r.raise_for_status()
        return self._docs(r.json())[:k]


def _env(spec: Dict[str, Any], key: str) -> Optional[str]:
    name = spec.get(key)
    return os.environ.get(name) if name else None


def build_llm(spec: Dict[str, Any], client_retries: Optional[int] = None):
    """``client_retries``: SDK-level retries (0 when FGAO's own resilience layer retries)."""
    p = (spec.get("provider") or "openai").lower()
    sdk_retries = spec.get("client_retries", client_retries)
    common = {k: spec[k] for k in ("max_tokens", "temperature", "system") if k in spec}
    if p in ("openai", "openai-compatible", "vllm", "ollama", "tgi"):
        from ..integrations.providers import OpenAILLM

        return OpenAILLM(spec["model"], base_url=spec.get("base_url"), api_key=_env(spec, "api_key_env") or spec.get("api_key"),
                         timeout=spec.get("timeout"), max_retries=sdk_retries, **common)
    if p == "anthropic":
        from ..integrations.providers import AnthropicLLM

        return AnthropicLLM(spec["model"], api_key=_env(spec, "api_key_env"), timeout=spec.get("timeout"),
                            max_retries=sdk_retries, **common)
    if p in ("huggingface", "hf"):
        from ..integrations.huggingface import HFInferenceLLM

        return HFInferenceLLM(spec["model"], token=_env(spec, "api_key_env") or os.environ.get("HF_TOKEN"),
                              provider=spec.get("hf_provider"), base_url=spec.get("base_url"),
                              timeout=spec.get("timeout"), **common)
    if p == "transformers":
        from ..integrations.huggingface import TransformersLLM

        return TransformersLLM(spec["model"], max_new_tokens=spec.get("max_tokens", 64))
    if p == "factory":
        return import_object(spec["factory"])(**spec.get("kwargs", {}))
    raise ValueError(f"unknown llm provider {p!r}")


def build_retriever(spec: Any, base_dir: str = "."):
    if isinstance(spec, str):
        spec = {"type": "bm25", "path": spec}
    t = spec.get("type", "bm25")
    path = spec.get("path")
    if path and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    if t == "bm25":
        return BM25Retriever(load_documents(path))
    if t == "dense":
        emb = spec.get("embedder", {})
        if emb.get("provider", "huggingface") in ("huggingface", "hf"):
            from ..integrations.huggingface import hf_embedder

            fn = hf_embedder(emb["model"], token=os.environ.get(emb.get("api_key_env", "HF_TOKEN")))
        else:
            from ..integrations.huggingface import st_embedder

            fn = st_embedder(emb["model"])
        return DenseRetriever(load_documents(path), fn)
    if t == "http":
        return HTTPRetriever(spec["url"], headers=spec.get("headers"), timeout=spec.get("timeout", 10.0))
    if t == "factory":
        return import_object(spec["factory"])(**spec.get("kwargs", {}))
    raise ValueError(f"unknown retriever type {t!r}")


def build_from_config(cfg: Dict[str, Any], base_dir: str = ".") -> Tuple[FGAORAG, Dict[str, Any]]:
    """Return (pipeline, server settings)."""
    if "factory" in cfg:  # a function returning a ready FGAORAG
        rag = import_object(cfg["factory"])(**cfg.get("kwargs", {}))
        return rag, dict(cfg.get("server", {}))
    res = cfg.get("resilience")
    llm = build_llm(cfg.get("llm", {}), client_retries=0 if res else None)
    if cfg.get("cache"):
        if cfg["cache"].get("path") or cfg["cache"].get("enabled", True):
            llm = CachedLLM(llm, cfg["cache"].get("path"))
    if res:
        llm = resilient(llm, name="llm", kind="llm", **res)

    def wrap(r, name):
        return resilient(r, name=f"retriever:{name}", kind="retriever", **res) if res else r

    costs = Costs(**cfg.get("costs", {}))
    if "sources" in cfg:
        sources = [Source(s["name"], wrap(build_retriever(s["retriever"], base_dir), s["name"]), k=s.get("k", 5),
                          cost=s.get("cost", 1.0), family=s.get("family", "kb"), rewrite=s.get("rewrite", False))
                   for s in cfg["sources"]]
    else:
        if "kb" not in cfg:
            raise ValueError("config needs 'kb' (and optionally 'web'), 'sources', or 'factory'")
        kb = wrap(build_retriever(cfg["kb"], base_dir), "kb")
        web = wrap(build_retriever(cfg["web"], base_dir), "web") if cfg.get("web") else None
        sources = default_sources(kb, web, costs)
    rag = FGAORAG(
        sources, llm, costs=costs, priors=Priors(**cfg.get("priors", {})), weights=Weights(**cfg.get("weights", {})),
        budget=float(cfg.get("budget", 16.0)), config=FGAOConfig(**cfg.get("controller", {})),
        on_error=cfg.get("on_error", "raise"),
    )
    return rag, dict(cfg.get("server", {}))


def load_config(path: str) -> Tuple[FGAORAG, Dict[str, Any]]:
    return build_from_config(load_file(path), os.path.dirname(os.path.abspath(path)))


def server_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: settings[k] for k in ("max_inflight", "max_queue", "request_timeout", "max_batch", "max_budget",
                                    "max_documents", "title")
           if k in settings}
    keys = list(settings.get("api_keys", []))
    env = settings.get("api_keys_env", "FGAO_API_KEYS")  # setting FGAO_API_KEYS always enables authentication
    if env and os.environ.get(env):
        keys += [k.strip() for k in os.environ[env].split(",") if k.strip()]
    if keys:
        out["api_keys"] = keys
    return out
