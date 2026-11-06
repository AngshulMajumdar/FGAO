"""Hugging Face components (install ``fgao[huggingface]``; ``transformers`` for local models).

* ``HFInferenceLLM``   chat model behind the Hugging Face Inference API / Providers
* ``TransformersLLM``  local ``transformers`` text-generation model
* ``hf_embedder``      embedding function for ``DenseRetriever`` (Inference API)
* ``st_embedder``      embedding function from a local ``sentence-transformers`` model
* ``load_hf_corpus``   documents from a Hugging Face dataset
"""
from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..rag.documents import Document
from ..rag.prompts import SYSTEM_PROMPT


class HFInferenceLLM:
    """``huggingface_hub`` chat completion (Inference API / Providers / a TGI endpoint) as an FGAO LLM.

    Sync calls use ``InferenceClient``, async calls ``AsyncInferenceClient``.
    The token defaults to ``HF_TOKEN``. ``temperature=None`` leaves the
    provider default; set ``temperature=0`` where the provider accepts it for
    deterministic calls. ``base_url`` targets a dedicated endpoint.
    """

    def __init__(self, model: str, *, token: Optional[str] = None, provider: Optional[str] = None,
                 base_url: Optional[str] = None, max_tokens: int = 128, temperature: Optional[float] = None,
                 system: Optional[str] = SYSTEM_PROMPT, timeout: Optional[float] = None, client: Any = None,
                 aclient: Any = None, **kwargs: Any) -> None:
        self.model = model
        self._ckw: dict = {"token": token or os.environ.get("HF_TOKEN")}
        for k, v in (("provider", provider), ("base_url", base_url), ("timeout", timeout)):
            if v is not None:
                self._ckw[k] = v
        self._client, self._aclient = client, aclient
        self.max_tokens, self.temperature, self.system, self.kwargs = max_tokens, temperature, system, kwargs

    @property
    def client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(**self._ckw)
        return self._client

    @property
    def aclient(self):
        if self._aclient is None:
            from huggingface_hub import AsyncInferenceClient

            self._aclient = AsyncInferenceClient(**self._ckw)
        return self._aclient

    def _args(self, prompt: str) -> dict:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [
            {"role": "user", "content": prompt}]
        kw = dict(self.kwargs)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        kw.update(messages=messages, max_tokens=self.max_tokens)
        if "base_url" not in self._ckw:
            kw["model"] = self.model
        return kw

    def complete(self, prompt: str) -> str:
        out = self.client.chat_completion(**self._args(prompt))
        return (out.choices[0].message.content or "").strip()

    async def acomplete(self, prompt: str) -> str:
        out = await self.aclient.chat_completion(**self._args(prompt))
        return (out.choices[0].message.content or "").strip()


class TransformersLLM:
    """A local ``transformers`` model (greedy decoding, chat template when available)."""

    def __init__(self, model: Any, *, max_new_tokens: int = 64, system: Optional[str] = SYSTEM_PROMPT,
                 **pipeline_kwargs: Any) -> None:
        if isinstance(model, str):
            from transformers import pipeline

            model = pipeline("text-generation", model=model, **pipeline_kwargs)
        self.pipe, self.max_new_tokens, self.system = model, max_new_tokens, system

    def complete(self, prompt: str) -> str:
        tok = getattr(self.pipe, "tokenizer", None)
        if tok is not None and getattr(tok, "chat_template", None):
            msgs = ([{"role": "system", "content": self.system}] if self.system else []) + [
                {"role": "user", "content": prompt}]
            out = self.pipe(msgs, max_new_tokens=self.max_new_tokens, do_sample=False)
            gen = out[0]["generated_text"]
            return (gen[-1]["content"] if isinstance(gen, list) else str(gen)).strip()
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)
        return out[0]["generated_text"].strip()


def hf_embedder(model: str, *, token: Optional[str] = None, client: Any = None) -> Callable[[List[str]], List[List[float]]]:
    """Embedding function backed by ``InferenceClient.feature_extraction``."""
    if client is None:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token or os.environ.get("HF_TOKEN"))

    def embed(texts: List[str]) -> List[List[float]]:
        rows = []
        for t in texts:
            v = client.feature_extraction(t, model=model)
            v = v.tolist() if hasattr(v, "tolist") else v
            while isinstance(v, list) and v and isinstance(v[0], list) and len(v) == 1:
                v = v[0]
            if v and isinstance(v[0], list):  # token embeddings: mean-pool
                v = [sum(col) / len(v) for col in zip(*v)]
            rows.append([float(x) for x in v])
        return rows

    return embed


def st_embedder(model: str, **kwargs: Any) -> Callable[[List[str]], Any]:
    """Embedding function from a local ``sentence-transformers`` model."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, **kwargs)
    return lambda texts: m.encode(list(texts), normalize_embeddings=True)


def load_hf_corpus(path: str, *, split: str = "train", text_field: str = "text", title_field: Optional[str] = None,
                   id_field: Optional[str] = None, limit: Optional[int] = None, **kwargs: Any) -> List[Document]:
    """Documents from a Hugging Face dataset (``datasets.load_dataset``)."""
    from datasets import load_dataset

    ds = load_dataset(path, split=split, **kwargs)
    out: List[Document] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(Document(text=str(row[text_field]), title=str(row[title_field]) if title_field else "",
                            id=str(row[id_field]) if id_field else str(i)))
    return out


__all__ = ["HFInferenceLLM", "TransformersLLM", "hf_embedder", "st_embedder", "load_hf_corpus"]
