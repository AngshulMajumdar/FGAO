"""OpenAI-compatible and Anthropic chat models as FGAO LLMs (sync and native async).

``OpenAILLM`` also serves any OpenAI-compatible endpoint (vLLM, TGI, Ollama,
LM Studio, ...) through ``base_url``. Clients are created lazily, so a process
that only uses the async API never opens a sync client and vice versa.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..rag.prompts import SYSTEM_PROMPT


def _messages(system: Optional[str], prompt: str) -> List[Dict[str, str]]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


class OpenAILLM:
    def __init__(self, model: str, *, client: Any = None, aclient: Any = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: int = 128, temperature: float = 0.0,
                 system: Optional[str] = SYSTEM_PROMPT, timeout: Optional[float] = None,
                 max_retries: Optional[int] = None) -> None:
        self.model, self.max_tokens, self.temperature, self.system = model, max_tokens, temperature, system
        self._kw = {k: v for k, v in (("base_url", base_url), ("api_key", api_key), ("timeout", timeout),
                                      ("max_retries", max_retries)) if v is not None}
        self._client, self._aclient = client, aclient

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(**self._kw)
        return self._client

    @property
    def aclient(self):
        if self._aclient is None:
            from openai import AsyncOpenAI

            self._aclient = AsyncOpenAI(**self._kw)
        return self._aclient

    def _args(self, prompt: str) -> Dict[str, Any]:
        return {"model": self.model, "messages": _messages(self.system, prompt), "max_tokens": self.max_tokens,
                "temperature": self.temperature}

    def complete(self, prompt: str) -> str:
        out = self.client.chat.completions.create(**self._args(prompt))
        return (out.choices[0].message.content or "").strip()

    async def acomplete(self, prompt: str) -> str:
        out = await self.aclient.chat.completions.create(**self._args(prompt))
        return (out.choices[0].message.content or "").strip()


class AnthropicLLM:
    def __init__(self, model: str, *, client: Any = None, aclient: Any = None, api_key: Optional[str] = None,
                 max_tokens: int = 128, temperature: float = 0.0, system: Optional[str] = SYSTEM_PROMPT,
                 timeout: Optional[float] = None, max_retries: Optional[int] = None) -> None:
        self.model, self.max_tokens, self.temperature, self.system = model, max_tokens, temperature, system
        self._kw = {k: v for k, v in (("api_key", api_key), ("timeout", timeout), ("max_retries", max_retries))
                    if v is not None}
        self._client, self._aclient = client, aclient

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(**self._kw)
        return self._client

    @property
    def aclient(self):
        if self._aclient is None:
            from anthropic import AsyncAnthropic

            self._aclient = AsyncAnthropic(**self._kw)
        return self._aclient

    def _args(self, prompt: str) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                              "messages": [{"role": "user", "content": prompt}]}
        if self.system:
            kw["system"] = self.system
        return kw

    def complete(self, prompt: str) -> str:
        out = self.client.messages.create(**self._args(prompt))
        return "".join(getattr(b, "text", "") for b in out.content).strip()

    async def acomplete(self, prompt: str) -> str:
        out = await self.aclient.messages.create(**self._args(prompt))
        return "".join(getattr(b, "text", "") for b in out.content).strip()


__all__ = ["OpenAILLM", "AnthropicLLM"]
