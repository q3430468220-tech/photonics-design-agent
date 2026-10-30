"""LLM provider layer - plain HTTP via `requests`, no vendor SDK required.

Providers (env PRISM_LLM_PROVIDER):
  anthropic : Messages API. ANTHROPIC_API_KEY, optional ANTHROPIC_BASE_URL
  openai    : Chat Completions (works for OpenAI, DeepSeek, Qwen/DashScope, relays).
              OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1)
  scripted  : replays canned responses (tests / offline demos)

Model name from PRISM_MODEL.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional


class LLMError(RuntimeError):
    pass


class LLM:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 4000, temperature: float = 0.3) -> str:
        raise NotImplementedError


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or os.environ.get("PRISM_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

    def complete(self, system, user, max_tokens=4000, temperature=0.3):
        import requests

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_with_retry(f"{self.base_url}/v1/messages", headers, body)
        try:
            return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected Anthropic response: {data}") from e


class OpenAICompatLLM(LLM):
    name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or os.environ.get("PRISM_MODEL", "gpt-4o")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY not set")

    def complete(self, system, user, max_tokens=4000, temperature=0.3):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        data = _post_with_retry(f"{self.base_url}/chat/completions", headers, body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected OpenAI-compatible response: {data}") from e


class ScriptedLLM(LLM):
    """Replays a fixed list of responses; raises when exhausted."""

    name = "scripted"

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[dict] = []

    def complete(self, system, user, max_tokens=4000, temperature=0.3):
        self.calls.append({"system": system, "user": user})
        if not self.responses:
            raise LLMError("ScriptedLLM has no responses left")
        return self.responses.pop(0)


def _post_with_retry(url: str, headers: dict, body: dict, retries: int = 4, timeout: int = 180) -> dict:
    import requests

    delay = 2.0
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504, 529):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        except requests.RequestException as e:
            last = str(e)
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"LLM request failed after {retries} attempts: {last}")


def get_llm(provider: Optional[str] = None, **kw) -> LLM:
    provider = (provider or os.environ.get("PRISM_LLM_PROVIDER", "anthropic")).lower()
    if provider == "anthropic":
        return AnthropicLLM(**kw)
    if provider in ("openai", "openai-compatible", "deepseek", "qwen"):
        return OpenAICompatLLM(**kw)
    if provider == "scripted":
        return ScriptedLLM(kw.get("responses", []))
    raise LLMError(f"unknown LLM provider {provider!r}")
