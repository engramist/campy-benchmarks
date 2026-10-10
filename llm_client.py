"""
campy-benchmarks / llm_client.py
Direct LLM access for the baselines (baselines.py), with no engine imports.

Mirrors how Campy's own `ask` reaches the model (campy/brain/llm/provider.py):
the base `[llm]` config section, an OpenAI-compatible chat.completions call,
temperature 0, the same default base URLs per provider. So a baseline and
Campy differ only in the context they put in the prompt.

One deliberate difference: for Ollama this calls the native /api/chat with an
explicit num_ctx sized to the prompt. The OpenAI-compatible endpoint uses the
server's default context window and silently drops the START of a longer
prompt (system prompt included), which would sabotage the full-transcript
baseline. The num_ctx used is recorded per call.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class LLMError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class BaselineLLM:
    def __init__(self, provider: str, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 300.0):
        if provider not in _DEFAULT_BASE_URLS:
            raise LLMError(f"baselines support providers {sorted(_DEFAULT_BASE_URLS)}, not {provider!r}")
        if not model:
            raise LLMError("no LLM model configured for baselines")
        self.provider = provider
        self.model = model
        self.base_url = (base_url or _DEFAULT_BASE_URLS[provider]).rstrip("/")
        self.api_key = api_key or os.environ.get(_KEY_ENV.get(provider, ""), "")
        self.timeout = timeout
        self.calls = 0

    @classmethod
    def from_config(cls, llm_cfg: Dict[str, Any], **overrides: Any) -> "BaselineLLM":
        """Build from a campy `[llm]` table (only the base section: `ask`
        uses create_llm_client(config), not a per-step override)."""
        merged = {k: v for k, v in llm_cfg.items() if not isinstance(v, dict)}
        merged.update({k: v for k, v in overrides.items() if v})
        return cls(
            provider=merged.get("provider", "ollama"),
            model=merged.get("model", ""),
            base_url=merged.get("base_url"),
            api_key=merged.get("api_key"),
            timeout=float(merged.get("timeout_seconds") or 300.0),
        )

    def describe(self) -> Dict[str, Any]:
        return {  # never the key
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": "ollama /api/chat (num_ctx sized per prompt)" if self.provider == "ollama"
                        else "chat/completions",
            "temperature": 0.0,
        }

    def _post(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            raise LLMError(f"{url} returned HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach {url}: {e}") from e

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Return {"text", "latency_ms", "prompt_tokens_est", "num_ctx"}. `max_tokens` caps the
        reply (None = no cap, the default, so answering calls are unchanged)."""
        self.calls += 1
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        t0 = time.perf_counter()
        num_ctx = None
        if self.provider == "ollama":
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            # headroom for tokenizer error (the 4-chars estimate) and the answer
            need = int(prompt_tokens * 1.5) + 1024
            num_ctx = 4096
            while num_ctx < need and num_ctx < 131072:
                num_ctx *= 2
            options: Dict[str, Any] = {"temperature": 0.0, "num_ctx": num_ctx}
            if max_tokens:
                options["num_predict"] = int(max_tokens)
            data = self._post(f"{root}/api/chat", {
                "model": self.model, "messages": messages, "stream": False, "options": options,
            })
            text = (data.get("message") or {}).get("content", "")
        else:
            body: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.0}
            if max_tokens:
                body["max_tokens"] = int(max_tokens)
            data = self._post(f"{self.base_url}/chat/completions", body)
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"unexpected chat response shape: {str(data)[:300]}") from e
        return {
            "text": text,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "prompt_tokens_est": prompt_tokens,
            "num_ctx": num_ctx,
        }
