"""LLM wrappers with pluggable backend (Anthropic / Ollama).

Public surface unchanged: pipeline.py still calls .complete() and .complete_json().
The `provider` argument selects which backend is used; defaults come from config.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import requests
import anthropic

from . import config


# ----- Backend implementations -------------------------------------------------


class _AnthropicBackend:
    """Wrapper around Anthropic's messages API."""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class _OllamaBackend:
    """Wrapper around the ZIB Ollama HTTP API.

    Uses /api/chat for system+user messages. Honors OLLAMA_API_KEY as a bearer
    token (the ZIB deployment is auth-gated). Per call we set num_predict to
    bound generation length.
    """

    def __init__(self, host: str, api_key: str, model: str, timeout_s: int):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,  # deterministic for evaluation
            },
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        # /api/chat returns {"message": {"role": "assistant", "content": "..."}, ...}
        return data.get("message", {}).get("content", "")


# ----- Public LLM facade -------------------------------------------------------


class LLM:
    """Provider-agnostic LLM facade used by the pipeline.

    Examples:
        LLM()                            # uses config defaults
        LLM(provider="ollama", model="qwen3-coder-next:latest")
        LLM(provider="anthropic", model="claude-sonnet-4-5")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()

        if self.provider == "anthropic":
            self.model = model or config.ANTHROPIC_MODEL
            self.backend = _AnthropicBackend(
                api_key=api_key or config.ANTHROPIC_API_KEY,
                model=self.model,
            )
        elif self.provider == "ollama":
            self.model = model or config.OLLAMA_MODEL
            self.backend = _OllamaBackend(
                host=config.OLLAMA_HOST,
                api_key=api_key or config.OLLAMA_API_KEY,
                model=self.model,
                timeout_s=config.OLLAMA_TIMEOUT_S,
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: {self.provider!r}. "
                "Set LLM_PROVIDER to 'anthropic' or 'ollama'."
            )

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        return self.backend.complete(system, user, max_tokens)

    def complete_json(self, system: str, user: str) -> dict:
        """Same as complete() but expects/repairs a JSON object output."""
        raw = self.complete(system, user, max_tokens=800)
        # Strip markdown fences if the model added them
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last-resort: grab the first {...} block.
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                return json.loads(m.group(0))
            raise

    def label(self) -> str:
        """Short human-readable identifier, useful for logs and result folders."""
        return f"{self.provider}/{self.model}"
