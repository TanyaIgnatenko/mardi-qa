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

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Set explicitly, because the API defaults to 1.0 while the Ollama
            # backend pins 0.0: left alone, every cross-backend comparison had
            # Anthropic sampling randomly against local models decoding greedily,
            # which is both unequal and unreproducible.
            temperature=self.temperature,
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class _OllamaBackend:
    """Wrapper around the ZIB Ollama HTTP API.

    Uses /api/chat for system+user messages. Honors OLLAMA_API_KEY as a bearer
    token (the ZIB deployment is auth-gated). Per call we set num_predict to
    bound generation length.
    """

    def __init__(self, host: str, api_key: str, model: str, timeout_s: int,
                 temperature: float = 0.0):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

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
            # Reasoning models put their chain of thought in a separate field and
            # spend the token budget on it: asked for JSON, qwen3.5 returned an
            # empty `content`, a full `thinking`, and done_reason "length" — it
            # never reached the answer. Turning thinking off makes these models
            # answer directly. Ollama ignores the flag for models without it.
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": self.temperature,
                # Pinned rather than left to Ollama's defaults (0.9 / 40), so the
                # decoding rule is the same one the Anthropic side uses and is
                # recorded here rather than inherited from a server version.
                "top_p": 1.0,
                "top_k": 0,
            },
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        if content:
            return content

        # Thinking was not suppressed after all. The answer, if any, trails the
        # reasoning — returning that beats returning nothing, and the caller's
        # JSON/SPARQL extraction will tell us whether it is usable.
        thinking = (message.get("thinking") or "").strip()
        if thinking:
            return thinking
        raise RuntimeError(
            f"{self.model} returned no content "
            f"(done_reason={data.get('done_reason')}, eval_count={data.get('eval_count')})"
        )


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
        temperature: float = 0.0,
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        # Greedy by default on both backends. Comparing models only means
        # something if the decoding rule is held fixed across them, and a fixed
        # rule also makes a run reproducible — a differing answer then points at
        # a changed prompt or changed graph data, not at the sampler.
        self.temperature = temperature

        if self.provider == "anthropic":
            self.model = model or config.ANTHROPIC_MODEL
            self.backend = _AnthropicBackend(
                api_key=api_key or config.ANTHROPIC_API_KEY,
                model=self.model,
                temperature=temperature,
            )
        elif self.provider == "ollama":
            self.model = model or config.OLLAMA_MODEL
            self.backend = _OllamaBackend(
                host=config.OLLAMA_HOST,
                api_key=api_key or config.OLLAMA_API_KEY,
                model=self.model,
                timeout_s=config.OLLAMA_TIMEOUT_S,
                temperature=temperature,
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
