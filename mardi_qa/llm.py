"""Minimal LLM wrapper. Swap this module out for Gemini/OpenAI if preferred."""

from __future__ import annotations

import json
import re

import anthropic

from . import config


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)
        self.model = model or config.ANTHROPIC_MODEL

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.complete(system, user, max_tokens=800)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last-resort: grab the first {...} block.
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                return json.loads(m.group(0))
            raise
