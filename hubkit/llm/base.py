"""LLM provider interface (ADR-005).

Nodes call ``provider.complete(...)`` and never import a vendor SDK directly, so
the same graph runs on the Claude subscription, the Anthropic API, or a fake
provider in tests.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    text: str
    data: Any = None  # parsed JSON when a schema was requested
    usage: dict = field(default_factory=dict)


class LLMError(Exception):
    """``usage`` carries tokens already spent when the call ran but its output was unusable."""

    def __init__(self, *args: Any, usage: dict | None = None):
        super().__init__(*args)
        self.usage = usage


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def complete(self, system: str, prompt: str, json_schema: dict | None = None, *,
                 model: str | None = None, effort: str | None = None) -> LLMResult:
        """Single-shot completion. With ``json_schema``, ``result.data`` holds the parsed object.

        ``model`` / ``effort`` come from ``hubkit.llm.routing.route(node)``; None means the
        provider's default."""


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in ``text`` (tolerates code fences and prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return valid JSON: {text[:200]!r}")


def strict_schema(schema: dict) -> dict:
    """Copy of a JSON schema with ``additionalProperties: false`` on every object, as structured
    outputs require."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: strict_schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    if out.get("type") == "object" or "properties" in out:
        out.setdefault("additionalProperties", False)
    return out


def json_instruction(schema: dict) -> str:
    return (
        "\n\nRespond with a single JSON object that matches this JSON Schema. "
        "No prose, no code fences.\n" + json.dumps(schema, ensure_ascii=False)
    )
