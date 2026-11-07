"""Minimal client for TypeSafe Jev (``POST /v1/systemone``).

Jev is a calibrated classifier, not a text generator: it reads a ``state`` and
answers typed questions (``noul`` = probability a statement is true, ``choice``
= one label from a set, ``score`` = position on an ordered scale) with
probabilities. That makes it a cheap evaluator for decisions an LLM judge would
otherwise make with a full generation.

Docs: https://docs.typesafe.ai/api · background: https://mastra.ai/articles/jev

Raw HTTP via ``httpx`` keeps the dependency surface small (``typesafe-sdk`` works
too). Configure with ``TYPESAFE_API_KEY`` and optionally ``JEV_MODEL``; pin a
version such as ``jev-1.13.0`` when you tune thresholds, since ``jev-latest``
moves with each release.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from hubkit.llm.usage import normalize

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# USD per million input tokens; output tokens are free. Source: mastra.ai/articles/jev
# (as of 2026-11). Update when TypeSafe changes pricing.
INPUT_PRICE_PER_M = 0.042

RETRY_STATUS = {429, 500, 502, 503, 504}


class JevError(Exception):
    """Jev could not answer (missing key, validation error, network failure)."""

    def __init__(self, message: str, usage: dict | None = None):
        super().__init__(message)
        self.usage = usage


# ---------------------------------------------------------------- question builders

def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    """Probability (0-1) that ``instructions`` (a yes/no question or statement) holds."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    """One label from ``criteria`` (label -> description)."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict:
    """Position on an ordered scale; ``criteria[i]`` describes level ``i`` (0-based)."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


# ---------------------------------------------------------------- client

@dataclass
class JevResult:
    answers: dict[str, dict]
    usage: dict  # hubkit.llm.usage.normalize() shape
    model: str
    raw: dict = field(default_factory=dict)

    def noul(self, key: str) -> float:
        return float(self.answers[key]["noul"])

    def choice(self, key: str) -> tuple[str, float]:
        a = self.answers[key]
        return a["choice"], float(a.get("confidence", 1.0))

    def score(self, key: str) -> tuple[float, float]:
        a = self.answers[key]
        return float(a["score"]), float(a.get("confidence", 1.0))


def jev_cost(input_tokens: int) -> float:
    return input_tokens * INPUT_PRICE_PER_M / 1_000_000


class JevClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, *, timeout: float = 30.0,
                 retries: int = 2, transport: httpx.BaseTransport | None = None, url: str = API_URL):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set (key from https://console.typesafe.ai/keys)")
        self.model = model or os.getenv("JEV_MODEL") or DEFAULT_MODEL
        self.url, self.retries = url, retries
        self._http = httpx.Client(timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {self.api_key}"})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def system_one(self, state: str | dict | list, questions: dict[str, dict]) -> JevResult:
        body = {"model": self.model, "state": state, "questions": questions}
        started = time.monotonic()
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = self._http.post(self.url, json=body)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if r.status_code == 200:
                    data = r.json()
                    tokens = data.get("usage") or {}
                    model = data.get("model") or self.model
                    usage = normalize(tokens, provider="typesafe", model=model,
                                      duration_ms=(time.monotonic() - started) * 1000,
                                      cost_usd=jev_cost(int(tokens.get("input_tokens") or 0)),
                                      cost_source="price_table")
                    return JevResult(answers=data.get("answers") or {}, usage=usage, model=model, raw=data)
                last = f"HTTP {r.status_code}: {_detail(r)}"
                if r.status_code not in RETRY_STATUS:
                    break  # 400/401/422: retrying will not help
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise JevError(f"Jev request failed: {last}",
                       usage=normalize({}, provider="typesafe", model=self.model,
                                       duration_ms=(time.monotonic() - started) * 1000,
                                       cost_usd=0.0, cost_source="price_table"))


def _detail(r: httpx.Response) -> str:
    try:
        return json.dumps(r.json().get("detail"), ensure_ascii=False)[:300]
    except Exception:
        return r.text[:300]
