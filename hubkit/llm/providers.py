"""Concrete LLM providers.

- ``ClaudeSubscriptionProvider`` (default): the Claude Agent SDK, which drives the
  locally installed Claude Code CLI and uses its login, so a Claude subscription
  works without an API key. Personal, local use only, within plan rate limits.
- ``AnthropicAPIProvider``: the Anthropic API with ``ANTHROPIC_API_KEY``; use for
  large eval sweeps.
- ``FakeProvider``: deterministic, for tests and offline demos.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable

from hubkit.config import env
from hubkit.llm.base import LLMError, LLMProvider, LLMResult, extract_json, strict_schema
from hubkit.llm.usage import normalize

log = logging.getLogger(__name__)

# The Claude Code CLI behind the Agent SDK can inject context we never asked for: tool definitions of
# claude.ai connectors and plugin skills, loaded in the background. Measured on the 2026-11-05/06 runs:
# 2-3 calls per run wrote 54k-169k cache tokens on 3-5k-token prompts, 46-83% of each run's cost.
CLI_ISOLATION_ENV = {"ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
CLI_BASE_TOKENS = 4000  # CLI scaffolding + structured-output tool (measured 1.1k-1.8k), with margin


def expected_tokens(system: str, prompt: str, schema: dict | None) -> int:
    """Rough input size of our own payload (about 3 characters per token for mixed en/vi text)."""
    return (len(system) + len(prompt) + (len(json.dumps(schema)) if schema else 0)) // 3 + CLI_BASE_TOKENS


def flag_overhead(usage: dict, expected: int) -> dict:
    """Mark a call whose input side is far larger than what we sent (injected CLI context)."""
    seen = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0))
    if seen > max(3 * expected, expected + 20000):
        usage["overhead_tokens"] = seen - expected
        log.warning("LLM call used %s input-side tokens for a ~%s-token prompt: injected CLI context?", seen, expected)
    return usage


def _run(coro):
    """Run an async coroutine from sync code, also when a loop is already running (Streamlit)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ClaudeSubscriptionProvider(LLMProvider):
    name = "claude_subscription"

    def __init__(self, model: str | None = None, max_turns: int = 3):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise LLMError("pip install claude-agent-sdk, and sign in once with `claude` (Claude Code CLI)") from exc
        self.model = model or env("MODEL")
        self.max_turns = max_turns

    async def _complete(self, system: str, prompt: str, json_schema: dict | None,
                        model: str | None, effort: str | None) -> LLMResult:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

        model = self.model or model  # an explicit constructor model / HUB_MODEL wins over routing
        options = ClaudeAgentOptions(
            system_prompt=system,
            tools=[],  # no built-in tools: this is pure reasoning; data access stays in our graph
            setting_sources=[],  # do not load the user's CLAUDE.md or settings into the analyst
            # isolation: no MCP servers, no claude.ai connectors, no plugins, skills or subagents
            mcp_servers={}, strict_mcp_config=True, plugins=[], skills=[], agents={},
            env=dict(CLI_ISOLATION_ENV),
            max_turns=self.max_turns,
            model=model,
            effort=effort,
            output_format={"type": "json_schema", "schema": json_schema} if json_schema else None,
        )
        text_parts: list[str] = []
        result: Any = None
        model_seen: str | None = None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                model_seen = getattr(msg, "model", None) or model_seen
                text_parts += [b.text for b in msg.content if isinstance(b, TextBlock)]
            elif isinstance(msg, ResultMessage):
                result = msg
        usage = self._usage(result, model_seen or model)
        if usage:
            flag_overhead(usage, expected_tokens(system, prompt, json_schema))
        if result is None or result.is_error:
            raise LLMError(f"Claude subscription call failed: {getattr(result, 'errors', None) or getattr(result, 'subtype', 'no result')}",
                           usage=usage)
        text = result.result or "".join(text_parts)
        data = None
        if json_schema:
            try:
                data = result.structured_output if result.structured_output is not None else extract_json(text)
            except LLMError as exc:
                raise LLMError(str(exc), usage=usage) from exc
        return LLMResult(text=text, data=data, usage=usage)

    def _usage(self, result: Any, model_seen: str | None) -> dict | None:
        if result is None:
            return None
        model_usage = getattr(result, "model_usage", None) or {}
        # the CLI adds a small Haiku side-call, so model_usage often lists two models: report ours
        model = model_seen or (next(iter(model_usage), None) if len(model_usage) == 1 else self.model)
        cost = getattr(result, "total_cost_usd", None)
        return normalize(getattr(result, "usage", None), provider=self.name, model=model,
                         duration_ms=getattr(result, "duration_ms", None),
                         cost_usd=cost, cost_source="sdk_estimate" if cost is not None else None)

    def complete(self, system: str, prompt: str, json_schema: dict | None = None, *,
                 model: str | None = None, effort: str | None = None) -> LLMResult:
        try:
            return _run(self._complete(system, prompt, json_schema, model, effort))
        except LLMError:
            raise
        except Exception as exc:  # SDK/CLI errors (e.g. ResultError on a plan usage limit) must not crash the graph
            msg = str(exc)
            if "limit" in msg.lower() and ("hit" in msg.lower() or "reset" in msg.lower()):
                msg = f"Claude plan usage limit reached: {msg}"
            raise LLMError(f"Claude subscription call failed: {type(exc).__name__}: {msg}"[:500]) from exc


class AnthropicAPIProvider(LLMProvider):
    name = "anthropic_api"

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("pip install anthropic") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.model = model or env("MODEL")
        self.max_tokens = max_tokens

    def request(self, system: str, prompt: str, json_schema: dict | None, model: str | None,
                effort: str | None) -> dict:
        """Messages API request body (separate so it can be tested without the network).

        - structured outputs (``output_config.format``) guarantee schema-valid JSON; the schema is
          made strict (``additionalProperties: false``)
        - top-level ``cache_control`` caches the prompt prefix: critic retries and re-asks resend the
          same prompt plus feedback, so they read the cache instead of paying full input again
        """
        output_config: dict = {}
        if effort:
            output_config["effort"] = effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": strict_schema(json_schema)}
        body = {
            "model": self.model or model or "claude-opus-5-5", "max_tokens": self.max_tokens, "system": system,
            "messages": [{"role": "user", "content": prompt}], "cache_control": {"type": "ephemeral"},
        }
        if output_config:
            body["output_config"] = output_config
        return body

    def complete(self, system: str, prompt: str, json_schema: dict | None = None, *,
                 model: str | None = None, effort: str | None = None) -> LLMResult:
        body = self.request(system, prompt, json_schema, model, effort)
        t0 = time.perf_counter()
        msg = self.client.messages.create(**body)
        usage = normalize(msg.usage, provider=self.name, model=getattr(msg, "model", None) or body["model"],
                          duration_ms=(time.perf_counter() - t0) * 1000)
        if getattr(msg, "stop_reason", None) == "refusal":
            raise LLMError("the model declined this request (refusal)", usage=usage)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        try:
            data = extract_json(text) if json_schema else None
        except LLMError as exc:
            raise LLMError(str(exc), usage=usage) from exc
        return LLMResult(text=text, data=data, usage=usage)


class FakeProvider(LLMProvider):
    """Calls ``handler(system, prompt, schema)``; the handler returns a dict (JSON) or a string."""

    name = "fake"

    def __init__(self, handler: Callable[[str, str, dict | None], Any], model: str | None = None):
        self.handler = handler
        self.model = model  # set a priced model id to exercise cost accounting in tests
        self.calls: list[dict] = []

    def _usage(self, system: str, prompt: str, text: str) -> dict:
        # deterministic stand-in: ~4 characters per token
        return normalize({"input_tokens": (len(system) + len(prompt)) // 4, "output_tokens": len(text) // 4},
                         provider=self.name, model=self.model, duration_ms=0)

    def complete(self, system: str, prompt: str, json_schema: dict | None = None, *,
                 model: str | None = None, effort: str | None = None) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt, "schema": json_schema, "model": model, "effort": effort})
        out = self.handler(system, prompt, json_schema)
        if isinstance(out, str):
            usage = self._usage(system, prompt, out)
            try:
                data = extract_json(out) if json_schema else None
            except LLMError as exc:
                raise LLMError(str(exc), usage=usage) from exc
            return LLMResult(text=out, data=data, usage=usage)
        import json

        text = json.dumps(out, ensure_ascii=False)
        return LLMResult(text=text, data=out, usage=self._usage(system, prompt, text))


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or env("LLM", "claude_subscription")).lower()
    if name == "claude_subscription":
        return ClaudeSubscriptionProvider()
    if name == "anthropic_api":
        return AnthropicAPIProvider()
    raise LLMError(f"Unknown LLM provider {name!r} (claude_subscription | anthropic_api)")
