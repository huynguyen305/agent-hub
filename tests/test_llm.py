import pytest

from hubkit.config import env
from hubkit.llm import FakeProvider, LLMError, ModelRouter
from hubkit.llm.base import strict_schema
from hubkit.llm.providers import AnthropicAPIProvider, ClaudeSubscriptionProvider, expected_tokens, flag_overhead
from hubkit.llm.usage import PRICES, normalize, one_line, price_cost, summarize, usage_entry

COACH = ModelRouter("coach", {"grade": ("claude-haiku-5-5", "low"), "explain": ("claude-sonnet-5-5", "medium")})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith(("HUB_MODEL", "HUB_EFFORT", "FIN_AGENT_MODEL", "FIN_AGENT_EFFORT")):
            monkeypatch.delenv(k)


def test_price_table_cost():
    tokens = {"input_tokens": 1_000_000, "output_tokens": 100_000,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}
    inp, out, _, read = PRICES["claude-opus-5-5"]
    assert price_cost("claude-opus-5-5", tokens) == pytest.approx(inp + out / 10 + read)
    assert price_cost("some-unknown-model", tokens) is None


def test_normalize_and_summarize():
    class U:
        input_tokens, output_tokens = 120, 30
        cache_creation_input_tokens, cache_read_input_tokens = None, 50

    u = normalize(U(), provider="anthropic_api", model="claude-sonnet-5-5")
    assert u["cost_source"] == "price_table" and u["cost_usd"] > 0
    rows = [usage_entry(u, "grade"), usage_entry({"input_tokens": 5}, "explain"), usage_entry(None, "grade", ok=False)]
    s = summarize(rows)
    assert s["calls"] == 3 and s["failed_calls"] == 1 and s["cost_partial"]
    assert s["by_node"]["grade"]["calls"] == 2 and "partial" in one_line(s)


def test_router_defaults_and_overrides(monkeypatch):
    assert COACH.route("grade") == {"model": "claude-haiku-5-5", "effort": "low"}
    monkeypatch.setenv("HUB_MODEL", "claude-sonnet-5-5")
    assert COACH.route("grade")["model"] == "claude-sonnet-5-5"
    monkeypatch.setenv("HUB_MODEL_GRADE", "claude-opus-5-5")
    assert COACH.route("grade")["model"] == "claude-opus-5-5"
    monkeypatch.setenv("HUB_MODEL_COACH_GRADE", "claude-haiku-5-5")
    assert COACH.route("grade")["model"] == "claude-haiku-5-5"  # agent + node wins
    monkeypatch.setenv("HUB_EFFORT_EXPLAIN", "turbo")
    with pytest.raises(ValueError, match="effort"):
        COACH.route("explain")
    monkeypatch.delenv("HUB_EFFORT_EXPLAIN")
    assert [(r["agent"], r["node"]) for r in COACH.table()] == [("coach", "grade"), ("coach", "explain")]


def test_fin_agent_env_names_still_work(monkeypatch):
    monkeypatch.setenv("FIN_AGENT_MODEL_EXPLAIN", "claude-opus-5-5")
    assert COACH.route("explain")["model"] == "claude-opus-5-5"
    monkeypatch.setenv("HUB_MODEL_EXPLAIN", "claude-haiku-5-5")
    assert COACH.route("explain")["model"] == "claude-haiku-5-5"  # HUB_ wins over FIN_AGENT_
    assert env("NOT_SET_ANYWHERE", "d") == "d"


def test_fake_provider_records_calls_and_usage():
    llm = FakeProvider(lambda s, p, schema: {"ok": True}, model="claude-haiku-5-5")
    res = llm.complete("sys", "prompt", {"type": "object"})
    assert res.data == {"ok": True} and len(llm.calls) == 1 and res.usage["input_tokens"] > 0


def test_overhead_detector_and_strict_schema():
    exp = expected_tokens("system", "x" * 9000, {"type": "object"})
    assert "overhead_tokens" not in flag_overhead({"cache_creation_input_tokens": exp}, exp)
    assert flag_overhead({"cache_creation_input_tokens": 169_200}, exp)["overhead_tokens"] > 150_000
    sch = strict_schema({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "object", "properties": {}}}}})
    assert sch["additionalProperties"] is False and sch["properties"]["a"]["items"]["additionalProperties"] is False


def test_api_request_body_uses_structured_outputs_effort_and_caching():
    p = AnthropicAPIProvider.__new__(AnthropicAPIProvider)
    p.model, p.max_tokens = None, 4096
    body = p.request("sys", "prompt", {"type": "object", "properties": {"x": {"type": "string"}}}, "claude-haiku-5-5", "low")
    assert body["model"] == "claude-haiku-5-5" and body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_subscription_sdk_errors_become_llm_errors(monkeypatch):
    import claude_agent_sdk

    class ResultError(Exception):
        pass

    async def boom(**kwargs):
        raise ResultError("You've hit your 5-hour limit · resets 8:30am")
        yield  # pragma: no cover

    monkeypatch.setattr(claude_agent_sdk, "query", boom)
    with pytest.raises(LLMError, match="usage limit reached"):
        ClaudeSubscriptionProvider().complete("s", "p", {"type": "object"}, model="claude-haiku-5-5")
