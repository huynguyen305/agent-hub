"""Token usage and cost accounting, one record per LLM call.

Providers return a normalized usage dict on ``LLMResult.usage`` (see
``normalize``). Graph nodes append ``usage_entry(...)`` records to the
``llm_usage`` state list; ``summarize`` turns that list into per-run totals.

Cost:
- ``claude_subscription``: the Claude Agent SDK reports ``total_cost_usd``. On a
  subscription this is an API-equivalent estimate, not money billed; usage counts
  against the plan's limits instead. ``cost_source`` is ``"sdk_estimate"``.
- ``anthropic_api``: computed from ``PRICES`` below (``cost_source="price_table"``).
  Unknown models get ``cost_usd=None`` rather than a guess.
"""

from __future__ import annotations

from typing import Any, Iterable

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

# USD per million tokens: (input, output, cache write 5m, cache read).
# First-party Claude API list prices as of 2026-10 (claude-api skill, cached 2026-10-06).
# Cache write = 1.25x input. Haiku 5.5: prompts up to 100K tokens (higher above that);
# its cache-read price is assumed at 0.1x input. Update when prices change.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def price_cost(model: str | None, tokens: dict) -> float | None:
    """Cost in USD from the price table, or None for an unknown model."""
    p = PRICES.get((model or "").strip())
    if p is None:
        return None
    return (
        tokens.get("input_tokens", 0) * p[0] + tokens.get("output_tokens", 0) * p[1]
        + tokens.get("cache_creation_input_tokens", 0) * p[2] + tokens.get("cache_read_input_tokens", 0) * p[3]
    ) / 1_000_000


def normalize(raw: dict | None, *, provider: str, model: str | None = None, duration_ms: float | None = None,
              cost_usd: float | None = None, cost_source: str | None = None) -> dict:
    """Canonical usage dict from a vendor ``usage`` object/dict."""
    raw = raw or {}
    get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
    out: dict[str, Any] = {k: _int(get(k)) for k in TOKEN_FIELDS}
    if cost_usd is None and cost_source is None:
        cost_usd = price_cost(model, out)
        cost_source = "price_table" if cost_usd is not None else None
    out.update(provider=provider, model=model, duration_ms=round(duration_ms) if duration_ms is not None else None,
               cost_usd=float(cost_usd) if cost_usd is not None else None, cost_source=cost_source)
    return out


def usage_entry(usage: dict | None, node: str, label: str | None = None, ok: bool = True) -> dict:
    """One ledger row for the graph state (``llm_usage``)."""
    u = usage or {}
    return {
        "node": node, "label": label, "ok": ok,
        **{k: _int(u.get(k)) for k in TOKEN_FIELDS},
        "total_tokens": sum(_int(u.get(k)) for k in TOKEN_FIELDS),
        "cost_usd": u.get("cost_usd"), "cost_source": u.get("cost_source"),
        "model": u.get("model"), "provider": u.get("provider"), "duration_ms": u.get("duration_ms"),
        "overhead_tokens": _int(u.get("overhead_tokens")),  # input-side tokens we did not send (CLI injection)
    }


def summarize(records: Iterable[dict] | None) -> dict:
    """Run totals. ``cost_usd`` is None when no call reported a cost; ``cost_partial`` flags gaps."""
    rows = list(records or [])
    totals: dict[str, Any] = {k: sum(_int(r.get(k)) for r in rows) for k in TOKEN_FIELDS}
    totals["total_tokens"] = sum(totals[k] for k in TOKEN_FIELDS)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    totals.update(
        calls=len(rows),
        failed_calls=sum(1 for r in rows if not r.get("ok", True)),
        cost_usd=sum(costs) if costs else None,
        cost_partial=bool(costs) and len(costs) < len(rows),
        cost_sources=sorted({r["cost_source"] for r in rows if r.get("cost_source")}),
        models=sorted({r["model"] for r in rows if r.get("model")}),
        duration_ms=sum(_int(r.get("duration_ms")) for r in rows),
        overhead_calls=sum(1 for r in rows if _int(r.get("overhead_tokens"))),
        overhead_tokens=sum(_int(r.get("overhead_tokens")) for r in rows),
        by_node={},
    )
    for r in rows:
        n = totals["by_node"].setdefault(r.get("node") or "?", {"calls": 0, "total_tokens": 0, "cost_usd": 0.0})
        n["calls"] += 1
        n["total_tokens"] += _int(r.get("total_tokens"))
        n["cost_usd"] += r.get("cost_usd") or 0.0
    return totals


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10_000 else f"{n:,}"


def fmt_cost(s: dict) -> str:
    if s.get("cost_usd") is None:
        return "n/a"
    est = " est." if "sdk_estimate" in s.get("cost_sources", []) else ""
    return f"${s['cost_usd']:.4f}{est}" + (" (partial)" if s.get("cost_partial") else "")


def one_line(s: dict) -> str:
    """e.g. ``8 calls · 41.2k tokens (in 35.0k, out 6.2k, cache read 0) · $0.2140 est. · 52.3s``"""
    return (
        f"{s['calls']} calls · {fmt_tokens(s['total_tokens'])} tokens "
        f"(in {fmt_tokens(s['input_tokens'])}, out {fmt_tokens(s['output_tokens'])}, "
        f"cache write {fmt_tokens(s['cache_creation_input_tokens'])}, cache read {fmt_tokens(s['cache_read_input_tokens'])}) "
        f"· {fmt_cost(s)} · {s['duration_ms'] / 1000:.1f}s"
        + (f" · WARNING {s['overhead_calls']} call(s) carried {fmt_tokens(s['overhead_tokens'])} injected tokens"
           if s.get("overhead_calls") else "")
    )
