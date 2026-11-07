"""Guards composed from the shared validators, one function per use.

Every function returns guardrail-log events as ``{"layer", "check", "action", "detail"}``
so the UI, the run log and the vault show exactly what fired.
"""

from __future__ import annotations

from guardrails import Guard, OnFailAction

from hubkit.guards.validators import PII, ConfidentialMarkers, Disclaimer, NoGuarantees, PromptInjection

_input_guard = Guard().use(PromptInjection(on_fail=OnFailAction.FIX), PII(on_fail=OnFailAction.FIX))
_content_guard = Guard().use(PromptInjection(on_fail=OnFailAction.FIX))
_confidential_guard = Guard().use(ConfidentialMarkers(on_fail=OnFailAction.NOOP))
_text_guard = Guard().use(NoGuarantees(on_fail=OnFailAction.NOOP))
_disclaimer_guard = Guard().use(Disclaimer(on_fail=OnFailAction.FIX))

CHECK_NAMES = {
    "NoGuarantees": "no_guarantees", "PromptInjection": "prompt_injection", "PII": "pii",
    "Disclaimer": "disclaimer", "ConfidentialMarkers": "confidential",
}


def failures(outcome) -> list[tuple[str, str]]:
    """(check, reason) for each failed validator in a Guardrails outcome."""
    return [
        (CHECK_NAMES.get(s.validator_name, s.validator_name), s.failure_reason or "")
        for s in (outcome.validation_summaries or []) if s.validator_status == "fail"
    ]


def guard_input(text: str, layer: str = "input") -> tuple[str, list[dict]]:
    """Clean user text (injection, PII). Returns cleaned text and log events."""
    out = _input_guard.validate(text)
    events = [{"layer": layer, "check": c, "action": "fix", "detail": r[:200]} for c, r in failures(out)]
    return (out.validated_output if out.validated_output is not None else text), events


def guard_fetched_content(text: str) -> tuple[str, list[dict]]:
    """For TIL notes, articles, PDFs and other fetched text: data only, injection stripped."""
    out = _content_guard.validate(text)
    events = [{"layer": "input", "check": c, "action": "strip", "detail": r[:200]} for c, r in failures(out)]
    return (out.validated_output if out.validated_output is not None else text), events


def check_confidential(text: str, internal_terms: list[str] | None = None) -> list[dict]:
    """ADR-H05: events when text looks employer-internal. The caller stops and asks the user to generalise."""
    out = _confidential_guard.validate(text, metadata={"internal_terms": internal_terms or []})
    return [{"layer": "input", "check": c, "action": "block", "detail": r[:200]} for c, r in failures(out)]


def check_language(text: str) -> list[str]:
    """Certainty or guarantee language in an output (empty list = passes)."""
    return [r for _, r in failures(_text_guard.validate(text))]


def ensure_disclaimer(text: str, disclaimers: dict[str, str], language: str = "en") -> str:
    """Append the agent's disclaimer for ``language`` (English if missing) unless already present."""
    if not text.strip():
        return disclaimers.get(language) or disclaimers.get("en", "")
    out = _disclaimer_guard.validate(text, metadata={"disclaimers": disclaimers, "language": language})
    return out.validated_output if out.validated_output is not None else text
