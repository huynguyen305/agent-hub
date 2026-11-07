"""Domain-free Guardrails AI validators shared by every agent.

Finance-only checks (numeric reconciliation against metrics, direction consistency,
risk balance) stay in fin-analyst-agent. These are the ones every agent needs:
prompt injection, PII, certainty language, a disclaimer, and confidential-work
markers (Agent Hub ADR-H05).

Guardrails AI ships hub telemetry that exports traces to a remote endpoint by
default. Prompts must stay local, so OpenTelemetry export and Guardrails metrics
are disabled before the library is imported.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from guardrails.settings import settings  # noqa: E402

settings.rc.enable_metrics = False

from guardrails.validators import (  # noqa: E402
    FailResult, PassResult, ValidationResult, Validator, register_validator,
)

# ---------------------------------------------------------------- input

INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)",
    r"disregard (?:the |your )?(?:system|previous) (?:prompt|instructions)",
    r"you are now\b", r"new instructions?:", r"reveal (?:your|the) (?:system )?prompt",
    r"</?(?:data|skill|system|knowledge|source|note)>",  # fence tags used in prompts
    r"bỏ qua (?:các |mọi )?(?:hướng dẫn|chỉ dẫn)", r"quên (?:các |mọi )?(?:hướng dẫn|quy tắc)",
]


@register_validator(name="hubkit/prompt-injection", data_type="string")
class PromptInjection(Validator):
    """Detects instruction-like text in user input or fetched content. FIX strips the matches."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        hits = [p for p in INJECTION_PATTERNS if re.search(p, value, re.I)]
        if not hits:
            return PassResult()
        cleaned = value
        for p in hits:
            cleaned = re.sub(p, "[removed]", cleaned, flags=re.I)
        return FailResult(error_message=f"Possible prompt injection: {hits}", fix_value=cleaned)


PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "vn_citizen_id": r"\b0\d{11}\b",  # CCCD: 12 digits starting with 0
    "phone": r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{8}(?!\d)",
    "account_number": r"\b\d{9,16}\b",
}


@register_validator(name="hubkit/pii", data_type="string")
class PII(Validator):
    """Redacts personal identifiers before text reaches prompts or logs. FIX replaces them."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        found, cleaned = [], value
        for label, pat in PII_PATTERNS.items():
            if re.search(pat, cleaned):
                found.append(label)
                cleaned = re.sub(pat, f"[{label}]", cleaned)
        if found:
            return FailResult(error_message=f"PII redacted: {found}", fix_value=cleaned)
        return PassResult()


# Labels and patterns that mark employer-confidential material (ADR-H05). Huy adds the
# employer's internal names locally through ``metadata["internal_terms"]``; they are
# never committed to the repo or the vault.
CONFIDENTIAL_PATTERNS = [
    r"\b(?:strictly )?confidential\b", r"\binternal (?:use )?only\b", r"\bdo not (?:share|distribute|forward)\b",
    r"\bproprietary\b", r"\bnot for (?:external )?distribution\b",
    r"\bmật\b", r"\bnội bộ\b", r"\bbảo mật\b", r"\bkhông (?:được )?chia sẻ\b",
    r"\b(?:jdbc|abfss|s3a?|dbfs)://\S+",  # connection strings and storage paths
    r"\b(?:CREATE|ALTER) TABLE\b", r"\bSELECT\b[\s\S]{0,200}\bFROM\b\s+\w+\.\w+",  # internal schemas and queries
]


@register_validator(name="hubkit/confidential-markers", data_type="string")
class ConfidentialMarkers(Validator):
    """Flags text that looks like employer-internal material. No fix: the user generalises it.

    metadata: ``internal_terms`` (optional list of internal system, project or client names).
    """

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        hits = [p for p in CONFIDENTIAL_PATTERNS if re.search(p, value, re.I)]
        for term in metadata.get("internal_terms") or []:
            if term and re.search(rf"\b{re.escape(term)}\b", value, re.I):
                hits.append(f"internal term #{(metadata['internal_terms']).index(term) + 1}")
        if hits:
            return FailResult(error_message=f"Looks like employer-internal material: {hits}")
        return PassResult()


# ---------------------------------------------------------------- output

GUARANTEE_PATTERNS = [
    r"\bguarantee[sd]?\b", r"\brisk[- ]free\b", r"\bcan(?:'|no)t lose\b", r"\bsure (?:to|bet)\b",
    r"\bwill (?:definitely|certainly) (?:rise|go up|double)\b", r"\bno downside\b",
    r"chắc chắn (?:tăng|lãi|có lời)", r"không có rủi ro", r"đảm bảo (?:lợi nhuận|lãi)", r"cam kết lợi nhuận",
]


@register_validator(name="hubkit/no-guarantees", data_type="string")
class NoGuarantees(Validator):
    """Blocks promises of returns or certainty (EN and VI)."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        hits = [p for p in GUARANTEE_PATTERNS if re.search(p, value, re.I)]
        if hits:
            return FailResult(error_message=f"Certainty or guarantee language found: {hits}")
        return PassResult()


@register_validator(name="hubkit/disclaimer", data_type="string")
class Disclaimer(Validator):
    """The agent's disclaimer must be present; FIX appends it.

    metadata: ``disclaimers`` ({language: text}) and ``language``.
    """

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        texts = metadata.get("disclaimers") or {}
        text = texts.get(metadata.get("language", "en")) or texts.get("en")
        if not text or text in value:
            return PassResult()
        return FailResult(error_message="Disclaimer missing", fix_value=f"{value.rstrip()}\n\n{text}")
