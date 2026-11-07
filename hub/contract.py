"""The shared contract every agent speaks (Agent Specs §5).

Request  { user_text, agent, intent, language, channel, profile_ref, budget }
Result   { text, items[], citations[], proposals[], genui_specs[], guardrail_log[], llm_usage[], status }
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

AGENTS = ("coach", "people", "intel", "fin")
AGENT_NAMES = {
    "coach": ("Knowledge Coach", "Huấn luyện kiến thức"),
    "people": ("People Advisor", "Cố vấn làm việc với con người"),
    "intel": ("World Intelligence", "Tình hình thế giới"),
    "fin": ("Fin Analyst", "Phân tích tài chính"),
}
STATUSES = ("ok", "asked_user", "escalated", "budget_stopped", "failed", "not_ready")


@dataclass
class Request:
    user_text: str
    agent: str
    intent: str = ""
    language: str = "en"
    channel: str = "ui"  # ui | email | cli | schedule
    profile_ref: str = "Projects/AgentHub/Design/Work Profile_Agent Hub"
    budget: float | None = None  # USD for this request; None = the agent's daily budget decides
    run_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    status: str
    text: str = ""
    items: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)  # vault PRs, after human review
    genui_specs: list[dict] = field(default_factory=list)
    guardrail_log: list[dict] = field(default_factory=list)
    llm_usage: list[dict] = field(default_factory=list)
    question: str = ""  # set when status == asked_user
    title: str = ""
    run_id: str = ""
    agent: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")

    def to_dict(self) -> dict:
        return asdict(self)
