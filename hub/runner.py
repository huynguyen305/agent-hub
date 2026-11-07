"""The hub: one entry point for every request (UI, CLI, schedule).

handle(text) →
  1. input guards: prompt injection and PII stripped; confidential-work markers block the
     request and ask Huy to generalise it (ADR-H05)
  2. route (rules first, Haiku only when unsure)
  3. budget check for the chosen agent (hard daily caps from the ledger)
  4. the agent runs and returns a ``Result`` (the shared contract)
  5. the run and every LLM call are written to the ledger; failures are never silent
  6. optional delivery by email (default channel, ADR-H07)

Agents that are not built yet answer ``not_ready`` with the phase that brings them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hubkit.guards import check_confidential, guard_input
from hubkit.llm.base import LLMError, LLMProvider
from hubkit.memory import BudgetExceeded, Budgets, HubStore

from channels.email_out import Delivery, Mailer, compose
from hub.contract import AGENT_NAMES, Request, Result
from hub.router import RouteDecision, Router, detect_language

NOT_READY = {
    "coach": ("Phase 2", "Giai đoạn 2"), "intel": ("Phase 3", "Giai đoạn 3"), "people": ("Phase 4", "Giai đoạn 4"),
}
CONFIDENTIAL_MSG = {
    "en": "This looks like internal work material, which the hub must not send to a personal-plan LLM (ADR-H05). "
          "Please describe it in general terms (no internal names, systems, schemas or figures) and ask again.",
    "vi": "Nội dung này có vẻ là tài liệu nội bộ, hub không được gửi tới LLM của gói cá nhân (ADR-H05). "
          "Bạn hãy mô tả lại một cách tổng quát (không tên nội bộ, hệ thống, schema hay số liệu) rồi hỏi lại.",
}


@dataclass
class Outcome:
    result: Result
    route: RouteDecision | None = None
    delivery: Delivery | None = None
    guard_events: list[dict] = field(default_factory=list)


class Hub:
    def __init__(self, llm: LLMProvider | None, store: HubStore | None = None, mailer: Mailer | None = None,
                 budgets: Budgets | None = None, internal_terms: list[str] | None = None,
                 agents: dict | None = None, route_with_llm: bool = True):
        self.llm = llm
        self.store = store or HubStore()
        self.mailer = mailer or Mailer()
        self.budgets = budgets or Budgets(self.store)
        self.router = Router(llm if route_with_llm else None)
        self.internal_terms = internal_terms or []
        self._agents = agents if agents is not None else {}

    # ---------------------------------------------------------------- agents
    def agent(self, name: str):
        if name not in self._agents and name == "fin":
            from agents.fin import FinAgent, available
            if available() and self.llm is not None:
                self._agents["fin"] = FinAgent(self.llm)
        return self._agents.get(name)

    # ---------------------------------------------------------------- requests
    def handle(self, text: str, *, channel: str = "ui", agent: str | None = None, email: bool = False,
               on_event: Callable[[str, dict], None] | None = None) -> Outcome:
        cleaned, events = guard_input(text)
        lang = detect_language(text)
        blocked = check_confidential(text, self.internal_terms)
        if blocked:
            rid = self.store.start_run("hub", "blocked", channel)
            self.store.finish_run(rid, "asked_user", summary="confidential markers")
            return Outcome(Result(status="asked_user", agent="hub", run_id=rid, question=CONFIDENTIAL_MSG[lang],
                                  guardrail_log=events + blocked), guard_events=events + blocked)

        if agent:
            decision = RouteDecision(agent, "", lang, method="explicit")
        else:
            try:
                self.budgets.check("router")
                decision = self.router.route(cleaned)
            except BudgetExceeded:
                decision = Router(None).route(cleaned)  # rules only once the router's budget is spent
        if decision.usage:
            rid = self.store.start_run("router", "route", channel)
            self.store.finish_run(rid, "ok", decision.usage, summary=f"-> {decision.agent or 'ask'}")
        if decision.asks:
            rid = self.store.start_run("hub", "clarify", channel)
            self.store.finish_run(rid, "asked_user", summary=decision.missing)
            return Outcome(Result(status="asked_user", agent="hub", run_id=rid, question=decision.missing),
                           decision, guard_events=events)
        return self._dispatch(cleaned, decision, channel, email, on_event, events)

    def resume(self, run_id: str, answer: str, *, email: bool = False, on_event=None) -> Outcome:
        run = self.store.run(run_id)
        impl = self.agent(run["agent"]) if run else None
        if not impl or not hasattr(impl, "resume"):
            return Outcome(Result(status="failed", agent="hub", run_id=run_id, text="Nothing to resume."))
        result = self._guarded(lambda: impl.resume(run_id, answer, on_event), run["agent"], run_id)
        self._finish(run_id, result)
        return Outcome(result, delivery=self._deliver(result, email))

    def _dispatch(self, text: str, decision: RouteDecision, channel: str, email: bool, on_event, events) -> Outcome:
        name = decision.agent
        rid = self.store.start_run(name, decision.intent, channel)
        try:
            self.budgets.check(name)
        except BudgetExceeded as e:
            result = Result(status="budget_stopped", agent=name, run_id=rid, text=str(e), title="Daily budget reached")
            self._finish(rid, result)
            return Outcome(result, decision, self._deliver(result, email), events)
        impl = self.agent(name)
        if impl is None:
            phase = NOT_READY.get(name, ("a later phase", "giai đoạn sau"))
            agent_name = AGENT_NAMES[name][1 if decision.language == "vi" else 0]
            msg = (f"{agent_name} chưa sẵn sàng; dự kiến có ở {phase[1]}." if decision.language == "vi"
                   else f"The {agent_name} isn't built yet; it arrives in {phase[0]}.")
            if name == "fin":
                msg = "Fin Analyst is not installed in this environment (pip install fin-analyst-agent)."
            result = Result(status="not_ready", agent=name, run_id=rid, text=msg, title="Not available yet")
        else:
            req = Request(user_text=text, agent=name, intent=decision.intent, language=decision.language,
                          channel=channel, run_id=rid)
            result = self._guarded(lambda: impl.handle(req, on_event), name, rid)
        result.guardrail_log = events + result.guardrail_log
        self._finish(rid, result)
        return Outcome(result, decision, self._deliver(result, email), events)

    @staticmethod
    def _guarded(fn, agent: str, run_id: str) -> Result:
        try:
            return fn()
        except LLMError as e:
            return Result(status="failed", agent=agent, run_id=run_id, text=f"LLM error: {e}",
                          llm_usage=[u for u in [getattr(e, "usage", None)] if u], title="Run failed")
        except Exception as e:  # noqa: BLE001 - the run log must record every failure
            return Result(status="failed", agent=agent, run_id=run_id, text=f"{type(e).__name__}: {e}"[:500],
                          title="Run failed")

    def _finish(self, run_id: str, result: Result) -> None:
        self.store.finish_run(run_id, result.status, result.llm_usage, summary=result.title or result.question,
                              error=result.text if result.status == "failed" else None)

    def _deliver(self, result: Result, email: bool) -> Delivery | None:
        if not email or result.status == "asked_user":
            return None
        return self.mailer.deliver(compose(result))

    # ---------------------------------------------------------------- reporting
    def cost_lines(self) -> list[str]:
        return [f"{s.agent:7} ${s.spent:6.3f} of ${s.limit:.2f} ({s.share:.0%})" for s in self.budgets.report()]

