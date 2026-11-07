"""Fin Analyst as agent #0: wraps ``fin_agent``'s LangGraph behind the hub contract.

The hub never writes to the vault here: ``sync`` stays off, so a run returns the report as
text (for the UI and email). Vault reports still go through Fin Analyst's own review and
PR flow when run from its UI or CLI.

When Fin Analyst needs a clarification (for example an unknown ticker), the result has
``status="asked_user"`` and the question; ``FinAgent.resume(run_id, answer)`` continues
the same run. Paused runs live in this process only.
"""

from __future__ import annotations

from typing import Callable

from hub.contract import Request, Result

EventFn = Callable[[str, dict], None]


def available() -> bool:
    try:
        import fin_agent  # noqa: F401
        return True
    except ImportError:
        return False


class FinAgent:
    name = "fin"

    def __init__(self, llm):
        from fin_agent.agent.graph import build_graph
        self.app = build_graph(llm)
        self.paused: set[str] = set()

    def handle(self, req: Request, on_event: EventFn | None = None) -> Result:
        text = req.user_text
        if text.lower().startswith(("/analyze", "/ask")):
            text = text.split(maxsplit=1)[1] if " " in text else ""
        if not text.strip():
            return Result(status="asked_user", agent=self.name, run_id=req.run_id,
                          question="Which company? For example: /analyze FPT 5 years")
        return self._run({"user_input": text, "sync": False, "run_id": req.run_id}, req.run_id, on_event)

    def resume(self, run_id: str, answer: str, on_event: EventFn | None = None) -> Result:
        from langgraph.types import Command
        if run_id not in self.paused:
            return Result(status="failed", agent=self.name, run_id=run_id, text="This run is no longer waiting.")
        self.paused.discard(run_id)
        return self._run(Command(resume=answer), run_id, on_event)

    def _run(self, payload, run_id: str, on_event: EventFn | None) -> Result:
        cfg = {"configurable": {"thread_id": run_id}}
        interrupt = None
        for chunk in self.app.stream(payload, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupt = update[0].value if update else {}
                if on_event:
                    on_event(node, update if isinstance(update, dict) else {})
        state = self.app.get_state(cfg).values
        usage = state.get("llm_usage") or []
        if interrupt is not None:
            self.paused.add(run_id)
            return Result(status="asked_user", agent=self.name, run_id=run_id, llm_usage=usage,
                          question=interrupt.get("question", "Could you clarify?"),
                          guardrail_log=state.get("guardrail_log") or [])
        return to_result(state, run_id)


def title_of(state: dict) -> str:
    req = state.get("request") or {}
    tickers = ", ".join(req.get("tickers") or [])
    if state.get("answer"):
        return f"{tickers}: answer"
    rec = state.get("recommendation") or {}
    if rec.get("call"):
        return f"{tickers}: {rec['call']} ({rec.get('confidence', '?')} confidence)"
    return tickers or "Fin Analyst"


def to_result(state: dict, run_id: str) -> Result:
    from fin_agent.cli import render
    usage = state.get("llm_usage") or []
    if state.get("error"):
        return Result(status="failed", agent="fin", run_id=run_id, text=f"Fin Analyst stopped: {state['error']}",
                      llm_usage=usage, guardrail_log=state.get("guardrail_log") or [], title="Run stopped")
    cites = []
    for ticker, ds in (state.get("datasets") or {}).items():
        meta = ds.get("meta") or {}
        cites.append(f"{ticker}: statements from {meta.get('source', '?')}"
                     + (f", prices to {meta['market_data']['price_date']}" if (meta.get("market_data") or {}).get("price_date") else ""))
    return Result(status="ok", agent="fin", run_id=run_id, text=render(state), title=title_of(state),
                  citations=cites, llm_usage=usage, guardrail_log=state.get("guardrail_log") or [])
