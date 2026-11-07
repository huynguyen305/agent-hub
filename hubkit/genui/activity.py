"""Agent activity -> component specs (AgenticGenUI protocol: ``{"type", "props"}``).

A domain-free version of Fin Analyst's ``AgentActivity``. An agent declares its
steps; ``Activity`` consumes LangGraph ``stream_mode="updates"`` events and
returns the specs that ``hubkit.genui.render.Renderer`` draws: a progress bar,
a timeline, an optional checklist (parallel workers), metric cards, banners and
toasts. Agents add domain detail by subclassing and defining ``_on_<node>``
handlers, which call ``banner``, ``toast``, ``card`` and ``item``.

``updates`` events only say a node *finished*; the running step is inferred from
the declared order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def spec(type_: str, **props: Any) -> dict:
    return {"type": type_, "props": props}


@dataclass
class Activity:
    steps: list[tuple[str, str]]  # (node, label) in pipeline order
    checklist_title: str = ""
    done: list[str] = field(default_factory=list)
    items: dict[str, dict] = field(default_factory=dict)  # id -> {label, status, detail, seq}
    banners: list[dict] = field(default_factory=list)
    toasts: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    finished: bool = False
    waiting: bool = False
    error: str | None = None
    seq: int = 0  # bumps on every event; items created in the latest event animate in
    repeat_nodes: tuple[str, ...] = ()  # fan-out nodes that fire many times (not marked done per event)

    # ---------------------------------------------------------------- events
    def update(self, node: str, payload: Any) -> None:
        self.seq += 1
        upd = payload if isinstance(payload, dict) else {}
        if node == "__interrupt__":
            self.waiting = True
            return
        self.usage += upd.get("llm_usage") or []
        if upd.get("error"):
            self.error = upd["error"]
        handler = getattr(self, f"_on_{node}", None)
        if handler:
            handler(upd)
        if node not in self.done and node not in self.repeat_nodes:
            self.done.append(node)

    def finish(self, waiting: bool = False) -> None:
        self.seq += 1
        self.finished, self.waiting = not waiting, waiting

    # ---------------------------------------------------------------- helpers for subclasses
    def banner(self, variant: str, title: str, description: str = "") -> None:
        self.banners.append(spec("InfoBanner", variant=variant, title=title, description=description, seq=self.seq))

    def toast(self, variant: str, message: str) -> None:
        self.toasts.append({"variant": variant, "message": message, "seq": self.seq})

    def card(self, title: str, value: str, description: str = "", **props: Any) -> None:
        self.cards.append(spec("MetricCard", title=title, value=value, description=description, seq=self.seq, **props))

    def item(self, id_: str, label: str, status: str, detail: str = "") -> None:
        """A checklist row: status pending | running | retrying | done | skipped | error."""
        self.items[id_] = {"label": label, "status": status, "detail": detail, "seq": self.seq}

    # ---------------------------------------------------------------- specs
    def specs(self) -> list[dict]:
        running_set = False
        events = []
        for key, label in self.steps:
            if key in self.done:
                status = "completed"
            elif self.error or (self.finished and not self.waiting):
                status = "skipped" if self.finished else "pending"
            elif not running_set:
                status, running_set = ("waiting" if self.waiting else "in-progress"), True
            else:
                status = "pending"
            events.append({"id": key, "title": label, "status": status, "detail": self.step_detail(key)})
        done = sum(1 for e in events if e["status"] == "completed")
        cost = sum(r.get("cost_usd") or 0 for r in self.usage)
        models = sorted({(r.get("model") or "").replace("claude-", "") for r in self.usage if r.get("model")})
        out = [spec("ProgressBar", value=round(100 * done / max(1, len(events))),
                    label="Error" if self.error else "Done" if self.finished else "Waiting for you" if self.waiting else "Working",
                    detail=f"{len(self.usage)} LLM calls · ${cost:.3f} est." + (f" · {', '.join(models)}" if models else ""),
                    variant="error" if self.error else "success" if self.finished else "default"),
               spec("Timeline", events=events)]
        if self.items:
            rows = [{"id": k, **v} for k, v in self.items.items()]
            out.append(spec("ChecklistWithProgress", title=self.checklist_title, items=rows,
                            completed=sum(1 for r in rows if r["status"] in ("done", "skipped", "error"))))
        if self.cards:
            out.append(spec("MetricCardGrid", cards=self.cards))
        out += self.banners
        if self.toasts:
            out.append(spec("ToastStack", toasts=self.toasts[-4:], currentSeq=self.seq))
        if self.error:
            out.append(spec("InfoBanner", variant="error", title="Run stopped", description=self.error[:300], seq=self.seq))
        return out

    def step_detail(self, key: str) -> str:
        rows = [r for r in self.usage if r.get("node") == key]
        if not rows:
            return ""
        cost = sum(r.get("cost_usd") or 0 for r in rows)
        model = (rows[-1].get("model") or "").replace("claude-", "")
        return " · ".join(x for x in (f"{len(rows)} call{'s' if len(rows) > 1 else ''}", model, f"${cost:.3f}") if x)
