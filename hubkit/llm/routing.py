"""Model routing: the cheapest model that does each node's job well.

Each agent declares its own route table, ``{node: (model, effort)}``, and gets a
``ModelRouter``. Environment overrides, most specific first:

- ``HUB_MODEL_<AGENT>_<NODE>`` / ``HUB_EFFORT_<AGENT>_<NODE>`` (one node of one agent)
- ``HUB_MODEL_<NODE>`` / ``HUB_EFFORT_<NODE>`` (that node name in any agent)
- ``HUB_MODEL`` (every node not overridden)

``FIN_AGENT_*`` works in place of ``HUB_*`` (see ``hubkit.config.env``).
"""

from __future__ import annotations

from hubkit.config import env

EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ModelRouter:
    def __init__(self, agent: str, routes: dict[str, tuple[str, str]]):
        self.agent = agent
        self.routes = dict(routes)

    def route(self, node: str) -> dict:
        """``{"model": ..., "effort": ...}`` for a node, after environment overrides."""
        default_model, default_effort = self.routes.get(node, (None, None))
        a, n = self.agent.upper(), node.upper()
        model = env(f"MODEL_{a}_{n}") or env(f"MODEL_{n}") or env("MODEL") or default_model
        effort = env(f"EFFORT_{a}_{n}") or env(f"EFFORT_{n}") or default_effort
        if effort and effort not in EFFORTS:
            raise ValueError(f"effort {effort!r} for {self.agent}.{node}: use one of {', '.join(EFFORTS)}")
        return {"model": model, "effort": effort}

    def table(self) -> list[dict]:
        """Resolved routing for display (UI, CLI)."""
        return [{"agent": self.agent, "node": n, **self.route(n)} for n in self.routes]
