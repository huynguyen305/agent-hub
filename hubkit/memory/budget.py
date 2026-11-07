"""Hard daily budgets per agent (Master Plan §5.5), enforced from the usage ledger.

Costs are the providers' figures: on the Claude subscription they are the SDK's
API-equivalent estimate (usage counts against plan limits, nothing is billed), on the
API they come from the price table. A run is refused before it starts once an agent
has spent its day's budget; the hub-wide cap applies on top.
"""

from __future__ import annotations

from dataclasses import dataclass

from hubkit.memory.store import HubStore

# USD per day (Master Plan §5.5). "hub" caps the total across agents.
DEFAULT_BUDGETS: dict[str, float] = {
    "intel": 0.40, "coach": 0.40, "people": 0.50, "fin": 1.00, "router": 0.05, "hub": 2.30,
}


class BudgetExceeded(Exception):
    def __init__(self, agent: str, spent: float, limit: float):
        super().__init__(f"{agent} has spent ${spent:.2f} of its ${limit:.2f} daily budget")
        self.agent, self.spent, self.limit = agent, spent, limit


@dataclass
class BudgetStatus:
    agent: str
    spent: float
    limit: float

    @property
    def left(self) -> float:
        return max(0.0, self.limit - self.spent)

    @property
    def share(self) -> float:
        return self.spent / self.limit if self.limit else 0.0


class Budgets:
    def __init__(self, store: HubStore, limits: dict[str, float] | None = None):
        self.store = store
        self.limits = {**DEFAULT_BUDGETS, **(limits or {})}

    def status(self, agent: str, day: str | None = None) -> BudgetStatus:
        limit = self.limits.get(agent, self.limits["hub"])
        return BudgetStatus(agent, self.store.spent(agent, day), limit)

    def check(self, agent: str) -> None:
        """Raise ``BudgetExceeded`` if the agent or the whole hub is out of today's budget."""
        s = self.status(agent)
        if s.spent >= s.limit:
            raise BudgetExceeded(agent, s.spent, s.limit)
        total = self.store.spent()
        if total >= self.limits["hub"]:
            raise BudgetExceeded("hub", total, self.limits["hub"])

    def report(self, day: str | None = None) -> list[BudgetStatus]:
        agents = [a for a in self.limits if a != "hub"]
        return [self.status(a, day) for a in agents] + [BudgetStatus("hub", self.store.spent(day=day), self.limits["hub"])]
