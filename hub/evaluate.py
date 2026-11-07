"""Router accuracy on the labelled golden set (``tests/golden/router.jsonl``).

A message labelled with an agent is correct when routed to that agent; one labelled
``ask`` is correct when the router asks instead of guessing. Results separate wrong
routes (harmful: the wrong agent answers) from unnecessary questions (only friction).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hub.router import Router

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "router.jsonl"


@dataclass
class RouterEval:
    total: int = 0
    correct: int = 0
    wrong: list[dict] = field(default_factory=list)  # routed to the wrong agent
    asked: list[dict] = field(default_factory=list)  # asked although the label is an agent
    by_label: dict[str, list[int]] = field(default_factory=dict)  # label -> [correct, total]
    methods: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        per = ", ".join(f"{k} {c}/{t}" for k, (c, t) in sorted(self.by_label.items()))
        return (f"accuracy {self.accuracy:.1%} ({self.correct}/{self.total}); wrong agent {len(self.wrong)}, "
                f"asked unnecessarily {len(self.asked)}; {per}; methods {self.methods}")


def load(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(router: Router, rows: list[dict] | None = None) -> RouterEval:
    ev = RouterEval()
    for row in rows or load():
        d = router.route(row["text"])
        got = d.agent or "ask"
        ok = got == row["label"]
        ev.total += 1
        ev.correct += ok
        ev.methods[d.method] = ev.methods.get(d.method, 0) + 1
        c, t = ev.by_label.get(row["label"], [0, 0])
        ev.by_label[row["label"]] = [c + ok, t + 1]
        if not ok:
            rec = {**row, "got": got, "scores": {k: v for k, v in d.scores.items() if v}}
            (ev.asked if got == "ask" else ev.wrong).append(rec)
    return ev
