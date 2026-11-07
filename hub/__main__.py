"""Agent Hub CLI.

    python -m hub ask "Phân tích FPT 5 năm" [--email] [--agent fin] [--llm fake]
    python -m hub route "message"              # show the routing decision only (no LLM)
    python -m hub eval-router                  # accuracy on tests/golden/router.jsonl (rules only)
    python -m hub cost [--days 7]              # spend per agent vs daily budgets
    python -m hub schedule list | check [--email]
    python -m hub outbox                        # latest emails written locally
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from hubkit.config import env, load_env

SCHEDULES = Path(__file__).with_name("schedules.yaml")


def _llm(name: str | None):
    from hubkit.llm.providers import get_provider
    return get_provider(name)


def cmd_ask(a) -> int:
    from hub.runner import Hub
    hub = Hub(_llm(a.llm), route_with_llm=not a.rules_only)
    out = hub.handle(a.text, channel="cli", agent=a.agent, email=a.email)
    r = out.result
    if out.route:
        print(f"[route] {out.route.agent or 'ask'} via {out.route.method}", file=sys.stderr)
    print(r.question if r.status == "asked_user" else r.text)
    if out.delivery:
        print(f"[email] {'sent' if out.delivery.sent else 'saved'}: {out.delivery.path}"
              + (f" ({out.delivery.error})" if out.delivery.error else ""), file=sys.stderr)
    print(f"[run {r.run_id}] status {r.status}", file=sys.stderr)
    return 0 if r.status in ("ok", "asked_user", "not_ready") else 1


def cmd_route(a) -> int:
    from hub.router import Router
    d = Router(None).route(a.text)
    print(f"agent={d.agent or 'ask'} method={d.method} language={d.language} scores={ {k: v for k, v in d.scores.items() if v} }")
    if d.missing:
        print(d.missing)
    return 0


def cmd_eval_router(a) -> int:
    from hub.evaluate import evaluate
    from hub.router import Router
    ev = evaluate(Router(None))
    print(ev.summary())
    for w in ev.wrong:
        print(f"  WRONG  {w['label']:>6} -> {w['got']:<6} {w['text']}")
    for w in ev.asked:
        print(f"  ASKED  {w['label']:>6}           {w['text']}")
    return 0 if ev.accuracy >= 0.9 else 1


def cmd_cost(a) -> int:
    from hubkit.memory import Budgets, HubStore
    store = HubStore()
    print("Today (Asia/Ho_Chi_Minh):")
    for s in Budgets(store).report():
        print(f"  {s.agent:7} ${s.spent:7.3f} of ${s.limit:.2f}  ({s.share:.0%})")
    if a.days > 1:
        print(f"\nLast {a.days} days:")
        for r in store.cost_report(a.days):
            print(f"  {r['day']} {r['agent']:7} {r['calls']:3} calls  ${r['cost_usd']:.3f}"
                  + (f"  ({r['unpriced']} unpriced)" if r["unpriced"] else ""))
    return 0


def cmd_schedule(a) -> int:
    from hubkit.memory import HubStore
    from hubkit.scheduling import Schedule, missed_runs
    sched = Schedule.load(SCHEDULES)
    if a.action == "list":
        for j in sched.jobs:
            ev = j.vault_glob or j.evidence
            print(f"{j.name:28} {j.agent:7} {j.runtime:6} {j.cron.expr:16} {'on ' if j.enabled else 'off'} evidence={ev}")
        return 0
    vault = Path(env("VAULT_PATH") or os.getenv("VAULT_PATH") or "")
    missed = missed_runs(sched, HubStore(), datetime.now(timezone.utc), vault if vault.name else None)
    if not missed:
        print("No missed runs.")
        return 0
    lines = [m.line() for m in missed]
    print("\n".join(lines))
    if a.email:
        from channels.email_out import Mailer, missed_runs_email
        d = Mailer().deliver(missed_runs_email(lines))
        print(f"[email] {'sent' if d.sent else 'saved'}: {d.path}" + (f" ({d.error})" if d.error else ""))
    return 2


def cmd_outbox(a) -> int:
    from channels.email_out import Mailer
    for p in Mailer().recent(a.limit):
        print(p)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="hub", description="Agent Hub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("text")
    p.add_argument("--agent", choices=["coach", "people", "intel", "fin"])
    p.add_argument("--email", action="store_true", help="deliver the result by email (outbox, SMTP if configured)")
    p.add_argument("--llm", default=None, help="claude_subscription (default) | anthropic_api")
    p.add_argument("--rules-only", action="store_true", help="route without the LLM fallback")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("route")
    p.add_argument("text")
    p.set_defaults(fn=cmd_route)
    sub.add_parser("eval-router").set_defaults(fn=cmd_eval_router)
    p = sub.add_parser("cost")
    p.add_argument("--days", type=int, default=1)
    p.set_defaults(fn=cmd_cost)
    p = sub.add_parser("schedule")
    p.add_argument("action", choices=["list", "check"])
    p.add_argument("--email", action="store_true")
    p.set_defaults(fn=cmd_schedule)
    p = sub.add_parser("outbox")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_outbox)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
