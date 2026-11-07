import email
import email.policy
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from channels.email_out import Mailer, compose, feedback_url, md_to_html, missed_runs_email
from hub.contract import Request, Result
from hub.evaluate import evaluate
from hub.router import Router, detect_language, tickers
from hub.runner import Hub
from hubkit.llm import FakeProvider
from hubkit.memory import BudgetExceeded, Budgets, HubStore, local_day
from hubkit.scheduling import Cron, Schedule, ScheduleError, missed_runs

ICT = timezone(timedelta(hours=7))
SCHEDULES = Path(__file__).resolve().parents[1] / "hub" / "schedules.yaml"


@pytest.fixture
def store(tmp_path):
    s = HubStore(tmp_path / "hub.sqlite3")
    yield s
    s.close()


# ---------------------------------------------------------------- store and budgets

def test_runs_usage_and_cost_by_agent(store):
    rid = store.start_run("fin", "analyze", "ui")
    store.finish_run(rid, "ok", [{"node": "lens", "model": "claude-sonnet-5-5", "input_tokens": 1000,
                                  "output_tokens": 200, "cost_usd": 0.012},
                                 {"node": "intake", "input_tokens": 50, "cost_usd": None}], summary="FPT: Hold")
    run = store.run(rid)
    assert run["status"] == "ok" and run["calls"] == 2 and run["cost_usd"] == pytest.approx(0.012)
    assert store.spent("fin") == pytest.approx(0.012) and store.spent("coach") == 0
    rep = store.cost_report(1)
    assert rep[0]["agent"] == "fin" and rep[0]["unpriced"] == 1
    store.add_feedback(rid, -1, channel="email")
    assert store.feedback(rid)[0]["vote"] == -1
    with pytest.raises(ValueError):
        store.add_feedback(rid, 5)
    store.put("coach", "state", {"cards": 3})
    assert store.get("coach", "state") == {"cards": 3} and store.get("coach", "missing", 0) == 0


def test_days_are_counted_in_vietnam_time():
    late_utc = datetime(2026, 11, 7, 18, 30, tzinfo=timezone.utc)  # 01:30 on the 8th in Hanoi
    assert local_day(late_utc) == "2026-11-08"


def test_budget_caps_agent_and_hub(store):
    b = Budgets(store, {"fin": 0.05, "hub": 0.08})
    rid = store.start_run("fin")
    store.finish_run(rid, "ok", [{"cost_usd": 0.05}])
    with pytest.raises(BudgetExceeded, match="fin"):
        b.check("fin")
    b.check("coach")
    rid = store.start_run("coach")
    store.finish_run(rid, "ok", [{"cost_usd": 0.04}])
    with pytest.raises(BudgetExceeded, match="hub"):
        b.check("coach")
    assert {s.agent for s in b.report()} >= {"fin", "coach", "hub"}


# ---------------------------------------------------------------- schedules

def test_cron_matching_and_previous():
    c = Cron("15 7 * * 1-5")  # 07:15 on weekdays
    sat = datetime(2026, 11, 7, 14, 0, tzinfo=ICT)
    assert c.previous(sat) == datetime(2026, 11, 6, 7, 15, tzinfo=ICT)  # Friday
    assert Cron("0 7,19 * * *").previous(datetime(2026, 11, 7, 19, 0, tzinfo=ICT)) == datetime(2026, 11, 7, 7, 0, tzinfo=ICT)
    assert Cron("0 20 * * 0").matches(datetime(2026, 11, 8, 20, 0, tzinfo=ICT))  # Sunday
    assert Cron("0 9 1 * *").previous(sat).day == 1
    with pytest.raises(ScheduleError):
        Cron("61 * * * *")
    with pytest.raises(ScheduleError):
        Cron("* * *")


def test_schedule_file_is_valid_and_people_never_in_cloud(tmp_path):
    sched = Schedule.load(SCHEDULES)
    assert {j.name for j in sched.jobs} >= {"til-research", "quiz-batch", "pre-market-bulletin", "missed-run-check"}
    assert all(j.owner == "routine" for j in sched.jobs if j.runtime == "cloud")
    bad = tmp_path / "s.yaml"
    bad.write_text("jobs:\n  - {name: p, agent: people, runtime: cloud, cron: '0 8 * * *'}\n", encoding="utf-8")
    with pytest.raises(ScheduleError, match="ADR-H12"):
        Schedule.load(bad)


def test_missed_runs_from_run_log_and_vault(store, tmp_path):
    vault = tmp_path / "vault"
    subprocess.run(["git", "init", "-q", "-b", "main", str(vault)], check=True)
    (vault / "Domains/DE/TIL").mkdir(parents=True)
    (vault / "Domains/DE/TIL/a.md").write_text("x")
    env = {"GIT_COMMITTER_DATE": "2026-11-06T23:07:00+07:00", "GIT_AUTHOR_DATE": "2026-11-06T23:07:00+07:00"}
    import os
    subprocess.run(["git", "add", "."], cwd=vault, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "til"], cwd=vault, check=True,
                   env={**os.environ, **env})
    sched_file = tmp_path / "s.yaml"
    sched_file.write_text("""jobs:
  - {name: til, agent: intel, runtime: cloud, owner: routine, cron: "0 7,19 * * *", evidence: {vault: "Domains/*/TIL/*.md"}}
  - {name: local, agent: hub, runtime: laptop, cron: "0 9 * * *", evidence: run_log}
  - {name: mail-only, agent: coach, runtime: cloud, owner: routine, cron: "0 6 * * *", evidence: none}
""", encoding="utf-8")
    sched = Schedule.load(sched_file)
    now = datetime(2026, 11, 7, 14, 40, tzinfo=ICT)
    missed = {m.job.name: m for m in missed_runs(sched, store, now, vault)}
    assert set(missed) == {"til", "local"}  # TIL last committed last night; the 07:00 run left nothing
    assert "last seen 2026-11-06 23:07" in missed["til"].line()
    store.start_run("hub", job="local", at=datetime(2026, 11, 7, 9, 1, tzinfo=ICT))
    store.finish_run(store.recent_runs(1)[0]["id"], "ok")
    assert {m.job.name for m in missed_runs(sched, store, now, vault)} == {"til"}


# ---------------------------------------------------------------- router

def test_router_golden_set_accuracy():
    ev = evaluate(Router())  # rules only, no LLM
    assert ev.accuracy >= 0.9, ev.summary()
    assert len(ev.wrong) <= 3, ev.wrong  # wrong agent is worse than asking
    assert ev.by_label["ask"][0] == ev.by_label["ask"][1]  # vague messages always get a question


def test_router_details():
    r = Router()
    assert r.route("/analyze FPT").method == "command"
    assert detect_language("Phân tích HPG 5 năm") == "vi" and detect_language("Analyze HPG") == "en"
    assert tickers("Explain CFA and NPV for FPT and VIB") == ["FPT", "VIB"]
    d = r.route("Teach me how to give feedback on a colleague's Spark code")
    assert d.asks and "?" in d.missing  # never guesses between People and Coach


def test_router_llm_fallback_and_never_guesses():
    llm = FakeProvider(lambda s, p, schema: {"agent": "intel", "intent": "ask", "language": "en",
                                             "urgency": "today", "missing": ""}, model="claude-haiku-5-5")
    d = Router(llm).route("Recent geopolitical tension in the South China Sea and markets")
    assert d.agent == "intel" and d.method == "llm" and d.usage and "<message>" in llm.calls[0]["prompt"]
    unsure = FakeProvider(lambda s, p, schema: {"agent": "unsure", "intent": "", "language": "en",
                                                "urgency": "whenever", "missing": "Learning or people?"})
    vague = Router(unsure).route("Thoughts on this?")
    assert vague.asks and vague.missing == "Learning or people?" and vague.method == "ask"
    n = len(llm.calls)
    assert Router(llm).route("Teach me how to give feedback on a colleague's Spark code").asks
    assert len(llm.calls) == n  # people vs coach: asks without spending an LLM call


# ---------------------------------------------------------------- email

def test_markdown_to_html():
    out = md_to_html("# FPT: Hold\n\nThesis with **bold** and <script>.\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<h2>FPT: Hold</h2>" in out and "<b>bold</b>" in out and "&lt;script&gt;" in out
    assert "<ul><li>one</li><li>two</li></ul>" in out and "<td>1</td>" in out


def test_compose_and_outbox(tmp_path, monkeypatch):
    for k in ("HUB_SMTP_HOST", "HUB_SMTP_USER", "HUB_SMTP_PASSWORD", "HUB_EMAIL_TO", "FIN_AGENT_SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)
    r = Result(status="ok", agent="fin", run_id="abc123", title="FPT: Hold", text="## Thesis\n\nSolid.",
               citations=["FPT: statements from vnstock"])
    msg = compose(r, language="vi", ui_url="http://localhost:8502")
    assert msg["Subject"] == "[Phân tích tài chính] FPT: Hold" and msg["X-Hub-Run"] == "abc123"
    html_part = msg.get_body(preferencelist=("html",)).get_content()
    assert feedback_url("http://localhost:8502", "abc123", 1).replace("&", "&amp;") in html_part
    assert "Hữu ích" in html_part and "Sources" in msg.get_body(preferencelist=("plain",)).get_content()
    d = Mailer(tmp_path / "outbox").deliver(msg)
    assert not d.sent and d.path.exists()  # no SMTP settings: saved locally, nothing leaves the machine
    back = email.message_from_bytes(d.path.read_bytes(), policy=email.policy.default)
    assert back["Subject"] == msg["Subject"]
    alert = missed_runs_email(["til-research (cloud, intel): expected 07:00"])
    assert "missed" in alert["Subject"] and "til-research" in alert.get_body(preferencelist=("plain",)).get_content()


# ---------------------------------------------------------------- hub end to end

class EchoAgent:
    def __init__(self):
        self.seen: list[Request] = []

    def handle(self, req: Request, on_event=None) -> Result:
        self.seen.append(req)
        return Result(status="ok", agent=req.agent, run_id=req.run_id, text=f"echo: {req.user_text}", title="Echo",
                      llm_usage=[{"node": "x", "cost_usd": 0.01, "input_tokens": 10}])


def make_hub(store, tmp_path, **kw):
    return Hub(None, store=store, mailer=Mailer(tmp_path / "outbox"), route_with_llm=False, **kw)


def test_hub_routes_records_and_emails(store, tmp_path):
    echo = EchoAgent()
    hub = make_hub(store, tmp_path, agents={"coach": echo})
    out = hub.handle("Explain how Delta Lake time travel works, my email is huy@example.com", email=True)
    assert out.result.status == "ok" and out.route.agent == "coach"
    assert "[email]" in echo.seen[0].user_text  # PII stripped before the agent sees it
    assert store.run(out.result.run_id)["status"] == "ok" and store.spent("coach") == pytest.approx(0.01)
    assert out.delivery and out.delivery.path.exists()


def test_hub_blocks_internal_material_and_asks(store, tmp_path):
    hub = make_hub(store, tmp_path, agents={"coach": EchoAgent()}, internal_terms=["Falcon"])
    out = hub.handle("Explain the Falcon pipeline: SELECT * FROM risk.exposures WHERE x")
    assert out.result.status == "asked_user" and "ADR-H05" in out.result.question
    assert hub.handle("Help").result.status == "asked_user"  # vague: one question back


def test_hub_not_ready_budget_and_failures(store, tmp_path):
    class Boom:
        def handle(self, req, on_event=None):
            raise RuntimeError("kaput")

    hub = make_hub(store, tmp_path, agents={"coach": Boom()}, budgets=Budgets(store, {"intel": 0.0}))
    assert hub.handle("What happened in the markets today?").result.status == "budget_stopped"
    assert hub.handle("Prep me for a 1:1 with my director tomorrow").result.status == "not_ready"
    failed = hub.handle("Explain convexity")
    assert failed.result.status == "failed" and "kaput" in failed.result.text
    assert store.run(failed.result.run_id)["status"] == "failed"  # never silent


# ---------------------------------------------------------------- Fin Analyst as agent #0 (UI → email round trip)

fin_agent = pytest.importorskip("fin_agent")


def fin_handler():
    from fin_agent.agent.nodes.intake import heuristic_parse
    from fin_agent.agent.prompts import INTAKE_SCHEMA, LENS_SCHEMA, RECOMMEND_SCHEMA
    line = re.compile(r"^(\S+\.(\w+):\S+) \| (.+?) \| ", re.M)

    def handler(system, prompt, schema):
        if schema is INTAKE_SCHEMA:
            return heuristic_parse(prompt)
        if schema is LENS_SCHEMA:
            lens = re.search(r"^Lens: (\w+)", prompt, re.M).group(1)
            m = [m for m in line.finditer(prompt) if "| reported" not in m.group(0)][-1]
            return {"summary": f"{lens} summary", "findings": [
                {"claim": f"Latest value is {m.group(3)}.", "metric_ids": [m.group(1)], "direction": "stable",
                 "severity": "neutral"}]}
        if schema is RECOMMEND_SCHEMA:
            return {"call": "Hold", "confidence": "high", "thesis": "Solid growth, weakening cash conversion.",
                    "bull_case": ["a"], "base_case": ["b"], "bear_case": ["c"],
                    "key_risks": ["receivables", "leverage", "margins"], "what_would_change_view": ["CFO recovers"]}
        raise AssertionError("unexpected schema")

    return handler


def test_fin_analysis_from_the_hub_arrives_as_email(store, tmp_path, monkeypatch):
    fixtures = Path(fin_agent.__file__).resolve().parents[1] / "tests" / "fixtures"
    monkeypatch.setenv("FIN_AGENT_FIXTURES", str(fixtures))
    monkeypatch.setenv("FIN_AGENT_KB", "off")
    monkeypatch.delenv("VAULT_PATH", raising=False)
    hub = Hub(FakeProvider(fin_handler(), model="claude-sonnet-5-5"), store=store, mailer=Mailer(tmp_path / "outbox"),
              route_with_llm=False)
    events = []
    out = hub.handle("Analyze DEMO, quick view", email=True, on_event=lambda n, u: events.append(n))
    r = out.result
    assert out.route.agent == "fin" and r.status == "ok", r.text
    assert r.title == "DEMO: Hold (high confidence)" and "not licensed" in r.text
    assert "intake" in events and "recommend" in events  # the UI panel gets live graph events
    assert store.spent("fin") > 0 and store.run(r.run_id)["calls"] == len(r.llm_usage) > 0
    msg = email.message_from_bytes(out.delivery.path.read_bytes(), policy=email.policy.default)
    assert msg["Subject"] == "[Fin Analyst] DEMO: Hold (high confidence)"
    assert f"fb={r.run_id}" in msg.get_body(preferencelist=("plain",)).get_content()


def test_fin_clarification_is_asked_and_resumed(store, tmp_path, monkeypatch):
    fixtures = Path(fin_agent.__file__).resolve().parents[1] / "tests" / "fixtures"
    monkeypatch.setenv("FIN_AGENT_FIXTURES", str(fixtures))
    monkeypatch.setenv("FIN_AGENT_KB", "off")
    hub = Hub(FakeProvider(fin_handler()), store=store, mailer=Mailer(tmp_path / "outbox"), route_with_llm=False)
    out = hub.handle("/analyze")
    assert out.result.status == "asked_user" and "Which company" in out.result.question
