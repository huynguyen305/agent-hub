"""Agent Hub UI (Phase 1 shell): ``streamlit run app/ui.py`` → http://localhost:8502.

One place to ask any agent, see what it is doing, read results, and check cost,
schedules and the email outbox. Feedback links in emails land here (``?fb=<run>&v=up``).
"""

from __future__ import annotations

import email
import email.policy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hubkit.config import load_env  # noqa: E402
from hubkit.genui import Renderer  # noqa: E402

from channels.email_out import Mailer  # noqa: E402
from hub.__main__ import SCHEDULES  # noqa: E402
from hub.contract import AGENT_NAMES  # noqa: E402
from hub.runner import Hub, Outcome  # noqa: E402

load_env()
st.set_page_config(page_title="Agent Hub", page_icon="🧭", layout="wide")


@st.cache_resource
def get_hub() -> Hub:
    from hubkit.llm.providers import get_provider
    return Hub(get_provider())


def ui_theme() -> str | None:
    try:
        return st.context.theme.type
    except Exception:  # noqa: BLE001
        return None


hub = get_hub()
ss = st.session_state
ss.setdefault("messages", [])  # (role, payload)
ss.setdefault("pending", None)  # run_id waiting for an answer

# ---------------------------------------------------------------- feedback from email links
qp = st.query_params
if "fb" in qp and qp.get("v") in ("up", "down"):
    run = hub.store.run(qp["fb"])
    if run:
        hub.store.add_feedback(qp["fb"], 1 if qp["v"] == "up" else -1, channel="email")
        st.toast(f"Feedback recorded for run {qp['fb']}: {'👍' if qp['v'] == 'up' else '👎'}")
    else:
        st.toast("Unknown run in the feedback link.")
    st.query_params.clear()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Agent Hub")
    st.caption("Coach · People · Intelligence · Fin Analyst")
    st.subheader("Today's budget")
    for s in hub.budgets.report():
        label = "Hub total" if s.agent == "hub" else AGENT_NAMES.get(s.agent, (s.agent.title(),))[0]
        st.progress(min(1.0, s.share), text=f"{label}: \\${s.spent:.3f} / \\${s.limit:.2f}")  # escaped $: not LaTeX
    st.caption("Costs are API-equivalent estimates; on the subscription they count against plan limits.")
    st.subheader("Email")
    st.caption("SMTP configured: emails are sent." if Mailer.smtp_configured()
               else "No SMTP settings: emails are saved to the outbox only (see README).")
    send_email = st.toggle("Email me the result", value=False)
    st.caption("Theme: ⋮ menu → Settings → Light, Dark or System.")


# ---------------------------------------------------------------- helpers
def show_result(out: Outcome) -> None:
    r = out.result
    if out.route and out.route.agent:
        st.caption(f"Routed to {AGENT_NAMES[out.route.agent][0]} ({out.route.method}) · run {r.run_id} · {r.status}")
    if r.status == "asked_user":
        st.info(r.question)
    elif r.status in ("failed", "budget_stopped"):
        st.error(r.text)
    elif r.status == "not_ready":
        st.warning(r.text)
    else:
        if r.title:
            st.subheader(r.title)
        st.markdown(r.text)
        if r.citations:
            st.caption("Sources: " + " · ".join(r.citations))
    if out.delivery:
        st.caption(("📧 Sent" if out.delivery.sent else "📥 Saved to outbox") + f": {out.delivery.path.name}"
                   + (f" ({out.delivery.error})" if out.delivery.error else ""))
    if r.run_id and r.status == "ok":
        c1, c2, _ = st.columns([1, 1, 8])
        if c1.button("👍", key=f"up-{r.run_id}"):
            hub.store.add_feedback(r.run_id, 1)
            st.toast("Thanks!")
        if c2.button("👎", key=f"down-{r.run_id}"):
            hub.store.add_feedback(r.run_id, -1)
            st.toast("Noted.")


def run_with_panel(fn) -> Outcome:
    """Run a hub call, drawing Fin Analyst's activity panel live when it streams events."""
    panel = st.empty()
    renderer = Renderer(theme=ui_theme())
    activity = None
    try:
        from fin_agent.ui.genui import AgentActivity
        activity = AgentActivity()
    except ImportError:
        pass

    def on_event(node: str, update: dict) -> None:
        if activity is not None:
            activity.update(node, update)
            panel.html(renderer.render(activity.specs(), activity.seq))

    with st.spinner("Working…"):
        out = fn(on_event)
    if activity is not None and activity.seq:
        activity.finish(waiting=out.result.status == "asked_user")
        panel.html(renderer.render(activity.specs(), activity.seq))
    else:
        panel.empty()
    return out


# ---------------------------------------------------------------- tabs
t_ask, t_agents, t_cost, t_sched, t_outbox = st.tabs(["Ask", "Agents", "Cost", "Schedules", "Outbox"])

with t_ask:
    st.caption("Ask in English or Vietnamese. Commands: /analyze, /ask, /quiz, /explain, /prep, /debrief, "
               "/draft, /brief, /watch. Internal work material is blocked (ADR-H05): describe it in general terms.")
    for role, payload in ss.messages:
        with st.chat_message(role):
            if role == "user":
                st.markdown(payload)
            else:
                show_result(payload)
    prompt = st.chat_input("Hỏi gì cũng được · Ask anything")
    if prompt:
        ss.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if ss.pending:
                rid, ss.pending = ss.pending, None
                out = run_with_panel(lambda ev: hub.resume(rid, prompt, email=send_email, on_event=ev))
            else:
                out = run_with_panel(lambda ev: hub.handle(prompt, email=send_email, on_event=ev))
            if out.result.status == "asked_user" and out.result.agent not in ("hub", ""):
                ss.pending = out.result.run_id  # the agent asked: the next message answers it
            show_result(out)
        ss.messages.append(("assistant", out))

with t_agents:
    cols = st.columns(4)
    status = {"fin": "Ready (agent #0)", "coach": "Phase 2", "intel": "Phase 3", "people": "Phase 4"}
    for col, key in zip(cols, ["coach", "people", "intel", "fin"]):
        with col:
            st.markdown(f"**{AGENT_NAMES[key][0]}**  \n{AGENT_NAMES[key][1]}")
            st.caption(status[key])
            runs = hub.store.recent_runs(5, agent=key)
            if runs:
                st.dataframe(pd.DataFrame(runs)[["started_at", "status", "summary", "cost_usd"]],
                             hide_index=True, use_container_width=True)

with t_cost:
    days = st.slider("Days", 1, 30, 7)
    rows = hub.store.cost_report(days)
    if rows:
        df = pd.DataFrame(rows)
        st.bar_chart(df.pivot_table(index="day", columns="agent", values="cost_usd", aggfunc="sum").fillna(0))
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No LLM calls recorded yet.")
    fb = hub.store.feedback()
    if fb:
        st.subheader("Feedback")
        st.dataframe(pd.DataFrame(fb), hide_index=True, use_container_width=True)

with t_sched:
    from hubkit.scheduling import Schedule, missed_runs
    from datetime import datetime, timezone
    import os

    sched = Schedule.load(SCHEDULES)
    st.dataframe(pd.DataFrame([{"job": j.name, "agent": j.agent, "runtime": j.runtime, "cron (ICT)": j.cron.expr,
                                "evidence": j.vault_glob or j.evidence, "enabled": j.enabled, "notes": j.notes}
                               for j in sched.jobs]), hide_index=True, use_container_width=True)
    vault = os.getenv("VAULT_PATH")
    if st.button("Check for missed runs", disabled=not vault, help=None if vault else "Set VAULT_PATH in .env"):
        missed = missed_runs(sched, hub.store, datetime.now(timezone.utc), Path(vault))
        if missed:
            for m in missed:
                st.warning(m.line())
        else:
            st.success("Every job with evidence ran on schedule.")

with t_outbox:
    files = hub.mailer.recent(30)
    if not files:
        st.info("No emails yet. Turn on “Email me the result” in the sidebar.")
    else:
        pick = st.selectbox("Email", files, format_func=lambda p: p.name)
        msg = email.message_from_bytes(pick.read_bytes(), policy=email.policy.default)
        st.markdown(f"**{msg['Subject']}**  \nTo: {msg['To'] or '(not set)'} · {msg['Date']}")
        body = msg.get_body(preferencelist=("html",))
        if body is not None:
            st.html(body.get_content())
