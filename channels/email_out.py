"""Email, the hub's default channel (ADR-H07).

``compose`` turns a ``Result`` into a multipart email (plain text plus simple HTML) with
👍/👎 links that open the hub UI and record feedback in the ledger. ``Mailer.deliver``
always writes a copy to the local outbox (``data/outbox/*.eml``) and sends it by SMTP
when configured:

    HUB_EMAIL_TO=<your address>          # recipient (required to send)
    HUB_SMTP_HOST=smtp.gmail.com
    HUB_SMTP_PORT=587
    HUB_SMTP_USER=<account>
    HUB_SMTP_PASSWORD=<app password>     # Gmail: an app password, never the account password
    HUB_UI_URL=http://localhost:8502     # where the feedback links point

Without SMTP settings nothing leaves the machine: emails stay in the outbox, which the
hub UI lists. Credentials are read from the environment (``.env``) only.
"""

from __future__ import annotations

import html
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from urllib.parse import urlencode

from hubkit.config import env
from hubkit.memory.store import TZ

from hub.contract import AGENT_NAMES, Result

FOOTER = {
    "en": ("Was this useful?", "Useful", "Not useful", "Open in the hub"),
    "vi": ("Nội dung này có hữu ích không?", "Hữu ích", "Chưa hữu ích", "Mở trong hub"),
}


def feedback_url(base: str, run_id: str, vote: int) -> str:
    return f"{base.rstrip('/')}/?{urlencode({'fb': run_id, 'v': 'up' if vote > 0 else 'down'})}"


def _inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)


def md_to_html(md: str) -> str:
    """Enough Markdown for agent output: headings, bullets, bold, code, links, pipe tables, paragraphs."""
    out, para, items, table = [], [], [], []

    def flush():
        if para:
            out.append("<p>" + "<br>".join(_inline(p) for p in para) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|?[\s:|-]+\|?", r)]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            if cells:
                head = "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
                body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
                out.append(f"<table><tr>{head}</tr>{body}</table>")
            table.clear()

    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
        elif m := re.match(r"^(#{1,4})\s+(.*)", s):
            flush()
            level = min(len(m.group(1)) + 1, 4)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif re.match(r"^\s*[-*]\s+", s):
            if para or table:
                flush()
            items.append(re.sub(r"^\s*[-*]\s+", "", s))
        elif s.lstrip().startswith("|"):
            if para or items:
                flush()
            table.append(s.strip())
        else:
            if items or table:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


STYLE = ("body{font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#14151A;max-width:720px;margin:auto;line-height:1.5}"
         "h2,h3,h4{color:#4C2FD9;margin:1.1em 0 .4em}table{border-collapse:collapse}td,th{border:1px solid #E2E2EC;"
         "padding:4px 8px;text-align:left}th{background:#F1F1F8}code{background:#F0EDFF;padding:0 3px;border-radius:3px}"
         ".fb a{display:inline-block;margin-right:10px;padding:6px 12px;border-radius:8px;text-decoration:none;"
         "background:#F0EDFF;color:#4C2FD9}.muted{color:#6B6F80;font-size:.85em}")


def compose(result: Result, *, subject: str | None = None, language: str = "en", to: str | None = None,
            sender: str | None = None, ui_url: str | None = None) -> EmailMessage:
    """Email for one agent result. ``language`` picks the footer; bilingual bodies come from the agent."""
    ui = ui_url or env("UI_URL", "http://localhost:8502")
    lang = language if language in FOOTER else "en"
    ask, up, down, open_ = FOOTER[lang]
    name = AGENT_NAMES.get(result.agent, (result.agent or "Agent Hub",) * 2)[1 if lang == "vi" else 0]
    subj = subject or f"[{name}] {result.title or result.status}"
    body = result.text or result.question or ""
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    msg = EmailMessage()
    msg["Subject"] = subj
    msg["From"] = sender or env("EMAIL_FROM") or env("SMTP_USER") or "agent-hub@localhost"
    if to or env("EMAIL_TO"):
        msg["To"] = to or env("EMAIL_TO")
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="agent-hub.local")
    if result.run_id:
        msg["X-Hub-Run"] = result.run_id
        msg["X-Hub-Agent"] = result.agent

    links = ""
    text_links = ""
    if result.run_id:
        u, d = feedback_url(ui, result.run_id, 1), feedback_url(ui, result.run_id, -1)
        links = (f'<p class="fb">{html.escape(ask)} <a href="{html.escape(u)}">👍 {html.escape(up)}</a>'
                 f'<a href="{html.escape(d)}">👎 {html.escape(down)}</a></p>')
        text_links = f"\n\n{ask}\n👍 {u}\n👎 {d}"
    cites = ""
    if result.citations:
        cites = "\n\nSources:\n" + "\n".join(f"- {c}" for c in result.citations)
    foot = f"{name} · run {result.run_id or '-'} · {when} ICT · {open_}: {ui}"

    msg.set_content(f"{body}{cites}{text_links}\n\n-- \n{foot}\n")
    msg.add_alternative(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style></head><body>"
        f"{md_to_html(body + cites)}{links}<p class='muted'>{html.escape(foot)}</p></body></html>",
        subtype="html")
    return msg


def missed_runs_email(lines: list[str], *, to: str | None = None) -> EmailMessage:
    """Alert email listing jobs that did not run (bilingual)."""
    body = ("## Missed scheduled runs / Lịch chạy bị bỏ lỡ\n\n" + "\n".join(f"- {l}" for l in lines)
            + "\n\nCheck the routine in Claude Code (claude.ai/code → Routines) or the laptop task.\n"
              "Kiểm tra routine trong Claude Code hoặc tác vụ trên laptop.")
    r = Result(status="failed", text=body, title=f"{len(lines)} missed run(s)", agent="hub")
    return compose(r, subject=f"[Agent Hub] ⚠ {len(lines)} missed run(s)", to=to)


@dataclass
class Delivery:
    path: Path
    sent: bool
    error: str | None = None


class Mailer:
    def __init__(self, outbox: str | Path | None = None):
        self.outbox = Path(outbox or env("OUTBOX") or Path.cwd() / "data" / "outbox")
        self.outbox.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def smtp_configured() -> bool:
        return bool(env("SMTP_HOST") and env("SMTP_USER") and env("SMTP_PASSWORD") and env("EMAIL_TO"))

    def deliver(self, msg: EmailMessage, send: bool | None = None) -> Delivery:
        """Write to the outbox; send by SMTP when configured (or ``send=False`` to only write)."""
        stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
        run = msg.get("X-Hub-Run") or "hub"
        path = self.outbox / f"{stamp}-{run}.eml"
        n = 1
        while path.exists():
            path, n = self.outbox / f"{stamp}-{run}-{n}.eml", n + 1
        path.write_bytes(bytes(msg))
        if send is False or (send is None and not self.smtp_configured()):
            return Delivery(path, False)
        if not msg.get("To"):
            msg["To"] = env("EMAIL_TO")
        try:
            with smtplib.SMTP(env("SMTP_HOST"), int(env("SMTP_PORT", "587")), timeout=30) as s:
                s.starttls()
                s.login(env("SMTP_USER"), env("SMTP_PASSWORD"))
                s.send_message(msg)
            return Delivery(path, True)
        except (OSError, smtplib.SMTPException) as e:
            return Delivery(path, False, f"{type(e).__name__}: {e}"[:300])

    def recent(self, limit: int = 20) -> list[Path]:
        return sorted(self.outbox.glob("*.eml"), reverse=True)[:limit]
