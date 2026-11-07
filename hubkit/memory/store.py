"""Hub memory: one local SQLite file for runs, LLM usage, feedback and small state.

- ``runs``: the episodic log. Every request or scheduled job writes one row
  (agent, intent, channel, status, cost), so nothing fails silently.
- ``usage``: one row per LLM call (``hubkit.llm.usage.usage_entry`` rows), tagged with
  the agent, so cost is visible per agent per day.
- ``feedback``: 👍/👎 from email links and the UI.
- ``kv``: small JSON state per namespace (for example a learner model checkpoint).

Days are counted in Asia/Ho_Chi_Minh, the hub's home time zone. Nothing here leaves
the machine; the vault only receives weekly roll-ups.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from hubkit.config import env

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_DB = Path.cwd() / "data" / "hub.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, agent TEXT NOT NULL, intent TEXT, channel TEXT, job TEXT,
  status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, day TEXT NOT NULL,
  calls INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0, summary TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS runs_day ON runs(day, agent);
CREATE INDEX IF NOT EXISTS runs_job ON runs(job, started_at);
CREATE TABLE IF NOT EXISTS usage (
  run_id TEXT, agent TEXT NOT NULL, node TEXT, label TEXT, model TEXT, ok INTEGER,
  input_tokens INTEGER, output_tokens INTEGER, cache_creation_input_tokens INTEGER,
  cache_read_input_tokens INTEGER, cost_usd REAL, cost_source TEXT, ts TEXT NOT NULL, day TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_day ON usage(day, agent);
CREATE TABLE IF NOT EXISTS feedback (
  run_id TEXT NOT NULL, item TEXT DEFAULT '', vote INTEGER NOT NULL, note TEXT, channel TEXT, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
  namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY (namespace, key)
);
"""

TOKENS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def now() -> datetime:
    return datetime.now(timezone.utc)


def local_day(ts: datetime | None = None) -> str:
    return (ts or now()).astimezone(TZ).date().isoformat()


class HubStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or env("DB") or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def close(self) -> None:
        self._db.close()

    # ---------------------------------------------------------------- runs
    def start_run(self, agent: str, intent: str = "", channel: str = "", job: str | None = None,
                  run_id: str | None = None, at: datetime | None = None) -> str:
        rid = run_id or uuid.uuid4().hex[:10]
        ts = at or now()
        with self._tx() as db:
            db.execute("INSERT INTO runs (id, agent, intent, channel, job, status, started_at, day) VALUES (?,?,?,?,?,?,?,?)",
                       (rid, agent, intent, channel, job, "running", ts.isoformat(), local_day(ts)))
        return rid

    def finish_run(self, run_id: str, status: str, usage: list[dict] | None = None, summary: str = "",
                   error: str | None = None, at: datetime | None = None) -> None:
        ts = at or now()
        rows = usage or []
        agent = self.run(run_id)["agent"]
        with self._tx() as db:
            for r in rows:
                db.execute(
                    "INSERT INTO usage (run_id, agent, node, label, model, ok, input_tokens, output_tokens, "
                    "cache_creation_input_tokens, cache_read_input_tokens, cost_usd, cost_source, ts, day) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, agent, r.get("node"), r.get("label"), r.get("model"), int(bool(r.get("ok", True))),
                     *(int(r.get(k) or 0) for k in TOKENS), r.get("cost_usd"), r.get("cost_source"),
                     ts.isoformat(), local_day(ts)))
            db.execute(
                "UPDATE runs SET status=?, finished_at=?, calls=?, input_tokens=?, output_tokens=?, cost_usd=?, "
                "summary=?, error=? WHERE id=?",
                (status, ts.isoformat(), len(rows), sum(int(r.get("input_tokens") or 0) for r in rows),
                 sum(int(r.get("output_tokens") or 0) for r in rows), sum(r.get("cost_usd") or 0 for r in rows),
                 summary[:500], (error or "")[:500] or None, run_id))

    def run(self, run_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def recent_runs(self, limit: int = 20, agent: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM runs", []
        if agent:
            q, args = q + " WHERE agent=?", [agent]
        rows = self._db.execute(q + " ORDER BY started_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def last_job_run(self, job: str, ok_only: bool = True) -> dict | None:
        q = "SELECT * FROM runs WHERE job=?" + (" AND status='ok'" if ok_only else "") + " ORDER BY started_at DESC LIMIT 1"
        row = self._db.execute(q, (job,)).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- cost
    def spent(self, agent: str | None = None, day: str | None = None) -> float:
        q, args = "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE day=?", [day or local_day()]
        if agent:
            q, args = q + " AND agent=?", [*args, agent]
        return float(self._db.execute(q, args).fetchone()[0])

    def cost_report(self, days: int = 1, until: str | None = None) -> list[dict]:
        """Per agent and day: calls, tokens, cost (newest day first)."""
        end = datetime.fromisoformat(until).date() if until else datetime.now(TZ).date()
        start = (end - timedelta(days=days - 1)).isoformat()
        rows = self._db.execute(
            "SELECT day, agent, COUNT(*) calls, SUM(1 - ok) failed, SUM(input_tokens) input_tokens, "
            "SUM(output_tokens) output_tokens, COALESCE(SUM(cost_usd), 0) cost_usd, "
            "SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) unpriced "
            "FROM usage WHERE day BETWEEN ? AND ? GROUP BY day, agent ORDER BY day DESC, agent",
            (start, end.isoformat())).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- feedback and state
    def add_feedback(self, run_id: str, vote: int, item: str = "", note: str = "", channel: str = "ui") -> None:
        if vote not in (-1, 1):
            raise ValueError("vote must be 1 (👍) or -1 (👎)")
        with self._tx() as db:
            db.execute("INSERT INTO feedback (run_id, item, vote, note, channel, ts) VALUES (?,?,?,?,?,?)",
                       (run_id, item, vote, note[:500], channel, now().isoformat()))

    def feedback(self, run_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM feedback", []
        if run_id:
            q, args = q + " WHERE run_id=?", [run_id]
        return [dict(r) for r in self._db.execute(q + " ORDER BY ts", args).fetchall()]

    def put(self, namespace: str, key: str, value: Any) -> None:
        with self._tx() as db:
            db.execute("INSERT INTO kv (namespace, key, value, updated_at) VALUES (?,?,?,?) "
                       "ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                       (namespace, key, json.dumps(value, ensure_ascii=False), now().isoformat()))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT value FROM kv WHERE namespace=? AND key=?", (namespace, key)).fetchone()
        return json.loads(row[0]) if row else default
