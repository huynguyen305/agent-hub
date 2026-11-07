"""``schedules.yaml``: the single source of truth for every scheduled job (Master Plan §3.2).

Each job names its agent, runtime (``cloud`` routine or ``laptop``), a cron expression in
the hub's time zone (Asia/Ho_Chi_Minh), and how to tell that it ran:

- ``evidence: run_log``: the job writes a row in the hub's run log (laptop jobs).
- ``evidence: {vault: "<glob>"}``: the job commits files to the vault (cloud routines);
  the latest git commit touching that path counts as the run.
- ``evidence: none``: the job leaves no trace the hub can read (for example an email-only
  routine); it is listed but never reported as missed.

``missed_runs`` compares the most recent expected fire time with the evidence and returns
the jobs that did not run, after a grace period. That feeds the alert email.

The cron parser is deliberately small: five fields (minute hour day month weekday), with
``*``, lists, ranges and steps. Weekday 0 or 7 is Sunday.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from hubkit.memory.store import TZ, HubStore

RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


class ScheduleError(Exception):
    pass


def _field(spec: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
        else:
            a = b = int(part)
        if not (lo <= a <= hi and lo <= b <= hi) or step < 1:
            raise ScheduleError(f"cron field {spec!r} out of range {lo}-{hi}")
        out.update(range(a, b + 1, step))
    return out


@dataclass(frozen=True)
class Cron:
    expr: str

    def __post_init__(self):
        parts = self.expr.split()
        if len(parts) != 5:
            raise ScheduleError(f"cron {self.expr!r}: need 5 fields (minute hour day month weekday)")
        sets = [_field(p, lo, hi) for p, (lo, hi) in zip(parts, RANGES)]
        if 7 in sets[4]:
            sets[4] = (sets[4] - {7}) | {0}
        object.__setattr__(self, "_sets", sets)
        object.__setattr__(self, "_dom_star", parts[2] == "*")
        object.__setattr__(self, "_dow_star", parts[4] == "*")

    def matches(self, t: datetime) -> bool:
        mi, ho, dom, mo, dow = self._sets
        if t.minute not in mi or t.hour not in ho or t.month not in mo:
            return False
        d_ok, w_ok = t.day in dom, (t.isoweekday() % 7) in dow
        if self._dom_star or self._dow_star:  # cron: when one is restricted, that one decides
            return d_ok and w_ok
        return d_ok or w_ok  # both restricted: either matches

    def previous(self, before: datetime, horizon_days: int = 35) -> datetime | None:
        """Latest fire time strictly before ``before`` (in the hub time zone), within the horizon."""
        t = before.astimezone(TZ).replace(second=0, microsecond=0)
        if t >= before.astimezone(TZ):
            t -= timedelta(minutes=1)
        stop = t - timedelta(days=horizon_days)
        while t > stop:
            if t.minute in self._sets[0] and t.hour in self._sets[1] and self.matches(t):
                return t
            t -= timedelta(minutes=1)
        return None


@dataclass
class Job:
    name: str
    agent: str
    runtime: str  # cloud | laptop
    cron: Cron
    channel: str = "email"
    owner: str = "hub"  # hub (built here) | routine (an existing Claude Code routine)
    evidence: dict | str = "run_log"
    grace_minutes: int = 45
    enabled: bool = True
    routine_id: str | None = None
    notes: str = ""

    @property
    def vault_glob(self) -> str | None:
        return self.evidence.get("vault") if isinstance(self.evidence, dict) else None


@dataclass
class Schedule:
    jobs: list[Job] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Schedule":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if (data.get("timezone") or "Asia/Ho_Chi_Minh") != "Asia/Ho_Chi_Minh":
            raise ScheduleError("schedules.yaml must use timezone Asia/Ho_Chi_Minh")
        jobs, seen = [], set()
        for raw in data.get("jobs") or []:
            name = raw.get("name")
            if not name or name in seen:
                raise ScheduleError(f"job name missing or duplicated: {name!r}")
            seen.add(name)
            if raw.get("runtime") not in ("cloud", "laptop"):
                raise ScheduleError(f"{name}: runtime must be cloud or laptop")
            ev = raw.get("evidence", "run_log")
            if not (ev in ("run_log", "none") or (isinstance(ev, dict) and set(ev) == {"vault"})):
                raise ScheduleError(f"{name}: evidence must be run_log, none or {{vault: <glob>}}")
            if raw.get("agent") == "people" and raw["runtime"] == "cloud" and raw.get("owner", "hub") == "hub":
                raise ScheduleError(f"{name}: People Advisor jobs never run in the cloud (ADR-H12)")
            jobs.append(Job(name=name, agent=raw["agent"], runtime=raw["runtime"], cron=Cron(str(raw["cron"])),
                            channel=raw.get("channel", "email"), owner=raw.get("owner", "hub"), evidence=ev,
                            grace_minutes=int(raw.get("grace_minutes", 45)), enabled=bool(raw.get("enabled", True)),
                            routine_id=raw.get("routine_id"), notes=raw.get("notes", "")))
        return cls(jobs)

    def job(self, name: str) -> Job:
        return next(j for j in self.jobs if j.name == name)


def last_vault_commit(vault: Path, glob: str) -> datetime | None:
    """Time of the latest commit touching files that match ``glob`` (git pathspec glob)."""
    res = subprocess.run(["git", "log", "-1", "--format=%cI", "--", f":(glob){glob}"], cwd=vault,
                         capture_output=True, text=True)
    out = res.stdout.strip()
    return datetime.fromisoformat(out) if res.returncode == 0 and out else None


@dataclass
class Missed:
    job: Job
    expected_at: datetime
    last_seen: datetime | None

    def line(self) -> str:
        seen = self.last_seen.astimezone(TZ).strftime("%Y-%m-%d %H:%M") if self.last_seen else "never"
        return (f"{self.job.name} ({self.job.runtime}, {self.job.agent}): expected "
                f"{self.expected_at.strftime('%Y-%m-%d %H:%M')} ICT, last seen {seen}")


def missed_runs(schedule: Schedule, store: HubStore, now: datetime, vault: Path | None = None,
                last_seen_fn=None) -> list[Missed]:
    """Enabled jobs whose latest expected run (older than the grace period) left no evidence."""
    last_seen_fn = last_seen_fn or (lambda job: _last_seen(job, store, vault))
    out = []
    for job in schedule.jobs:
        if not job.enabled or job.evidence == "none":
            continue
        expected = job.cron.previous(now - timedelta(minutes=job.grace_minutes))
        if expected is None:
            continue
        seen = last_seen_fn(job)
        if seen is None or seen < expected - timedelta(minutes=5):  # small clock skew allowed
            out.append(Missed(job, expected, seen))
    return out


def _last_seen(job: Job, store: HubStore, vault: Path | None) -> datetime | None:
    if job.evidence == "run_log":
        run = store.last_job_run(job.name)
        return datetime.fromisoformat(run["started_at"]) if run else None
    if job.vault_glob and vault:
        return last_vault_commit(vault, job.vault_glob)
    return None
