import json
import subprocess

import httpx
import pytest

from hubkit.evals.jev import INPUT_PRICE_PER_M, JevClient, JevError
from hubkit.genui import Activity, Renderer
from hubkit.genui.render import REGISTRY
from hubkit.vault import VaultSyncError, VaultWriter, check_paths, scan_secrets

COACH_PREFIXES = ("Context/Quiz_Log/", "Projects/AgentHub/Runs/")


# ---------------------------------------------------------------- vault writer

def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def vault(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    v = tmp_path / "vault"
    git(tmp_path, "init", "-q", "-b", "main", str(v))
    (v / "Context/Quiz_Log").mkdir(parents=True)
    (v / "Context/Quiz_Log/_MOC-Quiz-Log.md").write_text("# Quiz log\n")
    git(v, "add", ".")
    git(v, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed")
    git(v, "remote", "add", "origin", str(remote))
    git(v, "push", "-q", "origin", "main")
    monkeypatch.setattr("shutil.which", lambda name: None)  # no gh CLI in tests
    return v, remote


def test_writer_requires_prefixes_and_git(tmp_path, vault):
    with pytest.raises(VaultSyncError, match="allowed_prefixes"):
        VaultWriter(vault[0], allowed_prefixes=())
    with pytest.raises(VaultSyncError, match="git"):
        VaultWriter(tmp_path, allowed_prefixes=COACH_PREFIXES)


def test_path_allowlist_and_secret_scan():
    check_paths({"Context/Quiz_Log/2026-11-07.md": "x"}, COACH_PREFIXES)
    with pytest.raises(VaultSyncError):
        check_paths({"Home.md": "x"}, COACH_PREFIXES)
    with pytest.raises(VaultSyncError):
        check_paths({"Context/Quiz_Log/../../Home.md": "x"}, COACH_PREFIXES)
    with pytest.raises(VaultSyncError):
        scan_secrets({"a.md": "key sk-ant-abcdefghijklmnop"})


def test_sync_pushes_a_branch_with_the_agent_as_author(vault):
    v, remote = vault
    w = VaultWriter(v, allowed_prefixes=COACH_PREFIXES, author="Knowledge Coach", email="coach@localhost")

    def build(read):
        assert read("Context/Quiz_Log/_MOC-Quiz-Log.md") == "# Quiz log\n"
        return {"Context/Quiz_Log/2026-11-07.md": "# 2026-11-07\n\nSee [[Context/Quiz_Log/_MOC-Quiz-Log]] and [[Nowhere]].\n"}

    res = w.sync(build, branch="coach/2026-11-07", title="feat(coach): quiz log 2026-11-07")
    assert res.pushed and res.files == ["Context/Quiz_Log/2026-11-07.md"]
    assert res.warnings == ["broken wikilink: Context/Quiz_Log/2026-11-07.md -> Nowhere"]
    assert git(remote, "log", "-1", "--format=%an", "coach/2026-11-07").strip() == "Knowledge Coach"
    assert "coach/2026-11-07" not in git(v, "branch")  # local branch cleaned up; the user's checkout is untouched


# ---------------------------------------------------------------- genui

class CoachActivity(Activity):
    def _on_grade(self, upd):
        for card in upd.get("graded", []):
            self.item(card["id"], card["front"], "done", f"{card['grade']}/4")
        self.card("Due today", str(upd.get("due", 0)))


def test_activity_specs_follow_steps_and_usage():
    act = CoachActivity(steps=[("select", "Picking due cards"), ("grade", "Grading answers"), ("log", "Writing the log")],
                        checklist_title="Cards")
    act.update("select", {"llm_usage": []})
    first = {s["type"]: s["props"] for s in act.specs()}
    assert [e["status"] for e in first["Timeline"]["events"]] == ["completed", "in-progress", "pending"]
    act.update("grade", {"graded": [{"id": "c1", "front": "What is SCD2?", "grade": 3}], "due": 4,
                         "llm_usage": [{"node": "grade", "model": "claude-haiku-5-5", "cost_usd": 0.001}]})
    act.finish()
    specs = {s["type"]: s["props"] for s in act.specs()}
    assert specs["ProgressBar"]["label"] == "Done" and "1 LLM calls" in specs["ProgressBar"]["detail"]
    assert specs["ChecklistWithProgress"]["completed"] == 1
    assert specs["MetricCardGrid"]["cards"][0]["props"]["value"] == "4"
    assert "haiku-5-5" in specs["Timeline"]["events"][1]["detail"]


def test_activity_error_and_interrupt():
    act = Activity(steps=[("a", "A"), ("b", "B")])
    act.update("__interrupt__", None)
    assert act.waiting and act.specs()[0]["props"]["label"] == "Waiting for you"
    act.update("a", {"error": "budget stopped"})
    assert act.specs()[-1]["props"]["title"] == "Run stopped"


def test_renderer_escapes_falls_back_and_themes():
    specs = [{"type": "InfoBanner", "props": {"title": "<script>x</script>", "seq": 1}}, {"type": "Mystery", "props": {"x": "<b>"}}]
    out = Renderer(theme="dark").render(specs, current_seq=1)
    assert "<script>x" not in out and "ga-raw" in out and '<div class="ga ga-dark">' in out
    assert '<div class="ga">' in Renderer(theme="neon").render(specs)
    assert set(REGISTRY) >= {"ProgressBar", "Timeline", "ChecklistWithProgress", "MetricCardGrid", "ToastStack"}


# ---------------------------------------------------------------- jev

def fake_jev(status=200, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        if status != 200:
            return httpx.Response(status, json={"detail": "bad request"})
        answers = {k: {"type": "noul", "noul": 0.9} for k in body["questions"]}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 1000, "output_tokens": 50}})

    return httpx.MockTransport(handler)


def test_jev_client(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient()
    res = JevClient(api_key="k", transport=fake_jev()).system_one("state", {"q": {"type": "noul", "instructions": "x"}})
    assert res.noul("q") == 0.9 and res.usage["cost_usd"] == pytest.approx(1000 * INPUT_PRICE_PER_M / 1e6)
    calls = []
    with pytest.raises(JevError, match="422"):
        JevClient(api_key="k", transport=fake_jev(422, calls), retries=2).system_one("s", {"q": {"type": "noul"}})
    assert len(calls) == 1
