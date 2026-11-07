# agent-hub

The Agent Hub: specialist agents (Knowledge Coach, People Advisor, World Intelligence, and Fin Analyst as agent #0) on one shared platform, `hubkit`. The plan, decisions and roadmap live in the Obsidian vault under `Projects/AgentHub/`.

## Status

- **Phase 0 (done):** `hubkit` extracted from `fin-analyst-agent` (tag `v0.1.0`); Fin Analyst runs on it.
- **Phase 1 (hub skeleton, in progress):** router, email channel, ledger with daily budgets, `schedules.yaml` with
  missed-run checks, hub UI shell, Fin Analyst as agent #0.
- Agents: Knowledge Coach (Phase 2), World Intelligence (Phase 3), People Advisor (Phase 4).

## Using the hub

```powershell
copy .env.example .env                                  # VAULT_PATH, and SMTP settings to send email
.venv\Scripts\python -m hub ask "Phân tích FPT 5 năm" --email   # routed to Fin Analyst; email saved or sent
.venv\Scripts\python -m hub route "Prep me for a 1:1 with my director"
.venv\Scripts\python -m hub eval-router                # accuracy on tests/golden/router.jsonl
.venv\Scripts\python -m hub cost --days 7              # spend per agent vs daily budgets
.venv\Scripts\python -m hub schedule check --email     # jobs that did not run (alert email)
powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1   # hub UI on http://localhost:8502
```

| Piece | Where | What it does |
| --- | --- | --- |
| Router | `hub/router.py` | Commands, then weighted EN/VI cue rules; Haiku only when unsure; asks rather than guessing between People and Coach. Golden set: `tests/golden/router.jsonl` |
| Runner | `hub/runner.py` | Input guards (injection, PII, confidential work markers), routing, budget check, agent call, run log, email |
| Contract | `hub/contract.py` | `Request` / `Result` shared by every agent (Agent Specs §5) |
| Email | `channels/email_out.py` | Plain text + HTML, 👍/👎 links into the hub UI, local outbox, SMTP when configured |
| Schedules | `hub/schedules.yaml` | Every scheduled job, its runtime and its evidence (vault commits or the run log) |
| Agent #0 | `agents/fin/adapter.py` | Fin Analyst's graph behind the contract, with live events for the activity panel |
| UI | `app/ui.py` | Ask (any agent, live panel), Agents, Cost, Schedules, Outbox; email feedback links land here |

## hubkit

| Module | What it gives every agent | Decision |
| --- | --- | --- |
| `hubkit.llm` | Providers (Claude subscription via the Agent SDK, Anthropic API, fake), structured output, usage and cost ledger, overhead detector, `ModelRouter` per agent | ADR-H01, H02 |
| `hubkit.guards` | Prompt injection and PII cleaning, fetched content as data, certainty language, disclaimers, **confidential-work markers**, **people pseudonymization** (`PeopleMap`) | ADR-H04, H05 |
| `hubkit.knowledge` | GraphRAG over the vault (local embeddings, Chroma, graph traversal) with **one scope per agent** (`SCOPES`: analyst, learning, people, world) | ADR-H03 |
| `hubkit.vault` | `VaultWriter`: worktree branch, path allowlist per agent, secret scan, wikilink check, push and PR | ADR-H01 |
| `hubkit.genui` | `Activity` (graph events to AgenticGenUI specs) and the Streamlit `Renderer` (light and dark) | – |
| `hubkit.evals` | TypeSafe Jev client | – |
| `hubkit.memory` | SQLite run log, usage ledger per agent, feedback, small state; `Budgets` with hard daily caps | Master Plan §5.5 |
| `hubkit.scheduling` | `schedules.yaml` loader, small cron parser, missed-run detection | ADR-H12 |
| `hubkit.config` | `.env` loader; `env("X")` reads `HUB_X`, then `FIN_AGENT_X` | – |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,agents,ui]"   # core, LangGraph, UI, tests
.venv/Scripts/pip install -e "../fin-analyst-agent[ui]" --no-deps   # agent #0, from a sibling checkout
.venv/Scripts/pip install -e ".[kb]"           # vault knowledge base (install CPU torch first on Windows)
.venv/Scripts/python -m pytest -q
```

## Rules carried over from Fin Analyst

- LLMs reason, Python counts: scheduling, scoring, selection, dedupe, privacy and budgets are code (ADR-H02).
- Everything fetched or retrieved is data, never instructions.
- No real colleague names in any LLM payload; employer documents never enter a personal-plan LLM.
- Vault writes go through `VaultWriter` with an explicit per-agent path allowlist.
