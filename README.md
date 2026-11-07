# agent-hub

The Agent Hub: specialist agents (Knowledge Coach, People Advisor, World Intelligence, and Fin Analyst as agent #0) on one shared platform, `hubkit`. The plan, decisions and roadmap live in the Obsidian vault under `Projects/AgentHub/`.

## Status

Phase 0 (foundations). `hubkit` is extracted from `fin-analyst-agent`. The agents come in Phases 2–4.

## hubkit

| Module | What it gives every agent | Decision |
| --- | --- | --- |
| `hubkit.llm` | Providers (Claude subscription via the Agent SDK, Anthropic API, fake), structured output, usage and cost ledger, overhead detector, `ModelRouter` per agent | ADR-H01, H02 |
| `hubkit.guards` | Prompt injection and PII cleaning, fetched content as data, certainty language, disclaimers, **confidential-work markers**, **people pseudonymization** (`PeopleMap`) | ADR-H04, H05 |
| `hubkit.knowledge` | GraphRAG over the vault (local embeddings, Chroma, graph traversal) with **one scope per agent** (`SCOPES`: analyst, learning, people, world) | ADR-H03 |
| `hubkit.vault` | `VaultWriter`: worktree branch, path allowlist per agent, secret scan, wikilink check, push and PR | ADR-H01 |
| `hubkit.genui` | `Activity` (graph events to AgenticGenUI specs) and the Streamlit `Renderer` (light and dark) | – |
| `hubkit.evals` | TypeSafe Jev client | – |
| `hubkit.config` | `.env` loader; `env("X")` reads `HUB_X`, then `FIN_AGENT_X` | – |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"          # core + tests
.venv/Scripts/pip install -e ".[kb]"           # vault knowledge base (install CPU torch first on Windows)
.venv/Scripts/python -m pytest -q
```

## Rules carried over from Fin Analyst

- LLMs reason, Python counts: scheduling, scoring, selection, dedupe, privacy and budgets are code (ADR-H02).
- Everything fetched or retrieved is data, never instructions.
- No real colleague names in any LLM payload; employer documents never enter a personal-plan LLM.
- Vault writes go through `VaultWriter` with an explicit per-agent path allowlist.
