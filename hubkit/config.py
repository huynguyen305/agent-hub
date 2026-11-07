"""Configuration from environment variables (optionally a local ``.env`` file)."""

from __future__ import annotations

import os
import re
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ('"', "'"):
            value = value[1:].split(value[0], 1)[0]   # quoted: keep as-is, drop anything after
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # unquoted: drop inline comment
        os.environ.setdefault(key.strip(), value)


ENV_PREFIXES = ("HUB_", "FIN_AGENT_")  # FIN_AGENT_* still works, so fin-analyst-agent's .env keeps working


def env(name: str, default: str | None = None) -> str | None:
    """``HUB_<NAME>``, else ``FIN_AGENT_<NAME>``, else ``default``."""
    for prefix in ENV_PREFIXES:
        v = os.getenv(prefix + name)
        if v:
            return v
    return default
