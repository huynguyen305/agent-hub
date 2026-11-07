"""Stable pseudonyms for people and organisations (Agent Hub ADR-H04).

Before any LLM call, real names become role-based aliases ("Lead-A", "PM-B",
"Org-1"); after the call, aliases in the answer are mapped back locally. The
mapping is held by a ``PeopleMap`` (local SQLite in production, a dict in tests)
and never leaves the machine.

Matching is by known names only: the user registers people once with a role
("Anh Minh", role "Lead"), and every later mention is replaced, including
Vietnamese text written without diacritics. Unknown names are not guessed;
``assert_no_names`` is the CI check that no registered name reaches a payload.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field


def fold(text: str) -> str:
    """Lower-case and strip diacritics (Vietnamese đ included) so "Nguyễn" matches "nguyen"."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").lower()


@dataclass
class Person:
    name: str
    role: str  # "Lead", "PM", "Peer", "Report", "Stakeholder", "Org" ...
    alias: str = ""
    aliases_of_name: list[str] = field(default_factory=list)  # nicknames, short forms


class PeopleMap:
    """Registry of people and organisations with stable aliases per role."""

    def __init__(self) -> None:
        self.people: list[Person] = []

    def add(self, name: str, role: str, also: list[str] | None = None) -> str:
        """Register a person (or ``role="Org"`` for an organisation). Returns the stable alias."""
        for p in self.people:
            if fold(p.name) == fold(name):
                return p.alias
        same_role = sum(1 for p in self.people if p.role == role)
        alias = f"Org-{same_role + 1}" if role == "Org" else f"{role}-{string.ascii_uppercase[same_role % 26]}" + (
            str(same_role // 26) if same_role >= 26 else "")
        self.people.append(Person(name=name, role=role, alias=alias, aliases_of_name=list(also or [])))
        return alias

    def _names(self) -> list[tuple[str, Person]]:
        out = [(n, p) for p in self.people for n in [p.name, *p.aliases_of_name] if n.strip()]
        return sorted(out, key=lambda x: -len(x[0]))  # longest first: "Nguyen Van An" before "An"

    def pseudonymize(self, text: str) -> str:
        """Replace every registered name (with or without diacritics) by its alias."""
        folded = fold(text)
        spans: list[tuple[int, int, str]] = []
        for name, person in self._names():
            for m in re.finditer(rf"(?<!\w){re.escape(fold(name))}(?!\w)", folded):
                if not any(a < m.end() and m.start() < b for a, b, _ in spans):
                    spans.append((m.start(), m.end(), person.alias))
        # NFD folding can change length; map positions through a per-character index of the original
        if len(folded) != len(text):
            return self._pseudonymize_slow(text)
        for a, b, alias in sorted(spans, reverse=True):
            text = text[:a] + alias + text[b:]
        return text

    def _pseudonymize_slow(self, text: str) -> str:
        for name, person in self._names():
            pattern = "".join(_char_class(c) for c in name)
            text = re.sub(rf"(?<!\w){pattern}(?!\w)", person.alias, text, flags=re.I)
        return text

    def restore(self, text: str) -> str:
        """Map aliases in an LLM answer back to real names (local display only)."""
        for p in sorted(self.people, key=lambda p: -len(p.alias)):
            text = re.sub(rf"(?<![\w-]){re.escape(p.alias)}(?![\w-])", p.name, text)
        return text

    def leaked(self, payload: str) -> list[str]:
        """Registered names found in a payload (empty means safe to send)."""
        folded = fold(payload)
        return [n for n, _ in self._names() if re.search(rf"(?<!\w){re.escape(fold(n))}(?!\w)", folded)]

    def assert_no_names(self, payload: str) -> None:
        leaks = self.leaked(payload)
        if leaks:
            raise ValueError(f"{len(leaks)} registered name(s) would reach the LLM; pseudonymize first")


_VN_VARIANTS = {
    "a": "aàáảãạăằắẳẵặâầấẩẫậ", "e": "eèéẻẽẹêềếểễệ", "i": "iìíỉĩị", "o": "oòóỏõọôồốổỗộơờớởỡợ",
    "u": "uùúủũụưừứửữự", "y": "yỳýỷỹỵ", "d": "dđ",
}


def _char_class(c: str) -> str:
    base = fold(c)
    if base in _VN_VARIANTS:
        chars = _VN_VARIANTS[base] + _VN_VARIANTS[base].upper()
        return f"[{chars}]"
    return re.escape(c)
