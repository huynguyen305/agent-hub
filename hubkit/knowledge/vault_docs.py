"""Vault notes -> chunk Documents with graph metadata (no LLM extraction).

The knowledge graph is the vault's own topology, as in the Local Obsidian Graph
RAG project: explicit wikilinks, frontmatter relations, tags, and the agent's
own metadata (ticker, lens). These become metadata fields that
``langchain-graph-retriever`` turns into edges at query time:

- ``links`` (list): notes this chunk's note links to (body wikilinks + ``same``/``related``)
- ``note``: this chunk's note id (vault path without ``.md``), the target of links
- ``tags`` (list): frontmatter tags, minus "hub" tags shared by too many notes
- ``ticker``: company code for reports and data snapshots ("" otherwise)
- ``fa_lens``: lens for analyst skill notes ("" otherwise)

Hub pruning: ``up`` links and links to index/MOC/README notes are navigation,
not knowledge, and would connect everything to everything; they are dropped.
Ambiguous bare-name links are left unresolved rather than guessed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.documents import Document

# One index per agent scope (Agent Hub ADR-H03): each agent retrieves only its own folders.
SCOPES: dict[str, tuple[str, ...]] = {
    "analyst": (
        "Intelligence/Investment_Analysis",
        "Domains/Financial_Engineering",
        "Projects/FinAnalystAgent/Learnings",
    ),
    "learning": (
        "Domains/Data_Engineering",
        "Domains/Financial_Engineering",
        "Domains/AI_ML_Engineering",
        "Domains/Statistics",
        "Skills",
        "Projects/Techcombank_Data_Engineer",
    ),
    "people": (
        "Context/Stoic_Practice",
        "Skills/Career_Development",
    ),
    "world": (
        "Domains/AI_ML_Engineering/TIL",
        "Domains/Data_Engineering/TIL",
        "Domains/Financial_Engineering/TIL",
        "Intelligence",
    ),
}
DEFAULT_INCLUDE = SCOPES["analyst"]
EXCLUDE_NAMES = {"_templates", "_System", "copilot", "venv", "node_modules"}
FRONTMATTER_LINK_KEYS = ("same", "related", "see_also")  # "up" is hierarchy -> hub, skipped
REFERENCE_KINDS = {"concept", "skill", "learning"}
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 80
HUB_TAG_SHARE = 0.15  # a tag on more than 15% of notes (and 20+ notes) is a hub, not an edge
HUB_TAG_MIN_NOTES = 20

WIKILINK = re.compile(r"\[\[([^\]\[|#]+?)(?:#[^\]|]*)?(?:\\?\|[^\]]*)?\]\]")
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.S)
TICKER_PATH = re.compile(r"Company_Analysis/([A-Z0-9]{2,6})/")


@dataclass
class Note:
    id: str  # vault-relative path without .md
    path: str
    title: str
    meta: dict
    body: str
    raw_links: list[str] = field(default_factory=list)
    sha256: str = ""


def is_hub(note_id: str) -> bool:
    name = note_id.rsplit("/", 1)[-1].upper()
    return name.startswith(("_INDEX", "_MOC", "_README")) or name.endswith("_INDEX") or name == "README"


def list_notes(vault: Path, include: tuple[str, ...] = DEFAULT_INCLUDE) -> list[Path]:
    out = []
    for folder in include:
        base = vault / folder
        if not base.exists():
            continue
        for p in base.rglob("*.md"):
            parts = p.relative_to(vault).parts
            if any(x.startswith(".") or x in EXCLUDE_NAMES for x in parts):
                continue
            out.append(p)
    return sorted(set(out))


def read_note(vault: Path, p: Path) -> Note:
    raw = p.read_bytes()
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    meta, body = {}, text
    m = FRONTMATTER.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}  # malformed frontmatter: index the note anyway
        body = m.group(2)
        if not isinstance(meta, dict):
            meta = {}
    rel = p.relative_to(vault).as_posix()
    note_id = rel[:-3] if rel.endswith(".md") else rel
    title = next((h.group(2) for line in body.splitlines() if (h := HEADING.match(line)) and len(h.group(1)) == 1),
                 note_id.rsplit("/", 1)[-1])
    fm_links = []
    for key in FRONTMATTER_LINK_KEYS:
        vals = meta.get(key) or []
        for v in vals if isinstance(vals, list) else [vals]:
            fm_links += WIKILINK.findall(str(v))
    return Note(id=note_id, path=rel, title=str(title), meta=meta, body=body, raw_links=fm_links,
                sha256=hashlib.sha256(raw).hexdigest())


class LinkResolver:
    """Resolve ``[[target]]`` to a note id: full paths as-is, bare names only when unique."""

    def __init__(self, note_ids: list[str]):
        self.ids = set(note_ids)
        by_name: dict[str, list[str]] = {}
        for i in note_ids:
            by_name.setdefault(i.rsplit("/", 1)[-1].lower(), []).append(i)
        self.by_name = {k: v[0] for k, v in by_name.items() if len(v) == 1}

    def resolve(self, target: str) -> str | None:
        t = target.strip().rstrip("\\").removesuffix(".md").strip("/")
        if not t:
            return None
        if "/" in t:
            return t if t in self.ids else None  # links outside the indexed folders are not edges
        return self.by_name.get(t.lower())


def note_tags(meta: dict) -> list[str]:
    tags = meta.get("tags") or []
    tags = tags if isinstance(tags, list) else [tags]
    return sorted({str(t).strip().lstrip("#").lower() for t in tags if str(t).strip()})


def hub_tags(notes: list[Note]) -> set[str]:
    counts: dict[str, int] = {}
    for n in notes:
        for t in note_tags(n.meta):
            counts[t] = counts.get(t, 0) + 1
    limit = max(HUB_TAG_MIN_NOTES, HUB_TAG_SHARE * len(notes))
    return {t for t, c in counts.items() if c > limit}


def note_kind(n: Note) -> str:
    if is_hub(n.id):
        return "index"  # navigation (index/MOC/README): indexed for search, never background knowledge
    if "/Analyst_Skills/" in f"/{n.path}":
        return "skill"
    if "/Company_Analysis/" in f"/{n.path}":
        return "company_data" if "/Data/" in n.path else "company_report"
    if "/Learnings/" in n.path:
        return "learning"
    if "/Daily_Brief" in n.path:
        return "news"  # dated market briefs: searchable, but not background knowledge for a lens
    return "concept"


def sections(body: str) -> list[tuple[str, str]]:
    """(heading breadcrumb, text) per heading section, long sections split on paragraphs."""
    out, crumbs, buf = [], [], []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            crumb = " > ".join(c for _, c in crumbs)
            for piece in split_long(text):
                out.append((crumb, piece))
        buf.clear()

    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        h = None if in_code else HEADING.match(line)
        if h:
            flush()
            level = len(h.group(1))
            crumbs = [c for c in crumbs if c[0] < level] + [(level, h.group(2).strip())]
        else:
            buf.append(line)
    flush()
    return out


def split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) + 2 > limit:
            pieces.append(cur)
            cur = ""
        while len(para) > limit:  # one huge paragraph or table
            cut = para.rfind("\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(para[:cut])
            para = para[cut:].lstrip("\n")
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        pieces.append(cur)
    return pieces


def build_documents(vault: Path, include: tuple[str, ...] = DEFAULT_INCLUDE) -> tuple[list[Document], dict[str, str]]:
    """All chunk Documents for the included folders, plus ``{path: sha256}`` of the source notes."""
    notes = [read_note(vault, p) for p in list_notes(vault, include)]
    resolver = LinkResolver([n.id for n in notes])
    hubs = hub_tags(notes)
    docs: list[Document] = []
    for n in notes:
        fm_links = {r for t in n.raw_links if (r := resolver.resolve(t))}
        tags = [t for t in note_tags(n.meta) if t not in hubs]
        m = TICKER_PATH.search(n.path)
        ticker = str(n.meta.get("fa_ticker") or (m.group(1) if m else "")).upper()
        base = {
            "note": n.id, "path": n.path, "title": n.title, "kind": note_kind(n), "ticker": ticker,
            "fa_lens": str(n.meta.get("fa_lens") or ""), "fa_sector": str(n.meta.get("fa_sector") or ""),
            "date": str(n.meta.get("date") or ""),
            # reference knowledge (concepts, rules, learnings) vs dated material (news, company reports):
            # lens retrieval filters on this inside the store, so dated notes never become seeds
            "reference": note_kind(n) in REFERENCE_KINDS and not is_hub(n.id),  # index/MOC notes are navigation
        }
        # empty fields are omitted, not stored as "": an edge on ticker == "" would join every non-company note
        base = {k: v for k, v in base.items() if v}
        # links are note-level (Note LINKS_TO Note): links usually sit in a "Related" section, and a
        # chunk about RSI should still reach the notes its parent note points to
        body_links = {r for t in WIKILINK.findall(n.body) if (r := resolver.resolve(t))}
        links = sorted(x for x in body_links | fm_links if x != n.id and not is_hub(x))
        chunks = [(c, t) for c, t in sections(n.body) if len(t) >= MIN_CHUNK_CHARS]
        for i, (crumb, text) in enumerate(chunks):
            # the breadcrumb usually starts at the note's H1 (= title); don't repeat it
            head = crumb if crumb.startswith(n.title) else f"{n.title} > {crumb}" if crumb else n.title
            docs.append(Document(id=f"{n.id}#{i}", page_content=f"{head}\n{text}",
                                 metadata={**base, "heading": crumb, "links": links, "tags": tags}))
    return docs, {n.path: n.sha256 for n in notes}
