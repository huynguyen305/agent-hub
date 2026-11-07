"""Knowledge base: embeddings, vector store and graph retrieval over the vault.

Follows the LangChain Graph RAG integration (``langchain-graph-retriever``):
edges come from existing document metadata, traversal is ``Eager`` with a few
similarity seeds (``start_k``) expanded up to ``max_depth`` hops. Store choice
follows its guidance: Chroma (local, persistent) with the ``ShreddingTransformer``
for list metadata; ``InMemoryVectorStore`` only for tests.

Embeddings run locally (financial context stays on this machine, as in the
Local Obsidian Graph RAG project). Default ``intfloat/multilingual-e5-small``:
small enough for CPU and handles Vietnamese requests against English notes.

Indexing is incremental per chunk: unchanged text keeps its embedding, changed
metadata (e.g. a newly resolved link) is rewritten without re-embedding, and a
change of embedding model rebuilds the collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document

from hubkit.config import env
from hubkit.knowledge.vault_docs import DEFAULT_INCLUDE, build_documents

DEFAULT_EMBEDDINGS = "hf:intfloat/multilingual-e5-small"
DEFAULT_DIR = Path.cwd() / "data" / "knowledge"  # per process; override with HUB_KB_DIR
NESTED_FIELDS = {"links", "tags"}
MANIFEST_SCHEMA = 2  # bump when chunk metadata fields change: forces a metadata rewrite, not a re-embed

# (source field, target field): a doc links to a note; a note is linked from docs (backlinks);
# shared tags; same company; same lens
EDGES = [("links", "note"), ("note", "links"), ("tags", "tags"), ("ticker", "ticker"), ("fa_lens", "fa_lens")]


class KnowledgeError(Exception):
    """The knowledge base cannot be used (not installed, not indexed, store error)."""


# ---------------------------------------------------------------- embeddings

_EMB: dict[str, Any] = {}
_EMB_LOCK = threading.Lock()


def get_embeddings(spec: str | None = None):
    """``hf:<model>`` (default), ``ollama:<model>`` or ``fake`` (tests). Loaded once per process;
    the lock makes a retrieval that arrives during warm-up wait instead of loading a second copy."""
    spec = spec or env("EMBEDDINGS", DEFAULT_EMBEDDINGS)
    with _EMB_LOCK:
        if spec not in _EMB:
            _EMB[spec] = _load_embeddings(spec)
        return _EMB[spec]


def _load_embeddings(spec: str):
    kind, _, name = spec.partition(":")
    if kind == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=int(name or 64))
    if kind == "ollama":
        from langchain_ollama import OllamaEmbeddings  # type: ignore
        return OllamaEmbeddings(model=name)
    if kind == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise KnowledgeError('install the knowledge extra: pip install -e ".[kb]"') from exc
        kw: dict[str, Any] = {"normalize_embeddings": True}
        q_kw = dict(kw)
        if "e5" in name.lower():  # e5 models are trained with these prefixes
            kw["prompt"], q_kw["prompt"] = "passage: ", "query: "
        return HuggingFaceEmbeddings(model_name=name, encode_kwargs=kw, query_encode_kwargs=q_kw)
    raise KnowledgeError(f"unknown HUB_EMBEDDINGS {spec!r} (use hf:<model>, ollama:<model> or fake)")


def embeddings_spec() -> str:
    return env("EMBEDDINGS", DEFAULT_EMBEDDINGS)


# ---------------------------------------------------------------- Chroma 1.x adapter

def chroma_adapter(vector_store):
    """``langchain-graph-retriever`` 0.8's Chroma adapter imports ``IncludeEnum``, which
    chromadb 1.x removed; this subclass queries with plain strings. It also declares the
    list fields, without which edges on shredded lists (tags, backlinks) are never followed."""
    from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
    from langchain_graph_retriever.adapters.chroma import ChromaAdapter

    class Chroma1Adapter(ChromaAdapter):
        def _search(self, embedding, k=4, filter=None, **kwargs):
            k = min(k, self.vector_store._collection.count())
            if k <= 0:
                return []
            r = self.vector_store._collection.query(
                query_embeddings=embedding, n_results=k, where=filter,
                include=["documents", "metadatas", "embeddings"], **kwargs)
            return [Document(id=i, page_content=c, metadata={METADATA_EMBEDDING_KEY: list(e), **(m or {})})
                    for c, m, i, e in zip(r["documents"][0], r["metadatas"][0], r["ids"][0], r["embeddings"][0])]

    return Chroma1Adapter(vector_store, nested_metadata_fields=NESTED_FIELDS)


# ---------------------------------------------------------------- results

@dataclass
class Hit:
    id: str
    path: str
    title: str
    heading: str
    kind: str
    text: str
    depth: int  # 0 = similarity seed, 1+ = reached through graph edges
    score: float
    ticker: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SyncStats:
    notes: int = 0
    chunks: int = 0
    embedded: int = 0
    metadata_updated: int = 0
    deleted: int = 0
    rebuilt: bool = False
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _sha(obj: Any) -> str:
    raw = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _collection_name(spec: str) -> str:
    return ("vault_" + re.sub(r"[^a-zA-Z0-9]+", "_", spec).strip("_"))[:60].rstrip("_")


# ---------------------------------------------------------------- knowledge base

class KnowledgeBase:
    def __init__(self, store: str | None = None, directory: str | Path | None = None, embeddings: str | None = None):
        self.store_kind = store or env("KB_STORE", "chroma")
        self.dir = Path(directory or env("KB_DIR") or DEFAULT_DIR)
        self.spec = embeddings or embeddings_spec()
        self.collection = _collection_name(self.spec)
        self._vs = None
        self._lock = threading.RLock()

    # -- store
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def vector_store(self):
        with self._lock:
            return self._vector_store()

    def _vector_store(self):
        if self._vs is None:
            emb = get_embeddings(self.spec)
            if self.store_kind == "memory":
                from langchain_core.vectorstores import InMemoryVectorStore
                self._vs = InMemoryVectorStore(embedding=emb)
            elif self.store_kind == "chroma":
                try:
                    from langchain_chroma import Chroma
                except ImportError as exc:
                    raise KnowledgeError('install the knowledge extra: pip install -e ".[kb]"') from exc
                self.dir.mkdir(parents=True, exist_ok=True)
                self._vs = Chroma(collection_name=self.collection, embedding_function=emb,
                                  persist_directory=str(self.dir / "chroma"),
                                  collection_metadata={"hnsw:space": "cosine"})
            else:
                raise KnowledgeError(f"unknown HUB_KB_STORE {self.store_kind!r} (chroma | memory)")
        return self._vs

    def count(self) -> int:
        vs = self.vector_store()
        return vs._collection.count() if self.store_kind == "chroma" else len(vs.store)

    def is_ready(self) -> bool:
        if self.store_kind == "memory":
            return self._vs is not None and self.count() > 0
        m = self.manifest()
        return bool(m.get("chunks")) and m.get("embedding_model") == self.spec

    # -- indexing
    def sync(self, vault: str | Path, include: tuple[str, ...] = DEFAULT_INCLUDE,
             progress: Callable[[str], None] | None = None, full: bool = False) -> SyncStats:
        started = datetime.now(timezone.utc)
        say = progress or (lambda _m: None)
        docs, files = build_documents(Path(vault), include)
        stats = SyncStats(notes=len(files), chunks=len(docs))
        old = self.manifest()
        if self.store_kind == "memory":
            vs = self.vector_store()
            vs.store.clear()
            vs.add_documents(docs, ids=[d.id for d in docs])
            stats.embedded, stats.rebuilt = len(docs), True
            return stats

        if full or old.get("embedding_model") != self.spec or old.get("collection") != self.collection:
            self._reset()
            old, stats.rebuilt = {}, True
        old_chunks: dict[str, dict] = old.get("chunks", {})
        if old.get("schema") != MANIFEST_SCHEMA:  # metadata layout changed: rewrite every record's metadata once
            old_chunks = {i: {**c, "meta": ""} for i, c in old_chunks.items()}
        new_chunks = {d.id: {"text": _sha(d.page_content), "meta": _sha(d.metadata)} for d in docs}
        embed = [d for d in docs if old_chunks.get(d.id, {}).get("text") != new_chunks[d.id]["text"]]
        remeta = [d for d in docs if d.id in old_chunks and old_chunks[d.id]["text"] == new_chunks[d.id]["text"]
                  and old_chunks[d.id]["meta"] != new_chunks[d.id]["meta"]]
        gone = sorted(set(old_chunks) - set(new_chunks))

        from langchain_graph_retriever.transformers import ShreddingTransformer
        shred = ShreddingTransformer()
        vs = self.vector_store()
        if gone:
            vs.delete(ids=gone)
            stats.deleted = len(gone)
        # Chroma's upsert MERGES metadata into an existing record, so a removed tag or link (a shredded
        # "tags→x" key) would survive as a stale graph edge: always delete, then insert
        stale = [d.id for d in embed if d.id in old_chunks]
        if stale:
            vs.delete(ids=stale)
        for i in range(0, len(embed), 64):
            batch = list(shred.transform_documents(embed[i:i + 64]))
            vs.add_documents(batch, ids=[d.id for d in batch])
            stats.embedded += len(batch)
            say(f"embedded {stats.embedded}/{len(embed)} chunks")
        if remeta:  # same text, new links/tags: rewrite the record with its stored embedding
            col = vs._collection
            ids = [d.id for d in remeta]
            got = col.get(ids=ids, include=["embeddings"])
            emb = dict(zip(got["ids"], got["embeddings"]))
            shredded = list(shred.transform_documents(remeta))
            col.delete(ids=ids)  # not upsert: see above, metadata would be merged instead of replaced
            col.add(ids=ids, embeddings=[list(emb[i]) for i in ids],
                    metadatas=[d.metadata for d in shredded], documents=[d.page_content for d in shredded])
            stats.metadata_updated = len(remeta)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({
            "schema": MANIFEST_SCHEMA, "embedding_model": self.spec, "collection": self.collection, "vault": str(vault),
            "include": list(include), "built_at": started.isoformat(timespec="seconds"),
            "files": files, "chunks": new_chunks,
        }, ensure_ascii=False), encoding="utf-8")
        stats.seconds = (datetime.now(timezone.utc) - started).total_seconds()
        return stats

    def _reset(self) -> None:
        vs = self.vector_store()
        try:
            vs.delete_collection()
        except Exception:
            pass
        self._vs = None

    # -- retrieval
    def retriever(self, k: int, start_k: int = 4, adjacent_k: int = 8, max_depth: int = 2):
        from graph_retriever.strategies import Eager
        from langchain_graph_retriever import GraphRetriever

        vs = self.vector_store()
        store = chroma_adapter(vs) if self.store_kind == "chroma" else vs
        return GraphRetriever(store=store, edges=EDGES,
                              strategy=Eager(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth))

    def search(self, query: str, k: int = 4, *, ticker: str | None = None, exclude_paths: set[str] = frozenset(),
               exclude_kinds: set[str] = frozenset(), reference_only: bool = False, max_per_note: int = 2, max_depth: int = 2,
               start_k: int = 4) -> list[Hit]:
        """Graph retrieval: ``start_k`` similarity seeds, expanded along metadata edges.

        Over-fetches, then drops excluded notes/kinds (e.g. skill notes already in the prompt,
        which still serve as traversal seeds) and caps chunks per note for diversity.
        """
        if not self.is_ready():
            raise KnowledgeError("knowledge base is not indexed (run: python -m hubkit.knowledge index)")
        # filters apply inside the store (seeds and every traversal step), not after: a post-filter
        # empties the result when all seeds are of an unwanted kind
        flt = {"ticker": ticker.upper()} if ticker else {}
        if reference_only:
            flt["reference"] = True
        flt = flt or None
        docs = self.retriever(k=k * 4, start_k=start_k, max_depth=max_depth).invoke(query, filter=flt)
        # Eager returns documents in selection order; rank seeds first, then by hop, most similar first
        docs.sort(key=lambda d: (int(d.metadata.get("_depth", 0)), -float(d.metadata.get("_similarity_score", 0.0))))
        hits, per_note = [], {}
        for d in docs:
            m = d.metadata
            if m.get("path") in exclude_paths or m.get("kind") in exclude_kinds:
                continue
            if per_note.get(m.get("note"), 0) >= max_per_note:
                continue
            per_note[m.get("note")] = per_note.get(m.get("note"), 0) + 1
            hits.append(Hit(id=d.id or "", path=m.get("path", ""), title=m.get("title", ""), heading=m.get("heading", ""),
                            kind=m.get("kind", ""), text=d.page_content, depth=int(m.get("_depth", 0)),
                            score=round(float(m.get("_similarity_score", 0.0)), 4), ticker=m.get("ticker", "")))
            if len(hits) >= k:
                break
        return hits
