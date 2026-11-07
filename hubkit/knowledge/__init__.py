"""Vault knowledge base: GraphRAG over Obsidian notes (langchain-graph-retriever + Chroma).

``vault_docs`` turns notes into chunk Documents whose metadata carries the graph
(links, tags, ticker, lens); ``store`` embeds them locally and retrieves with
graph traversal. ``SCOPES`` gives each agent its own folders (Agent Hub ADR-H03).
"""
