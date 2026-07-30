"""SQLAlchemy ORM models for RepoMind's PostgreSQL persistence layer.

Mirrors the in-memory dataclasses produced by earlier phases —
`ingestion.repository_metadata.RepositoryMetadata`, `ingestion.source_file.SourceFile`,
and `models.schemas.CodeChunk` — plus tables for the knowledge graph built
in Phase 6 (`graph.graph_builder.RepositoryGraphBuilder`): a lossy SQL
reconstruction queried via `database.sqlite_client.DatabaseManager.load_graph`,
and the exact persisted snapshot (`GraphSnapshotRecord`) read/written by
`database.graph_store`. Those two modules are the only ones that should
import these models directly; every other layer interacts with plain
dataclasses/`networkx.DiGraph` objects through them.

`file_id` and `chunk_id` (see `ingestion.deterministic_ids`) are derived
only from a file's repository-relative path, so they are stable across
re-indexing runs of the *same* repository but are not globally unique
across *different* repositories (two repos can each have a `src/main.py`
that hashes to the same `file_id`). Every table below therefore scopes its
primary key and foreign keys by `repository_id` rather than relying on
`file_id`/`chunk_id` alone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by every RepoMind ORM model."""


class RepositoryRecord(Base):
    """One row per indexed repository.

    `repository_id` is a deterministic UUID5 derived from
    ``(owner, repository_name)`` (see
    `database.sqlite_client.DatabaseManager.compute_repository_id`), so
    re-indexing the same repository resolves to the same row instead of
    creating a duplicate.
    """

    __tablename__ = "repositories"

    repository_id: Mapped[str] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(nullable=False)
    repository_name: Mapped[str] = mapped_column(nullable=False)
    clone_url: Mapped[str] = mapped_column(nullable=False)
    local_path: Mapped[str] = mapped_column(nullable=False)
    default_branch: Mapped[str] = mapped_column(nullable=False)
    current_commit_hash: Mapped[str] = mapped_column(nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("owner", "repository_name", name="uq_repository_owner_name"),
    )


class SourceFileRecord(Base):
    """One row per discovered source file, scoped to its repository.

    Primary key is ``(repository_id, file_id)`` rather than `file_id`
    alone, since `file_id` is only unique within one repository.
    """

    __tablename__ = "source_files"

    repository_id: Mapped[str] = mapped_column(primary_key=True)
    file_id: Mapped[str] = mapped_column(primary_key=True)
    relative_path: Mapped[str] = mapped_column(nullable=False)
    language: Mapped[str] = mapped_column(nullable=False)
    extension: Mapped[str] = mapped_column(nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["repository_id"], ["repositories.repository_id"]),
    )


class CodeChunkRecord(Base):
    """One row per chunk (AST, sliding-window, or parent).

    Primary key is ``(repository_id, chunk_id)``. `parent_chunk_id`
    self-references another row of this same table within the same
    repository (small-to-big retrieval, Phase 5); it is only set for
    AST chunks that have an attached parent chunk.
    """

    __tablename__ = "code_chunks"

    repository_id: Mapped[str] = mapped_column(primary_key=True)
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    file_id: Mapped[str] = mapped_column(nullable=False)
    parent_chunk_id: Mapped[str | None] = mapped_column(nullable=True)
    chunk_type: Mapped[str] = mapped_column(nullable=False)
    function_name: Mapped[str | None] = mapped_column(nullable=True)
    class_name: Mapped[str | None] = mapped_column(nullable=True)
    parent_class: Mapped[str | None] = mapped_column(nullable=True)
    language: Mapped[str] = mapped_column(nullable=False)
    start_line: Mapped[int] = mapped_column(nullable=False)
    end_line: Mapped[int] = mapped_column(nullable=False)
    raw_code: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["repository_id", "file_id"],
            ["source_files.repository_id", "source_files.file_id"],
        ),
        ForeignKeyConstraint(
            ["repository_id", "parent_chunk_id"],
            ["code_chunks.repository_id", "code_chunks.chunk_id"],
        ),
    )


class GraphEdgeRecord(Base):
    """One row per directed edge between two chunks in the same repository.

    `edge_id` is a surrogate autoincrement key (edges have no natural
    identity of their own); the unique constraint on the
    ``(repository_id, source_chunk_id, target_chunk_id, edge_type)``
    tuple is what actually prevents duplicate edges.
    """

    __tablename__ = "graph_edges"

    edge_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repository_id: Mapped[str] = mapped_column(nullable=False)
    source_chunk_id: Mapped[str] = mapped_column(nullable=False)
    target_chunk_id: Mapped[str] = mapped_column(nullable=False)
    edge_type: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["repository_id", "source_chunk_id"],
            ["code_chunks.repository_id", "code_chunks.chunk_id"],
        ),
        ForeignKeyConstraint(
            ["repository_id", "target_chunk_id"],
            ["code_chunks.repository_id", "code_chunks.chunk_id"],
        ),
        UniqueConstraint(
            "repository_id", "source_chunk_id", "target_chunk_id", "edge_type",
            name="uq_graph_edge_triple",
        ),
    )


class EmbeddingRecord(Base):
    """One row per (chunk, embedding model) pair.

    `embedding_blob` stores the vector as raw little-endian float32 bytes
    (`numpy.ndarray.tobytes()`) rather than JSON or pickle - compact, and
    trivially reconstructed with `numpy.frombuffer(blob, dtype=np.float32)`
    given `embedding_dimension`. The unique constraint on
    ``(repository_id, chunk_id, model_name)`` is what actually prevents a
    chunk from being embedded twice by the same model; re-embedding with
    `force=True` updates this same row rather than inserting a new one.
    """

    __tablename__ = "embeddings"

    embedding_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repository_id: Mapped[str] = mapped_column(nullable=False)
    chunk_id: Mapped[str] = mapped_column(nullable=False)
    model_name: Mapped[str] = mapped_column(nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(nullable=False)
    embedding_blob: Mapped[bytes] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["repository_id", "chunk_id"],
            ["code_chunks.repository_id", "code_chunks.chunk_id"],
        ),
        UniqueConstraint(
            "repository_id", "chunk_id", "model_name",
            name="uq_embedding_chunk_model",
        ),
    )


class SemanticCacheRecord(Base):
    """One cached (query, response) pair for a repository (Phase 14).

    `query_embedding`/`embedding_dimension` follow the same storage
    convention as `EmbeddingRecord` (raw float32 bytes, reconstructed with
    `numpy.frombuffer`). `retrieved_chunk_ids` is a JSON-encoded list of
    chunk_id strings. The unique constraint on ``(repository_id, query)``
    is what makes "overwrite" well-defined: storing the same query text
    again for the same repository updates this row instead of duplicating
    it. `created_at` is not currently used for expiry, but is present so a
    future phase can add TTL-based invalidation without a schema change.
    """

    __tablename__ = "semantic_cache"

    cache_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repository_id: Mapped[str] = mapped_column(nullable=False)
    query: Mapped[str] = mapped_column(nullable=False)
    query_embedding: Mapped[bytes] = mapped_column(nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(nullable=False)
    response: Mapped[str] = mapped_column(nullable=False)
    retrieved_chunk_ids: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["repository_id"], ["repositories.repository_id"]),
        UniqueConstraint("repository_id", "query", name="uq_cache_repo_query"),
    )


class GraphSnapshotRecord(Base):
    """The one persisted `networkx.node_link_data` snapshot per repository (`database.graph_store`).

    Phase 33 storage migration: this table replaces the per-repository
    JSON file (`data/graph/{owner}_{name}.json`) `database.graph_store`
    used to read/write directly. `graph_data` stores the exact same
    `nx.node_link_data(graph)` dict `save_graph` always produced - only
    where it lives changed, not its shape - so `load_graph`'s round-trip
    via `nx.node_link_graph` is unaffected.
    """

    __tablename__ = "graph_snapshots"

    repository_id: Mapped[str] = mapped_column(primary_key=True)
    graph_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["repository_id"], ["repositories.repository_id"]),
    )
