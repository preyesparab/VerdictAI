"""PostgreSQL persistence adapter for structured metadata (Phase 7-8, 14).

`DatabaseManager` is the single public entry point for reading and writing
everything produced by earlier phases: repository metadata (Phase 2),
source files (Phase 3), chunks (Phases 4-5), the knowledge graph (Phase 6),
chunk embeddings (Phase 8), and semantic cache entries (Phase 14). Later
phases (retrieval, evaluation) load chunks/the graph/embeddings/cache
entries from here instead of re-running ingestion or re-embedding, and
instead of holding onto the in-memory objects a specific pipeline run
produced.

Backed by PostgreSQL via `psycopg2` (Phase 33 storage migration; originally
SQLite). `psycopg2` is a synchronous driver, deliberately chosen over
`asyncpg` so every method here keeps the same blocking `Session` API its
callers (`pipeline.py`, `api/main.py`, embedding/retrieval modules) already
depend on - no async rewrite at those call sites.

Every public method wraps exactly one transaction: all rows in a single
call are committed together, or none are (see `session_scope`). No
`sqlalchemy.exc` exception ever escapes this module — every failure is
re-raised as `core.exceptions.DatabaseError`.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from core.exceptions import DatabaseError
from core.logging import get_logger
from database.models import (
    Base,
    CodeChunkRecord,
    EmbeddingRecord,
    GraphEdgeRecord,
    RepositoryRecord,
    SemanticCacheRecord,
    SourceFileRecord,
)
from ingestion.chunker import ChunkingResult
from ingestion.deterministic_ids import compute_file_id
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.source_file import SourceFile
from models.schemas import CacheEntry, ChunkType, CodeChunk

logger = get_logger(__name__)

# Fixed, arbitrary namespace (generated once) used to derive a deterministic
# repository_id from (owner, repository_name) - same technique as
# ingestion.deterministic_ids, kept local to this module since repository_id
# is a persistence-layer concept with no ingestion-stage consumer.
_REPOSITORY_ID_NAMESPACE = uuid.UUID("c9f1a2b3-4d5e-4f6a-8b7c-1d2e3f4a5b6c")


_VALID_SCHEMA_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class DatabaseManager:
    """Owns the PostgreSQL engine/session for one RepoMind database.

    Each instance connects to `database_url` (a full database, shared by
    default across the whole process - PostgreSQL, unlike SQLite, has no
    single-file-per-database notion). Foreign keys are always enforced by
    PostgreSQL, unlike SQLite, so no connect-time PRAGMA is needed.

    `schema` scopes every table this instance creates/reads/writes to one
    PostgreSQL schema within that database via the connection's
    `search_path`, instead of a schema-qualifying every table in
    `database.models`. Production code leaves it unset (the connection's
    default `search_path`, normally ``public``); tests pass a unique
    per-test schema name so concurrent test runs never see each other's
    rows, replacing the old one-SQLite-file-per-test-via-`tmp_path`
    isolation.
    """

    def __init__(self, database_url: str | None = None, schema: str | None = None) -> None:
        """Initialize the manager without touching the database yet.

        Args:
            database_url: SQLAlchemy connection URL. Defaults to
                `settings.DATABASE_URL`.
            schema: If given, every table this instance touches is scoped
                to this PostgreSQL schema (via `search_path`) instead of
                the connection's default schema. Must be a valid bare SQL
                identifier - not user input. The schema itself is not
                created until `initialize_database` is called.

        Raises:
            ValueError: If `schema` is not a safe bare identifier.
        """
        if schema is not None and not _VALID_SCHEMA_NAME.match(schema):
            raise ValueError(f"Invalid schema name: {schema!r}")

        self._database_url = database_url or settings.DATABASE_URL
        self._schema = schema
        connect_args = {"options": f"-csearch_path={schema}"} if schema else {}
        self._engine: Engine = create_engine(
            self._database_url, future=True, pool_pre_ping=True, connect_args=connect_args
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    @staticmethod
    def compute_repository_id(owner: str, repository_name: str) -> str:
        """Derive the deterministic `repository_id` for `(owner, repository_name)`.

        Args:
            owner: Repository owner or organization.
            repository_name: Repository name.

        Returns:
            A UUID5 string, stable across repeated calls with the same
            arguments - re-indexing the same repository resolves to the
            same row instead of creating a duplicate.
        """
        return str(uuid.uuid5(_REPOSITORY_ID_NAMESPACE, f"{owner}/{repository_name}"))

    def initialize_database(self) -> None:
        """Create this instance's schema (if given) and every table in `database.models`.

        Idempotent: safe to call on an already-initialized database, since
        `CREATE SCHEMA IF NOT EXISTS` and `Base.metadata.create_all` both
        only create what does not already exist.

        Raises:
            DatabaseError: If the schema or its tables cannot be created.
        """
        try:
            if self._schema:
                # CREATE SCHEMA must run on a connection whose search_path
                # doesn't already point at the (possibly not-yet-existing)
                # target schema, so this uses a separate, schema-agnostic
                # connection rather than `self._engine`.
                with create_engine(self._database_url, future=True).connect() as connection:
                    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"'))
                    connection.commit()
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to initialize database (schema={self._schema}): {exc}") from exc
        logger.info("Database initialized (schema=%s)", self._schema or "default")

    def drop_schema(self) -> None:
        """Drop this instance's schema and everything in it.

        Only valid for an instance constructed with an explicit `schema`
        (test isolation); dropping the default/production schema this way
        is deliberately not supported.

        Raises:
            DatabaseError: If no `schema` was given, or the drop fails.
        """
        if not self._schema:
            raise DatabaseError("drop_schema() requires a DatabaseManager constructed with an explicit schema")
        try:
            with create_engine(self._database_url, future=True).connect() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{self._schema}" CASCADE'))
                connection.commit()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to drop schema {self._schema}: {exc}") from exc
        logger.info("Schema dropped: %s", self._schema)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Run one transaction: commit on success, roll back and re-raise on failure.

        Public so `database.graph_store` can share this instance's exact
        engine/schema (its `GraphSnapshotRecord` has a foreign key into
        `RepositoryRecord`, so it must use the same connection scope as
        every other table) rather than opening an independent connection.

        Yields:
            A `Session` for the caller to add/query rows on.

        Raises:
            DatabaseError: If any database operation within the block
                fails; the transaction is rolled back before this is raised.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
            logger.info("Transaction committed")
        except SQLAlchemyError as exc:
            session.rollback()
            logger.warning("Transaction rolled back: %s", exc)
            raise DatabaseError(str(exc)) from exc
        finally:
            session.close()

    # -- Store ---------------------------------------------------------------

    def store_repository(self, repository: RepositoryMetadata) -> str:
        """Insert `repository`, or update it if already stored.

        Args:
            repository: Metadata to persist, typically returned by
                `ingestion.repository_manager.RepositoryManager.get_repository`.

        Returns:
            The repository's deterministic `repository_id`.

        Raises:
            DatabaseError: If the write fails.
        """
        repository_id = self.compute_repository_id(repository.owner, repository.name)
        with self.session_scope() as session:
            record = session.get(RepositoryRecord, repository_id)
            if record is None:
                record = RepositoryRecord(repository_id=repository_id)
                session.add(record)
                logger.info("Repository stored: %s/%s", repository.owner, repository.name)
            else:
                logger.info("Repository already present, updating: %s/%s", repository.owner, repository.name)

            record.owner = repository.owner
            record.repository_name = repository.name
            record.clone_url = repository.clone_url
            record.local_path = str(repository.local_path)
            record.default_branch = repository.default_branch
            record.current_commit_hash = repository.commit_hash
            record.indexed_at = repository.last_updated

        return repository_id

    def store_source_files(self, repository_id: str, source_files: list[SourceFile]) -> int:
        """Insert every file in `source_files` not already stored for this repository.

        Args:
            repository_id: The owning repository's id, from `store_repository`.
            source_files: Files discovered by
                `ingestion.file_discovery.FileDiscovery`.

        Returns:
            The number of newly inserted rows (already-present files are skipped).

        Raises:
            DatabaseError: If the write fails.
        """
        stored = 0
        with self.session_scope() as session:
            existing_ids = set(
                session.scalars(
                    select(SourceFileRecord.file_id).where(
                        SourceFileRecord.repository_id == repository_id
                    )
                )
            )
            for source_file in source_files:
                file_id = compute_file_id(source_file.relative_path)
                if file_id in existing_ids:
                    continue
                session.add(
                    SourceFileRecord(
                        repository_id=repository_id,
                        file_id=file_id,
                        relative_path=source_file.relative_path.as_posix(),
                        language=source_file.language,
                        extension=source_file.extension,
                        size_bytes=source_file.size_bytes,
                    )
                )
                existing_ids.add(file_id)
                stored += 1

        logger.info(
            "Files stored: %d new (%d already present) for repository %s",
            stored, len(source_files) - stored, repository_id,
        )
        return stored

    def store_chunks(self, repository_id: str, chunking_results: dict[str, ChunkingResult]) -> int:
        """Insert every chunk in `chunking_results` not already stored for this repository.

        Args:
            repository_id: The owning repository's id, from `store_repository`.
            chunking_results: Per-file chunking output, keyed by
                repository-relative path, as produced by
                `pipeline.IndexingPipeline` (one `ChunkingResult` per file).

        Returns:
            The number of newly inserted rows (already-present chunks are skipped).

        Raises:
            DatabaseError: If the write fails - including if a chunk
                references a `file_id` not stored by `store_source_files`,
                which rolls back every chunk in this call, not just the
                offending one.
        """
        stored = 0
        with self.session_scope() as session:
            existing_ids = set(
                session.scalars(
                    select(CodeChunkRecord.chunk_id).where(
                        CodeChunkRecord.repository_id == repository_id
                    )
                )
            )
            for relative_path, result in chunking_results.items():
                file_id = compute_file_id(Path(relative_path))
                # Parent chunks are flushed before ast_chunks, which may carry
                # a parent_chunk_id pointing at one of them: the self-
                # referential FK on CodeChunkRecord requires the parent row
                # to exist first, and relying on Session.add() ordering alone
                # is not a guaranteed ordering without an explicit
                # relationship(), so flush explicitly between the two groups.
                for chunk in result.parent_chunks:
                    chunk_id = str(chunk.chunk_id)
                    if chunk_id in existing_ids:
                        continue
                    session.add(self._chunk_to_record(chunk, repository_id, file_id))
                    existing_ids.add(chunk_id)
                    stored += 1
                session.flush()

                for chunk in [*result.sliding_chunks, *result.ast_chunks]:
                    chunk_id = str(chunk.chunk_id)
                    if chunk_id in existing_ids:
                        continue
                    session.add(self._chunk_to_record(chunk, repository_id, file_id))
                    existing_ids.add(chunk_id)
                    stored += 1

        logger.info(
            "Chunks stored: %d new across %d file(s) for repository %s",
            stored, len(chunking_results), repository_id,
        )
        return stored

    def _chunk_to_record(self, chunk: CodeChunk, repository_id: str, file_id: str) -> CodeChunkRecord:
        """Build the `CodeChunkRecord` for one `CodeChunk`.

        Args:
            chunk: The chunk to convert.
            repository_id: The owning repository's id.
            file_id: The owning file's deterministic id.

        Returns:
            An unsaved `CodeChunkRecord` ready to be added to a session.
        """
        return CodeChunkRecord(
            repository_id=repository_id,
            chunk_id=str(chunk.chunk_id),
            file_id=file_id,
            parent_chunk_id=str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
            chunk_type=chunk.chunk_type.value,
            function_name=chunk.function_name,
            class_name=chunk.class_name,
            parent_class=chunk.parent_class,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            raw_code=chunk.raw_code,
        )

    def store_graph(self, repository_id: str, graph: nx.DiGraph) -> int:
        """Insert every chunk-to-chunk edge in `graph` not already stored for this repository.

        Only edges between two chunk nodes are persisted - `GraphEdgeRecord`
        foreign keys reference `CodeChunkRecord`, and `graph` (built by
        `graph.graph_builder.RepositoryGraphBuilder`) also contains file
        nodes (for import edges) and chunk nodes not yet stored via
        `store_chunks`; both are logged and skipped rather than raising,
        matching the graph builder's own policy of skipping unresolved
        references instead of failing construction.

        Args:
            repository_id: The owning repository's id, from `store_repository`.
            graph: The graph built by `RepositoryGraphBuilder.build_graph`.

        Returns:
            The number of newly inserted edges.

        Raises:
            DatabaseError: If the write fails.
        """
        stored = 0
        skipped = 0
        with self.session_scope() as session:
            stored_chunk_ids = set(
                session.scalars(
                    select(CodeChunkRecord.chunk_id).where(
                        CodeChunkRecord.repository_id == repository_id
                    )
                )
            )
            existing_triples = set(
                session.execute(
                    select(
                        GraphEdgeRecord.source_chunk_id,
                        GraphEdgeRecord.target_chunk_id,
                        GraphEdgeRecord.edge_type,
                    ).where(GraphEdgeRecord.repository_id == repository_id)
                )
            )

            for source, target, data in graph.edges(data=True):
                if source not in stored_chunk_ids or target not in stored_chunk_ids:
                    skipped += 1
                    continue
                edge_type = data.get("edge_type", "unknown")
                triple = (source, target, edge_type)
                if triple in existing_triples:
                    continue
                session.add(
                    GraphEdgeRecord(
                        repository_id=repository_id,
                        source_chunk_id=source,
                        target_chunk_id=target,
                        edge_type=edge_type,
                    )
                )
                existing_triples.add(triple)
                stored += 1

        logger.info(
            "Graph stored: %d new edge(s), %d skipped (file-level or unstored chunk) for repository %s",
            stored, skipped, repository_id,
        )
        return stored

    # -- Load ------------------------------------------------------------------

    def load_repository(self, repository_id: str) -> RepositoryMetadata | None:
        """Load a previously stored repository's metadata.

        Args:
            repository_id: The repository's id, from `store_repository` or
                `compute_repository_id`.

        Returns:
            The stored `RepositoryMetadata`, or None if no repository with
            this id has been stored.

        Raises:
            DatabaseError: If the read fails.
        """
        with self.session_scope() as session:
            record = session.get(RepositoryRecord, repository_id)
            if record is None:
                return None
            metadata = RepositoryMetadata(
                name=record.repository_name,
                owner=record.owner,
                clone_url=record.clone_url,
                default_branch=record.default_branch,
                local_path=Path(record.local_path),
                last_updated=record.indexed_at,
                commit_hash=record.current_commit_hash,
            )

        logger.info("Repository loaded: %s/%s", metadata.owner, metadata.name)
        return metadata

    def load_chunks(self, repository_id: str) -> list[CodeChunk]:
        """Load every chunk stored for `repository_id`.

        Args:
            repository_id: The repository's id, from `store_repository` or
                `compute_repository_id`.

        Returns:
            One `CodeChunk` per stored row, in no particular order. Empty
            if nothing has been stored for `repository_id`.

        Raises:
            DatabaseError: If the read fails.
        """
        with self.session_scope() as session:
            rows = session.execute(
                select(CodeChunkRecord, SourceFileRecord.relative_path).join(
                    SourceFileRecord,
                    (SourceFileRecord.repository_id == CodeChunkRecord.repository_id)
                    & (SourceFileRecord.file_id == CodeChunkRecord.file_id),
                ).where(CodeChunkRecord.repository_id == repository_id)
            ).all()

            chunks = [
                CodeChunk(
                    chunk_id=uuid.UUID(record.chunk_id),
                    file_id=record.file_id,
                    file_path=relative_path,
                    language=record.language,
                    chunk_type=ChunkType(record.chunk_type),
                    function_name=record.function_name,
                    class_name=record.class_name,
                    parent_class=record.parent_class,
                    start_line=record.start_line,
                    end_line=record.end_line,
                    raw_code=record.raw_code,
                    parent_chunk_id=uuid.UUID(record.parent_chunk_id) if record.parent_chunk_id else None,
                )
                for record, relative_path in rows
            ]

        logger.info("Chunks loaded: %d for repository %s", len(chunks), repository_id)
        return chunks

    def load_graph(self, repository_id: str) -> nx.DiGraph:
        """Load the stored chunk-to-chunk graph for `repository_id`.

        Args:
            repository_id: The repository's id, from `store_repository` or
                `compute_repository_id`.

        Returns:
            A `networkx.DiGraph` with one node per stored chunk and one
            edge per stored `GraphEdgeRecord`. File-level nodes/edges are
            not reconstructed - see `store_graph`.

        Raises:
            DatabaseError: If the read fails.
        """
        graph = nx.DiGraph()
        with self.session_scope() as session:
            chunk_records = session.scalars(
                select(CodeChunkRecord).where(CodeChunkRecord.repository_id == repository_id)
            )
            for record in chunk_records:
                graph.add_node(
                    record.chunk_id,
                    node_kind="chunk",
                    chunk_id=record.chunk_id,
                    file_id=record.file_id,
                    chunk_type=record.chunk_type,
                    function_name=record.function_name,
                    class_name=record.class_name,
                    parent_class=record.parent_class,
                    start_line=record.start_line,
                    end_line=record.end_line,
                )

            edge_records = session.scalars(
                select(GraphEdgeRecord).where(GraphEdgeRecord.repository_id == repository_id)
            )
            for edge_record in edge_records:
                graph.add_edge(
                    edge_record.source_chunk_id,
                    edge_record.target_chunk_id,
                    edge_type=edge_record.edge_type,
                )

        logger.info(
            "Graph loaded for repository %s: %d node(s), %d edge(s)",
            repository_id, graph.number_of_nodes(), graph.number_of_edges(),
        )
        return graph

    # -- Embeddings (Phase 8) ---------------------------------------------------

    def get_embedded_chunk_ids(self, repository_id: str, model_name: str) -> set[str]:
        """List chunk ids already embedded by `model_name` for this repository.

        Used by `embedding.embedding_manager.EmbeddingManager` to skip
        chunks that do not need re-embedding.

        Args:
            repository_id: The owning repository's id.
            model_name: The embedding model whose prior runs to check.

        Returns:
            The set of `chunk_id` strings with an existing embedding row
            for `(repository_id, model_name)`.

        Raises:
            DatabaseError: If the read fails.
        """
        with self.session_scope() as session:
            return set(
                session.scalars(
                    select(EmbeddingRecord.chunk_id).where(
                        EmbeddingRecord.repository_id == repository_id,
                        EmbeddingRecord.model_name == model_name,
                    )
                )
            )

    def store_embeddings(
        self,
        repository_id: str,
        model_name: str,
        chunk_vectors: list[tuple[str, np.ndarray]],
        *,
        force: bool = False,
    ) -> int:
        """Persist one embedding vector per `(chunk_id, vector)` pair.

        Args:
            repository_id: The owning repository's id.
            model_name: The embedding model that produced `chunk_vectors`.
            chunk_vectors: Pairs of chunk id and its embedding vector.
            force: If True, overwrite an already-stored embedding for a
                chunk instead of skipping it (updates the existing row in
                place; never inserts a second row for the same
                `(repository_id, chunk_id, model_name)`).

        Returns:
            The number of rows inserted or (with `force=True`) updated.

        Raises:
            DatabaseError: If the write fails - including if a chunk_id
                does not correspond to a stored `CodeChunkRecord`, which
                rolls back every vector in this call, not just the
                offending one.
        """
        stored = 0
        with self.session_scope() as session:
            existing_records = {
                record.chunk_id: record
                for record in session.scalars(
                    select(EmbeddingRecord).where(
                        EmbeddingRecord.repository_id == repository_id,
                        EmbeddingRecord.model_name == model_name,
                    )
                )
            }

            for chunk_id, vector in chunk_vectors:
                vector = np.asarray(vector, dtype=np.float32)
                existing_record = existing_records.get(chunk_id)

                if existing_record is not None:
                    if not force:
                        continue
                    existing_record.embedding_dimension = int(vector.shape[0])
                    existing_record.embedding_blob = vector.tobytes()
                    existing_record.created_at = datetime.now(timezone.utc)
                    stored += 1
                    continue

                session.add(
                    EmbeddingRecord(
                        repository_id=repository_id,
                        chunk_id=chunk_id,
                        model_name=model_name,
                        embedding_dimension=int(vector.shape[0]),
                        embedding_blob=vector.tobytes(),
                        created_at=datetime.now(timezone.utc),
                    )
                )
                stored += 1

        logger.info(
            "Embeddings stored: %d for repository %s (model=%s)",
            stored, repository_id, model_name,
        )
        return stored

    def load_embeddings(self, repository_id: str, model_name: str) -> dict[str, np.ndarray]:
        """Load every embedding stored by `model_name` for this repository.

        Args:
            repository_id: The owning repository's id.
            model_name: The embedding model whose vectors to load.

        Returns:
            A mapping from `chunk_id` to its embedding vector (dtype
            float32, shape ``(embedding_dimension,)``).

        Raises:
            DatabaseError: If the read fails.
        """
        with self.session_scope() as session:
            rows = session.execute(
                select(
                    EmbeddingRecord.chunk_id,
                    EmbeddingRecord.embedding_blob,
                    EmbeddingRecord.embedding_dimension,
                ).where(
                    EmbeddingRecord.repository_id == repository_id,
                    EmbeddingRecord.model_name == model_name,
                )
            ).all()

        embeddings = {
            chunk_id: np.frombuffer(blob, dtype=np.float32).reshape(dimension)
            for chunk_id, blob, dimension in rows
        }
        logger.info(
            "Embeddings loaded: %d for repository %s (model=%s)",
            len(embeddings), repository_id, model_name,
        )
        return embeddings

    # -- Semantic cache (Phase 14) -----------------------------------------------

    def store_cache_entry(
        self,
        repository_id: str,
        query: str,
        query_embedding: np.ndarray,
        response: str,
        retrieved_chunk_ids: list[str],
    ) -> int:
        """Insert a new cache entry, or overwrite the existing one for an exact query repeat.

        Args:
            repository_id: The repository this query was answered for.
            query: The query text this entry is cached under.
            query_embedding: `query`'s embedding vector.
            response: The generated response to cache.
            retrieved_chunk_ids: chunk_ids of the `RankedChunk`s used to
                produce `response`.

        Returns:
            The stored entry's `cache_id` (existing, if this exact query
            was already cached for `repository_id`; newly assigned
            otherwise).

        Raises:
            DatabaseError: If the write fails.
        """
        vector = np.asarray(query_embedding, dtype=np.float32)
        with self.session_scope() as session:
            existing = session.execute(
                select(SemanticCacheRecord).where(
                    SemanticCacheRecord.repository_id == repository_id,
                    SemanticCacheRecord.query == query,
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.query_embedding = vector.tobytes()
                existing.embedding_dimension = int(vector.shape[0])
                existing.response = response
                existing.retrieved_chunk_ids = json.dumps(retrieved_chunk_ids)
                existing.created_at = datetime.now(timezone.utc)
                session.flush()
                cache_id = existing.cache_id
                logger.info("Cache overwrite for repository %s: cache_id=%d", repository_id, cache_id)
            else:
                record = SemanticCacheRecord(
                    repository_id=repository_id,
                    query=query,
                    query_embedding=vector.tobytes(),
                    embedding_dimension=int(vector.shape[0]),
                    response=response,
                    retrieved_chunk_ids=json.dumps(retrieved_chunk_ids),
                    created_at=datetime.now(timezone.utc),
                )
                session.add(record)
                session.flush()
                cache_id = record.cache_id
                logger.info("Cache insert for repository %s: cache_id=%d", repository_id, cache_id)

        return cache_id

    def load_cache_entries(self, repository_id: str) -> list[CacheEntry]:
        """Load every cache entry stored for `repository_id`.

        Args:
            repository_id: The repository whose cache entries to load.

        Returns:
            One `CacheEntry` per stored row, in no particular order.
            Empty if nothing has been cached for `repository_id`.

        Raises:
            DatabaseError: If the read fails.
        """
        with self.session_scope() as session:
            records = session.scalars(
                select(SemanticCacheRecord).where(SemanticCacheRecord.repository_id == repository_id)
            ).all()

            entries = [
                CacheEntry(
                    cache_id=record.cache_id,
                    repository_id=record.repository_id,
                    query=record.query,
                    query_embedding=np.frombuffer(
                        record.query_embedding, dtype=np.float32
                    ).reshape(record.embedding_dimension),
                    response=record.response,
                    retrieved_chunk_ids=json.loads(record.retrieved_chunk_ids),
                    created_at=record.created_at,
                )
                for record in records
            ]

        logger.info("Cache entries loaded: %d for repository %s", len(entries), repository_id)
        return entries

    def clear_cache(self, repository_id: str) -> int:
        """Delete every cache entry stored for `repository_id`.

        Args:
            repository_id: The repository whose cache entries to clear.

        Returns:
            The number of entries deleted.

        Raises:
            DatabaseError: If the deletion fails.
        """
        with self.session_scope() as session:
            records = session.scalars(
                select(SemanticCacheRecord).where(SemanticCacheRecord.repository_id == repository_id)
            ).all()
            count = len(records)
            for record in records:
                session.delete(record)

        logger.info("Cache cleared for repository %s: %d entries removed", repository_id, count)
        return count
