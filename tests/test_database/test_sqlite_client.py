"""Tests for database.sqlite_client.DatabaseManager."""

from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from sqlalchemy import func, inspect, select

from core.exceptions import DatabaseError
from database.models import (
    CodeChunkRecord,
    EmbeddingRecord,
    GraphEdgeRecord,
    RepositoryRecord,
    SemanticCacheRecord,
    SourceFileRecord,
)
from database.sqlite_client import DatabaseManager
from ingestion.chunker import ChunkingResult
from ingestion.deterministic_ids import compute_file_id
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk


def _repository_metadata(owner: str = "acme", name: str = "demo") -> RepositoryMetadata:
    return RepositoryMetadata(
        name=name,
        owner=owner,
        clone_url=f"https://github.com/{owner}/{name}.git",
        default_branch="main",
        local_path=Path(f"/tmp/{owner}_{name}"),
        last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        commit_hash="a" * 40,
    )


def _source_file(relative_path: str = "src/main.py", language: str = "python") -> SourceFile:
    path = Path(relative_path)
    return SourceFile(
        absolute_path=Path("/repo") / path,
        relative_path=path,
        language=language,
        extension=path.suffix,
        size_bytes=123,
    )


def _chunk(
    file_id: str,
    file_path: str,
    *,
    chunk_type: ChunkType = ChunkType.FUNCTION,
    function_name: str | None = "foo",
    start_line: int = 1,
    end_line: int = 5,
    parent_chunk_id: uuid.UUID | None = None,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid4(),
        file_id=file_id,
        file_path=file_path,
        language="python",
        chunk_type=chunk_type,
        function_name=function_name,
        class_name=None,
        parent_class=None,
        start_line=start_line,
        end_line=end_line,
        raw_code=f"def {function_name}(): ...",
        parent_chunk_id=parent_chunk_id,
    )


def _row_count(manager: DatabaseManager, model: type) -> int:
    with manager._session_factory() as session:  # noqa: SLF001 - white-box test helper
        return session.scalar(select(func.count()).select_from(model))


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.initialize_database()
    return manager


def _seed_single_chunk(db: DatabaseManager) -> tuple[str, CodeChunk]:
    repository_id = db.store_repository(_repository_metadata())
    source_file = _source_file("src/a.py")
    db.store_source_files(repository_id, [source_file])
    file_id = compute_file_id(source_file.relative_path)

    chunk = _chunk(file_id, "src/a.py", function_name="a", start_line=1, end_line=2)
    result = ChunkingResult(ast_chunks=[chunk], sliding_chunks=[], parent_chunks=[])
    db.store_chunks(repository_id, {"src/a.py": result})
    return repository_id, chunk


class TestInitializeDatabase:
    def test_creates_database_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "repomind.db"
        manager = DatabaseManager(db_path=db_path)
        manager.initialize_database()
        assert db_path.exists()

    def test_creates_expected_tables(self, db: DatabaseManager) -> None:
        table_names = set(inspect(db._engine).get_table_names())  # noqa: SLF001
        assert table_names == {
            "repositories", "source_files", "code_chunks", "graph_edges", "embeddings",
            "semantic_cache",
        }

    def test_is_idempotent(self, db: DatabaseManager) -> None:
        db.initialize_database()
        db.initialize_database()


class TestStoreRepository:
    def test_inserts_new_repository(self, db: DatabaseManager) -> None:
        repository = _repository_metadata()
        repository_id = db.store_repository(repository)

        assert repository_id == DatabaseManager.compute_repository_id(repository.owner, repository.name)
        assert _row_count(db, RepositoryRecord) == 1

    def test_storing_same_repository_twice_does_not_duplicate(self, db: DatabaseManager) -> None:
        repository = _repository_metadata()
        first_id = db.store_repository(repository)
        second_id = db.store_repository(repository)

        assert first_id == second_id
        assert _row_count(db, RepositoryRecord) == 1

    def test_storing_same_repository_again_updates_fields(self, db: DatabaseManager) -> None:
        repository = _repository_metadata()
        repository_id = db.store_repository(repository)

        updated = dataclasses.replace(repository, commit_hash="b" * 40)
        db.store_repository(updated)

        loaded = db.load_repository(repository_id)
        assert loaded is not None
        assert loaded.commit_hash == "b" * 40


class TestStoreSourceFiles:
    def test_inserts_new_files(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        files = [_source_file("src/a.py"), _source_file("src/b.py")]

        stored = db.store_source_files(repository_id, files)

        assert stored == 2
        assert _row_count(db, SourceFileRecord) == 2

    def test_duplicate_files_are_skipped(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        files = [_source_file("src/a.py")]
        db.store_source_files(repository_id, files)

        second_call_stored = db.store_source_files(repository_id, files)

        assert second_call_stored == 0
        assert _row_count(db, SourceFileRecord) == 1


class TestStoreChunks:
    def test_inserts_ast_and_parent_chunks(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        source_file = _source_file("src/a.py")
        db.store_source_files(repository_id, [source_file])
        file_id = compute_file_id(source_file.relative_path)

        parent = _chunk(
            file_id, "src/a.py", chunk_type=ChunkType.PARENT, function_name=None, start_line=1, end_line=20
        )
        ast_chunk = _chunk(file_id, "src/a.py", start_line=2, end_line=5, parent_chunk_id=parent.chunk_id)
        result = ChunkingResult(ast_chunks=[ast_chunk], sliding_chunks=[], parent_chunks=[parent])

        stored = db.store_chunks(repository_id, {"src/a.py": result})

        assert stored == 2
        assert _row_count(db, CodeChunkRecord) == 2

    def test_duplicate_chunks_are_skipped(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        source_file = _source_file("src/a.py")
        db.store_source_files(repository_id, [source_file])
        file_id = compute_file_id(source_file.relative_path)

        chunk = _chunk(file_id, "src/a.py")
        result = ChunkingResult(ast_chunks=[chunk], sliding_chunks=[], parent_chunks=[])
        db.store_chunks(repository_id, {"src/a.py": result})

        second_call_stored = db.store_chunks(repository_id, {"src/a.py": result})

        assert second_call_stored == 0
        assert _row_count(db, CodeChunkRecord) == 1

    def test_chunk_referencing_unstored_file_rolls_back_entire_batch(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        # Deliberately skip store_source_files, so this file_id has no
        # matching SourceFileRecord and violates the chunk -> file FK.
        good_chunk = _chunk("bogus-file-id", "src/a.py", function_name="a", start_line=1, end_line=2)
        bad_chunk = _chunk("bogus-file-id", "src/a.py", function_name="b", start_line=3, end_line=4)
        result = ChunkingResult(ast_chunks=[good_chunk, bad_chunk], sliding_chunks=[], parent_chunks=[])

        with pytest.raises(DatabaseError):
            db.store_chunks(repository_id, {"src/a.py": result})

        assert _row_count(db, CodeChunkRecord) == 0


class TestStoreGraph:
    def _store_two_chunks(self, db: DatabaseManager) -> tuple[str, CodeChunk, CodeChunk, str]:
        repository_id = db.store_repository(_repository_metadata())
        source_file = _source_file("src/a.py")
        db.store_source_files(repository_id, [source_file])
        file_id = compute_file_id(source_file.relative_path)

        chunk_a = _chunk(file_id, "src/a.py", function_name="a", start_line=1, end_line=2)
        chunk_b = _chunk(file_id, "src/a.py", function_name="b", start_line=3, end_line=4)
        result = ChunkingResult(ast_chunks=[chunk_a, chunk_b], sliding_chunks=[], parent_chunks=[])
        db.store_chunks(repository_id, {"src/a.py": result})
        return repository_id, chunk_a, chunk_b, file_id

    def test_stores_only_chunk_to_chunk_edges(self, db: DatabaseManager) -> None:
        repository_id, chunk_a, chunk_b, file_id = self._store_two_chunks(db)

        graph = nx.DiGraph()
        graph.add_node(str(chunk_a.chunk_id), node_kind="chunk")
        graph.add_node(str(chunk_b.chunk_id), node_kind="chunk")
        graph.add_node(file_id, node_kind="file")
        graph.add_edge(str(chunk_a.chunk_id), str(chunk_b.chunk_id), edge_type="function_call")
        # File -> chunk edge: source is not a stored chunk, must be skipped.
        graph.add_edge(file_id, str(chunk_a.chunk_id), edge_type="imports")

        stored = db.store_graph(repository_id, graph)

        assert stored == 1
        assert _row_count(db, GraphEdgeRecord) == 1

    def test_duplicate_edges_are_skipped(self, db: DatabaseManager) -> None:
        repository_id, chunk_a, chunk_b, _file_id = self._store_two_chunks(db)

        graph = nx.DiGraph()
        graph.add_edge(str(chunk_a.chunk_id), str(chunk_b.chunk_id), edge_type="function_call")
        db.store_graph(repository_id, graph)

        second_call_stored = db.store_graph(repository_id, graph)

        assert second_call_stored == 0
        assert _row_count(db, GraphEdgeRecord) == 1


class TestLoadRepository:
    def test_returns_none_when_missing(self, db: DatabaseManager) -> None:
        assert db.load_repository("nonexistent") is None

    def test_round_trip(self, db: DatabaseManager) -> None:
        repository = _repository_metadata()
        repository_id = db.store_repository(repository)

        loaded = db.load_repository(repository_id)

        assert loaded is not None
        assert loaded.owner == repository.owner
        assert loaded.name == repository.name
        assert loaded.clone_url == repository.clone_url
        assert loaded.default_branch == repository.default_branch
        assert loaded.commit_hash == repository.commit_hash


class TestLoadChunks:
    def test_returns_empty_list_when_nothing_stored(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        assert db.load_chunks(repository_id) == []

    def test_round_trip(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        source_file = _source_file("src/a.py")
        db.store_source_files(repository_id, [source_file])
        file_id = compute_file_id(source_file.relative_path)

        chunk = _chunk(file_id, "src/a.py", function_name="a", start_line=1, end_line=2)
        result = ChunkingResult(ast_chunks=[chunk], sliding_chunks=[], parent_chunks=[])
        db.store_chunks(repository_id, {"src/a.py": result})

        loaded = db.load_chunks(repository_id)

        assert len(loaded) == 1
        loaded_chunk = loaded[0]
        assert loaded_chunk.chunk_id == chunk.chunk_id
        assert loaded_chunk.file_path == "src/a.py"
        assert loaded_chunk.function_name == "a"
        assert loaded_chunk.start_line == 1
        assert loaded_chunk.end_line == 2
        assert loaded_chunk.raw_code == chunk.raw_code


class TestLoadGraph:
    def test_returns_empty_graph_when_nothing_stored(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        graph = db.load_graph(repository_id)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_round_trip(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        source_file = _source_file("src/a.py")
        db.store_source_files(repository_id, [source_file])
        file_id = compute_file_id(source_file.relative_path)

        chunk_a = _chunk(file_id, "src/a.py", function_name="a", start_line=1, end_line=2)
        chunk_b = _chunk(file_id, "src/a.py", function_name="b", start_line=3, end_line=4)
        result = ChunkingResult(ast_chunks=[chunk_a, chunk_b], sliding_chunks=[], parent_chunks=[])
        db.store_chunks(repository_id, {"src/a.py": result})

        graph = nx.DiGraph()
        graph.add_edge(str(chunk_a.chunk_id), str(chunk_b.chunk_id), edge_type="function_call")
        db.store_graph(repository_id, graph)

        loaded = db.load_graph(repository_id)

        assert set(loaded.nodes) == {str(chunk_a.chunk_id), str(chunk_b.chunk_id)}
        assert loaded.has_edge(str(chunk_a.chunk_id), str(chunk_b.chunk_id))
        assert loaded[str(chunk_a.chunk_id)][str(chunk_b.chunk_id)]["edge_type"] == "function_call"


class TestStoreEmbeddings:
    def test_inserts_new_embedding(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        vector = np.ones(384, dtype=np.float32)

        stored = db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), vector)])

        assert stored == 1
        assert _row_count(db, EmbeddingRecord) == 1

    def test_duplicate_embedding_is_skipped_without_force(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        vector = np.ones(384, dtype=np.float32)
        db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), vector)])

        second_call_stored = db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), vector)])

        assert second_call_stored == 0
        assert _row_count(db, EmbeddingRecord) == 1

    def test_force_overwrites_existing_embedding_in_place(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), np.ones(384, dtype=np.float32))])

        new_vector = np.full(384, 2.0, dtype=np.float32)
        second_call_stored = db.store_embeddings(
            repository_id, "model-x", [(str(chunk.chunk_id), new_vector)], force=True
        )

        assert second_call_stored == 1
        assert _row_count(db, EmbeddingRecord) == 1
        loaded = db.load_embeddings(repository_id, "model-x")
        assert np.array_equal(loaded[str(chunk.chunk_id)], new_vector)

    def test_same_chunk_can_have_embeddings_from_two_models(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        db.store_embeddings(repository_id, "model-a", [(str(chunk.chunk_id), np.ones(384, dtype=np.float32))])
        db.store_embeddings(repository_id, "model-b", [(str(chunk.chunk_id), np.ones(768, dtype=np.float32))])

        assert _row_count(db, EmbeddingRecord) == 2

    def test_embedding_referencing_unstored_chunk_rolls_back_entire_batch(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.ones(384, dtype=np.float32)

        with pytest.raises(DatabaseError):
            db.store_embeddings(repository_id, "model-x", [("bogus-chunk-id", vector)])

        assert _row_count(db, EmbeddingRecord) == 0


class TestGetEmbeddedChunkIds:
    def test_returns_empty_set_when_nothing_stored(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        assert db.get_embedded_chunk_ids(repository_id, "model-x") == set()

    def test_returns_stored_chunk_ids_for_that_model_only(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), np.ones(384, dtype=np.float32))])

        assert db.get_embedded_chunk_ids(repository_id, "model-x") == {str(chunk.chunk_id)}
        assert db.get_embedded_chunk_ids(repository_id, "model-y") == set()


class TestLoadEmbeddings:
    def test_returns_empty_dict_when_nothing_stored(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        assert db.load_embeddings(repository_id, "model-x") == {}

    def test_round_trip_preserves_vector_values_and_dimension(self, db: DatabaseManager) -> None:
        repository_id, chunk = _seed_single_chunk(db)
        vector = np.linspace(0, 1, num=384, dtype=np.float32)
        db.store_embeddings(repository_id, "model-x", [(str(chunk.chunk_id), vector)])

        loaded = db.load_embeddings(repository_id, "model-x")

        assert set(loaded) == {str(chunk.chunk_id)}
        loaded_vector = loaded[str(chunk.chunk_id)]
        assert loaded_vector.shape == (384,)
        assert loaded_vector.dtype == np.float32
        assert np.allclose(loaded_vector, vector)


class TestStoreCacheEntry:
    def test_inserts_new_entry(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.ones(8, dtype=np.float32)

        cache_id = db.store_cache_entry(
            repository_id, "how does auth work", vector, "Auth works via...", ["chunk-1", "chunk-2"]
        )

        assert isinstance(cache_id, int)
        assert _row_count(db, SemanticCacheRecord) == 1

    def test_storing_same_query_again_overwrites(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.ones(8, dtype=np.float32)
        first_id = db.store_cache_entry(repository_id, "q", vector, "first response", ["a"])

        new_vector = np.full(8, 2.0, dtype=np.float32)
        second_id = db.store_cache_entry(repository_id, "q", new_vector, "second response", ["b"])

        assert first_id == second_id
        assert _row_count(db, SemanticCacheRecord) == 1
        entries = db.load_cache_entries(repository_id)
        assert entries[0].response == "second response"
        assert entries[0].retrieved_chunk_ids == ["b"]

    def test_different_queries_create_separate_entries(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.ones(8, dtype=np.float32)
        db.store_cache_entry(repository_id, "q1", vector, "r1", ["a"])
        db.store_cache_entry(repository_id, "q2", vector, "r2", ["b"])

        assert _row_count(db, SemanticCacheRecord) == 2


class TestLoadCacheEntries:
    def test_returns_empty_list_when_nothing_cached(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        assert db.load_cache_entries(repository_id) == []

    def test_round_trip_preserves_fields(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.linspace(0, 1, num=8, dtype=np.float32)
        db.store_cache_entry(repository_id, "q", vector, "response text", ["chunk-1", "chunk-2"])

        entries = db.load_cache_entries(repository_id)

        assert len(entries) == 1
        entry = entries[0]
        assert entry.query == "q"
        assert entry.response == "response text"
        assert entry.retrieved_chunk_ids == ["chunk-1", "chunk-2"]
        assert np.allclose(entry.query_embedding, vector)

    def test_isolated_by_repository(self, db: DatabaseManager) -> None:
        repo_a = db.store_repository(_repository_metadata(owner="acme", name="a"))
        repo_b = db.store_repository(_repository_metadata(owner="acme", name="b"))
        vector = np.ones(8, dtype=np.float32)
        db.store_cache_entry(repo_a, "q", vector, "response a", ["a"])
        db.store_cache_entry(repo_b, "q", vector, "response b", ["b"])

        entries_a = db.load_cache_entries(repo_a)
        assert len(entries_a) == 1
        assert entries_a[0].response == "response a"


class TestClearCache:
    def test_removes_all_entries_for_repository(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        vector = np.ones(8, dtype=np.float32)
        db.store_cache_entry(repository_id, "q1", vector, "r1", ["a"])
        db.store_cache_entry(repository_id, "q2", vector, "r2", ["b"])

        removed = db.clear_cache(repository_id)

        assert removed == 2
        assert db.load_cache_entries(repository_id) == []

    def test_does_not_affect_other_repositories(self, db: DatabaseManager) -> None:
        repo_a = db.store_repository(_repository_metadata(owner="acme", name="a"))
        repo_b = db.store_repository(_repository_metadata(owner="acme", name="b"))
        vector = np.ones(8, dtype=np.float32)
        db.store_cache_entry(repo_a, "q", vector, "r", ["a"])
        db.store_cache_entry(repo_b, "q", vector, "r", ["b"])

        db.clear_cache(repo_a)

        assert db.load_cache_entries(repo_a) == []
        assert len(db.load_cache_entries(repo_b)) == 1

    def test_returns_zero_when_nothing_to_clear(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        assert db.clear_cache(repository_id) == 0
