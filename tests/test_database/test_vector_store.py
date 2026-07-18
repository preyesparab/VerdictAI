"""Tests for database.vector_store.FaissIndexManager.

Embeddings are inserted directly via `DatabaseManager.store_embeddings`
with synthetic numpy vectors rather than real model inference - this
module only builds/searches/persists a FAISS index over whatever vectors
are already in SQLite, so no embedding model needs to run for these tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from core.constants import DEFAULT_MINILM_MODEL
from core.exceptions import RetrievalError
from database.sqlite_client import DatabaseManager
from database.vector_store import FaissIndexManager
from ingestion.chunker import ChunkingResult
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk

_TEST_MODEL = "test-model"


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


def _source_file(relative_path: str = "src/a.py") -> SourceFile:
    path = Path(relative_path)
    return SourceFile(
        absolute_path=Path("/repo") / path,
        relative_path=path,
        language="python",
        extension=path.suffix,
        size_bytes=10,
    )


def _chunk(function_name: str, start_line: int, end_line: int) -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid4(),
        file_id="f",
        file_path="src/a.py",
        language="python",
        chunk_type=ChunkType.FUNCTION,
        function_name=function_name,
        class_name=None,
        parent_class=None,
        start_line=start_line,
        end_line=end_line,
        raw_code=f"def {function_name}(): ...",
    )


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.initialize_database()
    return manager


@pytest.fixture
def indexes_dir(tmp_path: Path) -> Path:
    return tmp_path / "indexes"


def _seed_chunks_with_embeddings(
    db: DatabaseManager, model_name: str, count: int, dimension: int
) -> tuple[str, list[CodeChunk], dict[str, np.ndarray]]:
    repository_id = db.store_repository(_repository_metadata())
    source_file = _source_file()
    db.store_source_files(repository_id, [source_file])

    chunks = [_chunk(f"fn{i}", start_line=i + 1, end_line=i + 1) for i in range(count)]
    result = ChunkingResult(ast_chunks=chunks, sliding_chunks=[], parent_chunks=[])
    db.store_chunks(repository_id, {"src/a.py": result})

    rng = np.random.default_rng(seed=42)
    vectors = {str(chunk.chunk_id): rng.random(dimension).astype(np.float32) for chunk in chunks}
    db.store_embeddings(repository_id, model_name, list(vectors.items()))
    return repository_id, chunks, vectors


class TestBuildIndex:
    def test_creates_index_from_stored_embeddings(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, _chunks, _vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=3, dimension=8)
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)

        count = manager.build_index(repository_id)

        assert count == 3

    def test_handles_empty_repository_gracefully(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=DEFAULT_MINILM_MODEL)

        count = manager.build_index(repository_id)

        assert count == 0

    def test_raises_on_dimension_mismatch(self, db: DatabaseManager, indexes_dir: Path) -> None:
        # DEFAULT_MINILM_MODEL expects 384 dimensions; 10 is deliberately wrong.
        repository_id, _chunks, _vectors = _seed_chunks_with_embeddings(
            db, DEFAULT_MINILM_MODEL, count=1, dimension=10
        )
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=DEFAULT_MINILM_MODEL)

        with pytest.raises(RetrievalError):
            manager.build_index(repository_id)

    def test_ignores_duplicate_embedding_inserts(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, chunks, vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=1, dimension=8)
        chunk_id = str(chunks[0].chunk_id)
        # Re-inserting the same (repository_id, chunk_id, model_name) without
        # force is a no-op at the DB layer (Phase 7/8); confirm it stays a
        # no-op for the resulting index too (1 vector, not 2).
        db.store_embeddings(repository_id, _TEST_MODEL, [(chunk_id, vectors[chunk_id])])
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)

        count = manager.build_index(repository_id)

        assert count == 1


class TestSearch:
    def test_returns_results_sorted_by_similarity(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        db.store_source_files(repository_id, [_source_file()])
        chunk_a = _chunk("a", 1, 1)
        chunk_b = _chunk("b", 2, 2)
        chunk_c = _chunk("c", 3, 3)
        result = ChunkingResult(ast_chunks=[chunk_a, chunk_b, chunk_c], sliding_chunks=[], parent_chunks=[])
        db.store_chunks(repository_id, {"src/a.py": result})

        db.store_embeddings(
            repository_id,
            _TEST_MODEL,
            [
                (str(chunk_a.chunk_id), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                (str(chunk_b.chunk_id), np.array([0.9, 0.1, 0.0], dtype=np.float32)),
                (str(chunk_c.chunk_id), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ],
        )

        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        manager.build_index(repository_id)

        results = manager.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)

        assert [r.chunk_id for r in results] == [
            str(chunk_a.chunk_id), str(chunk_b.chunk_id), str(chunk_c.chunk_id),
        ]
        assert results[0].score > results[1].score > results[2].score
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_respects_top_k(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, _chunks, vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=5, dimension=8)
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        manager.build_index(repository_id)

        results = manager.search(next(iter(vectors.values())), top_k=2)

        assert len(results) == 2

    def test_raises_if_no_index_built_or_loaded(self, db: DatabaseManager, indexes_dir: Path) -> None:
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)

        with pytest.raises(RetrievalError):
            manager.search(np.zeros(8, dtype=np.float32), top_k=1)

    def test_raises_on_query_dimension_mismatch(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, _chunks, _vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=2, dimension=8)
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        manager.build_index(repository_id)

        with pytest.raises(RetrievalError):
            manager.search(np.zeros(4, dtype=np.float32), top_k=1)

    def test_empty_index_search_returns_empty_list(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=DEFAULT_MINILM_MODEL)
        manager.build_index(repository_id)

        results = manager.search(np.zeros(384, dtype=np.float32), top_k=5)

        assert results == []

    def test_raises_on_non_positive_top_k(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, _chunks, vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=2, dimension=8)
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        manager.build_index(repository_id)

        with pytest.raises(RetrievalError):
            manager.search(next(iter(vectors.values())), top_k=0)


class TestSaveAndLoadIndex:
    def test_save_creates_index_and_mapping_files(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id, _chunks, _vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=2, dimension=8)
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        manager.build_index(repository_id)

        index_path = manager.save_index(repository_id)

        repository = db.load_repository(repository_id)
        expected_stem = f"{repository.owner}_{repository.name}"
        assert index_path == indexes_dir / f"{expected_stem}.faiss"
        assert index_path.exists()
        assert (indexes_dir / f"{expected_stem}.json").exists()

    def test_load_index_restores_search_capability_in_a_new_manager(
        self, db: DatabaseManager, indexes_dir: Path
    ) -> None:
        repository_id, _chunks, vectors = _seed_chunks_with_embeddings(db, _TEST_MODEL, count=3, dimension=8)
        writer = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        writer.build_index(repository_id)
        writer.save_index(repository_id)

        reader = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)
        loaded_count = reader.load_index(repository_id)

        assert loaded_count == 3
        query_chunk_id, query_vector = next(iter(vectors.items()))
        results = reader.search(query_vector, top_k=1)
        assert results[0].chunk_id == query_chunk_id
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_load_index_raises_when_no_saved_index_exists(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)

        with pytest.raises(RetrievalError):
            manager.load_index(repository_id)

    def test_save_without_building_or_loading_raises(self, db: DatabaseManager, indexes_dir: Path) -> None:
        manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=_TEST_MODEL)

        with pytest.raises(RetrievalError):
            manager.save_index("nonexistent-repository-id")
