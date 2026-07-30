"""Tests for retrieval.sparse_retriever.BM25Manager.

Uses small synthetic repositories (a handful of hand-written chunks)
seeded directly through `DatabaseManager`, rather than real repositories -
no parsing or embedding model involved.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.exceptions import RetrievalError
from database.sqlite_client import DatabaseManager
from ingestion.chunker import ChunkingResult
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk
from retrieval.sparse_retriever import BM25Manager


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


def _chunk(
    function_name: str | None,
    raw_code: str,
    *,
    chunk_type: ChunkType = ChunkType.FUNCTION,
    class_name: str | None = None,
    start_line: int = 1,
    end_line: int = 5,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid4(),
        file_id="f",
        file_path="src/a.py",
        language="python",
        chunk_type=chunk_type,
        function_name=function_name,
        class_name=class_name,
        parent_class=None,
        start_line=start_line,
        end_line=end_line,
        raw_code=raw_code,
    )


@pytest.fixture
def db(pg_schema: str) -> Iterator[DatabaseManager]:
    manager = DatabaseManager(schema=pg_schema)
    manager.initialize_database()
    yield manager
    manager.drop_schema()


@pytest.fixture
def indexes_dir(tmp_path: Path) -> Path:
    return tmp_path / "indexes"


def _seed_repository(
    db: DatabaseManager,
    ast_chunks: list[CodeChunk],
    sliding_chunks: list[CodeChunk] | None = None,
    parent_chunks: list[CodeChunk] | None = None,
) -> str:
    repository_id = db.store_repository(_repository_metadata())
    db.store_source_files(repository_id, [_source_file()])
    result = ChunkingResult(
        ast_chunks=ast_chunks, sliding_chunks=sliding_chunks or [], parent_chunks=parent_chunks or []
    )
    db.store_chunks(repository_id, {"src/a.py": result})
    return repository_id


class TestBuildIndex:
    def test_creates_index_from_ast_chunks(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunks = [
            _chunk("fetch_user", "def fetch_user(user_id): return db.get(user_id)"),
            _chunk("delete_user", "def delete_user(user_id): db.remove(user_id)", start_line=6, end_line=10),
        ]
        repository_id = _seed_repository(db, chunks)
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        count = manager.build_index(repository_id)

        assert count == 2

    def test_only_ast_chunks_are_indexed(self, db: DatabaseManager, indexes_dir: Path) -> None:
        ast_chunk = _chunk("fetch_user", "def fetch_user(): ...")
        sliding_chunk = _chunk(None, "some sliding window text", chunk_type=ChunkType.SLIDING)
        parent_chunk = _chunk(None, "some parent context text", chunk_type=ChunkType.PARENT)
        repository_id = _seed_repository(db, [ast_chunk], sliding_chunks=[sliding_chunk], parent_chunks=[parent_chunk])
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        count = manager.build_index(repository_id)

        assert count == 1

    def test_handles_empty_repository_gracefully(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        count = manager.build_index(repository_id)

        assert count == 0

    def test_ignores_duplicate_chunks_defensively(
        self, db: DatabaseManager, indexes_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunk = _chunk("fetch_user", "def fetch_user(): ...")
        repository_id = _seed_repository(db, [chunk])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        monkeypatch.setattr(db, "load_chunks", lambda repository_id: [chunk, chunk])

        count = manager.build_index(repository_id)

        assert count == 1


class TestSearch:
    # Each test seeds >= 3 documents deliberately: BM25's IDF term
    # `log((N - n + 0.5) / (n + 0.5))` is exactly 0 for a term appearing in
    # precisely 1 of 2 documents, which would make these assertions
    # vacuously true regardless of whether search actually discriminates.
    def test_keyword_search_finds_matching_chunk(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunk_a = _chunk("fetch_user", "def fetch_user(user_id): return database.get(user_id)")
        chunk_b = _chunk(
            "delete_order", "def delete_order(order_id): database.remove(order_id)", start_line=6, end_line=10
        )
        chunk_c = _chunk(
            "cancel_shipment", "def cancel_shipment(shipment_id): database.cancel(shipment_id)",
            start_line=11, end_line=15,
        )
        repository_id = _seed_repository(db, [chunk_a, chunk_b, chunk_c])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        results = manager.search("fetch user", top_k=5)

        assert results[0].chunk_id == str(chunk_a.chunk_id)
        assert results[0].score > results[1].score

    def test_identifier_search_matches_camel_case_function_name(
        self, db: DatabaseManager, indexes_dir: Path
    ) -> None:
        chunk_a = _chunk("fetchUserByID", "def fetchUserByID(id): return repo.find(id)")
        chunk_b = _chunk("deleteOrder", "def deleteOrder(id): repo.remove(id)", start_line=6, end_line=10)
        chunk_c = _chunk(
            "cancelShipment", "def cancelShipment(id): repo.cancel(id)", start_line=11, end_line=15
        )
        repository_id = _seed_repository(db, [chunk_a, chunk_b, chunk_c])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        results = manager.search("fetch user by id", top_k=5)

        assert results[0].chunk_id == str(chunk_a.chunk_id)

    def test_results_sorted_by_score_descending(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunk_a = _chunk("fetch_user", "def fetch_user(user_id): return db.get(user_id)  # fetch user record")
        chunk_b = _chunk("fetch_order", "def fetch_order(order_id): return db.get(order_id)", start_line=6, end_line=10)
        chunk_c = _chunk("delete_order", "def delete_order(order_id): db.remove(order_id)", start_line=11, end_line=15)
        repository_id = _seed_repository(db, [chunk_a, chunk_b, chunk_c])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        results = manager.search("fetch user", top_k=3)

        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunks = [
            _chunk(f"fn{i}", f"def fn{i}(): return {i}", start_line=i, end_line=i) for i in range(1, 6)
        ]
        repository_id = _seed_repository(db, chunks)
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        results = manager.search("fn", top_k=2)

        assert len(results) == 2

    def test_empty_query_returns_no_results(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunk = _chunk("fetch_user", "def fetch_user(): ...")
        repository_id = _seed_repository(db, [chunk])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        assert manager.search("", top_k=5) == []
        assert manager.search("   !!!   ", top_k=5) == []

    def test_search_on_empty_repository_returns_no_results(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        assert manager.search("fetch user", top_k=5) == []

    def test_raises_if_no_index_built_or_loaded(self, db: DatabaseManager, indexes_dir: Path) -> None:
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        with pytest.raises(RetrievalError):
            manager.search("fetch user", top_k=5)

    def test_raises_on_non_positive_top_k(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunk = _chunk("fetch_user", "def fetch_user(): ...")
        repository_id = _seed_repository(db, [chunk])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        with pytest.raises(RetrievalError):
            manager.search("fetch user", top_k=0)


class TestSaveAndLoadIndex:
    def test_save_creates_index_file(self, db: DatabaseManager, indexes_dir: Path) -> None:
        chunk = _chunk("fetch_user", "def fetch_user(): ...")
        repository_id = _seed_repository(db, [chunk])
        manager = BM25Manager(db, indexes_dir=indexes_dir)
        manager.build_index(repository_id)

        index_path = manager.save_index(repository_id)

        repository = db.load_repository(repository_id)
        assert index_path == indexes_dir / f"{repository.owner}_{repository.name}_bm25.pkl"
        assert index_path.exists()

    def test_load_index_restores_search_capability_in_a_new_manager(
        self, db: DatabaseManager, indexes_dir: Path
    ) -> None:
        chunk_a = _chunk("fetch_user", "def fetch_user(user_id): return db.get(user_id)")
        chunk_b = _chunk(
            "delete_order", "def delete_order(order_id): db.remove(order_id)", start_line=6, end_line=10
        )
        chunk_c = _chunk(
            "cancel_shipment", "def cancel_shipment(shipment_id): db.cancel(shipment_id)",
            start_line=11, end_line=15,
        )
        repository_id = _seed_repository(db, [chunk_a, chunk_b, chunk_c])
        writer = BM25Manager(db, indexes_dir=indexes_dir)
        writer.build_index(repository_id)
        writer.save_index(repository_id)

        reader = BM25Manager(db, indexes_dir=indexes_dir)
        loaded_count = reader.load_index(repository_id)

        assert loaded_count == 3
        results = reader.search("fetch user", top_k=1)
        assert results[0].chunk_id == str(chunk_a.chunk_id)

    def test_save_and_load_empty_index_round_trip(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        writer = BM25Manager(db, indexes_dir=indexes_dir)
        writer.build_index(repository_id)
        writer.save_index(repository_id)

        reader = BM25Manager(db, indexes_dir=indexes_dir)
        loaded_count = reader.load_index(repository_id)

        assert loaded_count == 0
        assert reader.search("anything", top_k=5) == []

    def test_load_index_raises_when_no_saved_index_exists(self, db: DatabaseManager, indexes_dir: Path) -> None:
        repository_id = db.store_repository(_repository_metadata())
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        with pytest.raises(RetrievalError):
            manager.load_index(repository_id)

    def test_save_without_building_or_loading_raises(self, db: DatabaseManager, indexes_dir: Path) -> None:
        manager = BM25Manager(db, indexes_dir=indexes_dir)

        with pytest.raises(RetrievalError):
            manager.save_index("nonexistent-repository-id")
