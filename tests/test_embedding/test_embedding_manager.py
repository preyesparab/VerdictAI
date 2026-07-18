"""Tests for embedding.embedding_manager.EmbeddingManager.

Model inference is mocked throughout via `_FakeModel` - these tests verify
batching, skip/force logic, dimension validation, and SQLite persistence,
not real CodeBERT/MiniLM inference.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from core.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MINILM_MODEL
from core.exceptions import EmbeddingError
from database.sqlite_client import DatabaseManager
from embedding.embedding_manager import EmbeddingManager
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
    file_id: str,
    file_path: str,
    *,
    chunk_type: ChunkType = ChunkType.FUNCTION,
    function_name: str | None = "foo",
    start_line: int = 1,
    end_line: int = 5,
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
    )


class _FakeModel:
    """Deterministic fake embedding model: returns a fixed-dimension vector per text."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.encode_calls: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.encode_calls.append(list(sentences))
        return np.array(
            [[float(len(text) % 7 + 1)] * self.dimension for text in sentences], dtype=np.float32
        )


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.initialize_database()
    return manager


def _seed_repository_with_chunks(
    db: DatabaseManager, ast_chunks: list[CodeChunk], sliding_chunks: list[CodeChunk] | None = None,
    parent_chunks: list[CodeChunk] | None = None,
) -> str:
    repository_id = db.store_repository(_repository_metadata())
    source_file = _source_file()
    db.store_source_files(repository_id, [source_file])
    result = ChunkingResult(
        ast_chunks=ast_chunks, sliding_chunks=sliding_chunks or [], parent_chunks=parent_chunks or []
    )
    db.store_chunks(repository_id, {"src/a.py": result})
    return repository_id


class TestGenerateEmbeddingsWithMiniLM:
    def test_embeds_ast_chunks_and_persists(self, db: DatabaseManager) -> None:
        chunks = [
            _chunk("f", "src/a.py", function_name="a"),
            _chunk("f", "src/a.py", function_name="b", start_line=6, end_line=10),
        ]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL, batch_size=10)

        stored = manager.generate_embeddings(repository_id)

        assert stored == 2
        embeddings = db.load_embeddings(repository_id, DEFAULT_MINILM_MODEL)
        assert set(embeddings) == {str(c.chunk_id) for c in chunks}
        assert all(vector.shape == (384,) for vector in embeddings.values())


class TestGenerateEmbeddingsWithCodeBERT:
    def test_embeds_ast_chunks_and_persists(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=768)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_EMBEDDING_MODEL, batch_size=10)

        stored = manager.generate_embeddings(repository_id)

        assert stored == 1
        embeddings = db.load_embeddings(repository_id, DEFAULT_EMBEDDING_MODEL)
        assert next(iter(embeddings.values())).shape == (768,)


class TestOnlyAstChunksAreEmbedded:
    def test_sliding_and_parent_chunks_are_ignored(self, db: DatabaseManager) -> None:
        ast_chunk = _chunk("f", "src/a.py", function_name="a")
        parent_chunk = _chunk(
            "f", "src/a.py", chunk_type=ChunkType.PARENT, function_name=None, start_line=1, end_line=20
        )
        sliding_chunk = _chunk(
            "f", "src/a.py", chunk_type=ChunkType.SLIDING, function_name=None, start_line=1, end_line=50
        )
        repository_id = _seed_repository_with_chunks(
            db, [ast_chunk], sliding_chunks=[sliding_chunk], parent_chunks=[parent_chunk]
        )
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        stored = manager.generate_embeddings(repository_id)

        assert stored == 1
        assert model.encode_calls == [[ast_chunk.raw_code]]

    def test_repository_with_no_ast_chunks_returns_zero_gracefully(self, db: DatabaseManager) -> None:
        sliding_chunk = _chunk(
            "f", "src/a.py", chunk_type=ChunkType.SLIDING, function_name=None, start_line=1, end_line=50
        )
        repository_id = _seed_repository_with_chunks(db, [], sliding_chunks=[sliding_chunk])
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        stored = manager.generate_embeddings(repository_id)

        assert stored == 0
        assert model.encode_calls == []


class TestDuplicatePrevention:
    def test_second_call_skips_already_embedded_chunks(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        first = manager.generate_embeddings(repository_id)
        second = manager.generate_embeddings(repository_id)

        assert first == 1
        assert second == 0
        assert len(model.encode_calls) == 1
        assert len(db.load_embeddings(repository_id, DEFAULT_MINILM_MODEL)) == 1


class TestForceRegeneration:
    def test_force_reembeds_and_overwrites_without_duplicating(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)
        manager.generate_embeddings(repository_id)

        second = manager.generate_embeddings(repository_id, force=True)

        assert second == 1
        assert len(model.encode_calls) == 2
        assert len(db.load_embeddings(repository_id, DEFAULT_MINILM_MODEL)) == 1


class TestBatchProcessing:
    def test_processes_chunks_in_configured_batch_size(self, db: DatabaseManager) -> None:
        chunks = [
            _chunk("f", "src/a.py", function_name=f"fn{i}", start_line=i, end_line=i)
            for i in range(1, 6)
        ]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL, batch_size=2)

        stored = manager.generate_embeddings(repository_id)

        assert stored == 5
        assert [len(call) for call in model.encode_calls] == [2, 2, 1]

    def test_default_batch_size_comes_from_settings(
        self, db: DatabaseManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import embedding.embedding_manager as embedding_manager_module

        monkeypatch.setattr(embedding_manager_module.settings, "EMBEDDING_BATCH_SIZE", 3)
        chunks = [
            _chunk("f", "src/a.py", function_name=f"fn{i}", start_line=i, end_line=i)
            for i in range(1, 5)
        ]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        manager.generate_embeddings(repository_id)

        assert [len(call) for call in model.encode_calls] == [3, 1]


class TestValidation:
    def test_wrong_dimension_raises_embedding_error(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=10)  # wrong for MiniLM (384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        with pytest.raises(EmbeddingError):
            manager.generate_embeddings(repository_id)

    def test_model_inference_failure_raises_embedding_error(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)

        class _RaisingModel:
            def encode(self, sentences: list[str], **kwargs: object) -> None:
                raise RuntimeError("model exploded")

        manager = EmbeddingManager(db, model=_RaisingModel(), model_name=DEFAULT_MINILM_MODEL)

        with pytest.raises(EmbeddingError):
            manager.generate_embeddings(repository_id)


class TestLazyModelLoading:
    def test_model_loader_is_not_invoked_when_model_is_injected(self, db: DatabaseManager) -> None:
        chunks = [_chunk("f", "src/a.py", function_name="a")]
        repository_id = _seed_repository_with_chunks(db, chunks)
        model = _FakeModel(dimension=384)
        manager = EmbeddingManager(db, model=model, model_name=DEFAULT_MINILM_MODEL)

        manager.generate_embeddings(repository_id)

        assert manager._model is model  # noqa: SLF001 - white-box test: no reload happened
