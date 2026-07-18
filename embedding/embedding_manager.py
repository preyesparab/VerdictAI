"""EmbeddingManager: generates and persists embeddings for a repository's AST chunks.

Loads chunks from `database.sqlite_client.DatabaseManager` (Phase 7) rather
than re-parsing the repository, embeds only Level-1 AST chunks (see module
docstring in `models.schemas` for the AST vs. sliding/parent distinction),
and writes vectors back through `DatabaseManager.store_embeddings`. Never
touches `sentence_transformers` directly - model construction is delegated
to `embedding.model_loader`, so this module only orchestrates batching,
skip-logic, validation, and logging.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from core.exceptions import EmbeddingError
from core.logging import get_logger
from config import settings
from core.constants import EMBEDDING_DIMENSIONS
from database.sqlite_client import DatabaseManager
from embedding.model_loader import active_model_name, load_embedding_model
from models.schemas import ChunkType, CodeChunk

logger = get_logger(__name__)

# The AST-derived chunk types, as distinguished from SLIDING and PARENT in
# `models.schemas.ChunkType` - the only chunk types embedded by this module.
AST_CHUNK_TYPES = frozenset(
    {
        ChunkType.FUNCTION,
        ChunkType.ASYNC_FUNCTION,
        ChunkType.CLASS,
        ChunkType.METHOD,
        ChunkType.ARROW_FUNCTION,
    }
)


class EmbeddingModel(Protocol):
    """The subset of `sentence_transformers.SentenceTransformer` this module needs."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Encode `sentences` into a batch of embedding vectors."""
        ...


class EmbeddingManager:
    """Embeds a repository's AST chunks and persists the vectors to SQLite.

    The model is loaded lazily (on first `generate_embeddings` call) so
    constructing an `EmbeddingManager` never triggers a model download;
    tests can also inject a fake model directly via `model`.
    """

    def __init__(
        self,
        db: DatabaseManager,
        model: EmbeddingModel | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            db: Persistence layer to load chunks from and store vectors to.
            model: Pre-constructed embedding model. Overridable for testing;
                defaults to lazily loading `model_name` via
                `embedding.model_loader.load_embedding_model`.
            model_name: Embedding model identifier to load and to tag
                stored embeddings with. Defaults to
                `embedding.model_loader.active_model_name()` (selected by
                `settings.USE_CODEBERT`).
            batch_size: Chunks per `model.encode()` call. Defaults to
                `settings.EMBEDDING_BATCH_SIZE`.
        """
        self._db = db
        self._model_name = model_name or active_model_name()
        self._model = model
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    def _ensure_model(self) -> EmbeddingModel:
        """Lazily load the embedding model on first use."""
        if self._model is None:
            self._model = load_embedding_model(self._model_name)
        return self._model

    def generate_embeddings(self, repository_id: str, *, force: bool = False) -> int:
        """Embed and persist every AST chunk of `repository_id`.

        Args:
            repository_id: The repository to embed, previously indexed via
                `DatabaseManager.store_repository`/`store_chunks`.
            force: If True, re-embed and overwrite chunks that already have
                an embedding from this model instead of skipping them.

        Returns:
            The number of embeddings inserted or updated.

        Raises:
            EmbeddingError: If model inference fails, or if a produced
                vector's dimension does not match the active model's
                expected dimension.
        """
        logger.info(
            "Embedding generation started for repository %s (model=%s)",
            repository_id, self._model_name,
        )

        all_chunks = self._db.load_chunks(repository_id)
        ast_chunks = [chunk for chunk in all_chunks if chunk.chunk_type in AST_CHUNK_TYPES]

        if not ast_chunks:
            logger.info(
                "Embedding generation complete for repository %s: no AST chunk(s) to embed",
                repository_id,
            )
            return 0

        pending = ast_chunks
        if not force:
            already_embedded = self._db.get_embedded_chunk_ids(repository_id, self._model_name)
            pending = [chunk for chunk in ast_chunks if str(chunk.chunk_id) not in already_embedded]
            skipped = len(ast_chunks) - len(pending)
            if skipped:
                logger.info("Embedding skipped for %d already-embedded chunk(s)", skipped)

        if not pending:
            logger.info(
                "Embedding generation complete for repository %s: nothing new to embed",
                repository_id,
            )
            return 0

        model = self._ensure_model()
        expected_dimension = EMBEDDING_DIMENSIONS.get(self._model_name)
        stored = 0

        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            vectors = self._encode_batch(model, batch)

            chunk_vectors: list[tuple[str, np.ndarray]] = []
            for chunk, vector in zip(batch, vectors, strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                if expected_dimension is not None and vector.shape[0] != expected_dimension:
                    raise EmbeddingError(
                        f"Model {self._model_name!r} produced embedding dimension "
                        f"{vector.shape[0]}, expected {expected_dimension}"
                    )
                chunk_vectors.append((str(chunk.chunk_id), vector))

            stored += self._db.store_embeddings(
                repository_id, self._model_name, chunk_vectors, force=force
            )
            logger.info(
                "Batch completed: %d chunk(s) embedded (%d/%d total)",
                len(batch), min(start + self._batch_size, len(pending)), len(pending),
            )

        logger.info(
            "Embedding generation complete for repository %s: %d embedding(s) stored",
            repository_id, stored,
        )
        return stored

    def _encode_batch(self, model: EmbeddingModel, batch: list[CodeChunk]) -> Any:
        """Run one batch of chunks through `model.encode`.

        Args:
            model: The embedding model to run inference with.
            batch: The chunks to encode (their `raw_code` is embedded).

        Returns:
            One vector per chunk in `batch`, same order.

        Raises:
            EmbeddingError: If model inference fails.
        """
        texts = [chunk.raw_code for chunk in batch]
        try:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - third-party ML inference boundary
            raise EmbeddingError(f"Embedding inference failed: {exc}") from exc
