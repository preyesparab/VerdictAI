"""FAISS-backed dense vector index over a repository's chunk embeddings (Phase 9).

`FaissIndexManager` builds a FAISS index directly from embeddings already
stored by `database.sqlite_client.DatabaseManager.store_embeddings` (Phase
8) - it never calls an embedding model itself, so building or loading an
index never regenerates embeddings. It only indexes AST chunks: Phase 8's
`embedding.embedding_manager.EmbeddingManager` never embeds sliding/parent
chunks, so the `embeddings` table this reads from is already AST-only by
construction - no extra filtering is needed here.

Each repository gets its own index file under `settings.INDEXES_DIR`,
named from its owner/repository name when known (e.g.
``tiangolo_fastapi.faiss``), alongside a JSON sidecar mapping FAISS's
internal row order back to `chunk_id`s (FAISS itself only knows integer
row positions, not chunk identity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config import settings
from core.constants import EMBEDDING_DIMENSIONS
from core.exceptions import RetrievalError
from core.logging import get_logger
from database.sqlite_client import DatabaseManager
from embedding.model_loader import active_model_name

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """One search hit.

    Attributes:
        chunk_id: The matched chunk's id.
        score: Cosine similarity to the query (higher is more similar),
            in ``[-1, 1]``.
    """

    chunk_id: str
    score: float


class FaissIndexManager:
    """Builds, persists, and searches a FAISS index of one repository's chunk embeddings.

    Index construction is factored into `_create_index` so that swapping
    `faiss.IndexFlatIP` for an IVF/HNSW/PQ index later touches only that
    one method - `build_index`/`load_index`/`search`/`save_index` do not
    need to change.
    """

    def __init__(
        self,
        db: DatabaseManager,
        indexes_dir: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the manager without building or loading an index yet.

        Args:
            db: Persistence layer to load embeddings from and repository
                metadata (for index file naming) from.
            indexes_dir: Directory FAISS index files and their id-mapping
                sidecars are stored in. Defaults to `settings.INDEXES_DIR`
                (``data/indexes``).
            model_name: Which model's embeddings to index. Defaults to
                `embedding.model_loader.active_model_name()` (selected by
                `settings.USE_CODEBERT`) - should match whatever model
                `EmbeddingManager` was run with.
        """
        self._db = db
        self._indexes_dir = indexes_dir or settings.INDEXES_DIR
        self._indexes_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name or active_model_name()

        self._index: faiss.Index | None = None
        self._chunk_ids: list[str] = []
        self._dimension: int | None = None

    def _create_index(self, dimension: int) -> faiss.Index:
        """Construct a new, empty index of `dimension`.

        The sole extension point for adding IVF/HNSW/PQ indexes later
        without changing this class's public API.

        Args:
            dimension: The embedding vector dimension the index will hold.

        Returns:
            A new, untrained/empty `faiss.IndexFlatIP`.
        """
        return faiss.IndexFlatIP(dimension)

    def _index_stem(self, repository_id: str) -> str:
        """Derive this repository's index filename stem (without extension).

        Args:
            repository_id: The repository to name a file for.

        Returns:
            ``"<owner>_<repository_name>"`` if repository metadata is
            stored, else `repository_id` itself.
        """
        repository = self._db.load_repository(repository_id)
        if repository is None:
            return repository_id
        return f"{repository.owner}_{repository.name}"

    def _index_path(self, repository_id: str) -> Path:
        """Path to `repository_id`'s serialized FAISS index file."""
        return self._indexes_dir / f"{self._index_stem(repository_id)}.faiss"

    def _mapping_path(self, repository_id: str) -> Path:
        """Path to `repository_id`'s chunk-id mapping sidecar file."""
        return self._indexes_dir / f"{self._index_stem(repository_id)}.json"

    def build_index(self, repository_id: str) -> int:
        """Build an in-memory FAISS index from `repository_id`'s stored embeddings.

        Does not persist the index - call `save_index` afterward if it
        should survive the process. Replaces any index previously built
        or loaded into this manager instance.

        Args:
            repository_id: The repository whose embeddings to index.

        Returns:
            The number of vectors indexed (0 for a repository with no
            stored embeddings, handled gracefully rather than raising).

        Raises:
            RetrievalError: If an embedding's dimension does not match
                the active model's expected dimension, or index
                construction otherwise fails.
        """
        logger.info(
            "Index creation started for repository %s (model=%s)", repository_id, self._model_name
        )
        embeddings = self._db.load_embeddings(repository_id, self._model_name)
        logger.info("Embeddings loaded: %d for repository %s", len(embeddings), repository_id)

        expected_dimension = EMBEDDING_DIMENSIONS.get(self._model_name)
        chunk_ids = sorted(embeddings)  # deterministic row order

        for chunk_id in chunk_ids:
            actual_dimension = embeddings[chunk_id].shape[0]
            if expected_dimension is not None and actual_dimension != expected_dimension:
                raise RetrievalError(
                    f"Embedding for chunk {chunk_id!r} has dimension {actual_dimension}, "
                    f"expected {expected_dimension} for model {self._model_name!r}"
                )

        dimension = expected_dimension or (embeddings[chunk_ids[0]].shape[0] if chunk_ids else None)
        if dimension is None:
            raise RetrievalError(
                f"Cannot determine embedding dimension for model {self._model_name!r}: "
                "no stored embeddings and no known default dimension"
            )

        try:
            index = self._create_index(dimension)
            if chunk_ids:
                matrix = np.vstack([embeddings[chunk_id] for chunk_id in chunk_ids]).astype(np.float32)
                faiss.normalize_L2(matrix)
                index.add(matrix)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - third-party FAISS boundary, see module docstring
            raise RetrievalError(f"Failed to build FAISS index for repository {repository_id}: {exc}") from exc

        self._index = index
        self._chunk_ids = chunk_ids
        self._dimension = dimension

        logger.info("Vectors indexed: %d for repository %s", index.ntotal, repository_id)
        return index.ntotal

    def save_index(self, repository_id: str) -> Path:
        """Persist the currently built/loaded index to `settings.INDEXES_DIR`.

        Args:
            repository_id: The repository this index belongs to (used to
                name the index file and its id-mapping sidecar).

        Returns:
            The path the FAISS index file was written to.

        Raises:
            RetrievalError: If no index has been built/loaded yet, or the
                write fails.
        """
        if self._index is None:
            raise RetrievalError("No index to save: call build_index or load_index first.")

        index_path = self._index_path(repository_id)
        mapping_path = self._mapping_path(repository_id)

        try:
            faiss.write_index(self._index, str(index_path))
            mapping_path.write_text(
                json.dumps(
                    {
                        "repository_id": repository_id,
                        "model_name": self._model_name,
                        "dimension": self._dimension,
                        "chunk_ids": self._chunk_ids,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - third-party FAISS boundary, see module docstring
            raise RetrievalError(f"Failed to save FAISS index for repository {repository_id}: {exc}") from exc

        logger.info(
            "Index saved for repository %s: %s (%d vector(s))",
            repository_id, index_path, self._index.ntotal,
        )
        return index_path

    def load_index(self, repository_id: str) -> int:
        """Load a previously saved index from `settings.INDEXES_DIR`.

        Does not touch SQLite or regenerate anything - reads only the
        FAISS index file and its id-mapping sidecar written by
        `save_index`.

        Args:
            repository_id: The repository whose saved index to load.

        Returns:
            The number of vectors in the loaded index.

        Raises:
            RetrievalError: If no saved index exists for `repository_id`,
                or the files cannot be read/parsed.
        """
        index_path = self._index_path(repository_id)
        mapping_path = self._mapping_path(repository_id)

        if not index_path.exists() or not mapping_path.exists():
            raise RetrievalError(f"No saved FAISS index found for repository {repository_id} at {index_path}")

        try:
            index = faiss.read_index(str(index_path))
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - third-party FAISS boundary, see module docstring
            raise RetrievalError(f"Failed to load FAISS index for repository {repository_id}: {exc}") from exc

        self._index = index
        self._chunk_ids = mapping["chunk_ids"]
        self._dimension = mapping["dimension"]
        self._model_name = mapping.get("model_name", self._model_name)

        logger.info(
            "Index loaded for repository %s: %s (%d vector(s))",
            repository_id, index_path, index.ntotal,
        )
        return index.ntotal

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Find the `top_k` chunks most similar to `query_embedding`.

        Args:
            query_embedding: The query vector, same dimension as the
                indexed embeddings.
            top_k: Maximum number of results to return.

        Returns:
            Up to `top_k` `SearchResult`s, sorted by descending cosine
            similarity (FAISS's own search order for `IndexFlatIP`, so no
            extra sort is applied). Empty if the index has no vectors.

        Raises:
            RetrievalError: If no index has been built/loaded yet,
                `top_k` is not positive, `query_embedding`'s dimension
                does not match the index, or the search itself fails.
        """
        if self._index is None:
            raise RetrievalError("No index to search: call build_index or load_index first.")
        if top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {top_k}")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self._dimension is not None and query.shape[1] != self._dimension:
            raise RetrievalError(
                f"Query embedding has dimension {query.shape[1]}, expected {self._dimension}"
            )

        if self._index.ntotal == 0:
            logger.info("Search completed: 0 result(s) (empty index)")
            return []

        try:
            faiss.normalize_L2(query)
            effective_k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query, effective_k)
        except Exception as exc:  # noqa: BLE001 - third-party FAISS boundary, see module docstring
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

        results = [
            SearchResult(chunk_id=self._chunk_ids[index], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index != -1
        ]
        logger.info("Search completed: %d result(s)", len(results))
        return results
