"""BM25 sparse (keyword-based) retrieval over a repository's AST chunks (Phase 10).

`BM25Manager` builds a `rank_bm25.BM25Okapi` index directly from chunks
already stored by `database.sqlite_client.DatabaseManager` (Phase 7) - it
never re-parses a repository or touches embeddings, and only indexes AST
chunks (the same Level-1 units Phase 8/9 embed), never sliding/parent
chunks.

Complements `database.vector_store.FaissIndexManager` (Phase 9): FAISS
finds semantically similar chunks even with no shared words; BM25 finds
chunks sharing exact tokens with the query (identifiers, error strings,
config keys) that dense retrieval can under-rank. Phase 11 fuses both
rankings with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import settings
from core.exceptions import RetrievalError
from core.logging import get_logger
from database.sqlite_client import DatabaseManager
from embedding.embedding_manager import AST_CHUNK_TYPES
from models.schemas import CodeChunk
from retrieval.tokenizer import tokenize

logger = get_logger(__name__)


@dataclass(frozen=True)
class SparseSearchResult:
    """One BM25 search hit.

    Attributes:
        chunk_id: The matched chunk's id.
        score: BM25 relevance score (unbounded, can be negative for very
            common terms - higher is more relevant).
    """

    chunk_id: str
    score: float


class BM25Manager:
    """Builds, persists, and searches a BM25 index of one repository's AST chunks.

    An empty repository is tracked as "indexed, zero documents" rather
    than raising - `rank_bm25.BM25Okapi` cannot itself be constructed over
    an empty corpus (its average-document-length computation divides by
    corpus size), so this class never attempts to.
    """

    def __init__(self, db: DatabaseManager, indexes_dir: Path | None = None) -> None:
        """Initialize the manager without building or loading an index yet.

        Args:
            db: Persistence layer to load AST chunks from and repository
                metadata (for index file naming) from.
            indexes_dir: Directory BM25 index files are stored in.
                Defaults to `settings.INDEXES_DIR` (``data/indexes``).
        """
        self._db = db
        self._indexes_dir = indexes_dir or settings.INDEXES_DIR
        self._indexes_dir.mkdir(parents=True, exist_ok=True)

        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._is_indexed = False

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
        """Path to `repository_id`'s serialized BM25 index file."""
        return self._indexes_dir / f"{self._index_stem(repository_id)}_bm25.pkl"

    def _document_text(self, chunk: CodeChunk) -> str:
        """Build the BM25 document text for one chunk.

        `function_name`/`class_name`/`parent_class` are included both on
        their own and inside `raw_code` (which already contains them) -
        deliberately, so a query naming them exactly gets a higher term
        frequency, and therefore a higher BM25 score, than an incidental
        mention elsewhere in the code.

        Args:
            chunk: The chunk to build a document for.

        Returns:
            The text to tokenize and index for this chunk.
        """
        parts = [chunk.function_name, chunk.class_name, chunk.parent_class, chunk.raw_code]
        return " ".join(part for part in parts if part)

    def build_index(self, repository_id: str) -> int:
        """Build an in-memory BM25 index from `repository_id`'s stored AST chunks.

        Does not persist the index - call `save_index` afterward if it
        should survive the process. Replaces any index previously built
        or loaded into this manager instance.

        Args:
            repository_id: The repository whose AST chunks to index.

        Returns:
            The number of documents indexed (0 for a repository with no
            AST chunks, handled gracefully rather than raising).

        Raises:
            RetrievalError: If index construction fails.
        """
        logger.info("BM25 build started for repository %s", repository_id)

        all_chunks = self._db.load_chunks(repository_id)
        ast_chunks = [chunk for chunk in all_chunks if chunk.chunk_type in AST_CHUNK_TYPES]

        # Defensive de-duplication by chunk_id: DatabaseManager's primary
        # key already prevents duplicate rows, but a document must map to
        # exactly one BM25 corpus position, so this guarantee is kept
        # explicit and local rather than assumed from an upstream layer.
        seen_ids: set[str] = set()
        unique_chunks: list[CodeChunk] = []
        for chunk in ast_chunks:
            chunk_id = str(chunk.chunk_id)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)

        if not unique_chunks:
            self._bm25 = None
            self._chunk_ids = []
            self._is_indexed = True
            logger.info("Documents indexed: 0 for repository %s (empty repository)", repository_id)
            return 0

        tokenized_documents = [tokenize(self._document_text(chunk)) for chunk in unique_chunks]

        try:
            bm25 = BM25Okapi(tokenized_documents)
        except Exception as exc:  # noqa: BLE001 - third-party rank_bm25 boundary, see module docstring
            raise RetrievalError(f"Failed to build BM25 index for repository {repository_id}: {exc}") from exc

        self._bm25 = bm25
        self._chunk_ids = [str(chunk.chunk_id) for chunk in unique_chunks]
        self._is_indexed = True

        logger.info("Documents indexed: %d for repository %s", len(self._chunk_ids), repository_id)
        return len(self._chunk_ids)

    def save_index(self, repository_id: str) -> Path:
        """Persist the currently built/loaded index to `settings.INDEXES_DIR`.

        The document-id -> chunk_id mapping (`self._chunk_ids`, where
        position ``i`` is document ``i``'s chunk id) is pickled alongside
        the `BM25Okapi` object in the same file.

        Args:
            repository_id: The repository this index belongs to (used to
                name the index file).

        Returns:
            The path the index file was written to.

        Raises:
            RetrievalError: If no index has been built/loaded yet, or the
                write fails.
        """
        if not self._is_indexed:
            raise RetrievalError("No index to save: call build_index or load_index first.")

        index_path = self._index_path(repository_id)
        try:
            with index_path.open("wb") as fh:
                pickle.dump(
                    {"repository_id": repository_id, "bm25": self._bm25, "chunk_ids": self._chunk_ids},
                    fh,
                )
        except Exception as exc:  # noqa: BLE001 - third-party rank_bm25/pickle boundary
            raise RetrievalError(f"Failed to save BM25 index for repository {repository_id}: {exc}") from exc

        logger.info(
            "Index saved for repository %s: %s (%d document(s))",
            repository_id, index_path, len(self._chunk_ids),
        )
        return index_path

    def load_index(self, repository_id: str) -> int:
        """Load a previously saved index from `settings.INDEXES_DIR`.

        Does not touch SQLite or re-tokenize anything - reads only the
        pickle file written by `save_index`.

        Args:
            repository_id: The repository whose saved index to load.

        Returns:
            The number of documents in the loaded index.

        Raises:
            RetrievalError: If no saved index exists for `repository_id`,
                or the file cannot be read/unpickled.
        """
        index_path = self._index_path(repository_id)
        if not index_path.exists():
            raise RetrievalError(f"No saved BM25 index found for repository {repository_id} at {index_path}")

        try:
            with index_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - third-party rank_bm25/pickle boundary
            raise RetrievalError(f"Failed to load BM25 index for repository {repository_id}: {exc}") from exc

        self._bm25 = payload["bm25"]
        self._chunk_ids = payload["chunk_ids"]
        self._is_indexed = True

        logger.info(
            "Index loaded for repository %s: %s (%d document(s))",
            repository_id, index_path, len(self._chunk_ids),
        )
        return len(self._chunk_ids)

    def search(self, query: str, top_k: int) -> list[SparseSearchResult]:
        """Find the `top_k` chunks most relevant to `query` by BM25 score.

        Args:
            query: Free-text or keyword query.
            top_k: Maximum number of results to return.

        Returns:
            Up to `top_k` `SparseSearchResult`s, sorted by descending BM25
            score. Empty if the index has no documents, or if `query`
            tokenizes to nothing (e.g. empty or punctuation-only).

        Raises:
            RetrievalError: If no index has been built/loaded yet,
                `top_k` is not positive, or the search itself fails.
        """
        logger.info("Query received: %r", query)

        if not self._is_indexed:
            raise RetrievalError("No index to search: call build_index or load_index first.")
        if top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {top_k}")

        if not self._chunk_ids:
            logger.info("Retrieval completed: 0 result(s) (empty index)")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.info("Retrieval completed: 0 result(s) (empty query)")
            return []

        try:
            scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:  # noqa: BLE001 - third-party rank_bm25 boundary
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

        ranked = sorted(zip(self._chunk_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        results = [
            SparseSearchResult(chunk_id=chunk_id, score=float(score))
            for chunk_id, score in ranked[:top_k]
        ]

        logger.info("Retrieval completed: %d result(s)", len(results))
        return results
