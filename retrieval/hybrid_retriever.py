"""HybridRetriever: fuses dense (FAISS) and sparse (BM25) retrieval via RRF (Phase 11).

Runs `database.vector_store.FaissIndexManager` (Phase 9) and
`retrieval.sparse_retriever.BM25Manager` (Phase 10) independently over the
same query, then merges the two ranked candidate lists with Reciprocal
Rank Fusion: ``score(chunk) = sum(1 / (rrf_k + rank))`` over every list the
chunk appears in. RRF only uses each result's rank position, never its raw
score, which is what lets FAISS's cosine similarities and BM25's
unbounded term-frequency scores - two incomparable scales - be combined
without normalization or hand-tuned weights.

`HybridRetriever` owns no index lifecycle: it expects `faiss_manager` and
`bm25_manager` to already have an index built or loaded for whichever
repository is being queried (Phases 9/10's own responsibility). This is
intended to become the default retrieval engine the rest of the
application (context building, generation) calls into.
"""

from __future__ import annotations

import numpy as np

from config import settings
from core.exceptions import RetrievalError
from core.logging import get_logger
from database.vector_store import FaissIndexManager, SearchResult
from models.schemas import RetrievalSource, RetrievedChunk
from retrieval.sparse_retriever import BM25Manager, SparseSearchResult

logger = get_logger(__name__)


class HybridRetriever:
    """Combines `FaissIndexManager` and `BM25Manager` results via Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_manager: FaissIndexManager,
        bm25_manager: BM25Manager,
        rrf_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            faiss_manager: Dense retriever. Must already have an index
                built/loaded for whichever repository will be queried.
            bm25_manager: Sparse retriever. Must already have an index
                built/loaded for whichever repository will be queried.
            rrf_k: The `k` constant in RRF's ``1 / (k + rank)``. Larger
                values flatten the contribution curve across ranks
                (rank 1 matters relatively less); smaller values make
                rank 1 dominate more. Defaults to `settings.RRF_K`.
            top_k_dense: Candidates to request from `faiss_manager` per
                query. Defaults to `settings.TOP_K_DENSE`.
            top_k_bm25: Candidates to request from `bm25_manager` per
                query. Defaults to `settings.TOP_K_BM25`.
        """
        self._faiss_manager = faiss_manager
        self._bm25_manager = bm25_manager
        self._rrf_k = rrf_k or settings.RRF_K
        self._top_k_dense = top_k_dense or settings.TOP_K_DENSE
        self._top_k_bm25 = top_k_bm25 or settings.TOP_K_BM25

    def retrieve(
        self,
        repository_id: str,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse dense + sparse results for one query.

        Args:
            repository_id: The repository being queried (used only for
                logging context - index selection is the caller's
                responsibility, see class docstring).
            query: The query text, passed to `BM25Manager.search`.
            query_embedding: The query's dense embedding, passed to
                `FaissIndexManager.search`.
            top_k: Maximum number of fused results to return.

        Returns:
            Up to `top_k` `RetrievedChunk`s, sorted by descending
            `fused_score`. Empty if neither retriever found anything
            (including for a repository with no indexed chunks, which
            both retrievers already handle by returning empty lists).

        Raises:
            RetrievalError: If `top_k` is not positive, or either
                retriever fails.
        """
        if top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {top_k}")

        dense_results = self._run_dense(repository_id, query_embedding)
        sparse_results = self._run_sparse(repository_id, query)

        fused = self._fuse(dense_results, sparse_results)
        top_results = fused[:top_k]

        logger.info("Final candidate count: %d for repository %s", len(top_results), repository_id)
        return top_results

    def _run_dense(self, repository_id: str, query_embedding: np.ndarray) -> list[SearchResult]:
        """Run dense retrieval, translating any failure into a `RetrievalError`.

        Args:
            repository_id: The repository being queried (logging only).
            query_embedding: The query's dense embedding.

        Returns:
            Up to `self._top_k_dense` dense search results.

        Raises:
            RetrievalError: If dense retrieval fails.
        """
        try:
            results = self._faiss_manager.search(query_embedding, self._top_k_dense)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - uniform hybrid-retrieval failure boundary
            raise RetrievalError(f"Dense retrieval failed for repository {repository_id}: {exc}") from exc

        logger.info("Dense retrieval completed: %d result(s) for repository %s", len(results), repository_id)
        return results

    def _run_sparse(self, repository_id: str, query: str) -> list[SparseSearchResult]:
        """Run sparse retrieval, translating any failure into a `RetrievalError`.

        Args:
            repository_id: The repository being queried (logging only).
            query: The query text.

        Returns:
            Up to `self._top_k_bm25` sparse search results.

        Raises:
            RetrievalError: If sparse retrieval fails.
        """
        try:
            results = self._bm25_manager.search(query, self._top_k_bm25)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - uniform hybrid-retrieval failure boundary
            raise RetrievalError(f"Sparse retrieval failed for repository {repository_id}: {exc}") from exc

        logger.info("Sparse retrieval completed: %d result(s) for repository %s", len(results), repository_id)
        return results

    def _dedupe(self, results: list) -> list:
        """Keep only the first (highest-ranked) occurrence of each chunk_id.

        Defensive: neither retriever should return a chunk_id twice, but
        RRF's rank-based scoring would silently double-count one if it
        ever did, so this guarantee is kept explicit and local.

        Args:
            results: A dense or sparse retriever's result list, already
                sorted by descending relevance.

        Returns:
            `results` with later duplicates of an already-seen chunk_id
            removed.
        """
        seen: set[str] = set()
        deduped = []
        for result in results:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            deduped.append(result)
        return deduped

    def _fuse(
        self, dense_results: list[SearchResult], sparse_results: list[SparseSearchResult]
    ) -> list[RetrievedChunk]:
        """Merge dense and sparse results into one RRF-ranked list.

        Args:
            dense_results: `FaissIndexManager.search` output.
            sparse_results: `BM25Manager.search` output.

        Returns:
            Every distinct chunk_id from either list, as a
            `RetrievedChunk`, sorted by descending `fused_score`.
        """
        dense_results = self._dedupe(dense_results)
        sparse_results = self._dedupe(sparse_results)

        dense_by_chunk = {result.chunk_id: result.score for result in dense_results}
        sparse_by_chunk = {result.chunk_id: result.score for result in sparse_results}

        duplicate_chunk_ids = set(dense_by_chunk) & set(sparse_by_chunk)
        if duplicate_chunk_ids:
            logger.info("Duplicate chunks merged: %d", len(duplicate_chunk_ids))

        fused_scores: dict[str, float] = {}
        for rank, result in enumerate(dense_results, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
        for rank, result in enumerate(sparse_results, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)

        ordered_chunk_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)

        retrieved_chunks = []
        for rank, chunk_id in enumerate(ordered_chunk_ids, start=1):
            in_dense = chunk_id in dense_by_chunk
            in_sparse = chunk_id in sparse_by_chunk
            if in_dense and in_sparse:
                source = RetrievalSource.HYBRID
            elif in_dense:
                source = RetrievalSource.DENSE
            else:
                source = RetrievalSource.SPARSE

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    dense_score=dense_by_chunk.get(chunk_id),
                    bm25_score=sparse_by_chunk.get(chunk_id),
                    fused_score=fused_scores[chunk_id],
                    retrieval_source=source,
                    rank=rank,
                )
            )

        logger.info("RRF fusion completed: %d unique candidate(s)", len(retrieved_chunks))
        return retrieved_chunks
