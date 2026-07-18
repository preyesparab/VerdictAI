"""Tests for retrieval.hybrid_retriever.HybridRetriever.

Uses small fake dense/sparse managers (duck-typed `.search(...)`
collaborators, not real FAISS/BM25 indexes) so RRF fusion logic can be
tested against exact, hand-computed expected scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import RetrievalError
from database.vector_store import SearchResult
from models.schemas import RetrievalSource
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.sparse_retriever import SparseSearchResult


def _dense(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, score=score)


def _sparse(chunk_id: str, score: float) -> SparseSearchResult:
    return SparseSearchResult(chunk_id=chunk_id, score=score)


class _FakeFaissManager:
    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.search_calls: list[tuple] = []

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        self.search_calls.append((query_embedding, top_k))
        if self._error is not None:
            raise self._error
        return self._results[:top_k]


class _FakeBM25Manager:
    def __init__(self, results: list[SparseSearchResult] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.search_calls: list[tuple] = []

    def search(self, query: str, top_k: int) -> list[SparseSearchResult]:
        self.search_calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return self._results[:top_k]


class TestDenseOnly:
    def test_returns_dense_results_when_sparse_is_empty(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9), _dense("b", 0.8)])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager, rrf_k=60)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

        assert [r.chunk_id for r in results] == ["a", "b"]
        assert all(r.retrieval_source == RetrievalSource.DENSE for r in results)
        assert all(r.bm25_score is None for r in results)
        assert results[0].dense_score == 0.9


class TestSparseOnly:
    def test_returns_sparse_results_when_dense_is_empty(self) -> None:
        faiss_manager = _FakeFaissManager(results=[])
        bm25_manager = _FakeBM25Manager(results=[_sparse("a", 5.0), _sparse("b", 3.0)])
        retriever = HybridRetriever(faiss_manager, bm25_manager, rrf_k=60)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

        assert [r.chunk_id for r in results] == ["a", "b"]
        assert all(r.retrieval_source == RetrievalSource.SPARSE for r in results)
        assert all(r.dense_score is None for r in results)


class TestOverlappingResults:
    def test_chunk_in_both_lists_is_hybrid_and_outranks_single_source(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9), _dense("b", 0.8), _dense("c", 0.7)])
        bm25_manager = _FakeBM25Manager(results=[_sparse("b", 5.0), _sparse("d", 3.0)])
        retriever = HybridRetriever(faiss_manager, bm25_manager, rrf_k=60)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=10)

        by_id = {r.chunk_id: r for r in results}
        assert by_id["b"].retrieval_source == RetrievalSource.HYBRID
        assert by_id["b"].dense_score == 0.8
        assert by_id["b"].bm25_score == 5.0
        assert by_id["a"].retrieval_source == RetrievalSource.DENSE
        assert by_id["d"].retrieval_source == RetrievalSource.SPARSE
        # b is rank 2 dense + rank 1 sparse, so its summed RRF contribution
        # should outrank chunks appearing in only one list.
        assert results[0].chunk_id == "b"


class TestDuplicateMerging:
    def test_duplicate_chunk_within_a_single_retriever_is_deduped(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9), _dense("a", 0.5)])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager, rrf_k=60)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=10)

        assert len(results) == 1
        assert results[0].dense_score == 0.9  # first (highest-ranked) occurrence kept


class TestRRFScoreCorrectness:
    def test_fused_score_matches_rrf_formula(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9), _dense("b", 0.8)])
        bm25_manager = _FakeBM25Manager(results=[_sparse("b", 5.0), _sparse("a", 3.0)])
        rrf_k = 10
        retriever = HybridRetriever(faiss_manager, bm25_manager, rrf_k=rrf_k)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=10)
        by_id = {r.chunk_id: r for r in results}

        expected_a = 1 / (rrf_k + 1) + 1 / (rrf_k + 2)  # dense rank 1, sparse rank 2
        expected_b = 1 / (rrf_k + 2) + 1 / (rrf_k + 1)  # dense rank 2, sparse rank 1

        assert by_id["a"].fused_score == pytest.approx(expected_a)
        assert by_id["b"].fused_score == pytest.approx(expected_b)


class TestConfigurableRRFK:
    def test_different_rrf_k_produces_different_fused_scores(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9)])
        bm25_manager = _FakeBM25Manager(results=[])

        low_k_result = HybridRetriever(faiss_manager, bm25_manager, rrf_k=1).retrieve(
            "repo", "query", np.zeros(4), top_k=1
        )[0]
        high_k_result = HybridRetriever(faiss_manager, bm25_manager, rrf_k=1000).retrieve(
            "repo", "query", np.zeros(4), top_k=1
        )[0]

        assert low_k_result.fused_score == pytest.approx(1 / (1 + 1))
        assert high_k_result.fused_score == pytest.approx(1 / (1000 + 1))
        assert low_k_result.fused_score > high_k_result.fused_score

    def test_default_rrf_k_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import retrieval.hybrid_retriever as hybrid_retriever_module

        monkeypatch.setattr(hybrid_retriever_module.settings, "RRF_K", 5)
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9)])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager)

        result = retriever.retrieve("repo", "query", np.zeros(4), top_k=1)[0]

        assert result.fused_score == pytest.approx(1 / (5 + 1))


class TestEmptyRepositories:
    def test_both_retrievers_empty_returns_empty_list(self) -> None:
        faiss_manager = _FakeFaissManager(results=[])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

        assert results == []


class TestValidation:
    def test_raises_on_non_positive_top_k(self) -> None:
        retriever = HybridRetriever(_FakeFaissManager(), _FakeBM25Manager())

        with pytest.raises(RetrievalError):
            retriever.retrieve("repo", "query", np.zeros(4), top_k=0)

    def test_top_k_truncates_final_results(self) -> None:
        faiss_manager = _FakeFaissManager(results=[_dense("a", 0.9), _dense("b", 0.8), _dense("c", 0.7)])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager)

        results = retriever.retrieve("repo", "query", np.zeros(4), top_k=2)

        assert len(results) == 2

    def test_dense_failure_raises_retrieval_error(self) -> None:
        faiss_manager = _FakeFaissManager(error=RuntimeError("boom"))
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager)

        with pytest.raises(RetrievalError):
            retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

    def test_sparse_failure_raises_retrieval_error(self) -> None:
        faiss_manager = _FakeFaissManager(results=[])
        bm25_manager = _FakeBM25Manager(error=RuntimeError("boom"))
        retriever = HybridRetriever(faiss_manager, bm25_manager)

        with pytest.raises(RetrievalError):
            retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

    def test_passes_configured_top_k_dense_and_bm25_to_each_retriever(self) -> None:
        faiss_manager = _FakeFaissManager(results=[])
        bm25_manager = _FakeBM25Manager(results=[])
        retriever = HybridRetriever(faiss_manager, bm25_manager, top_k_dense=7, top_k_bm25=11)

        retriever.retrieve("repo", "query", np.zeros(4), top_k=5)

        assert faiss_manager.search_calls[0][1] == 7
        assert bm25_manager.search_calls[0][1] == 11
