"""Tests for retrieval.reranker.CrossEncoderReranker.

Model inference is mocked throughout via `_FakeCrossEncoder` - these tests
verify dedup, top-k selection, batching, and error handling, not real
cross-encoder inference.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import RetrievalError
from models.schemas import RerankCandidate, RetrievalSource
from retrieval.reranker import CrossEncoderReranker


def _candidate(
    chunk_id: str,
    raw_code: str,
    *,
    function_name: str | None = "foo",
    previous_score: float = 0.1,
    retrieval_source: RetrievalSource = RetrievalSource.HYBRID,
    graph_distance: int = 0,
) -> RerankCandidate:
    return RerankCandidate(
        chunk_id=chunk_id,
        raw_code=raw_code,
        file_path="src/a.py",
        function_name=function_name,
        retrieval_source=retrieval_source,
        graph_distance=graph_distance,
        previous_score=previous_score,
    )


class _FakeCrossEncoder:
    """Deterministic fake: scores by a caller-supplied function of (query, code)."""

    def __init__(self, score_fn=None) -> None:
        self._score_fn = score_fn or (lambda query, code: float(len(code)))
        self.predict_calls: list[tuple[list, int | None]] = []

    def predict(self, sentences, **kwargs):
        batch_size = kwargs.get("batch_size")
        self.predict_calls.append((list(sentences), batch_size))
        return np.array([self._score_fn(query, code) for query, code in sentences], dtype=np.float32)


class TestReranking:
    def test_sorts_by_cross_encoder_score_descending(self) -> None:
        scores = {"short": 0.1, "medium code here": 0.5, "very long matching code snippet": 0.9}
        model = _FakeCrossEncoder(score_fn=lambda q, c: scores[c])
        candidates = [
            _candidate("a", "short"),
            _candidate("b", "medium code here"),
            _candidate("c", "very long matching code snippet"),
        ]
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", candidates, top_k=3)

        assert [r.chunk_id for r in results] == ["c", "b", "a"]
        assert results[0].cross_encoder_score == pytest.approx(0.9)
        assert results[0].final_rank == 1
        assert results[2].final_rank == 3

    def test_previous_retrieval_score_and_source_are_carried_over(self) -> None:
        model = _FakeCrossEncoder()
        candidate = _candidate("a", "code", previous_score=0.42, retrieval_source=RetrievalSource.GRAPH)
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", [candidate], top_k=1)

        assert results[0].previous_retrieval_score == 0.42
        assert results[0].retrieval_source == RetrievalSource.GRAPH


class TestDuplicateHandling:
    def test_duplicate_chunk_ids_are_deduped_keeping_first(self) -> None:
        model = _FakeCrossEncoder()
        first = _candidate("a", "short code", previous_score=0.9)
        duplicate = _candidate("a", "different code text here", previous_score=0.1)
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", [first, duplicate], top_k=5)

        assert len(results) == 1
        assert results[0].previous_retrieval_score == 0.9
        scored_pairs = model.predict_calls[0][0]
        assert scored_pairs == [("query", "short code")]


class TestTopKSelection:
    def test_returns_only_top_k(self) -> None:
        model = _FakeCrossEncoder()
        candidates = [_candidate(f"c{i}", "x" * i) for i in range(1, 11)]
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", candidates, top_k=3)

        assert len(results) == 3

    def test_defaults_to_settings_top_k_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import retrieval.reranker as reranker_module

        monkeypatch.setattr(reranker_module.settings, "TOP_K_FINAL", 2)
        model = _FakeCrossEncoder()
        candidates = [_candidate(f"c{i}", "x" * i) for i in range(1, 6)]
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", candidates)

        assert len(results) == 2

    def test_raises_on_non_positive_top_k(self) -> None:
        model = _FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model)

        with pytest.raises(RetrievalError):
            reranker.rerank("query", [_candidate("a", "code")], top_k=0)


class TestEmptyCandidates:
    def test_returns_empty_list_without_invoking_model(self) -> None:
        model = _FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("query", [], top_k=5)

        assert results == []
        assert model.predict_calls == []


class TestBatchInference:
    def test_processes_in_configured_batch_size(self) -> None:
        model = _FakeCrossEncoder()
        candidates = [_candidate(f"c{i}", "x" * i) for i in range(1, 6)]
        reranker = CrossEncoderReranker(model=model, batch_size=2)

        reranker.rerank("query", candidates, top_k=5)

        assert [len(call[0]) for call in model.predict_calls] == [2, 2, 1]
        assert all(call[1] == 2 for call in model.predict_calls)

    def test_default_batch_size_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import retrieval.reranker as reranker_module

        monkeypatch.setattr(reranker_module.settings, "RERANKER_BATCH_SIZE", 3)
        model = _FakeCrossEncoder()
        candidates = [_candidate(f"c{i}", "x" * i) for i in range(1, 5)]
        reranker = CrossEncoderReranker(model=model)

        reranker.rerank("query", candidates, top_k=5)

        assert [len(call[0]) for call in model.predict_calls] == [3, 1]


class TestErrorHandling:
    def test_model_inference_failure_raises_retrieval_error(self) -> None:
        class _RaisingModel:
            def predict(self, sentences, **kwargs):
                raise RuntimeError("boom")

        reranker = CrossEncoderReranker(model=_RaisingModel())

        with pytest.raises(RetrievalError):
            reranker.rerank("query", [_candidate("a", "code")], top_k=1)

    def test_model_loading_failure_raises_retrieval_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import retrieval.reranker as reranker_module

        def _raise(model_name: str):
            raise RetrievalError("boom")

        monkeypatch.setattr(reranker_module, "load_cross_encoder", _raise)
        reranker = CrossEncoderReranker(model_name="fake-model")

        with pytest.raises(RetrievalError):
            reranker.rerank("query", [_candidate("a", "code")], top_k=1)


class TestLazyModelLoading:
    def test_model_loader_not_invoked_when_model_is_injected(self) -> None:
        model = _FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model)

        reranker.rerank("query", [_candidate("a", "code")], top_k=1)

        assert reranker._model is model  # noqa: SLF001 - white-box test: no reload happened
