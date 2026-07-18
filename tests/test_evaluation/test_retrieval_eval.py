"""Tests for evaluation.retrieval_eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.exceptions import EvaluationError
from evaluation.eval_dataset import EvalCase
from evaluation.retrieval_eval import (
    EvaluatedChunkInput,
    LLMRelevanceJudge,
    evaluate_retrieval,
    save_retrieval_scores,
)
from generation.llm_client import LLMCompletion


def _case(query: str = "how does auth work") -> EvalCase:
    return EvalCase(
        repository_id="repo-1",
        query=query,
        expected_answer="answer",
        expected_source_files=["src/a.py"],
        expected_functions=["foo"],
    )


def _chunk(chunk_id: str, raw_code: str = "def foo(): pass") -> EvaluatedChunkInput:
    return EvaluatedChunkInput(chunk_id=chunk_id, file_path="src/a.py", function_name="foo", raw_code=raw_code)


class _FakeLLMClient:
    def __init__(self, response_text: str = "4") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []
        self.raise_error: Exception | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_error:
            raise self.raise_error
        return LLMCompletion(
            text=self.response_text, prompt_tokens=1, completion_tokens=1, total_tokens=2, model_name="fake"
        )


class _FakeJudge:
    """Deterministic judge: scores by a caller-supplied mapping of chunk_id -> score."""

    def __init__(self, scores_by_chunk_id: dict[str, int]) -> None:
        self._scores = scores_by_chunk_id
        self.calls: list[tuple[str, str]] = []

    def score(self, query: str, chunk_code: str) -> int:
        self.calls.append((query, chunk_code))
        return self._scores[chunk_code]


class TestLLMRelevanceJudge:
    def test_parses_score_from_response(self) -> None:
        client = _FakeLLMClient(response_text="4")
        judge = LLMRelevanceJudge(client)

        assert judge.score("query", "code") == 4

    def test_parses_score_embedded_in_extra_text(self) -> None:
        client = _FakeLLMClient(response_text="I would say 3 out of 5.")
        judge = LLMRelevanceJudge(client)

        assert judge.score("query", "code") == 3

    def test_unparseable_response_defaults_to_min_score(self) -> None:
        client = _FakeLLMClient(response_text="I cannot determine this.")
        judge = LLMRelevanceJudge(client)

        assert judge.score("query", "code") == 1

    def test_llm_failure_raises_evaluation_error(self) -> None:
        client = _FakeLLMClient()
        client.raise_error = RuntimeError("boom")
        judge = LLMRelevanceJudge(client)

        with pytest.raises(EvaluationError):
            judge.score("query", "code")

    def test_sends_query_and_code_in_prompt(self) -> None:
        client = _FakeLLMClient(response_text="5")
        judge = LLMRelevanceJudge(client)

        judge.score("how does auth work", "def authenticate(): pass")

        _, user_prompt = client.calls[0]
        assert "how does auth work" in user_prompt
        assert "def authenticate(): pass" in user_prompt


class TestEvaluateRetrieval:
    def test_computes_precision_and_mean_relevance(self) -> None:
        chunks = [_chunk("a", "code-a"), _chunk("b", "code-b"), _chunk("c", "code-c")]
        judge = _FakeJudge({"code-a": 5, "code-b": 4, "code-c": 2})

        report = evaluate_retrieval(
            [_case()], retrieve_fn=lambda case: chunks, judge=judge, top_k=3, relevant_threshold=4
        )

        assert len(report.per_query) == 1
        result = report.per_query[0]
        assert [cs.relevance_score for cs in result.chunk_scores] == [5, 4, 2]
        assert result.precision_at_k == pytest.approx(2 / 3)
        assert result.mean_relevance_score == pytest.approx((5 + 4 + 2) / 3)
        assert report.mean_precision_at_k == pytest.approx(2 / 3)
        assert report.mean_relevance_score == pytest.approx((5 + 4 + 2) / 3)

    def test_respects_top_k_limit(self) -> None:
        chunks = [_chunk(f"c{i}", f"code-{i}") for i in range(10)]
        judge = _FakeJudge({f"code-{i}": 5 for i in range(10)})

        report = evaluate_retrieval([_case()], retrieve_fn=lambda case: chunks, judge=judge, top_k=3)

        assert len(report.per_query[0].chunk_scores) == 3
        assert len(judge.calls) == 3

    def test_averages_across_multiple_cases(self) -> None:
        judge = _FakeJudge({"code-a": 5, "code-b": 1})

        def retrieve(case: EvalCase) -> list[EvaluatedChunkInput]:
            return [_chunk("a", "code-a")] if case.query == "q1" else [_chunk("b", "code-b")]

        report = evaluate_retrieval(
            [_case("q1"), _case("q2")], retrieve_fn=retrieve, judge=judge, top_k=1, relevant_threshold=4
        )

        assert report.mean_precision_at_k == pytest.approx((1.0 + 0.0) / 2)
        assert report.mean_relevance_score == pytest.approx((5 + 1) / 2)

    def test_retrieve_fn_failure_raises_evaluation_error(self) -> None:
        def retrieve(case: EvalCase) -> list[EvaluatedChunkInput]:
            raise RuntimeError("boom")

        with pytest.raises(EvaluationError):
            evaluate_retrieval([_case()], retrieve_fn=retrieve, judge=_FakeJudge({}))


class TestSaveRetrievalScores:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        judge = _FakeJudge({"code-a": 5})
        report = evaluate_retrieval([_case()], retrieve_fn=lambda case: [_chunk("a", "code-a")], judge=judge, top_k=1)
        destination = tmp_path / "retrieval_scores.json"

        result_path = save_retrieval_scores(report, destination)

        assert result_path == destination
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["mean_precision_at_k"] == report.mean_precision_at_k
        assert len(payload["per_query"]) == 1

    def test_write_failure_raises_evaluation_error(self, tmp_path: Path) -> None:
        judge = _FakeJudge({"code-a": 5})
        report = evaluate_retrieval([_case()], retrieve_fn=lambda case: [_chunk("a", "code-a")], judge=judge, top_k=1)
        # A path whose parent is itself an existing file cannot be created as a directory.
        blocking_file = tmp_path / "blocking"
        blocking_file.write_text("x", encoding="utf-8")

        with pytest.raises(EvaluationError):
            save_retrieval_scores(report, blocking_file / "retrieval_scores.json")
