"""Tests for evaluation.ragas_eval.

`_run_ragas_evaluate` (the module's sole `ragas`/`datasets` import site) is
monkeypatched throughout - these tests verify dataset-row construction,
score aggregation, and error propagation, never a real RAGAS/LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.ragas_eval as ragas_eval_module
from core.exceptions import EvaluationError
from evaluation.ragas_eval import RagasCase, evaluate_generation, save_ragas_results


def _case(query: str = "how does auth work") -> RagasCase:
    return RagasCase(
        query=query,
        answer="Auth uses JWT.",
        retrieved_contexts=["def authenticate(): ..."],
        reference="Auth uses JSON Web Tokens.",
    )


class TestDatasetConstruction:
    def test_passes_one_row_per_case_in_ragas_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_rows = []

        def _fake_runner(rows, llm, embeddings):
            captured_rows.extend(rows)
            return [{"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0}]

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)

        evaluate_generation([_case()], llm="fake-llm", embeddings="fake-embeddings")

        assert captured_rows == [
            {
                "question": "how does auth work",
                "answer": "Auth uses JWT.",
                "contexts": ["def authenticate(): ..."],
                "ground_truth": "Auth uses JSON Web Tokens.",
            }
        ]

    def test_forwards_llm_and_embeddings_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def _fake_runner(rows, llm, embeddings):
            captured["llm"] = llm
            captured["embeddings"] = embeddings
            return [{"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0}]

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)

        evaluate_generation([_case()], llm="my-llm", embeddings="my-embeddings")

        assert captured == {"llm": "my-llm", "embeddings": "my-embeddings"}


class TestAggregation:
    def test_averages_scores_across_cases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_runner(rows, llm, embeddings):
            return [
                {"faithfulness": 1.0, "answer_relevancy": 0.8, "context_precision": 0.6, "context_recall": 0.4},
                {"faithfulness": 0.0, "answer_relevancy": 0.6, "context_precision": 0.4, "context_recall": 0.2},
            ]

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)

        report = evaluate_generation([_case("q1"), _case("q2")], llm="llm", embeddings="emb")

        assert report.faithfulness == pytest.approx(0.5)
        assert report.answer_relevancy == pytest.approx(0.7)
        assert report.context_precision == pytest.approx(0.5)
        assert report.context_recall == pytest.approx(0.3)
        assert [result.query for result in report.per_case] == ["q1", "q2"]


class TestErrorHandling:
    def test_empty_cases_raises_without_calling_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def _fake_runner(rows, llm, embeddings):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)

        with pytest.raises(EvaluationError):
            evaluate_generation([], llm="llm", embeddings="emb")

        assert called is False

    def test_runner_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raising_runner(rows, llm, embeddings):
            raise EvaluationError("ragas exploded")

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _raising_runner)

        with pytest.raises(EvaluationError):
            evaluate_generation([_case()], llm="llm", embeddings="emb")

    def test_missing_ragas_import_raises_evaluation_error(self) -> None:
        # Exercises the real (unmocked) `_run_ragas_evaluate` - `ragas` is
        # genuinely not installed in this project's environment (see this
        # module's docstring), so this documents that real behavior
        # directly rather than simulating it.
        with pytest.raises(EvaluationError):
            evaluate_generation([_case()], llm="llm", embeddings="emb")


class TestSaveRagasResults:
    def test_writes_json_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_runner(rows, llm, embeddings):
            return [{"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0}]

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)
        report = evaluate_generation([_case()], llm="llm", embeddings="emb")
        destination = tmp_path / "ragas_results.json"

        result_path = save_ragas_results(report, destination)

        assert result_path == destination
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["faithfulness"] == 1.0
        assert len(payload["per_case"]) == 1

    def test_write_failure_raises_evaluation_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_runner(rows, llm, embeddings):
            return [{"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0}]

        monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)
        report = evaluate_generation([_case()], llm="llm", embeddings="emb")
        blocking_file = tmp_path / "blocking"
        blocking_file.write_text("x", encoding="utf-8")

        with pytest.raises(EvaluationError):
            save_ragas_results(report, blocking_file / "ragas_results.json")
