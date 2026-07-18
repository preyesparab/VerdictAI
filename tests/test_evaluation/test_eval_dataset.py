"""Tests for evaluation.eval_dataset.load_eval_dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.exceptions import EvaluationError
from evaluation.eval_dataset import DEFAULT_EVAL_DATASET_PATH, EvalCase, load_eval_dataset

_VALID_ENTRY = {
    "repository_id": "repo-1",
    "query": "What does foo do?",
    "expected_answer": "It returns 42.",
    "expected_source_files": ["src/foo.py"],
    "expected_functions": ["foo"],
}


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoading:
    def test_loads_valid_entries(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [_VALID_ENTRY])

        cases = load_eval_dataset(path)

        assert cases == [
            EvalCase(
                repository_id="repo-1",
                query="What does foo do?",
                expected_answer="It returns 42.",
                expected_source_files=["src/foo.py"],
                expected_functions=["foo"],
            )
        ]

    def test_loads_multiple_entries_in_file_order(self, tmp_path: Path) -> None:
        second = {**_VALID_ENTRY, "query": "What does bar do?"}
        path = _write(tmp_path, [_VALID_ENTRY, second])

        cases = load_eval_dataset(path)

        assert [case.query for case in cases] == ["What does foo do?", "What does bar do?"]

    def test_empty_array_loads_as_empty_list(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [])

        assert load_eval_dataset(path) == []


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError):
            load_eval_dataset(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.json"
        path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(EvaluationError):
            load_eval_dataset(path)

    def test_non_array_json_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"not": "a list"})

        with pytest.raises(EvaluationError):
            load_eval_dataset(path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        incomplete = {key: value for key, value in _VALID_ENTRY.items() if key != "expected_answer"}
        path = _write(tmp_path, [incomplete])

        with pytest.raises(EvaluationError):
            load_eval_dataset(path)


class TestDefaultDataset:
    def test_default_path_points_at_shipped_dataset_file(self) -> None:
        assert DEFAULT_EVAL_DATASET_PATH.name == "eval_dataset.json"

    def test_shipped_dataset_has_at_least_ten_cases(self) -> None:
        cases = load_eval_dataset()

        assert len(cases) >= 10
        assert all(case.query.strip() for case in cases)
        assert all(case.repository_id.strip() for case in cases)
