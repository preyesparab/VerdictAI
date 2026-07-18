"""Tests for adjudicate.benchmark.baseline_reviewer."""

from __future__ import annotations

import json
from typing import Any

from adjudicate.benchmark.baseline_reviewer import MAX_ATTEMPTS, BaselineReviewer
from adjudicate.context_builder import ChangedFunction, ContextBundle
from core.exceptions import LLMGenerationError
from generation.llm_client import LLMCompletion


class _FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Any]] = []

    def complete(self, system_prompt: str, user_prompt: str, response_schema: Any | None = None) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt, response_schema))
        text = self._responses[len(self.calls) - 1]
        return LLMCompletion(text=text, prompt_tokens=10, completion_tokens=5, total_tokens=15, model_name="fake")


def _bundle() -> ContextBundle:
    return ContextBundle(
        changed_functions=[
            ChangedFunction(
                chunk_id="c1", file_path="app.py", function_name="foo", class_name=None,
                changed_start_line=1, changed_end_line=2,
            )
        ]
    )


def _review_json(has_issue: bool = False, explanation: str = "looks fine") -> str:
    return json.dumps({"has_issue": has_issue, "explanation": explanation})


def test_returns_parsed_review() -> None:
    fake = _FakeLLMClient([_review_json(has_issue=True, explanation="real bug here")])
    reviewer = BaselineReviewer(llm_client=fake)

    result = reviewer.review(_bundle(), "diff text")

    assert result.has_issue is True
    assert result.explanation == "real bug here"
    assert len(fake.calls) == 1


def test_retries_on_wrong_field_name_then_succeeds() -> None:
    """Regression for a real live failure: Groq's weaker JSON mode returned `description`
    instead of the requested `explanation` field name on the first attempt."""
    bad = json.dumps({"has_issue": False, "description": "wrong field name"})
    good = _review_json(has_issue=False, explanation="correct field name")
    fake = _FakeLLMClient([bad, good])
    reviewer = BaselineReviewer(llm_client=fake)

    result = reviewer.review(_bundle(), "diff text")

    assert len(fake.calls) == 2
    assert result.explanation == "correct field name"
    assert "could not be accepted" in fake.calls[1][1]


def test_raises_after_exhausting_all_retries() -> None:
    fake = _FakeLLMClient(["{}"] * MAX_ATTEMPTS)
    reviewer = BaselineReviewer(llm_client=fake)

    try:
        reviewer.review(_bundle(), "diff text")
        assert False, "expected LLMGenerationError"
    except LLMGenerationError as exc:
        assert str(MAX_ATTEMPTS) in str(exc)
    assert len(fake.calls) == MAX_ATTEMPTS


def test_requests_structured_output() -> None:
    fake = _FakeLLMClient([_review_json()])
    reviewer = BaselineReviewer(llm_client=fake)

    reviewer.review(_bundle(), "diff text")

    _, _, response_schema = fake.calls[0]
    assert response_schema is not None
