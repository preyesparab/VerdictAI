"""Tests for adjudicate.benchmark.token_tracking - a fake inner client, no real LLM calls."""

from __future__ import annotations

from adjudicate.benchmark.token_tracking import TokenTrackingLLMClient
from generation.llm_client import LLMCompletion


class _FakeInnerClient:
    def __init__(self, tokens_per_call: list[int]) -> None:
        self._tokens = list(tokens_per_call)
        self.calls = 0

    def complete(self, system_prompt, user_prompt, response_schema=None):
        text = "ok"
        tokens = self._tokens[self.calls]
        self.calls += 1
        return LLMCompletion(text=text, prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens, model_name="fake")


def test_tracks_call_count_and_total_tokens() -> None:
    tracker = TokenTrackingLLMClient(_FakeInnerClient([100, 50]))
    tracker.complete("s", "u")
    tracker.complete("s", "u")
    assert tracker.call_count == 2
    assert tracker.total_tokens == 150


def test_reset_zeroes_counters() -> None:
    tracker = TokenTrackingLLMClient(_FakeInnerClient([100, 50]))
    tracker.complete("s", "u")
    tracker.reset()
    assert tracker.call_count == 0
    assert tracker.total_tokens == 0
    tracker.complete("s", "u")
    assert tracker.call_count == 1
    assert tracker.total_tokens == 50


def test_passes_through_response_schema_and_returns_completion() -> None:
    tracker = TokenTrackingLLMClient(_FakeInnerClient([42]))
    result = tracker.complete("s", "u", response_schema=dict)
    assert result.total_tokens == 42
