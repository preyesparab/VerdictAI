"""Tests for adjudicate.benchmark.metrics - pure functions, no LLM involved."""

from __future__ import annotations

from adjudicate.benchmark.metrics import (
    CaseResult,
    average_llm_calls,
    average_rounds,
    average_tokens,
    catch_rate,
    claim_flip_rate,
    false_positive_rate,
)


def _result(case_id: str, buggy: bool, flagged: bool, llm_calls: int = 1, tokens: int = 100, rounds=None) -> CaseResult:
    return CaseResult(
        case_id=case_id, ground_truth_buggy=buggy, flagged=flagged,
        llm_calls=llm_calls, total_tokens=tokens, rounds_used=rounds,
    )


def test_catch_rate_all_buggy_cases_flagged() -> None:
    results = [_result("a", True, True), _result("b", True, True)]
    assert catch_rate(results) == 1.0


def test_catch_rate_half_buggy_cases_flagged() -> None:
    results = [_result("a", True, True), _result("b", True, False)]
    assert catch_rate(results) == 0.5


def test_catch_rate_ignores_clean_cases() -> None:
    results = [_result("a", True, True), _result("b", False, True)]
    assert catch_rate(results) == 1.0


def test_catch_rate_zero_buggy_cases_returns_zero_not_error() -> None:
    results = [_result("a", False, True)]
    assert catch_rate(results) == 0.0


def test_false_positive_rate_flags_a_clean_case() -> None:
    results = [_result("a", False, True), _result("b", False, False)]
    assert false_positive_rate(results) == 0.5


def test_false_positive_rate_ignores_buggy_cases() -> None:
    results = [_result("a", False, False), _result("b", True, True)]
    assert false_positive_rate(results) == 0.0


def test_false_positive_rate_zero_clean_cases_returns_zero_not_error() -> None:
    results = [_result("a", True, True)]
    assert false_positive_rate(results) == 0.0


def test_average_llm_calls() -> None:
    results = [_result("a", True, True, llm_calls=2), _result("b", True, True, llm_calls=4)]
    assert average_llm_calls(results) == 3.0


def test_average_llm_calls_empty_returns_zero() -> None:
    assert average_llm_calls([]) == 0.0


def test_average_tokens() -> None:
    results = [_result("a", True, True, tokens=100), _result("b", True, True, tokens=300)]
    assert average_tokens(results) == 200.0


def test_average_rounds_only_counts_results_with_rounds() -> None:
    results = [_result("a", True, True, rounds=2), _result("b", True, True, rounds=None)]
    assert average_rounds(results) == 2.0


def test_average_rounds_returns_none_when_nothing_has_rounds() -> None:
    results = [_result("a", True, True, rounds=None), _result("b", True, True, rounds=None)]
    assert average_rounds(results) is None


def test_claim_flip_rate_all_refuted() -> None:
    assert claim_flip_rate([True, True]) == 1.0


def test_claim_flip_rate_none_refuted() -> None:
    assert claim_flip_rate([False, False]) == 0.0


def test_claim_flip_rate_mixed() -> None:
    assert claim_flip_rate([True, False, True, False]) == 0.5


def test_claim_flip_rate_no_claims_returns_zero_not_error() -> None:
    assert claim_flip_rate([]) == 0.0
