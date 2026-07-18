"""Benchmark metrics (Phase 31): pure functions over already-collected results, zero LLM calls.

Every function here takes already-materialized per-case results and
computes a number - none of them call a model, a sandbox, or a network
resource, so they are fully deterministic and unit-testable without
mocking anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseResult:
    """One case's outcome under one condition.

    Attributes:
        case_id: Which benchmark case this is.
        ground_truth_buggy: Whether the case is actually buggy (the
            label established independently of any agent's output - see
            `adjudicate/benchmark/cases/*.json`'s own `ground_truth_rationale`).
        flagged: Whether this condition flagged the case as having an
            issue (definition differs per condition - see
            `adjudicate.benchmark.harness`'s own docstring for exactly
            what "flagged" means for each of the three conditions).
        llm_calls: Real LLM calls spent producing this result.
        total_tokens: Total tokens (prompt + completion) across those calls.
        rounds_used: Rebuttal rounds used (condition c only; None for a/b).
    """

    case_id: str
    ground_truth_buggy: bool
    flagged: bool
    llm_calls: int
    total_tokens: int
    rounds_used: int | None = None


def catch_rate(results: list[CaseResult]) -> float:
    """Fraction of genuinely buggy cases this condition flagged.

    Args:
        results: One `CaseResult` per case (any condition).

    Returns:
        `flagged_buggy_count / buggy_count`, or 0.0 if there are no buggy
        cases in `results` (rather than raising a division error).
    """
    buggy = [r for r in results if r.ground_truth_buggy]
    if not buggy:
        return 0.0
    return sum(1 for r in buggy if r.flagged) / len(buggy)


def false_positive_rate(results: list[CaseResult]) -> float:
    """Fraction of genuinely clean cases this condition flagged anyway.

    Args:
        results: One `CaseResult` per case (any condition).

    Returns:
        `flagged_clean_count / clean_count`, or 0.0 if there are no clean
        cases in `results`.
    """
    clean = [r for r in results if not r.ground_truth_buggy]
    if not clean:
        return 0.0
    return sum(1 for r in clean if r.flagged) / len(clean)


def average_llm_calls(results: list[CaseResult]) -> float:
    """Mean real LLM calls spent per case for this condition."""
    if not results:
        return 0.0
    return sum(r.llm_calls for r in results) / len(results)


def average_tokens(results: list[CaseResult]) -> float:
    """Mean total tokens spent per case for this condition."""
    if not results:
        return 0.0
    return sum(r.total_tokens for r in results) / len(results)


def average_rounds(results: list[CaseResult]) -> float | None:
    """Mean rebuttal rounds used, or None if no result in `results` recorded a round count
    (e.g. conditions a/b, which have no rebuttal-round concept)."""
    rounds = [r.rounds_used for r in results if r.rounds_used is not None]
    if not rounds:
        return None
    return sum(rounds) / len(rounds)


def claim_flip_rate(claim_was_refuted: list[bool]) -> float:
    """Fraction of Prosecutor claims (trusted at face value in condition b) that were
    REFUTED once actually checked by the real Verifier in condition c.

    Args:
        claim_was_refuted: One bool per claim raised across every case -
            True if the Verifier's real result for that claim was
            REFUTED, False otherwise (CONFIRMED or INCONCLUSIVE).

    Returns:
        `refuted_count / total_claim_count`, or 0.0 if no claims were
        raised at all (rather than raising a division error) - this is
        the single strongest number the benchmark produces: how often
        something that looked concerning without verification turned out
        to be unfounded once actually checked.
    """
    if not claim_was_refuted:
        return 0.0
    return sum(1 for refuted in claim_was_refuted if refuted) / len(claim_was_refuted)
