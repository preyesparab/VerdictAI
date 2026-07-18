"""Tests for adjudicate.orchestrator.rebuttal_loop.run_rebuttal_loop."""

from __future__ import annotations

from adjudicate.agents.defender import DefenderAgent
from adjudicate.context_builder import ContextBundle
from adjudicate.orchestrator import RebuttalLoopResult, run_rebuttal_loop
from adjudicate.orchestrator.rebuttal_loop import MAX_REBUTTAL_ROUNDS
from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from generation.llm_client import LLMCompletion


class _FakeLLMClient:
    """Returns each entry of `responses` in order, one per call; records every call's user_prompt."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        text = self._responses[len(self.calls) - 1]
        return LLMCompletion(text=text, prompt_tokens=10, completion_tokens=5, total_tokens=15, model_name="fake")


def _bundle() -> ContextBundle:
    return ContextBundle()


def _confirmed() -> VerificationResult:
    return VerificationResult(
        claim=ProsecutorClaim(
            claim_type=ClaimType.MISSING_NULL_CHECK, location="app.py:3",
            assertion="crashes on None", proposed_test="assert foo(None) is None",
        ),
        status=VerificationStatus.CONFIRMED, confidence=Confidence.HIGH,
        evidence="exit_code=1", strategy="proposed_test",
    )


def _inconclusive() -> VerificationResult:
    return VerificationResult(
        claim=ProsecutorClaim(
            claim_type=ClaimType.EXCEPTION_HANDLING, location="x.py:1",
            assertion="too broad", proposed_test="",
        ),
        status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
        evidence="no applicable check", strategy="exception_handling_fallback",
    )


def test_resolves_after_one_round_when_no_claims_are_inconclusive() -> None:
    fake = _FakeLLMClient(["I concede."])
    defender = DefenderAgent(llm_client=fake)

    result = run_rebuttal_loop(_bundle(), "diff", "original justification", [_confirmed()], defender=defender)

    assert result.ended_by == "resolution"
    assert result.rounds_used == 2  # round 1 (initial) + 1 rebuttal
    assert result.transcript == ["original justification", "I concede."]
    assert result.unresolved_claims == []
    assert len(fake.calls) == 1


def test_resolves_immediately_when_no_claims_at_all() -> None:
    """An empty claim list is vacuously resolved - nothing to litigate."""
    fake = _FakeLLMClient(["Nothing to respond to."])
    defender = DefenderAgent(llm_client=fake)

    result = run_rebuttal_loop(_bundle(), "diff", "original justification", [], defender=defender)

    assert result.ended_by == "resolution"
    assert result.rounds_used == 2


def test_hits_cap_when_a_claim_stays_inconclusive() -> None:
    fake = _FakeLLMClient(["first rebuttal attempt", "second rebuttal attempt"])
    defender = DefenderAgent(llm_client=fake)
    claim = _inconclusive()

    result = run_rebuttal_loop(_bundle(), "diff", "original justification", [claim], defender=defender)

    assert result.ended_by == "cap"
    assert result.rounds_used == MAX_REBUTTAL_ROUNDS + 1
    assert result.unresolved_claims == [claim]
    assert len(fake.calls) == MAX_REBUTTAL_ROUNDS


def test_transcript_chains_each_rounds_output_into_the_next_rounds_input() -> None:
    fake = _FakeLLMClient(["first rebuttal attempt", "second rebuttal attempt"])
    defender = DefenderAgent(llm_client=fake)

    result = run_rebuttal_loop(_bundle(), "diff", "original justification", [_inconclusive()], defender=defender)

    assert result.transcript == ["original justification", "first rebuttal attempt", "second rebuttal attempt"]
    # Round 2's prompt must reference round 1's own output, not the original justification again.
    _, second_round_prompt = fake.calls[1]
    assert "first rebuttal attempt" in second_round_prompt
    assert "original justification" not in second_round_prompt


def test_mixed_claims_resolve_once_none_are_inconclusive() -> None:
    fake = _FakeLLMClient(["conceding the confirmed one"])
    defender = DefenderAgent(llm_client=fake)

    result = run_rebuttal_loop(
        _bundle(), "diff", "original justification", [_confirmed()], defender=defender
    )

    assert result.ended_by == "resolution"
    assert len(fake.calls) == 1


def test_result_is_a_rebuttal_loop_result() -> None:
    fake = _FakeLLMClient(["x"])
    defender = DefenderAgent(llm_client=fake)
    result = run_rebuttal_loop(_bundle(), "diff", "original justification", [_confirmed()], defender=defender)
    assert isinstance(result, RebuttalLoopResult)
