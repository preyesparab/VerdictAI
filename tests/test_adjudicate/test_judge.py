"""Tests for adjudicate.agents.judge."""

from __future__ import annotations

import json
from typing import Any

from adjudicate.agents.judge import CAP_UNRESOLVED_CONFIDENCE_CEILING, JUDGE_MAX_TOKENS, MAX_ATTEMPTS, JudgeAgent, judge
from adjudicate.context_builder import ChangedFunction, ContextBundle
from adjudicate.schemas import ClaimType, JudgeVerdict, ProsecutorClaim, Verdict
from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from core.exceptions import LLMGenerationError
from generation.llm_client import LLMCompletion


def _verdict_json(
    verdict: str = "approve", confidence: float = 0.9,
    cited_evidence: list[str] | None = None, minority_report: str | None = None,
) -> str:
    return json.dumps(
        {
            "verdict": verdict, "confidence": confidence,
            "cited_evidence": cited_evidence if cited_evidence is not None else ["some real evidence"],
            "minority_report": minority_report,
        }
    )


class _FakeLLMClient:
    """Returns each entry of `responses` in order, one per call; records every call's kwargs."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Any]] = []

    def complete(self, system_prompt: str, user_prompt: str, response_schema: Any | None = None) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt, response_schema))
        text = self._responses[len(self.calls) - 1]
        return LLMCompletion(text=text, prompt_tokens=10, completion_tokens=5, total_tokens=15, model_name="fake")


_CHANGED_FUNCTION = ChangedFunction(
    chunk_id="c1", file_path="server/src/controllers/auth.controller.js", function_name="login",
    class_name=None, changed_start_line=40, changed_end_line=50,
)


def _bundle(**overrides: object) -> ContextBundle:
    defaults: dict[str, object] = {
        "changed_functions": [_CHANGED_FUNCTION], "callers": [], "callees": [], "related_tests": [],
    }
    defaults.update(overrides)
    return ContextBundle(**defaults)  # type: ignore[arg-type]


def _refuted_claim() -> VerificationResult:
    return VerificationResult(
        claim=ProsecutorClaim(
            claim_type=ClaimType.MISSING_NULL_CHECK, location="server/src/controllers/auth.controller.js:47",
            assertion="user may be null", proposed_test="assert login(...) does not throw",
        ),
        status=VerificationStatus.REFUTED, confidence=Confidence.HIGH,
        evidence="exit_code=0\nPASS: no crash", strategy="proposed_test",
    )


def _confirmed_claim() -> VerificationResult:
    return VerificationResult(
        claim=ProsecutorClaim(
            claim_type=ClaimType.MISSING_NULL_CHECK, location="user_service.py:3",
            assertion="crashes on None", proposed_test="assert foo(None) is None",
        ),
        status=VerificationStatus.CONFIRMED, confidence=Confidence.HIGH,
        evidence="exit_code=1\nTypeError: ...", strategy="proposed_test",
    )


def _inconclusive_claim() -> VerificationResult:
    return VerificationResult(
        claim=ProsecutorClaim(
            claim_type=ClaimType.EXCEPTION_HANDLING, location="x.py:1",
            assertion="too broad an except clause", proposed_test="",
        ),
        status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
        evidence="no applicable check", strategy="exception_handling_fallback",
    )


def test_judge_returns_parsed_verdict() -> None:
    fake = _FakeLLMClient([_verdict_json(verdict="approve", confidence=0.9)])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["justification"], [_refuted_claim()], "resolution")

    assert result.verdict == Verdict.APPROVE
    assert result.confidence == 0.9
    assert len(fake.calls) == 1


def test_module_level_wrapper_delegates_to_agent() -> None:
    fake = _FakeLLMClient([_verdict_json()])
    result = judge(_bundle(), "diff text", ["justification"], [_refuted_claim()], "resolution", llm_client=fake)
    assert isinstance(result, JudgeVerdict)
    assert len(fake.calls) == 1


def test_requests_structured_output_via_response_schema() -> None:
    fake = _FakeLLMClient([_verdict_json()])
    agent = JudgeAgent(llm_client=fake)

    agent.judge(_bundle(), "diff text", ["justification"], [_refuted_claim()], "resolution")

    _, _, response_schema = fake.calls[0]
    assert response_schema is not None


def test_prompt_includes_bundle_diff_transcript_and_termination_reason() -> None:
    fake = _FakeLLMClient([_verdict_json()])
    agent = JudgeAgent(llm_client=fake)

    agent.judge(_bundle(), "diff text here", ["round one text", "round two text"], [_refuted_claim()], "cap")

    _, user_prompt, _ = fake.calls[0]
    assert "auth.controller.js" in user_prompt
    assert "diff text here" in user_prompt
    assert "round one text" in user_prompt
    assert "round two text" in user_prompt
    assert "Termination Reason: cap" in user_prompt


def test_prompt_includes_verified_claim_status_and_evidence() -> None:
    fake = _FakeLLMClient([_verdict_json()])
    agent = JudgeAgent(llm_client=fake)

    agent.judge(_bundle(), "diff text", ["justification"], [_confirmed_claim()], "resolution")

    _, user_prompt, _ = fake.calls[0]
    assert "CONFIRMED" in user_prompt
    assert "TypeError" in user_prompt


def test_system_prompt_instructs_key_rules() -> None:
    fake = _FakeLLMClient([_verdict_json()])
    agent = JudgeAgent(llm_client=fake)

    agent.judge(_bundle(), "diff text", ["justification"], [_refuted_claim()], "resolution")

    system_prompt = fake.calls[0][0]
    lowered = system_prompt.lower()
    assert "never" in lowered and "second-guess" in lowered
    assert "reject" in lowered and "needs_human_review" in lowered
    assert "cap" in lowered and "minority_report" in lowered


def test_retries_on_malformed_json_then_succeeds() -> None:
    fake = _FakeLLMClient(["not json at all", _verdict_json(verdict="reject", confidence=0.8)])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["justification"], [_confirmed_claim()], "resolution")

    assert len(fake.calls) == 2
    assert result.verdict == Verdict.REJECT
    assert "could not be accepted" in fake.calls[1][1]


def test_raises_after_exhausting_all_retries_on_persistently_malformed_output() -> None:
    fake = _FakeLLMClient(["still not json"] * MAX_ATTEMPTS)
    agent = JudgeAgent(llm_client=fake)

    try:
        agent.judge(_bundle(), "diff text", ["justification"], [_confirmed_claim()], "resolution")
        assert False, "expected LLMGenerationError"
    except LLMGenerationError as exc:
        assert str(MAX_ATTEMPTS) in str(exc)
    assert len(fake.calls) == MAX_ATTEMPTS


# -- the cap/INCONCLUSIVE mechanical enforcement (the one novel behavior this phase needs) --


def test_high_confidence_without_minority_report_on_cap_with_inconclusive_is_rejected_and_retried() -> None:
    """The core rule: termination_reason='cap' + an unresolved INCONCLUSIVE claim + high confidence +
    no minority_report must never be accepted as-is - the loop must retry."""
    bad = _verdict_json(verdict="approve", confidence=0.95, minority_report=None)
    good = _verdict_json(verdict="needs_human_review", confidence=0.4, minority_report="genuinely uncertain")
    fake = _FakeLLMClient([bad, good])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["j", "r1"], [_inconclusive_claim()], "cap")

    assert len(fake.calls) == 2
    assert result.confidence == 0.4
    assert result.minority_report == "genuinely uncertain"
    assert "could not be accepted" in fake.calls[1][1]
    assert "cap" in fake.calls[1][1].lower()


def test_high_confidence_with_minority_report_on_cap_is_accepted() -> None:
    """A minority_report is a sufficient way to handle the uncertainty - confidence need not be lowered too."""
    text = _verdict_json(
        verdict="needs_human_review", confidence=0.85,
        minority_report="High confidence overall, but one claim could not be mechanically checked.",
    )
    fake = _FakeLLMClient([text])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["j"], [_inconclusive_claim()], "cap")

    assert len(fake.calls) == 1
    assert result.confidence == 0.85


def test_low_confidence_without_minority_report_on_cap_is_accepted() -> None:
    """Lowering confidence is also a sufficient way to handle the uncertainty - minority_report isn't required."""
    text = _verdict_json(verdict="needs_human_review", confidence=CAP_UNRESOLVED_CONFIDENCE_CEILING - 0.1)
    fake = _FakeLLMClient([text])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["j"], [_inconclusive_claim()], "cap")

    assert len(fake.calls) == 1
    assert result.confidence < CAP_UNRESOLVED_CONFIDENCE_CEILING


def test_high_confidence_is_fine_on_cap_when_nothing_is_actually_unresolved() -> None:
    """If every claim is CONFIRMED/REFUTED despite termination_reason='cap' (an edge case, but not
    contradictory - see RebuttalLoopResult's own definition), the cap-handling check doesn't apply."""
    fake = _FakeLLMClient([_verdict_json(verdict="approve", confidence=0.95, minority_report=None)])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["j"], [_refuted_claim()], "cap")

    assert len(fake.calls) == 1
    assert result.confidence == 0.95


def test_high_confidence_is_fine_on_resolution_even_with_no_minority_report() -> None:
    fake = _FakeLLMClient([_verdict_json(verdict="approve", confidence=0.95, minority_report=None)])
    agent = JudgeAgent(llm_client=fake)

    result = agent.judge(_bundle(), "diff text", ["j"], [_refuted_claim()], "resolution")

    assert len(fake.calls) == 1
    assert result.confidence == 0.95


def test_default_llm_client_uses_judge_max_tokens() -> None:
    """Regression for a real live failure: a genuinely long rebuttal transcript truncated the
    Judge's JSON response mid-field under the global default max_tokens - see JUDGE_MAX_TOKENS's
    own docstring. The default-constructed LLMClient must actually use the larger budget."""
    agent = JudgeAgent()
    assert agent._llm_client._max_tokens == JUDGE_MAX_TOKENS
