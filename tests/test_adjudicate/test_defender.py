"""Tests for adjudicate.agents.defender."""

from __future__ import annotations

from adjudicate.agents.defender import DefenderAgent, draft_justification, rebut
from adjudicate.context_builder import ChangedFunction, ContextBundle, GraphNeighbor, RelatedTest
from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from generation.llm_client import LLMCompletion


class _FakeLLMClient:
    """Records the (system_prompt, user_prompt) it was called with and returns a canned completion."""

    def __init__(self, text: str = "a justification") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        return LLMCompletion(
            text=self.text, prompt_tokens=10, completion_tokens=5, total_tokens=15, model_name="fake-model"
        )


_CHANGED_FUNCTION = ChangedFunction(
    chunk_id="c1",
    file_path="server/src/controllers/auth.controller.js",
    function_name="register",
    class_name=None,
    changed_start_line=10,
    changed_end_line=20,
)


def _bundle(**overrides: object) -> ContextBundle:
    defaults: dict[str, object] = {
        "changed_functions": [_CHANGED_FUNCTION],
        "callers": [],
        "callees": [],
        "related_tests": [],
    }
    defaults.update(overrides)
    return ContextBundle(**defaults)  # type: ignore[arg-type]


def test_draft_justification_returns_llm_text() -> None:
    fake = _FakeLLMClient(text="This change updates register() to hash passwords before storage.")
    agent = DefenderAgent(llm_client=fake)

    result = agent.draft_justification(_bundle(), "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+new\n")

    assert result == "This change updates register() to hash passwords before storage."
    assert len(fake.calls) == 1


def test_module_level_wrapper_delegates_to_agent() -> None:
    fake = _FakeLLMClient(text="justification via wrapper")
    result = draft_justification(_bundle(), "some diff", llm_client=fake)
    assert result == "justification via wrapper"
    assert len(fake.calls) == 1


def test_prompt_includes_changed_function_and_diff() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)
    diff_text = "@@ -1,2 +1,2 @@\n-return hash(pw)\n+return hash(pw, salt)\n"

    agent.draft_justification(_bundle(), diff_text)

    _, user_prompt = fake.calls[0]
    assert "auth.controller.js" in user_prompt
    assert "register" in user_prompt
    assert diff_text in user_prompt


def test_prompt_says_no_related_tests_when_bundle_has_none() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.draft_justification(_bundle(related_tests=[]), "diff text")

    _, user_prompt = fake.calls[0]
    assert "none found" in user_prompt.lower()


def test_prompt_lists_related_test_when_bundle_has_one() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)
    test = RelatedTest(
        chunk_id="t1", file_path="tests/auth.test.js", function_name="testRegister", found_via="graph"
    )

    agent.draft_justification(_bundle(related_tests=[test]), "diff text")

    _, user_prompt = fake.calls[0]
    assert "tests/auth.test.js" in user_prompt
    assert "testRegister" in user_prompt
    assert "found via graph" in user_prompt


def test_prompt_lists_callers_and_callees() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)
    caller = GraphNeighbor(
        node_id="n1", node_kind="chunk", file_path="Register.jsx", function_name="handleSubmit", class_name=None
    )
    callee = GraphNeighbor(
        node_id="n2", node_kind="chunk", file_path="auth.controller.js", function_name="sendToken", class_name=None
    )

    agent.draft_justification(_bundle(callers=[caller], callees=[callee]), "diff text")

    _, user_prompt = fake.calls[0]
    assert "Register.jsx" in user_prompt and "handleSubmit" in user_prompt
    assert "sendToken" in user_prompt


def test_system_prompt_instructs_grounding_and_no_invented_tests() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.draft_justification(_bundle(), "diff text")

    system_prompt, _ = fake.calls[0]
    assert "never invent" in system_prompt.lower()
    assert "context bundle" in system_prompt.lower()


# -- rebut (Phase 29) ----------------------------------------------------------

_CONFIRMED_CLAIM = VerificationResult(
    claim=ProsecutorClaim(
        claim_type=ClaimType.MISSING_NULL_CHECK, location="app.py:3",
        assertion="foo crashes on None", proposed_test="assert foo(None) is None",
    ),
    status=VerificationStatus.CONFIRMED, confidence=Confidence.HIGH,
    evidence="exit_code=1\nTypeError: ...", strategy="proposed_test",
)

_REFUTED_CLAIM = VerificationResult(
    claim=ProsecutorClaim(
        claim_type=ClaimType.MISSING_NULL_CHECK, location="auth.controller.js:47",
        assertion="user may be null", proposed_test="assert login(...) does not throw",
    ),
    status=VerificationStatus.REFUTED, confidence=Confidence.HIGH,
    evidence="exit_code=0\nPASS: no crash", strategy="proposed_test",
)

_INCONCLUSIVE_CLAIM = VerificationResult(
    claim=ProsecutorClaim(
        claim_type=ClaimType.EXCEPTION_HANDLING, location="x.py:1",
        assertion="too broad an except clause", proposed_test="",
    ),
    status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
    evidence="No verification strategy exists yet for exception_handling claims without a proposed_test.",
    strategy="exception_handling_fallback",
)


def test_rebut_returns_llm_text() -> None:
    fake = _FakeLLMClient(text="I concede the CONFIRMED claim.")
    agent = DefenderAgent(llm_client=fake)

    result = agent.rebut(_bundle(), "diff text", "original justification", [_CONFIRMED_CLAIM])

    assert result == "I concede the CONFIRMED claim."
    assert len(fake.calls) == 1


def test_module_level_rebut_wrapper_delegates_to_agent() -> None:
    fake = _FakeLLMClient(text="rebuttal via wrapper")
    result = rebut(_bundle(), "diff text", "original justification", [_CONFIRMED_CLAIM], llm_client=fake)
    assert result == "rebuttal via wrapper"
    assert len(fake.calls) == 1


def test_rebut_uses_separate_system_prompt_from_draft_justification() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.draft_justification(_bundle(), "diff text")
    agent.rebut(_bundle(), "diff text", "original justification", [_CONFIRMED_CLAIM])

    draft_system_prompt, _ = fake.calls[0]
    rebuttal_system_prompt, _ = fake.calls[1]
    assert draft_system_prompt != rebuttal_system_prompt


def test_rebuttal_prompt_includes_prior_statement_and_diff() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)
    diff_text = "@@ -1,2 +1,2 @@\n-return hash(pw)\n+return hash(pw, salt)\n"

    agent.rebut(_bundle(), diff_text, "the original justification text", [_CONFIRMED_CLAIM])

    _, user_prompt = fake.calls[0]
    assert "the original justification text" in user_prompt
    assert diff_text in user_prompt


def test_rebuttal_prompt_includes_verified_status_confidence_and_evidence() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.rebut(_bundle(), "diff text", "original justification", [_CONFIRMED_CLAIM])

    _, user_prompt = fake.calls[0]
    assert "CONFIRMED" in user_prompt
    assert "high confidence" in user_prompt
    assert "TypeError" in user_prompt
    assert "foo crashes on None" in user_prompt


def test_rebuttal_prompt_includes_all_three_statuses_when_present() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.rebut(
        _bundle(), "diff text", "original justification",
        [_CONFIRMED_CLAIM, _REFUTED_CLAIM, _INCONCLUSIVE_CLAIM],
    )

    _, user_prompt = fake.calls[0]
    assert "CONFIRMED" in user_prompt
    assert "REFUTED" in user_prompt
    assert "INCONCLUSIVE" in user_prompt


def test_rebuttal_prompt_says_no_claims_when_list_empty() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.rebut(_bundle(), "diff text", "original justification", [])

    _, user_prompt = fake.calls[0]
    assert "none" in user_prompt.lower()


def test_rebuttal_system_prompt_instructs_concede_confirmed_and_ground_inconclusive_pushback() -> None:
    fake = _FakeLLMClient()
    agent = DefenderAgent(llm_client=fake)

    agent.rebut(_bundle(), "diff text", "original justification", [_CONFIRMED_CLAIM])

    system_prompt, _ = fake.calls[0]
    lowered = system_prompt.lower()
    assert "concede" in lowered
    assert "inconclusive" in lowered
    assert "never invent" in lowered
