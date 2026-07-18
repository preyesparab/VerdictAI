"""Tests for adjudicate.agents.prosecutor."""

from __future__ import annotations

import json
from typing import Any

from adjudicate.agents.prosecutor import MAX_ATTEMPTS, ProsecutorAgent, raise_concerns
from adjudicate.context_builder import ChangedFunction, ContextBundle, GraphNeighbor, RelatedTest
from adjudicate.schemas import ClaimType, ProsecutorClaim
from core.exceptions import LLMGenerationError
from generation.llm_client import LLMCompletion


def _claim_json(
    claim_type: str = "missing_null_check",
    location: str = "server/src/controllers/auth.controller.js:47",
    assertion: str = "a concern",
    proposed_test: str = "expect(true).toBe(true);",
) -> str:
    return json.dumps(
        [{"claim_type": claim_type, "location": location, "assertion": assertion, "proposed_test": proposed_test}]
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


def test_raise_concerns_returns_parsed_claims() -> None:
    fake = _FakeLLMClient([_claim_json(assertion="user may be null here")])
    agent = ProsecutorAgent(llm_client=fake)

    claims = agent.raise_concerns(_bundle(), "diff text", "the Defender's justification")

    assert claims == [
        ProsecutorClaim(
            claim_type=ClaimType.MISSING_NULL_CHECK,
            location="server/src/controllers/auth.controller.js:47",
            assertion="user may be null here",
            proposed_test="expect(true).toBe(true);",
        )
    ]
    assert len(fake.calls) == 1


def test_requests_structured_output_via_response_schema() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)

    agent.raise_concerns(_bundle(), "diff text", "justification")

    _, _, response_schema = fake.calls[0]
    assert response_schema is not None


def test_module_level_wrapper_delegates_to_agent() -> None:
    fake = _FakeLLMClient([_claim_json(assertion="via wrapper")])
    claims = raise_concerns(_bundle(), "some diff", "justification", llm_client=fake)
    assert claims[0].assertion == "via wrapper"
    assert len(fake.calls) == 1


def test_prompt_includes_bundle_diff_and_defender_justification() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)
    diff_text = "@@ -1,2 +1,2 @@\n-return hash(pw)\n+return hash(pw, salt)\n"
    justification = "This change adds a salt to password hashing."

    agent.raise_concerns(_bundle(), diff_text, justification)

    _, user_prompt, _ = fake.calls[0]
    assert "auth.controller.js" in user_prompt
    assert "register" in user_prompt
    assert diff_text in user_prompt
    assert justification in user_prompt


def test_prompt_says_no_related_tests_when_bundle_has_none() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)

    agent.raise_concerns(_bundle(related_tests=[]), "diff text", "justification")

    _, user_prompt, _ = fake.calls[0]
    assert "none found" in user_prompt.lower()


def test_prompt_lists_related_test_when_bundle_has_one() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)
    test = RelatedTest(
        chunk_id="t1", file_path="tests/auth.test.js", function_name="testRegister", found_via="graph"
    )

    agent.raise_concerns(_bundle(related_tests=[test]), "diff text", "justification")

    _, user_prompt, _ = fake.calls[0]
    assert "tests/auth.test.js" in user_prompt
    assert "testRegister" in user_prompt


def test_prompt_lists_callers_and_callees() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)
    caller = GraphNeighbor(
        node_id="n1", node_kind="chunk", file_path="Register.jsx", function_name="handleSubmit", class_name=None
    )
    callee = GraphNeighbor(
        node_id="n2", node_kind="chunk", file_path="auth.controller.js", function_name="sendToken", class_name=None
    )

    agent.raise_concerns(_bundle(callers=[caller], callees=[callee]), "diff text", "justification")

    _, user_prompt, _ = fake.calls[0]
    assert "Register.jsx" in user_prompt and "handleSubmit" in user_prompt
    assert "sendToken" in user_prompt


def test_system_prompt_instructs_categories_scope_and_challenge() -> None:
    fake = _FakeLLMClient([_claim_json()])
    agent = ProsecutorAgent(llm_client=fake)

    agent.raise_concerns(_bundle(), "diff text", "justification")

    system_prompt = fake.calls[0][0]
    lowered = system_prompt.lower()
    assert "missing_null_check" in lowered
    assert "untested_branch" in lowered
    assert "type_mismatch" in lowered
    assert "exception_handling" in lowered
    assert "changed functions" in lowered
    assert "do not simply restate or agree" in lowered


def test_retries_once_on_malformed_json_then_succeeds() -> None:
    """The first response is unparseable (bad JSON); the second is valid - must retry, not fail immediately."""
    fake = _FakeLLMClient(["not json at all", _claim_json(assertion="recovered on retry")])
    agent = ProsecutorAgent(llm_client=fake)

    claims = agent.raise_concerns(_bundle(), "diff text", "justification")

    assert len(fake.calls) == 2
    assert claims[0].assertion == "recovered on retry"
    # The retry's prompt must include a corrective note referencing the failure.
    assert "could not be accepted" in fake.calls[1][1]


def test_retries_on_claim_referencing_a_file_outside_the_bundle() -> None:
    """Structurally valid JSON that fails the groundedness check must also trigger a retry."""
    ungrounded = _claim_json(location="server/src/controllers/some_other_file.js:5")
    fake = _FakeLLMClient([ungrounded, _claim_json(assertion="grounded this time")])
    agent = ProsecutorAgent(llm_client=fake)

    claims = agent.raise_concerns(_bundle(), "diff text", "justification")

    assert len(fake.calls) == 2
    assert claims[0].assertion == "grounded this time"


def test_raises_after_exhausting_all_retries_on_persistently_malformed_output() -> None:
    """Every attempt returns malformed JSON - must exhaust MAX_ATTEMPTS and raise, never silently accept."""
    fake = _FakeLLMClient(["still not json"] * MAX_ATTEMPTS)
    agent = ProsecutorAgent(llm_client=fake)

    try:
        agent.raise_concerns(_bundle(), "diff text", "justification")
        assert False, "expected LLMGenerationError"
    except LLMGenerationError as exc:
        assert str(MAX_ATTEMPTS) in str(exc)

    assert len(fake.calls) == MAX_ATTEMPTS
