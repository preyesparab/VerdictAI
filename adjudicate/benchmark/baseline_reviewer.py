"""Single-agent baseline reviewer (Phase 31): condition (a) of the benchmark.

No adversarial structure, no verifier - one LLM call, given the same
`ContextBundle` and diff every other condition sees, asked directly
whether the change has a real issue. This is the thing the rest of
Adjudicate (Defender/Prosecutor/Verifier/Rebuttal/Judge, Phases 25-30)
is being benchmarked *against* - it deliberately does not reuse any of
that machinery, since the whole point is measuring what the adversarial-
plus-verification structure buys over a single, unstructured review call.
"""

from __future__ import annotations

from typing import Any, Protocol

from adjudicate.config import AgentRole, adjudicate_settings
from adjudicate.context_builder import (
    ContextBundle,
    format_changed_functions,
    format_neighbors,
    format_related_tests,
)
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from generation.llm_client import LLMClient, LLMCompletion
from pydantic import BaseModel, Field, ValidationError

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
"""Total attempts (1 initial call + up to 2 retries) before giving up on a malformed
response - same bound as Phase 27/30's retry loops. Found necessary live, not
speculatively added: Groq's `response_format={"type": "json_object"}` mode (used when
Gemini's stricter `response_schema` isn't available - see `generation.llm_client`'s own
docstring) only guarantees valid JSON, not the exact field names asked for - a real
first-attempt Groq response used `description` instead of the requested `explanation`."""

SYSTEM_PROMPT: str = """You are a code reviewer looking at a single proposed change. Decide whether it has a \
real, concrete issue (a bug, a crash, a security problem, or genuinely incorrect behavior) - not a style \
preference or a hypothetical "could be better" comment. You must respond with a single JSON object matching \
the given schema exactly - the field names must be exactly `has_issue` and `explanation`, not synonyms - not \
prose.

Ground your judgment strictly in the "Context Bundle" and "Diff" supplied below - you have no other \
knowledge of this repository. If you are not confident there is a real, concrete issue, set `has_issue` to \
false rather than guessing.
"""


class BaselineReviewModel(BaseModel):
    """Structured output schema for the single-agent baseline (Phase 31 only - not a Phase 27 claim)."""

    has_issue: bool
    explanation: str = Field(min_length=1)


class LLMClientProtocol(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(
        self, system_prompt: str, user_prompt: str, response_schema: Any | None = None
    ) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


def _build_user_prompt(context_bundle: ContextBundle, diff: str) -> str:
    """Assemble the user turn: the same Context Bundle sections every other condition sees, plus the diff."""
    return (
        "Context Bundle:\n\n"
        f"Changed Functions:\n{format_changed_functions(context_bundle)}\n\n"
        f"Callers:\n{format_neighbors(context_bundle.callers, '(none found for any changed function)')}\n\n"
        f"Callees:\n{format_neighbors(context_bundle.callees, '(none found for any changed function)')}\n\n"
        f"Related Tests:\n{format_related_tests(context_bundle.related_tests)}\n\n"
        "Diff:\n"
        f"{diff}\n\n"
        "Review this change now, as a single JSON object matching the given schema."
    )


class BaselineReviewer:
    """Runs the single-agent baseline condition via one LLM call."""

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        """Initialize the reviewer.

        Args:
            llm_client: The provider client to generate through.
                Overridable for testing; defaults to a new
                `generation.llm_client.LLMClient` configured with
                `adjudicate.config.adjudicate_settings`'s Defender-role
                model override (there is no dedicated baseline-reviewer
                role - reusing the Defender's model override is an
                arbitrary but harmless choice, since this class has no
                role of its own in `adjudicate.config.AgentRole`).
        """
        self._llm_client = llm_client or LLMClient(
            gemini_model=adjudicate_settings.gemini_model_for(AgentRole.DEFENDER)
        )

    def review(self, context_bundle: ContextBundle, diff: str) -> BaselineReviewModel:
        """Review `diff` in one (possibly retried) call, grounded in `context_bundle`.

        Args:
            context_bundle: The `ContextBundle` built for this diff.
            diff: The raw unified diff text.

        Returns:
            The structured review - `has_issue` plus a brief explanation.

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, the request fails, or the
                response is still malformed after `MAX_ATTEMPTS` attempts.
        """
        base_prompt = _build_user_prompt(context_bundle, diff)

        last_error: ValidationError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            user_prompt = base_prompt if last_error is None else (
                f"{base_prompt}\n\nYour previous response could not be accepted: {last_error}. Return ONLY a "
                "JSON object with exactly the fields `has_issue` (bool) and `explanation` (string)."
            )
            completion = self._llm_client.complete(SYSTEM_PROMPT, user_prompt, response_schema=BaselineReviewModel)
            try:
                return BaselineReviewModel.model_validate_json(completion.text)
            except ValidationError as exc:
                logger.warning(
                    "Baseline reviewer attempt %d/%d produced an unparseable response: %s",
                    attempt, MAX_ATTEMPTS, exc,
                )
                last_error = exc

        raise LLMGenerationError(
            f"Baseline reviewer failed to produce a valid response after {MAX_ATTEMPTS} attempts: {last_error}"
        )
