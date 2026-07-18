"""Prosecutor agent (Phase 27): raises structured, falsifiable claims about a diff.

Given the same `ContextBundle` (Phase 24) the Defender used, the raw
diff, and the Defender's own justification (Phase 25),
`ProsecutorAgent.raise_concerns` makes one LLM call - using Gemini's
structured-output mode (`generation.llm_client.LLMClient.complete`'s new
`response_schema` parameter, Phase 27's own addition to that module) - to
produce a JSON array of `adjudicate.schemas.ProsecutorClaim`s instead of
free text (Phase 26's shape). Each claim is grounded the same way the
Defender is (bundle + diff only), and mechanically restricted to
functions in `changed_functions` by `adjudicate.schemas.parse_claims`,
not just prompted for - a claim about any other file is rejected, not
merely discouraged.

A malformed or ungrounded response is retried, with a corrective note
appended to the prompt, up to `MAX_ATTEMPTS` times before raising
`core.exceptions.LLMGenerationError` - Phase 27's explicit requirement
that bad structured output actually gets rejected, not silently accepted.

No verification of a claim's *truth* happens here - a claim can be
well-formed and still wrong (see `adjudicate.schemas`'s module docstring
for this session's own ground-truth example: the `login`/`missing_null_check`
claim that turned out to be false). Mechanically checking a claim against
the real code is Phase 28's Verifier, not this module's job.

Deviation (same as the Defender, Phase 25/26): does not subclass
`BaseAgent` - `review(context: Any)` takes one argument, but the
Prosecutor needs the bundle, the diff, and the Defender's justification.
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
from adjudicate.schemas import ClaimParsingError, ProsecutorClaim, ProsecutorClaimModel, parse_claims
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from generation.llm_client import LLMClient, LLMCompletion

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
"""Total attempts (1 initial call + up to 2 retries) before giving up on a malformed response."""

SYSTEM_PROMPT: str = """You are the Prosecutor agent in an adversarial code-review system. Your job is to \
raise specific, checkable concerns about a proposed code change - arguing against it, not summarizing it - \
given the same evidence the Defender used and the Defender's own justification. You must respond with a \
JSON array of claim objects conforming to the given schema - not prose.

You must ground every claim strictly in the "Context Bundle", "Diff", and "Defender's Justification" \
supplied below - you have no other knowledge of this repository. Rules you must follow:
1. Each claim's `claim_type` must be the category that actually fits: `missing_null_check` (a value that can \
be null/undefined/None is used without a guard), `untested_branch` (new logic - a branch, or an entire \
change - with no covering test), `type_mismatch` (code assumes a value's type without checking it), or \
`exception_handling` (exception handling that is missing, too broad, or silently swallows errors). Do not \
force a claim into a category it doesn't fit.
2. Each claim's `location` must be exactly `<file_path>:<line>`, where `<file_path>` is copied verbatim from \
one of the "Changed Functions" listed below and `<line>` is a real line number for that function. Never \
reference a file or function that is not explicitly listed under "Changed Functions" - the bundle may not \
capture every function this diff actually touched, but you must stick strictly to what you are given rather \
than speculate about what else might have changed.
3. Each claim's `assertion` must be a specific, falsifiable statement - never vague "this could be cleaner" \
or generic best-practice commentary. Each claim's `proposed_test` must be actual code (a test body, or a \
short repro script) that would concretely prove or disprove the assertion if run - not a description of a \
test.
4. Only cite callers, callees, or tests explicitly listed under "Callers", "Callees", or "Related Tests" \
below - never invent one. If "Related Tests" is empty, raise that absence itself as an `untested_branch` \
claim rather than assuming or inventing coverage to excuse the gap.
5. Directly challenge the Defender's justification where the bundle or diff contradicts or fails to fully \
support it - do not simply restate or agree with it. If, after genuinely scrutinizing everything, there is \
truly no specific, checkable concern, return an empty array rather than manufacturing a vague claim to \
appear thorough.
"""


class LLMClientProtocol(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(
        self, system_prompt: str, user_prompt: str, response_schema: Any | None = None
    ) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


def _build_user_prompt(context_bundle: ContextBundle, diff: str, defender_justification: str) -> str:
    """Assemble the user turn: Context Bundle sections, the raw diff, then the Defender's justification.

    Args:
        context_bundle: The `adjudicate.context_builder.ContextBundle` for
            this diff (Phase 24's output) - the only source of
            callers/callees/tests the Prosecutor may cite.
        diff: The raw unified diff text - the only source of "what
            specifically changed".
        defender_justification: `DefenderAgent.draft_justification`'s
            output (Phase 25) for the same bundle/diff - what the
            Prosecutor is arguing against.

    Returns:
        The user turn text.
    """
    return (
        "Context Bundle:\n\n"
        f"Changed Functions:\n{format_changed_functions(context_bundle)}\n\n"
        f"Callers:\n{format_neighbors(context_bundle.callers, '(none found for any changed function)')}\n\n"
        f"Callees:\n{format_neighbors(context_bundle.callees, '(none found for any changed function)')}\n\n"
        f"Related Tests:\n{format_related_tests(context_bundle.related_tests)}\n\n"
        "Diff:\n"
        f"{diff}\n\n"
        "Defender's Justification:\n"
        f"{defender_justification}\n\n"
        "Raise your concerns about this change now, as a JSON array of claims."
    )


class ProsecutorAgent:
    """Raises structured, falsifiable claims about a diff via one (possibly retried) LLM call.

    Never reads the repository directly - every fact it may cite comes
    from the `ContextBundle` it's given, the diff text, and the
    Defender's justification, all supplied by the caller.
    """

    role: str = "prosecutor"

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        """Initialize the agent.

        Args:
            llm_client: The provider client to generate through.
                Overridable for testing; defaults to a new
                `generation.llm_client.LLMClient` configured with
                `adjudicate.config.adjudicate_settings`'s Prosecutor-role
                model override (falls back to `config.settings.GEMINI_MODEL`
                if unset).
        """
        self._llm_client = llm_client or LLMClient(
            gemini_model=adjudicate_settings.gemini_model_for(AgentRole.PROSECUTOR)
        )

    def raise_concerns(
        self, context_bundle: ContextBundle, diff: str, defender_justification: str
    ) -> list[ProsecutorClaim]:
        """Raise structured concerns about `diff`, grounded in `context_bundle` and the Defender's justification.

        Args:
            context_bundle: The `ContextBundle` built for this diff
                (Phase 24) - the only source of callers/callees/related
                tests the Prosecutor may cite, and the set of file paths
                every claim's `location` is validated against.
            diff: The raw unified diff text this review is for.
            defender_justification: The Defender's (Phase 25) justification
                for the same diff - what the Prosecutor argues against.

        Returns:
            The claims, in the model's given order. Empty if the model
            found nothing to raise.

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, the request fails, or the
                response is still malformed/ungrounded after
                `MAX_ATTEMPTS` attempts.
        """
        logger.info(
            "Prosecutor raising concerns: %d changed function(s), %d caller(s), %d callee(s), %d related test(s)",
            len(context_bundle.changed_functions),
            len(context_bundle.callers),
            len(context_bundle.callees),
            len(context_bundle.related_tests),
        )
        valid_file_paths = {fn.file_path for fn in context_bundle.changed_functions}
        base_prompt = _build_user_prompt(context_bundle, diff, defender_justification)

        last_error: ClaimParsingError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            user_prompt = base_prompt if last_error is None else (
                f"{base_prompt}\n\n"
                f"Your previous response could not be accepted: {last_error}. Return ONLY a JSON array "
                "conforming to the schema, with every claim's location file_path copied verbatim from "
                "one of the Changed Functions listed above."
            )
            completion = self._llm_client.complete(
                SYSTEM_PROMPT, user_prompt, response_schema=list[ProsecutorClaimModel]
            )
            try:
                claims = parse_claims(completion.text, valid_file_paths)
            except ClaimParsingError as exc:
                logger.warning(
                    "Prosecutor attempt %d/%d produced unparseable claims: %s | raw response text: %r",
                    attempt, MAX_ATTEMPTS, exc, completion.text,
                )
                last_error = exc
                continue

            logger.info(
                "Prosecutor concerns generated: %d claim(s), %d total token(s)",
                len(claims), completion.total_tokens,
            )
            return claims

        raise LLMGenerationError(
            f"Prosecutor failed to produce valid structured claims after {MAX_ATTEMPTS} attempts: {last_error}"
        )


def raise_concerns(
    context_bundle: ContextBundle,
    diff: str,
    defender_justification: str,
    llm_client: LLMClientProtocol | None = None,
) -> list[ProsecutorClaim]:
    """Module-level convenience wrapper: raise concerns without constructing a `ProsecutorAgent` directly.

    Args:
        context_bundle: The `ContextBundle` built for this diff (Phase 24).
        diff: The raw unified diff text this review is for.
        defender_justification: The Defender's justification for the same diff.
        llm_client: Overridable provider client, forwarded to
            `ProsecutorAgent.__init__`; defaults to a real `LLMClient`.

    Returns:
        The claims - see `ProsecutorAgent.raise_concerns`.
    """
    return ProsecutorAgent(llm_client=llm_client).raise_concerns(context_bundle, diff, defender_justification)
