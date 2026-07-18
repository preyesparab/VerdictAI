"""Judge agent (Phase 30): rules on a reviewed change using only verified evidence.

Given the `ContextBundle` (Phase 24), the diff, the full rebuttal
transcript (Phase 25's initial justification plus every Phase 29
rebuttal round), the Phase 28 `VerificationResult`s, and Phase 29's
`termination_reason`, `JudgeAgent.judge` makes one (possibly retried)
LLM call to produce a structured `adjudicate.schemas.JudgeVerdict` - never
free text. The prompt reads the Defender's justification and every
claim's *verified* status/confidence/evidence, never a raw, unverified
Prosecutor claim on its own - the same "read the verified lens, not the
raw assertion" discipline the Phase 29 rebuttal already follows.

Deviation from the task's own shorthand signature (`judge(context_bundle,
diff, full_transcript, termination_reason)`), flagged rather than
silently followed to the letter: that signature has nowhere to carry
"every claim's verified status/confidence/evidence," which the same
task's own prompt-design section requires the Judge to read. Added
`verified_claims: list[VerificationResult]` as a required parameter -
the only way to satisfy that requirement, not an unrequested addition.

Mechanical enforcement of one specific rule, not just a prompt
instruction: if `termination_reason == "cap"` and at least one claim is
still INCONCLUSIVE, a verdict at or above `CAP_UNRESOLVED_CONFIDENCE_CEILING`
confidence *without* a `minority_report` is rejected and retried - this
is the one property Phase 29's own flagged gap makes critical (a "cap"
outcome must never be treated as a quieter version of "resolution"), and
it is cheaply, objectively checkable from structured data (is
`termination_reason` "cap", are there unresolved claims - a plain fact,
not an interpretation). Whether a CONFIRMED claim was "clearly and
explicitly conceded" in free-text rebuttal prose is not mechanically
enforced the same way - assessing that requires exactly the holistic
judgment an LLM (not a keyword/substring check) is suited for, so it
relies on the prompt instruction plus this phase's own live verification,
not a second, brittle text-matching heuristic layered on top.
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
from adjudicate.schemas import JudgeVerdict, JudgeVerdictModel, VerdictParsingError, parse_verdict
from adjudicate.verifier.models import VerificationResult, VerificationStatus, format_verified_claims
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from generation.llm_client import LLMClient, LLMCompletion

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
"""Total attempts (1 initial call + up to 2 retries) before giving up on a malformed
or miscalibrated verdict - same bound as Phase 27's Prosecutor retry loop."""

JUDGE_MAX_TOKENS = 2048
"""Overrides `settings.LLM_MAX_TOKENS` (1024) for the Judge specifically - found live, not assumed:
a real verdict call against a genuinely long rebuttal transcript (Phase 30's own cap/INCONCLUSIVE
verification) truncated mid-JSON (`'{"verdict": "needs_human_review", ..., "cited_'` - cut off
inside a string), reproducibly, across every retry, at the same output length. Gemini 2.5's
internal "thinking" tokens are drawn from the same output budget as the visible JSON, and the
Judge's prompt (a full multi-round transcript, not a single turn like the Defender's or
Prosecutor's) is long enough to need more headroom than the global default provides."""

CAP_UNRESOLVED_CONFIDENCE_CEILING = 0.7
"""If the rebuttal loop hit its round cap with a claim still INCONCLUSIVE, a verdict at or
above this confidence must explain why via `minority_report` - never silently confident, as
if the cap had quietly resolved the uncertainty. See this module's own docstring for why this
one property is mechanically enforced, not left to the prompt alone."""

SYSTEM_PROMPT: str = """You are the Judge agent in an adversarial code-review system. Your job is to rule on \
a proposed code change using ONLY verified evidence and the full rebuttal transcript - never a raw, \
unverified Prosecutor claim on its own. You must respond with a single JSON object matching the given \
schema - not prose.

You must ground your verdict strictly in the "Context Bundle", "Diff", "Verified Claims", "Rebuttal \
Transcript", and "Termination Reason" supplied below - you have no other knowledge of this repository. Rules \
you must follow:
1. Every claim below has already been mechanically checked by the Verifier: CONFIRMED means a real test or \
tool found the claimed problem is real; REFUTED means it was checked and found unfounded; INCONCLUSIVE means \
no tool could decisively check it. Never second-guess a CONFIRMED or REFUTED result, and never treat an \
INCONCLUSIVE claim as if it were resolved just because rebuttal rounds ran out.
2. If any CONFIRMED claim was not clearly and explicitly conceded in the Rebuttal Transcript, your verdict \
must be `reject` or `needs_human_review` - never `approve`.
3. If "Termination Reason" is `cap` and at least one claim below is still INCONCLUSIVE, you must either \
lower your `confidence` to reflect the genuine unresolved uncertainty, or explain it in `minority_report` - \
never rule with high confidence as if the round cap had quietly resolved it.
4. If "Termination Reason" is `resolution`, every claim is already CONFIRMED or REFUTED and has been \
addressed in the transcript - you may rule with a confidence calibrated to what the evidence actually shows, \
without the discount rule 3 requires.
5. `cited_evidence` must reference real evidence from the Verified Claims or Rebuttal Transcript below - \
never invent evidence, a caller, a callee, or a test that isn't listed.
6. `minority_report` must be null unless there is genuine, unresolved uncertainty worth flagging for a \
human - do not populate it reflexively on every verdict.
"""


class LLMClientProtocol(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(
        self, system_prompt: str, user_prompt: str, response_schema: Any | None = None
    ) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


def _format_transcript(full_transcript: list[str]) -> str:
    """Render the rebuttal transcript (Phase 25's justification plus every Phase 29 round) as readable text."""
    if not full_transcript:
        return "(empty)"
    lines = [f"Round 1 (Initial Justification):\n{full_transcript[0]}"]
    for round_number, text in enumerate(full_transcript[1:], start=2):
        lines.append(f"Round {round_number} (Rebuttal):\n{text}")
    return "\n\n".join(lines)


def _build_user_prompt(
    context_bundle: ContextBundle,
    diff: str,
    full_transcript: list[str],
    verified_claims: list[VerificationResult],
    termination_reason: str,
) -> str:
    """Assemble the user turn: Context Bundle, Diff, Verified Claims, Rebuttal Transcript, Termination Reason.

    Args:
        context_bundle: The same `ContextBundle` every agent in this
            review reads from.
        diff: The raw unified diff text.
        full_transcript: `adjudicate.orchestrator.rebuttal_loop.RebuttalLoopResult.transcript` -
            the initial justification plus every rebuttal round, in order.
        verified_claims: The Phase 28 `VerificationResult`s - never the
            Prosecutor's raw unverified claim text alone.
        termination_reason: `"resolution"` or `"cap"` - see
            `adjudicate.orchestrator.rebuttal_loop.RebuttalLoopResult.ended_by`.

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
        "Verified Claims:\n"
        f"{format_verified_claims(verified_claims)}\n\n"
        "Rebuttal Transcript:\n"
        f"{_format_transcript(full_transcript)}\n\n"
        f"Termination Reason: {termination_reason}\n\n"
        "Rule on this change now, as a single JSON object matching the given schema."
    )


def _cap_handling_violation(
    verdict: JudgeVerdict, termination_reason: str, unresolved_claims: list[VerificationResult]
) -> str | None:
    """Whether `verdict` illegally treats an unresolved `cap` outcome as if it were resolved.

    Returns:
        A description of the violation, or None if `verdict` handled the
        cap correctly (either a low-enough `confidence`, or a populated
        `minority_report`).
    """
    if termination_reason == "cap" and unresolved_claims and verdict.minority_report is None:
        if verdict.confidence >= CAP_UNRESOLVED_CONFIDENCE_CEILING:
            return (
                f"termination_reason='cap' with {len(unresolved_claims)} claim(s) still INCONCLUSIVE, but "
                f"confidence={verdict.confidence} >= {CAP_UNRESOLVED_CONFIDENCE_CEILING} and minority_report "
                "is null - genuine unresolved uncertainty must not be treated as resolved."
            )
    return None


class JudgeAgent:
    """Rules on a reviewed change via one (possibly retried) LLM call, producing a structured `JudgeVerdict`.

    Never reads the repository directly, never re-derives a verification
    result - every fact it may cite comes from the `ContextBundle`, the
    diff, the already-verified claims, and the already-produced rebuttal
    transcript, all supplied by the caller.
    """

    role: str = "judge"

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        """Initialize the agent.

        Args:
            llm_client: The provider client to generate through.
                Overridable for testing; defaults to a new
                `generation.llm_client.LLMClient` configured with
                `adjudicate.config.adjudicate_settings`'s Judge-role model
                override (falls back to `config.settings.GEMINI_MODEL` if
                unset) and `JUDGE_MAX_TOKENS` (see that constant's own
                docstring for why the Judge specifically needs more
                headroom than `settings.LLM_MAX_TOKENS`'s default).
        """
        self._llm_client = llm_client or LLMClient(
            gemini_model=adjudicate_settings.gemini_model_for(AgentRole.JUDGE), max_tokens=JUDGE_MAX_TOKENS
        )

    def judge(
        self,
        context_bundle: ContextBundle,
        diff: str,
        full_transcript: list[str],
        verified_claims: list[VerificationResult],
        termination_reason: str,
    ) -> JudgeVerdict:
        """Rule on `diff`, grounded in `verified_claims` and `full_transcript`.

        Args:
            context_bundle: The `ContextBundle` built for this diff
                (Phase 24) - the only source of callers/callees/related
                tests the Judge may cite.
            diff: The raw unified diff text.
            full_transcript: The initial justification plus every
                rebuttal round (Phase 29's `RebuttalLoopResult.transcript`).
            verified_claims: The Phase 28 `VerificationResult`s this
                verdict must be grounded in - never the Prosecutor's raw
                unverified claim text.
            termination_reason: `"resolution"` or `"cap"` (Phase 29's
                `RebuttalLoopResult.ended_by`) - governs how confidence
                must be calibrated when a claim is still INCONCLUSIVE.

        Returns:
            The structured verdict.

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, the request fails, or the response
                is still malformed or miscalibrated after `MAX_ATTEMPTS`
                attempts.
        """
        unresolved_claims = [c for c in verified_claims if c.status == VerificationStatus.INCONCLUSIVE]
        logger.info(
            "Judge ruling: %d verified claim(s) (%d unresolved), %d transcript round(s), termination_reason=%s",
            len(verified_claims), len(unresolved_claims), len(full_transcript), termination_reason,
        )
        base_prompt = _build_user_prompt(context_bundle, diff, full_transcript, verified_claims, termination_reason)

        last_error: VerdictParsingError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            user_prompt = base_prompt if last_error is None else (
                f"{base_prompt}\n\n"
                f"Your previous response could not be accepted: {last_error}. Return ONLY a JSON object "
                "conforming to the schema, calibrated correctly to the Termination Reason above."
            )
            completion = self._llm_client.complete(SYSTEM_PROMPT, user_prompt, response_schema=JudgeVerdictModel)

            try:
                verdict = parse_verdict(completion.text)
            except VerdictParsingError as exc:
                logger.warning("Judge attempt %d/%d produced an unparseable verdict: %s", attempt, MAX_ATTEMPTS, exc)
                last_error = exc
                continue

            violation = _cap_handling_violation(verdict, termination_reason, unresolved_claims)
            if violation is not None:
                logger.warning("Judge attempt %d/%d mishandled the round cap: %s", attempt, MAX_ATTEMPTS, violation)
                last_error = VerdictParsingError(violation)
                continue

            logger.info(
                "Judge verdict: %s (confidence=%.2f, %d cited, minority_report=%s)",
                verdict.verdict.value, verdict.confidence, len(verdict.cited_evidence),
                "yes" if verdict.minority_report else "no",
            )
            return verdict

        raise LLMGenerationError(
            f"Judge failed to produce a valid, properly-calibrated verdict after {MAX_ATTEMPTS} attempts: {last_error}"
        )


def judge(
    context_bundle: ContextBundle,
    diff: str,
    full_transcript: list[str],
    verified_claims: list[VerificationResult],
    termination_reason: str,
    llm_client: LLMClientProtocol | None = None,
) -> JudgeVerdict:
    """Module-level convenience wrapper: rule on a change without constructing a `JudgeAgent` directly.

    Args:
        context_bundle: The `ContextBundle` built for this diff (Phase 24).
        diff: The raw unified diff text.
        full_transcript: The initial justification plus every rebuttal round.
        verified_claims: The Phase 28 `VerificationResult`s to rule on.
        termination_reason: `"resolution"` or `"cap"`.
        llm_client: Overridable provider client, forwarded to
            `JudgeAgent.__init__`; defaults to a real `LLMClient`.

    Returns:
        The structured verdict - see `JudgeAgent.judge`.
    """
    return JudgeAgent(llm_client=llm_client).judge(
        context_bundle, diff, full_transcript, verified_claims, termination_reason
    )
