"""Defender agent (Phase 25; rebuttal added Phase 29): drafts a grounded justification for a diff.

Given a `ContextBundle` (Phase 24) and the raw diff it was built from,
`DefenderAgent.draft_justification` makes exactly one LLM call to produce
a plain-text argument for the change - what it does, why, and what's
already tested. The prompt reads only from the bundle's four fields
(`changed_functions`, `callers`, `callees`, `related_tests`) plus the diff
text itself, never the whole repository, and is explicitly instructed to
say so plainly when `related_tests` is empty rather than inventing
coverage that doesn't exist.

`DefenderAgent.rebut` (Phase 29) is the same agent's second capability:
given the Verifier's (Phase 28) real, mechanical results on the
Prosecutor's claims, respond appropriately - concede every CONFIRMED
claim outright (real test/tool evidence, not arguable), acknowledge
REFUTED ones, and push back on INCONCLUSIVE ones only with real
counter-evidence from the same `ContextBundle` - never by just restating
the original position. It reads the *verified* status/confidence/evidence,
never the Prosecutor's raw unverified claim text.

Deviation (flagged, not silently added): this does not subclass
`adjudicate.agents.base.BaseAgent`. `BaseAgent.review` takes a single
`context: Any` argument, but the Defender needs both the `ContextBundle`
and the raw diff text (the bundle alone doesn't say what specifically
changed within a changed function, only that it did) - forcing that into
one argument would mean inventing a wrapper type for no other reason than
fitting an interface no orchestrator yet drives through. `BaseAgent`
stays as-is for whichever later phase actually calls agents
polymorphically.

No claim schema lives here - Phase 27's `adjudicate.schemas` owns that.
Verification itself lives in `adjudicate.verifier` (Phase 28) and never
here - this module only reads a `VerificationResult`'s already-decided
status, never re-derives or second-guesses it.
"""

from __future__ import annotations

from typing import Protocol

from adjudicate.config import AgentRole, adjudicate_settings
from adjudicate.context_builder import (
    ContextBundle,
    format_changed_functions,
    format_neighbors,
    format_related_tests,
)
from adjudicate.verifier.models import VerificationResult, format_verified_claims
from core.logging import get_logger
from generation.llm_client import LLMClient, LLMCompletion

logger = get_logger(__name__)

SYSTEM_PROMPT: str = """You are the Defender agent in an adversarial code-review system. Your job is to \
draft a plain-text justification for a proposed code change: what it does, why it was made, and what test \
coverage already exists for it.

You must ground every claim strictly in the "Context Bundle" and "Diff" supplied below - you have no other \
knowledge of this repository. Rules you must follow:
1. Base "what it does" and "why" only on the diff itself and the listed changed function(s) - never \
speculate about behavior, intent, or history that isn't shown in the diff or the bundle.
2. Never state that a caller, callee, or relationship exists unless it is explicitly listed under \
"Callers" or "Callees" below. If a section is empty, say so plainly (e.g. "no callers were found for this \
change") instead of inventing one.
3. For test coverage, only cite tests listed under "Related Tests" below. If that section is empty, state \
explicitly that no existing tests were found for this change - never invent, assume, or imply test coverage \
that isn't listed.
4. Output plain, readable prose only. No JSON, no bullet-point claim schema, no scoring - just a clear, \
grounded justification a human reviewer could read directly.
"""


REBUTTAL_SYSTEM_PROMPT: str = """You are the Defender agent in an adversarial code-review system, now \
responding to the Verifier's real, mechanical results on the Prosecutor's claims about a code change you \
already argued for. Your job is to update your position honestly in light of real evidence, not to keep \
arguing regardless of what was found.

You must ground every statement strictly in the "Context Bundle", "Diff", "Your Prior Statement", and \
"Verified Claims" supplied below - you have no other knowledge of this repository. Rules you must follow:
1. For every claim marked CONFIRMED below, concede it outright and plainly - the Verifier ran a real test or \
tool and found the claimed problem is real. Never argue with, minimize, or ignore a CONFIRMED claim's \
evidence.
2. For every claim marked REFUTED below, you may note plainly that it was checked and found unfounded, \
citing the evidence - state it and move on, don't gloat or belabor it.
3. For every claim marked INCONCLUSIVE below, you may push back only with actual counter-evidence drawn \
from the Context Bundle (a real caller, callee, or test explicitly listed there) - never by simply restating \
your prior statement's position without new grounding. If you have no new grounded counter-evidence for an \
inconclusive claim, say so plainly rather than repeating yourself.
4. Never invent a caller, callee, test, or tool result beyond what's explicitly listed below.
5. Output plain, readable prose only. No JSON, no bullet-point schema, no scoring.
"""


class LLMClientProtocol(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


def _build_user_prompt(context_bundle: ContextBundle, diff: str) -> str:
    """Assemble the user turn: Context Bundle sections, in a fixed order, followed by the raw diff.

    Args:
        context_bundle: The `adjudicate.context_builder.ContextBundle` for
            this diff (Phase 24's output) - the only source of
            callers/callees/tests the Defender may cite.
        diff: The raw unified diff text - the only source of "what
            specifically changed" (the bundle names the enclosing
            function, not the edit itself).

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
        "Draft your justification for this change now."
    )


def _build_rebuttal_user_prompt(
    context_bundle: ContextBundle, diff: str, prior_statement: str, verified_claims: list[VerificationResult]
) -> str:
    """Assemble the user turn for a rebuttal round.

    Args:
        context_bundle: The same `ContextBundle` every agent in this
            review reads from.
        diff: The raw unified diff text.
        prior_statement: The statement being responded to - the original
            Phase 25 justification on the first rebuttal round, or the
            previous round's rebuttal on a later one (labeled "Your Prior
            Statement" either way, since it's accurate for both).
        verified_claims: The Phase 28 `VerificationResult`s to respond to.

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
        "Your Prior Statement:\n"
        f"{prior_statement}\n\n"
        "Verified Claims:\n"
        f"{format_verified_claims(verified_claims)}\n\n"
        "Respond now: concede every CONFIRMED claim, acknowledge REFUTED ones briefly, and only push back "
        "on INCONCLUSIVE claims with real counter-evidence from the Context Bundle above."
    )


class DefenderAgent:
    """Drafts a grounded, plain-text justification for a diff via one LLM call.

    Never reads the repository directly - every fact it may cite comes
    from the `ContextBundle` it's given plus the diff text, both supplied
    by the caller (`adjudicate.context_builder.AdjudicateContextBuilder`,
    Phase 24).
    """

    role: str = "defender"

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        """Initialize the agent.

        Args:
            llm_client: The provider client to generate through.
                Overridable for testing; defaults to a new
                `generation.llm_client.LLMClient` configured with
                `adjudicate.config.adjudicate_settings`'s Defender-role
                model override (falls back to `config.settings.GEMINI_MODEL`
                if unset).
        """
        self._llm_client = llm_client or LLMClient(
            gemini_model=adjudicate_settings.gemini_model_for(AgentRole.DEFENDER)
        )

    def draft_justification(self, context_bundle: ContextBundle, diff: str) -> str:
        """Draft a plain-text justification for `diff`, grounded in `context_bundle`.

        Args:
            context_bundle: The `ContextBundle` built for this diff
                (Phase 24) - the only source of callers/callees/related
                tests the Defender may cite.
            diff: The raw unified diff text this justification is for.

        Returns:
            The justification text, as returned by the LLM - what the
            change does, why, and what's already tested (or an explicit
            statement that no tests were found).

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, or the generation request fails.
        """
        logger.info(
            "Defender drafting justification: %d changed function(s), %d caller(s), %d callee(s), %d related test(s)",
            len(context_bundle.changed_functions),
            len(context_bundle.callers),
            len(context_bundle.callees),
            len(context_bundle.related_tests),
        )
        user_prompt = _build_user_prompt(context_bundle, diff)
        completion = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
        logger.info("Defender justification generated: %d total token(s)", completion.total_tokens)
        return completion.text

    def rebut(
        self,
        context_bundle: ContextBundle,
        diff: str,
        original_justification: str,
        verified_claims: list[VerificationResult],
    ) -> str:
        """Respond to the Verifier's real results on the Prosecutor's claims (Phase 29).

        Args:
            context_bundle: The `ContextBundle` built for this diff
                (Phase 24) - the only source of callers/callees/related
                tests the Defender may cite as counter-evidence.
            diff: The raw unified diff text.
            original_justification: The statement being responded to -
                the Defender's own prior justification (or, in a second
                rebuttal round, its own previous rebuttal - see
                `adjudicate.orchestrator.rebuttal_loop.run_rebuttal_loop`,
                which chains rounds this way).
            verified_claims: The Phase 28 `VerificationResult`s to
                respond to - never the Prosecutor's raw unverified claim
                text.

        Returns:
            The rebuttal text - concessions for CONFIRMED claims,
            brief acknowledgement of REFUTED ones, and grounded pushback
            (or an honest "no further evidence" admission) for
            INCONCLUSIVE ones.

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, or the generation request fails.
        """
        logger.info(
            "Defender rebutting: %d verified claim(s) (%s)",
            len(verified_claims),
            ", ".join(f"{r.status.value}" for r in verified_claims) or "none",
        )
        user_prompt = _build_rebuttal_user_prompt(context_bundle, diff, original_justification, verified_claims)
        completion = self._llm_client.complete(REBUTTAL_SYSTEM_PROMPT, user_prompt)
        logger.info("Defender rebuttal generated: %d total token(s)", completion.total_tokens)
        return completion.text


def draft_justification(
    context_bundle: ContextBundle, diff: str, llm_client: LLMClientProtocol | None = None
) -> str:
    """Module-level convenience wrapper: draft a justification without constructing a `DefenderAgent` directly.

    Args:
        context_bundle: The `ContextBundle` built for this diff (Phase 24).
        diff: The raw unified diff text this justification is for.
        llm_client: Overridable provider client, forwarded to
            `DefenderAgent.__init__`; defaults to a real `LLMClient`.

    Returns:
        The justification text - see `DefenderAgent.draft_justification`.
    """
    return DefenderAgent(llm_client=llm_client).draft_justification(context_bundle, diff)


def rebut(
    context_bundle: ContextBundle,
    diff: str,
    original_justification: str,
    verified_claims: list[VerificationResult],
    llm_client: LLMClientProtocol | None = None,
) -> str:
    """Module-level convenience wrapper: rebut without constructing a `DefenderAgent` directly.

    Args:
        context_bundle: The `ContextBundle` built for this diff (Phase 24).
        diff: The raw unified diff text.
        original_justification: The statement being responded to.
        verified_claims: The Phase 28 `VerificationResult`s to respond to.
        llm_client: Overridable provider client, forwarded to
            `DefenderAgent.__init__`; defaults to a real `LLMClient`.

    Returns:
        The rebuttal text - see `DefenderAgent.rebut`.
    """
    return DefenderAgent(llm_client=llm_client).rebut(context_bundle, diff, original_justification, verified_claims)
