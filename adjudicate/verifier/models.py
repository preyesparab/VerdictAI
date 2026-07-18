"""Verification result schema (Phase 28).

`VerificationResult` is every strategy's uniform return shape - the
Verifier's dispatch table (`adjudicate.verifier.strategies.DISPATCH`)
guarantees every claim comes back in this shape regardless of which
concrete strategy (a sandboxed test run, a static analyzer, the existing
test suite, or an honest "no applicable check yet") handled it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from adjudicate.schemas import ProsecutorClaim


class VerificationStatus(str, Enum):
    """The Verifier's verdict on one claim."""

    CONFIRMED = "confirmed"
    """The claimed problem was actually observed (a test failed, an analyzer hit)."""

    REFUTED = "refuted"
    """The claimed problem did not occur (a test passed, an analyzer found nothing)."""

    INCONCLUSIVE = "inconclusive"
    """No applicable check could produce a real answer - not silently defaulted to a verdict."""


class Confidence(str, Enum):
    """How much weight a later Judge should give this result.

    Tied to *what kind* of evidence produced the verdict, not to how the
    verdict came out - a REFUTED result from a real test is exactly as
    high-confidence as a CONFIRMED one from a real test.
    """

    HIGH = "high"
    """A real test (the claim's own `proposed_test`) actually ran and passed or failed."""

    MEDIUM = "medium"
    """A static analyzer (bandit, mypy) or the existing test suite ran and hit or missed."""

    LOW = "low"
    """`INCONCLUSIVE` only - no applicable check exists for this claim, explicitly labeled
    rather than silently treated as any other confidence level."""


@dataclass(frozen=True)
class VerificationResult:
    """One claim's mechanical verification outcome.

    Attributes:
        claim: The `ProsecutorClaim` this result is for.
        status: CONFIRMED, REFUTED, or INCONCLUSIVE.
        confidence: How much weight this result should carry.
        evidence: The actual tool/test output that produced `status` -
            real stdout/stderr/exit codes or analyzer findings, never a
            paraphrase or an LLM-generated summary (this module makes
            zero LLM calls - see `adjudicate.verifier.strategies`'s
            module docstring).
        strategy: Which dispatch strategy produced this result (e.g.
            ``"proposed_test"``, ``"bandit"``, ``"mypy"``,
            ``"existing_test_suite"``) - for observability/debugging,
            not part of the verdict itself.
    """

    claim: ProsecutorClaim
    status: VerificationStatus
    confidence: Confidence
    evidence: str
    strategy: str


def format_verified_claims(verified_claims: list[VerificationResult]) -> str:
    """Render a list of `VerificationResult`s as readable text for an LLM prompt.

    Shared by every agent that reads verified claims (the Defender's
    rebuttal, Phase 29; the Judge, Phase 30) so each renders the same
    data the same way - moved here (not left private to `adjudicate.agents
    .defender`, where it started) once a second real consumer needed it,
    the same "extract on the second real use" discipline
    `adjudicate.context_builder.format_changed_functions` followed in
    Phase 27. Always shows the *verified* status/confidence/evidence,
    never the Prosecutor's raw claim alone - the whole point of both
    phases that use this is responding to what was actually checked.
    """
    if not verified_claims:
        return "(none - the Prosecutor raised no claims for this change)"
    lines = []
    for result in verified_claims:
        claim = result.claim
        lines.append(
            f"- [{result.status.value.upper()}, {result.confidence.value} confidence] "
            f"{claim.claim_type.value} at `{claim.location}`: {claim.assertion}\n"
            f"  Evidence (via {result.strategy}):\n"
            + "\n".join(f"    {line}" for line in result.evidence.strip().splitlines())
        )
    return "\n".join(lines)
