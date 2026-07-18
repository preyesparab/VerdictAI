"""Verifier entry point (Phase 28): dispatches one claim to its verification strategy.

`verify_claim` is the only function the rest of Adjudicate (a future
orchestrator, Phase 29+) needs to call - it looks `claim.claim_type` up
in `adjudicate.verifier.strategies.DISPATCH` and runs the matching
strategy. No LLM call happens here or in anything this module calls.
"""

from __future__ import annotations

from pathlib import Path

from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from adjudicate.verifier.strategies import DISPATCH


def verify_claim(
    claim: ProsecutorClaim, repo_path: Path, test_command: list[str] | None = None
) -> VerificationResult:
    """Verify one `ProsecutorClaim` against the real, diff-applied code at `repo_path`.

    Args:
        claim: The claim to verify.
        repo_path: A real source tree with the diff already applied -
            every strategy runs real tools/tests against this, never
            against a description of the code.
        test_command: Required only for `ClaimType.BREAKING_CHANGE`
            claims - the command that runs the affected file's existing
            test suite (see `adjudicate.verifier.strategies.verify_breaking_change`).
            Ignored for every other claim type.

    Returns:
        The verification result. `VerificationStatus.INCONCLUSIVE` at
        `Confidence.LOW` if `claim.claim_type` has no dispatch entry, or
        if it's `BREAKING_CHANGE` and `test_command` wasn't given -
        never silently defaulted to CONFIRMED or REFUTED.
    """
    strategy = DISPATCH.get(claim.claim_type)
    if strategy is None:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.INCONCLUSIVE,
            confidence=Confidence.LOW,
            evidence=f"No verification strategy registered for claim_type {claim.claim_type!r}.",
            strategy="none",
        )

    if claim.claim_type == ClaimType.BREAKING_CHANGE:
        if test_command is None:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.INCONCLUSIVE,
                confidence=Confidence.LOW,
                evidence="breaking_change verification requires a test_command; none was given.",
                strategy="existing_test_suite",
            )
        return strategy(claim, repo_path, test_command)

    return strategy(claim, repo_path)
