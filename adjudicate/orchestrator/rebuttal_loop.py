"""Defender rebuttal loop (Phase 29): bounds and sequences the Defender's response to verified claims.

`run_rebuttal_loop` is pure round-coordination - it never drafts a
rebuttal itself, only calls `adjudicate.agents.defender.DefenderAgent.rebut`
(Phase 29's own new agent capability) and decides whether another round is
needed, matching the split this package's own Phase 23 scaffold already
described: "owns round sequencing and termination, not any single agent's
logic."

Round counting: round 1 is the initial justification (Phase 25's output,
already given as input here - not itself a `rebut` call). Up to
`MAX_REBUTTAL_ROUNDS` further rounds each call `rebut`, chaining each
round's own output back in as the next round's "prior statement" (so a
second rebuttal round responds to the first rebuttal, not back to the
original justification again).

Termination is decided mechanically, not by guessing whether an argument
was "persuasive" (that is Phase 30's job, not this one's): a round set is
"resolved" once every claim is already CONFIRMED or REFUTED - both are
final, real Verifier results nothing in this loop can change, so once
none remain INCONCLUSIVE there is nothing further to rebut. If at least
one claim is still INCONCLUSIVE once `MAX_REBUTTAL_ROUNDS` is exhausted,
the loop ends "by cap" instead - explicitly logged, since that distinction
is what the Judge (Phase 30) needs to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from adjudicate.agents.defender import DefenderAgent
from adjudicate.context_builder import ContextBundle
from adjudicate.verifier.models import VerificationResult, VerificationStatus
from core.logging import get_logger

logger = get_logger(__name__)

MAX_REBUTTAL_ROUNDS = 2
"""Up to 2 `rebut` calls on top of the initial (already-given) justification - 3 rounds total."""


@dataclass(frozen=True)
class RebuttalLoopResult:
    """Outcome of running the Defender rebuttal loop to completion.

    Attributes:
        transcript: ``[original_justification, rebuttal_1, (rebuttal_2)?]``,
            in round order.
        rounds_used: Total rounds counted, including the initial
            justification as round 1 - always 2 or 3, never more than
            ``MAX_REBUTTAL_ROUNDS + 1``.
        ended_by: ``"resolution"`` if every claim was already CONFIRMED or
            REFUTED (mechanically settled by the Verifier, needing only
            the Defender's acknowledgement) by the end of a round;
            ``"cap"`` if at least one claim was still INCONCLUSIVE when
            `MAX_REBUTTAL_ROUNDS` was reached.
        unresolved_claims: The INCONCLUSIVE `VerificationResult`s still
            open when the loop ended - empty if `ended_by == "resolution"`.
    """

    transcript: list[str]
    rounds_used: int
    ended_by: Literal["resolution", "cap"]
    unresolved_claims: list[VerificationResult]


def _is_resolved(verified_claims: list[VerificationResult]) -> bool:
    """Whether every claim is already mechanically settled (CONFIRMED or REFUTED).

    An INCONCLUSIVE claim is one Phase 28's own tools couldn't decisively
    check - nothing in this loop re-runs verification, so "resolved" here
    means "nothing is left that only a real test/tool result could
    settle," not "the Judge would find the Defender's rebuttal
    persuasive" (a judgment call, explicitly out of scope for this loop).
    """
    return all(result.status != VerificationStatus.INCONCLUSIVE for result in verified_claims)


def run_rebuttal_loop(
    context_bundle: ContextBundle,
    diff: str,
    original_justification: str,
    verified_claims: list[VerificationResult],
    defender: DefenderAgent | None = None,
) -> RebuttalLoopResult:
    """Run the Defender rebuttal loop, bounded to `MAX_REBUTTAL_ROUNDS` rebuttal calls.

    Args:
        context_bundle: The same `ContextBundle` every agent in this
            review reads from.
        diff: The raw unified diff text.
        original_justification: The Defender's Phase 25 justification -
            round 1, already produced, not itself a `rebut` call.
        verified_claims: The Phase 28 `VerificationResult`s to respond to.
        defender: Overridable for testing; defaults to a new
            `DefenderAgent`.

    Returns:
        The full transcript plus why the loop stopped.
    """
    agent = defender or DefenderAgent()
    transcript = [original_justification]

    for round_index in range(1, MAX_REBUTTAL_ROUNDS + 1):
        rebuttal = agent.rebut(context_bundle, diff, transcript[-1], verified_claims)
        transcript.append(rebuttal)
        rounds_used = round_index + 1  # +1 counts the initial justification as round 1

        if _is_resolved(verified_claims):
            logger.info("Rebuttal loop resolved after %d round(s)", rounds_used)
            return RebuttalLoopResult(
                transcript=transcript, rounds_used=rounds_used, ended_by="resolution", unresolved_claims=[]
            )

        logger.info(
            "Rebuttal round %d did not resolve every claim; %d round(s) remaining",
            rounds_used, MAX_REBUTTAL_ROUNDS - round_index,
        )

    unresolved = [result for result in verified_claims if result.status == VerificationStatus.INCONCLUSIVE]
    logger.info("Rebuttal loop hit the round cap with %d claim(s) still inconclusive", len(unresolved))
    return RebuttalLoopResult(
        transcript=transcript, rounds_used=MAX_REBUTTAL_ROUNDS + 1, ended_by="cap", unresolved_claims=unresolved
    )
