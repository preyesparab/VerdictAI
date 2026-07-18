"""Review orchestrator: coordinates the round-by-round adversarial flow.

Defender drafts (Phase 25) -> Prosecutor raises claims (Phase 26/27) ->
Verifier checks them (Phase 28) -> Defender rebuts (Phase 29,
`adjudicate.orchestrator.rebuttal_loop`) -> Judge renders a verdict
(Phase 30, not yet implemented). Owns round sequencing and termination,
not any single agent's logic - `rebuttal_loop.run_rebuttal_loop` never
drafts a rebuttal itself, only calls `adjudicate.agents.defender
.DefenderAgent.rebut` and decides whether another round is needed.
"""

from __future__ import annotations

from adjudicate.orchestrator.rebuttal_loop import RebuttalLoopResult, run_rebuttal_loop

__all__ = ["RebuttalLoopResult", "run_rebuttal_loop"]
