"""Verifier layer (Phase 28): mechanically checks Prosecutor claims against reality.

Takes each structured `adjudicate.schemas.ProsecutorClaim` (Phase 27) and
checks it against the real repository - a sandboxed run of the claim's
own `proposed_test`, a static analyzer (bandit/mypy), or the existing
test suite - instead of trusting either agent's prose. **Zero LLM calls
anywhere in this package** - see `adjudicate.verifier.strategies`'s
module docstring for what to do instead if a claim ever seems to need
one.

Public API: `verify_claim` (the dispatch entry point),
`VerificationResult`/`VerificationStatus`/`Confidence` (the result
schema). Sandboxing is `subprocess`-based (hard timeout, best-effort
network denial), not Docker - `adjudicate.verifier.sandbox`'s module
docstring flags real OS-level isolation as a follow-up, not something
this phase blocks on.
"""

from __future__ import annotations

from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from adjudicate.verifier.verifier import verify_claim

__all__ = ["Confidence", "VerificationResult", "VerificationStatus", "verify_claim"]
