"""Adjudicate: adversarial multi-agent code review, built on top of RepoMind (Part B of the roadmap).

Where `pipeline.Pipeline` (Part A) answers "what does this code do?",
Adjudicate answers "should this change be merged?" — a Defender and
Prosecutor agent argue over a diff using RepoMind's own graph/retrieval
context (via `adjudicate.repomind_client.RepoMindClient`, never by
reimplementing retrieval), a Verifier checks falsifiable claims against
the real repository instead of trusting either agent's prose, and a
Judge renders a final verdict.

This package depends on RepoMind (imports `config`, `core.exceptions`,
and calls the FastAPI backend in `api.main`) but RepoMind must never
depend back on `adjudicate` — the review layer is strictly additive.
"""

from __future__ import annotations
