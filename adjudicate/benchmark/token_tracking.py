"""A thin LLMClient wrapper that tallies real token usage (Phase 31).

`DefenderAgent`/`ProsecutorAgent`/`JudgeAgent`/`BaselineReviewer` return
only their parsed result, not the underlying `LLMCompletion`'s token
counts - reasonable for their own normal use, but the benchmark harness
needs real per-case cost numbers. Rather than change any agent's public
return type for this one benchmark-only need, this wraps the shared
`LLMClient` each agent is constructed with and records every completion
as it happens.
"""

from __future__ import annotations

from typing import Any

from generation.llm_client import LLMClient, LLMCompletion


class TokenTrackingLLMClient:
    """Wraps a real `LLMClient`, recording every call's token usage and count.

    Passed as the `llm_client` argument to every agent in a benchmark run
    so all of them share one real client (one provider, one set of
    credentials) while the harness can read off exactly how many calls/
    tokens a given stretch of agent calls actually used.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.call_count = 0
        self.total_tokens = 0

    def complete(
        self, system_prompt: str, user_prompt: str, response_schema: Any | None = None
    ) -> LLMCompletion:
        completion = self._inner.complete(system_prompt, user_prompt, response_schema)
        self.call_count += 1
        self.total_tokens += completion.total_tokens
        return completion

    def reset(self) -> None:
        """Zero the running counters - call before each case/condition group to get a per-case delta."""
        self.call_count = 0
        self.total_tokens = 0
