"""Base agent interface every Adjudicate agent implements against (Phase 23 scaffold).

Deliberately minimal: one abstract method, no prompt template, no claim
schema, no LLM call. Every later phase's agent (Defender, Prosecutor,
Judge, Documentation) subclasses `BaseAgent` and implements `review`;
this module exists only so those phases have a shared contract to
implement against from their first commit, instead of each inventing
its own call signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Common interface for every Adjudicate agent.

    Attributes:
        role: The agent's role name (e.g. ``"defender"``), used for
            logging and for `adjudicate.config.AdjudicateSettings`'s
            per-role model resolution.
    """

    role: str

    @abstractmethod
    def review(self, context: Any) -> Any:
        """Produce this agent's contribution to a review round.

        Args:
            context: Whatever this agent needs to do its job — for the
                Defender/Prosecutor, a `ContextBundle` (Phase 24); for
                the Judge, the accumulated claims/rebuttals of a round.
                Left as `Any` here since no phase has defined its real
                input/output types yet.

        Returns:
            This agent's output for the round. Left as `Any` for the
            same reason as `context`.
        """
        raise NotImplementedError
