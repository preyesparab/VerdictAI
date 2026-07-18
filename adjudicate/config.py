"""Adjudicate-specific runtime configuration (Phase 23).

Mirrors `config.Settings`'s pattern — a validated, `.env`-overridable
pydantic `BaseSettings` singleton — but scoped to the adversarial review
system instead of the chat/retrieval pipeline. Two concerns live here,
per this phase's scaffold-only scope (`docs/project _description.md`
Phase 23):

1. Per-agent-role LLM model overrides. A review's Judge may warrant a
   stronger (and separately billed) model than its Defender, so each
   role can independently override the model name. Provider selection
   itself (`USE_GEMINI`/`USE_OLLAMA`) is *not* duplicated per role here —
   Adjudicate reuses the core system's already-configured provider
   (`config.settings`) so a fresh deployment works without a second set
   of credentials; only the model name is overridable per role.
2. Sandbox limits for the Verifier (Phase 28). Declared now so the
   config *surface* exists before Phase 28 needs it — every field below
   is inert until that phase's sandbox execution reads it. No sandboxing
   is implemented in this module.

No LLM call is made from this module, and no agent prompt/schema logic
lives here — only configuration resolution. Building the actual
per-role LLM client (mirroring `generation.llm_client.LLMClient`) is
deferred to whichever phase first needs to make a call (Phase 25's
Defender).
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import settings as core_settings


class AgentRole(str, Enum):
    """The four agent roles Adjudicate's config can independently tune.

    Matches `docs/roadmap.md`'s Part B phases: Defender (25), Prosecutor
    (26), Judge (30), Documentation (35, stretch).
    """

    DEFENDER = "defender"
    PROSECUTOR = "prosecutor"
    JUDGE = "judge"
    DOCUMENTATION = "documentation"


class AdjudicateSettings(BaseSettings):
    """Typed, validated configuration for the Adjudicate review system.

    Populated the same way `config.Settings` is: field defaults, then a
    ``.env`` file at the project root, then process environment
    variables — sharing the one ``.env`` file rather than inventing a
    second config file format.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Per-role model overrides (Gemini only, for now — matches
    # `config.Settings.GEMINI_MODEL` being the default hosted provider).
    # `None` means "use the core system's `GEMINI_MODEL`".
    # ------------------------------------------------------------------
    DEFENDER_GEMINI_MODEL: str | None = Field(
        default=None, description="Gemini model for the Defender agent. Defaults to config.settings.GEMINI_MODEL."
    )
    PROSECUTOR_GEMINI_MODEL: str | None = Field(
        default=None, description="Gemini model for the Prosecutor agent. Defaults to config.settings.GEMINI_MODEL."
    )
    JUDGE_GEMINI_MODEL: str | None = Field(
        default=None, description="Gemini model for the Judge agent. Defaults to config.settings.GEMINI_MODEL."
    )
    DOCUMENTATION_GEMINI_MODEL: str | None = Field(
        default=None,
        description="Gemini model for the Documentation agent. Defaults to config.settings.GEMINI_MODEL.",
    )

    # ------------------------------------------------------------------
    # Verifier sandbox limits (Phase 28) — declared, not yet enforced.
    # ------------------------------------------------------------------
    VERIFIER_SANDBOX_TIMEOUT_SECONDS: float = Field(
        default=30.0, gt=0, description="Wall-clock limit for one Verifier sandbox execution."
    )
    VERIFIER_SANDBOX_MEMORY_LIMIT_MB: int = Field(
        default=512, gt=0, description="Memory limit for one Verifier sandbox execution."
    )

    def gemini_model_for(self, role: AgentRole) -> str:
        """Resolve the Gemini model to use for `role`.

        Args:
            role: The agent role requesting a model name.

        Returns:
            The role's override if one is configured, else
            `config.settings.GEMINI_MODEL`.
        """
        overrides: dict[AgentRole, str | None] = {
            AgentRole.DEFENDER: self.DEFENDER_GEMINI_MODEL,
            AgentRole.PROSECUTOR: self.PROSECUTOR_GEMINI_MODEL,
            AgentRole.JUDGE: self.JUDGE_GEMINI_MODEL,
            AgentRole.DOCUMENTATION: self.DOCUMENTATION_GEMINI_MODEL,
        }
        return overrides[role] or core_settings.GEMINI_MODEL


adjudicate_settings = AdjudicateSettings()
