"""Tests for adjudicate.config.AdjudicateSettings."""

from __future__ import annotations

import pytest

from adjudicate.config import AdjudicateSettings, AgentRole
from config import settings as core_settings


def test_defaults_are_valid() -> None:
    """Default sandbox limits should satisfy their own field constraints."""
    settings = AdjudicateSettings(_env_file=None)
    assert settings.VERIFIER_SANDBOX_TIMEOUT_SECONDS > 0
    assert settings.VERIFIER_SANDBOX_MEMORY_LIMIT_MB > 0


def test_role_model_defaults_to_core_gemini_model() -> None:
    """With no per-role override configured, every role falls back to config.settings.GEMINI_MODEL."""
    settings = AdjudicateSettings(_env_file=None)
    for role in AgentRole:
        assert settings.gemini_model_for(role) == core_settings.GEMINI_MODEL


def test_role_override_is_independent_per_role() -> None:
    """Overriding one role's model must not affect the others' fallback."""
    settings = AdjudicateSettings(_env_file=None, JUDGE_GEMINI_MODEL="gemini-2.5-pro")
    assert settings.gemini_model_for(AgentRole.JUDGE) == "gemini-2.5-pro"
    assert settings.gemini_model_for(AgentRole.DEFENDER) == core_settings.GEMINI_MODEL
    assert settings.gemini_model_for(AgentRole.PROSECUTOR) == core_settings.GEMINI_MODEL
    assert settings.gemini_model_for(AgentRole.DOCUMENTATION) == core_settings.GEMINI_MODEL


def test_rejects_non_positive_sandbox_timeout() -> None:
    """VERIFIER_SANDBOX_TIMEOUT_SECONDS must be strictly positive."""
    with pytest.raises(ValueError):
        AdjudicateSettings(_env_file=None, VERIFIER_SANDBOX_TIMEOUT_SECONDS=0)


def test_rejects_non_positive_sandbox_memory_limit() -> None:
    """VERIFIER_SANDBOX_MEMORY_LIMIT_MB must be strictly positive."""
    with pytest.raises(ValueError):
        AdjudicateSettings(_env_file=None, VERIFIER_SANDBOX_MEMORY_LIMIT_MB=0)
