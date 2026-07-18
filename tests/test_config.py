"""Tests for config.Settings."""

from __future__ import annotations

import pytest

from config import Settings


def test_defaults_are_valid() -> None:
    """Default settings should satisfy their own field constraints."""
    settings = Settings(_env_file=None)
    assert settings.TOP_K_DENSE > 0
    assert settings.TOP_K_BM25 > 0
    assert settings.TOP_K_FINAL > 0
    assert settings.MAX_CONTEXT_TOKENS > 0
    assert settings.RRF_K > 0
    assert 0.0 <= settings.CACHE_SIMILARITY_THRESHOLD <= 1.0


def test_llm_defaults_are_valid() -> None:
    """LLM settings should default to Gemini/gemini-2.5-flash, with Ollama as the configured fallback."""
    settings = Settings(_env_file=None)
    assert settings.USE_GEMINI is True
    assert settings.GEMINI_MODEL == "gemini-2.5-flash"
    assert settings.GEMINI_API_KEY is None
    assert settings.OLLAMA_MODEL == "mistral"
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert 0.0 <= settings.LLM_TEMPERATURE <= 2.0
    assert settings.LLM_MAX_TOKENS > 0
    assert settings.LLM_TIMEOUT_SECONDS > 0


def test_rejects_out_of_range_llm_temperature() -> None:
    """LLM_TEMPERATURE must stay within [0, 2]."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, LLM_TEMPERATURE=3.0)


def test_feature_flags_default_true_except_ollama() -> None:
    """Every retrieval feature and Gemini default on; Ollama defaults off (opt-in local LLM)."""
    settings = Settings(_env_file=None)
    assert settings.USE_CODEBERT is True
    assert settings.USE_BM25 is True
    assert settings.USE_RERANKER is True
    assert settings.USE_GRAPH_EXPANSION is True
    assert settings.USE_SMALL_TO_BIG is True
    assert settings.USE_SEMANTIC_CACHE is True
    assert settings.USE_GEMINI is True
    assert settings.USE_OLLAMA is False


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables must take priority over field defaults."""
    monkeypatch.setenv("TOP_K_DENSE", "5")
    monkeypatch.setenv("USE_OLLAMA", "true")
    settings = Settings(_env_file=None)
    assert settings.TOP_K_DENSE == 5
    assert settings.USE_OLLAMA is True


def test_invalid_log_level_raises() -> None:
    """An unrecognized LOG_LEVEL must fail validation, not silently default."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, LOG_LEVEL="NOT_A_LEVEL")


def test_log_level_is_case_insensitive() -> None:
    """Lower-case log levels should be normalized to upper-case."""
    settings = Settings(_env_file=None, LOG_LEVEL="debug")
    assert settings.LOG_LEVEL == "DEBUG"


def test_rejects_non_positive_top_k() -> None:
    """TOP_K_DENSE must be strictly positive."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, TOP_K_DENSE=0)


def test_rejects_out_of_range_cache_threshold() -> None:
    """CACHE_SIMILARITY_THRESHOLD must stay within [0, 1]."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, CACHE_SIMILARITY_THRESHOLD=1.5)


def test_data_subdirectories_nest_under_data_dir() -> None:
    """Derived data paths should live under DATA_DIR by default."""
    settings = Settings(_env_file=None)
    assert settings.REPOSITORIES_DIR.parent == settings.DATA_DIR
    assert settings.INDEXES_DIR.parent == settings.DATA_DIR
    assert settings.SQLITE_DB_PATH.parent == settings.SQLITE_DIR
