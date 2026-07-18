"""Tests for core.constants."""

from __future__ import annotations

from core.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RERANKER_MODEL,
    SUPPORTED_FILE_EXTENSIONS,
    SUPPORTED_LANGUAGES,
)


def test_every_extension_maps_to_a_supported_language() -> None:
    """Every file extension must resolve to a language declared as supported."""
    assert set(SUPPORTED_FILE_EXTENSIONS.values()) <= set(SUPPORTED_LANGUAGES)


def test_python_extension_is_supported() -> None:
    assert SUPPORTED_FILE_EXTENSIONS[".py"] == "python"


def test_default_model_names_are_non_empty_strings() -> None:
    for model_name in (DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, DEFAULT_OLLAMA_MODEL):
        assert isinstance(model_name, str)
        assert model_name.strip() != ""
