"""Tests for core.exceptions."""

from __future__ import annotations

import pytest

from core.exceptions import (
    DatabaseError,
    EmbeddingError,
    LLMGenerationError,
    ParsingError,
    RepoMindError,
    RepositoryCloneError,
    RetrievalError,
)

ALL_EXCEPTION_TYPES: tuple[type[RepoMindError], ...] = (
    RepositoryCloneError,
    ParsingError,
    EmbeddingError,
    RetrievalError,
    LLMGenerationError,
    DatabaseError,
)


@pytest.mark.parametrize("exc_type", ALL_EXCEPTION_TYPES)
def test_exception_is_subclass_of_repomind_error(exc_type: type[RepoMindError]) -> None:
    """Every custom exception must be catchable via the shared base class."""
    assert issubclass(exc_type, RepoMindError)


@pytest.mark.parametrize("exc_type", ALL_EXCEPTION_TYPES)
def test_exception_can_be_raised_and_carries_message(exc_type: type[RepoMindError]) -> None:
    """Each exception should behave like a normal Exception, preserving its message."""
    with pytest.raises(exc_type, match="boom"):
        raise exc_type("boom")


def test_base_error_is_a_python_exception() -> None:
    assert issubclass(RepoMindError, Exception)
