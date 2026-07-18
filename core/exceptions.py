"""Custom exception hierarchy for RepoMind.

Every layer raises one of these instead of a bare `Exception` or a
third-party library's exception type, so callers can catch failures by
pipeline stage (e.g., `except RetrievalError`) rather than by
implementation detail, and so a broad `except RepoMindError` at the
`app.py`/`pipeline.py` boundary catches every failure this system itself
can produce.
"""

from __future__ import annotations


class RepoMindError(Exception):
    """Base class for all RepoMind-specific exceptions."""


class RepositoryCloneError(RepoMindError):
    """Raised when cloning or loading a target repository fails."""


class ParsingError(RepoMindError):
    """Raised when Tree-sitter parsing of a source file fails."""


class EmbeddingError(RepoMindError):
    """Raised when generating an embedding for a chunk or query fails."""


class RetrievalError(RepoMindError):
    """Raised when a retrieval component (dense, sparse, graph, or reranker) fails."""


class LLMGenerationError(RepoMindError):
    """Raised when the LLM fails to generate a response."""


class DatabaseError(RepoMindError):
    """Raised when a database or persistence operation fails."""


class EvaluationError(RepoMindError):
    """Raised when a retrieval/generation evaluation or ablation run fails."""
