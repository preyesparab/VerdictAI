"""Static, environment-independent constants for RepoMind.

Values here never change between environments (local, CI, Docker,
production) — they are facts about the system, not configuration. Anything
that *should* vary by environment (paths, thresholds, feature flags)
belongs in `config.Settings` instead, which may use these constants as
defaults.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------
# Supported languages and file extensions
#
# Drives which files `ingestion.repo_loader` collects and which
# Tree-sitter grammar `ingestion.ast_parser` selects, in later phases.
# ---------------------------------------------------------------------
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = (
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "cpp",
    "c_sharp",
    "ruby",
    "php",
)

SUPPORTED_FILE_EXTENSIONS: Final[dict[str, str]] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
}

# ---------------------------------------------------------------------
# Default directory names (relative to project root)
#
# `config.Settings` composes these into absolute, overridable paths.
# ---------------------------------------------------------------------
DEFAULT_DATA_DIR_NAME: Final[str] = "data"
DEFAULT_REPOSITORIES_DIR_NAME: Final[str] = "repositories"
DEFAULT_INDEXES_DIR_NAME: Final[str] = "indexes"
DEFAULT_CACHE_DIR_NAME: Final[str] = "cache"
DEFAULT_EVALUATION_DIR_NAME: Final[str] = "evaluation"
DEFAULT_LOGS_DIR_NAME: Final[str] = "logs"

# ---------------------------------------------------------------------
# Default model names
#
# Fixed fallback identifiers for models used in later phases. These are
# intentionally not environment-configurable in this phase — only the
# feature flags that enable/disable the components using them are.
# ---------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL: Final[str] = "microsoft/codebert-base"
DEFAULT_MINILM_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL: Final[str] = "mistral"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL: Final[str] = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------
# Embedding dimensions
#
# Fixed output dimensionality of each supported embedding model, used by
# `embedding.embedding_manager.EmbeddingManager` to validate that a
# model's output matches what is expected before persisting it.
# ---------------------------------------------------------------------
EMBEDDING_DIMENSIONS: Final[dict[str, int]] = {
    DEFAULT_EMBEDDING_MODEL: 768,
    DEFAULT_MINILM_MODEL: 384,
}
