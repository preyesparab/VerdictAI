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
#
# DEFAULT_EMBEDDING_MODEL was originally a locally-loaded CodeBERT model
# (`microsoft/codebert-base`, via `sentence-transformers`) - replaced with
# Google's hosted Gemini embedding API to remove the 500MB+ resident-memory
# cost of loading a transformer in-process, which was causing the Render
# deployment to run out of memory (see docs/state/PROGRESS.md). This is a
# breaking change to any previously-stored embedding: the vector space and
# dimension are both different, so every previously-indexed repository
# must be re-indexed (`EmbeddingManager.generate_embeddings(..., force=True)`
# followed by a fresh `FaissIndexManager.build_index`) - there is no
# migration path between the two, by design (pre-launch, nothing indexed
# needs to survive).
# ---------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL: Final[str] = "gemini-embedding-001"
DEFAULT_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL: Final[str] = "mistral"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL: Final[str] = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------
# Embedding dimensions
#
# EMBEDDING_OUTPUT_DIMENSIONALITY is requested from the Gemini embedding
# API via `EmbedContentConfig.output_dimensionality` - the API truncates
# server-side using the model's own Matryoshka (MRL) training, not a naive
# post-hoc slice, so a truncated vector stays meaningful. 768 was chosen
# over the model's native 3072 to keep FAISS index size/search cost down;
# 1536/3072 remain available by raising this constant alone.
#
# EMBEDDING_DIMENSIONS is the fixed output dimensionality of each
# supported embedding model, used by
# `embedding.embedding_manager.EmbeddingManager` to validate that a
# model's output matches what is expected before persisting it.
# ---------------------------------------------------------------------
EMBEDDING_OUTPUT_DIMENSIONALITY: Final[int] = 768

EMBEDDING_DIMENSIONS: Final[dict[str, int]] = {
    DEFAULT_EMBEDDING_MODEL: EMBEDDING_OUTPUT_DIMENSIONALITY,
}
