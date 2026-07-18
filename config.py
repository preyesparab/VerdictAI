"""Centralized runtime configuration for RepoMind.

Every configurable value in the system — feature flags, retrieval
parameters, filesystem locations, and logging behavior — is defined on the
`Settings` model in this module and exposed as the single `settings`
singleton. No other module should read `os.environ` directly; it should
import `settings` from here instead. This keeps configuration:

- **Centralized**: one place to see every knob the system exposes.
- **Validated**: pydantic coerces types and validates ranges at startup
  (e.g., a similarity threshold outside ``[0, 1]``), failing fast on a bad
  ``.env`` value instead of surfacing a cryptic error deep inside a
  retrieval call.
- **Overridable**: every field can be set via an environment variable or a
  ``.env`` file without touching code, which later phases need for
  Docker/CI environments.

Static values that never change between environments (supported
languages, default model identifiers, directory *names*) live in
`core.constants` and are imported here only where a `Settings` field needs
a default.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    DEFAULT_CACHE_DIR_NAME,
    DEFAULT_DATA_DIR_NAME,
    DEFAULT_EVALUATION_DIR_NAME,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GRAPH_DIR_NAME,
    DEFAULT_GRAPH_FILE_NAME,
    DEFAULT_GROQ_MODEL,
    DEFAULT_INDEXES_DIR_NAME,
    DEFAULT_LOGS_DIR_NAME,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_REPOSITORIES_DIR_NAME,
    DEFAULT_SQLITE_DB_NAME,
    DEFAULT_SQLITE_DIR_NAME,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent

_VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class Settings(BaseSettings):
    """Typed, validated configuration for the RepoMind pipeline.

    Populated from, in increasing priority: field defaults, a ``.env``
    file at the project root, and process environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Feature flags
    #
    # Each flag independently toggles one pipeline component so that
    # retrieval strategies can be enabled, disabled, and benchmarked in
    # isolation once they are implemented in later phases.
    # ------------------------------------------------------------------
    USE_CODEBERT: bool = Field(
        default=True,
        description="Enable CodeBERT-based dense embeddings for retrieval.",
    )
    USE_BM25: bool = Field(
        default=True,
        description="Enable BM25 sparse retrieval.",
    )
    USE_RERANKER: bool = Field(
        default=True,
        description="Enable cross-encoder reranking of candidate results.",
    )
    USE_GRAPH_EXPANSION: bool = Field(
        default=True,
        description="Enable graph-based neighbor expansion of retrieved nodes.",
    )
    USE_SMALL_TO_BIG: bool = Field(
        default=True,
        description="Enable small-to-big retrieval (expand matched chunks to parent context).",
    )
    USE_SEMANTIC_CACHE: bool = Field(
        default=True,
        description="Enable semantic caching of retrieval/generation results.",
    )
    USE_GEMINI: bool = Field(
        default=True,
        description="Use Google Gemini as the hosted LLM provider. Checked before USE_GROQ and USE_OLLAMA.",
    )
    USE_GROQ: bool = Field(
        default=False,
        description="Use Groq as a hosted LLM provider. Only consulted if USE_GEMINI is False; checked before USE_OLLAMA.",
    )
    USE_OLLAMA: bool = Field(
        default=False,
        description="Use a local Ollama model instead of a hosted LLM provider. Only consulted if USE_GEMINI and USE_GROQ are both False.",
    )

    # ------------------------------------------------------------------
    # Retrieval parameters
    # ------------------------------------------------------------------
    TOP_K_DENSE: int = Field(
        default=20, gt=0, description="Candidates to retrieve from the dense index."
    )
    TOP_K_BM25: int = Field(
        default=20, gt=0, description="Candidates to retrieve from the sparse (BM25) index."
    )
    TOP_K_FINAL: int = Field(
        default=8, gt=0, description="Results kept after fusion/reranking, passed to generation."
    )
    MAX_CONTEXT_TOKENS: int = Field(
        default=4000, gt=0, description="Token budget for the assembled generation context."
    )
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a semantic cache hit.",
    )
    RRF_K: int = Field(
        default=60,
        gt=0,
        description="Constant k in reciprocal rank fusion: score = 1 / (k + rank).",
    )

    # ------------------------------------------------------------------
    # Embedding generation parameters
    # ------------------------------------------------------------------
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        gt=0,
        description="Number of chunks embedded per model.encode() call during embedding generation.",
    )

    # ------------------------------------------------------------------
    # Reranking parameters
    # ------------------------------------------------------------------
    RERANKER_MODEL: str = Field(
        default=DEFAULT_RERANKER_MODEL,
        description="Cross-encoder model used to rerank retrieved candidates.",
    )
    RERANKER_BATCH_SIZE: int = Field(
        default=32,
        gt=0,
        description="Number of (query, chunk) pairs scored per CrossEncoder.predict() call.",
    )

    # ------------------------------------------------------------------
    # LLM generation parameters (Phase 16; Gemini replacing OpenAI as of
    # the provider-layer update; Groq added Phase 27 follow-up as a
    # hosted fallback for when Gemini's free-tier quota is exhausted)
    #
    # Provider selection is a static, config-driven priority, not a
    # runtime fallback-on-failure: USE_GEMINI is checked first (Gemini is
    # the default hosted provider); if it is False, USE_GROQ is checked
    # next; if that is also False, USE_OLLAMA is checked; if none of the
    # three is True, LLMClient.complete raises a clear configuration
    # error rather than silently picking a provider.
    # ------------------------------------------------------------------
    OLLAMA_MODEL: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        description="Ollama model used for generation when USE_GEMINI and USE_GROQ are both False and USE_OLLAMA is True.",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    GEMINI_MODEL: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        description="Gemini model used for generation when USE_GEMINI is True.",
    )
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="Google Gemini API key. Required only when USE_GEMINI is True.",
    )
    GROQ_MODEL: str = Field(
        default=DEFAULT_GROQ_MODEL,
        description="Groq model used for generation when USE_GEMINI is False and USE_GROQ is True.",
    )
    GROQ_API_KEY: str | None = Field(
        default=None,
        description="Groq API key. Required only when USE_GROQ is True. Get one at https://console.groq.com/keys",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation. Kept low by default to favor grounded, factual answers over creative ones.",
    )
    LLM_MAX_TOKENS: int = Field(
        default=4096,
        gt=0,
        description="Maximum completion tokens requested per generation call. Raised from an original 1024 "
        "(Phase 27 follow-up, real reproduction against mkocabas/VIBE): a schema-constrained Gemini 2.5 Flash "
        "call's `max_output_tokens` budget is shared with the model's own internal 'thinking' tokens, which "
        "consumed 979 of a 1024-token budget on one real diff, truncating the visible JSON mid-string after "
        "only 28 output tokens. Disabling thinking for schema-constrained calls (see "
        "generation/llm_client.py::_complete_with_gemini) addresses the root cause; this increase is additional "
        "headroom for a full multi-claim JSON array (each with a real proposed_test code snippet), which can "
        "legitimately need more than 1024 tokens even with thinking removed from the budget entirely.",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for a single LLM provider request.",
    )

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    CORS_ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Origins the FastAPI backend (api/main.py) accepts browser requests from - "
        "Vite's default dev server ports by default; add the deployed frontend's origin (Phase 33) here.",
    )

    # ------------------------------------------------------------------
    # Filesystem locations
    #
    # Each defaults relative to the project root but is independently
    # overridable via its own environment variable.
    # ------------------------------------------------------------------
    BASE_DIR: Path = Field(default=PROJECT_ROOT, description="Project root directory.")
    DATA_DIR: Path = Field(default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME)
    REPOSITORIES_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_REPOSITORIES_DIR_NAME
    )
    INDEXES_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_INDEXES_DIR_NAME
    )
    CACHE_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_CACHE_DIR_NAME
    )
    SQLITE_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_SQLITE_DIR_NAME
    )
    SQLITE_DB_PATH: Path = Field(
        default=PROJECT_ROOT
        / DEFAULT_DATA_DIR_NAME
        / DEFAULT_SQLITE_DIR_NAME
        / DEFAULT_SQLITE_DB_NAME
    )
    EVALUATION_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_EVALUATION_DIR_NAME
    )
    LOG_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_LOGS_DIR_NAME
    )
    GRAPH_DIR: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_GRAPH_DIR_NAME
    )
    GRAPH_FILE_PATH: Path = Field(
        default=PROJECT_ROOT / DEFAULT_DATA_DIR_NAME / DEFAULT_GRAPH_DIR_NAME / DEFAULT_GRAPH_FILE_NAME
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and validate the configured log level.

        Args:
            value: Raw log level string from the environment or default.

        Returns:
            The upper-cased log level.

        Raises:
            ValueError: If `value` is not a recognized logging level.
        """
        upper_value = value.upper()
        if upper_value not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return upper_value

    def ensure_directories(self) -> None:
        """Create every configured filesystem directory if it does not exist.

        Not called automatically on import — importing `config` must not
        have filesystem side effects. Call this explicitly once, at
        process startup (e.g., from `app.py` or `pipeline.py`).
        """
        for directory in (
            self.DATA_DIR,
            self.REPOSITORIES_DIR,
            self.INDEXES_DIR,
            self.CACHE_DIR,
            self.SQLITE_DIR,
            self.EVALUATION_DIR,
            self.LOG_DIR,
            self.GRAPH_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
