"""Selects and loads the active chunk-embedding model.

Embeddings are generated through Google's hosted Gemini embedding API
(`core.constants.DEFAULT_EMBEDDING_MODEL`, currently ``gemini-embedding-001``
at `core.constants.EMBEDDING_OUTPUT_DIMENSIONALITY` dimensions) rather than a
locally-loaded transformer. `GeminiEmbeddingModel` wraps
`google.genai.Client.models.embed_content` behind the same
``.encode(sentences, **kwargs) -> np.ndarray`` interface a
`sentence_transformers.SentenceTransformer` used to expose, so callers
(`embedding.embedding_manager.EmbeddingManager`,
`retrieval.semantic_cache.SemanticCacheManager`, and `pipeline.py`'s
query-time embedding) needed no interface changes - only the new output
dimension, already reflected in `core.constants.EMBEDDING_DIMENSIONS`.

This replaced a locally-loaded CodeBERT/MiniLM model precisely because
loading a transformer in-process cost 500MB+ of resident memory, which was
causing the Render deployment to run out of memory - moving inference to a
hosted API removes that cost from this process's memory footprint entirely.
It is a breaking change to any previously-stored embedding (different
dimension, different vector space); see `core.constants`'s module comment
for why no migration path is provided.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from google import genai
from google.genai import types

from config import settings
from core.constants import DEFAULT_EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIMENSIONALITY
from core.exceptions import EmbeddingError
from core.logging import get_logger

logger = get_logger(__name__)


def active_model_name() -> str:
    """Return the active embedding model's identifier.

    Returns:
        `DEFAULT_EMBEDDING_MODEL` - the only embedding model this system
        currently supports.
    """
    return DEFAULT_EMBEDDING_MODEL


class GeminiEmbeddingModel:
    """Adapts `google.genai.Client.models.embed_content` to the `.encode(...)` interface callers already use.

    ``**kwargs`` (e.g. ``convert_to_numpy``, ``show_progress_bar`` - both
    `sentence_transformers`-specific) are accepted and silently ignored,
    since every existing caller still passes them unchanged.
    """

    def __init__(self, client: genai.Client, model_name: str, output_dimensionality: int) -> None:
        """Initialize the adapter.

        Args:
            client: An authenticated `google.genai.Client`.
            model_name: The Gemini embedding model to call.
            output_dimensionality: Requested output vector length, passed
                as `google.genai.types.EmbedContentConfig.output_dimensionality`
                - the model truncates server-side using its own
                Matryoshka (MRL) training, not a naive post-hoc slice.
        """
        self._client = client
        self._model_name = model_name
        self._output_dimensionality = output_dimensionality

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Embed `sentences` with one Gemini `embed_content` call.

        Args:
            sentences: Texts to embed.
            **kwargs: Ignored - see class docstring.

        Returns:
            A ``(len(sentences), output_dimensionality)`` float32 array,
            one row per input in `sentences`'s order.

        Raises:
            Exception: Whatever `google.genai` raises on a failed request -
                intentionally not wrapped here; every existing caller
                (`EmbeddingManager._encode_batch`,
                `SemanticCacheManager._embed_query`) already wraps
                `model.encode(...)` failures into its own module's
                exception type at its own error boundary.
        """
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=sentences,
            config=types.EmbedContentConfig(output_dimensionality=self._output_dimensionality),
        )
        return np.array([embedding.values for embedding in response.embeddings], dtype=np.float32)


def load_embedding_model(model_name: str | None = None) -> GeminiEmbeddingModel:
    """Construct a `GeminiEmbeddingModel` for `model_name`.

    Args:
        model_name: Embedding model to call. Defaults to `active_model_name()`.

    Returns:
        A ready-to-use `GeminiEmbeddingModel` exposing `.encode(...)`.

    Raises:
        EmbeddingError: If `settings.GEMINI_API_KEY` is not set, or the
            `google.genai.Client` cannot be constructed.
    """
    name = model_name or active_model_name()

    if not settings.GEMINI_API_KEY:
        raise EmbeddingError(
            "GEMINI_API_KEY is not set. Add it to your .env file - embeddings are "
            "generated through the Gemini API regardless of which provider "
            "USE_GEMINI/USE_GROQ/USE_OLLAMA selects for generation."
        )

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as exc:  # noqa: BLE001 - third-party client construction boundary
        raise EmbeddingError(f"Failed to construct Gemini client for embedding model {name!r}: {exc}") from exc

    logger.info("Model loaded: %s", name)
    return GeminiEmbeddingModel(client, name, EMBEDDING_OUTPUT_DIMENSIONALITY)
