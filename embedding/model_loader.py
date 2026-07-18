"""Selects and loads the active chunk-embedding model.

`settings.USE_CODEBERT` is the single switch between the two supported
models: CodeBERT (`microsoft/codebert-base`, code-pretrained, 768-dim) when
True, MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, general-purpose,
384-dim, much faster) when False. Both are exposed through the same
`sentence_transformers.SentenceTransformer` interface so callers (currently
`embedding.embedding_manager.EmbeddingManager`; later, query-time embedding
in the dense retriever) never need to branch on which model is active.

CodeBERT is not published as a ready-made `sentence-transformers` model, so
it is assembled from a plain `Transformer` + mean-`Pooling` module pair;
MiniLM already ships as a complete `sentence-transformers` model and is
loaded directly.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, models

from config import settings
from core.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MINILM_MODEL
from core.exceptions import EmbeddingError
from core.logging import get_logger

logger = get_logger(__name__)


def active_model_name() -> str:
    """Return the embedding model name selected by `settings.USE_CODEBERT`.

    Returns:
        `DEFAULT_EMBEDDING_MODEL` (CodeBERT) if `settings.USE_CODEBERT` is
        True, otherwise `DEFAULT_MINILM_MODEL`.
    """
    return DEFAULT_EMBEDDING_MODEL if settings.USE_CODEBERT else DEFAULT_MINILM_MODEL


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Construct the embedding model named `model_name`.

    Args:
        model_name: Model to load. Defaults to `active_model_name()`.

    Returns:
        A ready-to-use `SentenceTransformer` exposing `.encode(...)`.

    Raises:
        EmbeddingError: If the model cannot be constructed or downloaded.
            `sentence-transformers`/`transformers`/`huggingface_hub` can
            raise many different exception types here (network errors,
            missing cache, corrupt weights); all are wrapped uniformly so
            no third-party exception type crosses this module's boundary.
    """
    name = model_name or active_model_name()

    try:
        if name == DEFAULT_EMBEDDING_MODEL:
            word_embedding_model = models.Transformer(name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            model = SentenceTransformer(name)
    except Exception as exc:  # noqa: BLE001 - third-party ML load boundary, see docstring
        raise EmbeddingError(f"Failed to load embedding model {name!r}: {exc}") from exc

    logger.info("Model loaded: %s", name)
    return model
