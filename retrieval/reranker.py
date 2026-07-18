"""CrossEncoderReranker: reranks retrieval candidates with a cross-encoder (Phase 13).

Takes the candidate list produced by Hybrid Retrieval (Phase 11) + Graph
Expansion (Phase 12) - joined with each chunk's actual content from SQLite
into `models.schemas.RerankCandidate` by the caller - and re-scores every
``(query, raw_code)`` pair with a cross-encoder, which jointly attends
over the query and chunk together rather than comparing independently
computed vectors (see this phase's design discussion for why that ranks
better). Never re-runs retrieval or graph expansion; only reranks.
"""

from __future__ import annotations

from typing import Any, Protocol

from config import settings
from core.exceptions import RetrievalError
from core.logging import get_logger
from models.schemas import RankedChunk, RerankCandidate

logger = get_logger(__name__)


class CrossEncoderModel(Protocol):
    """The subset of `sentence_transformers.CrossEncoder` this module needs."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        """Score a batch of (query, passage) pairs."""
        ...


def load_cross_encoder(model_name: str) -> CrossEncoderModel:
    """Construct the cross-encoder model named `model_name`.

    Args:
        model_name: A `sentence_transformers.CrossEncoder`-compatible
            model identifier (e.g. `settings.RERANKER_MODEL`).

    Returns:
        A ready-to-use `CrossEncoder` exposing `.predict(...)`.

    Raises:
        RetrievalError: If the model cannot be constructed or downloaded.
            `sentence-transformers`/`transformers`/`huggingface_hub` can
            raise many different exception types here; all are wrapped
            uniformly so no third-party exception type crosses this
            module's boundary.
    """
    from sentence_transformers import CrossEncoder

    try:
        model = CrossEncoder(model_name)
    except Exception as exc:  # noqa: BLE001 - third-party ML load boundary, see docstring
        raise RetrievalError(f"Failed to load cross-encoder model {model_name!r}: {exc}") from exc

    logger.info("Model loaded: %s", model_name)
    return model


class CrossEncoderReranker:
    """Reranks candidates by cross-encoder relevance to a query.

    The model is loaded lazily (on first `rerank` call) and cached on the
    instance so repeated `rerank` calls never reload it; tests can also
    inject a fake model directly via `model`.
    """

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            model: Pre-constructed cross-encoder model. Overridable for
                testing; defaults to lazily loading `model_name` via
                `load_cross_encoder`.
            model_name: Cross-encoder model identifier to load. Defaults
                to `settings.RERANKER_MODEL`.
            batch_size: (query, chunk) pairs per `model.predict()` call.
                Defaults to `settings.RERANKER_BATCH_SIZE`.
        """
        self._model_name = model_name or settings.RERANKER_MODEL
        self._model = model
        self._batch_size = batch_size or settings.RERANKER_BATCH_SIZE

    def _ensure_model(self) -> CrossEncoderModel:
        """Lazily load the cross-encoder model on first use."""
        if self._model is None:
            self._model = load_cross_encoder(self._model_name)
        return self._model

    def _dedupe(self, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        """Keep only the first occurrence of each chunk_id.

        Args:
            candidates: Candidates to dedupe, in their original order.

        Returns:
            `candidates` with later duplicates of an already-seen
            chunk_id removed.
        """
        seen: set[str] = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            deduped.append(candidate)
        return deduped

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int | None = None,
    ) -> list[RankedChunk]:
        """Rerank `candidates` by cross-encoder relevance to `query`.

        Args:
            query: The user's query text.
            candidates: Candidates to rerank - typically Hybrid Retrieval
                + Graph Expansion's output, joined with chunk content.
            top_k: Maximum number of results to return. Defaults to
                `settings.TOP_K_FINAL`.

        Returns:
            Up to `top_k` `RankedChunk`s, sorted by descending
            `cross_encoder_score`. Empty if `candidates` is empty.

        Raises:
            RetrievalError: If `top_k` is not positive, the model cannot
                be loaded, or inference fails.
        """
        effective_top_k = top_k if top_k is not None else settings.TOP_K_FINAL
        if effective_top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {effective_top_k}")

        logger.info("Reranking started: %d candidate(s)", len(candidates))

        unique_candidates = self._dedupe(candidates)
        removed = len(candidates) - len(unique_candidates)
        if removed:
            logger.info("Duplicate candidates removed: %d", removed)

        if not unique_candidates:
            logger.info("Reranking completed: 0 candidate(s) scored (empty candidate list)")
            return []

        model = self._ensure_model()
        scores: list[float] = []

        for start in range(0, len(unique_candidates), self._batch_size):
            batch = unique_candidates[start : start + self._batch_size]
            pairs = [(query, candidate.raw_code) for candidate in batch]
            try:
                batch_scores = model.predict(pairs, batch_size=self._batch_size)
            except Exception as exc:  # noqa: BLE001 - third-party ML inference boundary
                raise RetrievalError(f"Cross-encoder inference failed: {exc}") from exc

            scores.extend(float(score) for score in batch_scores)
            logger.info(
                "Candidates scored: %d/%d", min(start + self._batch_size, len(unique_candidates)),
                len(unique_candidates),
            )

        scored = sorted(
            zip(unique_candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        top_candidates = scored[:effective_top_k]

        ranked_chunks = [
            RankedChunk(
                chunk_id=candidate.chunk_id,
                cross_encoder_score=score,
                previous_retrieval_score=candidate.previous_score,
                final_rank=rank,
                retrieval_source=candidate.retrieval_source,
            )
            for rank, (candidate, score) in enumerate(top_candidates, start=1)
        ]

        logger.info("Reranking completed: %d candidate(s) scored", len(unique_candidates))
        logger.info("Final top-k selected: %d", len(ranked_chunks))
        return ranked_chunks
