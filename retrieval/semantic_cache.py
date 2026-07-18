"""SemanticCacheManager: caches (query, response) pairs by embedding similarity (Phase 14).

Before Hybrid Retrieval (Phase 11) runs at all, `lookup` checks whether a
semantically similar query has already been answered for this repository
- if the closest cached query's cosine similarity meets
`settings.CACHE_SIMILARITY_THRESHOLD` **and** passes the lexical
compatibility check below, its cached response and chunk list are
returned directly, skipping retrieval, graph expansion, and reranking
entirely. `store` persists a new (query, response) pair via
`database.sqlite_client.DatabaseManager` after that pipeline runs.

Queries are embedded with the same model Phase 8/9 use
(`embedding.model_loader`), so cache similarity is measured the same way
retrieval relevance is. Comparisons are always scoped to one repository -
`DatabaseManager.load_cache_entries(repository_id)` never returns another
repository's entries, so a cache hit can never leak across repositories.

**Real over-match bug, closed out here (previously flagged in Phase 19,
never fixed)**: two clearly different questions ("What is the backend
tech stack?" vs "What is the frontend tech stack?") returned the
identical cached answer. Measured directly (not assumed): cosine
similarity between that exact pair is 0.9974 - CodeBERT (a *code*
embedding model, not tuned for natural-language sentence similarity)
collapses short, similarly-structured NL questions into a nearly
degenerate similarity range regardless of actual meaning. Measured
across 7 pairs spanning genuine near-duplicates and genuinely distinct
questions: the should-hit range (0.9861-0.9925) sits entirely *inside*
the should-miss range (0.9807-0.9987) - "register endpoint" vs "login
endpoint" (should miss) scored *higher* (0.9987) than a genuine
paraphrase (should hit, 0.9861). No single cosine threshold, at any
value, can separate these - raising `CACHE_SIMILARITY_THRESHOLD` cannot
fix this bug, full stop. Fixed instead with a second, independent gate -
`_is_lexically_compatible` - applied on top of the existing cosine
check, not in place of it.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

import numpy as np

from config import settings
from core.exceptions import DatabaseError, EmbeddingError, RetrievalError
from core.logging import get_logger
from database.sqlite_client import DatabaseManager
from embedding.model_loader import active_model_name, load_embedding_model
from models.schemas import CacheEntry, SemanticCacheHit

logger = get_logger(__name__)

# Stripped before computing lexical (Jaccard) overlap, so two questions
# sharing only boilerplate ("what", "is", "the", ...) don't look
# similar just because most NL questions share these words.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "what", "which", "who", "whom", "how", "where", "when", "why",
    "used", "use", "for", "to", "of", "in", "on", "at", "and", "or", "with", "about",
    "this", "that", "these", "those", "it", "its", "i", "you", "we", "they",
})

# Below this, two queries share too few real content words to be the
# same question, regardless of how high their embedding similarity is -
# catches the "totally unrelated topic" failure mode (measured at 0.0
# overlap for genuinely unrelated pairs vs 0.5+ for every genuine
# near-duplicate pair measured during this fix).
_LEXICAL_OVERLAP_THRESHOLD = 0.5

# Explicit, deliberately narrow list of confusable term categories for
# the specific failure mode this bug report is about: two queries that
# are lexically *almost identical* (so `_LEXICAL_OVERLAP_THRESHOLD`
# alone won't reject them - "backend"/"frontend" measured at 0.5
# lexical overlap, right at that threshold) but name different members
# of the same category. Each inner tuple is one category; each frozenset
# inside it is one alternative's synonym/phrase variants. A cache hit is
# rejected if the incoming query and a candidate's cached query each
# name a *different* alternative from the same category - not a general
# antonym/word-sense solution (a real, disclosed limitation - a
# confusable pair not listed here can still over-match), but a targeted,
# extensible fix for the reported bug and the class of question it
# represents (client/server split, HTTP verbs, environment, sync mode).
_DISTINGUISHING_CATEGORIES: tuple[tuple[frozenset[str], ...], ...] = (
    (
        frozenset({"backend", "back-end", "back end", "server-side", "serverside", "server"}),
        frozenset({"frontend", "front-end", "front end", "client-side", "clientside", "client"}),
    ),
    (
        frozenset({"login", "log in", "signin", "sign in"}),
        frozenset({"logout", "log out", "signout", "sign out"}),
        frozenset({"register", "registration", "signup", "sign up"}),
    ),
    (
        frozenset({"get"}), frozenset({"post"}), frozenset({"put"}),
        frozenset({"delete"}), frozenset({"patch"}),
    ),
    (frozenset({"read"}), frozenset({"write"})),
    (frozenset({"development", "dev"}), frozenset({"production", "prod"})),
    (frozenset({"synchronous", "sync"}), frozenset({"asynchronous", "async"})),
)


def _content_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace, drop `_STOPWORDS`."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in _STOPWORDS}


def _lexical_jaccard(query_a: str, query_b: str) -> float:
    """Jaccard similarity of `query_a`/`query_b`'s content-word sets.

    Returns 0.0 (not 1.0) if both are empty after stopword removal -
    two all-stopword queries share no real content, so they should not
    be treated as maximally similar by this measure.
    """
    words_a, words_b = _content_words(query_a), _content_words(query_b)
    if not words_a and not words_b:
        return 0.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def _matched_alternatives(query_lower: str, category: tuple[frozenset[str], ...]) -> set[int]:
    """Which alternative(s) within `category` appear in `query_lower` (by whole-word/phrase match).

    `\\b` anchors on the *first and last* character of the (possibly
    multi-word) escaped term, so this works for single words ("dev"
    correctly does not match inside "development" - `\\b` fails right
    after "dev" there, since "e" is still a word character) and
    space/hyphen-containing phrases ("front end", "back-end") alike.
    """
    matched = set()
    for i, alternative in enumerate(category):
        for term in alternative:
            if re.search(rf"\b{re.escape(term)}\b", query_lower):
                matched.add(i)
                break
    return matched


def _has_distinguishing_conflict(query_a: str, query_b: str) -> bool:
    """True if `query_a`/`query_b` name *different* alternatives of the same `_DISTINGUISHING_CATEGORIES` entry.

    Only a positive, two-sided conflict counts - if either query doesn't
    mention anything from a given category at all, that category says
    nothing about compatibility (e.g. "What is the tech stack?" vs "What
    is the backend tech stack?" isn't a conflict - the first query just
    isn't specific).
    """
    a_lower, b_lower = query_a.lower(), query_b.lower()
    for category in _DISTINGUISHING_CATEGORIES:
        matched_a = _matched_alternatives(a_lower, category)
        matched_b = _matched_alternatives(b_lower, category)
        if matched_a and matched_b and matched_a.isdisjoint(matched_b):
            return True
    return False


def _is_lexically_compatible(query_a: str, query_b: str) -> bool:
    """Second, independent gate on top of cosine similarity - see this module's own docstring for why.

    Args:
        query_a: The incoming query.
        query_b: A candidate cached query (already known to meet the
            cosine similarity threshold).

    Returns:
        False (reject the candidate as a cache hit) if either
        `_has_distinguishing_conflict` fires or the two queries' content
        words overlap below `_LEXICAL_OVERLAP_THRESHOLD`; True otherwise.
    """
    if _has_distinguishing_conflict(query_a, query_b):
        return False
    return _lexical_jaccard(query_a, query_b) >= _LEXICAL_OVERLAP_THRESHOLD


class EmbeddingModel(Protocol):
    """The subset of `sentence_transformers.SentenceTransformer` this module needs."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Encode `sentences` into a batch of embedding vectors."""
        ...


class SemanticCacheManager:
    """Looks up and stores (query, response) pairs by embedding similarity, per repository."""

    def __init__(
        self,
        db: DatabaseManager,
        model: EmbeddingModel | None = None,
        model_name: str | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            db: Persistence layer cache entries are stored in and loaded from.
            model: Pre-constructed embedding model. Overridable for
                testing; defaults to lazily loading `model_name` via
                `embedding.model_loader.load_embedding_model`.
            model_name: Embedding model identifier to load and to embed
                queries with. Defaults to
                `embedding.model_loader.active_model_name()` (selected by
                `settings.USE_CODEBERT`) - should match whatever model
                chunks were embedded with, so cache similarity is
                measured on the same embedding space retrieval uses.
            similarity_threshold: Minimum cosine similarity for a cache
                hit. Defaults to `settings.CACHE_SIMILARITY_THRESHOLD`.
        """
        self._db = db
        self._model_name = model_name or active_model_name()
        self._model = model
        self._similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.CACHE_SIMILARITY_THRESHOLD
        )

    def _ensure_model(self) -> EmbeddingModel:
        """Lazily load the embedding model on first use."""
        if self._model is None:
            try:
                self._model = load_embedding_model(self._model_name)
            except EmbeddingError as exc:
                raise RetrievalError(f"Failed to load embedding model for semantic cache: {exc}") from exc
        return self._model

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed `query` with the configured embedding model.

        Args:
            query: The query text to embed.

        Returns:
            `query`'s embedding vector, dtype float32.

        Raises:
            RetrievalError: If the model cannot be loaded or inference fails.
        """
        model = self._ensure_model()
        try:
            vector = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        except Exception as exc:  # noqa: BLE001 - third-party ML inference boundary
            raise RetrievalError(f"Failed to embed query for semantic cache: {exc}") from exc
        return np.asarray(vector, dtype=np.float32)

    def _cosine_similarity(self, query_vector: np.ndarray, query_norm: float, cached_vector: np.ndarray) -> float:
        """Cosine similarity between an already-normed query vector and a cached vector.

        Args:
            query_vector: The incoming query's embedding.
            query_norm: `query_vector`'s L2 norm, precomputed once per
                lookup rather than per comparison.
            cached_vector: A cached entry's embedding.

        Returns:
            Cosine similarity in ``[-1, 1]``, or 0.0 if `cached_vector` is
            a zero vector (undefined direction).
        """
        cached_norm = np.linalg.norm(cached_vector)
        if cached_norm == 0.0 or query_norm == 0.0:
            return 0.0
        return float(np.dot(query_vector, cached_vector) / (query_norm * cached_norm))

    def lookup(self, repository_id: str, query: str) -> SemanticCacheHit | None:
        """Check whether a semantically similar query was already answered for `repository_id`.

        A candidate must pass *two independent* gates to count as a hit:
        cosine similarity >= `settings.CACHE_SIMILARITY_THRESHOLD` (as
        before), and `_is_lexically_compatible(query, candidate.query)`
        (new - see this module's own docstring for the real over-match
        bug this closes). Candidates are checked in descending
        similarity order, so a high-cosine-but-lexically-incompatible
        entry doesn't shadow a slightly-lower-cosine entry that's a
        genuine match.

        Args:
            repository_id: The repository to check the cache for.
            query: The incoming query text.

        Returns:
            The best entry passing both gates, as a `SemanticCacheHit`;
            None if no entry passes both.

        Raises:
            RetrievalError: If loading cache entries or embedding `query` fails.
        """
        logger.info("Cache lookup for repository %s: %r", repository_id, query)
        started_at = time.perf_counter()

        try:
            entries: list[CacheEntry] = self._db.load_cache_entries(repository_id)
        except DatabaseError as exc:
            raise RetrievalError(f"Failed to load cache entries for repository {repository_id}: {exc}") from exc

        if not entries:
            logger.info("Cache miss for repository %s: no cached entries", repository_id)
            return None

        query_vector = self._embed_query(query)
        query_norm = float(np.linalg.norm(query_vector))

        scored = [
            (self._cosine_similarity(query_vector, query_norm, entry.query_embedding), entry) for entry in entries
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        best_raw_similarity = scored[0][0] if scored else -1.0
        rejected_incompatible = 0
        for similarity, entry in scored:
            if similarity < self._similarity_threshold:
                break  # sorted descending - nothing further can pass either
            if not _is_lexically_compatible(query, entry.query):
                rejected_incompatible += 1
                logger.info(
                    "Cache candidate rejected for repository %s: similarity=%.4f meets threshold but "
                    "lexically incompatible (query=%r, candidate=%r)",
                    repository_id, similarity, query, entry.query,
                )
                continue

            latency_saved_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Cache hit for repository %s: similarity=%.4f, matched query=%r, "
                "retrieval/expansion/reranking skipped (lookup took %.2fms)",
                repository_id, similarity, entry.query, latency_saved_ms,
            )
            return SemanticCacheHit(
                cache_id=entry.cache_id,
                query=entry.query,
                response=entry.response,
                retrieved_chunk_ids=entry.retrieved_chunk_ids,
                similarity=similarity,
                created_at=entry.created_at,
            )

        logger.info(
            "Cache miss for repository %s: best similarity %.4f (threshold %.4f), "
            "%d candidate(s) rejected as lexically incompatible",
            repository_id, max(best_raw_similarity, 0.0), self._similarity_threshold, rejected_incompatible,
        )
        return None

    def store(
        self,
        repository_id: str,
        query: str,
        response: str,
        chunk_ids: list[str],
    ) -> int:
        """Cache `response` under `query`, for future similar-query lookups.

        Args:
            repository_id: The repository this query was answered for.
            query: The query text to cache.
            response: The generated response to cache.
            chunk_ids: chunk_ids that produced/support `response` (e.g.
                `models.schemas.GeneratedAnswer.cited_chunks` in Phase 16)
                - persisted as-is, alongside the cached response.

        Returns:
            The stored entry's `cache_id`.

        Raises:
            RetrievalError: If embedding `query` or storing the entry fails.
        """
        query_vector = self._embed_query(query)

        try:
            cache_id = self._db.store_cache_entry(repository_id, query, query_vector, response, chunk_ids)
        except DatabaseError as exc:
            raise RetrievalError(f"Failed to store cache entry for repository {repository_id}: {exc}") from exc

        return cache_id

    def clear(self, repository_id: str) -> int:
        """Delete every cached entry for `repository_id`.

        Args:
            repository_id: The repository whose cache to clear.

        Returns:
            The number of entries removed.

        Raises:
            RetrievalError: If the deletion fails.
        """
        try:
            return self._db.clear_cache(repository_id)
        except DatabaseError as exc:
            raise RetrievalError(f"Failed to clear cache for repository {repository_id}: {exc}") from exc
