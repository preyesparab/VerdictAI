"""Synthesizes the final answer from assembled context via the LLM client (Phase 16).

`LLMService.generate_answer` is the only entry point the rest of the
application should call to get a repository-aware answer: it never
retrieves anything itself - it receives an already-built
`models.schemas.ContextDocument` (Phase 15's output) and only builds the
prompt, calls the configured LLM provider through
`generation.llm_client.LLMClient`, matches citations, and updates the
semantic cache. Retrieval, reranking, and context assembly are Phases
9-15's responsibility, not this module's - keeping generation a pure
function of "context in, answer out" is what keeps every claim in the
answer traceable to a specific piece of retrieved code instead of the
model's own training data.
"""

from __future__ import annotations

import time
from typing import Protocol

from core.exceptions import RetrievalError
from core.logging import get_logger
from generation.llm_client import LLMClient, LLMCompletion
from models.schemas import ChunkCitation, ContextDocument, GeneratedAnswer
from prompts.templates import build_prompt

logger = get_logger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE: str = (
    "The repository does not contain enough information to answer this question."
)


class LLMClientProtocol(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


class SemanticCache(Protocol):
    """The subset of `retrieval.semantic_cache.SemanticCacheManager` this module needs."""

    def store(self, repository_id: str, query: str, response: str, chunk_ids: list[str]) -> int:
        """Cache `response` under `query`, keyed by the chunk_ids that support it."""
        ...


def match_citations(answer: str, citations: list[ChunkCitation]) -> list[str]:
    """Match a generated answer's citations back to chunk_ids.

    A citation is considered present if `answer` mentions the citation's
    `file_path`, plus its `function_name` (or `class_name`, if
    `function_name` is None) when one is available. Plain substring
    containment is used rather than parsing the exact
    ``(source: `path`, function `name`)`` format the system prompt
    requests, since it is robust to whatever punctuation/formatting
    variation the model actually produces around a citation.

    Args:
        answer: The LLM's generated answer text.
        citations: `ContextDocument.chunk_citations` - the citable
            identity of every chunk that was actually in the prompt.

    Returns:
        chunk_ids of every citation `answer` mentions, in
        `citations`'s order. Empty if `answer` mentions none of them
        (including if `answer` states the context was insufficient).
    """
    matched: list[str] = []
    for citation in citations:
        if citation.file_path not in answer:
            continue
        if citation.function_name:
            if citation.function_name not in answer:
                continue
        elif citation.class_name and citation.class_name not in answer:
            continue
        matched.append(citation.chunk_id)
    return matched


class LLMService:
    """Generates repository-aware answers from a pre-built `ContextDocument`.

    Never calls Gemini or Ollama directly - all provider interaction goes
    through `generation.llm_client.LLMClient`. Never retrieves chunks or
    builds context itself - both are supplied by the caller via
    `ContextDocument` (Phase 15's output).
    """

    def __init__(
        self,
        semantic_cache: SemanticCache,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            semantic_cache: Where a successful generation's (query,
                answer, cited chunk_ids) is stored, for future similar-
                query cache hits. Required - unlike `llm_client`, it
                needs a `database.sqlite_client.DatabaseManager` that
                cannot be sensibly defaulted.
            llm_client: The provider client to generate through.
                Overridable for testing; defaults to a new
                `generation.llm_client.LLMClient` configured entirely
                from `config.settings`.
        """
        self._semantic_cache = semantic_cache
        self._llm_client = llm_client or LLMClient()

    def generate_answer(
        self, repository_id: str, query: str, context_document: ContextDocument
    ) -> GeneratedAnswer:
        """Generate a repository-aware answer for `query` from `context_document`.

        If `context_document.context` is empty (nothing was retrieved),
        the LLM is never called - a fixed "not enough information"
        answer is returned immediately, with zero token counts, and
        nothing is written to the cache (there was no generation to
        cache).

        Args:
            repository_id: The repository `query` is about.
            query: The user's question.
            context_document: `generation.context_builder.ContextBuilder.build_context`'s
                output for `(repository_id, query)`.

        Returns:
            The generated answer, including which of
            `context_document.chunk_citations` it actually cited.

        Raises:
            LLMGenerationError: If the configured provider's client
                cannot be constructed, or the generation request fails.
        """
        logger.info("Generation started for repository %s: query=%r", repository_id, query)
        started_at = time.perf_counter()

        if not context_document.context:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Generation skipped for repository %s: empty context, nothing was retrieved",
                repository_id,
            )
            return GeneratedAnswer(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                cited_chunks=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model_name="",
                latency_ms=latency_ms,
            )

        system_prompt, user_prompt = build_prompt(repository_id, context_document.context, query)
        logger.info(
            "Prompt built for repository %s: %d char(s) system, %d char(s) user",
            repository_id, len(system_prompt), len(user_prompt),
        )

        logger.info("Generation started: invoking LLM for repository %s", repository_id)
        completion = self._llm_client.complete(system_prompt, user_prompt)

        cited_chunks = match_citations(completion.text, context_document.chunk_citations)

        latency_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Generation completed for repository %s in %.2fms: %d cited chunk(s), %d total token(s)",
            repository_id, latency_ms, len(cited_chunks), completion.total_tokens,
        )

        generated = GeneratedAnswer(
            answer=completion.text,
            cited_chunks=cited_chunks,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            model_name=completion.model_name,
            latency_ms=latency_ms,
        )

        self._update_cache(repository_id, query, generated)

        return generated

    def _update_cache(self, repository_id: str, query: str, generated: GeneratedAnswer) -> None:
        """Store a successful generation in the semantic cache.

        A cache-write failure never fails the caller's request - the
        user already has their answer; caching is a best-effort side
        channel, not part of the answer's correctness. The failure is
        logged and swallowed.

        Args:
            repository_id: The repository `query` was answered for.
            query: The user's question.
            generated: The just-produced answer to cache.
        """
        try:
            self._semantic_cache.store(repository_id, query, generated.answer, generated.cited_chunks)
        except RetrievalError as exc:
            logger.warning("Cache update failed for repository %s: %s", repository_id, exc)
            return

        logger.info("Cache updated for repository %s", repository_id)
