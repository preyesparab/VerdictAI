"""ContextBuilder: assembles the final LLM-ready context from reranked chunks (Phase 15).

Takes `retrieval.reranker.CrossEncoderReranker.rerank`'s output - the
final ranked chunks - and turns them into the exact string
`generation.llm_client` sends to the LLM in Phase 16. Three
transformations happen, in order:

1. **Small-to-Big substitution**: each ranked (AST-granularity) chunk is
   replaced by its `parent` chunk (`models.schemas.CodeChunk.parent_chunk_id`,
   built by `ingestion.chunker.SemanticChunker`), which carries more
   surrounding code context than an isolated function/method - matching
   stays precise at AST granularity, but reasoning gets the wider window.
   Governed by `settings.USE_SMALL_TO_BIG`. The block's function/class
   name label is always the *originally matched* chunk's, though - never
   the parent's (which is always None, see `models.schemas.ChunkType`) -
   so a citation can always name a concrete function even when the code
   shown around it is a wider parent window.
2. **Deduplication**: multiple ranked chunks that substitute to the same
   parent (e.g. two methods of the same class) collapse to one block,
   labeled with the highest-ranked contributor's function/class name.
3. **Token budgeting**: blocks are kept in descending rerank order until
   the running total would exceed `settings.MAX_CONTEXT_TOKENS`; anything
   after that point (the lowest-ranked remainder) is dropped whole -
   never truncated mid-chunk.

This module never re-runs retrieval, reranking, or chunk parenting -
those are Phases 11-13 and 5's responsibility. It only assembles.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

from config import settings
from core.exceptions import DatabaseError, RetrievalError
from core.logging import get_logger
from models.schemas import ChunkCitation, ChunkType, CodeChunk, ContextDocument, RankedChunk

logger = get_logger(__name__)

# OpenAI's own rule of thumb: ~4 characters per token for typical English/
# code text. Deterministic and dependency-free (no BPE model to load or
# download), which keeps this module offline-testable like every other
# phase - the trade-off is an approximation rather than an exact count for
# any specific LLM's tokenizer.
CHARS_PER_TOKEN: int = 4

_BLOCK_SEPARATOR: str = "\n\n---\n\n"


def estimate_token_count(text: str) -> int:
    """Approximate the number of LLM tokens `text` would consume.

    Args:
        text: Text to estimate the token count of.

    Returns:
        0 for empty text; otherwise ``ceil(len(text) / CHARS_PER_TOKEN)``,
        floored at 1 for any non-empty text.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class ChunkStore(Protocol):
    """The subset of `database.sqlite_client.DatabaseManager` this module needs."""

    def load_chunks(self, repository_id: str) -> list[CodeChunk]:
        """Load every stored chunk (AST, sliding, and parent) for a repository."""
        ...


@dataclasses.dataclass(frozen=True)
class _RenderedBlock:
    """One block on its way to becoming context text - display code plus a citable label.

    Separates two things `CodeChunk` normally couples: the code actually
    *shown* (`raw_code`/`chunk_type`, the parent's if small-to-big
    substitution happened) and the identity a citation should *name*
    (`function_name`/`class_name`, always the originally matched chunk's,
    which carries the specific unit that was actually relevant even when
    the surrounding code shown is wider).
    """

    chunk_id: str
    file_path: str
    language: str
    chunk_type: ChunkType
    function_name: str | None
    class_name: str | None
    raw_code: str


def _format_block(block: _RenderedBlock, repository_id: str) -> str:
    """Format one block into its labeled context text.

    Args:
        block: The block to format.
        repository_id: The repository the block belongs to.

    Returns:
        Text with, in order: Repository, File Path, Language, Chunk
        Type, Function Name, Class Name, then the raw code.
    """
    lines = [
        f"Repository: {repository_id}",
        f"File Path: {block.file_path}",
        f"Language: {block.language}",
        f"Chunk Type: {block.chunk_type.value}",
        f"Function Name: {block.function_name or 'None'}",
        f"Class Name: {block.class_name or 'None'}",
        "Raw Code:",
        block.raw_code,
    ]
    return "\n".join(lines)


class ContextBuilder:
    """Builds small-to-big, deduplicated, token-budgeted context for the LLM.

    Stateless across calls other than its configuration; `build_context`
    can be called repeatedly for different repositories/queries.
    """

    def __init__(
        self,
        db: ChunkStore,
        max_context_tokens: int | None = None,
        use_small_to_big: bool | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            db: Loads every stored chunk for a repository. Overridable
                for testing via any `ChunkStore`-shaped fake; in
                production, a `database.sqlite_client.DatabaseManager`.
            max_context_tokens: Token budget for the assembled context.
                Defaults to `settings.MAX_CONTEXT_TOKENS`.
            use_small_to_big: Whether to substitute each ranked chunk with
                its parent chunk. Defaults to `settings.USE_SMALL_TO_BIG`.

        Raises:
            RetrievalError: If `max_context_tokens` is not positive.
        """
        self._db = db
        self._max_context_tokens = (
            max_context_tokens if max_context_tokens is not None else settings.MAX_CONTEXT_TOKENS
        )
        if self._max_context_tokens <= 0:
            raise RetrievalError(f"max_context_tokens must be positive, got {self._max_context_tokens}")
        self._use_small_to_big = (
            use_small_to_big if use_small_to_big is not None else settings.USE_SMALL_TO_BIG
        )

    def build_context(
        self, repository_id: str, query: str, ranked_chunks: list[RankedChunk]
    ) -> ContextDocument:
        """Assemble the final LLM context from reranked chunks.

        Args:
            repository_id: The repository `ranked_chunks` were retrieved
                from.
            query: The user's query text.
            ranked_chunks: `CrossEncoderReranker.rerank`'s output - not
                required to already be sorted, since this method sorts by
                `final_rank` itself.

        Returns:
            The assembled `ContextDocument`. If `ranked_chunks` is empty,
            `context` is `""`, `included_chunks`/`chunk_citations` are
            `[]`, `total_tokens` is 0, and `truncated` is False.

        Raises:
            RetrievalError: If chunks cannot be loaded for
                `repository_id`, or a ranked chunk's id is not among the
                repository's stored chunks (a broken chunk_id reference).
        """
        logger.info(
            "Context building started for repository %s: %d ranked chunk(s)",
            repository_id, len(ranked_chunks),
        )

        if not ranked_chunks:
            logger.info("Context building completed: empty ranked_chunks, returning empty context")
            return ContextDocument(
                repository_id=repository_id,
                query=query,
                context="",
                included_chunks=[],
                chunk_citations=[],
                total_tokens=0,
                truncated=False,
            )

        chunk_map = self._load_chunk_map(repository_id)
        ordered_ranked = sorted(ranked_chunks, key=lambda ranked: ranked.final_rank)

        substituted, substitutions = self._substitute_parents(ordered_ranked, chunk_map)
        logger.info("Parent chunks substituted: %d/%d", substitutions, len(substituted))

        deduped, duplicates_removed = self._dedupe(substituted)
        logger.info("Duplicate parent chunks removed: %d", duplicates_removed)

        included, truncated = self._enforce_token_budget(deduped, repository_id)
        logger.info(
            "Token budget enforced: %d/%d chunk(s) kept (budget=%d token(s))",
            len(included), len(deduped), self._max_context_tokens,
        )

        context = _BLOCK_SEPARATOR.join(_format_block(block, repository_id) for block in included)
        total_tokens = estimate_token_count(context)

        logger.info(
            "Context building completed for repository %s: %d chunk(s), %d token(s), truncated=%s",
            repository_id, len(included), total_tokens, truncated,
        )

        return ContextDocument(
            repository_id=repository_id,
            query=query,
            context=context,
            included_chunks=[block.chunk_id for block in included],
            chunk_citations=[
                ChunkCitation(
                    chunk_id=block.chunk_id,
                    file_path=block.file_path,
                    function_name=block.function_name,
                    class_name=block.class_name,
                )
                for block in included
            ],
            total_tokens=total_tokens,
            truncated=truncated,
        )

    def _load_chunk_map(self, repository_id: str) -> dict[str, CodeChunk]:
        """Load every stored chunk for `repository_id`, keyed by string chunk_id.

        Args:
            repository_id: The repository to load chunks for.

        Returns:
            A mapping from `str(chunk_id)` to `CodeChunk`, covering AST,
            sliding, and parent chunks alike.

        Raises:
            RetrievalError: If the underlying load fails.
        """
        try:
            chunks = self._db.load_chunks(repository_id)
        except DatabaseError as exc:
            raise RetrievalError(f"Failed to load chunks for repository {repository_id}: {exc}") from exc
        return {str(chunk.chunk_id): chunk for chunk in chunks}

    def _substitute_parents(
        self, ordered_ranked: list[RankedChunk], chunk_map: dict[str, CodeChunk]
    ) -> tuple[list[_RenderedBlock], int]:
        """Resolve each ranked chunk to the block that should represent it.

        Args:
            ordered_ranked: Ranked chunks, already sorted by `final_rank`.
            chunk_map: Every stored chunk for the repository, from
                `_load_chunk_map`.

        Returns:
            A tuple of (resolved blocks in the same order as
            `ordered_ranked`, number of chunks actually substituted with
            their parent).

        Raises:
            RetrievalError: If a ranked chunk's id is not in `chunk_map`.
        """
        resolved: list[_RenderedBlock] = []
        substitutions = 0

        for ranked in ordered_ranked:
            chunk = chunk_map.get(ranked.chunk_id)
            if chunk is None:
                raise RetrievalError(
                    f"Ranked chunk_id {ranked.chunk_id!r} not found among stored chunks for this repository"
                )

            display = chunk
            if self._use_small_to_big and chunk.parent_chunk_id is not None:
                parent = chunk_map.get(str(chunk.parent_chunk_id))
                if parent is not None:
                    display = parent
                    substitutions += 1
                else:
                    logger.warning(
                        "Parent chunk %s not found for chunk %s; using chunk directly",
                        chunk.parent_chunk_id, chunk.chunk_id,
                    )

            resolved.append(
                _RenderedBlock(
                    chunk_id=str(display.chunk_id),
                    file_path=display.file_path,
                    language=display.language,
                    chunk_type=display.chunk_type,
                    function_name=chunk.function_name,
                    class_name=chunk.class_name,
                    raw_code=display.raw_code,
                )
            )

        return resolved, substitutions

    def _dedupe(self, blocks: list[_RenderedBlock]) -> tuple[list[_RenderedBlock], int]:
        """Keep only the first (highest-ranked) occurrence of each chunk_id.

        Args:
            blocks: Blocks in descending rank order, post-substitution.

        Returns:
            A tuple of (deduplicated blocks, number of duplicates removed).
        """
        seen: set[str] = set()
        deduped: list[_RenderedBlock] = []
        duplicates = 0

        for block in blocks:
            if block.chunk_id in seen:
                duplicates += 1
                continue
            seen.add(block.chunk_id)
            deduped.append(block)

        return deduped, duplicates

    def _enforce_token_budget(
        self, blocks: list[_RenderedBlock], repository_id: str
    ) -> tuple[list[_RenderedBlock], bool]:
        """Keep blocks in order until the token budget would be exceeded.

        The highest-ranked block is always included whole, even if it
        alone exceeds `self._max_context_tokens` - an oversized single
        block should not collapse the context to nothing. Every
        subsequent block is added only if doing so keeps the running
        total (including the separator between blocks) within budget;
        the first block that would exceed it, and everything after, is
        dropped in full.

        Args:
            blocks: Deduplicated blocks in descending rank order.
            repository_id: The repository the blocks belong to (used to
                format each block for an accurate token count).

        Returns:
            A tuple of (blocks to include, whether any lower-ranked
            block was dropped to stay within budget).
        """
        included: list[_RenderedBlock] = []
        running_tokens = 0

        for block in blocks:
            block_tokens = estimate_token_count(_format_block(block, repository_id))
            separator_tokens = estimate_token_count(_BLOCK_SEPARATOR) if included else 0
            projected = running_tokens + separator_tokens + block_tokens

            if projected > self._max_context_tokens:
                if not included:
                    included.append(block)
                break

            included.append(block)
            running_tokens = projected

        truncated = len(included) < len(blocks)
        return included, truncated
