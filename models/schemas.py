"""Shared dataclasses/schemas used across layers.

`CodeChunk` is produced by `ingestion.ast_parser.TreeSitterParser` (Phase 4)
and `ingestion.chunker.SemanticChunker` (Phase 5), and will be consumed by
every downstream layer added in later phases: embedding generation
(retrieval), persistence (database), and evaluation. Defining it here —
rather than inside `ingestion/` — keeps those future layers from having
to import the ingestion package just to reference the type.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np


class ChunkType(str, Enum):
    """The kind of unit a `CodeChunk` represents.

    FUNCTION, ASYNC_FUNCTION, CLASS, METHOD, and ARROW_FUNCTION are
    AST-derived ("ast") chunk types, produced by
    `ingestion.ast_parser.TreeSitterParser`. SLIDING and PARENT are
    non-AST, window-based chunk types produced by
    `ingestion.chunker.SemanticChunker` to provide additional retrieval
    context.
    """

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ARROW_FUNCTION = "arrow_function"
    SLIDING = "sliding"
    PARENT = "parent"


@dataclass(frozen=True)
class CodeChunk:
    """A retrievable unit of source code — AST-derived, sliding-window, or parent.

    Attributes:
        chunk_id: Deterministic UUID. For AST chunks, derived from
            `(file_id, name, start_line)`; for sliding/parent chunks,
            derived from `(file_id, chunk_type, start_line, end_line)`.
            Stable across repeated runs over an unchanged file, so
            re-indexing can detect unchanged chunks.
        file_id: Deterministic identifier of the source file this chunk
            was extracted from (see `ingestion.deterministic_ids`).
        file_path: Repository-relative path of the source file.
        language: Source language (e.g., ``"python"``, ``"javascript"``).
        chunk_type: The kind of unit this chunk represents.
        function_name: Name of the function/method, or None for a class,
            sliding, or parent chunk.
        class_name: Name of the class this chunk defines (for a class
            chunk) or is lexically nested within (for a method or nested
            function); None if not applicable.
        parent_class: Base class(es) of `class_name`, comma-separated if
            multiple; None if `class_name` is None or has no bases.
        start_line: 1-indexed line the chunk starts on (inclusive).
        end_line: 1-indexed line the chunk ends on (inclusive).
        raw_code: The exact source text of the chunk, byte-for-byte as it
            appears in the file.
        parent_chunk_id: For an AST chunk, the `chunk_id` of the PARENT
            chunk providing its surrounding context (small-to-big
            retrieval). None for sliding and parent chunks themselves, or
            if no parent has been attached yet.
    """

    chunk_id: uuid.UUID
    file_id: str
    file_path: str
    language: str
    chunk_type: ChunkType
    function_name: str | None
    class_name: str | None
    parent_class: str | None
    start_line: int
    end_line: int
    raw_code: str
    parent_chunk_id: uuid.UUID | None = None


class RetrievalSource(str, Enum):
    """Which retriever(s) - or the knowledge graph - contributed a chunk.

    DENSE/SPARSE mean the chunk was returned by only
    `database.vector_store.FaissIndexManager`/`retrieval.sparse_retriever.BM25Manager`
    respectively; HYBRID means both retrievers returned it (Phase 11's
    `retrieval.hybrid_retriever.HybridRetriever` sums its RRF contribution
    from both). GRAPH means the chunk was not among the original hybrid
    retrieval results at all - it was discovered by Phase 12's
    `retrieval.graph_retriever.GraphExpander` traversing a one-hop
    relationship (call, containment, inheritance, or import) from a chunk
    that was. LOCATION means the chunk was not retrieved at all - it was
    looked up directly by file path/line (Phase 20's
    `GET /repos/{repo_id}/context`), for a caller (e.g. Adjudicate's
    future Context Builder) that already knows exactly which code it
    wants context for.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    LOCATION = "location"


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk's result from `retrieval.hybrid_retriever.HybridRetriever.retrieve`.

    Attributes:
        chunk_id: The matched chunk's id.
        dense_score: Cosine similarity from `FaissIndexManager.search`, or
            None if this chunk was not among the dense candidates.
        bm25_score: BM25 relevance from `BM25Manager.search`, or None if
            this chunk was not among the sparse candidates.
        fused_score: Reciprocal Rank Fusion score:
            ``sum(1 / (rrf_k + rank))`` over every retriever list this
            chunk appeared in. What the final ranking is sorted by.
        retrieval_source: Which retriever(s) contributed this chunk.
        rank: 1-indexed position in the final fused ranking (1 is most
            relevant).
    """

    chunk_id: str
    dense_score: float | None
    bm25_score: float | None
    fused_score: float
    retrieval_source: RetrievalSource
    rank: int


@dataclass(frozen=True)
class ExpandedRetrievedChunk:
    """One chunk's result from `retrieval.graph_retriever.GraphExpander.expand`.

    Every chunk in `HybridRetriever`'s input is always represented in the
    output too (`graph_distance=0`), unchanged and never outranked by a
    graph-discovered chunk - see `GraphExpander.expand`'s docstring for the
    priority guarantee.

    Attributes:
        chunk_id: The chunk's id - either an original hybrid retrieval
            result, or one discovered by graph traversal.
        retrieval_source: `RetrievalSource.DENSE`/`SPARSE`/`HYBRID`
            (carried over unchanged) for `graph_distance=0` chunks;
            `RetrievalSource.GRAPH` for anything graph-discovered.
        score: The original `RetrievedChunk.fused_score` for
            `graph_distance=0` chunks; a decayed expansion score
            (``originating_score * decay_factor ** graph_distance``) for
            graph-discovered chunks. Not directly comparable across the
            `graph_distance=0`/`>0` boundary - see `GraphExpander`'s
            two-tier sort.
        graph_distance: Hop count from the nearest original retrieved
            chunk that led here. 0 for an original hybrid retrieval
            result, 1+ for a graph-discovered chunk.
        originating_chunk_id: The chunk_id one hop closer to an original
            result that this chunk was discovered from. None for
            `graph_distance=0` chunks.
        edge_type: The knowledge-graph edge type connecting
            `originating_chunk_id` to `chunk_id` (e.g. ``"function_call"``,
            ``"method_call"``, ``"inherits"``, ``"contains"``,
            ``"imports"``). None for `graph_distance=0` chunks.
    """

    chunk_id: str
    retrieval_source: RetrievalSource
    score: float
    graph_distance: int
    originating_chunk_id: str | None
    edge_type: str | None


@dataclass(frozen=True)
class RerankCandidate:
    """One candidate handed to `retrieval.reranker.CrossEncoderReranker.rerank`.

    Produced by joining `retrieval.graph_retriever.GraphExpander.expand`'s
    output with the chunk's actual content from
    `database.sqlite_client.DatabaseManager.load_chunks` -
    `CrossEncoderReranker` only scores/ranks; it never loads chunk content
    itself.

    Attributes:
        chunk_id: The chunk's id.
        raw_code: The chunk's source text - what the cross-encoder scores
            against the query.
        file_path: Repository-relative path of the chunk's file.
        function_name: The chunk's function/method name, or None for a
            class chunk.
        retrieval_source: Which retriever/graph traversal produced this
            chunk (carried over unchanged into `RankedChunk`).
        graph_distance: Hop count from an original hybrid retrieval
            result (0 for one itself; see `ExpandedRetrievedChunk`).
        previous_score: This candidate's score prior to reranking
            (`ExpandedRetrievedChunk.score` - RRF fused score or decayed
            graph-expansion score), carried into `RankedChunk.previous_retrieval_score`
            for observability; not used for the reranked ordering.
    """

    chunk_id: str
    raw_code: str
    file_path: str
    function_name: str | None
    retrieval_source: RetrievalSource
    graph_distance: int
    previous_score: float


@dataclass(frozen=True)
class RankedChunk:
    """Final ranked result from `CrossEncoderReranker.rerank`.

    Attributes:
        chunk_id: The chunk's id.
        cross_encoder_score: The cross-encoder's relevance score for
            ``(query, raw_code)`` - what the final ranking is sorted by.
        previous_retrieval_score: `RerankCandidate.previous_score`, kept
            for observability/debugging; not used for ordering.
        final_rank: 1-indexed position in the final ranking (1 is most
            relevant).
        retrieval_source: Which retriever/graph traversal originally
            produced this chunk (carried over unchanged).
    """

    chunk_id: str
    cross_encoder_score: float
    previous_retrieval_score: float
    final_rank: int
    retrieval_source: RetrievalSource


@dataclass(frozen=True)
class CacheEntry:
    """One row loaded from the `SemanticCache` table (Phase 14).

    Attributes:
        cache_id: Autoincrement primary key.
        repository_id: The repository this cached query/response belongs to.
        query: The original query text this entry was cached under.
        query_embedding: `query`'s embedding vector, used to compute
            cosine similarity against a new incoming query.
        response: The previously-generated response text.
        retrieved_chunk_ids: chunk_ids of the `RankedChunk`s that were
            used to produce `response`.
        created_at: When this entry was stored (or last overwritten).
            Not currently used for expiry - present so a future phase can
            add TTL-based invalidation without a schema change.
    """

    cache_id: int
    repository_id: str
    query: str
    query_embedding: np.ndarray
    response: str
    retrieved_chunk_ids: list[str]
    created_at: datetime


@dataclass(frozen=True)
class ChunkCitation:
    """Citable identity of one block included in a `ContextDocument`.

    Produced alongside each included block by
    `generation.context_builder.ContextBuilder.build_context`, and
    consumed by `generation.answer_generator` (Phase 16) to match the
    file path/function name an LLM cites in its answer back to a
    chunk_id, so `models.schemas.GeneratedAnswer.cited_chunks` and the
    semantic cache's `retrieved_chunk_ids` reference real chunks instead
    of free text.

    Attributes:
        chunk_id: The chunk_id of the block this citation refers to -
            matches one entry in `ContextDocument.included_chunks`. When
            small-to-big substitution replaced the originally matched
            chunk with its parent, this is the *parent's* chunk_id (what
            is actually shown in the context), not the original match's.
        file_path: Repository-relative path of the originally matched
            chunk's file (identical whether or not substitution
            happened - a chunk and its parent always share a file).
        function_name: The *originally matched* chunk's function/method
            name - not the parent's, which is always None (see
            `models.schemas.ChunkType`). Preserved through substitution
            so the LLM always has a concrete function to cite even when
            the code shown around it comes from a wider parent window.
            None if the original match was a class, sliding, or parent
            chunk.
        class_name: The originally matched chunk's class name, by the
            same substitution-preserving rule as `function_name`. None
            if not applicable.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    class_name: str | None


@dataclass(frozen=True)
class ContextDocument:
    """Final assembled LLM context from `generation.context_builder.ContextBuilder.build_context`.

    Attributes:
        repository_id: The repository the context was built for.
        query: The user's query text.
        context: The assembled context string - small-to-big substituted
            chunks, deduplicated, ordered by descending rerank relevance,
            and filtered to fit `settings.MAX_CONTEXT_TOKENS` - ready to
            be handed to the LLM client in Phase 16. Empty string if
            nothing was retrieved.
        included_chunks: chunk_ids of every chunk whose formatted block is
            present in `context`, in the same order they appear. These
            are the post-substitution ids (a parent chunk's id, not the
            original AST chunk's id, whenever substitution happened) and
            already deduplicated.
        chunk_citations: One `ChunkCitation` per entry in
            `included_chunks`, same order, giving Phase 16 the file
            path/function name/class name to match an LLM's citations
            against without re-parsing `context`.
        total_tokens: Estimated token count of `context`, per
            `generation.context_builder.estimate_token_count`.
        truncated: True if one or more lower-ranked chunks were dropped
            to keep `context` within the token budget; False if every
            deduplicated chunk was included (even if the one highest-
            ranked chunk alone exceeds the budget - see
            `ContextBuilder._enforce_token_budget`).
    """

    repository_id: str
    query: str
    context: str
    included_chunks: list[str]
    chunk_citations: list[ChunkCitation]
    total_tokens: int
    truncated: bool


@dataclass(frozen=True)
class GeneratedAnswer:
    """Final LLM output from `generation.answer_generator.LLMService.generate_answer` (Phase 16).

    Attributes:
        answer: The generated answer text.
        cited_chunks: chunk_ids from `ContextDocument.chunk_citations`
            whose file path (and function/class name, if any) the answer
            actually mentions - see
            `generation.answer_generator.match_citations`. Empty if the
            answer cited nothing recognizable, or if generation was
            skipped entirely because `ContextDocument.context` was empty.
        prompt_tokens: Tokens consumed by the prompt, per the provider's
            own usage accounting. 0 if generation was skipped.
        completion_tokens: Tokens consumed by the generated answer, per
            the provider's own usage accounting. 0 if generation was
            skipped.
        total_tokens: `prompt_tokens + completion_tokens`.
        model_name: The provider model that generated `answer`
            (`settings.GEMINI_MODEL` or `settings.OLLAMA_MODEL`, whichever
            of `settings.USE_GEMINI`/`USE_OLLAMA` selected). Empty string
            if generation was skipped.
        latency_ms: Wall-clock time spent in `generate_answer`,
            including prompt building, the LLM call (if any), and the
            cache write.
    """

    answer: str
    cited_chunks: list[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    latency_ms: float


@dataclass(frozen=True)
class SemanticCacheHit:
    """Result of a cache hit from `retrieval.semantic_cache.SemanticCacheManager.lookup`.

    Attributes:
        cache_id: The matched `CacheEntry`'s id.
        query: The original cached query text that matched (not the
            incoming query - kept for observability/debugging).
        response: The cached response to return instead of re-running
            retrieval and generation.
        retrieved_chunk_ids: chunk_ids that produced the cached response.
        similarity: Cosine similarity between the incoming query and
            `query`'s embedding - always >= `settings.CACHE_SIMILARITY_THRESHOLD`.
        created_at: When the matched entry was stored.
    """

    cache_id: int
    query: str
    response: str
    retrieved_chunk_ids: list[str]
    similarity: float
    created_at: datetime
