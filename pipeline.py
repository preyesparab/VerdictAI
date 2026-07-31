"""End-to-end pipeline orchestration wiring ingestion, retrieval, and generation (Phase 19).

`Pipeline` is the single entrypoint every caller (CLI, UI, scripts) should
use from now on. It sequences the existing, independently-tested modules
from every backend phase — it contains no parsing, chunking, storage,
embedding, retrieval, or generation logic of its own:

    index_repository(url):
        RepositoryManager -> FileDiscovery -> TreeSitterParser ->
        SemanticChunker -> RepositoryGraphBuilder -> DatabaseManager ->
        EmbeddingManager -> FaissIndexManager -> BM25Manager

    query(question, repo_id):
        HybridRetriever (RRF) -> GraphExpander -> CrossEncoderReranker ->
        SemanticCacheManager -> ContextBuilder -> LLMService

This supersedes the older, Phase-5-only `IndexingPipeline` (acquire ->
discover -> parse -> chunk only) that used to live in this file: that
class's job is a strict subset of `index_repository` below, so nothing
from it is lost. It also replaces `ui.pipeline_service.PipelineService`,
which implemented this exact wiring for Phase 18's Streamlit UI before a
CLI existed; `app.py` and `ui/components.py` now import `Pipeline` from
here directly, and `ui/pipeline_service.py` has been deleted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from config import settings
from core.exceptions import ParsingError
from core.logging import get_logger
from database.graph_store import save_graph
from database.sqlite_client import DatabaseManager
from database.vector_store import FaissIndexManager
from embedding.embedding_manager import EmbeddingManager
from embedding.model_loader import active_model_name, load_embedding_model
from generation.answer_generator import LLMService
from generation.context_builder import ContextBuilder
from generation.llm_client import LLMClient
from graph.graph_builder import RepositoryGraphBuilder
from ingestion.ast_parser import TreeSitterParser
from ingestion.chunker import ChunkingResult, SemanticChunker
from ingestion.file_discovery import FileDiscovery
from ingestion.repository_manager import RepositoryManager
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.source_file import SourceFile
from models.schemas import (
    CodeChunk,
    ExpandedRetrievedChunk,
    GeneratedAnswer,
    RankedChunk,
    RerankCandidate,
    RetrievalSource,
    RetrievedChunk,
)
from retrieval.graph_retriever import GraphExpander
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.semantic_cache import SemanticCacheManager
from retrieval.sparse_retriever import BM25Manager

logger = get_logger(__name__)

# Progress stages reported during `Pipeline.index_repository`, in order -
# callers (CLI, UI) render one line per stage.
INDEXING_STAGES: tuple[str, ...] = (
    "Cloning repository",
    "Parsing source files",
    "Chunking",
    "Building knowledge graph",
    "Generating embeddings",
    "Building FAISS/BM25 indexes",
)

ProgressCallback = Callable[[str, str], None]  # (stage_name, status) where status is running/complete/error


class _NullSemanticCache:
    """No-op stand-in for `SemanticCacheManager`, used when `settings.USE_SEMANTIC_CACHE` is False.

    `generation.answer_generator.LLMService.generate_answer` always
    writes to whatever cache it is given after a successful generation;
    handing it this instead of a real `SemanticCacheManager` is how
    `Pipeline.query` honors the toggle without touching `LLMService` itself.
    """

    def store(self, repository_id: str, query: str, response: str, chunk_ids: list[str]) -> int:
        """Discard the write. Returns a placeholder id."""
        return -1


@dataclass(frozen=True)
class IndexingSummary:
    """Result of `Pipeline.index_repository`.

    Attributes:
        repository: The indexed repository's metadata.
        repository_id: The repository's deterministic id - the key every
            other `Pipeline` method takes as `repo_id`.
        files_discovered: Number of source files found.
        chunks_indexed: Number of chunks persisted (AST + sliding + parent).
        graph_nodes: Number of nodes in the built knowledge graph.
        graph_edges: Number of edges in the built knowledge graph.
        embedded_chunks: Number of chunks newly embedded this run.
    """

    repository: RepositoryMetadata
    repository_id: str
    files_discovered: int
    chunks_indexed: int
    graph_nodes: int
    graph_edges: int
    embedded_chunks: int


@dataclass(frozen=True)
class CitationDisplay:
    """One cited chunk, with everything a caller needs to render it.

    Attributes:
        chunk_id: The cited chunk's id.
        file_path: Repository-relative file path.
        function_name: Function/method name, or None.
        chunk_type: The chunk's `models.schemas.ChunkType` value (e.g.
            ``"function"``, ``"class"``, ``"parent"``).
        retrieval_source: Which retriever/graph traversal produced this
            chunk (e.g. ``"dense"``, ``"hybrid"``, ``"graph"``), or
            ``"cache"`` for a chunk cited by a cached response.
        relevance_score: The cross-encoder's relevance score, or 0.0 if
            reranking was disabled or the citation came from a cache hit.
        raw_code: The cited chunk's source code, for display.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    chunk_type: str
    retrieval_source: str
    relevance_score: float
    raw_code: str


@dataclass(frozen=True)
class SearchResult:
    """One retrieved-and-reranked chunk from `Pipeline.search` (retrieval only, no generation).

    Same shape as `CitationDisplay` minus the fields only generation
    produces (there is no answer to have cited this chunk) - deliberately
    kept structurally close to it so callers that already know how to
    render a `CitationDisplay` (e.g. `api.main`'s `CitationResponse`) need
    no new rendering logic.

    Attributes:
        chunk_id: The chunk's id.
        file_path: Repository-relative file path.
        function_name: Function/method name, or None.
        chunk_type: The chunk's `models.schemas.ChunkType` value.
        retrieval_source: Which retriever/graph traversal produced this
            chunk (e.g. ``"dense"``, ``"hybrid"``, ``"graph"``).
        relevance_score: The cross-encoder's relevance score, or the
            previous retrieval score if reranking was disabled.
        raw_code: The chunk's source code.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    chunk_type: str
    retrieval_source: str
    relevance_score: float
    raw_code: str


@dataclass(frozen=True)
class AskResult:
    """Result of `Pipeline.query`.

    Attributes:
        answer: The generated answer.
        citations: Display-ready detail for every chunk `answer` cited.
        retrieved_count: Number of chunks returned by the initial
            (dense or hybrid) retrieval stage.
        graph_expanded_count: Number of chunks discovered specifically by
            graph traversal (`graph_distance > 0`) - 0 if graph expansion
            was disabled or found nothing new.
        cache_hit: True if this answer was served from the semantic
            cache instead of a fresh generation.
        llm_model: The model that produced `answer` (or a fixed label
            when `cache_hit` is True).
        retrieval_latency_ms: Wall-clock time spent in retrieval +
            context building (0.0 for a cache hit, which skips both).
    """

    answer: GeneratedAnswer
    citations: list[CitationDisplay]
    retrieved_count: int
    graph_expanded_count: int
    cache_hit: bool
    llm_model: str
    retrieval_latency_ms: float


class Pipeline:
    """Wires every backend phase into one indexing flow and one query flow.

    Meant to live for the lifetime of a process (CLI invocation or UI
    session), so expensive collaborators (the embedding model, the
    cross-encoder, per-repository FAISS/BM25 indexes) are constructed once
    and reused across calls rather than reloaded every time.
    """

    def __init__(self, db: DatabaseManager | None = None) -> None:
        """Initialize the pipeline and ensure `settings`'s data directories exist.

        Args:
            db: Persistence layer. Defaults to a new `DatabaseManager`
                (`settings.DATABASE_URL`). Overridable for testing.
        """
        settings.ensure_directories()

        self._db = db or DatabaseManager()
        self._db.initialize_database()

        self._repository_manager = RepositoryManager()
        self._file_discovery = FileDiscovery()
        self._ast_parser = TreeSitterParser()
        self._semantic_chunker = SemanticChunker()

        self._embedding_model: Any = None
        self._reranker: CrossEncoderReranker | None = None
        self._semantic_cache: SemanticCacheManager | None = None
        self._faiss_managers: dict[str, FaissIndexManager] = {}
        self._bm25_managers: dict[str, BM25Manager] = {}

    # -- Indexing --------------------------------------------------------------

    def index_repository(self, url: str, on_progress: ProgressCallback | None = None) -> IndexingSummary:
        """Run the entire indexing pipeline for one repository.

        Order: repository management -> file discovery -> AST parsing ->
        chunking -> call graph construction -> SQLite storage -> embedding
        generation -> FAISS index build -> BM25 index build.

        Args:
            url: A GitHub repository URL
                (``https://github.com/<owner>/<repo>[.git]``).
            on_progress: Called as ``on_progress(stage_name, status)``
                before (``status="running"``) and after
                (``status="complete"``) each of `INDEXING_STAGES`, and
                once more with ``status="error"`` if a stage raises.

        Returns:
            A summary of what was indexed.

        Raises:
            RepositoryCloneError: If the URL is invalid or cloning fails.
            ParsingError: If the repository's local path cannot be scanned.
            EmbeddingError: If embedding generation fails.
            DatabaseError: If persistence fails.
        """

        def _report(stage: str, status: str) -> None:
            if on_progress is not None:
                on_progress(stage, status)

        try:
            _report(INDEXING_STAGES[0], "running")
            repository = self._repository_manager.get_repository(url)
            _report(INDEXING_STAGES[0], "complete")

            _report(INDEXING_STAGES[1], "running")
            source_files = self._file_discovery.discover(repository.local_path)
            parsed_chunks: dict[str, list] = {}
            for source_file in source_files:
                try:
                    parsed_chunks[source_file.relative_path.as_posix()] = self._ast_parser.parse(source_file)
                except ParsingError as exc:
                    logger.warning("Skipping %s during parsing: %s", source_file.relative_path, exc)
            _report(INDEXING_STAGES[1], "complete")

            _report(INDEXING_STAGES[2], "running")
            chunking_results = self._build_chunks(source_files, parsed_chunks)
            _report(INDEXING_STAGES[2], "complete")

            repository_id = self._db.store_repository(repository)
            self._db.store_source_files(repository_id, source_files)
            self._db.store_chunks(repository_id, chunking_results)

            _report(INDEXING_STAGES[3], "running")
            all_ast_chunks = [chunk for result in chunking_results.values() for chunk in result.ast_chunks]
            graph = RepositoryGraphBuilder().build_graph(source_files, all_ast_chunks)
            self._db.store_graph(repository_id, graph)
            save_graph(graph, repository_id, self._db)
            _report(INDEXING_STAGES[3], "complete")

            _report(INDEXING_STAGES[4], "running")
            embedding_manager = EmbeddingManager(self._db)
            embedded_count = embedding_manager.generate_embeddings(repository_id)
            _report(INDEXING_STAGES[4], "complete")

            _report(INDEXING_STAGES[5], "running")
            faiss_manager = FaissIndexManager(self._db)
            faiss_manager.build_index(repository_id)
            faiss_manager.save_index(repository_id)
            self._faiss_managers[repository_id] = faiss_manager

            bm25_manager = BM25Manager(self._db)
            bm25_manager.build_index(repository_id)
            bm25_manager.save_index(repository_id)
            self._bm25_managers[repository_id] = bm25_manager
            _report(INDEXING_STAGES[5], "complete")
        except Exception:
            _report("error", "error")
            raise

        chunks_indexed = sum(
            len(result.ast_chunks) + len(result.sliding_chunks) + len(result.parent_chunks)
            for result in chunking_results.values()
        )

        logger.info(
            "Repository indexed: %s/%s (repository_id=%s, %d file(s), %d chunk(s))",
            repository.owner, repository.name, repository_id, len(source_files), chunks_indexed,
        )

        return IndexingSummary(
            repository=repository,
            repository_id=repository_id,
            files_discovered=len(source_files),
            chunks_indexed=chunks_indexed,
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
            embedded_chunks=embedded_count,
        )

    def _build_chunks(
        self, source_files: list[SourceFile], parsed_chunks: dict[str, list]
    ) -> dict[str, ChunkingResult]:
        """Run `SemanticChunker` over every successfully parsed file."""
        results: dict[str, ChunkingResult] = {}
        for source_file in source_files:
            key = source_file.relative_path.as_posix()
            if key not in parsed_chunks:
                continue
            results[key] = self._semantic_chunker.build_chunks(source_file, parsed_chunks[key])
        return results

    # -- Querying ----------------------------------------------------------

    def query(self, question: str, repo_id: str) -> AskResult:
        """Run the complete retrieval + generation pipeline for one query.

        Order: hybrid retrieval (RRF) -> graph expansion -> cross-encoder
        reranking -> semantic cache check -> context builder
        (small-to-big) -> LLM answer generation. Honors
        `settings.USE_BM25`/`USE_GRAPH_EXPANSION`/`USE_RERANKER`/
        `USE_SMALL_TO_BIG`/`USE_SEMANTIC_CACHE` as currently set.

        Args:
            question: The user's natural-language question.
            repo_id: The repository to query - must already be indexed
                via `index_repository`.

        Returns:
            The generated answer plus retrieval/citation detail for display.

        Raises:
            RetrievalError: If the repository has no saved index, or
                retrieval fails.
            LLMGenerationError: If generation fails.
            DatabaseError: If loading chunks fails.
        """
        repository_id = repo_id
        query = question
        started_at = time.perf_counter()
        semantic_cache = self._get_semantic_cache()

        if settings.USE_SEMANTIC_CACHE:
            cache_hit = semantic_cache.lookup(repository_id, query)
            if cache_hit is not None:
                chunk_map = {str(chunk.chunk_id): chunk for chunk in self._db.load_chunks(repository_id)}
                return AskResult(
                    answer=GeneratedAnswer(
                        answer=cache_hit.response,
                        cited_chunks=cache_hit.retrieved_chunk_ids,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        model_name="(served from semantic cache)",
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                    ),
                    citations=self._build_citations(cache_hit.retrieved_chunk_ids, chunk_map, {}, source_label="cache"),
                    retrieved_count=0,
                    graph_expanded_count=0,
                    cache_hit=True,
                    llm_model="(served from semantic cache)",
                    retrieval_latency_ms=0.0,
                )

        retrieval_started_at = time.perf_counter()
        ranked, chunk_map, retrieved_count, graph_expanded_count = self._retrieve_and_rank(query, repository_id)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000

        context_builder = ContextBuilder(self._db, use_small_to_big=settings.USE_SMALL_TO_BIG)
        context_document = context_builder.build_context(repository_id, query, ranked)

        cache_for_generation = semantic_cache if settings.USE_SEMANTIC_CACHE else _NullSemanticCache()
        llm_service = LLMService(semantic_cache=cache_for_generation, llm_client=self._get_llm_client())
        answer = llm_service.generate_answer(repository_id, query, context_document)

        # `ranked` is keyed by the ORIGINAL (pre-substitution) chunk_ids the
        # reranker scored, but a cited chunk_id may be a PARENT's id (small-
        # to-big substitution happened inside ContextBuilder, after
        # reranking). Re-key by each chunk's *displayed* id - the same one
        # ContextBuilder computed - so a citation's retrieval source/score
        # can still be found; first-ranked wins for a shared parent, same
        # as ContextBuilder's own dedup.
        ranked_by_id: dict[str, RankedChunk] = {}
        for item in ranked:
            ranked_by_id.setdefault(self._display_chunk_id(item.chunk_id, chunk_map), item)
        citation_labels = {citation.chunk_id: citation for citation in context_document.chunk_citations}
        citations = self._build_citations(answer.cited_chunks, chunk_map, ranked_by_id, citation_labels=citation_labels)

        return AskResult(
            answer=answer,
            citations=citations,
            retrieved_count=retrieved_count,
            graph_expanded_count=graph_expanded_count,
            cache_hit=False,
            llm_model=answer.model_name,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    def search(self, query: str, repo_id: str) -> list[SearchResult]:
        """Run retrieval only (hybrid -> graph expansion -> reranking) - no generation, no semantic cache.

        For callers that need "what code relates to X" without an LLM
        call - e.g. Adjudicate's context builder (Phase 24), which needs
        this as its fallback when graph traversal can't find a
        relationship, such as a test that exercises a function only via
        a mock/string reference rather than a direct call edge.

        Args:
            query: A search phrase (need not be a natural-language
                question - identifier-heavy phrases work well against
                the Gemini embedding model's code embeddings).
            repo_id: The repository to search - must already be indexed.

        Returns:
            Up to `settings.TOP_K_FINAL` chunks, ranked best-first.

        Raises:
            RetrievalError: If the repository has no saved index, or
                retrieval fails.
            DatabaseError: If loading chunks fails.
        """
        ranked, chunk_map, _retrieved_count, _graph_expanded_count = self._retrieve_and_rank(query, repo_id)
        return [
            SearchResult(
                chunk_id=item.chunk_id,
                file_path=chunk_map[item.chunk_id].file_path,
                function_name=chunk_map[item.chunk_id].function_name,
                chunk_type=chunk_map[item.chunk_id].chunk_type.value,
                retrieval_source=item.retrieval_source.value,
                relevance_score=item.cross_encoder_score,
                raw_code=chunk_map[item.chunk_id].raw_code,
            )
            for item in ranked
            if item.chunk_id in chunk_map
        ]

    def _retrieve_and_rank(
        self, query: str, repository_id: str
    ) -> tuple[list[RankedChunk], dict[str, CodeChunk], int, int]:
        """Run hybrid retrieval -> graph expansion -> reranking, shared by `query` and `search`.

        Args:
            query: The search phrase or question.
            repository_id: The repository to retrieve against.

        Returns:
            A 4-tuple: the final ranked chunks, a chunk_id -> `CodeChunk`
            map for the repository, the initial retrieval count, and the
            number of chunks discovered specifically by graph expansion.
        """
        query_embedding = self._embed_query(query)

        faiss_manager = self._get_faiss_manager(repository_id)
        if settings.USE_BM25:
            bm25_manager = self._get_bm25_manager(repository_id)
            retrieved = HybridRetriever(faiss_manager, bm25_manager).retrieve(
                repository_id, query, query_embedding, settings.TOP_K_FINAL
            )
        else:
            dense_results = faiss_manager.search(query_embedding, settings.TOP_K_FINAL)
            retrieved = [
                RetrievedChunk(
                    chunk_id=result.chunk_id, dense_score=result.score, bm25_score=None,
                    fused_score=result.score, retrieval_source=RetrievalSource.DENSE, rank=rank,
                )
                for rank, result in enumerate(dense_results, start=1)
            ]
        retrieved_count = len(retrieved)

        if settings.USE_GRAPH_EXPANSION:
            expanded = GraphExpander(self._db).expand(repository_id, retrieved)
        else:
            expanded = [
                ExpandedRetrievedChunk(
                    chunk_id=item.chunk_id, retrieval_source=item.retrieval_source, score=item.fused_score,
                    graph_distance=0, originating_chunk_id=None, edge_type=None,
                )
                for item in retrieved
            ]
        graph_expanded_count = sum(1 for item in expanded if item.graph_distance > 0)

        chunk_map = {str(chunk.chunk_id): chunk for chunk in self._db.load_chunks(repository_id)}
        candidates = [
            RerankCandidate(
                chunk_id=item.chunk_id, raw_code=chunk_map[item.chunk_id].raw_code,
                file_path=chunk_map[item.chunk_id].file_path, function_name=chunk_map[item.chunk_id].function_name,
                retrieval_source=item.retrieval_source, graph_distance=item.graph_distance, previous_score=item.score,
            )
            for item in expanded
            if item.chunk_id in chunk_map
        ]

        if settings.USE_RERANKER:
            ranked = self._get_reranker().rerank(query, candidates, settings.TOP_K_FINAL)
        else:
            ranked = [
                RankedChunk(
                    chunk_id=candidate.chunk_id, cross_encoder_score=candidate.previous_score,
                    previous_retrieval_score=candidate.previous_score, final_rank=rank,
                    retrieval_source=candidate.retrieval_source,
                )
                for rank, candidate in enumerate(candidates[: settings.TOP_K_FINAL], start=1)
            ]
        return ranked, chunk_map, retrieved_count, graph_expanded_count

    def _display_chunk_id(self, chunk_id: str, chunk_map: dict[str, CodeChunk]) -> str:
        """The chunk_id `ContextBuilder` actually displays `chunk_id` under.

        Mirrors `generation.context_builder.ContextBuilder`'s own
        small-to-big substitution rule: a chunk with a parent is shown
        (and cited) under its *parent's* chunk_id when
        `settings.USE_SMALL_TO_BIG` is on and that parent is in the store;
        otherwise it is shown under its own id.

        Args:
            chunk_id: An original (pre-substitution) chunk_id, e.g. from
                a `RankedChunk`.
            chunk_map: chunk_id -> `CodeChunk` for the repository.

        Returns:
            The id `chunk_id` would actually be cited/displayed under.
        """
        chunk = chunk_map.get(chunk_id)
        if chunk is None or not settings.USE_SMALL_TO_BIG or chunk.parent_chunk_id is None:
            return chunk_id
        parent_id = str(chunk.parent_chunk_id)
        return parent_id if parent_id in chunk_map else chunk_id

    def _build_citations(
        self,
        chunk_ids: list[str],
        chunk_map: dict[str, CodeChunk],
        ranked_by_id: dict[str, RankedChunk],
        source_label: str | None = None,
        citation_labels: dict[str, Any] | None = None,
    ) -> list[CitationDisplay]:
        """Assemble display-ready `CitationDisplay`s for `chunk_ids`.

        Args:
            chunk_ids: The chunk_ids the generated answer actually cited.
            chunk_map: chunk_id -> `CodeChunk`, for chunk type and code -
                and the file path/function name fallback when
                `citation_labels` has no entry.
            ranked_by_id: chunk_id -> `RankedChunk`, for retrieval source
                and relevance score. May be missing entries (e.g. a cache
                hit, which has no fresh ranking).
            source_label: Fixed retrieval-source label to use when
                `ranked_by_id` has no entry for a chunk (e.g. ``"cache"``).
            citation_labels: chunk_id -> `models.schemas.ChunkCitation`,
                from `ContextDocument.chunk_citations`. Preferred over
                `chunk_map` for file path/function name: when small-to-big
                substitution replaced a chunk with its parent, `chunk_map`
                only has the *parent's* metadata (function_name is always
                None for a parent chunk - see `models.schemas.ChunkType`),
                while `chunk_citations` preserves the *originally matched*
                function/class name. None for a cache hit, which has no
                fresh `ContextDocument` to draw labels from.

        Returns:
            One `CitationDisplay` per chunk_id found in `chunk_map`
            (chunk_ids no longer present in the store are skipped).
        """
        citations = []
        for chunk_id in chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            ranked = ranked_by_id.get(chunk_id)
            label = citation_labels.get(chunk_id) if citation_labels else None
            citations.append(
                CitationDisplay(
                    chunk_id=chunk_id,
                    file_path=label.file_path if label else chunk.file_path,
                    function_name=label.function_name if label else chunk.function_name,
                    chunk_type=chunk.chunk_type.value,
                    retrieval_source=(ranked.retrieval_source.value if ranked else source_label or "unknown"),
                    relevance_score=(ranked.cross_encoder_score if ranked else 0.0),
                    raw_code=chunk.raw_code,
                )
            )
        return citations

    # -- Shared lazily-loaded collaborators ---------------------------------

    def _embed_query(self, query: str) -> Any:
        """Embed `query` with the active embedding model, loading it on first use."""
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model()
        return self._embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

    def _get_reranker(self) -> CrossEncoderReranker:
        """Lazily construct (and cache) the cross-encoder reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _get_semantic_cache(self) -> SemanticCacheManager:
        """Lazily construct (and cache) the semantic cache manager."""
        if self._semantic_cache is None:
            self._semantic_cache = SemanticCacheManager(self._db)
        return self._semantic_cache

    def _get_llm_client(self) -> LLMClient:
        """Construct a fresh `LLMClient` reflecting the current `settings.USE_OLLAMA`.

        Not cached on the instance (unlike the reranker/embedding model):
        it is cheap to construct and must pick up a live provider toggle
        immediately if `settings.USE_OLLAMA` changes between queries.
        """
        return LLMClient()

    def _get_faiss_manager(self, repository_id: str) -> FaissIndexManager:
        """Get (loading from disk if needed) the FAISS index for `repository_id`."""
        if repository_id not in self._faiss_managers:
            manager = FaissIndexManager(self._db)
            manager.load_index(repository_id)
            self._faiss_managers[repository_id] = manager
        return self._faiss_managers[repository_id]

    def _get_bm25_manager(self, repository_id: str) -> BM25Manager:
        """Get (loading from disk if needed) the BM25 index for `repository_id`."""
        if repository_id not in self._bm25_managers:
            manager = BM25Manager(self._db)
            manager.load_index(repository_id)
            self._bm25_managers[repository_id] = manager
        return self._bm25_managers[repository_id]

    # -- Repository/model info --------------------------------------------------

    def embedding_model_name(self) -> str:
        """The currently active embedding model name."""
        return active_model_name()

    def llm_model_name(self) -> str:
        """The currently active LLM's provider and model name (mirrors `LLMClient`'s selection order).

        Real bug found and fixed live during Phase 32 Part 2's own
        verification: this was missing the `USE_GROQ` branch entirely
        (checked `USE_GEMINI` then jumped straight to `USE_OLLAMA`), so
        `/info` reported "No LLM provider configured" for a
        Groq-configured process even though `generation.llm_client
        .LLMClient` itself correctly selects Gemini -> Groq -> Ollama, in
        that order (see that module's own docstring) - a display-only
        bug, not a functional one (real Groq calls were never affected,
        only this label).
        """
        if settings.USE_GEMINI:
            return f"Gemini ({settings.GEMINI_MODEL})"
        if settings.USE_GROQ:
            return f"Groq ({settings.GROQ_MODEL})"
        if settings.USE_OLLAMA:
            return f"Ollama ({settings.OLLAMA_MODEL})"
        return "No LLM provider configured"

    def load_repository_metadata(self, repository_id: str) -> RepositoryMetadata | None:
        """Load a previously indexed repository's metadata, or None if not indexed."""
        return self._db.load_repository(repository_id)
