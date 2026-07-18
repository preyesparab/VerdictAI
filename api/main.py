"""FastAPI HTTP layer for RepoMind (Phase 20; `/search` added in Phase 24).

Wraps `pipeline.Pipeline` (Phase 19) behind six endpoints so the CLI, a
future React frontend, and Adjudicate can all call indexing/query/
search/context/graph over HTTP instead of importing Python modules
directly. This module contains no ingestion/retrieval/generation logic
of its own - every endpoint either delegates straight to `Pipeline`
(`/query` and `/search` both call `Pipeline`'s retrieval steps; `/search`
stops before generation), or (for `/context` and `/graph`, which
`Pipeline` does not expose a method for) reuses
`database.sqlite_client.DatabaseManager`, `database.graph_store`, and
`generation.context_builder.ContextBuilder` directly, exactly as they
already are.

Indexing status is tracked in an in-memory registry (`_INDEX_STATE`),
populated by the background task's `on_progress` callback -
`Pipeline.index_repository` already reports its own progress this way,
so no pipeline change was needed to support `GET /status`. If the process
restarts, `GET /status` falls back to checking whether the repository
row exists in SQLite (a repository is only ever stored once every stage
of `index_repository` has run - see that method's body) - a best-effort
reconstruction, not a perfectly accurate progress replay.

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from adjudicate.orchestrator.live_review import run_live_review
from config import settings
from core.exceptions import (
    DatabaseError,
    EmbeddingError,
    LLMGenerationError,
    ParsingError,
    RepoMindError,
    RepositoryCloneError,
    RetrievalError,
)
from core.logging import get_logger
from database import graph_store
from database.sqlite_client import DatabaseManager
from generation.context_builder import ContextBuilder
from graph.blast_radius import compute_blast_radius
from ingestion.validators import validate_github_url
from models.schemas import RankedChunk, RetrievalSource
from pipeline import IndexingSummary, Pipeline

logger = get_logger(__name__)

QUERY_TIMEOUT_SECONDS = 120.0

app = FastAPI(title="RepoMind API", version="1.0.0")

# The Phase 32 React frontend (Vite dev server, localhost:5173 by default)
# calls this API directly from the browser - unlike the Streamlit UI
# (Phase 18), which never made a same-origin-policy-governed request
# since `ui/api_client.py`'s `httpx` calls run server-side. Origins are
# read from `settings` rather than hardcoded so a deployed frontend
# (Phase 33) only needs a config change, not a code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Shared singletons --------------------------------------------------------
# One `Pipeline` (and the `DatabaseManager` it wraps) per process, mirroring
# `app.py`'s `st.cache_resource` pattern: the SQLite connection, embedding
# model, cross-encoder, and per-repository FAISS/BM25 indexes are all
# expensive to reconstruct and must not be rebuilt on every request.
_pipeline: Pipeline | None = None
_db: DatabaseManager | None = None


def _get_pipeline() -> Pipeline:
    """Return (constructing on first use) the process-wide `Pipeline`."""
    global _pipeline, _db
    if _pipeline is None:
        _pipeline = Pipeline()
        _db = DatabaseManager()
    return _pipeline


def _get_db() -> DatabaseManager:
    """Return the process-wide `DatabaseManager`, for reads `Pipeline` has no method for."""
    _get_pipeline()
    assert _db is not None  # noqa: S101 - set alongside _pipeline in _get_pipeline
    return _db


# -- Indexing status registry -------------------------------------------------


@dataclass
class RepoIndexState:
    """In-memory indexing status for one repository, keyed by repo_id."""

    status: str  # "pending" | "indexing" | "ready" | "failed"
    stage: str | None = None
    summary: IndexingSummary | None = None
    error: str | None = None


_index_state: dict[str, RepoIndexState] = {}
_state_lock = threading.Lock()


def _run_indexing(repo_id: str, repo_url: str) -> None:
    """Background task: run `Pipeline.index_repository` and record its outcome.

    Args:
        repo_id: The repository's precomputed id (see `index_repository`
            below - derived from the URL before this task starts).
        repo_url: The repository URL to index.
    """
    with _state_lock:
        _index_state[repo_id].status = "indexing"

    def on_progress(stage: str, status: str) -> None:
        if status == "running":
            with _state_lock:
                _index_state[repo_id].stage = stage

    try:
        summary = _get_pipeline().index_repository(repo_url, on_progress=on_progress)
    except RepoMindError as exc:
        logger.error("Indexing failed for repository %s (%s): %s", repo_id, repo_url, exc)
        with _state_lock:
            _index_state[repo_id] = RepoIndexState(status="failed", error=str(exc))
        return

    logger.info("Indexing complete for repository %s (%s)", repo_id, repo_url)
    with _state_lock:
        _index_state[repo_id] = RepoIndexState(status="ready", summary=summary)


def _reconstruct_ready_state(repo_id: str) -> RepoIndexState | None:
    """Best-effort `RepoIndexState` for a repository indexed in a previous process.

    Args:
        repo_id: The repository to look up.

    Returns:
        A synthetic "ready" state if `repo_id` has a stored repository
        row, reconstructing stats from what is cheaply queryable;
        None if no such repository has ever been indexed.
    """
    db = _get_db()
    repository = db.load_repository(repo_id)
    if repository is None:
        return None

    chunks = db.load_chunks(repo_id)
    embedded_ids = db.get_embedded_chunk_ids(repo_id, _get_pipeline().embedding_model_name())

    # Same graph `GET /graph` reads - the JSON `save_graph` wrote from
    # `RepositoryGraphBuilder`'s original graph - not `DatabaseManager.
    # load_graph`'s lossy SQL reconstruction (see `get_graph`'s
    # docstring). Both must report identical node/edge counts.
    try:
        graph = graph_store.load_graph(path=_graph_json_path(repository.owner, repository.name))
        graph_nodes, graph_edges = graph.number_of_nodes(), graph.number_of_edges()
    except DatabaseError:
        graph_nodes = graph_edges = 0

    summary = IndexingSummary(
        repository=repository,
        repository_id=repo_id,
        files_discovered=len({chunk.file_id for chunk in chunks}),
        chunks_indexed=len(chunks),
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        embedded_chunks=len(embedded_ids),
    )
    return RepoIndexState(status="ready", summary=summary)


def _require_ready(repo_id: str) -> RepoIndexState:
    """Look up `repo_id`'s indexing state, raising an `HTTPException` if it isn't queryable yet.

    Args:
        repo_id: The repository to check.

    Returns:
        The repository's `RepoIndexState`, guaranteed `status == "ready"`.

    Raises:
        HTTPException: 404 if `repo_id` has never been indexed; 409 if
            indexing is still pending/in progress or previously failed.
    """
    with _state_lock:
        state = _index_state.get(repo_id)

    if state is None:
        state = _reconstruct_ready_state(repo_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Repository {repo_id!r} not found. POST /repos/index first.")

    if state.status == "failed":
        raise HTTPException(status_code=409, detail=f"Indexing failed for repository {repo_id!r}: {state.error}")
    if state.status in ("pending", "indexing"):
        raise HTTPException(
            status_code=409,
            detail=f"Repository {repo_id!r} is still being indexed (status={state.status}, stage={state.stage}).",
        )
    return state


# -- Request logging middleware ------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log method, path, status code, and latency for every request."""
    started_at = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    return response


def _repomind_error_status(exc: RepoMindError) -> int:
    """Map a `RepoMindError` subtype to the HTTP status code it should surface as."""
    if isinstance(exc, RepositoryCloneError):
        return 400
    if isinstance(exc, RetrievalError):
        return 404
    if isinstance(exc, (ParsingError, EmbeddingError, DatabaseError, LLMGenerationError)):
        return 500
    return 500


# -- Request/response models ---------------------------------------------------


class InfoResponse(BaseModel):
    embedding_model: str
    llm_model: str


class IndexRequest(BaseModel):
    repo_url: str


class IndexResponse(BaseModel):
    repo_id: str
    status: str


class StatusResponse(BaseModel):
    repo_id: str
    status: str
    stage: str | None = None
    error: str | None = None
    owner: str | None = None
    name: str | None = None
    commit_hash: str | None = None
    files_discovered: int | None = None
    chunks_indexed: int | None = None
    graph_nodes: int | None = None
    graph_edges: int | None = None
    embedded_chunks: int | None = None


class QueryRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    chunk_id: str
    file_path: str
    function_name: str | None
    chunk_type: str
    retrieval_source: str
    relevance_score: float
    raw_code: str


class QueryResponse(BaseModel):
    answer: str
    model: str
    cache_hit: bool
    retrieved_count: int
    graph_expanded_count: int
    latency_ms: float
    citations: list[CitationResponse]


class SearchResponse(BaseModel):
    results: list[CitationResponse]


class ReviewRequest(BaseModel):
    diff: str


class ChunkCitationResponse(BaseModel):
    chunk_id: str
    file_path: str
    function_name: str | None
    class_name: str | None
    chunk_type: str | None


class ContextResponse(BaseModel):
    repository_id: str
    query: str
    context: str
    included_chunks: list[str]
    chunk_citations: list[ChunkCitationResponse]
    matched_chunks: list[ChunkCitationResponse]
    total_tokens: int
    truncated: bool


# -- Endpoints ------------------------------------------------------------------


@app.get("/info", response_model=InfoResponse)
def get_info() -> InfoResponse:
    """Return the currently configured embedding/LLM model names.

    Not repo-scoped - these are global config, exposed so a pure HTTP
    client (the Streamlit UI, the future React frontend) can display them
    without reading Python config directly.
    """
    pipeline = _get_pipeline()
    return InfoResponse(embedding_model=pipeline.embedding_model_name(), llm_model=pipeline.llm_model_name())


@app.post("/repos/index", status_code=202, response_model=IndexResponse)
def index_repository(request: IndexRequest, background_tasks: BackgroundTasks) -> IndexResponse:
    """Trigger indexing for a repository, returning immediately with its repo_id.

    Indexing runs as a background task (`_run_indexing`); poll
    `GET /repos/{repo_id}/status` for progress and completion.
    """
    try:
        parsed = validate_github_url(request.repo_url)
    except RepositoryCloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    repo_id = DatabaseManager.compute_repository_id(parsed.owner, parsed.repo)

    with _state_lock:
        existing = _index_state.get(repo_id)
        if existing is not None and existing.status in ("pending", "indexing"):
            return IndexResponse(repo_id=repo_id, status=existing.status)
        _index_state[repo_id] = RepoIndexState(status="pending")

    background_tasks.add_task(_run_indexing, repo_id, request.repo_url)
    logger.info("Indexing queued for repository %s (%s)", repo_id, request.repo_url)
    return IndexResponse(repo_id=repo_id, status="pending")


@app.get("/repos/{repo_id}/status", response_model=StatusResponse)
def get_status(repo_id: str) -> StatusResponse:
    """Return indexing progress/state, and stats once ready."""
    with _state_lock:
        state = _index_state.get(repo_id)

    if state is None:
        state = _reconstruct_ready_state(repo_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Repository {repo_id!r} not found. POST /repos/index first.")

    summary = state.summary
    return StatusResponse(
        repo_id=repo_id,
        status=state.status,
        stage=state.stage,
        error=state.error,
        owner=summary.repository.owner if summary else None,
        name=summary.repository.name if summary else None,
        commit_hash=summary.repository.commit_hash if summary else None,
        files_discovered=summary.files_discovered if summary else None,
        chunks_indexed=summary.chunks_indexed if summary else None,
        graph_nodes=summary.graph_nodes if summary else None,
        graph_edges=summary.graph_edges if summary else None,
        embedded_chunks=summary.embedded_chunks if summary else None,
    )


@app.post("/repos/{repo_id}/query", response_model=QueryResponse)
async def query_repository(repo_id: str, request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question about an already-indexed repository."""
    _require_ready(repo_id)

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_get_pipeline().query, request.question, repo_id),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Query timed out after {QUERY_TIMEOUT_SECONDS}s") from exc
    except RepoMindError as exc:
        raise HTTPException(status_code=_repomind_error_status(exc), detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer.answer,
        model=result.llm_model,
        cache_hit=result.cache_hit,
        retrieved_count=result.retrieved_count,
        graph_expanded_count=result.graph_expanded_count,
        latency_ms=result.answer.latency_ms,
        citations=[
            CitationResponse(
                chunk_id=citation.chunk_id,
                file_path=citation.file_path,
                function_name=citation.function_name,
                chunk_type=citation.chunk_type,
                retrieval_source=citation.retrieval_source,
                relevance_score=citation.relevance_score,
                raw_code=citation.raw_code,
            )
            for citation in result.citations
        ],
    )


@app.get("/repos/{repo_id}/search", response_model=SearchResponse)
async def search_repository(repo_id: str, q: str) -> SearchResponse:
    """Run retrieval only (hybrid + graph expansion + reranking) - no LLM generation, no semantic cache.

    Exists for callers that need "what code relates to `q`" without
    paying for (or depending on) an LLM call - currently only
    Adjudicate's context builder (Phase 24), which falls back to this
    when graph traversal can't find a relationship (e.g. a test that
    exercises a function only via a mock/string reference, not a direct
    call edge `graph.graph_builder` would have captured). Reuses
    `pipeline.Pipeline.search`, the same retrieval steps `/query` runs
    before generation - no retrieval logic of its own.
    """
    _require_ready(repo_id)

    try:
        results = await asyncio.wait_for(
            run_in_threadpool(_get_pipeline().search, q, repo_id),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Search timed out after {QUERY_TIMEOUT_SECONDS}s") from exc
    except RepoMindError as exc:
        raise HTTPException(status_code=_repomind_error_status(exc), detail=str(exc)) from exc

    return SearchResponse(
        results=[
            CitationResponse(
                chunk_id=item.chunk_id,
                file_path=item.file_path,
                function_name=item.function_name,
                chunk_type=item.chunk_type,
                retrieval_source=item.retrieval_source,
                relevance_score=item.relevance_score,
                raw_code=item.raw_code,
            )
            for item in results
        ]
    )


@app.get("/repos/{repo_id}/context", response_model=ContextResponse)
def get_context(repo_id: str, file: str, line: int) -> ContextResponse:
    """Return raw context-builder output for a specific file:line location.

    Unlike `/query`, this does not run retrieval - it looks up whichever
    stored chunk(s) already enclose `file:line` directly (smallest
    enclosing chunk ranked first) and feeds them straight into the
    existing `ContextBuilder`, tagged `RetrievalSource.LOCATION` since
    they were found by location lookup, not retrieved. Intended for
    Adjudicate's context builder (Phase 24), which already knows exactly
    which code changed.

    `matched_chunks` (every raw match at this location, smallest-first)
    is distinct from `chunk_citations` (`ContextBuilder`'s own output,
    which - when `settings.USE_SMALL_TO_BIG` is on, the default -
    substitutes each match for its *parent* chunk, so it never actually
    contains the raw matched chunk_id): callers that need the literal
    enclosing chunk (e.g. Phase 24's blast-radius lookup, which needs a
    real graph node - a SLIDING/PARENT window chunk never is one) should
    read `matched_chunks`, not `chunk_citations`.
    """
    _require_ready(repo_id)
    db = _get_db()

    try:
        chunks = db.load_chunks(repo_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    matches = [chunk for chunk in chunks if chunk.file_path == file and chunk.start_line <= line <= chunk.end_line]
    if not matches:
        raise HTTPException(status_code=404, detail=f"No chunk found in repository {repo_id!r} at {file}:{line}")

    matches.sort(key=lambda chunk: chunk.end_line - chunk.start_line)
    ranked = [
        RankedChunk(
            chunk_id=str(chunk.chunk_id), cross_encoder_score=1.0, previous_retrieval_score=1.0,
            final_rank=rank, retrieval_source=RetrievalSource.LOCATION,
        )
        for rank, chunk in enumerate(matches, start=1)
    ]

    try:
        document = ContextBuilder(db).build_context(repo_id, f"{file}:{line}", ranked)
    except RetrievalError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # `document.chunk_citations` (built by `ContextBuilder`, possibly after
    # small-to-big substitution) carries no chunk_type of its own - look it up
    # from the full per-repository chunk list so callers (Adjudicate's context
    # builder, Phase 24) can tell an AST function/class/method chunk (a real
    # graph node) apart from a SLIDING/PARENT window chunk (never a graph node -
    # see `models.schemas.ChunkType`'s own docstring) without guessing.
    chunk_type_by_id = {str(chunk.chunk_id): chunk.chunk_type.value for chunk in chunks}

    return ContextResponse(
        repository_id=document.repository_id,
        query=document.query,
        context=document.context,
        included_chunks=document.included_chunks,
        chunk_citations=[
            ChunkCitationResponse(
                chunk_id=citation.chunk_id, file_path=citation.file_path,
                function_name=citation.function_name, class_name=citation.class_name,
                chunk_type=chunk_type_by_id.get(citation.chunk_id),
            )
            for citation in document.chunk_citations
        ],
        matched_chunks=[
            ChunkCitationResponse(
                chunk_id=str(chunk.chunk_id), file_path=chunk.file_path,
                function_name=chunk.function_name, class_name=chunk.class_name,
                chunk_type=chunk.chunk_type.value,
            )
            for chunk in matches
        ],
        total_tokens=document.total_tokens,
        truncated=document.truncated,
    )


@app.get("/repos/{repo_id}/graph")
def get_graph(repo_id: str, focus_node: str | None = None, hops: int = 2) -> dict[str, Any]:
    """Return the call graph (nodes + edges) for an indexed repository.

    Reads back the exact JSON `database.graph_store.save_graph` wrote at
    index time (from `RepositoryGraphBuilder`'s original in-memory
    graph), via `database.graph_store.load_graph` - not
    `DatabaseManager.load_graph`, which reconstructs a lossy approximation
    from SQL (every stored chunk as a node, including ones never in the
    graph, and missing every edge that touches a file-level node - see
    `DatabaseManager.store_graph`'s docstring). Without `focus_node`, this
    endpoint must return the same node/edge counts `GET /status` reports,
    since both describe the same graph.

    Args:
        repo_id: The indexed repository to read the graph for.
        focus_node: If given, return only the blast-radius subgraph
            around this node id (see `graph.blast_radius.compute_blast_radius`)
            instead of the full graph - same node-link response shape,
            just filtered, so callers (Phase 22, e.g. Adjudicate flagging
            a specific node, or `ui/graph_view.py`'s manual trigger) need
            no response-parsing changes. Reuses the graph already loaded
            for this request - no second load.
        hops: Traversal depth in each direction when `focus_node` is
            given. Ignored otherwise.

    Raises:
        HTTPException: 404 if `focus_node` is not a node in this
            repository's graph; 400 if `hops` is negative.
    """
    state = _require_ready(repo_id)
    assert state.summary is not None  # noqa: S101 - guaranteed by _require_ready's "ready" check
    repository = state.summary.repository

    try:
        graph = graph_store.load_graph(path=_graph_json_path(repository.owner, repository.name))
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if focus_node is not None:
        try:
            graph = compute_blast_radius(graph, focus_node, hops)
        except RetrievalError as exc:
            status_code = 400 if "hops" in str(exc) else 404
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return nx.node_link_data(graph)


def _graph_json_path(owner: str, name: str) -> Path:
    """The per-repository graph JSON path `Pipeline.index_repository` writes to (see its `_graph_path`)."""
    return settings.GRAPH_DIR / f"{owner}_{name}.json"


@app.post("/repos/{repo_id}/review")
def review_repository(repo_id: str, body: ReviewRequest, http_request: Request) -> StreamingResponse:
    """Stream a live adversarial review of `body.diff` as Server-Sent Events (Phase 32 Part 2).

    Runs the real Phases 24-30 pipeline (Context Builder -> Defender ->
    Prosecutor -> Verifier -> Rebuttal -> Judge) via
    `adjudicate.orchestrator.live_review.run_live_review`, emitting one
    ``data: <json>\\n\\n`` line per stage as it completes - this module
    contributes no pipeline logic of its own, only the HTTP/SSE framing,
    matching this file's own "no ingestion/retrieval/generation logic
    of its own" convention (see this module's docstring).

    Unlike every other endpoint here, this one is deliberately `POST`
    despite streaming a response - the diff text can be arbitrarily long
    and doesn't belong in a query string, so the browser's native
    `EventSource` (GET-only) isn't usable; the frontend reads this via
    `fetch` + a manual `ReadableStream` reader instead (see
    `frontend/src/api/client.js`'s `streamReview`).

    Args:
        repo_id: The indexed repository to review the diff against -
            must already be indexed and ready (same requirement every
            other repo-scoped endpoint here has).
        body: The raw unified diff to review.
        http_request: FastAPI's own request object - only used for
            `.base_url`, passed to `run_live_review` so
            `AdjudicateContextBuilder` can call this same server back
            over a real HTTP request (see that function's own docstring
            for why this isn't an in-process ASGI shortcut).

    Raises:
        HTTPException: 404/409 if the repository isn't indexed/ready
            (via `_require_ready`, before the stream starts); 400 if
            `body.diff` is blank.
    """
    state = _require_ready(repo_id)
    assert state.summary is not None  # noqa: S101 - guaranteed by _require_ready's "ready" check
    if not body.diff.strip():
        raise HTTPException(status_code=400, detail="diff must not be blank.")
    local_path = state.summary.repository.local_path
    if local_path is None:
        raise HTTPException(
            status_code=500, detail=f"Repository {repo_id!r} has no local path recorded on this server."
        )
    base_url = str(http_request.base_url)

    def event_stream() -> Any:
        for event in run_live_review(repo_id, local_path, body.diff, base_url):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
