"""HTTP client for `api.main`'s FastAPI backend (Phase 20), for the Streamlit UI.

Exposes the same `index_repository`/`query` interface `pipeline.Pipeline`
does, so `ui/components.py`'s rendering functions need no logic changes
to switch from an in-process `Pipeline` to this HTTP-backed client -
`app.py` only needs to construct an `APIClient` where it used to
construct a `Pipeline`. Indexing is polled to completion here (via
repeated `GET /repos/{repo_id}/status` calls) so callers still see the
same synchronous, progress-reporting `index_repository` call they did
before the API existed; the polling, not blocking, happens over HTTP now
instead of in-process.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from core.exceptions import RepoMindError, RepositoryCloneError, RetrievalError
from ingestion.repository_metadata import RepositoryMetadata
from models.schemas import GeneratedAnswer
from pipeline import AskResult, CitationDisplay, IndexingSummary, ProgressCallback

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 1.0


def _error_detail(response: httpx.Response) -> str:
    """Extract a human-readable error message from a non-2xx JSON response."""
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def _raise_for_status(response: httpx.Response) -> None:
    """Translate a non-2xx API response into the matching `RepoMindError` subtype.

    Args:
        response: The API response to check.

    Raises:
        RepositoryCloneError: On 400 (invalid repository URL).
        RetrievalError: On 404/409 (repository not found/not ready) or
            504 (query timeout).
        RepoMindError: On any other non-2xx status.
    """
    if response.is_success:
        return
    detail = _error_detail(response)
    if response.status_code == 400:
        raise RepositoryCloneError(detail)
    if response.status_code == 504:
        raise RetrievalError(f"Request timed out: {detail}")
    if response.status_code in (404, 409):
        raise RetrievalError(detail)
    raise RepoMindError(detail)


def _summary_from_status(repo_id: str, payload: dict) -> IndexingSummary:
    """Build an `IndexingSummary` from a `GET /repos/{repo_id}/status` response body."""
    return IndexingSummary(
        repository=RepositoryMetadata(
            name=payload["name"],
            owner=payload["owner"],
            clone_url=f"https://github.com/{payload['owner']}/{payload['name']}.git",
            default_branch="",
            local_path=None,
            last_updated=None,
            commit_hash=payload["commit_hash"],
        ),
        repository_id=repo_id,
        files_discovered=payload["files_discovered"],
        chunks_indexed=payload["chunks_indexed"],
        graph_nodes=payload["graph_nodes"],
        graph_edges=payload["graph_edges"],
        embedded_chunks=payload["embedded_chunks"],
    )


class APIClient:
    """Talks to the RepoMind FastAPI backend over HTTP.

    Mirrors `pipeline.Pipeline`'s public interface used by the UI
    (`index_repository`, `query`), so `ui/components.py` treats an
    instance of this class exactly like a `Pipeline`.
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: The API's base URL. Defaults to
                `DEFAULT_API_BASE_URL` (``http://127.0.0.1:8000``).
        """
        self._base_url = base_url or DEFAULT_API_BASE_URL
        self._client = httpx.Client(base_url=self._base_url, timeout=150.0)
        self._info: dict[str, str] | None = None

    def index_repository(self, url: str, on_progress: ProgressCallback | None = None) -> IndexingSummary:
        """Trigger indexing via `POST /repos/index`, polling `GET /status` until ready.

        Args:
            url: A GitHub repository URL.
            on_progress: Called as `on_progress(stage_name, status)`,
                approximating the same callback contract
                `Pipeline.index_repository` uses - one "running" call
                whenever the reported stage changes, one final
                "complete" call, or one "error" call on failure.

        Returns:
            A summary of what was indexed.

        Raises:
            RepositoryCloneError: If the URL is invalid.
            RepoMindError: If indexing fails, or the API is unreachable.
        """
        try:
            response = self._client.post("/repos/index", json={"repo_url": url})
        except httpx.ConnectError as exc:
            raise RepoMindError(f"Could not reach RepoMind API at {self._base_url}: {exc}") from exc
        _raise_for_status(response)
        repo_id = response.json()["repo_id"]

        last_stage: str | None = None
        while True:
            status_response = self._client.get(f"/repos/{repo_id}/status")
            _raise_for_status(status_response)
            payload = status_response.json()

            stage = payload.get("stage")
            if stage is not None and stage != last_stage and on_progress is not None:
                on_progress(stage, "running")
            last_stage = stage

            if payload["status"] == "ready":
                if on_progress is not None and last_stage is not None:
                    on_progress(last_stage, "complete")
                return _summary_from_status(repo_id, payload)
            if payload["status"] == "failed":
                if on_progress is not None:
                    on_progress("error", "error")
                raise RepoMindError(payload.get("error") or "Indexing failed")

            time.sleep(POLL_INTERVAL_SECONDS)

    def query(self, question: str, repo_id: str) -> AskResult:
        """Answer a question via `POST /repos/{repo_id}/query`.

        Args:
            question: The user's natural-language question.
            repo_id: The repository to query - must already be indexed.

        Returns:
            The generated answer plus retrieval/citation detail.

        Raises:
            RetrievalError: If the repository is not indexed/ready, or
                the query times out.
            RepoMindError: If generation fails, or the API is unreachable.
        """
        try:
            response = self._client.post(f"/repos/{repo_id}/query", json={"question": question})
        except httpx.ConnectError as exc:
            raise RepoMindError(f"Could not reach RepoMind API at {self._base_url}: {exc}") from exc
        _raise_for_status(response)
        payload = response.json()

        return AskResult(
            answer=GeneratedAnswer(
                answer=payload["answer"],
                cited_chunks=[citation["chunk_id"] for citation in payload["citations"]],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model_name=payload["model"],
                latency_ms=payload["latency_ms"],
            ),
            citations=[CitationDisplay(**citation) for citation in payload["citations"]],
            retrieved_count=payload["retrieved_count"],
            graph_expanded_count=payload["graph_expanded_count"],
            cache_hit=payload["cache_hit"],
            llm_model=payload["model"],
            retrieval_latency_ms=0.0,
        )

    def get_graph(
        self, repo_id: str, focus_node: str | None = None, hops: int | None = None
    ) -> dict[str, Any]:
        """Fetch the call graph via `GET /repos/{repo_id}/graph`.

        Args:
            repo_id: The repository to fetch the graph for - must
                already be indexed.
            focus_node: If given, fetch only the blast-radius subgraph
                around this node id (Phase 22) instead of the full graph.
            hops: Traversal depth in each direction when `focus_node` is
                given (server defaults to 2 if omitted). Ignored otherwise.

        Returns:
            The node-link JSON `database.graph_store.save_graph` already
            writes (``{"directed", "multigraph", "graph", "nodes", "edges"}``),
            unchanged - this method is a thin HTTP wrapper, not a new
            data shape. Filtered down to the blast-radius subgraph when
            `focus_node` is given, same shape either way.

        Raises:
            RetrievalError: If the repository is not indexed/ready, or
                `focus_node` is not a node in this repository's graph.
            RepoMindError: If the API is unreachable.
        """
        params: dict[str, Any] = {}
        if focus_node is not None:
            params["focus_node"] = focus_node
            if hops is not None:
                params["hops"] = hops

        try:
            response = self._client.get(f"/repos/{repo_id}/graph", params=params)
        except httpx.ConnectError as exc:
            raise RepoMindError(f"Could not reach RepoMind API at {self._base_url}: {exc}") from exc
        _raise_for_status(response)
        return response.json()

    def _get_info(self) -> dict[str, str]:
        """Fetch (and cache for this instance's lifetime) `GET /info`'s response."""
        if self._info is None:
            try:
                response = self._client.get("/info")
            except httpx.ConnectError as exc:
                raise RepoMindError(f"Could not reach RepoMind API at {self._base_url}: {exc}") from exc
            _raise_for_status(response)
            self._info = response.json()
        return self._info

    def embedding_model_name(self) -> str:
        """The currently active embedding model name."""
        return self._get_info()["embedding_model"]

    def llm_model_name(self) -> str:
        """The currently active LLM's provider and model name."""
        return self._get_info()["llm_model"]
