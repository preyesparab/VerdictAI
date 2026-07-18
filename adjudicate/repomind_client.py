"""Thin HTTP client for RepoMind's FastAPI backend, for Adjudicate (Phase 23).

Mirrors `ui/api_client.py`'s pattern (a thin `httpx`-backed wrapper that
translates non-2xx responses into `core.exceptions.RepoMindError`
subtypes) rather than inventing a second HTTP client convention. It is
deliberately narrow: Adjudicate only ever needs read access to an
already-indexed repository's context, graph, and (as a fallback -
Phase 24) retrieval, never indexing/chat, so `RepoMindClient` wraps
exactly four things — `GET /repos/{id}/context`, `GET /repos/{id}/graph`,
the blast-radius variant of that same endpoint (Phase 22's
`focus_node`/`hops` params), and `GET /repos/{id}/search` (Phase 24's
retrieval-only, no-generation endpoint) — and reimplements none of the
retrieval/graph logic behind them.

Unlike `ui/api_client.py`, this client accepts an injected `httpx.Client`
(not just a `base_url`) — this is what lets it be exercised in-process
against `api.main.app` via `httpx.ASGITransport` for verification,
without needing a live `uvicorn` process, the same dependency-injection
style already used by `generation.llm_client.LLMClient` for its provider
clients.
"""

from __future__ import annotations

from typing import Any

import httpx

from core.exceptions import RepoMindError, RepositoryCloneError, RetrievalError

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


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
        RepositoryCloneError: On 400 (bad request, e.g. negative `hops`).
        RetrievalError: On 404/409 (repository or node not found / not
            ready).
        RepoMindError: On any other non-2xx status.
    """
    if response.is_success:
        return
    detail = _error_detail(response)
    if response.status_code == 400:
        raise RepositoryCloneError(detail)
    if response.status_code in (404, 409):
        raise RetrievalError(detail)
    raise RepoMindError(detail)


class RepoMindClient:
    """Read-only HTTP client onto RepoMind's context/graph endpoints, for Adjudicate.

    Every method returns the API's JSON response body as-is (a plain
    `dict`), not a parsed model — the response schemas are defined
    locally inside `api/main.py` for request validation, not exported as
    a shared contract, so this client makes no assumption about their
    shape beyond what each method's docstring documents.
    """

    def __init__(self, base_url: str | None = None, http_client: httpx.Client | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: The API's base URL. Defaults to
                `DEFAULT_API_BASE_URL` (``http://127.0.0.1:8000``).
                Ignored if `http_client` is given.
            http_client: A pre-constructed `httpx.Client` to use instead
                of building one from `base_url` — lets callers (tests,
                or an in-process verification run) point this client at
                `api.main.app` directly via `httpx.ASGITransport`,
                without a live server process.
        """
        self._client = http_client or httpx.Client(base_url=base_url or DEFAULT_API_BASE_URL, timeout=60.0)

    def get_context(self, repo_id: str, file: str, line: int) -> dict[str, Any]:
        """Fetch `ContextBuilder` output for a specific file:line, via `GET /repos/{repo_id}/context`.

        Unlike a retrieval-based query, this looks up whichever stored
        chunk(s) already enclose `file:line` directly — the right call
        for Adjudicate, which already knows exactly which lines a diff
        touched.

        Args:
            repo_id: The indexed repository to look up.
            file: Repository-relative file path.
            line: 1-indexed line number within `file`.

        Returns:
            The raw JSON body: ``{"repository_id", "query", "context",
            "included_chunks", "chunk_citations", "matched_chunks",
            "total_tokens", "truncated"}``. ``matched_chunks`` (every raw
            match at this location, smallest-first, each tagged with its
            real ``chunk_type``) is what identifies the actual enclosing
            AST chunk - ``chunk_citations`` may have substituted it for
            a larger parent chunk (small-to-big) that is never itself a
            graph node.

        Raises:
            RetrievalError: If the repository is not indexed/ready, or
                no chunk encloses `file:line`.
            RepoMindError: If the API is unreachable.
        """
        try:
            response = self._client.get(f"/repos/{repo_id}/context", params={"file": file, "line": line})
        except httpx.ConnectError as exc:
            raise RepoMindError(f"Could not reach RepoMind API: {exc}") from exc
        _raise_for_status(response)
        return response.json()

    def get_graph(
        self, repo_id: str, focus_node: str | None = None, hops: int | None = None
    ) -> dict[str, Any]:
        """Fetch the call graph via `GET /repos/{repo_id}/graph`.

        Args:
            repo_id: The indexed repository to fetch the graph for.
            focus_node: If given, fetch only the blast-radius subgraph
                around this node id (Phase 22) instead of the full graph.
            hops: Traversal depth in each direction when `focus_node` is
                given (server defaults to 2 if omitted). Ignored
                otherwise.

        Returns:
            The node-link JSON `database.graph_store.save_graph` writes
            (``{"directed", "multigraph", "graph", "nodes", "edges"}``).

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
            raise RepoMindError(f"Could not reach RepoMind API: {exc}") from exc
        _raise_for_status(response)
        return response.json()

    def get_blast_radius(self, repo_id: str, focus_node: str, hops: int = 2) -> dict[str, Any]:
        """Fetch just `focus_node`'s N-hop neighborhood (Phase 22's blast radius).

        A thin, intention-revealing wrapper over `get_graph` — same
        endpoint, same response shape, no separate logic. Adjudicate's
        Verifier/Judge phases will call this once they know which node a
        Prosecutor claim implicates.

        Args:
            repo_id: The indexed repository to fetch the graph for.
            focus_node: The chunk or file node id to center the blast
                radius on.
            hops: Traversal depth in each direction. Defaults to 2,
                matching the server's own default.

        Returns:
            The induced subgraph's node-link JSON, same shape as
            `get_graph`.

        Raises:
            RetrievalError: If the repository is not indexed/ready, or
                `focus_node` is not a node in this repository's graph.
            RepoMindError: If the API is unreachable.
        """
        return self.get_graph(repo_id, focus_node=focus_node, hops=hops)

    def search(self, repo_id: str, query: str) -> list[dict[str, Any]]:
        """Run a targeted retrieval-only search via `GET /repos/{repo_id}/search`.

        Phase 24's fallback for relationships the graph can't capture -
        e.g. a test that exercises a function only through a mock/string
        reference or an HTTP-level test client, not a direct call edge.
        No LLM generation happens on the server for this call, unlike
        `POST /query`.

        Args:
            repo_id: The indexed repository to search.
            query: A search phrase - identifier-heavy phrases work best
                against the code embedding model in use.

        Returns:
            Up to the server's configured top-k results (each a raw
            citation dict: ``{"chunk_id", "file_path", "function_name",
            "chunk_type", "retrieval_source", "relevance_score",
            "raw_code"}``), ranked best-first.

        Raises:
            RetrievalError: If the repository is not indexed/ready.
            RepoMindError: If the API is unreachable.
        """
        try:
            response = self._client.get(f"/repos/{repo_id}/search", params={"q": query})
        except httpx.ConnectError as exc:
            raise RepoMindError(f"Could not reach RepoMind API: {exc}") from exc
        _raise_for_status(response)
        return response.json()["results"]
