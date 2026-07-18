"""Streamlit application entrypoint for RepoMind (Phase 18).

Thin entrypoint: this file only wires Streamlit widgets to
`ui.api_client.APIClient` (Phase 20) and renders results through
`ui.components` - it contains no retrieval/generation/ingestion logic of
its own, and no longer imports the pipeline directly: every call now
goes over HTTP to `api.main`'s FastAPI backend.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

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
from pipeline import AskResult
from ui.api_client import APIClient
from ui.components import render_chat_history, render_indexing_status, render_sidebar
from ui.graph_view import render_graph_tab

logger = get_logger(__name__)

st.set_page_config(page_title="RepoMind — GraphRAG for Repositories", page_icon="🧠", layout="wide")


@st.cache_resource
def get_pipeline_service() -> APIClient:
    """Construct (once per process) the `APIClient` every session shares.

    `st.cache_resource` is what makes this survive Streamlit's rerun-the-
    whole-script-on-every-interaction model - reuses one httpx connection
    pool (and its cached /info response) per session instead of opening a
    new one on every rerun. The embedding model, cross-encoder, and
    FAISS/BM25 indexes now live in the API process, not here.
    """
    return APIClient()


def init_session_state() -> None:
    """Initialize the cross-rerun state this app keeps in `st.session_state`."""
    st.session_state.setdefault("indexing_summary", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("cache_hits", 0)
    st.session_state.setdefault("cache_misses", 0)
    st.session_state.setdefault("retrieval_latencies_ms", [])


def _friendly_error_message(exc: Exception) -> str:
    """Map an exception to a user-facing message with no stack trace.

    Args:
        exc: The exception a pipeline call raised.

    Returns:
        A short, actionable message naming the failure category (never
        the raw exception's traceback).
    """
    if isinstance(exc, RepositoryCloneError):
        return f"Could not access this repository: {exc}"
    if isinstance(exc, ParsingError):
        return f"Failed to parse the repository's source files: {exc}"
    if isinstance(exc, EmbeddingError):
        return f"Failed to generate embeddings: {exc}"
    if isinstance(exc, RetrievalError):
        return f"Retrieval failed: {exc}"
    if isinstance(exc, LLMGenerationError):
        return f"The LLM failed to generate an answer: {exc}"
    if isinstance(exc, DatabaseError):
        return f"A storage error occurred: {exc}"
    if isinstance(exc, RepoMindError):
        return f"RepoMind encountered an error: {exc}"
    return "An unexpected error occurred. Please check your network connection and try again."


def handle_index_repository(service: APIClient, url: str) -> bool:
    """Index `url`, showing progress and updating session state, with errors displayed gracefully.

    Returns:
        True if indexing succeeded (safe to `st.rerun()` immediately);
        False if it was skipped or failed, in which case the warning/error
        already shown must survive this run - the caller must NOT rerun,
        or the message would be wiped before the user ever saw it.
    """
    if not url.strip():
        st.warning("Enter a repository URL first.")
        return False

    try:
        summary = render_indexing_status(service, url.strip())
    except Exception as exc:  # noqa: BLE001 - UI error boundary, see module docstring
        logger.error("Repository indexing failed for %r: %s", url, exc)
        st.error(_friendly_error_message(exc))
        return False

    st.session_state["indexing_summary"] = summary
    st.session_state["history"] = []
    logger.info("Repository indexed via UI: %s/%s", summary.repository.owner, summary.repository.name)
    st.success(f"Indexed {summary.repository.owner}/{summary.repository.name}: {summary.chunks_indexed} chunk(s).")
    return True


def handle_ask(service: APIClient, repository_id: str, query: str) -> bool:
    """Answer `query` against `repository_id`, updating session state, with errors displayed gracefully.

    Returns:
        True if generation succeeded (safe to `st.rerun()` immediately);
        False otherwise - see `handle_index_repository`'s docstring for why
        the caller must not rerun in that case.
    """
    if not query.strip():
        st.warning("Enter a question first.")
        return False

    logger.info("Query submitted: repository_id=%s query=%r", repository_id, query)
    try:
        with st.spinner("Retrieving context and generating an answer..."):
            result: AskResult = service.query(query.strip(), repository_id)
    except Exception as exc:  # noqa: BLE001 - UI error boundary, see module docstring
        logger.error("Query failed for %r: %s", query, exc)
        st.error(_friendly_error_message(exc))
        return False

    logger.info(
        "Answer generated: repository_id=%s latency_ms=%.1f cited_chunks=%d",
        repository_id, result.answer.latency_ms, len(result.citations),
    )

    if result.cache_hit:
        st.session_state["cache_hits"] += 1
    else:
        st.session_state["cache_misses"] += 1
        st.session_state["retrieval_latencies_ms"].append(result.retrieval_latency_ms)

    st.session_state["history"].append({"query": query.strip(), "result": result})
    return True


def main() -> None:
    """Render the RepoMind Streamlit application."""
    init_session_state()
    service = get_pipeline_service()

    st.title("🧠 RepoMind")
    st.caption("Ask natural-language questions about any public GitHub repository's code.")

    render_sidebar(service, st.session_state)

    with st.form("index_form"):
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
        index_submitted = st.form_submit_button("Index Repository")
    if index_submitted and handle_index_repository(service, repo_url):
        # The sidebar (repository info, feature toggles) already rendered
        # above using the pre-indexing session state this run - rerun so
        # it reflects the just-indexed repository immediately, the same
        # way handle_ask's result requires a rerun to reach the sidebar's
        # cache-hit-rate/latency stats. Only on success: a warning/error
        # `handle_index_repository` just displayed must survive this run,
        # not be wiped by an immediate rerun.
        st.rerun()

    st.divider()

    summary = st.session_state["indexing_summary"]
    if summary is None:
        st.info("Index a repository above to start asking questions.")
        return

    chat_tab, graph_tab = st.tabs(["💬 Chat", "🕸️ Graph"])

    with chat_tab:
        render_chat_history(st.session_state)

        with st.form("ask_form", clear_on_submit=True):
            query = st.text_input("Ask a question about this repository", placeholder="How does authentication work?")
            ask_submitted = st.form_submit_button("Ask RepoMind")
        if ask_submitted and handle_ask(service, summary.repository_id, query):
            st.rerun()

    with graph_tab:
        render_graph_tab(service, summary.repository_id)


if __name__ == "__main__":
    main()
