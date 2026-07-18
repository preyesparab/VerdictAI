"""Reusable Streamlit rendering functions for `app.py` (Phase 18).

Kept separate from `app.py` so the main entrypoint stays a thin
orchestrator (mirroring this project's "thin entrypoint" convention) and
each rendered section - sidebar, answer, citations, indexing progress,
chat history - can be read and modified independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from config import settings
from pipeline import AskResult, CitationDisplay, IndexingSummary
from ui.api_client import APIClient


def render_sidebar(service: APIClient, session: dict[str, Any]) -> None:
    """Render the sidebar: repository info, model config, cache/latency stats, feature toggles, eval metrics.

    Args:
        service: The pipeline service backing the current session.
        session: `st.session_state`-backed dict this app stores its
            cross-rerun state in (see `app.py`'s `init_session_state`).
    """
    with st.sidebar:
        st.header("Repository")
        summary: IndexingSummary | None = session.get("indexing_summary")
        if summary is None:
            st.caption("No repository indexed yet.")
        else:
            st.write(f"**{summary.repository.owner}/{summary.repository.name}**")
            st.caption(f"Commit `{summary.repository.commit_hash[:8]}`")
            st.metric("Files", summary.files_discovered)
            st.metric("Chunks", summary.chunks_indexed)
            st.metric("Graph nodes / edges", f"{summary.graph_nodes} / {summary.graph_edges}")

        st.divider()
        st.header("Models")
        st.write(f"**Embedding model:** {service.embedding_model_name()}")
        st.write(f"**LLM:** {service.llm_model_name()}")

        st.divider()
        st.header("Session stats")
        total_queries = session["cache_hits"] + session["cache_misses"]
        hit_rate = (session["cache_hits"] / total_queries) if total_queries else 0.0
        st.metric("Semantic cache hit rate", f"{hit_rate:.0%}", help=f"{session['cache_hits']}/{total_queries} queries")
        avg_latency = sum(session["retrieval_latencies_ms"]) / len(session["retrieval_latencies_ms"]) if session["retrieval_latencies_ms"] else 0.0
        st.metric("Avg. retrieval latency", f"{avg_latency:.0f} ms")

        st.divider()
        st.header("Feature toggles")
        st.caption("Changes apply to the next query immediately.")
        settings.USE_BM25 = st.toggle("BM25 (sparse retrieval)", value=settings.USE_BM25, key="toggle_bm25")
        settings.USE_GRAPH_EXPANSION = st.toggle("Graph Expansion", value=settings.USE_GRAPH_EXPANSION, key="toggle_graph")
        settings.USE_RERANKER = st.toggle("Cross-Encoder Reranking", value=settings.USE_RERANKER, key="toggle_reranker")
        settings.USE_SEMANTIC_CACHE = st.toggle("Semantic Cache", value=settings.USE_SEMANTIC_CACHE, key="toggle_cache")
        settings.USE_SMALL_TO_BIG = st.toggle("Small-to-Big Retrieval", value=settings.USE_SMALL_TO_BIG, key="toggle_small_to_big")

        st.divider()
        render_evaluation_metrics()


def render_evaluation_metrics() -> None:
    """Render evaluation metrics loaded from `evaluation/`'s saved result files, if present."""
    st.header("Evaluation metrics")

    # `evaluation/*.json` (Phase 17's saved reports) live alongside the
    # evaluation package itself, not under settings.EVALUATION_DIR (a
    # data/ output directory reserved for a different purpose) - resolve
    # directly against this project's evaluation/ folder.
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    ablation_file = eval_dir / "ablation_results.json"
    retrieval_file = eval_dir / "retrieval_scores.json"

    if not ablation_file.exists() and not retrieval_file.exists():
        st.caption("No evaluation results found. Run `evaluation/ablation.py` to generate them.")
        return

    if ablation_file.exists():
        try:
            data = json.loads(ablation_file.read_text(encoding="utf-8"))
            results = data.get("results", [])
            if results:
                best = max(results, key=lambda row: row.get("retrieval_precision_at_5", 0.0))
                st.write(f"**Best configuration:** {best.get('config_name', 'unknown')}")
                st.metric("Faithfulness", f"{best.get('faithfulness', 0.0):.2f}")
                st.metric("Retrieval Precision@5", f"{best.get('retrieval_precision_at_5', 0.0):.2f}")
                st.metric("Avg. latency", f"{best.get('average_latency_ms', 0.0):.0f} ms")
        except (OSError, ValueError, json.JSONDecodeError):
            st.caption("Could not read ablation_results.json.")

    if retrieval_file.exists():
        try:
            data = json.loads(retrieval_file.read_text(encoding="utf-8"))
            st.metric("Mean relevance score", f"{data.get('mean_relevance_score', 0.0):.2f}")
        except (OSError, ValueError, json.JSONDecodeError):
            st.caption("Could not read retrieval_scores.json.")


def render_indexing_status(service: APIClient, url: str) -> IndexingSummary | None:
    """Run `service.index_repository`, rendering live progress via `st.status`.

    Args:
        service: The pipeline service to run indexing through.
        url: The repository URL to index.

    Returns:
        The resulting `IndexingSummary`, or None if indexing failed (the
        error is already displayed to the user - see `app.py`'s
        exception handling around this call).
    """
    with st.status("Indexing repository...", expanded=True) as status:
        def _on_progress(stage: str, state: str) -> None:
            if state == "running":
                st.write(f"⏳ {stage}...")
            elif state == "complete":
                st.write(f"✅ {stage}")

        summary = service.index_repository(url, on_progress=_on_progress)
        status.update(label=f"Indexed {summary.repository.owner}/{summary.repository.name}", state="complete")
        return summary


def render_answer(result: AskResult) -> None:
    """Render a generated answer's text and headline metrics.

    Args:
        result: The query result to display.
    """
    st.markdown(result.answer.answer)

    columns = st.columns(4)
    columns[0].metric("Latency", f"{result.answer.latency_ms:.0f} ms")
    columns[1].metric("LLM used", result.llm_model)
    columns[2].metric("Chunks retrieved", result.retrieved_count)
    columns[3].metric("Graph-expanded chunks", result.graph_expanded_count)

    if result.cache_hit:
        st.info("Served from the semantic cache - retrieval and generation were skipped.")


def render_citations(citations: list[CitationDisplay]) -> None:
    """Render every cited chunk's metadata and an expandable code section.

    Args:
        citations: The citations to display (an answer's
            `AskResult.citations`).
    """
    if not citations:
        st.caption("No citations - the repository did not contain enough information for this answer.")
        return

    st.subheader("Source citations")
    for citation in citations:
        label = f"`{citation.file_path}` — {citation.function_name or citation.chunk_type}"
        with st.expander(label):
            columns = st.columns(4)
            columns[0].write(f"**File path**\n\n`{citation.file_path}`")
            columns[1].write(f"**Function**\n\n{citation.function_name or '—'}")
            columns[2].write(f"**Chunk type**\n\n{citation.chunk_type}")
            columns[3].write(f"**Retrieval source**\n\n{citation.retrieval_source}")
            st.write(f"**Relevance score:** {citation.relevance_score:.3f}")
            st.code(citation.raw_code, language="python")


def render_chat_history(session: dict[str, Any]) -> None:
    """Render every past (query, answer) turn as a chat transcript.

    Args:
        session: `st.session_state`-backed dict holding `"history"` - a
            list of ``{"query": str, "result": AskResult}`` entries, in
            chronological order.
    """
    for turn in session["history"]:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            render_answer(turn["result"])
            render_citations(turn["result"].citations)
