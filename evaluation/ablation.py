"""Ablation study across retrieval/generation feature flags (Phase 17).

Runs the evaluation dataset through five fixed configurations, each one
progressively enabling more of `config.Settings`'s retrieval feature
flags (`USE_BM25`, `USE_GRAPH_EXPANSION`, `USE_RERANKER`,
`USE_SMALL_TO_BIG`):

1. Dense Retrieval Only
2. Dense + BM25
3. Dense + BM25 + Graph Expansion
4. Dense + BM25 + Graph + Cross Encoder
5. Full Pipeline (every flag on)

For each configuration, every `evaluation.eval_dataset.EvalCase` is run
through retrieval, context building, and generation exactly as a live
query would be, then graded with `evaluation.retrieval_eval` (retrieval
quality) and `evaluation.ragas_eval` (generation quality), plus wall-clock
latency. This is what turns "we added a reranker" into a number: whether
Retrieval Precision@5/Faithfulness actually improved enough to justify the
extra latency, instead of an assumption.

This module never constructs its own retrieval/generation components -
it is handed a `PipelineComponents` bundle by the caller (a script wiring
up real `FaissIndexManager`/`BM25Manager`/`GraphExpander`/etc., or a test
wiring up fakes), the same dependency-injection pattern every other phase
in this codebase uses.

`USE_SEMANTIC_CACHE` is deliberately not part of the ablation dimension:
a cache hit would silently reuse the *previous* configuration's answer
instead of generating a fresh one, corrupting per-configuration metrics.
`PipelineComponents.llm_service` should be wired with a cache that never
serves a hit during evaluation (writes to it are otherwise harmless).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from config import settings
from core.exceptions import EvaluationError
from core.logging import get_logger
from evaluation.eval_dataset import EvalCase
from evaluation.ragas_eval import RagasCase, evaluate_generation
from evaluation.retrieval_eval import (
    DEFAULT_TOP_K,
    EvaluatedChunkInput,
    RelevanceJudge,
    evaluate_retrieval,
)
from generation.context_builder import ContextBuilder
from models.schemas import (
    CodeChunk,
    ContextDocument,
    ExpandedRetrievedChunk,
    GeneratedAnswer,
    RankedChunk,
    RerankCandidate,
    RetrievalSource,
    RetrievedChunk,
)

logger = get_logger(__name__)

DEFAULT_ABLATION_RESULTS_PATH: Path = Path(__file__).resolve().parent / "ablation_results.json"


@dataclass(frozen=True)
class AblationConfig:
    """One ablation configuration - which retrieval/generation flags are on.

    Attributes:
        name: Human-readable configuration name, used as the row label
            in the comparison table/plots.
        use_bm25: Mirrors `settings.USE_BM25`.
        use_graph_expansion: Mirrors `settings.USE_GRAPH_EXPANSION`.
        use_reranker: Mirrors `settings.USE_RERANKER`.
        use_small_to_big: Mirrors `settings.USE_SMALL_TO_BIG`.
    """

    name: str
    use_bm25: bool
    use_graph_expansion: bool
    use_reranker: bool
    use_small_to_big: bool


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig("Dense Retrieval Only", use_bm25=False, use_graph_expansion=False, use_reranker=False, use_small_to_big=False),
    AblationConfig("Dense + BM25", use_bm25=True, use_graph_expansion=False, use_reranker=False, use_small_to_big=False),
    AblationConfig("Dense + BM25 + Graph Expansion", use_bm25=True, use_graph_expansion=True, use_reranker=False, use_small_to_big=False),
    AblationConfig("Dense + BM25 + Graph + Cross Encoder", use_bm25=True, use_graph_expansion=True, use_reranker=True, use_small_to_big=False),
    AblationConfig("Full Pipeline", use_bm25=True, use_graph_expansion=True, use_reranker=True, use_small_to_big=True),
]


@dataclass(frozen=True)
class AblationResult:
    """One configuration's aggregate metrics.

    Attributes:
        config_name: The `AblationConfig.name` this result is for.
        faithfulness: Mean RAGAS faithfulness across the dataset.
        answer_relevancy: Mean RAGAS answer relevancy across the dataset.
        context_precision: Mean RAGAS context precision across the dataset.
        retrieval_precision_at_5: Mean retrieval Precision@5 (LLM-as-judge)
            across the dataset.
        average_latency_ms: Mean end-to-end (retrieval + context + generation)
            latency per query, in milliseconds.
    """

    config_name: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    retrieval_precision_at_5: float
    average_latency_ms: float


@dataclass(frozen=True)
class AblationReport:
    """The full ablation study result.

    Attributes:
        results: One `AblationResult` per `AblationConfig` run, in the
            same order as `configs` was given to `run_ablation`.
    """

    results: list[AblationResult]


class DenseSearcher(Protocol):
    """The subset of `database.vector_store.FaissIndexManager` this module needs."""

    def search(self, query_embedding: Any, top_k: int) -> list[Any]:
        """Dense-search for the `top_k` chunks most similar to `query_embedding`."""
        ...


class HybridRetrieverLike(Protocol):
    """The subset of `retrieval.hybrid_retriever.HybridRetriever` this module needs."""

    def retrieve(self, repository_id: str, query: str, query_embedding: Any, top_k: int) -> list[RetrievedChunk]:
        """Retrieve and fuse dense + sparse results for one query."""
        ...


class GraphExpanderLike(Protocol):
    """The subset of `retrieval.graph_retriever.GraphExpander` this module needs."""

    def expand(self, repository_id: str, retrieved_chunks: list[RetrievedChunk]) -> list[ExpandedRetrievedChunk]:
        """Expand `retrieved_chunks` with their graph neighbors."""
        ...


class RerankerLike(Protocol):
    """The subset of `retrieval.reranker.CrossEncoderReranker` this module needs."""

    def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RankedChunk]:
        """Rerank `candidates` by relevance to `query`."""
        ...


class LLMServiceLike(Protocol):
    """The subset of `generation.answer_generator.LLMService` this module needs."""

    def generate_answer(self, repository_id: str, query: str, context_document: ContextDocument) -> GeneratedAnswer:
        """Generate a repository-aware answer from `context_document`."""
        ...


class ChunkStore(Protocol):
    """The subset of `database.sqlite_client.DatabaseManager` this module needs."""

    def load_chunks(self, repository_id: str) -> list[CodeChunk]:
        """Load every stored chunk for a repository."""
        ...


@dataclass
class PipelineComponents:
    """Every collaborator needed to run one `EvalCase` through the pipeline.

    Attributes:
        db: Loads chunk content for whichever chunk_ids retrieval/graph
            expansion returns.
        dense_searcher: Used directly (bypassing `hybrid_retriever`) when
            a configuration has `use_bm25=False`.
        hybrid_retriever: Used when a configuration has `use_bm25=True`.
        graph_expander: Used when a configuration has
            `use_graph_expansion=True`.
        reranker: Used when a configuration has `use_reranker=True`.
        llm_service: Generates the final answer from each configuration's
            assembled context. Must be wired with a semantic cache that
            never serves a hit during evaluation (see module docstring).
        judge: Grades retrieved chunks' relevance for
            `evaluation.retrieval_eval.evaluate_retrieval`.
        embed_query: Embeds a query string into a dense vector, with
            whatever embedding model the indexes were built from.
        ragas_llm: A RAGAS/LangChain-compatible LLM wrapper, passed
            through to `evaluation.ragas_eval.evaluate_generation`.
        ragas_embeddings: A RAGAS/LangChain-compatible embeddings
            wrapper, passed through to `evaluate_generation`.
        top_k: Chunks retained at each retrieval stage and graded for
            Precision@k. Defaults to
            `evaluation.retrieval_eval.DEFAULT_TOP_K` (5).
    """

    db: ChunkStore
    dense_searcher: DenseSearcher
    hybrid_retriever: HybridRetrieverLike
    graph_expander: GraphExpanderLike
    reranker: RerankerLike
    llm_service: LLMServiceLike
    judge: RelevanceJudge
    embed_query: Callable[[str], Any]
    ragas_llm: Any
    ragas_embeddings: Any
    top_k: int = DEFAULT_TOP_K


@contextmanager
def _apply_config(config: AblationConfig) -> Iterator[None]:
    """Temporarily set `settings`'s retrieval flags to match `config`.

    Restores the previous values on exit (including on an exception), so
    one ablation run never leaks configuration into the next or into the
    rest of the process.

    Args:
        config: The configuration whose flags to apply.

    Yields:
        Nothing; used only for its enter/exit side effects.
    """
    previous = (settings.USE_BM25, settings.USE_GRAPH_EXPANSION, settings.USE_RERANKER, settings.USE_SMALL_TO_BIG)
    settings.USE_BM25 = config.use_bm25
    settings.USE_GRAPH_EXPANSION = config.use_graph_expansion
    settings.USE_RERANKER = config.use_reranker
    settings.USE_SMALL_TO_BIG = config.use_small_to_big
    try:
        yield
    finally:
        settings.USE_BM25, settings.USE_GRAPH_EXPANSION, settings.USE_RERANKER, settings.USE_SMALL_TO_BIG = previous


def _retrieve(
    config: AblationConfig, case: EvalCase, components: PipelineComponents
) -> tuple[list[RankedChunk], dict[str, CodeChunk]]:
    """Run retrieval (dense/hybrid, optional graph expansion, optional rerank) for one case.

    Args:
        config: Which retrieval stages to run.
        case: The evaluation case to retrieve for.
        components: The pipeline's collaborators.

    Returns:
        A tuple of (ranked chunks, chunk_id -> CodeChunk map for the case's
        repository - loaded once and reused for context building).
    """
    query_embedding = components.embed_query(case.query)

    if config.use_bm25:
        retrieved = components.hybrid_retriever.retrieve(
            case.repository_id, case.query, query_embedding, components.top_k
        )
    else:
        dense_results = components.dense_searcher.search(query_embedding, components.top_k)
        retrieved = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                dense_score=result.score,
                bm25_score=None,
                fused_score=result.score,
                retrieval_source=RetrievalSource.DENSE,
                rank=rank,
            )
            for rank, result in enumerate(dense_results, start=1)
        ]

    if config.use_graph_expansion:
        expanded = components.graph_expander.expand(case.repository_id, retrieved)
    else:
        expanded = [
            ExpandedRetrievedChunk(
                chunk_id=item.chunk_id,
                retrieval_source=item.retrieval_source,
                score=item.fused_score,
                graph_distance=0,
                originating_chunk_id=None,
                edge_type=None,
            )
            for item in retrieved
        ]

    chunk_map = {str(chunk.chunk_id): chunk for chunk in components.db.load_chunks(case.repository_id)}
    candidates = [
        RerankCandidate(
            chunk_id=item.chunk_id,
            raw_code=chunk_map[item.chunk_id].raw_code,
            file_path=chunk_map[item.chunk_id].file_path,
            function_name=chunk_map[item.chunk_id].function_name,
            retrieval_source=item.retrieval_source,
            graph_distance=item.graph_distance,
            previous_score=item.score,
        )
        for item in expanded
        if item.chunk_id in chunk_map
    ]

    if config.use_reranker:
        ranked = components.reranker.rerank(case.query, candidates, components.top_k)
    else:
        ranked = [
            RankedChunk(
                chunk_id=candidate.chunk_id,
                cross_encoder_score=candidate.previous_score,
                previous_retrieval_score=candidate.previous_score,
                final_rank=rank,
                retrieval_source=candidate.retrieval_source,
            )
            for rank, candidate in enumerate(candidates[: components.top_k], start=1)
        ]

    return ranked, chunk_map


def _run_case(
    config: AblationConfig, case: EvalCase, components: PipelineComponents
) -> tuple[list[RankedChunk], dict[str, CodeChunk], ContextDocument, GeneratedAnswer, float]:
    """Run one `EvalCase` end-to-end (retrieval -> context -> generation) under `config`.

    Args:
        config: The configuration to run.
        case: The evaluation case to run.
        components: The pipeline's collaborators.

    Returns:
        A tuple of (ranked chunks, chunk map, assembled context, generated
        answer, end-to-end latency in milliseconds).

    Raises:
        EvaluationError: If any pipeline stage fails.
    """
    started_at = time.perf_counter()
    try:
        ranked, chunk_map = _retrieve(config, case, components)
        context_builder = ContextBuilder(components.db, use_small_to_big=config.use_small_to_big)
        context_document = context_builder.build_context(case.repository_id, case.query, ranked)
        answer = components.llm_service.generate_answer(case.repository_id, case.query, context_document)
    except EvaluationError:
        raise
    except Exception as exc:  # noqa: BLE001 - uniform evaluation-layer failure boundary
        raise EvaluationError(
            f"Pipeline run failed for query {case.query!r} under configuration {config.name!r}: {exc}"
        ) from exc

    latency_ms = (time.perf_counter() - started_at) * 1000
    return ranked, chunk_map, context_document, answer, latency_ms


def _run_configuration(
    config: AblationConfig, cases: list[EvalCase], components: PipelineComponents
) -> AblationResult:
    """Run every case through the pipeline under one configuration and aggregate metrics.

    Args:
        config: The configuration to run.
        cases: The evaluation dataset.
        components: The pipeline's collaborators.

    Returns:
        The configuration's aggregate `AblationResult`.

    Raises:
        EvaluationError: If any case's pipeline run, retrieval grading,
            or RAGAS evaluation fails.
    """
    logger.info("Ablation configuration started: %s", config.name)

    evaluated_chunks_by_query: dict[str, list[EvaluatedChunkInput]] = {}
    ragas_cases: list[RagasCase] = []
    latencies: list[float] = []

    with _apply_config(config):
        for case in cases:
            ranked, chunk_map, context_document, answer, latency_ms = _run_case(config, case, components)
            latencies.append(latency_ms)

            evaluated_chunks_by_query[case.query] = [
                EvaluatedChunkInput(
                    chunk_id=item.chunk_id,
                    file_path=chunk_map[item.chunk_id].file_path,
                    function_name=chunk_map[item.chunk_id].function_name,
                    raw_code=chunk_map[item.chunk_id].raw_code,
                )
                for item in ranked
                if item.chunk_id in chunk_map
            ]

            retrieved_contexts = [
                chunk_map[chunk_id].raw_code for chunk_id in context_document.included_chunks if chunk_id in chunk_map
            ]
            ragas_cases.append(
                RagasCase(
                    query=case.query,
                    answer=answer.answer,
                    retrieved_contexts=retrieved_contexts or [""],
                    reference=case.expected_answer,
                )
            )

        retrieval_report = evaluate_retrieval(
            cases,
            retrieve_fn=lambda case: evaluated_chunks_by_query[case.query],
            judge=components.judge,
            top_k=components.top_k,
        )
        ragas_report = evaluate_generation(ragas_cases, llm=components.ragas_llm, embeddings=components.ragas_embeddings)

    result = AblationResult(
        config_name=config.name,
        faithfulness=ragas_report.faithfulness,
        answer_relevancy=ragas_report.answer_relevancy,
        context_precision=ragas_report.context_precision,
        retrieval_precision_at_5=retrieval_report.mean_precision_at_k,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )

    logger.info(
        "Ablation configuration completed: %s (precision@5=%.2f, faithfulness=%.2f, avg_latency=%.1fms)",
        config.name, result.retrieval_precision_at_5, result.faithfulness, result.average_latency_ms,
    )
    return result


def run_ablation(
    cases: list[EvalCase],
    components: PipelineComponents,
    configs: list[AblationConfig] | None = None,
) -> AblationReport:
    """Run the ablation study: every configuration, over every case.

    Args:
        cases: The evaluation dataset to run each configuration against.
        components: The pipeline's collaborators (shared across every
            configuration - only the feature flags change between runs).
        configs: Configurations to run. Defaults to `ABLATION_CONFIGS`
            (the five fixed configurations Phase 17 specifies).

    Returns:
        The full ablation report, one result per configuration.

    Raises:
        EvaluationError: If `cases` is empty, or any configuration's run fails.
    """
    if not cases:
        raise EvaluationError("run_ablation requires at least one evaluation case")

    effective_configs = configs if configs is not None else ABLATION_CONFIGS
    logger.info("Ablation study started: %d configuration(s), %d case(s)", len(effective_configs), len(cases))

    results = [_run_configuration(config, cases, components) for config in effective_configs]

    logger.info("Ablation study completed: %d configuration(s) evaluated", len(results))
    return AblationReport(results=results)


def print_comparison_table(report: AblationReport) -> None:
    """Print a formatted comparison table of every configuration's metrics.

    Args:
        report: The ablation report to display.
    """
    headers = ("Configuration", "Faithfulness", "Answer Rel.", "Context Prec.", "Precision@5", "Avg Latency (ms)")
    rows = [
        (
            result.config_name,
            f"{result.faithfulness:.2f}",
            f"{result.answer_relevancy:.2f}",
            f"{result.context_precision:.2f}",
            f"{result.retrieval_precision_at_5:.2f}",
            f"{result.average_latency_ms:.1f}",
        )
        for result in report.results
    ]

    widths = [max(len(header), *(len(row[i]) for row in rows)) if rows else len(header) for i, header in enumerate(headers)]

    def _format_row(cells: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    separator = "-+-".join("-" * width for width in widths)

    print(_format_row(headers))
    print(separator)
    for row in rows:
        print(_format_row(row))


def save_ablation_results(report: AblationReport, path: Path | None = None) -> Path:
    """Persist `report` as JSON.

    Args:
        report: The ablation report to save.
        path: Destination file. Defaults to `DEFAULT_ABLATION_RESULTS_PATH`
            (``evaluation/ablation_results.json``).

    Returns:
        The path the report was written to.

    Raises:
        EvaluationError: If `path` cannot be written.
    """
    destination = path or DEFAULT_ABLATION_RESULTS_PATH
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to write ablation results to {destination}: {exc}") from exc

    logger.info("Ablation results saved to %s", destination)
    return destination
