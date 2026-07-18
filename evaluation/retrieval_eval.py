"""Retrieval quality evaluation via LLM-as-judge (Phase 17).

For every `evaluation.eval_dataset.EvalCase`, a caller-supplied
`retrieve_fn` runs the live retrieval pipeline (however it is currently
configured - dense only, hybrid, graph-expanded, reranked, ...) and hands
back its top-k chunks. This module never runs retrieval itself: it only
grades what it is given, using an LLM as a relevance judge (1 = not
relevant, 5 = highly relevant), and computes Precision@k (the fraction of
the top-k judged >= a relevance threshold) and the mean relevance score.

Decoupling "how retrieval is run" from "how retrieval is graded" is what
lets `evaluation.ablation` reuse this exact grading logic across every
ablation configuration instead of five near-duplicate implementations.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from core.exceptions import EvaluationError
from core.logging import get_logger
from evaluation.eval_dataset import EvalCase
from evaluation.metrics import mean, precision_at_k
from generation.llm_client import LLMCompletion

logger = get_logger(__name__)

DEFAULT_RETRIEVAL_SCORES_PATH: Path = Path(__file__).resolve().parent / "retrieval_scores.json"

DEFAULT_TOP_K: int = 5
DEFAULT_RELEVANT_THRESHOLD: int = 4
MIN_SCORE: int = 1
MAX_SCORE: int = 5

_JUDGE_SYSTEM_PROMPT: str = (
    "You are grading search result relevance for a code retrieval system. "
    "Given a user query and one retrieved code snippet, rate how relevant the snippet "
    "is to answering the query on a scale of 1 to 5, where 1 means 'not relevant at all' "
    "and 5 means 'highly relevant, directly answers the query'. "
    "Respond with ONLY the single integer score - no explanation, no other text."
)

_SCORE_PATTERN = re.compile(r"[1-5]")


@dataclass(frozen=True)
class EvaluatedChunkInput:
    """One retrieved chunk, ready to be judged for relevance.

    Deliberately minimal and decoupled from `models.schemas.CodeChunk`/
    `RankedChunk`: `retrieve_fn` (supplied by the caller - a live pipeline
    script or `evaluation.ablation`) is responsible for building these
    from whatever concrete retrieval pipeline it ran, in best-first order.

    Attributes:
        chunk_id: The chunk's id.
        file_path: Repository-relative path of the chunk's file.
        function_name: The chunk's function/method name, or None.
        raw_code: The chunk's source text - what the judge actually reads.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    raw_code: str


@dataclass(frozen=True)
class ChunkRelevanceScore:
    """One judged chunk's relevance grade.

    Attributes:
        chunk_id: The graded chunk's id.
        file_path: The graded chunk's file path (for readability in the
            saved results, without needing to join back to the chunk store).
        function_name: The graded chunk's function/method name, or None.
        relevance_score: The judge's 1-5 relevance grade.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    relevance_score: int


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Retrieval evaluation result for one `EvalCase`.

    Attributes:
        repository_id: The repository the query was run against.
        query: The evaluation query.
        chunk_scores: Every retrieved chunk's relevance grade, in the
            order `retrieve_fn` returned them.
        precision_at_k: Fraction of `chunk_scores` graded
            `>= relevant_threshold`.
        mean_relevance_score: Mean of `chunk_scores`' relevance grades.
    """

    repository_id: str
    query: str
    chunk_scores: list[ChunkRelevanceScore]
    precision_at_k: float
    mean_relevance_score: float


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval evaluation result across an entire dataset.

    Attributes:
        per_query: One `RetrievalEvalResult` per evaluated `EvalCase`.
        mean_precision_at_k: Mean of every result's `precision_at_k`.
        mean_relevance_score: Mean of every result's `mean_relevance_score`.
    """

    per_query: list[RetrievalEvalResult]
    mean_precision_at_k: float
    mean_relevance_score: float


class LLMClientLike(Protocol):
    """The subset of `generation.llm_client.LLMClient` this module needs."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        """Complete a (system, user) prompt pair through the configured provider."""
        ...


class RelevanceJudge(Protocol):
    """Grades one (query, chunk) pair's relevance."""

    def score(self, query: str, chunk_code: str) -> int:
        """Return a 1-5 relevance grade for `chunk_code` against `query`."""
        ...


class LLMRelevanceJudge:
    """Grades chunk relevance by asking an LLM for a 1-5 score."""

    def __init__(self, llm_client: LLMClientLike) -> None:
        """Initialize the judge.

        Args:
            llm_client: The LLM to grade with - any
                `generation.llm_client.LLMClient`-shaped object (real or
                a test fake).
        """
        self._llm_client = llm_client

    def score(self, query: str, chunk_code: str) -> int:
        """Grade `chunk_code`'s relevance to `query` on a 1-5 scale.

        Args:
            query: The evaluation query.
            chunk_code: The retrieved chunk's source text.

        Returns:
            An integer in `[MIN_SCORE, MAX_SCORE]`. If the LLM's response
            does not contain a parseable digit in that range, defaults to
            `MIN_SCORE` (an unparseable judgment is treated as
            conservatively as a "not relevant" one, rather than raising
            or assuming maximum relevance).

        Raises:
            EvaluationError: If the underlying LLM call fails.
        """
        user_prompt = f"Query: {query}\n\nRetrieved code:\n{chunk_code}\n\nRelevance score (1-5):"
        try:
            completion = self._llm_client.complete(_JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001 - uniform evaluation-layer failure boundary
            raise EvaluationError(f"Relevance judge LLM call failed: {exc}") from exc

        match = _SCORE_PATTERN.search(completion.text)
        if match is None:
            logger.warning("Judge response not parseable as a 1-5 score: %r; defaulting to %d", completion.text, MIN_SCORE)
            return MIN_SCORE

        return max(MIN_SCORE, min(MAX_SCORE, int(match.group())))


def evaluate_retrieval(
    cases: list[EvalCase],
    retrieve_fn: Callable[[EvalCase], list[EvaluatedChunkInput]],
    judge: RelevanceJudge,
    top_k: int = DEFAULT_TOP_K,
    relevant_threshold: int = DEFAULT_RELEVANT_THRESHOLD,
) -> RetrievalEvaluationReport:
    """Evaluate retrieval quality over `cases` using an LLM-as-judge.

    Args:
        cases: The evaluation dataset (or a subset of it) to run.
        retrieve_fn: Runs the live retrieval pipeline for one `EvalCase`
            and returns its top-k chunks, best-first. Not called more
            than once per case.
        judge: Grades each returned chunk's relevance to its query.
        top_k: How many of `retrieve_fn`'s returned chunks to grade and
            compute `precision_at_k` over.
        relevant_threshold: Minimum relevance grade, inclusive, to count
            as "relevant" for `precision_at_k`.

    Returns:
        The evaluation report, with one result per case plus dataset-wide
        means.

    Raises:
        EvaluationError: If `retrieve_fn` or `judge` fails for any case.
    """
    logger.info("Retrieval evaluation started: %d case(s)", len(cases))

    results: list[RetrievalEvalResult] = []
    for case in cases:
        try:
            retrieved = retrieve_fn(case)
        except EvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001 - uniform evaluation-layer failure boundary
            raise EvaluationError(f"Retrieval failed for query {case.query!r}: {exc}") from exc

        top_chunks = retrieved[:top_k]
        chunk_scores = [
            ChunkRelevanceScore(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                function_name=chunk.function_name,
                relevance_score=judge.score(case.query, chunk.raw_code),
            )
            for chunk in top_chunks
        ]

        scores = [chunk.relevance_score for chunk in chunk_scores]
        result = RetrievalEvalResult(
            repository_id=case.repository_id,
            query=case.query,
            chunk_scores=chunk_scores,
            precision_at_k=precision_at_k(scores, top_k, relevant_threshold),
            mean_relevance_score=mean([float(score) for score in scores]),
        )
        results.append(result)
        logger.info(
            "Retrieval evaluated for query %r: precision@%d=%.2f, mean_relevance=%.2f",
            case.query, top_k, result.precision_at_k, result.mean_relevance_score,
        )

    report = RetrievalEvaluationReport(
        per_query=results,
        mean_precision_at_k=mean([result.precision_at_k for result in results]),
        mean_relevance_score=mean([result.mean_relevance_score for result in results]),
    )

    logger.info(
        "Retrieval evaluation completed: mean precision@%d=%.2f, mean relevance=%.2f across %d case(s)",
        top_k, report.mean_precision_at_k, report.mean_relevance_score, len(results),
    )
    return report


def _report_to_dict(report: RetrievalEvaluationReport) -> dict[str, Any]:
    """Convert `report` into a plain JSON-serializable dict."""
    return asdict(report)


def save_retrieval_scores(report: RetrievalEvaluationReport, path: Path | None = None) -> Path:
    """Persist `report` as JSON.

    Args:
        report: The evaluation report to save.
        path: Destination file. Defaults to `DEFAULT_RETRIEVAL_SCORES_PATH`
            (``evaluation/retrieval_scores.json``).

    Returns:
        The path the report was written to.

    Raises:
        EvaluationError: If `path` cannot be written.
    """
    destination = path or DEFAULT_RETRIEVAL_SCORES_PATH
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to write retrieval scores to {destination}: {exc}") from exc

    logger.info("Retrieval scores saved to %s", destination)
    return destination
