"""Generation quality evaluation via RAGAS (Phase 17).

Wraps the `ragas` library's `evaluate()` to score Faithfulness (does the
answer only assert what the retrieved context supports), Answer Relevancy
(does the answer actually address the query), Context Precision (how much
of the retrieved context was actually useful), and Context Recall (how
much of what the reference answer needed was present in the retrieved
context).

`ragas` itself is imported lazily, only inside `_run_ragas_evaluate` - not
at module import time - for two reasons:

1. It is a heavy, optional dependency; every other evaluation module
   (`retrieval_eval`, `ablation`) and this module's own dataset-building/
   error-wrapping logic must stay importable and unit-testable (via a
   monkeypatched `_run_ragas_evaluate`) without it installed.
2. At the time this phase was implemented, the latest `ragas` release
   (0.4.3) failed to import at all against the environment's installed
   `langchain-community` (`from langchain_community.chat_models.vertexai
   import ChatVertexAI` - a module langchain-community has since removed).
   Older `ragas` releases (e.g. 0.1.x) require correspondingly older,
   now-conflicting `langchain`/`langchain-community` pins. Rather than
   force a fragile, version-locked langchain stack onto every environment
   that imports this project's evaluation layer, real RAGAS scoring is an
   opt-in installed in whichever environment actually runs it (see
   `requirements.txt`'s comment on this).

`llm`/`embeddings` are passed through untouched to `ragas.evaluate` - they
must already be RAGAS/LangChain-compatible wrapper objects (e.g.
`ragas.llms.LangchainLLMWrapper`); constructing one is the caller's
responsibility, not this module's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.exceptions import EvaluationError
from core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RAGAS_RESULTS_PATH: Path = Path(__file__).resolve().parent / "ragas_results.json"

METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class RagasCase:
    """One (query, generated answer, retrieved context) triple to score.

    Attributes:
        query: The evaluation query.
        answer: The pipeline's generated answer
            (`models.schemas.GeneratedAnswer.answer`).
        retrieved_contexts: The raw code text of every chunk the answer
            was generated from (e.g. from
            `models.schemas.ContextDocument.chunk_citations`/`context`).
        reference: The human-written expected answer
            (`evaluation.eval_dataset.EvalCase.expected_answer`) - RAGAS's
            ground truth for Context Recall.
    """

    query: str
    answer: str
    retrieved_contexts: list[str]
    reference: str


@dataclass(frozen=True)
class RagasCaseResult:
    """One `RagasCase`'s per-metric scores.

    Attributes:
        query: The evaluation query.
        faithfulness: 0-1; how much of `answer` is supported by
            `retrieved_contexts`.
        answer_relevancy: 0-1; how relevant `answer` is to `query`.
        context_precision: 0-1; how much of `retrieved_contexts` was
            actually relevant.
        context_recall: 0-1; how much of `reference` is covered by
            `retrieved_contexts`.
    """

    query: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


@dataclass(frozen=True)
class RagasReport:
    """Aggregate RAGAS evaluation result across a dataset.

    Attributes:
        per_case: One `RagasCaseResult` per evaluated `RagasCase`.
        faithfulness: Mean faithfulness across `per_case`.
        answer_relevancy: Mean answer relevancy across `per_case`.
        context_precision: Mean context precision across `per_case`.
        context_recall: Mean context recall across `per_case`.
    """

    per_case: list[RagasCaseResult]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _case_to_row(case: RagasCase) -> dict[str, Any]:
    """Convert one `RagasCase` into a RAGAS 0.1.x-schema dataset row.

    Args:
        case: The case to convert.

    Returns:
        A row with RAGAS's ``question``/``answer``/``contexts``/
        ``ground_truth`` column names.
    """
    return {
        "question": case.query,
        "answer": case.answer,
        "contexts": case.retrieved_contexts,
        "ground_truth": case.reference,
    }


def _run_ragas_evaluate(rows: list[dict[str, Any]], llm: Any, embeddings: Any) -> list[dict[str, float]]:
    """Run the real `ragas.evaluate` call - the module's sole `ragas` import site.

    Args:
        rows: Dataset rows built by `_case_to_row`, one per `RagasCase`.
        llm: A RAGAS/LangChain-compatible LLM wrapper.
        embeddings: A RAGAS/LangChain-compatible embeddings wrapper.

    Returns:
        One dict of `METRIC_NAMES` -> score per row, same order as `rows`.

    Raises:
        EvaluationError: If `ragas`/`datasets` cannot be imported, or the
            evaluation call itself fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise EvaluationError(
            "ragas is not installed (or not compatible with this environment's langchain "
            f"packages) - see this module's docstring: {exc}"
        ) from exc

    dataset = Dataset.from_list(rows)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as exc:  # noqa: BLE001 - third-party evaluation boundary, see module docstring
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

    per_row = result.to_pandas()
    return [
        {metric: float(per_row.loc[i, metric]) for metric in METRIC_NAMES} for i in range(len(per_row))
    ]


def evaluate_generation(cases: list[RagasCase], llm: Any, embeddings: Any) -> RagasReport:
    """Score `cases` with RAGAS's Faithfulness/Answer Relevancy/Context Precision/Context Recall.

    Args:
        cases: The (query, answer, retrieved context, reference) triples
            to score - typically one per `evaluation.eval_dataset.EvalCase`
            that the live pipeline was actually run against.
        llm: A RAGAS/LangChain-compatible LLM wrapper used to judge
            faithfulness/relevancy.
        embeddings: A RAGAS/LangChain-compatible embeddings wrapper used
            for context precision/recall.

    Returns:
        The RAGAS report, with one result per case plus dataset-wide means.

    Raises:
        EvaluationError: If `cases` is empty, or the underlying RAGAS
            call fails (including `ragas` not being importable - see
            module docstring).
    """
    logger.info("RAGAS evaluation started: %d case(s)", len(cases))

    if not cases:
        raise EvaluationError("evaluate_generation requires at least one case")

    rows = [_case_to_row(case) for case in cases]
    scored_rows = _run_ragas_evaluate(rows, llm, embeddings)

    per_case = [
        RagasCaseResult(
            query=case.query,
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
        )
        for case, scores in zip(cases, scored_rows, strict=True)
    ]

    report = RagasReport(
        per_case=per_case,
        faithfulness=_mean(result.faithfulness for result in per_case),
        answer_relevancy=_mean(result.answer_relevancy for result in per_case),
        context_precision=_mean(result.context_precision for result in per_case),
        context_recall=_mean(result.context_recall for result in per_case),
    )

    logger.info(
        "RAGAS evaluation completed: faithfulness=%.2f, answer_relevancy=%.2f, "
        "context_precision=%.2f, context_recall=%.2f across %d case(s)",
        report.faithfulness, report.answer_relevancy, report.context_precision, report.context_recall,
        len(per_case),
    )
    return report


def _mean(values: Any) -> float:
    """Arithmetic mean of an iterable of floats, or 0.0 if empty."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def save_ragas_results(report: RagasReport, path: Path | None = None) -> Path:
    """Persist `report` as JSON.

    Args:
        report: The RAGAS report to save.
        path: Destination file. Defaults to `DEFAULT_RAGAS_RESULTS_PATH`
            (``evaluation/ragas_results.json``).

    Returns:
        The path the report was written to.

    Raises:
        EvaluationError: If `path` cannot be written.
    """
    destination = path or DEFAULT_RAGAS_RESULTS_PATH
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to write RAGAS results to {destination}: {exc}") from exc

    logger.info("RAGAS results saved to %s", destination)
    return destination
