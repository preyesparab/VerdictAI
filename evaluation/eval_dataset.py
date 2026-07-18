"""Loads the hand-written evaluation dataset (Phase 17).

`evaluation/eval_dataset.json` is a fixed set of manually written
question/expected-answer pairs for one target repository - the ground
truth every other evaluation module (`retrieval_eval`, `ragas_eval`,
`ablation`) measures the live pipeline against. Kept as a flat JSON file
(not generated) so the ground truth is auditable and stable across runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.exceptions import EvaluationError
from core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_EVAL_DATASET_PATH: Path = Path(__file__).resolve().parent / "eval_dataset.json"

_REQUIRED_FIELDS = (
    "repository_id",
    "query",
    "expected_answer",
    "expected_source_files",
    "expected_functions",
)


@dataclass(frozen=True)
class EvalCase:
    """One manually written question/expected-answer pair.

    Attributes:
        repository_id: The target repository this case was written
            against (see `database.sqlite_client.DatabaseManager.compute_repository_id`).
        query: The question to ask the pipeline.
        expected_answer: A human-written reference answer - RAGAS's
            "ground truth" for faithfulness/context recall, and what a
            human would compare a generated answer against.
        expected_source_files: Repository-relative file paths a good
            answer should be grounded in. Empty for a deliberately
            unanswerable query (the pipeline should say so instead of
            guessing).
        expected_functions: Function/method/class names a good answer
            should cite. Empty for a deliberately unanswerable query.
    """

    repository_id: str
    query: str
    expected_answer: str
    expected_source_files: list[str]
    expected_functions: list[str]


def _parse_case(raw: dict, index: int) -> EvalCase:
    """Validate and convert one raw JSON object into an `EvalCase`.

    Args:
        raw: The parsed JSON object for one dataset entry.
        index: The entry's position in the dataset (for error messages).

    Returns:
        The parsed `EvalCase`.

    Raises:
        EvaluationError: If `raw` is missing a required field.
    """
    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise EvaluationError(f"Eval dataset entry {index} is missing field(s): {missing}")

    return EvalCase(
        repository_id=raw["repository_id"],
        query=raw["query"],
        expected_answer=raw["expected_answer"],
        expected_source_files=list(raw["expected_source_files"]),
        expected_functions=list(raw["expected_functions"]),
    )


def load_eval_dataset(path: Path | None = None) -> list[EvalCase]:
    """Load and validate the evaluation dataset.

    Args:
        path: Dataset file to load. Defaults to `DEFAULT_EVAL_DATASET_PATH`
            (``evaluation/eval_dataset.json``).

    Returns:
        Every entry in the dataset, in file order.

    Raises:
        EvaluationError: If `path` does not exist, is not valid JSON, is
            not a JSON array, or any entry is missing a required field.
    """
    dataset_path = path or DEFAULT_EVAL_DATASET_PATH
    logger.info("Loading evaluation dataset from %s", dataset_path)

    try:
        raw_text = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to read evaluation dataset at {dataset_path}: {exc}") from exc

    try:
        raw_entries = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Evaluation dataset at {dataset_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw_entries, list):
        raise EvaluationError(
            f"Evaluation dataset at {dataset_path} must be a JSON array, got {type(raw_entries).__name__}"
        )

    cases = [_parse_case(entry, index) for index, entry in enumerate(raw_entries)]

    logger.info("Evaluation dataset loaded: %d case(s) from %s", len(cases), dataset_path)
    return cases
