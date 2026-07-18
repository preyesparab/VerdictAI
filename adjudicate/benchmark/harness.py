"""Benchmark harness (Phase 31): runs all three conditions across the curated case set.

Loads each case from `adjudicate/benchmark/cases/*.json`, materializes a
real sandbox for it (a temp dir with the target file, or - for the two
reused real-repo cases - a copy of the real cloned repo with the real
diff fixture applied via `patch` and any needed stub files), builds a
`ContextBundle` from the case's own already-verified embedded data (zero
LLM - no live retrieval needed, see each case JSON's `context_bundle`
field), and runs:

- **Condition (a)** - `adjudicate.benchmark.baseline_reviewer.BaselineReviewer`,
  a single unstructured LLM call. Flagged = `has_issue`.
- **Condition (b)** - the real `DefenderAgent.draft_justification` +
  `ProsecutorAgent.raise_concerns` (Phases 25-27, unmodified), with the
  Prosecutor's claims trusted at face value - no Verifier runs. Flagged =
  at least one claim was raised.
- **Condition (c)** - the same Defender/Prosecutor calls (shared with
  condition b, not re-run, both to save real LLM calls and to make the
  claim-flip-rate comparison a fair like-for-like), but every claim is
  now run through the real `adjudicate.verifier.verify_claim` (Phase 28,
  zero LLM), then a real `run_rebuttal_loop` (Phase 29) and a real
  `JudgeAgent.judge` (Phase 30) call. Flagged = verdict != approve.

`condition_b_and_c_result` returns both a `CaseResult` for (b), one for
(c), and the raw `VerificationResult`s (needed for the claim-flip-rate
metric, computed separately in `adjudicate.benchmark.metrics`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from adjudicate.agents.defender import DefenderAgent
from adjudicate.agents.judge import JudgeAgent
from adjudicate.agents.prosecutor import ProsecutorAgent
from adjudicate.benchmark.baseline_reviewer import BaselineReviewer
from adjudicate.benchmark.metrics import CaseResult
from adjudicate.benchmark.token_tracking import TokenTrackingLLMClient
from adjudicate.context_builder import ChangedFunction, ContextBundle, GraphNeighbor, RelatedTest
from adjudicate.orchestrator import run_rebuttal_loop
from adjudicate.schemas import ProsecutorClaim
from adjudicate.verifier import VerificationResult, verify_claim
from core.logging import get_logger

logger = get_logger(__name__)

CASES_DIR = Path(__file__).resolve().parent / "cases"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class BenchmarkCase:
    """One loaded, ready-to-run benchmark case."""

    case_id: str
    ground_truth_buggy: bool
    ground_truth_rationale: str
    diff: str
    context_bundle: ContextBundle
    raw: dict


def _bundle_from_json(data: dict) -> ContextBundle:
    """Build a `ContextBundle` from a case JSON's `context_bundle` field - zero LLM, pure data."""
    return ContextBundle(
        changed_functions=[
            ChangedFunction(
                chunk_id=f"benchmark-{i}", file_path=cf["file_path"], function_name=cf["function_name"],
                class_name=cf["class_name"], changed_start_line=cf["start_line"], changed_end_line=cf["end_line"],
            )
            for i, cf in enumerate(data["changed_functions"])
        ],
        callers=[
            GraphNeighbor(
                node_id=f"caller-{i}", node_kind=n["node_kind"], file_path=n["file_path"],
                function_name=n["function_name"], class_name=n["class_name"],
            )
            for i, n in enumerate(data["callers"])
        ],
        callees=[
            GraphNeighbor(
                node_id=f"callee-{i}", node_kind=n["node_kind"], file_path=n["file_path"],
                function_name=n["function_name"], class_name=n["class_name"],
            )
            for i, n in enumerate(data["callees"])
        ],
        related_tests=[
            RelatedTest(
                chunk_id=f"test-{i}", file_path=t["file_path"], function_name=t["function_name"],
                found_via=t["found_via"],
            )
            for i, t in enumerate(data["related_tests"])
        ],
    )


def load_case(path: Path) -> BenchmarkCase:
    """Load one case JSON into a `BenchmarkCase`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    diff = data["diff"] if "diff" in data else (FIXTURES_DIR / data["diff_file"]).read_text(encoding="utf-8")
    return BenchmarkCase(
        case_id=data["case_id"],
        ground_truth_buggy=data["ground_truth_buggy"],
        ground_truth_rationale=data["ground_truth_rationale"],
        diff=diff,
        context_bundle=_bundle_from_json(data["context_bundle"]),
        raw=data,
    )


def load_all_cases() -> list[BenchmarkCase]:
    """Load every case in `adjudicate/benchmark/cases/`, sorted by case_id for a stable run order."""
    return [load_case(p) for p in sorted(CASES_DIR.glob("*.json"))]


def materialize_sandbox(case: BenchmarkCase, tmp_path: Path) -> None:
    """Write whatever `case` needs into `tmp_path` so a `proposed_test` can run against real code.

    For a synthetic case, just writes `target_file_content` at
    `target_file_relative_path`. For a reused real-repo case, copies the
    real cloned source tree, applies the real diff fixture via `patch
    -p1`, and writes any needed stub files (see each case JSON's
    `sandbox_stubs`) - the same materialization Phase 28/29's own
    ground-truth verification used.
    """
    if case.raw.get("is_real_repo_case"):
        source = REPO_ROOT / case.raw["repo_copy_source"]
        dest = tmp_path / case.raw["repo_copy_dest_relative"]
        shutil.copytree(source, dest)
        diff_path = FIXTURES_DIR / case.raw["diff_file"]
        result = subprocess.run(
            ["patch", "-p1"], cwd=tmp_path, input=diff_path.read_text(encoding="utf-8"),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{case.case_id}: diff fixture failed to apply: {result.stderr}")
        for relative_path, content in case.raw.get("sandbox_stubs", {}).items():
            stub_path = tmp_path / relative_path
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text(content, encoding="utf-8")
    else:
        target_path = tmp_path / case.raw["target_file_relative_path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(case.raw["target_file_content"], encoding="utf-8")


@dataclass(frozen=True)
class ConditionAOutcome:
    result: CaseResult
    has_issue: bool
    explanation: str


def run_condition_a(case: BenchmarkCase, reviewer: BaselineReviewer, tracker: TokenTrackingLLMClient) -> ConditionAOutcome:
    """Run the single-agent baseline (condition a) for one case."""
    tracker.reset()
    review = reviewer.review(case.context_bundle, case.diff)
    return ConditionAOutcome(
        result=CaseResult(
            case_id=case.case_id, ground_truth_buggy=case.ground_truth_buggy,
            flagged=review.has_issue, llm_calls=tracker.call_count, total_tokens=tracker.total_tokens,
        ),
        has_issue=review.has_issue, explanation=review.explanation,
    )


@dataclass(frozen=True)
class ConditionBAndCOutcome:
    """Both conditions' results for one case - computed together since they share the
    Defender/Prosecutor calls (see this module's own docstring for why)."""

    result_b: CaseResult
    result_c: CaseResult
    claims: list[ProsecutorClaim]
    verified: list[VerificationResult]
    defender_justification: str
    rebuttal_transcript: list[str]
    verdict_value: str
    verdict_confidence: float
    verdict_minority_report: str | None


def run_condition_b_and_c(
    case: BenchmarkCase, repo_path: Path, defender: DefenderAgent, prosecutor: ProsecutorAgent,
    judge: JudgeAgent, tracker: TokenTrackingLLMClient,
) -> ConditionBAndCOutcome:
    """Run conditions (b) and (c) together for one case, sharing the Defender/Prosecutor calls."""
    tracker.reset()
    justification = defender.draft_justification(case.context_bundle, case.diff)
    claims = prosecutor.raise_concerns(case.context_bundle, case.diff, justification)
    b_llm_calls, b_tokens = tracker.call_count, tracker.total_tokens

    result_b = CaseResult(
        case_id=case.case_id, ground_truth_buggy=case.ground_truth_buggy,
        flagged=len(claims) > 0, llm_calls=b_llm_calls, total_tokens=b_tokens,
    )

    verified = [verify_claim(claim, repo_path=repo_path) for claim in claims]  # zero LLM - real tools only

    loop_result = run_rebuttal_loop(case.context_bundle, case.diff, justification, verified, defender=defender)
    verdict = judge.judge(
        case.context_bundle, case.diff, loop_result.transcript, verified, loop_result.ended_by
    )

    result_c = CaseResult(
        case_id=case.case_id, ground_truth_buggy=case.ground_truth_buggy,
        flagged=verdict.verdict.value != "approve", llm_calls=tracker.call_count, total_tokens=tracker.total_tokens,
        rounds_used=loop_result.rounds_used,
    )

    return ConditionBAndCOutcome(
        result_b=result_b, result_c=result_c, claims=claims, verified=verified,
        defender_justification=justification, rebuttal_transcript=loop_result.transcript,
        verdict_value=verdict.verdict.value, verdict_confidence=verdict.confidence,
        verdict_minority_report=verdict.minority_report,
    )


def run_case(
    case: BenchmarkCase, reviewer: BaselineReviewer, defender: DefenderAgent,
    prosecutor: ProsecutorAgent, judge: JudgeAgent, tracker: TokenTrackingLLMClient,
) -> tuple[ConditionAOutcome, ConditionBAndCOutcome]:
    """Run all three conditions for one case, materializing its sandbox exactly once."""
    logger.info("Running benchmark case: %s (ground_truth_buggy=%s)", case.case_id, case.ground_truth_buggy)
    with tempfile.TemporaryDirectory(prefix=f"benchmark_{case.case_id}_") as tmp:
        tmp_path = Path(tmp)
        materialize_sandbox(case, tmp_path)
        outcome_a = run_condition_a(case, reviewer, tracker)
        outcome_bc = run_condition_b_and_c(case, tmp_path, defender, prosecutor, judge, tracker)
    return outcome_a, outcome_bc
