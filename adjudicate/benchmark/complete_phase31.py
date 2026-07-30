"""Completes Phase 31's benchmark by running only the 13th case for real.

`adjudicate/benchmark/results/20260707T143527Z.json` has real, complete
three-condition results for 12 of the 13 curated cases; the 13th
(`type_mismatch_fixed`) failed both providers' quota that day and was
excluded from that report's own aggregate metrics (see its
`failed_cases` field). Re-running all 13 cases from scratch would waste
11 cases' worth of real LLM calls on results that already exist and
haven't changed - this script instead runs `type_mismatch_fixed` alone
and merges it into the prior report's aggregates algebraically (each
aggregate's stored mean, multiplied back out by its own case count,
plus the new case's value, divided by the new total) rather than
recomputing from raw per-case numbers the saved report never persisted.

This is safe specifically because the 13th case is clean
(`ground_truth_buggy=False`): catch rate (computed only over buggy
cases) is provably unchanged by adding it, so only false-positive rate,
the LLM-call/token/round averages, and the claim-flip counters need
recombining.

Usage: `python -m adjudicate.benchmark.complete_phase31`
"""

from __future__ import annotations

import json
from pathlib import Path

from adjudicate.agents.defender import DefenderAgent
from adjudicate.agents.judge import JudgeAgent
from adjudicate.agents.prosecutor import ProsecutorAgent
from adjudicate.benchmark.baseline_reviewer import BaselineReviewer
from adjudicate.benchmark.harness import CASES_DIR, load_case
from adjudicate.benchmark.run import (
    RESULTS_DIR,
    _run_case_with_backoff,
    print_report,
    save_report,
)
from adjudicate.benchmark.token_tracking import TokenTrackingLLMClient
from adjudicate.verifier import VerificationStatus
from core.logging import get_logger
from generation.llm_client import LLMClient

logger = get_logger(__name__)

PRIOR_REPORT_PATH = RESULTS_DIR / "20260707T143527Z.json"
TARGET_CASE_ID = "type_mismatch_fixed"


def _merged_condition_metrics(prior: dict, old_n: int, new_n: int, flagged_13: bool, rounds_13: int | None,
                               llm_calls_13: int, tokens_13: int) -> dict:
    """Recombine one condition's stored averages with the 13th case's real result."""
    old_clean_flagged = round(prior["false_positive_rate"] * (old_n - 6))  # old_n=12, 6 buggy, 6 clean
    new_clean_flagged = old_clean_flagged + (1 if flagged_13 else 0)
    new_false_positive_rate = new_clean_flagged / (new_n - 6)  # new_n=13, 6 buggy, 7 clean

    new_avg_llm_calls = (prior["avg_llm_calls"] * old_n + llm_calls_13) / new_n
    new_avg_tokens = (prior["avg_tokens"] * old_n + tokens_13) / new_n

    if prior["avg_rounds"] is None and rounds_13 is None:
        new_avg_rounds = None
    else:
        old_rounds_sum = (prior["avg_rounds"] or 0.0) * old_n
        rounds_n = old_n + (1 if rounds_13 is not None else 0)
        new_avg_rounds = (old_rounds_sum + (rounds_13 or 0)) / rounds_n if rounds_n else None

    return {
        "catch_rate": prior["catch_rate"],  # unaffected: case 13 is clean, catch rate is buggy-only
        "false_positive_rate": new_false_positive_rate,
        "avg_llm_calls": new_avg_llm_calls,
        "avg_tokens": new_avg_tokens,
        "avg_rounds": new_avg_rounds,
    }


def complete_and_merge() -> dict:
    prior = json.loads(PRIOR_REPORT_PATH.read_text(encoding="utf-8"))
    assert prior["failed_cases"] == [TARGET_CASE_ID], (
        f"Expected only {TARGET_CASE_ID} to be outstanding, found {prior['failed_cases']}"
    )
    old_n = prior["num_cases"] - 1  # 12 cases actually ran and contributed to prior['metrics']

    case = load_case(CASES_DIR / f"{TARGET_CASE_ID}.json")
    assert not case.ground_truth_buggy, "Merge shortcut assumes case 13 is clean - re-derive if this changes"

    llm_client = LLMClient()
    tracker = TokenTrackingLLMClient(llm_client)
    reviewer = BaselineReviewer(llm_client=tracker)
    defender = DefenderAgent(llm_client=tracker)
    prosecutor = ProsecutorAgent(llm_client=tracker)
    judge = JudgeAgent(llm_client=tracker)

    logger.info("Running the outstanding case for real: %s", TARGET_CASE_ID)
    outcome_a, outcome_bc = _run_case_with_backoff(case, reviewer, defender, prosecutor, judge, tracker)

    refuted_flags_13 = [vr.status == VerificationStatus.REFUTED for vr in outcome_bc.verified]

    case_detail_13 = {
        "case_id": case.case_id,
        "ground_truth_buggy": case.ground_truth_buggy,
        "ground_truth_rationale": case.ground_truth_rationale,
        "condition_a": {"has_issue": outcome_a.has_issue, "explanation": outcome_a.explanation},
        "condition_b": {"flagged": outcome_bc.result_b.flagged, "num_claims": len(outcome_bc.claims)},
        "condition_c": {
            "flagged": outcome_bc.result_c.flagged,
            "verdict": outcome_bc.verdict_value,
            "confidence": outcome_bc.verdict_confidence,
            "minority_report": outcome_bc.verdict_minority_report,
            "rounds_used": outcome_bc.result_c.rounds_used,
        },
        "claims": [
            {
                "claim_type": vr.claim.claim_type.value, "location": vr.claim.location,
                "assertion": vr.claim.assertion, "status": vr.status.value, "confidence": vr.confidence.value,
            }
            for vr in outcome_bc.verified
        ],
    }

    new_n = old_n + 1
    merged_metrics = {
        "condition_a_baseline": _merged_condition_metrics(
            prior["metrics"]["condition_a_baseline"], old_n, new_n,
            outcome_a.has_issue, None, outcome_a.result.llm_calls, outcome_a.result.total_tokens,
        ),
        "condition_b_adversarial_no_verifier": _merged_condition_metrics(
            prior["metrics"]["condition_b_adversarial_no_verifier"], old_n, new_n,
            outcome_bc.result_b.flagged, None, outcome_bc.result_b.llm_calls, outcome_bc.result_b.total_tokens,
        ),
        "condition_c_full_pipeline": _merged_condition_metrics(
            prior["metrics"]["condition_c_full_pipeline"], old_n, new_n,
            outcome_bc.result_c.flagged, outcome_bc.result_c.rounds_used,
            outcome_bc.result_c.llm_calls, outcome_bc.result_c.total_tokens,
        ),
        "total_claims_raised": prior["metrics"]["total_claims_raised"] + len(refuted_flags_13),
        "total_claims_refuted": prior["metrics"]["total_claims_refuted"] + sum(refuted_flags_13),
    }
    merged_metrics["claim_flip_rate_b_to_c"] = (
        merged_metrics["total_claims_refuted"] / merged_metrics["total_claims_raised"]
        if merged_metrics["total_claims_raised"] else 0.0
    )

    report = {
        "generated_at": prior["generated_at"],
        "merged_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "num_cases": 13,
        "failed_cases": [],
        "num_buggy": prior["num_buggy"],
        "num_clean": prior["num_clean"],
        "metrics": merged_metrics,
        "cases": prior["cases"] + [case_detail_13],
        "note": (
            "12 cases reused verbatim from 20260707T143527Z.json (no re-spent LLM calls); "
            f"only {TARGET_CASE_ID} was run live in this pass. Aggregates recombined "
            "algebraically from the prior report's own stored means, not recomputed from "
            "scratch - see this module's docstring."
        ),
    }
    return report


if __name__ == "__main__":
    final_report = complete_and_merge()
    print_report(final_report)
    saved_path = save_report(final_report)
    print(f"\nFinal 13-case report saved to {saved_path}")
