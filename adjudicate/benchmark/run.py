"""Benchmark run entry point (Phase 31): `python -m adjudicate.benchmark.run`.

Runs all three conditions across every case in `adjudicate/benchmark/cases/`,
prints a full results table, and saves a JSON report to
`adjudicate/benchmark/results/<timestamp>.json` for later reference -
this is a real, reusable entry point, not a throwaway script, so a
future session (or Phase 32's React frontend) can re-run or inspect past
runs without redoing the case-curation work.

Provider: this module does not force a provider - it uses whatever
`generation.llm_client.LLMClient` resolves to from the environment
(`config.settings`). See `docs/state/PROGRESS.md`'s Phase 31 entry for
which provider an actual run used and why.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from adjudicate.agents.defender import DefenderAgent
from adjudicate.agents.judge import JudgeAgent
from adjudicate.agents.prosecutor import ProsecutorAgent
from adjudicate.benchmark.baseline_reviewer import BaselineReviewer
from adjudicate.benchmark.harness import load_all_cases, run_case
from adjudicate.benchmark.metrics import (
    CaseResult,
    average_llm_calls,
    average_rounds,
    average_tokens,
    catch_rate,
    claim_flip_rate,
    false_positive_rate,
)
from adjudicate.benchmark.token_tracking import TokenTrackingLLMClient
from adjudicate.verifier import VerificationStatus
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from generation.llm_client import LLMClient

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_BACKOFF_ATTEMPTS = 4
_BACKOFF_DELAY_SECONDS = 15.0


def _run_case_with_backoff(*args, **kwargs):
    """Retry `run_case` on a transient provider error (503/429) - not a malformed-output
    retry (each agent already handles that internally), purely network/capacity flakiness,
    the same class of transient failure seen repeatedly earlier this session."""
    for attempt in range(1, _BACKOFF_ATTEMPTS + 1):
        try:
            return run_case(*args, **kwargs)
        except LLMGenerationError as exc:
            if attempt == _BACKOFF_ATTEMPTS:
                raise
            logger.warning(
                "Transient provider error on attempt %d/%d, waiting %.0fs: %s",
                attempt, _BACKOFF_ATTEMPTS, _BACKOFF_DELAY_SECONDS, exc,
            )
            time.sleep(_BACKOFF_DELAY_SECONDS)


def run_benchmark() -> dict:
    """Run every case through all three conditions and return the full raw + aggregated report."""
    llm_client = LLMClient()
    tracker = TokenTrackingLLMClient(llm_client)
    reviewer = BaselineReviewer(llm_client=tracker)
    defender = DefenderAgent(llm_client=tracker)
    prosecutor = ProsecutorAgent(llm_client=tracker)
    judge = JudgeAgent(llm_client=tracker)

    cases = load_all_cases()
    logger.info("Loaded %d benchmark cases (%d buggy, %d clean)",
                len(cases), sum(c.ground_truth_buggy for c in cases), sum(not c.ground_truth_buggy for c in cases))

    results_a: list[CaseResult] = []
    results_b: list[CaseResult] = []
    results_c: list[CaseResult] = []
    claim_flip_flags: list[bool] = []
    per_case_detail: list[dict] = []
    failed_cases: list[str] = []

    for case in cases:
        try:
            outcome_a, outcome_bc = _run_case_with_backoff(case, reviewer, defender, prosecutor, judge, tracker)
        except LLMGenerationError as exc:
            logger.error("Case %s failed after all retries, skipping: %s", case.case_id, exc)
            failed_cases.append(case.case_id)
            continue
        results_a.append(outcome_a.result)
        results_b.append(outcome_bc.result_b)
        results_c.append(outcome_bc.result_c)

        for verified_claim in outcome_bc.verified:
            claim_flip_flags.append(verified_claim.status == VerificationStatus.REFUTED)

        per_case_detail.append({
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
        })
        logger.info(
            "Case %s done: a=%s b=%s c=%s (%d claim(s))",
            case.case_id, outcome_a.has_issue, outcome_bc.result_b.flagged, outcome_bc.result_c.flagged,
            len(outcome_bc.claims),
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(cases),
        "failed_cases": failed_cases,
        "num_buggy": sum(c.ground_truth_buggy for c in cases),
        "num_clean": sum(not c.ground_truth_buggy for c in cases),
        "metrics": {
            "condition_a_baseline": {
                "catch_rate": catch_rate(results_a),
                "false_positive_rate": false_positive_rate(results_a),
                "avg_llm_calls": average_llm_calls(results_a),
                "avg_tokens": average_tokens(results_a),
                "avg_rounds": average_rounds(results_a),
            },
            "condition_b_adversarial_no_verifier": {
                "catch_rate": catch_rate(results_b),
                "false_positive_rate": false_positive_rate(results_b),
                "avg_llm_calls": average_llm_calls(results_b),
                "avg_tokens": average_tokens(results_b),
                "avg_rounds": average_rounds(results_b),
            },
            "condition_c_full_pipeline": {
                "catch_rate": catch_rate(results_c),
                "false_positive_rate": false_positive_rate(results_c),
                "avg_llm_calls": average_llm_calls(results_c),
                "avg_tokens": average_tokens(results_c),
                "avg_rounds": average_rounds(results_c),
            },
            "claim_flip_rate_b_to_c": claim_flip_rate(claim_flip_flags),
            "total_claims_raised": len(claim_flip_flags),
            "total_claims_refuted": sum(claim_flip_flags),
        },
        "cases": per_case_detail,
    }
    return report


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_report(report: dict) -> None:
    if report["failed_cases"]:
        print(f"\n!!! {len(report['failed_cases'])} case(s) failed after all retries and were skipped: "
              f"{report['failed_cases']} - metrics below are computed over the remaining cases only !!!")
    m = report["metrics"]
    print(f"\n{'Condition':<35} {'Catch Rate':>12} {'False Pos':>12} {'Avg Calls':>11} {'Avg Tokens':>12} {'Avg Rounds':>11}")
    for key, label in [
        ("condition_a_baseline", "(a) Single-agent baseline"),
        ("condition_b_adversarial_no_verifier", "(b) Adversarial, no verifier"),
        ("condition_c_full_pipeline", "(c) Full pipeline"),
    ]:
        d = m[key]
        rounds = f"{d['avg_rounds']:.2f}" if d["avg_rounds"] is not None else "N/A"
        print(f"{label:<35} {d['catch_rate']*100:>11.1f}% {d['false_positive_rate']*100:>11.1f}% "
              f"{d['avg_llm_calls']:>11.2f} {d['avg_tokens']:>12.1f} {rounds:>11}")
    print(f"\nClaim-flip rate (b -> c): {m['claim_flip_rate_b_to_c']*100:.1f}% "
          f"({m['total_claims_refuted']}/{m['total_claims_raised']} claims refuted once verified)")


if __name__ == "__main__":
    report = run_benchmark()
    print_report(report)
    saved_path = save_report(report)
    print(f"\nFull report saved to {saved_path}")
