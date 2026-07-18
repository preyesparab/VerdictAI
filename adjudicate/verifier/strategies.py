"""Verification strategies, dispatched by claim_type (Phase 28).

This module contains **zero LLM calls**, and must never gain one. Every
strategy here answers "is this claim true?" by running something real
against real code - a sandboxed test, a static analyzer, the existing
test suite - never by asking a model to interpret or judge the claim. If
a claim type ever seems to need an LLM to "help interpret" it, the
correct move is to return `VerificationStatus.INCONCLUSIVE` at
`Confidence.LOW` and flag the gap (see `verify_exception_handling`,
which does exactly this today), not to add a model call here.

Dispatch table (`DISPATCH`):
    UNTESTED_BRANCH, MISSING_NULL_CHECK -> `verify_via_proposed_test`
        Run the claim's own `proposed_test` in a sandbox against the
        real, diff-applied code. Failing = CONFIRMED, passing = REFUTED,
        both at HIGH confidence - this is a real test execution.
    TYPE_MISMATCH -> `verify_type_mismatch`
        Run `mypy` scoped to the claimed file. MEDIUM confidence - a
        static analyzer hit/miss, not a real execution.
    SECURITY -> `verify_security`
        Run `bandit` scoped to the claimed file (Python only - no
        equivalent wired up yet for other languages, see
        `verify_security`'s own docstring). MEDIUM confidence.
    BREAKING_CHANGE -> `verify_breaking_change`
        Run the project's existing test suite and check for failures.
        MEDIUM confidence - an honest approximation, not a true
        before/after diff (see the function's own docstring for why).
    EXCEPTION_HANDLING -> `verify_exception_handling`
        No dedicated tool exists for this category yet. Falls back to
        `verify_via_proposed_test` if the claim has one; otherwise
        returns INCONCLUSIVE at LOW confidence rather than forcing bandit
        or mypy onto a category neither actually fits.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier.models import Confidence, VerificationResult, VerificationStatus
from adjudicate.verifier.sandbox import DEFAULT_TIMEOUT_SECONDS, run_sandboxed
from core.logging import get_logger

logger = get_logger(__name__)

_LOCATION_RE = re.compile(r"^(?P<file>.+):(?P<line>\d+)$")

_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


def _parse_location(claim: ProsecutorClaim) -> tuple[str, int] | None:
    """Split `claim.location` into `(file_path, line)`, or None if it's malformed."""
    match = _LOCATION_RE.match(claim.location)
    if not match:
        return None
    return match.group("file"), int(match.group("line"))


def _language_for(file_path: str) -> str | None:
    """Guess a sandbox language from `file_path`'s extension, or None if unsupported."""
    for suffix, language in _LANGUAGE_BY_SUFFIX.items():
        if file_path.endswith(suffix):
            return language
    return None


def verify_via_proposed_test(claim: ProsecutorClaim, repo_path: Path) -> VerificationResult:
    """Run `claim.proposed_test` in a sandbox rooted at `repo_path`.

    Args:
        claim: The claim to verify - `claim_type` is `UNTESTED_BRANCH`,
            `MISSING_NULL_CHECK`, or (via `verify_exception_handling`'s
            fallback) `EXCEPTION_HANDLING`.
        repo_path: A real, already diff-applied copy of the source tree
            `claim.location` refers to - see `adjudicate.verifier.sandbox
            .run_sandboxed`'s docstring for why this must be real code,
            not a description of it.

    Returns:
        CONFIRMED (HIGH) if the test failed - the claimed problem was
        actually observed. REFUTED (HIGH) if it passed. INCONCLUSIVE
        (LOW) if there's no `proposed_test` to run, the claimed file's
        language has no sandbox runner, or the run timed out (a hang
        doesn't prove or disprove the specific claim either way).
    """
    if not claim.proposed_test.strip():
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.INCONCLUSIVE,
            confidence=Confidence.LOW,
            evidence="Claim has no proposed_test to run.",
            strategy="proposed_test",
        )

    location = _parse_location(claim)
    file_path = location[0] if location else claim.location
    language = _language_for(file_path)
    if language is None:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.INCONCLUSIVE,
            confidence=Confidence.LOW,
            evidence=f"No sandbox runner for {file_path!r}'s file type (supported: {sorted(_LANGUAGE_BY_SUFFIX)}).",
            strategy="proposed_test",
        )

    result = run_sandboxed(claim.proposed_test, language, cwd=repo_path)
    evidence = f"exit_code={result.exit_code}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"

    if result.timed_out:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.INCONCLUSIVE,
            confidence=Confidence.LOW,
            evidence=f"proposed_test timed out after {DEFAULT_TIMEOUT_SECONDS}s - not treated as a pass or fail.",
            strategy="proposed_test",
        )

    if result.exit_code == 0:
        return VerificationResult(
            claim=claim, status=VerificationStatus.REFUTED, confidence=Confidence.HIGH,
            evidence=evidence, strategy="proposed_test",
        )
    return VerificationResult(
        claim=claim, status=VerificationStatus.CONFIRMED, confidence=Confidence.HIGH,
        evidence=evidence, strategy="proposed_test",
    )


def verify_security(claim: ProsecutorClaim, repo_path: Path) -> VerificationResult:
    """Run `bandit` scoped to `claim.location`'s file (Python only).

    Args:
        claim: A `SECURITY`-typed claim.
        repo_path: The real source tree `claim.location` refers to.

    Returns:
        CONFIRMED (MEDIUM) if bandit reports a finding at or within 2
        lines of the claimed line. REFUTED (MEDIUM) if bandit runs clean
        on the file. INCONCLUSIVE (LOW) if the location is malformed, the
        file doesn't exist, or it isn't Python - bandit has no equivalent
        wired up for other languages yet, flagged rather than silently
        skipped.
    """
    location = _parse_location(claim)
    if location is None:
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"location {claim.location!r} is not '<file_path>:<line>'.", strategy="bandit",
        )
    file_path, line = location
    target = repo_path / file_path
    if not target.is_file() or target.suffix != ".py":
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"No static analyzer available for {file_path!r} (bandit is Python-only).",
            strategy="bandit",
        )

    completed = subprocess.run(
        [sys.executable, "-m", "bandit", "-f", "json", "-q", str(target)],
        capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"bandit did not return parseable JSON:\n{completed.stdout}\n{completed.stderr}",
            strategy="bandit",
        )

    results = report.get("results", [])
    nearby = [r for r in results if abs(r.get("line_number", -9999) - line) <= 2]
    evidence = json.dumps(results if nearby else report, indent=2)
    if nearby:
        return VerificationResult(
            claim=claim, status=VerificationStatus.CONFIRMED, confidence=Confidence.MEDIUM,
            evidence=evidence, strategy="bandit",
        )
    return VerificationResult(
        claim=claim, status=VerificationStatus.REFUTED, confidence=Confidence.MEDIUM,
        evidence=evidence, strategy="bandit",
    )


def verify_type_mismatch(claim: ProsecutorClaim, repo_path: Path) -> VerificationResult:
    """Run `mypy` scoped to `claim.location`'s file (Python only).

    Args:
        claim: A `TYPE_MISMATCH`-typed claim.
        repo_path: The real source tree `claim.location` refers to.

    Returns:
        CONFIRMED (MEDIUM) if mypy reports an error at or within 2 lines
        of the claimed line. REFUTED (MEDIUM) if mypy runs clean.
        INCONCLUSIVE (LOW) if the location is malformed, the file
        doesn't exist, or it isn't Python.
    """
    location = _parse_location(claim)
    if location is None:
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"location {claim.location!r} is not '<file_path>:<line>'.", strategy="mypy",
        )
    file_path, line = location
    target = repo_path / file_path
    if not target.is_file() or target.suffix != ".py":
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"No type checker available for {file_path!r} (mypy is Python-only).", strategy="mypy",
        )

    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--ignore-missing-imports", str(target)],
        capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SECONDS, cwd=repo_path,
    )
    error_line_re = re.compile(re.escape(str(target.name)) + r":(\d+):")
    nearby_lines = [
        out_line for out_line in completed.stdout.splitlines()
        if (m := error_line_re.search(out_line)) and abs(int(m.group(1)) - line) <= 2
    ]
    evidence = completed.stdout + completed.stderr
    if nearby_lines:
        return VerificationResult(
            claim=claim, status=VerificationStatus.CONFIRMED, confidence=Confidence.MEDIUM,
            evidence=evidence, strategy="mypy",
        )
    return VerificationResult(
        claim=claim, status=VerificationStatus.REFUTED, confidence=Confidence.MEDIUM,
        evidence=evidence, strategy="mypy",
    )


def verify_breaking_change(claim: ProsecutorClaim, repo_path: Path, test_command: list[str]) -> VerificationResult:
    """Run the project's existing test suite and check for failures.

    This is an honest approximation, not a true before/after diff: Phase
    27's `ContextBundle` doesn't carry a stored pre-change test-run
    baseline to compare against, only the diff-applied `repo_path` - so
    this checks whether the test suite passes *now*, at MEDIUM
    confidence, rather than claiming to know which failures are *new*.
    A real before/after comparison (run the suite on the pre-diff commit
    too, diff the two result sets) is a reasonable follow-up, not
    something to fake here.

    Args:
        claim: A `BREAKING_CHANGE`-typed claim.
        repo_path: The real, diff-applied source tree.
        test_command: The command to run the relevant test suite (e.g.
            ``["npm", "test"]`` or ``["pytest", "tests/some_test.py"]``) -
            there is no single universal "run the tests" command across
            languages/repos, so the caller supplies it.

    Returns:
        CONFIRMED (MEDIUM) if the test command exits non-zero. REFUTED
        (MEDIUM) if it exits zero. INCONCLUSIVE (LOW) if the command
        itself can't be run (e.g. the test runner isn't installed) or
        times out.
    """
    try:
        completed = subprocess.run(
            test_command, cwd=repo_path, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return VerificationResult(
            claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
            evidence=f"Could not run the existing test suite ({test_command}): {exc}",
            strategy="existing_test_suite",
        )

    evidence = f"command={test_command}\nexit_code={completed.returncode}\n{completed.stdout}\n{completed.stderr}"
    if completed.returncode == 0:
        return VerificationResult(
            claim=claim, status=VerificationStatus.REFUTED, confidence=Confidence.MEDIUM,
            evidence=evidence, strategy="existing_test_suite",
        )
    return VerificationResult(
        claim=claim, status=VerificationStatus.CONFIRMED, confidence=Confidence.MEDIUM,
        evidence=evidence, strategy="existing_test_suite",
    )


def verify_exception_handling(claim: ProsecutorClaim, repo_path: Path) -> VerificationResult:
    """`EXCEPTION_HANDLING` has no dedicated tool yet - fall back honestly rather than force a bad fit.

    Neither bandit (security-focused) nor mypy (type-focused) is a
    genuine fit for "this except clause is too broad" or "this exception
    path is untested" - forcing either would produce a confident-looking
    but meaningless result. The only thing that actually fits is the same
    proposed-test execution `UNTESTED_BRANCH`/`MISSING_NULL_CHECK` use,
    *if* the claim happens to have a `proposed_test` (the real colorama
    claims did - see `docs/state/PROGRESS.md`'s Phase 27 follow-up
    entry). If it doesn't, this returns INCONCLUSIVE at LOW confidence -
    an honest gap, not a silently-forced verdict.

    Args:
        claim: An `EXCEPTION_HANDLING`-typed claim.
        repo_path: The real, diff-applied source tree.

    Returns:
        Whatever `verify_via_proposed_test` returns, if `claim.proposed_test`
        is non-empty; otherwise INCONCLUSIVE (LOW).
    """
    if claim.proposed_test.strip():
        return verify_via_proposed_test(claim, repo_path)
    return VerificationResult(
        claim=claim, status=VerificationStatus.INCONCLUSIVE, confidence=Confidence.LOW,
        evidence=(
            "No verification strategy exists yet for exception_handling claims without a "
            "proposed_test - neither bandit nor mypy is a genuine fit for this category. "
            "Flagged as a real gap, not forced onto the wrong tool."
        ),
        strategy="exception_handling_fallback",
    )


DISPATCH: dict[ClaimType, Callable[..., VerificationResult]] = {
    ClaimType.UNTESTED_BRANCH: verify_via_proposed_test,
    ClaimType.MISSING_NULL_CHECK: verify_via_proposed_test,
    ClaimType.TYPE_MISMATCH: verify_type_mismatch,
    ClaimType.SECURITY: verify_security,
    ClaimType.BREAKING_CHANGE: verify_breaking_change,
    ClaimType.EXCEPTION_HANDLING: verify_exception_handling,
}
"""Maps every `ClaimType` to its verification strategy. `verify_breaking_change`
takes an extra `test_command` argument the others don't - see `adjudicate.verifier
.verifier.verify_claim` for how the dispatch call is shaped around that difference."""
