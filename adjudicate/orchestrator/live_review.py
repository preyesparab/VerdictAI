"""Live review streaming (Phase 32 Part 2): the real Phases 24-30 pipeline, one event per stage.

`run_live_review` is the data source behind `api/main.py`'s new
`POST /repos/{repo_id}/review` Server-Sent-Events endpoint. It wires
together exactly the same agents/functions every prior phase already
built and verified - `AdjudicateContextBuilder` (Phase 24),
`DefenderAgent` (Phase 25/29), `ProsecutorAgent` (Phase 26/27),
`verify_claim` (Phase 28), the rebuttal-loop termination logic (Phase 29,
`adjudicate.orchestrator.rebuttal_loop`), and `JudgeAgent` (Phase 30) -
and yields one plain, JSON-serializable event dict as each stage
completes. No new schema is invented and no data is simulated: every
event field is copied directly from the real dataclass each phase
already produces (`ContextBundle`, `ProsecutorClaim`, `VerificationResult`,
`JudgeVerdict`).

**Reuse, not duplication, of the rebuttal loop's termination logic**:
`adjudicate.orchestrator.rebuttal_loop.run_rebuttal_loop` runs its whole
loop atomically and only returns once finished - useful for the
benchmark harness (Phase 31), wrong for this module, which needs to
yield an event *after each round* as it happens (the "watching the case
build in real time" requirement this whole screen exists for). The loop
body below is a line-for-line mirror of `run_rebuttal_loop`'s own loop -
same `_is_resolved` check (imported, not reimplemented), same
`MAX_REBUTTAL_ROUNDS` bound, same round-counting - restructured only to
yield mid-loop instead of returning at the end. If `run_rebuttal_loop`'s
loop body ever changes, this one must change with it.

**Real bug found and fixed while wiring this up, not assumed away**:
`adjudicate.repomind_client.RepoMindClient`'s own module docstring
claims it can be pointed at `api.main.app` in-process via
`httpx.ASGITransport`, "without needing a live `uvicorn` process." That
claim does not hold with the `httpx` version this project has installed
(0.28.1): `ASGITransport` there implements only `handle_async_request`,
so pairing it with `RepoMindClient`'s sync `httpx.Client` fails with
``'ASGITransport' object has no attribute 'handle_request'`` - a latent,
pre-existing gap in that docstring's claim, not something introduced
here, caught live during this phase's own verification. Since this
endpoint runs *inside* the already-live server process handling the
request, there is no need for an in-process shortcut anyway - `base_url`
(the real ``http://host:port`` the current request itself arrived on,
via FastAPI's `Request.base_url`) is passed straight to a real
`RepoMindClient`, a genuine (if self-referential) HTTP call over
localhost, the same approach every other real client of this API
already uses.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from adjudicate.agents.defender import DefenderAgent
from adjudicate.agents.judge import JudgeAgent
from adjudicate.agents.prosecutor import ProsecutorAgent
from adjudicate.context_builder import AdjudicateContextBuilder, ContextBundle
from adjudicate.orchestrator.rebuttal_loop import MAX_REBUTTAL_ROUNDS, _is_resolved
from adjudicate.repomind_client import RepoMindClient
from adjudicate.schemas import ProsecutorClaim
from adjudicate.verifier import VerificationResult, verify_claim
from core.exceptions import RepoMindError
from core.logging import get_logger

logger = get_logger(__name__)

DIFF_APPLY_TIMEOUT_SECONDS = 30.0

# Matches a git-diff `index <old>..<new>[ <mode>]` header line - stripped
# before handing the diff to plain `patch` (see `_materialize_sandbox`'s
# own docstring for why).
_GIT_INDEX_LINE_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+(?: \d+)?\s*$", re.MULTILINE)


def _strip_git_index_lines(diff: str) -> str:
    """Remove any git-diff `index <old>..<new>` header line(s) from `diff`.

    Real reproduction (Phase 28 follow-up, `mkocabas/VIBE`): a diff
    combining a `diff --git a/X b/X` header with a placeholder
    `index 0000000..0000000 100644` line (the convention this project's
    own hand-constructed test/benchmark diffs commonly use, in lieu of a
    real `git diff`) makes GNU `patch` misinterpret an ordinary
    modification to an *existing* file as an attempt to *create* a new
    one - `0000000` on the old side is git's own "this file didn't
    exist before" signal, and patch reads it as such even though the
    hunk itself clearly modifies real, existing lines. Confirmed via
    direct isolation: `diff --git` alone or `index ...` alone apply
    fine; only the *combination* triggers this. Plain POSIX `patch`
    never actually needs the `index` line at all - only `---`/`+++`/`@@`
    drive hunk application - so removing it is safe for both a
    placeholder-hash diff (this bug) and a diff with real hashes (the
    line is simply redundant there).

    Args:
        diff: The raw unified diff text, possibly git-formatted.

    Returns:
        `diff` with any `index ...` line(s) removed; unchanged if none
        are present (a diff without git's extended headers, e.g. plain
        `diff -u` output, was never affected by this in the first place).
    """
    return _GIT_INDEX_LINE_RE.sub("", diff)


def _materialize_sandbox(local_path: Path, diff: str) -> Path:
    """Copy the real indexed repository's source tree to a temp dir and apply `diff` to it.

    Deliberately general - no per-repo/per-diff stub files or special
    casing (unlike `adjudicate/benchmark/harness.py`'s own
    `materialize_sandbox`, which hand-stubs specific fixtures for
    specific benchmark cases). A review submitted through the live UI is
    against whatever the repo and diff actually are; a claim whose
    `proposed_test` needs a dependency this sandbox doesn't have (e.g. a
    real `npm install` for a Node project) will genuinely fail to run,
    which `adjudicate.verifier.strategies.verify_via_proposed_test`
    already reports honestly (non-zero exit, real stdout/stderr in
    `evidence`) rather than something this function needs to paper over.

    This function only ever *reads* `local_path` (`shutil.copytree` into
    a brand-new `tempfile.mkdtemp()` directory) and applies the diff to
    that copy alone - `local_path`, the persistent indexed clone, is
    never opened for writing here, and the temp copy is the caller's
    (`run_live_review`'s) responsibility to `shutil.rmtree` afterward.
    Confirmed live (Phase 28 follow-up): repeated sandbox-creation
    attempts, including failing ones, leave the persistent clone's own
    `git status` clean throughout - there is no code path here that
    could leak a mutation back into it.

    Args:
        local_path: The indexed repository's real cloned path
            (`ingestion.repository_metadata.RepositoryMetadata.local_path`).
        diff: The raw unified diff to apply.

    Returns:
        The temp directory's path - a real, diff-applied copy of the repo.

    Raises:
        RepoMindError: If the diff does not apply cleanly, or `patch`
            itself doesn't finish within `DIFF_APPLY_TIMEOUT_SECONDS`.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="adjudicate_review_"))
    shutil.copytree(local_path, tmp_dir, dirs_exist_ok=True)
    patch_input = _strip_git_index_lines(diff)
    try:
        result = subprocess.run(
            ["patch", "-p1"], cwd=tmp_dir, input=patch_input, capture_output=True, text=True,
            timeout=DIFF_APPLY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoMindError(f"Applying the diff timed out after {DIFF_APPLY_TIMEOUT_SECONDS}s.") from exc
    if result.returncode != 0:
        raise RepoMindError(
            f"Diff did not apply cleanly to the indexed repository: {result.stderr or result.stdout}"
        )
    return tmp_dir


def _context_event(bundle: ContextBundle) -> dict[str, Any]:
    return {
        "type": "context",
        "changed_functions": [
            {
                "file_path": fn.file_path, "function_name": fn.function_name, "class_name": fn.class_name,
                "start_line": fn.changed_start_line, "end_line": fn.changed_end_line,
            }
            for fn in bundle.changed_functions
        ],
        "callers": [
            {
                "file_path": n.file_path, "function_name": n.function_name,
                "class_name": n.class_name, "node_kind": n.node_kind,
            }
            for n in bundle.callers
        ],
        "callees": [
            {
                "file_path": n.file_path, "function_name": n.function_name,
                "class_name": n.class_name, "node_kind": n.node_kind,
            }
            for n in bundle.callees
        ],
        "related_tests": [
            {"file_path": t.file_path, "function_name": t.function_name, "found_via": t.found_via}
            for t in bundle.related_tests
        ],
    }


def _claim_event(claim: ProsecutorClaim, index: int) -> dict[str, Any]:
    return {
        "index": index, "claim_type": claim.claim_type.value, "location": claim.location,
        "assertion": claim.assertion, "proposed_test": claim.proposed_test,
    }


def _verification_event(index: int, result: VerificationResult) -> dict[str, Any]:
    return {
        "type": "claim_verified", "index": index, "status": result.status.value,
        "confidence": result.confidence.value, "evidence": result.evidence, "strategy": result.strategy,
    }


def run_live_review(
    repo_id: str,
    local_path: Path,
    diff: str,
    base_url: str,
    *,
    context_builder: AdjudicateContextBuilder | None = None,
    defender: DefenderAgent | None = None,
    prosecutor: ProsecutorAgent | None = None,
    judge: JudgeAgent | None = None,
) -> Iterator[dict[str, Any]]:
    """Run the full Phases 24-30 pipeline against `diff`, yielding one event dict per stage.

    Args:
        repo_id: The indexed repository this diff is reviewed against.
        local_path: The repository's real cloned path on disk.
        diff: The raw unified diff text to review.
        base_url: The live API's own base URL (e.g.
            ``str(request.base_url)`` from the FastAPI endpoint calling
            this) - `AdjudicateContextBuilder` talks to it via a real
            `RepoMindClient` HTTP call back to the same server (see this
            module's own docstring for why this is a real network call,
            not an in-process shortcut). Ignored if `context_builder` is given.
        context_builder: Overridable for testing; defaults to a new
            `AdjudicateContextBuilder` wired to `base_url` via
            `RepoMindClient`. Injecting one directly (with a fake client,
            the same `_FakeRepoMindClient` pattern
            `tests/test_adjudicate/test_context_builder.py` already
            uses) lets tests exercise this function without a real
            network call.
        defender: Overridable for testing; defaults to a new
            `DefenderAgent` (same DI convention every agent already
            supports via its own `llm_client` param - see
            `adjudicate.agents.defender`).
        prosecutor: Overridable for testing; defaults to a new `ProsecutorAgent`.
        judge: Overridable for testing; defaults to a new `JudgeAgent`.

    Yields:
        One event dict per stage, in pipeline order: ``context`` ->
        ``defender`` -> ``claims`` -> one ``claim_verified`` per claim ->
        zero or more ``rebuttal`` -> ``judge`` -> ``done``. An ``error``
        event (and early return) replaces everything from the failing
        stage onward if any stage raises.
    """
    sandbox_path: Path | None = None
    try:
        if context_builder is None:
            context_builder = AdjudicateContextBuilder(RepoMindClient(base_url=base_url), repo_id)
        context_bundle = context_builder.build(diff)
        yield _context_event(context_bundle)

        if not context_bundle.changed_functions:
            yield {
                "type": "error",
                "message": "No changed function in this diff could be resolved against the indexed "
                "repository's call graph - the diff may target a file that hasn't been indexed, or lines "
                "that don't fall inside any function/class this repository's graph knows about.",
            }
            return

        defender = defender or DefenderAgent()
        justification = defender.draft_justification(context_bundle, diff)
        yield {"type": "defender", "justification": justification}

        prosecutor = prosecutor or ProsecutorAgent()
        claims = prosecutor.raise_concerns(context_bundle, diff, justification)
        yield {"type": "claims", "claims": [_claim_event(claim, i) for i, claim in enumerate(claims)]}

        verified: list[VerificationResult] = []
        if claims:
            sandbox_path = _materialize_sandbox(local_path, diff)
            for i, claim in enumerate(claims):
                result = verify_claim(claim, repo_path=sandbox_path)
                verified.append(result)
                yield _verification_event(i, result)

        # Mirrors adjudicate.orchestrator.rebuttal_loop.run_rebuttal_loop's own loop body
        # exactly (see this module's docstring) - only the mid-loop yield is new.
        transcript = [justification]
        ended_by = "cap"
        for round_index in range(1, MAX_REBUTTAL_ROUNDS + 1):
            rebuttal = defender.rebut(context_bundle, diff, transcript[-1], verified)
            transcript.append(rebuttal)
            round_number = round_index + 1
            yield {"type": "rebuttal", "round": round_number, "text": rebuttal}
            if _is_resolved(verified):
                ended_by = "resolution"
                break

        judge = judge or JudgeAgent()
        verdict = judge.judge(context_bundle, diff, transcript, verified, ended_by)
        yield {
            "type": "judge",
            "verdict": verdict.verdict.value,
            "confidence": verdict.confidence,
            "cited_evidence": verdict.cited_evidence,
            "minority_report": verdict.minority_report,
        }
        yield {"type": "done"}

    except RepoMindError as exc:
        logger.error("Live review failed for repository %s: %s", repo_id, exc)
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for a live streaming
        # endpoint: any unexpected failure must still end the stream with a clean "error"
        # event, not an unhandled exception mid-stream that leaves the frontend hanging on
        # a connection that silently died. Logged distinctly from the expected-RepoMindError
        # case above so an unexpected bug here is still loud in the server log.
        logger.error("Live review failed for repository %s with an unexpected error: %s", repo_id, exc, exc_info=True)
        yield {"type": "error", "message": f"Unexpected error: {exc}"}
    finally:
        if sandbox_path is not None:
            shutil.rmtree(sandbox_path, ignore_errors=True)
