"""Tests for adjudicate.orchestrator.live_review.run_live_review.

Real orchestration, real rebuttal-loop termination logic, and real
sandboxed claim verification (a real subprocess actually runs) - only
the LLM calls are faked (`DefenderAgent`/`ProsecutorAgent`/`JudgeAgent`
constructed with a fake `llm_client`, the same convention every other
agent test in this package already uses), matching this session's own
LLM-quota-discipline memory: mock the LLM boundary, never the real logic
around it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from adjudicate.agents.defender import DefenderAgent
from adjudicate.agents.judge import JudgeAgent
from adjudicate.agents.prosecutor import ProsecutorAgent
from adjudicate.context_builder import AdjudicateContextBuilder
from adjudicate.orchestrator.live_review import run_live_review
from core.exceptions import RetrievalError
from generation.llm_client import LLMCompletion


class _FakeRepoMindClient:
    """Same in-memory `RepoMindClient` stand-in `test_context_builder.py` already established."""

    def __init__(
        self,
        contexts: dict[tuple[str, int], dict[str, Any]],
        blast_radii: dict[str, dict[str, Any]],
    ) -> None:
        self._contexts = contexts
        self._blast_radii = blast_radii

    def get_context(self, repo_id: str, file: str, line: int) -> dict[str, Any]:
        key = (file, line)
        if key not in self._contexts:
            raise RetrievalError(f"no chunk at {file}:{line}")
        return self._contexts[key]

    def get_blast_radius(self, repo_id: str, focus_node: str, hops: int = 2) -> dict[str, Any]:
        return self._blast_radii[focus_node]

    def search(self, repo_id: str, query: str) -> list[dict[str, Any]]:
        return []


class _FakeLLMClient:
    """Returns each entry of `responses` in order; supports the `response_schema` kwarg every agent passes."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Any]] = []

    def complete(self, system_prompt: str, user_prompt: str, response_schema: Any | None = None) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt, response_schema))
        text = self._responses[len(self.calls) - 1]
        return LLMCompletion(text=text, prompt_tokens=10, completion_tokens=5, total_tokens=15, model_name="fake")


def _context_json(chunk_id: str, file_path: str, function_name: str) -> dict[str, Any]:
    return {
        "matched_chunks": [
            {
                "chunk_id": chunk_id, "file_path": file_path, "function_name": function_name,
                "class_name": None, "chunk_type": "function",
            }
        ]
    }


_NEW_FILE_DIFF = (
    "--- /dev/null\n+++ b/user_service.py\n@@ -0,0 +1,3 @@\n"
    '+def get_user_email(user):\n+    """docstring"""\n+    return user["email"]\n'
)

_CLAIM_JSON = json.dumps(
    [
        {
            "claim_type": "missing_null_check",
            "location": "user_service.py:3",
            "assertion": "user may be None, which raises when subscripted.",
            # No try/except: the sandbox convention is "let it crash if the bug is real" -
            # verify_via_proposed_test treats a non-zero exit (an uncaught real TypeError
            # here) as CONFIRMED, a clean exit as REFUTED. A try/except swallowing the
            # exception would invert this (silently REFUTE a real bug).
            "proposed_test": "import user_service\nuser_service.get_user_email(None)\n",
        }
    ]
)


def _judge_json(verdict: str, confidence: float, minority_report: str | None = None) -> str:
    return json.dumps(
        {"verdict": verdict, "confidence": confidence, "cited_evidence": ["real evidence"], "minority_report": minority_report}
    )


def _make_context_builder() -> AdjudicateContextBuilder:
    # _NEW_FILE_DIFF's hunk header is "@@ -0,0 +1,3 @@" - a pure-addition hunk (old-file
    # length 0), so parse_diff's own fallback sets old_start_line/old_end_line to the hunk's
    # old-file insertion point, 0 - _find_enclosing_chunk looks up line 0, not the new-file
    # line 1 (see context_builder.py's own ChangedLocation/parse_diff docstrings for why the
    # *old*-file side is what's looked up).
    client = _FakeRepoMindClient(
        contexts={("user_service.py", 0): _context_json("chunk-1", "user_service.py", "get_user_email")},
        blast_radii={
            "chunk-1": {
                "nodes": [
                    {"id": "chunk-1", "node_kind": "chunk", "file_path": "user_service.py",
                     "function_name": "get_user_email", "class_name": None}
                ],
                "edges": [],
            }
        },
    )
    return AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)


def test_full_pipeline_resolves_in_one_rebuttal_round(tmp_path) -> None:
    local_path = tmp_path / "repo"
    local_path.mkdir()

    defender = DefenderAgent(llm_client=_FakeLLMClient(["Justification text.", "I concede - confirmed by the test."]))
    prosecutor = ProsecutorAgent(llm_client=_FakeLLMClient([_CLAIM_JSON]))
    judge = JudgeAgent(llm_client=_FakeLLMClient([_judge_json("reject", 0.9)]))

    events = list(
        run_live_review(
            "repo-1", local_path, _NEW_FILE_DIFF, base_url="http://testserver/",
            context_builder=_make_context_builder(), defender=defender, prosecutor=prosecutor, judge=judge,
        )
    )

    types = [e["type"] for e in events]
    assert types == ["context", "defender", "claims", "claim_verified", "rebuttal", "judge", "done"]

    context_event = events[0]
    assert context_event["changed_functions"][0]["function_name"] == "get_user_email"

    assert events[1] == {"type": "defender", "justification": "Justification text."}

    claims_event = events[2]
    assert claims_event["claims"][0]["claim_type"] == "missing_null_check"
    assert claims_event["claims"][0]["index"] == 0

    # Real sandbox execution, not mocked: get_user_email(None) genuinely raises TypeError,
    # so the proposed_test genuinely fails (SystemExit(1) is never reached) -> CONFIRMED.
    verified_event = events[3]
    assert verified_event == {
        "type": "claim_verified", "index": 0, "status": "confirmed", "confidence": "high",
        "evidence": verified_event["evidence"], "strategy": "proposed_test",
    }
    assert "TypeError" in verified_event["evidence"]  # the real, uncaught exception's traceback

    rebuttal_event = events[4]
    assert rebuttal_event == {"type": "rebuttal", "round": 2, "text": "I concede - confirmed by the test."}

    judge_event = events[5]
    assert judge_event == {
        "type": "judge", "verdict": "reject", "confidence": 0.9,
        "cited_evidence": ["real evidence"], "minority_report": None,
    }

    assert events[6] == {"type": "done"}


def test_no_resolvable_changed_function_yields_error_and_stops() -> None:
    client = _FakeRepoMindClient(contexts={}, blast_radii={})
    context_builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)
    diff = "--- a/nowhere.py\n+++ b/nowhere.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"

    events = list(run_live_review("repo-1", None, diff, base_url="http://testserver/", context_builder=context_builder))

    assert [e["type"] for e in events] == ["context", "error"]
    assert "could be resolved" in events[1]["message"]


def test_no_claims_still_runs_one_rebuttal_round_and_a_judge_call(tmp_path) -> None:
    """Matches adjudicate.orchestrator.rebuttal_loop's own real behavior: even zero claims still
    get one rebuttal round (there's nothing to concede, but the loop is unconditional) - see
    tests/test_adjudicate/test_rebuttal_loop.py::test_resolves_immediately_when_no_claims_at_all."""
    local_path = tmp_path / "repo"
    local_path.mkdir()

    defender = DefenderAgent(llm_client=_FakeLLMClient(["Justification.", "Nothing to respond to."]))
    prosecutor = ProsecutorAgent(llm_client=_FakeLLMClient([json.dumps([])]))
    judge = JudgeAgent(llm_client=_FakeLLMClient([_judge_json("approve", 1.0)]))

    events = list(
        run_live_review(
            "repo-1", local_path, _NEW_FILE_DIFF, base_url="http://testserver/",
            context_builder=_make_context_builder(), defender=defender, prosecutor=prosecutor, judge=judge,
        )
    )

    types = [e["type"] for e in events]
    assert types == ["context", "defender", "claims", "rebuttal", "judge", "done"]
    assert events[2]["claims"] == []
    assert events[3] == {"type": "rebuttal", "round": 2, "text": "Nothing to respond to."}
    assert events[4]["verdict"] == "approve"


def test_still_inconclusive_claim_hits_the_round_cap(tmp_path) -> None:
    local_path = tmp_path / "repo"
    local_path.mkdir()

    # A real, disclosed gap in run_live_review's own dispatch, exercised here rather than
    # hidden: verify_claim's `test_command` (required for a real BREAKING_CHANGE verdict) is
    # never supplied by run_live_review (there's no general "run this repo's test suite"
    # command to infer for an arbitrary indexed repository), so every BREAKING_CHANGE claim
    # verifies as genuinely INCONCLUSIVE via the real, unmocked dispatch logic - not a claim
    # this test artificially malforms to force the outcome.
    inconclusive_claim_json = json.dumps(
        [
            {
                "claim_type": "breaking_change",
                "location": "user_service.py:3",
                "assertion": "an existing caller may rely on the old signature",
                "proposed_test": "import user_service\nuser_service.get_user_email({'email': 'a@b.com'})\n",
            }
        ]
    )
    defender = DefenderAgent(
        llm_client=_FakeLLMClient(["Justification.", "Rebuttal round 1.", "Rebuttal round 2."])
    )
    prosecutor = ProsecutorAgent(llm_client=_FakeLLMClient([inconclusive_claim_json]))
    judge = JudgeAgent(llm_client=_FakeLLMClient([_judge_json("needs_human_review", 0.4, "Genuinely unresolved.")]))

    events = list(
        run_live_review(
            "repo-1", local_path, _NEW_FILE_DIFF, base_url="http://testserver/",
            context_builder=_make_context_builder(), defender=defender, prosecutor=prosecutor, judge=judge,
        )
    )

    rebuttal_events = [e for e in events if e["type"] == "rebuttal"]
    assert [e["round"] for e in rebuttal_events] == [2, 3]  # both MAX_REBUTTAL_ROUNDS used - never resolved
    verified_event = next(e for e in events if e["type"] == "claim_verified")
    assert verified_event["status"] == "inconclusive"
    judge_event = next(e for e in events if e["type"] == "judge")
    assert judge_event["verdict"] == "needs_human_review"
    assert judge_event["minority_report"] == "Genuinely unresolved."


def test_repomind_error_yields_error_event_not_an_unhandled_exception() -> None:
    """`_find_enclosing_chunk` catches `RetrievalError` from `get_context` internally (so that
    path alone would just look like "no changed function resolved", not an exception) - raise
    it from `get_blast_radius` instead, which `AdjudicateContextBuilder.build` does not wrap in
    a try/except, so it genuinely propagates out of `.build()` uncaught."""

    class _RaisingClient:
        def get_context(self, repo_id: str, file: str, line: int) -> dict[str, Any]:
            return _context_json("chunk-1", "x.py", "target_fn")

        def get_blast_radius(self, repo_id: str, focus_node: str, hops: int = 2) -> dict[str, Any]:
            raise RetrievalError("graph lookup failed")

    context_builder = AdjudicateContextBuilder(_RaisingClient(), repo_id="repo-1", hops=1)
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"

    events = list(run_live_review("repo-1", None, diff, base_url="http://testserver/", context_builder=context_builder))

    assert len(events) == 1
    assert events[0]["type"] == "error"
