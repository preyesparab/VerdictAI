"""Tests for adjudicate.schemas."""

from __future__ import annotations

import json

import pytest

from adjudicate.schemas import (
    ClaimParsingError,
    ClaimType,
    JudgeVerdict,
    ProsecutorClaim,
    Verdict,
    VerdictParsingError,
    parse_claims,
    parse_verdict,
)

_VALID_PATHS = {"server/src/controllers/auth.controller.js"}


def _raw(claims: list[dict[str, object]]) -> str:
    return json.dumps(claims)


def test_parses_valid_single_claim() -> None:
    raw = _raw(
        [
            {
                "claim_type": "missing_null_check",
                "location": "server/src/controllers/auth.controller.js:47",
                "assertion": "user may be null here if User.findOne returns no match",
                "proposed_test": "expect(() => login(reqWithUnknownEmail)).not.toThrow();",
            }
        ]
    )

    claims = parse_claims(raw, _VALID_PATHS)

    assert claims == [
        ProsecutorClaim(
            claim_type=ClaimType.MISSING_NULL_CHECK,
            location="server/src/controllers/auth.controller.js:47",
            assertion="user may be null here if User.findOne returns no match",
            proposed_test="expect(() => login(reqWithUnknownEmail)).not.toThrow();",
        )
    ]


def test_empty_array_is_valid_and_means_no_concerns() -> None:
    assert parse_claims("[]", _VALID_PATHS) == []


def test_rejects_invalid_json_syntax() -> None:
    with pytest.raises(ClaimParsingError, match="not valid JSON"):
        parse_claims("{not json", _VALID_PATHS)


def test_rejects_non_array_json() -> None:
    with pytest.raises(ClaimParsingError, match="Expected a JSON array"):
        parse_claims(json.dumps({"claim_type": "missing_null_check"}), _VALID_PATHS)


def test_unwraps_single_key_object_wrapping_an_array() -> None:
    """Groq's (OpenAI-compatible) json_object mode can only return a top-level object, never
    a bare array - it wraps the requested array as `{"claims": [...]}`. Confirmed live against
    the real Groq API (docs/state/PROGRESS.md's Phase 27 closeout entry) - not a guess."""
    raw = json.dumps(
        {
            "claims": [
                {
                    "claim_type": "untested_branch",
                    "location": "server/src/controllers/auth.controller.js:47",
                    "assertion": "no test coverage",
                    "proposed_test": "it(...)",
                }
            ]
        }
    )

    claims = parse_claims(raw, _VALID_PATHS)

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.UNTESTED_BRANCH


def test_rejects_multi_key_object_even_if_one_value_is_a_list() -> None:
    """Only an unambiguous single-key-wrapping-a-list object is unwrapped - anything else
    still rejects rather than guessing which key is the real claims list."""
    raw = json.dumps({"claims": [], "other_field": [1, 2]})
    with pytest.raises(ClaimParsingError, match="Expected a JSON array"):
        parse_claims(raw, _VALID_PATHS)


def test_rejects_single_key_object_whose_value_is_not_a_list() -> None:
    raw = json.dumps({"claims": "not a list"})
    with pytest.raises(ClaimParsingError, match="Expected a JSON array"):
        parse_claims(raw, _VALID_PATHS)


def test_rejects_unknown_claim_type() -> None:
    raw = _raw(
        [
            {
                "claim_type": "not_a_real_category",
                "location": "server/src/controllers/auth.controller.js:47",
                "assertion": "x",
                "proposed_test": "y",
            }
        ]
    )
    with pytest.raises(ClaimParsingError, match="failed schema validation"):
        parse_claims(raw, _VALID_PATHS)


def test_rejects_missing_required_field() -> None:
    raw = _raw(
        [
            {
                "claim_type": "missing_null_check",
                "location": "server/src/controllers/auth.controller.js:47",
                "assertion": "x",
                # proposed_test omitted
            }
        ]
    )
    with pytest.raises(ClaimParsingError, match="failed schema validation"):
        parse_claims(raw, _VALID_PATHS)


def test_rejects_location_not_matching_file_colon_line() -> None:
    raw = _raw(
        [
            {
                "claim_type": "missing_null_check",
                "location": "server/src/controllers/auth.controller.js (somewhere near the end)",
                "assertion": "x",
                "proposed_test": "y",
            }
        ]
    )
    with pytest.raises(ClaimParsingError, match="not in '<file_path>:<line>' form"):
        parse_claims(raw, _VALID_PATHS)


def test_rejects_claim_referencing_a_file_outside_the_bundle() -> None:
    """The mechanical groundedness check: a claim about a file not in changed_functions must be rejected."""
    raw = _raw(
        [
            {
                "claim_type": "missing_null_check",
                "location": "server/src/controllers/some_other_file.js:10",
                "assertion": "x",
                "proposed_test": "y",
            }
        ]
    )
    with pytest.raises(ClaimParsingError, match="not one of this diff's changed_functions"):
        parse_claims(raw, _VALID_PATHS)


def test_parses_multiple_claims_in_order() -> None:
    raw = _raw(
        [
            {
                "claim_type": "type_mismatch",
                "location": "server/src/controllers/auth.controller.js:21",
                "assertion": "first",
                "proposed_test": "t1",
            },
            {
                "claim_type": "untested_branch",
                "location": "server/src/controllers/auth.controller.js:47",
                "assertion": "second",
                "proposed_test": "t2",
            },
        ]
    )

    claims = parse_claims(raw, _VALID_PATHS)

    assert [c.assertion for c in claims] == ["first", "second"]
    assert [c.claim_type for c in claims] == [ClaimType.TYPE_MISMATCH, ClaimType.UNTESTED_BRANCH]


# -- parse_verdict (Phase 30) ---------------------------------------------------


def test_parses_valid_verdict() -> None:
    raw = json.dumps(
        {
            "verdict": "approve",
            "confidence": 0.9,
            "cited_evidence": ["reset_all returned 401 without throwing"],
            "minority_report": None,
        }
    )

    verdict = parse_verdict(raw)

    assert verdict == JudgeVerdict(
        verdict=Verdict.APPROVE, confidence=0.9,
        cited_evidence=["reset_all returned 401 without throwing"], minority_report=None,
    )


def test_parses_verdict_with_minority_report() -> None:
    raw = json.dumps(
        {
            "verdict": "needs_human_review",
            "confidence": 0.4,
            "cited_evidence": ["one claim remains inconclusive"],
            "minority_report": "The exception-handling claim could not be mechanically checked.",
        }
    )

    verdict = parse_verdict(raw)

    assert verdict.minority_report == "The exception-handling claim could not be mechanically checked."


def test_verdict_rejects_invalid_json_syntax() -> None:
    with pytest.raises(VerdictParsingError, match="not valid JSON"):
        parse_verdict("{not json")


def test_rejects_non_object_json() -> None:
    with pytest.raises(VerdictParsingError, match="Expected a JSON object"):
        parse_verdict(json.dumps([{"verdict": "approve"}]))


def test_rejects_unknown_verdict_value() -> None:
    raw = json.dumps(
        {"verdict": "maybe", "confidence": 0.5, "cited_evidence": [], "minority_report": None}
    )
    with pytest.raises(VerdictParsingError, match="failed schema validation"):
        parse_verdict(raw)


def test_rejects_confidence_out_of_range() -> None:
    raw = json.dumps(
        {"verdict": "approve", "confidence": 1.5, "cited_evidence": [], "minority_report": None}
    )
    with pytest.raises(VerdictParsingError, match="failed schema validation"):
        parse_verdict(raw)


def test_verdict_rejects_missing_required_field() -> None:
    raw = json.dumps({"verdict": "approve", "confidence": 0.9})
    with pytest.raises(VerdictParsingError, match="failed schema validation"):
        parse_verdict(raw)
