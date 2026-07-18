"""Structured claim schema for the Prosecutor's concerns (Phase 27), and the Judge's verdict schema (Phase 30).

Phase 26's Prosecutor produced free-text prose - readable, but nothing
Phase 28's Verifier could mechanically check. This module defines the
fixed shape every concern must now take (`ProsecutorClaim`), grounded in
the actual concern categories the Prosecutor raised in its real Phase 26
output (see `docs/state/PROGRESS.md`'s Phase 26 entry), not a speculative
taxonomy:

- **colorama, `reset_all`**: an overly-broad ``except OSError: pass``
  masking unrelated errors (`ClaimType.EXCEPTION_HANDLING`), and neither
  cited test exercising the new exception path (`ClaimType.UNTESTED_BRANCH`).
- **NutriForge, `register`**: no type check before calling `.length` on
  `password` (`ClaimType.TYPE_MISMATCH`).
- **NutriForge, `login`**: a (factually wrong, but structurally the same
  *category* of claim) assertion that `user` isn't null-checked before
  `.save()` (`ClaimType.MISSING_NULL_CHECK`) - kept as a category despite
  that one instance being false, since a schema's job is to make a claim
  *checkable*, not to only accept claims that turn out true. This is the
  session's own ground-truth test case for Phase 28's Verifier: a
  structurally valid, well-formed claim that is nonetheless wrong, and
  must be mechanically refuted, not just parsed.
- **NutriForge, both functions**: no tests reference either function at
  all - also `ClaimType.UNTESTED_BRANCH` (a whole change with zero
  coverage is treated as the same category as a single untested branch
  within an otherwise-tested function, not a separate category, since
  only one example of the "whole-function" shape has been seen so far).

`BREAKING_CHANGE` and `SECURITY` were *not* included in Phase 27 - neither
had actually appeared in real Prosecutor output yet, and adding them then
would have been exactly the "speculative list" that phase was explicitly
told to avoid. Added now, in Phase 28, for a concrete, non-speculative
reason instead: the Verifier's dispatch table is explicitly specified to
route these two categories to their own strategies (`bandit` for
`SECURITY`, "run the existing test suite" for `BREAKING_CHANGE`), so the
enum needs to exist for the dispatch table to have somewhere to route
them - a purely additive change, like `models.schemas.RetrievalSource.LOCATION`
in Phase 20. The Prosecutor's own system prompt (Phase 27) is *not*
changed here and still only describes the original four categories -
it does not yet generate `BREAKING_CHANGE`/`SECURITY` claims itself;
that is a separate, not-yet-needed follow-up, flagged rather than done
speculatively alongside this change.

`Verdict`/`JudgeVerdict`/`JudgeVerdictModel`/`parse_verdict` (added Phase
30) follow the exact same dataclass/pydantic-model/parser split as
`ProsecutorClaim`, for the same reason - one LLM-response boundary, one
place pydantic is allowed to leak into. `Verdict`'s three values
(`approve`/`reject`/`needs_human_review`) don't need the same "grounded
in real output" justification `ClaimType`'s did - they're the fixed set
the task itself specifies, not an open-ended taxonomy to avoid guessing
at.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, ValidationError

_LOCATION_RE = re.compile(r"^(?P<file>.+):(?P<line>\d+)$")


class ClaimType(str, Enum):
    """The kind of problem one `ProsecutorClaim` raises.

    The first four values were grounded in a real Phase 26 Prosecutor
    output - see this module's docstring for exactly which concern
    produced which value. `BREAKING_CHANGE`/`SECURITY` were added in
    Phase 28 for the Verifier's dispatch table - see this module's
    docstring for why that's a different justification than the first
    four, not a relapse into a speculative list.
    """

    MISSING_NULL_CHECK = "missing_null_check"
    """A value that can be null/undefined/None is used without a guard."""

    UNTESTED_BRANCH = "untested_branch"
    """New logic (a branch, or an entire change) with no covering test."""

    TYPE_MISMATCH = "type_mismatch"
    """Code assumes a value's type without checking it first."""

    EXCEPTION_HANDLING = "exception_handling"
    """Exception handling that is missing, too broad, or silently swallows errors."""

    BREAKING_CHANGE = "breaking_change"
    """The change alters behavior an existing caller/test relies on."""

    SECURITY = "security"
    """A concrete security weakness (e.g. injection, unsafe deserialization, hardcoded secret)."""


@dataclass(frozen=True)
class ProsecutorClaim:
    """One structured, falsifiable concern raised by the Prosecutor (Phase 27).

    This is the shape used everywhere *except* at the LLM-response
    boundary itself (see `ProsecutorClaimModel` for that) - a plain
    frozen dataclass, matching every other schema in `models/schemas.py`.

    Attributes:
        claim_type: The category of problem - see `ClaimType`.
        location: ``"<file_path>:<line>"``. `parse_claims` enforces that
            `file_path` is one of the `ContextBundle`'s actual
            `changed_functions` - a claim about any other file is
            rejected, not just discouraged by the prompt.
        assertion: The specific, checkable claim itself - e.g. "`user`
            may be null here if `User.findOne` returns no match" - never
            generic advice.
        proposed_test: Code (a test body, or a short repro) that would
            concretely prove or disprove `assertion` if run - what
            Phase 28's Verifier is meant to actually execute.
    """

    claim_type: ClaimType
    location: str
    assertion: str
    proposed_test: str


class ProsecutorClaimModel(BaseModel):
    """Pydantic mirror of `ProsecutorClaim`, used only to drive the LLM's structured output.

    Passed to `generation.llm_client.LLMClient.complete` as
    ``response_schema=list[ProsecutorClaimModel]`` - Gemini's structured-
    output mode needs a pydantic model (or `list[Model]`) to constrain
    its decoding to, which a plain dataclass can't provide. Kept separate
    from `ProsecutorClaim` so pydantic stays confined to this one
    boundary rather than spreading through the rest of Adjudicate, which
    otherwise follows `models/schemas.py`'s plain-dataclass convention
    throughout.
    """

    claim_type: ClaimType
    location: str = Field(min_length=1)
    assertion: str = Field(min_length=1)
    proposed_test: str = Field(min_length=1)


class ClaimParsingError(ValueError):
    """Raised when an LLM's raw response cannot be parsed into valid `ProsecutorClaim`s.

    Kept distinct from `core.exceptions.LLMGenerationError` (a
    provider/network failure) - this is a contract violation on a
    response that came back successfully, and the Prosecutor's retry
    loop needs to tell the two apart: a `ClaimParsingError` is worth
    retrying with a corrective prompt, a provider failure is not.
    """


def parse_claims(raw_json: str, valid_file_paths: set[str]) -> list[ProsecutorClaim]:
    """Parse and validate a raw LLM response into `ProsecutorClaim`s.

    Two layers of validation, both required for every claim to be
    accepted - a single bad claim rejects the whole response (the caller
    retries the entire call, not a partial patch-up):

    1. **Structural**: `raw_json` must be a JSON array of objects, each
       matching `ProsecutorClaimModel`'s schema (a real `ClaimType`
       value, non-empty string fields).
    2. **Groundedness**: every claim's `location` must be
       ``"<file_path>:<line>"``, and `file_path` must be one of
       `valid_file_paths` - mechanically enforcing "never raise a
       concern about a function outside the bundle", rather than only
       asking the model nicely in the prompt.

    Args:
        raw_json: The LLM's raw response text - expected to be a JSON
            array (Gemini's `response_schema`-constrained output).
        valid_file_paths: `file_path` of every `ChangedFunction` in the
            `ContextBundle` this response was generated for.

    Returns:
        One `ProsecutorClaim` per parsed array entry, in the model's
        given order. Empty if the model legitimately found nothing to
        raise.

    Raises:
        ClaimParsingError: If `raw_json` is not valid JSON, is not a
            JSON array (or a single-key object wrapping one - see below),
            any entry fails schema validation, or any entry's `location`
            doesn't reference one of `valid_file_paths`.
    """
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ClaimParsingError(f"Response is not valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        # Provider-shape accommodation, not a guess: Groq's (and any OpenAI-
        # compatible) `response_format={"type": "json_object"}` mode can only
        # constrain the top-level response to an *object*, never a bare array -
        # unlike Gemini's `response_schema=list[Model]`, which can. Asked for a
        # JSON array regardless (see the system prompt), the model wraps it in
        # the only shape the API allows - confirmed live: Groq returns exactly
        # `{"claims": [...]}` for this prompt. Unwrapped only when unambiguous
        # (exactly one key, whose value is a list) - anything else still
        # rejects below, not silently guessed at.
        list_valued_keys = [key for key, value in payload.items() if isinstance(value, list)]
        if len(payload) == 1 and len(list_valued_keys) == 1:
            payload = payload[list_valued_keys[0]]
        else:
            raise ClaimParsingError(
                f"Expected a JSON array of claims (or a single-key object wrapping one), "
                f"got an object with keys {list(payload.keys())}"
            )

    if not isinstance(payload, list):
        raise ClaimParsingError(f"Expected a JSON array of claims, got {type(payload).__name__}")

    claims: list[ProsecutorClaim] = []
    for index, item in enumerate(payload):
        try:
            model = ProsecutorClaimModel.model_validate(item)
        except ValidationError as exc:
            raise ClaimParsingError(f"Claim {index} failed schema validation: {exc}") from exc

        match = _LOCATION_RE.match(model.location)
        if not match:
            raise ClaimParsingError(
                f"Claim {index}'s location {model.location!r} is not in '<file_path>:<line>' form"
            )
        file_path = match.group("file")
        if file_path not in valid_file_paths:
            raise ClaimParsingError(
                f"Claim {index} references {file_path!r}, which is not one of this diff's changed_functions "
                f"({sorted(valid_file_paths)}) - the Prosecutor may not raise concerns about functions "
                "outside the bundle it was given"
            )

        claims.append(
            ProsecutorClaim(
                claim_type=model.claim_type,
                location=model.location,
                assertion=model.assertion,
                proposed_test=model.proposed_test,
            )
        )

    return claims


class Verdict(str, Enum):
    """The Judge's (Phase 30) final ruling on a reviewed change."""

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass(frozen=True)
class JudgeVerdict:
    """The Judge's (Phase 30) final, structured ruling on a reviewed change.

    Attributes:
        verdict: `Verdict.APPROVE`/`REJECT`/`NEEDS_HUMAN_REVIEW`.
        confidence: 0.0-1.0 - calibrated to how much of the case was
            actually mechanically resolved (Phase 28's Verifier results,
            Phase 29's rebuttal-loop termination reason), not to how
            articulate the Defender sounded.
        cited_evidence: The specific evidence (drawn from the verified
            claims or the rebuttal transcript) the verdict is actually
            based on - never invented.
        minority_report: Non-None only when genuine, unresolved
            uncertainty remains for a human reviewer to look at (e.g. the
            rebuttal loop hit its round cap with a claim still
            INCONCLUSIVE) - not populated reflexively on every verdict.
    """

    verdict: Verdict
    confidence: float
    cited_evidence: list[str]
    minority_report: str | None


class JudgeVerdictModel(BaseModel):
    """Pydantic mirror of `JudgeVerdict`, used only to drive the LLM's structured output.

    Same split as `ProsecutorClaimModel`/`ProsecutorClaim` (Phase 27) -
    pydantic stays confined to this one LLM-response boundary rather than
    spreading through the rest of Adjudicate.
    """

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    cited_evidence: list[str]
    minority_report: str | None = None


class VerdictParsingError(ValueError):
    """Raised when an LLM's raw response cannot be parsed into a valid `JudgeVerdict`.

    Kept distinct from `core.exceptions.LLMGenerationError` for the same
    reason as `ClaimParsingError` - a malformed response is a contract
    violation on a call that technically succeeded, not a provider
    failure.
    """


def parse_verdict(raw_json: str) -> JudgeVerdict:
    """Parse and validate a raw LLM response into a `JudgeVerdict`.

    Structural validation only (a single JSON object matching
    `JudgeVerdictModel`'s schema) - unlike `parse_claims`, there is no
    location/bundle-membership check to make here, since a verdict
    doesn't reference a file:line the way a claim does.

    Args:
        raw_json: The LLM's raw response text - expected to be a JSON
            object (Gemini's `response_schema`-constrained output).

    Returns:
        The parsed `JudgeVerdict`.

    Raises:
        VerdictParsingError: If `raw_json` is not valid JSON, is not a
            JSON object, or fails schema validation.
    """
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise VerdictParsingError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise VerdictParsingError(f"Expected a JSON object, got {type(payload).__name__}")

    try:
        model = JudgeVerdictModel.model_validate(payload)
    except ValidationError as exc:
        raise VerdictParsingError(f"Verdict failed schema validation: {exc}") from exc

    return JudgeVerdict(
        verdict=model.verdict,
        confidence=model.confidence,
        cited_evidence=model.cited_evidence,
        minority_report=model.minority_report,
    )
