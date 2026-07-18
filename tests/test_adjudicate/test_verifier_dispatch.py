"""Tests for adjudicate.verifier.verifier.verify_claim (the dispatch entry point)."""

from __future__ import annotations

from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier import Confidence, VerificationStatus, verify_claim


def _claim(**overrides: object) -> ProsecutorClaim:
    defaults: dict[str, object] = {
        "claim_type": ClaimType.MISSING_NULL_CHECK,
        "location": "app.py:1",
        "assertion": "test assertion",
        "proposed_test": "",
    }
    defaults.update(overrides)
    return ProsecutorClaim(**defaults)  # type: ignore[arg-type]


def test_routes_missing_null_check_to_proposed_test_strategy(tmp_path) -> None:
    (tmp_path / "app.py").write_text("X = 1\n", encoding="utf-8")
    claim = _claim(claim_type=ClaimType.MISSING_NULL_CHECK, proposed_test="assert True\n")

    result = verify_claim(claim, repo_path=tmp_path)

    assert result.strategy == "proposed_test"
    assert result.status == VerificationStatus.REFUTED


def test_routes_security_to_bandit(tmp_path) -> None:
    (tmp_path / "app.py").write_text("X = 1\n", encoding="utf-8")
    claim = _claim(claim_type=ClaimType.SECURITY, location="app.py:1")

    result = verify_claim(claim, repo_path=tmp_path)

    assert result.strategy == "bandit"


def test_routes_type_mismatch_to_mypy(tmp_path) -> None:
    (tmp_path / "app.py").write_text("X: int = 1\n", encoding="utf-8")
    claim = _claim(claim_type=ClaimType.TYPE_MISMATCH, location="app.py:1")

    result = verify_claim(claim, repo_path=tmp_path)

    assert result.strategy == "mypy"


def test_breaking_change_without_test_command_is_inconclusive(tmp_path) -> None:
    claim = _claim(claim_type=ClaimType.BREAKING_CHANGE)

    result = verify_claim(claim, repo_path=tmp_path, test_command=None)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.confidence == Confidence.LOW
    assert "test_command" in result.evidence


def test_breaking_change_with_test_command_runs_existing_suite(tmp_path) -> None:
    (tmp_path / "test_thing.py").write_text("def test_it():\n    assert True\n", encoding="utf-8")
    claim = _claim(claim_type=ClaimType.BREAKING_CHANGE)

    result = verify_claim(claim, repo_path=tmp_path, test_command=["python", "-m", "pytest", "test_thing.py"])

    assert result.strategy == "existing_test_suite"
    assert result.status == VerificationStatus.REFUTED


def test_exception_handling_falls_back_correctly(tmp_path) -> None:
    claim = _claim(claim_type=ClaimType.EXCEPTION_HANDLING, proposed_test="")

    result = verify_claim(claim, repo_path=tmp_path)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.strategy == "exception_handling_fallback"
