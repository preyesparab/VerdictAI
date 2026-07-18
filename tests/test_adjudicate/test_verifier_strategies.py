"""Tests for adjudicate.verifier.strategies.

Real subprocess/bandit/mypy execution throughout - no LLM anywhere in
this module, so these exercise the actual tools against small, fast,
purpose-built fixture files rather than mocking anything.
"""

from __future__ import annotations

from adjudicate.schemas import ClaimType, ProsecutorClaim
from adjudicate.verifier.models import Confidence, VerificationStatus
from adjudicate.verifier.strategies import (
    DISPATCH,
    verify_breaking_change,
    verify_exception_handling,
    verify_security,
    verify_type_mismatch,
    verify_via_proposed_test,
)


def _claim(**overrides: object) -> ProsecutorClaim:
    defaults: dict[str, object] = {
        "claim_type": ClaimType.MISSING_NULL_CHECK,
        "location": "app.py:1",
        "assertion": "test assertion",
        "proposed_test": "",
    }
    defaults.update(overrides)
    return ProsecutorClaim(**defaults)  # type: ignore[arg-type]


class TestVerifyViaProposedTest:
    def test_confirmed_when_test_fails(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text("def get_email(user):\n    return user['email']\n", encoding="utf-8")
        claim = _claim(
            location="app.py:2",
            proposed_test=(
                "from app import get_email\n"
                "result = get_email(None)\n"
                "assert result is None\n"
            ),
        )

        result = verify_via_proposed_test(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.CONFIRMED
        assert result.confidence == Confidence.HIGH
        assert "TypeError" in result.evidence or "Error" in result.evidence

    def test_refuted_when_test_passes(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text(
            "def get_email(user):\n    return user['email'] if user else None\n", encoding="utf-8"
        )
        claim = _claim(
            location="app.py:2",
            proposed_test=(
                "from app import get_email\n"
                "result = get_email(None)\n"
                "assert result is None\n"
            ),
        )

        result = verify_via_proposed_test(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == Confidence.HIGH

    def test_inconclusive_when_no_proposed_test(self, tmp_path) -> None:
        claim = _claim(proposed_test="")
        result = verify_via_proposed_test(claim, repo_path=tmp_path)
        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW

    def test_inconclusive_for_unsupported_language(self, tmp_path) -> None:
        claim = _claim(location="app.rb:1", proposed_test="puts 'hi'")
        result = verify_via_proposed_test(claim, repo_path=tmp_path)
        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW

    def test_inconclusive_on_timeout(self, tmp_path) -> None:
        claim = _claim(proposed_test="import time\ntime.sleep(30)\n")
        result = verify_via_proposed_test(claim, repo_path=tmp_path)
        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW
        assert "timed out" in result.evidence


class TestVerifySecurity:
    def test_confirmed_on_real_bandit_finding(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text(
            "import subprocess\n"
            "def run(cmd):\n"
            "    subprocess.call(cmd, shell=True)\n",
            encoding="utf-8",
        )
        claim = _claim(claim_type=ClaimType.SECURITY, location="app.py:3", assertion="shell=True is unsafe")

        result = verify_security(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.CONFIRMED
        assert result.confidence == Confidence.MEDIUM
        assert "shell" in result.evidence.lower()

    def test_refuted_on_clean_file(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        claim = _claim(claim_type=ClaimType.SECURITY, location="app.py:2")

        result = verify_security(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == Confidence.MEDIUM

    def test_inconclusive_for_non_python_file(self, tmp_path) -> None:
        (tmp_path / "app.js").write_text("console.log('hi');\n", encoding="utf-8")
        claim = _claim(claim_type=ClaimType.SECURITY, location="app.js:1")

        result = verify_security(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW


class TestVerifyTypeMismatch:
    def test_confirmed_on_real_mypy_error(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n\nadd('x', 'y')\n", encoding="utf-8"
        )
        claim = _claim(claim_type=ClaimType.TYPE_MISMATCH, location="app.py:4")

        result = verify_type_mismatch(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.CONFIRMED
        assert result.confidence == Confidence.MEDIUM

    def test_refuted_on_clean_types(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n\nadd(1, 2)\n", encoding="utf-8"
        )
        claim = _claim(claim_type=ClaimType.TYPE_MISMATCH, location="app.py:4")

        result = verify_type_mismatch(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == Confidence.MEDIUM


class TestVerifyBreakingChange:
    def test_confirmed_when_suite_fails(self, tmp_path) -> None:
        (tmp_path / "test_thing.py").write_text("def test_it():\n    assert False\n", encoding="utf-8")
        claim = _claim(claim_type=ClaimType.BREAKING_CHANGE, location="test_thing.py:1")

        result = verify_breaking_change(
            claim, repo_path=tmp_path, test_command=["python", "-m", "pytest", "test_thing.py"]
        )

        assert result.status == VerificationStatus.CONFIRMED
        assert result.confidence == Confidence.MEDIUM

    def test_refuted_when_suite_passes(self, tmp_path) -> None:
        (tmp_path / "test_thing.py").write_text("def test_it():\n    assert True\n", encoding="utf-8")
        claim = _claim(claim_type=ClaimType.BREAKING_CHANGE, location="test_thing.py:1")

        result = verify_breaking_change(
            claim, repo_path=tmp_path, test_command=["python", "-m", "pytest", "test_thing.py"]
        )

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence == Confidence.MEDIUM

    def test_inconclusive_when_command_not_found(self, tmp_path) -> None:
        claim = _claim(claim_type=ClaimType.BREAKING_CHANGE)
        result = verify_breaking_change(claim, repo_path=tmp_path, test_command=["definitely-not-a-real-command"])
        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW


class TestVerifyExceptionHandling:
    def test_falls_back_to_proposed_test_when_present(self, tmp_path) -> None:
        (tmp_path / "app.py").write_text("def f():\n    pass\n", encoding="utf-8")
        claim = _claim(
            claim_type=ClaimType.EXCEPTION_HANDLING, location="app.py:1",
            proposed_test="raise RuntimeError('claim confirmed')\n",
        )

        result = verify_exception_handling(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.CONFIRMED
        assert result.confidence == Confidence.HIGH
        assert result.strategy == "proposed_test"

    def test_inconclusive_when_no_proposed_test_and_no_tool_fits(self, tmp_path) -> None:
        claim = _claim(claim_type=ClaimType.EXCEPTION_HANDLING, proposed_test="")

        result = verify_exception_handling(claim, repo_path=tmp_path)

        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == Confidence.LOW
        assert result.strategy == "exception_handling_fallback"


def test_dispatch_table_covers_every_claim_type() -> None:
    assert set(DISPATCH.keys()) == set(ClaimType)
