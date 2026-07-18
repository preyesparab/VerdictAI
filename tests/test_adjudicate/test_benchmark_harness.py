"""Tests for adjudicate.benchmark.harness - the non-LLM parts (loading, bundle construction,
sandbox materialization). No LLM calls anywhere in this file."""

from __future__ import annotations

from adjudicate.benchmark.harness import CASES_DIR, load_all_cases, load_case, materialize_sandbox
from adjudicate.context_builder import ContextBundle


def test_loads_all_thirteen_cases() -> None:
    cases = load_all_cases()
    assert len(cases) == 13


def test_case_set_has_both_buggy_and_clean_cases() -> None:
    cases = load_all_cases()
    buggy = [c for c in cases if c.ground_truth_buggy]
    clean = [c for c in cases if not c.ground_truth_buggy]
    assert len(buggy) >= 5
    assert len(clean) >= 5
    assert len(buggy) + len(clean) == len(cases)


def test_every_case_id_is_unique() -> None:
    cases = load_all_cases()
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_non_empty_rationale() -> None:
    for case in load_all_cases():
        assert case.ground_truth_rationale.strip(), f"{case.case_id} has no rationale"


def test_every_case_has_a_context_bundle_with_at_least_one_changed_function() -> None:
    for case in load_all_cases():
        assert isinstance(case.context_bundle, ContextBundle)
        assert len(case.context_bundle.changed_functions) >= 1, f"{case.case_id} has no changed_functions"


def test_load_case_parses_diff_field_directly() -> None:
    case = load_case(CASES_DIR / "divide_by_zero_bug.json")
    assert "def average" in case.diff


def test_load_case_resolves_diff_file_reference() -> None:
    case = load_case(CASES_DIR / "nutriforge_register_login_reused.json")
    assert "password.length < 8" in case.diff


def test_materialize_sandbox_writes_synthetic_target_file(tmp_path) -> None:
    case = load_case(CASES_DIR / "divide_by_zero_bug.json")
    materialize_sandbox(case, tmp_path)
    written = (tmp_path / "average.py").read_text(encoding="utf-8")
    assert "def average" in written


def test_materialize_sandbox_applies_real_diff_and_stubs(tmp_path) -> None:
    case = load_case(CASES_DIR / "nutriforge_register_login_reused.json")
    materialize_sandbox(case, tmp_path)
    controller = tmp_path / "server" / "src" / "controllers" / "auth.controller.js"
    assert controller.exists()
    content = controller.read_text(encoding="utf-8")
    assert "password.length < 8" in content
    assert "lastLoginAt" in content
    assert (tmp_path / "server" / "src" / "models" / "User.js").exists()
    assert (tmp_path / "node_modules" / "jsonwebtoken" / "index.js").exists()


def test_materialize_sandbox_for_colorama_case(tmp_path) -> None:
    case = load_case(CASES_DIR / "colorama_reset_all_reused.json")
    materialize_sandbox(case, tmp_path)
    initialise = tmp_path / "colorama" / "initialise.py"
    assert initialise.exists()
    assert "except OSError" in initialise.read_text(encoding="utf-8")
