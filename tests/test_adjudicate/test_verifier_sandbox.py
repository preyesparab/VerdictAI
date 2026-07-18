"""Tests for adjudicate.verifier.sandbox.

Real subprocess execution throughout - there is no LLM anywhere in this
module, so there is nothing to mock; these tests exercise the actual
sandboxing mechanism (timeout, network denial, exit codes) for real.
"""

from __future__ import annotations

import pytest

from adjudicate.verifier.sandbox import run_sandboxed


def test_python_success_exit_code_zero(tmp_path) -> None:
    result = run_sandboxed("print('hello')\n", "python", cwd=tmp_path)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_python_failure_nonzero_exit_code(tmp_path) -> None:
    result = run_sandboxed("raise ValueError('boom')\n", "python", cwd=tmp_path)
    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert not result.timed_out


def test_python_explicit_exit_code(tmp_path) -> None:
    result = run_sandboxed("import sys\nsys.exit(3)\n", "python", cwd=tmp_path)
    assert result.exit_code == 3


def test_python_timeout_is_reported_not_hung(tmp_path) -> None:
    result = run_sandboxed("import time\ntime.sleep(30)\n", "python", cwd=tmp_path, timeout_seconds=1.0)
    assert result.timed_out is True
    assert result.exit_code is None


def test_python_network_access_is_blocked(tmp_path) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80), timeout=2)\n"
        "    print('NETWORK_SUCCEEDED')\n"
        "except OSError as exc:\n"
        "    print('NETWORK_BLOCKED:', exc)\n"
    )
    result = run_sandboxed(code, "python", cwd=tmp_path)
    assert "NETWORK_BLOCKED" in result.stdout
    assert "NETWORK_SUCCEEDED" not in result.stdout


def test_javascript_success_exit_code_zero(tmp_path) -> None:
    result = run_sandboxed("console.log('hello');\n", "javascript", cwd=tmp_path)
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_javascript_failure_nonzero_exit_code(tmp_path) -> None:
    result = run_sandboxed("throw new Error('boom');\n", "javascript", cwd=tmp_path)
    assert result.exit_code != 0
    assert "boom" in result.stderr


def test_javascript_network_access_is_blocked(tmp_path) -> None:
    code = (
        "const http = require('http');\n"
        "try {\n"
        "  http.get('http://example.com', () => {});\n"
        "  console.log('NETWORK_SUCCEEDED');\n"
        "} catch (err) {\n"
        "  console.log('NETWORK_BLOCKED:', err.message);\n"
        "}\n"
    )
    result = run_sandboxed(code, "javascript", cwd=tmp_path)
    assert "NETWORK_BLOCKED" in result.stdout
    assert "NETWORK_SUCCEEDED" not in result.stdout


def test_script_file_removed_after_run(tmp_path) -> None:
    run_sandboxed("print('x')\n", "python", cwd=tmp_path)
    assert list(tmp_path.glob("_verifier_check.*")) == []


def test_unsupported_language_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported sandbox language"):
        run_sandboxed("puts 'hi'", "ruby", cwd=tmp_path)


def test_can_read_real_files_already_in_cwd(tmp_path) -> None:
    """Confirms the sandbox runs rooted at cwd, so a script can require/import a real
    file already placed there - this is how a claim's proposed_test reaches the actual
    diff-applied source it's testing."""
    (tmp_path / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    result = run_sandboxed("from helper import VALUE\nprint(VALUE)\n", "python", cwd=tmp_path)
    assert result.exit_code == 0
    assert "42" in result.stdout
