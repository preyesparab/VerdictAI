"""Sandboxed code execution for the Verifier (Phase 28).

`run_sandboxed` runs a `ProsecutorClaim.proposed_test` snippet as a real
subprocess against a real (already diff-applied) source tree - never
`exec()`/`eval()` in-process, which would run untrusted/LLM-authored code
directly inside this program with no boundary at all.

Two protections, both real but explicitly *not* a full sandbox:

1. **A hard wall-clock timeout** (`subprocess.run(..., timeout=...)`) -
   this is a genuine guarantee; a hung or infinite-looping test cannot
   block the Verifier indefinitely.
2. **Best-effort network denial** - the generated script is prefixed
   with a small guard that monkeypatches the language's own socket/HTTP
   primitives (`socket.socket.connect`/Node's `net.Socket.connect` and
   `http(s).request`/`.get`) to raise immediately. This blocks the
   realistic "test code accidentally calls out to the internet" case for
   well-behaved Python/Node code, but is **not a hard security
   boundary** - a determined subprocess could still reach the network
   through lower-level mechanisms (raw syscalls via `ctypes`, a
   different interpreter, etc.). Genuine isolation (a real network
   namespace, a memory cgroup, a read-only filesystem) needs OS-level
   sandboxing - Docker, specifically - which is not set up in this
   environment. Flagged here, not silently pretended away: **Docker-based
   sandboxing is a real follow-up for this phase**, not something this
   module claims to already provide.

Memory limiting (`adjudicate.config.AdjudicateSettings.VERIFIER_SANDBOX_MEMORY_LIMIT_MB`,
declared since Phase 23) is likewise not enforced here for the same
reason - real enforcement needs a cgroup (Linux) or a Job Object
(Windows), both meaningfully out of scope for a subprocess-only
implementation. Left inert, as it already was.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 15.0
"""Matches this phase's own suggested range (10-15s) - short enough that a hung
sandboxed test can't stall verification, long enough for a real test to run."""

_PYTHON_NETWORK_GUARD = """\
import socket as _verifier_socket


def _verifier_network_blocked(*_args, **_kwargs):
    raise OSError("network access is disabled inside the Verifier sandbox")


_verifier_socket.socket.connect = _verifier_network_blocked
_verifier_socket.create_connection = _verifier_network_blocked
"""

_JAVASCRIPT_NETWORK_GUARD = """\
const _verifierNet = require("net");
const _verifierHttp = require("http");
const _verifierHttps = require("https");

function _verifierNetworkBlocked() {
  throw new Error("network access is disabled inside the Verifier sandbox");
}

_verifierNet.Socket.prototype.connect = _verifierNetworkBlocked;
_verifierHttp.request = _verifierNetworkBlocked;
_verifierHttp.get = _verifierNetworkBlocked;
_verifierHttps.request = _verifierNetworkBlocked;
_verifierHttps.get = _verifierNetworkBlocked;
"""

_LANGUAGE_CONFIG: dict[str, tuple[str, str, list[str]]] = {
    # language -> (script filename, network guard prelude, interpreter command)
    "python": ("_verifier_check.py", _PYTHON_NETWORK_GUARD, [sys.executable]),
    "javascript": ("_verifier_check.js", _JAVASCRIPT_NETWORK_GUARD, ["node"]),
}


@dataclass(frozen=True)
class SandboxResult:
    """Raw outcome of one sandboxed run.

    Attributes:
        exit_code: The process's exit code, or None if it timed out
            (never ran to completion).
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: True if `DEFAULT_TIMEOUT_SECONDS` (or the given
            `timeout_seconds`) was exceeded - the process was killed,
            not allowed to finish.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_sandboxed(
    code: str, language: str, cwd: Path, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> SandboxResult:
    """Run `code` as a real subprocess rooted at `cwd`, never `exec()`/`eval()`.

    Args:
        code: The code to run - typically a `ProsecutorClaim.proposed_test`.
        language: ``"python"`` or ``"javascript"``.
        cwd: Working directory the subprocess runs in - should already
            contain the real (diff-applied) source tree the claim
            references, so a `require`/`import` of the actual changed
            file resolves correctly. The generated script is written
            here temporarily and removed afterward.
        timeout_seconds: Hard wall-clock limit. Defaults to
            `DEFAULT_TIMEOUT_SECONDS`.

    Returns:
        The raw result - interpreting exit code 0 as "claim refuted" vs.
        nonzero as "claim confirmed" is the caller's job (see
        `adjudicate.verifier.strategies.verify_via_proposed_test`), not
        this function's - this module only knows how to run code safely,
        not what a result means for any particular claim.

    Raises:
        ValueError: If `language` isn't one of the supported values.
    """
    if language not in _LANGUAGE_CONFIG:
        raise ValueError(f"Unsupported sandbox language: {language!r} (supported: {sorted(_LANGUAGE_CONFIG)})")

    script_name, network_guard, interpreter = _LANGUAGE_CONFIG[language]
    script_path = cwd / script_name
    script_path.write_text(network_guard + "\n" + code, encoding="utf-8")

    try:
        completed = subprocess.run(
            [*interpreter, script_name],
            cwd=cwd,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            exit_code=None,
            stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            timed_out=True,
        )
    finally:
        script_path.unlink(missing_ok=True)

    return SandboxResult(
        exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, timed_out=False
    )
