"""Tests for adjudicate.context_builder."""

from __future__ import annotations

from typing import Any

import pytest

from adjudicate.context_builder import (
    AdjudicateContextBuilder,
    ChangedFunction,
    ChangedLocation,
    ContextBundle,
    GraphNeighbor,
    RelatedTest,
    _fallback_search_query,
    _looks_like_test_file,
    format_changed_functions,
    format_neighbors,
    format_related_tests,
    parse_diff,
)
from core.exceptions import RetrievalError

# -- parse_diff ---------------------------------------------------------------

_SINGLE_HUNK_DIFF = """\
diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,4 @@ def foo():
 line one
-old line
+new line
+another new line
 line three
"""

_MULTI_HUNK_MULTI_FILE_DIFF = """\
diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,4 @@ def foo():
 line one
-old line
+new line
+another new line
 line three
@@ -50 +51,2 @@ def bar():
-old
+new
+extra
diff --git a/src/b.py b/src/b.py
index 333..444 100644
--- a/src/b.py
+++ b/src/b.py
@@ -1,2 +1,2 @@
-print(1)
+print(2)
 print(3)
"""

_NEW_FILE_DIFF = """\
diff --git a/src/new.py b/src/new.py
new file mode 100644
index 000000..111
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,3 @@
+def foo():
+    pass
+
"""

_PURE_DELETION_DIFF = """\
diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,0 @@ def foo():
-line one
-old line
-line three
"""


def test_parses_single_hunk() -> None:
    """The range is tight to the actual +/- edits (line 11-12), not the header's padded 10-13."""
    locations = parse_diff(_SINGLE_HUNK_DIFF)
    assert locations == [
        ChangedLocation(file="src/a.py", start_line=11, end_line=12, old_start_line=11, old_end_line=11)
    ]


def test_parses_multiple_hunks_across_multiple_files() -> None:
    locations = parse_diff(_MULTI_HUNK_MULTI_FILE_DIFF)
    assert locations == [
        ChangedLocation(file="src/a.py", start_line=11, end_line=12, old_start_line=11, old_end_line=11),
        ChangedLocation(file="src/a.py", start_line=51, end_line=52, old_start_line=50, old_end_line=50),
        ChangedLocation(file="src/b.py", start_line=1, end_line=1, old_start_line=1, old_end_line=1),
    ]


def test_new_file_diff_uses_the_new_path_not_dev_null() -> None:
    """A pure-addition hunk (old_lines=0) has no old-touched line, so old_start/end fall back to 0 (the header's insertion point) - harmless, since a brand-new file was never indexed either way."""
    locations = parse_diff(_NEW_FILE_DIFF)
    assert locations == [
        ChangedLocation(file="src/new.py", start_line=1, end_line=3, old_start_line=0, old_end_line=0)
    ]


def test_pure_deletion_hunk_contributes_no_location() -> None:
    """A hunk with new-file length 0 leaves nothing in the new file to look up."""
    assert parse_diff(_PURE_DELETION_DIFF) == []


def test_empty_diff_returns_no_locations() -> None:
    assert parse_diff("") == []


# -- _looks_like_test_file / _fallback_search_query ----------------------------


@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("colorama/tests/initialise_test.py", True),
        ("server/tests/auth.spec.js", True),
        ("src/test_utils.py", True),
        ("server/src/controllers/auth.controller.js", False),
        ("colorama/initialise.py", False),
    ],
)
def test_looks_like_test_file(file_path: str, expected: bool) -> None:
    assert _looks_like_test_file(file_path) is expected


def test_fallback_search_query_combines_name_and_file_stem() -> None:
    chunk = {"function_name": "reset_all", "file_path": "colorama/initialise.py"}
    assert _fallback_search_query(chunk) == "test reset_all initialise"


# -- format_changed_functions / format_neighbors / format_related_tests -------
# Shared by the Defender (Phase 25) and Prosecutor (Phase 26) to render a
# `ContextBundle` into the same prompt text for both agents.


def test_format_changed_functions_empty() -> None:
    assert "none" in format_changed_functions(ContextBundle()).lower()


def test_format_changed_functions_lists_file_and_lines() -> None:
    bundle = ContextBundle(
        changed_functions=[
            ChangedFunction(
                chunk_id="c1", file_path="src/a.py", function_name="foo", class_name=None,
                changed_start_line=10, changed_end_line=12,
            )
        ]
    )
    rendered = format_changed_functions(bundle)
    assert "src/a.py" in rendered and "foo" in rendered and "10-12" in rendered


def test_format_neighbors_empty_uses_given_note() -> None:
    assert format_neighbors([], "nothing here") == "nothing here"


def test_format_neighbors_lists_file_and_kind() -> None:
    neighbor = GraphNeighbor(
        node_id="n1", node_kind="chunk", file_path="src/caller.py", function_name="caller_fn", class_name=None
    )
    rendered = format_neighbors([neighbor], "none")
    assert "src/caller.py" in rendered and "caller_fn" in rendered and "chunk" in rendered


def test_format_related_tests_empty() -> None:
    assert "none found" in format_related_tests([]).lower()


def test_format_related_tests_lists_file_and_found_via() -> None:
    test = RelatedTest(chunk_id="t1", file_path="tests/test_a.py", function_name="test_foo", found_via="graph")
    rendered = format_related_tests([test])
    assert "tests/test_a.py" in rendered and "test_foo" in rendered and "found via graph" in rendered


def test_fallback_search_query_falls_back_to_class_name() -> None:
    chunk = {"function_name": None, "class_name": "AnsiToWin32", "file_path": "colorama/ansitowin32.py"}
    assert _fallback_search_query(chunk) == "test AnsiToWin32 ansitowin32"


# -- AdjudicateContextBuilder.build (orchestration, via a fake client) --------


class _FakeRepoMindClient:
    """In-memory stand-in for `RepoMindClient`, for testing orchestration logic without HTTP."""

    def __init__(
        self,
        contexts: dict[tuple[str, int], dict[str, Any]],
        blast_radii: dict[str, dict[str, Any]],
        search_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._contexts = contexts
        self._blast_radii = blast_radii
        self._search_results = search_results or {}
        self.search_calls: list[str] = []

    def get_context(self, repo_id: str, file: str, line: int) -> dict[str, Any]:
        key = (file, line)
        if key not in self._contexts:
            raise RetrievalError(f"no chunk at {file}:{line}")
        return self._contexts[key]

    def get_blast_radius(self, repo_id: str, focus_node: str, hops: int = 2) -> dict[str, Any]:
        return self._blast_radii[focus_node]

    def search(self, repo_id: str, query: str) -> list[dict[str, Any]]:
        self.search_calls.append(query)
        return self._search_results.get(query, [])


def _context_for(chunk_id: str, file_path: str, function_name: str | None) -> dict[str, Any]:
    return {
        "matched_chunks": [
            {
                "chunk_id": chunk_id, "file_path": file_path, "function_name": function_name,
                "class_name": None, "chunk_type": "function",
            }
        ]
    }


def test_find_enclosing_chunk_skips_non_ast_match_ranked_first() -> None:
    """A smaller SLIDING/PARENT match ranked first must be skipped for the first AST-typed one."""
    client = _FakeRepoMindClient(
        contexts={
            ("src/a.py", 10): {
                "matched_chunks": [
                    {"chunk_id": "sliding-1", "file_path": "src/a.py", "function_name": None, "class_name": None, "chunk_type": "sliding"},
                    {"chunk_id": "ast-1", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None, "chunk_type": "function"},
                ]
            }
        },
        blast_radii={"ast-1": {"nodes": [{"id": "ast-1", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None}], "edges": []}},
    )
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,1 +10,1 @@\n-x\n+y\n"
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert [f.chunk_id for f in bundle.changed_functions] == ["ast-1"]


def test_build_classifies_callers_and_callees_and_dedupes_across_hunks() -> None:
    """Two hunks landing in the same function must produce one changed_function, not two."""
    diff = (
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,2 +10,2 @@\n-x\n+y\n z\n"
        "@@ -15,1 +15,1 @@\n-a\n+b\n"
    )
    client = _FakeRepoMindClient(
        contexts={
            ("src/a.py", 10): _context_for("focus-1", "src/a.py", "target_fn"),
            ("src/a.py", 15): _context_for("focus-1", "src/a.py", "target_fn"),
        },
        blast_radii={
            "focus-1": {
                "nodes": [
                    {"id": "focus-1", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None},
                    {"id": "caller-1", "node_kind": "chunk", "file_path": "src/caller.py", "function_name": "caller_fn", "class_name": None},
                    {"id": "callee-1", "node_kind": "chunk", "file_path": "src/callee.py", "function_name": "callee_fn", "class_name": None},
                ],
                "edges": [
                    {"source": "caller-1", "target": "focus-1", "edge_type": "function_call"},
                    {"source": "focus-1", "target": "callee-1", "edge_type": "function_call"},
                    {"source": "focus-1", "target": "focus-1", "edge_type": "function_call"},  # self-loop, must be excluded
                ],
            }
        },
    )
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert [f.chunk_id for f in bundle.changed_functions] == ["focus-1"]
    assert [c.node_id for c in bundle.callers] == ["caller-1"]
    assert [c.node_id for c in bundle.callees] == ["callee-1"]
    assert bundle.related_tests == []
    # No caller found by the graph looked test-like, so the fallback fires (and
    # correctly finds nothing, since this fake has no search results configured).
    assert client.search_calls == ["test target_fn a"]


def test_build_falls_back_to_search_when_graph_finds_no_test_caller() -> None:
    client = _FakeRepoMindClient(
        contexts={("src/a.py", 10): _context_for("focus-1", "src/a.py", "target_fn")},
        blast_radii={
            "focus-1": {
                "nodes": [
                    {"id": "focus-1", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None},
                    {"id": "caller-1", "node_kind": "chunk", "file_path": "src/caller.py", "function_name": "caller_fn", "class_name": None},
                ],
                "edges": [{"source": "caller-1", "target": "focus-1", "edge_type": "function_call"}],
            }
        },
        search_results={
            "test target_fn a": [
                {"chunk_id": "test-1", "file_path": "tests/test_a.py", "function_name": "test_target_fn", "relevance_score": 5.0},
                {"chunk_id": "not-a-test", "file_path": "src/other.py", "function_name": "unrelated", "relevance_score": 4.0},
            ]
        },
    )
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,1 +10,1 @@\n-x\n+y\n"
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert client.search_calls == ["test target_fn a"]
    assert len(bundle.related_tests) == 1
    assert bundle.related_tests[0].chunk_id == "test-1"
    assert bundle.related_tests[0].found_via == "retrieval_fallback"


def test_build_does_not_fall_back_when_graph_already_found_a_test_caller() -> None:
    client = _FakeRepoMindClient(
        contexts={("src/a.py", 10): _context_for("focus-1", "src/a.py", "target_fn")},
        blast_radii={
            "focus-1": {
                "nodes": [
                    {"id": "focus-1", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None},
                    {"id": "test-caller-1", "node_kind": "chunk", "file_path": "tests/test_a.py", "function_name": "test_it", "class_name": None},
                ],
                "edges": [{"source": "test-caller-1", "target": "focus-1", "edge_type": "function_call"}],
            }
        },
    )
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,1 +10,1 @@\n-x\n+y\n"
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert client.search_calls == []
    assert len(bundle.related_tests) == 1
    assert bundle.related_tests[0].found_via == "graph"


def test_build_skips_location_with_no_enclosing_chunk() -> None:
    client = _FakeRepoMindClient(contexts={}, blast_radii={})
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,1 +10,1 @@\n-x\n+y\n"
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert bundle.changed_functions == []
    assert bundle.callers == []
    assert bundle.callees == []
    assert bundle.related_tests == []


def test_build_tries_end_line_when_start_line_has_no_chunk() -> None:
    """A legitimate multi-line replacement (not header context padding) still resolves via
    the old-line-range fallback when its first touched line has no registered chunk."""
    client = _FakeRepoMindClient(
        contexts={("src/a.py", 12): _context_for("focus-1", "src/a.py", "target_fn")},
        blast_radii={"focus-1": {"nodes": [{"id": "focus-1", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "target_fn", "class_name": None}], "edges": []}},
    )
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -10,3 +10,3 @@\n-x\n-p\n-q\n+y\n+r\n+s\n"
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert [f.chunk_id for f in bundle.changed_functions] == ["focus-1"]


def test_build_resolves_second_hunk_via_old_lines_despite_earlier_net_line_delta() -> None:
    """Regression for the multi-hunk line-drift bug: a file's second hunk must resolve against
    the pre-diff (old) line numbers, not the new-file numbers, once an earlier hunk in the same
    file has a non-zero net line delta. This is the exact drift that misattributed a real
    `login` edit to `me` during Phase 25 verification (see PROGRESS.md) - a pure-insertion first
    hunk (+4 net lines) shifts the second hunk's new-file numbers away from what's indexed,
    while its old-file numbers (what `bar`'s chunk is actually indexed at) stay correct.
    """
    diff = (
        "--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -5,0 +6,4 @@\n+p1\n+p2\n+p3\n+p4\n"
        "@@ -20,1 +24,1 @@\n-old_bar_line\n+new_bar_line\n"
    )
    client = _FakeRepoMindClient(
        contexts={("src/a.py", 20): _context_for("bar-chunk", "src/a.py", "bar")},
        blast_radii={
            "bar-chunk": {
                "nodes": [{"id": "bar-chunk", "node_kind": "chunk", "file_path": "src/a.py", "function_name": "bar", "class_name": None}],
                "edges": [],
            }
        },
    )
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    # The first hunk's insertion point (old line 5) has no registered chunk and is skipped
    # (best-effort); the second hunk must still resolve to `bar` via its old-line number (20),
    # not its drifted new-line number (24), which nothing is registered at.
    assert [f.chunk_id for f in bundle.changed_functions] == ["bar-chunk"]


def test_build_does_not_misattribute_a_boundary_line_edit_to_an_adjacent_function() -> None:
    """Regression for the context-spillover bug: an edit whose only touched line is a
    blank/separator line between two functions (no AST chunk covers it) must be honestly
    skipped, never silently attributed to whichever function happens to start a few lines
    later - the exact colorama `reset_all`/`init` misattribution from Phase 25 verification.
    """
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -9,1 +9,1 @@\n- \n+ \n"
    client = _FakeRepoMindClient(
        # Only the adjacent function's own chunk is registered (at line 12) - never the blank
        # separator line (9) the edit actually touches. Before the fix, a header-width fallback
        # range could have reached line 12 and wrongly attributed the edit to it.
        contexts={("src/a.py", 12): _context_for("bar-chunk", "src/a.py", "bar")},
        blast_radii={"bar-chunk": {"nodes": [], "edges": []}},
    )
    builder = AdjudicateContextBuilder(client, repo_id="repo-1", hops=1)

    bundle = builder.build(diff)

    assert bundle.changed_functions == []
