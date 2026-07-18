"""Context builder for Adjudicate reviews (Phase 24).

Given a unified diff, gathers exactly what a reviewer needs about each
changed location - the enclosing function/class, its real callers and
callees, and any existing tests that touch it - into a single
`ContextBundle`. Every later agent (Defender, Prosecutor - Phases 25/26)
reads only from this bundle, never the raw diff.

Graph traversal (via `RepoMindClient`'s `/context` and blast-radius
endpoints) is the primary mechanism for all three pieces: unlike a
general question, a diff already names its exact starting point (the
changed lines), so this is a structural lookup, not a similarity search.
Hybrid retrieval (`RepoMindClient.search`, Phase 24's new `/search`
endpoint) is called only as a fallback, and only for the "related
tests" piece - the one relationship the call graph systematically
misses: a test that exercises a function through a mock/string
reference (e.g. Python's ``@patch('module.func')``) or an HTTP-level
test client, not a direct call edge `graph.graph_builder` can capture.
No hybrid query is ever made for callers/callees - if blast radius finds
none, the bundle simply records none, which is the graph telling the
truth about an isolated node (see the Phase 4/6 diagnostic's own
isolated-node census), not a lookup failure to compensate for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from adjudicate.repomind_client import RepoMindClient
from core.exceptions import RetrievalError
from core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BLAST_RADIUS_HOPS = 1

_FILE_HEADER_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$")
_OLD_FILE_HEADER_PREFIX = "--- "
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE_MARKER_PREFIX = "\\"

# A file "looks like a test" if any path segment/filename contains one of
# these substrings - covers every convention seen across the repos this
# system has actually indexed so far (colorama's `*_test.py`, a
# hypothetical `test_*.py`, JS's `*.spec.js`/`__tests__/`).
_TEST_PATH_MARKERS = ("test", "spec")

# Mirrors `models.schemas.ChunkType`'s own AST-vs-window split: only these
# chunk types are ever added to the knowledge graph by
# `graph.graph_builder.RepositoryGraphBuilder` (see that module and
# `pipeline.Pipeline.index_repository`, which builds the graph from
# `ast_chunks` only) - a SLIDING or PARENT chunk is never a graph node, so
# blast radius on one would always 404.
_AST_CHUNK_TYPES = frozenset({"function", "async_function", "class", "method", "arrow_function"})


@dataclass(frozen=True)
class ChangedLocation:
    """One changed region from a diff, in both the new and the old (already-indexed) file.

    Attributes:
        file: Repository-relative file path.
        start_line: 1-indexed first *actually-changed* line (a real
            ``+``/``-`` line, never a hunk's surrounding context padding),
            in the *new* (post-change) file. For human-facing reporting
            only (`ChangedFunction.changed_start_line` - a reviewer reads
            a diff in new-file line numbers) - never used for an
            indexed-graph lookup.
        end_line: 1-indexed last actually-changed line, in the new file.
        old_start_line: 1-indexed first actually-changed line, in the
            *old* (pre-change) file - what the already-indexed graph's
            own line numbers refer to, since indexing happened before
            this diff was applied. This is the only range
            `AdjudicateContextBuilder._find_enclosing_chunk` looks up
            against; see `parse_diff`'s docstring for why the new-file
            side is unsafe for that.
        old_end_line: 1-indexed last actually-changed line, in the old
            file.
    """

    file: str
    start_line: int
    end_line: int
    old_start_line: int
    old_end_line: int


def _is_hunk_boundary(line: str) -> bool:
    """Whether `line` ends the current hunk's body (a new hunk, file, or diff section starts here)."""
    return bool(_HUNK_HEADER_RE.match(line)) or bool(_FILE_HEADER_RE.match(line)) or line.startswith(
        _OLD_FILE_HEADER_PREFIX
    ) or line.startswith("diff --git ")


def parse_diff(diff_text: str) -> list[ChangedLocation]:
    """Parse a standard unified diff into its changed (file, old_range, new_range) locations.

    Deliberately minimal: reads ``+++ b/<path>`` file headers and
    ``@@ -a,b +c,d @@`` hunk headers, plus each hunk's body, only.
    Anything a full diff parser would also handle (renames, binary files,
    context-diff format) is out of scope - every repository this system
    indexes is a git clone, and `git diff`/`git show` always emit this
    format.

    Two things a naive header-only reading gets wrong, both fixed here:

    1. **Which file's line numbers to trust.** A hunk header's new-file
       side (``+c,d``) describes the *proposed* file - which is not yet
       indexed (indexing happened against the pre-diff commit). Once a
       file has more than one hunk with a non-zero net line delta
       (almost any real multi-hunk edit), every hunk after the first has
       new-file line numbers that no longer line up with the *old*,
       already-indexed file's line numbers - silently resolving the
       wrong enclosing function for later hunks. The old-file side
       (``-a,b``) is exactly what's indexed, so this parses and keeps
       both `old_start_line`/`old_end_line` (for lookups) and
       `start_line`/`end_line` (new-file, for display) per location.
    2. **How wide a "changed location" is.** A hunk header's line counts
       include the surrounding context lines (3 by default), not just
       the actual ``+``/``-`` edits - so a small edit near a function's
       top or bottom can have a header-reported range that spills into
       an adjacent function's ``def`` line. This walks each hunk's body,
       tracking separate old-file/new-file line cursors, and keeps only
       the tight min/max of lines actually touched by a ``+`` or ``-``
       (context lines advance the cursors but are not part of the range).

    Args:
        diff_text: A unified diff, e.g. from ``git diff`` or a GitHub PR.

    Returns:
        One `ChangedLocation` per hunk with a non-empty new-file range,
        in diff order. A hunk that only deletes lines (new-file length 0)
        contributes nothing - there is no line range left in the new
        file to look up. A pure-addition hunk (old-file length 0, e.g. a
        brand new block of lines) has no old-touched line to bound a
        range with, so `old_start_line`/`old_end_line` both fall back to
        the hunk's old-file insertion point.
    """
    locations: list[ChangedLocation] = []
    current_file: str | None = None

    lines = diff_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        file_match = _FILE_HEADER_RE.match(line)
        if file_match:
            current_file = None if file_match.group(1) == "/dev/null" else file_match.group(1)
            index += 1
            continue

        hunk_match = _HUNK_HEADER_RE.match(line)
        if not hunk_match or current_file is None:
            index += 1
            continue

        old_start = int(hunk_match.group(1))
        old_lines = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
        new_start = int(hunk_match.group(3))
        new_lines = int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1

        old_cursor, new_cursor = old_start, new_start
        old_touched: list[int] = []
        new_touched: list[int] = []

        index += 1
        while index < len(lines) and not _is_hunk_boundary(lines[index]):
            body_line = lines[index]
            if body_line.startswith("+") and not body_line.startswith("+++"):
                new_touched.append(new_cursor)
                new_cursor += 1
            elif body_line.startswith("-") and not body_line.startswith("---"):
                old_touched.append(old_cursor)
                old_cursor += 1
            elif not body_line.startswith(_NO_NEWLINE_MARKER_PREFIX):
                old_cursor += 1
                new_cursor += 1
            index += 1

        if new_lines > 0:
            locations.append(
                ChangedLocation(
                    file=current_file,
                    start_line=min(new_touched) if new_touched else new_start,
                    end_line=max(new_touched) if new_touched else new_start,
                    old_start_line=min(old_touched) if old_touched else old_start,
                    old_end_line=max(old_touched) if old_touched else old_start,
                )
            )

    return locations


@dataclass(frozen=True)
class ChangedFunction:
    """The chunk enclosing one changed location.

    Attributes:
        chunk_id: The enclosing chunk's id.
        file_path: Repository-relative file path.
        function_name: Function/method name, or None (e.g. a
            module-scope change, like route-wiring).
        class_name: Enclosing class name, or None.
        changed_start_line: The diff's changed range start (not the
            chunk's own start line - the chunk may be larger).
        changed_end_line: The diff's changed range end.
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    class_name: str | None
    changed_start_line: int
    changed_end_line: int


@dataclass(frozen=True)
class GraphNeighbor:
    """A caller or callee found via blast-radius graph traversal.

    Attributes:
        node_id: The neighbor's node id (a chunk or file id).
        node_kind: ``"chunk"`` or ``"file"`` - a file-node neighbor means
            the relationship is module-level (an import, or route-style
            reference-by-name with no enclosing function).
        file_path: Repository-relative file path.
        function_name: Function/method name, or None for a file node or
            a module-scope chunk.
        class_name: Enclosing class name, or None.
    """

    node_id: str
    node_kind: str
    file_path: str
    function_name: str | None
    class_name: str | None


@dataclass(frozen=True)
class RelatedTest:
    """A test found to reference a changed function.

    Attributes:
        chunk_id: The test chunk's id.
        file_path: Repository-relative file path.
        function_name: Test function name, or None.
        found_via: ``"graph"`` if a direct call/reference edge led here
            (a caller from blast radius whose file looks test-like), or
            ``"retrieval_fallback"`` if the graph found nothing and a
            targeted hybrid search (`RepoMindClient.search`) found this
            instead.
        relevance_score: The fallback search's relevance score, or None
            for a graph-found test (graph edges aren't scored).
    """

    chunk_id: str
    file_path: str
    function_name: str | None
    found_via: str
    relevance_score: float | None = None


@dataclass(frozen=True)
class ContextBundle:
    """Everything a reviewer (or Defender/Prosecutor agent) needs about a diff.

    Attributes:
        changed_functions: The chunk enclosing each changed location
            (deduplicated - two hunks in the same function contribute
            one entry).
        callers: Every caller/importer/referencer found within
            `AdjudicateContextBuilder`'s configured hop count of any
            changed function, deduplicated across all of them.
        callees: Every callee/imported-module found the same way.
        related_tests: Existing tests that reference a changed function,
            found via the graph first and a hybrid-retrieval fallback
            second (see `found_via` on each `RelatedTest`).
    """

    changed_functions: list[ChangedFunction] = field(default_factory=list)
    callers: list[GraphNeighbor] = field(default_factory=list)
    callees: list[GraphNeighbor] = field(default_factory=list)
    related_tests: list[RelatedTest] = field(default_factory=list)


# -- Shared bundle-to-text formatting -----------------------------------------
# Used by every agent that prompts an LLM from a `ContextBundle` (Defender,
# Phase 25; Prosecutor, Phase 26) so each one renders the same four fields the
# same way, rather than each agent inventing its own text layout for the same
# data.


def format_changed_functions(bundle: ContextBundle) -> str:
    """Render `bundle.changed_functions` as a readable list, one line per changed function."""
    if not bundle.changed_functions:
        return "(none - no changed function could be resolved for this diff)"
    lines = []
    for fn in bundle.changed_functions:
        label = f"{fn.class_name}.{fn.function_name}" if fn.class_name and fn.function_name else (
            fn.function_name or "(module-level code)"
        )
        lines.append(f"- `{fn.file_path}` :: {label} (changed lines {fn.changed_start_line}-{fn.changed_end_line})")
    return "\n".join(lines)


def format_neighbors(neighbors: list[GraphNeighbor], empty_note: str) -> str:
    """Render a list of `GraphNeighbor` (callers or callees) as a readable list."""
    if not neighbors:
        return empty_note
    lines = []
    for neighbor in neighbors:
        label = f"{neighbor.class_name}.{neighbor.function_name}" if neighbor.class_name and neighbor.function_name else (
            neighbor.function_name or "(module-level)"
        )
        lines.append(f"- `{neighbor.file_path}` :: {label} ({neighbor.node_kind})")
    return "\n".join(lines)


def format_related_tests(tests: list[RelatedTest]) -> str:
    """Render `bundle.related_tests` as a readable list, tagging how each was found."""
    if not tests:
        return "(none found - no existing tests reference this change, via the call graph or the retrieval fallback)"
    lines = []
    for test in tests:
        label = test.function_name or "(module-level)"
        lines.append(f"- `{test.file_path}` :: {label} (found via {test.found_via})")
    return "\n".join(lines)


def _looks_like_test_file(file_path: str) -> bool:
    """Whether `file_path` matches a test-file naming convention (substring match, case-insensitive)."""
    lowered = file_path.lower()
    return any(marker in lowered for marker in _TEST_PATH_MARKERS)


def _node_to_neighbor(node: dict[str, Any]) -> GraphNeighbor:
    """Convert one `/graph`-shaped node dict into a `GraphNeighbor`."""
    return GraphNeighbor(
        node_id=node["id"],
        node_kind=node.get("node_kind", "chunk"),
        file_path=node.get("file_path", ""),
        function_name=node.get("function_name"),
        class_name=node.get("class_name"),
    )


def _fallback_search_query(chunk: dict[str, Any]) -> str:
    """Build a search phrase for the hybrid-retrieval test fallback.

    Combines the chunk's own name with its file's basename (no
    extension) - identifier-heavy phrasing that scores well against a
    code embedding model, matching the phrasing already confirmed to
    retrieve real mock-referenced tests (see this phase's PROGRESS.md
    entry).
    """
    name = chunk.get("function_name") or chunk.get("class_name") or ""
    file_stem = chunk["file_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"test {name} {file_stem}".strip()


class AdjudicateContextBuilder:
    """Builds a `ContextBundle` from a diff, via `RepoMindClient` only.

    Never calls hybrid retrieval as a primary lookup - graph/context
    endpoints come first for every piece of the bundle. Retrieval is
    used only as the documented fallback for related tests.
    """

    def __init__(self, client: RepoMindClient, repo_id: str, hops: int = DEFAULT_BLAST_RADIUS_HOPS) -> None:
        """Initialize the builder.

        Args:
            client: The RepoMind API client to gather context through.
            repo_id: The indexed repository the diff applies to.
            hops: Blast-radius traversal depth for callers/callees.
                Defaults to 1 (direct callers/callees only) - a review
                context is about this change's immediate neighborhood,
                not a wide, hard-to-read subgraph.
        """
        self._client = client
        self._repo_id = repo_id
        self._hops = hops

    def build(self, diff_text: str) -> ContextBundle:
        """Build a `ContextBundle` for every changed location in `diff_text`.

        Args:
            diff_text: A unified diff (see `parse_diff`).

        Returns:
            The assembled bundle. A changed location with no enclosing
            chunk (e.g. a change to a blank line, or a file with no
            indexed chunks) is skipped with a warning, not an error -
            the bundle is best-effort over whatever locations do resolve.
        """
        changed_functions: list[ChangedFunction] = []
        callers: dict[str, GraphNeighbor] = {}
        callees: dict[str, GraphNeighbor] = {}
        related_tests: dict[str, RelatedTest] = {}
        seen_chunk_ids: set[str] = set()

        for location in parse_diff(diff_text):
            chunk = self._find_enclosing_chunk(location)
            if chunk is None:
                logger.warning(
                    "No indexed chunk encloses %s (old lines %d-%d, new lines %d-%d); skipping this changed location",
                    location.file, location.old_start_line, location.old_end_line,
                    location.start_line, location.end_line,
                )
                continue

            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            changed_functions.append(
                ChangedFunction(
                    chunk_id=chunk_id,
                    file_path=chunk["file_path"],
                    function_name=chunk.get("function_name"),
                    class_name=chunk.get("class_name"),
                    changed_start_line=location.start_line,
                    changed_end_line=location.end_line,
                )
            )

            subgraph = self._client.get_blast_radius(self._repo_id, focus_node=chunk_id, hops=self._hops)
            found_test_via_graph = self._collect_neighbors(subgraph, chunk_id, callers, callees, related_tests)

            if not found_test_via_graph:
                self._fallback_to_retrieval(chunk, related_tests)

        return ContextBundle(
            changed_functions=changed_functions,
            callers=list(callers.values()),
            callees=list(callees.values()),
            related_tests=list(related_tests.values()),
        )

    def _find_enclosing_chunk(self, location: ChangedLocation) -> dict[str, Any] | None:
        """Look up the AST (function/class/method) chunk enclosing `location`.

        Looks up `location.old_start_line`/`old_end_line` - the *old*
        (pre-diff) file's line numbers - never `start_line`/`end_line`
        (new-file, display-only). The already-indexed graph was built
        from the pre-diff commit, so only the old-file side's line
        numbers are guaranteed to line up with it; the new-file side
        drifts as soon as an earlier hunk in the same file has a
        non-zero net line delta (see `parse_diff`'s docstring).

        `/context` ranks its citations smallest-first, but the smallest
        enclosing block is often a SLIDING/PARENT window chunk (Phase 5) -
        real for retrieval, but never a graph node (see
        `models.schemas.ChunkType`'s own docstring), so calling blast
        radius on one would always 404. This walks the ranked citations
        for the *first AST-typed* one instead of blindly trusting rank
        0, trying the old-line range's end too if its start's lookup
        404s outright or yields no AST-typed citation at all (e.g. a
        hunk that starts one line above a function's real body). Since
        `parse_diff` already tightens the range to just the actually
        ``+``/``-`` changed lines (not the hunk's surrounding context
        padding), this fallback no longer risks spilling into an
        adjacent function the way a header-width range could.
        """
        lines_to_try = (
            [location.old_start_line] if location.old_start_line == location.old_end_line
            else [location.old_start_line, location.old_end_line]
        )
        for line in lines_to_try:
            try:
                context = self._client.get_context(self._repo_id, file=location.file, line=line)
            except RetrievalError:
                continue
            # `matched_chunks` (raw matches, smallest-first), not
            # `chunk_citations` - the latter is `ContextBuilder`'s
            # small-to-big-substituted output and, when that setting is
            # on (the default), never contains the raw enclosing chunk.
            for match in context.get("matched_chunks") or []:
                if match.get("chunk_type") in _AST_CHUNK_TYPES:
                    return match
        return None

    def _collect_neighbors(
        self,
        subgraph: dict[str, Any],
        focus_id: str,
        callers: dict[str, GraphNeighbor],
        callees: dict[str, GraphNeighbor],
        related_tests: dict[str, RelatedTest],
    ) -> bool:
        """Classify a blast-radius subgraph's edges into callers/callees, tagging test-like callers.

        Returns:
            True if at least one caller's file looked test-like (the
            signal `build` uses to skip the retrieval fallback for this
            changed function).
        """
        nodes_by_id = {node["id"]: node for node in subgraph.get("nodes", [])}
        found_test = False

        for edge in subgraph.get("edges", []):
            source, target = edge["source"], edge["target"]
            if target == focus_id and source != focus_id:
                neighbor_node = nodes_by_id.get(source)
                if neighbor_node is None:
                    continue
                neighbor = _node_to_neighbor(neighbor_node)
                callers.setdefault(neighbor.node_id, neighbor)
                if _looks_like_test_file(neighbor.file_path):
                    related_tests.setdefault(
                        neighbor.node_id,
                        RelatedTest(
                            chunk_id=neighbor.node_id, file_path=neighbor.file_path,
                            function_name=neighbor.function_name, found_via="graph",
                        ),
                    )
                    found_test = True
            elif source == focus_id and target != focus_id:
                neighbor_node = nodes_by_id.get(target)
                if neighbor_node is None:
                    continue
                callees.setdefault(target, _node_to_neighbor(neighbor_node))

        return found_test

    def _fallback_to_retrieval(self, chunk: dict[str, Any], related_tests: dict[str, RelatedTest]) -> None:
        """Search for related tests via hybrid retrieval, since blast radius found none.

        Every result is kept only if its file looks test-like - the
        fallback query is about this chunk generally, not tests
        specifically, so non-test matches are expected and discarded
        here rather than filtered server-side.
        """
        query = _fallback_search_query(chunk)
        for result in self._client.search(self._repo_id, query):
            if not _looks_like_test_file(result["file_path"]):
                continue
            related_tests.setdefault(
                result["chunk_id"],
                RelatedTest(
                    chunk_id=result["chunk_id"], file_path=result["file_path"],
                    function_name=result.get("function_name"), found_via="retrieval_fallback",
                    relevance_score=result.get("relevance_score"),
                ),
            )
