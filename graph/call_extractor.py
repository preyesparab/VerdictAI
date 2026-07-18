"""Tree-sitter based call-site extraction from a single chunk's source text.

`CallExtractor` finds call expressions (``foo()``, ``self.foo()``,
``obj.foo()``, ``Foo.foo()``) within one `models.schemas.CodeChunk`'s
`raw_code` — the text of a single function/method/class definition. A
chunk's body is syntactically self-contained, so each chunk is parsed
independently rather than re-parsing (and re-walking) the whole file just
to find the calls made within it.

It also finds module-scope wiring - JavaScript handler-registration
idioms like Express's ``router.post("/x", handler)`` - via
`CallExtractor.extract_module_level`, which parses a *whole* file rather
than one chunk fragment: that code lives outside any function/class
body, so no chunk's `raw_code` ever contains it, regardless of how the
functions involved are named.

This only detects *syntactic* call sites and their textual qualifier; it
performs no type inference or symbol resolution — that happens in
`graph.graph_builder`, which turns a `CallSite`/`ReferenceSite` into a
graph edge (or logs it as an unresolved reference if no matching chunk
can be found).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from core.logging import get_logger

logger = get_logger(__name__)

_PYTHON_CALL_TYPE = "call"
_JS_CALL_TYPE = "call_expression"

# Qualifiers that indicate "call a method on my own instance/class", the
# strongest signal that a call resolves to a method of the caller's own
# enclosing class rather than some other object of the same attribute name.
_PYTHON_SELF_NAMES = frozenset({"self", "cls"})
_JS_SELF_NAMES = frozenset({"this"})

# Node types marking a function/class body's boundary. Module-scope
# scanning (`_walk_module_scope`) does not descend past these - that
# subtree gets its own dedicated per-chunk extraction (`extract`)
# elsewhere, so descending into it here would double-count the same
# call sites.
_JS_MODULE_SCOPE_BOUNDARY_TYPES = frozenset(
    {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "class_expression",
    }
)


@dataclass(frozen=True)
class CallSite:
    """One detected call expression within a chunk's body.

    Attributes:
        callee_name: The terminal identifier being called (e.g. ``"foo"``
            in both ``foo()`` and ``self.foo()`` and ``obj.foo()``).
        is_self_qualified: True if the call was of the form ``self.x()``/
            ``cls.x()`` (Python) or ``this.x()`` (JavaScript) — a strong
            signal the callee is a method on the caller's own class.
        qualifier_name: The identifier the call was qualified by, if any
            and if not self/cls/this (e.g. ``"obj"`` in ``obj.foo()``, or
            a class name in ``Foo.foo()``). None for a bare call, a
            self-qualified call, or a call qualified by a non-identifier
            expression (e.g. a chained ``a.b.foo()``).
    """

    callee_name: str
    is_self_qualified: bool
    qualifier_name: str | None


@dataclass(frozen=True)
class ReferenceSite:
    """A bare identifier passed as an argument to a module-scope call.

    Captures handler-registration idioms like Express's
    ``router.post("/x", handler)`` — ``handler`` is never invoked at that
    call site, just passed by reference, so it is not a `CallSite`. Only
    produced by `CallExtractor.extract_module_level`; the same pattern
    inside a function body is not scanned for references, keeping this
    to the wiring idiom it was added for rather than every callback
    passed anywhere in the codebase.

    Attributes:
        name: The referenced identifier's name.
    """

    name: str


@dataclass(frozen=True)
class ModuleLevelSites:
    """Call sites and bare-identifier references found at a file's module scope.

    Attributes:
        call_sites: Every `CallSite` found outside any function/class body.
        reference_sites: Every bare-identifier argument passed to a
            module-scope call (see `ReferenceSite`).
    """

    call_sites: list[CallSite]
    reference_sites: list[ReferenceSite]


def _walk(root: Node) -> Iterator[Node]:
    """Iteratively yield every node in `root`'s subtree, in document order."""
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _text(source: bytes, node: Node) -> str:
    """Decode the exact source text spanned by `node`."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_python_call_sites(root: Node, source: bytes) -> list[CallSite]:
    """Walk a parsed Python fragment and extract every `call` expression.

    Args:
        root: The root node of the parsed chunk fragment.
        source: The chunk's `raw_code`, as bytes.

    Returns:
        One `CallSite` per call expression found, including nested ones
        (e.g. calls inside a comprehension or a decorator argument).
    """
    call_sites: list[CallSite] = []

    for node in _walk(root):
        if node.type != _PYTHON_CALL_TYPE:
            continue
        function_node = node.child_by_field_name("function")
        if function_node is None:
            continue

        if function_node.type == "identifier":
            call_sites.append(
                CallSite(callee_name=_text(source, function_node), is_self_qualified=False, qualifier_name=None)
            )
        elif function_node.type == "attribute":
            attribute_node = function_node.child_by_field_name("attribute")
            object_node = function_node.child_by_field_name("object")
            if attribute_node is None:
                continue
            callee_name = _text(source, attribute_node)

            if object_node is not None and object_node.type == "identifier":
                object_name = _text(source, object_node)
                if object_name in _PYTHON_SELF_NAMES:
                    call_sites.append(CallSite(callee_name, is_self_qualified=True, qualifier_name=None))
                else:
                    call_sites.append(CallSite(callee_name, is_self_qualified=False, qualifier_name=object_name))
            else:
                # Chained (`a.b.foo()`) or otherwise non-identifier
                # qualifier — record the call with no qualifier rather
                # than dropping it entirely.
                call_sites.append(CallSite(callee_name, is_self_qualified=False, qualifier_name=None))
        # Any other callable form (subscript, call result, lambda, etc.)
        # is dynamic dispatch we cannot name statically — skip it.

    return call_sites


def _javascript_call_site_from_node(source: bytes, node: Node) -> CallSite | None:
    """Build a `CallSite` from one `call_expression` node, if syntactically nameable.

    Factored out of `_extract_javascript_call_sites` so `extract_module_level`
    can reuse the exact same per-call-site logic instead of duplicating it.

    Args:
        source: The full source (or fragment), as bytes.
        node: A `call_expression` node.

    Returns:
        The detected `CallSite`, or None for a dynamically-dispatched
        call form (subscript, call result, lambda, etc.) with no
        statically nameable callee.
    """
    function_node = node.child_by_field_name("function")
    if function_node is None:
        return None

    if function_node.type == "identifier":
        return CallSite(callee_name=_text(source, function_node), is_self_qualified=False, qualifier_name=None)

    if function_node.type == "member_expression":
        property_node = function_node.child_by_field_name("property")
        object_node = function_node.child_by_field_name("object")
        if property_node is None:
            return None
        callee_name = _text(source, property_node)

        if object_node is not None and object_node.type == "this":
            return CallSite(callee_name, is_self_qualified=True, qualifier_name=None)
        if object_node is not None and object_node.type == "identifier":
            return CallSite(callee_name, is_self_qualified=False, qualifier_name=_text(source, object_node))
        return CallSite(callee_name, is_self_qualified=False, qualifier_name=None)

    return None


def _extract_javascript_call_sites(root: Node, source: bytes) -> list[CallSite]:
    """Walk a parsed JavaScript fragment and extract every `call_expression`.

    Args:
        root: The root node of the parsed chunk fragment.
        source: The chunk's `raw_code`, as bytes.

    Returns:
        One `CallSite` per call expression found, including nested ones.
    """
    call_sites: list[CallSite] = []

    for node in _walk(root):
        if node.type != _JS_CALL_TYPE:
            continue
        call_site = _javascript_call_site_from_node(source, node)
        if call_site is not None:
            call_sites.append(call_site)

    return call_sites


def _walk_module_scope(root: Node, boundary_types: frozenset[str]) -> Iterator[Node]:
    """Yield every node in `root`'s subtree that is not nested inside a function/class body.

    Mirrors `_walk`'s traversal, but does not descend past a node whose
    type is in `boundary_types` - that subtree gets its own dedicated
    per-chunk extraction elsewhere (`extract`), so descending into it
    here would double-count the same call sites.

    Args:
        root: The node to start traversal from (always yielded, even if
            its own type is a boundary type).
        boundary_types: Node types marking a function/class body whose
            interior should not be walked.

    Yields:
        Each module-scope node, parents before children, left to right.
    """
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        if current is not root and current.type in boundary_types:
            continue
        stack.extend(reversed(current.children))


def _javascript_argument_references(source: bytes, call_node: Node) -> list[ReferenceSite]:
    """Extract bare-identifier arguments passed to one `call_expression`.

    Args:
        source: The full source file, as bytes.
        call_node: A `call_expression` node.

    Returns:
        One `ReferenceSite` per argument that is a plain identifier (e.g.
        ``register`` in ``router.post("/register", register)``) —
        excludes string/number/object/nested-call arguments, which are
        data, not function references.
    """
    arguments_node = call_node.child_by_field_name("arguments")
    if arguments_node is None:
        return []
    return [
        ReferenceSite(name=_text(source, child))
        for child in arguments_node.children
        if child.type == "identifier"
    ]


def _extract_javascript_module_level(root: Node, source: bytes) -> ModuleLevelSites:
    """Walk a parsed JavaScript file and extract module-scope calls and references.

    Args:
        root: The root node of the parsed file.
        source: The full source file, as bytes.

    Returns:
        Every call site and bare-identifier argument reference found
        outside any function/class body.
    """
    call_sites: list[CallSite] = []
    reference_sites: list[ReferenceSite] = []

    for node in _walk_module_scope(root, _JS_MODULE_SCOPE_BOUNDARY_TYPES):
        if node.type != _JS_CALL_TYPE:
            continue
        call_site = _javascript_call_site_from_node(source, node)
        if call_site is not None:
            call_sites.append(call_site)
        reference_sites.extend(_javascript_argument_references(source, node))

    return ModuleLevelSites(call_sites=call_sites, reference_sites=reference_sites)


@dataclass(frozen=True)
class _LanguageSupport:
    """A registered language: its grammar plus its call-site extractor function."""

    language: Language
    extractor: Callable[[Node, bytes], list[CallSite]]


class CallExtractor:
    """Detects call expressions within a single chunk's source text.

    Supports Python and JavaScript, following the same per-language
    registry pattern as `ingestion.ast_parser.TreeSitterParser`. Also
    detects JavaScript module-scope wiring via `extract_module_level`
    (see that method's docstring).
    """

    def __init__(self) -> None:
        """Load every supported language's grammar and build its parser."""
        support_by_language = {
            "python": _LanguageSupport(Language(tspython.language()), _extract_python_call_sites),
            "javascript": _LanguageSupport(
                Language(tsjavascript.language()), _extract_javascript_call_sites
            ),
        }
        self._support = support_by_language
        self._parsers = {
            language: Parser(support.language) for language, support in support_by_language.items()
        }

    def extract(self, language: str, raw_code: str) -> list[CallSite]:
        """Extract every call site within `raw_code`.

        Never raises: an unsupported language or a fragment Tree-sitter
        cannot make sense of yields no call sites rather than aborting
        graph construction, since call detection is a best-effort
        structural signal, not a correctness requirement.

        Args:
            language: The chunk's language (e.g. ``"python"``).
            raw_code: The chunk's exact source text.

        Returns:
            One `CallSite` per call expression found in `raw_code`.
        """
        support = self._support.get(language)
        if support is None:
            logger.debug("Call extraction skipped: unsupported language %r", language)
            return []

        try:
            source_bytes = raw_code.encode("utf-8")
            parser = self._parsers[language]
            tree = parser.parse(source_bytes)
            return support.extractor(tree.root_node, source_bytes)
        except Exception as exc:  # noqa: BLE001 - never let call detection abort graph construction
            logger.warning("Call extraction failed for a %s chunk: %s", language, exc)
            return []

    def extract_module_level(self, language: str, source_code: str) -> ModuleLevelSites:
        """Extract call sites and bare-identifier argument references made at a file's module scope.

        Unlike `extract` (which scans one already-isolated chunk
        fragment), this parses the WHOLE file and only visits nodes
        outside any function/class body — the top-level wiring code
        `extract` structurally cannot see, since chunking only ever
        produces one fragment per function/class, never one for a
        file's top-level statements.

        Never raises, for the same reason as `extract`: this is a
        best-effort structural signal, not a correctness requirement.

        Args:
            language: The file's language. Only ``"javascript"`` is
                supported — Python's decorator-based route registration
                (``@router.post(...)``) doesn't exhibit the same
                "handler passed as a bare argument" idiom, so it is not
                scanned here.
            source_code: The whole file's source text.

        Returns:
            Every module-scope call site and argument reference found.
            Empty for an unsupported language or a file Tree-sitter
            cannot make sense of.
        """
        if language != "javascript":
            return ModuleLevelSites(call_sites=[], reference_sites=[])

        try:
            source_bytes = source_code.encode("utf-8")
            parser = self._parsers[language]
            tree = parser.parse(source_bytes)
            return _extract_javascript_module_level(tree.root_node, source_bytes)
        except Exception as exc:  # noqa: BLE001 - never let call detection abort graph construction
            logger.warning("Module-level call extraction failed for a %s file: %s", language, exc)
            return ModuleLevelSites(call_sites=[], reference_sites=[])
