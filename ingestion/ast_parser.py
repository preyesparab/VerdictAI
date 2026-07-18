"""Tree-sitter based AST parsing of source files into semantic `CodeChunk` objects.

`TreeSitterParser` consumes the `SourceFile` objects produced by
`ingestion.file_discovery.FileDiscovery` and extracts one `CodeChunk` per
function, async function, class, method, or (named) arrow function found
in the file's AST — never arbitrary fixed-size windows. Each supported
language is registered once in `TreeSitterParser._LANGUAGES` as a
grammar plus an "extractor" function; adding a new language later means
writing one new extractor function and adding one registry entry, with
no changes to the parsing/orchestration logic itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from core.exceptions import ParsingError
from core.logging import get_logger
from ingestion.deterministic_ids import compute_chunk_id, compute_file_id
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk

logger = get_logger(__name__)

_PYTHON_FUNCTION_TYPE = "function_definition"
_PYTHON_CLASS_TYPE = "class_definition"
_PYTHON_DECORATED_TYPE = "decorated_definition"

_JS_FUNCTION_DECLARATION_TYPE = "function_declaration"
_JS_METHOD_TYPE = "method_definition"
_JS_ARROW_FUNCTION_TYPE = "arrow_function"
_JS_CLASS_TYPE = "class_declaration"


@dataclass(frozen=True)
class _ExtractedNode:
    """Intermediate representation of one matched AST node, before it is
    turned into a `CodeChunk` (which additionally needs `file_id` and
    `file_path`, not known to the language-specific extractors).
    """

    chunk_type: ChunkType
    function_name: str | None
    class_name: str | None
    parent_class: str | None
    span_node: Node


@dataclass(frozen=True)
class _LanguageSupport:
    """A registered language: its grammar plus the function that walks a
    parsed tree in that grammar and yields `_ExtractedNode` records.
    """

    language: Language
    extractor: Callable[[Node, bytes], list[_ExtractedNode]]


def _walk(root: Node) -> Iterator[Node]:
    """Iteratively yield every node in `root`'s subtree, in document order.

    Args:
        root: The node to start traversal from (included in the output).

    Yields:
        Each node in the subtree, parents before children, left to right.
    """
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _text(source: bytes, node: Node) -> str:
    """Decode the exact source text spanned by `node`.

    Args:
        source: The full source file, as bytes.
        node: The node whose span to extract.

    Returns:
        The decoded text of `node`'s span.
    """
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------
# Python extractor
# ---------------------------------------------------------------------


def _python_nearest_container_type(node: Node) -> str | None:
    """Find the type of the nearest enclosing function or class definition.

    Used to distinguish a top-level function from a class method (a
    function directly inside a class body) from a function nested inside
    another function (a closure, regardless of whether that outer
    function is itself a method).

    Args:
        node: A `function_definition` node.

    Returns:
        ``"class_definition"`` if `node` is a direct method of a class,
        ``"function_definition"`` if `node` is nested inside another
        function, or None if `node` is at module level.
    """
    current = node.parent
    while current is not None:
        if current.type in (_PYTHON_FUNCTION_TYPE, _PYTHON_CLASS_TYPE):
            return current.type
        current = current.parent
    return None


def _python_enclosing_class(node: Node) -> Node | None:
    """Find the nearest enclosing `class_definition`, at any nesting depth.

    Args:
        node: The node to search upward from.

    Returns:
        The nearest ancestor `class_definition` node, or None.
    """
    current = node.parent
    while current is not None:
        if current.type == _PYTHON_CLASS_TYPE:
            return current
        current = current.parent
    return None


def _python_base_classes(source: bytes, class_node: Node) -> str | None:
    """Extract a class's positional base classes as a comma-separated string.

    Args:
        source: The full source file, as bytes.
        class_node: A `class_definition` node.

    Returns:
        Comma-separated base class names (e.g., ``"Base1, Base2"``), or
        None if the class has no bases.
    """
    superclasses = class_node.child_by_field_name("superclasses")
    if superclasses is None:
        return None
    names = [
        _text(source, child)
        for child in superclasses.children
        if child.type in ("identifier", "attribute")
    ]
    return ", ".join(names) if names else None


def _python_span_node(node: Node) -> Node:
    """Widen a function/class node's span to include its decorators, if any.

    Args:
        node: A `function_definition` or `class_definition` node.

    Returns:
        The enclosing `decorated_definition` node if `node` is decorated,
        otherwise `node` itself.
    """
    parent = node.parent
    if parent is not None and parent.type == _PYTHON_DECORATED_TYPE:
        return parent
    return node


def _extract_python_chunks(root: Node, source: bytes) -> list[_ExtractedNode]:
    """Walk a parsed Python AST and extract function/async function/class/method chunks.

    Args:
        root: The root node of the parsed tree.
        source: The full source file, as bytes.

    Returns:
        One `_ExtractedNode` per function, async function, class, and
        method found, including nested ones.
    """
    extracted: list[_ExtractedNode] = []

    for node in _walk(root):
        if node.type == _PYTHON_FUNCTION_TYPE:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            container_type = _python_nearest_container_type(node)
            class_node = _python_enclosing_class(node)
            is_async = any(child.type == "async" for child in node.children)

            if container_type == _PYTHON_CLASS_TYPE:
                chunk_type = ChunkType.METHOD
            elif is_async:
                chunk_type = ChunkType.ASYNC_FUNCTION
            else:
                chunk_type = ChunkType.FUNCTION

            extracted.append(
                _ExtractedNode(
                    chunk_type=chunk_type,
                    function_name=_text(source, name_node),
                    class_name=_text(source, class_node.child_by_field_name("name"))
                    if class_node is not None
                    else None,
                    parent_class=_python_base_classes(source, class_node)
                    if class_node is not None
                    else None,
                    span_node=_python_span_node(node),
                )
            )

        elif node.type == _PYTHON_CLASS_TYPE:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            extracted.append(
                _ExtractedNode(
                    chunk_type=ChunkType.CLASS,
                    function_name=None,
                    class_name=_text(source, name_node),
                    parent_class=_python_base_classes(source, node),
                    span_node=_python_span_node(node),
                )
            )

    return extracted


# ---------------------------------------------------------------------
# JavaScript extractor
# ---------------------------------------------------------------------


def _js_enclosing_class(node: Node) -> Node | None:
    """Find the nearest enclosing `class_declaration`.

    Args:
        node: The node to search upward from.

    Returns:
        The nearest ancestor `class_declaration` node, or None.
    """
    current = node.parent
    while current is not None:
        if current.type == _JS_CLASS_TYPE:
            return current
        current = current.parent
    return None


def _js_superclass(source: bytes, class_node: Node) -> str | None:
    """Extract a class's `extends` target, if any.

    Args:
        source: The full source file, as bytes.
        class_node: A `class_declaration` node.

    Returns:
        The superclass expression's text, or None if the class does not
        extend anything.
    """
    for child in class_node.children:
        if child.type == "class_heritage":
            names = [_text(source, c) for c in child.children if c.type != "extends"]
            return ", ".join(names) if names else None
    return None


_JS_EXPORTS_OBJECT_NAME = "exports"


def _js_exports_property_name_node(source: bytes, node: Node) -> Node | None:
    """Find the property name node for a CommonJS handler export.

    Matches ``exports.foo = (...) => {...}`` / ``exports.foo = async
    (...) => {...}`` — the pattern used throughout Express controllers to
    export route handlers, which a plain ``variable_declarator`` check
    misses entirely since there is no ``const``/``let`` involved.

    Args:
        source: The full source file, as bytes.
        node: An `arrow_function` node.

    Returns:
        The exported property's identifier node, or None if `node`'s
        parent isn't an ``exports.<name> = ...`` assignment.
    """
    parent = node.parent
    if parent is None or parent.type != "assignment_expression":
        return None
    left = parent.child_by_field_name("left")
    if left is None or left.type != "member_expression":
        return None
    object_node = left.child_by_field_name("object")
    if object_node is None or object_node.type != "identifier":
        return None
    if _text(source, object_node) != _JS_EXPORTS_OBJECT_NAME:
        return None
    return left.child_by_field_name("property")


def _js_arrow_body_is_object(node: Node) -> bool:
    """Check whether an arrow function's body is an object-literal expression.

    True for ``(set, get) => ({ ... })`` (the state-shape returned by a
    Zustand/Redux-style store factory), false for a block-bodied callback
    like ``(e) => { doThing(); }`` — the distinguishing trait used by
    `_js_factory_call_name_node` to tell a state-store factory argument
    apart from an ordinary callback (e.g. one passed to `useCallback`).

    Args:
        node: An `arrow_function` node.

    Returns:
        True if `node`'s body is (possibly parenthesized) an object literal.
    """
    body = node.child_by_field_name("body")
    if body is None:
        return False
    if body.type == "object":
        return True
    return body.type == "parenthesized_expression" and any(
        child.type == "object" for child in body.children
    )


def _js_factory_call_name_node(node: Node) -> Node | None:
    """Find the variable name node for an arrow function nested in factory call(s).

    Matches state-management factory patterns such as Zustand's
    ``const useX = create((set, get) => ({...}))`` or, with middleware
    wrapping, ``const useX = create(persist((set, get) => ({...}), opts))``
    — the arrow function is not itself assigned to a variable, but is the
    innermost argument of one or more nested calls that ultimately are.

    Two restrictions keep this from over-matching ordinary callbacks that
    happen to share the same "nested in a call, result assigned to a
    const" shape:
      - `node`'s body must be an object literal (see
        `_js_arrow_body_is_object`) — excludes block-bodied callbacks like
        ``const handleDrop = useCallback((e) => { ... }, [deps])``.
      - Every call in the chain must be a bare-identifier call (``create(``,
        ``persist(``) — excludes method calls like
        ``const payload = items.map((i) => ({ id: i.id }))``, whose
        callee is a `member_expression` (``items.map``), not an identifier.

    Walks up through any number of ``arguments -> call_expression`` hops
    (one per layer of wrapping, e.g. ``persist(...)`` inside ``create(...)``),
    stopping as soon as the chain lands on a `variable_declarator`.

    Args:
        node: An `arrow_function` node.

    Returns:
        The identifier node the outermost factory call is assigned to, or
        None if `node` isn't nested in such a call-argument chain.
    """
    if not _js_arrow_body_is_object(node):
        return None

    current = node
    while True:
        parent = current.parent
        if parent is None or parent.type != "arguments":
            return None
        call_node = parent.parent
        if call_node is None or call_node.type != "call_expression":
            return None
        callee = call_node.child_by_field_name("function")
        if callee is None or callee.type != "identifier":
            return None
        current = call_node
        grandparent = current.parent
        if grandparent is not None and grandparent.type == "variable_declarator":
            return grandparent.child_by_field_name("name")


def _js_arrow_function_name_node(source: bytes, node: Node) -> Node | None:
    """Find the identifier node providing a named arrow function's name.

    An arrow function is treated as a standalone chunk when it is:
      - directly assigned to a variable (``const f = () => ...``),
      - a CommonJS handler export (``exports.f = (...) => ...``), or
      - the innermost callback of a factory-call chain itself assigned to
        a variable (``const useX = create(persist((set, get) => ({...}), opts))``).
    Anonymous arrow functions passed inline as callback arguments with no
    such stable name (e.g. ``arr.map((x) => x * 2)``) are skipped.

    Args:
        source: The full source file, as bytes.
        node: An `arrow_function` node.

    Returns:
        The identifier node naming this arrow function, or None if it has
        no resolvable stable name.
    """
    parent = node.parent
    if parent is not None and parent.type == "variable_declarator":
        return parent.child_by_field_name("name")

    exports_name = _js_exports_property_name_node(source, node)
    if exports_name is not None:
        return exports_name

    return _js_factory_call_name_node(node)


def _extract_javascript_chunks(root: Node, source: bytes) -> list[_ExtractedNode]:
    """Walk a parsed JavaScript AST and extract function/arrow function/class/method chunks.

    Args:
        root: The root node of the parsed tree.
        source: The full source file, as bytes.

    Returns:
        One `_ExtractedNode` per function declaration, named arrow
        function, class, and method found, including nested ones.
    """
    extracted: list[_ExtractedNode] = []

    for node in _walk(root):
        if node.type == _JS_CLASS_TYPE:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            extracted.append(
                _ExtractedNode(
                    chunk_type=ChunkType.CLASS,
                    function_name=None,
                    class_name=_text(source, name_node),
                    parent_class=_js_superclass(source, node),
                    span_node=node,
                )
            )

        elif node.type == _JS_METHOD_TYPE:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            class_node = _js_enclosing_class(node)
            class_name_node = class_node.child_by_field_name("name") if class_node else None
            extracted.append(
                _ExtractedNode(
                    chunk_type=ChunkType.METHOD,
                    function_name=_text(source, name_node),
                    class_name=_text(source, class_name_node) if class_name_node else None,
                    parent_class=_js_superclass(source, class_node) if class_node else None,
                    span_node=node,
                )
            )

        elif node.type == _JS_FUNCTION_DECLARATION_TYPE:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            extracted.append(
                _ExtractedNode(
                    chunk_type=ChunkType.FUNCTION,
                    function_name=_text(source, name_node),
                    class_name=None,
                    parent_class=None,
                    span_node=node,
                )
            )

        elif node.type == _JS_ARROW_FUNCTION_TYPE:
            name_node = _js_arrow_function_name_node(source, node)
            if name_node is None:
                continue
            extracted.append(
                _ExtractedNode(
                    chunk_type=ChunkType.ARROW_FUNCTION,
                    function_name=_text(source, name_node),
                    class_name=None,
                    parent_class=None,
                    span_node=node,
                )
            )

    return extracted


class TreeSitterParser:
    """Parses source files into AST-derived `CodeChunk` objects.

    Supports Python and JavaScript. Each language is registered with its
    compiled grammar and a dedicated extractor function; see the module
    docstring for how to add another language.
    """

    def __init__(self) -> None:
        """Load every supported language's grammar and build its parser."""
        support_by_language = {
            "python": _LanguageSupport(Language(tspython.language()), _extract_python_chunks),
            "javascript": _LanguageSupport(
                Language(tsjavascript.language()), _extract_javascript_chunks
            ),
        }
        self._support = support_by_language
        self._parsers = {
            language: Parser(support.language) for language, support in support_by_language.items()
        }

    def parse(self, source_file: SourceFile) -> list[CodeChunk]:
        """Parse `source_file` and extract its semantic `CodeChunk` objects.

        Args:
            source_file: A file discovered by `FileDiscovery`.

        Returns:
            One `CodeChunk` per function, async function, class, method,
            or named arrow function found in the file. Empty for a file
            with none (including an empty file).

        Raises:
            ParsingError: If the language is unsupported, the file cannot
                be read, Tree-sitter fails to parse it, or the resulting
                tree contains syntax errors.
        """
        support = self._support.get(source_file.language)
        if support is None:
            logger.error(
                "Parsing failed for %s: unsupported language %r",
                source_file.relative_path, source_file.language,
            )
            raise ParsingError(f"Unsupported language for AST parsing: {source_file.language!r}")

        logger.info("Parsing started: %s (%s)", source_file.relative_path, source_file.language)

        try:
            source_bytes = source_file.absolute_path.read_bytes()
        except OSError as exc:
            logger.error("Parsing failed for %s: %s", source_file.relative_path, exc)
            raise ParsingError(f"Failed to read {source_file.absolute_path}: {exc}") from exc

        parser = self._parsers[source_file.language]
        try:
            tree = parser.parse(source_bytes)
        except Exception as exc:  # noqa: BLE001 - translate any Tree-sitter failure
            logger.error("Parsing failed for %s: %s", source_file.relative_path, exc)
            raise ParsingError(
                f"Tree-sitter failed to parse {source_file.absolute_path}: {exc}"
            ) from exc

        if tree.root_node.has_error:
            logger.error(
                "Parsing failed for %s: syntax errors detected", source_file.relative_path
            )
            raise ParsingError(
                f"Syntax errors detected while parsing {source_file.absolute_path}"
            )

        file_id = compute_file_id(source_file.relative_path)
        extracted_nodes = support.extractor(tree.root_node, source_bytes)
        chunks = [
            self._build_chunk(source_file, file_id, extracted, source_bytes)
            for extracted in extracted_nodes
        ]

        logger.info(
            "Parsing completed: %s - %d chunk(s) extracted",
            source_file.relative_path, len(chunks),
        )
        return chunks

    def parse_many(self, source_files: list[SourceFile]) -> list[CodeChunk]:
        """Parse multiple files, skipping (and logging) any that fail.

        Args:
            source_files: Files to parse, typically the output of
                `FileDiscovery.discover`.

        Returns:
            The concatenated `CodeChunk` list from every file that parsed
            successfully. Files that raise `ParsingError` are skipped.
        """
        chunks: list[CodeChunk] = []
        failures = 0
        for source_file in source_files:
            try:
                chunks.extend(self.parse(source_file))
            except ParsingError as exc:
                failures += 1
                logger.warning("Skipping %s due to parsing failure: %s", source_file.relative_path, exc)

        logger.info(
            "Batch parsing complete: %d file(s) parsed, %d chunk(s) extracted, %d file(s) failed",
            len(source_files) - failures, len(chunks), failures,
        )
        return chunks

    def _build_chunk(
        self,
        source_file: SourceFile,
        file_id: str,
        extracted: _ExtractedNode,
        source_bytes: bytes,
    ) -> CodeChunk:
        """Turn one `_ExtractedNode` into a `CodeChunk`.

        Args:
            source_file: The file the chunk was extracted from.
            file_id: The file's deterministic identifier.
            extracted: The matched AST node and its extracted metadata.
            source_bytes: The full source file, as bytes.

        Returns:
            The assembled `CodeChunk`, with a deterministic `chunk_id`
            derived from `(file_id, name, start_line)`.
        """
        start_line = extracted.span_node.start_point[0] + 1
        end_line = extracted.span_node.end_point[0] + 1
        raw_code = source_bytes[
            extracted.span_node.start_byte : extracted.span_node.end_byte
        ].decode("utf-8", errors="replace")

        identity_name = (
            extracted.class_name if extracted.chunk_type == ChunkType.CLASS else extracted.function_name
        )
        chunk_id = compute_chunk_id(f"{file_id}:{identity_name}:{start_line}")

        return CodeChunk(
            chunk_id=chunk_id,
            file_id=file_id,
            file_path=source_file.relative_path.as_posix(),
            language=source_file.language,
            chunk_type=extracted.chunk_type,
            function_name=extracted.function_name,
            class_name=extracted.class_name,
            parent_class=extracted.parent_class,
            start_line=start_line,
            end_line=end_line,
            raw_code=raw_code,
        )
