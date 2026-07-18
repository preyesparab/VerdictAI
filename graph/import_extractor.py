"""Tree-sitter based import-statement extraction and intra-repository resolution.

Import statements live at file scope, outside any function/class
definition, so — unlike calls (`graph.call_extractor`) — they cannot be
recovered from a `CodeChunk`'s `raw_code`. `ImportExtractor` re-reads and
parses the whole file to find them, then `resolve_import` maps each one
to the `file_id` of another discovered file, when it refers to a module
inside this repository. Imports of external packages (``import os``,
``from django.db import models``) cannot resolve to a file and are
reported as unresolved rather than fabricated.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Callable, Iterator

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from core.exceptions import ParsingError
from core.logging import get_logger
from ingestion.source_file import SourceFile

logger = get_logger(__name__)

_PYTHON_IMPORT_TYPE = "import_statement"
_PYTHON_IMPORT_FROM_TYPE = "import_from_statement"
_JS_IMPORT_TYPE = "import_statement"
_JS_CALL_TYPE = "call_expression"
_JS_REQUIRE_CALL_NAME = "require"

# Extensions tried, in order, when resolving a JavaScript module
# specifier that omits its extension (e.g. `import x from "./foo"`).
_JS_RESOLUTION_SUFFIXES: tuple[str, ...] = ("", ".js", ".jsx", ".mjs", "/index.js")


@dataclass(frozen=True)
class ImportRef:
    """One detected import statement, before resolution to a file.

    Attributes:
        module_path: Dotted module path for Python (e.g. ``"pkg.sub"``),
            or the raw module specifier for JavaScript (e.g. ``"./utils"``
            or ``"react"``). May be empty for a bare `from . import x`.
        relative_level: For a Python relative import (``from . import x``,
            ``from ..pkg import y``), the number of leading dots. 0 for an
            absolute import. Always 0 for JavaScript, whose relativity is
            instead encoded in `module_path` starting with ``./``/``../``.
    """

    module_path: str
    relative_level: int = 0


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


def _extract_python_imports(root: Node, source: bytes) -> list[ImportRef]:
    """Walk a parsed Python file and extract every import statement.

    Args:
        root: The root node of the parsed file.
        source: The full source file, as bytes.

    Returns:
        One `ImportRef` per imported module (an `import a, b` statement
        yields two).
    """
    imports: list[ImportRef] = []

    for node in _walk(root):
        if node.type == _PYTHON_IMPORT_TYPE:
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(ImportRef(module_path=_text(source, child)))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        imports.append(ImportRef(module_path=_text(source, name_node)))

        elif node.type == _PYTHON_IMPORT_FROM_TYPE:
            module_node = node.child_by_field_name("module_name")
            if module_node is None:
                continue

            if module_node.type == "dotted_name":
                imports.append(ImportRef(module_path=_text(source, module_node)))
            elif module_node.type == "relative_import":
                dots = 0
                dotted_name = ""
                for child in module_node.children:
                    if child.type == "import_prefix":
                        dots = _text(source, child).count(".")
                    elif child.type == "dotted_name":
                        dotted_name = _text(source, child)
                imports.append(ImportRef(module_path=dotted_name, relative_level=dots))

    return imports


def _js_string_fragment_text(source: bytes, string_node: Node) -> str | None:
    """Extract the unquoted text of a JavaScript `string` node.

    Args:
        source: The full source file, as bytes.
        string_node: A `string` node.

    Returns:
        The decoded contents between the quotes, or None if `string_node`
        has no `string_fragment` child (e.g. an empty string literal).
    """
    fragment = next((child for child in string_node.children if child.type == "string_fragment"), None)
    return _text(source, fragment) if fragment is not None else None


def _js_require_module_path(source: bytes, call_node: Node) -> str | None:
    """Extract the module specifier from a CommonJS `require("...")` call.

    Matches a bare call to the `require` identifier with exactly one
    string-literal argument, regardless of what syntactically contains
    it — a plain `const x = require("./foo")`, a destructured
    `const { a, b } = require("./foo")`, or a chained
    `require("dotenv").config()` all resolve the same module path.

    Args:
        source: The full source file, as bytes.
        call_node: A `call_expression` node.

    Returns:
        The unquoted module specifier, or None if `call_node` isn't a
        single-string-argument call to a bare `require` identifier.
    """
    function_node = call_node.child_by_field_name("function")
    if function_node is None or function_node.type != "identifier":
        return None
    if _text(source, function_node) != _JS_REQUIRE_CALL_NAME:
        return None

    arguments_node = call_node.child_by_field_name("arguments")
    if arguments_node is None:
        return None
    string_args = [child for child in arguments_node.children if child.type == "string"]
    if len(string_args) != 1:
        return None

    return _js_string_fragment_text(source, string_args[0])


def _extract_javascript_imports(root: Node, source: bytes) -> list[ImportRef]:
    """Walk a parsed JavaScript file and extract every import.

    Covers both ES `import ... from "..."` statements and CommonJS
    `require("...")` calls (`const x = require("./foo")`, destructured
    `const { a, b } = require("./foo")`, or a bare/chained
    `require("./foo").anything()`) — Node.js backends overwhelmingly use
    the latter, so a JS import extractor that only recognized ES `import`
    would see zero dependencies anywhere in a typical CommonJS codebase.

    Args:
        root: The root node of the parsed file.
        source: The full source file, as bytes.

    Returns:
        One `ImportRef` per import found, with `module_path` set to the
        raw (unquoted) module specifier.
    """
    imports: list[ImportRef] = []

    for node in _walk(root):
        if node.type == _JS_IMPORT_TYPE:
            source_node = node.child_by_field_name("source")
            if source_node is None:
                continue
            module_path = _js_string_fragment_text(source, source_node)
            if module_path is not None:
                imports.append(ImportRef(module_path=module_path))

        elif node.type == _JS_CALL_TYPE:
            module_path = _js_require_module_path(source, node)
            if module_path is not None:
                imports.append(ImportRef(module_path=module_path))

    return imports


@dataclass(frozen=True)
class _LanguageSupport:
    """A registered language: its grammar plus its import extractor function."""

    language: Language
    extractor: Callable[[Node, bytes], list[ImportRef]]


class ImportExtractor:
    """Detects import statements within a whole source file.

    Supports Python and JavaScript, following the same per-language
    registry pattern as `ingestion.ast_parser.TreeSitterParser`.
    """

    def __init__(self) -> None:
        """Load every supported language's grammar and build its parser."""
        support_by_language = {
            "python": _LanguageSupport(Language(tspython.language()), _extract_python_imports),
            "javascript": _LanguageSupport(
                Language(tsjavascript.language()), _extract_javascript_imports
            ),
        }
        self._support = support_by_language
        self._parsers = {
            language: Parser(support.language) for language, support in support_by_language.items()
        }

    def extract(self, source_file: SourceFile) -> list[ImportRef]:
        """Extract every import statement in `source_file`.

        Args:
            source_file: The file to scan for import statements.

        Returns:
            One `ImportRef` per imported module. Empty if the language is
            unsupported or the file has no imports.

        Raises:
            ParsingError: If `source_file` cannot be read.
        """
        support = self._support.get(source_file.language)
        if support is None:
            logger.debug(
                "Import extraction skipped for %s: unsupported language %r",
                source_file.relative_path, source_file.language,
            )
            return []

        try:
            source_bytes = source_file.absolute_path.read_bytes()
        except OSError as exc:
            raise ParsingError(f"Failed to read {source_file.absolute_path}: {exc}") from exc

        parser = self._parsers[source_file.language]
        tree = parser.parse(source_bytes)
        return support.extractor(tree.root_node, source_bytes)


def resolve_import(
    import_ref: ImportRef,
    importing_file: SourceFile,
    language: str,
    file_id_by_relative_path: dict[str, str],
) -> str | None:
    """Resolve an `ImportRef` to the `file_id` of another discovered file.

    Only imports that refer to a file within this repository can resolve;
    external packages (``os``, ``react``, ...) have no corresponding file
    and correctly resolve to None.

    Args:
        import_ref: The import to resolve.
        importing_file: The file the import statement appeared in, used
            to resolve Python relative imports and JavaScript relative
            specifiers against its own directory.
        language: `importing_file`'s language, selecting the resolution
            strategy.
        file_id_by_relative_path: Every discovered file's POSIX-style
            repository-relative path (without extension for Python
            candidates, as-is for JavaScript candidates — see
            `_candidate_paths`) mapped to its `file_id`.

    Returns:
        The imported file's `file_id`, or None if it cannot be resolved
        to a file in this repository.
    """
    if language == "python":
        candidates = _python_candidate_paths(import_ref, importing_file)
    elif language == "javascript":
        candidates = _javascript_candidate_paths(import_ref, importing_file)
    else:
        return None

    for candidate in candidates:
        file_id = file_id_by_relative_path.get(candidate)
        if file_id is not None:
            return file_id
    return None


def _python_candidate_paths(import_ref: ImportRef, importing_file: SourceFile) -> list[str]:
    """Compute candidate repository-relative module paths for a Python import.

    Args:
        import_ref: The import to resolve.
        importing_file: The file the import statement appeared in.

    Returns:
        Candidate dotted-to-slash paths (POSIX-style, no extension) to
        look up in `file_id_by_relative_path`, most likely first. Empty
        if `import_ref` carries no usable module path at all.
    """
    segments = import_ref.module_path.split(".") if import_ref.module_path else []

    if import_ref.relative_level > 0:
        # `relative_level` dots: 1 dot means "this package" (the
        # importing file's own directory); each extra dot climbs one
        # more directory level.
        base = posixpath.dirname(importing_file.relative_path.as_posix())
        for _ in range(import_ref.relative_level - 1):
            base = posixpath.dirname(base)

        if not segments:
            return [f"{base}/__init__"] if base else ["__init__"]
        module_str = posixpath.join(base, *segments) if base else posixpath.join(*segments)
    else:
        if not segments:
            return []
        module_str = posixpath.join(*segments)

    return [module_str, f"{module_str}/__init__"]


def _javascript_candidate_paths(import_ref: ImportRef, importing_file: SourceFile) -> list[str]:
    """Compute candidate repository-relative file paths for a JavaScript import.

    Args:
        import_ref: The import to resolve.
        importing_file: The file the import statement appeared in.

    Returns:
        Candidate paths (POSIX-style, with extension) to look up in
        `file_id_by_relative_path`, most likely first. Empty for a bare
        specifier (e.g. ``"react"``), which is always external.
    """
    specifier = import_ref.module_path
    if not specifier.startswith((".", "/")):
        return []  # Bare specifier: a package, never a repository-local file.

    importing_dir = posixpath.dirname(importing_file.relative_path.as_posix())
    resolved = posixpath.normpath(posixpath.join(importing_dir, specifier))
    return [f"{resolved}{suffix}" for suffix in _JS_RESOLUTION_SUFFIXES]
