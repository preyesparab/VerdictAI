"""RepositoryGraphBuilder: constructs a directed knowledge graph from parsed chunks.

Builds one `networkx.DiGraph` per repository snapshot, with a node per
discovered file and per AST chunk, and directed edges for:

- Function calls (``caller_function -> callee_function``)
- Method calls (``caller_method -> callee_method``)
- Class inheritance (``ChildClass -> ParentClass``)
- Import relationships (``File A -> Imported File``)
- Class containment (``Class -> Method``)
- Module-scope calls/references (``File -> callee``/``File -> referenced``),
  for JavaScript handler-wiring code living outside any function/class
  body (e.g. Express's ``router.post("/x", handler)``)

Every edge type is resolved heuristically by name (see `_SymbolTable`
below and `graph.call_extractor`/`graph.import_extractor`) — there is no
type inference or full symbol resolution. A reference that cannot be
resolved (an external import, a dynamically dispatched call, an unknown
base class) is logged and skipped; it never raises, so a repository with
partially-resolvable references still produces a valid, usable graph.
"""

from __future__ import annotations

import posixpath
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from core.exceptions import ParsingError
from core.logging import get_logger
from graph.call_extractor import CallExtractor, CallSite
from graph.import_extractor import ImportExtractor, resolve_import
from ingestion.deterministic_ids import compute_file_id
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk

logger = get_logger(__name__)

NODE_KIND_FILE = "file"
NODE_KIND_CHUNK = "chunk"

EDGE_TYPE_FUNCTION_CALL = "function_call"
EDGE_TYPE_METHOD_CALL = "method_call"
EDGE_TYPE_INHERITS = "inherits"
EDGE_TYPE_IMPORTS = "imports"
EDGE_TYPE_CONTAINS = "contains"
EDGE_TYPE_REFERENCES = "references"

_CALLABLE_TYPES = frozenset(
    {ChunkType.FUNCTION, ChunkType.ASYNC_FUNCTION, ChunkType.METHOD, ChunkType.ARROW_FUNCTION}
)


@dataclass
class _SymbolTable:
    """Repo-wide name -> chunk_id indexes used to resolve edges by name.

    Built once per `build_graph` call from the full `ast_chunks` list.
    All lookups are heuristic (name-based, no type inference); see
    `resolve_call` for the exact preference order used to disambiguate a
    name that matches more than one chunk.
    """

    chunk_by_id: dict[str, CodeChunk] = field(default_factory=dict)
    classes_by_file_and_name: dict[tuple[str, str], str] = field(default_factory=dict)
    classes_by_name: dict[str, list[str]] = field(default_factory=dict)
    functions_by_name: dict[str, list[str]] = field(default_factory=dict)
    methods_by_class_and_name: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    methods_by_name: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, ast_chunks: list[CodeChunk]) -> "_SymbolTable":
        """Index every chunk by id and by the names other chunks might reference it by.

        Args:
            ast_chunks: Every AST chunk in the repository.

        Returns:
            A populated `_SymbolTable`.
        """
        chunk_by_id: dict[str, CodeChunk] = {}
        classes_by_file_and_name: dict[tuple[str, str], str] = {}
        classes_by_name: dict[str, list[str]] = defaultdict(list)
        functions_by_name: dict[str, list[str]] = defaultdict(list)
        methods_by_class_and_name: dict[tuple[str, str], list[str]] = defaultdict(list)
        methods_by_name: dict[str, list[str]] = defaultdict(list)

        for chunk in ast_chunks:
            node_id = str(chunk.chunk_id)
            chunk_by_id[node_id] = chunk

            if chunk.chunk_type == ChunkType.CLASS and chunk.class_name:
                classes_by_file_and_name.setdefault((chunk.file_id, chunk.class_name), node_id)
                classes_by_name[chunk.class_name].append(node_id)
            elif chunk.chunk_type == ChunkType.METHOD and chunk.function_name:
                methods_by_name[chunk.function_name].append(node_id)
                if chunk.class_name:
                    methods_by_class_and_name[(chunk.class_name, chunk.function_name)].append(node_id)
            elif (
                chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.ASYNC_FUNCTION, ChunkType.ARROW_FUNCTION)
                and chunk.function_name
            ):
                functions_by_name[chunk.function_name].append(node_id)

        return cls(
            chunk_by_id=chunk_by_id,
            classes_by_file_and_name=classes_by_file_and_name,
            classes_by_name=dict(classes_by_name),
            functions_by_name=dict(functions_by_name),
            methods_by_class_and_name=dict(methods_by_class_and_name),
            methods_by_name=dict(methods_by_name),
        )

    def class_in_file(self, file_id: str, class_name: str) -> str | None:
        """Find a class chunk by name, restricted to one file (for containment)."""
        return self.classes_by_file_and_name.get((file_id, class_name))

    def class_by_name(self, preferred_file_id: str, class_name: str) -> str | None:
        """Find a class chunk by name, preferring `preferred_file_id`, else repo-wide.

        Args:
            preferred_file_id: The file to look in first (e.g. a
                subclass's own file, since same-file base classes are
                far more common than cross-file ones).
            class_name: The class name to resolve.

        Returns:
            A matching class chunk id, or None if no class with this
            name was parsed anywhere in the repository (e.g. it is a
            standard-library or third-party base class).
        """
        same_file = self.classes_by_file_and_name.get((preferred_file_id, class_name))
        if same_file is not None:
            return same_file
        candidates = self.classes_by_name.get(class_name)
        return candidates[0] if candidates else None

    def resolve_call(self, caller: CodeChunk, call_site: CallSite) -> str | None:
        """Best-effort resolution of one call site to a chunk id.

        Preference order:
          1. A ``self``/``cls``/``this``-qualified call resolves to a
             method of the caller's own class.
          2. A call qualified by a known class name resolves to that
             class's method.
          3. Otherwise, prefer a plain function with a matching name
             (what an unqualified call syntactically is); fall back to
             any same-named method only if no function matches.
          4. Among remaining candidates, prefer one in the caller's own
             file (same-file references are far more common and far
             less ambiguous than cross-file ones).

        Args:
            caller: The chunk the call was made from.
            call_site: The detected call expression.

        Returns:
            A resolved callee chunk id, or None if `call_site.callee_name`
            does not match any chunk parsed in this repository.
        """
        name = call_site.callee_name

        if call_site.is_self_qualified and caller.class_name:
            same_class = self.methods_by_class_and_name.get((caller.class_name, name))
            if same_class:
                return same_class[0]

        if call_site.qualifier_name and call_site.qualifier_name in self.classes_by_name:
            qualified = self.methods_by_class_and_name.get((call_site.qualifier_name, name))
            if qualified:
                return qualified[0]

        candidate_pool = self.functions_by_name.get(name) or self.methods_by_name.get(name)
        if not candidate_pool:
            return None

        same_file = [cid for cid in candidate_pool if self.chunk_by_id[cid].file_id == caller.file_id]
        return same_file[0] if same_file else candidate_pool[0]

    def resolve_name(self, preferred_file_id: str, name: str) -> str | None:
        """Best-effort resolution of a bare name to a chunk id, preferring `preferred_file_id`.

        Used for module-scope call sites and argument references (see
        `graph.call_extractor.CallExtractor.extract_module_level`), which
        have no enclosing chunk - so no `self`/`cls`/`this` qualification
        or caller-class context is available, unlike `resolve_call`.

        Args:
            preferred_file_id: The file the reference was found in, to
                prefer for disambiguation (same-file references are more
                common and less ambiguous than cross-file ones).
            name: The identifier name to resolve.

        Returns:
            A resolved chunk id, or None if `name` matches no chunk
            parsed in this repository.
        """
        candidate_pool = self.functions_by_name.get(name) or self.methods_by_name.get(name)
        if not candidate_pool:
            return None

        same_file = [cid for cid in candidate_pool if self.chunk_by_id[cid].file_id == preferred_file_id]
        return same_file[0] if same_file else candidate_pool[0]


def _build_path_lookup(source_files: list[SourceFile]) -> dict[str, str]:
    """Index every discovered file by every path form an import might resolve to.

    Each file is indexed by its full relative path (for JavaScript
    specifiers, which include an extension) and, for Python files, also
    by its extension-stripped path (for dotted module paths).

    Args:
        source_files: Every file discovered by `FileDiscovery`.

    Returns:
        A mapping from candidate path string to `file_id`, for use with
        `graph.import_extractor.resolve_import`.
    """
    lookup: dict[str, str] = {}
    for source_file in source_files:
        file_id = compute_file_id(source_file.relative_path)
        relative_posix = source_file.relative_path.as_posix()
        lookup[relative_posix] = file_id

        if source_file.language == "python":
            stem, _ext = posixpath.splitext(relative_posix)
            lookup.setdefault(stem, file_id)

    return lookup


class RepositoryGraphBuilder:
    """Builds a directed knowledge graph of a repository's structural relationships.

    Nodes are files and AST chunks (functions, classes, methods); edges
    capture function/method calls, class inheritance, file imports, and
    class-to-method containment. Construction never raises on an
    unresolved reference — every extraction step logs and skips instead.
    """

    def __init__(
        self,
        call_extractor: CallExtractor | None = None,
        import_extractor: ImportExtractor | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            call_extractor: Detects call sites within a chunk's body.
                Defaults to a new `CallExtractor`. Overridable for testing.
            import_extractor: Detects import statements within a file.
                Defaults to a new `ImportExtractor`. Overridable for testing.
        """
        self._call_extractor = call_extractor or CallExtractor()
        self._import_extractor = import_extractor or ImportExtractor()

    def build_graph(self, source_files: list[SourceFile], ast_chunks: list[CodeChunk]) -> nx.DiGraph:
        """Build the full knowledge graph for one repository snapshot.

        Args:
            source_files: Every file discovered by `FileDiscovery`.
            ast_chunks: Every AST chunk extracted by `TreeSitterParser`
                across `source_files`.

        Returns:
            A `networkx.DiGraph` with one node per file and per chunk,
            and directed edges for calls, inheritance, imports, and
            class containment.
        """
        logger.info(
            "Graph construction started: %d file(s), %d chunk(s)", len(source_files), len(ast_chunks)
        )
        graph = nx.DiGraph()

        self._add_file_nodes(graph, source_files)
        self._add_chunk_nodes(graph, ast_chunks)

        symbols = _SymbolTable.build(ast_chunks)
        unresolved = 0
        unresolved += self._add_containment_edges(graph, ast_chunks, symbols)
        unresolved += self._add_inheritance_edges(graph, ast_chunks, symbols)
        unresolved += self._add_call_edges(graph, ast_chunks, symbols)
        unresolved += self._add_module_level_edges(graph, source_files, symbols)
        unresolved += self._add_import_edges(graph, source_files)

        logger.info(
            "Graph construction complete: %d node(s), %d edge(s), %d unresolved reference(s) total",
            graph.number_of_nodes(), graph.number_of_edges(), unresolved,
        )
        return graph

    # -- Nodes -------------------------------------------------------------

    def _add_file_nodes(self, graph: nx.DiGraph, source_files: list[SourceFile]) -> None:
        """Add one node per discovered file, skipping duplicates.

        Args:
            graph: The graph under construction.
            source_files: Every file discovered by `FileDiscovery`.
        """
        added = 0
        for source_file in source_files:
            file_id = compute_file_id(source_file.relative_path)
            if file_id in graph:
                continue
            graph.add_node(
                file_id,
                node_kind=NODE_KIND_FILE,
                file_id=file_id,
                file_path=source_file.relative_path.as_posix(),
                language=source_file.language,
            )
            added += 1
        logger.info("Added %d file node(s)", added)

    def _add_chunk_nodes(self, graph: nx.DiGraph, ast_chunks: list[CodeChunk]) -> None:
        """Add one node per AST chunk, skipping duplicates.

        Args:
            graph: The graph under construction.
            ast_chunks: Every AST chunk in the repository.
        """
        added = 0
        duplicates = 0
        for chunk in ast_chunks:
            node_id = str(chunk.chunk_id)
            if node_id in graph:
                duplicates += 1
                continue
            graph.add_node(
                node_id,
                node_kind=NODE_KIND_CHUNK,
                chunk_id=node_id,
                file_id=chunk.file_id,
                file_path=chunk.file_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type.value,
                function_name=chunk.function_name,
                class_name=chunk.class_name,
                parent_class=chunk.parent_class,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            added += 1
        logger.info("Added %d chunk node(s) (%d duplicate(s) ignored)", added, duplicates)

    # -- Edges ---------------------------------------------------------------

    def _add_edge_if_new(self, graph: nx.DiGraph, source: str, target: str, edge_type: str) -> bool:
        """Add a directed edge unless an edge already exists between these nodes.

        Args:
            graph: The graph under construction.
            source: Source node id.
            target: Target node id.
            edge_type: Relationship label stored on the edge.

        Returns:
            True if a new edge was added; False if one already existed
            (a duplicate edge, ignored).
        """
        if graph.has_edge(source, target):
            return False
        graph.add_edge(source, target, edge_type=edge_type)
        return True

    def _add_containment_edges(
        self, graph: nx.DiGraph, ast_chunks: list[CodeChunk], symbols: _SymbolTable
    ) -> int:
        """Add ``Class -> Method`` edges for every method with a resolvable containing class.

        Returns:
            The number of methods whose containing class could not be resolved.
        """
        added = 0
        unresolved = 0
        for chunk in ast_chunks:
            if chunk.chunk_type != ChunkType.METHOD or chunk.class_name is None:
                continue
            class_chunk_id = symbols.class_in_file(chunk.file_id, chunk.class_name)
            if class_chunk_id is None:
                unresolved += 1
                logger.debug(
                    "Unresolved containment: class %r not found for method %r in %s",
                    chunk.class_name, chunk.function_name, chunk.file_path,
                )
                continue
            if self._add_edge_if_new(graph, class_chunk_id, str(chunk.chunk_id), EDGE_TYPE_CONTAINS):
                added += 1
        logger.info("Added %d containment edge(s), %d unresolved", added, unresolved)
        return unresolved

    def _add_inheritance_edges(
        self, graph: nx.DiGraph, ast_chunks: list[CodeChunk], symbols: _SymbolTable
    ) -> int:
        """Add ``ChildClass -> ParentClass`` edges for every resolvable base class.

        Returns:
            The number of base classes that could not be resolved (e.g.
            standard-library or third-party base classes).
        """
        added = 0
        unresolved = 0
        for chunk in ast_chunks:
            if chunk.chunk_type != ChunkType.CLASS or not chunk.parent_class:
                continue
            base_names = [name.strip() for name in chunk.parent_class.split(",") if name.strip()]
            for base_name in base_names:
                parent_chunk_id = symbols.class_by_name(chunk.file_id, base_name)
                if parent_chunk_id is None:
                    unresolved += 1
                    logger.debug(
                        "Unresolved base class %r for %r in %s",
                        base_name, chunk.class_name, chunk.file_path,
                    )
                    continue
                if self._add_edge_if_new(graph, str(chunk.chunk_id), parent_chunk_id, EDGE_TYPE_INHERITS):
                    added += 1
        logger.info("Added %d inheritance edge(s), %d unresolved", added, unresolved)
        return unresolved

    def _add_call_edges(
        self, graph: nx.DiGraph, ast_chunks: list[CodeChunk], symbols: _SymbolTable
    ) -> int:
        """Add ``caller -> callee`` edges (function_call or method_call) for every resolvable call site.

        Returns:
            The number of call sites that could not be resolved to a
            parsed chunk (e.g. calls into external libraries).
        """
        added = 0
        unresolved = 0
        for chunk in ast_chunks:
            if chunk.chunk_type not in _CALLABLE_TYPES:
                continue

            call_sites = self._call_extractor.extract(chunk.language, chunk.raw_code)
            for call_site in call_sites:
                callee_chunk_id = symbols.resolve_call(chunk, call_site)
                if callee_chunk_id is None:
                    unresolved += 1
                    logger.debug(
                        "Unresolved call %r from %r in %s",
                        call_site.callee_name, chunk.function_name, chunk.file_path,
                    )
                    continue

                callee_chunk = symbols.chunk_by_id[callee_chunk_id]
                edge_type = (
                    EDGE_TYPE_METHOD_CALL if callee_chunk.chunk_type == ChunkType.METHOD
                    else EDGE_TYPE_FUNCTION_CALL
                )
                if self._add_edge_if_new(graph, str(chunk.chunk_id), callee_chunk_id, edge_type):
                    added += 1
        logger.info("Added %d call edge(s), %d unresolved", added, unresolved)
        return unresolved

    def _add_module_level_edges(
        self, graph: nx.DiGraph, source_files: list[SourceFile], symbols: _SymbolTable
    ) -> int:
        """Add ``File -> callee``/``File -> referenced`` edges for a file's top-level code.

        Handler-wiring idioms like Express's ``router.post("/x", handler)``
        live at module scope, outside any function - `_add_call_edges`
        only scans chunks of a callable type, so this code is invisible
        to the call graph regardless of how well the functions it
        references are named or chunked. `handler` here is passed by
        reference, not invoked, so it resolves to `EDGE_TYPE_REFERENCES`,
        not a call; the edge's source is the *file* node, since this
        code belongs to no function/method chunk.

        Args:
            graph: The graph under construction.
            source_files: Every file discovered by `FileDiscovery`.
            symbols: The repo-wide name index built from `ast_chunks`.

        Returns:
            The number of module-scope call sites/references that could
            not be resolved to a parsed chunk.
        """
        added = 0
        unresolved = 0

        for source_file in source_files:
            try:
                source_code = source_file.absolute_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Module-level scan skipped for %s: %s", source_file.relative_path, exc)
                continue

            file_id = compute_file_id(source_file.relative_path)
            sites = self._call_extractor.extract_module_level(source_file.language, source_code)

            for call_site in sites.call_sites:
                callee_chunk_id = symbols.resolve_name(file_id, call_site.callee_name)
                if callee_chunk_id is None:
                    unresolved += 1
                    continue
                callee_chunk = symbols.chunk_by_id[callee_chunk_id]
                edge_type = (
                    EDGE_TYPE_METHOD_CALL if callee_chunk.chunk_type == ChunkType.METHOD
                    else EDGE_TYPE_FUNCTION_CALL
                )
                if self._add_edge_if_new(graph, file_id, callee_chunk_id, edge_type):
                    added += 1

            for reference in sites.reference_sites:
                referenced_chunk_id = symbols.resolve_name(file_id, reference.name)
                if referenced_chunk_id is None:
                    unresolved += 1
                    continue
                if self._add_edge_if_new(graph, file_id, referenced_chunk_id, EDGE_TYPE_REFERENCES):
                    added += 1

        logger.info("Added %d module-level edge(s), %d unresolved", added, unresolved)
        return unresolved

    def _add_import_edges(self, graph: nx.DiGraph, source_files: list[SourceFile]) -> int:
        """Add ``File A -> Imported File`` edges for every resolvable intra-repository import.

        Returns:
            The number of imports that could not be resolved to a file
            in this repository (e.g. external packages).
        """
        path_lookup = _build_path_lookup(source_files)
        added = 0
        unresolved = 0

        for source_file in source_files:
            file_id = compute_file_id(source_file.relative_path)
            try:
                import_refs = self._import_extractor.extract(source_file)
            except ParsingError as exc:
                logger.warning("Import extraction failed for %s: %s", source_file.relative_path, exc)
                continue

            for import_ref in import_refs:
                target_file_id = resolve_import(
                    import_ref, source_file, source_file.language, path_lookup
                )
                if target_file_id is None:
                    unresolved += 1
                    logger.debug(
                        "Unresolved import %r in %s", import_ref.module_path, source_file.relative_path
                    )
                    continue
                if target_file_id == file_id:
                    continue
                if self._add_edge_if_new(graph, file_id, target_file_id, EDGE_TYPE_IMPORTS):
                    added += 1

        logger.info("Added %d import edge(s), %d unresolved", added, unresolved)
        return unresolved
