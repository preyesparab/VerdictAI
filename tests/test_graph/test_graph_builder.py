"""Tests for graph.graph_builder.RepositoryGraphBuilder.

Builds small synthetic repositories on disk and runs them through the
real `FileDiscovery` -> `TreeSitterParser` -> `RepositoryGraphBuilder`
pipeline, so these tests exercise real Tree-sitter parsing end to end
rather than hand-constructed `CodeChunk` fixtures.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from graph.graph_builder import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_FUNCTION_CALL,
    EDGE_TYPE_IMPORTS,
    EDGE_TYPE_INHERITS,
    EDGE_TYPE_METHOD_CALL,
    EDGE_TYPE_REFERENCES,
    NODE_KIND_CHUNK,
    NODE_KIND_FILE,
    RepositoryGraphBuilder,
)
from ingestion.ast_parser import TreeSitterParser
from ingestion.file_discovery import FileDiscovery
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk


def _write_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for relative_path, content in files.items():
        full_path = tmp_path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    return tmp_path


def _build(tmp_path: Path, files: dict[str, str]) -> tuple[list[SourceFile], list[CodeChunk], nx.DiGraph]:
    repo_path = _write_repo(tmp_path, files)
    source_files = FileDiscovery().discover(repo_path)
    ast_chunks = TreeSitterParser().parse_many(source_files)
    graph = RepositoryGraphBuilder().build_graph(source_files, ast_chunks)
    return source_files, ast_chunks, graph


def _find_chunk(ast_chunks: list[CodeChunk], **filters: object) -> CodeChunk:
    for chunk in ast_chunks:
        if all(getattr(chunk, key) == value for key, value in filters.items()):
            return chunk
    raise AssertionError(f"No chunk matched {filters}")


class TestFunctionCalls:
    def test_direct_function_call_creates_edge(self, tmp_path: Path) -> None:
        _, ast_chunks, graph = _build(
            tmp_path,
            {"main.py": "def helper():\n    return 1\n\n\ndef caller():\n    return helper()\n"},
        )
        caller = _find_chunk(ast_chunks, function_name="caller")
        helper = _find_chunk(ast_chunks, function_name="helper")

        assert graph.has_edge(str(caller.chunk_id), str(helper.chunk_id))
        assert graph[str(caller.chunk_id)][str(helper.chunk_id)]["edge_type"] == EDGE_TYPE_FUNCTION_CALL

    def test_call_to_undefined_function_does_not_crash(self, tmp_path: Path) -> None:
        _, ast_chunks, graph = _build(
            tmp_path, {"main.py": "def caller():\n    return undefined_external()\n"}
        )
        caller = _find_chunk(ast_chunks, function_name="caller")
        assert graph.out_degree(str(caller.chunk_id)) == 0


class TestRecursiveCalls:
    def test_recursive_call_creates_self_loop(self, tmp_path: Path) -> None:
        _, ast_chunks, graph = _build(
            tmp_path,
            {"main.py": "def fib(n):\n    return fib(n - 1) + fib(n - 2)\n"},
        )
        fib = _find_chunk(ast_chunks, function_name="fib")
        node_id = str(fib.chunk_id)

        assert graph.has_edge(node_id, node_id)
        assert graph[node_id][node_id]["edge_type"] == EDGE_TYPE_FUNCTION_CALL


class TestMethodCalls:
    def test_self_qualified_method_call_creates_edge(self, tmp_path: Path) -> None:
        source = (
            "class Service:\n"
            "    def run(self):\n"
            "        return self.helper()\n\n"
            "    def helper(self):\n"
            "        return 1\n"
        )
        _, ast_chunks, graph = _build(tmp_path, {"main.py": source})
        run = _find_chunk(ast_chunks, function_name="run", chunk_type=ChunkType.METHOD)
        helper = _find_chunk(ast_chunks, function_name="helper", chunk_type=ChunkType.METHOD)

        assert graph.has_edge(str(run.chunk_id), str(helper.chunk_id))
        assert graph[str(run.chunk_id)][str(helper.chunk_id)]["edge_type"] == EDGE_TYPE_METHOD_CALL

    def test_class_qualified_method_call_resolves_to_correct_class(self, tmp_path: Path) -> None:
        source = (
            "class A:\n"
            "    def target(self):\n"
            "        return 1\n\n"
            "class B:\n"
            "    def target(self):\n"
            "        return 2\n\n"
            "def caller():\n"
            "    return A.target()\n"
        )
        _, ast_chunks, graph = _build(tmp_path, {"main.py": source})
        caller = _find_chunk(ast_chunks, function_name="caller")
        target_a = _find_chunk(ast_chunks, function_name="target", class_name="A")
        target_b = _find_chunk(ast_chunks, function_name="target", class_name="B")

        assert graph.has_edge(str(caller.chunk_id), str(target_a.chunk_id))
        assert not graph.has_edge(str(caller.chunk_id), str(target_b.chunk_id))


class TestInheritance:
    def test_single_inheritance_creates_edge(self, tmp_path: Path) -> None:
        source = "class Base:\n    pass\n\n\nclass Child(Base):\n    pass\n"
        _, ast_chunks, graph = _build(tmp_path, {"main.py": source})
        base = _find_chunk(ast_chunks, class_name="Base")
        child = _find_chunk(ast_chunks, class_name="Child")

        assert graph.has_edge(str(child.chunk_id), str(base.chunk_id))
        assert graph[str(child.chunk_id)][str(base.chunk_id)]["edge_type"] == EDGE_TYPE_INHERITS

    def test_unresolved_base_class_does_not_crash(self, tmp_path: Path) -> None:
        _, ast_chunks, graph = _build(tmp_path, {"main.py": "class Child(ExternalBase):\n    pass\n"})
        child = _find_chunk(ast_chunks, class_name="Child")
        assert graph.out_degree(str(child.chunk_id)) == 0


class TestClassContainment:
    def test_class_contains_its_methods(self, tmp_path: Path) -> None:
        source = "class Service:\n    def a(self):\n        pass\n\n    def b(self):\n        pass\n"
        _, ast_chunks, graph = _build(tmp_path, {"main.py": source})
        service = _find_chunk(ast_chunks, class_name="Service", chunk_type=ChunkType.CLASS)
        method_a = _find_chunk(ast_chunks, function_name="a")
        method_b = _find_chunk(ast_chunks, function_name="b")

        assert graph.has_edge(str(service.chunk_id), str(method_a.chunk_id))
        assert graph.has_edge(str(service.chunk_id), str(method_b.chunk_id))
        assert graph[str(service.chunk_id)][str(method_a.chunk_id)]["edge_type"] == EDGE_TYPE_CONTAINS


class TestImports:
    def test_python_absolute_import_creates_file_edge(self, tmp_path: Path) -> None:
        source_files, _, graph = _build(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/helper.py": "def f():\n    pass\n",
                "main.py": "import pkg.helper\n",
            },
        )
        main_id = next(
            f for f in source_files if f.relative_path.as_posix() == "main.py"
        )
        helper_id = next(
            f for f in source_files if f.relative_path.as_posix() == "pkg/helper.py"
        )
        from ingestion.deterministic_ids import compute_file_id

        main_node = compute_file_id(main_id.relative_path)
        helper_node = compute_file_id(helper_id.relative_path)

        assert graph.has_edge(main_node, helper_node)
        assert graph[main_node][helper_node]["edge_type"] == EDGE_TYPE_IMPORTS

    def test_javascript_relative_import_creates_file_edge(self, tmp_path: Path) -> None:
        source_files, _, graph = _build(
            tmp_path,
            {
                "src/index.js": 'import { helper } from "./helper";\n',
                "src/helper.js": "export function helper() {}\n",
            },
        )
        from ingestion.deterministic_ids import compute_file_id

        index_node = compute_file_id(Path("src/index.js"))
        helper_node = compute_file_id(Path("src/helper.js"))

        assert graph.has_edge(index_node, helper_node)

    def test_external_import_does_not_crash_and_is_unresolved(self, tmp_path: Path) -> None:
        source_files, _, graph = _build(tmp_path, {"main.py": "import os\nimport requests\n"})
        from ingestion.deterministic_ids import compute_file_id

        main_node = compute_file_id(Path("main.py"))
        assert graph.out_degree(main_node) == 0


class TestModuleLevelWiring:
    def test_handler_reference_creates_file_to_chunk_edge(self, tmp_path: Path) -> None:
        source_files, ast_chunks, graph = _build(
            tmp_path,
            {
                "controller.js": 'exports.register = async (req, res) => {\n  return res.json({});\n};\n',
                "routes.js": (
                    'const { register } = require("./controller");\n'
                    'router.post("/register", register);\n'
                ),
            },
        )
        from ingestion.deterministic_ids import compute_file_id

        routes_node = compute_file_id(Path("routes.js"))
        register = _find_chunk(ast_chunks, function_name="register")

        assert graph.has_edge(routes_node, str(register.chunk_id))
        assert graph[routes_node][str(register.chunk_id)]["edge_type"] == EDGE_TYPE_REFERENCES

    def test_commonjs_require_creates_import_edge_alongside_reference(self, tmp_path: Path) -> None:
        source_files, ast_chunks, graph = _build(
            tmp_path,
            {
                "controller.js": 'exports.register = async (req, res) => {\n  return res.json({});\n};\n',
                "routes.js": (
                    'const { register } = require("./controller");\n'
                    'router.post("/register", register);\n'
                ),
            },
        )
        from ingestion.deterministic_ids import compute_file_id

        routes_node = compute_file_id(Path("routes.js"))
        controller_node = compute_file_id(Path("controller.js"))

        assert graph.has_edge(routes_node, controller_node)
        assert graph[routes_node][controller_node]["edge_type"] == EDGE_TYPE_IMPORTS

    def test_module_level_call_creates_function_call_edge_from_file(self, tmp_path: Path) -> None:
        source_files, ast_chunks, graph = _build(
            tmp_path,
            {"main.js": "function setup() {\n  return 1;\n}\nsetup();\n"},
        )
        from ingestion.deterministic_ids import compute_file_id

        main_node = compute_file_id(Path("main.js"))
        setup = _find_chunk(ast_chunks, function_name="setup")

        assert graph.has_edge(main_node, str(setup.chunk_id))
        assert graph[main_node][str(setup.chunk_id)]["edge_type"] == EDGE_TYPE_FUNCTION_CALL

    def test_reference_inside_function_body_is_not_scanned(self, tmp_path: Path) -> None:
        source_files, ast_chunks, graph = _build(
            tmp_path,
            {
                "controller.js": 'exports.handler = async (req, res) => {\n  return 1;\n};\n',
                "wiring.js": (
                    'const { handler } = require("./controller");\n'
                    "function setup() {\n"
                    "  router.post('/x', handler);\n"
                    "}\n"
                ),
            },
        )
        from ingestion.deterministic_ids import compute_file_id

        wiring_node = compute_file_id(Path("wiring.js"))
        handler = _find_chunk(ast_chunks, function_name="handler")

        assert not graph.has_edge(wiring_node, str(handler.chunk_id))


class TestGraphValidation:
    def test_duplicate_nodes_are_ignored(self, tmp_path: Path) -> None:
        _, ast_chunks, graph = _build(tmp_path, {"main.py": "def f():\n    pass\n"})
        chunk = _find_chunk(ast_chunks, function_name="f")
        node_count_before = graph.number_of_nodes()

        graph.add_node(str(chunk.chunk_id), node_kind=NODE_KIND_CHUNK)
        assert graph.number_of_nodes() == node_count_before

    def test_full_synthetic_repo_produces_connected_graph(self, tmp_path: Path) -> None:
        source = (
            "class Base:\n"
            "    def greet(self):\n"
            "        return 'hi'\n\n\n"
            "class Child(Base):\n"
            "    def greet_twice(self):\n"
            "        return self.greet() + self.greet()\n\n\n"
            "def standalone():\n"
            "    return standalone_helper()\n\n\n"
            "def standalone_helper():\n"
            "    return 42\n"
        )
        _, ast_chunks, graph = _build(tmp_path, {"main.py": source})

        assert graph.number_of_nodes() >= len(ast_chunks) + 1  # + the file node
        assert graph.number_of_edges() > 0
        assert all("node_kind" in graph.nodes[n] for n in graph.nodes)


def test_all_node_kinds_present(tmp_path: Path) -> None:
    _, _, graph = _build(tmp_path, {"main.py": "def f():\n    pass\n"})
    kinds = {data["node_kind"] for _, data in graph.nodes(data=True)}
    assert kinds == {NODE_KIND_FILE, NODE_KIND_CHUNK}
