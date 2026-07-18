"""Tests for ingestion.ast_parser.TreeSitterParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import ParsingError
from ingestion.ast_parser import TreeSitterParser
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk


def _write_source_file(tmp_path: Path, relative_name: str, content: str, language: str) -> SourceFile:
    absolute_path = tmp_path / relative_name
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    absolute_path.write_bytes(encoded)
    return SourceFile(
        absolute_path=absolute_path,
        relative_path=Path(relative_name),
        language=language,
        extension=absolute_path.suffix,
        size_bytes=len(encoded),
    )


def _by_name(chunks: list[CodeChunk], name: str) -> CodeChunk:
    for chunk in chunks:
        if chunk.function_name == name or chunk.class_name == name:
            return chunk
    raise AssertionError(f"No chunk named {name!r} in {[c.function_name or c.class_name for c in chunks]}")


@pytest.fixture
def parser() -> TreeSitterParser:
    return TreeSitterParser()


class TestUnsupportedAndErrors:
    def test_unsupported_language_raises_parsing_error(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.rs", "fn main() {}", "rust")
        with pytest.raises(ParsingError):
            parser.parse(source_file)

    def test_syntax_error_raises_parsing_error(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "broken.py", "def broken(:\n    pass\n", "python")
        with pytest.raises(ParsingError):
            parser.parse(source_file)

    def test_unreadable_file_raises_parsing_error(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        source_file = SourceFile(
            absolute_path=tmp_path / "missing.py",
            relative_path=Path("missing.py"),
            language="python",
            extension=".py",
            size_bytes=0,
        )
        with pytest.raises(ParsingError):
            parser.parse(source_file)


class TestEmptyFiles:
    def test_empty_python_file_returns_no_chunks(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "empty.py", "", "python")
        assert parser.parse(source_file) == []

    def test_empty_javascript_file_returns_no_chunks(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "empty.js", "", "javascript")
        assert parser.parse(source_file) == []

    def test_file_with_no_functions_or_classes_returns_no_chunks(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        source_file = _write_source_file(tmp_path, "constants.py", "X = 1\nY = 2\n", "python")
        assert parser.parse(source_file) == []


class TestPythonFunctions:
    def test_simple_function(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "def add(a, b):\n    return a + b\n"
        source_file = _write_source_file(tmp_path, "math_utils.py", code, "python")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.function_name == "add"
        assert chunk.class_name is None
        assert chunk.parent_class is None
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.raw_code == code.rstrip("\n")
        assert chunk.language == "python"
        assert chunk.file_path == "math_utils.py"

    def test_async_function(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "async def fetch(url):\n    return await get(url)\n"
        source_file = _write_source_file(tmp_path, "net.py", code, "python")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.ASYNC_FUNCTION
        assert chunk.function_name == "fetch"

    def test_nested_function_produces_two_chunks(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = (
            "def outer(x):\n"
            "    def inner(y):\n"
            "        return y * 2\n"
            "    return inner(x)\n"
        )
        source_file = _write_source_file(tmp_path, "nested.py", code, "python")

        chunks = parser.parse(source_file)

        assert {c.function_name for c in chunks} == {"outer", "inner"}
        outer = _by_name(chunks, "outer")
        inner = _by_name(chunks, "inner")
        assert outer.chunk_type == ChunkType.FUNCTION
        assert inner.chunk_type == ChunkType.FUNCTION
        assert inner.start_line == 2

    def test_class_and_methods(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = (
            "class Greeter:\n"
            "    def greet(self, name):\n"
            "        return f'hi {name}'\n"
            "\n"
            "    async def greet_async(self, name):\n"
            "        return f'hi {name}'\n"
        )
        source_file = _write_source_file(tmp_path, "greeter.py", code, "python")

        chunks = parser.parse(source_file)

        assert len(chunks) == 3
        cls = _by_name(chunks, "Greeter")
        greet = _by_name(chunks, "greet")
        greet_async = _by_name(chunks, "greet_async")

        assert cls.chunk_type == ChunkType.CLASS
        assert cls.function_name is None
        assert greet.chunk_type == ChunkType.METHOD
        assert greet.class_name == "Greeter"
        assert greet_async.chunk_type == ChunkType.METHOD
        assert greet_async.class_name == "Greeter"

    def test_class_with_base_classes(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "class Base:\n    pass\n\nclass Child(Base):\n    def run(self):\n        pass\n"
        source_file = _write_source_file(tmp_path, "hierarchy.py", code, "python")

        chunks = parser.parse(source_file)

        child = _by_name(chunks, "Child")
        run = _by_name(chunks, "run")
        assert child.parent_class == "Base"
        assert run.class_name == "Child"
        assert run.parent_class == "Base"

    def test_decorated_function_span_includes_decorator(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        code = "@staticmethod\ndef helper():\n    return 1\n"
        source_file = _write_source_file(tmp_path, "decorated.py", code, "python")

        [chunk] = parser.parse(source_file)

        assert chunk.start_line == 1
        assert chunk.raw_code.startswith("@staticmethod")

    def test_nested_function_inside_method_is_function_not_method(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        code = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        def helper():\n"
            "            return 1\n"
            "        return helper()\n"
        )
        source_file = _write_source_file(tmp_path, "foo.py", code, "python")

        chunks = parser.parse(source_file)

        helper = _by_name(chunks, "helper")
        assert helper.chunk_type == ChunkType.FUNCTION


class TestJavaScript:
    def test_function_declaration(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "function add(a, b) {\n    return a + b;\n}\n"
        source_file = _write_source_file(tmp_path, "math.js", code, "javascript")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.function_name == "add"
        assert chunk.language == "javascript"

    def test_named_arrow_function(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "const add = (a, b) => {\n    return a + b;\n};\n"
        source_file = _write_source_file(tmp_path, "arrow.js", code, "javascript")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.ARROW_FUNCTION
        assert chunk.function_name == "add"

    def test_anonymous_arrow_function_is_skipped(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "[1, 2, 3].map((x) => x * 2);\n"
        source_file = _write_source_file(tmp_path, "callback.js", code, "javascript")

        assert parser.parse(source_file) == []

    def test_commonjs_handler_export(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = 'exports.register = async (req, res) => {\n    return res.json({});\n};\n'
        source_file = _write_source_file(tmp_path, "controller.js", code, "javascript")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.ARROW_FUNCTION
        assert chunk.function_name == "register"

    def test_state_store_factory_export(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = (
            "const useAuthStore = create(\n"
            "  persist(\n"
            "    (set, get) => ({\n"
            "      user: null,\n"
            "      login: () => set({ user: 1 }),\n"
            "    }),\n"
            "    { name: 'auth' }\n"
            "  )\n"
            ");\n"
        )
        source_file = _write_source_file(tmp_path, "useAuthStore.js", code, "javascript")

        [chunk] = parser.parse(source_file)

        assert chunk.chunk_type == ChunkType.ARROW_FUNCTION
        assert chunk.function_name == "useAuthStore"

    def test_array_method_callback_assigned_to_const_is_skipped(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        code = "const payload = items.map((i) => ({ id: i.id }));\n"
        source_file = _write_source_file(tmp_path, "transform.js", code, "javascript")

        assert parser.parse(source_file) == []

    def test_block_bodied_hook_callback_is_skipped(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "const handleDrop = useCallback((e) => {\n    doThing();\n}, [dep]);\n"
        source_file = _write_source_file(tmp_path, "handler.js", code, "javascript")

        assert parser.parse(source_file) == []

    def test_class_declaration_and_methods(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = (
            "class Animal extends Base {\n"
            "    constructor(name) {\n"
            "        this.name = name;\n"
            "    }\n\n"
            "    speak() {\n"
            "        return this.name;\n"
            "    }\n"
            "}\n"
        )
        source_file = _write_source_file(tmp_path, "animal.js", code, "javascript")

        chunks = parser.parse(source_file)

        cls = _by_name(chunks, "Animal")
        speak = _by_name(chunks, "speak")
        assert cls.chunk_type == ChunkType.CLASS
        assert cls.parent_class == "Base"
        assert speak.chunk_type == ChunkType.METHOD
        assert speak.class_name == "Animal"
        assert speak.parent_class == "Base"


class TestDeterministicIds:
    def test_chunk_id_is_stable_across_parses(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        code = "def stable():\n    return 1\n"
        source_file = _write_source_file(tmp_path, "stable.py", code, "python")

        [first] = parser.parse(source_file)
        [second] = parser.parse(source_file)

        assert first.chunk_id == second.chunk_id
        assert first.file_id == second.file_id

    def test_different_files_with_same_function_name_get_different_chunk_ids(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        code = "def run():\n    return 1\n"
        file_a = _write_source_file(tmp_path, "a/run.py", code, "python")
        file_b = _write_source_file(tmp_path, "b/run.py", code, "python")

        [chunk_a] = parser.parse(file_a)
        [chunk_b] = parser.parse(file_b)

        assert chunk_a.chunk_id != chunk_b.chunk_id
        assert chunk_a.file_id != chunk_b.file_id


class TestParseMany:
    def test_continues_after_one_file_fails(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        good = _write_source_file(tmp_path, "good.py", "def ok():\n    return 1\n", "python")
        bad = _write_source_file(tmp_path, "bad.py", "def broken(:\n    pass\n", "python")

        chunks = parser.parse_many([bad, good])

        assert len(chunks) == 1
        assert chunks[0].function_name == "ok"

    def test_empty_list_returns_empty_list(self, parser: TreeSitterParser) -> None:
        assert parser.parse_many([]) == []
