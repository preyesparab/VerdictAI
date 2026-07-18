"""Tests for graph.import_extractor.ImportExtractor and resolve_import."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.source_file import SourceFile
from graph.import_extractor import ImportExtractor, ImportRef, resolve_import


def _write_source_file(tmp_path: Path, name: str, content: str, language: str) -> SourceFile:
    absolute_path = tmp_path / name
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    absolute_path.write_bytes(encoded)
    return SourceFile(
        absolute_path=absolute_path,
        relative_path=Path(name),
        language=language,
        extension=absolute_path.suffix,
        size_bytes=len(encoded),
    )


class TestPythonExtraction:
    def test_absolute_import(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.py", "import pkg.sub\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="pkg.sub", relative_level=0)]

    def test_aliased_import(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.py", "import pkg.sub as s\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="pkg.sub", relative_level=0)]

    def test_multiple_imports_in_one_statement(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.py", "import a, b.c\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [
            ImportRef(module_path="a", relative_level=0),
            ImportRef(module_path="b.c", relative_level=0),
        ]

    def test_from_import_absolute(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.py", "from pkg.sub import thing\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="pkg.sub", relative_level=0)]

    def test_from_import_relative_bare_dot(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "pkg/mod.py", "from . import sibling\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="", relative_level=1)]

    def test_from_import_relative_with_module(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "pkg/mod.py", "from ..other import thing\n", "python")
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="other", relative_level=2)]

    def test_no_imports_returns_empty(self, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "main.py", "x = 1\n", "python")
        assert ImportExtractor().extract(source_file) == []

    def test_unreadable_file_raises_parsing_error(self, tmp_path: Path) -> None:
        source_file = SourceFile(
            absolute_path=tmp_path / "missing.py",
            relative_path=Path("missing.py"),
            language="python",
            extension=".py",
            size_bytes=0,
        )
        from core.exceptions import ParsingError

        with pytest.raises(ParsingError):
            ImportExtractor().extract(source_file)


class TestJavaScriptExtraction:
    def test_default_import(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'import foo from "./foo";\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="./foo")]

    def test_named_import(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'import { a, b } from "../pkg";\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="../pkg")]

    def test_bare_specifier_import(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'import lodash from "lodash";\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="lodash")]

    def test_commonjs_require_plain(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'const foo = require("./foo");\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="./foo")]

    def test_commonjs_require_destructured(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'const { a, b } = require("../pkg");\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="../pkg")]

    def test_commonjs_require_chained_call_is_still_detected(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'require("dotenv").config();\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="dotenv")]

    def test_commonjs_require_bare_package_specifier(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", 'const express = require("express");\n', "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == [ImportRef(module_path="express")]

    def test_call_named_require_with_non_string_argument_is_ignored(self, tmp_path: Path) -> None:
        source_file = _write_source_file(
            tmp_path, "main.js", "const x = require(modulePath);\n", "javascript"
        )
        refs = ImportExtractor().extract(source_file)
        assert refs == []


class TestResolveImport:
    def test_python_absolute_import_resolves(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "main.py", "import pkg.sub\n", "python")
        lookup = {"pkg/sub": "sub-file-id"}
        target = resolve_import(
            ImportRef(module_path="pkg.sub"), importing_file, "python", lookup
        )
        assert target == "sub-file-id"

    def test_python_relative_bare_dot_resolves_to_package_init(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "pkg/mod.py", "from . import sibling\n", "python")
        lookup = {"pkg/__init__": "pkg-init-id"}
        target = resolve_import(
            ImportRef(module_path="", relative_level=1), importing_file, "python", lookup
        )
        assert target == "pkg-init-id"

    def test_python_relative_with_module_resolves(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "pkg/sub/mod.py", "", "python")
        lookup = {"pkg/other": "other-id"}
        target = resolve_import(
            ImportRef(module_path="other", relative_level=2), importing_file, "python", lookup
        )
        assert target == "other-id"

    def test_python_external_import_unresolved(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "main.py", "import os\n", "python")
        target = resolve_import(ImportRef(module_path="os"), importing_file, "python", {})
        assert target is None

    def test_javascript_relative_import_resolves_with_extension(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "src/index.js", "", "javascript")
        lookup = {"src/foo.js": "foo-id"}
        target = resolve_import(
            ImportRef(module_path="./foo"), importing_file, "javascript", lookup
        )
        assert target == "foo-id"

    def test_javascript_parent_relative_import_resolves(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "src/components/widget.js", "", "javascript")
        lookup = {"src/utils/helper.js": "helper-id"}
        target = resolve_import(
            ImportRef(module_path="../utils/helper"), importing_file, "javascript", lookup
        )
        assert target == "helper-id"

    def test_javascript_bare_specifier_unresolved(self, tmp_path: Path) -> None:
        importing_file = _write_source_file(tmp_path, "src/index.js", "", "javascript")
        target = resolve_import(
            ImportRef(module_path="react"), importing_file, "javascript", {"src/react.js": "wrong"}
        )
        assert target is None
