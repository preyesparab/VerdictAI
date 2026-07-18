"""Tests for ingestion.file_discovery.FileDiscovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.exceptions import ParsingError
from ingestion.file_discovery import IGNORED_DIRECTORIES, MAX_FILE_SIZE_BYTES, FileDiscovery


@pytest.fixture
def discovery() -> FileDiscovery:
    return FileDiscovery()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _relative_paths(source_files: list) -> set[str]:
    return {sf.relative_path.as_posix() for sf in source_files}


class TestDiscoverErrors:
    def test_nonexistent_path_raises_parsing_error(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        with pytest.raises(ParsingError):
            discovery.discover(tmp_path / "does-not-exist")

    def test_file_instead_of_directory_raises_parsing_error(
        self, discovery: FileDiscovery, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "not_a_dir.py"
        _write(file_path, b"print('hi')")
        with pytest.raises(ParsingError):
            discovery.discover(file_path)

    def test_os_error_during_walk_raises_parsing_error(
        self, discovery: FileDiscovery, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("ingestion.file_discovery.os.walk", side_effect=OSError("permission denied"))
        with pytest.raises(ParsingError):
            discovery.discover(tmp_path)


class TestDiscoverFiltering:
    def test_discovers_supported_files_recursively(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "main.py", b"print('hello')")
        _write(tmp_path / "src" / "utils.py", b"def f(): pass")
        _write(tmp_path / "src" / "app.js", b"console.log('hi');")

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"main.py", "src/utils.py", "src/app.js"}

    def test_ignores_unsupported_extensions(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "README.md", b"# hello")
        _write(tmp_path / "data.bin", b"\x00\x01\x02")
        _write(tmp_path / "main.py", b"print('hi')")

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"main.py"}

    @pytest.mark.parametrize("ignored_dir", sorted(IGNORED_DIRECTORIES))
    def test_ignores_each_ignored_directory(
        self, discovery: FileDiscovery, tmp_path: Path, ignored_dir: str
    ) -> None:
        _write(tmp_path / ignored_dir / "should_not_appear.py", b"print('hidden')")
        _write(tmp_path / "visible.py", b"print('visible')")

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"visible.py"}

    def test_ignores_oversized_files(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "small.py", b"x = 1")
        _write(tmp_path / "huge.py", b"x" * (MAX_FILE_SIZE_BYTES + 1))

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"small.py"}

    def test_file_exactly_at_size_limit_is_included(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "exact.py", b"x" * MAX_FILE_SIZE_BYTES)

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"exact.py"}

    def test_ignores_binary_files_with_supported_extension(
        self, discovery: FileDiscovery, tmp_path: Path
    ) -> None:
        _write(tmp_path / "looks_like_python.py", b"\x00\x01binary garbage\x00")
        _write(tmp_path / "real.py", b"print('real')")

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"real.py"}

    def test_empty_file_is_included(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "empty.py", b"")

        result = discovery.discover(tmp_path)

        assert _relative_paths(result) == {"empty.py"}

    def test_empty_repository_returns_empty_list(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        assert discovery.discover(tmp_path) == []


class TestSourceFileFields:
    def test_source_file_fields_are_correct(self, discovery: FileDiscovery, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "app.py", b"print('hi')")

        [source_file] = discovery.discover(tmp_path)

        assert source_file.absolute_path == tmp_path / "src" / "app.py"
        assert source_file.relative_path == Path("src") / "app.py"
        assert source_file.language == "python"
        assert source_file.extension == ".py"
        assert source_file.size_bytes == len(b"print('hi')")
