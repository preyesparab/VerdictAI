"""Tests for ingestion.chunker.SemanticChunker."""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

import pytest

from core.exceptions import ParsingError
from ingestion.chunker import ChunkingResult, SemanticChunker
from ingestion.deterministic_ids import compute_file_id
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk


def _write_source_file(tmp_path: Path, name: str, content: str, language: str = "python") -> SourceFile:
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


def _numbered_lines_file(tmp_path: Path, name: str, line_count: int) -> SourceFile:
    content = "".join(f"line{i}\n" for i in range(1, line_count + 1))
    return _write_source_file(tmp_path, name, content)


def _make_ast_chunk(source_file: SourceFile, start_line: int, end_line: int, name: str = "fn") -> CodeChunk:
    file_id = compute_file_id(source_file.relative_path)
    return CodeChunk(
        chunk_id=uuid.uuid4(),
        file_id=file_id,
        file_path=source_file.relative_path.as_posix(),
        language=source_file.language,
        chunk_type=ChunkType.FUNCTION,
        function_name=name,
        class_name=None,
        parent_class=None,
        start_line=start_line,
        end_line=end_line,
        raw_code=f"def {name}(): ...",
    )


@pytest.fixture
def chunker() -> SemanticChunker:
    return SemanticChunker()


class TestConstructorValidation:
    def test_rejects_non_positive_window_size(self) -> None:
        with pytest.raises(ValueError):
            SemanticChunker(window_size=0)

    def test_rejects_invalid_overlap_ratio(self) -> None:
        with pytest.raises(ValueError):
            SemanticChunker(overlap_ratio=1.0)
        with pytest.raises(ValueError):
            SemanticChunker(overlap_ratio=-0.1)

    def test_rejects_non_positive_parent_context_lines(self) -> None:
        with pytest.raises(ValueError):
            SemanticChunker(parent_context_lines=0)


class TestFileReadErrors:
    def test_missing_file_raises_parsing_error(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = SourceFile(
            absolute_path=tmp_path / "missing.py",
            relative_path=Path("missing.py"),
            language="python",
            extension=".py",
            size_bytes=0,
        )
        with pytest.raises(ParsingError):
            chunker.build_chunks(source_file, [])


class TestEmptyInputs:
    def test_empty_file_produces_no_chunks(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _write_source_file(tmp_path, "empty.py", "")
        result = chunker.build_chunks(source_file, [])
        assert result == ChunkingResult(ast_chunks=[], sliding_chunks=[], parent_chunks=[])

    def test_no_ast_chunks_still_produces_sliding_chunks(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "config.py", 30)
        result = chunker.build_chunks(source_file, [])
        assert result.ast_chunks == []
        assert result.parent_chunks == []
        assert len(result.sliding_chunks) == 1
        assert result.sliding_chunks[0].start_line == 1
        assert result.sliding_chunks[0].end_line == 30


class TestSlidingWindows:
    def test_file_shorter_than_window_produces_single_chunk(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "short.py", 10)
        result = chunker.build_chunks(source_file, [])
        assert len(result.sliding_chunks) == 1
        chunk = result.sliding_chunks[0]
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.chunk_type == ChunkType.SLIDING
        assert chunk.function_name is None
        assert chunk.class_name is None
        assert chunk.parent_class is None
        assert chunk.parent_chunk_id is None
        assert chunk.raw_code == "".join(f"line{i}\n" for i in range(1, 11))

    def test_file_exactly_window_size_produces_single_chunk(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "exact.py", 50)
        result = chunker.build_chunks(source_file, [])
        assert len(result.sliding_chunks) == 1
        assert result.sliding_chunks[0].start_line == 1
        assert result.sliding_chunks[0].end_line == 50

    def test_multiple_windows_with_correct_overlap(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "long.py", 120)
        result = chunker.build_chunks(source_file, [])

        ranges = [(c.start_line, c.end_line) for c in result.sliding_chunks]
        assert ranges == [(1, 50), (41, 90), (81, 120)]

        # 20% of a 50-line window is a 10-line overlap between consecutive windows.
        assert ranges[0][1] - ranges[1][0] + 1 == 10
        assert ranges[1][1] - ranges[2][0] + 1 == 10

    def test_last_window_reaches_exact_end_of_file(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "long.py", 137)
        result = chunker.build_chunks(source_file, [])
        assert result.sliding_chunks[-1].end_line == 137

    def test_no_duplicate_sliding_ranges(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "long.py", 253)
        result = chunker.build_chunks(source_file, [])
        ranges = [(c.start_line, c.end_line) for c in result.sliding_chunks]
        assert len(ranges) == len(set(ranges))

    def test_sliding_chunk_ids_are_deterministic(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "long.py", 120)
        first = chunker.build_chunks(source_file, [])
        second = chunker.build_chunks(source_file, [])
        assert [c.chunk_id for c in first.sliding_chunks] == [c.chunk_id for c in second.sliding_chunks]


class TestLineEndings:
    def test_crlf_line_endings_are_preserved_and_counted_correctly(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        # Regression test: chunker previously read files in text mode, which
        # silently translates \r\n -> \n, making sliding/parent raw_code
        # inconsistent with ast_parser's byte-exact raw_code on CRLF files.
        content = "".join(f"line{i}\r\n" for i in range(1, 11))
        source_file = _write_source_file(tmp_path, "crlf.py", content)

        result = chunker.build_chunks(source_file, [])

        assert len(result.sliding_chunks) == 1
        chunk = result.sliding_chunks[0]
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.raw_code == content
        assert "\r\n" in chunk.raw_code

    def test_parent_chunk_contains_crlf_ast_chunk_raw_code(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        content = "".join(f"line{i}\r\n" for i in range(1, 201))
        source_file = _write_source_file(tmp_path, "crlf_mid.py", content)
        ast_chunk = _make_ast_chunk(source_file, start_line=100, end_line=102)
        # Simulate ast_parser's byte-exact (CRLF-preserving) raw_code for this range.
        lines = content.splitlines(keepends=True)
        ast_chunk = dataclasses.replace(ast_chunk, raw_code="".join(lines[99:102]))

        result = chunker.build_chunks(source_file, [ast_chunk])

        parent = result.parent_chunks[0]
        assert result.ast_chunks[0].raw_code in parent.raw_code


class TestParentChunks:
    def test_parent_contains_child_in_middle_of_file(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "mid.py", 200)
        ast_chunk = _make_ast_chunk(source_file, start_line=100, end_line=102)

        result = chunker.build_chunks(source_file, [ast_chunk])

        assert len(result.parent_chunks) == 1
        parent = result.parent_chunks[0]
        assert parent.chunk_type == ChunkType.PARENT
        assert parent.start_line <= 100
        assert parent.end_line >= 102
        assert parent.end_line - parent.start_line + 1 == 50
        assert result.ast_chunks[0].parent_chunk_id == parent.chunk_id

    def test_parent_near_start_of_file_clamps_to_line_one(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "start.py", 200)
        ast_chunk = _make_ast_chunk(source_file, start_line=1, end_line=3)

        result = chunker.build_chunks(source_file, [ast_chunk])

        parent = result.parent_chunks[0]
        assert parent.start_line == 1
        assert parent.end_line - parent.start_line + 1 == 50
        assert parent.start_line <= 1
        assert parent.end_line >= 3

    def test_parent_near_end_of_file_clamps_to_last_line(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "end.py", 200)
        ast_chunk = _make_ast_chunk(source_file, start_line=198, end_line=200)

        result = chunker.build_chunks(source_file, [ast_chunk])

        parent = result.parent_chunks[0]
        assert parent.end_line == 200
        assert parent.end_line - parent.start_line + 1 == 50
        assert parent.start_line <= 198
        assert parent.end_line >= 200

    def test_parent_for_small_file_covers_whole_file(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "tiny.py", 5)
        ast_chunk = _make_ast_chunk(source_file, start_line=2, end_line=3)

        result = chunker.build_chunks(source_file, [ast_chunk])

        parent = result.parent_chunks[0]
        assert parent.start_line == 1
        assert parent.end_line == 5

    def test_ast_chunk_larger_than_target_gets_exact_parent(
        self, chunker: SemanticChunker, tmp_path: Path
    ) -> None:
        source_file = _numbered_lines_file(tmp_path, "big_fn.py", 200)
        ast_chunk = _make_ast_chunk(source_file, start_line=10, end_line=89)  # 80 lines > 50

        result = chunker.build_chunks(source_file, [ast_chunk])

        parent = result.parent_chunks[0]
        assert (parent.start_line, parent.end_line) == (10, 89)

    def test_duplicate_parent_ranges_are_deduplicated(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "dup.py", 200)
        chunk_a = _make_ast_chunk(source_file, start_line=100, end_line=102, name="a")
        chunk_b = _make_ast_chunk(source_file, start_line=100, end_line=102, name="b")

        result = chunker.build_chunks(source_file, [chunk_a, chunk_b])

        assert len(result.parent_chunks) == 1
        assert result.ast_chunks[0].parent_chunk_id == result.ast_chunks[1].parent_chunk_id

    def test_ast_chunks_order_and_count_preserved(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "order.py", 200)
        chunk_a = _make_ast_chunk(source_file, start_line=10, end_line=12, name="a")
        chunk_b = _make_ast_chunk(source_file, start_line=50, end_line=55, name="b")

        result = chunker.build_chunks(source_file, [chunk_a, chunk_b])

        assert [c.function_name for c in result.ast_chunks] == ["a", "b"]

    def test_parent_chunk_ids_are_deterministic(self, chunker: SemanticChunker, tmp_path: Path) -> None:
        source_file = _numbered_lines_file(tmp_path, "stable.py", 200)
        ast_chunk = _make_ast_chunk(source_file, start_line=100, end_line=102)

        first = chunker.build_chunks(source_file, [ast_chunk])
        second = chunker.build_chunks(source_file, [ast_chunk])

        assert first.parent_chunks[0].chunk_id == second.parent_chunks[0].chunk_id
        assert first.ast_chunks[0].parent_chunk_id == second.ast_chunks[0].parent_chunk_id
