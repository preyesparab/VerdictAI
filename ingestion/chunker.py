"""Semantic chunking: sliding-window chunks and parent (small-to-big) chunks.

`SemanticChunker` enriches the AST chunks produced by
`ingestion.ast_parser.TreeSitterParser` (Phase 4) with two complementary
chunk families:

- **Sliding-window chunks** cover every file uniformly (not just the parts
  Tree-sitter recognized as a function/class), so nothing is unreachable
  by retrieval regardless of how a file is structured.
- **Parent chunks** give every AST chunk a wider window of surrounding
  context (small-to-big retrieval): match precisely at AST granularity,
  then expand to the parent chunk before handing context to the LLM.

Sliding and parent chunks reuse `models.schemas.CodeChunk` — they are
structurally identical (a language, a line range, and raw source text),
distinguished only by `chunk_type`. AST-specific fields (`function_name`,
`class_name`, `parent_class`) are simply None for these chunk types.
"""

from __future__ import annotations

import dataclasses

from core.exceptions import ParsingError
from core.logging import get_logger
from ingestion.deterministic_ids import compute_chunk_id, compute_file_id
from ingestion.source_file import SourceFile
from models.schemas import ChunkType, CodeChunk

logger = get_logger(__name__)

SLIDING_WINDOW_SIZE: int = 50
SLIDING_WINDOW_OVERLAP_RATIO: float = 0.2
PARENT_CONTEXT_LINES: int = 50


@dataclasses.dataclass(frozen=True)
class ChunkingResult:
    """The three chunk families produced for one source file.

    Attributes:
        ast_chunks: The input AST chunks, each with `parent_chunk_id` set
            to the `chunk_id` of its corresponding entry in
            `parent_chunks` (unless no valid parent range could be
            computed, in which case it is left unchanged).
        sliding_chunks: Sliding-window chunks covering the entire file.
        parent_chunks: One parent chunk per distinct surrounding-context
            range needed by `ast_chunks` (deduplicated — multiple AST
            chunks whose computed windows coincide share one parent).
    """

    ast_chunks: list[CodeChunk]
    sliding_chunks: list[CodeChunk]
    parent_chunks: list[CodeChunk]


class SemanticChunker:
    """Builds sliding-window and parent chunks to complement AST chunks.

    Args:
        window_size: Target sliding-window size, in lines.
        overlap_ratio: Fraction of `window_size` consecutive sliding
            windows should overlap by.
        parent_context_lines: Target parent-chunk size, in lines.
    """

    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        overlap_ratio: float = SLIDING_WINDOW_OVERLAP_RATIO,
        parent_context_lines: int = PARENT_CONTEXT_LINES,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")
        if parent_context_lines <= 0:
            raise ValueError(f"parent_context_lines must be positive, got {parent_context_lines}")

        self._window_size = window_size
        self._overlap_lines = round(window_size * overlap_ratio)
        self._step = window_size - self._overlap_lines
        self._parent_context_lines = parent_context_lines

    def build_chunks(self, source_file: SourceFile, ast_chunks: list[CodeChunk]) -> ChunkingResult:
        """Build sliding-window and parent chunks for `source_file`.

        Args:
            source_file: The file to chunk (sliding chunks are generated
                for every file, regardless of `ast_chunks`).
            ast_chunks: The AST chunks previously extracted from this file
                by `TreeSitterParser`.

        Returns:
            A `ChunkingResult` with all three chunk families and their
            parent/child relationships established.

        Raises:
            ParsingError: If `source_file` cannot be read.
        """
        lines = self._read_lines(source_file)
        total_lines = len(lines)
        file_id = compute_file_id(source_file.relative_path)

        sliding_chunks, skipped_sliding = self._build_sliding_chunks(
            source_file, file_id, lines, total_lines
        )
        updated_ast_chunks, parent_chunks, skipped_parent = self._build_parent_chunks(
            source_file, file_id, lines, total_lines, ast_chunks
        )

        total_chunks = len(updated_ast_chunks) + len(sliding_chunks) + len(parent_chunks)
        logger.info(
            "Semantic chunking complete for %s: %d ast chunk(s), %d sliding chunk(s) created, "
            "%d parent chunk(s) created, %d total chunk(s), %d skipped",
            source_file.relative_path, len(updated_ast_chunks), len(sliding_chunks),
            len(parent_chunks), total_chunks, skipped_sliding + skipped_parent,
        )

        return ChunkingResult(
            ast_chunks=updated_ast_chunks,
            sliding_chunks=sliding_chunks,
            parent_chunks=parent_chunks,
        )

    def _read_lines(self, source_file: SourceFile) -> list[str]:
        """Read `source_file` as text, split into lines with terminators preserved.

        Args:
            source_file: The file to read.

        Returns:
            Each line of the file, including its original line terminator
            (so windows can be reassembled byte-for-byte).

        Raises:
            ParsingError: If the file cannot be read.
        """
        try:
            source_bytes = source_file.absolute_path.read_bytes()
        except OSError as exc:
            raise ParsingError(f"Failed to read {source_file.absolute_path}: {exc}") from exc
        # Decode raw bytes directly rather than using text-mode reading, which
        # performs universal-newline translation (\r\n -> \n) and would make
        # these chunks inconsistent with ast_parser's byte-exact raw_code on
        # any file using CRLF line endings.
        text = source_bytes.decode("utf-8", errors="replace")
        return text.splitlines(keepends=True)

    def _is_valid_range(self, start_line: int, end_line: int, total_lines: int) -> bool:
        """Check that a 1-indexed, inclusive line range is well-formed.

        Args:
            start_line: Proposed first line (inclusive).
            end_line: Proposed last line (inclusive).
            total_lines: Total number of lines in the file.

        Returns:
            True if ``1 <= start_line <= end_line <= total_lines``.
        """
        return 1 <= start_line <= end_line <= total_lines

    def _make_window_chunk(
        self,
        source_file: SourceFile,
        file_id: str,
        lines: list[str],
        start_line: int,
        end_line: int,
        chunk_type: ChunkType,
        identity_tag: str,
    ) -> CodeChunk:
        """Build a non-AST (sliding or parent) `CodeChunk` for a line range.

        Args:
            source_file: The file the chunk belongs to.
            file_id: The file's deterministic identifier.
            lines: The file's lines, with terminators preserved.
            start_line: First line of the window (1-indexed, inclusive).
            end_line: Last line of the window (1-indexed, inclusive).
            chunk_type: `ChunkType.SLIDING` or `ChunkType.PARENT`.
            identity_tag: Short tag (``"sliding"`` or ``"parent"``) mixed
                into the deterministic chunk_id so the two families never
                collide even if they share a line range.

        Returns:
            The assembled `CodeChunk`.
        """
        raw_code = "".join(lines[start_line - 1 : end_line])
        chunk_id = compute_chunk_id(f"{file_id}:{identity_tag}:{start_line}:{end_line}")
        return CodeChunk(
            chunk_id=chunk_id,
            file_id=file_id,
            file_path=source_file.relative_path.as_posix(),
            language=source_file.language,
            chunk_type=chunk_type,
            function_name=None,
            class_name=None,
            parent_class=None,
            start_line=start_line,
            end_line=end_line,
            raw_code=raw_code,
        )

    def _build_sliding_chunks(
        self, source_file: SourceFile, file_id: str, lines: list[str], total_lines: int,
    ) -> tuple[list[CodeChunk], int]:
        """Generate overlapping sliding-window chunks covering the whole file.

        Args:
            source_file: The file being chunked.
            file_id: The file's deterministic identifier.
            lines: The file's lines, with terminators preserved.
            total_lines: Total number of lines in the file.

        Returns:
            A tuple of (chunks, skipped_count).
        """
        chunks: list[CodeChunk] = []
        seen_ranges: set[tuple[int, int]] = set()
        skipped = 0

        start_idx = 0
        while start_idx < total_lines:
            end_idx = min(start_idx + self._window_size, total_lines)
            start_line = start_idx + 1
            end_line = end_idx

            range_key = (start_line, end_line)
            if self._is_valid_range(start_line, end_line, total_lines) and range_key not in seen_ranges:
                seen_ranges.add(range_key)
                chunks.append(
                    self._make_window_chunk(
                        source_file, file_id, lines, start_line, end_line, ChunkType.SLIDING, "sliding",
                    )
                )
            else:
                skipped += 1

            if end_idx >= total_lines:
                break
            start_idx += self._step

        return chunks, skipped

    def _compute_parent_range(
        self, start_line: int, end_line: int, total_lines: int,
    ) -> tuple[int, int]:
        """Compute a ~`parent_context_lines`-line window fully containing `[start_line, end_line]`.

        Extra context is split evenly before and after the child range,
        then shifted to whichever side has room if the file boundary
        would otherwise clip it — so the window stays close to the target
        size even near the start or end of a file. If the child range
        itself is already at least as large as the target, the parent
        range exactly matches the child (a parent can never be smaller
        than its child).

        Args:
            start_line: The AST chunk's first line (1-indexed, inclusive).
            end_line: The AST chunk's last line (1-indexed, inclusive).
            total_lines: Total number of lines in the file.

        Returns:
            The (parent_start_line, parent_end_line) window, guaranteed to
            satisfy ``parent_start_line <= start_line`` and
            ``parent_end_line >= end_line``.
        """
        chunk_length = end_line - start_line + 1
        extra = max(0, self._parent_context_lines - chunk_length)
        before = extra // 2
        after = extra - before

        parent_start = start_line - before
        parent_end = end_line + after

        if parent_start < 1:
            deficit = 1 - parent_start
            parent_start = 1
            parent_end = min(total_lines, parent_end + deficit)

        if parent_end > total_lines:
            surplus = parent_end - total_lines
            parent_end = total_lines
            parent_start = max(1, parent_start - surplus)

        return parent_start, parent_end

    def _build_parent_chunks(
        self,
        source_file: SourceFile,
        file_id: str,
        lines: list[str],
        total_lines: int,
        ast_chunks: list[CodeChunk],
    ) -> tuple[list[CodeChunk], list[CodeChunk], int]:
        """Build a deduplicated parent chunk for every AST chunk and link them.

        Args:
            source_file: The file being chunked.
            file_id: The file's deterministic identifier.
            lines: The file's lines, with terminators preserved.
            total_lines: Total number of lines in the file.
            ast_chunks: The AST chunks to attach parents to.

        Returns:
            A tuple of (updated_ast_chunks, parent_chunks, skipped_count).
            `updated_ast_chunks` has the same order and length as
            `ast_chunks`, with `parent_chunk_id` populated where a valid
            parent range was found.
        """
        parent_by_range: dict[tuple[int, int], CodeChunk] = {}
        updated_ast_chunks: list[CodeChunk] = []
        skipped = 0

        for ast_chunk in ast_chunks:
            parent_start, parent_end = self._compute_parent_range(
                ast_chunk.start_line, ast_chunk.end_line, total_lines,
            )

            if not self._is_valid_range(parent_start, parent_end, total_lines):
                skipped += 1
                updated_ast_chunks.append(ast_chunk)
                continue

            range_key = (parent_start, parent_end)
            parent_chunk = parent_by_range.get(range_key)
            if parent_chunk is None:
                parent_chunk = self._make_window_chunk(
                    source_file, file_id, lines, parent_start, parent_end, ChunkType.PARENT, "parent",
                )
                parent_by_range[range_key] = parent_chunk

            updated_ast_chunks.append(dataclasses.replace(ast_chunk, parent_chunk_id=parent_chunk.chunk_id))

        return updated_ast_chunks, list(parent_by_range.values()), skipped
