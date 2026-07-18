"""SourceFile dataclass describing a single discovered source file.

Produced by `ingestion.file_discovery.FileDiscovery` and consumed by the
Tree-sitter AST parser (a future phase), which needs the absolute path to
read the file and the detected language to select the correct grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    """Immutable description of one source file discovered in a repository.

    Attributes:
        absolute_path: Absolute filesystem path to the file.
        relative_path: Path relative to the repository root — a stable
            identifier independent of where the repository happens to be
            cloned locally, suitable for storage/display.
        language: Detected language, per
            `core.constants.SUPPORTED_FILE_EXTENSIONS`.
        extension: The file's extension, including the leading dot.
        size_bytes: File size in bytes.
    """

    absolute_path: Path
    relative_path: Path
    language: str
    extension: str
    size_bytes: int
