"""Recursive file discovery over a locally cloned repository.

Walks a repository directory, prunes ignored directories, and filters out
binary files, oversized files, and unsupported languages, returning the
resulting `SourceFile` objects. This is the bridge between
`RepositoryManager` (which produces a local repository path) and the
Tree-sitter AST parser (a future phase, which needs a concrete list of
parseable files) — see `ingestion/__init__.py` for the full pipeline
story.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.constants import SUPPORTED_FILE_EXTENSIONS
from core.exceptions import ParsingError
from core.logging import get_logger
from ingestion.source_file import SourceFile

logger = get_logger(__name__)

MAX_FILE_SIZE_BYTES: int = 2 * 1024 * 1024  # 2 MB

IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "target",
        "coverage",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
    }
)

# Bytes read from the start of a file to sniff for binary content — the
# same heuristic `git` itself uses (presence of a NUL byte).
_BINARY_SNIFF_SIZE: int = 8192


class FileDiscovery:
    """Discovers supported, parseable source files within a local repository."""

    def discover(self, repository_path: Path) -> list[SourceFile]:
        """Recursively discover supported source files under `repository_path`.

        Args:
            repository_path: Local filesystem path to a cloned repository,
                as returned by `RepositoryManager.get_repository().local_path`.

        Returns:
            One `SourceFile` per supported, non-binary, size-limited file
            found, in traversal order.

        Raises:
            ParsingError: If `repository_path` does not exist, is not a
                directory, or cannot be scanned (e.g., a permissions error).
        """
        if not repository_path.exists():
            raise ParsingError(f"Repository path does not exist: {repository_path}")
        if not repository_path.is_dir():
            raise ParsingError(f"Repository path is not a directory: {repository_path}")

        discovered: list[SourceFile] = []
        directories_scanned = 0
        files_ignored = 0
        directories_ignored = 0

        try:
            for current_dir, subdirs, filenames in os.walk(repository_path):
                directories_scanned += 1
                current_path = Path(current_dir)

                to_prune = [d for d in subdirs if d in IGNORED_DIRECTORIES]
                if to_prune:
                    directories_ignored += len(to_prune)
                    logger.debug("Ignoring directories %s in %s", to_prune, current_path)
                subdirs[:] = [d for d in subdirs if d not in IGNORED_DIRECTORIES]

                for filename in filenames:
                    source_file = self._evaluate_file(current_path / filename, repository_path)
                    if source_file is None:
                        files_ignored += 1
                        continue
                    logger.debug(
                        "Discovered source file: %s (%s)",
                        source_file.relative_path, source_file.language,
                    )
                    discovered.append(source_file)
        except OSError as exc:
            raise ParsingError(f"Failed to scan repository {repository_path}: {exc}") from exc

        logger.info(
            "File discovery complete for %s: %d file(s) discovered, "
            "%d director(ies) scanned, %d file(s) ignored, %d director(ies) ignored",
            repository_path, len(discovered), directories_scanned,
            files_ignored, directories_ignored,
        )
        return discovered

    def _evaluate_file(self, file_path: Path, repository_root: Path) -> SourceFile | None:
        """Evaluate one candidate file, returning a `SourceFile` if it qualifies.

        Args:
            file_path: Absolute path to the candidate file.
            repository_root: Root of the repository being scanned, used to
                compute the file's relative path.

        Returns:
            A `SourceFile` if the file has a supported extension, is
            within the size limit, and is not binary; otherwise None.
        """
        language = SUPPORTED_FILE_EXTENSIONS.get(file_path.suffix)
        if language is None:
            return None

        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            return None

        if size_bytes > MAX_FILE_SIZE_BYTES:
            logger.debug("Ignoring oversized file %s (%d bytes)", file_path, size_bytes)
            return None

        if self._is_binary(file_path):
            logger.debug("Ignoring binary file %s", file_path)
            return None

        return SourceFile(
            absolute_path=file_path,
            relative_path=file_path.relative_to(repository_root),
            language=language,
            extension=file_path.suffix,
            size_bytes=size_bytes,
        )

    def _is_binary(self, file_path: Path) -> bool:
        """Heuristically detect whether a file is binary.

        Reads the first chunk of the file and treats the presence of a
        NUL byte as a binary indicator.

        Args:
            file_path: Path to the file to inspect.

        Returns:
            True if the file appears to be binary or could not be read.
        """
        try:
            with file_path.open("rb") as fh:
                chunk = fh.read(_BINARY_SNIFF_SIZE)
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            return True
        return b"\0" in chunk
