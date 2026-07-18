"""Deterministic UUID derivation shared across ingestion stages.

Both `ast_parser.TreeSitterParser` (Phase 4) and `chunker.SemanticChunker`
(Phase 5) need `file_id` and `chunk_id` values that are stable across
repeated runs over an unchanged file — that stability is what lets
re-indexing detect unchanged chunks instead of re-embedding an entire
repository. Keeping the namespace constants and hashing scheme in one
place guarantees every stage derives IDs the same way.
"""

from __future__ import annotations

import uuid
from pathlib import Path

# Fixed, arbitrary namespaces (generated once) used to derive deterministic
# UUIDs. Using fixed namespaces (rather than uuid.NAMESPACE_DNS, etc.) just
# avoids any accidental collision with UUIDs generated elsewhere.
_FILE_ID_NAMESPACE = uuid.UUID("2f6e1c0a-3f0a-4c8e-9a3d-2f8c1a6b7e10")
_CHUNK_ID_NAMESPACE = uuid.UUID("7b9e2d4c-1a5f-4e3b-8c6d-9f1a2b3c4d5e")


def compute_file_id(relative_path: Path) -> str:
    """Derive a deterministic file identifier from a repository-relative path.

    Args:
        relative_path: The file's path relative to the repository root.

    Returns:
        A UUID5 string, stable across repeated runs as long as the file's
        relative path is unchanged.
    """
    return str(uuid.uuid5(_FILE_ID_NAMESPACE, relative_path.as_posix()))


def compute_chunk_id(identity: str) -> uuid.UUID:
    """Derive a deterministic chunk UUID from a pre-formatted identity string.

    Args:
        identity: A string that uniquely identifies the chunk (e.g.,
            ``f"{file_id}:{name}:{start_line}"`` for an AST chunk, or
            ``f"{file_id}:sliding:{start_line}:{end_line}"`` for a
            sliding-window chunk).

    Returns:
        A UUID5 stable across repeated calls with the same identity string.
    """
    return uuid.uuid5(_CHUNK_ID_NAMESPACE, identity)
