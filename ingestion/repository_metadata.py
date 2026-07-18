"""Metadata describing a locally cloned repository.

`RepositoryMetadata` is the return value of `RepositoryManager.get_repository`
and the record that will later be persisted to SQLite (`database/sqlite_client.py`,
a future phase) so that repeated queries against the same repository can
look up its local path without re-cloning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RepositoryMetadata:
    """Immutable snapshot of a locally cloned repository's identity and state.

    Attributes:
        name: Repository name (e.g., ``"fastapi"``).
        owner: Repository owner or organization (e.g., ``"tiangolo"``).
        clone_url: The normalized HTTPS clone URL.
        default_branch: The branch currently checked out locally.
        local_path: Absolute filesystem path to the cloned repository.
        last_updated: UTC timestamp of when this metadata was produced
            (i.e., when the clone/fetch/pull check last ran).
        commit_hash: Full SHA of the currently checked-out commit.
    """

    name: str
    owner: str
    clone_url: str
    default_branch: str
    local_path: Path
    last_updated: datetime
    commit_hash: str
