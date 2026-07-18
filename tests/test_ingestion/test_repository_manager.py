"""Tests for ingestion.repository_manager.RepositoryManager.

GitClient is fully mocked — these tests verify RepositoryManager's
orchestration logic (validate -> check cache -> clone/update -> build
metadata), never real git behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.exceptions import RepositoryCloneError
from ingestion.repository_manager import RepositoryManager
from ingestion.repository_metadata import RepositoryMetadata

VALID_URL = "https://github.com/tiangolo/fastapi"


@pytest.fixture
def mock_git_client() -> MagicMock:
    client = MagicMock()
    client.get_current_commit_hash.return_value = "abc123"
    client.get_default_branch.return_value = "main"
    return client


@pytest.fixture
def manager(tmp_path: Path, mock_git_client: MagicMock) -> RepositoryManager:
    return RepositoryManager(repositories_dir=tmp_path, git_client=mock_git_client)


def test_deterministic_local_path(manager: RepositoryManager, tmp_path: Path) -> None:
    path = manager._local_path_for("tiangolo", "fastapi")
    assert path == tmp_path / "tiangolo_fastapi"


def test_clones_new_repository(
    manager: RepositoryManager, mock_git_client: MagicMock, tmp_path: Path
) -> None:
    mock_git_client.is_existing_repository.return_value = False

    metadata = manager.get_repository(VALID_URL)

    mock_git_client.clone.assert_called_once_with(
        "https://github.com/tiangolo/fastapi.git", tmp_path / "tiangolo_fastapi"
    )
    mock_git_client.fetch_and_check_updates.assert_not_called()
    mock_git_client.pull.assert_not_called()
    assert isinstance(metadata, RepositoryMetadata)
    assert metadata.owner == "tiangolo"
    assert metadata.name == "fastapi"
    assert metadata.commit_hash == "abc123"
    assert metadata.default_branch == "main"
    assert metadata.local_path == tmp_path / "tiangolo_fastapi"


def test_existing_repository_up_to_date_skips_pull(
    manager: RepositoryManager, mock_git_client: MagicMock
) -> None:
    mock_git_client.is_existing_repository.return_value = True
    mock_git_client.fetch_and_check_updates.return_value = False

    manager.get_repository(VALID_URL)

    mock_git_client.clone.assert_not_called()
    mock_git_client.fetch_and_check_updates.assert_called_once()
    mock_git_client.pull.assert_not_called()


def test_existing_repository_with_updates_pulls(
    manager: RepositoryManager, mock_git_client: MagicMock
) -> None:
    mock_git_client.is_existing_repository.return_value = True
    mock_git_client.fetch_and_check_updates.return_value = True

    manager.get_repository(VALID_URL)

    mock_git_client.clone.assert_not_called()
    mock_git_client.pull.assert_called_once()


def test_invalid_url_raises_without_touching_git_client(
    manager: RepositoryManager, mock_git_client: MagicMock
) -> None:
    with pytest.raises(RepositoryCloneError):
        manager.get_repository("https://gitlab.com/owner/repo")

    mock_git_client.is_existing_repository.assert_not_called()
    mock_git_client.clone.assert_not_called()


def test_clone_failure_propagates(manager: RepositoryManager, mock_git_client: MagicMock) -> None:
    mock_git_client.is_existing_repository.return_value = False
    mock_git_client.clone.side_effect = RepositoryCloneError("network unreachable")

    with pytest.raises(RepositoryCloneError):
        manager.get_repository(VALID_URL)


def test_fetch_failure_propagates(manager: RepositoryManager, mock_git_client: MagicMock) -> None:
    mock_git_client.is_existing_repository.return_value = True
    mock_git_client.fetch_and_check_updates.side_effect = RepositoryCloneError("network unreachable")

    with pytest.raises(RepositoryCloneError):
        manager.get_repository(VALID_URL)


def test_repositories_dir_is_created(tmp_path: Path, mock_git_client: MagicMock) -> None:
    target = tmp_path / "does" / "not" / "exist" / "yet"
    RepositoryManager(repositories_dir=target, git_client=mock_git_client)
    assert target.exists()
