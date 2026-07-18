"""Tests for ingestion.git_client.

All GitPython interaction is mocked — these tests never touch the
network or a real git repository.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from git.exc import GitCommandError, InvalidGitRepositoryError

from core.exceptions import RepositoryCloneError
from ingestion.git_client import GitClient


@pytest.fixture
def client() -> GitClient:
    return GitClient()


class TestIsExistingRepository:
    def test_returns_false_when_path_missing(self, client: GitClient, tmp_path: Path) -> None:
        assert client.is_existing_repository(tmp_path / "nonexistent") is False

    def test_returns_true_for_valid_repository(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("ingestion.git_client.git.Repo", return_value=MagicMock())
        assert client.is_existing_repository(tmp_path) is True

    def test_returns_false_for_invalid_repository(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch(
            "ingestion.git_client.git.Repo",
            side_effect=InvalidGitRepositoryError("not a repo"),
        )
        assert client.is_existing_repository(tmp_path) is False


class TestClone:
    def test_success_delegates_to_gitpython(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mock_clone_from = mocker.patch("ingestion.git_client.git.Repo.clone_from")
        destination = tmp_path / "owner_repo"

        client.clone("https://github.com/owner/repo.git", destination)

        mock_clone_from.assert_called_once_with("https://github.com/owner/repo.git", destination)

    def test_git_command_error_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch(
            "ingestion.git_client.git.Repo.clone_from",
            side_effect=GitCommandError("git clone", 128, stderr=b"could not resolve host"),
        )
        with pytest.raises(RepositoryCloneError):
            client.clone("https://github.com/owner/repo.git", tmp_path / "owner_repo")

    def test_unexpected_error_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("ingestion.git_client.git.Repo.clone_from", side_effect=OSError("disk full"))
        with pytest.raises(RepositoryCloneError):
            client.clone("https://github.com/owner/repo.git", tmp_path / "owner_repo")


class TestFetchAndCheckUpdates:
    def _make_repo(self, *, local_sha: str, remote_sha: str, branch: str = "main") -> MagicMock:
        repo = MagicMock()
        repo.head.commit.hexsha = local_sha
        repo.active_branch.name = branch
        repo.remotes.origin.refs = {branch: MagicMock(commit=MagicMock(hexsha=remote_sha))}
        return repo

    def test_returns_false_when_up_to_date(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = self._make_repo(local_sha="abc123", remote_sha="abc123")
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        assert client.fetch_and_check_updates(tmp_path) is False
        repo.remotes.origin.fetch.assert_called_once()

    def test_returns_true_when_remote_has_new_commits(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = self._make_repo(local_sha="abc123", remote_sha="def456")
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        assert client.fetch_and_check_updates(tmp_path) is True

    def test_fetch_failure_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = MagicMock()
        repo.remotes.origin.fetch.side_effect = GitCommandError(
            "git fetch", 128, stderr=b"network unreachable"
        )
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        with pytest.raises(RepositoryCloneError):
            client.fetch_and_check_updates(tmp_path)

    def test_ref_comparison_failure_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = MagicMock()
        repo.active_branch.name = "main"
        repo.remotes.origin.refs = {}  # missing branch ref -> KeyError internally
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        with pytest.raises(RepositoryCloneError):
            client.fetch_and_check_updates(tmp_path)


class TestPull:
    def test_success_delegates_to_gitpython(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = MagicMock()
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        client.pull(tmp_path)

        repo.remotes.origin.pull.assert_called_once()

    def test_failure_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        repo = MagicMock()
        repo.remotes.origin.pull.side_effect = GitCommandError("git pull", 1, stderr=b"conflict")
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        with pytest.raises(RepositoryCloneError):
            client.pull(tmp_path)


class TestMetadataReads:
    def test_get_current_commit_hash(self, client: GitClient, tmp_path: Path, mocker: MagicMock) -> None:
        repo = MagicMock()
        repo.head.commit.hexsha = "deadbeef"
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        assert client.get_current_commit_hash(tmp_path) == "deadbeef"

    def test_get_default_branch(self, client: GitClient, tmp_path: Path, mocker: MagicMock) -> None:
        repo = MagicMock()
        repo.active_branch.name = "main"
        mocker.patch("ingestion.git_client.git.Repo", return_value=repo)

        assert client.get_default_branch(tmp_path) == "main"

    def test_open_invalid_repository_raises_repository_clone_error(
        self, client: GitClient, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mocker.patch(
            "ingestion.git_client.git.Repo",
            side_effect=InvalidGitRepositoryError("not a repo"),
        )
        with pytest.raises(RepositoryCloneError):
            client.get_current_commit_hash(tmp_path)
