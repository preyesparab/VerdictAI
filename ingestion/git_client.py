"""Thin wrapper around GitPython — the only module in RepoMind permitted to import it.

Every other module must go through `ingestion.repository_manager.RepositoryManager`
instead of talking to GitPython (or the `git` CLI) directly. Concentrating
the dependency here means every GitPython exception is translated into a
`RepositoryCloneError` at the boundary, so no caller ever needs to know
GitPython's exception types exist.
"""

from __future__ import annotations

from pathlib import Path

import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from core.exceptions import RepositoryCloneError
from core.logging import get_logger

logger = get_logger(__name__)


class GitClient:
    """Wraps the GitPython operations RepositoryManager needs.

    Every method raises `RepositoryCloneError` (never a GitPython
    exception) on failure.
    """

    def is_existing_repository(self, path: Path) -> bool:
        """Check whether `path` already holds a valid local git repository.

        Args:
            path: Candidate local repository directory.

        Returns:
            True if `path` exists and is a valid git working tree.
        """
        if not path.exists():
            return False
        try:
            git.Repo(path)
        except (InvalidGitRepositoryError, NoSuchPathError):
            return False
        return True

    def clone(self, clone_url: str, destination: Path) -> None:
        """Clone `clone_url` into `destination`.

        Args:
            clone_url: The repository's clone URL.
            destination: Local directory to clone into. Must not already
                exist as a non-empty directory.

        Raises:
            RepositoryCloneError: If the clone fails for any reason
                (network failure, invalid URL, authentication, disk error).
        """
        try:
            git.Repo.clone_from(clone_url, destination)
        except GitCommandError as exc:
            raise RepositoryCloneError(f"Failed to clone {clone_url}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - translate any unexpected GitPython failure
            raise RepositoryCloneError(f"Unexpected error cloning {clone_url}: {exc}") from exc

    def fetch_and_check_updates(self, path: Path) -> bool:
        """Fetch from origin and report whether the remote has new commits.

        Args:
            path: Local repository directory.

        Returns:
            True if the remote-tracking branch has commits not present
            locally (i.e., a pull is needed); False if already up to date.

        Raises:
            RepositoryCloneError: If fetching or comparing refs fails
                (e.g., network failure, detached HEAD with no upstream).
        """
        repo = self._open(path)
        try:
            origin = repo.remotes.origin
            origin.fetch()
        except GitCommandError as exc:
            raise RepositoryCloneError(f"Failed to fetch updates for {path}: {exc}") from exc

        try:
            local_commit = repo.head.commit.hexsha
            branch_name = repo.active_branch.name
            remote_commit = origin.refs[branch_name].commit.hexsha
        except Exception as exc:  # noqa: BLE001
            raise RepositoryCloneError(
                f"Failed to compare local/remote state for {path}: {exc}"
            ) from exc

        return local_commit != remote_commit

    def pull(self, path: Path) -> None:
        """Pull the latest changes for the currently checked-out branch.

        Args:
            path: Local repository directory.

        Raises:
            RepositoryCloneError: If the pull fails (merge conflicts,
                network failure, etc.).
        """
        repo = self._open(path)
        try:
            repo.remotes.origin.pull()
        except GitCommandError as exc:
            raise RepositoryCloneError(f"Failed to pull updates for {path}: {exc}") from exc

    def get_current_commit_hash(self, path: Path) -> str:
        """Return the full SHA of the currently checked-out commit.

        Args:
            path: Local repository directory.

        Returns:
            The 40-character commit SHA of HEAD.

        Raises:
            RepositoryCloneError: If the commit hash cannot be read.
        """
        repo = self._open(path)
        try:
            return repo.head.commit.hexsha
        except Exception as exc:  # noqa: BLE001
            raise RepositoryCloneError(f"Failed to read commit hash for {path}: {exc}") from exc

    def get_default_branch(self, path: Path) -> str:
        """Return the name of the currently checked-out branch.

        Args:
            path: Local repository directory.

        Returns:
            The active branch name (the branch GitHub's default HEAD
            pointed to at clone time).

        Raises:
            RepositoryCloneError: If the branch name cannot be determined
                (e.g., detached HEAD).
        """
        repo = self._open(path)
        try:
            return repo.active_branch.name
        except Exception as exc:  # noqa: BLE001
            raise RepositoryCloneError(f"Failed to determine default branch for {path}: {exc}") from exc

    def _open(self, path: Path) -> git.Repo:
        """Open an existing local repository, translating failures.

        Args:
            path: Local repository directory.

        Returns:
            The opened `git.Repo`.

        Raises:
            RepositoryCloneError: If `path` is not a valid git repository.
        """
        try:
            return git.Repo(path)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise RepositoryCloneError(f"{path} is not a valid local git repository: {exc}") from exc
