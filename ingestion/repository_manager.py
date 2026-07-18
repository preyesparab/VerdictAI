"""RepositoryManager: the sole public interface for repository acquisition and caching.

Downstream layers (AST parsing, chunking, and everything built on top of
them) must obtain a local repository path exclusively through
`RepositoryManager.get_repository` — never by importing `GitClient` or
GitPython directly. This keeps the clone/cache/update policy in one
place and makes every other module trivially testable against a fake
local path instead of a real git repository.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.logging import get_logger
from ingestion.git_client import GitClient
from ingestion.repository_metadata import RepositoryMetadata
from ingestion.validators import ParsedGitHubUrl, validate_github_url

logger = get_logger(__name__)


class RepositoryManager:
    """Acquires and caches GitHub repositories on the local filesystem.

    Repositories are stored under a deterministic ``<owner>_<repo>``
    folder inside `settings.REPOSITORIES_DIR`. An already-cloned
    repository is fetched and compared against its remote before
    deciding whether a pull is needed — it is never deleted and
    re-cloned.
    """

    def __init__(
        self,
        repositories_dir: Path | None = None,
        git_client: GitClient | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            repositories_dir: Root directory under which each repository
                gets its own cache folder. Defaults to
                `settings.REPOSITORIES_DIR`.
            git_client: The git implementation to use. Defaults to a new
                `GitClient`. Overridable for testing.
        """
        self._repositories_dir = repositories_dir or settings.REPOSITORIES_DIR
        self._repositories_dir.mkdir(parents=True, exist_ok=True)
        self._git_client = git_client or GitClient()

    def get_repository(self, url: str) -> RepositoryMetadata:
        """Ensure `url` is cloned locally and up to date, returning its metadata.

        Args:
            url: A GitHub repository URL
                (``https://github.com/<owner>/<repo>[.git]``).

        Returns:
            Metadata describing the local repository, including the path
            downstream modules should read from.

        Raises:
            RepositoryCloneError: If the URL is invalid, or cloning/fetching
                /pulling fails.
        """
        parsed = validate_github_url(url)
        local_path = self._local_path_for(parsed.owner, parsed.repo)

        if self._git_client.is_existing_repository(local_path):
            logger.info(
                "Repository %s/%s already exists at %s; checking for updates",
                parsed.owner, parsed.repo, local_path,
            )
            self._update_repository(local_path, parsed)
        else:
            logger.info("Cloning %s/%s into %s", parsed.owner, parsed.repo, local_path)
            self._git_client.clone(parsed.clone_url, local_path)
            logger.info("Cloned %s/%s successfully", parsed.owner, parsed.repo)

        return self._build_metadata(parsed, local_path)

    def _local_path_for(self, owner: str, repo: str) -> Path:
        """Compute the deterministic local cache path for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            ``<repositories_dir>/<owner>_<repo>``.
        """
        return self._repositories_dir / f"{owner}_{repo}"

    def _update_repository(self, local_path: Path, parsed: ParsedGitHubUrl) -> None:
        """Fetch and pull `local_path` only if the remote has new commits.

        Args:
            local_path: Local repository directory.
            parsed: The validated URL components, used for logging.
        """
        has_updates = self._git_client.fetch_and_check_updates(local_path)
        if has_updates:
            logger.info("Updates found for %s/%s; pulling", parsed.owner, parsed.repo)
            self._git_client.pull(local_path)
            logger.info("Updated %s/%s", parsed.owner, parsed.repo)
        else:
            logger.info("%s/%s is already up to date", parsed.owner, parsed.repo)

    def _build_metadata(self, parsed: ParsedGitHubUrl, local_path: Path) -> RepositoryMetadata:
        """Assemble a `RepositoryMetadata` snapshot for the given local repository.

        Args:
            parsed: The validated URL components.
            local_path: Local repository directory.

        Returns:
            The assembled metadata.
        """
        return RepositoryMetadata(
            name=parsed.repo,
            owner=parsed.owner,
            clone_url=parsed.clone_url,
            default_branch=self._git_client.get_default_branch(local_path),
            local_path=local_path,
            last_updated=datetime.now(timezone.utc),
            commit_hash=self._git_client.get_current_commit_hash(local_path),
        )
