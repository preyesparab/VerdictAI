"""GitHub repository URL validation.

Parses and validates that a URL points to a well-formed, public GitHub
repository before any filesystem or network operation is attempted.
Rejects non-GitHub hosts (GitLab, Bitbucket, SSH remotes, etc.) and
malformed URLs early, so `RepositoryManager` never passes a bad URL
through to `git clone`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.exceptions import RepositoryCloneError

# Matches https://github.com/<owner>/<repo>, with an optional trailing
# ".git" suffix and/or trailing slash. Owner and repo segments follow
# GitHub's allowed character set (alphanumerics, hyphens, dots, underscores).
_GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?/?$"
)


@dataclass(frozen=True)
class ParsedGitHubUrl:
    """A validated GitHub repository URL, broken into its components.

    Attributes:
        owner: The repository owner (user or organization).
        repo: The repository name, with any ".git" suffix stripped.
        clone_url: The normalized, canonical clone URL
            (``https://github.com/<owner>/<repo>.git``).
    """

    owner: str
    repo: str
    clone_url: str


def validate_github_url(url: str) -> ParsedGitHubUrl:
    """Validate that `url` is a well-formed, public GitHub repository URL.

    Accepts ``https://github.com/<owner>/<repo>`` and
    ``https://github.com/<owner>/<repo>.git`` (with an optional trailing
    slash). Rejects any other host, malformed URLs, and URLs missing a
    repository name.

    Args:
        url: The candidate repository URL.

    Returns:
        A `ParsedGitHubUrl` with the owner, repo name, and normalized
        clone URL.

    Raises:
        RepositoryCloneError: If `url` is empty, not a string, or does not
            match a supported GitHub repository URL.
    """
    if not isinstance(url, str) or not url.strip():
        raise RepositoryCloneError(f"Repository URL must be a non-empty string, got {url!r}")

    match = _GITHUB_URL_PATTERN.match(url.strip())
    if not match:
        raise RepositoryCloneError(
            f"Unsupported or malformed repository URL: {url!r}. "
            "Only URLs of the form https://github.com/<owner>/<repo>[.git] are supported."
        )

    owner = match.group("owner")
    repo = match.group("repo")

    return ParsedGitHubUrl(
        owner=owner,
        repo=repo,
        clone_url=f"https://github.com/{owner}/{repo}.git",
    )
