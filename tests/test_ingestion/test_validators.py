"""Tests for ingestion.validators."""

from __future__ import annotations

import pytest

from core.exceptions import RepositoryCloneError
from ingestion.validators import ParsedGitHubUrl, validate_github_url


@pytest.mark.parametrize(
    ("url", "expected_owner", "expected_repo"),
    [
        ("https://github.com/tiangolo/fastapi", "tiangolo", "fastapi"),
        ("https://github.com/tiangolo/fastapi.git", "tiangolo", "fastapi"),
        ("https://github.com/tiangolo/fastapi/", "tiangolo", "fastapi"),
        ("https://github.com/tiangolo/fastapi.git/", "tiangolo", "fastapi"),
        ("https://github.com/octocat/Hello-World", "octocat", "Hello-World"),
        ("https://github.com/psf/black.js", "psf", "black.js"),
    ],
)
def test_valid_urls_parse_correctly(url: str, expected_owner: str, expected_repo: str) -> None:
    parsed = validate_github_url(url)
    assert isinstance(parsed, ParsedGitHubUrl)
    assert parsed.owner == expected_owner
    assert parsed.repo == expected_repo
    assert parsed.clone_url == f"https://github.com/{expected_owner}/{expected_repo}.git"


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/tiangolo/fastapi",
        "https://bitbucket.org/tiangolo/fastapi",
        "https://github.com/tiangolo",
        "https://github.com/tiangolo/",
        "not a url",
        "",
        "git@github.com:tiangolo/fastapi.git",
        "ftp://github.com/tiangolo/fastapi",
        "https://github.com/tiangolo/fastapi/extra/path",
        "https://github.com//fastapi",
    ],
)
def test_invalid_urls_raise_repository_clone_error(url: str) -> None:
    with pytest.raises(RepositoryCloneError):
        validate_github_url(url)


def test_none_url_raises_repository_clone_error() -> None:
    with pytest.raises(RepositoryCloneError):
        validate_github_url(None)  # type: ignore[arg-type]
