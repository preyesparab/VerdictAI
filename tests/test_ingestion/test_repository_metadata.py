"""Tests for ingestion.repository_metadata.RepositoryMetadata."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingestion.repository_metadata import RepositoryMetadata


def _make_metadata(**overrides: object) -> RepositoryMetadata:
    defaults: dict[str, object] = {
        "name": "fastapi",
        "owner": "tiangolo",
        "clone_url": "https://github.com/tiangolo/fastapi.git",
        "default_branch": "master",
        "local_path": Path("/data/repositories/tiangolo_fastapi"),
        "last_updated": datetime.now(timezone.utc),
        "commit_hash": "a" * 40,
    }
    defaults.update(overrides)
    return RepositoryMetadata(**defaults)  # type: ignore[arg-type]


def test_fields_round_trip() -> None:
    metadata = _make_metadata()
    assert metadata.name == "fastapi"
    assert metadata.owner == "tiangolo"
    assert metadata.commit_hash == "a" * 40


def test_is_immutable() -> None:
    metadata = _make_metadata()
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.name = "other"  # type: ignore[misc]


def test_local_path_is_path_instance() -> None:
    metadata = _make_metadata()
    assert isinstance(metadata.local_path, Path)
