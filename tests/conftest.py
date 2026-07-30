"""Shared pytest fixtures for tests that need an isolated Postgres schema.

Phase 33 replaced the one-SQLite-file-per-test pattern (`tmp_path /
"test.db"`) with one PostgreSQL schema per test, against a single shared
dev/test database (`settings.DATABASE_URL`). `pg_schema` hands out a unique
schema name per test; individual test files still own constructing and
tearing down their `DatabaseManager` (via `manager.drop_schema()`) so each
file's fixture keeps full control of when `initialize_database` runs.
"""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture
def pg_schema() -> str:
    """A unique PostgreSQL schema name, safe to pass to `DatabaseManager(schema=...)`."""
    return f"test_{uuid.uuid4().hex[:16]}"
