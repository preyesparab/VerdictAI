"""Tests for database.graph_store.save_graph / load_graph."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pytest

from core.exceptions import DatabaseError
from database.graph_store import load_graph, save_graph
from database.sqlite_client import DatabaseManager
from ingestion.repository_metadata import RepositoryMetadata


def _repository_metadata(owner: str = "acme", name: str = "demo") -> RepositoryMetadata:
    return RepositoryMetadata(
        name=name,
        owner=owner,
        clone_url=f"https://github.com/{owner}/{name}.git",
        default_branch="main",
        local_path=Path(f"/tmp/{owner}_{name}"),
        last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        commit_hash="a" * 40,
    )


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("a", node_kind="chunk", function_name="foo")
    graph.add_node("b", node_kind="chunk", function_name="bar")
    graph.add_edge("a", "b", edge_type="function_call")
    return graph


@pytest.fixture
def db(pg_schema: str) -> Iterator[DatabaseManager]:
    manager = DatabaseManager(schema=pg_schema)
    manager.initialize_database()
    yield manager
    manager.drop_schema()


@pytest.fixture
def repository_id(db: DatabaseManager) -> str:
    # graph_snapshots.repository_id is a foreign key into repositories, so
    # a row must exist there first, same as every other table keyed by
    # repository_id.
    return db.store_repository(_repository_metadata())


def test_save_graph_writes_snapshot_row(sample_graph: nx.DiGraph, db: DatabaseManager, repository_id: str) -> None:
    save_graph(sample_graph, repository_id, db)

    loaded = load_graph(repository_id, db)
    assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert loaded.number_of_edges() == sample_graph.number_of_edges()


def test_load_graph_round_trip_preserves_structure(
    sample_graph: nx.DiGraph, db: DatabaseManager, repository_id: str
) -> None:
    save_graph(sample_graph, repository_id, db)

    loaded = load_graph(repository_id, db)

    assert loaded.is_directed()
    assert set(loaded.nodes) == set(sample_graph.nodes)
    assert set(loaded.edges) == set(sample_graph.edges)
    assert loaded.nodes["a"]["function_name"] == "foo"
    assert loaded["a"]["b"]["edge_type"] == "function_call"


def test_save_graph_overwrites_existing_snapshot(
    sample_graph: nx.DiGraph, db: DatabaseManager, repository_id: str
) -> None:
    save_graph(sample_graph, repository_id, db)

    replacement = nx.DiGraph()
    replacement.add_node("c", node_kind="chunk", function_name="baz")
    save_graph(replacement, repository_id, db)

    loaded = load_graph(repository_id, db)
    assert set(loaded.nodes) == {"c"}


def test_load_graph_missing_snapshot_raises_database_error(db: DatabaseManager, repository_id: str) -> None:
    with pytest.raises(DatabaseError):
        load_graph(repository_id, db)


def test_load_graph_unknown_repository_raises_database_error(db: DatabaseManager) -> None:
    with pytest.raises(DatabaseError):
        load_graph("never-indexed-repository-id", db)
