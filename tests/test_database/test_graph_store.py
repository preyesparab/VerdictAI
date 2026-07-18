"""Tests for database.graph_store.save_graph / load_graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from core.exceptions import DatabaseError
from database.graph_store import load_graph, save_graph


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("a", node_kind="chunk", function_name="foo")
    graph.add_node("b", node_kind="chunk", function_name="bar")
    graph.add_edge("a", "b", edge_type="function_call")
    return graph


def test_save_graph_writes_json_file(sample_graph: nx.DiGraph, tmp_path: Path) -> None:
    destination = tmp_path / "graph.json"
    result_path = save_graph(sample_graph, destination)

    assert result_path == destination
    assert destination.exists()
    assert "\"directed\": true" in destination.read_text(encoding="utf-8")


def test_save_graph_creates_parent_directories(sample_graph: nx.DiGraph, tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "dir" / "graph.json"
    save_graph(sample_graph, destination)
    assert destination.exists()


def test_load_graph_round_trip_preserves_structure(sample_graph: nx.DiGraph, tmp_path: Path) -> None:
    destination = tmp_path / "graph.json"
    save_graph(sample_graph, destination)

    loaded = load_graph(destination)

    assert loaded.is_directed()
    assert set(loaded.nodes) == set(sample_graph.nodes)
    assert set(loaded.edges) == set(sample_graph.edges)
    assert loaded.nodes["a"]["function_name"] == "foo"
    assert loaded["a"]["b"]["edge_type"] == "function_call"


def test_load_graph_missing_file_raises_database_error(tmp_path: Path) -> None:
    with pytest.raises(DatabaseError):
        load_graph(tmp_path / "does_not_exist.json")


def test_load_graph_invalid_json_raises_database_error(tmp_path: Path) -> None:
    destination = tmp_path / "graph.json"
    destination.write_text("not valid json", encoding="utf-8")
    with pytest.raises(DatabaseError):
        load_graph(destination)


def test_save_graph_default_path_uses_settings(sample_graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from config import settings

    default_path = tmp_path / "default_graph.json"
    monkeypatch.setattr(settings, "GRAPH_FILE_PATH", default_path)

    result_path = save_graph(sample_graph)

    assert result_path == default_path
    assert default_path.exists()
