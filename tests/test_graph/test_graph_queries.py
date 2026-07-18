"""Tests for graph.graph_queries.

Uses a small hand-built `nx.DiGraph` rather than real parsing — these
tests verify the query primitives in isolation from graph construction.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph.graph_builder import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_FUNCTION_CALL,
    EDGE_TYPE_IMPORTS,
    EDGE_TYPE_INHERITS,
    EDGE_TYPE_METHOD_CALL,
)
from graph.graph_queries import get_callees, get_callers, get_child_methods, get_neighbors, get_parent_class


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    # caller -> callee (function call), callee -> helper (method call)
    graph.add_edge("caller", "callee", edge_type=EDGE_TYPE_FUNCTION_CALL)
    graph.add_edge("callee", "helper", edge_type=EDGE_TYPE_METHOD_CALL)
    # child class -> parent class
    graph.add_edge("child_class", "parent_class", edge_type=EDGE_TYPE_INHERITS)
    # class -> its methods
    graph.add_edge("a_class", "method_1", edge_type=EDGE_TYPE_CONTAINS)
    graph.add_edge("a_class", "method_2", edge_type=EDGE_TYPE_CONTAINS)
    # file -> imported file
    graph.add_edge("file_a", "file_b", edge_type=EDGE_TYPE_IMPORTS)
    return graph


def test_get_callers(sample_graph: nx.DiGraph) -> None:
    assert get_callers(sample_graph, "callee") == ["caller"]
    assert get_callers(sample_graph, "caller") == []


def test_get_callers_excludes_non_call_edges(sample_graph: nx.DiGraph) -> None:
    assert get_callers(sample_graph, "parent_class") == []


def test_get_callees(sample_graph: nx.DiGraph) -> None:
    assert get_callees(sample_graph, "caller") == ["callee"]
    assert get_callees(sample_graph, "callee") == ["helper"]
    assert get_callees(sample_graph, "helper") == []


def test_get_neighbors_includes_all_edge_types_both_directions(sample_graph: nx.DiGraph) -> None:
    assert get_neighbors(sample_graph, "callee") == ["caller", "helper"]


def test_get_neighbors_unknown_node_returns_empty(sample_graph: nx.DiGraph) -> None:
    assert get_neighbors(sample_graph, "does-not-exist") == []


def test_get_parent_class(sample_graph: nx.DiGraph) -> None:
    assert get_parent_class(sample_graph, "child_class") == "parent_class"


def test_get_parent_class_none_when_no_base(sample_graph: nx.DiGraph) -> None:
    assert get_parent_class(sample_graph, "parent_class") is None


def test_get_parent_class_unknown_node_returns_none(sample_graph: nx.DiGraph) -> None:
    assert get_parent_class(sample_graph, "does-not-exist") is None


def test_get_child_methods(sample_graph: nx.DiGraph) -> None:
    assert set(get_child_methods(sample_graph, "a_class")) == {"method_1", "method_2"}


def test_get_child_methods_empty_for_non_class(sample_graph: nx.DiGraph) -> None:
    assert get_child_methods(sample_graph, "method_1") == []


def test_accepts_uuid_chunk_id() -> None:
    import uuid

    node_id = uuid.uuid4()
    graph = nx.DiGraph()
    graph.add_edge(str(node_id), "other", edge_type=EDGE_TYPE_FUNCTION_CALL)

    assert get_callees(graph, node_id) == ["other"]
