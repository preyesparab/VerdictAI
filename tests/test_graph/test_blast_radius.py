"""Tests for graph.blast_radius.compute_blast_radius.

Graphs are built by hand with `networkx` (same node/edge attribute
conventions as `graph.graph_builder.RepositoryGraphBuilder`), mirroring
`tests/test_retrieval/test_graph_retriever.py`'s approach for the same
persisted-graph shape.
"""

from __future__ import annotations

import networkx as nx
import pytest

from core.exceptions import RetrievalError
from graph.blast_radius import compute_blast_radius


def _node(graph: nx.DiGraph, node_id: str) -> None:
    graph.add_node(node_id, node_kind="chunk", chunk_id=node_id, function_name=node_id)


def _chain_graph() -> nx.DiGraph:
    """A -> B -> C -> D -> E (function_call edges), plus isolated Z."""
    graph = nx.DiGraph()
    for node_id in "ABCDEZ":
        _node(graph, node_id)
    graph.add_edge("A", "B", edge_type="function_call")
    graph.add_edge("B", "C", edge_type="function_call")
    graph.add_edge("C", "D", edge_type="function_call")
    graph.add_edge("D", "E", edge_type="function_call")
    return graph


class TestHopBounds:
    def test_zero_hops_returns_only_the_node(self) -> None:
        subgraph = compute_blast_radius(_chain_graph(), "C", hops=0)
        assert set(subgraph.nodes) == {"C"}

    def test_one_hop_returns_direct_predecessor_and_successor(self) -> None:
        subgraph = compute_blast_radius(_chain_graph(), "C", hops=1)
        assert set(subgraph.nodes) == {"B", "C", "D"}

    def test_two_hops_returns_the_full_two_hop_neighborhood(self) -> None:
        subgraph = compute_blast_radius(_chain_graph(), "C", hops=2)
        assert set(subgraph.nodes) == {"A", "B", "C", "D", "E"}

    def test_hops_beyond_graph_extent_does_not_crash(self) -> None:
        subgraph = compute_blast_radius(_chain_graph(), "C", hops=10)
        assert set(subgraph.nodes) == {"A", "B", "C", "D", "E"}


class TestDirectionality:
    def test_traversal_is_bidirectional(self) -> None:
        # From A (the chain's start), 1 hop should still reach B even
        # though A has no predecessors - successors alone must work.
        subgraph = compute_blast_radius(_chain_graph(), "A", hops=1)
        assert set(subgraph.nodes) == {"A", "B"}

        # From E (the chain's end), 1 hop must reach D via predecessors
        # alone, since E has no successors.
        subgraph = compute_blast_radius(_chain_graph(), "E", hops=1)
        assert set(subgraph.nodes) == {"D", "E"}


class TestIsolatedNode:
    def test_isolated_node_returns_only_itself_at_any_hop_count(self) -> None:
        subgraph = compute_blast_radius(_chain_graph(), "Z", hops=5)
        assert set(subgraph.nodes) == {"Z"}


class TestInducedSubgraphIncludesLateralEdges:
    def test_edge_between_two_neighbors_not_on_the_traversal_path_is_included(self) -> None:
        graph = _chain_graph()
        # A lateral edge between B and D - both already in C's 1-hop
        # radius via C itself, not via this edge - should still show up
        # in the induced subgraph, since it's a real edge between two
        # included nodes.
        graph.add_edge("B", "D", edge_type="function_call")

        subgraph = compute_blast_radius(graph, "C", hops=1)
        assert set(subgraph.nodes) == {"B", "C", "D"}
        assert subgraph.has_edge("B", "D")


class TestErrors:
    def test_unknown_node_raises_retrieval_error(self) -> None:
        with pytest.raises(RetrievalError):
            compute_blast_radius(_chain_graph(), "nonexistent", hops=2)

    def test_negative_hops_raises_retrieval_error(self) -> None:
        with pytest.raises(RetrievalError):
            compute_blast_radius(_chain_graph(), "C", hops=-1)
