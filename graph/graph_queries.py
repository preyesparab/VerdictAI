"""Read-only query helpers over a knowledge graph built by `RepositoryGraphBuilder`.

Each function takes the `networkx.DiGraph` and a chunk id (accepting
either the `uuid.UUID` or the `str` form used as the graph's node id) and
returns other chunk ids — never node attribute dicts — so callers can
compose queries (e.g. "callers of the callers") and look up attributes
via ``graph.nodes[chunk_id]`` only when actually needed. These are the
primitives a future retrieval phase's one-hop graph expansion will call
after Hybrid Retrieval (FAISS + BM25) produces its initial candidates.
"""

from __future__ import annotations

import uuid

import networkx as nx

from graph.graph_builder import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_FUNCTION_CALL,
    EDGE_TYPE_INHERITS,
    EDGE_TYPE_METHOD_CALL,
)

_CALL_EDGE_TYPES = frozenset({EDGE_TYPE_FUNCTION_CALL, EDGE_TYPE_METHOD_CALL})


def _node_id(chunk_id: uuid.UUID | str) -> str:
    """Normalize a chunk id to the string form used as the graph's node id."""
    return str(chunk_id)


def get_callers(graph: nx.DiGraph, chunk_id: uuid.UUID | str) -> list[str]:
    """Find every chunk that calls `chunk_id` (function or method calls only).

    Args:
        graph: A graph built by `RepositoryGraphBuilder.build_graph`.
        chunk_id: The callee's chunk id.

    Returns:
        Chunk ids of direct callers. Empty if `chunk_id` is not in the
        graph or has no known callers.
    """
    node_id = _node_id(chunk_id)
    if node_id not in graph:
        return []
    return [
        caller
        for caller, _, data in graph.in_edges(node_id, data=True)
        if data.get("edge_type") in _CALL_EDGE_TYPES
    ]


def get_callees(graph: nx.DiGraph, chunk_id: uuid.UUID | str) -> list[str]:
    """Find every chunk that `chunk_id` calls (function or method calls only).

    Args:
        graph: A graph built by `RepositoryGraphBuilder.build_graph`.
        chunk_id: The caller's chunk id.

    Returns:
        Chunk ids of direct callees. Empty if `chunk_id` is not in the
        graph or makes no resolvable calls.
    """
    node_id = _node_id(chunk_id)
    if node_id not in graph:
        return []
    return [
        callee
        for _, callee, data in graph.out_edges(node_id, data=True)
        if data.get("edge_type") in _CALL_EDGE_TYPES
    ]


def get_neighbors(graph: nx.DiGraph, chunk_id: uuid.UUID | str) -> list[str]:
    """Find every node directly connected to `chunk_id`, regardless of edge type or direction.

    This is the primitive one-hop graph expansion uses: given a chunk
    matched by dense/sparse retrieval, pull in everything structurally
    adjacent to it (callers, callees, parent class, containing class,
    imported/importing files) as additional candidates.

    Args:
        graph: A graph built by `RepositoryGraphBuilder.build_graph`.
        chunk_id: The node id to expand from.

    Returns:
        Neighbor ids (predecessors and successors, deduplicated). Empty
        if `chunk_id` is not in the graph.
    """
    node_id = _node_id(chunk_id)
    if node_id not in graph:
        return []
    neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
    return sorted(neighbors)


def get_parent_class(graph: nx.DiGraph, chunk_id: uuid.UUID | str) -> str | None:
    """Find the base class chunk id `chunk_id` inherits from, if any.

    Args:
        graph: A graph built by `RepositoryGraphBuilder.build_graph`.
        chunk_id: A class chunk id.

    Returns:
        The parent class's chunk id, or None if `chunk_id` is not in the
        graph or has no resolved base class.
    """
    node_id = _node_id(chunk_id)
    if node_id not in graph:
        return None
    for _, target, data in graph.out_edges(node_id, data=True):
        if data.get("edge_type") == EDGE_TYPE_INHERITS:
            return target
    return None


def get_child_methods(graph: nx.DiGraph, chunk_id: uuid.UUID | str) -> list[str]:
    """Find every method chunk id contained in the class `chunk_id`.

    Args:
        graph: A graph built by `RepositoryGraphBuilder.build_graph`.
        chunk_id: A class chunk id.

    Returns:
        Chunk ids of methods this class contains. Empty if `chunk_id` is
        not in the graph or has no resolved methods.
    """
    node_id = _node_id(chunk_id)
    if node_id not in graph:
        return []
    return [
        target
        for _, target, data in graph.out_edges(node_id, data=True)
        if data.get("edge_type") == EDGE_TYPE_CONTAINS
    ]
