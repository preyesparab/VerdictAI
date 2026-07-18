"""Blast-radius subgraph extraction from an already-built knowledge graph (Phase 22).

Given a specific node (typically a function/class chunk implicated in a
flagged review claim), `compute_blast_radius` finds every node reachable
within N hops of it - via the graph's own edges (function/method calls,
class inheritance/containment, references, file-level imports) in
*either* direction, callers/importers upstream and callees/imported code
downstream - and returns the induced subgraph for highlighting.

Reuses the bidirectional, hop-bounded frontier traversal
`retrieval.graph_retriever.GraphExpander.expand` (Phase 12) already
established over this exact graph structure - same "walk predecessors
and successors, one hop at a time, stop at N" shape - generalized to
start from one arbitrary node instead of a set of retrieved chunks, and
deliberately *without* that module's retrieval-specific scoring/decay or
its "imported file's every chunk" broadening: Adjudicate's actual
downstream consumer for this (Phase 24) wants "real callers/callees",
not retrieval-style fuzzy broadening, and that broadening would make the
highlighted set unreadable for any node in a heavily-imported file.
"""

from __future__ import annotations

import networkx as nx

from core.exceptions import RetrievalError
from core.logging import get_logger

logger = get_logger(__name__)


def compute_blast_radius(graph: nx.DiGraph, node_id: str, hops: int = 2) -> nx.DiGraph:
    """Find `node_id` and every node within `hops` of it, in either direction.

    Args:
        graph: The repository's full knowledge graph, as built by
            `graph.graph_builder.RepositoryGraphBuilder` and persisted by
            `database.graph_store.save_graph`/loaded by
            `database.graph_store.load_graph`.
        node_id: The focus node's id - a chunk or file id present in
            `graph`.
        hops: Maximum traversal depth in each direction. Must be
            non-negative; 0 returns just `node_id` alone.

    Returns:
        The induced subgraph over `node_id` and every node reachable
        within `hops` hops via any of `graph`'s edges. Both predecessors
        (callers, importers, containing classes) and successors
        (callees, imported modules, contained methods) are traversed -
        "blast radius" means everything affected by a change to
        `node_id`, not just what it calls. `node_id` itself is always
        included, even at `hops=0`. The induced subgraph includes every
        edge between two included nodes, not only the edges actually
        used to reach them during traversal.

    Raises:
        RetrievalError: If `node_id` is not in `graph`, or `hops` is
            negative.
    """
    if node_id not in graph:
        raise RetrievalError(f"Node {node_id!r} not found in graph")
    if hops < 0:
        raise RetrievalError(f"hops must be non-negative, got {hops}")

    visited = {node_id}
    frontier = {node_id}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for current in frontier:
            next_frontier.update(graph.predecessors(current))
            next_frontier.update(graph.successors(current))
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier

    logger.info(
        "Blast radius computed for node %s: %d node(s) within %d hop(s)",
        node_id, len(visited), hops,
    )
    return graph.subgraph(visited).copy()
