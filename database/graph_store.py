"""Persistence adapter for the code structure/dependency graph.

Serializes the `networkx.DiGraph` produced by
`graph.graph_builder.RepositoryGraphBuilder` to/from JSON using
NetworkX's node-link format — a plain, human-inspectable format that
needs no database engine, appropriate for this phase (SQLite/vector
storage are later phases). `save_graph` and `load_graph` are the only
public entry points; callers never touch the JSON structure directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from config import settings
from core.exceptions import DatabaseError
from core.logging import get_logger

logger = get_logger(__name__)


def save_graph(graph: nx.DiGraph, path: Path | None = None) -> Path:
    """Serialize `graph` to JSON using NetworkX's node-link format.

    Args:
        graph: The graph to persist, typically built by
            `RepositoryGraphBuilder.build_graph`.
        path: Destination file. Defaults to `settings.GRAPH_FILE_PATH`
            (``data/graph/graph.json``).

    Returns:
        The path the graph was written to.

    Raises:
        DatabaseError: If `path` cannot be written.
    """
    destination = path or settings.GRAPH_FILE_PATH
    data = nx.node_link_data(graph)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        raise DatabaseError(f"Failed to write graph to {destination}: {exc}") from exc

    logger.info(
        "Graph serialization complete: %d node(s), %d edge(s) written to %s",
        graph.number_of_nodes(), graph.number_of_edges(), destination,
    )
    return destination


def load_graph(path: Path | None = None) -> nx.DiGraph:
    """Load a graph previously written by `save_graph`.

    Args:
        path: Source file. Defaults to `settings.GRAPH_FILE_PATH`
            (``data/graph/graph.json``).

    Returns:
        The deserialized `networkx.DiGraph`.

    Raises:
        DatabaseError: If `path` does not exist or does not contain
            valid node-link JSON.
    """
    source = path or settings.GRAPH_FILE_PATH

    try:
        raw_text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatabaseError(f"Failed to read graph from {source}: {exc}") from exc

    try:
        data = json.loads(raw_text)
        graph = nx.node_link_graph(data, directed=True, multigraph=False)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatabaseError(f"Failed to parse graph JSON from {source}: {exc}") from exc

    logger.info(
        "Graph load complete: %d node(s), %d edge(s) loaded from %s",
        graph.number_of_nodes(), graph.number_of_edges(), source,
    )
    return graph
