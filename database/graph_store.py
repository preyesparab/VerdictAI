"""Persistence adapter for the code structure/dependency graph.

Serializes the `networkx.DiGraph` produced by
`graph.graph_builder.RepositoryGraphBuilder` to/from a `GraphSnapshotRecord`
row (`database.models`) using NetworkX's node-link format — the same plain,
JSON-shaped format used before the Phase 33 storage migration, just stored
in a `JSONB` column instead of a file at `data/graph/{owner}_{name}.json`.
`save_graph` and `load_graph` are the only public entry points; callers
never touch the JSON structure or the ORM row directly.

Both functions take the caller's `DatabaseManager` rather than opening an
independent connection: `GraphSnapshotRecord.repository_id` is a foreign
key into `RepositoryRecord`, so a snapshot must be written/read through the
exact same engine/schema every other table for that repository uses (in
particular, the per-test schema isolation `tests/conftest.py` sets up) -
not a second, independently-configured connection to the same database
that could silently point at a different schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import networkx as nx

from core.exceptions import DatabaseError
from core.logging import get_logger
from database.models import GraphSnapshotRecord

if TYPE_CHECKING:
    from database.sqlite_client import DatabaseManager

logger = get_logger(__name__)


def save_graph(graph: nx.DiGraph, repository_id: str, db: DatabaseManager) -> None:
    """Serialize `graph` and upsert it as the snapshot for `repository_id`.

    Args:
        graph: The graph to persist, typically built by
            `RepositoryGraphBuilder.build_graph`.
        repository_id: The owning repository's id (from
            `db.store_repository` or `db.compute_repository_id`) - must
            already have a stored `RepositoryRecord`, since this table's
            `repository_id` is a foreign key into it.
        db: The `database.sqlite_client.DatabaseManager` whose engine/schema
            to write through.

    Raises:
        DatabaseError: If the write fails.
    """
    data = nx.node_link_data(graph)

    with db.session_scope() as session:
        record = session.get(GraphSnapshotRecord, repository_id)
        if record is None:
            record = GraphSnapshotRecord(repository_id=repository_id)
            session.add(record)
        record.graph_data = data
        record.updated_at = datetime.now(timezone.utc)

    logger.info(
        "Graph snapshot saved for repository %s: %d node(s), %d edge(s)",
        repository_id, graph.number_of_nodes(), graph.number_of_edges(),
    )


def load_graph(repository_id: str, db: DatabaseManager) -> nx.DiGraph:
    """Load the graph snapshot previously written by `save_graph`.

    Args:
        repository_id: The repository whose snapshot to load.
        db: The `database.sqlite_client.DatabaseManager` whose engine/schema
            to read through - must be the same one `save_graph` wrote with.

    Returns:
        The deserialized `networkx.DiGraph`.

    Raises:
        DatabaseError: If no snapshot is stored for `repository_id`, or
            the stored JSON cannot be deserialized.
    """
    with db.session_scope() as session:
        record = session.get(GraphSnapshotRecord, repository_id)

    if record is None:
        raise DatabaseError(f"No graph snapshot stored for repository {repository_id}")

    try:
        graph = nx.node_link_graph(record.graph_data, directed=True, multigraph=False)
    except (KeyError, TypeError) as exc:
        raise DatabaseError(f"Failed to parse graph snapshot for repository {repository_id}: {exc}") from exc

    logger.info(
        "Graph snapshot loaded for repository %s: %d node(s), %d edge(s)",
        repository_id, graph.number_of_nodes(), graph.number_of_edges(),
    )
    return graph
