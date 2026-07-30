"""GraphExpander: expands hybrid retrieval results over the knowledge graph (Phase 12).

For every chunk `retrieval.hybrid_retriever.HybridRetriever` (Phase 11)
already retrieved, discovers its direct callers/callees, containing/child
class relationships, and imported-module neighbors by traversing the
knowledge graph `graph.graph_builder.RepositoryGraphBuilder` built (Phase
6). Never rebuilds the graph or re-runs retrieval - it only loads the
already-persisted graph (`database.graph_store.load_graph`) and expands.

Unlike `database.sqlite_client.DatabaseManager.load_graph` (Phase 7, which
deliberately reconstructs chunk-to-chunk edges only), this module loads
the original full graph - file nodes included - because "imported
modules" expansion needs the file-to-file `imports` edges Phase 7 does not
persist. Each repository's graph is loaded directly by `repository_id`
(Phase 33 storage migration - `database.graph_store` is PostgreSQL-backed,
keyed the same way every other table is, so no filename derivation is
needed here anymore).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from core.exceptions import DatabaseError, RetrievalError
from core.logging import get_logger
from database.graph_store import load_graph
from database.sqlite_client import DatabaseManager
from graph.graph_builder import (
    EDGE_TYPE_IMPORTS,
    NODE_KIND_CHUNK,
)
from models.schemas import ExpandedRetrievedChunk, RetrievalSource, RetrievedChunk

logger = get_logger(__name__)

DEFAULT_DECAY_FACTOR: float = 0.5


@dataclass(frozen=True)
class _Neighbor:
    """One one-hop neighbor discovered during traversal."""

    chunk_id: str
    edge_type: str


class GraphExpander:
    """Expands hybrid retrieval results using one-hop knowledge-graph traversal.

    Priority guarantee: every chunk passed into `expand` is always present
    in its output at `graph_distance=0`, with its original
    `retrieval_source` and `fused_score` untouched, and always sorted
    ahead of every graph-discovered chunk (`graph_distance>0`) regardless
    of either's numeric score - see `expand`'s docstring.
    """

    def __init__(self, db: DatabaseManager, decay_factor: float = DEFAULT_DECAY_FACTOR) -> None:
        """Initialize the expander.

        Args:
            db: Used to load the persisted graph snapshot through the same
                engine/schema every other table for this repository uses
                (see `database.graph_store.load_graph`).
            decay_factor: Multiplier applied per hop when computing a
                graph-discovered chunk's score
                (``originating_score * decay_factor ** graph_distance``).
                Must be in ``(0, 1)`` so score strictly decreases with
                distance.

        Raises:
            ValueError: If `decay_factor` is not in ``(0, 1)``.
        """
        if not 0.0 < decay_factor < 1.0:
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")

        self._db = db
        self._decay_factor = decay_factor

    def _load_graph(self, repository_id: str) -> nx.DiGraph:
        """Load `repository_id`'s knowledge graph.

        Args:
            repository_id: The repository whose graph to load.

        Returns:
            The graph built by `RepositoryGraphBuilder.build_graph` and
            persisted by `database.graph_store.save_graph`.

        Raises:
            RetrievalError: If no snapshot is stored for `repository_id`
                or it cannot be deserialized.
        """
        try:
            graph = load_graph(repository_id, self._db)
        except DatabaseError as exc:
            raise RetrievalError(f"Failed to load graph for repository {repository_id}: {exc}") from exc

        logger.info(
            "Graph loaded for repository %s: %d node(s), %d edge(s)",
            repository_id, graph.number_of_nodes(), graph.number_of_edges(),
        )
        return graph

    def _validate_node(self, graph: nx.DiGraph, node_id: str) -> None:
        """Raise if `node_id` is a phantom node (edge reference with no node data).

        A well-formed graph never produces this from normal traversal -
        every node reachable via `in_edges`/`out_edges` was explicitly
        added with attributes by `RepositoryGraphBuilder`. It can only
        happen if the persisted JSON was corrupted/hand-edited such that
        an edge references a node id absent from the node list - NetworkX
        silently materializes such a node with no attributes at all
        rather than raising, so this check is what turns that into a
        loud, diagnosable failure instead of a `KeyError` deep in
        traversal.

        Args:
            graph: The graph being traversed.
            node_id: A neighbor node id discovered via an edge.

        Raises:
            RetrievalError: If `node_id` has no attributes.
        """
        if not graph.nodes[node_id]:
            raise RetrievalError(
                f"Graph contains an invalid node reference: {node_id!r} has no attributes (corrupted graph)"
            )

    def _index_chunks_by_file(self, graph: nx.DiGraph) -> dict[str, list[str]]:
        """Group every chunk node id by the file_id it belongs to.

        Args:
            graph: The repository's knowledge graph.

        Returns:
            A mapping from `file_id` to the chunk node ids in that file,
            used to resolve "imported modules" expansion (chunk -> file ->
            imported file -> that file's chunks).
        """
        index: dict[str, list[str]] = defaultdict(list)
        for node_id, data in graph.nodes(data=True):
            if data.get("node_kind") == NODE_KIND_CHUNK:
                file_id = data.get("file_id")
                if file_id:
                    index[file_id].append(node_id)
        return dict(index)

    def _get_neighbors(
        self, graph: nx.DiGraph, chunk_id: str, chunks_by_file: dict[str, list[str]]
    ) -> list[_Neighbor]:
        """Find every one-hop neighbor of `chunk_id`.

        Covers callers/callees (predecessors/successors via
        function_call/method_call edges), containing/child class
        relationships (predecessors/successors via contains/inherits
        edges), and imported modules (via `chunk_id`'s file's `imports`
        edges, resolved to every chunk in each imported file).

        Args:
            graph: The repository's knowledge graph.
            chunk_id: The chunk to find neighbors of.
            chunks_by_file: Output of `_index_chunks_by_file`.

        Returns:
            Every one-hop neighbor, possibly with duplicate chunk_ids if
            reachable via more than one edge/relationship.

        Raises:
            RetrievalError: If a discovered neighbor node is corrupted
                (see `_validate_node`).
        """
        if chunk_id not in graph:
            return []  # not indexed / disconnected from the graph: no neighbors, not an error

        neighbors: list[_Neighbor] = []

        for predecessor, _, data in graph.in_edges(chunk_id, data=True):
            self._validate_node(graph, predecessor)
            neighbors.append(_Neighbor(chunk_id=predecessor, edge_type=data.get("edge_type", "unknown")))

        for _, successor, data in graph.out_edges(chunk_id, data=True):
            self._validate_node(graph, successor)
            neighbors.append(_Neighbor(chunk_id=successor, edge_type=data.get("edge_type", "unknown")))

        file_id = graph.nodes[chunk_id].get("file_id")
        if file_id and file_id in graph:
            for _, imported_file_id, data in graph.out_edges(file_id, data=True):
                if data.get("edge_type") != EDGE_TYPE_IMPORTS:
                    continue
                self._validate_node(graph, imported_file_id)
                for imported_chunk_id in chunks_by_file.get(imported_file_id, []):
                    neighbors.append(_Neighbor(chunk_id=imported_chunk_id, edge_type=EDGE_TYPE_IMPORTS))

        return neighbors

    def expand(
        self,
        repository_id: str,
        retrieved_chunks: list[RetrievedChunk],
        max_hops: int = 1,
    ) -> list[ExpandedRetrievedChunk]:
        """Expand `retrieved_chunks` with their one-hop (or `max_hops`-hop) graph neighbors.

        Every chunk in `retrieved_chunks` is always present in the
        returned list at `graph_distance=0`, carrying its original
        `retrieval_source`/`fused_score` unchanged - graph expansion never
        replaces an original result, even if graph traversal also reaches
        the same chunk_id. The returned list is sorted with every
        `graph_distance=0` chunk ahead of every `graph_distance>0` chunk,
        and by descending `score` within each of those two groups.

        Args:
            repository_id: The repository whose graph to expand over.
            retrieved_chunks: `HybridRetriever.retrieve`'s output - the
                candidates to expand from.
            max_hops: Maximum traversal depth. 1 (the default) is what
                this phase's expansion strategy is designed and tested
                for; the loop is hop-count-generic so a later phase can
                raise this without a redesign.

        Returns:
            `retrieved_chunks`, plus every distinct graph-discovered
            neighbor within `max_hops`, as `ExpandedRetrievedChunk`s.

        Raises:
            RetrievalError: If `max_hops` is not positive, the graph
                cannot be loaded, or the graph contains an invalid node
                reference (see `_validate_node`).
        """
        if max_hops < 1:
            raise RetrievalError(f"max_hops must be at least 1, got {max_hops}")

        logger.info(
            "Expansion started for repository %s: %d retrieved chunk(s), max_hops=%d",
            repository_id, len(retrieved_chunks), max_hops,
        )

        graph = self._load_graph(repository_id)
        chunks_by_file = self._index_chunks_by_file(graph)

        results: dict[str, ExpandedRetrievedChunk] = {}
        for retrieved in retrieved_chunks:
            results[retrieved.chunk_id] = ExpandedRetrievedChunk(
                chunk_id=retrieved.chunk_id,
                retrieval_source=retrieved.retrieval_source,
                score=retrieved.fused_score,
                graph_distance=0,
                originating_chunk_id=None,
                edge_type=None,
            )

        original_chunk_ids = set(results)
        frontier = [(retrieved.chunk_id, retrieved.fused_score) for retrieved in retrieved_chunks]
        duplicates_removed = 0

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, float]] = []

            for chunk_id, originating_score in frontier:
                neighbors = self._get_neighbors(graph, chunk_id, chunks_by_file)
                logger.info("Neighbors discovered: %d for chunk %s (hop %d)", len(neighbors), chunk_id, hop)

                for neighbor in neighbors:
                    if neighbor.chunk_id in original_chunk_ids:
                        duplicates_removed += 1
                        continue  # graph-expanded chunks never replace an original result

                    # `originating_score` already carries the decay accumulated by
                    # every prior hop, so one more `* decay_factor` here is what
                    # makes the total decay `decay_factor ** graph_distance`
                    # relative to the original chunk - not `** hop` applied fresh
                    # at each step, which would double-decay hop 2+.
                    candidate_score = originating_score * self._decay_factor
                    existing = results.get(neighbor.chunk_id)
                    if existing is not None:
                        duplicates_removed += 1
                        if candidate_score <= existing.score:
                            continue  # keep the better-scoring path already found

                    results[neighbor.chunk_id] = ExpandedRetrievedChunk(
                        chunk_id=neighbor.chunk_id,
                        retrieval_source=RetrievalSource.GRAPH,
                        score=candidate_score,
                        graph_distance=hop,
                        originating_chunk_id=chunk_id,
                        edge_type=neighbor.edge_type,
                    )
                    next_frontier.append((neighbor.chunk_id, candidate_score))

            frontier = next_frontier

        logger.info("Duplicates removed: %d", duplicates_removed)

        ordered = sorted(
            results.values(),
            key=lambda item: (0 if item.graph_distance == 0 else 1, -item.score),
        )

        logger.info(
            "Expansion completed: %d total candidate(s) for repository %s", len(ordered), repository_id
        )
        return ordered
