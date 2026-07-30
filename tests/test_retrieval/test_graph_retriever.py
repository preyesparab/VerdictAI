"""Tests for retrieval.graph_retriever.GraphExpander.

Graphs are built by hand with `networkx` (same node/edge attribute
conventions as `graph.graph_builder.RepositoryGraphBuilder`) rather than
via real parsing - this module only traverses an already-persisted graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pytest

from core.exceptions import RetrievalError
from database.graph_store import save_graph
from database.sqlite_client import DatabaseManager
from ingestion.repository_metadata import RepositoryMetadata
from models.schemas import RetrievalSource, RetrievedChunk
from retrieval.graph_retriever import GraphExpander


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


def _file_node(graph: nx.DiGraph, file_id: str) -> None:
    graph.add_node(file_id, node_kind="file", file_id=file_id, file_path=f"{file_id}.py", language="python")


def _chunk_node(
    graph: nx.DiGraph,
    chunk_id: str,
    file_id: str,
    *,
    chunk_type: str = "function",
    function_name: str | None = None,
    class_name: str | None = None,
) -> None:
    graph.add_node(
        chunk_id,
        node_kind="chunk",
        chunk_id=chunk_id,
        file_id=file_id,
        language="python",
        chunk_type=chunk_type,
        function_name=function_name,
        class_name=class_name,
        start_line=1,
        end_line=5,
    )


def _retrieved(chunk_id: str, fused_score: float, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        dense_score=None,
        bm25_score=None,
        fused_score=fused_score,
        retrieval_source=RetrievalSource.HYBRID,
        rank=rank,
    )


def _find(results: list, chunk_id: str):
    for result in results:
        if result.chunk_id == chunk_id:
            return result
    raise AssertionError(f"{chunk_id!r} not found in results: {[r.chunk_id for r in results]}")


def _build_rich_graph() -> nx.DiGraph:
    """A small graph covering every one-hop relationship GraphExpander handles.

    file1: A (function, calls B), D (class Child, inherits C, contains E),
           E (method of D)
    file2: B (function, callee of A), C (class Base), F (standalone function)
    file3: Z (isolated - no edges, and file3 has no import relationship to
           anything, so Z gains no neighbors via the import shortcut either)
    file4: G (function, callee of B) - deliberately NOT importable from
           file1/file2, so the only path from A to G is the two-hop
           function_call chain A->B->G, not a same-hop import shortcut.

    Edges: A->B function_call, B->G function_call, D->C inherits,
           D->E contains, file1->file2 imports.
    """
    graph = nx.DiGraph()
    _file_node(graph, "file1")
    _file_node(graph, "file2")
    _file_node(graph, "file3")
    _file_node(graph, "file4")

    _chunk_node(graph, "A", "file1", function_name="handler")
    _chunk_node(graph, "D", "file1", chunk_type="class", class_name="Child")
    _chunk_node(graph, "E", "file1", chunk_type="method", function_name="method_e", class_name="Child")

    _chunk_node(graph, "B", "file2", function_name="helper")
    _chunk_node(graph, "C", "file2", chunk_type="class", class_name="Base")
    _chunk_node(graph, "F", "file2", function_name="standalone")

    _chunk_node(graph, "Z", "file3", function_name="isolated")

    _chunk_node(graph, "G", "file4", function_name="deep_helper")

    graph.add_edge("A", "B", edge_type="function_call")
    graph.add_edge("B", "G", edge_type="function_call")
    graph.add_edge("D", "C", edge_type="inherits")
    graph.add_edge("D", "E", edge_type="contains")
    graph.add_edge("file1", "file2", edge_type="imports")

    return graph


@pytest.fixture
def db(pg_schema: str) -> Iterator[DatabaseManager]:
    manager = DatabaseManager(schema=pg_schema)
    manager.initialize_database()
    yield manager
    manager.drop_schema()


def _seed_repository_with_graph(db: DatabaseManager, graph: nx.DiGraph) -> str:
    repository = _repository_metadata()
    repository_id = db.store_repository(repository)
    save_graph(graph, repository_id, db)
    return repository_id


class TestFunctionCallExpansion:
    def test_callee_is_discovered(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=1)

        callee = _find(results, "B")
        assert callee.edge_type == "function_call"
        assert callee.graph_distance == 1
        assert callee.originating_chunk_id == "A"
        assert callee.retrieval_source == RetrievalSource.GRAPH
        assert callee.score == pytest.approx(0.9 * 0.5)

    def test_caller_is_discovered(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("B", 0.9)], max_hops=1)

        caller = _find(results, "A")
        assert caller.edge_type == "function_call"
        assert caller.graph_distance == 1


class TestInheritanceExpansion:
    def test_parent_class_is_discovered_via_inherits(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("D", 0.8)], max_hops=1)

        base_class = _find(results, "C")
        assert base_class.edge_type == "inherits"
        assert base_class.graph_distance == 1


class TestContainmentExpansion:
    def test_child_method_is_discovered_from_class(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("D", 0.8)], max_hops=1)

        method = _find(results, "E")
        assert method.edge_type == "contains"
        assert method.graph_distance == 1

    def test_containing_class_is_discovered_from_method(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("E", 0.8)], max_hops=1)

        containing_class = _find(results, "D")
        assert containing_class.edge_type == "contains"
        assert containing_class.graph_distance == 1


class TestImportExpansion:
    def test_chunk_in_imported_file_is_discovered(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=1)

        imported = _find(results, "F")
        assert imported.edge_type == "imports"
        assert imported.graph_distance == 1
        assert imported.originating_chunk_id == "A"


class TestDuplicateRemoval:
    def test_chunk_reachable_from_two_originals_keeps_the_better_path(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)
        # A and D are both in file1, which imports file2 - both reach F via
        # imports expansion. A has a higher score, so F should end up
        # attributed to A, not to D's weaker path.
        retrieved = [_retrieved("A", 0.9), _retrieved("D", 0.5)]

        results = expander.expand(repository_id, retrieved, max_hops=1)

        matches = [r for r in results if r.chunk_id == "F"]
        assert len(matches) == 1  # not duplicated
        assert matches[0].originating_chunk_id == "A"
        assert matches[0].score == pytest.approx(0.9 * 0.5)

    def test_graph_expanded_chunk_never_replaces_an_original(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)
        # B is both an original hybrid result AND a graph neighbor of A.
        retrieved = [_retrieved("A", 0.9), _retrieved("B", 0.1)]

        results = expander.expand(repository_id, retrieved, max_hops=1)

        matches = [r for r in results if r.chunk_id == "B"]
        assert len(matches) == 1
        assert matches[0].graph_distance == 0
        assert matches[0].score == 0.1
        assert matches[0].retrieval_source == RetrievalSource.HYBRID


class TestDisconnectedNodes:
    def test_isolated_chunk_produces_no_expansions(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("Z", 0.6)], max_hops=1)

        assert len(results) == 1
        assert results[0].chunk_id == "Z"
        assert results[0].graph_distance == 0

    def test_chunk_not_present_in_graph_produces_no_expansions(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("never-indexed", 0.5)], max_hops=1)

        assert len(results) == 1
        assert results[0].chunk_id == "never-indexed"
        assert results[0].graph_distance == 0


class TestEmptyGraph:
    def test_empty_graph_returns_only_originals(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, nx.DiGraph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("anything", 0.5)], max_hops=1)

        assert len(results) == 1
        assert results[0].graph_distance == 0

    def test_empty_retrieved_chunks_returns_empty_list(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [], max_hops=1)

        assert results == []


class TestConfigurableHopCount:
    def test_two_hop_neighbor_absent_at_max_hops_one(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        results = expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=1)

        assert not any(r.chunk_id == "G" for r in results)

    def test_two_hop_neighbor_present_at_max_hops_two_with_compounded_decay(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db, decay_factor=0.5)

        results = expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=2)

        deep = _find(results, "G")
        assert deep.graph_distance == 2
        assert deep.originating_chunk_id == "B"
        assert deep.score == pytest.approx(0.9 * 0.5 * 0.5)

    def test_raises_on_non_positive_max_hops(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db)

        with pytest.raises(RetrievalError):
            expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=0)


class TestPriorityGuarantee:
    def test_original_outranks_expansion_regardless_of_numeric_score(self, db: DatabaseManager) -> None:
        repository_id = _seed_repository_with_graph(db, _build_rich_graph())
        expander = GraphExpander(db, decay_factor=0.9)
        # A weak original (score 0.01) alongside a strong original (score
        # 100) whose decayed neighbor (100 * 0.9 = 90) would numerically
        # exceed the weak original if sorted by score alone.
        retrieved = [_retrieved("D", 100.0), _retrieved("Z", 0.01)]

        results = expander.expand(repository_id, retrieved, max_hops=1)

        z_index = next(i for i, r in enumerate(results) if r.chunk_id == "Z")
        c_index = next(i for i, r in enumerate(results) if r.chunk_id == "C")  # D's expanded neighbor
        assert z_index < c_index  # original (Z) ranks ahead of expansion (C) despite lower score


class TestValidation:
    def test_raises_when_graph_snapshot_is_missing(self, db: DatabaseManager) -> None:
        repository_id = db.store_repository(_repository_metadata())
        expander = GraphExpander(db)

        with pytest.raises(RetrievalError):
            expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=1)

    def test_raises_on_corrupted_graph_with_invalid_node_reference(self, db: DatabaseManager) -> None:
        graph = nx.DiGraph()
        _file_node(graph, "file1")
        _chunk_node(graph, "A", "file1", function_name="handler")
        # Simulate a corrupted persisted graph: an edge to a node that was
        # never added with attributes (NetworkX materializes it silently
        # with none, rather than raising).
        graph.add_edge("A", "phantom-node", edge_type="function_call")
        repository_id = _seed_repository_with_graph(db, graph)
        expander = GraphExpander(db)

        with pytest.raises(RetrievalError):
            expander.expand(repository_id, [_retrieved("A", 0.9)], max_hops=1)

    def test_decay_factor_must_be_between_zero_and_one_exclusive(self, db: DatabaseManager) -> None:
        with pytest.raises(ValueError):
            GraphExpander(db, decay_factor=1.0)
        with pytest.raises(ValueError):
            GraphExpander(db, decay_factor=0.0)
