"""Tests for generation.context_builder.ContextBuilder.

`_FakeChunkStore` stands in for `database.sqlite_client.DatabaseManager` -
these tests verify small-to-big substitution, deduplication, ordering,
token budgeting, and error handling, never a real SQLite database.
"""

from __future__ import annotations

import uuid

import pytest

import generation.context_builder as context_builder_module
from core.exceptions import DatabaseError, RetrievalError
from generation.context_builder import ContextBuilder, estimate_token_count
from models.schemas import ChunkType, CodeChunk, RankedChunk, RetrievalSource


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, seed)


def _chunk(
    seed: str,
    *,
    raw_code: str = "def foo(): pass",
    chunk_type: ChunkType = ChunkType.FUNCTION,
    function_name: str | None = "foo",
    class_name: str | None = None,
    parent_chunk_id: uuid.UUID | None = None,
    file_path: str = "src/a.py",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=_uuid(seed),
        file_id="file-1",
        file_path=file_path,
        language="python",
        chunk_type=chunk_type,
        function_name=function_name,
        class_name=class_name,
        parent_class=None,
        start_line=1,
        end_line=10,
        raw_code=raw_code,
        parent_chunk_id=parent_chunk_id,
    )


def _ranked(chunk_id: uuid.UUID, rank: int) -> RankedChunk:
    return RankedChunk(
        chunk_id=str(chunk_id),
        cross_encoder_score=1.0 / rank,
        previous_retrieval_score=0.5,
        final_rank=rank,
        retrieval_source=RetrievalSource.HYBRID,
    )


class _FakeChunkStore:
    def __init__(self, chunks: list[CodeChunk], *, raise_error: bool = False) -> None:
        self._chunks = chunks
        self._raise_error = raise_error

    def load_chunks(self, repository_id: str) -> list[CodeChunk]:
        if self._raise_error:
            raise DatabaseError("boom")
        return self._chunks


def _rendered_block(chunk: CodeChunk) -> context_builder_module._RenderedBlock:
    """Build the `_RenderedBlock` `ContextBuilder` would produce for `chunk` with no substitution."""
    return context_builder_module._RenderedBlock(
        chunk_id=str(chunk.chunk_id),
        file_path=chunk.file_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        function_name=chunk.function_name,
        class_name=chunk.class_name,
        raw_code=chunk.raw_code,
    )


class TestParentSubstitution:
    def test_substitutes_parent_chunk_when_present(self) -> None:
        parent = _chunk("parent", raw_code="class Foo:\n    def foo(self): pass\n")
        child = _chunk("child", raw_code="def foo(self): pass", parent_chunk_id=parent.chunk_id)
        builder = ContextBuilder(_FakeChunkStore([parent, child]))

        doc = builder.build_context("repo-1", "how does foo work", [_ranked(child.chunk_id, 1)])

        assert doc.included_chunks == [str(parent.chunk_id)]
        assert "class Foo:" in doc.context

    def test_uses_chunk_directly_when_no_parent(self) -> None:
        chunk = _chunk("solo")
        builder = ContextBuilder(_FakeChunkStore([chunk]))

        doc = builder.build_context("repo-1", "q", [_ranked(chunk.chunk_id, 1)])

        assert doc.included_chunks == [str(chunk.chunk_id)]

    def test_uses_chunk_directly_when_small_to_big_disabled(self) -> None:
        parent = _chunk("parent2", raw_code="class Bar: ...")
        child = _chunk("child2", raw_code="def bar(self): pass", parent_chunk_id=parent.chunk_id)
        builder = ContextBuilder(_FakeChunkStore([parent, child]), use_small_to_big=False)

        doc = builder.build_context("repo-1", "q", [_ranked(child.chunk_id, 1)])

        assert doc.included_chunks == [str(child.chunk_id)]

    def test_falls_back_to_chunk_when_parent_missing_from_store(self) -> None:
        missing_parent_id = _uuid("missing-parent")
        child = _chunk("child3", parent_chunk_id=missing_parent_id)
        builder = ContextBuilder(_FakeChunkStore([child]))

        doc = builder.build_context("repo-1", "q", [_ranked(child.chunk_id, 1)])

        assert doc.included_chunks == [str(child.chunk_id)]


class TestDuplicateRemoval:
    def test_multiple_children_sharing_one_parent_collapse_to_one_block(self) -> None:
        parent = _chunk("shared-parent", raw_code="class Baz:\n    ...\n")
        method_a = _chunk("method-a", function_name="a", parent_chunk_id=parent.chunk_id)
        method_b = _chunk("method-b", function_name="b", parent_chunk_id=parent.chunk_id)
        builder = ContextBuilder(_FakeChunkStore([parent, method_a, method_b]))

        doc = builder.build_context(
            "repo-1", "q", [_ranked(method_a.chunk_id, 1), _ranked(method_b.chunk_id, 2)]
        )

        assert doc.included_chunks == [str(parent.chunk_id)]
        assert doc.context.count("class Baz:") == 1


class TestOrdering:
    def test_preserves_final_rank_order_regardless_of_input_order(self) -> None:
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        builder = ContextBuilder(_FakeChunkStore([c1, c2, c3]))

        # Passed out of rank order on purpose.
        doc = builder.build_context(
            "repo-1",
            "q",
            [_ranked(c3.chunk_id, 3), _ranked(c1.chunk_id, 1), _ranked(c2.chunk_id, 2)],
        )

        assert doc.included_chunks == [str(c1.chunk_id), str(c2.chunk_id), str(c3.chunk_id)]


class TestTokenBudget:
    def test_all_chunks_fit_within_default_budget(self) -> None:
        chunks = [_chunk(f"fit-{i}") for i in range(3)]
        builder = ContextBuilder(_FakeChunkStore(chunks))

        doc = builder.build_context(
            "repo-1", "q", [_ranked(chunk.chunk_id, i + 1) for i, chunk in enumerate(chunks)]
        )

        assert len(doc.included_chunks) == 3
        assert doc.truncated is False

    def test_drops_lowest_ranked_chunks_exceeding_budget(self) -> None:
        chunks = [_chunk(f"budget-{i}", raw_code="same code body") for i in range(3)]
        block_tokens = estimate_token_count(
            context_builder_module._format_block(_rendered_block(chunks[0]), "repo-1")
        )
        builder = ContextBuilder(_FakeChunkStore(chunks), max_context_tokens=block_tokens)

        doc = builder.build_context(
            "repo-1", "q", [_ranked(chunk.chunk_id, i + 1) for i, chunk in enumerate(chunks)]
        )

        assert doc.included_chunks == [str(chunks[0].chunk_id)]
        assert doc.truncated is True

    def test_keeps_single_oversized_top_chunk_whole(self) -> None:
        chunk = _chunk("oversized", raw_code="x" * 500)
        block_tokens = estimate_token_count(
            context_builder_module._format_block(_rendered_block(chunk), "repo-1")
        )
        builder = ContextBuilder(_FakeChunkStore([chunk]), max_context_tokens=block_tokens - 1)

        doc = builder.build_context("repo-1", "q", [_ranked(chunk.chunk_id, 1)])

        assert doc.included_chunks == [str(chunk.chunk_id)]
        assert doc.truncated is False  # nothing else existed to drop
        assert doc.total_tokens > builder._max_context_tokens  # noqa: SLF001 - white-box: budget was exceeded by design

    def test_oversized_top_chunk_still_drops_lower_ranked_remainder(self) -> None:
        chunks = [_chunk(f"over-{i}", raw_code="x" * 500) for i in range(3)]
        block_tokens = estimate_token_count(
            context_builder_module._format_block(_rendered_block(chunks[0]), "repo-1")
        )
        builder = ContextBuilder(_FakeChunkStore(chunks), max_context_tokens=block_tokens - 1)

        doc = builder.build_context(
            "repo-1", "q", [_ranked(chunk.chunk_id, i + 1) for i, chunk in enumerate(chunks)]
        )

        assert doc.included_chunks == [str(chunks[0].chunk_id)]
        assert doc.truncated is True


class TestEmptyContext:
    def test_empty_ranked_chunks_returns_empty_context(self) -> None:
        builder = ContextBuilder(_FakeChunkStore([]))

        doc = builder.build_context("repo-1", "q", [])

        assert doc.context == ""
        assert doc.included_chunks == []
        assert doc.chunk_citations == []
        assert doc.total_tokens == 0
        assert doc.truncated is False


class TestErrorHandling:
    def test_missing_chunk_id_raises_retrieval_error(self) -> None:
        builder = ContextBuilder(_FakeChunkStore([]))

        with pytest.raises(RetrievalError):
            builder.build_context("repo-1", "q", [_ranked(_uuid("ghost"), 1)])

    def test_database_error_wrapped_as_retrieval_error(self) -> None:
        builder = ContextBuilder(_FakeChunkStore([], raise_error=True))

        with pytest.raises(RetrievalError):
            builder.build_context("repo-1", "q", [_ranked(_uuid("any"), 1)])

    def test_non_positive_max_context_tokens_raises(self) -> None:
        with pytest.raises(RetrievalError):
            ContextBuilder(_FakeChunkStore([]), max_context_tokens=0)


class TestContextFormat:
    def test_block_contains_required_fields_in_order(self) -> None:
        chunk = _chunk(
            "formatted",
            raw_code="def handler(): return 1",
            chunk_type=ChunkType.METHOD,
            function_name="handler",
            class_name="Handler",
        )
        builder = ContextBuilder(_FakeChunkStore([chunk]))

        doc = builder.build_context("repo-42", "q", [_ranked(chunk.chunk_id, 1)])

        labels_in_order = [
            "Repository: repo-42",
            "File Path: src/a.py",
            "Language: python",
            "Chunk Type: method",
            "Function Name: handler",
            "Class Name: Handler",
            "Raw Code:",
            "def handler(): return 1",
        ]
        positions = [doc.context.index(label) for label in labels_in_order]
        assert positions == sorted(positions)


class TestChunkCitations:
    def test_citation_uses_original_function_name_after_parent_substitution(self) -> None:
        parent = _chunk("cite-parent", raw_code="class Foo:\n    def foo(self): pass\n")
        child = _chunk(
            "cite-child", function_name="foo", class_name="Foo", parent_chunk_id=parent.chunk_id
        )
        builder = ContextBuilder(_FakeChunkStore([parent, child]))

        doc = builder.build_context("repo-1", "q", [_ranked(child.chunk_id, 1)])

        assert len(doc.chunk_citations) == 1
        citation = doc.chunk_citations[0]
        assert citation.chunk_id == str(parent.chunk_id)  # displayed block's id, not the child's
        assert citation.function_name == "foo"  # the original match's name, not the parent's (None)
        assert citation.class_name == "Foo"
        assert citation.file_path == child.file_path

    def test_citation_matches_chunk_directly_when_no_substitution(self) -> None:
        chunk = _chunk("cite-solo", function_name="solo_fn", class_name="Solo")
        builder = ContextBuilder(_FakeChunkStore([chunk]))

        doc = builder.build_context("repo-1", "q", [_ranked(chunk.chunk_id, 1)])

        assert doc.chunk_citations == [
            context_builder_module.ChunkCitation(
                chunk_id=str(chunk.chunk_id),
                file_path=chunk.file_path,
                function_name="solo_fn",
                class_name="Solo",
            )
        ]

    def test_shared_parent_citation_labeled_by_highest_ranked_contributor(self) -> None:
        parent = _chunk("cite-shared-parent", raw_code="class Baz:\n    ...\n")
        method_a = _chunk("cite-method-a", function_name="a", parent_chunk_id=parent.chunk_id)
        method_b = _chunk("cite-method-b", function_name="b", parent_chunk_id=parent.chunk_id)
        builder = ContextBuilder(_FakeChunkStore([parent, method_a, method_b]))

        doc = builder.build_context(
            "repo-1", "q", [_ranked(method_a.chunk_id, 1), _ranked(method_b.chunk_id, 2)]
        )

        assert len(doc.chunk_citations) == 1
        assert doc.chunk_citations[0].function_name == "a"

    def test_citations_omit_chunks_dropped_by_token_budget(self) -> None:
        chunks = [_chunk(f"cite-budget-{i}", raw_code="same code body") for i in range(3)]
        block_tokens = estimate_token_count(
            context_builder_module._format_block(_rendered_block(chunks[0]), "repo-1")
        )
        builder = ContextBuilder(_FakeChunkStore(chunks), max_context_tokens=block_tokens)

        doc = builder.build_context(
            "repo-1", "q", [_ranked(chunk.chunk_id, i + 1) for i, chunk in enumerate(chunks)]
        )

        assert [citation.chunk_id for citation in doc.chunk_citations] == [str(chunks[0].chunk_id)]


class TestTokenEstimation:
    def test_empty_text_is_zero_tokens(self) -> None:
        assert estimate_token_count("") == 0

    def test_short_text_is_at_least_one_token(self) -> None:
        assert estimate_token_count("a") == 1

    def test_scales_with_length(self) -> None:
        assert estimate_token_count("a" * 4) == 1
        assert estimate_token_count("a" * 5) == 2
        assert estimate_token_count("a" * 8) == 2


class TestDefaults:
    def test_defaults_come_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(context_builder_module.settings, "MAX_CONTEXT_TOKENS", 4000)
        monkeypatch.setattr(context_builder_module.settings, "USE_SMALL_TO_BIG", False)
        parent = _chunk("default-parent")
        child = _chunk("default-child", parent_chunk_id=parent.chunk_id)
        builder = ContextBuilder(_FakeChunkStore([parent, child]))

        doc = builder.build_context("repo-1", "q", [_ranked(child.chunk_id, 1)])

        assert doc.included_chunks == [str(child.chunk_id)]  # small-to-big disabled by settings
