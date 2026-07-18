"""Tests for the pure display-logic helpers on pipeline.Pipeline.

These specifically guard the two bugs found while manually verifying the
Streamlit app: citations losing the originally-matched function name, and
retrieval source/relevance score becoming unrecoverable, both after
small-to-big substitution replaces a cited chunk_id with its parent's.
Anything requiring a real repository/index/LLM is covered by the manual
CLI testing instructions instead (see this phase's PROGRESS.md summary).

`Pipeline` supersedes the older Phase-5-only `IndexingPipeline` that used
to be tested from this file (acquire -> discover -> parse -> chunk only);
that behavior is now a subset of `Pipeline.index_repository`, exercised
end-to-end via the CLI rather than with mocked collaborators.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from config import settings
from database.sqlite_client import DatabaseManager
from models.schemas import ChunkCitation, ChunkType, CodeChunk, RankedChunk, RetrievalSource
from pipeline import Pipeline


def _chunk(seed: str, *, parent_chunk_id: uuid.UUID | None = None, function_name: str | None = "foo") -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, seed),
        file_id="file-1",
        file_path="src/a.py",
        language="python",
        chunk_type=ChunkType.FUNCTION,
        function_name=function_name,
        class_name=None,
        parent_class=None,
        start_line=1,
        end_line=5,
        raw_code="def foo(): pass",
        parent_chunk_id=parent_chunk_id,
    )


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(db=DatabaseManager(db_path=tmp_path / "test.db"))


@pytest.fixture(autouse=True)
def _restore_small_to_big():
    original = settings.USE_SMALL_TO_BIG
    yield
    settings.USE_SMALL_TO_BIG = original


class TestDisplayChunkId:
    def test_returns_own_id_when_no_parent(self, pipeline: Pipeline) -> None:
        chunk = _chunk("solo")
        chunk_map = {str(chunk.chunk_id): chunk}

        assert pipeline._display_chunk_id(str(chunk.chunk_id), chunk_map) == str(chunk.chunk_id)

    def test_returns_parent_id_when_substituted(self, pipeline: Pipeline) -> None:
        settings.USE_SMALL_TO_BIG = True
        parent = _chunk("parent")
        child = _chunk("child", parent_chunk_id=parent.chunk_id)
        chunk_map = {str(parent.chunk_id): parent, str(child.chunk_id): child}

        assert pipeline._display_chunk_id(str(child.chunk_id), chunk_map) == str(parent.chunk_id)

    def test_returns_own_id_when_small_to_big_disabled(self, pipeline: Pipeline) -> None:
        settings.USE_SMALL_TO_BIG = False
        parent = _chunk("parent2")
        child = _chunk("child2", parent_chunk_id=parent.chunk_id)
        chunk_map = {str(parent.chunk_id): parent, str(child.chunk_id): child}

        assert pipeline._display_chunk_id(str(child.chunk_id), chunk_map) == str(child.chunk_id)

    def test_returns_own_id_when_parent_missing_from_store(self, pipeline: Pipeline) -> None:
        settings.USE_SMALL_TO_BIG = True
        missing_parent_id = uuid.uuid5(uuid.NAMESPACE_DNS, "missing")
        child = _chunk("child3", parent_chunk_id=missing_parent_id)
        chunk_map = {str(child.chunk_id): child}

        assert pipeline._display_chunk_id(str(child.chunk_id), chunk_map) == str(child.chunk_id)

    def test_unknown_chunk_id_returns_itself(self, pipeline: Pipeline) -> None:
        assert pipeline._display_chunk_id("ghost", {}) == "ghost"


class TestBuildCitations:
    def test_prefers_citation_label_over_chunk_map_for_function_name(self, pipeline: Pipeline) -> None:
        parent = _chunk("parent4", function_name=None)  # a real parent chunk: no function_name
        chunk_map = {str(parent.chunk_id): parent}
        # The original match's name, preserved by ContextBuilder's ChunkCitation - not the parent's None.
        labels = {
            str(parent.chunk_id): ChunkCitation(
                chunk_id=str(parent.chunk_id), file_path="src/a.py", function_name="original_fn", class_name=None
            )
        }

        citations = pipeline._build_citations([str(parent.chunk_id)], chunk_map, {}, citation_labels=labels)

        assert citations[0].function_name == "original_fn"
        assert citations[0].chunk_type == "function"  # still sourced from chunk_map

    def test_falls_back_to_chunk_map_when_no_citation_label(self, pipeline: Pipeline) -> None:
        chunk = _chunk("solo2", function_name="direct_fn")
        chunk_map = {str(chunk.chunk_id): chunk}

        citations = pipeline._build_citations([str(chunk.chunk_id)], chunk_map, {})

        assert citations[0].function_name == "direct_fn"

    def test_uses_ranked_entry_for_retrieval_source_and_score(self, pipeline: Pipeline) -> None:
        chunk = _chunk("ranked1")
        chunk_map = {str(chunk.chunk_id): chunk}
        ranked = RankedChunk(
            chunk_id=str(chunk.chunk_id), cross_encoder_score=2.5, previous_retrieval_score=0.1,
            final_rank=1, retrieval_source=RetrievalSource.HYBRID,
        )

        citations = pipeline._build_citations([str(chunk.chunk_id)], chunk_map, {str(chunk.chunk_id): ranked})

        assert citations[0].retrieval_source == "hybrid"
        assert citations[0].relevance_score == 2.5

    def test_falls_back_to_source_label_when_not_ranked(self, pipeline: Pipeline) -> None:
        chunk = _chunk("unranked1")
        chunk_map = {str(chunk.chunk_id): chunk}

        citations = pipeline._build_citations([str(chunk.chunk_id)], chunk_map, {}, source_label="cache")

        assert citations[0].retrieval_source == "cache"
        assert citations[0].relevance_score == 0.0

    def test_skips_chunk_ids_not_in_store(self, pipeline: Pipeline) -> None:
        citations = pipeline._build_citations(["ghost-id"], {}, {})

        assert citations == []

    def test_preserves_order(self, pipeline: Pipeline) -> None:
        first, second = _chunk("order-a"), _chunk("order-b")
        chunk_map = {str(first.chunk_id): first, str(second.chunk_id): second}

        citations = pipeline._build_citations([str(second.chunk_id), str(first.chunk_id)], chunk_map, {})

        assert [c.chunk_id for c in citations] == [str(second.chunk_id), str(first.chunk_id)]


class TestModelInfo:
    def test_llm_model_name_reflects_provider_selection(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "USE_GEMINI", True)
        monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
        assert pipeline.llm_model_name() == "Gemini (gemini-2.5-flash)"

        monkeypatch.setattr(settings, "USE_GEMINI", False)
        monkeypatch.setattr(settings, "USE_GROQ", True)
        monkeypatch.setattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
        assert pipeline.llm_model_name() == "Groq (llama-3.3-70b-versatile)"

        monkeypatch.setattr(settings, "USE_GROQ", False)
        monkeypatch.setattr(settings, "USE_OLLAMA", True)
        monkeypatch.setattr(settings, "OLLAMA_MODEL", "mistral")
        assert pipeline.llm_model_name() == "Ollama (mistral)"

        monkeypatch.setattr(settings, "USE_OLLAMA", False)
        assert pipeline.llm_model_name() == "No LLM provider configured"


class TestQueryArgumentOrder:
    def test_query_takes_question_then_repo_id(self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
        """`query`'s spec'd signature is `query(question, repo_id)` - the reverse of the
        old `ask(repository_id, query)` this logic was moved from. Guard the order
        directly against the semantic-cache lookup call, which is the first thing
        `query` does and takes `(repository_id, query)` positionally."""
        seen: dict[str, str] = {}

        class _FakeCache:
            def lookup(self, repository_id: str, query: str):
                seen["repository_id"] = repository_id
                seen["query"] = query
                return None

        pipeline._semantic_cache = _FakeCache()
        settings.USE_SEMANTIC_CACHE = True
        try:
            with pytest.raises(Exception):  # no FAISS index for this repo - fine, we only care about the cache call
                pipeline.query("What does this repo do?", "repo-123")
        finally:
            settings.USE_SEMANTIC_CACHE = True

        assert seen == {"repository_id": "repo-123", "query": "What does this repo do?"}
