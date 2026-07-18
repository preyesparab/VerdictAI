"""Tests for evaluation.ablation.

RAGAS's real evaluation call is mocked via `evaluation.ragas_eval._run_ragas_evaluate`
(same seam `test_ragas_eval.py` uses); retrieval/generation components are
all fakes - these tests verify per-configuration flag branching (which
stages run/are skipped), metric aggregation, and result persistence, never
a real pipeline or LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.ragas_eval as ragas_eval_module
from config import settings
from core.exceptions import EvaluationError
from evaluation.ablation import (
    ABLATION_CONFIGS,
    AblationConfig,
    PipelineComponents,
    _apply_config,
    print_comparison_table,
    run_ablation,
    save_ablation_results,
)
from evaluation.eval_dataset import EvalCase
from models.schemas import (
    ChunkType,
    CodeChunk,
    ExpandedRetrievedChunk,
    GeneratedAnswer,
    RankedChunk,
    RetrievalSource,
    RetrievedChunk,
)


@pytest.fixture(autouse=True)
def _mock_ragas(monkeypatch: pytest.MonkeyPatch):
    def _fake_runner(rows, llm, embeddings):
        return [
            {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}
            for _ in rows
        ]

    monkeypatch.setattr(ragas_eval_module, "_run_ragas_evaluate", _fake_runner)


def _chunk(chunk_id: str, function_name: str, raw_code: str) -> CodeChunk:
    import uuid

    return CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id),
        file_id="file-1",
        file_path="src/a.py",
        language="python",
        chunk_type=ChunkType.FUNCTION,
        function_name=function_name,
        class_name=None,
        parent_class=None,
        start_line=1,
        end_line=5,
        raw_code=raw_code,
        parent_chunk_id=None,
    )


_CHUNK_A = _chunk("a", "foo", "def foo(): pass")
_CHUNK_B = _chunk("b", "bar", "def bar(): pass")
_CHUNK_ID_A = str(_CHUNK_A.chunk_id)
_CHUNK_ID_B = str(_CHUNK_B.chunk_id)


def _case() -> EvalCase:
    return EvalCase(
        repository_id="repo-1",
        query="what does foo do",
        expected_answer="It does foo things.",
        expected_source_files=["src/a.py"],
        expected_functions=["foo"],
    )


class _FakeChunkStore:
    def __init__(self, chunks: list[CodeChunk]) -> None:
        self._chunks = chunks

    def load_chunks(self, repository_id: str) -> list[CodeChunk]:
        return self._chunks


class _FakeDenseSearcher:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query_embedding, top_k: int):
        self.calls += 1
        from database.vector_store import SearchResult

        return [SearchResult(chunk_id=_CHUNK_ID_A, score=0.9)][:top_k]


class _FakeHybridRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, repository_id, query, query_embedding, top_k):
        self.calls += 1
        return [
            RetrievedChunk(
                chunk_id=_CHUNK_ID_A, dense_score=0.9, bm25_score=0.5, fused_score=0.9,
                retrieval_source=RetrievalSource.HYBRID, rank=1,
            )
        ][:top_k]


class _FakeGraphExpander:
    def __init__(self) -> None:
        self.calls = 0

    def expand(self, repository_id, retrieved_chunks):
        self.calls += 1
        expanded = [
            ExpandedRetrievedChunk(
                chunk_id=item.chunk_id, retrieval_source=item.retrieval_source, score=item.fused_score,
                graph_distance=0, originating_chunk_id=None, edge_type=None,
            )
            for item in retrieved_chunks
        ]
        expanded.append(
            ExpandedRetrievedChunk(
                chunk_id=_CHUNK_ID_B, retrieval_source=RetrievalSource.GRAPH, score=0.4,
                graph_distance=1, originating_chunk_id=_CHUNK_ID_A, edge_type="function_call",
            )
        )
        return expanded


class _FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, candidates, top_k):
        self.calls += 1
        return [
            RankedChunk(
                chunk_id=candidate.chunk_id, cross_encoder_score=1.0, previous_retrieval_score=candidate.previous_score,
                final_rank=rank, retrieval_source=candidate.retrieval_source,
            )
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]


class _FakeLLMService:
    def __init__(self) -> None:
        self.calls = []

    def generate_answer(self, repository_id, query, context_document):
        self.calls.append(context_document)
        return GeneratedAnswer(
            answer="foo does foo things", cited_chunks=[], prompt_tokens=1, completion_tokens=1,
            total_tokens=2, model_name="fake", latency_ms=0.5,
        )


class _FakeJudge:
    def score(self, query: str, chunk_code: str) -> int:
        return 5


def _components(**overrides) -> tuple[PipelineComponents, dict]:
    fakes = {
        "dense_searcher": _FakeDenseSearcher(),
        "hybrid_retriever": _FakeHybridRetriever(),
        "graph_expander": _FakeGraphExpander(),
        "reranker": _FakeReranker(),
        "llm_service": _FakeLLMService(),
    }
    components = PipelineComponents(
        db=_FakeChunkStore([_CHUNK_A, _CHUNK_B]),
        dense_searcher=fakes["dense_searcher"],
        hybrid_retriever=fakes["hybrid_retriever"],
        graph_expander=fakes["graph_expander"],
        reranker=fakes["reranker"],
        llm_service=fakes["llm_service"],
        judge=_FakeJudge(),
        embed_query=lambda query: [0.1, 0.2],
        ragas_llm="fake-llm",
        ragas_embeddings="fake-embeddings",
        top_k=5,
    )
    return components, fakes


class TestAblationConfigs:
    def test_five_fixed_configurations(self) -> None:
        assert [config.name for config in ABLATION_CONFIGS] == [
            "Dense Retrieval Only",
            "Dense + BM25",
            "Dense + BM25 + Graph Expansion",
            "Dense + BM25 + Graph + Cross Encoder",
            "Full Pipeline",
        ]

    def test_flags_progressively_enable(self) -> None:
        dense_only, plus_bm25, plus_graph, plus_reranker, full = ABLATION_CONFIGS

        assert (dense_only.use_bm25, dense_only.use_graph_expansion, dense_only.use_reranker, dense_only.use_small_to_big) == (False, False, False, False)
        assert (plus_bm25.use_bm25, plus_bm25.use_graph_expansion, plus_bm25.use_reranker) == (True, False, False)
        assert (plus_graph.use_bm25, plus_graph.use_graph_expansion, plus_graph.use_reranker) == (True, True, False)
        assert (plus_reranker.use_bm25, plus_reranker.use_graph_expansion, plus_reranker.use_reranker) == (True, True, True)
        assert (full.use_bm25, full.use_graph_expansion, full.use_reranker, full.use_small_to_big) == (True, True, True, True)


class TestApplyConfig:
    def test_sets_and_restores_settings_flags(self) -> None:
        original = (settings.USE_BM25, settings.USE_GRAPH_EXPANSION, settings.USE_RERANKER, settings.USE_SMALL_TO_BIG)
        config = AblationConfig("test", use_bm25=False, use_graph_expansion=False, use_reranker=False, use_small_to_big=False)

        with _apply_config(config):
            assert settings.USE_BM25 is False
            assert settings.USE_GRAPH_EXPANSION is False

        assert (settings.USE_BM25, settings.USE_GRAPH_EXPANSION, settings.USE_RERANKER, settings.USE_SMALL_TO_BIG) == original

    def test_restores_on_exception(self) -> None:
        original = settings.USE_BM25
        config = AblationConfig("test", use_bm25=not original, use_graph_expansion=False, use_reranker=False, use_small_to_big=False)

        with pytest.raises(RuntimeError):
            with _apply_config(config):
                raise RuntimeError("boom")

        assert settings.USE_BM25 == original


class TestRunAblation:
    def test_dense_only_skips_hybrid_graph_and_reranker(self) -> None:
        components, fakes = _components()
        config = AblationConfig("Dense Only", use_bm25=False, use_graph_expansion=False, use_reranker=False, use_small_to_big=False)

        run_ablation([_case()], components, configs=[config])

        assert fakes["dense_searcher"].calls == 1
        assert fakes["hybrid_retriever"].calls == 0
        assert fakes["graph_expander"].calls == 0
        assert fakes["reranker"].calls == 0
        assert len(fakes["llm_service"].calls) == 1

    def test_full_pipeline_uses_every_stage(self) -> None:
        components, fakes = _components()
        config = AblationConfig("Full", use_bm25=True, use_graph_expansion=True, use_reranker=True, use_small_to_big=True)

        run_ablation([_case()], components, configs=[config])

        assert fakes["dense_searcher"].calls == 0
        assert fakes["hybrid_retriever"].calls == 1
        assert fakes["graph_expander"].calls == 1
        assert fakes["reranker"].calls == 1

    def test_result_contains_expected_metrics(self) -> None:
        components, _ = _components()

        report = run_ablation([_case()], components, configs=[ABLATION_CONFIGS[0]])

        assert len(report.results) == 1
        result = report.results[0]
        assert result.config_name == "Dense Retrieval Only"
        assert result.faithfulness == pytest.approx(0.9)
        assert result.answer_relevancy == pytest.approx(0.8)
        assert result.context_precision == pytest.approx(0.7)
        assert result.retrieval_precision_at_5 == pytest.approx(1.0)  # fake judge always scores 5
        assert result.average_latency_ms >= 0.0

    def test_runs_every_default_configuration(self) -> None:
        components, _ = _components()

        report = run_ablation([_case()], components)

        assert len(report.results) == len(ABLATION_CONFIGS)

    def test_empty_cases_raises(self) -> None:
        components, _ = _components()

        with pytest.raises(EvaluationError):
            run_ablation([], components)

    def test_pipeline_failure_raises_evaluation_error(self) -> None:
        class _RaisingSearcher:
            def search(self, query_embedding, top_k):
                raise RuntimeError("index unavailable")

        components, _ = _components()
        components.dense_searcher = _RaisingSearcher()
        config = AblationConfig("Dense Only", use_bm25=False, use_graph_expansion=False, use_reranker=False, use_small_to_big=False)

        with pytest.raises(EvaluationError):
            run_ablation([_case()], components, configs=[config])


class TestPrintComparisonTable:
    def test_prints_header_and_rows(self, capsys: pytest.CaptureFixture[str]) -> None:
        components, _ = _components()
        report = run_ablation([_case()], components, configs=[ABLATION_CONFIGS[0]])

        print_comparison_table(report)

        output = capsys.readouterr().out
        assert "Configuration" in output
        assert "Dense Retrieval Only" in output


class TestSaveAblationResults:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        components, _ = _components()
        report = run_ablation([_case()], components, configs=[ABLATION_CONFIGS[0]])
        destination = tmp_path / "ablation_results.json"

        result_path = save_ablation_results(report, destination)

        assert result_path == destination
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert len(payload["results"]) == 1
        assert payload["results"][0]["config_name"] == "Dense Retrieval Only"

    def test_write_failure_raises_evaluation_error(self, tmp_path: Path) -> None:
        components, _ = _components()
        report = run_ablation([_case()], components, configs=[ABLATION_CONFIGS[0]])
        blocking_file = tmp_path / "blocking"
        blocking_file.write_text("x", encoding="utf-8")

        with pytest.raises(EvaluationError):
            save_ablation_results(report, blocking_file / "ablation_results.json")
