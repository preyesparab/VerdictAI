"""Phase 16 demonstration: a real repository URL -> a generated, cited answer.

Exercises the full backend pipeline for real, against a tiny public
GitHub repository (``navdeep-G/samplemod``):

    RepositoryManager -> FileDiscovery -> TreeSitterParser -> SemanticChunker
    -> DatabaseManager (SQLite) -> RepositoryGraphBuilder -> EmbeddingManager
    -> FaissIndexManager + BM25Manager -> HybridRetriever -> GraphExpander
    -> CrossEncoderReranker -> ContextBuilder -> LLMService

Every component above is the real, production implementation - nothing is
faked except the final network call to an LLM provider: this environment
has no local Ollama server running and no OpenAI API key configured, so
`_StubLLMClient` stands in for that one call. It is not a canned string -
it reads the *real* assembled context `LLMService` hands it and cites
whatever function/file is actually first in it, so the citation-matching
step downstream is exercised against genuine data, not a fixture.

All artifacts (cloned repo, SQLite db, FAISS/BM25 indexes, graph JSON) are
written under a scratch directory, not this project's `data/` folder, so
running this script has no side effects on the repository.

Run: .venv/Scripts/python.exe scripts/demo_phase16_end_to_end.py
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# This script is invoked directly (`python scripts/demo_phase16_end_to_end.py`),
# not as a package module, so the project root is not on sys.path by
# default. Add it before importing any project package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import DEFAULT_MINILM_MODEL  # noqa: E402
from core.exceptions import ParsingError  # noqa: E402
from core.logging import get_logger  # noqa: E402
from database.graph_store import save_graph  # noqa: E402
from database.sqlite_client import DatabaseManager  # noqa: E402
from database.vector_store import FaissIndexManager  # noqa: E402
from embedding.embedding_manager import EmbeddingManager  # noqa: E402
from embedding.model_loader import load_embedding_model  # noqa: E402
from generation.answer_generator import LLMService  # noqa: E402
from generation.context_builder import ContextBuilder  # noqa: E402
from generation.llm_client import LLMCompletion  # noqa: E402
from graph.graph_builder import RepositoryGraphBuilder  # noqa: E402
from ingestion.ast_parser import TreeSitterParser  # noqa: E402
from ingestion.chunker import SemanticChunker  # noqa: E402
from ingestion.file_discovery import FileDiscovery  # noqa: E402
from ingestion.repository_manager import RepositoryManager  # noqa: E402
from models.schemas import RerankCandidate  # noqa: E402
from retrieval.graph_retriever import GraphExpander  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
from retrieval.semantic_cache import SemanticCacheManager  # noqa: E402
from retrieval.sparse_retriever import BM25Manager  # noqa: E402

logger = get_logger(__name__)

REPOSITORY_URL = "https://github.com/navdeep-G/samplemod"
QUERY = "What does the hmm function do?"

# Kept small/cached deliberately so this demo runs in seconds without
# downloading multi-hundred-MB models: MiniLM (dense embeddings) and the
# default MiniLM cross-encoder reranker are both already present in the
# local HuggingFace cache.
EMBEDDING_MODEL_NAME = DEFAULT_MINILM_MODEL


class _StubLLMClient:
    """Stands in for `generation.llm_client.LLMClient` - see module docstring.

    Reads the real "File Path:"/"Function Name:" of the first block in
    the received context and cites it, so downstream citation matching
    runs against genuine retrieved content.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        file_path = _first_match(user_prompt, r"File Path: (.+)")
        function_name = _first_match(user_prompt, r"Function Name: (.+)")
        text = (
            f"The `{function_name}` function prints a thought if an answer is available "
            f"(source: `{file_path}`, function `{function_name}`)."
        )
        return LLMCompletion(
            text=text,
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(user_prompt) + len(text)) // 4,
            model_name="demo-stub (no local Ollama/OpenAI available)",
        )


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "unknown"


def main() -> None:
    scratch_dir = Path(tempfile.mkdtemp(prefix="repomind_demo_"))
    print(f"Scratch directory: {scratch_dir}\n")

    try:
        _run(scratch_dir)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _run(scratch_dir: Path) -> None:
    repositories_dir = scratch_dir / "repositories"
    indexes_dir = scratch_dir / "indexes"
    schema = f"demo_{uuid.uuid4().hex[:16]}"

    # -- Phase 2-5: acquire, discover, parse, chunk -----------------------
    print(f"[1/9] Cloning {REPOSITORY_URL} ...")
    repository_manager = RepositoryManager(repositories_dir=repositories_dir)
    repository = repository_manager.get_repository(REPOSITORY_URL)
    print(f"      -> {repository.owner}/{repository.name} @ {repository.commit_hash[:8]}")

    source_files = FileDiscovery().discover(repository.local_path)
    print(f"[2/9] Discovered {len(source_files)} source file(s)")

    ast_parser = TreeSitterParser()
    semantic_chunker = SemanticChunker()
    chunking_results = {}
    for source_file in source_files:
        try:
            ast_chunks = ast_parser.parse(source_file)
            chunking_results[source_file.relative_path.as_posix()] = semantic_chunker.build_chunks(
                source_file, ast_chunks
            )
        except ParsingError as exc:
            logger.warning("Skipping %s: %s", source_file.relative_path, exc)

    total_ast_chunks = sum(len(r.ast_chunks) for r in chunking_results.values())
    print(f"      -> {total_ast_chunks} AST chunk(s) parsed across {len(chunking_results)} file(s)")

    # -- Phase 7: persist repository/files/chunks -------------------------
    db = DatabaseManager(schema=schema)
    db.initialize_database()
    repository_id = db.store_repository(repository)
    db.store_source_files(repository_id, source_files)
    db.store_chunks(repository_id, chunking_results)
    print(f"[3/9] Persisted to PostgreSQL (schema={schema}) as repository_id={repository_id}")

    # -- Phase 6: knowledge graph ------------------------------------------
    all_ast_chunks = [chunk for result in chunking_results.values() for chunk in result.ast_chunks]
    graph = RepositoryGraphBuilder().build_graph(source_files, all_ast_chunks)
    db.store_graph(repository_id, graph)
    save_graph(graph, repository_id, db)
    print(f"[4/9] Graph built: {graph.number_of_nodes()} node(s), {graph.number_of_edges()} edge(s)")

    # -- Phase 8-10: embed + index (dense and sparse) ---------------------
    embedding_manager = EmbeddingManager(db, model_name=EMBEDDING_MODEL_NAME)
    embedded_count = embedding_manager.generate_embeddings(repository_id)
    print(f"[5/9] Embedded {embedded_count} chunk(s) with {EMBEDDING_MODEL_NAME}")

    faiss_manager = FaissIndexManager(db, indexes_dir=indexes_dir, model_name=EMBEDDING_MODEL_NAME)
    faiss_manager.build_index(repository_id)
    bm25_manager = BM25Manager(db, indexes_dir=indexes_dir)
    bm25_manager.build_index(repository_id)
    print("      -> FAISS + BM25 indexes built")

    # -- Phase 11-13: hybrid retrieval, graph expansion, reranking --------
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    query_embedding = embedding_model.encode([QUERY], convert_to_numpy=True, show_progress_bar=False)[0]

    hybrid_retriever = HybridRetriever(faiss_manager, bm25_manager)
    retrieved = hybrid_retriever.retrieve(repository_id, QUERY, query_embedding, top_k=5)
    print(f"[6/9] Hybrid retrieval: {len(retrieved)} candidate(s) for {QUERY!r}")

    graph_expander = GraphExpander(db)
    expanded = graph_expander.expand(repository_id, retrieved)
    print(f"      -> Graph expansion: {len(expanded)} candidate(s) after 1-hop traversal")

    chunk_map = {str(chunk.chunk_id): chunk for chunk in db.load_chunks(repository_id)}
    candidates = [
        RerankCandidate(
            chunk_id=item.chunk_id,
            raw_code=chunk_map[item.chunk_id].raw_code,
            file_path=chunk_map[item.chunk_id].file_path,
            function_name=chunk_map[item.chunk_id].function_name,
            retrieval_source=item.retrieval_source,
            graph_distance=item.graph_distance,
            previous_score=item.score,
        )
        for item in expanded
        if item.chunk_id in chunk_map
    ]

    reranker = CrossEncoderReranker()
    ranked = reranker.rerank(QUERY, candidates, top_k=5)
    print(f"[7/9] Reranked to top {len(ranked)} chunk(s)")

    # -- Phase 15: context assembly ---------------------------------------
    context_builder = ContextBuilder(db)
    context_document = context_builder.build_context(repository_id, QUERY, ranked)
    print(
        f"[8/9] Context built: {len(context_document.included_chunks)} chunk(s), "
        f"~{context_document.total_tokens} token(s), truncated={context_document.truncated}"
    )

    # -- Phase 14 + 16: generation + semantic cache -----------------------
    semantic_cache = SemanticCacheManager(db, model_name=EMBEDDING_MODEL_NAME)
    llm_service = LLMService(semantic_cache=semantic_cache, llm_client=_StubLLMClient())
    answer = llm_service.generate_answer(repository_id, QUERY, context_document)

    print(f"[9/9] Answer generated in {answer.latency_ms:.2f}ms via {answer.model_name}\n")
    print("=" * 70)
    print(f"Q: {QUERY}")
    print(f"A: {answer.answer}")
    print(f"Cited chunk_ids: {answer.cited_chunks}")
    print(f"Tokens: prompt={answer.prompt_tokens} completion={answer.completion_tokens} total={answer.total_tokens}")
    print("=" * 70)

    cache_hit = semantic_cache.lookup(repository_id, "what does the hmm function do")
    print(f"\nCache verification - similar follow-up query hit cache: {cache_hit is not None}")
    if cache_hit is not None:
        print(f"  -> served cached answer without re-running retrieval/generation: {cache_hit.response!r}")

    db.drop_schema()


if __name__ == "__main__":
    main()
