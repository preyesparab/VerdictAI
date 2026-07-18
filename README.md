# RepoMind

A production-grade GraphRAG system for software repositories: Tree-sitter AST
parsing, graph-based retrieval, hybrid (dense + sparse) retrieval, cross-encoder
reranking, semantic caching, small-to-big retrieval, and RAGAS-based evaluation.

Every retrieval component is implemented manually — no LangChain, LlamaIndex, or
Haystack.

## Status

Repository initialized. No business logic implemented yet.

## Structure

```
RepoMind/
├── app.py            # Application entrypoint
├── pipeline.py        # End-to-end pipeline orchestration
├── config.py           # Centralized configuration
├── ingestion/          # Repo loading, AST parsing, chunking
├── retrieval/          # Dense/sparse/hybrid/graph retrieval, reranking, caching
├── generation/          # Context assembly and LLM-based answer synthesis
├── evaluation/          # RAGAS evaluation harness
├── database/            # SQLite, graph store, vector store adapters
├── models/               # Shared schemas
├── prompts/               # Prompt templates
├── utils/                  # Logging, constants, exceptions
├── tests/                   # Test suite (mirrors source tree)
├── docs/                     # Architecture notes
├── scripts/                   # CLI utilities
└── data/                       # Runtime artifacts (gitignored)
```

## Development

See `CLAUDE.md` for coding rules and the required development workflow.
