# Verdict AI (RepoMind + Adjudicate)

A GraphRAG system for software repositories, built entirely from first
principles — no LangChain, LlamaIndex, or Haystack — plus **Adjudicate**, an
adversarial multi-agent code review pipeline built on top of it.

Two related problems, one shared retrieval engine:

1. **"I don't understand this codebase."** Point it at a GitHub URL and chat
   with it in natural language, with every answer grounded in cited,
   retrievable source code — never a hallucinated file path — plus an
   interactive call/import/inheritance graph with blast-radius highlighting.
2. **"Is this diff actually safe to merge?"** Submit a unified diff and get an
   adversarial review: a **Prosecutor** agent raises claims about the change,
   a deterministic **Verifier** checks each one against real sandboxed test
   execution and static analysis (zero LLM calls), a **Defender** rebuts, and
   a **Judge** renders a final verdict — streamed live through the UI, stage
   by stage.

## Why manual, not a framework

Every retrieval component — chunking, embedding, FAISS indexing, BM25,
Reciprocal Rank Fusion, graph expansion, cross-encoder reranking, semantic
caching — is implemented by hand. This is a demonstration project for
ML/GenAI engineering interviews: the goal is showing how each of these pieces
actually works, not wiring together a framework that hides them. See
`docs/project_description.md` for the full design rationale behind every
non-obvious choice.

## The review pipeline (Adjudicate)

```
Diff in → Context Builder (graph-first, vector fallback)
        → Defender drafts a justification for the change
        → Prosecutor raises structured claims against it
        → Verifier checks each claim for real (sandboxed pytest/mypy/bandit — no LLM)
        → Defender/Verifier rebuttal loop (capped at 3 rounds)
        → Judge renders a final verdict: approve / reject / needs_human_review
```

The core finding from the benchmark harness (`adjudicate/benchmark/`, all 13
curated cases — synthetic bug/fix pairs plus reused real-repo diffs — have
real, complete results): of every claim the Prosecutor raised and would have
reached a reviewer unverified, **50.0% (25/50) were refuted once actually
checked** by the Verifier. That's the number that justifies the Verifier
layer's existence — see `docs/state/PROGRESS.md`'s Phase 31 entry for the
full three-condition breakdown (single-agent baseline vs.
adversarial-without-verification vs. the full verified pipeline).

## The retrieval engine underneath it

- **Ingestion**: clone → discover files → Tree-sitter AST parsing (Python,
  JavaScript) → semantic chunking (function/class-level "parent" chunks plus
  sliding-window fallback for everything else).
- **Graph**: a NetworkX call/import/inheritance graph, persisted alongside
  SQLite (the source of truth for chunks/repos/metadata).
- **Hybrid retrieval**: dense (FAISS, CodeBERT/MiniLM embeddings) + sparse
  (BM25), combined via Reciprocal Rank Fusion, expanded outward along the
  call graph, then narrowed back down with a cross-encoder reranker.
- **Small-to-big**: retrieve at chunk granularity, expand to the enclosing
  parent (function/class/file) before handing context to the LLM.
- **Semantic cache**: embedding-similarity cache in front of the LLM call,
  gated by a lexical-compatibility check to avoid near-duplicate-but-wrong
  cache hits.
- **Evaluation**: a RAGAS-based harness plus a five-way ablation study
  (dense-only → +BM25 → +graph → +reranker → full pipeline) quantifying what
  each component actually contributes.

## Setup

```bash
git clone <this-repo>
cd RepoMind
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env   # then set GEMINI_API_KEY and/or GROQ_API_KEY
```

At least one hosted LLM provider key is required (`GEMINI_API_KEY` or
`GROQ_API_KEY`; `USE_GEMINI`/`USE_GROQ`/`USE_OLLAMA` in `.env` control
priority — Gemini is checked first, then Groq, then a local Ollama model).

### Run the API + frontend (primary UI)

```bash
uvicorn api.main:app --reload --port 8000       # backend
cd frontend && npm install && npm run dev        # frontend (Vite)
```

Open the Vite dev URL it prints, paste a GitHub repo URL to index it, then
use the Graph / Chat / "Review a PR" views.

### Or use the CLI directly (no frontend needed)

```bash
python cli.py index https://github.com/psf/requests
python cli.py query <repository_id> "How does the Session class work?"
```

### Run the benchmark harness

```bash
python -m adjudicate.benchmark.run      # 3-condition adversarial-review benchmark
```

The RAGAS/ablation evaluation harness (`evaluation/ablation.py`'s
`run_ablation()`, `evaluation/ragas_eval.py`) is a library, not a CLI entry
point — invoke it from a script or notebook against an indexed repository's
`PipelineComponents`.

### Run the tests

```bash
pytest
```

## Status

Core retrieval infrastructure (Phases 0–20) and the Adjudicate review
pipeline (Phases 21–32) are complete, including a full live end-to-end
verification of the review screen through the real browser UI. The
benchmark harness has real, complete results for all 13 curated cases.
Deployment (Phase 33) has not started — this has only run locally so far.
See `docs/roadmap.md` for the phase-by-phase status and
`docs/state/PROGRESS.md` for the detailed history behind each one.

## Structure

```
RepoMind/
├── core/            # Cross-cutting plumbing: constants, exceptions, logging
├── ingestion/       # GitHub URL → cloned repo → discovered files → AST chunks
├── graph/           # NetworkX call/import/inheritance graph + blast radius
├── embedding/       # CodeBERT/MiniLM embedding generation
├── database/        # SQLite (SQLAlchemy), FAISS, BM25, JSON graph store
├── retrieval/       # Dense/sparse/hybrid/graph-expansion/reranker/semantic cache
├── generation/      # Context builder (small-to-big) + LLM client + answer generator
├── prompts/         # Chat system/user prompt templates
├── models/          # Shared dataclasses (CodeChunk, RetrievedChunk, ContextDocument, ...)
├── evaluation/       # RAGAS, Precision@K, 5-way ablation, plots
├── api/              # FastAPI app — the sole HTTP orchestration layer
├── adjudicate/        # The adversarial multi-agent review subsystem
├── ui/                 # Legacy Streamlit dev UI (thin HTTP client over api/)
├── frontend/            # React + Vite SPA (the current/primary UI)
├── scripts/              # One-off demo scripts
├── tests/                 # Test suite, mirrors the top-level package structure
├── docs/                   # roadmap.md, state/PROGRESS.md, project_description.md
├── pipeline.py             # Pipeline — the single index/query/search entrypoint
├── cli.py                  # Thin argparse CLI over Pipeline
├── app.py                  # Legacy Streamlit entrypoint
└── config.py               # Settings — the single source of runtime config
```

## Development

See `CLAUDE.md` for coding rules and the required development workflow, and
`docs/project_description.md` for the full architecture reference (tech
stack, security/scalability trade-offs, and the reasoning behind every major
design decision).
