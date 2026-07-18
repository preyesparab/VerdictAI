# RepoMind + Adjudicate — Engineering Interview Reference

**Purpose of this document.** This is not a README. It is a complete internal
engineering reference intended to let someone who has never seen this
codebase — an interviewer, or another LLM preparing interview notes — ask
any technical question about it and get a grounded answer. Every claim below
is derived from reading the actual source, not from the roadmap's aspirational
description of what *should* exist. Where the code diverges from the plan, or
contains a known bug/limitation, that is called out explicitly, because those
are exactly the details a real engineering interview probes for.

**What this project actually is.** RepoMind is a hand-built GraphRAG
(Graph-augmented Retrieval-Augmented Generation) system for source-code
repositories: point it at a GitHub URL, it parses the code with Tree-sitter,
builds a call/import/inheritance graph, indexes it with dense (FAISS) and
sparse (BM25) retrieval fused by Reciprocal Rank Fusion, expands results
across the graph, reranks with a cross-encoder, and answers natural-language
questions about the repo with the LLM grounded in retrieved, cited code.
Built on top of it is **Adjudicate**, an adversarial multi-agent code-review
system: given a diff, a Prosecutor agent raises structured, falsifiable
claims about it, a deterministic (non-LLM) Verifier checks each claim against
real sandboxed test execution and static analyzers, a Defender agent rebuts
or concedes based on that verified evidence, and a Judge agent rules — with
every step streamed live to a React frontend over Server-Sent Events.

No LangChain, LlamaIndex, or Haystack is used anywhere. Every retrieval
component — the BM25 tokenizer, Reciprocal Rank Fusion, graph expansion,
semantic caching, small-to-big context assembly — is hand-implemented, which
is precisely the point: this project exists to demonstrate that these
techniques are understood at the implementation level, not just as
library calls.

**A note on this document's provenance.** An earlier version of this file
(same path, dated 2026-07-08) was a phase-by-phase build plan mirroring
`docs/roadmap.md`, written for whoever picks up development next, not for an
interview audience. That content still exists verbatim in
`docs/state/PROGRESS.md`'s history and in `docs/roadmap.md`'s phase list —
nothing about the plan is lost, it just isn't duplicated in prose form here.
This revision replaces it with an interview-oriented reference covering the
system as actually implemented today.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Complete Tech Stack](#2-complete-tech-stack)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Project Folder Structure](#4-project-folder-structure)
5. [Authentication & Authorization (or the honest lack thereof)](#5-authentication--authorization)
6. [Repository Indexing Workflow](#6-repository-indexing-workflow)
7. [AI Workflows: RAG Chat & Adversarial Code Review](#7-ai-workflows-rag-chat--adversarial-code-review)
8. [Database & Persistence Design](#8-database--persistence-design)
9. [Caching Architecture (and why there is no Redis)](#9-caching-architecture)
10. [Backend Optimizations](#10-backend-optimizations)
11. [Indexing & Query Optimization (FAISS / BM25 / SQLite)](#11-indexing--query-optimization)
12. [API Design](#12-api-design)
13. [Security](#13-security)
14. [Performance](#14-performance)
15. [Scalability](#15-scalability)
16. [Deployment](#16-deployment)
17. [Future Improvements](#17-future-improvements)
18. [Important Design Decisions](#18-important-design-decisions)
19. [Interview Questions](#19-interview-questions)
20. [Common Follow-up Questions](#20-common-follow-up-questions)

---

## 1. Project Overview

### 1.1 What problem this solves

Two related but distinct problems:

1. **"I don't understand this codebase."** Dropping a large, unfamiliar
   repository on an engineer and asking them to answer "how does auth work
   here" or "what calls this function" is slow. Plain-text search (grep) has
   no semantic understanding; a vanilla RAG-over-files system has no
   structural understanding either — it doesn't know that function A calls
   function B, or that a test only exercises a function indirectly through
   a mock. RepoMind indexes a repository at the *syntactic and structural*
   level (via Tree-sitter ASTs and a call graph), not just the *textual*
   level, and answers questions with citations grounded in real code.

2. **"Is this pull request actually safe to merge?"** A single LLM reviewing
   a diff will confidently assert things that are not true — "this branch is
   untested," "this could throw a null pointer" — with no mechanism to check
   whether the assertion is correct. Adjudicate's core idea is that an LLM's
   claims about code should be treated as *hypotheses*, not facts, and
   checked the same way a human reviewer would: run the actual test, run the
   actual static analyzer, look at the actual call graph — before an LLM
   (the Judge) ever renders a verdict on them.

### 1.2 Motivation

This is explicitly built as a demonstration project for ML/GenAI engineering
interviews (see `CLAUDE.md`). The goal is not "does it work" but "does it
demonstrate research-quality understanding of the individual RAG components
most interview loops probe": AST parsing, graph construction, hybrid
dense+sparse retrieval, cross-encoder reranking, semantic caching,
small-to-big retrieval, and rigorous evaluation (RAGAS + ablation studies).
Building every component by hand, rather than calling `VectorStoreIndex()`
from a framework, is the deliberate choice that makes this possible — see
[§18](#18-important-design-decisions) for the full rationale.

### 1.3 Target users

There is no external "end user" persona in the product sense — this is a
single-operator, locally-run tool (see [§5](#5-authentication--authorization)
for why there is no authentication). The realistic users are: (a) the
project author demonstrating it live in an interview, and (b) a developer
running it locally against their own repository to chat with it or get an
adversarial review of a diff before opening a real PR.

### 1.4 Key features

- Point at any public GitHub repo URL → clone, parse, chunk, graph, embed,
  and index it end to end.
- Chat with the repo in natural language, with every answer grounded in
  cited, retrievable source code (never hallucinated file paths).
- A rendered, interactive call/import/inheritance graph of the whole
  codebase, with blast-radius highlighting around any node.
- Submit a unified diff for adversarial multi-agent review: Prosecutor
  raises claims → a deterministic Verifier checks them against real
  sandboxed execution and static analysis → Defender rebuts → Judge rules —
  streamed live, not hidden behind a single "please wait" spinner.
- A five-way ablation study and RAGAS-based evaluation harness quantifying
  exactly what each retrieval component (dense-only → +BM25 → +graph →
  +reranker → full pipeline) contributes.
- A benchmark harness comparing three review strategies (single-agent
  baseline, adversarial-without-verification, full verified pipeline) on
  real bug-fix commits and injected mutations, producing a concrete
  claim-flip rate: **46.8% of Prosecutor claims that sound right are refuted
  once actually checked** — the single number that justifies the entire
  Verifier layer's existence.

### 1.5 High-level architecture (one diagram)

```
                       ┌─────────────────────────┐
                       │   React Frontend (Vite)  │
                       │  Chat | Graph | Review    │
                       └────────────┬─────────────┘
                                    │ fetch / SSE (HTTP, JSON)
                                    ▼
                       ┌─────────────────────────┐
                       │   FastAPI Backend         │
                       │   (api/main.py)           │
                       │   — the ONLY orchestration │
                       │      layer in the system   │
                       └────────────┬─────────────┘
                                    │ in-process Python calls
                       ┌────────────┴─────────────┐
                       ▼                            ▼
        ┌───────────────────────────┐  ┌───────────────────────────┐
        │   pipeline.Pipeline        │  │  adjudicate/ (agents,      │
        │  index_repository / query  │  │  verifier, orchestrator)   │
        │  / search                  │  │  built ON TOP of Pipeline  │
        └───────────┬────────────────┘  │  via HTTP (RepoMindClient) │
                    │                    └──────────────┬────────────┘
     ┌──────────────┼──────────────┬─────────────┐      │
     ▼              ▼              ▼             ▼      │
 Tree-sitter   NetworkX graph   FAISS (dense) BM25(sparse)
 parsing/chunk  builder                                  │
     │              │              │             │       │
     └──────────────┴──────┬───────┴─────────────┘       │
                            ▼                              │
                    ┌──────────────┐                       │
                    │   SQLite      │◄── source of truth    │
                    │ (SQLAlchemy)  │    for everything      │
                    └──────────────┘                        │
                            │                                │
                            ▼                                ▼
                    ┌──────────────────────────────────────────┐
                    │  LLM Client — Gemini → Groq → Ollama       │
                    │  (static priority, not retry-fallback)     │
                    └──────────────────────────────────────────┘
```

### 1.6 Why these technologies were selected

The short version, expanded per-technology in [§2](#2-complete-tech-stack)
and with full trade-off discussion in [§18](#18-important-design-decisions):
Python end-to-end because every retrieval/ML library needed (Tree-sitter
bindings, sentence-transformers, FAISS, NetworkX) is Python-native, and a
polyglot split (a Node orchestration layer calling out to a Python ML
service, as is common in other projects) would add a network hop and a
second language for zero benefit here — there is no reason to orchestrate in
JavaScript when the thing being orchestrated *is* Python. React + Vite for
the frontend because the live SSE review screen and the interactive graph
(vis-network) both need real client-side state and animation that a
server-rendered template can't give cheaply. SQLite because this is a
single-process, single-operator tool — there is no concurrent-writer
scenario Postgres/MySQL would need to solve, and SQLite's zero-ops,
single-file nature matches the "local tool" deployment story exactly.

---

## 2. Complete Tech Stack

### Frontend
- **React 19.2.7 + Vite 8.1.1** — component model + fast dev server/HMR.
  No React Router: the app has exactly three views (Graph / Chat / Review)
  toggled by local state, not URL-addressable routes, so a router would be
  pure overhead.
- **vis-network 10.1.0 + vis-data 8.0.4** — graph rendering. Swapped in from
  an earlier Cytoscape.js implementation; the codebase still carries the
  design rationale in `visStyle.js`'s comments (physics tuning validated
  against two real indexed repos — a 227-node repo and a 657-node repo that
  never stabilized under vis.js's defaults, requiring a size-scaled physics
  config).
- **Tailwind CSS 3.4.19** — utility-first styling, no separate CSS-in-JS
  runtime.
- **oxlint** — linting (a Rust-based ESLint-compatible linter), not ESLint
  itself.
- No TypeScript. No state-management library (no Redux/Zustand/MobX) — a
  single React Context (`RepoContext.jsx`) is sufficient given the app's
  small, mostly-flat state shape.

### Backend
- **FastAPI 0.115+ / uvicorn[standard] 0.30+** — the single HTTP
  orchestration layer for the entire system (`api/main.py`). There is no
  separate Node.js/Express layer — see [§3](#3-high-level-architecture) for
  why that's a deliberate departure from a common polyglot pattern.
- **httpx 0.27+** — used both as FastAPI's test-friendly HTTP client
  dependency and as the transport for `adjudicate.repomind_client.RepoMindClient`
  and `ui.api_client.APIClient` (Streamlit) and `frontend/src/api/client.js`
  (fetch-based, the JS equivalent).

### AI / ML Layer
- **Tree-sitter 0.23 + tree-sitter-python + tree-sitter-javascript** — real
  AST parsing (not regex-based heuristics) for chunk extraction and call/
  import graph construction. Only these two grammars are wired up; see
  [§6](#6-repository-indexing-workflow) for the documented TypeScript gap.
- **sentence-transformers 3.0+ / torch 2.6+** — embedding generation.
  `microsoft/codebert-base` (768-dim, code-pretrained) by default, with
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim, faster, general-purpose)
  as a config-toggleable alternative (`USE_CODEBERT`).
- **faiss-cpu 1.8+** — dense vector similarity search (`IndexFlatIP`, exact
  brute-force cosine via L2-normalized inner product).
- **rank-bm25 0.2+** — sparse lexical retrieval (`BM25Okapi`), paired with a
  fully hand-written camelCase/snake_case-aware tokenizer.
- **cross-encoder/ms-marco-MiniLM-L-6-v2** (via `sentence-transformers.CrossEncoder`)
  — joint query+candidate reranking.
- **google-genai 1.0+** (Gemini, default provider), **groq 0.11+** (hosted
  fallback for Gemini free-tier quota exhaustion), **ollama 0.3+** (local
  fallback, not currently the active provider).
- **NetworkX 3.2+** — the call/import/inheritance graph, an in-memory
  `nx.DiGraph`.
- **ragas** (intentionally *not* pinned in `requirements.txt` — see
  [§7](#7-ai-workflows-rag-chat--adversarial-code-review)) — generation
  quality metrics (faithfulness, answer relevancy, context precision/recall).
- **bandit, mypy** — static analyzers driven by Adjudicate's Verifier for
  `SECURITY` and `TYPE_MISMATCH` claim types respectively.

### Database / Persistence
- **SQLite (via SQLAlchemy 2.0+ ORM)** — the single relational source of
  truth: repositories, files, chunks, graph edges (chunk↔chunk only —
  see the documented divergence in [§8](#8-database--persistence-design)),
  embeddings (raw float32 blobs), and the semantic cache.
- **FAISS binary index + JSON sidecar**, one pair per repository, on disk
  under `data/indexes/`.
- **BM25 pickle**, one per repository, alongside the FAISS files.
- **A separate flat JSON graph store** (`database/graph_store.py`,
  `nx.node_link_data`) — the *lossless* full graph (including file nodes and
  import edges), independent of and richer than what SQLite persists.

### Caching
- **A hand-built semantic cache**, persisted in the same SQLite database
  (`semantic_cache` table) — not Redis, not an in-memory LRU. See
  [§9](#9-caching-architecture) for the full design, including a real,
  documented over-matching bug and its fix.

### Authentication
- **None.** See [§5](#5-authentication--authorization) — this is deliberate
  and explained, not an oversight.

### Deployment
- **Docker + docker-compose scaffolds exist but are stale** (still launch
  the legacy Streamlit UI, not the FastAPI+React stack) — see
  [§16](#16-deployment). No CI/CD pipeline exists yet. Real deployment
  (Vercel + Render, per the roadmap) is Phase 33, not started.

### Libraries (notable, not otherwise categorized)
- `pydantic 2.6+` / `pydantic-settings 2.2+` — typed, validated, `.env`-driven
  configuration (`config.Settings`, `adjudicate.config.AdjudicateSettings`).
- `GitPython 3.1+` — the *only* module permitted to import it is
  `ingestion/git_client.py` (an explicit, enforced architectural boundary).
- `streamlit 1.38+` / `streamlit-agraph` — the legacy dev UI, superseded by
  but still coexisting with the React frontend.
- `pytest 8.0+` / `pytest-mock 3.14+` — the test suite (693 tests currently
  passing).

### State Management
React Context only (frontend). No server-side session state beyond an
in-memory indexing-status registry in `api/main.py` (a real scalability
limitation — see [§15](#15-scalability)).

### External APIs
- **GitHub** — via `git clone`/`git fetch`/`git pull` (not the GitHub REST
  API — repository acquisition is pure git, no API token required, which
  also means it only works against public repos).
- **Google Gemini API**, **Groq API** — hosted LLM inference.
- **Ollama** — local LLM inference (optional, disabled by default).

---

## 3. High-Level Architecture

### 3.1 The chain, and why it is NOT a Node→Python microservice split

A common pattern in similar full-stack AI projects is: browser → Node/Express
(orchestration + auth + business logic) → a separate Python/FastAPI
"AI service" (ML-heavy work) → an LLM provider. **RepoMind deliberately does
not do this.** There is exactly one backend process, `api/main.py`
(FastAPI), and it *is* the orchestration layer *and* the ML layer at once.

The reasoning, stated plainly: a Node orchestration layer earns its keep
when the orchestration logic (auth, request shaping, rate limiting, business
rules) is meaningfully separate from the ML work, and especially when the ML
work is CPU/GPU-bound and benefits from being isolated in its own scalable
service. Here, every single piece of "business logic" — validate a GitHub
URL, run the retrieval pipeline, build context, call an LLM — **is** the ML
work. There is no separate concern for a second language/runtime to own.
Splitting them would add a network hop, a second dependency-management
surface, and a second deployment target, for a system that has no auth,
no multi-tenant business rules, and no reason to scale the "orchestration"
and "ML" tiers independently. `api/main.py`'s FastAPI process plays the role
both the Express layer and the FastAPI layer play in a polyglot design,
simultaneously.

### 3.2 Full request flow: chat

```
Browser (React)
  │  POST /repos/{id}/query {question}
  ▼
FastAPI (api/main.py)
  │  _require_ready(repo_id)  → 404/409 gate
  │  run_in_threadpool(Pipeline.query, timeout=120s)
  ▼
pipeline.Pipeline.query()
  │
  ├─ 1. SemanticCacheManager.lookup()        ── cache HIT → skip to step 5
  │
  ├─ 2. Pipeline._retrieve_and_rank()
  │     ├─ embed query (CodeBERT/MiniLM)
  │     ├─ HybridRetriever.retrieve()  (FAISS + BM25, fused via RRF)
  │     ├─ GraphExpander.expand()      (1-hop graph neighbors, score decay)
  │     └─ CrossEncoderReranker.rerank() (top ~40 → top 8)
  │
  ├─ 3. ContextBuilder.build_context()  (small-to-big substitution,
  │                                       dedup, ~4000-token budget)
  │
  ├─ 4. LLMService.generate_answer()    (Gemini/Groq/Ollama, cite-grounded)
  │     └─ writes result back into the semantic cache
  │
  └─ 5. AskResult{answer, citations[], cache_hit, latency_ms, ...}
  ▼
FastAPI → JSON → React renders answer + clickable citation chips
  (a citation resolves to a graph node → clicking it jumps to Graph view
   with blast-radius highlight; a small-to-big-substituted PARENT/SLIDING
   chunk is never a graph node and renders as a disabled chip)
```

### 3.3 Full request flow: repository indexing

```
Browser              POST /repos/index {repo_url}
  │                        │
  ▼                        ▼
FastAPI  ── computes repo_id up front (deterministic UUID5) ──► 202 Accepted
  │                                                    {repo_id, status:"pending"}
  │  (BackgroundTasks.add_task — fire-and-forget, in-process)
  ▼
Pipeline.index_repository(url, on_progress=...)
  1. RepositoryManager.get_repository(url)     — clone/update via git
  2. FileDiscovery.discover(local_path)        — walk, filter, language-tag
  3. TreeSitterParser.parse() per file         — AST → chunks (strict: any
                                                  syntax error drops the file)
  4. SemanticChunker.build_chunks()            — + sliding-window + parent
                                                  chunks
  5. DatabaseManager.store_*()                 — persist to SQLite
  6. RepositoryGraphBuilder.build_graph()      — NetworkX call/import graph
                                                  → graph_store.save_graph()
                                                    (JSON, lossless)
                                                  → db.store_graph()
                                                    (SQLite, lossy — chunk-
                                                    only edges)
  7. EmbeddingManager.generate_embeddings()    — batched, dedup by chunk_id
  8. FaissIndexManager / BM25Manager           — build + persist both indexes
  ▼
Browser polls GET /repos/{id}/status every 1s until status == "ready"
  (both the React `useIndexingStatus` hook and the Streamlit `APIClient`
   independently reimplement this identical 1-second poll)
```

### 3.4 Full request flow: adversarial review (Adjudicate)

```
Browser (ReviewView)     POST /repos/{id}/review {diff}   (SSE stream)
  │                             │
  ▼                             ▼
FastAPI  StreamingResponse(run_live_review(...), media_type="text/event-stream")
  │
  ▼
adjudicate.orchestrator.live_review.run_live_review()  — a Python generator,
  each yielded dict becomes one `data: {...}\n\n` SSE frame:

  1. event "context"   ← AdjudicateContextBuilder.build(diff)
                          (graph-first: enclosing chunk lookup + 1-hop
                           blast radius for callers/callees/related tests;
                           retrieval fallback ONLY if zero test-like callers
                           found via the graph)
  2. event "defender"  ← DefenderAgent.draft_justification(bundle, diff)
  3. event "claims"    ← ProsecutorAgent.raise_concerns(bundle, diff, ...)
                          (structured JSON, schema-validated, retried up to
                           3x on malformed output)
  4. event "claim_verified" × N
                        ← verify_claim() per claim — ZERO LLM CALLS, ever:
                          sandboxed test execution / bandit / mypy /
                          existing test suite, per claim_type
  5. event "rebuttal" × 0-2
                        ← run_rebuttal_loop() — Defender concedes CONFIRMED
                           claims, rebuts INCONCLUSIVE ones with evidence
                           from the bundle, capped at 3 rounds total
  6. event "judge"     ← JudgeAgent.judge(...) — verdict/confidence/cited
                           evidence, reading ONLY verified results, never
                           raw unverified claims
  7. event "done"
  ▼
React ReviewView — append-only card feed, each event updates one component
  (ContextCard, DefenderCard, ClaimCard×N updating "Verifying…" → CONFIRMED/
   REFUTED/INCONCLUSIVE in place, RebuttalCard×N, JudgeCard pinned above
   the feed once the verdict lands)
```

### 3.5 Responsibilities of every major component

| Component | Responsibility |
|---|---|
| `api/main.py` | The single HTTP boundary. Owns request validation, status-code mapping from domain exceptions, the in-memory indexing-status registry, SSE streaming, and CORS. |
| `pipeline.Pipeline` | The single call-order authority for indexing and querying. Nothing outside it decides *what order* retrieval stages run in. |
| `ingestion/` | Everything between "a GitHub URL" and "a list of `SourceFile`s and `CodeChunk`s." |
| `graph/` | Builds and queries the NetworkX call/import/inheritance graph; computes blast radius. |
| `database/` | All persistence: SQLite (relational), FAISS (dense vectors), BM25 (sparse index), and the separate lossless JSON graph store. |
| `retrieval/` | Dense (via `database.vector_store`), sparse (BM25), fusion (RRF), graph expansion, reranking, semantic cache. |
| `generation/` | Context assembly (small-to-big) and LLM invocation/citation-matching. |
| `adjudicate/` | The entire adversarial review subsystem — agents, deterministic verifier, rebuttal orchestration, benchmark harness. Depends on RepoMind (via HTTP, through `RepoMindClient`) but RepoMind never depends on it — a one-way dependency enforced as a design invariant. |
| `frontend/` | The React SPA — three views (Graph/Chat/Review), talking to `api/main.py` exclusively over HTTP/SSE. |
| `ui/` | The legacy Streamlit dev UI, kept alive alongside the React frontend, calling the identical FastAPI backend. |
| `evaluation/` | RAGAS scoring, LLM-as-judge Precision@K, the five-way ablation study, plotting. |

---

## 4. Project Folder Structure

```
RepoMind/
├── core/                # Cross-cutting plumbing: constants, exceptions, logging
├── ingestion/            # GitHub URL → cloned repo → discovered files → AST chunks
├── graph/                # NetworkX call/import/inheritance graph + blast radius
├── embedding/            # CodeBERT/MiniLM embedding generation
├── database/             # SQLite (SQLAlchemy), FAISS, BM25, JSON graph store
├── retrieval/             # Dense/sparse/hybrid/graph-expansion/reranker/semantic cache
├── generation/            # Context builder (small-to-big) + LLM client + answer generator
├── prompts/               # Chat system/user prompt templates
├── models/                # Shared dataclasses (CodeChunk, RetrievedChunk, ContextDocument, ...)
├── evaluation/             # RAGAS, Precision@K, 5-way ablation, plots
├── api/                   # FastAPI app — the sole HTTP orchestration layer
├── adjudicate/             # The adversarial multi-agent review subsystem (see below)
├── ui/                     # Legacy Streamlit dev UI (thin HTTP client over api/)
├── frontend/                # React + Vite SPA (the current/primary UI)
├── scripts/                 # One-off demo scripts
├── tests/                   # 693 tests, mirrors the top-level package structure
├── docs/                    # roadmap.md, PROGRESS.md, this file
├── pipeline.py              # Pipeline — the single index/query/search entrypoint
├── cli.py                   # Thin argparse CLI over Pipeline
├── app.py                   # Streamlit entrypoint
└── config.py                # Settings — the single source of runtime config
```

### `adjudicate/` in detail (the most structurally complex package)

```
adjudicate/
├── config.py               # AdjudicateSettings — per-role Gemini model override
├── schemas.py               # ProsecutorClaim / JudgeVerdict + LLM-boundary pydantic mirrors
├── repomind_client.py        # RepoMindClient — the ONLY way Adjudicate touches RepoMind (HTTP)
├── context_builder.py         # parse_diff() + AdjudicateContextBuilder.build() → ContextBundle
├── agents/
│   ├── base.py                # BaseAgent ABC (deliberately unused by every concrete agent — see §18)
│   ├── defender.py             # draft_justification() + rebut()
│   ├── prosecutor.py            # raise_concerns() — structured claims, retry-on-malformed
│   ├── judge.py                  # judge() — verdict, mechanically-checked cap-handling rule
│   └── documentation.py           # STUB ONLY — Phase 35, not implemented
├── verifier/
│   ├── models.py               # VerificationStatus / Confidence / VerificationResult
│   ├── sandbox.py                # run_sandboxed() — subprocess + timeout + best-effort net-deny
│   ├── strategies.py              # per-claim-type verification — ZERO LLM CALLS, ever
│   └── verifier.py                 # verify_claim() — the single dispatch entry point
├── orchestrator/
│   ├── rebuttal_loop.py           # run_rebuttal_loop() — 3-round-capped Defender/Verifier loop
│   └── live_review.py              # run_live_review() — the full pipeline as an SSE event generator
└── benchmark/
    ├── baseline_reviewer.py        # condition (a): single unstructured LLM call, no adversarial structure
    ├── harness.py                    # runs all 3 conditions per case in an isolated sandbox
    ├── metrics.py                    # catch rate, false-positive rate, claim-flip rate
    ├── token_tracking.py              # wraps LLMClient to tally calls/tokens per condition
    └── run.py                          # entrypoint: python -m adjudicate.benchmark.run
```

### `frontend/src/` in detail

```
frontend/src/
├── App.jsx                 # <RepoProvider><AppShell/></RepoProvider> — no router
├── api/client.js             # fetch-based HTTP client; streamReview() does manual SSE parsing
├── state/RepoContext.jsx      # the ONLY state store — plain React Context, no reducer library
├── hooks/
│   ├── useIndexingStatus.js    # POST /repos/index then poll /status every 1000ms
│   ├── useChat.js                # request/response only — /query has no streaming endpoint
│   ├── useGraph.js                # fetch graph once per repoId, share via context
│   └── useLiveReview.js            # drives streamReview(), real SSE, NOT native EventSource
└── components/
    ├── Layout/     # AppShell (crossfade landing↔workspace), ViewSwitcher, LandingView
    ├── Sidebar/     # RepoInputPanel, IndexingProgress
    ├── Chat/         # ChatView, ChatHistoryPanel, ChatMessage, CitationItem, ChatInputBar
    ├── Graph/         # GraphView (vis-network), NodeTooltip, blastRadius.js, visStyle.js
    └── Review/         # DiffInputPanel, ContextCard, DefenderCard, ClaimCard, RebuttalCard, JudgeCard
```

---

## 5. Authentication & Authorization

**There is none.** No login, no signup, no JWTs, no sessions, no API keys
gating access to the FastAPI backend itself. This is a deliberate scope
decision, not an oversight, and it is worth being direct about in an
interview: a locally-run, single-operator tool with no multi-tenant data has
no *user* to authenticate against — every repository indexed is visible to
whoever can reach the FastAPI process, and that process is assumed to run on
`localhost` (the CORS allowlist defaults to exactly `http://localhost:5173`
and `http://127.0.0.1:5173`, Vite's dev-server origins).

**What credentials the system *does* manage**: LLM provider API keys
(`GEMINI_API_KEY`, `GROQ_API_KEY`), loaded via `pydantic-settings` from a
`.env` file, validated at process startup (fail fast on a malformed value),
and never logged or echoed back in any response — that is the actual
security-sensitive surface in this system, and it's handled the standard
way (environment variables, `.gitignore`d `.env`, no secrets in source).

**What would be needed to productionize this** (a fair interview follow-up,
answered honestly rather than deflected): per-user repository isolation
(the SQLite schema already scopes every table by `repository_id`, which
would need to additionally scope by `user_id`), a real auth layer (JWT or
session-based, likely reintroducing something like the Express-orchestration
pattern this project deliberately avoided at its current scale — see
[§3.1](#31-the-chain-and-why-it-is-not-a-nodepython-microservice-split)),
rate limiting per user (currently there is none at all — see
[§13](#13-security)), and moving the Verifier's sandbox from process-level
isolation to a real container boundary before ever letting untrusted
third-party diffs be submitted by the public (see [§13](#13-security) and
[§17](#17-future-improvements)).

---

## 6. Repository Indexing Workflow

Step by step, exactly as implemented in `pipeline.Pipeline.index_repository()`:

### Step 1 — URL validation and cloning (`ingestion/`)
`ingestion.validators.validate_github_url()` enforces a strict regex:
`https://github.com/<owner>/<repo>[.git][/]` only — GitLab, Bitbucket, SSH
remotes, and malformed URLs are all rejected before `git` is ever invoked.
`ingestion.repository_manager.RepositoryManager` is the *sole* permitted
caller of `ingestion.git_client.GitClient` (the only module allowed to
import GitPython at all — an enforced boundary). Repos are cached at a
deterministic local path (`data/repositories/{owner}_{repo}/`); a
re-submission fetches and only pulls if new commits actually exist — it
never re-clones or deletes an existing checkout.

### Step 2 — File discovery (`ingestion/file_discovery.py`)
A pruned `os.walk` skips `.git`, `node_modules`, `dist`, `build`,
`__pycache__`, `venv`, `.idea`, `.vscode`, etc. Files over 2 MB are skipped.
Binary files are detected the same way `git` itself does — a NUL byte
anywhere in the first 8 KB. Every surviving file becomes a `SourceFile`
(path, language, size).

### Step 3 — Tree-sitter AST parsing (`ingestion/ast_parser.py`)
Only **Python** and **JavaScript** grammars are actually registered, despite
`core/constants.py` listing 11 "supported" languages (TypeScript, Java, Go,
Rust, C/C++, C#, Ruby, PHP are *discovered* but not *parsed* — they produce
zero AST chunks and zero call-graph edges, a documented, deliberate gap; see
[§17](#17-future-improvements)). Parsing is **strict**: a single syntax
error anywhere in a file drops *that entire file's* chunks (not just the
broken region) — `parse_many()` catches this per-file, logs it, and
continues rather than aborting the whole indexing run.

The extractor logic has real, documented edge-case handling worth knowing
in depth for an interview:
- **Python**: functions/async functions/classes/methods, decorators folded
  into the chunk span (`@app.route(...)` is captured as part of the chunk),
  base classes extracted from `superclasses`.
- **JavaScript**: standard `const f = () => {}` naming, **plus** two
  real-world patterns added after a diagnostic against a live-indexed repo
  found they produced *zero* chunks under the naive check: (1) CommonJS
  handler exports (`exports.foo = async (req, res) => {...}`, used
  throughout Express controllers), and (2) Zustand-style factory calls
  (`const useX = create(persist((set, get) => ({...}), opts))`) — matched by
  walking up through nested `arguments → call_expression` hops as long as
  the callee at every hop is a bare identifier and the arrow body is an
  object literal, deliberately excluding ordinary callbacks like
  `items.map((i) => ({...}))` (member-expression callee) and
  `useCallback((e) => {...}, [dep])` (block-bodied, not object-literal).

### Step 4 — Semantic chunking (`ingestion/chunker.py`)
Three chunk families, all sharing one `CodeChunk` dataclass:
1. **AST chunks** (from step 3) — one function/class = one chunk. Only
   these are ever embedded or become graph nodes.
2. **Sliding-window chunks** — 50 lines, 20% overlap (10-line step of 40),
   covering every file uniformly regardless of AST structure, so nothing is
   unreachable by retrieval even in odd/unparseable file regions.
3. **Parent chunks** — a ~50-line window around each AST chunk, used later
   for small-to-big context expansion at generation time. The window
   algorithm shifts overflow to the far side near file boundaries, so a
   function near line 1 still gets close to the full target parent size
   rather than shrinking, and a parent is *never* smaller than its own child.

Chunk IDs are **deterministic**: `UUID5(namespace, f"{file_id}:{name}:{start_line}")`.
This is the mechanism that makes re-indexing idempotent — an unchanged
function gets the identical ID on every re-index run, so
`EmbeddingManager.generate_embeddings()` can skip re-embedding it entirely
(see [§10](#10-backend-optimizations)).

### Step 5 — Persistence (`database/sqlite_client.py`)
Repository, files, and chunks are stored via SQLAlchemy. A real, documented
ordering constraint: **parent chunks are flushed to the DB before AST/sliding
chunks**, because an AST chunk's `parent_chunk_id` foreign key must already
exist — `Session.add()` alone doesn't guarantee insert ordering without an
explicit `relationship()`, so the code flushes explicitly between the two
groups.

### Step 6 — Graph construction (`graph/graph_builder.py`)
`RepositoryGraphBuilder.build_graph()` builds one `nx.DiGraph`: file nodes +
chunk nodes, edges for `function_call`, `method_call`, `inherits`,
`imports`, `contains`, and `references` (the last added specifically to
capture Express-style module-scope route wiring — `router.post("/x", handler)`
— which lives outside any function body and is otherwise structurally
invisible). Call/reference resolution is **pure name-based heuristic
matching, with no type inference** — this is explicitly documented as a
known limitation, not hidden: two unrelated classes each with a `save()`
method can resolve to the wrong target if referenced unqualified. Nothing
about graph construction ever raises on an unresolved reference — it's
logged and skipped, so partial resolution never blocks indexing (contrast
with step 3's strict parsing — a deliberate asymmetry: structural extraction
is best-effort, syntactic correctness is not).

The resulting graph is persisted **twice, to two different, non-interchangeable
stores** — this divergence is a genuine, documented architectural subtlety
covered in full in [§8](#8-database--persistence-design).

### Step 7 — Embedding generation (`embedding/embedding_manager.py`)
Only AST chunks are embedded (never sliding/parent chunks). Already-embedded
chunk IDs are skipped by default (`force=False`) — since chunk IDs are
deterministic, an unchanged function across re-indexing runs is never
re-embedded. Batched at 32 chunks/call by default, each batch persisted
immediately (not one giant transaction for the whole repo).

### Step 8 — Building the FAISS and BM25 indexes (`database/vector_store.py`, `retrieval/sparse_retriever.py`)
FAISS: vectors L2-normalized, added to a flat `IndexFlatIP` (inner-product
index on normalized vectors == cosine similarity — see
[§11](#11-indexing--query-optimization)). BM25: a custom tokenizer
(camelCase/snake_case-aware — `fetch_user_by_id` and `fetchUserByID` both
tokenize to `["fetch","user","by","id"]`) feeds `rank_bm25.BM25Okapi`. Both
are persisted per-repository (`{owner}_{repo}.faiss`/`.json` sidecar,
`{owner}_{repo}_bm25.pkl`).

Progress through all 8 stages is reported to the frontend via a callback
(`on_progress(stage, status)`) that both the React `useIndexingStatus` hook
and the Streamlit `APIClient` poll for once per second via
`GET /repos/{id}/status`.

---

## 7. AI Workflows: RAG Chat & Adversarial Code Review

RepoMind has two distinct LLM-driven workflows. They share the LLM client
and provider-priority logic but are otherwise structurally independent.

### 7.1 RAG chat generation

**Frontend → FastAPI → Pipeline → LLM, no separate AI microservice.**
Unlike systems that route "chat" through a distinct AI service, everything
here happens inside the one FastAPI process (see
[§3.1](#31-the-chain-and-why-it-is-not-a-nodepython-microservice-split)).
The exact call sequence is documented in full in
[§3.2](#32-full-request-flow-chat) — the short version: semantic cache
lookup (skip everything below on a hit) → hybrid retrieval (dense+sparse
fused by RRF) → graph expansion (1-hop) → cross-encoder reranking (top ~40
→ top 8) → small-to-big context assembly (~4000-token budget) → LLM
generation with a fixed, cite-or-refuse system prompt.

The system prompt (`prompts/templates.py::SYSTEM_PROMPT`) enforces four
rules mechanically reinforced elsewhere in the pipeline, not just asked for
in prose: answer only from supplied context, never invent file paths or
function names, cite every claim in a fixed
`(source: \`<file_path>\`, function \`<name>\`)` format, and explicitly say
so if context is insufficient rather than guessing. **If retrieval returns
nothing** (`context_document.context == ""`), the LLM is never even called —
`LLMService.generate_answer()` short-circuits to a fixed
"the repository does not contain enough information" message with zero
token cost, and nothing is written to cache (no generation occurred).

**Citation matching is deliberately loose, not a strict parser**:
`match_citations()` considers a citation "present" if the model's raw answer
text contains the citation's file path as a plain substring (and function or
class name), rather than parsing the exact requested format — a robustness
choice, since models don't always reproduce punctuation/formatting exactly.

### 7.2 Provider selection: static priority, not retry-on-failure

`generation/llm_client.py::LLMClient` picks a provider by a **fixed
config-driven priority**, checked once per call, not a runtime
fallback-on-error chain:

```
if USE_GEMINI:  Gemini (gemini-2.5-flash by default)
elif USE_GROQ:  Groq (llama-3.3-70b-versatile) — hosted fallback for
                Gemini free-tier quota exhaustion
elif USE_OLLAMA: Ollama (local, mistral)
else: raise LLMGenerationError
```

A real, documented bug and its fix are worth knowing cold for this project:
Gemini 2.5's `max_output_tokens` budget is **shared with the model's
internal "thinking" tokens**. A real reproduction against a schema-constrained
Prosecutor call spent 979 of a 1024-token budget on invisible thinking,
truncating the visible JSON after only 28 tokens. The fix has two parts:
(1) `thinking_config=ThinkingConfig(thinking_budget=0)` is set **only** when
a `response_schema` is supplied (schema-constrained extraction doesn't need
chain-of-thought; free-text completions like the Defender's justification
are unaffected), and (2) `LLM_MAX_TOKENS` was raised from 1024 to 4096 as
additional headroom. The Judge agent separately overrides this again to
2048 tokens for its own calls, having independently reproduced the same
truncation against a long rebuttal transcript.

### 7.3 Adversarial code review (Adjudicate) — the project's core novel contribution

Given a unified diff, `adjudicate.orchestrator.live_review.run_live_review()`
drives six stages, each streamed to the frontend as a distinct SSE event
(full flow diagram in
[§3.4](#34-full-request-flow-adversarial-review-adjudicate)). The engineering
substance is concentrated in three places:

**1. Context building is graph-first, retrieval is only a fallback.**
`AdjudicateContextBuilder.build()` parses the diff (`parse_diff()` — with two
documented, real bug-shaped edge cases fixed: multi-hunk cumulative
line-number drift between the diff's new-file numbering and the
already-indexed old-file numbering, and default 3-line git-diff context
spilling a hunk's reported range into an adjacent function), finds the
enclosing AST chunk for each changed location, and walks the call graph
1-hop via `compute_blast_radius()` to find real callers/callees. **Hybrid
retrieval is invoked only when zero test-like callers were found via the
graph for a given changed function** — e.g. a test that references behavior
only through a string-based mock (`@patch('colorama.initialise.reset_all')`),
which is genuinely invisible to a static call graph. This gating is
deliberate: an isolated node with zero real callers should stay isolated in
the output (the graph telling the truth), not be papered over with
retrieval-style fuzzy matches.

**2. The Verifier makes zero LLM calls, by hard rule.** This is the single
most interview-relevant design decision in the whole project. Prosecutor
claims carry a `claim_type` from a fixed enum
(`missing_null_check`, `untested_branch`, `type_mismatch`, `exception_handling`,
`breaking_change`, `security`), and each type dispatches to a **deterministic**
strategy:

| Claim type | Verification mechanism | Confidence |
|---|---|---|
| `untested_branch`, `missing_null_check` | Run the Prosecutor's own `proposed_test` in a real sandbox; nonzero exit = the claim is CONFIRMED | HIGH |
| `type_mismatch` | `mypy` scoped to the file, match findings within ±2 lines of the claimed location | MEDIUM |
| `security` | `bandit` scoped to the file, same ±2-line matching | MEDIUM |
| `breaking_change` | Run the existing test suite now (not a true before/after diff — a documented, honest limitation) | MEDIUM |
| `exception_handling` | Falls back to the proposed-test strategy if one exists, else INCONCLUSIVE — neither bandit nor mypy is a genuine fit, and forcing one would produce a confident-looking but meaningless result | HIGH or LOW |

Confidence tiers are **evidence-based, not outcome-based**: a claim REFUTED
by a real failing test is exactly as HIGH-confidence as one CONFIRMED by a
real passing test. Any claim type with no registered strategy, or a
`breaking_change` claim with no supplied `test_command`, resolves to
INCONCLUSIVE/LOW — it never silently guesses.

Sandboxing itself (`verifier/sandbox.py`) is honestly scoped: real
subprocess execution with a hard wall-clock timeout and a monkeypatched
network-denial guard, explicitly documented as **process-level, not a hard
container security boundary** — "a determined subprocess could still reach
the network through lower-level mechanisms." Docker-based isolation is
flagged as a real follow-up, not claimed as already solved.

**3. Rebuttal and judgment read verified evidence, never raw claims.**
`run_rebuttal_loop()` feeds the Verifier's actual results back to the
Defender — CONFIRMED claims must be conceded, not argued with; INCONCLUSIVE
claims may be countered only with real evidence from the context bundle,
never invented — capped at 3 total rounds, and whether it ended by genuine
resolution (every claim CONFIRMED or REFUTED) or by hitting the round cap is
logged and passed to the Judge. The Judge reads the full transcript plus
verified evidence *only*, and one specific rule is **mechanically enforced,
not just prompted**: if the loop ended by hitting the round cap with claims
still unresolved, the Judge cannot render high-confidence approval without
either a `minority_report` or a confidence below 0.7 — checked in code
(`_cap_handling_violation`) and retried with a corrective prompt if violated.
Other rules (e.g. "never approve if a CONFIRMED claim wasn't conceded") are
prompt-only, because verifying that a concession genuinely happened in prose
is exactly the kind of holistic judgment an LLM, not a text-matching
heuristic, is suited for.

### 7.4 Evaluation: RAGAS + the five-way ablation study

`evaluation/ablation.py` runs the *same* dataset of evaluation cases through
five progressively richer retrieval configurations — Dense Only → +BM25 →
+Graph Expansion → +Cross-Encoder → Full Pipeline — measuring RAGAS
faithfulness/answer relevancy/context precision, LLM-judge Precision@5, and
average end-to-end latency for each, so the marginal contribution of every
component is a concrete number rather than an assumption.
`USE_SEMANTIC_CACHE` is deliberately excluded from the ablation dimensions —
a cache hit would silently reuse a *previous configuration's* answer,
corrupting per-configuration metrics.

`ragas` itself is **intentionally not pinned in `requirements.txt`** — it's
imported lazily, only inside the one function that needs it, because at
implementation time the latest release failed to import at all against a
`langchain-community` module it depended on that had since been removed, and
the last known-compatible older release requires downgrading
`langchain`/`openai` to versions that conflict with the rest of the
environment. Every other evaluation module stays fully importable and
unit-testable without `ragas` installed at all — a deliberate,
documented isolation of an unstable third-party dependency.

---

## 8. Database & Persistence Design

### 8.1 Why SQLite, and the ID scheme that makes it work across repos

There is exactly one relational database, one file, no server process. Every
table is explicitly scoped by `repository_id`, because chunk/file IDs are
`UUID5`-derived purely from a repo-relative path — **not globally unique
across different repositories** (two different repos both containing
`src/main.py` hash identically). Three separate, deliberately independent
fixed UUID5 namespaces exist across the codebase (`ingestion/deterministic_ids.py`
for file/chunk IDs, and a third, module-local one in `sqlite_client.py`
for `repository_id` itself) — explicitly chosen as fixed custom namespaces
rather than a standard one, to avoid accidental collision with UUIDs
generated elsewhere in the system.

### 8.2 Schema (every table)

| Table | Primary Key | Notable columns | Constraints |
|---|---|---|---|
| `repositories` | `repository_id` (UUID5 of owner+repo) | owner, repository_name, clone_url, local_path, default_branch, current_commit_hash, indexed_at | `UNIQUE(owner, repository_name)` — re-indexing resolves to the same row |
| `source_files` | composite `(repository_id, file_id)` | relative_path, language, extension, size_bytes | FK → repositories |
| `code_chunks` | composite `(repository_id, chunk_id)` | file_id, parent_chunk_id (self-FK), chunk_type, function_name, class_name, parent_class, start_line, end_line, raw_code | FK → source_files; self-FK → code_chunks (small-to-big link) |
| `graph_edges` | surrogate `edge_id` | source_chunk_id, target_chunk_id, edge_type | `UNIQUE(repository_id, source_chunk_id, target_chunk_id, edge_type)`; **both FKs point only at code_chunks** (see §8.3) |
| `embeddings` | surrogate `embedding_id` | chunk_id, model_name, embedding_dimension, embedding_blob (raw float32 bytes), created_at | `UNIQUE(repository_id, chunk_id, model_name)` |
| `semantic_cache` | surrogate `cache_id` | query, query_embedding (float32 blob), response, retrieved_chunk_ids (JSON list), created_at | `UNIQUE(repository_id, query)` — exact re-cache overwrites |

Embeddings and cached query vectors are stored as **raw little-endian
float32 bytes** (`ndarray.tobytes()`), not JSON or pickle — compact, and
trivially reconstructed with `numpy.frombuffer(blob, dtype=np.float32)`.
SQLite's foreign-key enforcement is off by default; a `"connect"` event
listener runs `PRAGMA foreign_keys=ON` on every new connection to actually
enforce every FK declared above — easy to miss, and a real thing worth
knowing SQLite requires explicitly.

### 8.3 The graph_store vs. sqlite_client divergence — a genuine architectural subtlety

This is the single most interview-worthy "real bug shape" fact in the
persistence layer, and it's worth understanding precisely rather than
glossing over. There are **two independently-named `save_graph`/`load_graph`
pairs** with different semantics:

- **`database.graph_store.save_graph`/`load_graph`** — a flat JSON file
  (`nx.node_link_data`), **lossless**: every node (file *and* chunk) and every
  edge type, including `imports` and file-sourced `references` edges.
- **`database.sqlite_client.DatabaseManager.store_graph`/`load_graph`** —
  relational, **lossy by construction**: `graph_edges.source_chunk_id`/
  `target_chunk_id` are foreign keys into `code_chunks` only. File nodes and
  every edge that touches one (all `imports` edges, plus module-scope
  `function_call`/`references` edges sourced from a file rather than a
  chunk) are silently skipped on write and consequently absent on read.

Both are named identically and shaped identically — nothing in the type
system prevents a caller from grabbing the wrong one. This surfaced as a
real live bug: `/status`'s DB-fallback reconstruction path and `/graph` once
disagreed on node/edge counts for the same repository (184/283 vs 344/255)
because they went through different loaders. It was fixed by making both
paths in `api/main.py` consistently use `database.graph_store.load_graph`
(the full, correct source), and `graph.blast_radius.compute_blast_radius`'s
own docstring now explicitly states which loader it expects. This is a
documentation-enforced guard, not a compiler-enforced one — a good example
of a real trade-off between "add a type distinction" and "just document it
very clearly," made because the two representations genuinely serve
different purposes (SQLite for relational chunk-level queries, JSON for a
complete graph round-trip) and collapsing them into one would lose that.

### 8.4 FAISS and BM25 as parallel, non-relational indexes

Neither the FAISS index nor the BM25 index lives inside SQLite — both are
separate on-disk artifacts (`data/indexes/{owner}_{repo}.faiss` +
`.json` sidecar; `{owner}_{repo}_bm25.pkl`), rebuilt from SQLite's
`embeddings`/`code_chunks` tables at index time and loaded independently at
query time. This is a deliberate separation-of-concerns: SQLite is the
durable source of truth chunks/embeddings are derived from; FAISS/BM25 are
disposable, rebuildable query accelerators over that truth, never
themselves the thing being backed up or migrated.

---

## 9. Caching Architecture

**There is no Redis anywhere in this system**, and that is a scope-matched
decision, not a missing feature — worth stating plainly rather than trying
to force-fit the question. RepoMind is a single-process, single-machine
tool; there is no second application server that would need a *shared,
distributed* cache to stay consistent with. The one form of caching this
system does — a semantic cache over near-duplicate natural-language
queries — is implemented directly against the same SQLite database
everything else already uses (`semantic_cache` table), which is sufficient
because there is exactly one writer.

### 9.1 What the semantic cache actually does

`retrieval/semantic_cache.py::SemanticCacheManager.lookup()` runs **before**
hybrid retrieval, graph expansion, or reranking even start — a hit skips the
entire retrieval+generation pipeline. It embeds the incoming query with the
same model used for chunk embeddings (CodeBERT or MiniLM, matching whichever
`USE_CODEBERT` selects), computes cosine similarity against every cached
query for that repository (never across repositories), and returns the
highest-similarity entry above `CACHE_SIMILARITY_THRESHOLD` (default 0.95).

### 9.2 A real, documented over-matching bug — and why raising the threshold cannot fix it

This is worth knowing in real depth, because it's a genuine lesson in why
embedding similarity alone is not always the right tool. Two clearly
different questions — *"What is the backend tech stack?"* vs. *"What is the
frontend tech stack?"* — returned the **identical cached answer** in
production use. Measured cosine similarity between that exact pair: **0.9974**.
A broader 7-pair measurement made the scale of the problem explicit: the
range of similarity scores for genuine near-duplicate questions
(0.9861–0.9925) sat **entirely inside** the range for genuinely distinct
questions (0.9807–0.9987) — "register endpoint" vs. "login endpoint" (should
miss) scored *higher* (0.9987) than an actual paraphrase pair (should hit,
0.9861). The documented conclusion: **no single cosine threshold, at any
value, can separate these** — because CodeBERT is a *code*-pretrained
embedding model, not tuned for natural-language sentence similarity, so it
compresses semantically-different-but-structurally-similar NL sentences into
an almost indistinguishable region of the embedding space.

**The fix is a second, independent gate**, applied on top of (not instead
of) the cosine check — `_is_lexically_compatible()`:
1. A small, hand-curated set of **distinguishing-term categories**
   (backend/frontend, login/logout/register, HTTP verbs, read/write,
   dev/prod, sync/async) — a conflict fires only if both queries positively
   match a *different* term from the same category (two-sided; a query that
   doesn't mention the category at all doesn't conflict).
2. **Jaccard similarity of content words** (stopwords stripped) between the
   two queries, with a 0.5 floor. Notably, "backend" vs. "frontend" alone
   measures right at the 0.5 boundary — which is exactly why check (1) is
   needed *in addition to* Jaccard, not instead of it.

This is explicitly disclosed as **not a general antonym/word-sense solution**
— a targeted, extensible fix for the reported bug class, not a claimed
general solve. That kind of honest scoping is worth reproducing verbatim if
asked about it in an interview: the fix solves the measured problem, and
says so, rather than overclaiming generality it doesn't have.

### 9.3 What Redis *would* add if this were productized

A fair, direct answer to "why not Redis": if this became a multi-replica
service (multiple FastAPI processes behind a load balancer — see
[§15](#15-scalability)), the semantic cache and the in-memory indexing-status
registry (`api/main.py`'s `_index_state` dict, currently process-local and
lost on restart) would both need to move to a shared store so every replica
sees the same state — Redis is the standard choice there, and would also
enable TTL-based cache expiry (the `semantic_cache.created_at` column
already exists specifically for this, but is explicitly documented as "not
currently used for expiry" — schema headroom for a feature not yet built)
and cross-process rate limiting. None of that is needed at the current
single-process scale, which is exactly why it isn't there.

---

## 10. Backend Optimizations

Each of these is a genuine engineering decision with a stated reason, not
just "because it's faster":

- **Reciprocal Rank Fusion over hand-tuned score weighting.** FAISS cosine
  similarity (`[-1,1]`) and BM25's unbounded term-frequency scores live on
  two incomparable scales. RRF (`score = Σ 1/(rrf_k + rank)`, `rrf_k=60`)
  uses only rank position from each retriever's own list, never the raw
  score — sidestepping normalization entirely rather than hand-tuning a
  dense/sparse weight that would need re-tuning per corpus.
- **Graph expansion with per-hop score decay, never outranking original
  hits.** `GraphExpander.expand()` guarantees every chunk found by hybrid
  retrieval sorts ahead of every graph-discovered chunk regardless of
  numeric score (a two-tier sort key, not a single blended score) —
  structural neighbors *augment* the result set, they never displace a
  chunk that was actually a strong lexical/semantic match.
- **Cross-encoder reranking as a final, expensive-but-small-batch pass.**
  A cross-encoder that jointly reads (query, candidate) pairs is far more
  accurate than bi-encoder cosine similarity but too slow to run over an
  entire corpus — so it only ever sees the ~40 candidates hybrid retrieval
  + graph expansion already narrowed down to, batched at 32 pairs/call.
- **Small-to-big context assembly, never truncating the top-ranked block.**
  `ContextBuilder` substitutes a small matched chunk for its wider parent
  before sending anything to the LLM, but always keeps the single
  highest-ranked block whole even if it alone exceeds the token budget —
  lower-ranked blocks are dropped wholesale (never truncated mid-chunk) to
  make room, an explicit "don't collapse the context to nothing" design
  choice.
- **Deterministic chunk IDs enabling idempotent, incremental re-indexing.**
  Because a chunk's ID is a pure function of `(file_id, name, start_line)`,
  `EmbeddingManager` can skip re-embedding every unchanged function on a
  re-index run — only genuinely new/moved/renamed chunks pay the embedding
  cost.
- **Lazy model loading, cached on first use.** The embedding model,
  cross-encoder, and every LLM provider client are constructed only on
  first real use, not at import or app-startup time — so importing a module
  never triggers a multi-hundred-MB model download or GPU allocation as a
  side effect, which matters both for test collection speed and for
  `Pipeline`'s constructor staying cheap.
- **Uniform third-party exception wrapping at every module boundary.**
  Every external library touchpoint (GitPython, Tree-sitter, SQLAlchemy,
  FAISS, sentence-transformers, the three LLM SDKs) is wrapped into one of
  seven `RepoMindError` subtypes at exactly one module each — no raw
  third-party exception type ever crosses a layer boundary, which is what
  lets `api/main.py`'s single `_repomind_error_status()` mapper turn *any*
  domain failure into the correct HTTP status code without knowing anything
  about FAISS or GitPython internals.
- **Async request handling with a threadpool + hard timeout for CPU-bound
  work.** `/query` and `/search` run the (synchronous, CPU/IO-bound)
  `Pipeline` methods via `run_in_threadpool`, wrapped in
  `asyncio.wait_for(timeout=120s)` — so a pathological query can't hang a
  request indefinitely; it returns `504` instead.

---

## 11. Indexing & Query Optimization

This project doesn't use MongoDB, so there's no compound-index/B-tree story
in the sense the question implies — but the equivalent "how do lookups
actually stay fast" story is genuinely richer here, spanning three separate
index structures.

### 11.1 SQLite-level indexing
Every table's primary key is a composite `(repository_id, <entity>_id)` —
SQLite auto-indexes primary keys, so every chunk/file/embedding lookup
scoped by repository is already index-backed. The
`UNIQUE(repository_id, source_chunk_id, target_chunk_id, edge_type)`
constraint on `graph_edges` doubles as both the de-duplication guarantee
*and* a queryable index for "does this edge already exist." Foreign keys
are enforced (via the `PRAGMA foreign_keys=ON` connection-event listener
noted in [§8](#8-database--persistence-design)) but SQLite does not
auto-index FK columns the way some engines do — for this project's access
patterns (always scoped by the PK's leading `repository_id` column) this
hasn't required an additional explicit index, since every real query
already benefits from the composite PK's leftmost-prefix.

### 11.2 FAISS: exact brute-force search via a normalization trick, not an ANN index
`FaissIndexManager` uses `faiss.IndexFlatIP` — an **exact**, brute-force
inner-product index, not an approximate one (no IVF/HNSW/PQ). The way it
becomes cosine similarity is a specific, deliberate trick: both the stored
vectors and every query vector are `faiss.normalize_L2()`'d before ever
touching the index, so inner product on unit vectors *is* cosine similarity
by definition — cheaper than implementing cosine distance directly, using
FAISS's fastest primitive. `_create_index()` is deliberately factored into
its own one-line method specifically so swapping in an IVF/HNSW index later
touches only that method — a documented, real extension point, not a
premature abstraction. The trade-off is explicit: exact search is O(n)
per query and doesn't scale to millions of vectors, which is fine at
per-repository chunk counts (hundreds to low thousands) but would need
revisiting for a very large monorepo — see [§15](#15-scalability).

### 11.3 BM25: hand-built tokenizer, library-provided scoring
The scoring algorithm itself (`rank_bm25.BM25Okapi`) is a third-party
library — what's hand-built is the tokenizer feeding it
(`retrieval/tokenizer.py`), specifically because identifier naming
conventions (`camelCase` vs `snake_case`) need to be normalized to the same
token stream for either convention to match a query written in the other.
A two-alternative lookaround regex splits camelCase boundaries
(`fetchUser`→`fetch|User`) while correctly keeping acronym runs intact
(`HTTPServer`→`HTTP|Server`, not `HTTPS|erver`); a separate non-alphanumeric
split handles snake_case and punctuation for free. `fetch_user_by_id` and
`fetchUserByID` both tokenize to `["fetch","user","by","id"]` — a small,
directly-testable, hand-verified property that's exactly the kind of thing
worth walking through live in an interview.

### 11.4 Query optimization discipline: skip-when-empty everywhere
A recurring pattern across the retrieval stack: every stage short-circuits
cheaply on empty input rather than paying setup cost for no work — an empty
BM25 corpus is tracked as "indexed, zero documents" rather than constructed
(since `BM25Okapi`'s average-document-length calculation divides by corpus
size and would error on an empty one); `ContextBuilder.build_context()`
returns an all-empty result without touching the database at all for an
empty ranked-chunk list; `EmbeddingManager.generate_embeddings()` returns
early without ever loading the embedding model if there's nothing pending
to embed. None of these are premature optimizations — they're correctness
fixes for genuinely-occurring edge cases (a freshly cloned but empty repo,
a query that returns zero hits) that happen to also avoid unnecessary work.

---

## 12. API Design

Base: `FastAPI(title="RepoMind API", version="1.0.0")`. CORS restricted to
Vite's dev-server origins by default. A request-logging middleware times
every request and logs `method path -> status (latency_ms)`.

| Method | Path | Auth | Purpose | Key status codes |
|---|---|---|---|---|
| `GET` | `/info` | none | Global config (active embedding model, active LLM model name) — not repo-scoped | 200 |
| `POST` | `/repos/index` | none | `{repo_url}` → validates URL, computes a deterministic `repo_id` up front, queues indexing as a `BackgroundTasks` job, returns immediately | `202 Accepted`; `400` on invalid URL |
| `GET` | `/repos/{id}/status` | none | Polled every 1s by both frontends; in-memory state with a SQLite-reconstruction fallback if the process restarted mid-index | `404` if never indexed |
| `POST` | `/repos/{id}/query` | none | `{question}` → runs the full RAG chat pipeline via a threadpool with a 120s timeout | `404`/`409` (not ready), `504` (timeout) |
| `GET` | `/repos/{id}/search?q=` | none | Retrieval-only — no generation, no semantic cache. Built specifically for Adjudicate's context-builder fallback, so it never pays for an LLM call just to get ranked chunks | same as `/query` minus generation-only fields |
| `GET` | `/repos/{id}/context?file=&line=` | none | **Location-based, not retrieval-based** lookup: finds the smallest chunk enclosing a given file:line. Returns both `chunk_citations` (post small-to-big substitution) and `matched_chunks` (raw, pre-substitution — what Adjudicate's blast-radius lookup actually needs, since substituted PARENT/SLIDING chunks are never real graph nodes) | `404` if nothing encloses that location |
| `GET` | `/repos/{id}/graph?focus_node=&hops=` | none | Returns the full graph (`nx.node_link_data`), or just the blast-radius subgraph around `focus_node` if supplied. Reads the **lossless** JSON graph store, not SQLite's lossy reconstruction — see [§8.3](#83-the-graph_store-vs-sqlite_client-divergence--a-genuine-architectural-subtlety) | `404` if `focus_node` not in graph, `400` if `hops` negative |
| `POST` | `/repos/{id}/review` | none | `{diff}` → **Server-Sent Events** stream of the full Adjudicate pipeline. Deliberately POST, not GET, since a raw diff body doesn't fit `EventSource`'s GET-only constraint — the frontend reads it via a manual `fetch` + `ReadableStream` reader instead | `404`/`409` (not ready), `400` (blank diff) |

**Central error-to-status mapping** (`_repomind_error_status()`):
`RepositoryCloneError → 400`, `RetrievalError → 404`,
`ParsingError`/`EmbeddingError`/`DatabaseError`/`LLMGenerationError → 500`,
anything unrecognized → `500`. `504` is applied separately, only for an
actual `asyncio.TimeoutError` on `/query`/`/search`, outside this mapper.

---

## 13. Security

An honest accounting, not an inflated one — this is a locally-run research/
demo tool, and it's worth being precise about what security posture that
actually implies rather than describing controls that don't exist.

**What exists:**
- **Input validation at the trust boundary that matters most**: every
  GitHub URL is checked against a strict regex before `git clone` is ever
  invoked, rejecting SSH remotes, other hosts, and malformed input — the
  one place user-controlled input reaches a subprocess-spawning operation
  during indexing.
- **CORS is restricted**, not wildcard-open — the allowlist defaults to
  exactly the Vite dev-server origins, and would need to explicitly add a
  deployed frontend origin (a real, intentional gate, not an oversight).
- **Secrets stay in environment variables**, validated at process startup
  via `pydantic-settings`, never logged or echoed in any API response.
- **The Verifier's sandbox provides real, if partial, isolation for the one
  place this system executes code it did not write**: a hard subprocess
  timeout and a monkeypatched network-denial guard around the Prosecutor's
  proposed tests. This is explicitly documented as **process-level, not a
  hard security boundary** — "a determined subprocess could still reach the
  network through lower-level mechanisms," and memory limiting is declared
  in config but genuinely unenforced (would need a cgroup or Windows Job
  Object). Docker-based isolation is flagged as necessary before ever
  accepting diffs from untrusted third parties, not claimed as already
  solved.
- **Uniform exception wrapping** (see [§10](#10-backend-optimizations))
  means no raw third-party stack trace or internal path ever leaks through
  an API response — every error surfaces as one of seven typed
  `RepoMindError` subtypes with a controlled message.

**What genuinely does not exist, stated plainly:**
- **No authentication or authorization** — see [§5](#5-authentication--authorization)
  for the full reasoning.
- **No rate limiting anywhere** — not Redis-backed, not in-memory, not at
  all. A public deployment without adding this would be trivially
  abusable (unlimited free indexing/LLM-generation requests against
  whoever's API key is configured).
- **No input sanitization against prompt injection** in retrieved code
  content that gets embedded into LLM prompts — retrieved chunks are
  trusted repository content, and the system prompt's cite-or-refuse
  constraint is the only defense against a maliciously crafted repository
  attempting to manipulate model behavior via its own source comments.
- **No container-level sandbox boundary** for the Verifier, as noted above
  — this is the single most important gap to close before any public,
  multi-tenant deployment, and the roadmap (Phase 33) explicitly
  recommends restricting submittable repos to an allowlist unless this is
  fully hardened first.

---

## 14. Performance

- **Semantic cache hits skip the entire pipeline**, not just generation —
  retrieval, graph expansion, and reranking are all bypassed on a hit,
  making a repeated (or lexically-compatible near-duplicate) query
  effectively free after the first ask.
- **Batched, deduplicated embedding generation** avoids ever re-embedding
  unchanged chunks across re-indexing runs (see
  [§10](#10-backend-optimizations)), which is the dominant cost of indexing
  a large repository the first time and near-zero on every subsequent
  re-index.
- **The reranker only ever scores ~40 candidates**, never the full corpus —
  cross-encoders are accurate but too slow to run at corpus scale, so the
  expensive step only ever touches what cheap dense+sparse+graph retrieval
  already narrowed down.
- **A fixed ~4-chars-per-token heuristic**, not a real tokenizer, drives
  the context builder's token-budget enforcement — a deliberate accuracy/
  dependency-footprint trade-off (no BPE tokenizer needs to be loaded just
  to estimate a budget), acceptable because the budget only needs to be
  approximately right, not exact.
- **`/query` and `/search` run behind a 120-second hard timeout** via
  `asyncio.wait_for` over a threadpool — a pathological query degrades to a
  clean `504` rather than hanging the request indefinitely or blocking the
  event loop (the actual work is synchronous/CPU-bound, so it must run off
  the event loop entirely, not just be `await`ed).
- **FAISS's exact search is fast at current scale precisely because it's
  exact and small** — brute-force cosine over a few hundred to a few
  thousand vectors per repository is faster in practice than the setup/
  approximation overhead of an ANN index would justify; this stops being
  true well before a monorepo-scale codebase, which is why
  `_create_index()` is factored out as an explicit swap point (see
  [§11.2](#112-faiss-exact-brute-force-search-via-a-normalization-trick-not-an-ann-index)).

---

## 15. Scalability

Stated directly: **this system does not currently scale horizontally, and
knows it doesn't.** The honest list of what would need to change, and why
each one is currently a real limiter:

- **The FastAPI process holds real in-memory state** — the indexing-status
  registry (`_index_state`) and every lazily-loaded model/index (embedding
  model, cross-encoder, per-repo FAISS/BM25 managers) live on the single
  process instance. Running a second replica behind a load balancer would
  immediately produce inconsistent `/status` responses depending on which
  replica handled which request, since nothing propagates that state across
  processes. This is the single largest concrete change needed before
  horizontal scaling is possible — see [§9.3](#93-what-redis-would-add-if-this-were-productized)
  for why Redis is the natural answer once this becomes necessary.
- **Indexing runs as an in-process `BackgroundTasks` job, not a queued
  worker.** There's no message queue (no Celery/RQ/SQS) — a long-running
  indexing job ties up the same process serving live query traffic, and a
  process restart loses in-flight indexing progress entirely (recoverable
  only via the best-effort SQLite-reconstruction fallback on `/status`,
  which is explicitly documented as "not a perfectly accurate progress
  replay"). A real production version would move indexing to a proper task
  queue with persisted job state.
- **FAISS's exact search doesn't scale to very large indexes** — see
  [§11.2](#112-faiss-exact-brute-force-search-via-a-normalization-trick-not-an-ann-index);
  the swap point for an approximate index exists but hasn't been needed
  yet at the repository sizes this has actually been tested against.
- **SQLite has no concurrent-writer story** — fine for one process, a hard
  wall the moment there's more than one writer, which is exactly when
  Postgres (or similar) would become the right choice, not a "nice to have."
- **The backend is otherwise close to stateless per-request** — every piece
  of durable state lives in SQLite/FAISS/BM25/the JSON graph store, none of
  it in request-scoped memory — so once the in-memory registry above is
  externalized, the remaining path to horizontal scaling is comparatively
  short: N stateless FastAPI replicas behind a load balancer, a shared
  Redis for cache/status, and a real job queue for indexing.
- **No Kubernetes/container orchestration exists yet** — the current
  Docker/docker-compose scaffold (see [§16](#16-deployment)) is a
  single-container development convenience, not a scaling story.

---

## 16. Deployment

**Current state, precisely:** a `Dockerfile` and `docker-compose.yml` exist
at the repo root, but both are early-phase scaffolds that predate the
FastAPI+React architecture — the `Dockerfile`'s `CMD` still launches the
**legacy Streamlit `app.py`**, not `api/main.py` plus a built React bundle,
and `docker-compose.yml` defines a single `app` service with no separate
frontend/API/database services. Both carry their own comments acknowledging
this: "Minimal scaffold — build steps will expand as dependencies are
added." **There is no CI/CD pipeline** — no `.github/workflows/`, no
`render.yaml`, no `vercel.json`, nothing. This system has never been
deployed anywhere beyond a local machine.

This is exactly what the roadmap expects at this stage: **Phase 33
(Deployment) is explicitly marked not-started (⬜)**, gated behind Phase 30
(the Judge agent) being complete and verified, which it now is — meaning
deployment is the *next* real piece of work, not a forgotten one.

**What's planned, per the roadmap, honestly labeled as planned rather than
done:** frontend on Vercel, FastAPI backend on Render — explicitly following
the same deployment pattern used by a companion project referenced in the
roadmap notes. The one deployment concern called out as needing resolution
*before* committing to Render specifically: the Verifier executes untrusted
repository code and needs strict container resource/time/network limits
with genuinely no network access from inside the sandbox — the roadmap
explicitly says to confirm Render supports this before committing, with a
dedicated small VM as the fallback if it doesn't, and to restrict
submittable repositories to an allowlist for any public deployment unless
the sandbox security posture is fully hardened first (directly connecting
back to the [§13](#13-security) gap).

**Environment variables** (via `.env`, loaded by `pydantic-settings`):
`GEMINI_API_KEY`, `GROQ_API_KEY` (LLM providers), `USE_GEMINI`/`USE_GROQ`/
`USE_OLLAMA` (provider priority toggles), `OLLAMA_BASE_URL`,
`CORS_ALLOWED_ORIGINS` (would need the deployed frontend's real origin
added), plus every retrieval/feature-flag setting in `config.Settings` —
all overridable without a code change, which is exactly what a Docker/CI
environment needs.

---

## 17. Future Improvements

Ranked roughly by how directly each connects to something already flagged
as a known gap in the code itself (not invented for this document):

- **TypeScript grammar support** (`tree-sitter-typescript`) — the single
  largest concrete coverage gap: `.ts`/`.tsx` files are discovered but never
  parsed, so a TypeScript-only repository currently indexes with zero AST
  chunks and zero call-graph edges. Explicitly deprioritized behind the
  benchmark harness, frontend, and deployment work, not forgotten.
- **Docker-based Verifier sandboxing**, replacing the current process-level
  subprocess isolation — the most important security hardening step before
  any public/multi-tenant deployment (see [§13](#13-security)).
- **A true before/after diff for `breaking_change` claims** — currently
  verified by running the existing suite against the post-diff state only;
  a real comparison would run the suite against the pre-diff commit too and
  diff the two result sets, rather than approximating "does it pass now."
  Explicitly documented as a reasonable, not-yet-built follow-up.
- **Closing the Prosecutor's claim-taxonomy gaps surfaced by the benchmark
  harness** — real missed bugs in the 12/13-case benchmark run (a shared
  mutable default argument, an off-by-one slice) have **no matching
  `ClaimType` in the enum at all**, a structural vocabulary gap rather than
  a verification failure; separately, `BREAKING_CHANGE`/`SECURITY` claim
  types exist in the Verifier's dispatch table but the Prosecutor's own
  system prompt was never updated to actually generate them.
- **Completing benchmark case 13/13** — currently blocked on same-day
  exhaustion of both Groq's and Gemini's free-tier quotas, not a code
  defect; the roadmap deliberately keeps this phase at "in progress" rather
  than declaring it done on 12/13 provisional data.
- **An approximate (IVF/HNSW) FAISS index** once per-repository chunk
  counts grow past where exact brute-force search stays cheap — the swap
  point (`_create_index()`) already exists as a one-method change.
- **Moving the in-memory indexing-status registry and semantic cache to
  Redis**, and indexing itself to a real job queue — the concrete
  prerequisites for horizontal scaling (see [§15](#15-scalability)).
- **A pinned, known-compatible `ragas` installation path** (isolated from
  this project's main environment) so RAGAS scoring can actually run
  end-to-end rather than being structurally supported but practically
  blocked by an upstream dependency conflict.
- **Observability**: structured logging exists (`core/logging.py`) but
  there's no metrics/tracing layer — Prometheus/Grafana or an equivalent
  would be the natural next step once this runs anywhere beyond a laptop.
- **CI/CD**: no pipeline exists at all yet; even a minimal
  lint+test-on-PR GitHub Actions workflow would be a meaningful next step
  before real deployment.
- **The Documentation agent (Phase 35)** — currently a docstring-only stub
  with zero implementation and zero tests; the last unbuilt piece of the
  originally-scoped agent roster.

---

## 18. Important Design Decisions

Framed as the questions an interviewer would actually ask, answered the way
the code itself justifies them.

**Why hand-build every retrieval component instead of LangChain/LlamaIndex?**
Because the entire point of this project (per its own stated goal) is to
demonstrate implementation-level understanding of RAG internals for an
interview context — RRF fusion, a camelCase-aware BM25 tokenizer, graph
expansion with score decay, small-to-big context assembly. Calling
`VectorStoreIndex.from_documents()` would produce a working system but
answer none of the questions this project exists to let someone answer.

**Why NetworkX with pure name-based call resolution, instead of real symbol
resolution (a language server / type checker)?** Real symbol resolution
(scope-aware, type-aware) is a much larger undertaking per language — this
project instead makes a deliberate, honestly-documented trade-off: name-based
heuristic matching gets the large majority of real call relationships right
at a fraction of the implementation cost, and every place it can go wrong
(same-named methods on unrelated classes, cross-service HTTP calls invisible
to a static graph, framework-dispatched routing) is explicitly documented as
a known boundary rather than silently mis-resolved and left unexplained.

**Why FAISS `IndexFlatIP` (exact) instead of an approximate index from day
one?** Premature optimization for a problem that doesn't exist yet at
current scale — per-repository chunk counts are in the hundreds to low
thousands, where exact search is both fast enough and simpler than tuning
an ANN index's recall/speed trade-off. The swap point is a single factored
method, deliberately kept ready rather than built speculatively.

**Why is there both a JSON graph store and a lossy SQLite graph
reconstruction — why not just pick one?** Because they serve genuinely
different needs: SQLite's relational shape is what lets `graph_edges` be
queried/joined against `code_chunks` directly, while a full lossless
round-trip (needed for the interactive graph visualization and blast-radius
computation, which must include file nodes and import edges) is far simpler
as a flat JSON dump than as a relational schema that would need file-node
tables and nullable polymorphic foreign keys to represent the same thing.
The trade-off — two stores that look similarly named but aren't
interchangeable — is real, and is resolved by documentation and consistent
call-site discipline (see [§8.3](#83-the-graph_store-vs-sqlite_client-divergence--a-genuine-architectural-subtlety))
rather than by collapsing them into one.

**Why Gemini before Groq before Ollama, as a static priority rather than
try-Gemini-then-catch-and-fallback?** A runtime try/catch fallback would
silently degrade answer quality/latency characteristics on a transient
error without anyone noticing which provider actually served a given
request. A static, config-driven priority makes provider selection
explicit and observable (logged, and reflected in the `/info` endpoint —
itself the subject of a real, found-live bug where the display-only
provider-name logic once missed the Groq branch entirely) rather than an
implicit runtime decision.

**Why does the Verifier make zero LLM calls, as a hard rule rather than a
guideline?** This is the entire reason Adjudicate exists as more than "an
LLM that argues with itself." An LLM asked to verify another LLM's claim is
still just an LLM guessing, with no more grounding than the original claim
had. Real test execution, real static analyzers, and a real call graph are
the only things in this system that can turn "sounds plausible" into
"actually true" — the 46.8% claim-flip rate the benchmark harness measured
is the concrete proof this distinction matters, not a hypothetical one.

**Why Server-Sent Events for the live review screen, not WebSockets?** The
data flow is strictly one-directional (server → client, one event stream
per review request) with no need for the client to send messages mid-stream
— SSE is the simpler primitive that fits exactly that shape, at the cost of
needing a manual `fetch`+`ReadableStream` reader on the frontend instead of
the browser's native `EventSource`, because `EventSource` is GET-only and
the diff payload needs to go in a POST body.

**Why React Context instead of Redux/Zustand?** The app's state shape is
small and mostly flat — one repo ID, one graph, one chat history, one
citation-to-graph-focus link — and doesn't need action-based state
transitions, time-travel debugging, or cross-slice selectors. A single
Context with plain `useState` is the simplest thing that's actually
sufficient, not a placeholder for "we'll add Redux later."

**Why Vite over Create React App/webpack directly?** Vite's dev-server HMR
speed matters concretely here because the graph view (`visStyle.js`) and
the SSE review screen both involve iterative visual/animation tuning that
benefits from near-instant reload feedback — a slower dev loop would have
directly slowed down exactly the kind of empirical tuning documented in
`visStyle.js`'s physics-scaling comments (validated against two real
differently-sized indexed repositories).

**Why is there no Node.js/Express orchestration layer at all?** Covered in
full in [§3.1](#31-the-chain-and-why-it-is-not-a-nodepython-microservice-split)
— restated briefly: there is no orchestration concern in this system that
is separate from the ML work itself, so splitting them into two languages/
processes would add real cost (a network hop, a second dependency surface)
for no corresponding benefit.

---

## 19. Interview Questions

### Frontend
1. Why does this app have no router, and what would change if it needed one?
2. Walk through how `useLiveReview.js` parses an SSE stream manually instead
   of using `EventSource` — why is that necessary here specifically?
3. How does `RepoContext`'s `focusRequestId` counter work, and why is it
   needed even for re-clicking the *same* graph node?
4. Why does `AppShell` keep all three views mounted simultaneously instead
   of unmounting the inactive ones?
5. How does blast-radius highlighting animate in `GraphView.jsx`, given
   vis-network has no built-in `.animate()` API?
6. Why was Cytoscape.js replaced with vis-network, and what had to be
   re-tuned as a result?
7. Why is a citation chip sometimes rendered disabled/unclickable?

### Backend / API
8. Walk through the full status-code decision tree in
   `_repomind_error_status()` — why does `RetrievalError` map to 404 and not
   500?
9. Why is `/repos/{id}/review` a POST endpoint even though it streams SSE?
10. What happens to an in-flight indexing job if the FastAPI process
    restarts, and how does `/status` recover from it?
11. Why does `/context` return both `chunk_citations` and `matched_chunks`
    instead of just one?
12. Why is `/search` a separate endpoint from `/query` instead of a query
    parameter on `/query`?
13. Walk through why the request-logging middleware measures latency the
    way it does.

### Retrieval / RAG
14. Explain Reciprocal Rank Fusion and why it's rank-based rather than
    score-based.
15. What does the "imported file's every chunk is a neighbor" graph
    expansion behavior actually do, and what's the coarseness trade-off?
16. Why does the cross-encoder only ever see ~40 candidates instead of the
    whole corpus?
17. Walk through the semantic cache over-matching bug end to end — why
    couldn't raising the threshold fix it?
18. What is small-to-big retrieval, and how does `ContextBuilder` decide
    when a chunk gets substituted for its parent?
19. Why is `dense_retriever.py` a placeholder while the real dense-retrieval
    logic lives in `database/vector_store.py`?
20. How does the camelCase/snake_case tokenizer work, and why does it
    matter for BM25 specifically?

### Graph
21. How is the call graph actually built — what's the resolution algorithm,
    and what are its documented failure modes?
22. Why does the `references` edge type exist, and what problem does it
    solve that `function_call` doesn't?
23. Explain `compute_blast_radius`'s algorithm and how it differs from
    `GraphExpander.expand`'s traversal, despite sharing the same underlying
    shape.
24. Why does `_add_edge_if_new`'s deduplication (by source/target pair only,
    ignoring edge_type) matter, and what could it silently hide?

### Database
25. Walk through the `graph_store` vs `sqlite_client` divergence in full —
    what's lost, and why wasn't it just unified into one representation?
26. Why are embeddings stored as raw float32 bytes instead of JSON?
27. Explain the three independent UUID5 namespaces in this codebase and why
    they're kept separate.
28. Why must parent chunks be flushed to SQLite before AST/sliding chunks?
29. Why does every table scope by `repository_id` even though chunk IDs are
    already UUIDs?

### Adjudicate / Agents
30. Why does the Verifier make zero LLM calls, and what would be lost if it
    didn't follow that rule?
31. Walk through the CONFIRMED/REFUTED/INCONCLUSIVE confidence-tier logic —
    why is confidence tied to evidence kind, not verdict direction?
32. What's the documented limitation of `breaking_change` verification, and
    what would a correct fix look like?
33. Why doesn't any concrete agent subclass `BaseAgent`?
34. Explain the rebuttal loop's termination conditions and how "resolution"
    differs from "cap."
35. What does the Judge's `_cap_handling_violation` check actually enforce,
    and why is it mechanical rather than prompt-only?
36. Walk through the claim-flip rate metric — why is it described as the
    single strongest number the benchmark produces?
37. Why is sandboxing process-level rather than container-level here, and
    what specifically does that not protect against?
38. What are the two documented diff-parsing edge cases in
    `context_builder.py::parse_diff`, and how does each get fixed?

### AI / LLM
39. Explain the Gemini "thinking token" truncation bug in full — root
    cause, symptom, and both parts of the fix.
40. Why is provider selection a static priority rather than a
    retry-on-failure fallback chain?
41. How does citation matching work, and why is it substring-based rather
    than a strict parser?
42. Why does the system never call the LLM at all when retrieval returns
    nothing?

### System Design
43. Why is there no separate Node/Express orchestration layer in this
    architecture?
44. What would need to change for this system to run as multiple stateless
    replicas behind a load balancer?
45. Why SQLite instead of Postgres/MongoDB, and when would that stop being
    the right call?
46. What's the single largest concrete blocker to horizontal scaling in
    this codebase today?

### Deployment
47. What's actually deployed today versus only planned, and how do you know
    (what's the evidence in the repo)?
48. What deployment-specific security concern does the roadmap flag before
    committing to a Render-hosted Verifier sandbox?

### Optimization / Evaluation
49. Walk through the five-way ablation study design — why is
    `USE_SEMANTIC_CACHE` deliberately excluded from it?
50. Why is `ragas` not pinned in `requirements.txt`, and how does the rest
    of the evaluation module stay testable without it?
51. Explain the LLM-as-judge Precision@K scoring approach and its failure
    mode when the judge's response is unparseable.
52. What's the actual measured contribution of graph expansion in the
    ablation results, and how would you find out if you didn't already
    know?

---

## 20. Common Follow-up Questions

**Decision: hand-building retrieval instead of using a framework.**
- *Interviewer:* "Isn't this just reinventing what LangChain already gives
  you for free?"
- *Ideal answer:* For a production system shipping fast, yes, a framework
  is usually the right call. This project's explicit goal is different — it
  exists to demonstrate that the person who built it understands RRF,
  BM25 tokenization, graph expansion, and small-to-big retrieval at the
  implementation level, which is exactly the kind of knowledge an ML/GenAI
  interview probes for and a framework call site can't demonstrate.
- *Trade-offs:* More code to maintain, more surface for bugs (and this
  project has real, documented ones — the semantic cache over-match, the
  Gemini thinking-token truncation). A framework would have avoided some of
  those categories of bug entirely, at the cost of hiding exactly the
  mechanics this project is trying to show.
- *Alternative:* LlamaIndex or LangChain's retrieval abstractions, which
  would cut implementation time substantially but reduce the project to
  "knows how to call an API," not "understands RAG."

**Decision: no authentication.**
- *Interviewer:* "How would you add multi-tenant auth to this?"
- *Ideal answer:* The SQLite schema already scopes every table by
  `repository_id`, which is most of the way to per-tenant isolation — it
  would need a `user_id` added alongside it, a real auth layer (JWT is the
  natural fit given the FastAPI backend), and per-user rate limiting, which
  currently doesn't exist at all in any form.
- *Trade-offs:* Adding auth now, before there's a real multi-user need,
  would be speculative complexity — the project correctly scoped it out
  given its actual current use case (a local single-operator tool).
- *Alternative:* An API-key-only gate (simpler than full user accounts)
  would be the minimum viable step if this needed to be shared with a small
  trusted group before a full auth system was justified.

**Decision: the Verifier's process-level sandbox, not Docker.**
- *Interviewer:* "Doesn't running arbitrary code from a diff in a subprocess
  worry you?"
- *Ideal answer:* Yes, and the code says so directly — it's explicitly
  documented as best-effort, not a hard security boundary, with the
  specific gap named (a determined subprocess could bypass the
  monkeypatched network guard via lower-level syscalls). It's adequate for
  a local, single-operator tool reviewing diffs the operator already
  trusts enough to be looking at; it is explicitly *not* adequate for a
  public deployment accepting arbitrary third-party diffs, which is exactly
  why the roadmap gates public deployment on either Docker-based isolation
  or an allowlist of submittable repos.
- *Trade-offs:* Docker gives real container-level isolation but adds
  meaningful deployment complexity (needs Docker-in-Docker or a sidecar
  pattern on most hosting platforms) — which is exactly the open question
  the roadmap flags about Render's support before committing to it.
- *Alternative:* A dedicated small VM per sandbox run, called out explicitly
  in the roadmap as the fallback if Render doesn't support the needed
  container isolation.

**Decision: RRF over a learned or hand-tuned fusion weight.**
- *Interviewer:* "Why not just weight dense 0.7 / sparse 0.3 and tune it on
  a validation set?"
- *Ideal answer:* That requires the two scores to be on comparable scales
  to begin with, which cosine similarity and BM25 term-frequency scores
  fundamentally aren't, and it requires re-tuning per corpus/domain. RRF
  sidesteps both problems by using only rank position, at the cost of
  discarding the magnitude information a learned weight could in principle
  exploit.
- *Trade-offs:* A learned fusion (even simple logistic regression over both
  scores) could in principle outperform RRF given enough labeled relevance
  data — this project doesn't have that data at the scale needed, and RRF's
  "works reasonably well with zero tuning" property matters more given
  that constraint.
- *Alternative:* A hand-tuned weighted sum, evaluated in this project's own
  ablation framework — the infrastructure to measure whether it would
  actually help already exists (`evaluation/ablation.py`), it's just not
  been done.

**Decision: two non-interchangeable graph persistence paths.**
- *Interviewer:* "This seems like a bug waiting to happen — why not just
  pick one representation?"
- *Ideal answer:* It already caused one real, findable bug (a `/status` vs
  `/graph` count mismatch), which is exactly why the fix wasn't "add a type
  wrapper" but "make every call site consistently use the lossless loader,
  and say so explicitly in the docstring of the function most likely to be
  called against the wrong one." The two representations exist because they
  serve genuinely different access patterns (relational chunk-level queries
  vs. a complete graph round-trip for visualization) — collapsing them into
  one schema would mean either giving SQLite nullable polymorphic file/chunk
  edges (ugly, loses referential integrity) or giving the JSON store a
  query engine it doesn't need.
- *Trade-offs:* The chosen fix (strong documentation + consistent call-site
  discipline) is weaker than a type-system guarantee — nothing stops a
  future change from reintroducing the same mismatch. A stricter fix would
  wrap each in a distinct type (e.g. `FullGraph` vs `PartialGraph`) so the
  compiler/type-checker catches a misuse; this project accepted the weaker
  guarantee given Python's dynamic typing makes that enforcement partial at
  best anyway.
- *Alternative:* Migrate `graph_edges` to a schema that can represent file
  nodes and import edges too (e.g. a polymorphic `node_id` column with a
  `node_kind` discriminator), unifying both into one SQLite-backed
  representation — more invasive, not yet done.

**Decision: Server-Sent Events over WebSockets for the review screen.**
- *Interviewer:* "Why not WebSockets, given real-time feels like the more
  standard choice?"
- *Ideal answer:* WebSockets solve bidirectional communication; this
  feature is strictly one-directional (the client submits one diff, then
  only receives events until the stream ends) — SSE is the narrower,
  simpler primitive that exactly fits that shape, with automatic
  reconnection semantics built into the protocol that a hand-rolled
  WebSocket client would have to reimplement.
- *Trade-offs:* SSE's one real limitation here is that `EventSource` is
  GET-only, and the diff payload needs a POST body — solved with a manual
  `fetch` + `ReadableStream` reader instead of the browser's native
  `EventSource`, which is more code than a one-line `new EventSource(url)`
  would have been.
- *Alternative:* A WebSocket would remove that specific workaround (the
  diff could be sent as the first message over an already-open socket) at
  the cost of needing to hand-manage connection lifecycle/reconnection that
  SSE gives for free.
