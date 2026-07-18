# Roadmap — RepoMind (infra) → Adjudicate (agentic layer)

## Part A — RepoMind core infra

✅ Phase 0  Project Initialization

✅ Phase 1  Core Infrastructure

✅ Phase 2  Repository Management

✅ Phase 3  File Discovery

✅ Phase 4  Tree-sitter AST Parsing

✅ Phase 5  Semantic Chunking (Sliding Window + Parent Chunks)

✅ Phase 6  Call Graph Construction

✅ Phase 7  SQLite Storage

✅ Phase 8  Embedding Generation

✅ Phase 9  FAISS Index

✅ Phase 10 BM25 Index

✅ Phase 11 Hybrid Retrieval (RRF)

✅ Phase 12 Graph Expansion

✅ Phase 13 Cross-Encoder Re-ranking

✅ Phase 14 Semantic Cache

✅ Phase 15 Context Builder

✅ Phase 16 LLM Integration

✅ Phase 17 Evaluation Framework

✅ Phase 18 Streamlit UI

✅ Phase 19 Pipeline Wiring (pipeline.py + CLI)

✅ Phase 20 FastAPI Backend

## Part B — Adjudicate: Adversarial Multi-Agent Code Review

✅ Phase 21 Graph Rendering Service

✅ Phase 22 Blast-Radius Highlighting

✅ Phase 23 Adjudicate Project Scaffold

✅ Phase 24 Context Builder Integration (graph-first, vector fallback)

✅ Phase 25 Defender Agent (naive)

✅ Phase 26 Prosecutor Agent (naive)

✅ Phase 27 Structured Claim Schema

✅ Phase 28 Verifier Layer

✅ Phase 29 Defender Rebuttal Loop

✅ Phase 30 Judge Agent

🟡 Phase 31 Benchmark Harness (in progress — 12/13 cases have real, complete results; 1 case blocked on daily provider quota, see docs/state/PROGRESS.md for the full provisional results table and why it is not yet ✅)

🟡 Phase 32 React Frontend (Part 1 of 2 done — Chat + Graph scaffold, real-verified against a live indexed repo; Part 2, the live Defender/Prosecutor/Verifier/Judge review screen + SSE, still pending. See docs/state/PROGRESS.md. Live observability is built directly into this phase, not a separate Streamlit prototype step; see docs/project _description.md's Phase 32 entry)

⬜ Phase 33 Deployment (Vercel + Render/FastAPI + Sandboxed Verifier Hosting)

⬜ Phase 34 Documentation Agent (stretch)

⬜ Phase 35 Dependency Upgrade Scanner

⬜ Phase 36 Specialist/Routing/Memory Extension (stretch — pick one)