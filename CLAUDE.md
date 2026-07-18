# Project

This repository contains a production-grade GraphRAG system for software repositories.

The objective is NOT to build a demo.

The objective is to build a research-quality Retrieval-Augmented Generation pipeline that demonstrates:

- Tree-sitter AST parsing
- Graph-based retrieval
- Hybrid Retrieval (Dense + Sparse)
- Cross Encoder reranking
- Semantic caching
- Small-to-Big retrieval
- Evaluation using RAGAS
- Production engineering

This project will be used in ML/GenAI interviews.

---

# Coding Rules

Never use:

- LangChain
- LlamaIndex
- Haystack

Every retrieval component must be implemented manually.

Always explain design decisions before writing code.

Never create placeholder implementations.

Every file must contain docstrings.

Every public function must contain type hints.

Every module must be independently testable.

Never continue to another phase unless I explicitly approve.

---
## Session start protocol
Before doing anything else, read:
1. docs/roadmap.md — current phase status
2. docs/state/PROGRESS.md — what's been built and why, in every prior phase
3. docs/project_description.md — full architecture reference (only if 
   you need deep context on a module you haven't touched yet)

After completing a phase, append a summary entry to 
docs/state/PROGRESS.md following the format shown in that file. 
Do not skip this — it's how the next session avoids re-reading everything.

# Development Workflow

For every task:

1. Explain the architecture.
2. Explain why this design is chosen.
3. Show the folder structure affected.
4. Generate production-quality code.
5. Explain how to test.
6. Wait for approval.

---

# Goal

The code should resemble something written by an experienced ML Engineer at OpenAI, Anthropic, or Google DeepMind.

Optimize for:

- readability
- modularity
- extensibility
- correctness
