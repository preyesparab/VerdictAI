# RepoMind / Adjudicate (VerdictAI) — Interview Preparation Guide

## How to use this document

This is a study document, not a pitch deck. It is organized phase-by-phase,
following `docs/roadmap.md`'s actual numbering (Phase 0 through Phase 33,
the current frontier of real, verified work). Every phase section has four
parts:

1. **What was built** — plain-English, as if explaining it out loud.
2. **Why it was built this way** — the actual design reasoning, pulled from
   `docs/project_description.md` and `docs/state/PROGRESS.md`.
3. **Real bugs found and fixed** — pulled directly from PROGRESS.md's real
   entries. Where a phase has no dedicated PROGRESS.md bug history (mostly
   the earliest infra phases, built before this file's per-phase logging
   convention existed), that is stated plainly rather than invented.
4. **Likely interview questions**, with real, defensible answers.

Two closing sections anchor everything else: **"The single most important
thing to say about this project"** and **"Numbers to have ready."** Read
those two first if short on time before an interview, then drill into
whichever phase is likely to come up.

Nothing here is invented or rounded up. Where something is provisional,
partially verified, or a known/disclosed limitation, this document says so
the same way PROGRESS.md does — that honesty is itself part of the
interview material.

---

## Table of Contents

**Part A — RepoMind core infra**
[Phase 0](#phase-0--project-initialization) ·
[Phase 1](#phase-1--core-infrastructure) ·
[Phase 2](#phase-2--repository-management) ·
[Phase 3](#phase-3--file-discovery) ·
[Phase 4](#phase-4--tree-sitter-ast-parsing) ·
[Phase 5](#phase-5--semantic-chunking) ·
[Phase 6](#phase-6--call-graph-construction) ·
[Phase 7](#phase-7--sqlite-storage) ·
[Phase 8](#phase-8--embedding-generation) ·
[Phase 9](#phase-9--faiss-index) ·
[Phase 10](#phase-10--bm25-index) ·
[Phase 11](#phase-11--hybrid-retrieval-rrf) ·
[Phase 12](#phase-12--graph-expansion) ·
[Phase 13](#phase-13--cross-encoder-re-ranking) ·
[Phase 14](#phase-14--semantic-cache) ·
[Phase 15](#phase-15--context-builder) ·
[Phase 16](#phase-16--llm-integration) ·
[Phase 17](#phase-17--evaluation-framework) ·
[Phase 18](#phase-18--streamlit-ui) ·
[Phase 19](#phase-19--pipeline-wiring) ·
[Phase 20](#phase-20--fastapi-backend)

**Part B — Adjudicate: adversarial multi-agent code review**
[Phase 21](#phase-21--graph-rendering-service) ·
[Phase 22](#phase-22--blast-radius-highlighting) ·
[Phase 23](#phase-23--adjudicate-project-scaffold) ·
[Phase 24](#phase-24--context-builder-integration) ·
[Phase 25](#phase-25--defender-agent-naive) ·
[Phase 26](#phase-26--prosecutor-agent-naive) ·
[Phase 27](#phase-27--structured-claim-schema) ·
[Phase 28](#phase-28--verifier-layer) ·
[Phase 29](#phase-29--defender-rebuttal-loop) ·
[Phase 30](#phase-30--judge-agent) ·
[Phase 31](#phase-31--benchmark-harness) ·
[Phase 32](#phase-32--react-frontend) ·
[Phase 33](#phase-33--deployment-in-progress)

**Closing**
[The single most important thing to say about this project](#the-single-most-important-thing-to-say-about-this-project) ·
[Numbers to have ready](#numbers-to-have-ready)

---

# Part A — RepoMind core infra

## Phase 0 — Project Initialization

**What was built.** The bare repository scaffold: package layout
(`core/`, `ingestion/`, `graph/`, `database/`, `embedding/`, `retrieval/`,
`generation/`, `models/`, `api/`, `ui/`, `tests/`), `config.py`'s
`pydantic-settings`-based `Settings` object as the single source of runtime
configuration, and the project's guiding constraints (no LangChain/
LlamaIndex/Haystack, hand-build every retrieval component, docstrings and
type hints everywhere) as codified in `CLAUDE.md`.

**Why built this way.** The entire project exists to demonstrate
implementation-level understanding of RAG internals in an ML/GenAI
interview context (`CLAUDE.md`'s stated goal), so the scaffold was chosen
to keep every subsystem "independently testable" from day one — a flat,
purpose-named package layout rather than a generic `src/` or `app/`
wrapper, so `tests/` can mirror it directly.

**Real bugs found and fixed.** None recorded — this phase predates
PROGRESS.md's per-phase logging convention and is pure scaffolding with no
executable behavior to break.

**Likely interview questions.**
- *"Why no framework at all, not even a lightweight one, from the start?"*
  Because the project's whole value proposition is demonstrating that RRF,
  a camelCase-aware BM25 tokenizer, graph expansion, and small-to-big
  context assembly are understood at the implementation level — see
  [§18 of project_description.md](../project_description.md), restated
  in the closing section below. A framework would produce a working system
  but defeat the purpose.
- *"How do you enforce 'every module independently testable' in practice?"*
  Every package under the top level has a corresponding `tests/test_<pkg>/`
  directory, and cross-package imports are deliberately one-directional
  (e.g. `adjudicate/` depends on RepoMind only through an HTTP client,
  never the reverse — see Phase 23).

---

## Phase 1 — Core Infrastructure

**What was built.** `core/constants.py` (supported languages/extensions,
default model names), `core/exceptions.py` (the `RepoMindError` hierarchy —
`RepositoryCloneError`, `ParsingError`, `EmbeddingError`, `DatabaseError`,
`RetrievalError`, `LLMGenerationError`, and friends), and `core/logging.py`
(structured logging setup).

**Why built this way.** The seven `RepoMindError` subtypes exist so that
*every* external library touchpoint (GitPython, Tree-sitter, SQLAlchemy,
FAISS, sentence-transformers, the three LLM SDKs) gets wrapped at exactly
one module boundary each — no raw third-party exception ever crosses a
layer. This is what later lets `api/main.py`'s single
`_repomind_error_status()` function map *any* domain failure to the
correct HTTP status code without knowing anything about FAISS or GitPython
internals (see Phase 20).

**Real bugs found and fixed.** None recorded in PROGRESS.md for this
phase specifically.

**Likely interview questions.**
- *"Walk through the status-code decision tree for a domain exception."*
  `RepositoryCloneError → 400`, `RetrievalError → 404`,
  `ParsingError`/`EmbeddingError`/`DatabaseError`/`LLMGenerationError → 500`,
  anything unrecognized → `500`; `504` is handled separately, only for a
  real `asyncio.TimeoutError` on `/query`/`/search` (see Phase 20).
- *"Why does `RetrievalError` map to 404, not 500?"* Because it represents
  "nothing found for a valid request" (e.g. an unresolved node in a graph
  lookup), which is a client-facing "not found" condition, not a server
  malfunction.

---

## Phase 2 — Repository Management

**What was built.** `ingestion/repository_manager.py::RepositoryManager` —
the sole permitted caller of `ingestion/git_client.py::GitClient` (the
*only* module in the codebase allowed to import GitPython — an enforced
architectural boundary). Repos are cached at a deterministic local path
(`data/repositories/{owner}_{repo}/`); a re-submission fetches and only
pulls if new commits actually exist — it never re-clones or deletes an
existing checkout.

**Why built this way.** Repository acquisition is pure `git clone`/
`fetch`/`pull`, not the GitHub REST API — no API token required, which
also means it only works against public repositories (a disclosed
trade-off, not an oversight). Confining GitPython usage to one module is
the same "wrap at the boundary" discipline as Phase 1's exception
hierarchy, applied to a specific risky dependency.

**Real bugs found and fixed.** None recorded specifically for this phase.

**Likely interview questions.**
- *"What happens if I submit the same repo URL twice?"* The second call
  resolves to the same `repository_id` (deterministic UUID5 of
  owner+repo — see Phase 7) and the existing clone is fetched/pulled, not
  re-cloned; `repositories.UNIQUE(owner, repository_name)` guarantees the
  same DB row is reused.
- *"Why not use the GitHub API instead of shelling out to git?"* No API
  token needed, and it works for any public git remote reachable by URL —
  simpler and broader than authenticating against GitHub specifically,
  at the cost of GitHub-specific metadata (stars, PR info) never being
  available, which this project never needed anyway.

---

## Phase 3 — File Discovery

**What was built.** `ingestion/file_discovery.py` — a pruned `os.walk`
that skips `.git`, `node_modules`, `dist`, `build`, `__pycache__`, `venv`,
`.idea`, `.vscode`, etc.; files over 2 MB are skipped; binary files are
detected the same way `git` itself does (a NUL byte anywhere in the first
8 KB). Every surviving file becomes a `SourceFile` (path, language, size).

**Why built this way.** Mirroring `git`'s own binary-detection heuristic
(NUL-byte sniffing) means the discovery step doesn't need a heavyweight
MIME-type library and behaves the way developers already expect `git`
itself to behave.

**Real bugs found and fixed.** None recorded for this phase specifically;
`core/constants.py`'s `SUPPORTED_FILE_EXTENSIONS` correctly maps `.ts`/
`.tsx` to `"typescript"` at this stage even though nothing downstream
parses it yet — that gap belongs to Phase 4, not discovery (confirmed
directly during the TypeScript-grammar diagnostic — see Phase 4's bug
section).

**Likely interview questions.**
- *"How do you decide a file is binary without a MIME-type library?"* A
  NUL byte anywhere in the first 8 KB — the same heuristic `git` itself
  uses, cheap and dependency-free.
- *"What happens to a discovered `.ts` file today?"* It's discovered and
  DB-recorded correctly (`source_files.language = "typescript"`), but
  produces zero chunks and zero graph edges downstream because Tree-sitter
  only has Python/JavaScript grammars registered — a documented, deferred
  gap (see Phase 4).

---

## Phase 4 — Tree-sitter AST Parsing

**What was built.** `ingestion/ast_parser.py::TreeSitterParser` — real AST
parsing (not regex heuristics) for **Python** and **JavaScript** only,
despite `core/constants.py` listing 11 "supported" languages. Parsing is
**strict**: a single syntax error anywhere in a file drops that entire
file's chunks; `parse_many()` catches this per-file, logs it, and
continues rather than aborting the whole indexing run. Python extraction
covers functions/async functions/classes/methods, decorators folded into
the chunk span, base classes from `superclasses`. JavaScript extraction
covers standard `const f = () => {}` naming plus two patterns added after
a real diagnostic: CommonJS handler exports (`exports.foo = async (req,
res) => {...}`) and Zustand-style factory calls (`const useX =
create(persist((set, get) => ({...}), opts))`).

**Why built this way.** Real AST parsing (vs. regex) is precisely the
kind of implementation-level RAG-adjacent skill this project exists to
demonstrate. Strict per-file failure (vs. best-effort partial parsing) was
chosen deliberately asymmetric with Phase 6's graph resolution: syntactic
correctness must not be guessed at, but structural relationship resolution
is allowed to be best-effort since an unresolved reference is far less
dangerous than silently mis-parsed code.

**Real bugs found and fixed.**
- **The JS/TS chunker missed two real function-definition patterns.**
  Found via a NutriForge graph diagnostic: `_js_arrow_function_name_node`
  only recognized `const f = () => {}` (`variable_declarator`). Two real
  patterns in NutriForge's actual source produced **zero chunks at all**
  (not a graph bug — the function never became a node): CommonJS handler
  exports (4× in `auth.controller.js`/`aiController.js`) and Zustand
  factory exports (2× in `useAuthStore.js`/`useCartStore.js`). Diagnosed
  by first surveying every other export style actually present in the
  repo (`module.exports = { foo, bar }` was already handled; ES `export
  const foo = () =>` had zero occurrences and was deliberately left
  unfixed — "fix what's actually observed, not a speculative list").
  Fixed by adding `_js_exports_property_name_node` and
  `_js_factory_call_name_node` (the latter walking up through nested
  `arguments → call_expression` hops, restricted to bare-identifier calls
  with an object-literal arrow body — deliberately excluding
  `items.map((i) => ({...}))` and `useCallback((e) => {...}, [dep])`,
  both confirmed false-positive shapes during testing). Verified by a
  full re-index: NutriForge's graph went from 221 nodes/158 chunks/82
  edges to 227 nodes/164 chunks/102 edges (+6 chunk nodes, +20
  `function_call` edges), with `imports` edges unchanged (36) as
  expected. 154/154 tests passed in the affected test directories (25
  chunker tests, up from 21).
- **A downstream blocker surfaced, not fixed, in the same session**: the
  embedding stage failed loading `microsoft/codebert-base` because the
  installed `torch` (2.4.1) predated `transformers`' CVE-2025-32434 guard
  requiring `torch>=2.6`. Fixed in a separate, explicitly approved
  follow-up (`pip install "torch>=2.6,<3"`), confirmed via a dry-run that
  it changed zero other installed packages.
- **The TypeScript grammar gap is a real, diagnosed, deliberately deferred
  limitation** (not fixed as of this writing): `TreeSitterParser` only
  registers `"python"` and `"javascript"` — a real repro
  (`github/accessibility-scanner-alt-text-plugin`) indexed to "39 nodes, 1
  chunk, 0 edges" because every `.ts` file failed per-file with `unsupported
  language 'typescript'`, caught and logged, not crashed. The downstream
  effect was also verified: `AdjudicateContextBuilder.build()` degrades to
  an empty `changed_functions` list for a diff touching only `.ts` files,
  and the React frontend correctly renders a clear, styled error card
  rather than a wrong or partial review.

**Likely interview questions.**
- *"Why did the JS chunker miss those two patterns, and how did you find
  it?"* A real diagnostic against a live-indexed repo (NutriForge) showed
  a graph edge (`auth.routes.js → auth.controller.js`) that should have
  existed but didn't; tracing backward showed the target function never
  became a chunk at all, because the arrow-function-name detector only
  matched one syntactic shape. Fixed by extending the name-detection
  logic for the two real patterns found, with explicit false-positive
  exclusions verified against real code.
- *"Why is TypeScript not supported yet, and what's the actual blast
  radius of that gap?"* Deprioritized behind the benchmark harness,
  frontend, and deployment work — not forgotten. Concretely: any
  TypeScript-only repository indexes with zero AST chunks/edges, and any
  Adjudicate review touching only `.ts` files degrades cleanly to an
  explicit "couldn't resolve a changed function" error rather than a
  wrong answer — verified live, not just asserted.
- *"Why strict-fail per file on a syntax error instead of best-effort
  partial parsing?"* A partially-parsed file with silently-dropped chunks
  is a worse failure mode than a cleanly-skipped file that's logged and
  visible — the asymmetry with Phase 6's best-effort graph resolution is
  deliberate: syntax correctness must not be guessed at, but structural
  reference resolution tolerates ambiguity by design.

---

## Phase 5 — Semantic Chunking

**What was built.** `ingestion/chunker.py` — three chunk families sharing
one `CodeChunk` dataclass: (1) **AST chunks** (one function/class per
chunk — the only chunks ever embedded or turned into graph nodes), (2)
**sliding-window chunks** (50 lines, 20% overlap — a 40-line step —
covering every file uniformly so nothing is unreachable by retrieval even
in odd/unparseable regions), and (3) **parent chunks** (a ~50-line window
around each AST chunk, used later for small-to-big expansion — the window
algorithm shifts overflow to the far side near file boundaries so a
function near line 1 still gets close to full parent size, and a parent is
never smaller than its own child). Chunk IDs are deterministic:
`UUID5(namespace, f"{file_id}:{name}:{start_line}")`.

**Why built this way.** Deterministic chunk IDs are the single mechanism
that makes re-indexing idempotent: an unchanged function gets the
identical ID on every re-index, so `EmbeddingManager` can skip
re-embedding it entirely (Phase 8) — a real, load-bearing performance
decision, not incidental. Three separate chunk families exist because they
serve different consumers: AST chunks are precise units for retrieval and
the call graph; sliding-window chunks are a safety net for text that
Tree-sitter can't structurally capture; parent chunks exist purely to
support small-to-big expansion at generation time (Phase 15) without
polluting the embedding index with redundant, overlapping vectors.

**Real bugs found and fixed.** None recorded in PROGRESS.md specifically
for the chunker's window-sizing logic; the JS/TS *extraction* bugs that
feed into AST chunking are documented under Phase 4, since the root cause
was in `ast_parser.py`, not `chunker.py`.

**Likely interview questions.**
- *"What is small-to-big retrieval, concretely, and where does it start?"*
  It starts here: every AST chunk gets a parent chunk computed at index
  time and linked via `parent_chunk_id`, so `ContextBuilder` (Phase 15)
  can later substitute a small matched chunk for its wider parent without
  a second retrieval pass.
- *"Why are chunk IDs deterministic instead of random UUIDs?"* So that an
  unchanged function's ID doesn't move across re-indexing runs — this is
  what lets embedding generation, and downstream FAISS/BM25 rebuilds,
  skip redundant work on a re-index (Phase 8).
- *"Why sliding-window chunks at all, if AST chunks are more precise?"*
  AST chunks only exist where Tree-sitter successfully parses structure —
  module-level code, malformed regions, or any language without a
  registered grammar (see Phase 4's TypeScript gap) would be entirely
  unreachable by retrieval without a uniform fallback layer.

---

## Phase 6 — Call Graph Construction

**What was built.** `graph/graph_builder.py::RepositoryGraphBuilder` — one
`nx.DiGraph` with file nodes + chunk nodes, edges for `function_call`,
`method_call`, `inherits`, `imports`, `contains`, and `references` (the
last capturing module-scope wiring like Express's
`router.post("/x", handler)`, which lives outside any function body).
Resolution is **pure name-based heuristic matching, with no type
inference** — a known, explicitly documented limitation, not hidden.
Nothing about graph construction raises on an unresolved reference; it's
logged and skipped.

**Why built this way.** Real symbol resolution (scope-aware, type-aware,
essentially a language server) is a much larger undertaking per language.
Name-based heuristic matching gets the large majority of real call
relationships right at a fraction of the cost, and every failure mode
(same-named methods on unrelated classes, cross-service HTTP calls,
framework-dispatched routing) is documented as a known boundary rather
than silently mis-resolved.

**Real bugs found and fixed.**
- **Zero `imports` edges anywhere in a CommonJS codebase.**
  `graph/import_extractor.py::_extract_javascript_imports` only matched ES
  `import_statement` nodes; `require(...)` (100% of NutriForge's
  `server/`, a plain Express backend) produced zero `imports` edges at
  all. Fixed by adding `_js_require_module_path`, matching a bare
  `require` identifier call with exactly one string argument regardless of
  what contains it (plain, destructured, or chained `.config()` calls),
  reusing the existing `resolve_import` path unchanged. 6 new tests.
- **Module-scope route wiring was invisible to the graph.**
  `_add_call_edges` only scanned chunks of a callable type — Express-style
  `router.post("/x", handler)` lives at module scope (no enclosing
  function) and passes `handler` *by reference*, not as a call, so it
  wasn't even a `CallSite`. Fixed by adding
  `CallExtractor.extract_module_level` (JS only), walking a whole file
  without descending into function/class bodies, producing both
  module-scope `CallSite`s and a new `ReferenceSite`; a new
  `EDGE_TYPE_REFERENCES = "references"` edge type was added, sourced from
  the *file* node (this code belongs to no chunk). Refactored the
  per-node call-site logic into a shared helper so both the existing
  per-chunk path and the new module-scope path use it identically —
  confirmed behavior-preserving (all 57 pre-existing `test_graph` tests
  passed unchanged before any new test was added). 10 new tests.
  Re-indexing NutriForge after both fixes: graph went from 227/102 to
  227 nodes/**170** edges (76 `imports`, was 36; 67 `function_call`; 2
  `method_call`; 25 new `references` edges, spread across nearly every
  `*.routes.js` file — confirming the fix wasn't overfit to one file).
  474/474 tests passed (8 previously-failing Gemini tests started passing
  as a side effect of an unrelated `google-genai` install, not this fix).
- **Independent re-verification, later, with concrete hand-checks (not
  re-trusting the aggregate counts):** hand-verified a file with genuinely
  zero internal relationships (`WorkoutLog.js`, only an external
  `mongoose` import) correctly produced zero edges — a real negative
  result, not a gap. Hand-verified a fan-out file
  (`workout.controller.js`) whose one real relationship
  (`require("../models/Workout")`) was the only edge shown — matched
  exactly. Census of 103/227 isolated nodes: hand-verified 5 deliberately
  chosen to be *likely* to expose a miss — all 5 confirmed correctly
  isolated via `grep` (framework-decorator-dispatched handlers with no
  in-repo caller, genuinely dead exported code, an orphaned page never
  imported anywhere).
- **A known, disclosed name-collision limitation surfaced, not fixed**: a
  caller referencing `login` by bare name resolved to the wrong
  `login` — the client-side Zustand store's method, not the server route
  handler — a coincidental same-name collision inherent to name-only
  resolution, documented as a known heuristic tradeoff rather than a bug
  to chase down.

**Likely interview questions.**
- *"Walk through the call-graph resolution algorithm and its documented
  failure modes."* Pure name-based matching against a symbol table built
  per-file/per-class; failure modes are same-named methods on unrelated
  classes resolving to the wrong target, cross-service HTTP calls
  (invisible to a static graph), and framework-dispatched routing (URL
  routing is a runtime mechanism, not a statically resolvable reference) —
  all explicitly documented as a "known boundary," not silently
  mis-resolved.
- *"Why does the `references` edge type exist, and what does it solve
  that `function_call` doesn't?"* It captures module-scope code that
  passes a function *by reference* rather than calling it (Express route
  wiring being the canonical example) — `function_call`/`method_call`
  only fire for genuine `call_expression` nodes, which a bare identifier
  argument is not.
- *"Why real symbol resolution instead of a full language server?"*
  Cost/benefit — a scope-aware, type-aware resolver is a much larger
  undertaking per language; name-based heuristics get most real
  relationships right cheaply, and every failure mode is documented rather
  than hidden, which is itself defensible engineering judgment for a
  demonstration-scale project.

---

## Phase 7 — SQLite Storage

**What was built.** `database/sqlite_client.py::DatabaseManager` — the
relational source of truth via SQLAlchemy: `repositories`, `source_files`,
`code_chunks` (self-FK `parent_chunk_id` for small-to-big), `graph_edges`
(chunk-only FKs), `embeddings` (raw float32 blobs), `semantic_cache`.
Three independent, deliberately separate fixed UUID5 namespaces exist
(`ingestion/deterministic_ids.py` for file/chunk IDs, a module-local one
in `sqlite_client.py` for `repository_id`) to avoid accidental collision.
`PRAGMA foreign_keys=ON` is set explicitly via a `"connect"` event
listener, since SQLite disables FK enforcement by default.

**Why built this way.** Single-process, single-operator tool → SQLite's
zero-ops, single-file nature matches the deployment story exactly; every
table is scoped by `repository_id` because chunk/file IDs are UUID5-derived
from repo-relative paths, **not globally unique across different repos**
(two repos both containing `src/main.py` hash identically).

**Real bugs found and fixed.**
- **The `graph_store` vs. `sqlite_client` divergence — the single most
  interview-worthy "real bug shape" in the whole persistence layer.**
  There are two independently-named `save_graph`/`load_graph` pairs with
  different semantics: `database.graph_store` (a flat JSON file,
  **lossless** — every node including files, every edge type including
  `imports`) vs. `DatabaseManager.store_graph`/`load_graph` (relational,
  **lossy by construction** — `graph_edges`' FKs point only at
  `code_chunks`, so file nodes and every edge touching one are silently
  skipped on write). Both are named and shaped identically — nothing in
  the type system stops a caller from grabbing the wrong one. This
  surfaced as a real live bug during Phase 20 verification: `/status`'s
  DB-fallback path and `/graph` disagreed on node/edge counts for the same
  repository (184/283 vs 344/255) because they went through different
  loaders. Root cause confirmed by direct DB/file inspection before
  touching code. **Fixed by making both `api/main.py` call sites
  consistently use `database.graph_store.load_graph`** (the full, correct
  source), with `graph.blast_radius.compute_blast_radius`'s own docstring
  updated to state explicitly which loader it expects — a
  documentation-enforced guard, not a compiler-enforced one, because the
  two representations genuinely serve different needs (SQLite's relational
  shape for chunk-level queries; JSON's flat shape for a lossless
  round-trip) and collapsing them would mean either giving SQLite
  nullable polymorphic file/chunk edges or giving JSON a query engine it
  doesn't need. Re-confirmed **still holding** after the Postgres
  migration (Phase 33) — both paths remain distinct, intentionally
  different, Postgres-backed functions, and `api/main.py`/
  `retrieval.graph_retriever.GraphExpander` still call only the lossless
  one.
- **Ordering constraint**: parent chunks must be flushed to the DB before
  AST/sliding chunks, because an AST chunk's `parent_chunk_id` FK must
  already exist — `Session.add()` alone doesn't guarantee insert ordering
  without an explicit `relationship()`, so the code flushes explicitly
  between the two groups.

**Likely interview questions.**
- *"Walk through the `graph_store` vs `sqlite_client` divergence in
  full — what's lost, and why wasn't it just unified into one
  representation?"* See above in full — lead with the real observed
  symptom (184/283 vs 344/255), the root cause (two independently-named,
  identically-shaped loader pairs with different loss characteristics),
  the fix (consistent call-site discipline + docstring guard, not a type
  wrapper), and why unification was rejected (the two representations
  serve genuinely different access patterns).
- *"This seems like a bug waiting to happen — why not just pick one
  representation?"* It already caused exactly the bug you'd predict — and
  the response was to make the fix at the call-site-discipline level
  because the alternative (a type-system-enforced distinction) is weaker
  value in Python's dynamic type system than it sounds, and the two shapes
  really do serve different consumers.
- *"Why must parent chunks be flushed before AST/sliding chunks?"* A
  self-referencing foreign key (`parent_chunk_id`) needs its target row to
  exist first, and SQLAlchemy's `Session.add()` doesn't guarantee insert
  order without an explicit ORM relationship — an explicit flush boundary
  is the simpler fix.
- *"Why does every table scope by `repository_id` even though chunk IDs
  are already UUIDs?"* Because those UUIDs are content-derived (UUID5 of a
  repo-relative path), not globally random — two different repositories
  can and do produce colliding chunk/file IDs, so `repository_id` is the
  only thing making them globally unambiguous.

---

## Phase 8 — Embedding Generation

**What was built.** `embedding/embedding_manager.py::EmbeddingManager` —
only AST chunks are ever embedded (never sliding/parent chunks).
Already-embedded chunk IDs are skipped by default (`force=False`), which
works because chunk IDs are deterministic (Phase 5). Batched at 32
chunks/call, each batch persisted immediately. Default model
`microsoft/codebert-base` (768-dim, code-pretrained), with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim) as a config-toggleable
alternative (`USE_CODEBERT`).

**Why built this way.** CodeBERT was chosen because it's code-pretrained
— appropriate for embedding source-code chunks for retrieval. This choice
has a real, documented downstream cost: CodeBERT is *not* tuned for
natural-language sentence similarity, which is the direct root cause of
the Phase 14 semantic-cache over-matching bug (see below) and a
separately-flagged, still-open concern about natural-language query
ranking quality in general.

**Real bugs found and fixed.**
- **A real environment blocker, found and fixed as an approved follow-up
  to the Phase 4 chunker fix**: `microsoft/codebert-base` failed to load
  because the installed `torch` (2.4.1) predated `transformers`'
  CVE-2025-32434 guard (`torch.load` on non-safetensors weights requires
  `torch>=2.6`). Fixed with a scoped `pip install "torch>=2.6,<3"`,
  explicitly checked via dry-run to avoid cascading downgrades to
  `sentence-transformers`/`transformers` given how far the live
  environment had drifted from `requirements.txt`'s own pins.
- **The root cause behind two later, separately-diagnosed bugs traces
  back to this phase's model choice, not this phase's code**: CodeBERT's
  weak natural-language similarity is the underlying reason the Phase 14
  semantic cache over-matched, and the reason a natural-language chat
  query once failed to retrieve a file whose chunks were real, embedded,
  and graph-connected (a code-flavored query using literal identifiers
  retrieved it correctly instead) — flagged as an open, separate,
  not-yet-fixed retrieval-ranking-quality concern.

**Likely interview questions.**
- *"Why CodeBERT over a general sentence-embedding model, and what did
  that trade-off actually cost you?"* CodeBERT is pretrained on code,
  which is the right target for embedding source-code chunks — but it's
  measurably weak at natural-language sentence similarity, which directly
  caused a real semantic-cache over-matching bug (Phase 14) and a
  documented, still-open natural-language retrieval-ranking concern. The
  `USE_CODEBERT` toggle to MiniLM exists specifically as an escape hatch,
  though it hasn't been adopted as the default.
- *"How does re-indexing avoid re-embedding everything every time?"*
  Deterministic chunk IDs (Phase 5) mean an unchanged function has the
  identical ID across runs — `force=False` skips any chunk ID already
  present in the `embeddings` table.

---

## Phase 9 — FAISS Index

**What was built.** `database/vector_store.py::FaissIndexManager` — one
`faiss.IndexFlatIP` per repository, an **exact**, brute-force
inner-product index (no IVF/HNSW/PQ). Both stored and query vectors are
`faiss.normalize_L2()`'d before touching the index, so inner product on
unit vectors *is* cosine similarity by definition. `_create_index()` is
deliberately factored into its own one-line method as an explicit,
documented extension point for swapping in an approximate index later.
Persisted per repository (`{owner}_{repo}.faiss` + JSON sidecar).

**Why built this way.** Premature optimization avoided deliberately:
per-repository chunk counts are hundreds to low thousands, where exact
search is both fast enough and simpler than tuning an ANN index's
recall/speed trade-off. The normalization trick is the specific reason
`IndexFlatIP` (the fastest FAISS primitive) can be used for cosine
similarity without implementing cosine distance directly.

**Real bugs found and fixed.** None recorded in PROGRESS.md for this
phase specifically.

**Likely interview questions.**
- *"Why `IndexFlatIP` (exact) instead of an approximate index from day
  one?"* At current per-repository scale (hundreds to low thousands of
  chunks), exact brute-force search is faster in practice than the
  setup/approximation overhead an ANN index would justify — this stops
  being true well before monorepo scale, which is exactly why
  `_create_index()` is factored out as a one-method swap point rather than
  hard-coded inline.
- *"How does normalizing vectors turn an inner-product index into cosine
  similarity?"* Inner product of two unit vectors equals the cosine of
  the angle between them by definition — `normalize_L2()` on both stored
  and query vectors makes `IndexFlatIP`'s raw inner product *be* cosine
  similarity, with zero extra computation.

---

## Phase 10 — BM25 Index

**What was built.** `retrieval/tokenizer.py` (hand-built) feeding
`rank_bm25.BM25Okapi` (third-party scoring library). The tokenizer is
camelCase/snake_case-aware: a two-alternative lookaround regex splits
camelCase boundaries (`fetchUser`→`fetch|User`) while keeping acronym runs
intact (`HTTPServer`→`HTTP|Server`, not `HTTPS|erver`); a separate
non-alphanumeric split handles snake_case and punctuation for free.
`fetch_user_by_id` and `fetchUserByID` both tokenize identically to
`["fetch","user","by","id"]`. Persisted per repository
(`{owner}_{repo}_bm25.pkl`).

**Why built this way.** The scoring algorithm itself is a well-understood,
appropriately off-the-shelf library call; what's genuinely hand-built —
and what actually demonstrates understanding — is the tokenizer, because
identifier naming conventions need to be normalized to the same token
stream for either convention to match a query written in the other.

**Real bugs found and fixed.** None recorded in PROGRESS.md; a real
edge-case discipline is baked into the design itself — an empty BM25
corpus is tracked as "indexed, zero documents" rather than constructed,
since `BM25Okapi`'s average-document-length calculation divides by corpus
size and would error on an empty one (Phase 11/general query-optimization
discipline, see `project_description.md` §11.4).

**Likely interview questions.**
- *"How does the camelCase/snake_case tokenizer work, and why does it
  matter for BM25 specifically?"* BM25 is a pure lexical/term-frequency
  method — if `fetchUser` and `fetch_user` tokenize differently, BM25 can
  never match a query in one convention against code written in the
  other. The regex-based splitter normalizes both to the same token
  stream, verified with a directly-testable, hand-verified property.
- *"Why hand-build the tokenizer but not the BM25 scoring algorithm
  itself?"* BM25's math (IDF, term-frequency saturation, length
  normalization) is a standard, well-specified algorithm where a library
  call demonstrates nothing extra; the tokenizer is the actually
  code-specific, non-generic piece worth building by hand.

---

## Phase 11 — Hybrid Retrieval (RRF)

**What was built.** Reciprocal Rank Fusion combining FAISS (dense) and
BM25 (sparse) result lists: `score = Σ 1/(rrf_k + rank)`, `rrf_k = 60`,
using only rank position from each retriever's own list, never the raw
score.

**Why built this way.** FAISS cosine similarity (`[-1,1]`) and BM25's
unbounded term-frequency scores live on two incomparable scales — RRF
sidesteps normalization entirely by using only rank, at the cost of
discarding magnitude information a learned/hand-tuned weighted sum could
in principle exploit. This project's own ablation infrastructure
(`evaluation/ablation.py`) exists specifically to measure whether that
trade-off is actually costing anything, though a hand-tuned alternative
has not been built or measured.

**Real bugs found and fixed.** None recorded in PROGRESS.md for this
phase specifically.

**Likely interview questions.**
- *"Explain RRF and why it's rank-based rather than score-based."* Two
  retrievers' raw scores aren't comparable on the same scale; RRF uses
  only each retriever's own rank ordering, summed with a smoothing
  constant (`rrf_k=60`) that dampens the impact of small rank differences
  at the tail of each list.
- *"Why not just hand-tune dense 0.7 / sparse 0.3 and validate it?"* That
  requires the two scores to already be comparable, which they
  fundamentally aren't, and requires re-tuning per corpus/domain — RRF's
  "works reasonably well with zero tuning" property matters more given
  this project doesn't have labeled relevance data at the scale a learned
  weight would need. The infrastructure to actually measure this
  trade-off (`evaluation/ablation.py`) exists; the experiment itself
  hasn't been run.

---

## Phase 12 — Graph Expansion

**What was built.** `retrieval/graph_retriever.py::GraphExpander` — walks
one hop from each hybrid-retrieval hit, with per-hop score decay.
Critically: every chunk found by hybrid retrieval sorts ahead of every
graph-discovered chunk regardless of numeric score (a two-tier sort key,
not a single blended score) — structural neighbors augment the result
set, never displace a genuinely strong lexical/semantic hit. Also
broadens: "an imported file's every chunk is a neighbor."

**Why built this way.** The two-tier sort key is a deliberate design
choice to prevent graph expansion from silently degrading precision — a
chunk that's structurally *related* but wasn't independently a strong
match should never outrank one that genuinely was.

**Real bugs found and fixed.** None recorded in PROGRESS.md for this
phase specifically. (Phase 22's `compute_blast_radius` deliberately
diverges from this module's traversal shape for a different consumer —
see Phase 22 for why.)

**Likely interview questions.**
- *"What does the 'imported file's every chunk is a neighbor' expansion
  behavior actually do, and what's the coarseness trade-off?"* Any chunk
  in a file that's imported by a hit's file becomes a graph-expansion
  candidate — coarse (an imported file might have 30 unrelated chunks),
  but cheap and directionally useful; the two-tier sort ensures this
  coarseness can never outrank a genuine hybrid-retrieval match, only
  supplement it.
- *"How does `compute_blast_radius` differ from this module's traversal,
  despite sharing the same underlying shape?"* Both are bidirectional,
  hop-bounded frontier traversals over the same graph, but
  `GraphExpander` includes retrieval-only decay scoring and the
  "imported file → all its chunks" broadening, while `compute_blast_radius`
  (Phase 22) deliberately skips both — Adjudicate wants "real
  callers/callees" specifically, and the broadening would make a
  highlighted blast radius unreadable for any node in a heavily-imported
  file.

---

## Phase 13 — Cross-Encoder Re-ranking

**What was built.** `retrieval/reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2`
(via `sentence-transformers.CrossEncoder`) jointly scores (query,
candidate) pairs for the ~40 candidates hybrid retrieval + graph
expansion already narrowed down to, batched at 32 pairs/call, cutting to
the final top ~8.

**Why built this way.** A cross-encoder reads query and candidate jointly
(far more accurate than bi-encoder cosine similarity) but is too slow to
run over an entire corpus — so it only ever touches the small candidate
set cheaper stages already produced, which is the standard "cheap recall,
expensive precision" retrieval funnel pattern, hand-implemented here
rather than imported.

**Real bugs found and fixed.** None recorded in PROGRESS.md for this
phase specifically.

**Likely interview questions.**
- *"Why does the cross-encoder only ever see ~40 candidates instead of
  the whole corpus?"* Cross-encoders are O(n) full joint-attention passes
  per candidate — far too slow to run against a full corpus per query;
  restricting it to what cheap dense+sparse+graph retrieval already
  narrowed down is what makes the accuracy gain affordable.

---

## Phase 14 — Semantic Cache

**What was built.** `retrieval/semantic_cache.py::SemanticCacheManager`
— persisted in the same SQLite/Postgres database (`semantic_cache`
table), not Redis, not an in-memory LRU. `lookup()` runs *before* hybrid
retrieval, graph expansion, or reranking even start; a hit skips the
entire retrieval+generation pipeline. Embeds the incoming query with the
same model used for chunk embeddings, computes cosine similarity against
every cached query for that repository, returns the highest-similarity
entry above `CACHE_SIMILARITY_THRESHOLD` (default 0.95).

**Why built this way.** No Redis, because this is a single-process,
single-machine tool with exactly one writer — a shared distributed cache
solves a problem that doesn't exist yet at this scale (see Phase 33's
scalability discussion for what *would* change this).

**Real bugs found and fixed.**
- **A real, genuinely un-fixable-by-threshold over-matching bug —
  first flagged in Phase 19, never actually investigated until it visibly
  broke real usage.** "What is the backend tech stack?" and "What is the
  frontend tech stack?" returned the identical cached answer. Measured
  directly against the real model: cosine similarity **0.9974** — clears
  the 0.95 threshold by a wide margin. A broader 7-pair measurement made
  the scale of the problem explicit and decisive: the range of scores for
  genuine near-duplicates (0.9861–0.9925) sat **entirely inside** the
  range for genuinely distinct questions (0.9807–0.9987) —
  "register endpoint" vs. "login endpoint" (should miss) scored *higher*
  (0.9987) than an actual paraphrase pair (should hit, 0.9861). Decisive
  finding: `min(should-hit) > max(should-miss)` is **False** — no single
  cosine threshold, at any value, separates these. Root cause: CodeBERT
  (Phase 8's deliberate, correct-for-code choice) is not tuned for
  natural-language sentence similarity, and compresses short,
  similarly-structured NL questions into a nearly degenerate similarity
  band regardless of actual meaning. **The fix is a second, independent
  gate applied on top of the cosine check**, `_is_lexically_compatible()`:
  (1) a small, explicit, extensible list of distinguishing-term categories
  (backend/frontend, login/logout/register, HTTP verbs, read/write,
  dev/prod, sync/async) — a conflict fires only on positive, two-sided
  evidence (both queries name a *different* alternative from the same
  category); (2) content-word Jaccard similarity with a 0.5 floor, needed
  *in addition to* category-matching because "backend" vs. "frontend"
  alone measured exactly 0.5 Jaccard — right at the boundary, too close
  to trust alone. Explicitly disclosed as **not a general antonym/
  word-sense solution** — a targeted, extensible fix for the reported bug
  class, not a claim of completeness. 14 new tests (11 pure-function, 3
  integration against real near-identical fake vectors proving query
  *text*, not the embedding, now forces the miss). 3 pre-existing tests
  had to be fixed because their placeholder query text ("cached query"/
  "new query") legitimately failed the new Jaccard gate — fixed by giving
  them real, lexically-plausible text rather than loosening the gate.
  Verified live against the real model, not just fake-vector unit tests:
  a frontend query after a cached backend answer correctly missed;
  the exact same question asked twice still correctly hit
  (similarity ≈1.0). Full suite: 688 passed (was 674).
- **A separate, related bug found later during frontend work**: a real
  `/query` call crashed with `ValueError: shapes (768,) and (384,) not
  aligned` inside cosine-similarity — stale `semantic_cache` rows from an
  earlier session using a 384-dim model (MiniLM) were being compared
  against current 768-dim CodeBERT vectors. Fixed by clearing the 5 stale
  rows (a data cleanup, not a code change); flagged as a real, unfixed
  gap: nothing currently invalidates cached query embeddings when the
  configured embedding model changes.
- **Re-verified after the Postgres migration (Phase 33)**: the same
  backend/frontend lexical-gate regression pair re-run against the real
  Postgres-backed cache — correctly missed (0.9963 cosine, rejected by
  the lexical gate) and correctly hit (exact repeat, similarity=1.0) —
  confirming the fix's logic is storage-backend-independent.

**Likely interview questions.**
- *"Walk through the semantic cache over-matching bug end to end — why
  couldn't raising the threshold fix it?"* This is one of the strongest
  stories in the project — walk through the measured 7-pair table showing
  should-hit and should-miss ranges overlap entirely, then the two-part
  lexical gate, then the explicit disclosure that it's a targeted fix, not
  a general solution.
- *"Why is there no Redis-backed cache?"* Single-process, single-writer
  tool — a distributed cache solves a consistency problem that doesn't
  exist yet. If this became a multi-replica service, both the semantic
  cache and the in-memory indexing-status registry would need to move to
  a shared store, and Redis would be the natural choice there (see Phase
  33).
- *"What would you do differently, or what's the known limitation here?"*
  The distinguishing-term category list is deliberately narrow — a
  confusable pair outside it (e.g. two lexically-near-identical but
  semantically different proper nouns) could still over-match; and nothing
  invalidates cached embeddings when the configured embedding model
  changes, which is what caused the separate 768-vs-384-dim crash.

---

## Phase 15 — Context Builder

**What was built.** `generation/context_builder.py::ContextBuilder` —
small-to-big context assembly: substitutes a small matched chunk for its
wider parent before sending anything to the LLM, but always keeps the
single highest-ranked block *whole* even if it alone exceeds the token
budget; lower-ranked blocks are dropped wholesale (never truncated
mid-chunk) to make room. Token budget (~4000 tokens) is enforced via a
fixed ~4-chars-per-token heuristic, not a real tokenizer.

**Why built this way.** "Never collapse the context to nothing" is an
explicit design choice — truncating the top-ranked block mid-function
would risk cutting off exactly the code the answer most depends on; a
fixed chars-per-token heuristic avoids loading a full BPE tokenizer just
to estimate a budget, which only needs to be approximately right, not
exact.

**Real bugs found and fixed.**
- **A real bug found during Phase 24 (Adjudicate) integration, but
  belonging to this module**: `GET /context`'s `chunk_citations` never
  contains the raw enclosing AST chunk when `USE_SMALL_TO_BIG` is on (the
  default) — every match is substituted for its parent chunk, which is
  never a graph node. This silently broke Adjudicate's blast-radius
  lookup (every call 404'd) until a new `matched_chunks` field (raw,
  pre-substitution matches) was added to the API response — `ContextBuilder`
  itself was correct; the gap was that no existing consumer had ever
  needed the raw, pre-substitution match list until Adjudicate did.

**Likely interview questions.**
- *"What is small-to-big retrieval, and how does `ContextBuilder` decide
  when a chunk gets substituted for its parent?"* Every retrieved AST
  chunk has a precomputed parent (Phase 5); at generation time, the
  smaller precise match is swapped for the wider parent to give the LLM
  more surrounding context, but the substitution happens per-chunk within
  the existing token budget, never expanding past it for a lower-ranked
  item.
- *"Why does `/context` return both `chunk_citations` and
  `matched_chunks`?"* Because small-to-big substitution means the
  "citation" shown to a chat user (parent-substituted) is a different
  object from the "raw match" a structural consumer like Adjudicate's
  blast-radius lookup needs (which must be a real, unsubstituted AST
  chunk that's actually a graph node) — this distinction was invisible
  until Adjudicate's Phase 24 integration exposed it as a real bug.

---

## Phase 16 — LLM Integration

**What was built.** `generation/llm_client.py::LLMClient` — a static,
config-driven provider priority (not retry-on-failure): Gemini →
Groq → Ollama → raise, checked once per call. `prompts/templates.py`'s
system prompt enforces cite-or-refuse behavior mechanically reinforced
elsewhere in the pipeline: answer only from supplied context, cite every
claim in a fixed format, explicitly say so if context is insufficient. If
retrieval returns nothing, the LLM is never even called —
`generate_answer()` short-circuits to a fixed refusal message with zero
token cost, and nothing is written to cache.

**Why built this way.** A runtime try/catch fallback would silently
degrade answer quality/latency on a transient error without anyone
noticing which provider actually served a request. A static, explicit
priority (logged, and reflected in `/info`) makes provider selection
observable rather than an implicit runtime decision — a real, found-live
bug (below) is a direct consequence of taking this distinction seriously.

**Real bugs found and fixed.**
- **The Gemini "thinking token" truncation bug — the single most
  interview-relevant LLM-plumbing bug in the project, reproduced twice,
  independently, in two different agents.** Gemini 2.5's
  `max_output_tokens` budget is **shared with the model's internal
  "thinking" tokens**. First reproduced against the Judge agent (Phase
  30): 6 consecutive real attempts against a long transcript all
  truncated at the identical output length; `usage_metadata` confirmed
  `finish_reason=MAX_TOKENS`. Reproduced *again*, independently, against
  the Prosecutor (a much later session, against a real `mkocabas/VIBE`
  diff): raw response text was cut off mid-string (`'"location": "lib/data_'`);
  the raw `google-genai` response object confirmed
  `thoughts_token_count=979` of a `max_output_tokens=1024` budget — 979
  tokens went to invisible reasoning, leaving 28 for the actual JSON, far
  short of even one complete claim object. Crucially: the existing
  retry loop only appends a corrective note to the prompt; it never
  increases the token budget, so the *same* truncation point recurred on
  every retry regardless of attempt count — retries alone could never
  have fixed this. **The fix has two parts, applied at the root cause**:
  (1) `thinking_config=ThinkingConfig(thinking_budget=0)` is set **only**
  when a `response_schema` is supplied — schema-constrained extraction
  doesn't need chain-of-thought, while free-text completions (the
  Defender's justification, the rebuttal loop) are untouched; (2)
  `LLM_MAX_TOKENS` was raised from 1024 to 4096 as additional headroom
  (not the primary fix — disabling thinking already frees the whole prior
  budget, but a full multi-claim JSON array with real code snippets can
  legitimately need more). The Judge separately overrides this to
  `JUDGE_MAX_TOKENS = 2048`, since its own long multi-round transcript
  prompt is longer than the single-turn Defender/Prosecutor prompts the
  global default was already proven against. A JSON-repair/sanitization
  alternative was explicitly considered and **rejected**: a genuinely
  truncated response has no recoverable content past the cut point, and
  fabricating claim text the model never produced directly conflicts with
  the Prosecutor's whole grounded-claims design. Verified live, real
  Gemini calls, on the exact diff that failed 3/3 before the fix:
  succeeded on the first attempt post-fix. Verified no regression on a
  known-good diff (colorama's `reset_all`) producing a legitimate empty
  claims array, not a parse failure. Permanent diagnosability was added
  too: the retry loop's warning log now includes the raw response text,
  so a future parse failure of any kind is debuggable from logs alone.
- **A real, found-live bug in the display-only provider-name logic**: the
  `/info` endpoint's provider-name display once missed the Groq branch
  entirely (referenced directly in `project_description.md`'s design
  rationale for why static-priority provider selection matters — makes
  provider choice observable and therefore auditable).
- **A real regression in the test suite, exposed by a live provider
  switch**: four pre-existing `LLMClient` tests constructed clients with
  `use_gemini=False, use_ollama=True` without pinning `use_groq=False`,
  silently falling back to whatever the real `.env`'s `USE_GROQ` said —
  once Groq was flipped on in the real environment, three of these tests
  started making *real* live API calls instead of exercising the fake
  Ollama client they were written for. Fixed by explicitly pinning
  `use_groq=False` in all four — a latent test-isolation fragility that
  existed before Groq was even added, exposed only once something
  finally triggered it.
- **A real cross-provider structured-output gap, Groq-specific**: Groq's
  (and any OpenAI-compatible) `json_object` mode can only constrain the
  top-level response to an *object*, never a bare array — asked for "a
  JSON array of claims," Groq wrapped it the only way its API allows:
  `{"claims": [...]}`. Fixed in `parse_claims` by unwrapping a top-level
  object only when unambiguous (exactly one key, whose value is a list);
  anything else still rejects exactly as before.

**Likely interview questions.**
- *"Explain the Gemini thinking-token truncation bug in full — root
  cause, symptom, and both parts of the fix."* This is a marquee
  question — walk through both independent reproductions (Judge, then
  Prosecutor), the `usage_metadata`/`finish_reason` diagnostic evidence,
  why retries alone couldn't fix it, and the two-part fix
  (`thinking_budget=0` gated on `response_schema`, plus a raised token
  ceiling), plus the explicitly-rejected JSON-repair alternative and why.
- *"Why is provider selection a static priority rather than a
  retry-on-failure fallback chain?"* A silent runtime fallback hides
  which provider actually served a request, which matters for debugging
  quality/latency regressions — a static, logged, observable priority
  makes provider choice an explicit, auditable decision instead.
- *"Why does the system never call the LLM at all when retrieval returns
  nothing?"* Calling an LLM with empty context either produces a
  hallucinated answer or an expensive no-op refusal — short-circuiting to
  a fixed message costs zero tokens and is strictly more honest.
- *"How does citation matching work, and why is it substring-based rather
  than a strict parser?"* `match_citations()` treats a citation as
  "present" if the raw answer text contains the citation's file path (and
  function/class name) as a plain substring — a robustness choice, since
  models don't always reproduce the exact requested punctuation/format.

---

## Phase 17 — Evaluation Framework

**What was built.** `evaluation/ablation.py` — a five-way ablation study
(Dense Only → +BM25 → +Graph Expansion → +Cross-Encoder → Full Pipeline)
over the same evaluation dataset, measuring RAGAS faithfulness/answer
relevancy/context precision, LLM-judge Precision@5, and average latency
per configuration. `USE_SEMANTIC_CACHE` is deliberately excluded from the
ablation dimensions — a cache hit would silently reuse a *previous
configuration's* answer, corrupting per-configuration metrics.
`ragas` itself is **intentionally not pinned** in `requirements.txt`,
imported lazily only inside the one function that needs it.

**Why built this way.** At implementation time, the latest `ragas`
release failed to import at all against a `langchain-community` module it
depended on that had since been removed, and the last known-compatible
older release requires downgrading `langchain`/`openai` to versions that
conflict with the rest of the environment. Rather than pin a broken
dependency or abandon RAGAS scoring, every other evaluation module stays
fully importable and unit-testable without `ragas` installed at all — a
deliberate, documented isolation of one unstable third-party dependency,
not a silent gap.

**Real bugs found and fixed.** None recorded specifically for this phase
in PROGRESS.md beyond the disclosed `ragas` import/dependency-conflict
issue itself, which is a documented environment limitation rather than a
bug in this project's own code.

**Likely interview questions.**
- *"Why is `ragas` not pinned in `requirements.txt`, and how does the
  rest of the evaluation module stay testable without it?"* The current
  release chain has a real, upstream, unresolved dependency conflict
  (`langchain-community` removal vs. an older compatible `ragas` release
  needing `langchain`/`openai` downgrades) — rather than force a
  system-wide downgrade or drop RAGAS scoring entirely, the import is
  lazy and scoped to exactly the one function that needs it, so nothing
  else in `evaluation/` depends on it being installed.
- *"Why is `USE_SEMANTIC_CACHE` deliberately excluded from the ablation
  dimensions?"* A cache hit returns a *previous* configuration's answer
  verbatim — mixing that into a "what does configuration X actually
  produce" measurement would corrupt every downstream metric for that
  configuration.
- *"What's the actual measured contribution of graph expansion in the
  ablation results, and how would you find out if you didn't already
  know?"* Run `evaluation/ablation.py` and compare the +BM25 row against
  the +Graph Expansion row's faithfulness/Precision@5/latency deltas —
  the harness exists precisely so this isn't a guess.

---

## Phase 18 — Streamlit UI

**What was built.** `app.py` + `ui/` — the original, legacy dev UI, later
superseded by (but still coexisting alongside) the React frontend
(Phase 32). `ui/pipeline_service.py` originally implemented the exact
index/query wiring that later moved into `pipeline.py` (Phase 19).
`streamlit-agraph` (vis.js under the hood) was used for the graph tab
added in Phase 21.

**Why built this way.** Streamlit was the fast, low-effort path to a
working UI while the core retrieval pipeline was still being built —
`streamlit-agraph` was chosen over `pyvis`/`networkx+matplotlib`
specifically because `agraph()` returns a clicked node's ID directly as a
Python value, which the alternatives don't offer out of the box.

**Real bugs found and fixed.** No dedicated PROGRESS.md entry exists for
this phase's original build; the two `streamlit-agraph`-related bugs (the
`groups={}` validation-error fix, and the stale-process false-negative
during Phase 22 verification) are documented under Phase 21/22, since
they surfaced during the graph-tab work, not this phase's own build.

**Likely interview questions.**
- *"Why does this project have two frontends?"* Streamlit was the fastest
  path to a working UI while the backend was still under active
  development; the React frontend (Phase 32) was built once the backend's
  API contract and Adjudicate's event shapes were stable enough to justify
  the investment. Both call the identical FastAPI backend — `ui/api_client.py`
  and `frontend/src/api/client.js` are the JS/Python equivalents of the
  same HTTP client.
- *"Was Streamlit ever going to be the final UI?"* No — it's explicitly
  the legacy/dev UI, kept alive because it still works and costs nothing
  to maintain, not because it was ever the target production experience.

---

## Phase 19 — Pipeline Wiring

**What was built.** `pipeline.py::Pipeline` — the single call-order
authority for indexing (`index_repository`: repo mgmt → file discovery →
AST parse → chunk → call graph → SQLite → embeddings → FAISS → BM25) and
querying (`query`: semantic cache → hybrid retrieval RRF → graph expansion
→ reranker → context builder → LLM generation). `cli.py` — a thin
argparse CLI (`index <repo_url>`, `query <repo_id> "<question>"`) calling
`Pipeline` directly. This replaced an older, narrower `IndexingPipeline`
(Phase 5-only scope) and `ui/pipeline_service.py`'s duplicate
implementation.

**Why built this way.** Prior to this phase, indexing/query wiring
existed twice (once in the Phase-5-only `IndexingPipeline`, once in
`ui/pipeline_service.py` for Streamlit) — consolidating into one
`Pipeline` class, imported directly by both the CLI and the Streamlit app,
eliminated duplicated call-order logic that could drift apart.

**Real bugs found and fixed.**
- No new bugs introduced by the consolidation itself (verified via the
  full 471-test suite passing, plus a real re-index and 3 fresh CLI
  queries against `tartley/colorama` with citations manually confirmed
  against the real cloned repo via `grep`).
- **First flagged the semantic cache over-matching issue** (later fixed
  properly in Phase 14's dedicated entry, months later, once it visibly
  broke real usage) — `USE_SEMANTIC_CACHE` had to be manually disabled
  during this phase's own manual query verification to get distinct
  fresh answers, an early, correctly-logged signal of the bug that wasn't
  actually root-caused until much later.

**Likely interview questions.**
- *"Why was the old `IndexingPipeline` replaced rather than extended?"*
  Its scope (Phase 5-only: acquire/discover/parse/chunk) was a strict
  subset of the new `index_repository` — extending it in place would have
  meant maintaining two overlapping wiring implementations
  (`IndexingPipeline` and `ui/pipeline_service.py`) that had already begun
  drifting; replacing both with one `Pipeline` class removed the
  duplication at its source.
- *"What's the single call-order authority for indexing, and why does
  that framing matter?"* `Pipeline.index_repository` — the point being
  that *nothing outside it* decides what order retrieval/indexing stages
  run in; both the CLI and the API layer call into the same ordering
  logic rather than each re-implementing it.

---

## Phase 20 — FastAPI Backend

**What was built.** `api/main.py` — the single HTTP orchestration layer
for the entire system, wrapping `Pipeline` behind endpoints:
`POST /repos/index` (202, `BackgroundTasks`-driven, `repo_id` computed
up front so it's returned before cloning even starts), `GET /repos/{id}/status`
(polled every 1s by both frontends, in-memory state with a SQLite/Postgres
reconstruction fallback if the process restarted), `POST /repos/{id}/query`
(threadpool + 120s timeout), `GET /repos/{id}/context?file=&line=`,
`GET /repos/{id}/graph`. A request-logging middleware times every request.

**Why built this way.** There is deliberately **no separate Node/Express
orchestration layer** — see the closing thesis section below for the full
argument; in short, every piece of "business logic" in this system (validate
a URL, run retrieval, build context, call an LLM) *is* the ML work, so
splitting orchestration from ML work into two languages/processes would
add a network hop and a second dependency surface for no corresponding
benefit.

**Real bugs found and fixed.**
- **The `graph_store` vs. `sqlite_client` divergence, first surfaced
  live here.** `/status` reported `graph_nodes=184/graph_edges=283` for
  colorama while `/graph` reported `344/255` for the identical
  repository. Root cause and fix are the full Phase 7 bug entry above —
  both `/graph` and `/status`'s DB-fallback path were calling
  `DatabaseManager.load_graph` (lossy — reconstructs a node for every
  *stored chunk*, all 344, instead of just the 161 AST chunks the graph
  builder actually used, and drops the 28 edges touching file-level
  nodes) instead of `database.graph_store.load_graph` (the exact JSON
  `save_graph` wrote from the real in-memory graph). Fixed both call
  sites; all three paths (live in-memory `/status`, DB-fallback `/status`,
  `/graph`) confirmed to agree afterward: 184 nodes/283 edges.
- **A real deviation, flagged and approved, not silently added**: the
  `/context` endpoint needed a `RankedChunk` for an arbitrary file:line
  lookup, but no existing `RetrievalSource` enum value fit "found by
  direct location lookup, not retrieval" — added
  `RetrievalSource.LOCATION` (purely additive, no existing member/logic
  changed).
- Verified end-to-end via a real Playwright-driven browser session
  against both `uvicorn api.main:app` and `streamlit run app.py`: the
  `st.status` block visibly grew a second progress line between two
  screenshots ~1s apart, proving indexing genuinely polls the HTTP API
  asynchronously rather than blocking on an in-process call.

**Likely interview questions.**
- *"Why is there no separate Node/Express orchestration layer in this
  architecture?"* See the closing thesis section — restated briefly: no
  orchestration concern here is separate from the ML work itself, so a
  polyglot split would cost a network hop and a second dependency surface
  for zero corresponding benefit at this system's actual scale (no auth,
  no multi-tenant business rules).
- *"Walk through the full status-code decision tree in
  `_repomind_error_status()` — why does `RetrievalError` map to 404 and
  not 500?"* See Phase 1/7 — `RetrievalError` represents "nothing found
  for a valid request," a client-facing not-found condition, distinct
  from a genuine server malfunction (`DatabaseError`, `EmbeddingError` →
  500). `504` is handled entirely separately, only for a real
  `asyncio.TimeoutError`.
- *"What happens to an in-flight indexing job if the process restarts?"*
  The in-memory `_index_state` registry is lost, but `/status` falls back
  to reconstructing a "ready" state from the database if indexing had
  actually completed — explicitly documented as "not a perfectly accurate
  progress replay" for jobs that were genuinely in-flight at restart time,
  a real, disclosed scalability limitation (see Phase 33).
- *"Why is `/repos/{id}/review` a POST endpoint even though it streams
  SSE?"* Covered fully under Phase 32 — `EventSource` is GET-only, and
  the diff payload needs to go in a request body.

---

# Part B — Adjudicate: adversarial multi-agent code review

## Phase 21 — Graph Rendering Service

**What was built.** `ui/graph_view.py` — fetches `GET /repos/{repo_id}/graph`
and renders via `streamlit_agraph.agraph()`. Clustering: each chunk node
gets `group=file_path` (vis.js's automatic per-group coloring, no custom
clustering logic); file nodes get a fixed dark color/square shape as
structural anchors; edges colored/weighted by `edge_type` (calls bold
blue, inherits dashed purple, imports/contains light and thin).

**Why built this way.** `streamlit-agraph` was chosen (Phase 18) because
`agraph()` returns a clicked node's id directly as a usable Python value —
matplotlib produces a static image with zero click events, and pyvis's
generated HTML has no built-in bridge back into Streamlit's session state.

**Real bugs found and fixed.**
- **A real, browser-console-visible bug**: `streamlit_agraph.Config`
  defaults `groups` to `None`, which vis.js's option validator rejects
  ("Invalid type received for 'groups'. Expected: object. Received
  null."). Fixed by passing `groups={}` explicitly — confirmed the
  per-node `group` coloring was already working correctly regardless
  (driven by each node's own `group` property, not the network-level
  override), so the fix only silenced the invalid-null validation error,
  it didn't change any rendering behavior.
- Verified end-to-end via Playwright: indexed `tartley/colorama`, graph
  caption read "184 node(s), 283 edge(s)" (matching Phase 20's fixed
  `/graph` count exactly), clicked a node via grid-search over canvas
  coordinates (vis.js draws to a `<canvas>`, no per-node DOM elements),
  and confirmed the returned node data (`is_a_tty`, line 9-10) against
  the real cloned file via direct inspection.
- **A disclosed, unfixed legibility limitation**: at 184 nodes/23 files,
  vis.js's default group-color palette has ~20 distinct colors, so some
  files necessarily share a color — flagged as not worth fixing for a
  first validation pass, though physics-based spatial separation still
  keeps visually-adjacent clusters apart even when their colors collide.

**Likely interview questions.**
- *"Why was `streamlit-agraph` chosen over `pyvis`/`networkx+matplotlib`?"*
  It's the only one of the three that returns a clicked node's data
  directly as a Python value usable in Streamlit's own session state —
  matplotlib has zero interactivity, pyvis has no built-in bridge back.
- *"What's the known legibility limitation at this node count, and why
  wasn't it fixed?"* ~20 distinct palette colors for 23+ files means some
  files necessarily collide — judged as over-engineering to build a full
  distinct-color-assignment scheme for a first validation pass; worth
  revisiting for a much larger repo (later actually became relevant, see
  Phase 32's folder-based recoloring work).

---

## Phase 22 — Blast-Radius Highlighting

**What was built.** `graph/blast_radius.py::compute_blast_radius(graph,
node_id, hops=2)` — bidirectional, hop-bounded frontier traversal over an
already-loaded graph, returning the induced subgraph. `GET /repos/{repo_id}/graph`
gained optional `focus_node`/`hops` query params (no new endpoint); the
same already-loaded graph is passed through `compute_blast_radius` before
serializing.

**Why built this way.** Deliberately reuses `GraphExpander`'s traversal
*pattern* (Phase 12), not the class itself — skips the retrieval-only
decay scoring and the "imported file's every chunk is a neighbor"
broadening, because Adjudicate's actual consumer (Phase 24) wants "real
callers/callees," and that broadening would make the highlighted set
unreadable for any node in a heavily-imported file.

**Real bugs found and fixed.**
- **A verification-process bug, not a code bug, caught before it could
  produce a false pass**: initial Playwright runs showed no "Blast
  radius" UI section at all, no error either — traced to several stale
  duplicate `streamlit`/`uvicorn` processes left running on the same
  ports from earlier in the session, so the browser was hitting an old
  process without this session's code changes. Fixed by killing every
  stale process and confirming a single clean listener per port before
  re-verifying — a real lesson about environment hygiene, not a logic
  fix.
- Verified against Phase 4/6's already-confirmed ground truth (not a
  fresh guess): every one of 16 returned edges for a live blast-radius
  query matched an edge already hand-verified earlier in the same
  diagnostic chain — nothing new or unexplained.
- **An independent re-verification pass, in a later session, per explicit
  instruction not to trust the prior log at face value**: reproduced the
  15-node/16-edge blast radius directly from the persisted graph JSON
  (not through any cached result) and cross-checked every edge against a
  fresh `grep` of the real cloned source — all matched. The live-UI-click
  check could **not** be completed this time (no browser automation tool
  available in that session) — explicitly flagged as an open gap rather
  than assumed to still hold, and the roadmap's ✅ was left resting on
  the original Playwright-verified pass, not this partial one.

**Likely interview questions.**
- *"Explain `compute_blast_radius`'s algorithm and how it differs from
  `GraphExpander.expand`'s traversal, despite sharing the same underlying
  shape."* Both are bidirectional, hop-bounded frontier BFS over the same
  `nx.DiGraph`; `compute_blast_radius` deliberately omits retrieval-only
  decay scoring and the imported-file broadening that `GraphExpander`
  needs for retrieval recall but that would make a highlighted blast
  radius noisy and unreadable for Adjudicate's purpose.
- *"How do you know verification claims in a project like this are real
  and not just 'it produced output'?"* This phase is a good concrete
  example: results were cross-checked against already-established ground
  truth from a *separate*, earlier diagnostic chain (Phase 4/6), not
  freshly asserted — and a later independent session explicitly
  re-verified from scratch and disclosed exactly which part it could and
  couldn't re-confirm, rather than blindly trusting the prior log.

---

## Phase 23 — Adjudicate Project Scaffold

**What was built.** The `adjudicate/` package skeleton: `agents/base.py`
(`BaseAgent` ABC, one abstract `review` method — deliberately unused by
every later concrete agent, see below), empty docstring-only stubs for
`defender.py`/`prosecutor.py`/`judge.py`/`documentation.py`,
`config.py::AdjudicateSettings` (per-role Gemini model override, provider
selection deliberately *not* duplicated — reuses the core system's
already-configured `USE_GEMINI`/`USE_OLLAMA`), and
`repomind_client.py::RepoMindClient` — a thin `httpx` wrapper and the
**only** way Adjudicate ever touches RepoMind (a one-way dependency
enforced as a design invariant: RepoMind never imports from `adjudicate/`).

**Why built this way.** The one-way HTTP-only dependency exists so that
Adjudicate is provably a layer *on top of* RepoMind, never entangled with
its internals — `RepoMindClient` accepts an injected `httpx.Client` (the
same dependency-injection style `LLMClient` already uses for its provider
clients), which is what made real, non-mocked verification possible
without a long-lived server process.

**Real bugs found and fixed.** None recorded — this phase is pure
scaffolding, verified via a real transient `uvicorn` server started and
cleanly shut down in-process (confirmed via `netstat` showing `TIME_WAIT`,
not a live listener — avoiding the exact stale-process trap Phase 22's
log had just flagged).

**Likely interview questions.**
- *"Why doesn't any concrete agent subclass `BaseAgent`?"* Every real
  agent (Defender, Prosecutor, Judge) needs more than one input — the
  Defender needs both the context bundle *and* the raw diff, since only
  the diff shows what specifically changed within a named function.
  Forcing that into `BaseAgent.review(context: Any)`'s single argument
  would mean inventing a wrapper type purely to fit an interface no
  orchestrator ever actually drives through polymorphically — so
  `BaseAgent` was left deliberately unused rather than forced to fit,
  flagged explicitly each time rather than silently ignored.
- *"Why is `RepoMindClient` HTTP-based rather than a direct Python
  import of `Pipeline`?"* Enforces the one-way dependency as a structural
  fact, not just a convention — Adjudicate genuinely cannot reach into
  RepoMind's internals even by accident, and it also matches the real
  deployment shape (Adjudicate's orchestrator calling the same FastAPI
  backend the frontend does).

---

## Phase 24 — Context Builder Integration

**What was built.** `adjudicate/context_builder.py` — `parse_diff()` (a
minimal unified-diff parser) and `AdjudicateContextBuilder.build(diff_text)
→ ContextBundle`. For each parsed location: finds the enclosing AST chunk,
walks the blast radius 1-hop for callers/callees, tags test-like callers
into `related_tests`. **Hybrid retrieval is invoked only when zero
test-like callers were found via the graph** — e.g. a test referencing
behavior only through a string-based mock (`@patch(...)`), genuinely
invisible to a static call graph. A new `Pipeline.search()` method
(retrieval-only, no generation, no cache) and `GET /repos/{repo_id}/search`
endpoint were added specifically so this fallback never pays for an LLM
call just to get ranked chunks.

**Why built this way.** The graph-first-with-narrow-fallback design is
deliberate: an isolated node with zero real callers should stay isolated
in the output — the graph telling the truth — rather than being papered
over with retrieval-style fuzzy matches that could misrepresent real
coverage.

**Real bugs found and fixed.**
- **A real bug found before it could produce silently-wrong results**:
  `GET /context`'s `chunk_citations` never contains the raw enclosing AST
  chunk when small-to-big substitution is on (the default) — every match
  is substituted for a `"parent"`-typed chunk, never a real graph node.
  `AdjudicateContextBuilder.build()` on a real diff returned an *empty*
  bundle because of this. Fixed by adding a new `matched_chunks` field
  (raw, pre-substitution, smallest-first, each tagged with its real
  `chunk_type`) to the API response; `_find_enclosing_chunk` now reads
  `matched_chunks` and picks the first AST-typed entry, trying the
  location's end line too if the start line's lookup fails. Confirmed via
  direct inspection of `/context`'s raw JSON before writing the fix, not
  guessed.
- Verified against already-hand-derived ground truth on two real repos:
  NutriForge (3-hunk diff, callers/callees hand-summed from each
  function's own blast radius and confirmed to match the union exactly)
  and colorama (a genuine test-collision case, and a direct isolated test
  of the fallback mechanism proving it works correctly even when the
  graph-first gating correctly chose not to invoke it).

**Likely interview questions.**
- *"Walk through what happens when a diff touches a function whose only
  test coverage is a string-based mock."* The graph-first blast radius
  finds zero test-like callers (a `@patch('module.func')` string has no
  static call-graph edge to the function it patches); the fallback then
  fires, calling `Pipeline.search()` with a name+filename-derived query,
  keeping only test-like results — verified live against colorama's
  `reset_all`, which retrieved the two real tests that reference it only
  via `@patch`.
- *"Why does the fallback gate on zero test-like callers specifically,
  rather than always running retrieval alongside the graph?"* Always
  running retrieval would blur "the graph found nothing" (a real,
  meaningful signal about test coverage) with "retrieval found something
  loosely related" — the explicit gating preserves that distinction
  rather than papering over genuine isolation.

---

## Phase 25 — Defender Agent (naive)

**What was built.** `adjudicate/agents/defender.py::DefenderAgent.draft_justification`
— one LLM call, grounded strictly in the diff + the bundle's listed
changed functions/callers/callees/related tests; instructed never to state
a caller/callee not explicitly listed, and to say plainly if related tests
is empty rather than inventing coverage.

**Why built this way.** "Naive" here means single-pass, non-adversarial —
this phase exists specifically to prove the context bundle from Phase 24
is sufficient to ground a non-hallucinated narrative *before* introducing
the harder adversarial machinery (Prosecutor, Verifier) on top of it.

**Real bugs found and fixed.**
- **Two real diff-parsing edge cases were found during this phase's
  verification, in `context_builder.py` (Phase 24's module), not the
  Defender itself**: (1) a `git diff`'s default 3-line context can make a
  hunk's reported line range spill past the actual edit into an adjacent
  function's `def` line, and since the enclosing-chunk lookup falls back
  to the hunk's `end_line` whenever `start_line` yields no AST match, it
  picked colorama's `init` instead of `reset_all` for a 1-line edit
  genuinely inside `reset_all`. (2) Across multiple hunks in one file,
  `parse_diff` reported *new-file* line numbers, which drift from the
  *indexed* (pre-diff) file's real line numbers by the cumulative net
  line delta of every preceding hunk — a real NutriForge diff mis-resolved
  its second hunk to `me` instead of `login` for exactly this reason.
  Both were worked around for this phase's verification (using
  `git diff -U0` and net-zero-line-delta edits) rather than fixed inline,
  and explicitly flagged for a real fix — which happened immediately
  afterward, see below.
- **The fix, done as an approved follow-up before Phase 26 started**:
  `_HUNK_HEADER_RE` now captures both old-file and new-file hunk-header
  numbers; `_find_enclosing_chunk` now resolves against `old_start_line`/
  `old_end_line` (the indexed file's real numbering) instead of the
  drifting new-file side. `parse_diff` was rewritten to walk each hunk's
  *body*, not just its header, tracking separate old/new line cursors so
  both ranges are the tight min/max of lines actually touched by `+`/`-`,
  never the header's context-padded full span. Explicitly **not** fixed,
  by instruction: a hunk whose true edit spans two functions still
  resolves to only one `ChangedFunction` — flagged as a real, scoped-out
  future improvement. Re-verified against the exact original repro cases:
  colorama's `reset_all`/`init` misattribution now resolves correctly to
  `reset_all`; NutriForge's drifted hunk now resolves correctly to `login`
  via its old-line anchor, hand-confirmed by showing what the *pre-fix*
  code would have (wrongly) resolved to at that same old line number.
- Both live Defender outputs (NutriForge, colorama) were manually
  grounding-checked against known ground truth — every caller/callee/test
  claim traced back to the bundle, with nothing extra or invented; the
  empty-test-coverage claim for NutriForge was independently confirmed
  true via a real file search of the repo.

**Likely interview questions.**
- *"What are the two documented diff-parsing edge cases in
  `parse_diff`, and how does each get fixed?"* Context-padded line ranges
  spilling into an adjacent function (fixed by using the tight
  actually-touched-line range, not the header's full span) and cumulative
  new-file line-number drift across multiple hunks (fixed by resolving
  against the pre-diff file's old-line numbers instead). Both were found
  via real repro cases, not hypothetically.
- *"What's still not fixed, even after this pass?"* A single hunk whose
  real edit spans two functions still resolves to only one
  `ChangedFunction` — explicitly deferred, not silently left broken.

---

## Phase 26 — Prosecutor Agent (naive)

**What was built.** `adjudicate/agents/prosecutor.py::ProsecutorAgent.raise_concerns`
— one LLM call, instructed to raise only specific, checkable problems
(missing null/undefined checks, untested branches, breaking changes to
listed callers, security issues), scoped strictly to bundle-listed changed
functions, and instructed to directly challenge the Defender's
justification where the bundle/diff contradicts it.

**Why built this way.** This is the first genuinely adversarial agent in
the pipeline — its system prompt explicitly treats an empty "Related
Tests" section as itself a legitimate concern to raise, and instructs it
to challenge rather than restate the Defender, distinguishing it from a
second summarizer.

**Real bugs found and fixed.**
- **A real, honestly-reported factual error in live Prosecutor output —
  not hidden, and explicitly identified as the concrete motivating case
  for Phase 28's Verifier.** Against a real NutriForge diff, the
  Prosecutor claimed `login` had no null check before `user.save()` —
  factually wrong: the diff's own context lines, given directly to the
  Prosecutor, show a guard clause
  (`if (!user || !(await user.matchPassword(password))) { return
  res.status(401)...}`) immediately above the flagged lines, which already
  guarantees `user` is non-null by that point. This was not a
  bundle-hallucination — every name used was real and bundle-listed — but
  a reasoning error: it missed a guard clause visible in the exact text it
  was given. Explicitly **not** silently fixed by further prompt
  tweaking: "this is exactly the class of mechanically-checkable claim
  Phase 28's Verifier is meant to catch, and re-prompting to patch one
  observed error without a systematic check would be whack-a-mole, not a
  real fix." This false claim became the project's own canonical
  ground-truth Verifier test case for the next several phases.

**Likely interview questions.**
- *"Why is a false Prosecutor claim treated as a feature of the design
  rather than a bug to prompt-engineer away?"* Because the entire premise
  of Adjudicate is that LLM claims are hypotheses to be checked, not
  facts — a naive, occasionally-wrong Prosecutor is exactly the
  realistic baseline the Verifier (Phase 28) needs to demonstrate value
  against. Patching this one observed error with a sharper prompt would
  have been whack-a-mole against an unbounded space of possible reasoning
  errors.
- *"Walk through the false claim itself."* The `login`/`missing_null_check`
  case — claimed no null check before `user.save()`; the diff's own
  visible context lines contain a guard clause that already handles it.
  This became the project's standing ground-truth test case for the
  Verifier, the Rebuttal loop, and the Judge across the next four phases.

---

## Phase 27 — Structured Claim Schema

**What was built.** `adjudicate/schemas.py::ClaimType` (initially
`MISSING_NULL_CHECK`, `UNTESTED_BRANCH`, `TYPE_MISMATCH`,
`EXCEPTION_HANDLING` — each grounded in a real Phase 26 concern, not
speculative), `ProsecutorClaim`, and `parse_claims(raw_json,
valid_file_paths)` — two independent validation layers: (1) structural
(valid JSON, schema-conformant) and (2) **groundedness** (a claim's
`location` file must be one of the bundle's actual `changed_functions`
files — mechanically enforced, not just prompted). `generation/llm_client.py`
gained `response_schema` support (Gemini's true constrained-decoding JSON
mode). A `MAX_ATTEMPTS = 3` corrective-reprompt retry loop was added to
the Prosecutor for malformed output.

**Why built this way.** Two independent validation layers exist because
they catch different failure classes: schema conformance can't catch a
Prosecutor claiming about a function outside the bundle, and groundedness
checking can't catch malformed JSON — each layer mechanically enforces
exactly one property Phase 26 could previously only *ask for* in a prompt.
`BREAKING_CHANGE`/`SECURITY` were deliberately **not** added yet — neither
had appeared in real Prosecutor output at this point, and speculatively
adding claim types not yet observed would violate the project's own
"ground every addition in real output" discipline (they were added in
Phase 28, once the Verifier's dispatch table needed concrete targets).

**Real bugs found and fixed.**
- **A real test-suite integrity bug, caught by noticing a total-count
  mismatch, not by assuming green meant correct**: two new
  `parse_verdict` test functions (added in Phase 30, but the underlying
  discipline belongs here) were accidentally named identically to two
  pre-existing `parse_claims` tests — Python silently keeps only the
  later definition, so the original tests were being shadowed and never
  actually executed. Caught by the test count not matching expectations.
- **Groq's structured-output wrapping gap**, described fully under Phase
  16 (`{"claims": [...]}` instead of a bare array) — fixed in
  `parse_claims` with an unambiguous single-key-list unwrap.
- Deterministic proof, not just live capture, that the schema can express
  a well-formed but *false* claim: the real Phase 26
  `login`/`missing_null_check` wording was fed through `parse_claims`
  directly and confirmed to parse as a structurally valid
  `ProsecutorClaim` — proving the schema's job (structural/groundedness
  validity) is correctly separate from the claim's *truth*, which is
  exactly what Phase 28 exists to check.

**Likely interview questions.**
- *"Walk through the two independent validation layers in `parse_claims`
  and what each one catches that the other can't."* Structural (valid
  JSON, matches the pydantic schema) catches malformed output; groundedness
  (the claimed file must be in the bundle's real changed-functions list)
  catches a claim about code outside what was actually reviewed — Phase
  26 could only ask for the second property in a system prompt; this
  phase made it a hard, code-enforced rejection.
- *"Why weren't `BREAKING_CHANGE`/`SECURITY` added to the enum in this
  phase?"* Neither had actually appeared in real Prosecutor output yet —
  adding categories the model had never actually reached for would be
  exactly the "speculative list" the project's own conventions explicitly
  avoid. They were added one phase later, once the Verifier's dispatch
  table needed concrete, justified targets.

---

## Phase 28 — Verifier Layer

**What was built.** `adjudicate/verifier/` — `verify_claim(claim,
repo_path, test_command=None)`, the single dispatch entry point, **making
zero LLM calls anywhere**, confirmed both statically (grepped the whole
package for any LLM-related import — zero matches) and live (ran a real
verification with every LLM provider env var explicitly disabled,
confirmed `LLMClient` itself raises in that state as a negative control,
then confirmed `verify_claim` still completes successfully). Dispatch
table: `UNTESTED_BRANCH`/`MISSING_NULL_CHECK` → run the claim's own
`proposed_test` in a real sandbox (`sandbox.py::run_sandboxed` — real
`subprocess`, hard wall-clock timeout, monkeypatched network-denial guard,
explicitly documented as process-level, **not** a hard security
boundary); `SECURITY` → real `bandit` scoped to the file; `TYPE_MISMATCH`
→ real `mypy` scoped to the file (both match findings within ±2 lines of
the claimed location); `BREAKING_CHANGE` → run the existing test suite
now (an honest approximation, not a true before/after diff);
`EXCEPTION_HANDLING` → falls back to the proposed-test strategy if one
exists, else INCONCLUSIVE at LOW confidence — explicitly not forced onto
`bandit`/`mypy`, neither of which is a genuine fit. Confidence tiers are
**evidence-based, not outcome-based**: a claim REFUTED by a real failing
test is exactly as HIGH-confidence as one CONFIRMED by a real passing
test.

**Why built this way.** This is the single most important design decision
in the whole project. An LLM asked to verify another LLM's claim is still
just an LLM guessing, with no more grounding than the original claim had —
real test execution, real static analyzers, and a real call graph are the
only things in this system that can turn "sounds plausible" into "actually
true." The Phase 26 false claim (`login`/`missing_null_check`) is the
concrete motivating case, and the benchmark's later 46.8%→50.0% claim-flip
rate (Phase 31) is the quantitative proof this distinction matters.

**Real bugs found and fixed.** This phase is primarily about proving
correctness on constructed and real ground-truth cases rather than fixing
a pre-existing bug, but the verification methodology itself is worth
knowing cold:
- **Case 1 — the known false claim, correctly REFUTED with real,
  inspectable evidence.** Against a fresh temp copy of the real NutriForge
  `server/` tree with the real diff applied via `patch -p1`: `STATUS:
  refuted, CONFIDENCE: high, STRATEGY: proposed_test`, with real stdout
  showing the sandboxed script called the actual, unmodified, diff-applied
  `login` handler and got a clean 401 — the guard clause the false claim
  overlooked really does intercept before the flagged line is ever
  reached.
- **Case 2 — a constructed true positive, correctly CONFIRMED with a
  real crash.** A deliberately broken function (`get_user_email(None)` →
  `TypeError: 'NoneType' object is not subscriptable`), correctly reported
  CONFIRMED with the real traceback as evidence — proving the Verifier
  isn't just biased toward REFUTED.
- **Case 3 — the zero-LLM-calls guarantee, proven live, not just by
  reading the code.** Cleared every provider env var; first confirmed the
  negative control (`LLMClient().complete()` correctly raises with no
  provider configured); then ran a real `verify_claim` call in the exact
  same zero-provider process — it completed successfully, proving no code
  path in `adjudicate/verifier/` touches any LLM client.
- **Disclosed limitations, stated plainly rather than glossed over**:
  sandboxing is subprocess-based, not Docker — network denial is
  best-effort and memory-limit enforcement is declared but inert (needs a
  cgroup/Job Object a bare subprocess can't provide). `verify_breaking_change`
  checks "does the suite pass now," not "did this diff introduce a new
  failure" — a true before/after comparison needs a stored pre-change
  baseline Adjudicate doesn't have yet. `EXCEPTION_HANDLING` claims
  without a `proposed_test` have no real check at all (INCONCLUSIVE/LOW,
  by design). `bandit`/`mypy` are Python-only — a claim about a
  non-Python file routed to `SECURITY`/`TYPE_MISMATCH` is always
  INCONCLUSIVE/LOW.

**Likely interview questions.**
- *"Why does the Verifier make zero LLM calls, and what would be lost if
  it didn't follow that rule?"* An LLM verifying another LLM's claim is
  still ungrounded guessing — it would add a second opinion, not a check.
  The whole reason Adjudicate is more than "an LLM that argues with
  itself" is that real test execution/static analysis/call-graph facts
  are the only things that can actually falsify a claim.
- *"Walk through the CONFIRMED/REFUTED/INCONCLUSIVE confidence-tier logic
  — why is confidence tied to evidence kind, not verdict direction?"*
  Because a real failing test is equally strong evidence whether it
  confirms or refutes a claim — tying confidence to *which way* the
  verdict went would bias the system toward treating "the code has a bug"
  as inherently more trustworthy than "the code is fine," which isn't
  true and isn't what evidence quality actually measures.
- *"Why is sandboxing process-level rather than container-level here, and
  what specifically does that not protect against?"* A real subprocess
  timeout is a genuine guarantee; network denial is a monkeypatched guard
  around `socket.connect`/Node's `net`/`http(s)` — explicitly documented
  as bypassable by a determined subprocess via lower-level syscalls, and
  memory limiting is declared in config but inert without OS-level
  integration. Adequate for a local, single-operator tool reviewing
  diffs the operator already trusts enough to look at; explicitly not
  adequate for public deployment accepting arbitrary third-party diffs
  (see Phase 33).
- *"What's the documented limitation of `breaking_change` verification,
  and what would a correct fix look like?"* It runs the existing suite
  against the post-diff state only and calls that confirmation — a
  correct version would run the suite against the pre-diff commit too and
  diff the two result sets, which needs a stored pre-change baseline this
  system doesn't currently persist.

---

## Phase 29 — Defender Rebuttal Loop

**What was built.** `DefenderAgent.rebut()` (a separate
`REBUTTAL_SYSTEM_PROMPT`: concede every CONFIRMED claim outright, briefly
acknowledge REFUTED ones citing evidence without gloating, push back on
INCONCLUSIVE ones only with real counter-evidence from the bundle or admit
plainly there is none) and `adjudicate/orchestrator/rebuttal_loop.py::run_rebuttal_loop`
— pure round-coordination, never drafts a rebuttal itself. Round 1 is the
initial justification; up to `MAX_REBUTTAL_ROUNDS = 2` further rounds each
call `rebut`, chaining each round's output into the next round's input.
Termination is decided **mechanically**: "resolved" once every claim is
CONFIRMED or REFUTED (both are final, nothing left to rebut); if at least
one claim is still INCONCLUSIVE after the round cap, ends "by cap"
instead.

**Why built this way.** The loop deliberately does **not** attempt to
judge whether the Defender's counter-evidence was "satisfactorily
countered" in prose — that requires a value judgment genuinely out of
scope for a coordination loop; it belongs to the Judge (Phase 30).

**Real bugs found and fixed.** No code bugs; the phase's own honestly
disclosed structural property is the interview-relevant fact: **a claim
that starts INCONCLUSIVE always ends the loop "by cap,"** never
"resolution," because nothing in this phase re-verifies a claim mid-loop
— a claim's status can never change from INCONCLUSIVE once set within a
single loop run. The Defender still gets a genuine second attempt (round
2's prompt contains round 1's own rebuttal text, not the original
justification again — confirmed by a dedicated test), but it mechanically
cannot flip `ended_by` to "resolution." Flagged explicitly as a real
future extension (re-verifying Defender counter-evidence mid-round), not
faked or hidden.

**Likely interview questions.**
- *"Explain the rebuttal loop's termination conditions and how
  'resolution' differs from 'cap.'"* Resolution = every claim is CONFIRMED
  or REFUTED (both final Verifier outcomes, nothing left to argue about);
  cap = the round limit was hit with at least one claim still
  INCONCLUSIVE. Because nothing re-verifies a claim mid-loop, any
  genuinely INCONCLUSIVE claim structurally guarantees the loop ends by
  cap, not resolution — this is a known, disclosed property, not a bug.
- *"What happens when a claim is INCONCLUSIVE and the rebuttal loop hits
  its cap?"* The Defender still gets a real second round (with the prior
  round's actual text as context, not the original justification again),
  but the claim's status cannot change within this loop — `ended_by="cap"`
  is passed to the Judge, which (Phase 30) is mechanically forbidden from
  rendering high-confidence approval in that state without either a
  `minority_report` or confidence below 0.7.

---

## Phase 30 — Judge Agent

**What was built.** `adjudicate/agents/judge.py::JudgeAgent.judge(context_bundle,
diff, full_transcript, verified_claims, termination_reason) → JudgeVerdict`
— reads the Defender's justification, every claim's *verified*
status/confidence/evidence, and the full rebuttal exchange; never a raw
unverified Prosecutor claim alone. `Verdict` = approve/reject/
needs_human_review. **`_cap_handling_violation` is mechanically enforced,
not just prompted**: rejects (and retries) any verdict where
`termination_reason == "cap"`, at least one claim is still INCONCLUSIVE,
confidence is `>= 0.7` (`CAP_UNRESOLVED_CONFIDENCE_CEILING`), *and*
`minority_report` is null — a cheaply, objectively checkable fact from
structured data. Deliberately **not** extended to "was a CONFIRMED claim
clearly conceded" — assessing that requires holistic judgment over free
text, which is exactly what an LLM (not a keyword heuristic) is suited
for, and relies on prompt instruction plus live verification instead.

**Why built this way.** The mechanical cap-handling check exists because
it's the one property this phase's own flagged Phase 29 gap makes
critical: an unresolved INCONCLUSIVE claim always ends the loop "by cap,"
and it would be a real correctness failure for the Judge to render
confident approval over a claim nothing ever actually resolved — this is
too important to leave to a prompt instruction alone, so it's checked in
code and retried with a corrective prompt if violated.

**Real bugs found and fixed.**
- **The same Gemini thinking-token truncation bug (Phase 16), first
  reproduced here.** The Judge's response was reproducibly truncated
  mid-JSON at the *identical* output length across 6 consecutive real
  attempts against the same long transcript — `settings.LLM_MAX_TOKENS`'s
  global default (1024) was being consumed almost entirely by invisible
  thinking tokens, since the Judge's prompt (a full multi-round
  transcript) is longer than the single-turn prompts that default had
  previously been proven against. Fixed with a Judge-specific
  `JUDGE_MAX_TOKENS = 2048`, confirmed live afterward (Case 3 below
  succeeded on the first attempt once applied) — see the full mechanism
  under Phase 16.
- **A real test-suite integrity bug caught while writing new tests**: two
  new `parse_verdict` tests were accidentally named identically to two
  pre-existing `parse_claims` tests in the same module — Python silently
  keeps only the later definition, so the originals were shadowed and
  never executed. Caught by noticing the total test count didn't match
  expectations, not by assuming green meant correct; renamed and
  confirmed all 19 (12 original + 7 new) genuinely ran.
- **Three required live verdict cases, each producing directionally
  correct, evidence-cited output**: (1) the REFUTED
  `login`/`missing_null_check` case → `approve, confidence=1.0`,
  correctly not penalizing the PR for a claim the Verifier already
  disproved. (2) the CONFIRMED, conceded true-positive case →
  `reject, confidence=1.0`, citing both the real CONFIRMED evidence and
  the Defender's own explicit concession — never approving a conceded
  CONFIRMED claim. (3) a constructed INCONCLUSIVE `exception_handling`
  claim with no `proposed_test`, driven through a real rebuttal loop to
  its cap for the first time live (`rounds_used=3, ended_by="cap"`) →
  `needs_human_review, confidence=0.7` (exactly at the enforced ceiling),
  with a substantive, real `minority_report` explaining *why* — this is
  real Gemini output that passed the mechanical `_cap_handling_violation`
  check on its own merits via the minority-report path, not hand-waved.

**Likely interview questions.**
- *"What does the Judge's `_cap_handling_violation` check actually
  enforce, and why is it mechanical rather than prompt-only?"* If the
  rebuttal loop ended by hitting its round cap with an unresolved
  INCONCLUSIVE claim, the Judge cannot render confidence ≥0.7 approval
  without either a `minority_report` or a lower confidence — checked
  against plain structured data (termination reason, claim statuses,
  confidence, minority_report presence), which is objective enough to be
  code-enforced rather than left to a prompt instruction alone, unlike
  "was a CONFIRMED claim clearly conceded," which is a holistic judgment
  call better suited to the LLM itself.
- *"Explain the Gemini thinking-token truncation bug in full."* See Phase
  16 — but note this is where it was *first* independently reproduced
  (6 consecutive identical-length truncations), which is what led to the
  fix being generalized (later reproduced again, independently, against
  the Prosecutor on a real VIBE diff).
- *"Walk through what happens when the rebuttal loop hits its cap with an
  unresolved claim, end to end."* The loop passes `ended_by="cap"` and
  the unresolved claim list to the Judge; the Judge's own reasoning is
  unconstrained, but its output is mechanically checked afterward —
  real Gemini output in this exact scenario produced
  `needs_human_review, confidence=0.7` with a substantive
  `minority_report`, which passed the check via the minority-report path
  (satisfying "either lower confidence or explain why").

---

## Phase 31 — Benchmark Harness

**What was built.** `adjudicate/benchmark/` — a 13-case benchmark (5
synthetic bug/fix pairs = 10 cases, each ground-truthed independently of
any LLM call via direct execution or static analysis; 2 reused real-repo
cases carried over from Phases 27-29's own live verification; 1 additional
clean case, `type_mismatch_fixed`) comparing three review conditions per
case: (a) a single-agent baseline (no adversarial structure), (b)
adversarial without verification (Prosecutor + Defender, no Verifier), (c)
the full verified pipeline. Metrics: catch rate, false-positive rate,
average LLM calls/tokens/rounds, and the **claim-flip rate** — the
fraction of Prosecutor claims that were REFUTED once actually verified.

**Why built this way.** The case set mixes synthetic (precise, cheaply
reproducible ground truth via direct execution) and real-repo (authentic
LLM behavior against genuine code) cases specifically so neither
weakness — synthetic cases being toy-like, real cases being harder to
ground-truth — dominates the result on its own.

**Real bugs found and fixed.**
- **A real bug found before a single case could even complete**: the
  first launch attempt (against Groq) crashed with a `pydantic.ValidationError`
  inside the baseline reviewer's response model — Groq's `json_object`
  mode only guarantees valid JSON, not the exact requested field names,
  and returned a field starting with `des...` (`description`) instead of
  the requested `explanation`. Root cause: unlike every other
  structured-output agent in the codebase (Prosecutor, Judge),
  `baseline_reviewer.py::review()` had **no** retry/validation loop at
  all — a bare `model_validate_json()` call. Fixed by adding the same
  `MAX_ATTEMPTS=3` corrective-reprompt loop already established elsewhere,
  plus a stronger system-prompt instruction naming the exact required
  field names, covered by a direct regression test reproducing the exact
  failure via a fake client.
- **A systematic, absorbed-but-flagged issue observed during the real
  run**: Groq returned `"confidence": "high"` (a string) instead of a
  number on **6 separate cases**, failing `JudgeVerdictModel`'s schema on
  attempt 1 every time — the existing Judge retry loop caught and
  recovered on attempt 2 every time, so no case was lost, but this is
  flagged as a real, systematic (not one-off) gap worth a dedicated fix
  (a more forceful numeric-type hint in Groq's schema prompt) before using
  Groq at this volume again.
- **The GNU `patch` "Reversed (or previously applied) patch" bug** —
  found later, after this phase's own fixtures were already in production
  use, but directly relevant to trusting this benchmark's own results: a
  diff with *both* a `diff --git a/X b/X` header line and a placeholder
  `index 0000000..0000000` line makes `patch` misinterpret an ordinary
  modification to an *existing* file as an attempt to create a new one.
  Explicitly checked, not assumed, whether this affected Phase 31's own
  numbers: both real benchmark fixtures use genuine non-placeholder git
  hashes (real `git diff` output), so the bug's precondition never
  applied to them — Phase 31's reported numbers are unaffected. Fixed
  generally (`_strip_git_index_lines`, since plain POSIX `patch` never
  actually reads the `index` line at all) in the live-review sandbox path
  this bug was actually found in (see Phase 32/33's bug list).

**Reading the raw condition (c) numbers honestly — the single most
interview-critical piece of analysis in this project.** Taken at face
value, condition (c) looks *worse* than (a) and (b) on both catch rate and
false-positive rate. Tracing every miss and every false positive
individually (not summarized) shows this is **not** a Verifier or Judge
defect:
- **All 3 missed bugs are Prosecutor recall misses, not Verifier/Judge
  failures.** In every case, the Prosecutor never raised a claim of the
  type that would have caught the real defect. Two concrete, distinct
  gaps: `mutable_default_bug` and `off_by_one_bug` are defect classes
  with **no matching `ClaimType` in the taxonomy at all** — a structural
  vocabulary gap, independent of model reasoning quality.
  `sql_injection_bug` *does* have a matching `SECURITY` claim type (and
  `bandit` would have flagged it, per the case's own ground truth), but
  the Prosecutor simply never raised one — a real recall miss, not a
  taxonomy gap.
- **4 of the 5 false positives in (c) are real, verified, CONFIRMED
  findings about the file — just about a different, out-of-scope issue**
  than the one specific defect each case's ground-truth label targets. A
  synthetic single-function fixture with no test suite genuinely has
  `untested_branch`/`missing_null_check` issues, confirmed by real
  sandboxed execution — the Judge correctly rejected given genuinely
  confirmed problems in front of it; it has no way to know an external
  ground-truth label considers those issues out of scope. Contrast with
  `sql_injection_fixed` (and later `type_mismatch_fixed`), where the same
  generic claim types were raised but genuinely REFUTED — proof the
  mechanism *can* discriminate correctly when the underlying claims are
  actually unfounded.
- **Condition (b)'s 100%/100% is not "always right" — it's "almost always
  says something."** `flagged = len(claims) > 0`, and the Prosecutor
  raised at least one claim on every case regardless of ground truth — it
  "catches" bugs for the wrong reason (*some* claim was raised, not the
  *correct* one), the same mechanism driving its false-positive rate to
  100%. This is precisely the gap the Verifier exists to close.
- **The claim-flip rate is the single strongest, cleanest number this
  benchmark produces, and it is not undermined by any of the above**: of
  50 total claims the Prosecutor raised (trusting them at face value),
  real verification refuted 25 of them (**50.0%**) — unfounded claims
  that would have reached a human reviewer as-is under condition (b),
  caught before they did under condition (c).

**Likely interview questions.**
- *"Walk through the claim-flip rate metric — why is it described as the
  single strongest number the benchmark produces?"* It directly isolates
  the Verifier's marginal contribution: of every claim the Prosecutor
  raised confidently, half were refuted once actually checked against
  real tests/tools — a concrete, non-confounded measurement of "how often
  would an unverified LLM reviewer have been wrong," independent of the
  taxonomy-coverage and ground-truth-scope confounds that make condition
  (c)'s raw catch/FP rate misleading on its own.
- *"Why does condition (c)'s raw catch rate look worse than the
  baseline, and why isn't that the real story?"* Because every miss traces
  to a Prosecutor recall/taxonomy gap (two defect classes have no
  matching `ClaimType` at all; one has a matching type but wasn't raised),
  not a Verifier/Judge failure — and most of (c)'s "false positives" are
  real, confirmed findings about a different, out-of-scope issue in the
  same file, which a single-binary-label-per-case benchmark can't
  distinguish from a genuine false alarm. The claim-flip rate is
  unconfounded by either issue and is the number to lead with.
- *"How do you know the Verifier isn't just theater?"* Two independent
  lines of evidence: Phase 28's own zero-LLM-calls proof (grep + a live
  no-provider-configured run), and this phase's claim-flip rate — 50% of
  raised claims were mechanically refuted by real test execution/static
  analysis, which is only possible because the Verifier is checking
  something real, not re-asking an LLM to vote.
- *"What would a v2 of the Prosecutor's claim taxonomy need to fix,
  concretely?"* Add `ClaimType` entries for shared-mutable-default-argument
  and off-by-one-slice-arithmetic defect classes (currently
  unrepresentable), and separately investigate why `SECURITY` claims
  aren't being raised even when `bandit` would confirm one — a real
  recall gap distinct from the taxonomy gap.

---

## Phase 32 — React Frontend

**What was built.** Both parts: **Part 1** — Chat + Graph scaffold
(`frontend/src/` — `RepoContext` as the sole state store, `useIndexingStatus`/
`useChat`/`useGraph` hooks, graph rendering, blast-radius animation,
citation-to-graph linking). **Part 2** — the live Defender/Prosecutor/
Verifier/Judge review screen, driven by `POST /repos/{id}/review`'s SSE
stream via a manual `fetch`+`ReadableStream` reader (`EventSource` can't
be used — this endpoint is POST). Confirmed with a real, full,
Playwright-driven browser run reaching and hand-verifying the Judge card.

**Why built this way.** Graph library choice went through three real
iterations, each a genuine engineering decision, not indecision:
Cytoscape.js (first choice — first-class `.animate()` API for blast-radius
fades) → `cytoscape-cola` (a real Cytoscape extension for a stronger
spring-physics feel, per explicit instruction to check Cytoscape's own
layout options before switching libraries) → **vis-network** (final,
per an explicit user decision, not a unilateral engineering call — to
recover the exact physics feel the original `streamlit-agraph`/vis.js
Streamlit view already had, which approximating inside Cytoscape's
physics model was judged insufficient to replicate). SSE (not WebSockets)
because the review data flow is strictly one-directional. React Context
(not Redux/Zustand) because the app's state shape is small and flat. No
router because the app has exactly three fixed views (Graph/Chat/Review)
toggled by local state, not URL-addressable routes.

**Real bugs found and fixed.**
- **A real crash on the very first live `/query` call**: `ValueError:
  shapes (768,) and (384,) not aligned` inside the semantic cache's
  cosine-similarity computation — stale `semantic_cache` rows from an
  earlier session using a 384-dim model (MiniLM) were being compared
  against current 768-dim CodeBERT vectors. Fixed by clearing the stale
  rows directly; flagged as a real, unfixed gap that nothing currently
  invalidates cached embeddings when the configured model changes.
- **A CORS gap**: `api/main.py` had no CORS middleware at all — needed
  for the Vite dev server (a different origin) to call it from the
  browser. Added `CORSMiddleware` plus `Settings.CORS_ALLOWED_ORIGINS`
  (config, not hard-coded) so a deployed frontend origin is a config
  change later, not a code change.
- **A real, hand-caught interpolation bug while rewriting the blast-radius
  animation for vis-network**: the fade-back tween initially read each
  node's "current" opacity from inside the same per-frame callback that
  also *writes* the new interpolated opacity — reading "current" on frame
  2 actually read frame 1's already-partially-interpolated value,
  compounding the interpolation incorrectly instead of producing a smooth
  fade. Caught by re-reading the function before moving on, not assumed
  correct because it looked reasonable; fixed by snapshotting starting
  opacities into a plain `Map` once, before the tween begins.
- **A real graph-physics non-convergence bug on larger repos** (pawn_ai,
  657 nodes/802 edges never visibly settling): confirmed via
  instrumenting vis-network's real `stabilizationIterationsDone`/
  `stabilized` events (read from the installed library's own source to
  confirm real semantics before trusting them) — the iteration cap was
  being hit *before* real velocity-based convergence, dropping the
  simulation into a slow, indefinitely-running live tick loop.
  **Disclosed honestly, not glossed over**: even the "reference working"
  case (NutriForge, 227 nodes) never technically reaches `stabilized`
  either — it only *looks* settled because residual velocity is small
  enough to be visually imperceptible. Fixed with node-count-gated
  scaling (`computePhysicsScale`, threshold 250 nodes — comfortably above
  NutriForge's 227, mathematically identical to the old hardcoded values
  at or below that threshold, so provably zero regression risk for small
  repos) — `iterations`/`damping`/`minVelocity` all scale linearly above
  the threshold. Verified with two byte-for-byte-identical screenshots
  taken 2 seconds apart after settling — genuinely at rest, not just
  visually close.
- **A real, disclosed font-rendering bug**: vis-network renders labels to
  a `<canvas>`, which does not automatically re-render when a linked web
  font (Inter) finishes loading asynchronously — labels could render in
  the fallback system font on first paint and silently stay that way.
  Fixed with `document.fonts.load(...).then(() => network.redraw())`
  right after constructing the `Network`.
- **The GNU `patch` "Reversed (or previously applied) patch" bug** — the
  user's own reasonable first hypothesis (the persistent clone had been
  mutated by an earlier failed run) was checked thoroughly and ruled
  out two independent ways: reading `_materialize_sandbox`'s code (only
  ever copies *into* a fresh temp dir, never writes back) and, as the
  strongest possible check, cloning the real repo fresh from GitHub at
  the identical commit and running `diff -rq` against the persistent
  clone — zero differences. The **real** cause: a diff with both a
  `diff --git a/X b/X` header and a placeholder `index
  0000000..0000000` line makes `patch` misinterpret an ordinary edit to
  an existing file as a new-file creation — and this placeholder
  convention is exactly what this project's own hand-constructed test
  diffs had used throughout (a very natural shortcut when writing a
  synthetic diff without running real `git diff`). Confirmed a real
  `git diff` (genuine hashes) was never affected. Fixed generally with
  `_strip_git_index_lines`, since plain POSIX `patch` never actually
  reads the `index` line at all.
- **The Prosecutor JSON-parse failure against a real repo (`mkocabas/VIBE`)**
  was the second independent reproduction of the Gemini thinking-token
  truncation bug (see Phase 16/30) — the user's own reasonable hypothesis
  (unescaped code/quotes in the diff) was checked and found wrong; the raw
  response was genuinely cut off mid-string, confirmed via
  `finish_reason=MAX_TOKENS` and `thoughts_token_count=979/1024`.
- **Live end-to-end verification was, by disclosure, only partially
  completed at first**: a real Playwright + Chromium run reached Context
  and Defender stages correctly (matching Phase 24/25's own hand-verified
  ground truth for the exact same node), then hit a real `429
  RESOURCE_EXHAUSTED` from Gemini's free-tier daily quota before reaching
  Prosecutor/Verifier/Rebuttal/Judge — reported honestly as "code-complete
  and unit-tested, but not fully live-verified end-to-end," not marked ✅
  prematurely. **Closed out for real in a later session**: a full
  Playwright run reached **Judge: REJECT, 80% confidence**, citing two
  `[CONFIRMED, high]` claims — hand-checked against the diff (both real
  and accurate) and cross-validated against the standalone Phase 31
  benchmark's own stored result for the identical case
  (`colorama_reset_all_reused: reject, 0.8 confidence`) — independently
  confirming both paths agree.

**Likely interview questions.**
- *"Why was Cytoscape.js replaced with vis-network, and what had to be
  re-tuned as a result?"* Cytoscape won initially for its first-class
  `.animate()` API; it was replaced (an explicit user decision, following
  an interim `cytoscape-cola` attempt that still didn't fully replicate
  the target feel) specifically to recover the real vis.js physics the
  original Streamlit view already had validated. The final physics config
  deliberately reused vis.js's own real BarnesHut defaults (unmodified
  from what `streamlit-agraph` had never overridden) rather than
  re-tuning from scratch, on the reasoning that re-tuning away from an
  already-confirmed-good feel risks moving away from the target, not
  toward it — the one deliberate change was raising `avoidOverlap` from 0
  to 0.6 for label legibility.
- *"Walk through how `useLiveReview.js` parses an SSE stream manually
  instead of using `EventSource` — why is that necessary here
  specifically?"* `EventSource` is GET-only; the diff payload must go in
  a POST body, so the frontend reads the response via a manual
  `fetch` + `ReadableStream` reader instead, handling buffered,
  chunk-spanning `text/event-stream` framing by hand.
- *"Why does a citation chip sometimes render disabled/unclickable?"*
  Because small-to-big substitution (Phase 15) frequently replaces a
  citation with its `"parent"` window chunk, which is never a real graph
  node — `CitationItem` resolves each citation against the loaded graph
  and renders non-resolving ones as plain inert text with an explanatory
  tooltip rather than a fake or dead click target. This is not
  hypothetical — it was hit on the very first live query during
  verification.
- *"Walk through the graph physics non-convergence bug end to end."*
  Real vis-network events (`stabilizationIterationsDone` vs `stabilized`)
  were instrumented to confirm the iteration cap was firing before real
  velocity-based convergence on a 657-node repo; the fix scaled
  iterations/damping/minVelocity by node count above a threshold set
  comfortably above the known-good case, with the small-repo path proven
  mathematically unchanged, not just tested.

---

## Phase 33 — Deployment (in progress)

**What was built (the completed slice).** SQLite → PostgreSQL migration,
scoped deliberately narrow: only the relational storage layer
(`database/sqlite_client.py`, `database/graph_store.py`,
`database/models.py`, `config.py`). FAISS, BM25, Redis, task queues, and
`pipeline.py`/agent/API-contract logic were explicitly untouched, migrated
incrementally with a full regression run after every step. Driver:
`psycopg2` (sync), not `asyncpg` — every `DatabaseManager` method is a
blocking SQLAlchemy `Session` call, and going async would require
threading `await` through `pipeline.py`/`api/main.py`/every retrieval
module, explicitly out of scope for a storage-layer swap. `graph_store.py`
moved from a flat JSON file to a real table (`GraphSnapshotRecord`,
`graph_data JSONB`), not just SQLite→Postgres — it was never SQLite-backed
to begin with. Test isolation redesigned from one-SQLite-file-per-test to
one PostgreSQL **schema** per test (`DatabaseManager(schema=...)`, scoped
via `search_path`).

**Why built this way.** `psycopg2` over `asyncpg`: the task was explicitly
framed as "storage-layer swap underneath, not a behavior change" — an
async driver would cascade into an async engine, `AsyncSession`, and
`await` at every call site across the codebase, which is a much larger,
different-shaped change than what was asked for. Schema-per-test (not a
separate database-per-test or truncate-between-tests): the old
one-file-per-test pattern has no direct Postgres equivalent since all
tests now share one real instance; per-test schemas via `search_path`
give the same hard isolation a separate SQLite file gave, without needing
a separate physical database per test.

**Real bugs found and fixed.**
- **A real correctness fix bundled into the migration, not scope creep**:
  `indexed_at`/`created_at` columns are now `DateTime(timezone=True)` —
  SQLite silently tolerated storing timezone-aware
  `datetime.now(timezone.utc)` values without actually being
  timezone-aware itself, but PostgreSQL's default `DateTime` maps to
  `TIMESTAMP WITHOUT TIME ZONE` and would have silently dropped the
  tzinfo on write.
- **A real bug found and fixed during the migration itself, caught by the
  full regression suite, not shipped broken**: the first pass gave
  `graph_store.py` its own independent module-level engine/sessionmaker
  "for independent testability." This immediately failed with
  `psycopg2.errors.UndefinedTable: relation "graph_snapshots" does not
  exist` across every graph-store/graph-retriever test, because that
  independent engine had no `search_path` override and connected to the
  default schema, while each test's own `DatabaseManager(schema=pg_schema)`
  had only created the table inside the test's own schema.
  Root-cause understood before fixing (not papered over): `GraphSnapshotRecord`'s
  foreign key into `RepositoryRecord` makes "its own connection"
  *structurally* wrong, not just a test-isolation inconvenience — fixed
  by threading the caller's `DatabaseManager` (and its `session_scope`)
  through both `save_graph`/`load_graph` instead.
- **The Phase 7 `graph_store` vs. `sqlite_client` divergence was
  explicitly re-verified to still hold after the migration**, per direct
  instruction, not assumed: both functions exist as distinct,
  intentionally different Postgres-backed paths, and `api/main.py`/
  `GraphExpander` still call only the lossless one — confirmed by reading
  the code, not by re-running the original repro (which would require
  reverting the fix to reproduce).
- **Full end-to-end verification against a real dev Postgres instance,
  not mocked**: a fresh `Pipeline()` re-indexed `tartley/colorama` from
  scratch and produced an **exact match** to the known-good SQLite-era
  numbers (`files_discovered=23, chunks_indexed=344, graph_nodes=184,
  graph_edges=283`). A real chat query returned a correct, grounded
  answer through the full stack (real embeddings, FAISS/BM25/graph-
  expansion/reranking, real LLM generation — Gemini fell through to the
  Groq static-priority fallback live during this check). The Phase 14
  semantic-cache lexical-gate fix was re-run against the real
  Postgres-backed cache and held identically. A fresh `uvicorn` process
  (simulating a restart, not reusing in-memory state) confirmed
  `Pipeline`, `/status`, and `/graph` all still agree — exactly the
  invariant the original Phase 7 bug fix established.

**What's explicitly still pending, stated plainly.** Vercel (frontend)
and Render (FastAPI backend) hosting, and — the one deployment concern
flagged as needing resolution *before* committing to Render specifically
— the Verifier's sandbox needs strict container resource/time/network
limits with genuinely no network access, and the roadmap explicitly says
to confirm Render supports this before committing, with a dedicated small
VM as the fallback if it doesn't. Submittable repositories should be
restricted to an allowlist for any public deployment unless the sandbox
security posture is fully hardened first. FAISS/pgvector migration,
Redis, and a real task queue for indexing are all out of this slice's
scope. Roadmap status is deliberately left 🟡 (in progress), not ✅, since
the phase as a whole (deployment) is not complete — only its storage-layer
prerequisite is.

**Likely interview questions.**
- *"What's actually deployed today versus only planned, and how do you
  know?"* Nothing is deployed anywhere beyond a local machine — no
  CI/CD pipeline exists (`docker-compose.yml`/`Dockerfile` are stale,
  still launching the legacy Streamlit app), and this is directly
  reflected in `docs/roadmap.md`'s own 🟡 marker plus this phase's own
  explicit "not done" list, not a claim being made from memory.
- *"What deployment-specific security concern does the roadmap flag
  before committing to a Render-hosted Verifier sandbox?"* Container-level
  resource/time/network isolation with genuinely no network access from
  inside the sandbox — confirm Render supports this before committing,
  with a dedicated small VM as the documented fallback if it doesn't.
- *"Why `psycopg2` (sync) instead of `asyncpg`?"* The task was explicitly
  scoped as a storage-layer swap, not a behavior change — every
  `DatabaseManager` call site across `pipeline.py`, `api/main.py`, and
  every retrieval module is synchronous today; an async driver would
  cascade into an async engine/session/await chain across the whole
  codebase, a much larger, differently-shaped change than what was asked
  for.
- *"How did you verify the migration didn't silently change behavior?"*
  Incremental migration with a full regression suite after every step
  (693 passed throughout, both before and after the riskiest step), plus
  an end-to-end real-Postgres re-index of a known repository producing an
  *exact* match to the previously-recorded SQLite-era node/edge/chunk
  counts, and explicit re-verification (not assumption) that the Phase 7
  bug fix and the Phase 14 semantic-cache fix both still hold against the
  new backend.
- *"What would need to change for this system to run as multiple
  stateless replicas behind a load balancer?"* The in-memory
  indexing-status registry (`_index_state` in `api/main.py`) and the
  now-Postgres-backed-but-still-single-instance-assumption pieces would
  need externalizing to a shared store (Redis is the natural choice);
  indexing would need to move from an in-process `BackgroundTasks` job to
  a real queued worker with persisted job state. The database layer
  itself (now Postgres) is already the right shape for multi-writer
  access — SQLite was the actual single-writer wall, and this phase
  already removed it.

---

# The single most important thing to say about this project

**Adversarial claims, verified by deterministic tools, before they ever
reach the judge.** RepoMind is a hand-built GraphRAG system — Tree-sitter
AST parsing, a hand-resolved call graph, hybrid dense+sparse retrieval
fused by RRF, graph expansion, cross-encoder reranking, a semantic cache,
small-to-big context assembly — built manually specifically to demonstrate
that these techniques are understood at the implementation level, not just
as library calls. But the project's real thesis lives one layer up, in
Adjudicate: an LLM's claims about a code diff are treated as *hypotheses*,
never facts. A Prosecutor agent raises structured, falsifiable claims; a
Verifier — making **zero LLM calls, ever**, confirmed both statically and
by a live no-provider-configured run — checks each one against real
sandboxed test execution, `bandit`, `mypy`, or the real call graph; a
Defender concedes what's genuinely CONFIRMED and only contests what real
evidence actually supports; and a Judge renders a verdict reading verified
evidence only, with one critical failure mode (rendering confident
approval over an unresolved claim) mechanically blocked in code, not just
asked for in a prompt. The concrete proof this distinction is not
academic is the benchmark's own headline number: of every claim the
Prosecutor raised confidently, **50% were refuted once actually checked**
— unfounded claims that would have reached a human reviewer as-is under a
naive adversarial-without-verification design, caught before they did
under the full pipeline. Every other design decision in this project — the
static LLM provider priority for observability, SSE for one-directional
streaming, the graph-first-with-narrow-fallback context builder, the
mechanically-enforced cap-handling rule — exists in service of the same
idea: don't trust an LLM's assertion about anything checkable without
actually checking it.

---

# Numbers to have ready

**Tests.** 693 passed, 0 failed — the current, final count as of the
Postgres migration (Phase 33's storage-layer slice), unchanged from
immediately before it, confirming the storage-backend swap introduced
zero regressions.

**Benchmark (Phase 31, final, 13/13 real cases — report:
`adjudicate/benchmark/results/20260719T070702Z.json`):**

| Condition | Catch Rate (buggy, n=6) | False Positive Rate (clean, n=7) | Avg LLM Calls | Avg Tokens | Avg Rounds |
|---|---|---|---|---|---|
| (a) Single-agent baseline | 83.3% (5/6) | 14.3% (1/7) | 1.00 | 416.4 | N/A |
| (b) Adversarial, no Verifier | 100.0% (6/6) | 100.0% (7/7) | 2.08 | 2156.7 | N/A |
| (c) Full verified pipeline | 50.0% (3/6) | 71.4% (5/7) | 5.08 | 7359.7 | 2.00 |

**Claim-flip rate: 50.0% (25/50)** — of every claim the Prosecutor raised
across all 13 cases, half were REFUTED once actually verified against real
tests/tools. This is the headline number, not condition (c)'s raw
catch/FP rate — see Phase 31's full analysis above for exactly why (every
miss traces to a Prosecutor claim-taxonomy/recall gap, not a Verifier/Judge
defect; most of (c)'s "false positives" are real, confirmed findings about
a different, out-of-scope issue in the same file).

**Case set composition.** 13 cases: 5 synthetic bug/fix pairs (10 cases,
each ground-truthed by direct deterministic execution or static analysis
against the committed file, independent of any LLM call) + 2 reused
real-repo cases (ground truth carried over from real Phase 27-29 live
verification, diffs re-applied via `patch -p1` against a real clone) + 1
additional clean case (`type_mismatch_fixed`, completed in a follow-up
session via `adjudicate/benchmark/complete_phase31.py` once provider
quota reset). 6 buggy / 7 clean by ground truth.

**Real graph sizes indexed during this project's own verification work**
(useful for grounding "how big a repo has this actually been tested
against" if asked): `tartley/colorama` — 344 chunks, 184 graph
nodes/283 edges. `preyesparab/NutriForge` — 227 nodes/170 edges (after
the Phase 4/6 chunker+graph fixes). `harshal31718/pawn_ai` — 657
nodes/802 edges (the graph-physics stabilization-tuning case). Per-repo
FAISS/BM25 indexes and the JSON graph snapshot are all rebuildable
artifacts, not the source of truth (SQLite/PostgreSQL is).

**Key config constants worth having cold:**
- RRF: `rrf_k = 60`.
- Semantic cache: `CACHE_SIMILARITY_THRESHOLD = 0.95` (cosine) **plus** a
  lexical-compatibility gate (Jaccard ≥ 0.5 and no positive
  distinguishing-term conflict) — the cosine threshold alone is
  provably insufficient (see Phase 14).
- Chunking: AST chunks (functions/classes only), sliding-window chunks
  (50 lines, 40-line step / 20% overlap), parent chunks (~50 lines around
  each AST chunk, for small-to-big).
- Context builder: ~4000-token budget, ~4 chars/token heuristic.
- Reranker: ~40 candidates in, batched 32/call, top ~8 out.
- LLM token budgets: `LLM_MAX_TOKENS` default 1024 → 4096 (Phase 16 fix);
  `JUDGE_MAX_TOKENS = 2048` (Phase 30's own, larger override).
  `thinking_config.thinking_budget=0` set only when `response_schema` is
  supplied.
- Retry loops: `MAX_ATTEMPTS = 3` (Prosecutor, Judge, and — after the
  Phase 31 fix — the baseline reviewer).
- Rebuttal loop: `MAX_REBUTTAL_ROUNDS = 2` further rounds after the
  initial justification (3 rounds total max).
- Judge cap-handling: `CAP_UNRESOLVED_CONFIDENCE_CEILING = 0.7`.
- Graph physics: `PHYSICS_SCALE_NODE_THRESHOLD = 250` nodes (vis-network
  iteration/damping/minVelocity scaling kicks in above this, provably
  unchanged at or below it).
- Provider priority: Gemini → Groq → Ollama → raise (static, config-driven,
  not retry-on-failure — a deliberate, documented choice, see Phase 16).

**Known, disclosed, currently-unfixed limitations to have ready (never
minimize these — they are some of the strongest "what would you do
differently" material in the project):** TypeScript grammar unsupported
(`.ts`/`.tsx` files discovered, never parsed — zero chunks, zero graph
edges); Verifier sandboxing is process-level, not Docker (best-effort
network denial, inert memory limiting); `breaking_change` verification
checks "does the suite pass now," not a true before/after diff; a single
diff hunk whose real edit spans two functions still resolves to only one
`ChangedFunction`; `bandit`/`mypy` are Python-only, so `SECURITY`/
`TYPE_MISMATCH` claims about non-Python files are always
INCONCLUSIVE/LOW; nothing invalidates cached query embeddings when the
configured embedding model changes; no authentication, no rate limiting,
anywhere; no CI/CD pipeline exists yet.
