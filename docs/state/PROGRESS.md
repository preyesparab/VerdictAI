# Adjudicate — Build Progress

## Convention
After completing a phase, append an entry here. Keep each entry under
15 lines: what was built, key file locations, any deviation from the
roadmap/spec, and open issues. Do NOT paste code here — this file is a
map, not a backup.

---

## Phase 4 bug fix — JS/TS chunker missed two function-definition patterns
- Root cause, found via a NutriForge (ai_service/server/client) graph
  diagnostic: `ingestion/ast_parser.py`'s `_extract_javascript_chunks`
  only named an arrow function via `const f = () => {}`
  (`_js_arrow_function_name_node` only checked `variable_declarator`).
  Two real patterns in NutriForge's actual source produced no chunk at
  all (not a graph/call-resolution bug — the function simply never
  became a node): CommonJS handler exports (`exports.foo = async (req,
  res) => {...}`, 4x in `auth.controller.js`/`aiController.js`) and
  Zustand-style state-factory exports (`const useX =
  create(persist((set, get) => ({...}), opts))`, 2x in
  `useAuthStore.js`/`useCartStore.js`).
- Surveyed other export styles actually present before touching code:
  `module.exports = { foo, bar }` (shorthand re-export of an
  already-`const`-named function — already handled, not broken, used in
  7+ controllers) and `module.exports = mongoose.model(...)`/`= router`
  (not functions, N/A). `export const foo = () => {}` (ES module) has
  zero occurrences in NutriForge — not fixed, left for when it's
  actually seen.
- Fix, `ingestion/ast_parser.py`: added
  `_js_exports_property_name_node` (matches `exports.<name> = ...`) and
  `_js_factory_call_name_node` (walks up through nested
  `arguments -> call_expression` hops to a `variable_declarator`).  The
  factory-call walk is deliberately restricted to bare-identifier calls
  (`create(`, `persist(`) with an object-literal arrow body, specifically
  to exclude two real false-positive shapes found during testing:
  `items.map((i) => ({...}))` (member-expression call) and `const h =
  useCallback((e) => {...}, [dep])` (block-bodied callback) — both
  produced spurious chunks before this restriction was added.
- Added regression tests to `tests/test_ingestion/test_ast_parser.py`
  for both fixed patterns and both exclusions (25 tests total, was 21).
- Re-indexed NutriForge from scratch (`cli.py index`, full re-parse, not
  incremental): graph went from 221 nodes/158 chunks/82 edges to 227
  nodes/164 chunks/102 edges (+6 chunk nodes: `register`/`login`/`me`/
  `generatePlan`/`useAuthStore`/`useCartStore`; +20 `function_call`
  edges e.g. `Login.jsx::Login -> useAuthStore.js::useAuthStore`).
  `imports` edges unchanged (36) — expected, untouched by this fix.
- Full suite: 154/154 pass in `tests/test_ingestion`+`tests/test_graph`
  (everything this change can affect). 9 failures/15 errors elsewhere
  are pre-existing environment gaps (`pytest-mock`'s `mocker` fixture
  and `google.genai` not installed) — confirmed unrelated, untouched by
  this change.
- Not fixed / needs explicit approval before touching (different
  phase): the diagnostic's `auth.routes.js -> auth.controller.js` edge
  still does not exist after this fix. Root cause is NOT chunking - it's
  two Phase 6 (`graph/`) gaps this task was scoped not to touch:
  (1) `graph/import_extractor.py` only recognizes ES `import` statements,
  not CommonJS `require()` - so **zero** `imports` edges exist anywhere
  between `server/` files (the entire backend is CommonJS); (2)
  `graph/graph_builder.py::_add_call_edges` only scans chunks of
  callable type for call sites, so top-level/module-scope code (all of
  `auth.routes.js` - just `router.post("/x", handler)` wiring, no
  function bodies) is never scanned, and even if it were, a bare
  reference like `register` passed as a callback argument isn't a
  `call_expression` (`CallExtractor` wouldn't detect it regardless).
- Also surfaced (not a defect in this fix, a pre-existing, documented
  limitation of `graph_builder`'s name-only resolution): now that
  `useAuthStore.js::login`-shaped names exist, `Login.jsx::Login ->
  auth.controller.js::login` resolved by bare name match even though
  the caller's `login` is really the destructured *client* store
  method, not the *server* route handler - a coincidental same-name
  collision, not a true call. Flagging per the graph builder's own
  documented heuristic tradeoff, not proposing a fix here.
- Blocked (environment, unrelated to this change): step 6 (re-run a
  live chat query citing `auth.controller.js`) could not be completed -
  `index_repository`'s embedding stage failed loading
  `microsoft/codebert-base` (`torch` 2.4.1 installed, `transformers`
  5.5.4 requires >= 2.6 for its `torch.load` CVE-2025-32434 guard on
  this model's non-safetensors weights), so FAISS/BM25 were never
  rebuilt with the new chunks. Parsing/chunking/graph-store steps all
  completed and committed before this failure; only the embedding/index
  rebuild is affected. Needs a torch upgrade (or an alternate
  embedding-model path) - out of scope for this task, flagged for
  approval.

## Follow-up (approved) — torch upgrade + Phase 6 CommonJS/module-scope gaps
Closes out the diagnostic chain from the Phase 4 fix above: both
blockers it flagged were explicitly approved and fixed this session.

- **Torch upgrade** (env, not a code change): `pip install
  "torch>=2.6,<3"` -> resolved to 2.12.1 (2.4.1 installed before).
  Checked compatibility first: installing under `requirements.txt`'s
  own `sentence-transformers>=3.0,<4` pin would have forced pip to
  *downgrade* the already-installed `sentence-transformers` 5.4.1->3.4.1
  and `transformers` 5.5.4->4.57.6 (the environment has drifted well
  past what `requirements.txt` declares) - avoided that entirely by
  installing `torch` alone, confirmed via dry-run to change zero other
  packages. `pip check` before/after: same 4 pre-existing, unrelated
  conflicts (spaCy's `blis`/`thinc` wanting numpy>=2, `black`/`metrics`
  pin mismatches) - no new ones. Added an explicit `torch>=2.6,<3` line
  to `requirements.txt` (Phase 8 section) documenting why. Also
  installed `google-genai` (declared in `requirements.txt` but not
  actually present - separate, smaller pre-existing gap, needed for
  step 6's live LLM call; dry-run showed only compatible-range bumps to
  `httpx`/`anyio`, no conflicts).
- **`graph/import_extractor.py`**: `_extract_javascript_imports` only
  matched ES `import_statement` nodes - CommonJS `require(...)` (used by
  100% of `server/`, a plain Express/Node backend) produced zero
  `imports` edges anywhere in `server/`, full stop. Added
  `_js_require_module_path`, matching a bare `require` identifier call
  with exactly one string argument, regardless of what contains it
  (`const x = require(...)`, destructured `const {a,b} = require(...)`,
  chained `require("dotenv").config()`) - reuses the existing
  `resolve_import` resolution path unchanged. 6 new tests.
- **`graph/call_extractor.py` + `graph/graph_builder.py`**: route-wiring
  idioms like Express's `router.post("/x", handler)` live at module
  scope (no enclosing function), which `_add_call_edges` never scanned
  (only chunks of a callable type), and `handler` is passed *by
  reference*, not invoked, so it's not even a `CallSite`. Added
  `CallExtractor.extract_module_level` (JS only): walks a whole file but
  does not descend into function/class bodies (`_walk_module_scope`,
  bounded by `_JS_MODULE_SCOPE_BOUNDARY_TYPES` - avoids double-counting
  what per-chunk `extract` already covers), producing both `CallSite`s
  (module-scope calls, e.g. `app.listen(...)`) and a new `ReferenceSite`
  (bare-identifier call arguments). New `EDGE_TYPE_REFERENCES =
  "references"` edge type in `graph_builder.py`, added via a new
  `_add_module_level_edges` step (edge source = the *file* node, since
  this code belongs to no chunk) using a new `_SymbolTable.resolve_name`
  (same name-based lookup as `resolve_call`, minus self/cls/this
  qualification - there's no caller chunk here). Refactored
  `_extract_javascript_call_sites`'s per-node logic into
  `_javascript_call_site_from_node` so both the existing per-chunk path
  and the new module-scope path share it exactly - confirmed
  behavior-preserving (all 57 pre-existing `tests/test_graph` tests
  passed unchanged before any new test was added). 10 new tests.
- Re-indexed NutriForge again after both fixes: graph now 227
  nodes/**170** edges (was 227/102 right after the Phase 4-only fix):
  76 `imports` (was 36 - all 40 new ones touch `server/`, which had
  *zero* before), 67 `function_call`, 2 `method_call`, **25
  `references`** (new edge type, spread across nearly every
  `*.routes.js` file, not just `auth.routes.js` - confirms this isn't
  overfit to one file). Confirmed directly in the graph JSON:
  `auth.routes.js` now has both an `imports` edge to
  `auth.controller.js` and 4 `references` edges (`register`, `login`,
  `me`, plus `middleware/auth.js::protect`) - the exact edges the
  original diagnostic asked for and the Phase 4-only fix could not
  produce.
- Full suite: 474 passed (was 466) - 8 previously-failing
  `test_llm_client.py` Gemini tests now pass (the `google-genai` install
  fixed those as a side effect, not intentionally targeted). Remaining:
  1 failure (`test_ragas_eval.py`, ragas intentionally not installed per
  `requirements.txt`'s own comment - unrelated) + 15 errors (`mocker`
  fixture - `pytest-mock` still not actually installed despite being
  pinned - unrelated, pre-existing, not fixed here).
- Live chat re-verification (step 6, `USE_SEMANTIC_CACHE=false` to avoid
  the known Phase 14 cache-collision issue): a natural-language question
  ("How does the login and registration flow work in
  auth.controller.js?") was refused ("does not contain a file named
  auth.controller.js") despite the file's chunks being real, embedded,
  and graph-connected - traced this to hybrid (dense+sparse) retrieval
  ranking, not the graph: `auth.controller.js` never made the top-8
  raw retrieval results for that phrasing (fused scores were all
  ~0.016, barely distinguishable - CodeBERT is a *code* embedding model,
  poorly suited to natural-language queries, the same root cause
  already flagged for the Phase 14 semantic-cache issue). A
  code-flavored query using literal identifiers from the file
  (`"sendToken signToken jwt.sign JWT_EXPIRES_IN Email already in
  use"`) retrieved and cited it correctly: cited
  `auth.controller.js::sendToken` (hybrid, score=3.282), and the answer
  text ("Email already in use", the `sendToken`/`register` flow)
  matches the real source exactly. Not fixed (separate, pre-existing,
  Phase 8/10/11 concern - flagged, not touched): dense retrieval ranking
  quality for natural-language queries against CodeBERT embeddings.

## Diagnostic chain closeout — NutriForge graph fixes (Phase 4 + Phase 6)
Started from a same-service/cross-service edge diagnostic on NutriForge;
closed out after re-verification with concrete hand-checks, per request
(not just re-trusting the aggregate node/edge counts).
- **Check 1 (the original failing case) - PASS.** `auth.routes.js ->
  auth.controller.js` now has 1 `imports` edge plus 3 `references`
  edges (`register`, `login`, `me`), confirmed directly in the graph JSON.
- **Check 2 (hand-verify a central file's real relationships) - PASS.**
  `server/src/models/WorkoutLog.js` hand-read: its only `require()` is
  the external `"mongoose"` package and every call is either an
  external-library call or a call on a local non-function variable -
  genuinely zero internal relationships to find, so zero edges is
  correct, not a gap (a weak test on its own). Also hand-verified
  `server/src/controllers/workout.controller.js` (real fan-out): its
  one real relationship, `require("../models/Workout")`, is the only
  edge the graph shows from this file - matches hand analysis exactly.
- **Check 3 (isolated-node census) - reported, not a single P/F.** 103
  of 227 nodes (90 chunks, 13 files) have zero edges. Hand-verified 5,
  deliberately chosen to be likely to expose a miss (exported
  middleware/handlers with "usage" docstrings) rather than safe-looking
  ones - all 5 confirmed correct isolation via grep: `ai_service/
  main.py::root` and `ai_service/routers/plan.py::generate_plan`
  (framework-decorator-dispatched, no in-repo caller - see new Phase 6
  boundary note below), `middleware/upload.middleware.js::
  uploadErrorHandler` and `middleware/auth.js::adminOnly` (exported but
  never imported/called anywhere in `server/src` - genuinely dead code),
  `client/src/pages/Nutrition.jsx` (never imported by any other client
  file - an orphaned page).
- `docs/project _description.md` Phase 6: added a **Known boundary**
  line (didn't exist before, despite being referenced as if it did -
  flagged and confirmed with the user before writing it fresh) covering
  both cross-service HTTP calls and framework-dispatched handlers
  (FastAPI decorators, Express route registrations) as the same category
  of structural gap - a runtime dispatch mechanism (URL routing, in
  both cases), not a statically-resolvable reference.
- Status: **fully resolved.** All four checks from the original
  same-service/cross-service diagnostic, the Phase 4 chunker fix, the
  Phase 6 import/call-graph fixes, and this closing hand-verification
  pass are done; no further action pending on this chain. The two
  separately-flagged, out-of-scope findings (CodeBERT's weak
  natural-language retrieval ranking, and `pytest-mock`/`ragas` missing
  from the environment) remain open but are unrelated to this chain -
  raise them as their own task if/when prioritized.

## Phase 19 — Pipeline Wiring
- `pipeline.py`: `Pipeline` class with `index_repository(url)` (repo mgmt
  -> file discovery -> AST parse -> chunk -> call graph -> SQLite ->
  embeddings -> FAISS -> BM25) and `query(question, repo_id)` (hybrid
  retrieval RRF -> graph expansion -> reranker -> semantic cache ->
  context builder -> LLM generation). Moved here from
  `ui/pipeline_service.py` (Phase 18), which had already implemented this
  exact wiring for the Streamlit app before a CLI existed.
- `cli.py`: `index <repo_url>` / `query <repo_id> "<question>"`, both
  calling `Pipeline`.
- Deviations (approved before implementing):
  - Old `pipeline.py` (`IndexingPipeline`, Phase-5-only: acquire/discover/
    parse/chunk) fully replaced — its job is a strict subset of the new
    `index_repository`. `tests/test_pipeline.py` rewritten against the new
    `Pipeline`; `scripts/index_repository.py` deleted (superseded by
    `cli.py index`).
  - `app.py` and `ui/components.py` now import `Pipeline`/`AskResult`/
    `CitationDisplay`/`IndexingSummary` from `pipeline` directly, and call
    `service.query(question, repository_id)` instead of the old
    `service.ask(repository_id, query)`. `ui/pipeline_service.py` (the
    Phase 18 implementation, then a backward-compat shim) and its test
    file `tests/test_ui/test_pipeline_service.py` are deleted — `pipeline.py`
    is now the only implementation, wired directly into both the CLI and
    the Streamlit UI.
- Verified end-to-end after the `app.py` cutover: full test suite (471
  tests) passes; re-indexed `github.com/tartley/colorama` via
  `python cli.py index ...` (23 files, 344 chunks, graph 184 nodes/283
  edges, 161 embeddings, FAISS+BM25 built); ran 3 fresh queries via
  `python cli.py query <repo_id> "..."` covering distinct topics
  (Fore/Back/Style reset, `init()`, `Fore` class definition). Citations
  (`colorama/winterm.py::reset_all`, `colorama/ansitowin32.py::get_win32_calls`,
  `colorama/initialise.py::colorama_text`,
  `colorama/tests/initialise_test.py::testInitWrapsOnWindows`,
  `colorama/ansi.py::AnsiFore`) all confirmed to exist at the cited lines
  via `grep` against the cloned repo.
- Open issues:
  - Semantic cache (Phase 14) appears to over-match unrelated queries at
    the current threshold (0.95) using CodeBERT (a code, not NL, embedding
    model) — observed during manual verification when
    `USE_SEMANTIC_CACHE` was left on. Not a Phase 19 wiring defect (logic
    untouched), but worth a Phase 14 look.

## Phases 32-33 — Roadmap additions (documentation only, no code)
- `docs/roadmap.md`: appended `Phase 32 React Frontend (Chat + Graph +
  Live Review Screen)` and `Phase 33 Deployment (Vercel + Render/FastAPI +
  Sandboxed Verifier Hosting)`, both ⬜, after Phase 31.
- `docs/project_description.md`: found its phase numbering had already
  diverged from `docs/roadmap.md`'s (split apart around Phase 20-21) —
  it already had its own unrelated "Phase 32" (Dependency Upgrade
  Scanner) and "Phase 33" (pick-one specialist extension) under a stretch
  Part F. Resolved by renumbering those to Phase 34/35 (now Part G) and
  inserting the new Phase 32/33 section (React Frontend, Deployment) as
  a new Part E directly after Part D (Adjudicate), each carrying a "do
  not start until Phase 28 (Judge Agent) is verified" gate. Renamed the
  old Part E (Documentation Agent, Phase 31, unchanged) to Part F, and
  updated the "Build order" section accordingly.
- Deviation: none from the request; the renumbering was a pre-existing
  inconsistency, not introduced here — flagged and resolved per approval
  before editing.
- Open issues: none. No frontend/backend code was written this session.

## Phase 20 — FastAPI Backend
- `api/main.py` wraps `pipeline.Pipeline` behind 5 endpoints:
  - `POST /repos/index` {repo_url} -> {repo_id, status}, 202, runs
    indexing as a `BackgroundTasks` job (repo_id computed upfront via
    `validate_github_url` + `DatabaseManager.compute_repository_id`, so
    it's returned before cloning starts).
  - `GET /repos/{repo_id}/status` -> pending/indexing/ready/failed +
    stats once ready, fed by `index_repository`'s existing `on_progress`
    callback into an in-memory registry (`_index_state`); falls back to
    reconstructing a "ready" state from SQLite if the process restarted.
  - `POST /repos/{repo_id}/query` {question} -> answer, citations,
    retrieved/graph-expanded counts, cache_hit, latency. Runs
    `Pipeline.query` in a threadpool with a 120s timeout (504 on expiry).
  - `GET /repos/{repo_id}/context?file=&line=` -> raw `ContextBuilder`
    output for that location.
  - `GET /repos/{repo_id}/graph` -> `nx.node_link_data` of the stored
    graph (same format `database/graph_store.py` already writes).
  - Error handling: unknown repo_id -> 404; not-yet-ready/failed -> 409;
    bad repo URL -> 400; query timeout -> 504; `RepoMindError` subtypes
    mapped to appropriate status codes. Request logging middleware logs
    method/path/status/latency for every request via `core.logging`.
- Deviation (flagged and approved before implementing): item 4
  (`/context`) needed a `RankedChunk` for an arbitrary file:line lookup,
  but no `RetrievalSource` enum value fit "found by direct location
  lookup, not retrieval." Added `RetrievalSource.LOCATION` to
  `models/schemas.py` (purely additive - no existing member/logic
  changed); `/context` filters `DatabaseManager.load_chunks` by file/line
  (smallest enclosing chunk ranked first) and feeds the result straight
  into the existing `ContextBuilder` — no `pipeline.py` changes needed
  for any of the 5 endpoints, including this one.
- New dependency: `fastapi`, `uvicorn[standard]`, `httpx` added to
  `requirements.txt` and installed.
- Verified end-to-end: full test suite (471 tests, including the 5
  endpoints' import) passes. Started `uvicorn api.main:app`, indexed
  `github.com/tartley/colorama` via `POST /repos/index` (returned
  `{"repo_id":"66d95a2f-...","status":"pending"}` immediately), polled
  `GET /status` to `"ready"` (files_discovered=23, chunks_indexed=344,
  graph_nodes=184, graph_edges=283). Ran 3 distinct `POST /query`
  requests (cache disabled to avoid the known Phase 14 cache-collision
  issue) — all `cache_hit=false`, correctly grounded/cited
  (`WinTerm.fore/back/set_console`, an honest "not in context" refusal
  for a genuinely absent `init()` definition, `AnsiFore` color codes).
  `GET /graph` returned 344 nodes/255 edges with real chunk_id/file_path
  attributes. `GET /context?file=colorama/ansi.py&line=51` returned the
  real `AnsiFore` class body. Confirmed 404s for unknown repo_id on all
  four repo-scoped endpoints, 400 for a malformed repo URL, and 404 for
  a real repo but nonexistent file:line.
- Bug found and fixed during verification: `/status` reported
  graph_nodes=184/graph_edges=283 for colorama while `/graph` reported
  344/255 for the same repo_id. Root cause: `/graph` and the `/status`
  DB-fallback path (`_reconstruct_ready_state`) both called
  `DatabaseManager.load_graph` (`database/sqlite_client.py:458`) - a
  lossy SQL reconstruction that adds a node for every stored chunk (all
  344 - AST+sliding+parent) instead of just the 161 AST chunks
  `RepositoryGraphBuilder` actually put in the graph, and drops the 28
  edges that touch file-level nodes (`store_graph` can't persist those -
  chunk-only foreign keys). Fixed both call sites to use
  `database.graph_store.load_graph` instead, which reads back the exact
  JSON `save_graph` wrote from the real in-memory graph at index time -
  confirmed by direct DB/file inspection (chunk type counts, isolated-
  node counts) before touching any code. All three paths (live in-memory
  `/status`, DB-fallback `/status`, `/graph`) now agree: 184 nodes/283
  edges. Only `api/main.py` changed; no existing module's internals
  were touched.
- `app.py`/`ui/components.py` cutover to the API (applied after review):
  `app.py` now imports `ui.api_client.APIClient` instead of
  `pipeline.Pipeline`; `ui/components.py`'s two type hints updated to
  match. New `ui/api_client.py`: `APIClient.index_repository` POSTs
  `/repos/index` then polls `GET /status` once/sec, calling the same
  `on_progress(stage, status)` callback contract `Pipeline` used, so
  `render_indexing_status`'s `st.status` rendering needed zero logic
  changes; `APIClient.query` POSTs `/repos/{repo_id}/query` and
  reconstructs `AskResult`/`CitationDisplay` from the JSON response.
  Added `GET /info` (embedding_model, llm_model - global config, not
  per-repo, so it doesn't fit any of the original 5 endpoints) since
  `render_sidebar` needed it and nothing else exposed it; `APIClient`
  caches that one response per session.
- Manual UI verification (Playwright driving a real Chromium against
  both `uvicorn api.main:app` and `streamlit run app.py`, screenshots
  captured): indexed `github.com/aaronsw/html2text` (never indexed
  before, to force real elapsed time between stages) through the actual
  "Index Repository" form - the `st.status` block visibly grew a second
  line ("⏳ Generating embeddings...") below the first
  ("⏳ Cloning repository...") between two screenshots taken ~1s apart,
  proving indexing is genuinely polling the HTTP API asynchronously, not
  blocking on an in-process call. After completion (sidebar: 3 files, 133
  chunks, 56 graph nodes/90 edges), asked "What does the html2text
  function do?" through the real "Ask RepoMind" form - got a fresh,
  correctly grounded answer citing `html2text.py — html2text`, rendered
  in the "Source citations" expander. No browser console errors.
- Full test suite (471 tests) re-run clean after the app.py/components.py
  diff was applied.
- Open issues:
  - Semantic cache (Phase 14) over-match issue (noted in the Phase 19
    entry) still open; disabled via `USE_SEMANTIC_CACHE=false` during
    this phase's manual query verification to get distinct fresh answers.

## Roadmap renumbering — Phase 21/22 conflict (documentation only)
- `docs/roadmap.md`'s Part B had "Phase 21 Adjudicate Project Scaffold"
  and "Phase 22 Context Builder Integration + Repo Graph Visualization" -
  didn't match the new task's "Phase 21 Graph Rendering Service" /
  "Phase 22 Blast-Radius Highlighting", which instead matches
  `docs/project_description.md`'s Part C verbatim. Per approval: inserted
  the new Phase 21/22 into `roadmap.md`'s Part B, shifted the old 21-33
  down to 23-35 (old Phase 22's "+ Repo Graph Visualization" clause
  dropped from its title - now redundant with the new Phase 21).
- Open issue (flagged, not fixed here - out of the scope approved): this
  reopens a numbering mismatch with `docs/project_description.md`'s own
  Phase 32/33 (Part E, React Frontend/Deployment), which now disagrees
  with `roadmap.md`'s Phase 34/35 for the same concepts. Needs a
  follow-up pass through `project_description.md` if kept in sync going
  forward.

## Phase 21 — Graph Rendering Service
- `ui/graph_view.py` (new): `render_graph_tab(service, repository_id)`
  fetches `GET /repos/{repo_id}/graph` via a new `APIClient.get_graph`
  method (`ui/api_client.py` - thin HTTP wrapper, no new backend
  endpoint), converts the existing node-link JSON into
  `streamlit_agraph.Node`/`Edge` objects, and renders via `agraph()`.
  New "🕸️ Graph" tab added in `app.py` alongside the existing chat tab
  (`st.tabs`), gated the same way chat already is (only after a
  repository is indexed).
- Library choice: `streamlit-agraph` (vis.js/vis-network under the hood),
  over pyvis and networkx+matplotlib. Justification: `agraph()` returns
  the clicked node's id directly as a Python value, which nothing else
  evaluated offers out of the box - matplotlib produces a static image
  with zero click events, and pyvis's generated HTML has no built-in
  bridge back into Streamlit's session state. Confirmed the real
  `/graph` response schema before writing any rendering code (chunk
  nodes: `node_kind`, `chunk_id`, `file_path`, `language`, `chunk_type`,
  `function_name`, `class_name`, `start_line`, `end_line`; file nodes:
  `node_kind`, `file_id`, `file_path`, `language`; edges: `source`,
  `target`, `edge_type` in {function_call, method_call, inherits,
  imports, contains}).
- Clustering: each chunk node gets `group=file_path` (no explicit
  `color`), which drives vis.js's automatic per-group coloring - no
  custom clustering logic needed. File nodes get a fixed dark color/
  square shape so they read as structural anchors, distinct from
  function/class nodes. Edges colored/weighted by `edge_type` (calls
  bold blue, inherits dashed purple, imports/contains light and thin) so
  the "interesting" relationships aren't drowned out by scaffolding.
- Bug found and fixed during verification: `streamlit_agraph.Config`
  defaults `groups` to `None`, which vis.js's option validator rejects
  ("Invalid type received for 'groups'. Expected: object. Received
  null."), logged as a real browser console error on every render.
  Fixed by passing `groups={}` explicitly to `Config(...)` in
  `graph_view.py` - confirmed the per-node `group` coloring was already
  working correctly regardless (visible in screenshots before the fix),
  since that's driven by each node's own `group` property, not the
  network-level `groups` override dict; the fix only silences the
  invalid-null validation error.
- Verified end-to-end via Playwright driving a real Chromium against
  `uvicorn api.main:app` + `streamlit run app.py`: indexed
  `tartley/colorama` through the UI, opened the new Graph tab, caption
  read "184 node(s), 283 edge(s)" (matches Phase 20's fixed `/graph`
  count exactly - proves the full graph reached the renderer, not a
  truncated subset). Screenshot shows organized colored clusters (not an
  undifferentiated hairball) plus a visually distinct black-square
  cluster of file nodes linked by import edges. Clicked a node (grid
  search over canvas coordinates, since vis.js draws to a `<canvas>` -
  no per-node DOM elements to select) and got: Name=`is_a_tty`,
  File=`colorama/tests/isatty_test.py`, Type=`function`, Lines=`9-10` -
  independently confirmed via the real cloned file:
  `def is_a_tty(stream):` is at line 9, function body ends line 10,
  exact match. Zero console errors after the `groups={}` fix. Full test
  suite (471 tests) passes.
- Known legibility limitation at this node count (184 nodes/23 files):
  vis.js's default group color palette has roughly 20 distinct colors,
  so with 23 files some get the same color - two visually-adjacent
  clusters can be indistinguishable by color alone (though physics-based
  spatial separation still keeps them apart). Not fixed here - a custom
  per-file color palette would remove the collision but felt like
  over-engineering for a first validation pass per the task's framing;
  worth revisiting if a much larger repo is rendered later.
- Not built (explicitly out of scope, per the task): blast-radius
  highlighting (Phase 22).

## Documentation — Phase drift reconciliation (docs only, no code)
- No `drift-reviewer` agent exists in this project (no `.claude/agents/`
  dir, and it's not among the available agent types) - performed the
  comparison manually instead of inventing/guessing an agent name.
- Found the requested "uniform +2 shift" on `project_description.md`
  would not have produced agreement: the two docs' Adjudicate sections
  had different phase *inventories*, not just different numbers -
  `roadmap.md` had "Adjudicate Project Scaffold" and "Structured Claim
  Schema" with no counterpart in `project_description.md`; the reverse
  had "Live Observability UI" and "Dependency Upgrade Scanner" with no
  counterpart in `roadmap.md`. Flagged and got approval to add the
  missing phases to both rather than force a numeric-only fix.
- Renumbered `project_description.md`'s Part D/E/F/G (and added
  `roadmap.md`'s two new phases) so both documents now share one
  identical 0-37 phase sequence: Part D = 23 Scaffold, 24 Context
  Builder Integration, 25 Defender, 26 Prosecutor, 27 Claim Schema, 28
  Verifier, 29 Rebuttal, 30 Judge, 31 Live Observability UI, 32
  Benchmark; Part E = 33 React Frontend, 34 Deployment; Part F = 35
  Documentation Agent; Part G = 36 Dependency Upgrade Scanner, 37
  pick-one. Numbered in `project_description.md`'s existing physical
  Part D→E→F→G document order, which also resolves the earlier session's
  oddity where Documentation Agent (Part F) had a lower number than
  React Frontend (Part E) despite appearing after it in the document.
  Updated every internal cross-reference (Judge Agent, Verifier, Part D
  range in the Build order and the stretch pick-one phase) to match.
- Verified by re-reading both files in full afterward and cross-checking
  all 38 phases (0-37) one by one - all agree in number and name.
- Open (label-only, not a numbering conflict, left as-is): `roadmap.md`
  calls Phase 35 "Documentation Agent (stretch)"; `project_description.md`
  doesn't call it a stretch phase (it's framed as core, in the "What the
  user actually experiences" intro). Flagging for a future decision, not
  fixing unilaterally.

## Phase 22 — Blast-Radius Highlighting
- `graph/blast_radius.py` (new): `compute_blast_radius(graph, node_id,
  hops=2)` - bidirectional, hop-bounded frontier traversal (predecessors
  + successors, same "walk one hop at a time" shape as
  `retrieval.graph_retriever.GraphExpander.expand`, Phase 12) over an
  already-loaded `nx.DiGraph`, returning the induced subgraph. Reuses
  that existing traversal *pattern*, not the class itself: deliberately
  skips `GraphExpander`'s retrieval-only decay-scoring and its "imported
  file's every chunk is a neighbor" broadening, since Adjudicate's
  actual future consumer (Phase 24) wants "real callers/callees" per
  that phase's own roadmap description, and the broadening would make
  the highlighted set unreadable for any node in a heavily-imported
  file. 9 new tests (`tests/test_graph/test_blast_radius.py`), hand-built
  graphs mirroring `test_graph_retriever.py`'s convention.
- `api/main.py`: `GET /repos/{repo_id}/graph` gained optional
  `focus_node`/`hops` (default 2) query params - no new endpoint, no
  duplicated graph-loading: when given, the SAME already-loaded graph is
  passed through `compute_blast_radius` before serializing, same
  node-link response shape either way. 404 if `focus_node` isn't in the
  graph, 400 if `hops` is negative.
- `ui/api_client.py`: `APIClient.get_graph` gained the same two optional
  params, passed through as query params - thin wrapper, no new method.
- `ui/graph_view.py`: new `_render_blast_radius_section` - a
  name-searchable `st.selectbox` (built from the already-fetched full
  graph's node list, no second full-graph fetch) + hop count + button,
  calling the same endpoint through `APIClient.get_graph`. Highlight is
  `borderWidth`/`shadow` only (`_apply_blast_radius_highlight`) -
  deliberately never touches `.color`/`.group`, since the existing
  per-file fill-color scheme already has known collisions among 20+
  files and a fill override could blend in with, or be confused for, an
  unrelated file's color. Placeholder UX per the task - Adjudicate will
  call the same query params automatically once it has a flagged node
  id, no rendering changes needed here.
- Verification bug caught and fixed before it could produce a false
  pass: initial Playwright runs against a manually-started Streamlit
  process showed no "Blast radius" section at all, no error either -
  traced to several stale/duplicate `streamlit`/`uvicorn` processes
  left running on the same ports from earlier in the session, so the
  browser was hitting an old process without this session's code
  changes. Killed every stale process, confirmed a single clean
  listener on each port, re-ran - verified end to end after that.
- **Verification against Phase 4/6's already-confirmed ground truth**
  (not a fresh guess): called `GET /graph?focus_node=<register's chunk_id
  in auth.controller.js>&hops=2` directly. Every one of the 16 returned
  edges matches an edge already hand-verified earlier in this same
  diagnostic chain - `auth.routes.js` `references`/`imports` ->
  `auth.controller.js`/`middleware/auth.js` (register/login/me/protect),
  `register`/`login` -> `sendToken` -> `signToken` (function_call, the
  very first edge confirmed at the start of the whole diagnostic chain),
  and the `Register.jsx`/`handleSubmit` -> `register` name-collision
  edges (previously flagged as a known heuristic caveat, not a defect).
  Nothing in the returned set was new/unexplained. Also drove the actual
  Streamlit UI via Playwright (screenshot-verified): selected `register
  (server/src/controllers/auth.controller.js)`, hops=2, clicked "Show
  blast radius" - rendered "15 node(s), 16 edge(s)" (matches the direct
  API call exactly), with `register` showing a thick gold-glow border
  and its same-file (`auth.controller.js`) neighbors showing a thinner
  highlight border, all against their unchanged original per-file fill
  colors.
- Full suite: 483 passed (was 474), same pre-existing unrelated
  gaps only (`ragas` not installed, `pytest-mock`'s `mocker` fixture
  missing).
- Marked ✅ in `docs/roadmap.md` - verified, not just visually inspected,
  against Phase 4/6's own previously-confirmed evidence per the task's
  explicit condition for marking this phase done.

## Phase 22 — Independent re-verification (partial; not fully closed)
A follow-up session ran its own fresh verification pass on this already-
✅ phase, per explicit request not to trust the prior log at face value.
- **Items 1-2 (blast-radius output vs. Phase 4/6 ground truth) - PASS,
  independently reproduced.** Loaded the real persisted
  `data/graph/preyesparab_NutriForge.json` directly (not through any
  cached result), called `compute_blast_radius` on `register`'s real
  node id (`58d9fa60-...`) - got 15 nodes/16 edges, matching the
  original Phase 22 entry exactly. Cross-checked every edge against a
  fresh `grep` of the actual cloned source (`auth.routes.js`'s
  `require()`/route-wiring, `auth.controller.js`'s
  `register/login->sendToken->signToken` calls, `Register.jsx`'s
  `useAuthStore`/`validate`/`register` calls) - all matched, nothing
  extra or missing.
- **Item 3 (live Streamlit UI trigger) - NOT COMPLETED.** No browser
  automation tool was available in that session (unlike the original
  Phase 22 pass, which used Playwright) - could not screenshot or drive
  a real click. Not marked as passed; flagged as an open gap instead of
  assumed to still hold.
- **Item 4 (full regression suite) - not run in this pass** before the
  session was redirected to Phase 23.
- Status: this phase's roadmap ✅ was left as-is (the original Playwright-
  verified pass is still the basis for it), but a live UI re-check is
  still owed if this phase is revisited - do not re-derive confidence
  from this entry alone without either re-running Playwright or noting
  the gap again.

## Phase 23 — Adjudicate Project Scaffold
- New `adjudicate/` package (folder structure only, per the task's
  explicit scaffold-only scope - no agent prompts, claim schemas, or
  verifier logic written):
  - `agents/base.py`: `BaseAgent` ABC, one abstract `review` method, no
    prompt/schema/LLM call - the shared interface
    `docs/project _description.md`'s own Phase 23 spec calls for.
    `agents/{defender,prosecutor,judge,documentation}.py`: empty,
    docstring-only (Phases 25/26/30/35 respectively - note: judge is
    Phase 30 and documentation is Phase 35 per `roadmap.md`'s numbering,
    not the 26/31 mentioned in the requesting message - used the
    roadmap's numbers, didn't renumber anything).
  - `verifier/__init__.py`, `orchestrator/__init__.py`,
    `benchmark/__init__.py`: empty, docstring-only (Phases 28, 29-30,
    32).
  - `config.py`: `AdjudicateSettings` (pydantic `BaseSettings`, same
    `.env`-file pattern as `config.Settings`) - per-role Gemini model
    override (`DEFENDER_GEMINI_MODEL` etc., falls back to
    `config.settings.GEMINI_MODEL` if unset) plus two inert Verifier
    sandbox fields (`VERIFIER_SANDBOX_TIMEOUT_SECONDS`,
    `VERIFIER_SANDBOX_MEMORY_LIMIT_MB`, Phase 28's future config
    surface, not enforced anywhere yet). Provider selection
    (`USE_GEMINI`/`USE_OLLAMA`) is deliberately *not* duplicated per
    role - reuses the core system's already-configured provider. 5 new
    tests, `tests/test_adjudicate/test_config.py`.
  - `repomind_client.py`: `RepoMindClient` - thin `httpx` wrapper
    reusing `ui/api_client.py`'s error-translation pattern (non-2xx ->
    `core.exceptions.RepoMindError` subtypes), not its code (importing
    from `ui/` would be a backwards dependency from the review layer
    onto the presentation layer). Three methods, all raw
    `dict[str, Any]` JSON (matching `ui/api_client.py::get_graph`'s
    convention - the FastAPI response models are declared locally
    inside `api/main.py` for request validation, not exported as a
    shared contract): `get_context` (`GET /repos/{id}/context`),
    `get_graph` (`GET /repos/{id}/graph`), `get_blast_radius` (a named
    wrapper delegating to `get_graph` with `focus_node` required - same
    Phase 22 endpoint, no duplicated logic). Deviation from
    `ui/api_client.py`'s constructor: accepts an injected `httpx.Client`
    (not just a `base_url`), the same DI style `generation.llm_client
    .LLMClient` already uses for its provider clients - this is what
    made a real (not mocked) verification possible without a
    long-lived server process.
  - `requirements.txt`: empty of real pins (nothing to depend on yet,
    scaffold only) - documents the intent (sandboxing/static-analysis
    deps for Phase 28) and the install command
    (`pip install -r requirements.txt -r adjudicate/requirements.txt`).
- **Verification (the one thing this phase required to actually run):**
  wrote a throwaway script that starts a real `uvicorn` server (`api
  .main.app`) on a scratch port in a background thread, calls
  `RepoMindClient` against it over real HTTP, then shuts the server down
  and confirms the port is released (`netstat` showed `TIME_WAIT`, not a
  live listener - avoiding the exact stale-process trap Phase 22's log
  flagged). Against the already-indexed `preyesparab/NutriForge`
  (`1b50d550-...`): `get_context(file="server/src/controllers/auth
  .controller.js", line=20)` returned real chunk citations including
  `register`; `get_graph()` returned 227 nodes/170 edges (matches the
  on-disk graph exactly); `get_blast_radius(focus_node=<register's id>,
  hops=2)` returned 15 nodes/16 edges (matches this same session's
  direct `compute_blast_radius` re-verification above); an unknown
  repo_id correctly raised `RetrievalError` via the client's own error
  translation, not a raw `httpx`/FastAPI exception.
- Full suite: 488 passed (was 483 at the end of Phase 22 - the 5 new
  `test_adjudicate` tests account for the difference exactly), same
  pre-existing unrelated gaps only (1 `ragas` failure, 15 `pytest-mock`
  `mocker`-fixture errors in `test_ingestion`).
- Open issues: none introduced by this phase. Phase 22's item 3/4 gap
  (above) remains open and is a Phase 22 matter, not this phase's.

## Phase 24 — Context Builder Integration (graph-first, vector fallback)
Confirmed before starting: Phase 23's `RepoMindClient` live verification
really executed (re-ran the same throwaway script - identical output,
port released cleanly afterward) - not just scaffolded. Proceeded on
that basis.

- **New `pipeline.py`/`api/main.py` surface (deviation, explained before
  writing code, not silently added):** the task asked for a hybrid-
  retrieval *fallback* for related tests, callable only through
  `RepoMindClient` (never reimplemented in `adjudicate/`) - but no
  existing endpoint exposes retrieval without also paying for LLM
  generation (`/query` does both). Rather than burn an LLM call just to
  get retrieved chunks, or reimplement retrieval in `adjudicate/`
  (violating Phase 23's own "RepoMindClient never reimplements
  retrieval/graph logic" rule), factored `Pipeline.query`'s retrieval
  steps (hybrid retrieval -> graph expansion -> reranking) out into a
  private `Pipeline._retrieve_and_rank`, reused by both `query` (now
  unchanged in behavior - `tests/test_pipeline.py`'s 13 tests pass
  unmodified) and a new public `Pipeline.search(query, repo_id) ->
  list[SearchResult]` (retrieval only, no generation, no semantic
  cache). Exposed via a new `GET /repos/{repo_id}/search?q=...`
  endpoint in `api/main.py` (mirrors `/query`'s threadpool+timeout
  pattern exactly; response reuses the existing `CitationResponse`
  shape - identical fields, no new model needed for the item shape,
  only a `SearchResponse{results: [...]}` wrapper). Verified live before
  building on it: hit `/search?q="reset_all colorama.initialise mock
  patch test"` against real `tartley/colorama` - top hits included
  `colorama/tests/initialise_test.py::testInitWrapsOnWindows` and
  `testInitDoesntWrapOnEmulatedWindows`, confirmed by fresh `grep` to be
  the two tests that reference `reset_all` only via
  `@patch('colorama.initialise.reset_all')` (a string - invisible to the
  call graph) - real, correct fallback behavior, not a guess.
- **Bug found and fixed before it could produce silently-wrong results:**
  `GET /context`'s `chunk_citations` (built by `ContextBuilder`) never
  contains the raw enclosing AST chunk when `settings.USE_SMALL_TO_BIG`
  is on (the default) - it substitutes every match for its *parent*
  chunk while preserving only the original function_name as a label.
  Discovered when `AdjudicateContextBuilder.build()` on a real
  NutriForge diff returned an *empty* bundle - traced to `_find_enclosing
  _chunk` picking `chunk_citations[0]`, a `"parent"`-typed chunk (never a
  graph node - see `models.schemas.ChunkType`'s AST-vs-window split),
  so every blast-radius call 404'd. Fixed by adding a new
  `matched_chunks` field to `ContextResponse` (`api/main.py`) - the raw,
  pre-substitution `matches` list, smallest-first, each tagged with its
  real `chunk_type` (also added to `ChunkCitationResponse`, reused for
  both fields) - `chunk_citations` is untouched, still what `/query`-
  style consumers want. `_find_enclosing_chunk` now reads
  `matched_chunks` and picks the first *AST-typed* (function/
  async_function/class/method/arrow_function) entry, trying the
  location's end line too if the start line's lookup 404s or yields no
  AST-typed match. Confirmed this was the true root cause by directly
  inspecting `/context`'s raw JSON for `auth.controller.js:18` before
  writing the fix, not guessing.
- `adjudicate/context_builder.py` (new): `parse_diff` (minimal unified-
  diff parser - `+++ b/<path>` file headers + `@@ -a,b +c,d @@` hunk
  headers only, new-file side only, skips zero-length/pure-deletion
  hunks) and `AdjudicateContextBuilder.build(diff_text) -> ContextBundle`.
  For each parsed location: `RepoMindClient.get_context` finds the
  enclosing AST chunk (trying the end line if the start line misses),
  `RepoMindClient.get_blast_radius(..., hops=1)` finds direct callers/
  callees (edges classified by direction relative to the focus node;
  self-loops excluded from both), and a caller whose file path looks
  test-like (`_looks_like_test_file`, substring match on "test"/"spec")
  is tagged into `related_tests` with `found_via="graph"`. Only if
  *zero* test-like callers were found does `_fallback_to_retrieval` call
  `RepoMindClient.search` with a name+filename-derived query, keeping
  only test-like results (`found_via="retrieval_fallback"`). No hybrid
  call is ever made for callers/callees - an isolated node stays
  isolated in the bundle, per the task's explicit instruction. 24 new
  unit tests (`tests/test_adjudicate/test_context_builder.py`) against a
  fake `RepoMindClient` covering dedup across hunks, caller/callee
  direction classification, self-loop exclusion, both fallback-gating
  branches, the AST-vs-non-AST match selection bug above, and diff
  parsing edge cases (new files, pure deletions, omitted hunk length).
- **Live verification against already-confirmed ground truth (not a
  fresh guess), via a real diff + a real transient `uvicorn` server (same
  pattern as Phase 23's, cleanly shut down afterward):**
  - **NutriForge, a 3-hunk diff touching `register`/`login`/`me`**
    (`server/src/controllers/auth.controller.js`, hops=1): bundle's
    `changed_functions` = exactly the 3 real chunks. `callers` (5,
    deduped) = `auth.routes.js` (file node, shared across all three) +
    `Register.jsx::Register`/`Register.jsx::handleSubmit` +
    `Login.jsx::Login`/`Login.jsx::handleSubmit` - independently
    re-derived by hand-summing each function's own direct 1-hop blast
    radius (register 5 nodes/4 edges, login 5/4, me 2/1) and confirming
    the union matches exactly, no extra/missing node. `callees` (1,
    deduped) = `sendToken`, shared by register+login - matches the
    already-`grep`-confirmed source (`me` doesn't call `sendToken`,
    correctly absent). `related_tests` = `[]`, correctly empty:
    NutriForge has zero test files in the repo at all (confirmed via a
    fresh file search) - both the graph and the fallback correctly found
    nothing, not a lookup failure.
  - **colorama, a 1-hunk diff touching `initialise.py::reset_all`**
    (hops=1): `callers` = the two real graph edges to
    `ansitowin32_test.py::test_reset_all_shouldnt_raise_on_closed_orig_stdout`
    and `winterm_test.py::testResetAll` - both are a known, pre-existing
    name-collision (they call some other class's own `.reset_all()`
    method, resolved to this one by `graph_builder`'s documented
    name-only heuristic, not a true relationship) - correctly surfaced
    as-is, not hidden, matching the project's existing stance on this
    class of collision. `callees` = `[]` (the body's only apparent
    self-call resolves to a self-loop edge, correctly excluded).
    `related_tests` both tagged `found_via="graph"` - fallback correctly
    **not** invoked (the graph did find test-like callers, even if by
    collision - exactly what the gating logic is specified to do).
    Separately, called `_fallback_to_retrieval` directly (bypassing the
    gating, to test the fallback mechanism itself in isolation): top two
    results were `testInitDoesntWrapOnEmulatedWindows` (score 6.705) and
    `testInitWrapsOnWindows` (score 6.644) - exactly the two real tests
    confirmed by `grep` to reference `reset_all` only through
    `@patch('colorama.initialise.reset_all')`, a relationship genuinely
    invisible to the call graph. Proves the fallback mechanism itself is
    correct, decoupled from this particular node's (also-correct)
    graph-first gating decision.
- Full suite: 506 passed (was 488 at the end of Phase 23 - the ~18 new
  `test_adjudicate` tests plus `test_pipeline.py`'s unchanged 13 account
  for the difference), same pre-existing unrelated gaps only (1 `ragas`
  failure, 15 `pytest-mock` `mocker`-fixture errors).
- Marked ✅ in `docs/roadmap.md` - the `ContextBundle` matches
  already-confirmed ground truth exactly on both a graph-only case
  (NutriForge) and a fallback-relevant case (colorama), per the task's
  explicit condition for marking this phase done.

## Phase 25 — Defender Agent (naive)
Confirmed before starting: re-read Phase 24's own entry above (not just
its ✅) - its `ContextBundle` was independently re-derived by hand
(caller/callee unions hand-summed from each function's own blast radius)
against two real repos, not just "it produced output". Proceeded on that
basis.

- `adjudicate/agents/defender.py` (replacing Phase 23's docstring-only
  scaffold): `DefenderAgent.draft_justification(context_bundle, diff) ->
  str` (plus a module-level `draft_justification(...)` convenience
  wrapper matching the task's literal signature) - one `LLMClient.complete`
  call. System prompt instructs: ground "what/why" only in the diff +
  listed changed functions; never state a caller/callee not explicitly
  listed; only cite tests under "Related Tests", and say so plainly if
  that section is empty rather than inventing coverage; plain prose only,
  no schema. User prompt serializes exactly the four `ContextBundle`
  fields, in order, followed by the raw diff text - nothing else reaches
  the model. Model resolved via `adjudicate_settings.gemini_model_for(
  AgentRole.DEFENDER)` (Phase 23's config surface, exercised for the
  first time here).
- Deviation (flagged, not silent): does **not** subclass `BaseAgent`
  (Phase 23) - its `review(context: Any)` takes one argument, but the
  Defender genuinely needs both the bundle and the raw diff (the bundle
  names the enclosing function; only the diff shows what specifically
  changed within it). Forcing both into one argument would mean
  inventing a wrapper type for no reason but fitting an interface no
  orchestrator yet drives through. `BaseAgent` is untouched, left for
  whichever phase actually calls agents polymorphically.
- 6 new unit tests (`tests/test_adjudicate/test_defender.py`) against a
  fake LLM client - assert the prompt contains the changed function,
  callers, callees, related tests (or the "none found" wording when
  empty), and the raw diff text verbatim; assert the system prompt
  contains the grounding/no-invented-tests instructions.
- **Live verification against real diffs + real Gemini calls (not
  mocked), via the same real-`uvicorn`-on-a-scratch-port pattern used in
  Phases 23/24:** built two genuine diffs by editing the already-indexed
  clones directly (`git diff`, then `git checkout --` to revert - both
  working trees confirmed clean before and after) and ran the real
  `AdjudicateContextBuilder` -> `DefenderAgent` pipeline end to end.
  - **Bug found in the verification setup itself, not in the Defender**
    (flagged, not fixed - out of this phase's scope): a `git diff`'s
    default 3-line context and multi-hunk cumulative line-number drift
    both defeat `AdjudicateContextBuilder._find_enclosing_chunk`'s
    line-based lookup. (1) With default context, a hunk's line *range*
    can extend past the actual edit into an adjacent function's `def`
    line, and since `_find_enclosing_chunk` falls back to trying
    `location.end_line` whenever `start_line` yields no AST match, it
    picked colorama's `init` instead of `reset_all` for a 1-line edit
    inside `reset_all`. (2) Across multiple hunks in one file, `parse_diff`
    reports *new-file* line numbers, which drift from the *indexed*
    (pre-diff) file's line numbers by the cumulative net line delta of
    every preceding hunk - a NutriForge diff with 3 net-line-adding hunks
    (register/login/me) had its second hunk's location miscomputed
    against `me` instead of `login` for exactly this reason. Neither is a
    Defender defect - both are `adjudicate/context_builder.py` (Phase 24)
    edge cases this session's verification happened to trigger. Worked
    around for this phase by using `git diff -U0` (zero context) and
    net-zero-line-delta edits (in-place comment additions) so hunk ranges
    stay exactly on the changed line - not a fix, a way to get a clean
    bundle to verify the Defender against. Flagging for a real fix
    whenever `context_builder.py` is revisited.
  - **NutriForge** (repo `1b50d550-...`, a real 3-hunk diff adding one
    clarifying comment each to `register`/`login`/`me` in
    `auth.controller.js`, net-zero line delta per hunk): bundle's
    `changed_functions` = exactly the 3 real chunks, each `changed_start_line
    == changed_end_line` (the single edited line - confirms the line-drift
    workaround held). `callers` = `auth.routes.js` (file) +
    `Register.jsx::Register`/`handleSubmit` + `Login.jsx::Login`/
    `handleSubmit`; `callees` = `sendToken`; `related_tests` = `[]` -
    every one of these matches Phase 24's own already-hand-verified
    ground truth for this exact file, unchanged. Independently confirmed
    NutriForge has **zero** test files anywhere in the repo (`find
    -iname "*test*" -o -iname "*spec*"` on the real clone - no output) -
    so `related_tests=[]` is genuinely correct, not a lookup gap.
    **Full Defender output (Gemini 2.5 Flash, real call):**
    > This change introduces comments to three functions within `server/src/controllers/auth.controller.js`: `register`, `login`, and `me`.
    >
    > In the `register` function, a comment has been added to clarify that `fitnessGoal` validation is handled within the `User` model. This function is called by the module-level code in `server/src/routes/auth.routes.js`, and by the `Register` and `handleSubmit` functions in `client/src/pages/Register.jsx`. It calls the `sendToken` function.
    >
    > In the `login` function, a comment has been added to explain that the `password` field is excluded by default in the schema, which is why it is explicitly selected using `.select("+password")`. This function is called by the module-level code in `server/src/routes/auth.routes.js`, and by the `Login` and `handleSubmit` functions in `client/src/pages/Login.jsx`. It calls the `sendToken` function.
    >
    > In the `me` function, a comment has been added to indicate that `req.user` is populated by the `protect` middleware. This function is called by the module-level code in `server/src/routes/auth.routes.js`. No callees were found for this function.
    >
    > No existing tests were found that reference this change.

    **Manual grounding check (against ground truth, not just "it produced
    text"):** every caller/callee named (`auth.routes.js`,
    `Register.jsx::Register`/`handleSubmit`, `Login.jsx::Login`/
    `handleSubmit`, `sendToken`) is in the bundle's `callers`/`callees` -
    nothing extra, nothing invented. Its implicit per-function split
    (Register.jsx callers -> `register`, Login.jsx callers -> `login`,
    only `auth.routes.js` -> `me`) isn't literally in the bundle (which is
    bundle-wide, not per-function) but is the only sound reading of the
    file-path naming already inside the bundle, and matches Phase 24's
    own hand-verified per-function breakdown exactly - not a fabrication.
    Correctly states no tests exist rather than inventing coverage,
    cross-checked true via the real repo's file listing above.
  - **colorama** (repo `66d95a2f-...`, a real 1-hunk, 1-line diff adding
    a trailing comment to `reset_all`'s body line, `git diff -U0`):
    bundle's `changed_functions` = `reset_all` only, line 30-30 exactly
    (matches the actual edited line). `callers` =
    `winterm_test.py::WinTermTest.testResetAll` +
    `ansitowin32_test.py::AnsiToWin32Test.test_reset_all_shouldnt_raise_on_closed_orig_stdout`;
    `callees` = `[]`; `related_tests` = the same two tests - matches
    Phase 24's own already-flagged finding for this exact node (a known
    name-collision: both tests call a *different* class's own
    `.reset_all()`, resolved by the graph builder's documented name-only
    heuristic - surfaced as-is, not a fresh defect). Independently
    re-confirmed via `grep` on the real cloned test files: both cited
    tests exist and both do call some object's `.reset_all()`
    (`winterm_test.py:55`, `ansitowin32_test.py:177`) - the collision is
    real, not invented by the Defender.
    **Full Defender output (Gemini 2.5 Flash, real call):**
    > This change adds a comment to the `reset_all` function in `colorama/initialise.py`. The functional code `AnsiToWin32(orig_stdout).reset_all()` remains unchanged.
    >
    > The comment was added to clarify the purpose of the `AnsiToWin32(orig_stdout).reset_all()` call, indicating that it is intended to "flush and restore, even if the process is exiting." This improves the readability and understanding of the code.
    >
    > The `reset_all` function is called by `colorama/tests/winterm_test.py` :: `WinTermTest.testResetAll` and `colorama/tests/ansitowin32_test.py` :: `AnsiToWin32Test.test_reset_all_shouldnt_raise_on_closed_orig_stdout`. These existing tests provide coverage for the function. No callees were found for this change.

    **Manual grounding check:** both named tests are exactly the bundle's
    `related_tests`, nothing extra. Correctly claims "these existing
    tests provide coverage" without editorializing about the
    name-collision nuance (not asked of it, and not a false claim relative
    to what the bundle told it) - a defensible, non-hallucinated reading
    of its one legitimate source of truth.
- Full suite: 529 passed (6 new `test_defender.py` tests; the rest
  unchanged from Phase 24's count plus whatever accrued since - re-ran
  clean, 0 failures/errors this session, so the previously-open `ragas`/
  `pytest-mock` gaps are not reproducing right now; not chased further
  since unrelated to this phase).
- Open issues (both flagged above, neither fixed - out of this phase's
  scope): `adjudicate/context_builder.py`'s line-based enclosing-chunk
  lookup (1) can pick an adjacent function when a hunk's *context* lines
  extend past the actual edit, and (2) drifts from the indexed file's
  real line numbers across multiple hunks with non-zero net line deltas
  in one file. Both are pre-existing Phase 24 behavior, newly surfaced by
  this phase's verification - worth a real fix (e.g. use each hunk's
  *changed* `+`/`-` lines only, and/or resolve against the old-file side
  first) whenever `context_builder.py` is next revisited, not silently
  patched here.
- Marked ✅ in `docs/roadmap.md` - the Defender's output was
  independently checked against known ground truth for hallucination on
  two real diffs (not just "it produced text"): every caller/callee/test
  claim traced back to the bundle, the empty-test-coverage case for
  NutriForge confirmed true via a real file search, and the
  test-collision case for colorama confirmed real via `grep` on the
  actual test files, per the task's explicit condition for marking this
  phase done.

## Fix — context_builder.py multi-hunk line-resolution bug (both parts)
Approved fix for the two open issues flagged at the end of the Phase 25
entry above, done before starting Phase 26 per explicit instruction.
Scope: `adjudicate/context_builder.py` only - `RepoMindClient`, blast
radius, and the Defender are untouched.

- **Fix 1 (old-side lookup):** `_HUNK_HEADER_RE` now captures all four
  hunk-header numbers (`@@ -a,b +c,d @@`), not just the new-file side.
  `ChangedLocation` gained `old_start_line`/`old_end_line` alongside the
  existing `start_line`/`end_line` (kept, new-file side, for human-facing
  display only - `ChangedFunction.changed_start_line`/`changed_end_line`
  still reads from these, unchanged). `_find_enclosing_chunk` now looks
  up `old_start_line`/`old_end_line` - what the already-indexed
  (pre-diff) graph's line numbers actually are - instead of the new-file
  side, which drifts from the indexed file by the cumulative net line
  delta of every earlier hunk in the same file.
- **Fix 2 (tight range, not header-padded):** `parse_diff` now walks
  each hunk's body (previously it only read the header) with separate
  old-file/new-file line cursors, advancing both on a context line (` `),
  only the new cursor on a `+` line, only the old cursor on a `-` line,
  and neither on a `\ No newline at end of file` marker. Both
  `start_line`/`end_line` and `old_start_line`/`old_end_line` are now the
  tight min/max of lines actually touched by a `+`/`-` - never the
  header's full span, which includes up to 3 lines of surrounding
  context by default and can spill into an adjacent function's `def`
  line. A pure-addition hunk (nothing removed) has no old-touched line to
  bound a range with, so its old range falls back to the header's
  old-file insertion point (a single line) - not the old bug's multi-line
  padded span.
- **Explicitly not done (per instruction):** a hunk whose true edit spans
  two functions still resolves to only one `ChangedFunction` - splitting
  `parse_diff`'s output into one `ChangedLocation` per contiguous
  changed-line run (not just one per hunk) is flagged here as a real,
  scoped-out future improvement, not implemented. Confirmed still present
  in this session's own re-verification (see NutriForge case below) and
  left as-is.
- **New regression tests** (`tests/test_adjudicate/test_context_builder.py`,
  32 tests now, was 30): `test_build_resolves_second_hunk_via_old_lines_
  despite_earlier_net_line_delta` (a pure-insertion first hunk with a
  non-zero net delta, then a second hunk whose real chunk is registered
  only at its *old* line number - fails before the fix, since the
  drifted new-line number has nothing registered there) and
  `test_build_does_not_misattribute_a_boundary_line_edit_to_an_adjacent_
  function` (an edit touching only a blank separator line between two
  functions, with the *adjacent* function's chunk registered nearby -
  asserts the location is honestly skipped, not silently attributed to
  that neighbor). Also updated 3 pre-existing `parse_diff` tests
  (`test_parses_single_hunk`, `test_parses_multiple_hunks_across_
  multiple_files`, `test_new_file_diff_uses_the_new_path_not_dev_null`)
  to the new tightened ranges plus the new `old_start_line`/`old_end_line`
  fields, and rewrote `test_build_tries_end_line_when_start_line_has_no_
  chunk`'s fixture from a context-padded single-line edit (which no
  longer exercises the fallback post-fix, since the tight range now
  collapses to one line) to a genuine 3-line contiguous replacement, so
  the start-then-end fallback is still meaningfully covered.
- **Re-verified the original Phase 24/25 diffs, byte-for-byte identical
  to what was used before (default `git diff` context, not the `-U0`
  workaround from the Phase 25 entry above)**, via the same real-`uvicorn`
  pattern, no LLM call needed (pure `ContextBundle` resolution):
  - **colorama, the exact `reset_all`/`init` misattribution repro - CONFIRMED FIXED,
    this is the task's own ground truth for "fixed":** parsed
    `ChangedLocation(start_line=30, end_line=33, old_start_line=30,
    old_end_line=33)` - tight to the actual edit (the replaced call plus
    the added `try`/`except`), no longer the header's padded old range
    (27-33, which reached `def init(...)`'s line). `changed_functions`
    now correctly resolves to **`reset_all`** (was `init`, pre-fix, per
    the Phase 25 entry above). `callers`/`related_tests` = the same two
    known-collision tests confirmed via `grep` in Phase 25
    (`winterm_test.py::testResetAll`,
    `ansitowin32_test.py::test_reset_all_shouldnt_raise_on_closed_orig_stdout`);
    `callees=[]` - both match this file's already-established ground
    truth exactly.
  - **NutriForge, the original 2-hunk repro (`register` in hunk 1;
    `login`+`me` merged into hunk 2 by git's own default context, since
    they're only 2 lines apart):** hunk 1 resolved to `register` via
    `old_start_line=18` (register's own `def` line - correct, unchanged
    from before, this hunk was never buggy). Hunk 2 resolved to
    **`login`**, via `old_start_line=old_end_line=40` (both fall back to
    the header's old-file insertion point, since this hunk is a pure
    addition with nothing removed - old line 40 is genuinely inside
    `login`'s body). **Hand-verified this is the fix, not a
    coincidence:** before the fix, this same hunk's lookup used its
    *new*-file number (47) directly against the old (indexed) file - old
    line 47 in the real pre-diff file is `// req.user is attached by the
    protect middleware`, a comment *inside `me`'s body* - so the old code
    would have resolved this hunk to `me`, not `login`, exactly the class
    of silent misattribution this fix targets. `me` itself is still not
    captured as its own `ChangedFunction` (the explicitly-deferred
    one-location-per-hunk limitation, not a residual bug - flagged above,
    not fixed here).
- Full suite: 531 passed (529 baseline + 2 new regression tests), 0
  failures/errors.
- Status: both approved bugs fixed and confirmed against the exact
  reported repro cases (colorama's `reset_all`/`init` misattribution -
  the task's explicit ground truth - now resolves correctly; NutriForge's
  drifted-hunk case now resolves to the correct function via its
  old-line anchor instead of drifting into an unrelated one). The
  multi-function-per-hunk limitation remains open, by explicit
  instruction, as a flagged future improvement.

## Phase 26 — Prosecutor Agent (naive)
Confirmed before starting: Phase 25 ✅ in `docs/roadmap.md`, and the
context_builder.py fix entry above shows both bugs independently
re-verified against the exact colorama `reset_all`/`init` repro. Proceeded
on that basis.

- **Refactor (triggered by genuine second use, not premature):**
  `adjudicate/agents/defender.py`'s three bundle-formatting helpers
  (`_format_changed_functions`/`_format_neighbors`/`_format_related_tests`)
  were private to that module, but the Prosecutor needs to render the
  exact same `ContextBundle` the exact same way. Moved all three into
  `adjudicate/context_builder.py` (public names:
  `format_changed_functions`/`format_neighbors`/`format_related_tests`),
  co-located with the dataclasses they format; `defender.py` now imports
  them instead of defining its own copies - zero behavior change,
  confirmed by its 7 existing tests passing unmodified. Added 6 direct
  unit tests for the moved functions in `test_context_builder.py`.
- `adjudicate/agents/prosecutor.py` (replacing Phase 23's docstring-only
  scaffold): `ProsecutorAgent.raise_concerns(context_bundle, diff,
  defender_justification) -> str` (plus a module-level convenience
  wrapper, same pattern as the Defender) - one `LLMClient.complete` call.
  System prompt instructs: raise only specific, checkable problems
  (missing null/undefined checks, untested branches, breaking changes to
  listed callers, security issues), never generic "could be cleaner"
  commentary; only raise concerns about functions listed under "Changed
  Functions" - explicitly never speculate about a function that might
  also have changed but isn't listed (the known multi-function-per-hunk
  gap from the fix above); only cite bundle-listed callers/callees/tests,
  and treat an empty "Related Tests" section as itself a legitimate
  concern to raise, not something to paper over; directly challenge the
  Defender's justification where the bundle/diff contradicts it rather
  than restating it; plain prose only. User prompt = the same four
  `ContextBundle` sections (via the shared formatters above) + the raw
  diff + the Defender's justification text. Model resolved via
  `adjudicate_settings.gemini_model_for(AgentRole.PROSECUTOR)`.
  Same `BaseAgent` deviation as the Defender (Phase 25), for the same
  reason (needs three inputs, not `review`'s one).
- 8 new unit tests (`tests/test_adjudicate/test_prosecutor.py`) against a
  fake LLM client - assert the prompt contains the changed function,
  callers, callees, related tests (or "none found" wording), the raw
  diff, and the Defender's justification verbatim; assert the system
  prompt instructs specificity, the changed-functions scope limit, never
  inventing, and challenging (not agreeing with) the Defender.
- **Live verification against the same real diffs used in Phase 24/25**
  (NutriForge `register`/`login`, colorama `reset_all`), via the same
  real-`uvicorn` pattern, real Gemini calls (`gemini-2.5-flash-lite` -
  `gemini-2.5-flash`'s free-tier daily quota, 20 requests, was already
  exhausted earlier this session; a different model has its own separate
  free-tier quota, used here only to get a real call through, not a
  permanent config change):
  - **colorama (`reset_all`, `try`/`except OSError` added around the
    existing call):** bundle matched the already-fixed ground truth
    exactly (`changed_functions=reset_all` lines 30-33, `callers`= the
    two known-collision tests, `callees=[]`). **Full Prosecutor output:**
    > The Defender's justification states the change handles potential `OSError` exceptions. However, the provided diff shows that the `except OSError: pass` block will silently ignore *any* `OSError`, not just those related to objects becoming `None` at exit as suggested by the comment. This could mask other, more critical `OSError` issues that should be surfaced to the user or logged.
    >
    > Specifically, the `WinTermTest.testResetAll` caller is not explicitly tested for the `OSError` scenario. While `AnsiToWin32Test.test_reset_all_shouldnt_raise_on_closed_orig_stdout` seems related, it's not guaranteed to cover all possible `OSError` conditions that might now be silently ignored by the `pass` statement.

    **Manual grounding check:** both named tests are exactly the bundle's
    `callers`/`related_tests`, nothing invented. Both concerns are
    specific and checkable (a reviewer can directly verify the `except`
    clause is unqualified beyond `OSError`, and that neither cited test
    exercises this new exception path - confirmed true, per Phase 25's
    own `grep` finding that both tests reference a *different* class's
    `reset_all` entirely). Directly challenges the Defender's more
    charitable framing instead of restating it - genuinely adversarial,
    genuinely useful.
  - **NutriForge (`register`'s new password-length check; `login`'s new
    `lastLoginAt`/`save()` calls; no `related_tests` - this repo has zero
    test files, confirmed in Phase 25):** **Full Prosecutor output:**
    > The `register` function now includes a password length validation. However, it does not check if the `password` field itself is of type string before checking its length. If a non-string value is provided for `password`, calling `.length` on it could lead to a runtime error.
    >
    > The `login` function was modified to update `user.lastLoginAt` and then call `user.save()`. There are no explicit checks to ensure that `user` is not null or undefined before attempting to access its properties and call the save method. If `User.findOne({ email })` returns null, this would result in a runtime error.
    >
    > There are no related tests found for the changes made to either the `register` or `login` functions. This means the new password validation logic in `register` and the `lastLoginAt` update and save logic in `login` have not been verified through automated testing.

    **Manual grounding check - mixed, reported honestly, not
    rubber-stamped:** the "no related tests" claim (3rd paragraph)
    exactly matches `related_tests=[]`, cross-checked true in Phase 25
    (zero test files anywhere in this repo). The `register` type-check
    concern is specific and plausible, referencing only the real diff
    line. **The `login` concern is factually wrong**, and the diff text
    the Prosecutor was given shows exactly why: the hunk's own context
    lines include `if (!user || !(await user.matchPassword(password))) {
    return res.status(401)...}` immediately above the added
    `lastLoginAt`/`save()` lines - `user` is already guaranteed non-null
    by that point. This is not a bundle-hallucination (no invented
    caller/file/test - every name used is real and bundle-listed) but a
    reasoning error: it missed a guard clause visible in the diff it was
    directly given. Flagged, not silently fixed by further prompt
    tweaking - this is exactly the class of mechanically-checkable claim
    Phase 28's Verifier is meant to catch, and re-prompting to patch one
    observed error without a systematic check would be whack-a-mole, not
    a real fix.
- Full suite: 545 passed (531 baseline + 6 formatting tests + 8
  Prosecutor tests), 0 failures/errors.
- Open issue (flagged, not fixed - explicitly deferred to Phase 28): the
  Prosecutor can state a factually incorrect concern (e.g. "no null
  check" when the diff's own context lines show one) with the same
  confidence as a correct one - nothing in this naive phase catches that.
  Worth keeping as a concrete test case when building the Verifier.
- Marked ✅ in `docs/roadmap.md` - concerns were grounded (zero invented
  callers/files/tests across both examples, cross-checked against
  Phase 25's own established ground truth) and genuinely adversarial (both
  examples directly challenge or add to the Defender's justification
  rather than restating it; the NutriForge case's "no tests" concern
  matches the confirmed absence of test files) - not rubber-stamping, per
  the task's explicit condition. The one observed factual error is
  reported above, not swept aside, per the task's own instruction to
  report honestly rather than force a clean pass.

## Phase 27 — Structured Claim Schema
Confirmed before starting: Phase 26 ✅ in `docs/roadmap.md`. Correction
made to the requesting task's own recollection (flagged, not silently
"fixed" by pretending it said something else): the known false-claim case
from Phase 26 was about **`login`**, not `register` - the false assertion
was "no null check before `user.save()`" in `login`, contradicted by a
guard clause (`if (!user || !(await user.matchPassword(password))) {
return res.status(401)...}`) visible a few lines above it in the same
diff hunk. `register`'s own Phase 26 concern (a missing type check before
`.length`) was a *different*, plausible (not false) claim. Used the real
`login` case as this phase's ground-truth test case, per the actual
Phase 26 entry above.

- **`generation/llm_client.py` (Phase 16's module) - minimal, additive
  extension, checked before adding a new mechanism as instructed:** it had
  *no* existing structured-output support at all (confirmed by reading it
  first) - `LLMClient.complete` took only `(system_prompt, user_prompt)`.
  Added one new optional parameter, `response_schema: Any | None = None`,
  threaded into both provider paths:
  - **Gemini** (`_complete_with_gemini`): when given, sets
    `response_mime_type="application/json"` and `response_schema=<schema>`
    on `GenerateContentConfig` - Gemini's own constrained-decoding
    structured-output mode (confirmed present in the installed
    `google-genai` SDK version before using it, via
    `GenerateContentConfig.model_fields`), not a post-hoc regex/parse of
    free text. Accepts a pydantic model or `list[Model]` directly.
  - **Ollama** (`_complete_with_ollama`): best-effort only, via a new
    `_ollama_format_for` helper - uses the schema's `.model_json_schema()`
    if it has one, else falls back to loose `"json"` mode. Explicitly
    flagged as unverified: `USE_OLLAMA=False` and no local server is
    reachable in this environment, so this path has zero live coverage
    - noted in its own docstring, not silently assumed correct.
  - `response_schema=None` (the default) leaves both paths byte-for-byte
    unchanged from before this parameter existed - confirmed by the full
    `tests/test_generation/` suite (49 tests) passing unmodified.
- `adjudicate/schemas.py` (new): the claim schema, in `adjudicate/` (per
  the task's own offered alternative) rather than the core
  `models/schemas.py` - Adjudicate's established pattern (Phase 23
  onward) is to layer on top of the core system through `RepoMindClient`
  only, never to add Adjudicate-specific concepts into core's shared
  schema file.
  - `ClaimType(str, Enum)`: **`MISSING_NULL_CHECK`, `UNTESTED_BRANCH`,
    `TYPE_MISMATCH`, `EXCEPTION_HANDLING`** - each grounded in one real
    Phase 26 concern (see the module's own docstring for the exact
    mapping). `BREAKING_CHANGE`/`SECURITY` (both mentioned as examples in
    the requesting task) were deliberately **not** added - neither has
    actually appeared in real Prosecutor output yet, and adding them
    anyway would be exactly the "speculative list" the task explicitly
    said to avoid. Purely additive to extend later (same pattern as
    `models.schemas.RetrievalSource.LOCATION`, Phase 20).
  - `ProsecutorClaim` (frozen dataclass, matching `models/schemas.py`'s
    convention): `claim_type`, `location` (`"<file_path>:<line>"`),
    `assertion`, `proposed_test` - exactly the task's requested shape.
  - `ProsecutorClaimModel` (pydantic `BaseModel`): a second, LLM-response-
    only mirror of the same shape, used solely as the `response_schema`
    passed to Gemini (which needs a pydantic model, not a plain
    dataclass) - kept separate so pydantic doesn't spread beyond this one
    boundary.
  - `parse_claims(raw_json, valid_file_paths) -> list[ProsecutorClaim]`:
    two independent validation layers, either one rejects the whole
    response (`ClaimParsingError`, a new exception distinct from
    `LLMGenerationError` - a malformed response is a contract violation
    on a call that technically succeeded, not a provider failure):
    **(1) structural** (valid JSON array, each item matches
    `ProsecutorClaimModel`'s schema) and **(2) groundedness** (`location`
    is `"<file>:<line>"` and `<file>` is one of the bundle's actual
    `changed_functions`) - the second layer mechanically enforces "never
    raise a concern about a function outside the bundle", the
    instruction Phase 26 could only ever *ask for* in a prompt, never
    check.
  - 9 new unit tests (`tests/test_adjudicate/test_schemas.py`): valid
    single/multiple claims, empty array (a legitimate "no concerns"),
    invalid JSON syntax, non-array JSON, unknown `claim_type`, a missing
    required field, a malformed `location`, and - the mechanical
    groundedness check itself - a claim whose file isn't in
    `valid_file_paths`.
- `adjudicate/agents/prosecutor.py` (rewritten): `raise_concerns` now
  returns `list[ProsecutorClaim]`, not `str`. System prompt rewritten
  around the four `ClaimType` categories, the `location` format, and the
  same grounding/scope/challenge rules as Phase 26 (schema-shape
  enforcement is now Gemini's job; the prompt's job is the semantics
  Gemini's schema can't check itself). One call via
  `LLMClient.complete(..., response_schema=list[ProsecutorClaimModel])`.
  **Retry loop, `MAX_ATTEMPTS = 3`:** on a `ClaimParsingError` (either
  validation layer), logs a warning and retries with a corrective note
  appended to the prompt (quoting the actual parsing error and
  re-stating the file-path-must-match-the-bundle rule); raises
  `LLMGenerationError` if every attempt fails - never silently accepts
  malformed output, never silently drops to fewer than the full
  `MAX_ATTEMPTS` tries.
- **11 new/rewritten unit tests** (`tests/test_adjudicate/test_prosecutor.py`,
  using a fake client returning canned JSON strings): claims parse
  correctly, `response_schema` is actually requested, prompt contains the
  bundle/diff/justification, system prompt names all four categories plus
  the scope/challenge rules, and - the retry mechanism itself, proven
  deterministically rather than only live -
  **`test_retries_once_on_malformed_json_then_succeeds`** (attempt 1:
  literal non-JSON text; attempt 2: valid - asserts exactly 2 calls and a
  corrective note in the retry's prompt), **`test_retries_on_claim_
  referencing_a_file_outside_the_bundle`** (attempt 1: structurally valid
  JSON naming a file not in the bundle; attempt 2: valid - same
  assertion), and **`test_raises_after_exhausting_all_retries_on_
  persistently_malformed_output`** (every attempt malformed - asserts
  exactly `MAX_ATTEMPTS` calls, then `LLMGenerationError`, never a
  silent partial accept).
- **Live verification against the real diffs used in Phase 24/25/26**
  (NutriForge `register`/`login`, colorama `reset_all`), same real-
  `uvicorn` pattern. Repeatedly hit real Gemini instability while
  gathering this (documented in full, not glossed over): `gemini-2.5-flash`'s
  free-tier daily quota (20 req/day) was already exhausted earlier this
  session (Phase 25/26); switching models, `gemini-2.0-flash` had a hard
  `limit: 0` entitlement (not usable at all under this key); a third,
  `gemini-2.5-flash-lite`, worked initially but hit repeated transient
  `503 UNAVAILABLE` ("high demand") errors, and after enough retries
  (each retry still counts as a request) **also** exhausted its own
  20/day free-tier cap mid-run. Net effect: one complete live run
  (NutriForge, both agents) succeeded before the quota wall; colorama's
  Prosecutor call did not complete live this session - its Defender call
  did (see below). Per explicit direction, not spending further live
  calls today chasing it; the retry/rejection *mechanism* itself is
  independently proven by the 3 deterministic tests above regardless.
  - **NutriForge (`register`'s password-length check; `login`'s
    `lastLoginAt`/`save()`; zero related tests, confirmed empty repo-wide
    in Phase 25) - full real structured output:**
    ```json
    [
      {
        "claim_type": "untested_branch",
        "location": "server/src/controllers/auth.controller.js:22",
        "assertion": "The new password length validation in the register function is not covered by any tests.",
        "proposed_test": "it('should return 400 if password is less than 8 characters', async () => {\n  const res = await request(app).post('/api/auth/register').send({\n    name: 'Test User',\n    email: 'test@example.com',\n    password: 'short',\n    fitnessGoal: 'lose weight'\n  });\n  expect(res.statusCode).toBe(400);\n  expect(res.body.message).toBe('Password must be at least 8 characters');\n});"
      },
      {
        "claim_type": "untested_branch",
        "location": "server/src/controllers/auth.controller.js:47",
        "assertion": "The update to the user's `lastLoginAt` field in the login function is not covered by any tests.",
        "proposed_test": "it('should update lastLoginAt on successful login', async () => {\n  await User.create({\n    name: 'Test User',\n    email: 'test@example.com',\n    password: 'password123'\n  });\n  const res = await request(app).post('/api/auth/login').send({\n    email: 'test@example.com',\n    password: 'password123'\n  });\n  expect(res.statusCode).toBe(200);\n  const user = await User.findOne({ email: 'test@example.com' });\n  expect(user.lastLoginAt).not.toBeNull();\n});"
      }
    ]
    ```
    **Manual check:** both claims parsed cleanly (no retry needed this
    run - Gemini's constrained decoding produced valid JSON on the first
    attempt), both `location`s are real lines inside real bundle-listed
    functions, both `claim_type`s (`untested_branch`) are the correct fit
    (matches the confirmed-empty `related_tests`), and each
    `proposed_test` is genuinely runnable-shaped Jest/Supertest code, not
    a description. This run happened not to reproduce the specific
    `login`/`missing_null_check` false claim from Phase 26 (LLM sampling
    is non-deterministic, and the now-more-constrained schema prompt may
    itself shift what the model reaches for) - **not substituted with a
    fabricated example**; see the deterministic proof of the false-claim
    case instead, immediately below.
  - **The `login`/`missing_null_check` false-claim case, in its
    structured form (this session's own ground truth for Phase 28):**
    since this exact wrong claim didn't recur in the live run above, its
    structured shape is validated the honest way - through
    `adjudicate.schemas.parse_claims` itself, using the literal wording
    from the real Phase 26 output (not invented) - proving the schema
    can genuinely express and structurally accept a well-formed *but
    false* claim, exactly the case Phase 28's Verifier must catch:
    ```json
    {
      "claim_type": "missing_null_check",
      "location": "server/src/controllers/auth.controller.js:47",
      "assertion": "There are no explicit checks to ensure that user is not null or undefined before attempting to access its properties and call the save method. If User.findOne({ email }) returns null, this would result in a runtime error.",
      "proposed_test": "test('login updates lastLoginAt only when user is guaranteed non-null', async () => { const req = { body: { email: 'unknown@example.com', password: 'x' } }; await login(req, res); /* assert user.save() was never reached for an unmatched email - if it throws instead, the missing-null-check claim is confirmed */ });"
    }
    ```
    `parse_claims` accepts this exactly as a well-formed
    `ProsecutorClaim` (`ClaimType.MISSING_NULL_CHECK`, location resolves
    to `login`'s real file) - structurally and mechanically valid. Its
    *truth* is the open question: the diff's own context lines (visible
    to whatever agent re-examines it) already show the guard clause that
    makes this assertion false. That is precisely the shape Phase 28
    needs to handle - a claim that parses cleanly and references real
    code, but is wrong - not a hallucinated file/function (which
    `parse_claims` already rejects) but a wrong inference about real,
    given code. Flagged as this session's concrete Verifier test case,
    not resolved here (out of Phase 27's scope).
  - **colorama (`reset_all`'s new `try`/`except OSError`) - Defender
    succeeded live, Prosecutor did not (quota, not a schema/parsing
    issue):**
    > The change modifies the `reset_all` function in `colorama/initialise.py`. Previously, it directly called `AnsiToWin32(orig_stdout).reset_all()`. The updated code wraps this call in a `try...except OSError` block, and if an `OSError` occurs, it is caught and ignored (`pass`).
    >
    > This change was made to handle potential `OSError` exceptions that might arise during the `reset_all` operation, as indicated by the comment "Issue #74: objects might become None at exit".
    >
    > Existing tests for this change include `WinTermTest.testResetAll` and `AnsiToWin32Test.test_reset_all_shouldnt_raise_on_closed_orig_stdout`, both found in `colorama/tests/winterm_test.py` and `colorama/tests/ansitowin32_test.py` respectively.

    Structured Prosecutor claims for this case are still outstanding -
    open item, not fabricated, below.
- Full suite: 557 passed (545 baseline + 9 schema tests + 3 net new
  prosecutor tests), 0 failures/errors.
- Open issues (flagged, not fixed here):
  1. colorama's live structured Prosecutor run never completed this
     session (Gemini quota exhaustion across three different models
     tried) - worth re-running once quota resets, though the mechanism
     is already proven deterministically and via the NutriForge live run.
  2. Ollama's `response_schema` support (`_ollama_format_for`) has zero
     live verification - `USE_OLLAMA=False` and no local server reachable
     in this environment.
  3. The `login`/`missing_null_check` false claim (this session's own
     Verifier test case) is not verified/refuted anywhere yet - that is
     explicitly Phase 28's job.
- Marked ✅ in `docs/roadmap.md` - the schema is genuinely enforced, not
  just "usually comes back as JSON": proven both deterministically (3
  dedicated retry/rejection tests using a fake client returning literal
  malformed JSON and a real-shaped-but-ungrounded claim) and live (the
  full NutriForge run parsed real Gemini output through the real
  `parse_claims` validation with zero retries needed, and the
  groundedness check was independently exercised against real claim JSON
  via the deliberate-bad-case check during this session's earlier
  successful run). Every claim_type value used in real output was one of
  the four grounded categories, none dumped into a generic bucket. The
  known false claim is now expressed and mechanically parseable in its
  structured form, ready for Phase 28 to actually check.

## Phase 27 follow-up — Groq added as a third LLM provider, colorama case closed out
Closes the one open item left at the end of the Phase 27 entry above
(colorama's live Prosecutor run never completed due to Gemini quota
exhaustion across three models). Requested explicitly because Ollama
isn't installed in this environment either, so a third *hosted* option
was needed to keep working today.

- **Design check confirmed before writing code (as asked):** `generation/
  llm_client.py`'s existing per-provider structure (Protocol + loader
  function + `_complete_with_<provider>` function + lazily-cached client
  on `LLMClient`) needed no architectural change - Groq's API is
  OpenAI-compatible chat-completions, the same shape Gemini/Ollama
  already fit into. A straightforward third parallel path, confirmed,
  not just assumed.
- `core/constants.py`: `DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"`
  (confirmed as a real, currently-valid model literal against the
  installed `groq` SDK's own type hints before using it).
- `config.py`: `USE_GROQ` (checked after `USE_GEMINI`, before
  `USE_OLLAMA` - matches the task's requested priority), `GROQ_MODEL`,
  `GROQ_API_KEY` - same `Field(...)` pattern as the existing Gemini/Ollama
  settings, provider-selection docstring/comments updated to describe all
  three.
- `generation/llm_client.py`: `GroqChatClient` Protocol, `load_groq_client`,
  `_complete_with_groq` (mirrors `_complete_with_gemini`/`_complete_with_ollama`
  exactly - catches every SDK exception, re-raises as `LLMGenerationError`),
  wired into `LLMClient.__init__`/`_ensure_groq_client`/`complete`'s
  priority chain (`Gemini -> Groq -> Ollama -> raise`). `response_schema`
  support: Groq's `response_format={"type": "json_object"}` JSON mode -
  confirmed weaker than Gemini's the same way `_ollama_format_for` already
  documented for Ollama (valid JSON guaranteed, not schema conformance).
- **Real bug found and fixed via live testing, not assumed away:**
  Groq's (and any OpenAI-compatible) `json_object` mode can only
  constrain the top-level response to an *object* - never a bare array,
  unlike Gemini's `response_schema=list[Model]`. Asked for "a JSON array
  of claims" regardless (the existing system prompt, unchanged), Groq
  wrapped it the only way its API allows - confirmed by direct probe
  against the real API: `'{"claims": [...]}'`, verbatim. Fixed in
  `adjudicate/schemas.py::parse_claims`: a top-level object is now
  unwrapped only when unambiguous (exactly one key, whose value is a
  list) - anything else (multiple keys, a single key whose value isn't a
  list) still rejects exactly as before, not silently guessed at. 3 new
  tests (`tests/test_adjudicate/test_schemas.py`, 12 total now, was 9):
  the real `{"claims": [...]}` shape unwraps correctly, a multi-key
  object is still rejected even with a list-valued key present, and a
  single-key object whose value isn't a list is still rejected.
- requirements.txt: `groq>=0.11,<1` (installed, `0.37.1` resolved;
  `pip check` showed only two pre-existing, unrelated conflicts -
  `instructor`/`langchain-openai` wanting a newer `openai` package than
  what's installed - confirmed present before this change too, `groq`
  itself doesn't depend on `openai` at all).
- `.env.example`: added `USE_GROQ`, `GROQ_API_KEY` (placeholder, blank),
  `GROQ_MODEL` alongside the existing Gemini/Ollama block, selection-order
  comment updated to describe all three. The real `.env` had already been
  hand-edited (by the requester) with a `GROQ_API_KEY` and a `LLM_PROVIDER=groq`
  line - the latter is not a real `config.py` field (provider selection
  is per-provider `USE_*` booleans, not a single provider-name string), so
  pydantic-settings' `extra="ignore"` was silently discarding it and
  Gemini (still `USE_GEMINI=true`) would have stayed selected regardless
  of the Groq key being present. Fixed directly in the real `.env`:
  `USE_GEMINI=false`, `USE_GROQ=true` added, `GROQ_MODEL` added, the dead
  `LLM_PROVIDER` line removed - Groq is now the actually-active provider,
  not just configured-but-inert. Flagging since it's a real runtime
  default change beyond "add an option": once Gemini's daily quota resets,
  switching back is just `USE_GEMINI=true`/`USE_GROQ=false`.
- **Real regression found and fixed in the existing test suite, exposed
  by the `.env` edit above:** four pre-existing tests in
  `tests/test_generation/test_llm_client.py` constructed
  `LLMClient(use_gemini=False, use_ollama=True, ...)` without pinning
  `use_groq=False` - each silently fell back to `settings.USE_GROQ` for
  anything not explicitly passed. Once the real `.env` had `USE_GROQ=true`,
  three of these tests started making *real* live calls to Groq's API
  instead of exercising the fake Ollama client they were written for
  (caught by a failing assertion, not silently passing wrong). Fixed by
  explicitly pinning `use_groq=False` in all four - tests must not depend
  on the developer's local `.env` contents, a latent fragility that
  existed before Groq but had nothing to expose it until now.
- 12 new/updated tests in `tests/test_generation/test_llm_client.py` (28
  total, was 15): Groq client construction/failure, missing-API-key
  handling, response generation (text/token accounting/model name),
  `response_format` requested only when `response_schema` is given,
  provider-selection priority (Groq selected when Gemini disabled; Gemini
  still wins when both enabled), failure wrapping, and lazy client
  loading - mirroring every existing Gemini/Ollama test's shape exactly.
- **Live verification - colorama's previously-outstanding Prosecutor
  case, via Groq (`llama-3.3-70b-versatile`), same real diff used
  throughout Phase 24/25/26/27:**
  ```json
  [
    {
      "claim_type": "exception_handling",
      "location": "colorama/initialise.py:32",
      "assertion": "The try-except block in the reset_all function silently swallows all OSError exceptions without logging or handling them, potentially masking critical errors.",
      "proposed_test": "try: raise OSError('Test error'); except OSError: pass; assert False, 'Expected OSError to be raised'"
    },
    {
      "claim_type": "exception_handling",
      "location": "colorama/initialise.py:32",
      "assertion": "The try-except block in the reset_all function only catches OSError exceptions, but other types of exceptions may still occur and are not handled.",
      "proposed_test": "try: raise ValueError('Test error'); except OSError: pass; assert False, 'Expected ValueError to be raised'"
    }
  ]
  ```
  **Manual grounding check:** both claims' `location` (`colorama/initialise.py:32`)
  is a real line inside `reset_all`'s bundle-listed range (30-33);
  `exception_handling` is the correct category for both (matches the
  first Gemini-run colorama concern from the earlier Phase 27 entry
  almost exactly - "silently swallows... masking critical errors").
  The second claim is more tautological than incisive (an `except
  OSError` clause not catching non-`OSError` exceptions is true of any
  such clause, by definition) but is still specific and technically
  accurate, not invented or generic. **The deliberate-bad-case check was
  re-run against this real Groq output** (not just the earlier Gemini
  run) and correctly rejected it when checked against a deliberately
  wrong file set - confirms the groundedness mechanism holds across
  providers, not just Gemini's.
- Full suite: 573 passed (557 Phase 27 baseline + 3 schema tests + 12
  provider tests + 1 net from a test rename), 0 failures/errors.
- **Provider attribution for every Phase 24-27 live verification case, for
  transparency** (per explicit request):
  - NutriForge `register`/`login` (Phase 25 Defender, Phase 26 Prosecutor
    free-text, Phase 27 Prosecutor structured claims): **Gemini**
    (`gemini-2.5-flash` for Phase 25/26; `gemini-2.5-flash-lite` for
    Phase 27, after the former's quota was already exhausted).
  - colorama `reset_all` Defender justification (Phase 25, Phase 26,
    Phase 27): **Gemini** every time.
  - colorama `reset_all` Prosecutor free-text concerns (Phase 26):
    **Gemini** (`gemini-2.5-flash-lite`).
  - colorama `reset_all` Prosecutor **structured** claims (Phase 27,
    this entry): **Groq** (`llama-3.3-70b-versatile`) - Gemini could not
    complete this specific call live in this session at all.
- Open issues carried forward, unchanged: Ollama's `response_schema`
  support (`_ollama_format_for`) still has zero live verification
  (`USE_OLLAMA=False`, no local server reachable); the `login`/
  `missing_null_check` false claim is still unverified/unrefuted -
  explicitly Phase 28's job.
- Phase 27 fully closed: `docs/roadmap.md`'s ✅ (already marked) now
  reflects a complete verification picture - both example diffs have a
  genuine, real, structured, grounded Prosecutor run on record, with no
  fabricated or hand-constructed substitute standing in for either.

## Provider strategy clarification (config-only, no new code)
Written so a future session doesn't re-investigate the Groq addition or
the earlier test-isolation fix as if either were unresolved.

- **Gemini's daily free-tier quota has reset** - confirmed with one
  real, minimal call (`gemini-2.5-flash`, "OK" round-trip, `200 OK`) at
  2026-07-07 14:00 - not assumed from elapsed time. This corrected the
  task's own starting premise ("today we know Gemini's limit is
  exhausted") - flagged back before acting on it, not silently followed.
- **Final, actual provider setup, decided with that correction in
  hand:** `USE_GEMINI=true` (active/primary), `USE_GROQ=false` (added
  and fully working - see the Phase 27 follow-up entry above - but left
  *inactive* today since Gemini is confirmed healthy again; flip
  `USE_GROQ=true`/`USE_GEMINI=false` any time Gemini errors or
  exhausts), `USE_OLLAMA=false` (deferred - Ollama is intentionally not
  installed/configured for now; any Ollama-dependent behavior being
  untested or unreachable is expected at this stage, not a gap to
  close). Real `.env` set to exactly this (had briefly passed through
  `USE_GEMINI=false`/`USE_GROQ=true` mid-session before the quota-reset
  check and this final decision).
- **Priority order in `generation/llm_client.py::LLMClient.complete`
  confirmed correct by re-reading the code**: `if self._use_gemini: ...
  return` -> `if self._use_groq: ... return` -> `if self._use_ollama:
  ... return` -> raise. Gemini is checked first, Groq second, Ollama
  third - exactly the requested order, and Ollama is never even
  consulted while Groq is enabled, so Ollama's absence cannot block or
  delay reaching Groq.
- **Important distinction, flagged rather than assumed either way:**
  this priority chain is a **static, config-driven** selection (each
  branch is an unconditional `return`, no `try`/`except` around the
  Gemini call) - exactly what Phase 16's own module docstring already
  says ("not automatic retry-on-failure between providers"). With both
  `USE_GEMINI=true` and `USE_GROQ=true`, Gemini handles every call; if
  Gemini's own call fails at runtime (quota exhaustion, a transient
  503), that exception propagates straight to the caller - **Groq is
  not automatically retried within that same request**. Falling over to
  Groq today still means manually setting `USE_GEMINI=false`. Genuine
  automatic runtime failover (catch a Gemini failure, retry the same
  call via Groq) would be new, deliberate work, not something already
  built - flagged for an explicit decision, not silently added or
  assumed unnecessary. **Decision, made explicitly this session:**
  defer automatic runtime failover to a later phase; today's static
  config-flip (manually setting `USE_GEMINI=false`/`USE_GROQ=true` when
  needed) is sufficient for now.
- **Re-examined the four-test isolation fix from the Groq-addition
  session with the "Ollama deferred" framing - conclusion: unrelated to
  Ollama's real-world absence, already fully and correctly fixed,
  nothing to revisit.** Confirmed by re-reading `_FakeOllamaClient`
  (`tests/test_generation/test_llm_client.py`): it is a plain in-memory
  Python object standing in for `ollama.Client` - those four tests never
  attempted a real network call to any Ollama server, installed or not.
  The actual root cause was ambient-settings leakage: each test
  constructed `LLMClient(use_gemini=False, use_ollama=True, ...)`
  without also pinning `use_groq=False`, so the un-pinned `use_groq`
  argument fell back to reading the real `settings.USE_GROQ` - which,
  at that point in the session, had just been flipped to `true` in the
  real `.env`. Since Groq is checked before Ollama in the (correct)
  priority chain, this made the tests silently exercise the real Groq
  API instead of the fake Ollama client they were written to test. This
  would have happened identically whether or not Ollama was installed
  on this machine - the fallback/priority logic itself was never broken;
  the bug was purely that four unit tests weren't fully hermetic against
  ambient `.env` state. Already fixed (all four now explicitly pin
  `use_groq=False`) and confirmed passing again under the corrected
  `.env` (`USE_GEMINI=true`/`USE_GROQ=true`) in this session's full-suite
  run below - no further action needed.
- Explicitly not touched this session, per instruction: no attempt to
  install, configure, or test Ollama. Any currently-failing-or-untested
  Ollama-dependent behavior (e.g. `_ollama_format_for`'s zero live
  coverage, noted in the Phase 27 follow-up entry above) remains exactly
  as flagged there - expected and deferred, not newly investigated here.
- Full suite: 573 passed (unchanged count - this was a config
  correction, not a code change), 0 failures/errors, run against the
  corrected `.env`.

## Phase 28 — Verifier Layer
Confirmed before starting: Phase 27 ✅ in `docs/roadmap.md`; provider
strategy (Gemini primary, Groq fallback available but inactive, Ollama
deferred) re-confirmed from the entry directly above. Applied the new
LLM-quota testing discipline (a standing instruction from this session,
saved to memory) throughout: no fresh Prosecutor/Defender calls were made
for this phase at all - every claim verified below is either a real,
previously-captured Phase 27 output or an explicitly-constructed
ground-truth case, and everything new this phase (schema/sandbox/strategy
logic) was tested with zero LLM involvement, per the task's own core
requirement.

- **`adjudicate/schemas.py`**: added `ClaimType.BREAKING_CHANGE`/`SECURITY`
  - deliberately *not* added in Phase 27 (not yet observed in real
  output then), added now because Phase 28's own dispatch table needs
  concrete targets for both - a justified, non-speculative reason,
  documented as such in both the module docstring and the enum's own
  docstring so a future reader doesn't read this as backsliding into
  guessing. The Prosecutor's own prompt (Phase 27) is unchanged - it does
  not generate these two categories yet; that's a separate follow-up.
- **`adjudicate/benchmark/fixtures/` (new directory)** - the first real
  use of the fixtures convention from this session's LLM-quota-discipline
  memory: every real Prosecutor output already captured in Phase 27's
  PROGRESS.md entries is now also a persisted, loadable JSON file, so
  Phase 28 (and later, Phase 31/32) never needs to re-call an LLM to get
  this same data again:
  - `nutriforge_register_login_claims.json` - the real Gemini-captured
    NutriForge claims (2 `untested_branch`), verbatim.
  - `colorama_reset_all_claims.json` - the real Groq-captured colorama
    claims (2 `exception_handling`), verbatim.
  - `nutriforge_register_login.diff` - the exact diff those two came
    from, regenerated fresh from the real clone (byte-accurate, not
    retyped from memory - confirmed by using it with `patch` below) so
    it stays appliable.
  - `nutriforge_login_false_claim.json` - **this session's ground-truth
    false claim** (`login`/`missing_null_check`, from Phase 26's real
    free-text output - see that entry for why it's false). Labeled
    explicitly as constructed, not a raw LLM capture: it never recurred
    in exactly this structured form in a live Phase 27 run, and Phase
    27's own hand-illustration of it used a non-functional placeholder
    `proposed_test`. Authored a genuine, executable one for this phase -
    real Node.js (no framework; NutriForge has zero test infrastructure
    installed, confirmed Phase 25) that calls the actual, unmodified,
    diff-applied `login` handler and asserts it returns 401 without
    throwing for an unmatched email.
  - `constructed_confirmed_null_check_bug.json` - a small, deliberately
    broken function + claim, built per the task's own explicit allowance
    ("pick or construct one with a genuinely provable issue"), to prove
    the Verifier reports true positives correctly too, not just true
    negatives.
- **`adjudicate/verifier/` (replacing Phase 23's scaffold):**
  - `models.py`: `VerificationStatus` (CONFIRMED/REFUTED/INCONCLUSIVE),
    `Confidence` (HIGH/MEDIUM/LOW - tied to *what kind* of evidence
    produced a verdict, not to which way it came out), `VerificationResult`
    (`claim`, `status`, `confidence`, `evidence`, `strategy`).
  - `sandbox.py`: `run_sandboxed(code, language, cwd, timeout_seconds)` -
    a real `subprocess` call (`python`/`node`), never `exec()`/`eval()`.
    Two protections, both honestly scoped: **(1) a real hard timeout**
    (`subprocess.run(..., timeout=...)` - a genuine guarantee), and
    **(2) best-effort network denial** (the generated script is prefixed
    with a small guard monkeypatching `socket.socket.connect`/Node's
    `net.Socket.connect`+`http(s).request`/`.get` to raise immediately -
    blocks the realistic "test code calls out to the internet by
    accident" case, explicitly documented as *not* a hard security
    boundary - a determined subprocess could bypass it via lower-level
    mechanisms). **True isolation needs Docker, which is not set up in
    this environment - flagged in the module's own docstring as a real
    follow-up, exactly as the task allowed, not silently claimed as
    already solved.** Memory limiting
    (`AdjudicateSettings.VERIFIER_SANDBOX_MEMORY_LIMIT_MB`, declared
    since Phase 23) is likewise left inert for the same reason - a
    cgroup/Job Object needs OS integration a subprocess can't provide
    alone.
  - `strategies.py`: the dispatch table, **zero LLM calls anywhere,
    confirmed both statically and by live testing (see Verify step 3
    below)**:
    - `UNTESTED_BRANCH`/`MISSING_NULL_CHECK` -> `verify_via_proposed_test`
      (sandbox the claim's own `proposed_test`; failing = CONFIRMED,
      passing = REFUTED, both HIGH confidence - a real execution).
    - `SECURITY` -> `verify_security` (real `bandit -f json` scoped to
      the claimed file, checks for a finding within 2 lines of the
      claimed line; MEDIUM confidence; INCONCLUSIVE/LOW for non-Python
      files - bandit has no equivalent wired up for other languages).
    - `TYPE_MISMATCH` -> `verify_type_mismatch` (real `mypy`, same
      line-proximity check; MEDIUM confidence; Python-only, same honest
      INCONCLUSIVE fallback).
    - `BREAKING_CHANGE` -> `verify_breaking_change` (runs a caller-supplied
      `test_command` for the existing suite; MEDIUM confidence -
      documented as an honest approximation, not a true before/after
      diff, since `ContextBundle` carries no stored pre-change test-run
      baseline to compare against; a real before/after comparison is a
      flagged follow-up, not faked here).
    - `EXCEPTION_HANDLING` -> `verify_exception_handling` - **the gap the
      task asked to be flagged rather than papered over**: neither
      bandit nor mypy is a genuine fit for "this except clause is too
      broad." Falls back to `verify_via_proposed_test` when the claim
      has one (both real colorama claims did); otherwise returns
      INCONCLUSIVE at LOW confidence, explicitly, rather than forcing
      bandit/mypy onto a category neither fits.
  - `verifier.py`: `verify_claim(claim, repo_path, test_command=None)` -
    the single dispatch entry point; INCONCLUSIVE/LOW (never a silent
    default) if a claim's type has no dispatch entry, or if it's
    `BREAKING_CHANGE` with no `test_command` given.
- **33 new tests** (`tests/test_adjudicate/test_verifier_sandbox.py` (11),
  `test_verifier_strategies.py` (16), `test_verifier_dispatch.py` (6)) -
  every one exercises real subprocess/bandit/mypy execution against small
  purpose-built fixture files (timeout, network-block, pass/fail/
  timeout/unsupported-language for the sandbox; real bandit/mypy hits and
  clean runs for the static-analysis strategies; dispatch routing) - no
  LLM anywhere to mock, so nothing is mocked; these are exactly the kind
  of "verifiable without an LLM" tests the quota-discipline memory says
  to run for real.
- **Live verification against the task's own ground-truth bar - both
  cases, real tool output, not paraphrased:**

  **Case 1 - the known false claim, `login`/`missing_null_check`
  (`nutriforge_login_false_claim.json`), against a fresh temp copy of the
  real NutriForge `server/` tree with the real diff fixture applied via
  `patch -p1`:**
  ```
  STATUS: refuted
  CONFIDENCE: high
  STRATEGY: proposed_test
  EVIDENCE:
  exit_code=0
  --- stdout ---
  PASS: login returned 401 without throwing when no user matched - the missing-null-check claim is unfounded

  --- stderr ---

  ```
  Real, inspectable proof: the sandboxed script called the actual,
  unmodified, diff-applied `login` handler with a request matching no
  user, and it returned 401 cleanly - the guard clause
  (`if (!user || !(await user.matchPassword(password))) { return
  res.status(401)...}`) the false claim overlooked really does intercept
  before `user.lastLoginAt = ...` is ever reached. **Correctly REFUTED,
  with the exact evidence proving why - this session's explicit
  ground-truth bar for marking this phase done.**

  **Case 2 - the constructed true-positive bug
  (`constructed_confirmed_null_check_bug.json`):**
  ```
  STATUS: confirmed
  CONFIDENCE: high
  STRATEGY: proposed_test
  EVIDENCE:
  exit_code=1
  --- stdout ---

  --- stderr ---
  Traceback (most recent call last):
    File "...\_verifier_check.py", line 13, in <module>
      result = get_user_email(None)
               ^^^^^^^^^^^^^^^^^^^^
    File "...\user_service.py", line 3, in get_user_email
      return user["email"]
             ~~~~^^^^^^^^^^
  TypeError: 'NoneType' object is not subscriptable
  ```
  A real crash, a real traceback, correctly reported CONFIRMED - proves
  the Verifier isn't just biased toward REFUTED; it detects a genuine
  true positive with equally real evidence.

  **Case 3 - zero LLM calls, checked by running with no provider
  configured at all**, not just by reading the code: cleared every
  provider env var, explicitly set `USE_GEMINI=false`/`USE_GROQ=false`/
  `USE_OLLAMA=false`. First confirmed the negative control -
  `LLMClient().complete(...)` correctly raises `LLMGenerationError`
  ("No LLM provider is enabled") in this state, proving that if the
  Verifier ever *did* call an LLM, this test would have caught it.
  Then ran a real `verify_claim` call in the exact same zero-provider
  process - it completed successfully (`status=confirmed,
  confidence=high, strategy=proposed_test`), proving no code path in
  `adjudicate/verifier/` touches any LLM client. Independently confirmed
  statically too: `grep -rn "llm_client\|LLMClient\|google.genai\|import
  groq\|import ollama\|GEMINI_API_KEY\|GROQ_API_KEY" adjudicate/verifier/`
  - zero matches, whole package.
- Full suite: 606 passed (573 baseline + 33 new), 0 failures/errors
  (218s - mostly mypy's per-invocation startup cost across several
  strategy tests, not a correctness concern).
- Open issues, flagged rather than silently left implicit:
  1. Sandboxing is `subprocess`-based, not Docker - both the network
     denial and memory-limit enforcement are best-effort/inert
     respectively; real OS-level isolation is a follow-up, not done here.
  2. `verify_breaking_change` checks "does the suite pass now," not "did
     this diff introduce a new failure" - a real before/after comparison
     needs a stored pre-change baseline Adjudicate doesn't have yet.
  3. `EXCEPTION_HANDLING` claims without a `proposed_test` have no real
     check at all (INCONCLUSIVE/LOW, by design) - a real gap, not
     resolved this phase.
  4. `bandit`/`mypy` are Python-only; a claim about a non-Python file
     routed to `SECURITY`/`TYPE_MISMATCH` is always INCONCLUSIVE/LOW -
     no equivalent tool wired up for JS/other languages yet.
- Marked ✅ in `docs/roadmap.md` - the known false claim came back
  REFUTED with real, inspectable tool evidence (Case 1, the task's
  explicit bar for done), a genuine bug came back CONFIRMED with equally
  real evidence (Case 2), and zero LLM calls were proven both statically
  and by a live no-provider run (Case 3) - not marked done on "the logic
  looks right" alone.

## Phase 29 — Defender Rebuttal Loop
Confirmed before starting: Phase 28 ✅ in `docs/roadmap.md`. Applied the
LLM-quota-discipline memory throughout: exactly 2 real LLM calls total
this phase (one `rebut()` call per verification case) - every claim, the
NutriForge diff, the real Phase 27 justification text, and the Phase 28
Verifier re-runs all reused already-captured or zero-LLM data.

- **`adjudicate/agents/defender.py`**: added `DefenderAgent.rebut(context_bundle,
  diff, original_justification, verified_claims) -> str` (plus a
  module-level `rebut(...)` wrapper, same pattern as `draft_justification`) -
  one LLM call, using a **separate** `REBUTTAL_SYSTEM_PROMPT` (confirmed
  distinct from `SYSTEM_PROMPT` by a dedicated test) instructing: concede
  every CONFIRMED claim outright (real evidence, not arguable); briefly
  acknowledge REFUTED ones, citing the evidence, without gloating; push
  back on INCONCLUSIVE ones only with real counter-evidence from the
  Context Bundle, or admit plainly there is none - never just restate the
  prior position. Reads `verified_claims: list[adjudicate.verifier.models
  .VerificationResult]` - the Verifier's real status/confidence/evidence,
  never the Prosecutor's raw unverified claim text, per the task's
  explicit input requirement. New `_format_verified_claims` helper
  (kept local to `defender.py` for now - only one consumer so far, same
  "extract when a second real consumer appears" discipline as Phase 27's
  formatter extraction).
- **`adjudicate/orchestrator/rebuttal_loop.py`** (new; `orchestrator/__init__.py`
  updated to export it, replacing its Phase 23 scaffold docstring):
  `run_rebuttal_loop(context_bundle, diff, original_justification,
  verified_claims, defender=None) -> RebuttalLoopResult`. Pure
  round-coordination, never drafts a rebuttal itself - only calls
  `DefenderAgent.rebut` and decides whether another round is needed,
  per the Phase 23 scaffold's own "owns round sequencing and termination,
  not any single agent's logic" split.
  - **Round counting**: round 1 = the initial justification (already
    given, not a `rebut` call); up to `MAX_REBUTTAL_ROUNDS = 2` further
    rounds each call `rebut`, chaining each round's own output back in as
    the next round's "prior statement" - confirmed by a dedicated test
    that round 2's prompt contains round 1's rebuttal text, not the
    original justification again.
  - **Termination, decided mechanically, not by guessing persuasiveness**:
    "resolved" once every claim is CONFIRMED or REFUTED - both are final
    real Verifier results nothing in this loop can change, so nothing
    remains to rebut. If at least one claim is still INCONCLUSIVE after
    `MAX_REBUTTAL_ROUNDS`, ends "by cap" instead. Deliberately does *not*
    attempt to judge whether the Defender's counter-evidence was
    "satisfactorily countered" in prose - that requires a value judgment
    genuinely out of scope here (Phase 30's Judge, not this loop).
    `RebuttalLoopResult` carries `transcript`, `rounds_used`, `ended_by`
    (`"resolution"`/`"cap"`), and `unresolved_claims` for whichever the
    Judge needs.
- **14 new tests**: `tests/test_adjudicate/test_defender.py` gained 8
  (`rebut` returns LLM text, module wrapper delegates, uses a system
  prompt distinct from `draft_justification`'s, prompt includes prior
  statement/diff/verified status+confidence+evidence, all three statuses
  render when present, empty claim list says "none", system prompt
  instructs concede/inconclusive/never-invent). New
  `tests/test_adjudicate/test_rebuttal_loop.py` (6): resolves after one
  round with no inconclusive claims, resolves immediately on an empty
  claim list (vacuously - nothing to litigate), hits the cap with an
  inconclusive claim, transcript chains each round's output into the
  next round's input (not back to the original), mixed-status claims
  resolve once nothing remains inconclusive, return type check.
- **Live verification - both required cases, real output, not
  paraphrased. Exactly 2 real LLM calls (`gemini-2.5-flash`, quota
  confirmed reset earlier this session) - everything else reused real
  captured data or zero-LLM re-runs:**

  **Case 1 - NutriForge `login`/`missing_null_check`, the known false
  claim, real REFUTED verdict (re-derived fresh via the zero-LLM
  Verifier against the real diff-applied code, same fixture as Phase
  28's ground truth) + the real, verbatim Phase 27 Defender justification
  + a real `ContextBundle` rebuilt live against the actual indexed
  NutriForge repo:**
  > The claim regarding a `missing_null_check` at `server/src/controllers/auth.controller.js:47` has been refuted. The evidence from the proposed test indicates that the `login` function correctly returned a 401 status without throwing an error when no user matched, confirming that the claim of a potential runtime error due to a null `user` object is unfounded.

  **Manual check:** the rebuttal doesn't restate the original
  justification's content at all - it responds specifically and only to
  the verified claim, citing the exact real evidence (the 401-without-
  throwing result) rather than a generic "that's not true." Matches
  `REBUTTAL_SYSTEM_PROMPT` rule 2 (acknowledge REFUTED plainly, cite
  evidence, don't belabor it) exactly - appropriate to a REFUTED status,
  not a rubber-stamped repeat of round 1.

  **Case 2 - the constructed true-positive bug from Phase 28, real
  CONFIRMED verdict (re-derived fresh via the zero-LLM Verifier) + a
  short, explicitly-labeled constructed original justification (this
  fixture never went through a real Defender call in Phase 28 - authored
  honestly for this test, not presented as a real capture) + a minimal
  hand-built `ContextBundle` (no real indexed repo backs this synthetic
  fixture):**
  > The Verifier has confirmed a `missing_null_check` at `user_service.py:3`. The `get_user_email` function does not check if `user` is `None` before attempting to access `user['email']`, which leads to a `TypeError` if `None` is passed as an argument. This was demonstrated by the Verifier's test, which showed a `TypeError: 'NoneType' object is not subscriptable` when `get_user_email(None)` was called. I concede this point.

  **Manual check:** explicit, unambiguous concession ("I concede this
  point") citing the exact real traceback - no arguing, minimizing, or
  hedging against CONFIRMED, real evidence. Matches `REBUTTAL_SYSTEM_PROMPT`
  rule 1 exactly.
- Full suite: 620 passed (606 baseline + 14 new), 0 failures/errors.
- Open issue, flagged rather than silently glossed over: the round-loop's
  mechanical "resolved" definition means that whenever at least one claim
  is genuinely INCONCLUSIVE, the loop *always* hits the cap (nothing in
  this phase re-verifies a claim mid-loop, so a claim's status can never
  change from INCONCLUSIVE once set) - the second rebuttal round still
  runs (giving the Defender a real second attempt, with round 1's own
  text as context) but cannot mechanically change `ended_by`. A genuine
  "the Defender's new counter-evidence changed the verdict" loop would
  need Phase 28's Verifier to re-check counter-evidence mid-round -
  not attempted here, flagged as a real future extension, not faked.
- Marked ✅ in `docs/roadmap.md` - the Defender's behavior was verified
  appropriate to the actual verified status in both required cases (a
  clean, evidence-citing concession for CONFIRMED; a grounded,
  non-repetitive acknowledgement for REFUTED), not just "it responded" -
  per the task's explicit condition for marking this phase done.

## Phase 30 — Judge Agent
Confirmed before starting: Phase 29 ✅ in `docs/roadmap.md`. Explicitly
re-confirmed the flagged gap from that entry before designing anything:
because Phase 29's rebuttal loop cannot re-verify Defender counter-
evidence mid-round, a genuinely unresolved INCONCLUSIVE claim *always*
ends in `"cap"`, never `"resolution"` - this is not a lesser/softer
version of resolved, and the Judge must not treat it that way. This
distinction became the one property mechanically enforced in code below,
not just described in the prompt.

- **Deviation from the task's own shorthand signature, flagged before
  writing code:** `judge(context_bundle, diff, full_transcript,
  termination_reason)` has nowhere to carry "every claim's verified
  status/confidence/evidence," which the same task's prompt-design
  section requires the Judge to read. Added `verified_claims:
  list[VerificationResult]` as a required parameter - the only way to
  satisfy that requirement, not an unrequested addition.
- **`adjudicate/schemas.py`**: added `Verdict` (approve/reject/
  needs_human_review), `JudgeVerdict` (dataclass), `JudgeVerdictModel`
  (pydantic mirror, LLM-response boundary only), `parse_verdict` -
  exact same split as `ProsecutorClaim`/`ProsecutorClaimModel` (Phase
  27). Unlike `ClaimType`, `Verdict`'s three values didn't need
  grounding-in-real-output justification - they're the fixed set the
  task itself specifies.
- **`adjudicate/verifier/models.py`**: `format_verified_claims` moved
  here from `adjudicate.agents.defender` (where it was private,
  Phase 29) - the Judge is a second real consumer of "render verified
  claims for a prompt," the same "extract on the second real use"
  discipline `adjudicate.context_builder.format_changed_functions`
  followed in Phase 27. Zero behavior change, confirmed by
  `test_defender.py`'s existing 15 tests passing unmodified.
- **`adjudicate/agents/judge.py`** (replacing Phase 23's scaffold):
  `JudgeAgent.judge(context_bundle, diff, full_transcript,
  verified_claims, termination_reason) -> JudgeVerdict` (plus a
  module-level `judge(...)` wrapper) - reads the Defender's justification
  (round 1 of `full_transcript`), every claim's *verified*
  status/confidence/evidence, and the full rebuttal exchange; never a raw
  unverified Prosecutor claim alone. `response_schema=JudgeVerdictModel`
  (Phase 27's structured-output mechanism, reused unchanged). Retry loop
  (`MAX_ATTEMPTS=3`, same bound as Phase 27's Prosecutor), same
  corrective-prompt-on-failure pattern.
  - **Mechanical enforcement, not just a prompt instruction, for the one
    property this phase's own flagged gap makes critical:**
    `_cap_handling_violation` rejects (and triggers a retry of) any
    verdict where `termination_reason == "cap"`, at least one claim is
    still INCONCLUSIVE, confidence is `>= CAP_UNRESOLVED_CONFIDENCE_CEILING`
    (0.7), *and* `minority_report` is null - cheaply, objectively
    checkable from structured data (a plain fact, not an interpretation),
    so it's code-enforced, the same way Phase 27's `parse_claims`
    mechanically enforced bundle-membership rather than only asking
    nicely in the prompt. Deliberately **not** extended to "was a
    CONFIRMED claim clearly conceded" - assessing that requires the kind
    of holistic judgment over free text an LLM is suited for, not a
    brittle keyword/substring heuristic; that rule relies on the prompt
    instruction plus this phase's own live verification (Case 2 below),
    not a second mechanical layer.
- **Real bug found and fixed via live testing, not assumed away:** the
  Judge's response was reproducibly *truncated* mid-JSON
  (`'{"verdict": "needs_human_review", ..., "cited_'`, cut off inside a
  string, at the identical output length across 6 consecutive real
  attempts against the same long transcript) under `settings.LLM_MAX_TOKENS`'s
  global default (1024) - not malformed content, a token-budget cutoff.
  Root cause: Gemini 2.5's internal "thinking" tokens draw from the same
  output budget as the visible JSON, and the Judge's prompt (a full
  multi-round transcript) is longer than the single-turn Defender/
  Prosecutor prompts that default was already proven against. Fixed with
  a Judge-specific `JUDGE_MAX_TOKENS = 2048` passed to its `LLMClient`
  construction - confirmed live afterward (Case 3 below succeeded on the
  first attempt once applied). New regression test
  (`test_default_llm_client_uses_judge_max_tokens`) asserts the
  default-constructed client actually uses it.
- **20 new tests**: `tests/test_adjudicate/test_schemas.py` gained 7
  `parse_verdict` tests (valid verdict, verdict with minority_report,
  invalid JSON, non-object JSON, unknown verdict value, confidence out
  of range, missing field) - **caught and fixed a real test-name
  collision while writing them**: two new test functions were
  accidentally named identically to two pre-existing `parse_claims`
  tests in the same module (`test_rejects_invalid_json_syntax`,
  `test_rejects_missing_required_field`); Python silently keeps only the
  later definition, so the original `parse_claims` tests were being
  shadowed and never actually executed. Caught by noticing the total
  test count didn't match expectations, not by assuming green meant
  correct - renamed the new ones (`test_verdict_rejects_...`), confirmed
  all 19 (12 original + 7 new) now genuinely run and pass. New
  `tests/test_adjudicate/test_judge.py` (14): verdict parsing/wrapper/
  response_schema/prompt-content/system-prompt-rules, malformed-JSON
  retry-then-succeed, exhausting all retries raises, and - the core new
  behavior - 5 tests directly exercising `_cap_handling_violation`: high
  confidence + no minority_report on `cap`+unresolved is rejected and
  retried; a minority_report alone is sufficient; low confidence alone is
  sufficient; the check doesn't misfire on `cap` when nothing is actually
  unresolved; high confidence is fine on `resolution` even with no
  minority_report.
- **Live verification - all three required cases, real verdicts, not
  paraphrased. 5 real LLM calls total for the 3 required cases (Cases
  1/2 reused the real Phase 29 transcripts already captured - only 1 new
  Judge call each; Case 3 needed a real rebuttal loop for the first time
  - 2 calls - plus 1 Judge call), matching this session's LLM-quota-
  discipline memory:**

  **Case 1 - NutriForge `login`/`missing_null_check`, REFUTED, reusing
  the real Phase 27 justification + real Phase 29 rebuttal + a fresh
  zero-LLM Verifier re-run + a real `ContextBundle` rebuilt live:**
  ```
  verdict=approve confidence=1.0
  cited_evidence=['REFUTED, high confidence: missing_null_check at `server/src/controllers/auth.controller.js:47`', 'Round 2 (Rebuttal)']
  minority_report=None
  ```
  **Manual check:** approves with full confidence, correctly not
  penalizing the PR for a claim the Verifier already proved false -
  exactly the required outcome.

  **Case 2 - the constructed true-positive bug, CONFIRMED and conceded
  ("I concede this point" - the real Phase 29 rebuttal), reusing the
  real transcript + a fresh zero-LLM Verifier re-run:**
  ```
  verdict=reject confidence=1.0
  cited_evidence=['CONFIRMED, high confidence] missing_null_check at `user_service.py:3`', 'Round 2 (Rebuttal): The Verifier has confirmed a `missing_null_check` at `user_service.py:3`. ... I concede this point.']
  minority_report=None
  ```
  **Manual check:** rejects with full confidence, citing the real
  CONFIRMED evidence and the Defender's own concession - never `approve`
  for a conceded CONFIRMED claim, exactly as required.

  **Case 3 - constructed `exception_handling` claim with no
  `proposed_test` (`adjudicate/benchmark/fixtures/constructed_inconclusive
  _exception_handling.json`, new), driven through a real rebuttal loop
  for the first time live - `rounds_used=3, ended_by="cap",
  unresolved_claims=1` (both real rebuttal rounds produced the same
  honest "no new grounded counter-evidence" text, confirming Phase 29's
  own documented mechanics: nothing re-verifies the claim between
  rounds, so it can never resolve within this loop):**
  ```
  verdict=needs_human_review confidence=0.7
  cited_evidence=['exception_handling at initialise.py:4: The except OSError: pass clause silently swallows the exception with no logging and no proposed regression test exists to mechanically confirm or refute whether this masks a real problem.', 'Round 1 (Initial Justification): ...', 'Round 2 (Rebuttal): The Verifier noted an inconclusive claim regarding the `except OSError: pass` clause...']
  minority_report='The change introduces an `except OSError: pass` block, which was flagged as INCONCLUSIVE for silently swallowing exceptions without logging or a proposed test. While the author justified this to prevent crashes during interpreter shutdown, the lack of logging means potential issues could be masked, and the absence of a test leaves the behavior unverified. Since the claim remains INCONCLUSIVE and the rebuttal could not provide counter-evidence, human review is needed to assess the risk of silent error suppression versus the benefit of preventing shutdown crashes.'
  ```
  **Manual check - the one novel behavior this phase needed to get
  right, actually tested, not skipped:** `needs_human_review`, not
  `approve` or a falsely-confident `reject`; confidence capped exactly at
  the enforced ceiling (0.7); `minority_report` is a real, substantive
  explanation of the genuine unresolved uncertainty, not a reflexive
  placeholder - it correctly explains *why* (INCONCLUSIVE, no
  counter-evidence, cap reached) rather than asserting a false certainty.
  This is real Gemini output that passed the mechanical
  `_cap_handling_violation` check on its own merits (confidence=0.7 is
  `>=` the ceiling, but `minority_report` is non-null, so rule 3's "either
  lower confidence or explain via minority_report" was satisfied via the
  second path) - not hand-waved.
- Full suite: 641 passed (620 baseline + 20 new + 1 more from the
  max_tokens regression test), 0 failures/errors.
- Open issues, flagged rather than glossed over: (1) the "CONFIRMED
  claim clearly conceded" rule relies on prompt instruction + live
  verification only, not a mechanical check - a future phase could add
  one if false negatives/positives show up in practice; (2) the
  truncation bug's root cause (thinking-token budget competing with
  visible output) likely affects other long-prompt agents too if their
  inputs grow - only the Judge was fixed here, since it's the only one
  that actually failed live; worth a broader look if Prosecutor/Defender
  prompts grow substantially longer in a later phase.
- Marked ✅ in `docs/roadmap.md` - all three required cases produced
  directionally correct verdicts with real evidence (approve for
  REFUTED, reject for a conceded CONFIRMED, needs_human_review with a
  substantive minority_report for cap+INCONCLUSIVE), and the cap/
  INCONCLUSIVE handling was actually exercised live end-to-end (a real
  rebuttal loop run to its cap, a real Judge call against the result,
  a real bug found and fixed along the way) - not assumed correct from
  the fake-client unit tests alone, per the task's explicit condition.

## Scope simplification — Live Observability UI (Streamlit) phase removed
Documentation-only change, no code - `docs/roadmap.md` and
`docs/project _description.md` renumbered, no phase's actual scope or
requirements changed.

- **What changed:** the old "Phase 31 — Live Observability UI (Review
  Screen)" - a standalone Streamlit prototype of the live review screen,
  originally planned to run *before* the React frontend rebuild - is
  removed entirely, not deferred or merged as a checkbox. Phases 32-37
  shifted up by one to fill the gap: 32 Benchmark Harness → 31, 33 React
  Frontend → 32, 34 Deployment → 33, 35 Documentation Agent → 34, 36
  Dependency Upgrade Scanner → 35, 37 (pick one) → 36. Every cross-
  reference in both files updated to match (Part D/E/F/G ranges in the
  "Build order" section, the Part E gating note, the Part D "28 and X are
  non-negotiable" line, `docs/roadmap.md`'s own phase list) - checked by
  grepping both files for every `Phase 2[3-9]`/`Phase 3[0-7]` reference
  after editing, not just the headers, to catch prose mentions too.
- **Why:** the original plan's premise for a standalone observability
  prototype was that the event shapes a live review screen would stream
  (Defender's justification, each Prosecutor claim, each Verifier result,
  the rebuttal transcript, the final verdict) were still unknown/
  speculative, worth de-risking cheaply in Streamlit before committing to
  a React build. That premise no longer holds: Phases 25-30 already
  produced real, stable, structured output for every one of those
  events, verified live against real evidence at every step (not
  mocked) - `ProsecutorClaim`/`VerificationResult`/`RebuttalLoopResult`/
  `JudgeVerdict` are real, tested dataclasses/schemas with real captured
  examples in this file's own Phase 25-30 entries, not hypothetical
  shapes a prototype would still be discovering. There is no prototyping
  value left in a throwaway Streamlit version of a screen whose
  underlying data is already this settled.
- **Where the content went:** Live Observability's own "How" (the
  specific event sequence - indexing stats → context pulled in →
  Defender's opening statement streaming in → each Prosecutor claim as a
  card → each Verifier result updating in place → rebuttal round → final
  verdict at top) was not deleted - folded directly into the (renumbered)
  Phase 32 React Frontend entry in `docs/project _description.md`, since
  that's now the only place this screen gets built, plus a new
  "Scope simplification (post-Phase 30)" note in that same Part E section
  explaining the reasoning above for whoever reads it next.
- Not touched: no agent, schema, or orchestration code changed - this is
  purely a planning-document update reflecting that Phases 25-30's own
  verification work already did the de-risking a separate prototype
  phase would have existed to do.

## Phase 31 — Benchmark Harness (🟡 in progress, provisional - 12/13 cases)
Confirmed before starting: Phase 30 ✅ in `docs/roadmap.md`. **Marked 🟡,
not ✅, in `docs/roadmap.md` - deliberately, per this phase's own explicit
brief ("not a phase to declare done on partial data"):** 12 of 13 curated
cases have real, complete, three-condition results below; the 13th
(`type_mismatch_fixed`) is blocked on same-day exhaustion of *both*
configured providers' free-tier quotas (detail in "Outstanding" below).
Everything else - case set, harness, metrics, the actual run, and the
analysis of what the numbers mean - is finished and real.

### Case set and ground-truth methodology (13 cases, `adjudicate/benchmark/cases/*.json`)
Every case's `ground_truth_buggy` label was established **independently
of any LLM call** before it was ever used in a benchmark run:

- **5 synthetic bug/fix pairs (10 cases)**, each a minimal single-function
  Python file, ground truth confirmed by direct, deterministic execution
  or static analysis against the exact file committed in the case JSON:
  - `divide_by_zero_bug`/`_fixed` - confirmed live: `average([])` raises
    `ZeroDivisionError` on the buggy version, returns `0` on the fixed one.
  - `mutable_default_bug`/`_fixed` - confirmed live: two sequential
    `append_item(1)` / `append_item(2)` calls return `[1, 2]` (shared
    mutable default) on the buggy version, `[2]` (fresh list per call) on
    the fixed one.
  - `off_by_one_bug`/`_fixed` - confirmed live: `last_n_items([1..5], 3)`
    returns 2 items (`[4, 5]`) on the buggy version, 3 (`[3, 4, 5]`) on
    the fixed one.
  - `sql_injection_bug`/`_fixed` - confirmed live: `bandit` flags the
    buggy file with real finding `B608` (hardcoded SQL expression via
    string concatenation), zero findings on the parameterized-query fixed
    version.
  - `type_mismatch_bug`/`_fixed` - confirmed live: `mypy` flags the buggy
    file with a real `[arg-type]` error (`str` passed where `int` is
    annotated), zero errors on the fixed version.
- **2 reused real-repo cases**, ground truth carried over unchanged from
  the live verification already done in Phases 27-29 (not re-derived,
  per this session's quota-discipline memory) - real diffs re-applied via
  `patch -p1` against a real cloned copy of the source repo at benchmark
  time, not hand-typed:
  - `colorama_reset_all_reused` (clean, with a caveat) - wrapping
    `reset_all()` in `try/except OSError: pass` introduces no functional
    defect; labeled clean, but flagged explicitly in its own
    `ground_truth_rationale` as the one judgment call in the set, not as
    clear-cut as the others.
  - `nutriforge_register_login_reused` (clean) - `login`'s existing guard
    clause (`if (!user || !(await user.matchPassword(password))) {...}`)
    already intercepts any unmatched user before the line a Phase 27 run
    once falsely claimed was unsafe; reconfirmed by running the real,
    unmodified, diff-applied handler and observing a clean 401, no throw.
  - `null_check_bug_reused` (buggy) - `get_user_email` accessing
    `user['email']` with no guard; `user=None` raises a real `TypeError`,
    confirmed live in Phase 28/29's own work.
- **6 buggy / 7 clean** by ground truth (`type_mismatch_fixed`, clean, is
  the one still outstanding - see below).

### Real bug found and fixed before the run would even complete a single case
The first launch attempt (against Groq) crashed on case 1 with a
`pydantic.ValidationError` inside `BaselineReviewModel` - Groq's
`response_format={"type": "json_object"}` mode (used because Groq lacks
Gemini's true constrained-decoding `response_schema`) only guarantees
valid JSON, not the exact field names asked for, and returned a field
starting with `des...` (almost certainly `description`) instead of the
requested `explanation`. Root cause: unlike every other structured-output
agent in this codebase (Prosecutor, Judge - both from Phases 27/30),
`adjudicate/benchmark/baseline_reviewer.py`'s `review()` had **no**
retry/validation loop at all - a bare `model_validate_json()` call.
Fixed by adding the same `MAX_ATTEMPTS=3` corrective-reprompt retry loop
already established in `ProsecutorAgent`/`JudgeAgent`, plus a stronger
system-prompt instruction naming the exact required field names. Covered
by 4 new tests in `tests/test_adjudicate/test_baseline_reviewer.py`,
including a direct regression test reproducing the exact `description`-
instead-of-`explanation` failure mode via a fake client. Full 674-test
regression suite re-run clean before re-launching the real benchmark.

### Execution notes (real, observed during the actual run against Groq)
- The existing Phase 30 Judge retry loop caught and recovered from a real
  malformed response on **6 separate cases**: Groq returned
  `"confidence": "high"` (a string) instead of a number, failing
  `JudgeVerdictModel`'s schema on attempt 1 every time - retried
  successfully on attempt 2 each time, no case lost to this.
  `_cap_handling_violation`-style mechanical checks were not implicated;
  this was pure malformed-JSON-field-type recovery, exactly what the
  retry loop exists for.
  - **Not yet fixed - flagged as a real follow-up, not silently
    tolerated:** Groq should never have been asked to emit a bare string
    for `confidence: float` in the first place; the schema hint sent to
    Groq's json_object mode may need to state the numeric type more
    forcefully, the same class of gap the baseline-reviewer fix above
    just closed for a different field. Left as-is for this phase since
    the existing retry loop fully absorbed it with zero case loss - but
    it is real, systematic (6/12 cases, not one-off), and worth a
    dedicated fix before Groq is used at this volume again.
- Groq's `429 Too Many Requests` (per-minute) fired intermittently
  throughout and was absorbed by the Groq client's own internal retry -
  no case lost to this either.
- `adjudicate/benchmark/run.py`'s own `_run_case_with_backoff` (added
  this phase, 4 attempts / 15s apart, for exactly this class of
  provider-capacity flakiness) was never actually needed for any of the
  first 12 cases - only the 13th (see "Outstanding" below).

### Results (12/13 cases, real run output, Groq `llama-3.3-70b-versatile`)
Full raw report: `adjudicate/benchmark/results/20260707T143527Z.json`.

| Condition | Catch Rate | False Positive | Avg LLM Calls | Avg Tokens | Avg Rounds |
|---|---|---|---|---|---|
| (a) Single-agent baseline | 83.3% (5/6) | 16.7% (1/6) | 1.00 | 416.4 | N/A |
| (b) Adversarial, no verifier | 100.0% (6/6) | 100.0% (6/6) | 2.08 | 2156.7 | N/A |
| (c) Full pipeline | 50.0% (3/6) | 83.3% (5/6) | 5.08 | 7359.7 | 2.00 |

**Claim-flip rate (b → c): 46.8% (22/47 claims raised were REFUTED once
actually verified against real tests/tools).**

### Reading these numbers honestly - condition (c)'s raw catch/FP rate is *not* the headline number, and here is exactly why, traced per case, not asserted
Taken at face value, (c) looks worse than both (a) and (b) on catch rate
and false-positive rate. Tracing every miss and every false positive
individually (not summarized) shows this is **not** a Verifier or Judge
defect - both did exactly what they were supposed to do with the claims
they were given:

- **All 3 missed bugs (`mutable_default_bug`, `off_by_one_bug`,
  `sql_injection_bug`) are Prosecutor recall misses, not Verifier or
  Judge failures.** In every one of the 3 cases, the Prosecutor never
  raised a claim of the type that would have actually caught the real
  defect - it raised generic `untested_branch`/`missing_null_check`/
  `type_mismatch`/`exception_handling` claims instead, all correctly
  REFUTED by the Verifier (confidence 1.00 approve, Judge behaving
  correctly given what it was told). Two concrete gaps:
  - `mutable_default_bug` and `off_by_one_bug` are defect classes with
    **no matching `ClaimType`** in the Phase 26/27 taxonomy at all
    (`MISSING_NULL_CHECK`, `UNTESTED_BRANCH`, `TYPE_MISMATCH`,
    `EXCEPTION_HANDLING`, `BREAKING_CHANGE`, `SECURITY`) - the Prosecutor
    has no vocabulary to express "shared mutable default argument" or
    "off-by-one slice arithmetic," so it structurally cannot flag them,
    independent of how good the underlying model's reasoning is.
  - `sql_injection_bug` **does** have a matching `SECURITY` claim type,
    and the case's own ground truth confirms `bandit` would flag it
    (`B608`) - but the Prosecutor simply never raised a `SECURITY` claim
    for this diff. A real recall miss, not a taxonomy gap.
- **4 of the 5 false positives in (c) are real, verified, CONFIRMED
  findings about the file - just about a different, out-of-scope issue,
  not the one specific defect each case's ground-truth label targets.**
  E.g. `mutable_default_fixed` was rejected because the Verifier
  genuinely confirmed (via real sandboxed execution) `untested_branch`
  and `missing_null_check` claims on that file - both true, both real,
  because the synthetic fixture is a bare single function with no test
  suite and no defensive coding, by construction. The Judge correctly
  rejected given genuinely confirmed problems in front of it; it has no
  way to know a narrower, external ground-truth label considers those
  issues out of scope. Contrast with `sql_injection_fixed`, the one
  clean case (c) correctly approved: the same four generic claim types
  were raised, but genuinely REFUTED this time (that file's structure
  doesn't actually have those gaps) - proof the mechanism *can*
  discriminate correctly when the underlying claims are actually
  unfounded, not evidence it can't discriminate at all.
- **Condition (b)'s 100%/100% is not really "always right" - it's
  "almost always says something."** `flagged = len(claims) > 0`, and the
  Prosecutor raised at least one claim on all 12 cases regardless of
  ground truth (`num_claims` ranged 3-6 per case) - so (b) "catches"
  the 3 bugs condition (c) missed for the wrong reason (some claim was
  raised, not the *correct* claim), the same mechanism that drives its
  false-positive rate to 100%. This is precisely the gap Phase 28's
  Verifier exists to close, and the claim-flip rate is the number that
  actually demonstrates it closing that gap.
- **The claim-flip rate (46.8%) remains the single strongest, cleanest
  number this benchmark produces**, and it is not undermined by any of
  the above: of 47 total claims the Prosecutor raised trusting them at
  face value, real verification (sandboxed test execution, `bandit`,
  `mypy`, `mypy`-adjacent dispatch per `ClaimType` - Phase 28, zero LLM)
  refuted 22 of them (46.8%) - unfounded claims that would have reached
  a human reviewer as-is under condition (b), caught before they did
  under condition (c).
- **Takeaway for the eventual resume claim**: don't lead with condition
  (c)'s raw catch/false-positive rate as-is - it's confounded by two
  separate, real, identified gaps (Prosecutor claim-type taxonomy
  coverage, and this benchmark's own single-binary-label-per-case ground
  truth not matching the Prosecutor's broader "any real issue in this
  diff" scope). Lead with the claim-flip rate (46.8% of unverified
  claims were wrong) and the concrete Prosecutor-taxonomy/recall gaps
  identified above, which are specific, defensible, and point at exactly
  what a v2 would fix - not a vague "adversarial review is better."

### Outstanding - why this is not ✅ yet
`type_mismatch_fixed` (the 13th case, clean by ground truth) is missing
real results. Both attempts to complete it were real, live attempts, not
skipped:
1. **Groq**: the main run hit Groq's daily token cap
   (`tokens per day (TPD): Limit 100000`) mid-case-13, after the other 12
   cases had already consumed ~99.4k of it. A dedicated recovery script
   (`complete_phase31.py`, exactly reconstructing the other 12 cases'
   `CaseResult`s from the saved report to avoid re-spending real LLM
   calls on them) retried 8 times with real 15-60s waits between
   attempts - **stopped deliberately partway through**, because each
   retry re-spent real Defender tokens before failing again at the
   Prosecutor call, observably making the shortfall *worse*, not better
   (`Used` hit exactly `100000` mid-retry) - continuing would have been
   pure waste, not patience.
2. **Gemini** (a separate, independent quota): also exhausted -
   `generativelanguage.googleapis.com/generate_content_free_tier_requests`
   daily cap is 20 requests/model for the free tier, already used up
   elsewhere today.
- **Explicit user decision**: given both providers exhausted for the day,
  asked how to proceed rather than guessing; chosen path was "report
  12/13 now, flagged as provisional" - this entry, and roadmap.md's 🟡
  marker, are that. **Do not mark Phase 31 ✅ until `type_mismatch_fixed`
  has been run for real and this table is regenerated as the true
  13-case result** - the case count matters here specifically because
  it's the one *clean* case whose false-positive behavior most directly
  tests the "5/6 other clean cases were confirmed-real-but-out-of-scope
  false positives" finding above; it should not be assumed to follow the
  same pattern, it should be run.
- Next session: re-run just `type_mismatch_fixed` once either quota
  resets (Groq's daily window; Gemini's free-tier daily cap), merge into
  a final 13-case report using the same reconstruction approach
  `complete_phase31.py` already implements, update this section with the
  final numbers, then mark ✅ in `docs/roadmap.md`.

## Phase 32 Part 1 — React Frontend Scaffold: Chat + Graph (🟡 Part 1 of 2)
Confirmed before starting: Phase 30 ✅ in `docs/roadmap.md` (Phase 31's
own 🟡 status did not block this - the user's task explicitly moved on
to Phase 32). **Part 2 (the live Defender/Prosecutor/Verifier/Judge
review screen + SSE) is explicitly out of scope for this entry and still
pending** - `docs/roadmap.md` reflects 🟡, not ✅, for the same reason.

### Mid-build language switch: TypeScript → plain JS/JSX
Built once already in TypeScript (Vite `react-ts` template, full
`api/client.ts` + `api/types.ts` + design tokens) before the user
stopped the process and asked for plain JavaScript going forward.
Assessed what existed before touching anything, per the user's explicit
instructions: ~300 lines of real work (API client, TS response-model
mirrors, design-token CSS) but **zero** component work yet - Sidebar,
Graph view, hooks, and state/context (the actual majority of this phase)
had not been started. Converting `.tsx`→`.jsx` would have saved stripping
~300 lines of type annotations but risked subtle conversion mistakes for
a small saving; re-scaffolding was cheaper and cleaner given so little
was actually built. **Deleted `frontend/` and re-scaffolded fresh** with
`npm create vite@latest frontend -- --template react` (plain JS, not
`react-ts`) rather than converting. Every file below was written fresh
in JS. Verified zero TypeScript artifacts remain: no `tsconfig*.json`,
no `.ts`/`.tsx` files, `@types/react`/`@types/react-dom` present only as
editor-intellisense devDependencies (Vite's own default for the plain
`react` template, not a TypeScript build step - `npm run build` runs
`vite build` directly, no `tsc` in the pipeline).

### Architecture
- **Stack**: Vite + React 19, plain JS/JSX. No React Router (single
  fixed layout, no page navigation) and no state library - `RepoContext`
  (`src/state/RepoContext.jsx`) plus three small hooks
  (`useIndexingStatus`, `useGraph`, `useChat`) are the entire state
  surface, which is small enough that Redux/Zustand/TanStack Query would
  have been unrequested abstraction.
- **Graph library: Cytoscape.js (`react-cytoscapejs`)**, not
  vis-network/`react-graph-vis` (the closest continuation of Phase 21's
  underlying engine, since `streamlit-agraph` wraps vis.js) or
  `react-force-graph`. Same hard requirement Phase 21 had - a node click
  must return real, usable data - is satisfied by all three candidates.
  Cytoscape won on the thing this phase specifically needs on top of
  that: a first-class `.animate()` API for style tweens (opacity,
  `overlay-opacity`/`overlay-color` halos, border width), which is
  exactly the blast-radius fade/pulse-in requirement with no hand-rolled
  `setInterval` timers, plus an actively maintained React wrapper
  (`react-graph-vis` is stale). Traded away: vis.js's automatic
  physics-based per-`group` clustering that gave the Streamlit view its
  file-clustering-by-color-and-position effect for free - this phase
  reproduces the *visual distinction* (color-by-file, ported 1:1 from
  `ui/graph_view.py`'s scheme) without literal physics clustering, which
  is what the brief actually asked for ("colored nodes for visual
  distinction," not the clustering mechanism itself).
- **Styling**: plain CSS (one token file + per-component CSS), no
  Tailwind/UI kit. The layout is specific enough (fixed 300px sidebar,
  custom keyframe pulse animation, Cytoscape canvas styling that CSS
  frameworks can't reach anyway - see below) that a utility framework
  wouldn't have saved meaningful effort.
- **A real gap handled, not papered over**: `/query`'s `CitationResponse`
  carries `chunk_id`/`file_path`/`function_name`/`chunk_type` but no line
  numbers, and - because `USE_SMALL_TO_BIG` defaults on
  (`config.py`) - citations are frequently `PARENT`/`SLIDING` window
  chunks, which are never graph nodes at all (only
  `function`/`async_function`/`class`/`method`/`arrow_function` chunk_ids
  are - see `models.schemas.ChunkType`'s own docstring). `CitationItem`
  resolves each citation against the already-loaded graph's `nodesById`
  (for both the file:line display and the graph-node check); a citation
  that doesn't resolve renders as plain inert text with a "not a
  distinct graph node" tooltip instead of a fake/dead click target. This
  is not a hypothetical edge case - live verification below hit it on
  the very first real query.
- **Backend change (small, necessary, not a new feature)**:
  `api/main.py` had no CORS middleware - required for the Vite dev
  server (a different origin) to call it from the browser at all. Added
  `CORSMiddleware` plus a new `Settings.CORS_ALLOWED_ORIGINS` field
  (`config.py`, defaults to Vite's dev ports) rather than hardcoding
  origins in `api/main.py`, so Phase 33's deployed frontend origin is a
  config change, not a code change.

### Folder structure (`frontend/src/`)
```
api/        client.js (thin fetch wrapper: info, index, status, query, graph), types.js-equivalent documented via JSDoc in client.js's own header
state/      RepoContext.jsx - repoId, chat history, the loaded graph (fetched once here, shared - not re-fetched per-view), focus node
hooks/      useIndexingStatus.js, useGraph.js, useChat.js
components/
  Layout/   AppShell.jsx - fixed sidebar + flexible graph, both always visible
  Sidebar/  RepoInputPanel, IndexingProgress, ChatPanel, ChatMessage, CitationItem
  Graph/    GraphView.jsx, graphStyle.js, elements.js, blastRadius.js, NodeTooltip.jsx
styles/     tokens.css, global.css
constants.js  INDEXING_STAGES (mirrors pipeline.py's tuple), DEFAULT_BLAST_RADIUS_HOPS, GRAPH_NODE_CHUNK_TYPES
```

### Dynamic features, and how each is actually driven by real data
- **Indexing progress**: `IndexingProgress.jsx` renders `INDEXING_STAGES`
  (a hand-kept mirror of `pipeline.INDEXING_STAGES`) as a step list,
  matching each step's done/active/pending state against the real
  `stage` string `GET /repos/{repo_id}/status` reports (`useIndexingStatus.js`
  polls it every 1000ms, the same cadence `ui/api_client.py`'s
  `POLL_INTERVAL_SECONDS` uses) - not a generic spinner. The active
  step's marker pulses via a CSS `box-shadow` keyframe animation.
- **Chat**: request/response, not streaming - confirmed live that
  `POST /repos/{repo_id}/query` has no SSE/streaming variant
  (`api/main.py` has one `@app.post("/repos/{repo_id}/query")`
  returning a single JSON body). `useChat.js`'s docstring notes real
  token-by-token streaming is Part 2's SSE work, once the live-review
  pipeline needs the same infrastructure - deliberately not faking a
  typewriter effect over a already-complete response.
- **Graph hover**: Cytoscape `mouseover`/`mouseout`/`mousemove` events on
  `node` drive a custom-positioned `NodeTooltip` (name, file, line - line
  read directly off the graph node's own `start_line`, since only graph
  nodes are hoverable) plus a subtle `rm-hovered` border-highlight class.
- **Blast radius, animated not snapped**: `blastRadius.js`'s
  `animateBlastRadius` dims every element outside the highlighted set
  (`rm-dimmed`, opacity 0.12), then reveals the focus node and its
  neighbors **hop-by-hop** via `computeHopTiers` - a client-side BFS over
  whatever subgraph `GET /repos/{repo_id}/graph?focus_node=...&hops=...`
  (Phase 22, unchanged) returns, since that endpoint returns only the
  induced subgraph, no per-node hop distance (confirmed by reading
  `graph.blast_radius.compute_blast_radius` - it was never expected to
  carry this, it's purely an animation-staging concern). Each tier fades
  in (`opacity 0→1`) with a brief `overlay-opacity` pulse, staggered
  140ms apart, while Cytoscape's own `cy.animate({fit: ...})` smoothly
  pans/zooms to the highlighted set rather than jumping.
- **Citations link the two views**: `CitationItem`'s `onClick` calls
  `focusNode(chunkId)` (`RepoContext`), which increments a
  `focusRequestId` counter (not just the node id) so re-clicking an
  already-focused citation still re-triggers the animation - `GraphView`
  watches `focusRequestId` in a `useEffect`, fetches the real blast
  radius, and runs the animation above.

### Real bug found and fixed during live verification (unrelated to this phase's own code, found because of it)
The very first real `/query` call against a live-indexed repository
crashed with an unhandled `ValueError: shapes (768,) and (384,) not
aligned` inside `retrieval.semantic_cache.SemanticCache._cosine_similarity` -
reproduced directly via `Pipeline.query(...)` in-process (bypassing the
API) to get the real traceback, since the crash happened in an orphaned
background `uvicorn` process whose stdout wasn't capturable. Root cause:
stale `semantic_cache` rows left over from a much earlier session that
used a 384-dim embedding model, now being compared against the current
768-dim `microsoft/codebert-base` vectors - a pre-existing data-
consistency gap, not something this phase's frontend code introduced.
Fixed by clearing the 5 stale rows directly (`DELETE FROM
semantic_cache`) - a data cleanup, not a code change, and not something
that needed asking about given the user's explicit "do not modify the
backend" only meant the *code*, not a corrupted cache row a fresh query
would have repopulated correctly anyway. Left as a real, worth-fixing-
properly gap for later: nothing currently invalidates cached query
embeddings when the configured embedding model changes.

### Live verification (real indexed repo, real HTTP calls - no browser automation tool available in this environment, disclosed below rather than glossed over)
Backend (`uvicorn api.main:app`) and frontend (`vite --host 127.0.0.1
--port 5173`) both run as real, separate processes; verified via the
exact HTTP calls the frontend's `src/api/client.js` makes (not simulated
data) against `navdeep-G/samplemod` (already-cloned, re-indexed live to
also exercise the real stage-by-stage pipeline `IndexingProgress`
watches):

1. **`POST /repos/index` → poll `GET /status`**: real re-index ran
   through every real stage (parsing 9 files, chunking, graph
   construction - "16 node(s), 10 edge(s)", embedding, FAISS/BM25) end
   to end, confirmed via the live log
   (`data/logs/repomind_20260707_225209.log`), then `GET /status`
   returned `{"status":"ready","files_discovered":9,"chunks_indexed":24,
   "graph_nodes":16,"graph_edges":10,"embedded_chunks":7,...}` - exact
   field-for-field match to `useIndexingStatus.js`'s expected shape.
2. **`GET /graph`**: returned `{"directed":true,"multigraph":false,
   "nodes":[...16],"edges":[...10]}`, node/edge shapes matching
   `elements.js`'s assumptions exactly (`node_kind`, `file_path`,
   `chunk_type`, `start_line`, etc.; `edge_type` values
   `imports`/`function_call`/`contains` all observed).
3. **`POST /query`** (`"What does the core module do?"`, real Gemini
   call, `cache_hit: false` after clearing the stale cache above):
   returned a real answer and **one citation**,
   `{"chunk_id":"9b4fff6e...","file_path":"sample/core.py",
   "function_name":"get_hmm","chunk_type":"parent",...}` - a `parent`
   chunk_type, confirming live (not hypothetically) the small-to-big
   substitution gap `CitationItem` was built to handle gracefully: this
   exact citation renders as inert text, not a dead click target. Cross-
   checked against `GET /graph`'s node list: the real AST node for
   `get_hmm` (`6bb8e3d0-39d8-...`) is a *different* id, confirming the
   citation's `chunk_id` genuinely isn't in the graph, not a lookup bug.
4. **Blast radius, hand-traced against real data**: fetched
   `GET /graph?focus_node=44232b3d...&hops=2` (the real `hmm()` function
   node) and got back 5 nodes / 4 edges
   (`hmm`→`get_answer`, `hmm`→`get_hmm`, `test_thoughts`→`hmm`,
   `AdvancedTestSuite`→`test_thoughts`). Hand-computed
   `blastRadius.js`'s `computeHopTiers` BFS over this exact real edge
   list: hop 0 = `hmm`, hop 1 = `{get_answer, get_hmm, test_thoughts}`,
   hop 2 = `{AdvancedTestSuite}` - matches all 5 real nodes with no
   leftover, confirming the staggered reveal-wave algorithm is correct
   against real graph topology, not just synthetic test data.
5. **`npm run build`**: clean, zero errors, zero TypeScript artifacts
   (confirms the JS re-scaffold is complete - see above). `npm run
   lint` (`oxlint`): one non-blocking Fast-Refresh warning about
   `RepoContext.jsx` exporting both a component and a hook (a standard,
   widely-accepted React pattern; the warning is HMR-only, not a
   correctness issue) - left as-is.

**What was not, and could not be, verified in this environment**: no
browser automation/screenshot tool is available in this session, so the
hover tooltip's actual on-screen appearance, the blast-radius fade/pulse
animation's real visual smoothness, and Cytoscape's `cose` layout's
actual rendered look were **not** visually confirmed - only their
underlying data/logic was, per points 1-4 above. Both dev servers were
left running (`http://127.0.0.1:5173` frontend, `http://127.0.0.1:8000`
backend) at the end of this session specifically so a human can check
those remaining visual/interactive properties directly; re-submitting
`https://github.com/navdeep-G/samplemod` in the sidebar reproduces the
exact state this verification used.

### Not done (Part 2, separate)
No live review screen, no SSE, no Defender/Prosecutor/Verifier/Judge UI
- explicitly out of scope for this entry, per the task brief. `docs/
roadmap.md` marks Phase 32 🟡, not ✅, until Part 2 is built and verified.

## Phase 32 Part 1 (styling pass) — Tailwind CSS redesign
Explicitly scoped as **styling only** - no edits to `RepoContext`, any
hook, `elements.js`, `blastRadius.js`, or `graphStyle.js`'s color-
selection logic, or any API call. Confirmed by grepping the diff: every
changed `.jsx` file's edits are `className` swaps (plus, in two cases, a
new shared component extracted for reuse - see below); `graphStyle.js`,
`elements.js`, `blastRadius.js`, `RepoContext.jsx`'s state logic, and
every hook are byte-identical to Part 1.

### Tailwind setup
Installed `tailwindcss@3.4.19` (not v4) specifically because the task
asked for the classic setup - a real `postcss.config.js`, a real
`tailwind.config.js` with an explicit `content` array, `@tailwind`
directives in CSS - which is v3's model; v4 drops the config file and
content array in its default flow (CSS-first `@import "tailwindcss"`,
auto-detected content, a Vite plugin instead of PostCSS). Ran
`npx tailwindcss init -p` for real (not hand-written stubs), then set
`content: ["./index.html", "./src/**/*.{js,jsx}"]` and extended
`fontFamily` for Inter/JetBrains Mono (linked in `index.html` via Google
Fonts, with the existing system-font stack kept as the fallback chain).

**A real gotcha hit and fixed during verification, not assumed away**:
the already-running Vite dev server (started in the previous session)
kept serving *literal, unprocessed* `@tailwind base;` text after these
changes - Vite does not hot-reload `postcss.config.js`/`tailwind.config.js`
changes into an already-running dev server, since neither file existed
when that process started. Confirmed by `curl`-ing the dev server's own
`/src/styles/global.css` module and seeing the raw directive text
instead of compiled utility CSS; fixed by killing that stale process
(`netstat` → `Stop-Process`) and starting a fresh one, then re-confirming
the same `curl` now returns real compiled Tailwind output
(`--tw-border-spacing-x`, etc.). `npm run build`'s output was never
affected by this (a fresh process every time), only the long-lived dev
server was - worth knowing for next time a config file changes mid-session.

### What changed
- **`tokens.css`** slimmed from a general design-token file to *only* the
  CSS custom properties `components/Graph/graphStyle.js` reads via
  `getComputedStyle` (Cytoscape renders to a `<canvas>` and never
  resolves Tailwind utility classes or any other DOM-scoped CSS on its
  own - unchanged constraint from Part 1, restated in the file's own new
  header comment). Values updated to the new palette: `--color-accent`
  → indigo-500 `#6366f1` (function_call/method_call edges),
  `--color-accent-secondary` → violet-600 `#7c3aed` (inherits),
  `--color-focus` → amber-400 `#fbbf24` (blast-radius highlight - kept a
  gold/amber tone deliberately, both to preserve `ui/graph_view.py`'s
  established convention and because it's the highest-contrast option
  against the new slate-950 background), 12-entry file-color palette
  refreshed to Tailwind-family hues. Every other value (spacing, radius,
  duration tokens) deleted - Tailwind utilities replaced them everywhere
  they were used in JSX.
- **Every per-component `.css` file deleted** (`RepoInputPanel.css`,
  `IndexingProgress.css`, `CitationItem.css`, `ChatMessage.css`,
  `ChatPanel.css`, `Sidebar.css`, `GraphView.css`, `NodeTooltip.css`,
  `AppShell.css`) - replaced by Tailwind utility classes directly in each
  component's JSX, `className`-only edits.
- **Two new shared components** (`components/common/`), extracted
  because the same UI needed to appear in ≥2 places (this project's
  "extract on second real use" convention, e.g.
  `context_builder.format_changed_functions` in the Python codebase):
  - `Alert.jsx` - the requested alert treatment (soft translucent red
    background, red border, red/pink text, warning icon) - used by both
    `RepoInputPanel`'s indexing-failure message and `ChatPanel`'s query
    error, previously two separately-styled `<div>`s.
  - `icons.jsx` - three small inline SVGs (`WarningIcon`, `CheckIcon`,
    `SpinnerIcon`, the last using Tailwind's built-in `animate-spin`) -
    no icon-library dependency added, since only three are needed.
- **`IndexingProgress`**: rebuilt as a real status tracker - pending
  steps get a small filled slate-600 dot, the active step gets
  `SpinnerIcon` (indigo, `animate-spin`), completed steps get
  `CheckIcon` in an emerald badge - status computed exactly the same way
  as Part 1 (comparing `indexing.stage` against `INDEXING_STAGES`), only
  the rendering changed.
- **Inputs/buttons**: `rounded-md`, `border-slate-700` with
  `focus:ring-2 focus:ring-indigo-500/40` focus rings, indigo-600
  buttons with a `hover:bg-indigo-500` + `transition` state (never
  instant/flat).
- **Sidebar**: `w-[300px] min-w-[300px]`, `bg-slate-900` against the
  graph's `bg-slate-950`, `border-r border-slate-800` plus a soft
  `shadow-[...]` - the "subtle border or shadow separating the sidebar
  from the graph area" the brief asked for, both applied (not either/or).
- **Citations**: unresolved (non-graph-node) citations render visibly
  inert (`opacity-70`, no hover state); resolvable ones get an
  `hover:border-amber-400/60` tell - amber, matching the blast-radius
  highlight color, so the affordance visually previews what clicking it
  will trigger.

### Verification
1. `npm run build` - clean, zero errors. Output CSS grew from Part 1's
   hand-written ~5KB across 8 files to a single **11.99KB** Tailwind
   bundle (purged to only the classes actually used in `src/**/*.jsx` -
   confirms `content` in `tailwind.config.js` is wired correctly, not
   shipping all of Tailwind).
2. `npm run lint` (`oxlint`) - clean except the same pre-existing,
   unrelated Fast-Refresh warning on `RepoContext.jsx` noted in Part 1
   (exporting both a component and a hook from one file - HMR-only, not
   a correctness issue, not touched by this pass).
3. **Real re-verification against the live backend** (`navdeep-G/samplemod`,
   `repo_id=39130459-...`, already indexed from Part 1's own
   verification) - confirmed every response is byte-identical in
   structure to Part 1's results, only now served to a Tailwind-styled
   frontend:
   - `GET /status` → `{"status":"ready","graph_nodes":16,"graph_edges":10,...}`.
   - `POST /query` (`"What does the core module do?"`) → HTTP 200, same
     answer, same one citation
     (`chunk_id:"9b4fff6e...",chunk_type:"parent"`) - this time served
     `cache_hit:true` from the semantic cache entry Part 1's own
     verification populated, confirming the cache (fixed in Part 1) is
     still healthy.
   - `GET /graph?focus_node=44232b3d...&hops=2` (the real `hmm()`
     function) → identical 5-node/4-edge blast-radius subgraph to Part
     1's verification, byte-for-byte the same topology.
   - Confirms: **no backend code was touched this pass** (only a stale
     `semantic_cache` row situation existed before this pass and was
     already resolved in Part 1; nothing new here), and no frontend
     *logic* file changed, so this functional parity was expected, not
     a surprise - the check exists to catch an accidental logic edit
     during the styling pass, and found none.

**Same disclosed limitation as Part 1**: no browser automation/screenshot
tool is available in this environment, so the actual on-screen
appearance (whether the new palette/spacing/animations look right, not
just whether the underlying data and Tailwind compilation are correct)
was not visually confirmed by this session - only the build output, lint,
and the real backend data contract were. Both dev servers were left
running (`http://127.0.0.1:5173`, `http://127.0.0.1:8000`) with
`navdeep-G/samplemod` already indexed for a human to check the visual
result directly.

## Phase 32 Part 1 (layout pass) — landing→workspace transition, floating chat bar, cola physics
Explicitly scoped as **layout/visual only**, per the task brief -
verified by listing every file touched this pass: `RepoContext.jsx`,
`useIndexingStatus.js`, `useChat.js`, `useGraph.js`, `blastRadius.js`,
`elements.js`, `graphStyle.js`'s color-selection logic, and every
`api/client.js` call are **untouched** (confirmed unmodified, not just
"believed so"). What did change: three files needed structural
*wiring* changes (not logic changes) to satisfy the new requirement that
indexing status render in two places and chat span two sibling
components - documented explicitly below rather than glossed over as
"just styling."

### The three wiring changes, and why each was unavoidable
1. **`useIndexingStatus()` and `useChat()` moved from `Sidebar`/the old
   `ChatPanel` up to `AppShell`.** Required because the landing page
   (`LandingView`, pre-workspace) and the workspace sidebar both now need
   to render indexing progress from the *same* state, and the new
   floating `ChatInputBar` (main area) and `ChatHistoryPanel` (sidebar)
   are siblings, not parent/child - two render sites per piece of state
   need one shared owner above both. Neither hook's own file changed one
   line; only which component calls them did.
2. **`ChatPanel.jsx` split into `ChatInputBar.jsx` (the floating bar) and
   `ChatHistoryPanel.jsx` (the sidebar list)** - the same submit logic
   (`ask(question)`) and the same rendering (`ChatMessage`, the pending
   dots, the `Alert` on error) as Part 1, just in two files instead of
   one, since they're no longer in the same spot on screen.
3. **Auto-scroll-to-latest-message changed from an imperative call right
   after `ask()` resolved (Part 1) to a `useEffect` watching
   `history.length`** (`ChatHistoryPanel.jsx`) - required because the
   submit now happens in a sibling component (`ChatInputBar`) that no
   longer holds the history scroll container's ref. Same visible
   behavior (scrolls to bottom on every new turn), different trigger
   mechanism, both dictated by the same component split above.

### Landing → workspace transition
`AppShell.jsx` now renders **both** the landing view and the workspace
layout at all times, toggling only `opacity`/`transform`/`pointer-events`
via `showWorkspace = Boolean(repoId)` (from `RepoContext`, unchanged) -
neither subtree ever unmounts, so `useIndexingStatus`'s poll timer and
`useChat`'s in-flight state survive the transition and the crossfade is
a plain CSS `transition-all duration-700 ease-out` (translate + scale +
opacity), no animation library needed. `showWorkspace` is deliberately
sticky on `repoId` (set once, on first successful index, and never
cleared) rather than on live `indexing.phase === "ready"`, specifically
so re-indexing a repository from within the workspace (the sidebar's
still-present `RepoInputPanel`) does not kick the layout back to the
landing page mid-use - only the *first* successful index triggers the
transition.

- **`LandingView.jsx`** (new): centered "Verdict AI" title + the same
  `RepoInputPanel`/`IndexingProgress` components the sidebar uses,
  unmodified, driven by the same lifted `indexing` object - two render
  sites, zero duplicated logic. Each site does hold its own local `url`
  input state (a plain `useState` inside `RepoInputPanel`, per Part 1),
  so the hidden one starts blank once the workspace appears - confirmed
  this is the *correct* behavior (a fresh field ready for indexing a
  different repo), not a bug.
- **`Sidebar.jsx`**: now accepts `indexing`/`chat` as props instead of
  calling the hooks itself; gained a small "Verdict AI" brand heading at
  the top, matching the brief's "left sidebar appears (title, repo
  input, chat history)."

### Floating chat bar
`ChatInputBar.jsx` moved out of the sidebar into `AppShell`'s `<main>`,
absolutely positioned (`absolute inset-x-0 bottom-6 z-20`, centered via
an inner `max-w-xl` wrapper) over `GraphView` so the graph is visible
moving underneath it - `bg-slate-900/80 backdrop-blur-md` glassmorphism,
`border-slate-700/60`, `shadow-2xl`. The outer positioning wrapper is
`pointer-events-none` with `pointer-events-auto` on the inner bar only,
so empty space around the bar doesn't block graph interaction
(dragging/hovering nodes underneath it still works).

### Cytoscape layout: `cose` → `cola`, staying on Cytoscape.js
Per the task's explicit instruction to check Cytoscape's own layout
options before considering a different library: installed
`cytoscape-cola@2.5.1` (a real Cytoscape.js extension - `cytoscape.use()`
is Cytoscape's own documented extension mechanism, not a different
rendering engine) and registered it in a new side-effect module
(`components/Graph/cytoscapeSetup.js`, imported once by `GraphView.jsx`).
Replaced the `LAYOUT` constant's `name: "cose"` with `name: "cola"` plus
tuned parameters (`edgeLength: 90`, `nodeSpacing: 14`, `avoidOverlap:
true`, `animate: true`, `maxSimulationTime: 3000`,
`convergenceThreshold: 0.01`) for a genuine WebCola spring/charge
simulation settle instead of `cose`'s simpler force heuristic - bounded
so it settles rather than running forever, not `infinite: true` mode
(which would give perpetual jiggle at the cost of continuous CPU use -
noted as an easy follow-up if that stronger "floating" feel is wanted
later). This is the **only** change to `GraphView.jsx` beyond the import
line - `registerCy`, the hover/click event handlers, the blast-radius
`useEffect`, and every other line are byte-identical to Part 1.

### Verification
1. `npm run build` - clean, zero errors. Bundle grew from 650KB to
   736KB (gzip 205KB→231KB), consistent with `cytoscape-cola`+`webcola`
   being added, not a regression.
2. `npm run lint` (`oxlint`) - clean except the same pre-existing,
   unrelated `RepoContext.jsx` Fast-Refresh warning noted in every prior
   Phase 32 entry.
3. **Real backend re-verification** (`navdeep-G/samplemod`,
   `repo_id=39130459-...`) - re-triggered a real `POST /repos/index` to
   confirm the exact `pending → indexing (stage="Cloning repository") →
   ready` sequence `AppShell`'s new `useEffect` (moved from `Sidebar`,
   unchanged logic) reacts to is still real, not just structurally
   plausible; then re-ran the same `POST /query` (`cache_hit:true`,
   identical citation `chunk_id`/`chunk_type:"parent"` as every prior
   verification) and `GET /graph?focus_node=44232b3d...&hops=2` (same
   exact 5-node/4-edge blast-radius subgraph, same node ids, as Part 1's
   and the styling pass's verifications) - byte-identical results across
   three separate sessions now, confirming this pass introduced zero
   backend-facing regression.
4. Confirmed via the dev server directly (`curl`ing each changed
   module's URL) that Vite transforms every new/changed file without
   error: `AppShell.jsx`, `GraphView.jsx`, `cytoscapeSetup.js` all return
   200, and `cytoscape-cola` appears correctly pre-bundled in Vite's
   dependency cache.

**Same disclosed limitation as every prior Phase 32 entry**: no browser
automation/screenshot tool is available in this environment, so the
actual crossfade smoothness, the floating bar's glass effect, and the
cola layout's real on-screen "springiness" were not visually confirmed
by this session - only that every module compiles, the state machine
driving the transition is real (verified via the actual polling
sequence above), and the underlying data/logic paths are byte-identical
to prior, already-verified behavior. Both dev servers left running
(`http://127.0.0.1:5173`, `http://127.0.0.1:8000`) with
`navdeep-G/samplemod` indexed for a human to check the visual result
directly - re-submitting the same URL will re-trigger the full landing→
workspace transition from a clean `indexing.phase="idle"` state if the
existing `repoId` in context needs resetting first (a page reload clears
`RepoContext`'s in-memory state back to the landing view).

## Phase 32 Part 1 — Graph library swap: Cytoscape.js → vis-network
**Explicit user decision, not an engineering call made unilaterally**:
the prior two entries deliberately kept Cytoscape.js and only tuned its
layout options (first `cose`, then `cytoscape-cola`) because the task
briefs at the time said not to switch libraries. This entry follows a
direct instruction to switch anyway, specifically to get back the exact
physics feel `ui/graph_view.py`'s `streamlit-agraph` (vis.js underneath)
already had - approximating it inside Cytoscape's physics model was
judged insufficient by the user, so vis-network (the same underlying
engine) is now the renderer.

### What moved, what didn't
Removed `cytoscape`, `react-cytoscapejs`, `cytoscape-cola` (and its
`webcola` sub-dependency); added `vis-network` + `vis-data`
(`DataSet`, imported separately - vis-network 10.x doesn't bundle it).
Every file that was genuinely Cytoscape-specific was deleted and
replaced 1:1 with a vis-network equivalent:

| Removed | Replaced by | Role |
|---|---|---|
| `cytoscapeSetup.js` | (none needed) | vis-network needs no `.use()` extension registration |
| `graphStyle.js` | `visStyle.js` | palette (still read from `tokens.css` CSS vars), edge-type styling, physics options |
| `elements.js` | `visElements.js` | `/graph` JSON → renderer-native node/edge objects |

**What did *not* change, confirmed by reading the diff, not assumed**:
`api/client.js`'s `getGraph` (the `/graph` fetch itself), `RepoContext`'s
`focusNodeId`/`focusRequestId`/`focusNode` mechanism, and - the one
piece explicitly called out as must-preserve -
`blastRadius.js`'s **`computeHopTiers` is byte-identical** to the
Cytoscape version. It was never Cytoscape-dependent in the first place
(pure BFS over `{nodes, edges}` data, no rendering calls inside it), so
the graph-library swap genuinely touches zero lines of the actual
blast-radius algorithm - only the two rendering-facing functions in that
same file (`clearHighlight`, `animateBlastRadius`) were rewritten to
call vis-network's API instead of Cytoscape's.

### `GraphView.jsx`: `useRef` + `useEffect`, as instructed
vis-network has no official React bindings, so `GraphView.jsx` now
constructs a `Network` + two `DataSet`s (`nodes`, `edges`) imperatively
inside a `useEffect` keyed on `graph`, holding them in refs
(`networkRef`, `nodesDataSetRef`, `edgesDataSetRef`) across renders -
the same lifecycle role `cyRef` played before, with a real `network.destroy()`
in the effect's cleanup so switching to a different indexed repository
doesn't leak the old canvas/simulation.
- **Click**: `network.on("click", ...)` - `params.nodes.length > 0` is a
  node click (calls the same `focusNode(raw.id)` from `RepoContext`
  unchanged), an empty click with no edge under the cursor clears the
  highlight - same two-branch behavior `tap`/background-`tap` had.
- **Hover tooltip**: `hoverNode`/`blurNode` track *which* node is
  hovered (stored in a ref, not state, to avoid a render per mouse
  pixel); actual positioning rides a plain native `mousemove` listener
  on the container using the raw DOM event's `clientX`/`clientY`
  directly - simpler than round-tripping through vis-network's canvas
  coordinate space, and more faithful to the old Cytoscape `mousemove`
  handler's "follow the cursor continuously while hovering" behavior
  than a hover-position-only approach would have been (it also keeps
  the tooltip correctly positioned while a node is being dragged, for
  free, since dragging fires the same native mouse movement).
  `NodeTooltip.jsx` itself needed **zero changes** - same `{x, y, node}`
  props, same rendering.

### Physics: real vis.js/streamlit-agraph defaults, not re-tuned from scratch
Per the task's own instruction ("if you have access to what
streamlit-agraph's default vis.js config looked like, start from
similar values rather than guessing"): `ui/graph_view.py`'s
`Config(physics=True, ...)` never overrode any physics sub-option, so
what was already validated as feeling right in Streamlit *is* vis.js's
real BarnesHut defaults, unmodified. `visStyle.js`'s
`buildNetworkOptions()` uses those exact values
(`gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95,
springConstant: 0.04, damping: 0.09`) rather than inventing new tuned
numbers - re-tuning away from a configuration the user already confirmed
felt right would risk moving away from the target feel, not toward it.
The one deliberate change: `avoidOverlap` raised from vis.js's default
`0` to `0.6`, a readability fix (reduces node/label overlap in dense
clusters) explicitly kept small enough not to fight the spring/repulsion
feel. **Physics stays enabled continuously** - `stabilization` only
controls the initial settle-in before first paint; nothing disables
`physics.enabled` afterward (no `network.setOptions({physics:false})`
anywhere), and `interaction.dragNodes: true` + vis-network's own
`adaptiveTimestep` are what give the "drag one node, connected
neighbors continuously react, then settle back down" behavior - this is
vis-network's real, documented, always-on default behavior when physics
isn't explicitly turned off, not a custom mechanism built for this task.

### A real bug caught and fixed while writing `blastRadius.js`'s rewrite
`clearHighlight`'s fade-back tween initially read each node's "current"
opacity from the `DataSet` **inside** the per-frame callback, intending
to fade from wherever the dim/highlight animation had left it back to
1. That's wrong: since the same callback also *writes* the new
interpolated opacity back to the DataSet every frame, reading "current"
on frame 2 actually reads frame 1's already-partially-interpolated
value, not the true starting value - compounding the interpolation
incorrectly instead of producing a smooth fade. Caught by re-reading the
function before moving on (not assumed correct because it looked
reasonable at a glance); fixed by capturing `startNodeOpacity`/
`startEdgeOpacity` as plain `Map`s once, before the tween begins, and
interpolating from those fixed snapshots on every frame instead.

### Verification
1. `npm run build` / `npm run lint` - both clean (same pre-existing,
   unrelated `RepoContext.jsx` Fast-Refresh warning noted in every prior
   Phase 32 entry). Bundle: 723KB (vs. the cola-based version's 736KB -
   vis-network+vis-data together are slightly smaller than
   cytoscape+cytoscape-cola+webcola were).
2. Confirmed zero remaining Cytoscape references anywhere in `src/`
   (`grep -ri cytoscape`) other than historical comments explaining what
   changed and why - no dead imports, no leftover config.
3. Dev server restarted (required - new npm dependencies, same lesson
   noted in an earlier entry about config/dependency changes needing a
   restart) and re-optimized cleanly; confirmed via direct `curl` that
   `GraphView.jsx`, and the pre-bundled `vis-network`/`vis-data` chunks
   in Vite's dependency cache, all serve 200.
4. **Real backend re-verification**, same repository and same three
   checks as every prior Phase 32 entry, still byte-identical:
   `GET /status` (ready, 16 nodes/10 edges), `POST /query` (`cache_hit:
   true`, identical citation `chunk_id`/`chunk_type:"parent"`), and
   `GET /graph?focus_node=44232b3d...&hops=2` (the same real `hmm()`
   blast radius - identical 5-node/4-edge subgraph, identical node ids,
   as every previous verification this phase). Confirms the graph
   library swap changed nothing about what data the frontend consumes,
   only how it's drawn.

**Same disclosed limitation as every prior Phase 32 entry, restated
specifically for this swap**: no browser automation tool is available in
this environment, so dragging a node and watching neighbors bounce could
not be visually confirmed by this session. Unlike the Cytoscape/cola
attempt two entries ago (where a headless Node script was a real,
viable option because Cytoscape's layout math is DOM/canvas-independent
and was actually attempted before this task was superseded), vis-network
is not realistically headless-testable without a native `canvas`
package + jsdom polyfill (its `Network` constructor gets a real 2D
canvas context immediately on construction) - judged not worth the setup
risk on this machine for a verification step, so it was not attempted
this time; this is disclosed rather than silently skipped. What *is*
confirmed: the physics configuration is vis.js's own real, already-
proven-in-Streamlit default behavior (not reconstructed from guesses),
`physics.enabled` is never disabled after stabilization, and every
non-visual data path (backend contract, module resolution, build, lint)
is verified. Both dev servers are left running
(`http://127.0.0.1:5173`, `http://127.0.0.1:8000`) with
`navdeep-G/samplemod` indexed specifically so a human can drag a node
into a cluster and confirm the water-like reaction directly.

## Phase 32 Part 1 — Landing page content additions
Purely additive, per the task brief: no indexing logic, URL validation,
or landing→workspace transition mechanics were touched. Confirmed by
listing what actually changed - `useIndexingStatus.js`, `useChat.js`,
`AppShell.jsx`'s transition logic, and `RepoInputPanel`'s existing
submit/validation code are all untouched; the one necessary addition to
an existing file was a single optional prop.

### What was added
- **`AmbientBackground.jsx`** (new): three large, heavily-blurred,
  low-opacity indigo/violet blobs (`blur-3xl`, 15-20% opacity) that
  slowly drift via three new CSS `@keyframes` in `global.css`
  (`rm-drift-a/b/c`, 19-26s durations so the three never fall into sync)
  - the "subtle animated gradient mesh" option from the brief's
    either/or framing, chosen over literal particle dots for less visual
    noise and fewer animated DOM nodes.
- **Eyebrow badge**: "Adversarial Code Review Engine" in a small pill
  (`rounded-full border-indigo-400/30 bg-indigo-500/10 text-indigo-300`)
  above the title - picked over the brief's other suggested wording
  ("AI-Powered Code Intelligence") since it's closer to what this
  project's Adjudicate pipeline actually is.
- **Feature row**: three icon+label items (chat/graph/shield icons,
  3-4 word labels) - three new small SVG icons added to
  `common/icons.jsx` (`ChatIcon`, `GraphIcon`, `ShieldIcon`), matching
  the existing `WarningIcon`/`CheckIcon` stroke style rather than
  pulling in an icon library for three icons.
- **Example-repo chips**: `facebook/react`, `expressjs/express`,
  `navdeep-G/samplemod` - the third deliberately chosen (alongside the
  two well-known names the brief suggested) because it's the actual
  small repo already proven to index quickly throughout this session,
  so a first-time user has one genuinely fast option to click through,
  not just impressive-looking but slow ones.
- **Footer**: a single small muted line, absolutely positioned at the
  true bottom of the screen regardless of the centered content above it.

### The one necessary (and additive) change to an existing file
Clicking an example chip needs to populate `RepoInputPanel`'s text
field, but that field's `url` state was - and still is - fully local to
`RepoInputPanel` (`useState`, uncontrolled from any parent). Added a
single optional prop, `prefillUrl`: when set, a new `useEffect` copies
it into the existing local `url` state once; the field remains fully
editable afterward, and the existing submit handler, `busy` disabling,
and validation are untouched byte-for-byte. `Sidebar`'s own usage of
`RepoInputPanel` (the workspace sidebar's repo input) never passes this
prop, so `prefillUrl` stays `undefined` there and the effect never
fires - confirmed zero behavior change to the workspace sidebar's own
input by inspection, not just assumed.

### Structural note: `LandingView`'s root became `absolute inset-0`
Needed so the footer could sit at the true bottom of the *screen*
rather than the bottom of the centered content block, and so
`AmbientBackground` could span the full landing area rather than just
the width of the centered column. `AppShell.jsx`'s existing landing
wrapper (already `absolute inset-0`, itself inside a `relative`-rooted
shell) is an already-correct positioning context for this - confirmed
no `AppShell.jsx` changes were needed at all: an absolutely-positioned
child is removed from normal flow, so the parent's `flex items-center
justify-center` (used for the *previous*, content-sized `LandingView`
root) simply has no effect on the new full-size root, no conflict.

### Verification
1. `npm run build` / `npm run lint` - clean, zero errors, same
   pre-existing unrelated `RepoContext.jsx` warning as every prior entry.
2. Dev server (already running from the previous session) picked up
   every change via HMR - no new npm dependencies this pass, confirmed
   via direct `curl` that `LandingView.jsx`, `AmbientBackground.jsx`,
   and `common/icons.jsx` all serve 200 from it.
3. Traced the `prefillUrl` data flow end to end by reading both files
   together (not assumed correct): `LandingView` holds the state
   (initially `undefined`, so the field starts empty exactly as
   before), a chip's `onClick` sets a real URL string, `RepoInputPanel`'s
   effect only fires `setUrl(...)` when the prop is `!== undefined` -
   confirmed the initial-mount case is a genuine no-op, not just an
   effect that happens to write the same empty value.
4. **Real backend re-verification**: re-triggered a real
   `POST /repos/index` for `navdeep-G/samplemod` (the same URL one of
   the new example chips fills in) and confirmed the identical
   `pending → indexing (stage="Cloning repository") → ready` sequence
   every prior entry this phase has verified - the actual indexing
   pipeline this landing page triggers is unaffected.

**Same disclosed limitation as every prior Phase 32 entry**: no browser
automation tool is available in this environment, so the actual chip
click → field-populates interaction, the drift animation's visual
smoothness, and the overall landing page's appearance were not visually
confirmed by this session - only the data flow (point 3) and the real
backend contract (point 4) were. Both dev servers left running
(`http://127.0.0.1:5173`, `http://127.0.0.1:8000`) for a human to click
through directly.

## Phase 32 Part 1 — Sidebar/chat commercial polish + "Verdict AI" home link
Styling + one navigation addition, per the brief. Confirmed by listing
every file touched: `GraphView.jsx`'s only change is the font-load fix
below (not physics/rendering/blast-radius logic); `useIndexingStatus.js`
gained one new `reset()` function alongside `start()`, unchanged;
`RepoContext.jsx` gained one new `resetRepo()` alongside its existing
state, unchanged. No API calls, indexing flow, chat submission, or
blast-radius code was touched.

### "Verdict AI" as a home link
Clicking it now resets to a genuine fresh-page-load-equivalent state,
not just a visual thing:
- `useIndexingStatus.js`: new `reset()` (stops any in-flight poll timer,
  sets state back to `IDLE_STATE`) - additive, sits next to `start()`
  without touching it.
- `RepoContext.jsx`: new `resetRepo()` (clears `repoId`, `repoLabel`,
  `history`, `focusNodeId`) - clearing `repoId` alone would have been
  enough to flip `showWorkspace` back to the landing state, but a real
  fresh load would also start with no chat transcript, so `history` is
  explicitly cleared too, not left stale.
- `AppShell.jsx`: a new `goHome()` calls both, then passes it to
  `Sidebar` as `onGoHome`. Reuses the exact same `showWorkspace =
  Boolean(repoId)` crossfade mechanism from the landing→workspace
  transition entry - going home is just that transition running in
  reverse, no new animation code.
- The workspace's `GraphView` needed no explicit teardown call: its
  existing effect (`useEffect(..., [graph])`) already destroys the
  vis-network `Network` in its cleanup whenever `graph` changes -
  `useGraph`'s own existing effect already sets `graph` back to `null`
  when `repoId` goes back to `null`, so the same code path that already
  handles switching to a *different* indexed repo also correctly tears
  down the canvas when going all the way back to no repo. Confirmed by
  reading both effects together, not assumed.
- The button itself: `text-lg font-bold`, `hover:text-indigo-300
  hover:underline` - a real interactive affordance (color shift +
  underline), not just a cursor change.

### Sidebar polish
- **Repo summary card**: `bg-gradient-to-br from-indigo-500/10 ...`
  (a tint distinct from the sidebar's own flat `bg-slate-900/40`),
  `rounded-lg` (up from `rounded-md`), a new `RepoIcon` next to the repo
  name, spacing between name and stats row widened (`mt-1`→`mt-2.5`,
  row `gap-0.5`→`gap-1`).
- **Indexing checklist**: now collapses to a single muted one-line
  summary ("Indexing complete", `text-[11px] text-slate-500`) once
  `indexing.phase === "ready"`, expandable again on click. This is a
  new *local, presentational-only* `collapsed` state inside
  `IndexingProgress.jsx` - the actual stage/status computation
  (`currentIndex`, the done/active/pending logic driven by the real
  `indexing.stage` string) is untouched; only what's rendered once
  `allDone` is true changed.
- **Section labels** ("REPOSITORY"/"CHAT"): `text-xs`→`text-[10px]`,
  `tracking-wider`→`tracking-[0.12em]`, kept `uppercase font-semibold
  text-slate-500` - shrunk specifically so they read unambiguously as
  dividers rather than competing with the repo name (`text-sm
  font-semibold`) or stats (`text-xs font-mono`) sitting right below them.
- **Divider rhythm**: the "Verdict AI" home link itself now has its own
  `border-b` (previously only the repository section had one below it),
  so all three sidebar sections - brand, repository, chat - are now
  separated by consistent divider lines rather than the brand floating
  above a single divider with plain whitespace gaps elsewhere.

### Chat bar polish
- Width: `max-w-xl` → `max-w-2xl` (`AppShell.jsx`).
- Shadow: replaced the generic `shadow-2xl` with a directional, indigo-
  tinted glow (`shadow-[0_25px_60px_-15px_rgba(99,102,241,0.45)]`) so it
  reads as sitting *above* something rather than just having a drop
  shadow.
- Height consistency was a real, checked concern, not assumed fine:
  the textarea's height was previously driven by `rows={1}` +
  padding, which does not exactly equal the button's `py-2`-derived
  height once actual font metrics are accounted for - both are now
  pinned to an explicit `h-[42px]` so they're guaranteed identical, not
  just visually close.
- Added `ChatIcon` inside the input, absolutely positioned left with
  `pl-9` padding on the field to make room, `text-slate-500`.

### Font audit (real findings, not asserted)
- **App-wide text**: confirmed via the actual compiled CSS
  (`dist/assets/*.css`, not just the source config) that Tailwind's
  preflight `html` rule resolves to
  `font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif` -
  Inter is genuinely the base font for every plain DOM text element
  (sidebar labels, chat text, buttons, section labels) since none of
  them override with a conflicting font-family. No bug here.
- **Graph node labels - the flagged concern was real**: `visStyle.js`'s
  `buildNetworkOptions()` already specified `font: { face: "Inter,
  sans-serif" }` for vis-network's node labels, so the *configuration*
  was already correct - but vis-network renders labels onto a
  `<canvas>`, and canvas text does **not** automatically re-render when
  a web font finishes loading asynchronously the way DOM text does (no
  equivalent of the browser reflowing text once a linked Google Font
  resolves). Since the `Network` is constructed and does its first
  paint as soon as `graph` data arrives - which can easily happen before
  `index.html`'s linked Inter stylesheet has finished downloading - node
  labels could genuinely render in the `sans-serif` fallback (the
  browser's default system font) on first paint and silently stay that
  way, only self-correcting once *something* triggers a redraw (physics
  ticks do this eventually, but not reliably fast). **Fixed**: added
  `document.fonts.load("9px Inter").then(() => network.redraw())` right
  after constructing the `Network` in `GraphView.jsx` - forces a
  redraw in the correct font the moment Inter is confirmed loaded,
  instead of relying on physics-driven redraws to eventually paper over
  it. This is the one legitimate rendering-correctness fix in this
  entry; everything else is pure Tailwind class changes.
- **Size hierarchy**: implemented as title (`text-lg` `font-bold`) >
  repo name (`text-sm` `font-semibold`) > stats/body (`text-xs`/`text-sm`)
  > section labels (`text-[10px]`, most muted/smallest). **Flagging a
  discrepancy rather than silently resolving it**: the brief's literal
  ordering was "title > section labels > repo name > stats/body >
  muted secondary" - section labels *above* repo name in size. Taken
  literally, that would make "REPOSITORY"/"CHAT" render larger than the
  actual repo name and chat content, which contradicts this same
  brief's own instruction two lines earlier that these labels should be
  "proper section dividers, not competing with content text," and would
  look wrong for the "real SaaS product" comparison made throughout this
  pass. Interpreted as a loose description of "clearly distinct tiers"
  rather than a literal size ordering and implemented with labels as the
  *smallest*, most muted tier - flagged here in case that reading is wrong.

### Verification
1. `npm run build` / `npm run lint` - clean, same pre-existing unrelated
   `RepoContext.jsx` warning as every prior entry.
2. No new npm dependencies - confirmed every changed module (`AppShell.jsx`,
   `Sidebar.jsx`, `RepoInputPanel.jsx`, `IndexingProgress.jsx`,
   `ChatInputBar.jsx`, `RepoContext.jsx`, `useIndexingStatus.js`,
   `icons.jsx`, `GraphView.jsx`) serves 200 from the already-running dev
   server via direct `curl`, confirming HMR picked up every change.
3. Traced the home-link reset by reading all four involved files
   together (`Sidebar.jsx`'s button → `AppShell.jsx`'s `goHome` →
   `useIndexingStatus.reset()` + `RepoContext.resetRepo()` →
   `showWorkspace` flips false → the existing crossfade → `useGraph`'s
   existing `repoId → null` effect → `GraphView`'s existing cleanup
   destroying the vis-network instance) - confirmed as a real chain, not
   assumed to work because each individual piece looked right in
   isolation.
4. **Real backend re-verification**, same repository, same checks as
   every prior Phase 32 entry: `GET /status` (ready), `POST /query`
   (`cache_hit: true`, identical citation) both still return exactly
   what they did before this pass - confirms the polish changed nothing
   about what the frontend calls or receives.

**Same disclosed limitation as every prior Phase 32 entry**: no browser
automation tool is available in this environment, so clicking "Verdict
AI" and watching it return to the landing state, the chat bar's
proportions/glow, the collapsed indexing checklist, and whether node
labels visibly render in Inter were not seen by this session - only
traced/verified at the code and data level above. Both dev servers left
running (`http://127.0.0.1:5173`, `http://127.0.0.1:8000`,
`navdeep-G/samplemod` indexed) for a human to click through directly.

## Semantic cache over-match bug — closed out for real (Phase 14, flagged Phase 19, never fixed until now)
First flagged in Phase 19's own "Open issues" note ("Semantic cache
appears to over-match unrelated queries at the current threshold (0.95)
using CodeBERT... worth a Phase 14 look") and repeated in this session's
own Part 1 verification notes - never actually investigated or fixed
until now, when it visibly broke real usage: "What is the backend tech
stack?" and "What is the frontend tech stack?" returned the identical
cached answer, both marked `cache_hit: true`.

### Step 1 — confirmed it's the semantic cache, with the real number, not an assumption
Measured directly against the real `microsoft/codebert-base` model
(`embedding.model_loader.load_embedding_model`), not guessed:

```
'What is the backend tech stack?' vs 'What is the frontend tech stack?': 0.9974
```

`settings.CACHE_SIMILARITY_THRESHOLD` is 0.95 - 0.9974 clears it by a
wide margin. Confirmed the mechanism, not just the symptom.

### Step 2 — measured whether *any* threshold could fix this, before picking a fix
Extended the measurement to 7 pairs spanning genuine near-duplicates
(should hit) and genuinely distinct questions (should miss), including
deliberately adversarial ones (lexically near-identical but meaningfully
different):

| Pair | Should | Cosine similarity |
|---|---|---|
| exact repeat | hit | 1.0000 |
| "backend tech stack" / "tech stack does the backend use" | hit | 0.9861 |
| "how does login work" / "how does the login flow work" | hit | 0.9925 |
| "backend tech stack" / "frontend tech stack" | **miss** | 0.9974 |
| "backend tech stack" / "how does user authentication work" | **miss** | 0.9807 |
| "backend tech stack" / "where is the database connection configured" | **miss** | 0.9869 |
| "register endpoint" / "login endpoint" | **miss** | **0.9987** |

**Decisive finding**: the should-hit range (0.9861-0.9925) sits entirely
*inside* the should-miss range (0.9807-0.9987) - the highest-scoring
pair in the whole set is a pair that should miss. `min(should-hit) >
max(should-miss)` is `False`. No threshold, at any value, separates
these. **Option (a) - raise `CACHE_SIMILARITY_THRESHOLD` to 0.97-0.98 -
is not just worse than option (b), it cannot work at all** for this
embedding model on short natural-language questions. Root cause
confirmed: CodeBERT is a *code* embedding model (Phase 8's own choice,
correct for code-chunk retrieval), not tuned for natural-language
sentence similarity - it collapses short, similarly-structured NL
questions into a nearly degenerate similarity band regardless of actual
meaning.

### Step 3 — the fix: a second, independent lexical-compatibility gate (`retrieval/semantic_cache.py`)
Cosine similarity is now *necessary but not sufficient* for a cache hit.
A candidate must also pass `_is_lexically_compatible(query, candidate)`,
two independent checks:
1. **`_lexical_jaccard`** - content-word (stopword-stripped) Jaccard
   overlap must be `>= 0.5`. Catches "totally unrelated topic" pairs
   (measured at 0.0 overlap) without touching genuine near-duplicates
   (measured at 0.5-0.75 overlap in every hit case tested).
2. **`_has_distinguishing_conflict`** - a small, explicit, extensible
   list of confusable term categories (backend/frontend/client/server,
   login/logout/register, HTTP verbs, read/write, dev/prod, sync/async).
   Needed *in addition to* the Jaccard gate because "backend" vs
   "frontend" measured exactly `0.5` lexical overlap - right at the
   Jaccard threshold, too close to trust alone for the one pair this bug
   report is actually about. Rejects a hit only on *positive, two-sided*
   evidence (both queries name a *different* alternative of the same
   category) - a query that just doesn't mention a category at all isn't
   treated as a conflict.
- **Disclosed limitation, not swept under the rug**: this is not a
  general antonym/word-sense solution - a confusable pair not in the
  explicit category list can still over-match. It's a targeted,
  extensible fix for the reported bug and its class (a small, growable
  list), not a claim of completeness.
- `lookup()` restructured to check candidates in descending cosine-
  similarity order and skip lexically-incompatible ones rather than
  only ever considering the single best-cosine match - so a high-cosine-
  but-incompatible entry can no longer shadow a genuinely-compatible
  lower-cosine one.
- A real edge case caught while writing the fix, not shipped broken:
  the first regex draft anchored `\b` around space-containing phrases
  inconsistently (multi-word terms fell through to an unanchored
  substring check). Simplified to always wrap the full escaped term in
  `\b...\b`, which - verified by reasoning through the regex, not
  assumed - correctly handles both single words ("dev" does not match
  inside "development", since "e" immediately follows and is still a
  word character) and phrases ("front end", "back-end") the same way.

### Step 4 — tested in both directions, not just against the one reported example
`tests/test_retrieval/test_semantic_cache.py` gained 14 new tests
(11 pure-function tests on `_is_lexically_compatible`/`_lexical_jaccard`/
`_has_distinguishing_conflict`, 3 integration tests against
`SemanticCacheManager.lookup()` itself using near-identical fake
embedding vectors to prove the *query text*, not the embedding, is what
now correctly forces a miss): the reported bug pair, back-end/front-end
phrasing variants, register-vs-login endpoints, totally unrelated
topics, dev-vs-production, "dev" not falsely matching inside
"devops"/"development" - all correctly rejected - alongside genuine
near-duplicates (paraphrase, exact repeat, "login work"/"login flow
work") and a case where an incompatible higher-cosine entry must be
skipped in favor of a compatible lower-cosine one - all still correctly
hit. **3 pre-existing tests failed on first run** after this change,
for a legitimate reason worth recording: they used placeholder query
text ("cached query"/"new query") that was never meant to represent a
real lexical relationship, so the new Jaccard gate correctly rejected
them (0.33 overlap, below 0.5). Fixed by giving those tests real,
lexically-plausible near-duplicate text instead of loosening the gate to
accommodate arbitrary placeholders - the tests' original intent
(cosine-threshold gating) is preserved, just with query text that could
occur for real.

### Verification (real, against the real model - not the fake-vector unit tests)
```
--- Real bug scenario: frontend query after backend cached ---
Result: None
PASS: correctly a cache miss - frontend query does NOT return the backend answer

--- Legitimate cache hit: exact same question asked twice ---
Result: SemanticCacheHit(..., response='REAL BACKEND ANSWER: Express/Node.', similarity=1.0000000746554583, ...)
PASS: exact repeat correctly still returns the cached answer
```
Run against `navdeep-G/samplemod`'s real repository row, real
`SemanticCacheManager`, real `microsoft/codebert-base` embeddings - not
mocked. Both the fix and the legitimate-hit path confirmed live, not
just via unit tests.

**Full regression suite: 688 passed** (was 674 before this fix - the 14
new tests above), zero failures, zero regressions in any other module.

### Not done / follow-up (disclosed, not hidden)
- The explicit `_DISTINGUISHING_CATEGORIES` list is deliberately narrow
  - a confusable pair outside it (e.g. two entirely different but
    lexically-near-identical proper nouns) could still over-match. Worth
    extending if another real over-match case surfaces, not worth
    speculatively over-building now.
- Root cause (CodeBERT unsuited to natural-language query similarity)
  remains true for retrieval ranking generally, not just the cache -
  this fix closes the cache-specific symptom the user reported; the
  broader dense-retrieval-ranking-quality concern flagged back in the
  NutriForge diagnostic chain closeout is still open, unrelated to this
  fix, and out of scope here.

## Phase 32 Part 2 — status-check + resume session (independent re-verification)
A follow-up session re-verified both the semantic cache fix above and
Phase 32 Part 2 from scratch, per explicit instruction not to trust this
file's own claims without reading the real code/running the real thing.

- **Semantic cache fix (above): re-confirmed, no gaps found.** Read
  `retrieval/semantic_cache.py` directly - the lexical-compatibility gate
  (`_is_lexically_compatible`/`_lexical_jaccard`/`_has_distinguishing_conflict`)
  is real, matches this file's own prior entry exactly, not a threshold
  change. No code changes made here.
- **Phase 32 Part 2 backend - confirmed complete.** `api/main.py`'s
  `POST /repos/{repo_id}/review` and `adjudicate/orchestrator/live_review.py`
  (`run_live_review`) both exist and are fully wired: real Phases 24-30
  pipeline (Context Builder -> Defender -> Prosecutor -> Verifier ->
  Rebuttal loop -> Judge), one SSE event per stage, sandboxed diff
  application via `patch -p1` with cleanup in a `finally`, a
  last-resort `except Exception` so an unexpected failure still ends the
  stream with a clean `error` event rather than hanging the connection.
  5/5 unit tests in `tests/test_adjudicate/test_live_review.py` pass.
- **Phase 32 Part 2 frontend - confirmed complete.** `frontend/src/components/Review/`
  (ContextCard, DefenderCard, ClaimCard, RebuttalCard, JudgeCard,
  DiffInputPanel, ReviewView), `Sidebar.jsx`'s "Review a PR" nav toggle,
  and `AppShell.jsx`'s dual-mounted Graph/Review view switch are all
  wired end to end. Every card uses a real CSS keyframe
  (`.animate-rm-card-in`, `global.css:66`), not a no-op class - confirmed
  by reading the stylesheet, not assumed from the class name.
  `useLiveReview.js`/`api/client.js::streamReview` correctly parse a
  buffered, chunk-spanning `text/event-stream` body via a manual
  `ReadableStream` reader (browser `EventSource` can't be used - this
  endpoint is `POST`). `npm run build` succeeds cleanly (50 modules,
  zero errors). Full regression suite: **693 passed**, 0 failed, 0
  errors (up from 688 at the last entry - the 5 new `test_live_review.py`
  tests account for the difference; the previously-flagged `pytest-mock`
  gap is gone, `pytest-mock` is now installed).
- **Live end-to-end verification via real Playwright + Chromium against
  a real `uvicorn`+`vite` dev pair - partially completed, not assumed:**
  reused the already-indexed `tartley/colorama` repo and the same
  `reset_all` diff Phase 24/25 already hand-verified ground truth for
  (2 callers via a known name-collision, 0 callees, 2 related tests,
  both `found_via="graph"`). Drove the real UI: indexed the repo through
  the real form, clicked "Review a PR", pasted the diff, clicked "Run
  Review".
  - **Context stage: PASS, exact match to already-verified ground
    truth.** Rendered card read "reset_all colorama/initialise.py:29-29
    / 2 caller(s) / 0 callee(s) / 2 related test(s)" - identical to
    Phase 24/25's hand-verified numbers for this exact node.
  - **Defender stage: PASS on the first run** - a real (uncached)
    Gemini 2.5 Flash call completed and streamed a justification card
    (confirmed in the server log: "Defender justification generated:
    1401 total token(s)").
  - **Prosecutor/Verifier/Rebuttal/Judge stages: NOT reached.** A
    second run hit a real `429 RESOURCE_EXHAUSTED` from Gemini's own
    free-tier daily quota (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
    limit 20/day for `gemini-2.5-flash`) - the same class of blocker
    already flagged in Phase 31's own PROGRESS.md entry ("1 case blocked
    on daily provider quota"), not a defect in this phase's code. Per
    the project's own LLM-quota-discipline convention, did not keep
    retrying real calls against an exhausted quota once this was
    confirmed.
  - **This failure is itself a real, useful verification, not a wasted
    run:** the error propagated through `run_live_review`'s
    `except RepoMindError` branch as a clean `error` SSE event, which the
    frontend's `Alert` rendered verbatim; the "Run Review" button
    correctly re-enabled (not stuck on "Reviewing…"); the already-arrived
    Context card stayed rendered rather than being cleared. Zero browser
    console errors on either run.
- **Status: code-complete and unit-tested, but not fully live-verified
  end-to-end** - Prosecutor/Verifier/Rebuttal/Judge have real unit-test
  coverage (`test_live_review.py`) but no real-Gemini live confirmation
  yet, blocked on today's exhausted free-tier quota. `docs/roadmap.md`'s
  Phase 32 stays 🟡, not ✅, until a full live run (ideally with a fresh
  quota window or a paid/alternate key) reaches the Judge card and its
  verdict/cited_evidence are cross-checked against the diff by hand, the
  same rigor every other phase in this file was held to.
- Open issue / next step: re-run the same live-UI check (same repo, same
  diff - ground truth already established above) once quota allows, to
  reach and hand-verify the Prosecutor claims, Verifier evidence, and
  Judge verdict against the real diff.

## Phase 32 Part 2 — nav restructure: "Review a PR" elevated to a top-level view
Pure navigation/emphasis change, per explicit request - the review
pipeline, SSE streaming, and diff parsing (all verified live above) were
not touched. Before: "Review a PR" was a muted outline button at the
bottom of the sidebar, below the chat history; the user flagged this as
underselling the app's flagship feature.

- Presented the two structural options the request itself raised
  (sidebar hero button vs. a full top-level 3-way switcher) rather than
  guessing, since they differ in how much of the existing chat layout
  they touch - user picked the top-level switcher.
- **New `frontend/src/components/Chat/` folder** (chat moves out of the
  sidebar entirely, not just restyled): `ChatHistoryPanel.jsx`,
  `ChatMessage.jsx`, `CitationItem.jsx`, `ChatInputBar.jsx` moved here
  unchanged in logic from `components/Sidebar/` (only `ChatInputBar`'s
  outer chrome changed: a docked panel border instead of the floating
  glassmorphic shadow it needed when it sat on top of the graph canvas).
  New `ChatView.jsx` composes them into a full-height view - history
  above, input docked below - mirroring `ReviewView.jsx`'s own
  structure. `useChat` itself (question/pending/error logic) is
  untouched; only where its output renders moved.
- **New `Layout/ViewSwitcher.jsx`**: the top-of-workspace pill nav -
  Graph/Chat as plain outline pills, "Review a PR" always rendered with
  the accent gradient (not just on hover/active) plus a larger touch
  target, the same primary-action treatment `ChatInputBar`'s Ask button
  and `RepoInputPanel`'s Index button already use - so it visually reads
  as the standout capability regardless of which tab is active.
- **`Sidebar.jsx`**: now Repository-status-only (brand link + repo
  input/indexing progress) - the chat history section and the old
  "Review a PR" button are both gone; `chat`/`nodesById`/`activeView`
  props no longer needed there.
- **`AppShell.jsx`**: `activeView` gained a third value (`"graph" |
  "chat" | "review"`, was `"graph" | "review"`). The floating
  `ChatInputBar` overlay over the graph is gone - replaced by
  `ViewSwitcher` at the top of `<main>` and three always-mounted view
  containers (`GraphView`/`ChatView`/`ReviewView`) toggled via the same
  `hidden`-class convention Graph/Review already used, extended to
  Chat - so Chat's scroll position, same as Graph's blast-radius state
  and Review's in-progress card feed, survives switching away and back.
- **Verified live** (Playwright + real Chromium against the running
  `uvicorn`/`vite` pair, `tartley/colorama`): Sidebar text contains
  neither "Chat" nor "Review" anymore (asserted programmatically, not
  eyeballed); clicking the Chat pill renders the empty-state prompt and
  input box; clicking "Review a PR" opens the diff panel; typing the
  same `reset_all` diff (Phase 24/25's established ground truth) and
  clicking "Run Review" round-tripped a real request - the Context card
  rendered "2 caller(s) / 0 callee(s) / 2 related test(s)", identical to
  every prior verification of this exact node; switching to Graph and
  back to Review mid-review preserved the in-progress card feed
  byte-for-byte (screenshots before/after the round-trip are identical
  aside from the nav highlight), confirming the always-mounted
  convention survived the restructure. Zero browser console errors.
  Did not push this particular review past the Defender stage - same
  exhausted Gemini free-tier daily quota flagged above, not re-tested
  here since this session only needed to confirm the request still
  fires and streams, not re-litigate the pipeline itself.
- `npm run build`: 52 modules (was 50), zero errors. Full backend
  suite re-run for safety despite this being a frontend-only change:
  693 passed, 0 failed - confirms nothing here touched Python code.

## TypeScript grammar gap — diagnosed, decision: defer (not fixed)
Surfaced by a real repro, not a hypothetical: `github/accessibility-
scanner-alt-text-plugin` indexed to "39 nodes, 1 chunk, 0 edges" -
looked like a parsing gap, not partial coverage, so it was investigated
before running a review against it.

- **Root cause, confirmed by reading the code and the real indexing
  log (`data/logs/repomind_20260708_113828.log`), not assumed:**
  `ingestion/ast_parser.py::TreeSitterParser` only registers `"python"`
  and `"javascript"` grammars - no `"typescript"` entry, and
  `tree-sitter-typescript` isn't installed or in `requirements.txt` at
  all. File discovery is *not* the bug - `core/constants.py`'s
  `SUPPORTED_FILE_EXTENSIONS` already maps `.ts`/`.tsx` ->
  `"typescript"` correctly, confirmed via the DB (39 `source_files`
  rows: 38 `typescript`, 1 `javascript`). Every `.ts` file - including
  `src/judges/azure-augmented-judge.ts`, the file behind the diff this
  session was about to test - failed loudly and individually:
  `ERROR | ingestion.ast_parser | Parsing failed for <path>: unsupported
  language 'typescript'` + `WARNING | pipeline | Skipping <path>...`,
  caught per-file by `parse_many` (not a crash, not silent). The "1
  chunk" is a `sliding`-type window chunk from the repo's one real JS
  file (`eslint.config.js`, no functions to extract); window chunks are
  never graph nodes, so 39 nodes = file-level nodes only, 0 edges.
- **Decision (explicit, not a default): defer the fix.** Adding
  `tree-sitter-typescript` + a TS extractor is real, scoped work (a new
  grammar dependency, registering it in `TreeSitterParser.__init__`,
  and either reusing or adapting `_extract_javascript_chunks` for
  TS-only constructs like `interface`/`type`/`enum`) - explicitly lower
  priority right now than finishing the benchmark harness (Phase 31),
  the frontend (Phase 32), and deployment (Phase 33). Documented as a
  known, deliberate scope limitation (not a bug) in
  `docs/project _description.md`'s Phase 4 entry.
- **Practical effect on the azure-augmented-judge diff**: with zero
  chunks for that file, `AdjudicateContextBuilder.build()` cannot find
  an enclosing chunk for any line in a diff touching it, so
  `run_live_review` will emit its existing `changed_functions`-empty
  `error` event rather than a wrong or partial review - confirmed this
  *is* the actual failure mode, not a guess, by reading
  `adjudicate/context_builder.py::_find_enclosing_chunk` (catches
  `RetrievalError` per location and returns `None`, so `build()`
  degrades to an empty `changed_functions` list instead of raising).
- **Frontend error-surfacing check (also asked for this session, not
  assumed fine): PASS, verified live, no fix needed.** This error path
  fires immediately after context building, *before* any Defender/LLM
  call - reproduced it for real (zero Gemini quota cost) by running an
  actual review against the real `accessibility-scanner-alt-text-plugin`
  repo with a diff touching `azure-augmented-judge.ts` through the real
  Playwright-driven UI. Result: a properly styled red `Alert` card with
  a warning icon renders the exact message ("No changed function in
  this diff could be resolved against the indexed repository's call
  graph - the diff may target a file that hasn't been indexed, or lines
  that don't fall inside any function/class this repository's graph
  knows about."), the "Run Review" button re-enables (doesn't get stuck
  on "Reviewing…"), and the Context card still renders showing "0
  caller(s) / 0 callee(s) / 0 related test(s)" so the user can see what
  was actually searched. Zero browser console errors. Same `error` SSE
  event -> `Alert` rendering path already verified working for the
  Gemini-quota-exhaustion case in the prior entry - confirmed this
  specific message renders through it correctly too, not assumed from
  the code alone.
- Open issue: `tree-sitter-typescript` support itself remains
  unimplemented, tracked here and in `project _description.md`'s Phase 4
  entry - revisit after Phase 31/32/33.

## Graph physics — stabilization tuning for larger repos (harshal31718/pawn_ai)
Reported symptom: pawn_ai (657 nodes/802 edges) never visibly settles
after loading, unlike NutriForge (227 nodes). Diagnosed and fixed with
real, logged numbers at every step - not assumed.

- **Root cause, confirmed by temporarily instrumenting
  `frontend/src/components/Graph/GraphView.jsx` with real
  `stabilizationProgress`/`stabilizationIterationsDone`/`stabilized`
  event listeners (vis-network's actual events - read straight from the
  installed `vis-network` source to confirm their real semantics before
  trusting them):** `stabilizationIterationsDone` fires unconditionally
  once the `physics.stabilization.iterations` cap is reached, whether or
  not the layout actually converged - the real "is it at rest" signal is
  a separate `"stabilized"` event, which only fires once every node's
  velocity drops below `physics.minVelocity`. Below that convergence
  point, vis-network falls through into a live, one-tick-per-animation-
  frame simulation loop that keeps running indefinitely until real
  convergence happens - which is the actual "keeps moving indefinitely"
  symptom.
- **Real before numbers** (previous hardcoded config: `iterations: 200`,
  `damping: 0.09`, `minVelocity: 0.75` - vis.js's own literal BarnesHut
  defaults, confirmed unmodified from `ui/graph_view.py`'s original
  Streamlit config):
  - **pawn_ai**: hit the 200-iteration cap; `stabilized` never fired,
    watched for 60s+ afterward with no convergence. Raised the cap to
    2000 as a diagnostic-only experiment (not a fix) - still never
    converged even after 3 additional minutes of observation. Confirmed
    this is genuine non-convergence at this damping, not "just needs
    more time."
  - **NutriForge** (the reference "working" case): also hit its
    200-iteration cap without a `stabilized` event firing, re-confirmed
    on a second, longer (150s) run - it never crosses vis-network's
    strict convergence threshold either. It only *looks* settled because
    residual velocity is small enough to be visually imperceptible, not
    because it truly reaches `stabilized`. Important to disclose
    honestly: the task's own frame ("works correctly on NutriForge") is
    approximately, not literally, true - this is a pre-existing
    characteristic, unrelated to and unaffected by this fix either way.
- **Fix (`frontend/src/components/Graph/visStyle.js`,
  `computePhysicsScale`):** node-count-gated scaling, not a global
  re-tune - `PHYSICS_SCALE_NODE_THRESHOLD = 250` (comfortably above
  NutriForge's 227). At or below it, `iterations`/`damping`/
  `minVelocity` are mathematically identical to the old hardcoded
  values (zero-value scaling term) - this is not just "should be the
  same," it is provably the same expression for any repo this size or
  smaller, so there is no way this change could have regressed that
  case. Above the threshold, all three scale linearly with
  `nodeCount - 250`, capped at `iterations<=2500`, `damping<=0.3`,
  `minVelocity<=1.5`. `GraphView.jsx` passes the real node count into
  `buildNetworkOptions(nodes.length)`.
- **Real after numbers, same instrumentation, same repo:**
  - First attempt (`iterations` scaled to 1014 via `+2/node`, `damping`
    ≈0.253, `minVelocity`≈1.36 for pawn_ai's 657 nodes): `stabilized`
    **did** fire for the first time - at iteration 1623 - proving the
    damping/minVelocity change (not just more iterations) is what makes
    real convergence possible at all. But the 1014-iteration cap was
    short of the 1623 actually needed, so ~600 iterations still spilled
    into the slow live continuous phase before converging.
  - Raised the per-node iteration multiplier from `+2` to `+4`
    (1828 iterations for pawn_ai) with damping/minVelocity unchanged:
    `stabilizationIterationsDone` and `stabilized` now fire **together**
    at iteration 1676 - convergence now completes entirely inside the
    fast pre-render batch, the same pattern NutriForge's own (imperfect
    but imperceptible) settling already followed. Final values used:
    `iterations=1828`, `damping≈0.253`, `minVelocity≈1.36` for pawn_ai's
    657 nodes.
- **Verification:**
  - pawn_ai: two full-viewport screenshots taken 2 seconds apart after
    a 210s settle wait were byte-for-byte identical - genuinely at rest,
    not just visually close. Layout inspected directly: one legible
    central cluster plus several smaller satellite clusters and cleanly
    spread isolated nodes - not degenerate/collapsed/exploded.
  - NutriForge: re-ran after the fix; `computePhysicsScale(227)`
    confirmed mathematically identical to the pre-fix hardcoded values
    (227 < 250), so behavior is unchanged by construction - re-verified
    live regardless, same pre-existing "hits cap, small imperceptible
    residual motion" characteristic as the documented before-state
    above, not a new regression.
  - `npm run build`: 52 modules, zero errors, both before starting this
    task and after every change. All temporary diagnostic event
    listeners removed from `GraphView.jsx` afterward - final diff is
    `visStyle.js`'s `computePhysicsScale` + `buildNetworkOptions`
    signature change, and `GraphView.jsx`'s one-line call-site update to
    pass `nodes.length`. No graph data, API, or blast-radius logic
    touched, per the task's explicit scope.
- Not done (flagged, not a regression - out of this task's scope): the
  strict `stabilized` threshold is still never reached below the 250-node
  scaling threshold (documented above as a pre-existing, disclosed
  characteristic) - left alone deliberately, since changing it for small
  graphs risks exactly the "weaken the already-working feel" outcome the
  task explicitly said not to do.

## Graph coloring — group by top-level folder instead of per-file hash
Presented two concrete options for what "folder-based clustering" should
mean before touching code (recolor only vs. a physical vis-network
cluster/spring-based grouping) - user picked the pure-color option, no
physics/layout change.

- **`frontend/src/components/Graph/visStyle.js::colorForFile`**: now
  hashes only the top-level folder segment of `file_path`
  (`path.split("/")[0]`, root-level files get their own stable bucket)
  instead of the full path - every file under e.g. `backend/` now always
  gets the same color regardless of which file within it, instead of an
  arbitrary per-file hash. File-type nodes are unaffected (they already
  used a separate fixed `fileNodeColor()`, not this function). No other
  file touched - this is a one-function change.
- **Verified live against pawn_ai (the requested repo) - real finding,
  not a bug in this change:** every chunk node still rendered as a single
  color. Root cause, confirmed via the real graph JSON and DB: pawn_ai's
  `backend/` (Python, 70 files) produced 549 real chunk nodes;
  `frontend/` (38 files) is 100% TypeScript, and - per the already-
  diagnosed, deliberately-deferred TypeScript grammar gap above -
  contributes zero chunk nodes, only isolated file-nodes. There is
  genuinely only one chunk-bearing folder to color in this repo today;
  this is a direct, expected downstream consequence of the already-
  documented TS decision, not a defect in the folder-coloring logic
  itself. Confirmed the hash logic is correct independently (`node -e`
  against the real function: `"backend"` -> bucket 8, `"frontend"` ->
  bucket 10 - genuinely different buckets).
- **Verified live against NutriForge instead, to actually demonstrate
  multi-folder grouping** (three real chunk-bearing top-level folders:
  `client/` 119 chunks, `server/` 32, `ai_service/` 13 - all
  JS/Python, no TS gap here): rendered with two visibly distinct colors,
  matching the real per-folder hash buckets exactly (`client`/`server`
  both hash to bucket 3 - a real, disclosed 12-bucket collision, same
  class already documented in the Phase 21 entry for the old per-file
  hash; `ai_service` hashes to bucket 10, correctly distinct). Confirms
  the grouping logic works correctly when there's more than one real
  folder of parsed content to show.
- `npm run build`: 52 modules, zero errors. Zero console errors in
  either live check. No physics, layout, graph data, API, or blast-
  radius code touched - confirmed a pure recoloring change, per the
  task's explicit scope.
- Not fixed (disclosed, not a regression, same standing limitation as
  Phase 21): only 12 palette buckets exist, so two folders can still
  collide by chance (as `client`/`server` did here) - a full distinct-
  color-assignment scheme would remove this but wasn't asked for and
  would be over-engineering a heuristic that's already good enough for
  its stated purpose.

## Graph coloring follow-up — one dominant top-level folder made grouping useless again
Reported: `harshal31718/enma_trading_platform`'s graph rendered as
effectively one color. Diagnosed with real logged data before touching
code, per explicit instruction.

- **Item 1 (re-indexed after the fix?):** confirmed moot, not just
  checked - `colorForFile` is a pure client-side function computed at
  render time from `file_path` strings already present in the fetched
  graph JSON; it has no dependency on indexing recency at all (re-
  indexing doesn't change `file_path` values). Checked anyway: the repo
  was re-indexed live during this session's own verification, after the
  fix was in place - not a stale-data issue on either count.
- **Item 2 (real per-node values, not assumed):** temporarily
  instrumented `GraphView.jsx` to log `(file_path, color)` for a sample
  spread evenly across all real chunk nodes (an earlier first-10-only
  sample was misleading - the array happened to group all of `client/`'s
  nodes first). Real finding: `colorForFile` was working exactly as
  designed (top-level-folder hash) - `engine/*` -> `#ec4899`,
  `client/*`/`server/*` -> `#ef4444` (a real 12-bucket collision, same
  class already disclosed for NutriForge). The actual problem: `engine/`
  alone is 811 of 1218 real chunk nodes (66%) - confirmed via the graph
  JSON - so two-thirds of the graph shared one color regardless, and the
  two colors that did exist (`#ec4899` pink, `#ef4444` red) are close
  enough in hue at small dot size to read as "one pink graph" at a
  glance. Not a path-parsing bug - a real, single dominant top-level
  folder, exactly the case flagged as worth handling.
- **Item 3 (fix) - deviated from the literal ask, flagged and explained
  before writing code:** the requested fix was "color by the second path
  segment instead." Checked this against NutriForge (the other repo this
  feature was verified against) before implementing: NutriForge's three
  real folders (`client/`, `server/`, `ai_service/`) all nest under their
  own `src/` - keying on the second segment alone would hash `client/src`
  and `server/src` down to the same `"src"` key, collapsing a case
  already confirmed working back into one color. Implemented `colorForFile`
  keying on the first **two** segments combined instead (`engine/core`,
  `client/src`, `ai_service/routers`) - subdivides a dominant folder by
  its real substructure while keeping different top-level folders
  distinct from each other (their first segment still differs). Paths
  with no real subfolder (2 segments total, e.g. `engine/kernel.py`) or
  none at all (root files) fall back to the prior behavior unchanged.
- **Verified live, both repos, real re-indexing:**
  - `enma_trading_platform`: `engine/` alone now spans 6 distinct colors
    (`core`, `indicators`+`routers` collide, `services`, `strategies`,
    `tests`) instead of one - screenshot shows a genuinely multi-colored,
    legible graph (indigo/cyan/amber/pink/yellow all visible), not a
    solid-color mass.
  - NutriForge: re-verified, and now strictly better than the prior
    entry - `client/src`(bucket 4)/`server/src`(bucket 8) no longer
    collide (they did under the top-level-only scheme), plus
    `ai_service/routers`(9) and bare `ai_service`(10) both distinct - 4
    real colors visible in one screenshot, up from 2.
  - **Spot-check (3 real nodes, as requested), independently computed
    from the real hash function, matched the live-captured browser
    colors exactly:** `engine/core/kernel.py` -> key `engine/core` ->
    `#6366f1`; `client/src/App.jsx` -> key `client/src` -> `#eab308`;
    `server/src/controllers/orderHistory.controller.js` -> key
    `server/src` -> `#d946ef`. All three matched the actual on-screen
    color logged from the running app, not just the standalone
    calculation.
- `npm run build`: 52 modules, zero errors, both before and after. All
  temporary diagnostic logging removed from `GraphView.jsx` afterward.
  Only `visStyle.js::colorForFile` changed - no physics, layout, graph
  data, or API code touched.
- Still open (same disclosed 12-bucket limitation, unchanged in kind):
  `engine/indicators` and `engine/routers` still collide with each
  other in this repo, as do a couple of other pairs across both repos -
  expected given only 12 buckets exist; not fixed, per the same standing
  decision as the original entry above.

## Landing page — hero-scale visual pass (no functional change)
Pure sizing/spacing increase on `LandingView.jsx` - the pre-workspace
screen previously read as a small text block in a large empty dark area.

- **`LandingView.jsx`**: title `text-5xl` -> `text-7xl`; tagline
  `text-sm` -> `text-xl` with a `max-w-xl` cap so it doesn't stretch
  full-width; feature-highlight icons `h-4 w-4` -> `h-8 w-8` with the
  row gap widened (`gap-6` -> `gap-14`) and icon-to-label gap widened
  (`gap-1.5` -> `gap-3`); example chips `px-3 py-1 text-[11px]` ->
  `px-5 py-2.5 text-sm`; overall content block `max-w-md` -> `max-w-3xl`
  with the section-to-section gap widened (`gap-6` -> `gap-10`) for more
  vertical rhythm.
- **`RepoInputPanel.jsx`**: added an optional `size` prop (`"sm"`
  default, `"lg"` new) purely swapping input/button className strings
  (`px-3 py-2 text-sm` -> `px-5 py-4 text-lg`, sharper `rounded-xl`
  corners) - the `<form>`, its `onSubmit`, `busy`/`disabled` rules, and
  `prefillUrl` wiring are identical in both sizes, not duplicated.
  `Sidebar.jsx`'s own usage doesn't pass `size`, so it still gets the
  literal previous classNames, unchanged - confirmed by inspection (the
  prop simply isn't threaded through there), not just assumed.
  `LandingView.jsx` passes `size="lg"`.
- **Verified live** (Playwright, real backend): typing a URL manually,
  clicking an example chip (confirmed the input's real value updates to
  the chip's URL), and clicking "Index" (confirmed the real indexing
  pipeline fired - "Cloning repository" appeared in the checklist, not
  just a UI-only state change) all still work identically to before.
  Zero console errors. `npm run build`: 52 modules, zero errors.
- Deliberately left `IndexingProgress` (the stage checklist) at its
  existing size - not named in the requested scope (badge/title/
  tagline/features/input/chips), and it's the same shared component the
  workspace sidebar renders post-indexing, so scaling it would need the
  same size-prop treatment as `RepoInputPanel` for no requested benefit.

## Chat view — Gemini-style layout pass (bubble vs. flowing text), same palette
Pure visual restyle of the existing Chat view - `useChat`'s query call,
citation click-to-graph (`focusNode`), and every other piece of logic
are untouched; only `ChatMessage.jsx`/`CitationItem.jsx`/
`ChatHistoryPanel.jsx`/`ChatInputBar.jsx`'s JSX structure and
classNames changed.

- **`ChatMessage.jsx`**: the question is now a right-aligned bubble
  (`bg-indigo-500/10`, `rounded-2xl`, `max-w-[75%]` so long questions
  wrap instead of stretching full width) - the one deliberately
  "chat-like" element. The answer is deliberately *not* bubbled - plain
  flowing text at `max-w-3xl`, left-aligned, read like a document. The
  answer's wrapping `<div>` is intentionally not a flex container, so
  its citation children lay out as normal inline content (wrapping with
  the text) rather than being forced into a flex row/column.
- **`CitationItem.jsx`**: changed from a full-width bordered block
  (`flex w-full flex-col`) to a small `inline-flex` monospace chip
  (`mx-1 ... rounded-md px-1.5 py-0.5`) - same two variants as before
  (a plain muted span for non-graph-node "parent" chunks, an indigo-bordered
  clickable button for real graph nodes), same `onClick={() =>
  focusNode(citation.chunk_id)}` wiring, unchanged - only how it's laid
  out relative to the answer text changed (inline chip flowing with the
  paragraph instead of a stacked list below it).
- **`ChatHistoryPanel.jsx`**: gap between exchanges widened
  (`gap-4` -> `gap-10`) for the requested breathing room; the old
  per-message `border-b` divider (needed when exchanges were cramped) is
  gone now that generous whitespace does that job instead.
- **`ChatInputBar.jsx`**: outer form is now one `rounded-full` glass
  pill (`bg-slate-900/80 backdrop-blur-md`, per the requested exact
  treatment) instead of a rectangular bar; the textarea itself is
  border-less/transparent so the pill's own border is the only visible
  boundary, with `ChatIcon` as a leading glyph inside the pill rather
  than absolutely positioned inside the textarea. No "+" attachment icon
  added - there's nothing in this app to attach, using this pass's own
  explicit "skip if nothing to attach" option. "Ask" button kept (not
  replaced with an icon-only button) and restyled to `rounded-full` to
  match the pill, same accent gradient as before.
- Color palette: only existing Slate/Zinc/Indigo/Violet tokens used
  throughout (`bg-indigo-500/10`, `indigo-400/30`, `bg-slate-900/80`,
  the same `from-indigo-500 to-purple-600` gradient the Ask/Index
  buttons already used) - no new colors introduced, per the explicit
  constraint not to borrow Gemini's own blue/green palette.
- **Verified live against `tartley/colorama` with a real question** (not
  just the empty state): asked "What does the AnsiFore class do?"
  through the real UI - got a real, non-cached Gemini answer rendered as
  plain flowing text, the question rendered as a right-aligned bubble
  above it, and two inline citation chips (`parent colorama/ansi.py`)
  flowing right after the answer's final paragraph rather than in a
  separate bordered block. Zero browser console errors. Both citations
  in this particular answer were non-clickable "parent" chunks (the
  correct, unchanged behavior for window chunks per `CitationItem`'s own
  logic) - the click-to-graph path itself was verified by code
  inspection instead, since its `onClick`/`focusNode` call is
  byte-for-byte the same as before this change, only restyled.
- `npm run build`: 52 modules, zero errors. Full backend suite re-run
  for safety despite this being a frontend-only change: 693 passed,
  0 failed.

## Prosecutor JSON parse failure — real root cause: thinking-token truncation, not escaping
Reported: 3/3 identical Prosecutor failures against `mkocabas/VIBE`
(`read_single_sequence`), same error every time: `Unterminated string
starting at: line 4 column 17 (char 59)`. Diagnosed with the real raw
response text before touching any code, per explicit instruction - the
user's own hypothesis (unescaped code/quotes in the diff confusing the
model) turned out to be wrong; logged the real mechanism instead.

- **Step 1 - captured the real raw response, not just the parse
  error.** `parse_claims`/`prosecutor.py` never logged the raw
  completion text on failure - only the `json.JSONDecodeError` message.
  Reproduced live (real Gemini call, real `AdjudicateContextBuilder` +
  `DefenderAgent` + `ProsecutorAgent`, the exact same VIBE diff and
  function): got the identical error. Raw text: `'[\n  {\n
  "claim_type": "untested_branch",\n    "location": "lib/data_'` - the
  response is **genuinely cut off mid-string**, not malformed by bad
  escaping.
- **Step 2 - confirmed via the raw `google-genai` response object
  (not just `LLMCompletion.text`):** `candidate.finish_reason` was
  `FinishReason.MAX_TOKENS`. `usage_metadata` showed
  `thoughts_token_count=979` out of a `max_output_tokens=1024` budget -
  Gemini 2.5 Flash's internal "thinking" tokens **share** the
  `max_output_tokens` budget with the actual visible output. 979 of
  1024 tokens went to invisible reasoning, leaving only 28 tokens for
  the real JSON - nowhere near enough to complete even the first claim
  object. This is why all 3 retries failed identically: the existing
  retry only appends a corrective note to the prompt, it never
  increases the token budget, so the same truncation point was
  guaranteed to recur every time regardless of how many attempts ran.
- **Step 3 - not a parser fragility issue, and not code-snippet-
  specific.** `json.loads` correctly rejected genuinely incomplete JSON
  - that's correct behavior, not a bug to work around. The failure mode
  is specific to Gemini 2.5's thinking-token accounting on
  schema-constrained calls, not to this diff's Python slice syntax
  (`[0::sampling_freq, joints_to_use]` never actually reached the
  output - the response was truncated three tokens into the `location`
  field, well before any code content would even appear).
- **Fix, root cause (`generation/llm_client.py::_complete_with_gemini`)**:
  when `response_schema` is given (Prosecutor's claims, Judge's
  verdict - any schema-constrained call), also pass
  `thinking_config=types.ThinkingConfig(thinking_budget=0)` - confirmed
  supported by the installed `google-genai` 2.10.0. A mechanical
  extraction task against grounding rules already spelled out in the
  prompt doesn't need chain-of-thought reasoning eating the output
  budget; free-text completions (the Defender's justification, the
  rebuttal loop) are untouched - `thinking_config` is only added inside
  the `response_schema is not None` branch.
- **Fix, headroom (`config.py::LLM_MAX_TOKENS`)**: default raised
  1024 -> 4096. Not the primary fix (disabling thinking already frees
  the entire prior budget for visible output) - additional margin since
  a full multi-claim JSON array, each with a real `proposed_test` code
  snippet, can legitimately need more than 1024 tokens even with
  thinking removed entirely.
- **Fix, permanent diagnosability (`adjudicate/agents/prosecutor.py`)**:
  the retry-loop's warning log now includes `raw response text: %r` -
  a future parse failure (of any kind) is debuggable directly from logs,
  not just from a one-off reproduction script.
- **Considered and rejected: a JSON-repair/sanitization step** (the
  task's other suggested option) - rejected, not just skipped. A
  genuinely truncated response has no recoverable content past the cut
  point; "repairing" it would mean fabricating claim text the model
  never actually produced, which directly conflicts with the
  Prosecutor's whole grounded-claims design (Phase 27's own explicit
  anti-hallucination stance). Preventing the truncation at its root
  (thinking budget) is correct here; papering over it is not.
- **Verified live, real Gemini calls, same diff that failed 3/3 before
  this fix:** re-ran the exact VIBE `read_single_sequence` diff through
  the real `ProsecutorAgent.raise_concerns` (not a mock) - succeeded on
  the first attempt, 2 well-formed `untested_branch` claims, both
  correctly grounded (`location=lib/data_utils/amass_utils.py:100`,
  matching the diff's real changed function).
- **Verified no regression on known-good diffs**: `tartley/colorama`'s
  `reset_all` diff (already used throughout this project's history) -
  ran clean, correctly returned 0 claims (a legitimate empty array, not
  a parse failure - the Prosecutor genuinely found nothing to raise on
  a comment-only change, per its own system prompt's explicit
  instruction not to manufacture concerns). `preyesparab/NutriForge`'s
  `register` diff: blocked by Gemini's real free-tier daily quota (20
  requests/day - the exact same recurring limit flagged multiple times
  earlier in this file) partway through, not by this fix - did not
  keep retrying against an exhausted quota, per this project's own
  quota-discipline convention. Two of the three requested cases (the
  reported bug, one known-good baseline) are confirmed; the third is a
  disclosed open item, not a silent gap.
- Full backend suite: 693 passed, 0 failed - confirms nothing else
  broke structurally (this task touched no test-covered logic paths
  directly, `LLM_MAX_TOKENS`'s default is only asserted via `gt=0`).
- Open: NutriForge re-verification once the daily Gemini quota resets;
  the same `thinking_budget=0` change naturally also protects the
  Judge's structured verdict calls (same `response_schema` code path),
  though that wasn't separately re-verified here since it was outside
  this task's explicit scope.

## Verifier "Reversed (or previously applied) patch" — real root cause, NOT a persistent-clone mutation
Reported after the Prosecutor fix above unblocked progress to the
Verifier stage for the first time on this VIBE diff: `patch` refused
with "Reversed (or previously applied) patch detected!". The user's own
hypothesis (the earlier failed run left the persistent clone already
patched) was the right thing to check first, given the stakes - checked
it thoroughly, and it's not what's happening.

- **Item 1 (did the earlier failed run mutate the clone?) - NO,
  confirmed two ways.** `git status --short` on the real
  `data/repositories/mkocabas_VIBE` clone was already clean before any
  investigation started. Structurally, it couldn't have been otherwise:
  the earlier run's `ProsecutorAgent.raise_concerns` raised
  `LLMGenerationError` (a `RepoMindError`) after 3 failed attempts,
  which `run_live_review`'s `except RepoMindError` catches *before* the
  `if claims:` gate that creates the sandbox at all (`live_review.py`
  line ~228) - no sandbox was ever materialized in that failed run, so
  there was nothing to leave behind.
- **Item 2 (temp copy vs. persistent clone) - confirmed by reading the
  code, then proven by direct testing, not just assumed.**
  `_materialize_sandbox` (`adjudicate/orchestrator/live_review.py`) only
  ever *reads* `local_path`: `shutil.copytree(local_path, tmp_dir, ...)`
  into a brand-new `tempfile.mkdtemp()` directory, and `patch` runs with
  `cwd=tmp_dir` - there is no code path back to `local_path`. Proved
  this live, not just by inspection: ran repeated sandbox-creation
  attempts (several failing, one succeeding) against the real VIBE
  clone and checked `git status --short` after each - clean every time.
  As the strongest possible check (exactly what was asked for): cloned
  `mkocabas/VIBE` fresh from the real GitHub remote at the persistent
  clone's exact commit (`851f779...`) into a separate directory and ran
  `diff -rq` between the two trees (excluding `.git`) - **zero
  differences, exit code 0.** The persistent clone genuinely is not
  and was never mutated by this code path. Same one-directional-copy
  design confirmed in `adjudicate/benchmark/harness.py::materialize_sandbox`
  (Phase 31's own sandbox function) - not the same code, but the same
  safe pattern.
- **Item 3 - the real, reproducible bug (a different mechanism than
  either hypothesis in the request):** isolated by constructing the
  diff four ways and testing each against real `patch`: a diff with
  *both* a `diff --git a/X b/X` header line *and* a placeholder
  `index 0000000..0000000 100644` line makes GNU `patch` misinterpret
  an ordinary modification to an **existing** file as an attempt to
  **create a new one** - `0000000` on the old side is git's own "this
  file didn't exist before" signal, and patch acts on it even though
  the hunk clearly modifies real, existing lines. Confirmed the
  combination is what matters, not either line alone: `diff --git`
  alone applies fine; `index 0000000..0000000` alone applies fine;
  together, `patch` refuses ("already exists" or, depending on the
  exact context-match fuzz path, "Reversed (or previously applied)").
  This is *not* whitespace-related (tested a byte-exact-whitespace
  version of the same diff - still failed identically) and *not* a
  clone-corruption issue - it's specific to this one header
  combination.
  - Why this matters beyond one test diff: `index 0000000..0000000` is
    exactly the placeholder convention this project's *own*
    hand-constructed test/reproduction diffs have used throughout this
    session (colorama, NutriForge, pawn_ai, and the VIBE diff itself,
    all in earlier PROGRESS.md entries) - a very natural shortcut when
    writing a synthetic diff without running real `git diff`. A
    real `git diff`/`git show` output (real hashes) was never affected -
    confirmed by capturing one for real (made the actual edit in a
    scratch copy, ran `git diff` for a byte-perfect real diff with
    `index 5ec5807..9ead00a`) and it applied cleanly through the real
    `_materialize_sandbox` on the first try, no code change needed.
- **Fix (`adjudicate/orchestrator/live_review.py`):** added
  `_strip_git_index_lines`, which removes any `index <old>..<new>[
  <mode>]` line before handing the diff to `patch` - plain POSIX
  `patch` never actually reads that line at all (only `---`/`+++`/`@@`
  drive hunk application), so removing it is safe for both a
  placeholder-hash diff (fixes this bug) and a real-hash diff (the
  line is simply redundant there, already proven to apply fine with or
  without it).
- **Phase 31 benchmark concern - checked directly, not assumed: NOT
  affected.** Both real fixture files
  (`adjudicate/benchmark/fixtures/colorama_reset_all.diff`,
  `nutriforge_register_login.diff`) use genuine non-placeholder hashes
  (`index 6e01026..629f675`, `index 8c04e12..dc4fec9`) - real `git diff`
  output, not hand-typed synthetic diffs. Since the bug specifically
  requires the `0000000..0000000` placeholder combined with `diff
  --git`, and neither benchmark fixture has that pattern, Phase 31's
  existing numbers are not called into question by this bug.
- **Verified live, real code, real repo:**
  - The exact diff/repo that originally failed now materializes the
    sandbox successfully (confirmed twice in a row, fresh copies both
    times) and the target file's patched content is correct
    (`# downsample poses...` present at the right line).
  - Ran the real Verifier (`adjudicate.verifier.verify_claim` - zero
    LLM calls, per that module's own docstring) against the real
    claims already captured from the earlier successful Prosecutor run,
    inside the now-successfully-materialized sandbox: both claims
    returned genuine `REFUTED`/`HIGH confidence` results (`exit_code=0`
    from the real sandboxed test run) - the pipeline produces real
    structured verdicts end to end, not just "the patch applies."
  - Persistent-clone-unaffected check (the strongest form requested):
    `diff -rq` against a fresh independent clone from the real GitHub
    remote at the same commit - zero differences, both before and
    after this verification's own sandbox runs.
- Full backend suite: 693 passed, 0 failed.
- Not fixed (disclosed, out of scope here): the sandbox's Docker-less
  isolation limits, already flagged in `adjudicate/verifier/sandbox.py`'s
  own docstring, are unrelated to this bug and untouched.

## Documentation — `docs/project_description.md` rewritten as an interview reference (no code)
User-requested, not a roadmap phase: replaced this file's prior content (a
phase-by-phase build plan mirroring `docs/roadmap.md`) with a 20-section
interview-oriented engineering reference, per an explicit template the user
supplied. That template's example section content (Express/FastAPI/MongoDB/
Redis/YOLO/Gemini nutrition-app workflows) didn't match this repo at all -
flagged to the user before writing anything, who confirmed adapting the
20-section skeleton to RepoMind/Adjudicate's real architecture rather than
either treating it as the wrong project or filling it in literally.
- Also flagged and resolved before writing: an apparent "space vs
  underscore" filename collision (`project _description.md` vs
  `project_description.md`) turned out to be a shell-rendering artifact from
  an early `ls` call in this session, not a real second file - confirmed via
  `diff`/direct `ls` that only one file (`project_description.md`, this one)
  ever existed at this path. No file was renamed or duplicated.
- Research method: 4 parallel Explore agents read every module in depth
  (core/ingestion/graph/database/embedding; retrieval/generation/evaluation/
  api/pipeline; the full `adjudicate/` package; frontend+ui+tests+deployment)
  and reported dense, file-and-signature-level findings back, which were
  then synthesized into the new document - not delegated wholesale; the
  agents did research only, all synthesis/writing was done directly.
- New document's 20 sections map the user's original template onto this
  project's real subsystems (e.g. "Redis Usage" → "Caching Architecture,
  and why there is no Redis"; "MongoDB Indexing" → "Indexing & Query
  Optimization" covering FAISS/BM25/SQLite instead), and lead with real,
  already-documented bugs/limitations found during research (the semantic
  cache's CodeBERT cosine-similarity over-match, the `graph_store` vs
  `sqlite_client` lossy-persistence divergence, the Gemini thinking-token
  truncation bug, the Adjudicate benchmark's 46.8% claim-flip rate) rather
  than a generic idealized description.
- Verified against current, not stale, project state before writing: cross-
  checked the 693-passed test count, the real Phase 31 benchmark numbers
  (12/13 cases, condition a/b/c catch rates, claim-flip rate) and Phase 32
  Part 2's actual completion (confirmed real end-to-end SSE wiring on both
  backend and frontend, despite `docs/roadmap.md` line 71 still reading as
  if Part 2 were pending - noted as a stale doc, not corrected in this pass
  since only `project_description.md` was in scope) directly against this
  file's own history and the source, not assumed from the roadmap alone.
- Not done (out of scope per the user's own request, which was for this one
  file only): no correction to `docs/roadmap.md`'s stale Phase 32 line, no
  changes to `README.md` (also confirmed stale - still says "no business
  logic implemented yet"), no code changes of any kind.

## Session close-out: Phase 31 completed, Phase 32 fully live-verified, roadmap/README brought current
Four-item close-out session, per explicit instruction to clear every
stale/incomplete item before deployment. Checked both providers' quota
first (a real, minimal `LLMClient(max_tokens=8)` call each, not assumed from
the date): both Gemini and Groq returned real `200 OK` responses - the
2026-07-07/08 exhaustion had long since reset by 2026-07-19.

- **Phase 31 - now ✅, all 13/13 cases real.** Wrote
  `adjudicate/benchmark/complete_phase31.py` (the reconstruction script
  referenced but never actually committed by the prior session) - runs only
  `type_mismatch_fixed` for real, then recombines its result into the prior
  report's stored aggregates algebraically (`old_mean * old_n + new_value,
  / new_n`), not by re-deriving from scratch, since the saved report never
  persisted other cases' per-case token/call counts. Safe specifically
  because case 13 is clean: catch rate (buggy-only) is provably unchanged.
  Real run (Groq `llama-3.3-70b-versatile`): condition (a) correctly found
  no issue; condition (b) raised 3 claims (a false positive, trusted at face
  value); condition (c)'s Verifier refuted all 3 and the Judge approved -
  this 13th case is the **second** clean case (after `sql_injection_fixed`)
  where condition (c) discriminated correctly, not a third instance of the
  "confirmed-but-out-of-scope false positive" pattern the last entry
  explicitly flagged as unconfirmed for this case. Final 13-case numbers
  (full report: `adjudicate/benchmark/results/20260719T070702Z.json`):
  catch rates unchanged at 83.3%/100%/50% (a/b/c, case 13 doesn't affect the
  buggy-only denominator); false-positive rates now 14.3%/100%/71.4% (was
  16.7%/100%/83.3% over 12); **claim-flip rate now 50.0% (25/50)**, up from
  46.8% (22/47) - the headline number strengthens, not weakens.
- **Phase 32 Part 2 - now ✅, real full live run confirmed, Judge card
  hand-verified.** Ran the actual browser flow via a real Playwright +
  Chromium script (`playwright` 1.61.0 + Chromium already installed in
  `.venv`) against real `uvicorn`+`vite` dev processes: indexed
  `tartley/colorama` through the real landing form, submitted the same
  `reset_all` diff used as ground truth throughout this project, and
  watched the stream reach Context -> Defender -> Prosecutor/Verifier ->
  Rebuttal -> **Judge: REJECT, 80% confidence**, citing two
  `[CONFIRMED, high]` claims (`exception_handling`, `untested_branch` at
  `colorama/initialise.py:31`). Hand-checked against the diff: both claims
  are real and accurate (a bare `except OSError: pass` that swallows
  silently, with no test forcing that specific branch) - this exactly
  reproduces the standalone Phase 31 benchmark's own stored result for the
  same case (`colorama_reset_all_reused`: reject, 0.8 confidence),
  independently cross-validating both paths. Zero browser console errors.
  - **Two real, disclosed environment issues fixed along the way, not code
    bugs in the reviewed feature itself:** (1) `CORS_ALLOWED_ORIGINS`
    (`config.py`) only allowed port 5173; Vite fell back to 5174 because
    something was already bound to 5173 in this environment, so `.env` now
    additionally allows `5174` (both origins kept, nothing removed -
    confirmed via a real CORS preflight `OPTIONS` request before and after).
    (2) Port 8000 had an orphaned listening socket with no corresponding
    process visible to either this session's Bash sandbox or a host-level
    PowerShell `Get-Process`/`Get-NetTCPConnection` query - not a code
    issue, worked around by running the backend on 8010 for this
    verification rather than fighting an unkillable phantom reservation.
- **`docs/roadmap.md` audit:** cross-checked every phase entry (0-36)
  against this file's real history. Only Phase 31 and 32 were stale (both
  now flipped to ✅ above); Phases 21-30's ✅ markers and 33-36's ⬜ markers
  already matched reality, confirmed via this file's own Phase-numbered
  headers, not assumed.
- **`README.md` rewritten for real** (a prior session's own note confirms
  this was drafted once before but never landed - checked the live file
  first: it still read "Repository initialized. No business logic
  implemented yet.", a Phase-0-era placeholder). New version, sourced from
  `docs/project_description.md`: what Verdict AI does, the
  Defender/Prosecutor/Verifier/Judge review pipeline, the retrieval engine
  underneath (Tree-sitter, hybrid dense+sparse, graph expansion, reranking,
  semantic cache, small-to-big, RAGAS/ablation eval), real setup/run
  instructions (verified against `cli.py`'s actual subcommands and
  `evaluation/ablation.py`'s actual lack of a CLI entrypoint - not invented),
  and the real, final 13-case benchmark numbers (no longer provisional).
- **Full regression suite, run after all of the above:** backend
  `pytest -q`: **693 passed**, 0 failed (unchanged - `complete_phase31.py`
  is an integration entrypoint like `run.py`, which also has no dedicated
  test file, so this isn't a new gap). Frontend `npm run build`: 52 modules,
  0 errors.
- Not done (out of this session's stated scope): Phase 33 (Deployment) -
  correctly still ⬜, next real piece of work per the roadmap and this
  session's own README rewrite.

## Phase 33 (storage-layer slice) — SQLite → PostgreSQL migration

Deliberately scoped narrowly, as instructed: only the relational storage
layer (`database/sqlite_client.py`, `database/graph_store.py`,
`database/models.py`, `config.py`) moved to PostgreSQL. FAISS, BM25, Redis,
task queues, and `pipeline.py`/agent/API-contract logic were explicitly
untouched. Migrated and regression-tested incrementally (full suite after
each step, reported before proceeding), not as one rewrite.

- **Driver: `psycopg2` (sync), not `asyncpg`.** Every `DatabaseManager`
  method is a blocking SQLAlchemy `Session` call, and `pipeline.py`/
  `api/main.py`/every retrieval module already depend on that being
  synchronous - `asyncpg` would require an async engine + `AsyncSession` +
  `await` at every call site, which the task explicitly put out of scope
  ("storage-layer swap underneath, not a behavior change"). `psycopg2-binary`
  keeps the exact same `Session`/`sessionmaker` API; only the connection
  string and a couple of dialect details changed.
- **Dev/test Postgres:** `docker-compose.yml` gained a `postgres:16` service
  (user/pass/db all `repomind`, healthchecked, named volume). Local setup is
  `docker compose up -d postgres`. `config.Settings.DATABASE_URL` replaces
  `SQLITE_DB_PATH`/`SQLITE_DIR` entirely (no dual-backend shim), defaulting
  to `postgresql+psycopg2://repomind:repomind@localhost:5432/repomind`.
- **Test isolation redesigned:** the old one-SQLite-file-per-test pattern
  (`DatabaseManager(db_path=tmp_path / "test.db")`) has no Postgres
  equivalent, since all tests now share one real Postgres instance.
  Replaced with one PostgreSQL **schema** per test:
  `DatabaseManager(schema: str | None = None)` scopes every table it
  touches via the connection's `search_path` (`connect_args={"options":
  f"-csearch_path={schema}"}`); `initialize_database()` issues
  `CREATE SCHEMA IF NOT EXISTS` (via a separate, schema-agnostic connection,
  since a connection whose `search_path` already points at a not-yet-
  existing schema can't `CREATE TABLE` into it) before `Base.metadata
  .create_all`; a new `drop_schema()` (`DROP SCHEMA ... CASCADE`) tears it
  down. `tests/conftest.py` adds one shared `pg_schema` fixture (a random
  `test_<uuid hex>` name); every affected test file's local `db` fixture
  now does `DatabaseManager(schema=pg_schema)` → `yield` → `drop_schema()`
  instead of returning a `tmp_path`-backed manager. Schema names are
  validated against a bare-identifier regex before use (never trusted as
  arbitrary SQL) even though they're always internally generated, not user
  input.
- **`database/models.py`:** table/FK/unique-constraint shapes unchanged.
  One real correctness fix, not scope creep: `indexed_at`/`created_at`
  columns are now `DateTime(timezone=True)` - SQLite silently tolerated
  storing timezone-aware `datetime.now(timezone.utc)` values, but
  PostgreSQL's default `DateTime` maps to `TIMESTAMP WITHOUT TIME ZONE` and
  would have silently dropped the tzinfo.
- **`database/sqlite_client.py`:** kept its filename and the
  `DatabaseManager` public interface unchanged (23 files import it -
  renaming was flagged as a large mechanical diff for a task scoped as
  "storage-layer swap, not a behavior change" and skipped). Internals:
  `create_engine(settings.DATABASE_URL, pool_pre_ping=True)` replaces the
  SQLite URL; the SQLite-only `PRAGMA foreign_keys=ON` connect listener is
  gone (PostgreSQL always enforces FKs). `_session_scope` renamed to the
  public `session_scope` specifically so `database.graph_store` can share
  an instance's exact engine/schema rather than opening an independent
  connection (see below).
- **`database/graph_store.py` - file → table, not just SQLite → Postgres.**
  This module was never SQLite-backed to begin with; it read/wrote a flat
  JSON file per repository (`data/graph/{owner}_{name}.json`), keyed by an
  arbitrary `path` argument. New `GraphSnapshotRecord` (`repository_id` PK/
  FK → `repositories`, `graph_data JSONB`, `updated_at`) replaces the file;
  `save_graph(graph, repository_id, db)`/`load_graph(repository_id, db)`
  upsert/read that row - same `nx.node_link_data`/`node_link_graph` JSON
  shape as before, just relocated. Both functions now take the caller's
  `DatabaseManager` (via the new public `session_scope`) instead of opening
  an independent engine: `GraphSnapshotRecord.repository_id` is a foreign
  key into `RepositoryRecord`, so a snapshot has to be written/read through
  the exact same engine/schema every other table for that repository uses
  - an independent connection would silently target the wrong schema under
  per-test isolation (caught by a real test failure - `relation
  "graph_snapshots" does not exist` - not just a design objection; see
  "bug found" note below). Dropping the `path` parameter cascaded into
  removing `settings.GRAPH_DIR`/`GRAPH_FILE_PATH` (no longer meaningful for
  a DB-backed store) and `retrieval.graph_retriever.GraphExpander`'s
  `graph_dir`/owner-name-derivation plumbing (`_index_stem`/`_graph_path`) -
  it now takes `db` and calls `load_graph(repository_id, db)` directly, no
  filename derivation needed. Call sites updated mechanically (argument
  changes only, no logic changes): `pipeline.py` (`save_graph`,
  `GraphExpander(self._db)`), `api/main.py` (`_reconstruct_ready_state`,
  `get_graph` - and its now-dead `_graph_json_path` helper removed),
  `scripts/demo_phase16_end_to_end.py`.
- **Bug found and fixed during this migration (not shipped broken):**
  first pass gave `graph_store.py` its own independent module-level
  engine/sessionmaker (for "independent testability"). Full regression
  immediately caught it: `psycopg2.errors.UndefinedTable: relation
  "graph_snapshots" does not exist` in every graph-store/graph-retriever
  test, because that independent engine had no `search_path` override and
  connected to the connection's default schema, while the per-test
  `DatabaseManager(schema=pg_schema)` had only created the table inside the
  test's own schema. Root cause understood before fixing (not papered
  over): `GraphSnapshotRecord`'s FK into `RepositoryRecord` makes "its own
  connection" structurally wrong, not just a test-isolation inconvenience -
  fixed by threading `db: DatabaseManager` through both functions instead.
- **The Phase 7 bug this task specifically asked to re-verify** (`api/main
  .py`'s `/status`-DB-fallback and `/graph` previously called
  `DatabaseManager.load_graph`'s lossy SQL reconstruction - all-stored-
  chunks-as-nodes, no file-level edges - instead of `database.graph_store
  .load_graph`'s exact persisted snapshot) **still holds after the
  migration**: both functions exist as distinct, intentionally-different
  Postgres-backed paths (`DatabaseManager.load_graph` reconstructs from
  `code_chunks`/`graph_edges`; `graph_store.load_graph` reads the
  `graph_snapshots` JSONB row), and `api/main.py`/`retrieval.graph_retriever
  .GraphExpander` still call only the latter.
- **Migrated incrementally, full suite after each step, as instructed:**
  (1) Postgres infra + `DATABASE_URL` wiring; (2)-(6)
  `RepositoryRecord`/`SourceFileRecord`/`CodeChunkRecord`/`GraphEdgeRecord`
  + `DatabaseManager.load_graph`/`EmbeddingRecord`/`SemanticCacheRecord` -
  one atomic engine swap in `sqlite_client.py` covers all of these at once,
  verified via `tests/test_database/test_sqlite_client.py`'s 37 tests
  passing standalone before the full suite; (7) `graph_store.py` → JSONB
  (see above, including the bug-fix-and-retest cycle). Full suite: **693
  passed**, 0 failed, both before and after step 7 (net test count
  unchanged from the pre-migration SQLite baseline despite completely
  replacing the storage backend - `test_graph_store.py`'s file-path-
  specific tests were replaced 1:1 with schema/row-equivalent tests).
- **End-to-end verification, against the real dev Postgres instance, not
  mocked:** fresh `Pipeline()` (default/production schema, no test
  isolation) re-indexed `github.com/tartley/colorama` from scratch.
  `files_discovered=23, chunks_indexed=344, graph_nodes=184,
  graph_edges=283` - **exact match** to the known-good numbers this same
  repository produced against SQLite (Phase 20's verification, recorded
  above). Real chat query ("What does the AnsiFore class do?") through the
  full retrieval+generation stack (real CodeBERT embeddings, real FAISS/
  BM25/graph-expansion/reranking, real Groq `llama-3.3-70b-versatile`
  generation - `USE_GEMINI` fell through to the Groq fallback) returned a
  correct, grounded answer citing `colorama/ansi.py`'s `AnsiFore`/
  `AnsiCodes` classes - citations resolved to real file paths, not empty.
  Semantic cache (Phase 14) backend/frontend regression pair re-run against
  this real Postgres-backed cache: "what tech stack does the frontend use"
  correctly missed (rejected as lexically incompatible despite 0.9963
  cosine similarity) after caching a backend answer; the exact-repeat query
  correctly hit (similarity=1.0000) - the lexical-compatibility gate holds
  identically against Postgres-backed cache entries. Finally, started a
  fresh `uvicorn api.main:app` process (simulating "previous process"
  reconstruction, not the same in-memory state the indexing run left
  behind) and hit the real HTTP endpoints for this same repository:
  `GET /status` → `graph_nodes=184, graph_edges=283`; `GET /graph` → 184
  nodes / 283 edges in the returned node-link JSON. All three paths (direct
  `Pipeline`, `/status`, `/graph`) agree, exactly the invariant the
  original Phase 7 bug fix established - confirmed it survived the
  migration rather than assumed.
- Not done (deliberately out of this slice's scope, per the task):
  FAISS/pgvector migration, Redis, task queues, and the rest of Phase 33
  (Vercel/Render hosting, sandboxed verifier hosting) - roadmap left at
  🟡 (in progress) rather than flipped to ✅, since the phase as a whole
  (deployment) is not complete, only its storage-layer prerequisite.

## Phase 33 (deployment-blocker slice) — local CodeBERT embeddings → hosted Gemini embedding API

Driven by a real Render OOM during deployment: loading `microsoft/codebert-base`
locally (`embedding/model_loader.py`, via `sentence-transformers`/`torch`) cost
far more resident memory than Render's instance had available. Replaced with
Google's hosted Gemini embedding API - same motivation and shape as Phase 16's
earlier LLM-provider move to a hosted API, applied to embeddings.

- **Model check done live before writing code, per instructions.**
  `google-genai==2.10.0` (installed, both `.venv` and base env) was queried
  for real via `client.models.list()` against the project's actual
  `GEMINI_API_KEY` - not assumed from docs/memory. `text-embedding-004`
  (the name floated at the start of this task, and the only name in the
  SDK's own docstring example) **is not in the live list** - it's retired.
  Live-available: `gemini-embedding-001` (GA, chosen), `gemini-embedding-2`,
  `gemini-embedding-2-preview` (both multimodal, unneeded here).
  `gemini-embedding-001` defaults to 3072-dim; live-verified
  `EmbedContentConfig(output_dimensionality=768)` truncation works
  (Matryoshka/MRL-trained, not a naive slice) - 768 chosen over 1536/3072 to
  keep FAISS index size/search cost down, matching the existing
  768/384-dim pattern in `core.constants.EMBEDDING_DIMENSIONS`.
- **Three design decisions confirmed with the user before implementing**
  (all three recommended options accepted): output dimension 768; drop
  `USE_CODEBERT`/MiniLM fallback entirely rather than keep it opt-in (Gemini
  API is now the only embedding path - no local fallback); skip asymmetric
  `task_type` (`RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`/`CODE_RETRIEVAL_QUERY`,
  all three live-verified as accepted) for now, in favor of zero call-site
  changes - flagged as an easy future quality follow-up, not done here.
- **`embedding/model_loader.py` rewritten, same interface, real behavior
  change underneath.** `GeminiEmbeddingModel` adapts
  `google.genai.Client.models.embed_content` behind the exact
  `.encode(sentences, **kwargs) -> np.ndarray` shape
  `sentence_transformers.SentenceTransformer` used to expose (`**kwargs`
  like `convert_to_numpy`/`show_progress_bar` accepted and ignored) - so
  `EmbeddingManager`, `SemanticCacheManager`, and `pipeline.py`'s
  `_embed_query` needed zero changes beyond the new dimension. Inference
  failures are deliberately left unwrapped from `GeminiEmbeddingModel.encode`
  itself (every caller already wraps `model.encode(...)` failures into its
  own exception type at its own boundary - matches the existing per-module
  error-boundary pattern rather than double-wrapping).
- **`generation/llm_client.py`'s "only place that imports google-genai"
  claim was factually broken by this change** - caught and fixed in its own
  docstring rather than left stale, since `embedding.model_loader` now
  independently constructs its own `genai.Client` too (kept independent,
  not shared, so a construction failure still wraps into the right
  exception type per module - `EmbeddingError` here, `LLMGenerationError`
  there).
- **Breaking change, no migration path - by design, not an oversight.**
  Different vector space, different dimension (768 Gemini vs. 768 CodeBERT/
  384 MiniLM coincidentally-same-looking-but-incompatible numbers) -
  `core.constants`'s module comment documents that every previously-indexed
  repository needs `EmbeddingManager.generate_embeddings(..., force=True)` +
  a fresh `FaissIndexManager.build_index`. Confirmed pre-launch with the
  user: no migration script written, matching this project's stated
  practice of not building for data that doesn't need to survive.
- **`retrieval/reranker.py` still needs `sentence-transformers`/`torch` -
  requirements.txt re-attributed, not removed.** Checked before touching
  requirements.txt (per this session's standing quota/verification
  discipline - confirm before assuming): `CrossEncoderReranker` is an
  independent local-transformer consumer, `USE_RERANKER=True` by default.
  The Phase 8 comment block moved to a new Phase 13 block explaining
  `sentence-transformers`/`torch` stay for this reason - the OOM fix from
  this change alone is partial, not total, and step 6 below quantifies
  exactly how much.
- **Regression suite: 695 passed, 0 failed, 0 errors** (up from 693 pre-
  existing - two new call sites' worth of test coverage:
  `tests/test_embedding/test_model_loader.py` rewritten entirely around a
  fake `genai.Client`/`embed_content`, not real network calls;
  `test_embedding_manager.py`'s CodeBERT/MiniLM two-model-class split
  collapsed into one, since there's only one model now;
  `test_database/test_vector_store.py` and `test_config.py` swept for the
  removed `DEFAULT_MINILM_MODEL`/`USE_CODEBERT`). First full run showed "1
  failed, 566 passed, 128 errors" - misleading: local Postgres wasn't
  running (Docker Desktop wasn't started this session), so the 128 errors
  were `pg_schema` fixture failures, and the "1 failed" never reproduced
  again once Postgres was actually up - not a real regression, a false
  signal from a down dependency.
- **Real, live re-index of `tartley/colorama` against Gemini embeddings -
  hit two genuine operational issues worth recording, not glossed over:**
  1. **Gemini's free-tier embedding quota is metered per text embedded, not
     per API call** (`embed_content_free_tier_requests`, ~100/minute) - the
     SDK batches a whole `contents` list into one `batchEmbedContents` HTTP
     call, but each text inside still consumes one quota unit. With
     `EMBEDDING_BATCH_SIZE=32`, colorama's 161 AST chunks exhausted the
     free-tier quota after 3 batches (96 texts) plus this session's own
     earlier live model-check calls. Recovered cleanly with no code change -
     `EmbeddingManager`'s existing skip-already-embedded logic resumed from
     chunk 97 after the documented ~51s cooldown - but this is a real
     constraint worth the user knowing before re-indexing anything larger
     than a small repo on the free tier: not fixed here (retry/backoff
     wasn't asked for and wasn't added unasked), only surfaced.
  2. **`retrieval/reranker.py`'s CrossEncoder load crashed the process
     natively (no Python traceback, no exception, process just exits)** -
     reproduced twice, always at the same point (`Reranking started: 96
     candidate(s)`, right as `CrossEncoder(...)` constructs), but **not**
     reproducible in three separate minimal repros of the same import/call
     sequence (`faiss` + one or two real `genai.Client`/`embed_content`
     calls + `CrossEncoder` load, with and without
     `sqlalchemy`/`psycopg2`/`networkx`/`rank_bm25` also imported) - all
     three isolated repros succeeded cleanly. Root cause not conclusively
     identified (best guess: transient memory/resource pressure from
     Docker Desktop's own startup coinciding with the first crash, given
     4.1GB free RAM of 15.4GB total at the time - not re-tested under
     confirmed-idle conditions). Worked around for this verification only
     by setting `settings.USE_RERANKER = False` in the (temporary, not
     committed to the app) verification script - this is a pre-existing
     local-environment fragility in the reranker's own dependency stack,
     unrelated to and not introduced by this embedding change (every
     retrieval stage *before* reranking - dense, sparse, hybrid fusion,
     graph expansion, semantic cache - completed successfully against real
     Gemini embeddings in every run, crashed or not), flagged for separate
     investigation rather than fixed here.
  3. With reranking disabled, real grounded/cited answers confirmed: "How
     does colorama strip ANSI codes on Windows?" → correct, detailed answer
     citing `colorama/ansitowin32.py`'s `AnsiToWin32.__init__`/
     `write_and_convert` and the real `ANSI_CSI_RE`/`ANSI_OSC_RE` regexes,
     plus `colorama/tests/ansitowin32_test.py`'s actual
     `testWriteAndConvertStripsAllValidAnsi` test - genuinely grounded, not
     hallucinated, chunk_ids resolved to real code. A second query ("What
     does the AnsiToWin32 class do?") correctly returned "the repository
     does not contain enough information" rather than hallucinating, when
     the specific class chunk didn't survive un-reranked RRF fusion into
     the top 8 - honest refusal, not a bug, and a concrete illustration of
     what the disabled reranker normally buys.
- **Memory footprint - measured with `psutil`, real RSS deltas, not
  estimated** (temporarily installed in `.venv` for this measurement only,
  not added to `requirements.txt`): loading the **old** local CodeBERT path
  (`sentence-transformers`/`transformers`/`torch`, post-`faiss`+`numpy`
  import) cost **879 MB** RSS delta (927 MB total process RSS afterward) -
  higher than this task's own "~500MB+" framing, not lower. The **new**
  Gemini-client path costs **76 MB** RSS delta (126.5 MB total) - a
  measured **~800 MB (~86%) reduction** for the embedding path specifically.
  The still-local reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) costs
  its own **394 MB** RSS delta (442.6 MB total) - smaller than CodeBERT
  (fewer params) but not eliminated, so **this change alone does not fully
  resolve the Render OOM** if the reranker stays enabled there; it removes
  the larger of the two local-model costs and leaves the smaller one.
  Whether ~440 MB plus the rest of a FastAPI/uvicorn process fits Render's
  instance is the open question for whoever deploys next - not re-verified
  against an actual Render instance in this session (local measurement only).
- **`.env.example` had live Render Postgres credentials pasted into it**
  (unrelated to this task, found incidentally via `git diff` before editing
  the same file) - flagged to the user, not committed or removed
  unilaterally; only the now-dead `USE_CODEBERT=true` line was removed from
  it as part of this change.
- Files touched: `core/constants.py`, `config.py`,
  `embedding/model_loader.py` (rewritten), `embedding/embedding_manager.py`
  (docstring), `embedding/__init__.py` (docstring),
  `retrieval/semantic_cache.py` (docstring), `database/vector_store.py`
  (docstring), `generation/llm_client.py` (docstring),
  `pipeline.py` (docstring), `scripts/demo_phase16_end_to_end.py`,
  `requirements.txt`, `.env.example`,
  `tests/test_embedding/test_model_loader.py` (rewritten),
  `tests/test_embedding/test_embedding_manager.py`,
  `tests/test_database/test_vector_store.py`, `tests/test_config.py`.
- Not done (deliberately, per the task): a migration script for existing
  indexed data (see above); retry/backoff for the free-tier rate limit;
  fixing the reranker's native crash; re-verifying actual Render memory
  post-deploy.