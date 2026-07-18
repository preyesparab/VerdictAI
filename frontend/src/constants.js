/**
 * Mirrors `pipeline.INDEXING_STAGES` (Python) - the fixed, ordered list of
 * stage names `on_progress` reports during `POST /repos/index` (relayed
 * through `GET /repos/{repo_id}/status`'s `stage` field). Kept in sync by
 * hand; if `pipeline.py` changes this tuple, this list must change too.
 */
export const INDEXING_STAGES = [
  "Cloning repository",
  "Parsing source files",
  "Chunking",
  "Building knowledge graph",
  "Generating embeddings",
  "Building FAISS/BM25 indexes",
];

/** Default blast-radius traversal depth (matches `graph.blast_radius`'s own default). */
export const DEFAULT_BLAST_RADIUS_HOPS = 2;

/** AST-derived chunk types that are real graph nodes (see api/client.js's GraphNode doc). */
export const GRAPH_NODE_CHUNK_TYPES = new Set(["function", "async_function", "class", "method", "arrow_function"]);
