/**
 * Thin HTTP client for `api/main.py`'s FastAPI backend (Phase 20/32).
 *
 * Mirrors `ui/api_client.py`'s contract (same base URL default, same
 * endpoints) so the two clients - Python for Streamlit, JS for this
 * React frontend - stay behaviorally interchangeable even though they
 * are separate implementations; there is no shared-codegen step in this
 * project.
 *
 * Response shapes (documented here, not enforced by a type system - this
 * project uses plain JS/JSX, not TypeScript):
 *
 * StatusResponse: { repo_id, status: "pending"|"indexing"|"ready"|"failed",
 *   stage, error, owner, name, commit_hash, files_discovered,
 *   chunks_indexed, graph_nodes, graph_edges, embedded_chunks }
 * QueryResponse: { answer, model, cache_hit, retrieved_count,
 *   graph_expanded_count, latency_ms, citations: CitationResponse[] }
 * CitationResponse: { chunk_id, file_path, function_name, chunk_type,
 *   retrieval_source, relevance_score, raw_code }
 * GraphResponse (database.graph_store's node-link JSON): { directed,
 *   multigraph, nodes: GraphNode[], edges: GraphEdge[] }
 * GraphNode: { id, node_kind: "file"|"chunk", file_path, language,
 *   chunk_type, function_name, class_name, parent_class, start_line,
 *   end_line } - only "function"|"async_function"|"class"|"method"|
 *   "arrow_function" chunk_types are graph nodes; "sliding"/"parent"
 *   window chunks (possible on a citation) never are.
 * GraphEdge: { source, target, edge_type: "function_call"|"method_call"|
 *   "inherits"|"imports"|"contains"|"references" }
 *
 * Live review SSE events (Phase 32 Part 2, `POST /repos/{repo_id}/review`
 * - see `adjudicate.orchestrator.live_review.run_live_review`'s own
 * docstring for the authoritative per-stage shape):
 * ContextEvent: { type: "context", changed_functions, callers, callees, related_tests }
 * DefenderEvent: { type: "defender", justification }
 * ClaimsEvent: { type: "claims", claims: [{ index, claim_type, location, assertion, proposed_test }] }
 * ClaimVerifiedEvent: { type: "claim_verified", index, status: "confirmed"|"refuted"|"inconclusive",
 *   confidence: "high"|"medium"|"low", evidence, strategy }
 * RebuttalEvent: { type: "rebuttal", round, text }
 * JudgeEvent: { type: "judge", verdict: "approve"|"reject"|"needs_human_review",
 *   confidence, cited_evidence, minority_report }
 * DoneEvent: { type: "done" } | ErrorEvent: { type: "error", message }
 */

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL;

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, init) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    throw new ApiError(`Could not reach RepoMind API at ${BASE_URL}: ${String(cause)}`, 0);
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body) => body.detail)
      .catch(() => response.statusText);
    throw new ApiError(detail ?? response.statusText, response.status);
  }

  return response.json();
}

export function getInfo() {
  return request("/info");
}

export function indexRepository(repoUrl) {
  return request("/repos/index", {
    method: "POST",
    body: JSON.stringify({ repo_url: repoUrl }),
  });
}

export function getStatus(repoId) {
  return request(`/repos/${encodeURIComponent(repoId)}/status`);
}

export function queryRepository(repoId, question) {
  return request(`/repos/${encodeURIComponent(repoId)}/query`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

/**
 * Fetch the call graph, or (when `focusNode` is given) just the
 * blast-radius subgraph around it (Phase 22) - same node-link shape
 * either way, see `api/main.py`'s `GET /repos/{repo_id}/graph` docstring.
 */
export function getGraph(repoId, focusNode, hops) {
  const params = new URLSearchParams();
  if (focusNode !== undefined) {
    params.set("focus_node", focusNode);
    if (hops !== undefined) params.set("hops", String(hops));
  }
  const query = params.toString();
  return request(`/repos/${encodeURIComponent(repoId)}/graph${query ? `?${query}` : ""}`);
}

/**
 * Stream a live adversarial review via `POST /repos/{repo_id}/review` (Phase 32 Part 2).
 *
 * The endpoint is `POST` (the diff can be arbitrarily long, so it can't
 * ride a query string), which rules out the browser's native
 * `EventSource` (GET-only) - this reads the `text/event-stream` body
 * directly via `fetch` + a manual `ReadableStream` reader instead,
 * calling `onEvent(parsedEvent)` for each ``data: <json>\n\n`` block as
 * it arrives. Buffers partial chunks across `reader.read()` calls, since
 * a single SSE event can be split across TCP/stream chunk boundaries.
 *
 * @param repoId - The indexed repository to review the diff against.
 * @param diff - The raw unified diff text.
 * @param onEvent - Called once per parsed SSE event, in arrival order.
 * @param signal - Optional `AbortSignal` to cancel an in-flight review.
 */
export async function streamReview(repoId, diff, onEvent, signal) {
  let response;
  try {
    response = await fetch(`${BASE_URL}/repos/${encodeURIComponent(repoId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ diff }),
      signal,
    });
  } catch (cause) {
    throw new ApiError(`Could not reach RepoMind API at ${BASE_URL}: ${String(cause)}`, 0);
  }

  if (!response.ok || !response.body) {
    const detail = await response
      .json()
      .then((body) => body.detail)
      .catch(() => response.statusText);
    throw new ApiError(detail ?? response.statusText, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? ""; // the last block may be incomplete - keep it buffered
    for (const block of blocks) {
      const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      onEvent(JSON.parse(dataLine.slice("data: ".length)));
    }
  }
}
