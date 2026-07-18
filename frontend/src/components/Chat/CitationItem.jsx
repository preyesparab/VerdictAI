import { GRAPH_NODE_CHUNK_TYPES } from "../../constants";
import { useRepo } from "../../state/RepoContext";

/**
 * One cited chunk, rendered as a small inline monospace chip that flows
 * with the answer's text (its parent isn't a flex container, so this
 * `inline-flex` element wraps like a word rather than stacking as a
 * separate block) - not a detached citations list underneath the answer.
 *
 * Only clickable when it resolves to a real graph node: `/query`'s
 * citations carry a `chunk_id` but no line numbers, and - because
 * `USE_SMALL_TO_BIG` defaults on - are frequently PARENT/SLIDING window
 * chunks, which are never graph nodes at all (see api/client.js's
 * GraphNode doc). `nodesById` (the already-loaded full graph) is the
 * source of truth for both the line-number lookup and the graph-node
 * check - there is no separate line-number field to fall back on.
 */
export function CitationItem({ citation, nodesById }) {
  const { focusNode } = useRepo();
  const graphNode = nodesById[citation.chunk_id];
  const isGraphNode = GRAPH_NODE_CHUNK_TYPES.has(citation.chunk_type) && graphNode !== undefined;

  const label = citation.function_name || citation.chunk_type;
  const location = graphNode
    ? `${citation.file_path}:${graphNode.start_line}`
    : citation.file_path;

  if (!isGraphNode) {
    return (
      <span
        className="mx-1 inline-flex items-baseline gap-1 rounded-md border border-slate-800 bg-slate-900/60 px-1.5 py-0.5 align-middle font-mono text-xs text-slate-400 opacity-70"
        title="Not a distinct graph node (window chunk)"
      >
        {label}
        <span className="text-slate-500">{location}</span>
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => focusNode(citation.chunk_id)}
      title="Jump to this node in the graph"
      className="mx-1 inline-flex items-baseline gap-1 rounded-md border border-indigo-400/30 bg-indigo-500/10 px-1.5 py-0.5 align-middle font-mono text-xs text-indigo-300 transition hover:border-indigo-400/60 hover:bg-indigo-500/20"
    >
      {label}
      <span className="text-indigo-400/80">{location}</span>
    </button>
  );
}
