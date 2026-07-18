import { useCallback, useEffect, useState } from "react";
import { getGraph } from "../api/client";

/**
 * Loads the full call graph for `repoId` once it becomes available, and
 * exposes `nodesById` (built once per load) so citation-resolution
 * (`CitationItem`) and the graph view's own lookups don't each re-derive
 * it separately.
 */
export function useGraph(repoId) {
  const [graph, setGraph] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    if (!repoId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getGraph(repoId);
      setGraph(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [repoId]);

  useEffect(() => {
    setGraph(null);
    if (repoId) reload();
  }, [repoId, reload]);

  const nodesById = graph ? Object.fromEntries(graph.nodes.map((node) => [node.id, node])) : {};

  return { graph, nodesById, loading, error, reload };
}
