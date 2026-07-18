import { createContext, useContext, useMemo, useState } from "react";
import { useGraph } from "../hooks/useGraph";

/**
 * Cross-view state shared by the sidebar and graph view (Phase 32 Part 1):
 * which repository is loaded, the chat transcript, the loaded graph
 * (fetched once here via `useGraph` and shared - not re-fetched
 * separately by the sidebar's citation resolution and the graph view),
 * and which graph node is currently focused (set either by clicking a
 * node directly or by clicking a citation in the chat - the mechanism
 * that links the two views together per this phase's brief).
 */
const RepoContext = createContext(null);

export function RepoProvider({ children }) {
  const [repoId, setRepoId] = useState(null);
  const [repoLabel, setRepoLabel] = useState(null);
  const [history, setHistory] = useState([]);
  const [focusNodeId, setFocusNodeId] = useState(null);
  const [focusRequestId, setFocusRequestId] = useState(0);
  const { graph, nodesById, loading: graphLoading, error: graphError } = useGraph(repoId);

  /**
   * Focus a graph node. `focusRequestId` increments on every call (even
   * re-focusing the same node) so GraphView can re-trigger its
   * pan/zoom + blast-radius animation from a `useEffect` keyed on it,
   * rather than missing a re-click on an already-focused node.
   */
  function focusNode(nodeId) {
    setFocusNodeId(nodeId);
    setFocusRequestId((id) => id + 1);
  }

  function addChatTurn(turn) {
    setHistory((prev) => [...prev, turn]);
  }

  /**
   * Return to the pre-indexing landing state, same as a fresh page
   * load - clears the loaded repo (which also clears `graph`/`nodesById`
   * via `useGraph`'s own effect reacting to `repoId` going back to
   * null), the chat transcript, and any graph focus/highlight. Added for
   * the "Verdict AI" home link (Sidebar.jsx) - AppShell.jsx pairs this
   * with `useIndexingStatus`'s own new `reset()` so both halves of
   * "fresh page load" state go back to idle together.
   */
  function resetRepo() {
    setRepoId(null);
    setRepoLabel(null);
    setHistory([]);
    setFocusNodeId(null);
  }

  const value = useMemo(
    () => ({
      repoId,
      setRepoId,
      repoLabel,
      setRepoLabel,
      history,
      addChatTurn,
      focusNodeId,
      focusRequestId,
      focusNode,
      resetRepo,
      graph,
      nodesById,
      graphLoading,
      graphError,
    }),
    [repoId, repoLabel, history, focusNodeId, focusRequestId, graph, nodesById, graphLoading, graphError],
  );

  return <RepoContext.Provider value={value}>{children}</RepoContext.Provider>;
}

export function useRepo() {
  const ctx = useContext(RepoContext);
  if (ctx === null) {
    throw new Error("useRepo() must be called within a <RepoProvider>");
  }
  return ctx;
}
