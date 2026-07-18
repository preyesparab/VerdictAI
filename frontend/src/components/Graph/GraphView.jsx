import { useEffect, useRef, useState } from "react";
import { DataSet } from "vis-data";
import { Network } from "vis-network";
import { DEFAULT_BLAST_RADIUS_HOPS } from "../../constants";
import { getGraph } from "../../api/client";
import { useRepo } from "../../state/RepoContext";
import { animateBlastRadius, clearHighlight } from "./blastRadius";
import { toVisElements } from "./visElements";
import { buildNetworkOptions } from "./visStyle";
import { NodeTooltip } from "./NodeTooltip";

/**
 * Full-height graph visualization: the main content area, right of the
 * sidebar. Rendered with vis-network (swapped from Cytoscape.js) via the
 * standard `useRef` + `useEffect` integration pattern - vis-network has
 * no official React bindings, so the `Network` instance and its two
 * `DataSet`s (nodes/edges) are constructed imperatively here and kept in
 * refs across renders, same lifecycle role `cyRef` played before.
 *
 * `/graph` fetching (`useRepo`'s `graph`), the blast-radius hop-BFS
 * (`computeHopTiers` in `./blastRadius.js`), and the citation-to-graph
 * linking mechanism (`focusNodeId`/`focusRequestId`/`focusNode` from
 * `RepoContext`) are all unchanged - only how the resulting highlight is
 * *rendered* (vis-network DataSet updates instead of Cytoscape classes/
 * `.animate()`) and how click/hover are *wired* (vis-network's
 * `click`/`hoverNode`/`blurNode` events instead of Cytoscape's
 * `tap`/`mouseover`/`mouseout`) changed.
 */
export function GraphView() {
  const { repoId, focusNodeId, focusRequestId, focusNode, graph, graphLoading, graphError } = useRepo();
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const nodesDataSetRef = useRef(null);
  const edgesDataSetRef = useRef(null);
  const hoveredNodeRef = useRef(null);
  const [tooltip, setTooltip] = useState(null); // { x, y, node } | null
  const [highlightActive, setHighlightActive] = useState(false);

  // Build (or rebuild, if `graph` changes - e.g. a different repo was
  // indexed) the vis-network Network. Runs once per real `graph` object,
  // same frequency the old `useMemo(() => toCytoscapeElements(graph),
  // [graph])` recomputed at.
  useEffect(() => {
    if (!graph || !containerRef.current) return;

    const { nodes, edges } = toVisElements(graph);
    const nodesDataSet = new DataSet(nodes);
    const edgesDataSet = new DataSet(edges);
    nodesDataSetRef.current = nodesDataSet;
    edgesDataSetRef.current = edgesDataSet;

    const network = new Network(
      containerRef.current, { nodes: nodesDataSet, edges: edgesDataSet }, buildNetworkOptions(nodes.length),
    );
    networkRef.current = network;

    // vis-network draws labels onto a <canvas> - unlike DOM text, canvas
    // text does not automatically re-render when a web font finishes
    // loading asynchronously, so if Inter (linked in index.html) is
    // still downloading when this first paints, node labels can render
    // in the `sans-serif` fallback and silently stay that way even
    // after Inter loads. `document.fonts.load` + a `redraw()` once it
    // resolves forces a repaint in the correct font instead of relying
    // on physics-driven redraws to eventually paper over it.
    if (typeof document !== "undefined" && document.fonts?.load) {
      document.fonts.load("9px Inter").then(() => {
        if (networkRef.current === network) network.redraw();
      });
    }

    network.on("click", (params) => {
      if (params.nodes.length > 0) {
        const raw = nodesDataSet.get(params.nodes[0])?.raw;
        if (raw?.node_kind === "chunk") focusNode(raw.id);
      } else if (params.edges.length === 0) {
        clearHighlight(network, nodesDataSet, edgesDataSet);
        setHighlightActive(false);
      }
    });

    network.on("hoverNode", (params) => {
      hoveredNodeRef.current = nodesDataSet.get(params.node)?.raw ?? null;
    });
    network.on("blurNode", () => {
      hoveredNodeRef.current = null;
      setTooltip(null);
    });

    // Raw viewport coordinates (NodeTooltip is `position: fixed`) -
    // simpler and more faithful to the original Cytoscape `mousemove`
    // handler's "follow the cursor while hovering" behavior than
    // converting through vis-network's canvas coordinate space; this
    // also naturally keeps the tooltip correct while a node is being
    // dragged, since dragging fires the same native mouse movement.
    const container = containerRef.current;
    function handleMouseMove(event) {
      if (!hoveredNodeRef.current) return;
      setTooltip({ x: event.clientX, y: event.clientY, node: hoveredNodeRef.current });
    }
    container.addEventListener("mousemove", handleMouseMove);

    return () => {
      container.removeEventListener("mousemove", handleMouseMove);
      network.destroy();
      networkRef.current = null;
      nodesDataSetRef.current = null;
      edgesDataSetRef.current = null;
      hoveredNodeRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  // Re-fetch the blast radius and animate it in whenever a citation or a
  // node click focuses a new node - `focusRequestId` (not just
  // `focusNodeId`) is the effect key so re-clicking the same node still
  // re-triggers the animation (see state/RepoContext.jsx's own doc).
  useEffect(() => {
    if (!focusNodeId || !repoId || !networkRef.current) return;
    let cancelled = false;

    getGraph(repoId, focusNodeId, DEFAULT_BLAST_RADIUS_HOPS)
      .then((subgraph) => {
        if (cancelled || !networkRef.current) return;
        animateBlastRadius(networkRef.current, nodesDataSetRef.current, edgesDataSetRef.current, subgraph, focusNodeId);
        setHighlightActive(true);
      })
      .catch(() => {
        // A stale/removed focus node (e.g. from a previous repo) - nothing to highlight.
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRequestId, repoId]);

  function handleClearHighlight() {
    if (networkRef.current) clearHighlight(networkRef.current, nodesDataSetRef.current, edgesDataSetRef.current);
    setHighlightActive(false);
  }

  const emptyStateClasses = "flex h-full w-full items-center justify-center bg-[#0B0F19] text-sm text-slate-500";

  if (!repoId) {
    return <div className={emptyStateClasses}>Index a repository to see its call graph.</div>;
  }
  if (graphLoading) {
    return <div className={emptyStateClasses}>Loading graph…</div>;
  }
  if (graphError) {
    return <div className={`${emptyStateClasses} text-red-400`}>{graphError}</div>;
  }
  if (!graph || graph.nodes.length === 0) {
    return <div className={emptyStateClasses}>This repository's graph is empty.</div>;
  }

  return (
    <div className="relative flex h-full w-full flex-col bg-[#0B0F19]">
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% 45%, rgba(99,102,241,0.16), transparent 65%)",
        }}
      />
      <div className="pointer-events-none absolute inset-x-4 top-3 z-10 flex items-center justify-between">
        <span className="rounded-md bg-slate-950/70 px-2 py-1 font-mono text-xs text-slate-400">
          {graph.nodes.length} nodes · {graph.edges.length} edges
        </span>
        {highlightActive && (
          <button
            type="button"
            onClick={handleClearHighlight}
            className="pointer-events-auto rounded-md border border-slate-700 bg-slate-900 px-3 py-1 text-xs text-slate-200 transition hover:border-amber-400/60 hover:text-amber-300"
          >
            Clear highlight
          </button>
        )}
      </div>
      <div ref={containerRef} className="relative h-full w-full" />
      {tooltip && <NodeTooltip x={tooltip.x} y={tooltip.y} node={tooltip.node} />}
    </div>
  );
}
