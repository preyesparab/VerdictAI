import { edgeElementId } from "./visElements";

const STAGGER_MS = 140;
const REVEAL_DURATION_MS = 320;
const DIM_OPACITY = 0.12;

/**
 * BFS hop distance from `focusId` over `edges`, treated as undirected -
 * matching `graph.blast_radius.compute_blast_radius`'s own bidirectional
 * traversal (callers/importers upstream, callees/imported code
 * downstream, both count as "1 hop away"). The endpoint itself returns
 * only the induced subgraph, no per-node distance - this is purely a
 * client-side animation-staging concern, computed over whatever subgraph
 * the server already returned.
 *
 * Unchanged since the Cytoscape version - this function has no
 * rendering-library dependency at all, pure data in, data out.
 */
export function computeHopTiers(nodes, edges, focusId) {
  const adjacency = new Map(nodes.map((n) => [n.id, []]));
  for (const edge of edges) {
    adjacency.get(edge.source)?.push(edge.target);
    adjacency.get(edge.target)?.push(edge.source);
  }

  const hopById = new Map([[focusId, 0]]);
  let frontier = [focusId];
  while (frontier.length > 0) {
    const next = [];
    for (const id of frontier) {
      const hop = hopById.get(id);
      for (const neighbor of adjacency.get(id) ?? []) {
        if (!hopById.has(neighbor)) {
          hopById.set(neighbor, hop + 1);
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  return hopById;
}

/** requestAnimationFrame-driven tween - vis-network has no built-in `.animate()` (unlike Cytoscape), so this replaces it. */
function tween(durationMs, onFrame, { delayMs = 0 } = {}) {
  const start = performance.now() + delayMs;
  function step(now) {
    if (now < start) {
      requestAnimationFrame(step);
      return;
    }
    const t = Math.min(1, (now - start) / durationMs);
    onFrame(t);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function tweenNodeOpacity(nodesDataSet, ids, from, to, durationMs, delayMs = 0) {
  if (ids.length === 0) return;
  tween(
    durationMs,
    (t) => {
      const opacity = lerp(from, to, t);
      nodesDataSet.update(ids.map((id) => ({ id, opacity })));
    },
    { delayMs },
  );
}

function tweenEdgeOpacity(edgesDataSet, ids, from, to, durationMs, delayMs = 0) {
  if (ids.length === 0) return;
  tween(
    durationMs,
    (t) => {
      const opacity = lerp(from, to, t);
      edgesDataSet.update(
        ids.map((id) => {
          const edge = edgesDataSet.get(id);
          return { id, color: { ...edge.color, opacity } };
        }),
      );
    },
    { delayMs },
  );
}

/** Reset every node/edge to its normal, non-highlighted appearance, fading back from whatever opacity it's currently at. */
export function clearHighlight(network, nodesDataSet, edgesDataSet) {
  const allNodeIds = nodesDataSet.getIds();
  const allEdgeIds = edgesDataSet.getIds();

  // Capture starting opacities once, before the tween begins - reading
  // them fresh on every frame would lerp from an already-partially-
  // faded value each frame instead of the true start, compounding
  // incorrectly instead of interpolating smoothly.
  const startNodeOpacity = new Map(allNodeIds.map((id) => [id, nodesDataSet.get(id).opacity ?? 1]));
  const startEdgeOpacity = new Map(allEdgeIds.map((id) => [id, edgesDataSet.get(id).color?.opacity ?? 0.85]));

  tween(200, (t) => {
    nodesDataSet.update(allNodeIds.map((id) => ({ id, opacity: lerp(startNodeOpacity.get(id), 1, t) })));
    edgesDataSet.update(
      allEdgeIds.map((id) => {
        const edge = edgesDataSet.get(id);
        return { id, color: { ...edge.color, opacity: lerp(startEdgeOpacity.get(id), 0.85, t) } };
      }),
    );
  });

  nodesDataSet.update(allNodeIds.map((id) => ({ id, borderWidth: 0, shadow: false })));
}

/**
 * Dim everything outside the blast radius, then reveal the focus node and
 * its neighbors hop-by-hop with a fade + brief highlight glow (a border/
 * shadow pulse, vis-network's equivalent of Cytoscape's `overlay-opacity`)
 * instead of snapping the highlight in instantly - and smoothly pans/
 * zooms to fit the highlighted set via vis-network's own animated `fit`.
 *
 * Args:
 *   network: The live vis-network `Network` instance (the full graph is
 *     already rendered in it - this only restyles existing DataSet
 *     items, it does not add or remove any).
 *   nodesDataSet: The `vis-data` `DataSet` backing `network`'s nodes.
 *   edgesDataSet: Same, for edges.
 *   subgraph: The blast-radius response (`{ nodes, edges }`) from
 *     `GET /repos/{repo_id}/graph?focus_node=...&hops=...`.
 *   focusId: The focus node's id.
 */
export function animateBlastRadius(network, nodesDataSet, edgesDataSet, subgraph, focusId) {
  const hopById = computeHopTiers(subgraph.nodes, subgraph.edges, focusId);
  const nodeIds = new Set(subgraph.nodes.map((n) => n.id));
  const edgeIds = new Set(subgraph.edges.map(edgeElementId));

  const allNodeIds = nodesDataSet.getIds();
  const allEdgeIds = edgesDataSet.getIds();
  const dimmedNodeIds = allNodeIds.filter((id) => !nodeIds.has(id));
  const dimmedEdgeIds = allEdgeIds.filter((id) => !edgeIds.has(id));

  tweenNodeOpacity(nodesDataSet, dimmedNodeIds, 1, DIM_OPACITY, 250);
  tweenEdgeOpacity(edgesDataSet, dimmedEdgeIds, 0.85, DIM_OPACITY, 250);

  const focusHex = getFocusColorFromDom();
  nodesDataSet.update(
    [...nodeIds].map((id) => ({
      id,
      borderWidth: id === focusId ? 6 : 3,
      color: { ...nodesDataSet.get(id).color, border: focusHex },
      shadow: { enabled: true, color: focusHex, size: id === focusId ? 20 : 10, x: 0, y: 0 },
    })),
  );

  network.fit({ nodes: [...nodeIds], animation: { duration: 500, easingFunction: "easeOutCubic" } });

  const maxHop = Math.max(...Array.from(hopById.values()));
  for (let hop = 0; hop <= maxHop; hop++) {
    const tierNodeIds = [...hopById.entries()].filter(([, h]) => h === hop).map(([id]) => id);
    const tierEdgeIds = [...edgeIds].filter((id) => {
      const edge = edgesDataSet.get(id);
      return tierNodeIds.includes(edge.from) || tierNodeIds.includes(edge.to);
    });
    const delay = hop * STAGGER_MS;

    nodesDataSet.update(tierNodeIds.map((id) => ({ id, opacity: 0 })));
    tweenNodeOpacity(nodesDataSet, tierNodeIds, 0, 1, REVEAL_DURATION_MS, delay);

    edgesDataSet.update(tierEdgeIds.map((id) => ({ id, color: { ...edgesDataSet.get(id).color, opacity: 0 } })));
    tweenEdgeOpacity(edgesDataSet, tierEdgeIds, 0, 0.9, REVEAL_DURATION_MS, delay);
  }
}

function getFocusColorFromDom() {
  return getComputedStyle(document.documentElement).getPropertyValue("--color-focus").trim();
}
