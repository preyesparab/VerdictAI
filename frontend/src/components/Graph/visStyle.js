/**
 * vis-network styling + physics config for the graph view.
 *
 * Colors are read from the same CSS custom properties `tokens.css`
 * already defines (single source of truth shared with the rest of the
 * app's Tailwind palette) via `getComputedStyle` - vis-network renders
 * to a <canvas>, same constraint Cytoscape had, so it never resolves
 * CSS custom properties on its own either.
 *
 * Edge colors/widths are the same mapping `ui/graph_view.py`'s
 * `_EDGE_STYLE` established and the Cytoscape version (`graphStyle.js`,
 * now removed) carried forward: calls drawn bolder, containment/imports
 * lighter. "references" stays unstyled (falls to the default edge color).
 *
 * Physics: below `PHYSICS_SCALE_NODE_THRESHOLD` nodes, every value is
 * vis.js's own real BarnesHut default, byte-for-byte - this is
 * deliberately the exact configuration `ui/graph_view.py`'s
 * `streamlit_agraph.Config(physics=True, ...)` already used, validated
 * as feeling right on repos like NutriForge (227 nodes). Measured
 * directly (not assumed) that this default configuration never
 * converges at all on a larger, similarly-dense repo (pawn_ai, 657
 * nodes/802 edges): logging the real `stabilizationIterationsDone` vs.
 * `stabilized` events showed the 200-iteration cap is reached without
 * `stabilized` ever firing, even after raising the cap to 2000
 * iterations and watching for another 3 minutes - confirming this is a
 * genuine non-convergence (persistent low-level oscillation from more
 * total system energy at higher node count), not just "needs more time"
 * at the same damping. `computePhysicsScale` below scales
 * `stabilization.iterations`/`damping`/`minVelocity` up only once node
 * count exceeds the threshold NutriForge sits comfortably under - so
 * every repo at or below that size gets the exact previously-validated
 * feel, unchanged, and only larger repos get the adjustment needed to
 * actually settle. The one other deliberate addition, independent of
 * this scaling, is `avoidOverlap`, left at 0 in vis.js's own default but
 * raised here to reduce label-obscuring overlap, a readability fix, not
 * a physics-feel change.
 */

// NutriForge-sized repos and smaller get vis.js's literal defaults
// (unchanged); scaling kicks in above this so there is zero behavior
// change for every graph size already validated as feeling right.
const PHYSICS_SCALE_NODE_THRESHOLD = 250;

const BASE_STABILIZATION_ITERATIONS = 200;
const BASE_DAMPING = 0.09; // vis.js's own barnesHut default
const BASE_MIN_VELOCITY = 0.75; // vis.js's own default

/**
 * Scale stabilization budget/damping/minVelocity by node count, above
 * `PHYSICS_SCALE_NODE_THRESHOLD` only. Values below the threshold are
 * untouched vis.js defaults; above it, more nodes get proportionally
 * more iterations to settle in, faster velocity decay (damping), and a
 * slightly more forgiving "close enough to at rest" threshold
 * (minVelocity) - all capped so an extreme node count still stabilizes
 * in a bounded number of iterations rather than growing unbounded.
 */
export function computePhysicsScale(nodeCount) {
  const over = Math.max(0, nodeCount - PHYSICS_SCALE_NODE_THRESHOLD);
  return {
    iterations: Math.min(BASE_STABILIZATION_ITERATIONS + over * 4, 2500),
    damping: Math.min(BASE_DAMPING + over * 0.0004, 0.3),
    minVelocity: Math.min(BASE_MIN_VELOCITY + over * 0.0015, 1.5),
  };
}

const FILE_COLOR_VAR_COUNT = 12;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Deterministic file_path -> palette color, keyed by up to the first
 * *two* path segments (e.g. `engine/core/kernel.py` and
 * `engine/core/params.py` both key on `"engine/core"`) rather than just
 * the top-level folder alone.
 *
 * A pure top-level-only key (tried first, see this repo's own
 * `docs/state/PROGRESS.md`) reads as useless on repos where one folder
 * holds most of the codebase - confirmed on a real repo
 * (`harshal31718/enma_trading_platform`): `engine/` alone was 811 of
 * 1218 real chunk nodes (66%), so nearly the whole graph rendered in
 * one color regardless of that folder's own real substructure
 * (`engine/core`, `engine/indicators`, `engine/routers`, ...).
 *
 * Using only the *second* segment instead (e.g. `"core"`, `"judges"`)
 * was considered and rejected: measured against another already-working
 * repo (NutriForge - `client/src/...`, `server/src/...`,
 * `ai_service/routers/...`), every one of its top-level folders nests
 * under its own `src/`, so keying on the second segment alone would
 * collapse `client`/`server` back into one shared `"src"` bucket -
 * a regression on a case already confirmed working. Keying on the first
 * *two* segments together avoids this: `client/src` and `server/src`
 * still differ (their first segment differs), while `engine/core` and
 * `engine/indicators` also now differ (their second segment differs) -
 * both repos get real, useful grouping this way.
 *
 * A path with only one real segment beyond the top folder (e.g.
 * `engine/kernel.py` - the "second segment" is just the filename, not a
 * real subfolder) keys on the top-level folder alone, same as before -
 * there's no real subfolder to subdivide by. Root-level files (no `/`
 * at all) get their own stable bucket.
 */
export function colorForFile(filePath) {
  const segments = filePath.split("/");
  let key;
  if (segments.length <= 1) {
    key = "\0root\0";
  } else if (segments.length === 2) {
    key = segments[0];
  } else {
    key = `${segments[0]}/${segments[1]}`;
  }
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  return cssVar(`--color-file-${hash % FILE_COLOR_VAR_COUNT}`);
}

const EDGE_STYLE_BY_TYPE = {
  function_call: { color: () => cssVar("--color-accent"), width: 1.5, dashes: false },
  method_call: { color: () => cssVar("--color-accent"), width: 1.5, dashes: false },
  inherits: { color: () => cssVar("--color-accent-secondary"), width: 2, dashes: true },
  imports: { color: () => cssVar("--color-text-muted"), width: 1, dashes: true },
  contains: { color: () => cssVar("--color-border"), width: 0.75, dashes: false },
};

export function edgeStyleFor(edgeType) {
  const spec = EDGE_STYLE_BY_TYPE[edgeType];
  const defaultColor = cssVar("--color-border-muted");
  if (!spec) return { color: defaultColor, width: 0.75, dashes: false };
  return { color: spec.color(), width: spec.width, dashes: spec.dashes };
}

export function focusColor() {
  return cssVar("--color-focus");
}

export function fileNodeColor() {
  return cssVar("--color-node-file");
}

export function textColor() {
  return cssVar("--color-text");
}

/**
 * vis-network `options` - physics + interaction + rendering, no
 * manipulation/navigation UI.
 *
 * @param {number} nodeCount - Node count of the graph about to be
 *   rendered, used only to scale the physics stabilization budget (see
 *   `computePhysicsScale`) - 0 (vis.js's untouched defaults) if omitted.
 */
export function buildNetworkOptions(nodeCount = 0) {
  const scale = computePhysicsScale(nodeCount);
  return {
    autoResize: true,
    interaction: {
      hover: true,
      tooltipDelay: 100000, // disable vis-network's own basic title tooltip - NodeTooltip.jsx renders a richer one
      zoomView: true,
      dragView: true,
      dragNodes: true,
    },
    physics: {
      enabled: true,
      solver: "barnesHut",
      barnesHut: {
        // gravitationalConstant/centralGravity/springLength unchanged
        // from vis.js's real defaults at every graph size - only
        // damping is scaled (see `computePhysicsScale`/module docstring).
        gravitationalConstant: -2000,
        centralGravity: 0.3,
        springLength: 95,
        springConstant: 0.04,
        damping: scale.damping,
        avoidOverlap: 0.6,
      },
      minVelocity: scale.minVelocity,
      stabilization: { enabled: true, iterations: scale.iterations, fit: true },
      adaptiveTimestep: true,
    },
    nodes: {
      shape: "dot",
      font: { color: textColor(), size: 9, face: "Inter, sans-serif" },
      borderWidth: 0,
      shadow: false,
    },
    edges: {
      smooth: { enabled: true, type: "continuous", roundness: 0.4 },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      shadow: false,
    },
    layout: { improvedLayout: true },
  };
}
