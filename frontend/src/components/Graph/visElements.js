import { colorForFile, edgeStyleFor, fileNodeColor } from "./visStyle";

const CLASS_NODE_SIZE = 16;
const CHUNK_NODE_SIZE = 10;
const FILE_NODE_SIZE = 20;

/** Deterministic edge id (source/target/type) - vis-network edges need a stable id for DataSet updates. */
export function edgeElementId(edge) {
  return `${edge.source}->${edge.target}:${edge.edge_type}`;
}

function nodeLabel(node) {
  if (node.node_kind === "file") return node.file_path.split("/").pop();
  return node.function_name || node.class_name || node.chunk_type || "chunk";
}

function nodeSize(node) {
  if (node.node_kind === "file") return FILE_NODE_SIZE;
  return node.chunk_type === "class" ? CLASS_NODE_SIZE : CHUNK_NODE_SIZE;
}

/** Convert one `/graph` node into a vis-network DataSet node item. The full raw node is kept under `raw` for click/hover handlers, exactly as Cytoscape's `data.raw` did. */
export function toVisNode(node) {
  const isFile = node.node_kind === "file";
  return {
    id: node.id,
    label: nodeLabel(node),
    shape: isFile ? "square" : "dot",
    size: nodeSize(node),
    color: {
      background: isFile ? fileNodeColor() : colorForFile(node.file_path),
      border: isFile ? fileNodeColor() : colorForFile(node.file_path),
      highlight: { background: isFile ? fileNodeColor() : colorForFile(node.file_path) },
    },
    raw: node,
  };
}

export function toVisEdge(edge) {
  const style = edgeStyleFor(edge.edge_type);
  return {
    id: edgeElementId(edge),
    from: edge.source,
    to: edge.target,
    color: { color: style.color, opacity: 0.85 },
    width: style.width,
    dashes: style.dashes,
    edge_type: edge.edge_type,
  };
}

export function toVisElements(graph) {
  const nodeIds = new Set(graph.nodes.map((n) => n.id));
  return {
    nodes: graph.nodes.map(toVisNode),
    edges: graph.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target)).map(toVisEdge),
  };
}
