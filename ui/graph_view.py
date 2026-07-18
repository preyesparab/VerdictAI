"""Streamlit call-graph visualization for RepoMind (Phase 21).

Renders the node-link JSON `GET /repos/{repo_id}/graph` already returns
(unchanged since Phase 20 - `database.graph_store`'s exact node-link
format, confirmed against a live response before writing this: chunk
nodes carry `node_kind`, `chunk_id`, `file_path`, `language`,
`chunk_type`, `function_name`, `class_name`, `parent_class`,
`start_line`, `end_line`; file nodes carry `node_kind`, `file_id`,
`file_path`, `language`; edges carry `source`, `target`, `edge_type`) as
an interactive graph, via `streamlit-agraph`.

Chosen over pyvis and a plain networkx+matplotlib spring layout for one
reason neither alternative offers out of the box: `agraph()` returns the
id of whichever node the user clicks, directly as a Python value -
matplotlib produces a static image with no click events at all, and
pyvis's generated HTML has no built-in bridge back into Streamlit's
session state (you'd have to hand-roll a custom component). Per-file
clustering rides vis.js's native `group` attribute - nodes sharing a
`group` are colored alike automatically - so the physics-based layout
visually pulls a file's functions/classes together without any custom
clustering logic here.

Also renders blast-radius highlighting (Phase 22): given a focus node
and a hop count, `GET /repos/{repo_id}/graph?focus_node=...&hops=...`
(same endpoint, same response shape - see `graph.blast_radius`) returns
just that node's N-hop neighborhood, rendered as its own small graph
with the focus node and its neighbors made visually distinct via a
border/shadow treatment layered on top of the existing per-file `group`
fill color - not a fill-color override, since that scheme already has
known collisions among 20+ files and a fill highlight would blend in or
conflict with an unrelated file's color. The manual node-picker
(`render_blast_radius_section`) is a placeholder trigger to verify the
mechanism end to end; Adjudicate will call the same endpoint
automatically once a flagged node id is known, with no rendering changes
needed here.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from ui.api_client import APIClient

# vis.js edge color/width by the edge_type `graph.graph_builder.RepositoryGraphBuilder`
# assigns: calls/inheritance are the relationships worth reading a codebase by, so they're
# drawn bolder; containment/imports are structural scaffolding, drawn lighter so they don't
# dominate the picture.
_EDGE_STYLE: dict[str, dict[str, Any]] = {
    "function_call": {"color": "#4C78A8", "width": 1.5},
    "method_call": {"color": "#4C78A8", "width": 1.5},
    "inherits": {"color": "#9D5BD2", "width": 2, "dashes": True},
    "imports": {"color": "#B0B0B0", "width": 1, "dashes": True},
    "contains": {"color": "#E0E0E0", "width": 0.5},
}

_FILE_NODE_COLOR = "#2E2E2E"

# Blast-radius highlight (Phase 22): border/shadow only, layered on top
# of `_build_node`'s existing per-file `group` fill color - never an
# override, per the file-color-collision caveat in the module docstring.
_FOCUS_BORDER_WIDTH = 6
_FOCUS_SHADOW_COLOR = "#FFD700"
_NEIGHBOR_BORDER_WIDTH = 3
_DEFAULT_HOPS = 2


def render_graph_tab(service: APIClient, repository_id: str) -> None:
    """Render the whole-repository call graph, with basic info shown for a clicked node.

    Args:
        service: The API client to fetch the graph through (reuses the
            existing `GET /repos/{repo_id}/graph` endpoint - no new API
            call path).
        repository_id: The indexed repository to render.
    """
    try:
        with st.spinner("Loading graph..."):
            graph_data = service.get_graph(repository_id)
    except Exception as exc:  # noqa: BLE001 - UI error boundary, matches app.py's convention
        st.error(f"Could not load the graph: {exc}")
        return

    raw_nodes = graph_data.get("nodes", [])
    raw_edges = graph_data.get("edges", [])
    st.caption(f"{len(raw_nodes)} node(s), {len(raw_edges)} edge(s) — drag to explore, click a node for details.")

    if not raw_nodes:
        st.info("This repository's graph is empty.")
        return

    nodes_by_id = {node["id"]: node for node in raw_nodes}
    nodes = [_build_node(node) for node in raw_nodes]
    edges = [
        Edge(source=edge["source"], target=edge["target"], **_EDGE_STYLE.get(edge.get("edge_type"), {}))
        for edge in raw_edges
        if edge["source"] in nodes_by_id and edge["target"] in nodes_by_id
    ]

    # `groups={}` avoids vis.js's own validator rejecting `Config`'s default
    # `groups: None` (a real console error observed during manual verification) -
    # per-node `group=file_path` (set in `_build_node`) still drives automatic
    # per-file coloring on its own; this only supplies the (empty) network-level
    # override dict vis.js expects instead of `null`.
    config = Config(height=700, width=1000, directed=True, physics=True, hierarchical=False, groups={})
    selected_id = agraph(nodes=nodes, edges=edges, config=config)

    if selected_id:
        _render_node_info(nodes_by_id.get(selected_id))
    else:
        st.caption("Click a node above to see its function/class name, file path, and line range.")

    st.divider()
    _render_blast_radius_section(service, repository_id, raw_nodes)


def _build_node(node: dict[str, Any]) -> Node:
    """Convert one `/graph` node dict into a `streamlit_agraph.Node`.

    File nodes get a distinct fixed color/shape so they read as
    structural anchors, not code units. Chunk nodes are colored by
    `group=file_path` (not given an explicit `color`) so vis.js's
    automatic per-group coloring visually clusters each file's
    functions/classes together.
    """
    if node.get("node_kind") == "file":
        return Node(
            id=node["id"], label=node["file_path"].rsplit("/", 1)[-1],
            title=node["file_path"], shape="square", size=18, color=_FILE_NODE_COLOR,
            group=node["file_path"],
        )

    label = node.get("function_name") or node.get("class_name") or node.get("chunk_type", "chunk")
    size = 14 if node.get("chunk_type") == "class" else 10
    return Node(
        id=node["id"], label=label, title=f"{node.get('file_path')}:{node.get('start_line')}",
        shape="dot", size=size, group=node.get("file_path"),
    )


def _render_node_info(node: dict[str, Any] | None) -> None:
    """Render a selected node's function/class name, file path, and line range."""
    if node is None:
        st.warning("Selected node not found in the loaded graph.")
        return

    if node.get("node_kind") == "file":
        st.info(f"**File:** `{node['file_path']}`  ·  **Language:** {node.get('language', 'unknown')}")
        return

    name = node.get("function_name") or node.get("class_name") or "(unnamed)"
    columns = st.columns(4)
    columns[0].metric("Name", name)
    columns[1].write(f"**File**\n\n`{node.get('file_path')}`")
    columns[2].write(f"**Type**\n\n{node.get('chunk_type')}")
    columns[3].write(f"**Lines**\n\n{node.get('start_line')}–{node.get('end_line')}")
    if node.get("class_name") and node.get("function_name"):
        st.caption(f"Method of class `{node['class_name']}`")


def _chunk_label(node: dict[str, Any]) -> str:
    """Build a disambiguated dropdown label for one chunk node: ``name (file_path)``.

    Plain function/class names collide often across files (e.g. many
    files each define their own ``handleSubmit``) - the file path makes
    each dropdown entry unambiguous.
    """
    name = node.get("function_name") or node.get("class_name") or node.get("chunk_type", "chunk")
    return f"{name}  ({node.get('file_path')})"


def _apply_blast_radius_highlight(node_obj: Node, *, is_focus: bool) -> Node:
    """Layer a border/shadow highlight onto an already-built `Node`.

    Deliberately does not touch `.color`/`.group` - the existing
    per-file fill-color scheme (`_build_node`) already has known
    collisions among 20+ files, so a fill-color highlight could blend in
    with, or be confused for, an unrelated file's color. `borderWidth`
    and `shadow` are independent vis.js Node options that layer on top
    of whatever fill color `group` already assigned.

    Args:
        node_obj: A `Node` already built by `_build_node`.
        is_focus: True for the blast radius's center node (thickest
            border plus a gold shadow/glow); False for one of its
            N-hop neighbors (a thinner, but still elevated, border).

    Returns:
        `node_obj`, mutated in place, for convenient chaining.
    """
    if is_focus:
        node_obj.borderWidth = _FOCUS_BORDER_WIDTH
        node_obj.shadow = {"enabled": True, "color": _FOCUS_SHADOW_COLOR, "size": 20}
    else:
        node_obj.borderWidth = _NEIGHBOR_BORDER_WIDTH
    return node_obj


def _render_blast_radius_section(
    service: APIClient, repository_id: str, raw_nodes: list[dict[str, Any]]
) -> None:
    """Manual trigger for Phase 22 blast-radius highlighting.

    Placeholder UX, not the final one: Adjudicate will eventually call
    `GET /repos/{repo_id}/graph?focus_node=...&hops=...` automatically
    once it already knows a flagged node's id, with no rendering changes
    needed here. This search/dropdown + hop count + button is just
    enough to verify that mechanism (endpoint, traversal, and highlight
    rendering) works end to end.

    Args:
        service: The API client the full graph was already fetched
            through - reused for the blast-radius request too.
        repository_id: The indexed repository being visualized.
        raw_nodes: The full graph's node list (already fetched by
            `render_graph_tab`), used to populate the picker - avoids a
            second full-graph fetch just to build the dropdown.
    """
    st.subheader("Blast radius")
    chunk_nodes = [node for node in raw_nodes if node.get("node_kind") == "chunk"]
    if not chunk_nodes:
        st.caption("No functions/classes to pick from.")
        return

    id_by_label = {_chunk_label(node): node["id"] for node in chunk_nodes}
    picked_label = st.selectbox(
        "Pick a function or class to see what depends on it",
        sorted(id_by_label),
        index=None,
        placeholder="Search by name...",
    )
    hops = st.number_input("Hops", min_value=0, max_value=10, value=_DEFAULT_HOPS, step=1)

    if not picked_label:
        st.caption("Select a node above, then click below to render its blast radius.")
        return

    if not st.button("Show blast radius"):
        return

    focus_node_id = id_by_label[picked_label]
    try:
        with st.spinner("Computing blast radius..."):
            subgraph_data = service.get_graph(repository_id, focus_node=focus_node_id, hops=int(hops))
    except Exception as exc:  # noqa: BLE001 - UI error boundary, matches render_graph_tab's convention
        st.error(f"Could not compute blast radius: {exc}")
        return

    sub_nodes_raw = subgraph_data.get("nodes", [])
    sub_edges_raw = subgraph_data.get("edges", [])
    st.caption(
        f"{len(sub_nodes_raw)} node(s), {len(sub_edges_raw)} edge(s) within {int(hops)} "
        f"hop(s) of **{picked_label}**."
    )

    sub_nodes_by_id = {node["id"]: node for node in sub_nodes_raw}
    sub_nodes = [
        _apply_blast_radius_highlight(_build_node(node), is_focus=(node["id"] == focus_node_id))
        for node in sub_nodes_raw
    ]
    sub_edges = [
        Edge(source=edge["source"], target=edge["target"], **_EDGE_STYLE.get(edge.get("edge_type"), {}))
        for edge in sub_edges_raw
        if edge["source"] in sub_nodes_by_id and edge["target"] in sub_nodes_by_id
    ]

    config = Config(height=500, width=1000, directed=True, physics=True, hierarchical=False, groups={})
    agraph(nodes=sub_nodes, edges=sub_edges, config=config)
