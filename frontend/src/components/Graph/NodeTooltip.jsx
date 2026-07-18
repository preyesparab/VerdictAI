/** Subtle hover tooltip: name/file/line, shown before a full click (Phase 32 brief). */
export function NodeTooltip({ x, y, node }) {
  if (!node) return null;

  const isFile = node.node_kind === "file";
  const name = isFile ? node.file_path.split("/").pop() : node.function_name || node.class_name || node.chunk_type;

  return (
    <div
      className="pointer-events-none fixed z-20 max-w-xs rounded-md border border-slate-700 bg-slate-900 px-3 py-2 shadow-xl shadow-black/40"
      style={{ left: x + 14, top: y + 14 }}
    >
      <div className="text-xs font-semibold text-slate-100">{name}</div>
      <div className="mt-0.5 font-mono text-[11px] text-slate-400">
        {node.file_path}
        {!isFile && node.start_line ? `:${node.start_line}` : ""}
      </div>
    </div>
  );
}
