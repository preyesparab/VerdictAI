/** The real Context Builder output (Phase 24) - what the rest of the review is grounded in. */
export function ContextCard({ context }) {
  return (
    <div className="animate-rm-card-in rounded-lg border border-white/10 bg-slate-900/40 p-4 backdrop-blur-xl">
      <h3 className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-500">Context gathered</h3>
      <div className="mt-2 flex flex-col gap-1.5">
        {context.changed_functions.map((fn, i) => (
          <div key={i} className="font-mono text-xs text-slate-300">
            <span className="text-indigo-400">{fn.function_name || fn.class_name || "(module-level)"}</span>{" "}
            <span className="text-slate-500">
              {fn.file_path}:{fn.start_line}-{fn.end_line}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex gap-4 text-[11px] text-slate-500">
        <span>{context.callers.length} caller(s)</span>
        <span>{context.callees.length} callee(s)</span>
        <span>{context.related_tests.length} related test(s)</span>
      </div>
    </div>
  );
}
