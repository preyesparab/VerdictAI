import { useEffect, useState } from "react";
import { INDEXING_STAGES } from "../../constants";
import { CheckIcon, SpinnerIcon } from "../common/icons";

/**
 * Step-by-step indexing checklist driven entirely by the real `stage`
 * string `GET /repos/{repo_id}/status` reports (relaying
 * `Pipeline.index_repository`'s own `on_progress` callback) - not a
 * generic spinner. A stage is "done" once a later stage has been
 * reported (the backend does not emit a distinct per-stage "complete"
 * event through polling, only the current running stage - see
 * `api/main.py`'s `_run_indexing`), "active" while it's the current
 * `stage`, "pending" otherwise.
 *
 * Collapses to a single muted summary line once `indexing.phase ===
 * "ready"` (a local, presentational-only `collapsed` state - no change
 * to how `indexing.stage`/`.phase` themselves are computed), expandable
 * again on click, so the completed checklist stops visually competing
 * with the chat section below it once it's no longer the active focus.
 */
export function IndexingProgress({ indexing }) {
  const allDone = indexing.phase === "ready";
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (allDone) setCollapsed(true);
  }, [allDone]);

  if (indexing.phase === "idle") return null;

  const currentIndex = indexing.stage ? INDEXING_STAGES.indexOf(indexing.stage) : -1;

  if (allDone && collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="flex w-full items-center gap-2 rounded-md border border-white/5 bg-slate-900/20 px-3 py-1.5 text-left text-[11px] text-slate-500 transition hover:border-white/10 hover:text-slate-300"
      >
        <CheckIcon className="h-3 w-3 flex-shrink-0 text-emerald-500/60" />
        <span>Indexing complete</span>
        <span className="ml-auto text-slate-600">Show steps</span>
      </button>
    );
  }

  return (
    <ol className="flex flex-col gap-2 rounded-md border border-white/10 bg-slate-900/40 p-3 backdrop-blur-xl">
      {allDone && (
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          className="mb-0.5 self-end text-[10px] text-slate-500 transition hover:text-slate-300"
        >
          Collapse
        </button>
      )}
      {INDEXING_STAGES.map((stage, i) => {
        const status = allDone || i < currentIndex ? "done" : i === currentIndex ? "active" : "pending";
        return (
          <li key={stage} className="flex items-center gap-2.5 text-xs">
            <span
              className={
                "flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full " +
                (status === "done"
                  ? "bg-emerald-500/20 text-emerald-400"
                  : status === "active"
                    ? "text-indigo-400 shadow-[0_0_8px_2px_rgba(99,102,241,0.6)]"
                    : "bg-slate-800 text-slate-600")
              }
            >
              {status === "done" && <CheckIcon className="h-3 w-3" />}
              {status === "active" && <SpinnerIcon className="h-3.5 w-3.5" />}
              {status === "pending" && <span className="h-1.5 w-1.5 rounded-full bg-slate-600" />}
            </span>
            <span
              className={
                status === "pending"
                  ? "text-slate-500"
                  : status === "active"
                    ? "font-semibold text-white"
                    : "text-slate-400"
              }
            >
              {stage}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
