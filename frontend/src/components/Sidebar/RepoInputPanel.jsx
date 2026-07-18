import { useEffect, useState } from "react";
import { Alert } from "../common/Alert";
import { RepoIcon } from "../common/icons";

/**
 * GitHub URL input + trigger, plus the repo summary once indexed.
 * `indexing` carries `{ phase, stage, error, repoId, summary }` from
 * `useIndexingStatus` - this component only renders it, `IndexingProgress`
 * (a sibling) renders the animated stage checklist.
 *
 * `prefillUrl` is optional and additive: when set (by LandingView's
 * example-repo chips), it's copied into the field's own local `url`
 * state via an effect - the field is still fully editable afterward,
 * this only seeds it. `Sidebar`'s own usage never passes this prop, so
 * this is a no-op there - zero behavior change to the existing
 * submit/validation/indexing flow either way.
 *
 * `size` ("sm" default, "lg" for LandingView's hero-scale pass) only
 * swaps className strings for the input/button - the form element, its
 * `onSubmit`, and every validation/disabled rule are identical in both
 * sizes. `Sidebar` never passes this prop, so its compact rendering is
 * byte-for-byte unchanged.
 */
export function RepoInputPanel({ indexing, onSubmit, prefillUrl, size = "sm" }) {
  const [url, setUrl] = useState("");
  const busy = indexing.phase === "indexing";
  const isLarge = size === "lg";

  useEffect(() => {
    if (prefillUrl !== undefined) setUrl(prefillUrl);
  }, [prefillUrl]);

  function handleSubmit(event) {
    event.preventDefault();
    if (!url.trim() || busy) return;
    onSubmit(url.trim());
  }

  return (
    <div className="flex flex-col gap-3">
      <form onSubmit={handleSubmit} className={isLarge ? "flex gap-3" : "flex gap-2"}>
        <input
          type="text"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          disabled={busy}
          className={
            isLarge
              ? "min-w-0 flex-1 rounded-xl border border-white/10 bg-slate-950/50 px-5 py-4 text-lg text-white placeholder:text-slate-400 transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50"
              : "min-w-0 flex-1 rounded-md border border-white/10 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-400 transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50"
          }
        />
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className={
            isLarge
              ? "whitespace-nowrap rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-7 py-4 text-lg font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:scale-105 hover:brightness-110 disabled:cursor-default disabled:opacity-40 disabled:hover:scale-100 disabled:hover:brightness-100"
              : "whitespace-nowrap rounded-md bg-gradient-to-r from-indigo-500 to-purple-600 px-3 py-2 text-sm font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:scale-105 hover:brightness-110 disabled:cursor-default disabled:opacity-40 disabled:hover:scale-100 disabled:hover:brightness-100"
          }
        >
          {busy ? "Indexing…" : "Index"}
        </button>
      </form>

      {indexing.phase === "ready" && indexing.summary && (
        <div className="rounded-lg border border-white/10 bg-gradient-to-br from-indigo-500/10 via-slate-900/50 to-slate-900/50 p-3.5 backdrop-blur-xl">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-white">
            <RepoIcon className="h-3.5 w-3.5 flex-shrink-0 text-indigo-400" />
            <span className="truncate">
              {indexing.summary.owner}/{indexing.summary.name}
            </span>
          </div>
          <div className="mt-2.5 flex flex-col gap-1 font-mono text-xs text-slate-400">
            <span>{indexing.summary.files_discovered} files</span>
            <span>{indexing.summary.chunks_indexed} chunks</span>
            <span>
              {indexing.summary.graph_nodes}/{indexing.summary.graph_edges} nodes/edges
            </span>
          </div>
        </div>
      )}

      {indexing.phase === "failed" && <Alert>{indexing.error}</Alert>}
    </div>
  );
}
