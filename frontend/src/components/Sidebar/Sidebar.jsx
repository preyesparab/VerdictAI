import { IndexingProgress } from "./IndexingProgress";
import { RepoInputPanel } from "./RepoInputPanel";

/**
 * Fixed-width left panel: brand + repo input/status - always visible
 * once the workspace layout is showing (see AppShell.jsx for the
 * landing/workspace transition). `indexing` is lifted to AppShell and
 * passed down as a prop rather than called here, since
 * `IndexingProgress`/`RepoInputPanel` also need to render on the landing
 * page (LandingView.jsx) driven by the exact same state - one hook
 * instance, two render sites.
 *
 * `onGoHome`: clicking the "Verdict AI" brand resets back to the
 * landing state, same as a fresh page load - see AppShell.jsx's
 * `goHome()`, which pairs `useIndexingStatus`'s `reset()` with
 * `RepoContext`'s `resetRepo()`.
 *
 * Chat and "Review a PR" moved out of this sidebar into the top-level
 * `ViewSwitcher`/Chat+Review views (Phase 32 Part 2 nav restructure) -
 * this component is now Repository-status-only, no `chat`/`nodesById`/
 * `activeView` props needed here anymore.
 */
export function Sidebar({ indexing, onGoHome }) {
  return (
    <aside className="flex h-full w-[300px] min-w-[300px] flex-col gap-5 overflow-hidden border border-white/10 bg-slate-900/40 p-4 shadow-2xl backdrop-blur-xl">
      <button
        type="button"
        onClick={onGoHome}
        title="Back to landing"
        className="flex-shrink-0 self-start border-b border-white/10 pb-4 text-lg font-bold tracking-tight text-white transition hover:text-indigo-300 hover:underline hover:decoration-indigo-400/60 hover:underline-offset-4"
      >
        Verdict AI
      </button>

      <div className="flex flex-shrink-0 flex-col gap-3">
        <h2 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Repository</h2>
        <RepoInputPanel indexing={indexing} onSubmit={indexing.start} />
        <IndexingProgress indexing={indexing} />
      </div>
    </aside>
  );
}
