import { useEffect, useState } from "react";
import { useChat } from "../../hooks/useChat";
import { useIndexingStatus } from "../../hooks/useIndexingStatus";
import { useRepo } from "../../state/RepoContext";
import { ChatView } from "../Chat/ChatView";
import { GraphView } from "../Graph/GraphView";
import { ReviewView } from "../Review/ReviewView";
import { Sidebar } from "../Sidebar/Sidebar";
import { LandingView } from "./LandingView";
import { ViewSwitcher } from "./ViewSwitcher";

/**
 * State-driven landing -> workspace layout transition.
 *
 * `useIndexingStatus` and `useChat` are lifted here (previously owned by
 * `Sidebar` and the old combined `ChatPanel` respectively, both Part 1)
 * because their state now needs to render in two different places each:
 * indexing status on the landing page AND the workspace sidebar, chat on
 * `ChatView` AND (in principle) anywhere else that reacts to it. Same
 * two hooks, same calls, only the component that owns them moved - no
 * behavior change (confirmed by keeping every hook's own file untouched).
 *
 * Both the landing view and the workspace layout are always mounted -
 * `showWorkspace` only toggles opacity/transform/pointer-events, never
 * unmounts either subtree - so `useIndexingStatus`'s poll timer and
 * `useChat`'s in-flight request state survive the transition instead of
 * being reset by a remount, and the crossfade can be a plain CSS
 * transition instead of needing an animation library.
 *
 * `activeView` (Phase 32 Part 2 nav restructure): three equally-primary
 * views - Graph, Chat, Review a PR - switched via the top-of-workspace
 * `ViewSwitcher` (previously Chat lived in the sidebar + a floating bar
 * over the graph, and "Review a PR" was a muted button at the sidebar's
 * bottom). All three stay mounted regardless of which is active (same
 * "hidden, never unmounted" convention Graph/Review already used before
 * this change) so Chat's scroll position, Graph's blast-radius
 * highlight/zoom, and Review's in-progress card feed all survive
 * switching away and back.
 */
export function AppShell() {
  const { repoId, setRepoId, setRepoLabel, addChatTurn, resetRepo } = useRepo();
  const indexing = useIndexingStatus();
  const chat = useChat(repoId, addChatTurn);
  const showWorkspace = Boolean(repoId);
  const [activeView, setActiveView] = useState("graph"); // "graph" | "chat" | "review"

  useEffect(() => {
    if (indexing.phase === "ready" && indexing.repoId) {
      setRepoId(indexing.repoId);
      setRepoLabel(`${indexing.summary.owner}/${indexing.summary.name}`);
    }
  }, [indexing.phase, indexing.repoId, indexing.summary, setRepoId, setRepoLabel]);

  /**
   * "Verdict AI" home link (Sidebar.jsx) - returns to the landing state,
   * same as a fresh page load: resets both halves of the app's
   * pre-indexing state (the indexing hook's own phase/stage, and the
   * shared repo/chat/focus context) back to idle/empty together.
   */
  function goHome() {
    indexing.reset();
    resetRepo();
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#0B0F19]">
      <div
        className={
          "absolute inset-0 flex items-center justify-center transition-all duration-700 ease-out " +
          (showWorkspace ? "pointer-events-none -translate-y-4 scale-95 opacity-0" : "opacity-100")
        }
      >
        <LandingView indexing={indexing} />
      </div>

      <div
        className={
          "flex h-full w-full transition-all duration-700 ease-out " +
          (showWorkspace ? "opacity-100" : "pointer-events-none translate-y-4 opacity-0")
        }
      >
        <Sidebar indexing={indexing} onGoHome={goHome} />
        <main className="flex h-full min-w-0 flex-1 flex-col">
          <ViewSwitcher
            activeView={activeView} onSelectView={setActiveView}
            disabled={indexing.phase !== "ready"}
          />
          {/* All three views stay mounted - only `hidden` toggles, so switching back to Graph
              keeps its blast-radius highlight/zoom, switching back to Chat keeps its scroll
              position, and switching back to Review keeps its in-progress card feed; none
              ever remounts from scratch. */}
          <div className="relative min-h-0 flex-1">
            <div className={activeView === "graph" ? "h-full" : "hidden"}>
              <GraphView />
            </div>
            <div className={activeView === "chat" ? "h-full" : "hidden"}>
              <ChatView chat={chat} disabled={indexing.phase !== "ready"} />
            </div>
            <div className={activeView === "review" ? "h-full" : "hidden"}>
              <ReviewView />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
