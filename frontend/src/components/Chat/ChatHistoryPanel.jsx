import { useEffect, useRef } from "react";
import { useRepo } from "../../state/RepoContext";
import { Alert } from "../common/Alert";
import { ChatMessage } from "./ChatMessage";

/**
 * Scrollable answer/citation history - `ChatView`'s main content, with
 * `ChatInputBar` docked below it. Moved out of the sidebar into its own
 * top-level view (Phase 32 Part 2 nav restructure); `pending`/`error`
 * are still the same lifted `useChat` state as before, passed down from
 * `AppShell` via `ChatView`.
 *
 * Auto-scroll-to-latest reacts to `history.length` changing (any new
 * turn, from anywhere) rather than being called imperatively right after
 * a submit inside this same component, since the submit itself happens
 * in a sibling component (`ChatInputBar`).
 */
export function ChatHistoryPanel({ nodesById, pending, error, disabled }) {
  const { history } = useRepo();
  const historyRef = useRef(null);

  useEffect(() => {
    requestAnimationFrame(() => {
      historyRef.current?.scrollTo({ top: historyRef.current.scrollHeight, behavior: "smooth" });
    });
  }, [history.length]);

  return (
    <div ref={historyRef} className="scrollbar-thin flex min-h-0 flex-1 flex-col gap-10 overflow-y-auto py-3">
      {history.length === 0 && (
        <div className="text-xs text-slate-500">
          {disabled ? "Index a repository to start chatting." : "Ask a question about this repository."}
        </div>
      )}
      {history.map((turn, i) => (
        <ChatMessage key={i} turn={turn} nodesById={nodesById} />
      ))}
      {pending && (
        <div className="flex items-center gap-2 text-xs italic text-slate-500">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400" />
          </span>
          Thinking…
        </div>
      )}
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
