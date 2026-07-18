import { useRepo } from "../../state/RepoContext";
import { ChatHistoryPanel } from "./ChatHistoryPanel";
import { ChatInputBar } from "./ChatInputBar";

/**
 * Full-height chat view (Phase 32 Part 2 nav restructure) - `chat`
 * ({ ask, pending, error }, from `useChat`) is lifted to `AppShell` and
 * passed down as a prop, same lifted-state pattern `GraphView`/
 * `ReviewView`'s siblings already use, so the in-flight request state
 * survives switching away to another view and back (this view stays
 * mounted, only `hidden` toggles - see `AppShell.jsx`).
 */
export function ChatView({ chat, disabled }) {
  const { nodesById } = useRepo();

  return (
    <div className="flex h-full w-full flex-col gap-3 overflow-hidden bg-[#0B0F19] p-6">
      <ChatHistoryPanel nodesById={nodesById} pending={chat.pending} error={chat.error} disabled={disabled} />
      <ChatInputBar ask={chat.ask} pending={chat.pending} disabled={disabled} />
    </div>
  );
}
