import { useState } from "react";
import { ChatIcon } from "../common/icons";

/**
 * The chat question input, docked at the bottom of `ChatView` (Phase 32
 * Part 2 nav restructure moved chat from a floating bar over the graph
 * into its own top-level view - see `AppShell.jsx`). Same submit logic
 * as before (calls the lifted `ask` from `useChat`, passed down from
 * `AppShell`); only the outer chrome changed - a single rounded-full
 * "pill" glass bar (Gemini-style layout pass) instead of a
 * rectangular docked bar. The textarea itself is borderless/transparent
 * so the pill's own border is the only visible boundary; `ChatIcon` sits
 * as a leading glyph inside the pill rather than absolutely positioned
 * inside the textarea. No "+" attachment icon - there's nothing in this
 * app to attach, per this pass's own "skip if nothing to attach" option.
 *
 * Textarea and button both fixed to `h-[42px]` (rather than letting
 * `rows`/padding drive the textarea's height independently) so they're
 * guaranteed the same height regardless of font metrics, not just
 * visually close.
 */
export function ChatInputBar({ ask, pending, disabled }) {
  const [question, setQuestion] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim() || pending) return;
    const asked = question;
    setQuestion("");
    await ask(asked);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-shrink-0 items-center gap-2 rounded-full border border-white/10 bg-slate-900/80 p-2 pl-4 shadow-lg shadow-black/20 backdrop-blur-md"
    >
      <ChatIcon className="h-4 w-4 flex-shrink-0 text-slate-500" />
      <textarea
        placeholder={disabled ? "Index a repository to start chatting…" : "Ask about this codebase…"}
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
          }
        }}
        disabled={disabled || pending}
        rows={1}
        className="h-[42px] min-w-0 flex-1 resize-none overflow-hidden border-none bg-transparent py-2.5 text-sm leading-relaxed text-white placeholder:text-slate-400 focus:outline-none focus:ring-0 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || pending || !question.trim()}
        className="h-[42px] flex-shrink-0 whitespace-nowrap rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 px-5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:scale-105 hover:brightness-110 disabled:cursor-default disabled:opacity-40 disabled:hover:scale-100 disabled:hover:brightness-100"
      >
        Ask
      </button>
    </form>
  );
}
