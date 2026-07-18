import { CitationItem } from "./CitationItem";

/**
 * One (question, answer) turn - `turn` is `{ question, response }` from
 * `useChat`'s `addChatTurn`. Gemini-style layout: the question is a
 * right-aligned, bubbled chip (it's short, a distinct user action);
 * the answer is deliberately *not* bubbled - plain flowing text at
 * near-full width, left-aligned, read like a document rather than a
 * chat message. Citations render inline (`CitationItem` is an
 * `inline-flex` chip, not a block), flowing with the answer text
 * instead of stacking below it as a separate list - this `<div>` is
 * intentionally not a flex container so its span/button children wrap
 * inline like normal text.
 */
export function ChatMessage({ turn, nodesById }) {
  const { question, response } = turn;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl bg-indigo-500/10 px-4 py-2.5 text-sm leading-relaxed text-white">
          {question}
        </div>
      </div>

      <div className="max-w-3xl whitespace-pre-wrap text-[15px] leading-relaxed text-slate-300">
        {response.answer}
        {response.cache_hit && (
          <span className="ml-2 inline-block rounded border border-emerald-500 px-1.5 py-0.5 align-middle text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
            cached
          </span>
        )}
        {response.citations.map((citation, i) => (
          <CitationItem key={`${citation.chunk_id}-${i}`} citation={citation} nodesById={nodesById} />
        ))}
      </div>
    </div>
  );
}
