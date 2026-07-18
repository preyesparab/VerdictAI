/** One Defender rebuttal round, responding to the Verifier's real results (Phase 29). */
export function RebuttalCard({ round, text }) {
  return (
    <div className="animate-rm-card-in rounded-lg border border-white/10 bg-slate-900/40 p-4 backdrop-blur-xl">
      <div className="flex items-center gap-2">
        <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-purple-300">
          Rebuttal · Round {round}
        </span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{text}</p>
    </div>
  );
}
