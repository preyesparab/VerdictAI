import { GavelIcon, WarningIcon } from "../common/icons";

const VERDICT_STYLE = {
  approve: { label: "Approve", border: "border-emerald-500/40", glow: "shadow-emerald-900/40", badge: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40" },
  reject: { label: "Reject", border: "border-red-500/40", glow: "shadow-red-900/40", badge: "bg-red-500/15 text-red-300 border-red-500/40" },
  needs_human_review: { label: "Needs Human Review", border: "border-amber-500/40", glow: "shadow-amber-900/40", badge: "bg-amber-500/15 text-amber-300 border-amber-500/40" },
};

/**
 * The Judge's final, structured verdict (Phase 30) - the most visually
 * prominent card in the feed, per this screen's brief. `ReviewView.jsx`
 * pins this outside the scrollable card list once it arrives (not via
 * any special logic in this component - it's just rendered in a fixed
 * position by its parent).
 */
export function JudgeCard({ verdict }) {
  const style = VERDICT_STYLE[verdict.verdict];
  const confidencePercent = Math.round(verdict.confidence * 100);

  return (
    <div
      className={`animate-rm-card-in rounded-xl border-2 bg-slate-900/70 p-5 shadow-2xl backdrop-blur-xl ${style.border} ${style.glow}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <GavelIcon className="h-5 w-5 text-slate-300" />
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">Judge's Verdict</span>
        </div>
        <span className={`rounded-full border px-3 py-1 text-sm font-bold uppercase tracking-wide ${style.badge}`}>
          {style.label}
        </span>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
        <span className="font-mono text-xs text-slate-400">{confidencePercent}% confidence</span>
      </div>

      {verdict.cited_evidence.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1">
          {verdict.cited_evidence.map((item, i) => (
            <li key={i} className="text-sm leading-relaxed text-slate-300 before:mr-1.5 before:text-indigo-400 before:content-['—']">
              {item}
            </li>
          ))}
        </ul>
      )}

      {verdict.minority_report && (
        <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2">
          <WarningIcon className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-300">Minority report</div>
            <p className="mt-0.5 text-sm leading-relaxed text-amber-100/90">{verdict.minority_report}</p>
          </div>
        </div>
      )}
    </div>
  );
}
