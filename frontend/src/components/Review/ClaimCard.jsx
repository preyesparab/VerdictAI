import { useState } from "react";
import { CheckIcon, SpinnerIcon, WarningIcon, XIcon } from "../common/icons";

const STATUS_STYLE = {
  confirmed: {
    label: "Confirmed", icon: XIcon,
    badge: "border-red-500/30 bg-red-500/10 text-red-300",
    ring: "border-l-red-500/60",
  },
  refuted: {
    label: "Refuted", icon: CheckIcon,
    badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    ring: "border-l-emerald-500/60",
  },
  inconclusive: {
    label: "Inconclusive", icon: WarningIcon,
    badge: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    ring: "border-l-amber-500/60",
  },
};

/**
 * One Prosecutor claim (Phase 27), starting in a "verifying…" state and
 * updating in place once its real `claim_verified` event arrives
 * (`claim.verification` is null until then - see `useLiveReview.js`).
 * Border color and status badge are intentionally the loudest visual
 * signal on the card: CONFIRMED (a real problem, red) vs REFUTED (the
 * concern didn't hold up, green) vs INCONCLUSIVE (amber) - matching this
 * screen's brief.
 */
export function ClaimCard({ claim }) {
  const [showTest, setShowTest] = useState(false);
  const verification = claim.verification;
  const style = verification ? STATUS_STYLE[verification.status] : null;
  const StatusIcon = style?.icon;

  return (
    <div
      className={
        "animate-rm-card-in rounded-lg border border-white/10 border-l-4 bg-slate-900/40 p-4 backdrop-blur-xl transition-colors " +
        (style ? style.ring : "border-l-slate-600")
      }
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-white/10 bg-slate-800/60 px-2 py-0.5 font-mono text-[10px] text-slate-300">
            {claim.claim_type}
          </span>
          <span className="font-mono text-[11px] text-slate-500">{claim.location}</span>
        </div>

        {verification ? (
          <span className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style.badge}`}>
            <StatusIcon className="h-3 w-3" />
            {style.label}
          </span>
        ) : (
          <span className="flex items-center gap-1 rounded-full border border-slate-600 bg-slate-800/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
            <SpinnerIcon className="h-3 w-3" />
            Verifying…
          </span>
        )}
      </div>

      <p className="mt-2 text-sm leading-relaxed text-slate-200">{claim.assertion}</p>

      <button
        type="button"
        onClick={() => setShowTest((v) => !v)}
        className="mt-2 text-[11px] text-slate-500 transition hover:text-slate-300"
      >
        {showTest ? "Hide" : "Show"} proposed test
      </button>
      {showTest && (
        <pre className="scrollbar-thin mt-1 max-h-40 overflow-auto rounded-md border border-white/10 bg-slate-950/60 p-2 font-mono text-[11px] text-slate-300">
          {claim.proposed_test}
        </pre>
      )}

      {verification && (
        <div className="mt-3 border-t border-white/10 pt-2">
          <div className="text-[11px] text-slate-500">via {verification.strategy} · {verification.confidence} confidence</div>
          <pre className="scrollbar-thin mt-1 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-slate-400">
            {verification.evidence}
          </pre>
        </div>
      )}
    </div>
  );
}
