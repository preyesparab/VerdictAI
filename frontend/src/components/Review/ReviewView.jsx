import { useRepo } from "../../state/RepoContext";
import { useLiveReview } from "../../hooks/useLiveReview";
import { Alert } from "../common/Alert";
import { ClaimCard } from "./ClaimCard";
import { ContextCard } from "./ContextCard";
import { DefenderCard } from "./DefenderCard";
import { DiffInputPanel } from "./DiffInputPanel";
import { JudgeCard } from "./JudgeCard";
import { RebuttalCard } from "./RebuttalCard";

/**
 * The live adversarial review screen (Phase 32 Part 2) - an append-only
 * card feed rendering the real Phases 24-30 pipeline as it streams in
 * over SSE (`useLiveReview`). Once the Judge's verdict arrives, its card
 * is pinned above the scrollable feed (not inside it) - "visually the
 * most prominent" and always visible, per this screen's brief - while
 * every earlier card (context, Defender, claims, rebuttals) stays
 * reachable by scrolling below it.
 */
export function ReviewView() {
  const { repoId } = useRepo();
  const { status, context, defenderText, claims, rebuttals, verdict, error, runReview } = useLiveReview(repoId);

  return (
    <div className="flex h-full w-full flex-col gap-4 overflow-hidden bg-[#0B0F19] p-6">
      <DiffInputPanel onRunReview={runReview} disabled={status === "streaming"} />

      {error && <Alert>{error}</Alert>}

      {verdict && <JudgeCard verdict={verdict} />}

      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
        {status === "idle" && !context && (
          <div className="flex flex-1 items-center justify-center text-sm text-slate-600">
            Paste a diff above and run a review to watch the case build in real time.
          </div>
        )}
        {context && <ContextCard context={context} />}
        {defenderText && <DefenderCard text={defenderText} />}
        {claims.map((claim) => (
          <ClaimCard key={claim.index} claim={claim} />
        ))}
        {rebuttals.map((rebuttal) => (
          <RebuttalCard key={rebuttal.round} round={rebuttal.round} text={rebuttal.text} />
        ))}
      </div>
    </div>
  );
}
