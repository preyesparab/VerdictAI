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
 * over SSE (`useLiveReview`). The diff input panel and every card
 * (Judge, context, Defender, claims, rebuttals) are sequential content
 * in one naturally-scrolling page - no sub-region gets its own scroll
 * container, so scrolling this view moves everything together, same
 * as a long article or chat log would.
 */
export function ReviewView() {
  const { repoId } = useRepo();
  const { status, context, defenderText, claims, rebuttals, verdict, error, runReview } = useLiveReview(repoId);

  return (
    <div className="scrollbar-thin flex h-full w-full flex-col gap-4 overflow-y-auto bg-[#0B0F19] p-6">
      <DiffInputPanel onRunReview={runReview} disabled={status === "streaming"} />

      {error && <Alert>{error}</Alert>}

      {verdict && <JudgeCard verdict={verdict} />}

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
  );
}
