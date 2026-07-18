import { useCallback, useRef, useState } from "react";
import { streamReview } from "../api/client";

const IDLE_STATE = { context: null, defenderText: null, claims: [], rebuttals: [], verdict: null, error: null };

/**
 * Drives the live review SSE stream (`POST /repos/{repo_id}/review`,
 * Phase 32 Part 2) into React state the card feed renders from directly.
 * Each `claim_verified` event updates that one claim's `verification`
 * field in place (matched by `index`) rather than appending a new
 * entry - the same claim card is meant to transition from "verifying…"
 * to its real result, not spawn a second card.
 */
export function useLiveReview(repoId) {
  const [status, setStatus] = useState("idle"); // idle | streaming | done | error
  const [state, setState] = useState(IDLE_STATE);
  const abortRef = useRef(null);

  const runReview = useCallback(
    async (diff) => {
      if (!repoId || !diff.trim()) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus("streaming");
      setState(IDLE_STATE);

      try {
        await streamReview(
          repoId,
          diff,
          (event) => {
            switch (event.type) {
              case "context":
                setState((prev) => ({ ...prev, context: event }));
                break;
              case "defender":
                setState((prev) => ({ ...prev, defenderText: event.justification }));
                break;
              case "claims":
                setState((prev) => ({
                  ...prev,
                  claims: event.claims.map((claim) => ({ ...claim, verification: null })),
                }));
                break;
              case "claim_verified":
                setState((prev) => ({
                  ...prev,
                  claims: prev.claims.map((claim) =>
                    claim.index === event.index ? { ...claim, verification: event } : claim,
                  ),
                }));
                break;
              case "rebuttal":
                setState((prev) => ({ ...prev, rebuttals: [...prev.rebuttals, event] }));
                break;
              case "judge":
                setState((prev) => ({ ...prev, verdict: event }));
                break;
              case "done":
                setStatus("done");
                break;
              case "error":
                setState((prev) => ({ ...prev, error: event.message }));
                setStatus("error");
                break;
              default:
                break;
            }
          },
          controller.signal,
        );
      } catch (err) {
        if (err.name === "AbortError") return;
        setState((prev) => ({ ...prev, error: err.message }));
        setStatus("error");
      }
    },
    [repoId],
  );

  return { status, ...state, runReview };
}
