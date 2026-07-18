import { useCallback, useRef, useState } from "react";
import { getStatus, indexRepository } from "../api/client";

const POLL_INTERVAL_MS = 1000; // matches ui/api_client.py's POLL_INTERVAL_SECONDS

const IDLE_STATE = { phase: "idle", repoId: null, stage: null, error: null, summary: null };

/**
 * Drives repository indexing end to end: triggers `POST /repos/index`,
 * then polls `GET /repos/{repo_id}/status` (same 1s cadence
 * `ui/api_client.py` uses) until the repository is "ready" or "failed",
 * exposing the live stage name at every step so the UI can render real,
 * incremental progress instead of a generic spinner.
 *
 * `repoId` is set on `state` as soon as `POST /repos/index` responds
 * (not only once "ready") so a caller reacting to state changes (rather
 * than the fire-and-forget `start()` call itself) always has it.
 */
export function useIndexingStatus() {
  const [state, setState] = useState(IDLE_STATE);
  const pollTimer = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollTimer.current !== null) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  const start = useCallback(
    async (repoUrl) => {
      stopPolling();
      setState({ ...IDLE_STATE, phase: "indexing" });

      let repoId;
      try {
        const response = await indexRepository(repoUrl);
        repoId = response.repo_id;
      } catch (err) {
        setState({ ...IDLE_STATE, phase: "failed", error: err.message });
        return;
      }
      setState((prev) => ({ ...prev, repoId }));

      const poll = async () => {
        let payload;
        try {
          payload = await getStatus(repoId);
        } catch (err) {
          setState((prev) => ({ ...prev, phase: "failed", error: err.message }));
          return;
        }

        if (payload.status === "ready") {
          setState({ phase: "ready", repoId, stage: payload.stage, error: null, summary: payload });
          return;
        }
        if (payload.status === "failed") {
          setState({ phase: "failed", repoId, stage: payload.stage, error: payload.error, summary: null });
          return;
        }
        setState({ phase: "indexing", repoId, stage: payload.stage, error: null, summary: null });
        pollTimer.current = setTimeout(poll, POLL_INTERVAL_MS);
      };

      await poll();
    },
    [stopPolling],
  );

  /**
   * Return to the pre-indexing idle state - stops any in-flight poll and
   * clears `state` back to `IDLE_STATE`. Added for the "Verdict AI" home
   * link (Sidebar.jsx): going back to the landing page should look like
   * a fresh page load, not leave a stale "ready"/completed checklist
   * behind. Purely additive - `start`'s own logic is untouched.
   */
  const reset = useCallback(() => {
    stopPolling();
    setState(IDLE_STATE);
  }, [stopPolling]);

  return { ...state, start, stopPolling, reset };
}
