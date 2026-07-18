import { useCallback, useState } from "react";
import { queryRepository } from "../api/client";

/**
 * Sends chat questions via `POST /repos/{repo_id}/query`. Request/response
 * only for now - `/query` has no streaming endpoint yet (see
 * `api/main.py`), so `pending` drives a loading state rather than an
 * incremental typewriter render. Real token-by-token streaming is Part
 * 2's SSE work, once the live-review pipeline needs it too.
 */
export function useChat(repoId, addChatTurn) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);

  const ask = useCallback(
    async (question) => {
      if (!repoId || !question.trim()) return;
      setPending(true);
      setError(null);
      try {
        const response = await queryRepository(repoId, question);
        addChatTurn({ question, response });
      } catch (err) {
        setError(err.message);
      } finally {
        setPending(false);
      }
    },
    [repoId, addChatTurn],
  );

  return { ask, pending, error };
}
