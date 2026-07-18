import { useState } from "react";

const PLACEHOLDER = `diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,6 @@ def get_user_email(user):
     return user["email"]
+
+def get_user_role(user):
+    return user["role"]`;

/**
 * Raw unified-diff input + "Run Review" trigger - the format
 * `adjudicate.context_builder.parse_diff` actually expects (confirmed by
 * reading that function directly, not assumed): `+++ b/<path>` file
 * headers and `@@ -a,b +c,d @@` hunk headers, exactly what a real
 * `git diff`/`git show` emits.
 */
export function DiffInputPanel({ onRunReview, disabled }) {
  const [diff, setDiff] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    if (!diff.trim() || disabled) return;
    onRunReview(diff);
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded-lg border border-white/10 bg-slate-900/40 p-4 backdrop-blur-xl">
      <label className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-500">Unified diff</label>
      <textarea
        value={diff}
        onChange={(event) => setDiff(event.target.value)}
        placeholder={PLACEHOLDER}
        disabled={disabled}
        rows={8}
        spellCheck={false}
        className="scrollbar-thin resize-y rounded-md border border-white/10 bg-slate-950/50 p-3 font-mono text-xs leading-relaxed text-slate-200 placeholder:text-slate-600 transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || !diff.trim()}
        className="self-end rounded-md bg-gradient-to-r from-indigo-500 to-purple-600 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:scale-105 hover:brightness-110 disabled:cursor-default disabled:opacity-40 disabled:hover:scale-100 disabled:hover:brightness-100"
      >
        {disabled ? "Reviewing…" : "Run Review"}
      </button>
    </form>
  );
}
