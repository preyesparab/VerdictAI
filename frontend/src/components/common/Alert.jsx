import { WarningIcon } from "./icons";

/** Soft translucent red alert for error messages - used by RepoInputPanel and ChatHistoryPanel. */
export function Alert({ children }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
      <WarningIcon className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
      <span>{children}</span>
    </div>
  );
}
