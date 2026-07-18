import { ChatIcon, GavelIcon, GraphIcon } from "../common/icons";

const PLAIN_VIEWS = [
  { id: "graph", label: "Graph", icon: GraphIcon },
  { id: "chat", label: "Chat", icon: ChatIcon },
];

/**
 * Top-of-workspace pill switcher between the app's three primary views
 * (Phase 32 Part 2 nav restructure - previously "Review a PR" was a
 * muted button buried at the bottom of the sidebar; this puts Graph,
 * Chat, and Review a PR on equal footing at the top of the main
 * workspace area, one click away from wherever the user already is).
 *
 * "Review a PR" is deliberately styled apart from the other two: always
 * the accent gradient (not just on hover/active, unlike the plain
 * outline pills), a larger touch target, and a drop shadow - the same
 * "primary action" treatment `ChatInputBar`'s Ask button and
 * `RepoInputPanel`'s Index button already use elsewhere in this app -
 * signaling it's the flagship capability, not just a third equal tab.
 */
export function ViewSwitcher({ activeView, onSelectView, disabled }) {
  return (
    <nav className="flex flex-shrink-0 items-center gap-2 border-b border-white/10 bg-slate-900/30 px-6 py-3 backdrop-blur-xl">
      {PLAIN_VIEWS.map(({ id, label, icon: Icon }) => {
        const isActive = activeView === id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onSelectView(id)}
            disabled={disabled}
            className={
              "flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold transition disabled:cursor-default disabled:opacity-40 " +
              (isActive
                ? "border-indigo-400/40 bg-indigo-500/15 text-indigo-200"
                : "border-white/10 bg-slate-900/40 text-slate-300 hover:border-indigo-400/30 hover:text-white")
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        );
      })}

      <button
        type="button"
        onClick={() => onSelectView("review")}
        disabled={disabled}
        title="Review a PR"
        className={
          "flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:scale-105 hover:brightness-110 disabled:cursor-default disabled:opacity-40 disabled:hover:scale-100 disabled:hover:brightness-100 " +
          (activeView === "review" ? "ring-2 ring-white/40" : "")
        }
      >
        <GavelIcon className="h-4 w-4" />
        Review a PR
      </button>
    </nav>
  );
}
